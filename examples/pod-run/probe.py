"""Probe trajectory: the same inputs through the base student and every epoch checkpoint.

The per-step geometry export pools the hidden states of whichever prompt a step trained on, so
its trajectory mostly shows prompt identity. This replays one fixed set (each prompt with the
response the teacher scored, from the dialogue cache) through the base student and through
checkpoint-1..N, so any movement between checkpoints is the training's, not the sampling's.

Writes, in the run's output directory:
  probe_states.npz        states[checkpoint, prompt, hidden] (float32), names, prompts
  probe_geometry.json     a geometry export, checkpoint-major, prompts in a fixed order
  probe_summary.json      per-checkpoint drift numbers

Usage (from the repository root): python examples/pod-run/probe.py outputs/<run>
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from aspire.config import AspireConfig
from aspire.geometry import GeometryRecorder, pool_hidden_states
from aspire.judge import encode_exchanges

run = Path(sys.argv[1])
checkpoints = sorted(run.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
cfg = AspireConfig.from_yaml(checkpoints[-1] / "config.yaml")

pairs, scores, teacher_scores = [], [], []
for f in sorted((run / "dialogue_cache").glob("*.json")):
    d = json.loads(f.read_text(encoding="utf-8"))
    pairs.append((d["prompt"], d["final_response"]))
    ev = d["final_evaluation"]
    scores.append({x["dimension"]: float(x["score"]) for x in ev["dimension_scores"]})
    meta = ev.get("metadata") or {}
    ts = meta.get("teacher_scores") or {meta.get("teacher", "teacher"): ev["overall_score"]}
    teacher_scores.append({k: float(v) for k, v in ts.items()})
# A fixed order: by prompt text.
order = sorted(range(len(pairs)), key=lambda i: pairs[i][0])
pairs = [pairs[i] for i in order]
scores = [scores[i] for i in order]
teacher_scores = [teacher_scores[i] for i in order]

quant = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
) if cfg.student.load_in_4bit else None


def states_for(model, tokenizer):
    model.eval()
    out = []
    with torch.no_grad():
        for prompt, response in pairs:
            ids, mask = encode_exchanges(tokenizer, [prompt], [response], cfg.student.max_length)
            ids, mask = ids.to(model.device), mask.to(model.device)
            hidden = model(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states[-1]
            out.append(pool_hidden_states(hidden.float(), mask))
    return np.stack(out).astype(np.float32)


tokenizer = AutoTokenizer.from_pretrained(checkpoints[-1] / "student")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
base = AutoModelForCausalLM.from_pretrained(
    cfg.student.model_name_or_path, quantization_config=quant, device_map={"": "cuda"}, dtype=torch.bfloat16
)
names, all_states = ["base"], [states_for(base, tokenizer)]

from peft import PeftModel  # noqa: E402

for ckpt in checkpoints:
    model = PeftModel.from_pretrained(base, ckpt / "student")
    names.append(ckpt.name)
    all_states.append(states_for(model, tokenizer))
    base = model.unload()  # back to the bare base for the next adapter
    print(f"{ckpt.name}: done", flush=True)

states = np.stack(all_states)  # [checkpoints, prompts, hidden]
np.savez_compressed(run / "probe_states.npz", states=states, names=np.array(names),
                    prompts=np.array([p for p, _ in pairs]))

# Drift: how far each prompt's state moved from base, against how far apart prompts are.
def cos_dist(a, b):
    return 1 - (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))

between = []
for i in range(states.shape[1]):
    for j in range(i + 1, states.shape[1]):
        between.append(cos_dist(states[0, i], states[0, j]))
summary = {"checkpoints": names, "prompts": len(pairs),
           "between_prompt_cosine_distance_at_base": float(np.mean(between)), "per_checkpoint": []}
for k, name in enumerate(names):
    shift = cos_dist(states[k], states[0])
    step = cos_dist(states[k], states[k - 1]) if k else np.zeros(len(pairs))
    centroid_shift = float(np.linalg.norm(states[k].mean(0) - states[0].mean(0)))
    summary["per_checkpoint"].append({
        "name": name,
        "mean_cosine_shift_from_base": float(shift.mean()),
        "max_cosine_shift_from_base": float(shift.max()),
        "mean_cosine_shift_from_previous": float(step.mean()),
        "centroid_shift_from_base_l2": centroid_shift,
    })
(run / "probe_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
print(json.dumps(summary, indent=1))

recorder = GeometryRecorder(run_id=f"{cfg.experiment_name}-probe",
                            condition=f"probe: {len(pairs)} fixed exchanges x {', '.join(names)}",
                            seed=cfg.seed, window=cfg.training.geometry_window)
for k in range(states.shape[0]):
    for i in range(states.shape[1]):
        recorder.record_step(states[k, i], scores[i], teacher_scores[i])
recorder.write(run / "probe_geometry.json", training_items=len(pairs), cycles=len(checkpoints))
print("PROBE-OK", run)
