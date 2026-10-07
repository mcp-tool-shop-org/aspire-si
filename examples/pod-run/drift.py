"""Drift export: what training changed, with prompt identity removed.

Reads probe_states.npz (states[checkpoint, prompt, hidden], checkpoint 0 = base) and writes
drift_geometry.json: one step per (checkpoint >= 1, prompt), checkpoint-major, whose state is
that prompt's hidden state minus its own state in the base model. Scalars and professors are
the teacher's scores for that prompt, as in the per-step export.

Usage: python examples/pod-run/drift.py outputs/<run>   (after probe.py)
"""

import json
import sys
from pathlib import Path

import numpy as np

from aspire.geometry import GeometryRecorder

run = Path(sys.argv[1])
probe = np.load(run / "probe_states.npz")
states, names, prompts = probe["states"].astype(np.float64), list(probe["names"]), list(probe["prompts"])

by_prompt = {}
for f in (run / "dialogue_cache").glob("*.json"):
    d = json.loads(f.read_text(encoding="utf-8"))
    ev = d["final_evaluation"]
    meta = ev.get("metadata") or {}
    by_prompt[d["prompt"]] = (
        {x["dimension"]: float(x["score"]) for x in ev["dimension_scores"]},
        {k: float(v) for k, v in (meta.get("teacher_scores") or {meta.get("teacher", "teacher"): ev["overall_score"]}).items()},
    )

summary = json.loads((run / "probe_summary.json").read_text(encoding="utf-8"))
recorder = GeometryRecorder(
    run_id=f"{run.name}-drift",
    condition=f"drift from base: {len(prompts)} fixed exchanges x {', '.join(names[1:])}",
    window=4,
)
for k in range(1, states.shape[0]):
    for i, prompt in enumerate(prompts):
        scores, teachers = by_prompt[str(prompt)]
        recorder.record_step(states[k, i] - states[0, i], scores, teachers)
out = recorder.write(run / "drift_geometry.json", training_items=len(prompts), cycles=states.shape[0] - 1)
doc = json.loads(out.read_text(encoding="utf-8"))
print(out, "steps", len(doc["trajectory"]["timesteps"]), "explained", [round(x, 3) for x in doc["reduction"]["explained_variance"]])
drift = states[1:] - states[0]
norms = np.linalg.norm(drift, axis=-1)
print("drift norm by checkpoint (mean over prompts):", [round(float(n), 4) for n in norms.mean(1)])
cos = []
for k in range(drift.shape[0] - 1):
    a, b = drift[k], drift[k + 1]
    cos.append(float(np.mean((a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)))))
print("direction kept between checkpoints (mean cosine of each prompt's drift):", [round(c, 3) for c in cos])
mean_dirs = drift.mean(1)
common = [float(np.mean(drift[k] @ mean_dirs[k] / (np.linalg.norm(drift[k], axis=-1) * np.linalg.norm(mean_dirs[k])))) for k in range(drift.shape[0])]
print("how shared the drift is across prompts (mean cosine to the average drift):", [round(c, 3) for c in common])
