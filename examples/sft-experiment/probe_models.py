"""Probe any sequence of students on one fixed set of exchanges, and write drift exports.

Each entry is `name=model` or `name=model+adapter` (a base model, or a base with a LoRA adapter on
top). The exchanges are the (prompt, scored response) pairs of a dialogue cache: the control's,
so every stage of the experiment is measured on the same 32 inputs as the control.

Writes to --out:
  states.npz                       states[entry, exchange, hidden], names, prompts
  summary.json                     drift numbers per entry, from --drift-from and from the first entry
  drift-from-<name>.geometry.json  a schema 1.1 drift export (checkpoint_by_item) per --drift-from

Usage: python probe_models.py --pairs <run>/dialogue_cache --out probe
         base=Qwen/Qwen2.5-1.5B-Instruct sft=sft/merged sft-aspire-1=sft/merged+outputs/x/checkpoint-1/student
         --drift-from base --drift-from sft
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))


def load_pairs(cache: Path) -> tuple[list[tuple[str, str]], list[dict], list[dict]]:
    """Exchanges sorted by prompt, with their dimension scores and per-teacher scores."""
    rows = []
    for f in cache.glob("*.json"):
        d = json.loads(f.read_text(encoding="utf-8"))
        ev = d["final_evaluation"]
        meta = ev.get("metadata") or {}
        teachers = meta.get("teacher_scores") or {meta.get("teacher", "teacher"): ev["overall_score"]}
        rows.append(
            (
                d["prompt"],
                d["final_response"],
                {x["dimension"]: float(x["score"]) for x in ev["dimension_scores"]},
                {k: float(v) for k, v in teachers.items()},
            )
        )
    rows.sort(key=lambda r: r[0])
    return [(r[0], r[1]) for r in rows], [r[2] for r in rows], [r[3] for r in rows]


def parse_entry(text: str) -> tuple[str, str, str | None]:
    name, _, spec = text.partition("=")
    if not name or not spec:
        raise SystemExit(f"entry {text!r} must be name=model or name=model+adapter")
    model, _, adapter = spec.partition("+")
    return name, model, adapter or None


def drift_summary(states: np.ndarray, names: list[str], ref: int) -> dict:
    def cos_dist(a, b):
        return 1 - (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))

    out = {"from": names[ref], "per_entry": []}
    for k, name in enumerate(names):
        drift = states[k] - states[ref]
        shift = cos_dist(states[k], states[ref])
        mean_dir = drift.mean(0)
        norm = np.linalg.norm(mean_dir)
        shared = (
            float(np.mean(drift @ mean_dir / (np.linalg.norm(drift, axis=-1) * norm + 1e-12)))
            if norm
            else 0.0
        )
        out["per_entry"].append(
            {
                "name": name,
                "mean_cosine_shift": float(shift.mean()),
                "mean_drift_norm": float(np.linalg.norm(drift, axis=-1).mean()),
                "shared_direction": shared,
            }
        )
    return out


def alignment(
    states: np.ndarray, ref: int, k: int, scores: np.ndarray, nulls: int = 500, seed: int = 0
) -> dict:
    """r(position along the mean drift from ref to k, teacher score), with a random-direction null."""
    drift = states[k] - states[ref]
    u = drift.mean(0)
    if not np.linalg.norm(u):
        return {"r": None, "null_95": None}
    u = u / np.linalg.norm(u)
    centred = states[k] - states[k].mean(0)
    r = float(np.corrcoef(centred @ u, scores)[0, 1])
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(nulls):
        v = rng.standard_normal(states.shape[2])
        null.append(abs(np.corrcoef(centred @ (v / np.linalg.norm(v)), scores)[0, 1]))
    return {"r": r, "null_95": float(np.percentile(null, 95))}


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - needs models
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("entries", nargs="+")
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--drift-from", action="append", default=[])
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from aspire.geometry import GeometryRecorder, pool_hidden_states
    from aspire.judge import encode_exchanges

    pairs, dims, teachers = load_pairs(args.pairs)
    entries = [parse_entry(e) for e in args.entries]
    quant = (
        None
        if args.no_4bit or args.device == "cpu"
        else BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    )
    tokenizer = AutoTokenizer.from_pretrained(entries[0][1])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_states = []
    loaded: dict[str, object] = {}
    for name, model_path, adapter in entries:
        base = loaded.get(model_path)
        if base is None:
            loaded.clear()
            base = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quant,
                device_map={"": args.device},
                dtype=torch.bfloat16 if args.device == "cuda" else torch.float32,
            )
            loaded[model_path] = base
        model = base
        if adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(base, adapter)
        model.eval()
        rows = []
        with torch.no_grad():
            for prompt, response in pairs:
                ids, mask = encode_exchanges(tokenizer, [prompt], [response], args.max_length)
                ids, mask = ids.to(model.device), mask.to(model.device)
                hidden = model(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states[
                    -1
                ]
                rows.append(pool_hidden_states(hidden.float(), mask))
        all_states.append(np.stack(rows).astype(np.float32))
        if adapter:
            loaded[model_path] = model.unload()
        print(f"{name}: done", flush=True)

    names = [e[0] for e in entries]
    states = np.stack(all_states).astype(np.float64)
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out / "states.npz",
        states=states.astype(np.float32),
        names=np.array(names),
        prompts=np.array([p for p, _ in pairs]),
    )
    overall = np.array([sum(t.values()) / len(t) for t in teachers])
    summary = {"exchanges": len(pairs), "entries": names, "drift": [], "alignment": {}}
    for ref_name in args.drift_from or [names[0]]:
        ref = names.index(ref_name)
        summary["drift"].append(drift_summary(states, names, ref))
        later = [k for k in range(len(names)) if k > ref]
        summary["alignment"][ref_name] = {names[k]: alignment(states, ref, k, overall) for k in later}
        rec = GeometryRecorder(
            run_id=f"drift-from-{ref_name}",
            condition=(
                f"drift from {ref_name}: {len(pairs)} fixed exchanges x "
                + ", ".join(names[k] for k in later)
            ),
            window=4,
            step_axis="checkpoint_by_item",
            checkpoints=len(later),
            scalar_source="fixed_per_item",
        )
        for k in later:
            for i in range(len(pairs)):
                rec.record_step(states[k, i] - states[ref, i], dims[i], teachers[i])
        rec.write(
            args.out / f"drift-from-{ref_name}.geometry.json", training_items=len(pairs), cycles=len(later)
        )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    print("PROBE-OK", args.out)


if __name__ == "__main__":
    main()
