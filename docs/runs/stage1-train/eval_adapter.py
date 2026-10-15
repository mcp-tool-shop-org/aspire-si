"""Evaluate a trained adapter with the pinned eval harness, unchanged.

    python eval_adapter.py --adapter <round dir>/adapter --tasks <tasks.jsonl> --out <run dir> --seed N
                           [--phase A-after] [--arm ROLE] [--limit N]

The base model comes from student.load_student() and the adapter is attached unmerged, so the base weights
and the inference path are the ones the baseline used. Generation, parsing and grading are harness.run_task,
whose file is untouched (its sha256 is recorded beside the adapter's). Output has the same raw.jsonl format,
so score.py, summarize.py and the mapper read it as they read the baseline.

A sealed sitting (the tasks file is the frozen sealed set) refuses an adapter that wasn't trained on the whole
frozen dataset: one whose manifest.json (beside the adapter) is missing, or records freeze_report.partial_set
(a smoke or a test trained with --partial-set). A smoke adapter must never spend a sealed read (R&D).
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _evaldir import EVAL_DIR  # noqa: E402

sys.path.insert(0, str(EVAL_DIR))
import harness  # noqa: E402
import mapper  # noqa: E402
import student  # noqa: E402


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


SEALED_SHA256 = "e1e515b4201b3341c363d0cf1061356acf7775c3d4b5b5194afa9790223fc683"  # the sealed-freeze record


def is_sealed(tasks: Path) -> bool:
    return hashlib.sha256(tasks.read_bytes().replace(b"\r\n", b"\n")).hexdigest() == SEALED_SHA256


def sealed_read_allowed(tasks: Path, adapter: Path) -> str | None:
    """None when the sitting may run; otherwise why a sealed read is refused for this adapter."""
    if not is_sealed(tasks):
        return None  # not the sealed set
    return frozen_adapter_problem(adapter)


def frozen_adapter_problem(adapter: Path) -> str | None:
    """None when the adapter was trained on the whole frozen dataset; otherwise why not. Every sealed read (the
    sealed tasks here, the sealed interview questions in interview_adapter.py) requires it."""
    manifest = adapter.parent / "manifest.json"
    if not manifest.exists():
        return "no manifest.json beside the adapter: can't show it was trained on the frozen dataset"
    if (json.loads(manifest.read_text(encoding="utf-8")).get("freeze_report") or {}).get("partial_set"):
        return "the adapter was trained with --partial-set (a smoke or a test), not on the frozen dataset"
    return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--phase", default="A-after")
    ap.add_argument("--arm", default="ROLE")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args(argv)

    adapter = Path(a.adapter)
    why = sealed_read_allowed(Path(a.tasks), adapter)
    if why:
        raise SystemExit(f"refusing a sealed sitting: {why}")
    tasks = [json.loads(line) for line in open(a.tasks, encoding="utf-8")][: a.limit]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = {**harness.settings("sampled", a.seed), "tasks_file": Path(a.tasks).name,
           "tasks_sha256": sha256_file(Path(a.tasks)), "phase": a.phase, "arm": a.arm,
           "adapter_sha256": sha256_file(adapter / "adapter_model.safetensors"),
           "eval_adapter_sha256": sha256_file(Path(__file__))}
    cfg_path = out / "run-config.json"
    if cfg_path.exists() and json.loads(cfg_path.read_text(encoding="utf-8")) != cfg:
        raise SystemExit("refusing to resume under different settings")
    cfg_path.write_text(json.dumps(cfg, indent=1), encoding="utf-8")

    from peft import PeftModel
    print(json.dumps(student.verify_weights()), flush=True)
    tok = student.load_tokenizer()
    model = PeftModel.from_pretrained(student.load_student(), str(adapter))
    model.eval()
    raw = out / "raw.jsonl"
    done = {json.loads(line)["task_id"] for line in open(raw, encoding="utf-8")} if raw.exists() else set()
    for t in tasks:
        if t["id"] in done:
            continue
        rec = harness.run_task(model, tok, t, "sampled", a.seed)
        rec.update({"phase": a.phase, "arm": a.arm, "seed": a.seed, "harness_sha256": cfg["harness_sha256"],
                    "adapter_sha256": cfg["adapter_sha256"]})
        with open(raw, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(t["id"], rec["final"]["verdict"], "ok" if rec["grade"]["verdict_correct"] else "WRONG",
              sum(x["new_tokens"] for x in rec["turns"]), "tok", flush=True)
    mapper.map_run(out, tasks, tok)
    if not is_sealed(Path(a.tasks)):  # descriptive; the sealed outputs stay unopened until round F
        import format_report
        format_report.write(raw)
    return 0


if __name__ == "__main__":
    sys.exit(main())
