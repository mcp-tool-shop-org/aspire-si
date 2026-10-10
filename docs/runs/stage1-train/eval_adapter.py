"""Evaluate a trained adapter with the pinned eval harness, unchanged.

    python eval_adapter.py --adapter <round dir>/adapter --tasks <tasks.jsonl> --out <run dir> --seed N
                           [--phase A-after] [--arm ROLE] [--limit N]

The base model comes from student.load_student() and the adapter is attached unmerged, so the base weights
and the inference path are the ones the baseline used. Generation, parsing and grading are harness.run_task,
whose file is untouched (its sha256 is recorded beside the adapter's). Output has the same raw.jsonl format,
so score.py, summarize.py and the mapper read it as they read the baseline.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "stage1_eval"))
import harness  # noqa: E402
import mapper  # noqa: E402
import student  # noqa: E402


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


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
    return 0


if __name__ == "__main__":
    sys.exit(main())
