"""The pre-interview for a trained adapter, with interview.py unchanged.

    python interview_adapter.py --adapter <round dir>/adapter --out <run dir> [--seeds 0 1 2]
                                --sitting round1 --arm ROLE

interview.py asks its 12 sealed questions of whatever `student.load_student()` returns. This wrapper runs
interview.main with that one call wrapped: the base model loads as before (weights verified, same loader), and the
adapter is attached unmerged, as eval_adapter attaches it for the tasks. The frame, the questions, the decoding and
the records are interview.py's own, and its file is untouched (its sha256 is in interview-config.json as before).

Beside interview.py's config, `adapter-config.json` records the adapter's sha256 and this file's. A resume under a
different adapter is refused, so one run directory never mixes two adapters' answers.

The questions are sealed, so the adapter must have been trained on the whole frozen dataset (eval_adapter's
rule): one with no manifest.json, or trained with --partial-set, is refused before any question is asked.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _evaldir import EVAL_DIR  # noqa: E402
from eval_adapter import frozen_adapter_problem  # noqa: E402

sys.path.insert(0, str(EVAL_DIR))
import interview  # noqa: E402
import student  # noqa: E402


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def with_adapter(load, adapter: Path):
    """`load` (student.load_student) with the adapter attached unmerged to what it returns."""
    def load_with_adapter(*args, **kwargs):
        from peft import PeftModel
        model = PeftModel.from_pretrained(load(*args, **kwargs), str(adapter))
        model.eval()
        return model
    return load_with_adapter


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    a, rest = ap.parse_known_args(argv)
    adapter = Path(a.adapter)
    why = frozen_adapter_problem(adapter)
    if why:
        raise SystemExit(f"refusing an interview sitting: {why}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = {"adapter_sha256": sha256_file(adapter / "adapter_model.safetensors"),
           "interview_adapter_sha256": sha256_file(Path(__file__)),
           "interview_py_sha256": sha256_file(EVAL_DIR / "interview.py")}
    p = out / "adapter-config.json"
    if p.exists() and json.loads(p.read_text(encoding="utf-8")) != cfg:
        raise SystemExit("refusing to resume under a different adapter")
    p.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    load = student.load_student
    student.load_student = with_adapter(load, adapter)
    try:
        return interview.main(["--out", str(out), *rest])
    finally:
        student.load_student = load


if __name__ == "__main__":
    sys.exit(main())
