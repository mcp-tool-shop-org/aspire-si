"""Where the eval package lives: `stage1-eval` in the repository (docs/runs/), `stage1_eval` in a working copy."""

from pathlib import Path

HERE = Path(__file__).resolve().parent
EVAL_DIR = next(p for p in (HERE.parent / "stage1-eval", HERE.parent / "stage1_eval") if (p / "harness.py").exists())
