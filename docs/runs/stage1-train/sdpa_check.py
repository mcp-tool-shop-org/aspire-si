"""Argon's finding 2: which SDPA pin reproduces A_0. Run on the card only after Argon's re-review, before G0.

    python sdpa_check.py --a0-raw <A_0 pilot-30-s0 raw.jsonl> --out <dir> [--task rv-bc9c8c5439]

A_0 ran the harness with the default SDPA backends (flash, efficient and math all enabled). det_run.py pins FLASH
alone. If A_0's decode steps (q_len 1, with an attention mask) ran on another kernel, or FLASH refuses that shape,
the pin changes the numerics or crashes. G0 would then read (b), a false re-baseline of about 6.6 h.

**One process, the model loaded once.** It takes one DEV task (rv-bc9c8c5439) at seed 0, with det_run's
deterministic settings (CUBLAS workspace, deterministic algorithms with warn_only, cuDNN flags), and runs it twice
under each setting:
- the default backends;
- FLASH alone;
- FLASH and EFFICIENT together.

Each run's turns (text and new_tokens) are compared with the setting's other run and with A_0's stored record for
that task at seed 0.

**The pin, by Argon's rule** (`decide`):
- **FLASH:** it ran twice equal (deterministic) and matches A_0.
- **[FLASH, EFFICIENT]:** FLASH alone raised or didn't match, and the pair ran twice equal and matches A_0.
- **No pin settled:** neither did. det_run's pin is then a question for R&D; nothing is changed here.

`sdpa-check.json` records every run's outcome, the comparisons, the warned ops, the machine line and the decision.
"""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # before torch loads, as det_run.py sets it

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _evaldir import EVAL_DIR  # noqa: E402

sys.path.insert(0, str(EVAL_DIR))
SETTINGS = ("default", "flash", "flash+efficient")
TASK_FILES = ("taskset-review-10.jsonl", "taskset-pilot-20.jsonl")


def turns_of(rec: dict) -> list:
    """What a run is compared on: each turn's generated text and new_tokens."""
    return [(t.get("text"), t.get("new_tokens")) for t in rec["turns"]]


def compare(results: dict, a0: dict) -> dict:
    """{setting: {"ran", "repeat_equal", "equals_a0", "error"}} from {setting: [run record or {"error"}]}."""
    out = {}
    for name, runs in results.items():
        errors = [r["error"] for r in runs if "error" in r]
        if errors:
            out[name] = {"ran": False, "repeat_equal": False, "equals_a0": False, "error": errors[0]}
            continue
        a, b = (turns_of(r) for r in runs)
        out[name] = {"ran": True, "repeat_equal": a == b, "equals_a0": a == turns_of(a0), "error": None}
    return out


def decide(cmp: dict) -> str:
    """Argon's rule: FLASH if it's deterministic and matches A_0; else the pair if it is; else no pin settled."""
    def good(name):
        c = cmp.get(name) or {}
        return c.get("ran") and c.get("repeat_equal") and c.get("equals_a0")
    if good("flash"):
        return "flash"
    if good("flash+efficient"):
        return "flash+efficient"
    return "none: no pinned setting is deterministic and matches A_0; R&D decides"


def main(argv=None) -> int:  # pragma: no cover - runs on the card
    ap = argparse.ArgumentParser()
    ap.add_argument("--a0-raw", required=True, help="A_0's pilot-30 seed-0 raw.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--task", default="rv-bc9c8c5439")
    a = ap.parse_args(argv)
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    import det_run
    import harness
    import student
    tasks = {}
    for f in TASK_FILES:
        for line in (HERE.parent / "stage1-taskset" / f).read_text(encoding="utf-8").splitlines():
            if line.strip():
                t = json.loads(line)
                tasks[t["id"]] = t
    task = tasks[a.task]
    a0 = next(json.loads(x) for x in Path(a.a0_raw).read_text(encoding="utf-8").splitlines()
              if x.strip() and json.loads(x)["task_id"] == a.task)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.use_deterministic_algorithms(True, warn_only=True)  # det_run's settings; warn_only so a run reports
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    print(json.dumps(student.verify_weights()), flush=True)
    tok = student.load_tokenizer()
    model = student.load_student()
    model.eval()
    pins = {"flash": [SDPBackend.FLASH_ATTENTION],
            "flash+efficient": [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]}
    results, warned = {}, {}
    for name in SETTINGS:
        results[name] = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for k in range(2):
                try:
                    if name == "default":
                        rec = harness.run_task(model, tok, task, "sampled", 0)
                    else:
                        with sdpa_kernel(pins[name]):
                            rec = harness.run_task(model, tok, task, "sampled", 0)
                    results[name].append(rec)
                except RuntimeError as e:
                    results[name].append({"error": str(e)[:300]})
                print(f"{name} run {k + 1}: {'error' if 'error' in results[name][-1] else 'ok'}", flush=True)
        warned[name] = sorted({str(w.message) for w in caught if "deterministic" in str(w.message).lower()})
    cmp = compare(results, a0)
    report = {"task": a.task, "seed": 0, "comparison": cmp, "decision": decide(cmp), "warned_ops": warned,
              "machine_line": det_run.machine_line(), "a0_raw": Path(a.a0_raw).name}
    (out / "sdpa-check.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({"comparison": cmp, "decision": report["decision"]}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
