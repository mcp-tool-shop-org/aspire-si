"""The determinism wrapper for every batch-1 sitting (DEV, sealed, interview), with the harness unchanged
(R&D's settlement of Argon's round-1 review, 2026-10-15).

    python det_run.py --machine-line <frozen.json> [--warn-only] -- <target> <the target's own arguments>
    python det_run.py --write-machine-line <frozen.json> [--warn-only] -- <target> ...     (G0 only)

<target> is harness, eval_adapter or interview_adapter. The wrapper calls the target's own main(argv) in this
process. The harness's bytes stay as pinned (harness identity 8d63a541, harness.py 4fa52448), and nothing here
edits a target.

Before anything is imported from torch, it sets CUBLAS_WORKSPACE_CONFIG. Then, around the whole sitting:
- torch.use_deterministic_algorithms(True), or with warn_only=True under --warn-only (R&D: acceptable when the
  strict mode raises; every warned op is logged);
- cudnn.deterministic = True and cudnn.benchmark = False;
- the SDPA backend pinned to FLASH with torch.nn.attention.sdpa_kernel. The pin is the record: sdpa_kernel errors
  if flash can't run, and that error is a stop, not a fallback. `--sdpa efficient` exists only for the ruled case
  where flash won't run on this card, and it's recorded.

**The machine line, a hard stop (run sheet §3; Argon's Q2).** The GPU name, the driver, torch's CUDA, torch and
transformers are read first. With --machine-line, any field that differs from the frozen line stops the sitting
before a model loads (exit 5). G0's M1 alone writes the frozen line, with --write-machine-line, which refuses
an existing file (exit 7). M2 and every later sitting check it.

`wrapper-config.json` in the target's --out directory records:
- the settings, the machine line, and this file's sha256;
- the sha256 of the CPU and CUDA RNG states at the start (the harness re-seeds before every generation);
- the warned ops.

A resume under a different wrapper config is refused (exit 6), like the harness's own resume check.

Exit codes:
- the target's own on success;
- 4: a strict deterministic op raised; rerun with --warn-only into a fresh directory;
- 5: the machine line differs;
- 6: a resume under a different wrapper config;
- 7: --write-machine-line names a file that already exists (the frozen line is never overwritten).
"""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # before torch initialises CUDA

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _evaldir import EVAL_DIR  # noqa: E402

sys.path.insert(0, str(EVAL_DIR))
TARGETS = ("harness", "eval_adapter", "interview_adapter")
MACHINE_FIELDS = ("gpu", "driver", "cuda", "torch", "transformers")
EXIT = {"strict_raised": 4, "machine_line": 5, "resume": 6, "line_exists": 7}


def gpu_line() -> dict:
    """GPU 0's name and driver from nvidia-smi (no shell); empty strings when it doesn't answer."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
        name, driver = (x.strip() for x in out[0].split(","))
        return {"gpu": name, "driver": driver}
    except (OSError, subprocess.SubprocessError, IndexError, ValueError):
        return {"gpu": "", "driver": ""}


def machine_line() -> dict:
    import torch
    import transformers
    return {**gpu_line(), "cuda": torch.version.cuda or "", "torch": torch.__version__,
            "transformers": transformers.__version__}


def line_differences(now: dict, frozen: dict) -> list:
    return [f"{k}: {frozen.get(k)!r} -> {now.get(k)!r}" for k in MACHINE_FIELDS if now.get(k) != frozen.get(k)]


def rng_digest() -> dict:
    import torch
    out = {"cpu": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()}
    if torch.cuda.is_available():
        out["cuda"] = hashlib.sha256(torch.cuda.get_rng_state().numpy().tobytes()).hexdigest()
    return out


def out_dir(target_args: list) -> Path:
    if "--out" not in target_args or target_args.index("--out") + 1 >= len(target_args):
        raise SystemExit("the target's arguments need --out <dir>")
    return Path(target_args[target_args.index("--out") + 1])


def is_determinism_error(e: BaseException) -> bool:
    return isinstance(e, RuntimeError) and "deterministic" in str(e).lower()


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" not in argv:
        raise SystemExit("usage: det_run.py [options] -- <target> <target arguments>")
    cut = argv.index("--")
    own, rest = argv[:cut], argv[cut + 1:]
    ap = argparse.ArgumentParser()
    line = ap.add_mutually_exclusive_group(required=True)
    line.add_argument("--machine-line", help="the frozen machine line (JSON); any difference stops the sitting")
    line.add_argument("--write-machine-line", help="G0 only: write this machine's line as the frozen line")
    ap.add_argument("--warn-only", action="store_true")
    ap.add_argument("--sdpa", choices=("flash", "efficient"), default="flash")
    a = ap.parse_args(own)
    if not rest or rest[0] not in TARGETS:
        raise SystemExit(f"the target must be one of {', '.join(TARGETS)}")
    target, target_args = rest[0], rest[1:]
    out = out_dir(target_args)

    now = machine_line()
    if a.machine_line:
        frozen = json.loads(Path(a.machine_line).read_text(encoding="utf-8"))
        diff = line_differences(now, frozen)
        if diff:
            print("machine line differs from the frozen line, stopping: " + "; ".join(diff), flush=True)
            return EXIT["machine_line"]
    else:
        target_line = Path(a.write_machine_line)
        if target_line.exists():  # a wrong flag must never re-freeze the line and pass (R&D)
            print(f"refusing to overwrite the frozen machine line {target_line.name}; a new line needs a dated "
                  "amendment and a new G0", flush=True)
            return EXIT["line_exists"]
        target_line.write_text(json.dumps(now, indent=1), encoding="utf-8")

    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    backend = {"flash": SDPBackend.FLASH_ATTENTION, "efficient": SDPBackend.EFFICIENT_ATTENTION}[a.sdpa]
    cfg = {"wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "target": target,
           "deterministic_algorithms": True, "warn_only": a.warn_only,
           "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"], "cudnn_deterministic": True,
           "cudnn_benchmark": False, "sdpa_backend": a.sdpa, "machine_line": now}
    out.mkdir(parents=True, exist_ok=True)
    p = out / "wrapper-config.json"
    if p.exists():
        old = json.loads(p.read_text(encoding="utf-8"))
        if {k: old.get(k) for k in cfg} != cfg:
            print("refusing to resume under a different wrapper config", flush=True)
            return EXIT["resume"]
    record = {**cfg, "rng_at_start": rng_digest(), "warned_ops": [], "outcome": "running"}

    def save():
        p.write_text(json.dumps(record, indent=1), encoding="utf-8")
    save()

    torch.use_deterministic_algorithms(True, warn_only=a.warn_only)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    module = __import__(target)
    caught: list = []
    try:
        with warnings.catch_warnings(record=True) as caught, sdpa_kernel([backend]):
            warnings.simplefilter("always")
            code = module.main(target_args)
        record["outcome"] = f"exit {code or 0}"
        return code or 0
    except RuntimeError as e:
        if is_determinism_error(e) and not a.warn_only:
            record["outcome"] = f"strict deterministic op raised: {e}"
            print(f"{record['outcome']}; rerun with --warn-only into a fresh --out", flush=True)
            return EXIT["strict_raised"]
        record["outcome"] = f"error: {e}"
        raise
    finally:
        record["warned_ops"] = sorted({str(w.message) for w in caught or []
                                       if "deterministic" in str(w.message).lower()})
        save()
        for w in caught or []:  # recording them must not hide them: every warning is shown as usual
            warnings.showwarning(w.message, w.category, w.filename, w.lineno)


if __name__ == "__main__":
    sys.exit(main())
