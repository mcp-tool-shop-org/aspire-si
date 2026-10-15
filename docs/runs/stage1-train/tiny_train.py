"""Test helper: train.py on a tiny random Qwen3, on the CPU, with the student's real tokenizer.

    python tiny_train.py <train.py arguments>

The runner's tests start this in place of train.py. The model has the student's architecture and vocabulary
but 2 layers of width 64, so a step takes about a second on the CPU. The 5090 is hidden (CUDA_VISIBLE_DEVICES is
emptied before torch loads) and the weights check is skipped: there are no student weights here. Everything
else is train.py, unchanged: the data gates, the sequences, the loop, the checkpoints and the exit codes.

Fault injection, for the runner's tests only (environment variables):
- TINY_CRASH_AT_STEP=N: a hard crash (os._exit 1) as step N starts, once (a marker file in --out remembers);
- TINY_OOM_AT_STEP=N: an out-of-memory error at step N;
- TINY_HANG_AT_STEP=N: step N never finishes (it hangs inside the forward pass, in the step's own work, where a
  real stall happens: the stall dump times the work, not the supervisor's check before it);
- TINY_IO_FAIL=N: the first N loss-log writes raise a transient OSError.
- TINY_SLOW_S=X: every step waits X seconds first (so a test can land an event mid-run).
- TINY_SLOW_LOG_S=X: every loss-log write takes X seconds (slow I/O after the step's work).
"""

import os
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch  # noqa: E402

import train  # noqa: E402
import student  # noqa: E402  (train put the eval directory on the path)

TINY = {"hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2, "num_attention_heads": 4,
        "num_key_value_heads": 2, "head_dim": 16}


def tiny_student(device: str = "cpu"):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(1234)  # the same tiny weights in every process
    cfg = Qwen3Config(vocab_size=len(student.load_tokenizer()), max_position_embeddings=8192, **TINY)
    m = Qwen3ForCausalLM(cfg).to(torch.float32)
    m.model.register_forward_pre_hook(_oom_hook)
    return m


def out_dir(argv) -> Path:
    return Path(argv[argv.index("--out") + 1])


def install_faults(argv):
    out = out_dir(argv)
    crash, oom, hang = (int(os.environ.get(k, 0)) for k in ("TINY_CRASH_AT_STEP", "TINY_OOM_AT_STEP",
                                                             "TINY_HANG_AT_STEP"))
    marker = out / ".tiny-crashed"
    real_check = train.Supervisor.check
    step = {"n": None}
    start = resumed_from(out)
    slow = float(os.environ.get("TINY_SLOW_S", 0))

    def check(self):
        paused = real_check(self)
        # Supervisor.check runs once before the model loads (k = 0), then at the start of every step (k = 1, 2, ...),
        # so the step about to run is start + k, counted from 1.
        step["n"] = 0 if step["n"] is None else step["n"] + 1
        n = start + step["n"] if step["n"] else None
        if crash and n == crash and not marker.exists():
            marker.write_text("1")
            os._exit(1)
        FAULT["oom"] = bool(oom and n == oom)  # raised inside the forward pass, where a real one happens
        FAULT["hang"] = bool(hang and n == hang)  # likewise: the step's work hangs, not the check before it
        if n and slow:
            time.sleep(slow)
        return paused

    train.Supervisor.check = check
    fail = {"left": int(os.environ.get("TINY_IO_FAIL", 0))}
    slow_log = float(os.environ.get("TINY_SLOW_LOG_S", 0))
    real_retry = train.with_retry

    def with_retry(fn, what, events, sleep=time.sleep):
        def flaky():
            if what == "loss.jsonl" and slow_log:
                time.sleep(slow_log)
            if what == "loss.jsonl" and fail["left"] > 0:
                fail["left"] -= 1
                raise OSError(5, "injected transient I/O error")
            return fn()
        return real_retry(flaky, what, events, sleep=lambda s: None)  # no real waiting in tests

    train.with_retry = with_retry


def resumed_from(out: Path) -> int:
    """The step the run resumes from (0 for a fresh start), read before train.py touches the checkpoint."""
    import json
    st = out / "checkpoint" / "state.json"
    return json.loads(st.read_text(encoding="utf-8"))["step"] if st.exists() else 0


FAULT = {"oom": False, "hang": False}


def _oom_hook(module, args):
    if FAULT["oom"]:
        raise torch.OutOfMemoryError("injected: CUDA out of memory")
    while FAULT["hang"]:
        time.sleep(1)


def main(argv):
    student.verify_weights = lambda d=None: {"weights": "skipped (tiny test model)"}
    student.load_student = tiny_student
    student.identity = lambda: {"repo": "tiny-test", "revision": None}
    install_faults(argv)
    # The tests train on a few files, never the whole dataset, so the freeze checks are reported, not enforced.
    return train.main(argv + ["--partial-set"])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
