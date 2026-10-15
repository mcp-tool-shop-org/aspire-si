"""Stage-1 trainer: LoRA SFT on the student, self-contained (torch + peft only, no TRL, no new installs).

    python train.py --data <items.jsonl> --out <round dir>
                    [--init-adapter <dir>] [--held-out a.jsonl b.jsonl ...] [--max-steps N] [--dry-run]
                    [--status <run dir>/status.json] [--ckpt-every 25]
                    [--order shuffled|listed|round-robin] [--lesson-order T01,T04,...]

- Loads the student through stage1-eval/student.py (the same loader and weights check as the evaluation).
- LoRA rank 64, alpha 128, dropout 0.05 on every linear layer; AdamW lr 1e-4, cosine with 3% warm-up,
  effective batch 16 sequences, max length 4096, gradient checkpointing, bf16.
- Loss on the assistant target only. One sequence per assistant turn (see data.py).
- The training order is a setting (--order), recorded in the manifest, and a resume under a different order is
  refused, because the order changes what is trained when:
  - shuffled (the default; the pre-registered order): one seeded, shuffled pass over all of the session's lessons;
  - listed: one lesson at a time, in --lesson-order (default: the order lessons first appear in the data files),
    each lesson's sequences shuffled within it by the seed;
  - round-robin: one sequence from each lesson in turn, in --lesson-order, each lesson's own sequences shuffled
    by the seed, until every lesson is used up.
  Every order trains every sequence once per epoch: only the sequence in time changes. A non-default order is
  a measured change: pre-register it before the run it's used in.
- Refuses the data on any schema error, a leakage hit against the held-out sets or the pre-interview, or a
  sequence over the length limits (target <= 3,000 tokens, whole sequence <= 4,096): refused, never truncated.
- Writes: adapter/, manifest.json (data sha256s, seed, settings, adapter sha256, machine line, timing),
  loss.jsonl, trainer-events.jsonl. Resumable from the last checkpoint in the round dir: the checkpoint holds the
  adapter, the optimizer and the CPU, CUDA and Python RNG states, is written to a temporary directory and renamed
  into place, and loss.jsonl is cut back to the checkpoint's step on resume, so a resumed run logs and trains
  exactly as an uninterrupted one.
- Run under run.py (the supervisor), it reads the supervisor's status.json between steps: it stops itself if the
  supervisor's heartbeat is older than 120 s, and waits while the supervisor asks for a heat pause (the paused
  time is excluded from train_seconds). It never changes what is trained: it either trains the step as planned or
  stops with one of the exit codes below.

Exit codes: 0 done; 71 VRAM self-stop (28 GB); 72 non-finite loss; 73 loss spike (3 steps over 3x the running
median); 74 disk full; 75 supervisor heartbeat lost; 76 I/O still failing after 5 tries; 77 the card grant ended
(the supervisor asked for a stop: the checkpoint is written at the last finished step first, so the run can be
resumed under a new grant). Anything else is a crash.
"""

import argparse
import errno
import faulthandler
import hashlib
import json
import math
import os
import platform
import random
import shutil
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _evaldir import EVAL_DIR  # noqa: E402

sys.path.insert(0, str(EVAL_DIR))
import data  # noqa: E402  (also puts the aspire-si-gates package on the path)
import student  # noqa: E402
from aspire_si_gates import gates_identity  # noqa: E402

LORA = {"r": 64, "lora_alpha": 128, "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]}
VRAM_CAP_GB = 28  # self-stop line; jobs are planned at or below about 26 GB
OPT = {"lr": 1e-4, "warmup_frac": 0.03, "weight_decay": 0.0, "effective_batch": 16, "max_len": 4096,
       "max_target": 3000, "epochs": 1, "seed": 0}
EXIT = {"vram": 71, "nonfinite": 72, "spike": 73, "disk_full": 74, "orphan": 75, "io": 76, "grant_end": 77}
HEARTBEAT_MAX_AGE_S = 120
SPIKE = {"factor": 3.0, "steps": 3, "window": 20}  # stop: 3 straight steps over 3x the median of the last 20
IO_BACKOFF_S = (1, 2, 4, 8, 16)  # 5 tries per operation
NEAR_CAP = 0.9  # sequences within 10% of a length cap are listed for the round's diagnostics


class StopRequested(Exception):
    """The supervisor asked for a stop (the card grant ended): checkpoint first, then exit 77."""


class Stop(Exception):
    def __init__(self, reason: str, code: int, **detail):
        super().__init__(reason)
        self.reason, self.code, self.detail = reason, code, detail


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def code_hash() -> str:
    h = hashlib.sha256()
    for f in ("train.py", "data.py", "_evaldir.py"):
        h.update((HERE / f).read_bytes())
    h.update((EVAL_DIR / "student.py").read_bytes())
    h.update(gates_identity()["sha256"].encode())  # the gates package that passed the data
    return h.hexdigest()


def machine() -> dict:
    import torch
    import transformers
    return {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cuda": torch.version.cuda, "torch": torch.__version__, "transformers": transformers.__version__,
            "python": platform.python_version()}


class Events:
    """trainer-events.jsonl: one line per operational event (I/O retry, pause, stop). run.py copies them into the
    run's actions.jsonl."""

    def __init__(self, path: Path):
        self.path = path

    def __call__(self, event: str, **kw):
        line = json.dumps({"t": round(time.time(), 3), "event": event, **kw})
        with open(self.path, "a", encoding="utf-8", newline="\n") as f:
            f.write(line + "\n")


def with_retry(fn, what: str, events, sleep=time.sleep):
    """Run fn(); on an OSError retry after 1, 2, 4, 8 and 16 s. A full disk stops at once (nothing is deleted)."""
    for i, wait in enumerate((0,) + IO_BACKOFF_S):
        if wait:
            sleep(wait)
        try:
            return fn()
        except OSError as e:
            if e.errno == errno.ENOSPC:
                free = shutil.disk_usage(Path.cwd()).free
                raise Stop("disk full", EXIT["disk_full"], what=what, free_bytes=free) from e
            events("io_retry", what=what, attempt=i + 1, error=f"{type(e).__name__}: {e}")
            last = e
    raise Stop("I/O still failing after 5 tries", EXIT["io"], what=what, error=str(last))


def write_atomic(path: Path, text: str):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def build(rows, tok, max_len, max_target):
    seqs, refused = [], []
    for r in rows:
        for s in data.sequences(r, tok):
            n, t = len(s["prompt_ids"]) + len(s["target_ids"]), len(s["target_ids"])
            if n > max_len or t > max_target:
                refused.append((r["id"], n, t))
                continue
            seqs.append({**s, "id": r["id"], "lesson": r.get("lesson")})
    if refused:
        raise SystemExit(f"items over the limits (sequence <= {max_len}, target <= {max_target} tokens; shorten "
                         f"or split them, never truncate): {refused[:10]}")
    return seqs


ORDERS = ("shuffled", "listed", "round-robin")


def lesson_sequence(seqs, lesson_order: list | None) -> list:
    """The lessons in the order to train them: --lesson-order if given (it must name every lesson in the data,
    each once, and nothing else), else the order lessons first appear in the data."""
    present = list(dict.fromkeys(s["lesson"] for s in seqs))
    if None in present:
        raise SystemExit("an item has no lesson field: a lesson order can't place it")
    if not lesson_order:
        return present
    if sorted(lesson_order) != sorted(present) or len(set(lesson_order)) != len(lesson_order):
        missing, extra = set(present) - set(lesson_order), set(lesson_order) - set(present)
        raise SystemExit(f"--lesson-order must name every lesson in the data once: missing {sorted(missing)}, "
                         f"not in the data {sorted(extra)}")
    return list(lesson_order)


def training_order(seqs, mode: str, lesson_order: list | None, rng) -> list:
    """Indices into seqs, every one exactly once, in the order to train them. `rng` is the seeded random module
    (or a random.Random); "shuffled" draws exactly as the trainer always has (one shuffle of 0..n-1)."""
    if mode not in ORDERS:
        raise SystemExit(f"--order must be one of {', '.join(ORDERS)}")
    if mode == "shuffled":
        order = list(range(len(seqs)))
        rng.shuffle(order)
        return order
    lessons = lesson_sequence(seqs, lesson_order)
    groups = {ls: [i for i, s in enumerate(seqs) if s["lesson"] == ls] for ls in lessons}
    for ls in lessons:
        rng.shuffle(groups[ls])
    if mode == "listed":
        return [i for ls in lessons for i in groups[ls]]
    order, k = [], 0  # round-robin
    while any(groups.values()):
        ls = lessons[k % len(lessons)]
        if groups[ls]:
            order.append(groups[ls].pop(0))
        k += 1
    return order


def near_caps(seqs, max_len, max_target):
    out = []
    for s in seqs:
        n, t = len(s["prompt_ids"]) + len(s["target_ids"]), len(s["target_ids"])
        if n > NEAR_CAP * max_len or t > NEAR_CAP * max_target:
            out.append({"item_id": s["id"], "prompt_tokens": len(s["prompt_ids"]), "target_tokens": t})
    return out


# ---------------------------------------------------------------- checkpoints

def find_checkpoint(out: Path):
    """The checkpoint to resume from. checkpoint/ is complete once renamed into place; checkpoint.tmp/ is an
    unfinished save (discarded); checkpoint.old/ is the previous one, kept until the new one is in place."""
    ckpt, tmp, old = out / "checkpoint", out / "checkpoint.tmp", out / "checkpoint.old"
    if tmp.exists():
        shutil.rmtree(tmp)
    if (ckpt / "state.json").exists():
        if old.exists():
            shutil.rmtree(old)
        return ckpt
    if (old / "state.json").exists():  # a crash between the two renames
        os.replace(old, ckpt)
        return ckpt
    return None


def save_checkpoint(out: Path, model, opt, state: dict, torch):
    tmp, ckpt, old = out / "checkpoint.tmp", out / "checkpoint", out / "checkpoint.old"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    model.save_pretrained(tmp / "adapter")
    torch.save(opt.state_dict(), tmp / "optimizer.pt")
    rng = {"python": random.getstate(), "torch_cpu": torch.get_rng_state(),
           "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
    torch.save(rng, tmp / "rng.pt")
    (tmp / "state.json").write_text(json.dumps(state), encoding="utf-8")  # written last: marks the save complete
    if ckpt.exists():
        os.replace(ckpt, old)
    os.replace(tmp, ckpt)
    if old.exists():
        shutil.rmtree(old)


def trim_loss_log(path: Path, step: int):
    """Keep only the lines up to `step`, so steps repeated after a crash aren't logged twice."""
    if not path.exists():
        return 0
    lines = path.read_text(encoding="utf-8").splitlines()
    keep = [x for x in lines if x.strip() and json.loads(x)["step"] <= step]
    write_atomic(path, "".join(x + "\n" for x in keep))
    return len(lines) - len(keep)


# ---------------------------------------------------------------- the supervisor channel

class Supervisor:
    """Reads run.py's status.json between steps: its heartbeat, and its heat-pause request."""

    def __init__(self, status: Path | None, events, max_age=HEARTBEAT_MAX_AGE_S, sleep=time.sleep):
        self.status, self.events, self.max_age, self.sleep = status, events, max_age, sleep

    def _read(self):
        try:
            return json.loads(self.status.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None  # mid-replace or briefly locked; the heartbeat age decides

    def check(self) -> float:
        """Returns the seconds spent paused. Raises Stop if the supervisor's heartbeat is too old."""
        if self.status is None:
            return 0.0
        paused, t0, announced = 0.0, time.time(), False
        while True:
            st = self._read()
            hb = (st or {}).get("supervisor_heartbeat")
            if hb is None or time.time() - hb > self.max_age:
                age = None if hb is None else round(time.time() - hb, 1)
                raise Stop("supervisor heartbeat lost", EXIT["orphan"], heartbeat_age_s=age)
            if st.get("stop_requested"):
                raise StopRequested(st.get("stop_reason") or "stop requested")
            if not st.get("pause"):
                if announced:
                    self.events("pause_end", paused_s=round(paused, 1))
                return paused
            if not announced:
                self.events("pause_start", reason=st.get("pause_reason"))
                announced = True
            self.sleep(2)
            paused = time.time() - t0


# ---------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--init-adapter")
    ap.add_argument("--held-out", nargs="*", default=[])
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--status", help="the supervisor's status.json (heartbeat and heat pause)")
    ap.add_argument("--ckpt-every", type=int, default=25)
    ap.add_argument("--order", choices=ORDERS, default="shuffled", help="the training order (see the docstring)")
    ap.add_argument("--lesson-order", help="comma-separated lesson ids, for --order listed or round-robin")
    ap.add_argument("--closed-form-carry", choices=("verbatim", "paraphrase", "either"),
                    help="how a closed-form written variant's keyed option carries the key point (default: the gates' "
                         "default, R&D's ruling: paraphrase)")
    ap.add_argument("--closed-form-share", type=float, help="the keyed option's least content-word share")
    ap.add_argument("--closed-form-distractor-max", type=float, help="the near-copy guard: no distractor reaches it")
    ap.add_argument("--closed-form-distractor-margin", type=float,
                    help="every distractor's share is below the keyed share minus this")
    ap.add_argument("--closed-form-span", type=int, help="the span length no variant may share with a trained turn")
    ap.add_argument("--closed-form-exempt-keyed-key-points", action="store_true", default=None,
                    help="a closed-form keyed option may repeat the item's own key points (off by default)")
    ap.add_argument("--partial-set", action="store_true",
                    help="train on part of the data (a smoke or a test): the whole-dataset freeze checks are reported "
                         "in the manifest instead of refusing; every other check still refuses")
    a = ap.parse_args(argv)
    lesson_order = [x.strip() for x in a.lesson_order.split(",") if x.strip()] if a.lesson_order else None
    if lesson_order and a.order == "shuffled":
        ap.error("--lesson-order applies to --order listed or round-robin; shuffled ignores lessons")

    files = {p: data.load(p) for p in a.data}
    rows = [r for rs in files.values() for r in rs]
    # Per item and per lesson over everything, the batch checks per batch file, and the whole-dataset freeze
    # checks (the Dataset Specialist's ruling, 2026-10-11).
    closed_form = data.closed_form_settings({k: v for k, v in (
        ("carry", a.closed_form_carry), ("share", a.closed_form_share),
        ("distractor_max", a.closed_form_distractor_max), ("distractor_margin", a.closed_form_distractor_margin),
        ("span", a.closed_form_span), ("exempt_keyed_key_points", a.closed_form_exempt_keyed_key_points))
        if v is not None})
    freeze = data.validate_training_set(files, require_freeze=not a.partial_set, closed_form=closed_form)
    held = [h for p in a.held_out for h in data.load(p)]
    from interview import QUESTIONS  # the pre-interview's 12 questions are off-limits to training
    leaks = data.leakage(rows, held, QUESTIONS)
    if leaks:
        raise SystemExit("leakage against the held-out sets:\n" + "\n".join(leaks[:30]))
    tok = student.load_tokenizer()
    seqs = build(rows, tok, OPT["max_len"], OPT["max_target"])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"code_sha256": code_hash(), "gates": gates_identity(), "student": student.identity(),
                "lora": LORA, "opt": OPT,
                "data": {p: sha256_file(Path(p)) for p in a.data},
                "held_out": {p: sha256_file(Path(p)) for p in a.held_out}, "examples": len(rows),
                "sequences": len(seqs), "target_tokens": sum(len(s["target_ids"]) for s in seqs),
                "near_length_caps": near_caps(seqs, OPT["max_len"], OPT["max_target"]),
                "order": {"mode": a.order, "seed": OPT["seed"],
                          "lessons": None if a.order == "shuffled" else lesson_sequence(seqs, lesson_order)},
                "freeze_report": freeze, "closed_form": closed_form,
                "closed_form_figures": data.closed_form_figures(rows),
                "init_adapter": a.init_adapter}
    if a.dry_run:
        print(json.dumps({k: v for k, v in manifest.items() if k != "student"}, indent=1))
        return 0
    events = Events(out / "trainer-events.jsonl")
    try:
        return train(a, seqs, manifest, out, events)
    except Stop as s:
        events("stop", reason=s.reason, code=s.code, **s.detail)
        print("stopped:", s.reason, json.dumps(s.detail), flush=True)
        return s.code


def train(a, seqs, manifest, out, events):
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model

    cuda = torch.cuda.is_available()
    sup = Supervisor(Path(a.status) if a.status else None, events)
    try:
        sup.check()  # no supervisor heartbeat, no model load
    except StopRequested as e:
        raise Stop("card grant ended", EXIT["grant_end"], step=None, checkpoint_step=None, request=str(e)) from e
    print(json.dumps(student.verify_weights()), flush=True)
    if cuda:
        # The trainer stops itself at 28 GB (the maintainer's policy): past this limit an allocation fails with a
        # clean out-of-memory error, well before the rig's watchdog line at 31.2 GB.
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, VRAM_CAP_GB * 2**30 / total))
    random.seed(OPT["seed"])
    torch.manual_seed(OPT["seed"])
    model = student.load_student()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    if a.init_adapter:
        model = PeftModel.from_pretrained(model, a.init_adapter, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", bias="none", **LORA))
    model.train()
    inner = model.get_base_model()          # the causal-LM with the LoRA layers injected
    backbone, lm_head = inner.model, inner.lm_head
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=OPT["lr"], weight_decay=OPT["weight_decay"])

    order = training_order(seqs, manifest["order"]["mode"], manifest["order"]["lessons"], random)
    accum = OPT["effective_batch"]
    total_steps = math.ceil(len(order) * OPT["epochs"] / accum)
    if a.max_steps:
        total_steps = min(total_steps, a.max_steps)
    warm = max(1, int(total_steps * OPT["warmup_frac"]))

    def lr_at(step):
        if step < warm:
            return OPT["lr"] * (step + 1) / warm
        prog = (step - warm) / max(1, total_steps - warm)
        return OPT["lr"] * 0.5 * (1 + math.cos(math.pi * prog))

    dev = model.device
    loss_path = out / "loss.jsonl"
    start_step, prior_seconds, prior_paused = 0, 0.0, 0.0
    ckpt = find_checkpoint(out)
    if ckpt is not None:
        st = json.loads((ckpt / "state.json").read_text(encoding="utf-8"))
        if st["code_sha256"] != manifest["code_sha256"] or st["data"] != manifest["data"]:
            raise SystemExit("refusing to resume: code or data differ from the checkpoint")
        if st.get("order", manifest["order"]) != manifest["order"]:
            raise SystemExit("refusing to resume: the training order differs from the checkpoint's")
        model.load_adapter(str(ckpt / "adapter"), adapter_name="default", is_trainable=True)
        opt.load_state_dict(torch.load(ckpt / "optimizer.pt", map_location=dev))
        rng = torch.load(ckpt / "rng.pt", weights_only=False)
        random.setstate(rng["python"])
        torch.set_rng_state(rng["torch_cpu"])
        if cuda and rng["torch_cuda"] is not None:
            torch.cuda.set_rng_state_all(rng["torch_cuda"])
        start_step = st["step"]
        prior_seconds = st.get("train_seconds", 0.0)
        prior_paused = st.get("paused_seconds", 0.0)
        dropped = trim_loss_log(loss_path, start_step)
        events("resumed", step=start_step, loss_lines_dropped=dropped)
        print("resumed at step", start_step, flush=True)
    else:
        dropped = trim_loss_log(loss_path, 0)  # a crash before the first checkpoint: start the log again
        if dropped:
            events("restarted", loss_lines_dropped=dropped)

    history = [json.loads(x)["loss"] for x in loss_path.read_text(encoding="utf-8").splitlines() if x.strip()] \
        if loss_path.exists() else []
    step_times = []
    stall_file = open(out / "stall-stack.txt", "a", encoding="utf-8")
    t0 = time.time()
    paused = 0.0
    over = 0
    saved_at = start_step
    for step in range(start_step, total_steps):
        faulthandler.cancel_dump_traceback_later()  # the supervisor's pause and any wait are never timed as a stall
        try:
            paused += sup.check()
        except StopRequested as e:
            if step > saved_at:  # flush: the steps finished since the last checkpoint are kept
                state = {"step": step, "code_sha256": manifest["code_sha256"], "data": manifest["data"], "order": manifest["order"],
                         "train_seconds": prior_seconds + time.time() - t0 - paused,
                         "paused_seconds": prior_paused + paused}
                with_retry(lambda: save_checkpoint(out, model, opt, state, torch), "checkpoint", events)
                events("checkpoint", step=step)
            faulthandler.cancel_dump_traceback_later()
            raise Stop("card grant ended", EXIT["grant_end"], step=step, checkpoint_step=step, request=str(e)) from e
        if len(step_times) >= 3:
            # A stalled step writes its own stack: the dump fires if this step's work (armed after the supervisor's
            # check, so pauses don't count) takes 5x the median step time.
            faulthandler.dump_traceback_later(5 * statistics.median(step_times), repeat=False, file=stall_file)
        t_step = time.time()
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        batch = [seqs[order[(step * accum + k) % len(order)]] for k in range(accum)]
        tgt_total = sum(len(s["target_ids"]) for s in batch)
        step_loss = 0.0
        seq_peaks = []
        for s in batch:
            n_p, n_t = len(s["prompt_ids"]), len(s["target_ids"])
            try:
                if cuda:
                    torch.cuda.reset_peak_memory_stats()
                ids = torch.tensor([s["prompt_ids"] + s["target_ids"]], device=dev)
                # The LM head runs only on the positions that predict the target: full-vocabulary logits over a
                # 4k-token prompt cost several GB (the first smoke tripped the VRAM watchdog on exactly that).
                hidden = backbone(input_ids=ids).last_hidden_state[:, n_p - 1:n_p - 1 + n_t]
                logits = lm_head(hidden).float()
                tgt = torch.tensor([s["target_ids"]], device=dev)
                loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1),
                                                         reduction="sum") / tgt_total
                loss.backward()
            except torch.OutOfMemoryError as e:
                peak = round(torch.cuda.max_memory_allocated() / 2**30, 2) if cuda else None
                raise Stop("VRAM self-stop (28 GB)", EXIT["vram"], step=step + 1, item_id=s["id"],
                           prompt_tokens=n_p, target_tokens=n_t, peak_gb=peak, error=str(e)[:300]) from e
            step_loss += loss.item()
            del hidden, logits, loss
            seq_peaks.append([n_p, n_t, round(torch.cuda.max_memory_allocated() / 2**30, 2) if cuda else 0.0])
        if not math.isfinite(step_loss):
            raise Stop("non-finite loss", EXIT["nonfinite"], step=step + 1, loss=str(step_loss))
        recent = history[-SPIKE["window"]:]
        if len(recent) >= SPIKE["window"] // 2 and step_loss > SPIKE["factor"] * statistics.median(recent):
            over += 1
            if over >= SPIKE["steps"]:
                raise Stop("loss spike", EXIT["spike"], step=step + 1, loss=round(step_loss, 5),
                           running_median=round(statistics.median(recent), 5))
        else:
            over = 0
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        # The step's work ends here: the loss log's and the checkpoint's retries (up to 31 s of backoff) are I/O with
        # their own retry and disk-full rules, never timed against 5x the median step (R&D on #125).
        faulthandler.cancel_dump_traceback_later()
        history.append(step_loss)
        step_times.append(time.time() - t_step)
        rec = {"step": step + 1, "of": total_steps, "loss": round(step_loss, 5), "lr": lr_at(step),
               "target_tokens": tgt_total, "step_s": round(step_times[-1], 3),
               "elapsed_s": round(time.time() - t0 - paused, 1),
               "peak_vram_gb": max(x[2] for x in seq_peaks),
               "seq_peaks_prompt_target_gb": sorted({tuple(x) for x in seq_peaks})}
        line = json.dumps(rec) + "\n"

        def _log():
            with open(loss_path, "a", encoding="utf-8", newline="\n") as f:
                f.write(line)
        with_retry(_log, "loss.jsonl", events)
        print(json.dumps(rec), flush=True)
        if (step + 1) % a.ckpt_every == 0 or step + 1 == total_steps:
            state = {"step": step + 1, "code_sha256": manifest["code_sha256"], "data": manifest["data"], "order": manifest["order"],
                     "train_seconds": prior_seconds + time.time() - t0 - paused,
                     "paused_seconds": prior_paused + paused}
            with_retry(lambda: save_checkpoint(out, model, opt, state, torch), "checkpoint", events)
            saved_at = step + 1
            events("checkpoint", step=step + 1)
    faulthandler.cancel_dump_traceback_later()
    elapsed = prior_seconds + time.time() - t0 - paused  # cumulative across resumes, pauses excluded
    with_retry(lambda: model.save_pretrained(out / "adapter"), "adapter", events)
    adapter_file = out / "adapter" / "adapter_model.safetensors"
    manifest.update({"steps": total_steps, "machine": machine(), "train_seconds": round(elapsed, 1),
                     "paused_seconds": round(prior_paused + paused, 1),
                     "gpu_minutes": round(elapsed / 60, 1),
                     "examples_per_minute": round(min(total_steps * accum, len(seqs) * OPT["epochs"])
                                                  / max(1e-9, elapsed / 60), 2),
                     "peak_vram_gb": max(json.loads(x)["peak_vram_gb"]
                                         for x in loss_path.read_text(encoding="utf-8").splitlines() if x.strip()),
                     "adapter_sha256": sha256_file(adapter_file) if adapter_file.exists() else None})
    with_retry(lambda: write_atomic(out / "manifest.json", json.dumps(manifest, indent=1)), "manifest", events)
    print("done", json.dumps({k: manifest[k] for k in ("steps", "train_seconds", "peak_vram_gb", "adapter_sha256")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
