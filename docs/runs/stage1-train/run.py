"""Stage 1 runner: supervises one training run of train.py (one round, arm and seed). Plain code, not a model.

    python run.py --round 1 --arm ROLE --seed 0 --grant-id G-... --grant-end 2026-10-12T18:00:00Z \
                  --planned-peak-gb 24 --planned-minutes 90 -- --data items.jsonl [other train.py arguments]

It writes, in outputs/stage1-train/<round>/<arm>/<seed>/:
- status.json, atomically, every second: the live state the Publisher reads (fields below), and the channel to
  train.py (the supervisor heartbeat, and the heat-pause request);
- actions.jsonl: one line per automatic action (time, trigger, action, count against its cap, detail);
- diagnostics.json at the end, plus its row in receipts/diagnostics.jsonl (schema stage1-receipt-v1);
- train/: train.py's own output (adapter, checkpoints, loss.jsonl, manifest.json).

**The rule.** Operational actions are automatic, each with a named trigger, a cap and an actions.jsonl line.
Anything that changes what is measured (data, learning rate or schedule, LoRA, sequence order, the ledger's rules,
the evaluation) is never touched: the run stops and a person decides. The automatic actions:
1. A crash or kill not caused by memory: resume from the last checkpoint after 60 s, at most 3 times a run. Each
   resume first needs the rig watchdog alive (heartbeat under 60 s old), the grant's planned end not passed,
   and the card's used memory at or below the baseline measured at the start plus 1 GB. A failed precondition
   stops the run.
2. A transient I/O error in train.py: retried after 1, 2, 4, 8 and 16 s (train.py does it and logs each try).
   A full disk stops the run and nothing is deleted.
3. VRAM over 28 GB (train.py's self-stop) or a watchdog kill: stop. Never resumed: it's a sizing defect.
4. GPU at or above 85 C for 60 s: pause between steps until it is below 75 C. At most 30 minutes of pause in a
   run, then stop. Paused time is excluded from train_seconds; wall time is reported beside it.
5. A stall (no step within 5x the median step time; before the 3rd step, the startup allowance): train.py
   dumps its own stack, then the run stops. Never restarted by itself.
6. The card grant's planned end passes: train.py is asked to stop, writes its checkpoint at the last finished
   step, and exits; the run stops and is never resumed by itself. Extending a grant is the Publisher's call, made
   before the end, never the runner's.
Also stopped, never acted on: a non-finite loss, a loss spike (3 straight steps over 3x the running median).
There is no rest or cooldown wait before or after a run: rest between runs is the Publisher's scheduling. The
only waits are the heat pause (a temperature guard) and the 60 s before a crash resume.
The device state at the start (GPU temperature, SM clock, power, memory used) is recorded in status.json and the
diagnostics receipt.

**Ownership.** On Windows train.py runs inside a Job Object with kill-on-close, so if this process dies for any
reason, the OS ends train.py too. Second guard: train.py stops itself when status.json's heartbeat is older than
120 s.

Nothing here touches the GPU except reading nvidia-smi. The card itself is granted by the Publisher: the grant
id and its planned end are required arguments, so a run can't start without them.
"""

import argparse
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import receipts  # noqa: E402

ARMS = ("ROLE", "NO-ROLE")
REPO = Path(__file__).resolve().parents[3]
OUT_ROOT = REPO / "outputs" / "stage1-train"
STUCK_RULE = HERE.parent / "stage1-stuck-rule.json"
CAPS = {"resumes": 3, "pause_s": 30 * 60}
HEAT = {"pause_c": 85, "hold_s": 60, "resume_c": 75}
STALL_FACTOR = 5
WATCHDOG_MAX_AGE_S = 60
CARD_SLACK_MIB = 1024
EXIT_REASON = {71: "VRAM self-stop (28 GB)", 72: "non-finite loss", 73: "loss spike", 74: "disk full",
               75: "supervisor heartbeat lost", 76: "I/O still failing after 5 tries", 77: "card grant ended"}


def now_iso(t=None) -> str:
    return datetime.fromtimestamp(time.time() if t is None else t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso(s: str) -> float:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def write_atomic(path: Path, obj):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8", newline="\n")
    for i in range(5):  # a reader can hold the target open for a moment on Windows
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.05 * (i + 1))
    os.replace(tmp, path)


# ---------------------------------------------------------------- the rig

def nvidia_smi() -> dict | None:
    """Temperature (C), power (W), SM clock (MHz) and used memory (MiB) of GPU 0, or None if nvidia-smi can't
    answer."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,power.draw,clocks.sm,memory.used",
                              "--format=csv,noheader,nounits", "-i", "0"], capture_output=True, text=True,
                             timeout=10).stdout
        temp, power, sm, mem = (x.strip() for x in out.strip().splitlines()[0].split(","))
        return {"temp_c": float(temp), "power_w": float(power), "sm_clock_mhz": float(sm),
                "mem_used_mib": float(mem)}
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def file_probe(path: Path):
    """For the runner's tests: the 'GPU' is a JSON file the test writes."""
    def probe():
        try:
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
    return probe


class Watchdog:
    """The rig's VRAM watchdog: alive when its heartbeat file is fresh; a kill leaves _watchdog_TRIPPED and a
    KILL line in _watchdog_KILL.log."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def heartbeat_age(self) -> float | None:
        hb = self.root / "_watchdog_HEARTBEAT"
        return time.time() - hb.stat().st_mtime if hb.exists() else None

    def killed_since(self, t: float) -> bool:
        trip = self.root / "_watchdog_TRIPPED"
        if trip.exists() and trip.stat().st_mtime >= t:
            return True
        log = self.root / "_watchdog_KILL.log"
        if log.exists() and log.stat().st_mtime >= t:
            for line in log.read_text(encoding="utf-8", errors="replace").splitlines()[-50:]:
                try:
                    when = datetime.strptime(line[:19], "%Y-%m-%d %H:%M:%S").timestamp()
                except ValueError:
                    continue
                if when >= t and " KILL " in f" {line[19:]} ":
                    return True
        return False


class Job:
    """train.py's process. run.py has already joined its own kill-on-close Job Object (own_children), so train.py
    and the interpreter a venv launcher starts are inside it from birth, and the OS ends them if run.py dies for
    any reason. kill() ends train.py's process tree only."""

    def __init__(self, cmd: list, env: dict, log: Path):
        self.log = open(log, "a", encoding="utf-8")
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        self.proc = subprocess.Popen(cmd, env=env, stdout=self.log, stderr=subprocess.STDOUT, creationflags=flags)

    def poll(self):
        return self.proc.poll()

    def kill(self):
        if self.proc.poll() is None:
            if os.name == "nt":
                subprocess.run(["taskkill", "/T", "/F", "/PID", str(self.proc.pid)], capture_output=True)
            else:
                self.proc.kill()
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass
        self.close()

    def close(self):
        if not self.log.closed:
            self.log.close()


_JOB = None


def own_children():
    """Put this process in a Job Object with kill-on-close, once. Its children are born inside it, and when this
    process ends (a crash, a hard kill) its handle closes and Windows ends every process left in the job. The
    handle is held for the life of the process and never closed by hand: closing it would end run.py too."""
    global _JOB
    if os.name != "nt" or _JOB is not None:
        return _JOB
    import ctypes
    from ctypes import wintypes

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in ("ReadOperationCount", "WriteOperationCount",
                                                       "OtherOperationCount", "ReadTransferCount",
                                                       "WriteTransferCount", "OtherTransferCount")]

    class BASIC(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class EXTENDED(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", BASIC), ("IoInfo", IO_COUNTERS),
                    ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    job = k32.CreateJobObjectW(None, None)
    info = EXTENDED()
    info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not k32.SetInformationJobObject(wintypes.HANDLE(job), 9, ctypes.byref(info), ctypes.sizeof(info)):
        raise OSError(ctypes.get_last_error(), "SetInformationJobObject failed")
    if not k32.AssignProcessToJobObject(wintypes.HANDLE(job), wintypes.HANDLE(k32.GetCurrentProcess())):
        raise OSError(ctypes.get_last_error(), "AssignProcessToJobObject failed")
    _JOB = job
    return job


# ---------------------------------------------------------------- the runner

class Runner:
    def __init__(self, a, probe, watchdog: Watchdog, trainer: list, sleep=time.sleep):
        self.a, self.probe, self.watchdog, self.trainer, self.sleep = a, probe, watchdog, trainer, sleep
        self.dir = Path(a.out_root) / str(a.round) / a.arm / str(a.seed)
        self.train_dir = self.dir / "train"
        self.run_id = f"r{a.round}-s{a.seed}-{a.arm}"
        self.counts = {"resumes": 0, "pause_s": 0.0}
        self.actions: list = []
        self.status: dict = {}
        self.steps: list = []          # loss.jsonl records seen
        self.events_seen = 0
        self.ema = None

    # -- records

    def act(self, trigger: str, action: str, count=None, cap=None, **detail):
        rec = {"t": now_iso(), "trigger": trigger, "action": action, "count": count, "cap": cap, **detail}
        self.actions.append(rec)
        with open(self.dir / "actions.jsonl", "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(rec) + "\n")

    def write_status(self, **kw):
        self.status.update(kw)
        self.status["supervisor_heartbeat"] = time.time()
        self.status["updated_at"] = now_iso()
        self.status["actions"] = self.actions[-50:]
        write_atomic(self.dir / "status.json", self.status)

    # -- train.py's files

    def read_progress(self):
        loss = self.train_dir / "loss.jsonl"
        if loss.exists():
            lines = [x for x in loss.read_text(encoding="utf-8").splitlines() if x.strip()]
            recs = []
            for x in lines:
                try:
                    recs.append(json.loads(x))
                except ValueError:
                    break  # a line being written
            if len(recs) < len(self.steps):  # train.py cut the log back to its checkpoint on resume
                self.steps = self.steps[:len(recs)]
            for r in recs[len(self.steps):]:
                self.steps.append(r)
                self.ema = r["loss"] if self.ema is None else 0.9 * self.ema + 0.1 * r["loss"]
        ev = self.train_dir / "trainer-events.jsonl"
        if ev.exists():
            lines = [x for x in ev.read_text(encoding="utf-8").splitlines() if x.strip()]
            for x in lines[self.events_seen:]:
                try:
                    e = json.loads(x)
                except ValueError:
                    break
                self.events_seen += 1
                if e["event"] == "io_retry":
                    self.act("transient I/O error", "retry", count=e["attempt"], cap=5, what=e["what"],
                             error=e["error"])
                elif e["event"] in ("resumed", "restarted", "stop"):
                    self.act(f"trainer: {e['event']}", "recorded", **{k: v for k, v in e.items()
                                                                       if k not in ("t", "event")})

    def last_stop_event(self):
        ev = self.train_dir / "trainer-events.jsonl"
        if not ev.exists():
            return None
        stops = [json.loads(x) for x in ev.read_text(encoding="utf-8").splitlines()
                 if x.strip() and '"event": "stop"' in x]
        return stops[-1] if stops else None

    # -- preconditions

    def preconditions(self, first: bool) -> str | None:
        age = self.watchdog.heartbeat_age()
        if age is None or age > WATCHDOG_MAX_AGE_S:
            return f"the rig watchdog isn't alive (heartbeat age {None if age is None else round(age)} s)"
        if time.time() >= parse_iso(self.a.grant_end):
            return "the grant's planned end has passed"
        g = self.probe()
        if g is None:
            return "nvidia-smi didn't answer"
        if first:
            self.status["card_baseline_mib"] = g["mem_used_mib"]
            self.status["device_at_start"] = {"at": now_iso(), **{k: g.get(k) for k in
                                              ("temp_c", "sm_clock_mhz", "power_w", "mem_used_mib")}}
        elif g["mem_used_mib"] > self.status["card_baseline_mib"] + CARD_SLACK_MIB:
            return (f"the card isn't empty: {g['mem_used_mib']:.0f} MiB used, baseline "
                    f"{self.status['card_baseline_mib']:.0f} + {CARD_SLACK_MIB}")
        return None

    # -- one attempt of train.py

    def attempt(self) -> tuple:
        """Run train.py until it exits or a stop; returns (state, reason)."""
        cmd = self.trainer + ["--out", str(self.train_dir), "--status", str(self.dir / "status.json")] \
            + self.a.train_args
        env = dict(os.environ)
        job = Job(cmd, env, self.dir / "train.log")
        self.write_status(state="running", pid=job.proc.pid)
        started, last_step_at, last_probe = time.time(), time.time(), 0.0
        hot_since, pause_started = None, None
        n_steps = len(self.steps)
        grant_end = parse_iso(self.a.grant_end)
        new_steps = 0  # steps this attempt has completed (a resumed trainer reloads the model first)
        try:
            while True:
                code = job.poll()
                self.read_progress()
                t = time.time()
                if len(self.steps) != n_steps:
                    new_steps += max(0, len(self.steps) - n_steps)
                    n_steps, last_step_at = len(self.steps), t
                # the grant's end: ask train.py to checkpoint and stop (once)
                if t >= grant_end and code is None and not self.status.get("stop_requested"):
                    self.act("the card grant's planned end passed", "stop at the next checkpoint",
                             grant_end=self.a.grant_end)
                    self.write_status(stop_requested=True, stop_reason="card grant ended")
                # heat
                if t - last_probe >= self.a.probe_every:
                    last_probe = t
                    g = self.probe()
                    if g:
                        self.status.update(gpu_temp_c=g.get("temp_c"), gpu_power_w=g.get("power_w"),
                                           gpu_mem_used_mib=g.get("mem_used_mib"))
                        temp = g.get("temp_c") or 0
                        if pause_started is None:
                            hot_since = (hot_since or t) if temp >= HEAT["pause_c"] else None
                            if hot_since is not None and t - hot_since >= HEAT["hold_s"]:
                                pause_started = t
                                self.act(f"GPU at or above {HEAT['pause_c']} C for {HEAT['hold_s']} s",
                                         "pause between steps", count=round(self.counts["pause_s"]),
                                         cap=CAPS["pause_s"], temp_c=temp)
                                self.write_status(pause=True, pause_reason=f"GPU {temp:.0f} C", state="paused")
                        elif temp < HEAT["resume_c"]:
                            self.counts["pause_s"] += t - pause_started
                            last_step_at += t - pause_started  # paused time isn't stall time
                            pause_started, hot_since = None, None
                            self.act(f"GPU below {HEAT['resume_c']} C", "resume", count=round(self.counts["pause_s"]),
                                     cap=CAPS["pause_s"], temp_c=temp)
                            self.write_status(pause=False, pause_reason=None, state="running")
                if pause_started is not None and self.counts["pause_s"] + (t - pause_started) > CAPS["pause_s"]:
                    self.counts["pause_s"] += t - pause_started
                    self.act("heat pauses over 30 minutes", "stop", count=round(self.counts["pause_s"]),
                             cap=CAPS["pause_s"])
                    job.kill()
                    return "stopped", "heat: pause cap reached"
                # stall
                if code is None and pause_started is None:
                    times = [r["step_s"] for r in self.steps[-min(new_steps, 50):] if "step_s" in r]                         if new_steps else []
                    if len(times) >= 3:
                        limit = STALL_FACTOR * statistics.median(times) + self.a.stall_slack
                    else:
                        limit = self.a.startup_timeout
                    if t - last_step_at > limit:
                        self.sleep(3)  # let train.py's own stack dump land
                        stack = self.train_dir / "stall-stack.txt"
                        self.act("no step within the stall limit", "stop", limit_s=round(limit, 1),
                                 stack=stack.read_text(encoding="utf-8", errors="replace")[-4000:]
                                 if stack.exists() else None)
                        job.kill()
                        return "stopped", "stall"
                self.write_live(t, started, pause_started)
                if code is not None:
                    job.close()
                    return self.exit_state(code, started)
                self.sleep(self.a.poll_every)
        finally:
            if job.poll() is None:
                job.kill()

    def exit_state(self, code: int, started: float) -> tuple:
        if code == 0:
            return "done", None
        if code in EXIT_REASON:
            stop = self.last_stop_event() or {}
            self.status["stop_detail"] = {k: v for k, v in stop.items() if k not in ("t", "event")}
            return "stopped", EXIT_REASON[code]
        if self.watchdog.killed_since(started):
            self.act("the rig watchdog killed train.py", "stop (never resumed)", exit_code=code)
            return "stopped", "watchdog kill"
        return "crashed", f"exit code {code}"

    def write_live(self, t, started, pause_started):
        last = self.steps[-1] if self.steps else {}
        times = [r["step_s"] for r in self.steps[-50:] if "step_s" in r]
        med = statistics.median(times) if times else None
        total = last.get("of")
        paused_now = (t - pause_started) if pause_started else 0.0
        self.write_status(
            step=last.get("step", 0), total_steps=total, loss_last=last.get("loss"),
            loss_ema20=None if self.ema is None else round(self.ema, 5), lr=last.get("lr"),
            target_tokens_per_s=round(last["target_tokens"] / last["step_s"], 1)
            if last.get("step_s") else None,
            vram_step_peak_gb=last.get("peak_vram_gb"),
            vram_run_peak_gb=max((r.get("peak_vram_gb", 0) for r in self.steps), default=None),
            eta_s=round(med * (total - last["step"])) if med and total else None,
            wall_s=round(t - self.started, 1), paused_s=round(self.counts["pause_s"] + paused_now, 1),
            resumes=self.counts["resumes"])

    # -- the whole run

    def run(self) -> int:
        a = self.a
        self.dir.mkdir(parents=True, exist_ok=True)
        if (self.dir / "diagnostics.json").exists():
            raise SystemExit(f"{self.dir} already holds a finished run")
        self.started = time.time()
        import train
        from aspire_si_gates import gates_identity
        data_files = a.train_args[a.train_args.index("--data") + 1:] if "--data" in a.train_args else []
        data_files = [x for x in _until_flag(data_files)]
        self.status = {
            "run_id": self.run_id, "round": a.round, "arm": a.arm, "seed": a.seed, "state": "starting",
            "grant_id": a.grant_id, "grant_end": a.grant_end, "planned_peak_gb": a.planned_peak_gb,
            "planned_minutes": a.planned_minutes, "started_at": now_iso(self.started),
            "code_sha256": train.code_hash(), "runner_sha256": sha256_file(Path(__file__)),
            "data_sha256": {Path(p).name: sha256_file(Path(p)) for p in data_files},
            "gates": gates_identity(), "stuck_rule_sha256": sha256_file(STUCK_RULE),
            "ledger": None, "pause": False, "pause_reason": None, "stop_reason": None,
        }
        why = self.preconditions(first=True)
        if why:
            self.act("start precondition", "refused to start", reason=why)
            self.write_status(state="stopped", stop_reason=f"not started: {why}")
            return 2
        self.act("start", "train.py started", grant_id=a.grant_id)
        while True:
            state, reason = self.attempt()
            if state != "crashed":
                break
            n = self.counts["resumes"] + 1
            if n > CAPS["resumes"]:
                self.act(f"crash ({reason})", "stop: resume cap reached", count=n - 1, cap=CAPS["resumes"])
                state, reason = "stopped", f"crash after {CAPS['resumes']} resumes ({reason})"
                break
            self.write_status(state="resuming", stop_reason=None)
            self.sleep(a.resume_delay)
            why = self.preconditions(first=False)
            if why:
                self.act(f"crash ({reason})", "stop: resume precondition failed", count=n - 1,
                         cap=CAPS["resumes"], reason=why)
                state, reason = "stopped", f"crash, not resumed: {why}"
                break
            self.counts["resumes"] = n
            self.act(f"crash ({reason})", "resume from the last checkpoint", count=n, cap=CAPS["resumes"])
        self.read_progress()
        self.write_status(state=state, stop_reason=reason, finished_at=now_iso())
        self.finish(state, reason)
        return 0 if state == "done" else 1

    # -- the end of the run

    def finish(self, state, reason):
        manifest_p = self.train_dir / "manifest.json"
        manifest = json.loads(manifest_p.read_text(encoding="utf-8")) if manifest_p.exists() else {}
        losses = [r["loss"] for r in self.steps]
        spikes = []
        for i, r in enumerate(self.steps):
            window = losses[max(0, i - 20):i]
            if len(window) >= 10 and r["loss"] > 3 * statistics.median(window):
                spikes.append({"step": r["step"], "loss": r["loss"],
                               "running_median": round(statistics.median(window), 5)})
        near = manifest.get("near_length_caps", [])
        proposals = []
        if near:
            proposals.append({"text": f"{len(near)} training sequence(s) are within 10% of the length caps "
                                      "(4,096 tokens, or 3,000 for the target); shorten or split them before "
                                      "they grow past the cap", "status": "needs approval: Publisher and maintainer"})
        if spikes:
            proposals.append({"text": f"the loss spiked at {len(spikes)} step(s) without stopping the run; check "
                                      "the batches at those steps", "status": "needs approval: Publisher and maintainer"})
        if self.counts["resumes"] or self.counts["pause_s"]:
            proposals.append({"text": f"{self.counts['resumes']} resume(s) and {self.counts['pause_s'] / 60:.1f} "
                                      "minutes of heat pause in this run; check cooling and stability before the "
                                      "next grant", "status": "needs approval: Publisher and maintainer"})
        diag = {
            "run_id": self.run_id, "state": state, "stop_reason": reason,
            "steps": self.steps[-1]["step"] if self.steps else 0,
            "train_seconds": manifest.get("train_seconds"), "wall_seconds": round(time.time() - self.started, 1),
            "paused_seconds": round(self.counts["pause_s"], 1), "resumes": self.counts["resumes"],
            "peak_vram_gb": max((r.get("peak_vram_gb", 0) for r in self.steps), default=None),
            "planned_peak_gb": self.a.planned_peak_gb, "grant_id": self.a.grant_id,
            "device_at_start": self.status.get("device_at_start"),
            "order": manifest.get("order"),
            "quiz": None, "stuck_by_cell": [], "hedge_summary": None,
            "near_length_caps": near, "loss_spikes": spikes, "backup": self.backup(),
            "proposals": proposals,
        }
        write_atomic(self.dir / "diagnostics.json", diag)
        fw = receipts.Firewall.load(sealed=self.a.sealed, **({"sealed_sha256": self.a.sealed_sha256}
                                                              if self.a.sealed_sha256 else {}))
        w = receipts.ReceiptWriter(self.a.out_root, self.a.round, self.a.arm, self.a.seed, self.run_id,
                                   manifest.get("adapter_sha256"), self.status["gates"]["sha256"],
                                   self.status["stuck_rule_sha256"], fw)
        w.write("diagnostics", {k: v for k, v in diag.items() if k not in ("run_id",)})

    def backup(self) -> dict:
        """Copy the run's small records and the adapter to the backup drive, if it's mounted. Never fatal."""
        root = Path(self.a.backup_root) if self.a.backup_root else None
        if root is None:
            return {"status": "skipped", "reason": "no backup root set"}
        if not Path(root.anchor or root).exists():
            return {"status": "skipped", "reason": "the archive drive isn't mounted"}
        rel = f"stage1-train/{self.a.round}/{self.a.arm}/{self.a.seed}"
        dest = root / rel
        try:
            dest.mkdir(parents=True, exist_ok=True)
            for name in ("status.json", "actions.jsonl"):
                if (self.dir / name).exists():
                    shutil.copy2(self.dir / name, dest / name)
            for name in ("manifest.json", "loss.jsonl", "trainer-events.jsonl"):
                if (self.train_dir / name).exists():
                    shutil.copy2(self.train_dir / name, dest / name)
            if (self.train_dir / "adapter").exists():
                shutil.copytree(self.train_dir / "adapter", dest / "adapter", dirs_exist_ok=True)
            return {"status": "copied", "to": f"external-archive:{rel}"}  # receipts carry no absolute paths
        except OSError as e:
            return {"status": "failed", "reason": f"{type(e).__name__} (errno {e.errno})"}


def _until_flag(xs):
    for x in xs:
        if x.startswith("--"):
            return
        yield x


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--grant-id", required=True, help="the Publisher's card grant")
    ap.add_argument("--grant-end", required=True, help="the grant's planned end, ISO 8601 UTC")
    ap.add_argument("--planned-peak-gb", type=float, required=True)
    ap.add_argument("--planned-minutes", type=float, required=True)
    ap.add_argument("--out-root", default=str(OUT_ROOT))
    ap.add_argument("--sealed", default=str(receipts.SEALED_FILE), help="the frozen sealed file, for the firewall")
    ap.add_argument("--sealed-sha256", help=argparse.SUPPRESS)  # tests plant their own sealed file
    ap.add_argument("--watchdog-dir", default=os.environ.get("ASPIRE_WATCHDOG_DIR"),
                    help="the rig watchdog's directory (its heartbeat file); default $ASPIRE_WATCHDOG_DIR")
    ap.add_argument("--backup-root", default=os.environ.get("ASPIRE_BACKUP_ROOT"),
                    help="where the run is copied at the end, if that drive is mounted; default $ASPIRE_BACKUP_ROOT")
    ap.add_argument("--trainer", default=str(HERE / "train.py"), help="train.py (tests use tiny_train.py)")
    ap.add_argument("--gpu-probe", default="nvidia-smi", help="nvidia-smi, or file:<json> for the tests")
    ap.add_argument("--resume-delay", type=float, default=60)
    ap.add_argument("--startup-timeout", type=float, default=30 * 60, help="allowance before the 3rd step")
    ap.add_argument("--stall-slack", type=float, default=60, help="seconds added to 5x the median step time")
    ap.add_argument("--probe-every", type=float, default=10)
    ap.add_argument("--poll-every", type=float, default=1)
    ap.add_argument("train_args", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    if a.train_args[:1] == ["--"]:
        a.train_args = a.train_args[1:]
    if not a.watchdog_dir:
        ap.error("--watchdog-dir (or ASPIRE_WATCHDOG_DIR) is required: a run never starts without the rig watchdog")
    if "--out" in a.train_args or "--status" in a.train_args:
        ap.error("run.py sets train.py's --out and --status itself")
    parse_iso(a.grant_end)
    return a


def main(argv=None):
    a = parse(argv)
    probe = nvidia_smi if a.gpu_probe == "nvidia-smi" else file_probe(Path(a.gpu_probe.split(":", 1)[1]))
    trainer = [sys.executable, a.trainer]
    own_children()
    return Runner(a, probe, Watchdog(Path(a.watchdog_dir)), trainer).run()


if __name__ == "__main__":
    sys.exit(main())
