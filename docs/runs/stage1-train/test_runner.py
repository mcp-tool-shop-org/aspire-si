"""The runner's automatic actions, tested on the CPU with a tiny model (tiny_train.py). No GPU is used: the GPU
probe is a JSON file the test writes, the rig watchdog is a heartbeat file in a temporary directory, and
CUDA_VISIBLE_DEVICES is emptied in every trainer process.

Each test runs the real run.py and train.py; only the model is tiny and the faults are injected.
"""

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run  # noqa: E402

DATA = [str(HERE.parent / "stage1-training-data" / f) for f in
        ("pilot-t04.jsonl", "batch2-t01.jsonl", "batch2-t14.jsonl", "batch2-t16.jsonl")]
TINY = str(HERE / "tiny_train.py")
PY = sys.executable

pytestmark = pytest.mark.skipif(os.name != "nt", reason="the runner's process ownership is the Windows Job Object")


def tokenizer_available():
    """The tiny trainer uses the student's real tokenizer, from the local model cache."""
    sys.path.insert(0, str(HERE.parent / "stage1-eval"))
    try:
        import student
        return (student.snapshot_dir() / "tokenizer.json").exists()
    except BaseException:  # snapshot_dir() exits when HF_HOME isn't set
        return False


needs_tokenizer = pytest.mark.skipif(not tokenizer_available(), reason="the student's tokenizer isn't cached here")


@pytest.fixture
def rig(tmp_path):
    """A watchdog heartbeat kept fresh, a planted sealed file, and a GPU probe file."""
    wd = tmp_path / "watchdog"
    wd.mkdir()
    hb = wd / "_watchdog_HEARTBEAT"
    stop = threading.Event()

    def beat():
        while not stop.is_set():
            hb.write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))
            stop.wait(2)
    threading.Thread(target=beat, daemon=True).start()
    sealed = tmp_path / "sealed.jsonl"
    sealed.write_text(json.dumps({"id": "sx-planted0001", "material": "the planted sealed line says the north "
                                  "gate opens at seven every morning except holidays"}) + "\n", encoding="utf-8")
    gpu = tmp_path / "gpu.json"
    gpu.write_text(json.dumps({"temp_c": 50, "power_w": 100, "mem_used_mib": 1000}))
    yield {"wd": wd, "sealed": sealed, "gpu": gpu, "root": tmp_path / "out", "stop": stop}
    stop.set()


def runner_args(rig, *extra, steps=8, ckpt=2, grant_end="2099-01-01T00:00:00Z", **kw):
    import hashlib
    args = ["--round", "1", "--arm", "ROLE", "--seed", "0", "--grant-id", "G-test", "--grant-end", grant_end,
            "--planned-peak-gb", "1", "--planned-minutes", "5", "--out-root", str(rig["root"]),
            "--sealed", str(rig["sealed"]), "--sealed-sha256", hashlib.sha256(rig["sealed"].read_bytes()).hexdigest(),
            "--watchdog-dir", str(rig["wd"]), "--backup-root", unmounted_drive() + "/aspire-si-backup",
            "--trainer", TINY, "--gpu-probe", f"file:{rig['gpu']}", "--resume-delay", "0",
            "--startup-timeout", str(kw.get("startup", 300)), "--stall-slack", str(kw.get("slack", 0)),
            "--probe-every", "0.5", "--poll-every", "0.25", *extra,
            "--", "--data", *DATA, "--max-steps", str(steps), "--ckpt-every", str(ckpt)]
    return args


def unmounted_drive() -> str:
    """A drive letter with nothing mounted, so the backup step takes its skip-and-log path."""
    import string
    return next(f"{c}:" for c in reversed(string.ascii_uppercase) if not Path(f"{c}:/").exists())


def run_runner(rig, env_extra=None, *extra, **kw):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", **(env_extra or {})}
    p = subprocess.run([PY, str(HERE / "run.py"), *runner_args(rig, *extra, **kw)], env=env, capture_output=True,
                       text=True, timeout=900)
    d = rig["root"] / "1" / "ROLE" / "0"
    status = json.loads((d / "status.json").read_text(encoding="utf-8"))
    actions = [json.loads(x) for x in (d / "actions.jsonl").read_text(encoding="utf-8").splitlines()] \
        if (d / "actions.jsonl").exists() else []
    return p, d, status, actions


def losses(d):
    return [(r["step"], r["loss"]) for r in
            (json.loads(x) for x in (d / "train" / "loss.jsonl").read_text(encoding="utf-8").splitlines() if x)]


def plain_run(tmp_path, steps=8, ckpt=2):
    out = tmp_path / "plain"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    subprocess.run([PY, TINY, "--data", *DATA, "--out", str(out), "--max-steps", str(steps), "--ckpt-every",
                    str(ckpt)], env=env, check=True, capture_output=True, timeout=900)
    return [(r["step"], r["loss"]) for r in
            (json.loads(x) for x in (out / "loss.jsonl").read_text(encoding="utf-8").splitlines() if x)]


@needs_tokenizer
def test_a_crash_resumes_to_the_same_losses_as_an_uninterrupted_run(rig, tmp_path):
    want = plain_run(tmp_path)
    p, d, status, actions = run_runner(rig, {"TINY_CRASH_AT_STEP": "6"})
    assert p.returncode == 0, p.stdout + p.stderr
    assert status["state"] == "done" and status["resumes"] == 1
    got = losses(d)
    assert [s for s, _ in got] == list(range(1, 9))  # step 5 ran twice, and is logged once
    assert got == want  # RNG, optimizer and data order restored: identical losses
    resumed = [x for x in actions if x["action"] == "resume from the last checkpoint"]
    assert len(resumed) == 1 and resumed[0]["count"] == 1 and resumed[0]["cap"] == 3
    diag = json.loads((d / "diagnostics.json").read_text(encoding="utf-8"))
    assert diag["resumes"] == 1 and diag["backup"]["status"] == "skipped"
    row = json.loads((d / "receipts" / "diagnostics.jsonl").read_text(encoding="utf-8"))
    assert row["schema"] == "stage1-receipt-v1" and row["source"] == "diagnostics" and row["round"] == 1
    assert row["arm"] == "ROLE" and row["seed"] == 0 and row["grant_id"] == "G-test"


@needs_tokenizer
def test_the_vram_self_stop_is_never_resumed(rig):
    p, d, status, actions = run_runner(rig, {"TINY_OOM_AT_STEP": "3"})
    assert p.returncode == 1
    assert status["state"] == "stopped" and status["stop_reason"] == "VRAM self-stop (28 GB)"
    assert status["resumes"] == 0
    assert status["stop_detail"]["step"] == 3 and "prompt_tokens" in status["stop_detail"]
    assert not any(x["action"].startswith("resume") for x in actions)


@needs_tokenizer
def test_a_transient_io_error_is_retried_and_logged(rig):
    p, d, status, actions = run_runner(rig, {"TINY_IO_FAIL": "2"}, steps=3)
    assert p.returncode == 0, p.stdout + p.stderr
    retries = [x for x in actions if x["trigger"] == "transient I/O error"]
    assert [x["count"] for x in retries] == [1, 2] and all(x["cap"] == 5 for x in retries)
    assert [s for s, _ in losses(d)] == [1, 2, 3]


@needs_tokenizer
def test_a_stall_stops_with_the_trainers_own_stack(rig):
    p, d, status, actions = run_runner(rig, {"TINY_HANG_AT_STEP": "5"}, slack=2)
    assert p.returncode == 1
    assert status["state"] == "stopped" and status["stop_reason"] == "stall"
    stall = [x for x in actions if x["trigger"] == "no step within the stall limit"][0]
    assert stall["stack"] and "File" in stall["stack"]  # faulthandler's dump of the hung step
    assert status["resumes"] == 0


def test_three_resumes_then_stop(rig, tmp_path):
    # A trainer that always crashes: the 4th crash stops the run.
    crasher = tmp_path / "crasher.py"
    crasher.write_text("import sys; sys.exit(3)\n")
    p, d, status, actions = run_runner(rig, None, "--trainer", str(crasher))
    assert status["state"] == "stopped" and status["stop_reason"].startswith("crash after 3 resumes")
    assert [x["count"] for x in actions if x["action"] == "resume from the last checkpoint"] == [1, 2, 3]


def test_no_resume_when_the_card_is_not_empty(rig, tmp_path):
    crasher = tmp_path / "crasher.py"
    # Crashes once, after raising the 'card' memory: the resume precondition sees a busy card and stops.
    crasher.write_text(f"import json; json.dump({{'temp_c': 50, 'power_w': 100, 'mem_used_mib': 9000}}, "
                       f"open(r'{rig['gpu']}', 'w')); raise SystemExit(3)\n")
    p, d, status, actions = run_runner(rig, None, "--trainer", str(crasher))
    assert status["state"] == "stopped" and "the card isn't empty" in status["stop_reason"]
    assert status["resumes"] == 0


def test_refuses_to_start_without_a_live_watchdog_or_after_the_grant(rig):
    rig["stop"].set()
    time.sleep(0.2)
    old = time.time() - 120
    os.utime(rig["wd"] / "_watchdog_HEARTBEAT", (old, old))
    p, d, status, _ = run_runner(rig)
    assert p.returncode == 2 and "watchdog isn't alive" in status["stop_reason"]
    os.utime(rig["wd"] / "_watchdog_HEARTBEAT", None)
    (d / "status.json").unlink()
    p, d, status, _ = run_runner(rig, grant_end="2020-01-01T00:00:00Z")
    assert p.returncode == 2 and "planned end has passed" in status["stop_reason"]


def test_a_watchdog_kill_is_never_resumed(rig, tmp_path):
    crasher = tmp_path / "crasher.py"
    trip = rig["wd"] / "_watchdog_TRIPPED"
    crasher.write_text(f"open(r'{trip}', 'w').write('KILL'); raise SystemExit(1)\n")
    p, d, status, actions = run_runner(rig, None, "--trainer", str(crasher))
    assert status["state"] == "stopped" and status["stop_reason"] == "watchdog kill" and status["resumes"] == 0


@needs_tokenizer
def test_heat_pauses_between_steps_and_resumes(rig, monkeypatch, tmp_path):
    monkeypatch.setattr(run, "HEAT", {"pause_c": 85, "hold_s": 0, "resume_c": 75})
    d = rig["root"] / "1" / "ROLE" / "0"
    hot = {"from": None}

    def probe():
        # Hot for 4 seconds from the first logged step, so the pause lands between steps, not during the load.
        if hot["from"] is None and (d / "train" / "loss.jsonl").exists():
            hot["from"] = time.time()
        temp = 90 if hot["from"] is not None and time.time() - hot["from"] < 4 else 60
        return {"temp_c": temp, "power_w": 300, "mem_used_mib": 1000}
    a = run.parse(runner_args(rig, steps=8))
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    code = run.Runner(a, probe, run.Watchdog(rig["wd"]), [PY, TINY]).run()
    actions = [json.loads(x) for x in (d / "actions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert code == 0
    assert [x["action"] for x in actions if x["action"] in ("pause between steps", "resume")] == \
        ["pause between steps", "resume"]
    diag = json.loads((d / "diagnostics.json").read_text(encoding="utf-8"))
    assert diag["paused_seconds"] > 0
    events = (d / "train" / "trainer-events.jsonl").read_text(encoding="utf-8")
    assert "pause_start" in events and "pause_end" in events
    manifest = json.loads((d / "train" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["paused_seconds"] > 0


@needs_tokenizer
def test_killing_the_runner_ends_the_trainer(rig):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "TINY_HANG_AT_STEP": "3"}
    p = subprocess.Popen([PY, str(HERE / "run.py"), *runner_args(rig, startup=600)], env=env,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    d = rig["root"] / "1" / "ROLE" / "0"
    deadline = time.time() + 300
    while time.time() < deadline and not (d / "train" / "loss.jsonl").exists():
        time.sleep(0.5)
    assert (d / "train" / "loss.jsonl").exists()
    marker = str(rig["root"])
    assert trainer_processes(marker), "the trainer should be running"
    subprocess.run(["taskkill", "/F", "/PID", str(p.pid)], capture_output=True)  # run.py only, not its tree
    end = time.time() + 5
    while time.time() < end and trainer_processes(marker):
        time.sleep(0.25)
    assert not trainer_processes(marker), "train.py outlived run.py"


@needs_tokenizer
def test_the_trainer_stops_itself_when_the_heartbeat_goes_stale(tmp_path):
    out, status = tmp_path / "t", tmp_path / "status.json"
    status.write_text(json.dumps({"supervisor_heartbeat": time.time(), "pause": False}))
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    p = subprocess.Popen([PY, TINY, "--data", *DATA, "--out", str(out), "--status", str(status), "--max-steps", "8"],
                         env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + 300
    while time.time() < deadline and not (out / "loss.jsonl").exists():
        status.write_text(json.dumps({"supervisor_heartbeat": time.time(), "pause": False}))
        time.sleep(0.5)
    status.write_text(json.dumps({"supervisor_heartbeat": time.time() - 500, "pause": False}))  # frozen
    assert p.wait(timeout=120) == 75
    steps = (out / "loss.jsonl").read_text(encoding="utf-8").splitlines()
    assert 1 <= len(steps) < 8
    assert '"supervisor heartbeat lost"' in (out / "trainer-events.jsonl").read_text(encoding="utf-8")


def trainer_processes(marker: str) -> list:
    # python processes only: this PowerShell query's own command line holds both strings too
    ps = ("Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' -and "
          "$_.CommandLine -like '*tiny_train.py*' -and "
          f"$_.CommandLine -like '*{marker}*' }} | ForEach-Object {{ $_.ProcessId }}")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True).stdout
    return [x for x in out.split() if x.strip().isdigit()]
