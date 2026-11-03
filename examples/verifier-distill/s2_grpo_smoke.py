"""Smoke test S2 of the verifier plan (docs/runs/2026-10-10-verifier-distill-prereg-draft.md, the smoke list):
does a 20-step GRPO on the RTX 5090 fit under the watchdog line, and how fast, on each rollout path?

It measures the trainer's memory and speed on this rig. It is not a training result, and nothing it produces
is kept as a model. Pre-registration: docs/runs/2026-10-10-smoke-s2-prereg.md (the pass line is fixed there,
before any run).

  python-gpu s2_grpo_smoke.py --path hf        --gold <dir> --rnd-commit <sha> --out <dir>
  python-gpu s2_grpo_smoke.py --path llamacpp  --gold <dir> --rnd-commit <sha> --out <dir>
      --llama-server <exe> --gguf <file> --convert-script <py>
  python       s2_grpo_smoke.py --dry --gold <dir> --rnd-commit <sha> --out <dir>      # CPU rehearsal

Everything except the model code is plain Python: torch, transformers and peft are imported only inside the
functions that need them, so the pass line, the prompt rule, the reward and the receipt run on any Python.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import importlib.metadata as metadata
import json
import math
import platform
import re
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Callable, Sequence
from pathlib import Path

# -- the pre-registered numbers (docs/runs/2026-10-10-smoke-s2-prereg.md) ------------------------------------
STUDENT = "Qwen/Qwen3-8B"
STUDENT_REVISION = (
    "b968826d9c46"  # the plan's pin, as far as the plan wrote it; the run records the resolved commit
)
STEPS = 20
PROMPTS = 32
GROUPS_PER_STEP = 2
G = 8
TEMPERATURE = 0.7
LORA_R, LORA_ALPHA = 16, 32
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")
LEARNING_RATE = 1e-5
BETA = 0.04
EPSILON = 0.2
MAX_NEW_TOKENS = 2048
VRAM_LINE_MIB = 31_200  # the rig's watchdog line
SAMPLER_PERIOD_S = 1.0
HEARTBEAT_MAX_AGE_S = 60
READY_TIMEOUT_S = 300.0
READY_POLL_S = 1.0
HEARTBEAT_ENV = "WATCHDOG_HEARTBEAT"
PROMPT_SPLIT = "tune"
GOLD_FILES = ("grounded.jsonl", "prs/grounded-prs.jsonl", "diffs/reasoning-diffs.jsonl")
VERDICTS = ("supported", "unsupported", "cannot_tell")
ENVIRONMENT_FIELDS = (
    "python",
    "python_executable",
    "platform",
    "gpu_name",
    "driver",
    "cuda",
    "torch",
    "transformers",
    "peft",
    "bitsandbytes",
    "dtype",
    "batch",
    "gpu_exclusive",
    "exclusive_guard",
)


# -- the data: the prompts -----------------------------------------------------------------------------------
def read_gold(gold_dir: Path) -> tuple[list[dict], dict[str, str]]:
    """Every row of the three gold files, and each file's sha256 (taken from the bytes that were parsed)."""
    rows, hashes = [], {}
    for name in GOLD_FILES:
        data = (gold_dir / name).read_bytes()
        hashes[name] = hashlib.sha256(data).hexdigest()
        rows += [json.loads(x) for x in data.decode("utf-8").splitlines() if x.strip()]
    return rows, hashes


def select_prompts(rows: list[dict], n: int = PROMPTS, split: str = PROMPT_SPLIT) -> list[dict]:
    """The rule: the first n rows of the tune split by sorted claim id. A gold with fewer stops the run."""
    mine = sorted((r for r in rows if r.get("split") == split), key=lambda r: r["id"])
    if len(mine) < n:
        raise SystemExit(f"the gold has {len(mine)} {split} claims; S2 needs {n}")
    return mine[:n]


def evidence_of(row: dict) -> str:
    return "\n".join(f"[{c.get('source', '')}]\n{c.get('text', '')}" for c in row["context"])


def build_prompt(row: dict) -> str:
    """The user message: the task, the reply contract, the evidence and the claim."""
    return (
        "Decide whether the evidence supports the claim. First think, then answer with one JSON object:\n"
        '{"verdict": "supported" | "unsupported" | "cannot_tell", "evidence_quotes": [{"part": "<the part of '
        'the claim>", "verdict": "<its verdict>", "quote": "<text copied exactly from the evidence>"}]}\n'
        "Give 1 to 4 evidence_quotes (a simple claim is one part). A quote must appear in the evidence.\n\n"
        f"Evidence:\n{evidence_of(row)}\n\nClaim: {row['claim']}\n"
    )


# -- the reward: the plan's mechanical reward, reduced as the pre-registration declares ----------------------
def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_reply(text: str) -> tuple[str, str]:
    """(thinking, answer): the thinking is what precedes </think>, when there is one."""
    if "</think>" in text:
        head, _, tail = text.partition("</think>")
        return head.replace("<think>", "").strip(), tail
    return "", text


def parse_reply(text: str) -> dict | None:
    """The answer as contract v3, or None when not schema-valid (1 to 4 parts, each with a known verdict)."""
    _, answer = split_reply(text)
    start, end = answer.find("{"), answer.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(answer[start : end + 1])
    except json.JSONDecodeError:
        return None
    parts = obj.get("evidence_quotes") if isinstance(obj, dict) else None
    if obj.get("verdict") not in VERDICTS or not isinstance(parts, list) or not 1 <= len(parts) <= 4:
        return None
    for p in parts:
        if not isinstance(p, dict) or p.get("verdict") not in VERDICTS or not isinstance(p.get("quote"), str):
            return None
    return obj


def loop_detected(text: str, ngram: int = 8, repeats: int = 40) -> bool:
    """offrig's detector: an 8-word n-gram repeated 40 times."""
    words = text.split()
    counts: dict[tuple[str, ...], int] = {}
    for i in range(len(words) - ngram + 1):
        key = tuple(words[i : i + ngram])
        counts[key] = counts.get(key, 0) + 1
        if counts[key] >= repeats:
            return True
    return False


def part_rule(parts: list[dict]) -> str:
    """The three-valued part rule: any part unsupported, else any cannot_tell, else supported."""
    verdicts = {p["verdict"] for p in parts}
    return (
        "unsupported"
        if "unsupported" in verdicts
        else "cannot_tell"
        if "cannot_tell" in verdicts
        else "supported"
    )


def macro(verdict: str, gold: str) -> tuple[float, bool]:
    """(the claim outcome, whether the verdict is correct). On gold cannot_tell, unsupported also counts."""
    if verdict == gold or (gold == "cannot_tell" and verdict == "unsupported"):
        return 1.0, True
    if verdict == "supported":
        return -3.0, False
    if verdict == "unsupported":
        return -1.0, False
    return -0.3, False  # cannot_tell on a decidable gold


def reward(reply: str, gold: str, evidence: str, thinking_tokens: int, *, truncated: bool = False) -> float:
    """The gated reward of the plan (:336-412), reduced as declared in the pre-registration.

    Run: the format gate; the macro outcome; the micro quote rule (an unfound quote on a supported part is a
    false accept); credit for found quotes (extras, at most +0.3, correct verdicts only); the meso consistency
    penalty (-0.5); the length penalty on correct samples (capped -0.5); the loop penalty (-0.5, and the
    verdict
    is read as cannot_tell); the hard cap at 8,192 thinking tokens (-1.0).
    Not run, because the gold has no part labels or tiers: part correctness, part false accepts, and the easy
    and quick credit.
    """

    if truncated or thinking_tokens >= 8192:
        return -1.0
    parsed = parse_reply(reply)
    if parsed is None:
        return -1.0
    verdict = parsed["verdict"]
    looped = loop_detected(reply)
    if looped:
        verdict = "cannot_tell"
    outcome, correct = macro(verdict, gold)
    evidence_n = normalise(evidence)
    found = [
        normalise(p["quote"]) in evidence_n and bool(normalise(p["quote"])) for p in parsed["evidence_quotes"]
    ]
    unfound_supported = any(
        p["verdict"] == "supported" and not f for p, f in zip(parsed["evidence_quotes"], found, strict=True)
    )
    if verdict == "supported" and unfound_supported:
        outcome, correct = -3.0, False
    meso = -0.5 if part_rule(parsed["evidence_quotes"]) != parsed["verdict"] else 0.0
    loop_pen = -0.5 if looped else 0.0
    if not correct:
        return outcome + meso + loop_pen
    budget = 256 if verdict == "unsupported" else 1024
    length = max(-0.5, -0.1 * (thinking_tokens - budget) / budget) if thinking_tokens > budget else 0.0
    penalties = max(meso + length + loop_pen, -1.0)
    extras = min(0.05 * sum(found), 0.3)
    return outcome + penalties + extras


def advantages(rewards: Sequence[float]) -> list[float]:
    """Group-normalised. A group whose rewards are all equal has advantage 0 (zero-variance)."""
    sd = statistics.pstdev(rewards)
    if sd == 0:
        return [0.0] * len(rewards)
    mean = statistics.fmean(rewards)
    return [(r - mean) / (sd + 1e-6) for r in rewards]


def zero_variance(rewards: Sequence[float]) -> bool:
    return len(set(rewards)) <= 1


# -- the readout, the pass line and the exit status ----------------------------------------------------------
def readout(
    steps: list[dict],
    *,
    peak_mib: int | None,
    steps_requested: int,
    stopped: dict | None,
    oom: bool,
    cut: bool,
    dry: bool,
    series: Sequence[int] = (),
) -> dict:
    """The pass decision, per path. A run that was cut, stopped or dry never passes.

    whole = the registered shape ran: no --steps/--prompts/--g cut, and not a dry rehearsal.
    pass  = whole, no stop, no OOM, every step completed, and the sampler's peak under the watchdog line.
    """

    completed = len(steps)
    whole = not cut and not dry and steps_requested == STEPS
    step_s = [s["step_s"] for s in steps]
    tokens = sum(s["tokens"] for s in steps)
    rollout_s = sum(s["rollout_s"] for s in steps)
    fits = peak_mib is not None and peak_mib < VRAM_LINE_MIB
    complete = completed == steps_requested and stopped is None and not oom
    return {
        "steps_requested": steps_requested,
        "steps_completed": completed,
        "peak_vram_mib_sampler": peak_mib,
        "sampler_samples": len(series),
        "vram_mib_sampler_series": list(series),
        "peak_vram_mib_torch": max((s.get("peak_alloc_mib") or 0 for s in steps), default=None)
        if steps
        else None,
        "fits": fits,
        "oom": oom,
        "rollout_tok_s": round(tokens / rollout_s, 2) if rollout_s > 0 else None,
        "step_s": {"p50": statistics.median(step_s), "max": max(step_s)} if step_s else None,
        "wall_s": round(sum(step_s), 2),
        "zero_variance_groups": sum(s["zero_variance_groups"] for s in steps),
        "groups": sum(s["groups"] for s in steps),
        "complete": complete and whole,
        "pass": bool(complete and whole and fits),
    }


def verdict_line(result: dict, *, stopped: dict | None, cut: bool, dry: bool) -> str:
    """S2-OK only for a whole run that fits. S2-DRY: a rehearsal. S2-LIMITED: a cut. S2-STOPPED: a stop."""
    if dry:
        return (
            "S2-DRY"
            if not stopped and result["steps_completed"] == result["steps_requested"]
            else "S2-STOPPED"
        )
    if stopped:
        return "S2-STOPPED"
    if cut:
        return "S2-LIMITED"
    if result["pass"]:
        return "S2-OK"
    return "S2-FAIL"  # an OOM, an incomplete run, or a peak over the line: the measurement is the result


def exit_status(line: str) -> int:
    """0 for S2-OK and for a finished rehearsal; anything else is non-zero."""
    return 0 if line in ("S2-OK", "S2-DRY") else 1


# -- the environment, the guards and the receipt -------------------------------------------------------------
def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def environment(
    *,
    gpu: dict | None,
    dtype: str,
    batch: int,
    exclusive: bool | None,
    cuda: str | None,
    guard: dict | None = None,
) -> dict:
    """What the receipt records about the machine. No package is imported to read a version."""
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "gpu_name": gpu["name"] if gpu else None,
        "driver": gpu["driver"] if gpu else None,
        "cuda": cuda,
        "torch": package_version("torch"),
        "transformers": package_version("transformers"),
        "peft": package_version("peft"),
        "bitsandbytes": package_version("bitsandbytes"),
        "dtype": dtype,
        "batch": batch,
        "gpu_exclusive": exclusive,
        "exclusive_guard": guard,
    }


def require_single_nvidia_gpu(gpus: list[dict]) -> dict:
    """The one NVIDIA GPU, or a refusal: nothing else is ever used."""
    if len(gpus) != 1:
        raise SystemExit(f"refusing to start: {len(gpus)} GPUs visible; S2 uses the single NVIDIA GPU only")
    if not gpus[0]["name"].startswith("NVIDIA"):
        raise SystemExit(f"refusing to start: {gpus[0]['name']!r} is not an NVIDIA GPU")
    return gpus[0]


def require_exclusive(apps: list[str]) -> None:
    """The plain refuse-on-any rule. Superseded by ``check_exclusive`` for the real run (see the prereg):
    under Windows WDDM the compute-apps list always holds the desktop, so this refused every time."""
    if apps:
        raise SystemExit(f"refusing to start: {len(apps)} other process(es) hold the GPU")


# The exclusive-GPU guard (rules a and b, fail closed). Measured on the rig, 2026-11-03: idle memory.used read
# 2,238 to 2,572 MiB (desktop only) and the compute-apps list always held about 26 desktop processes of type
# "C+G" with memory "N/A". The baseline is the measured idle maximum plus a margin of about 1.5 GB.
COMPUTE_RUNNER_PATTERNS = (
    "python*",
    "ollama*",
    "llama-server*",
)  # image names, lower case; a card adds a name
EXCLUSIVE_BASELINE_MIB = 4096
DESKTOP_PROCESS_TYPES = ("C+G", "G")  # type "C" (compute only) and any other type refuse
SmiRunner = Callable[[Sequence[str]], str]


def image_name(process_name: str) -> str:
    """The file name only: a Windows or POSIX path is cut to its last part, so no path is ever recorded."""
    text = (process_name or "").strip()
    if text.upper() in ("", "N/A", "[NOT FOUND]"):
        return ""
    return re.split(r"[\\/]", text)[-1]


def parse_gpu_processes(xml_text: str) -> list[dict]:
    """The processes from ``nvidia-smi -q -x``: ``[{"image": <file name>, "type": "C" | "G" | "C+G"}]``.

    That XML carries each process's type and name (the compute-apps query has no type and, on Windows,
    no usable memory figure). Output that is not that XML is a refusal, never an empty list.

    The shapes that refuse, each named in the message: not XML, a wrong root, no ``<gpu>``, more than one
    ``<gpu>``, a ``<gpu>`` with no ``<processes>``, ``<processes>`` holding text but no ``<process_info>``
    (for example "N/A"), and an unknown process type. Only a present, truly empty ``<processes/>`` means
    "nothing on the card".

    If the real rig ever returns one of the refused shapes, it is amended by a card with the real fixture,
    never by relaxing this parser."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise SystemExit(
            f"refusing to start: nvidia_smi_garbage: the process list is not XML ({exc})"
        ) from exc
    if root.tag != "nvidia_smi_log":
        raise SystemExit(f"refusing to start: nvidia_smi_garbage: unexpected root element {root.tag!r}")
    gpus = root.findall("gpu")
    if not gpus:
        raise SystemExit("refusing to start: nvidia_smi_garbage: no <gpu> element")
    if len(gpus) > 1:
        raise SystemExit(f"refusing to start: nvidia_smi_garbage: more than one <gpu> element ({len(gpus)})")
    listing = gpus[0].find("processes")
    if listing is None:
        raise SystemExit("refusing to start: nvidia_smi_garbage: the <gpu> has no <processes> element")
    infos = listing.findall("process_info")
    if not infos and (listing.text or "").strip():
        raise SystemExit(
            f"refusing to start: nvidia_smi_garbage: <processes> holds text {listing.text.strip()!r} "
            "but no <process_info> entries"
        )
    procs = []
    for info in infos:
        ptype = (info.findtext("type") or "").strip().upper()
        if ptype not in ("C", "G", "C+G"):
            raise SystemExit(f"refusing to start: nvidia_smi_garbage: unknown process type {ptype!r}")
        procs.append({"image": image_name(info.findtext("process_name") or ""), "type": ptype})
    return procs


def parse_memory_used(text: str) -> int:
    """``memory.used`` in MiB from ``--query-gpu=memory.used --format=csv,noheader,nounits``.

    Exactly one line (the guard allows one GPU) holding a plain non-negative integer; anything else refuses.
    Rule (b) reads only this source; the XML's ``fb_memory_usage`` is not read."""
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    if not lines:
        raise SystemExit("refusing to start: nvidia_smi_garbage: no memory.used reading")
    if len(lines) != 1:
        raise SystemExit(
            f"refusing to start: nvidia_smi_garbage: {len(lines)} memory.used readings for one GPU"
        )
    if not lines[0].isdigit():
        raise SystemExit(
            f"refusing to start: nvidia_smi_garbage: memory.used is {lines[0]!r}, not a MiB integer"
        )
    return int(lines[0])


def run_smi(args: Sequence[str]) -> str:  # pragma: no cover - runs the real nvidia-smi
    done = subprocess.run(
        ["nvidia-smi", *args],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
        stdin=subprocess.DEVNULL,
    )
    return done.stdout


def ask_smi(smi: SmiRunner, args: Sequence[str]) -> str:
    """One nvidia-smi call, with every way it can fail turned into a named refusal."""
    try:
        return smi(args)
    except FileNotFoundError as exc:
        raise SystemExit("refusing to start: nvidia_smi_missing: nvidia-smi is not on the PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise SystemExit("refusing to start: nvidia_smi_timeout: nvidia-smi did not answer in time") from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"refusing to start: nvidia_smi_failed: nvidia-smi exited {exc.returncode}") from exc
    except OSError as exc:
        raise SystemExit(f"refusing to start: nvidia_smi_unavailable: {type(exc).__name__}") from exc


def parse_gpus(text: str) -> list[dict]:
    """``--query-gpu=name,driver_version --format=csv,noheader,nounits`` as ``[{"name", "driver"}]``."""
    gpus = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = [x.strip() for x in line.split(",")]
        if len(parts) != 2 or not all(parts):
            raise SystemExit(f"refusing to start: nvidia_smi_garbage: GPU line {line!r}")
        gpus.append({"name": parts[0], "driver": parts[1]})
    return gpus


def check_exclusive(smi: SmiRunner, *, source: str = "nvidia-smi") -> dict:
    """Refuse unless nobody who matters holds the card; return what was seen, for the receipt.

    (a) refuse on a listed process whose image name matches a compute runner, or whose type is "C" alone;
    (b) refuse when the total ``memory.used`` is above ``EXCLUSIVE_BASELINE_MIB``.
    A missing, failing, slow or unreadable nvidia-smi refuses with a named reason: it never passes by default.
    ``smi`` takes the nvidia-smi arguments and returns its stdout (injected: tests never run the real one)."""
    procs = parse_gpu_processes(ask_smi(smi, ["-q", "-x"]))
    used = parse_memory_used(ask_smi(smi, ["--query-gpu=memory.used", "--format=csv,noheader,nounits"]))
    runners = sorted(
        {
            f"{p['image'] or '<unnamed>'} (type {p['type']})"
            for p in procs
            if p["type"] == "C"
            or any(fnmatch.fnmatchcase(p["image"].lower(), pat) for pat in COMPUTE_RUNNER_PATTERNS)
        }
    )
    rule_a = not runners
    rule_b = used <= EXCLUSIVE_BASELINE_MIB
    record = {
        "source": source,
        "processes": sorted(p["image"] or "<unnamed>" for p in procs),
        "process_types": sorted({p["type"] for p in procs}),
        "memory_used_mib": used,
        "baseline_mib": EXCLUSIVE_BASELINE_MIB,
        "runner_patterns": list(COMPUTE_RUNNER_PATTERNS),
        "rules": {
            "a_no_compute_runner": "passed" if rule_a else "FAILED",
            "b_memory_baseline": "passed" if rule_b else "FAILED",
        },
        "result": "passed" if rule_a and rule_b else "refused",
    }
    fired = []
    if not rule_a:
        fired.append(f"rule (a): compute runner(s) hold the GPU: {', '.join(runners)}")
    if not rule_b:
        fired.append(f"rule (b): memory.used is {used} MiB, above the {EXCLUSIVE_BASELINE_MIB} MiB baseline")
    if fired:
        raise SystemExit("refusing to start: " + "; ".join(fired))
    return record


DRY_DESKTOP_IMAGES = (
    "Antigravity.exe ApplicationFrameHost.exe chrome.exe claude.exe CrossDeviceResume.exe explorer.exe "
    "IntelGraphicsSoftware.exe msedgewebview2.exe Notepad.exe OmenCommandCenterBackground.exe SearchHost.exe "
    "ShellExperienceHost.exe ShellHost.exe SnippingTool.exe StartMenuExperienceHost.exe steamwebhelper.exe "
    "SystemSettings.exe TextInputHost.exe WidgetBoard.exe WindowsTerminal.exe "
    "chrome.exe chrome.exe chrome.exe msedgewebview2.exe msedgewebview2.exe msedgewebview2.exe"
).split()  # the 20 desktop images the rig showed, and 6 repeats: 26 processes of type "C+G"
DRY_IDLE_MIB = 2400


def dry_smi(args: Sequence[str]) -> str:
    """A canned nvidia-smi for ``--dry``: the rig's desktop case. It never starts the real nvidia-smi."""
    if list(args) == ["-q", "-x"]:
        infos = "".join(
            f"<process_info><pid>{1000 + i}</pid><type>C+G</type>"
            f"<process_name>C:\\Windows\\{name}</process_name><used_memory>N/A</used_memory></process_info>"
            for i, name in enumerate(DRY_DESKTOP_IMAGES)
        )
        return f"<nvidia_smi_log><gpu><processes>{infos}</processes></gpu></nvidia_smi_log>"
    return f"{DRY_IDLE_MIB}\n"


def require_heartbeat(mtime: float | None, now: float, max_age_s: float = HEARTBEAT_MAX_AGE_S) -> None:
    """The VRAM watchdog must be alive: its heartbeat file's modified time (never opened) is recent."""
    if mtime is None:
        raise SystemExit(f"refusing to start: no watchdog heartbeat (set {HEARTBEAT_ENV} to its file)")
    if now - mtime > max_age_s:
        raise SystemExit(f"refusing to start: the watchdog heartbeat is {now - mtime:.0f}s old")


def build_receipt(
    *,
    path: str,
    dry: bool,
    args: dict,
    rnd_commit: str,
    gold_hashes: dict[str, str],
    prompt_ids: list[str],
    env: dict,
    revision: str | None,
    llama: dict | None,
    result: dict,
    stopped: dict | None,
    verdict: str,
) -> dict:
    return {
        "test": "S2",
        "path": path,
        "dry": dry,
        "model": STUDENT,
        "revision_registered": STUDENT_REVISION,
        "revision_resolved": revision,
        "registered": {
            "steps": STEPS,
            "prompts": PROMPTS,
            "groups_per_step": GROUPS_PER_STEP,
            "G": G,
            "temperature": TEMPERATURE,
            "lora": {"r": LORA_R, "alpha": LORA_ALPHA, "targets": list(LORA_TARGETS)},
            "learning_rate": LEARNING_RATE,
            "beta": BETA,
            "epsilon": EPSILON,
            "max_new_tokens": MAX_NEW_TOKENS,
        },
        "run": args,
        "rnd_commit": rnd_commit,
        "gold_files_sha256": gold_hashes,
        "prompt_split": PROMPT_SPLIT,
        "prompt_ids": prompt_ids,
        "environment": env,
        "llama_cpp": llama,
        "pass_line": {"vram_line_mib": VRAM_LINE_MIB, "steps": STEPS, "oom": False},
        "stopped": stopped,
        "readout": result,
        "verdict": verdict,
    }


# -- the VRAM sampler (nvidia-smi, once a second) ------------------------------------------------------------
class VramSampler:
    """Polls the card's used memory and keeps the peak. ``query`` gives MiB used, or None if unreadable."""

    def __init__(self, query: Callable[[], int | None], period_s: float = SAMPLER_PERIOD_S) -> None:
        self.query, self.period_s = query, period_s
        self.peak_mib: int | None = None
        self.samples = 0
        self.series: list[int] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def sample_once(self) -> None:
        used = self.query()
        if used is not None:
            self.samples += 1
            self.series.append(used)
            self.peak_mib = used if self.peak_mib is None else max(self.peak_mib, used)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.sample_once()
            self._stop.wait(self.period_s)

    def __enter__(self) -> VramSampler:
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=5)
        self.sample_once()


# -- torch, imported only where it is used -------------------------------------------------------------------
def require_torch():
    """Import torch. Absent: a clear refusal. Present but broken (a DLL, an ABI clash): the real error."""
    try:
        import torch
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise SystemExit(
                "torch is not installed in this Python; install it, or run the pure parts only"
            ) from exc
        raise
    return torch


def completion_logprobs(model, input_ids, attention_mask, prompt_len: int):
    """Log-probability of each completion token given what precedes it: (batch, completion_len)."""
    torch = require_torch()
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :].float()
    logp = torch.log_softmax(logits, dim=-1).gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    return logp[:, prompt_len - 1 :]


def grpo_loss(logp, old_logp, ref_logp, adv, mask, *, beta: float = BETA, eps: float = EPSILON):
    """The clipped-ratio policy loss plus a k3 KL penalty to the reference, masked, averaged over tokens."""
    torch = require_torch()
    ratio = torch.exp(logp - old_logp)
    a = adv.unsqueeze(-1)
    policy = -torch.min(ratio * a, torch.clamp(ratio, 1 - eps, 1 + eps) * a)
    delta = ref_logp - logp
    kl = torch.exp(delta) - delta - 1
    return ((policy + beta * kl) * mask).sum() / mask.sum().clamp(min=1)


def hf_rollout(model, tok, prompt_text: str, g: int, max_new: int, temperature: float):
    """Path (a): Hugging Face generate, G samples of one prompt from the model that is being trained."""
    torch = require_torch()
    enc = tok(prompt_text, return_tensors="pt")
    prompt_ids = enc["input_ids"].to(model.device)
    with torch.no_grad():
        out = model.generate(
            prompt_ids.repeat(g, 1),
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new,
            pad_token_id=tok.pad_token_id,
        )
    comps = []
    for row in out[:, prompt_ids.shape[1] :]:
        ids = row.tolist()
        while ids and ids[-1] == tok.pad_token_id:
            ids.pop()
        comps.append(ids)
    return prompt_ids[0].tolist(), comps


def http_json(base_url: str, path: str, payload, timeout_s: float, *, urlopen=urllib.request.urlopen):
    """POST ``payload`` as JSON (a GET when None). A socket timeout always becomes TimeoutError."""
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(base_url + path, data=data, headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=timeout_s) as resp:
            return json.loads(resp.read().decode())
    except TimeoutError as exc:
        raise TimeoutError(f"{path} did not answer in {timeout_s:g}s") from exc
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, TimeoutError):
            raise TimeoutError(f"{path} did not answer in {timeout_s:g}s") from exc
        raise


def wait_ready(
    get: Callable[[str, float], dict],
    poll: Callable[[], int | None],
    *,
    timeout_s: float = READY_TIMEOUT_S,
    control_timeout_s: float = 10.0,
    interval_s: float = READY_POLL_S,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], float] = time.monotonic,
) -> None:
    """Poll the server's /health until it says ok. A server that exited, or is never ok in time, is a stop."""
    deadline = now() + timeout_s
    while True:
        code = poll()
        if code is not None:
            raise SystemExit(f"llama_server_exited: the server ended with code {code} before it was ready")
        try:
            reply = get("/health", control_timeout_s)
            if isinstance(reply, dict) and reply.get("status") == "ok":
                return
        except (urllib.error.URLError, TimeoutError, OSError, ValueError):
            pass  # still loading the model (503), not listening yet, or a slow answer: poll again
        if now() >= deadline:
            raise SystemExit(f"llama_server_not_ready: no ok from /health within {timeout_s:g}s")
        sleep(interval_s)


def terminate_and_wait(proc, timeout_s: float = 10.0) -> None:
    """Stop a child process, and make sure it is gone: terminate, wait, then kill."""
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def file_record(path: Path) -> dict:
    return {"path": str(path), "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def start_llama_server(
    a: argparse.Namespace,
    *,
    popen=subprocess.Popen,
    urlopen=urllib.request.urlopen,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], float] = time.monotonic,
    port: int = 8091,
):
    """Start the server and wait until it is ready. Returns (process, post). A server that is not ready is
    stopped before the error is raised, so it never outlives the run."""
    proc = popen(
        [
            str(a.llama_server),
            "-m",
            str(a.gguf),
            "--port",
            str(port),
            "-ngl",
            "99",
            "--lora-init-without-apply",
        ]
    )
    base = f"http://127.0.0.1:{port}"
    try:
        wait_ready(
            lambda path, timeout_s: http_json(base, path, None, timeout_s, urlopen=urlopen),
            proc.poll,
            timeout_s=a.ready_timeout,
            control_timeout_s=a.timeout,
            sleep=sleep,
            now=now,
        )
    except BaseException:
        terminate_and_wait(proc)
        raise
    return proc, lambda path, payload, timeout_s: http_json(base, path, payload, timeout_s, urlopen=urlopen)


class LlamaCppRollout:
    """Path (b): a llama.cpp server samples G completions; the adapter is exported and reloaded each step.

    ``post(path, payload, timeout_s)`` is the transport (injected, so the control calls can be tested).
    ``export_adapter(step)`` writes the current LoRA as a GGUF adapter and returns its path.
    """

    def __init__(
        self, post, export_adapter, tok, timeout_s: float = 10.0, gen_timeout_s: float = 120.0
    ) -> None:
        self.post, self.export_adapter, self.tok = post, export_adapter, tok
        self.timeout_s, self.gen_timeout_s = timeout_s, gen_timeout_s

    def reload(self, step: int) -> None:
        adapter = self.export_adapter(step)
        self.post("/lora-adapters", [{"id": 0, "path": str(adapter), "scale": 1.0}], self.timeout_s)

    def __call__(self, prompt_text: str, g: int, max_new: int, temperature: float):
        prompt_ids = self.tok(prompt_text)["input_ids"]
        comps = []
        for _ in range(g):
            reply = self.post(
                "/completion",
                {
                    "prompt": prompt_ids,
                    "n_predict": max_new,
                    "temperature": temperature,
                    "return_tokens": True,
                },
                self.gen_timeout_s,
            )
            comps.append(list(reply["tokens"]))
        return list(prompt_ids), comps


# -- one run -------------------------------------------------------------------------------------------------
def run_steps(
    *,
    model,
    ref_adapter_off: Callable,
    tok,
    optimiser,
    prompts: list[dict],
    rollout: Callable,
    steps: int,
    g: int,
    max_new: int,
    reward_fn: Callable[[str, dict, int], float],
    device_sync: Callable[[], None],
    peak_alloc_mib: Callable[[], int | None],
    on_step: Callable[[dict], None] | None = None,
    before_step: Callable[[int], None] | None = None,
    mode: Callable[[bool], None] | None = None,
    state: Callable[[], dict] | None = None,
) -> tuple[list[dict], dict | None, bool]:
    """Run GRPO steps. Returns (step rows, a stop record or None, whether it ran out of memory).

    Each step is GROUPS_PER_STEP prompt groups of G samples, one group at a time (gradients accumulate), so
    peak memory is one group's. The reference is the same model with the adapter off.

    ``mode(False)`` puts the model in rollout mode (gradient checkpointing off, the KV cache on) and
    ``mode(True)`` in training mode (checkpointing on, cache off, as the plan's memory table assumes);
    ``state()`` reports whether checkpointing is on. Both states are recorded in every step row.
    """

    torch = require_torch()
    rows: list[dict] = []
    oom_types = tuple(t for t in (getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None),) if t)
    for step in range(steps):
        try:
            started = time.time()
            if before_step:
                before_step(step)
            roll_s = train_s = 0.0
            tokens = zero_var = 0
            rewards_all: list[float] = []
            optimiser.zero_grad(set_to_none=True)
            for k in range(GROUPS_PER_STEP):
                row = prompts[(step * GROUPS_PER_STEP + k) % len(prompts)]
                if mode:
                    mode(False)
                ckpt_rollout = state()["grad_ckpt"] if state else None
                t0 = time.time()
                prompt_ids, comps = rollout(build_prompt(row), g, max_new, TEMPERATURE)
                device_sync()
                roll_s += time.time() - t0
                tokens += sum(len(c) for c in comps)
                rewards = []
                for c in comps:
                    text = tok.decode(c, skip_special_tokens=True)
                    think, _ = split_reply(text)
                    rewards.append(reward_fn(text, row, len(tok.encode(think)) if think else 0))
                zero_var += zero_variance(rewards)
                rewards_all += rewards
                adv = torch.tensor(advantages(rewards), dtype=torch.float32)
                t1 = time.time()
                if mode:
                    mode(True)
                ckpt_train = state()["grad_ckpt"] if state else None
                width = max(len(c) for c in comps) or 1
                pad = tok.pad_token_id
                ids = torch.tensor([prompt_ids + c + [pad] * (width - len(c)) for c in comps])
                att = torch.tensor([[1] * (len(prompt_ids) + len(c)) + [0] * (width - len(c)) for c in comps])
                mask = torch.tensor([[1.0] * len(c) + [0.0] * (width - len(c)) for c in comps])
                ids, att, mask, adv = (x.to(model.device) for x in (ids, att, mask, adv))
                with torch.no_grad():
                    old = completion_logprobs(model, ids, att, len(prompt_ids))
                    with ref_adapter_off():
                        ref = completion_logprobs(model, ids, att, len(prompt_ids))
                logp = completion_logprobs(model, ids, att, len(prompt_ids))
                loss = grpo_loss(logp, old, ref, adv, mask) / GROUPS_PER_STEP
                loss.backward()
                device_sync()
                train_s += time.time() - t1
            t2 = time.time()
            optimiser.step()
            device_sync()
            train_s += time.time() - t2
            if not math.isfinite(float(loss.detach())):
                return rows, {"step": step, "error": "non-finite loss"}, False
            rec = {
                "step": step,
                "groups": GROUPS_PER_STEP,
                "zero_variance_groups": zero_var,
                "tokens": tokens,
                "rollout_s": round(roll_s, 4),
                "train_s": round(train_s, 4),
                "step_s": round(time.time() - started, 4),
                "mean_reward": round(statistics.fmean(rewards_all), 4),
                "loss": round(float(loss.detach()) * GROUPS_PER_STEP, 6),
                "peak_alloc_mib": peak_alloc_mib(),
                "grad_ckpt": ckpt_train,
                "grad_ckpt_rollout": ckpt_rollout,
            }
            rows.append(rec)
            if on_step:
                on_step(rec)
        except oom_types:
            return rows, {"step": step, "error": "out of memory"}, True
        except TimeoutError as exc:
            return rows, {"step": step, "error": f"timeout: {exc}"}, False
    return rows, None, False


def make_mode(model) -> tuple[Callable[[bool], None], Callable[[], dict]]:
    """(mode, state) for a PEFT model: training = checkpointing on and cache off; rollout = the reverse."""
    inner = model.get_base_model()

    def mode(training: bool) -> None:
        if training:
            inner.config.use_cache = False
            model.train()
            inner.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        else:
            inner.gradient_checkpointing_disable()
            inner.config.use_cache = True
            model.eval()

    def state() -> dict:
        return {"grad_ckpt": bool(inner.is_gradient_checkpointing)}

    return mode, state


# -- the dry rehearsal: a tiny random model on the CPU, no download, no network ------------------------------
class DryTokenizer:
    """Bytes as tokens (ids 2 to 257; 0 pads, 1 ends), so the rehearsal needs no tokenizer download."""

    pad_token_id, eos_token_id, vocab_size = 0, 1, 258

    def encode(self, text: str) -> list[int]:
        return [b + 2 for b in text.encode("utf-8")]

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        return bytes(i - 2 for i in ids if i >= 2).decode("utf-8", errors="replace")

    def __call__(self, text: str, return_tensors: str | None = None) -> dict:
        ids = self.encode(text)
        if return_tensors == "pt":
            torch = require_torch()
            return {"input_ids": torch.tensor([ids])}
        return {"input_ids": ids}


def dry_reward(text: str, row: dict, thinking_tokens: int) -> float:
    """Dry only: a reward that varies between samples, so the advantage and loss path runs. Not the plan's."""
    return (sum(text.encode("utf-8")) % 7) / 7.0 - 0.5


def dry_bundle(a: argparse.Namespace) -> dict:
    """A 2-layer Llama with the real LoRA targets (q/k/v/o), random weights, on the CPU."""
    torch = require_torch()
    from peft import LoraConfig, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    tok = DryTokenizer()
    cfg = LlamaConfig(
        vocab_size=tok.vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=4096,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )
    model = get_peft_model(
        LlamaForCausalLM(cfg),
        LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, target_modules=list(LORA_TARGETS), lora_dropout=0.0),
    )
    params = [p for p in model.parameters() if p.requires_grad]
    mode, state = make_mode(model)
    return {
        "model": model,
        "tok": tok,
        "mode": mode,
        "state": state,
        "optimiser": torch.optim.AdamW(params, lr=LEARNING_RATE),
        "ref_adapter_off": model.disable_adapter,
        "rollout": lambda text, g, max_new, temp: hf_rollout(model, tok, text, g, max_new, temp),
        "reward_fn": dry_reward,
        "sync": lambda: None,
        "peak_alloc_mib": lambda: None,
        "revision": None,
        "llama": None,
    }


# -- the rig's side: the GPU, the watchdog, the real model (needs the 5090, never run in the build) ----------
def _smi(*query: str) -> list[list[str]]:  # pragma: no cover - needs nvidia-smi
    out = subprocess.run(
        ["nvidia-smi", f"--query-gpu={','.join(query)}", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    return [[x.strip() for x in line.split(",")] for line in out.stdout.splitlines() if line.strip()]


class Rig:
    """The machine's side of a real run: the guards, the VRAM query, the CUDA version.

    ``smi`` is the nvidia-smi runner (injected, so the guards are tested without the real one)."""

    def __init__(self, a: argparse.Namespace, smi: SmiRunner = run_smi) -> None:
        self.a = a
        self.smi = smi

    def preflight(self) -> dict:
        import os

        beat = self.a.heartbeat or (
            Path(os.environ[HEARTBEAT_ENV]) if os.environ.get(HEARTBEAT_ENV) else None
        )
        mtime = beat.stat().st_mtime if beat is not None and beat.exists() else None  # never opened
        require_heartbeat(mtime, time.time())
        gpus = parse_gpus(
            ask_smi(self.smi, ["--query-gpu=name,driver_version", "--format=csv,noheader,nounits"])
        )
        gpu = require_single_nvidia_gpu(gpus)
        return {**gpu, "guard": check_exclusive(self.smi)}

    def vram_query(self) -> int | None:  # pragma: no cover - needs the 5090
        try:
            return int(_smi("memory.used")[0][0])
        except (subprocess.SubprocessError, ValueError, IndexError, OSError):
            return None

    def cuda_version(self) -> str | None:  # pragma: no cover - needs the 5090
        return require_torch().version.cuda


def real_bundle(a: argparse.Namespace) -> dict:  # pragma: no cover - needs the 5090 and the model
    """Qwen3-8B on a 4-bit base with a LoRA (r 16, alpha 32, q/k/v/o); path (a) samples with the same copy."""
    torch = require_torch()
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(STUDENT, revision=a.revision)
    base = AutoModelForCausalLM.from_pretrained(
        STUDENT,
        revision=a.revision,
        quantization_config=quant,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
    )
    revision = getattr(base.config, "_commit_hash", None)
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    model = get_peft_model(
        base, LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, target_modules=list(LORA_TARGETS), lora_dropout=0.0)
    )
    model.print_trainable_parameters()
    torch.cuda.reset_peak_memory_stats()
    optimiser = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE)

    def render(text: str) -> str:
        return tok.apply_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=True,
        )

    mode, state = make_mode(model)
    bundle = {
        "model": model,
        "tok": tok,
        "mode": mode,
        "state": state,
        "optimiser": optimiser,
        "ref_adapter_off": model.disable_adapter,
        "reward_fn": lambda text, row, tt: reward(text, row["label"], evidence_of(row), tt),
        "sync": torch.cuda.synchronize,
        "peak_alloc_mib": lambda: int(torch.cuda.max_memory_allocated() / 2**20),
        "revision": revision,
        "llama": None,
    }
    if a.path == "hf":
        # generation runs with checkpointing off and the cache on, and training with it on: run_steps switches
        # the mode around every rollout and every training pass
        bundle["rollout"] = lambda text, g, max_new, temp: hf_rollout(
            model, tok, render(text), g, max_new, temp
        )
        return bundle
    if not (a.llama_server and a.gguf and a.convert_script):
        raise SystemExit("--path llamacpp needs --llama-server, --gguf and --convert-script")
    server_sha = hashlib.sha256(a.llama_server.read_bytes()).hexdigest()
    gguf_record, convert_record = file_record(a.gguf), file_record(a.convert_script)
    proc, post = start_llama_server(a)

    def export(step: int) -> Path:
        out = a.out / f"adapter-{step}"
        model.save_pretrained(out)
        gguf = out / "adapter.gguf"
        subprocess.run([sys.executable, str(a.convert_script), str(out), "--outfile", str(gguf)], check=True)
        return gguf

    rollout = LlamaCppRollout(post, export, tok, a.timeout, a.gen_timeout)
    bundle["rollout"] = lambda text, g, max_new, temp: rollout(render(text), g, max_new, temp)
    bundle["before_step"] = rollout.reload
    bundle["llama"] = {
        "path": str(a.llama_server),
        "sha256": server_sha,
        "pid": proc.pid,
        "gguf": gguf_record,
        "convert_script": convert_record,
    }
    bundle["close"] = lambda: terminate_and_wait(proc)
    return bundle


# -- the run -------------------------------------------------------------------------------------------------
def is_cut(a: argparse.Namespace) -> bool:
    """Whether the run is not the registered shape: --steps, --prompts, --g or --max-new changed."""
    return a.steps != STEPS or a.prompts != PROMPTS or a.g != G or a.max_new != MAX_NEW_TOKENS


def execute(a: argparse.Namespace, *, build: Callable[[argparse.Namespace], dict], rig=None) -> int:
    """One S2 run: guards, prompts, steps, readout, receipt. ``build`` makes the model bundle; ``rig`` is the
    machine's side (the GPU query, the heartbeat, the sampler), None for a dry rehearsal."""
    rows, gold_hashes = read_gold(a.gold)
    prompts = select_prompts(rows, a.prompts)
    cut = is_cut(a)
    gpu = None
    if a.dry:
        guard = check_exclusive(dry_smi, source="dry: a canned nvidia-smi output, never the real one")
    else:
        gpu = rig.preflight()
        guard = gpu.get("guard")
    bundle = build(a)
    try:
        sampler = VramSampler(rig.vram_query) if rig is not None and not a.dry else None

        def run_it() -> tuple[list[dict], dict | None, bool]:
            return run_steps(
                model=bundle["model"],
                ref_adapter_off=bundle["ref_adapter_off"],
                tok=bundle["tok"],
                optimiser=bundle["optimiser"],
                prompts=prompts,
                rollout=bundle["rollout"],
                steps=a.steps,
                g=a.g,
                max_new=a.max_new,
                reward_fn=bundle["reward_fn"],
                device_sync=bundle["sync"],
                peak_alloc_mib=bundle["peak_alloc_mib"],
                before_step=bundle.get("before_step"),
                mode=bundle.get("mode"),
                state=bundle.get("state"),
            )

        if sampler is not None:
            with sampler:
                steps, stopped, oom = run_it()
        else:
            steps, stopped, oom = run_it()
        peak = sampler.peak_mib if sampler is not None else None
        result = readout(
            steps,
            peak_mib=peak,
            steps_requested=a.steps,
            stopped=stopped,
            oom=oom,
            cut=cut,
            dry=a.dry,
            series=sampler.series if sampler is not None else (),
        )
        line = verdict_line(result, stopped=stopped, cut=cut, dry=a.dry)
        env = environment(
            gpu=gpu,
            dtype="float32 (dry, CPU)" if a.dry else "bfloat16 compute on a 4-bit base",
            batch=a.g,
            exclusive=None if a.dry else True,
            cuda=rig.cuda_version() if rig is not None and not a.dry else None,
            guard=guard,
        )
        receipt = build_receipt(
            path=a.path,
            dry=a.dry,
            args={"steps": a.steps, "prompts": a.prompts, "g": a.g, "max_new": a.max_new, "dry": a.dry},
            rnd_commit=a.rnd_commit,
            gold_hashes=gold_hashes,
            prompt_ids=[r["id"] for r in prompts],
            env=env,
            revision=bundle.get("revision"),
            llama=bundle.get("llama"),
            result=result,
            stopped=stopped,
            verdict=line,
        )
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / f"s2-{a.path}.json").write_text(
            json.dumps(receipt, indent=1), encoding="utf-8", newline="\n"
        )
        (a.out / f"s2-{a.path}-steps.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in steps), encoding="utf-8", newline="\n"
        )
        keys = (
            "steps_completed",
            "peak_vram_mib_sampler",
            "rollout_tok_s",
            "step_s",
            "zero_variance_groups",
            "oom",
        )
        print(json.dumps({"path": a.path, "stopped": stopped, **{k: result[k] for k in keys}}))
        print(line)
        return exit_status(line)
    finally:
        close = bundle.get("close")
        if close:
            close()


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--path", choices=("hf", "llamacpp"), default="hf")
    ap.add_argument("--steps", type=int, default=STEPS, help="a value other than 20 makes the run S2-LIMITED")
    ap.add_argument(
        "--prompts", type=int, default=PROMPTS, help="a value other than 32 makes the run S2-LIMITED"
    )
    ap.add_argument("--g", type=int, default=G, help="a value other than 8 makes the run S2-LIMITED")
    ap.add_argument(
        "--max-new", type=int, default=MAX_NEW_TOKENS, help="a value other than 2048 makes the run S2-LIMITED"
    )
    ap.add_argument("--gold", type=Path, required=True, help="rnd experiments/verifier-gold at --rnd-commit")
    ap.add_argument("--rnd-commit", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dry", action="store_true", help="CPU only: a tiny random model, 2 steps, never CUDA")
    ap.add_argument("--revision", default=STUDENT_REVISION)
    ap.add_argument(
        "--heartbeat", type=Path, default=None, help=f"the watchdog heartbeat file (else ${HEARTBEAT_ENV})"
    )
    ap.add_argument("--llama-server", type=Path, default=None, help="path to the rig's llama-server (path b)")
    ap.add_argument("--gguf", type=Path, default=None, help="the base model's GGUF for the llama.cpp server")
    ap.add_argument("--convert-script", type=Path, default=None, help="llama.cpp's convert_lora_to_gguf.py")
    ap.add_argument("--timeout", type=float, default=10.0, help="seconds per llama.cpp control call")
    ap.add_argument(
        "--ready-timeout", type=float, default=READY_TIMEOUT_S, help="seconds to wait for /health"
    )
    ap.add_argument("--gen-timeout", type=float, default=120.0, help="seconds per llama.cpp generation call")
    return ap


def apply_dry_defaults(a: argparse.Namespace) -> argparse.Namespace:
    """A dry rehearsal is 2 steps of G = 2 on a short completion cap, unless the caller says otherwise."""
    if a.path != "hf":
        raise SystemExit("--dry rehearses the Hugging Face path only")
    a.steps = 2 if a.steps == STEPS else a.steps
    a.g = 2 if a.g == G else a.g
    a.max_new = 24 if a.max_new == MAX_NEW_TOKENS else a.max_new
    return a


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - the real run needs the rig
    a = build_parser().parse_args(argv)
    if a.dry:
        return execute(apply_dry_defaults(a), build=dry_bundle)
    return execute(a, build=real_bundle, rig=Rig(a))


if __name__ == "__main__":
    sys.exit(main())
