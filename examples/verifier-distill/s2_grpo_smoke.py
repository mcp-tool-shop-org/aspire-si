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
import csv
import fnmatch
import gc
import hashlib
import importlib.metadata as metadata
import json
import math
import os
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
from xml.parsers import expat

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
DRY_PREPASS_ITEMS = 4
PREPASS_BASE_SEED = 0
PREPASS_RECORD_KEYS = (
    "id",
    "gold_file",
    "gold_verdict",
    "g",
    "k_correct",
    "supported",
    "unsupported",
    "cannot_tell",
    "malformed",
    "truncations",
    "quote_pass",
    "reward_mean",
    "reward_std",
    "zero_variance",
    "prompt_tokens",
    "completion_tokens_mean",
)
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
        for x in data.decode("utf-8").splitlines():
            if x.strip():
                row = json.loads(x)
                row.setdefault("gold_file", name)
                rows.append(row)
    return rows, hashes


def select_prompts(rows: list[dict], n: int | None = PROMPTS, split: str = PROMPT_SPLIT) -> list[dict]:
    """The rule: the first n rows of the tune split by sorted claim id (every row when n is None)."""
    mine = sorted((r for r in rows if r.get("split") == split), key=lambda r: r["id"])
    if n is None:
        if not mine:
            raise SystemExit(f"the gold has 0 {split} claims; S2 needs at least 1")
        return mine
    if len(mine) < n:
        raise SystemExit(f"the gold has {len(mine)} {split} claims; S2 needs {n}")
    return mine[:n]


def evidence_of(row: dict) -> str:
    return "\n".join(f"[{c.get('source', '')}]\n{c.get('text', '')}" for c in row["context"])


def passages_of(row: dict) -> list[str]:
    """Each passage's text from row['context'] kept as a separate list element (never joined)."""
    return [str(c.get("text", "")) for c in row["context"]]


def build_prompt(row: dict) -> str:
    """The user message: the task, the reply contract, the evidence and the claim."""
    return (
        "Decide whether the evidence supports the claim. First think, then answer with one JSON object:\n"
        '{"verdict": "supported" | "unsupported" | "cannot_tell", "evidence_quotes": [{"part": "<the part of '
        'the claim>", "verdict": "<its verdict>", "quote": "<text copied exactly from the evidence>"}]}\n'
        "Give 1 to 4 evidence_quotes (a simple claim is one part). A quote must appear in the evidence.\n\n"
        f"Evidence:\n{evidence_of(row)}\n\nClaim: {row['claim']}\n"
    )


# -- the reward: the plan's mechanical reward, with offrig rule 2 and optional v2 schedule -------------------
try:
    from quote_rule import (  # noqa: F401
        MIN_LINE_QUOTE_CHARS,
        MIN_QUOTE_CHARS,
        QUOTE_RULE,
        WATERFALL_RUNGS,
        as_passages,
        classify_quote_rung,
        normalise,
        normalise_unmarked,
        quote_credit_v2,
        quote_found,
        quote_is_relevant,
        strip_markers,
    )
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from quote_rule import (  # noqa: F401
        MIN_LINE_QUOTE_CHARS,
        MIN_QUOTE_CHARS,
        QUOTE_RULE,
        WATERFALL_RUNGS,
        as_passages,
        classify_quote_rung,
        normalise,
        normalise_unmarked,
        quote_credit_v2,
        quote_found,
        quote_is_relevant,
        strip_markers,
    )


def split_reply(text: str) -> tuple[str, str]:
    """(thinking, answer): the thinking precedes </think>; an unclosed thinking block has no answer."""
    if "</think>" in text:
        head, _, tail = text.partition("</think>")
        return head.replace("<think>", "").strip(), tail
    return text.replace("<think>", "").strip(), ""


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


def reward_details(
    reply: str,
    gold: str,
    evidence: str | Sequence[str | dict],
    thinking_tokens: int,
    *,
    truncated: bool = False,
    quote_reward: str = "v1",
) -> dict:
    """Detailed breakdown of the reward for logging and diagnostics."""
    if truncated:
        return {
            "total": -1.0,
            "verdict_score": -1.0,
            "quote_credit": 0.0,
            "correct": False,
            "truncated_or_malformed": True,
            "verdict": None,
        }
    parsed = parse_reply(reply)
    if parsed is None:
        return {
            "total": -1.0,
            "verdict_score": -1.0,
            "quote_credit": 0.0,
            "correct": False,
            "truncated_or_malformed": True,
            "verdict": None,
        }
    verdict = parsed["verdict"]
    looped = loop_detected(reply)
    if looped:
        verdict = "cannot_tell"
    passages = as_passages(evidence)
    parts = parsed["evidence_quotes"]
    quotes = [p["quote"] for p in parts]
    found = [quote_found(q, passages) for q in quotes]
    meso = -0.5 if part_rule(parts) != parsed["verdict"] else 0.0
    loop_pen = -0.5 if looped else 0.0

    if quote_reward == "v2":
        k = len(parts)
        m = sum(found)
        eligible_for_credit = False
        if verdict == "supported" and gold == "supported":
            correct = True
            if m == k:
                outcome = 1.0
                eligible_for_credit = True
            elif m > 0:
                outcome = -0.2 + 1.2 * (m / k)
            else:
                outcome = -0.2
        elif verdict == "supported" and gold != "supported":
            correct = False
            outcome = -3.0 if m == k else -3.2
        elif verdict == "unsupported" and gold in ("unsupported", "cannot_tell"):
            correct = True
            given_found = [f for q, f in zip(quotes, found, strict=True) if normalise(q)]
            if not given_found:
                outcome = 1.0
            elif all(given_found):
                outcome = 1.0
                eligible_for_credit = True
            else:
                outcome = 0.8
        elif verdict == "unsupported" and gold == "supported":
            correct = False
            outcome = -1.0
        elif verdict == "cannot_tell" and gold == "cannot_tell":
            correct = True
            outcome = 1.0
        else:
            correct = False
            outcome = -0.3

        if not correct:
            return {
                "total": outcome + meso + loop_pen,
                "verdict_score": outcome,
                "quote_credit": 0.0,
                "correct": False,
                "truncated_or_malformed": False,
                "verdict": verdict,
            }
        budget = 256 if verdict == "unsupported" else 1024
        length = max(-0.5, -0.1 * (thinking_tokens - budget) / budget) if thinking_tokens > budget else 0.0
        penalties = max(meso + length + loop_pen, -1.0)
        extras = (
            min(quote_credit_v2(quotes, passages, verdict=verdict, correct=True), 0.3)
            if eligible_for_credit
            else 0.0
        )
        return {
            "total": outcome + penalties + extras,
            "verdict_score": outcome,
            "quote_credit": extras,
            "correct": True,
            "truncated_or_malformed": False,
            "verdict": verdict,
        }

    if quote_reward != "v1":
        raise SystemExit(f"unknown --quote-reward {quote_reward!r}")

    outcome, correct = macro(verdict, gold)
    unfound_supported = any(p["verdict"] == "supported" and not f for p, f in zip(parts, found, strict=True))
    if verdict == "supported" and unfound_supported:
        outcome, correct = -3.0, False
    if not correct:
        return {
            "total": outcome + meso + loop_pen,
            "verdict_score": outcome,
            "quote_credit": 0.0,
            "correct": False,
            "truncated_or_malformed": False,
            "verdict": verdict,
        }
    budget = 256 if verdict == "unsupported" else 1024
    length = max(-0.5, -0.1 * (thinking_tokens - budget) / budget) if thinking_tokens > budget else 0.0
    penalties = max(meso + length + loop_pen, -1.0)
    extras = min(0.05 * sum(found), 0.3)
    return {
        "total": outcome + penalties + extras,
        "verdict_score": outcome,
        "quote_credit": extras,
        "correct": True,
        "truncated_or_malformed": False,
        "verdict": verdict,
    }


def reward(
    reply: str,
    gold: str,
    evidence: str | Sequence[str | dict],
    thinking_tokens: int,
    *,
    truncated: bool = False,
    quote_reward: str = "v1",
) -> float:
    """The gated reward of the plan (:336-412), with offrig quote rule 2 and optional v2 quote schedule."""
    return float(
        reward_details(
            reply,
            gold,
            evidence,
            thinking_tokens,
            truncated=truncated,
            quote_reward=quote_reward,
        )["total"]
    )


def advantages(rewards: Sequence[float], *, mode: str = "std") -> list[float]:
    """Group advantages. 'std' normalises by group std; 'dr-grpo' subtracts the group mean only."""
    if not rewards:
        return []
    sd = statistics.pstdev(rewards)
    if sd == 0:
        return [0.0] * len(rewards)
    mean = statistics.fmean(rewards)
    if mode == "dr-grpo":
        return [r - mean for r in rewards]
    if mode != "std":
        raise SystemExit(f"unknown --advantage mode {mode!r}")
    return [(r - mean) / (sd + 1e-6) for r in rewards]


def zero_variance(rewards: Sequence[float]) -> bool:
    return len(set(rewards)) <= 1


def classify_zero_variance_group(samples: Sequence[dict], gold: str | None = None) -> str:
    """Classify a zero-variance group into one of the four required diagnostic labels."""
    _ = gold
    if not samples or all(
        bool(s.get("truncated_or_malformed") or s.get("truncated") or s.get("malformed")) for s in samples
    ):
        return "all truncated or malformed"
    if all(bool(s.get("correct")) for s in samples):
        if all(float(s.get("quote_credit") or 0.0) > 0.0 for s in samples):
            return "all correct with full credit"
        return "all correct without quote credit"
    return "all wrong"


def completion_eval_outcome(
    reply: str,
    row: dict,
    *,
    truncated: bool = False,
    reward_val: float = 0.0,
) -> dict:
    """Extract text-free evaluation outcomes for one completion: verdict, per-quote found/missed status,
    waterfall rung, and relevance boolean. Never stores item, claim, or quote text.
    """
    if truncated:
        return {
            "verdict": None,
            "reward": float(reward_val),
            "quote_outcomes": [],
        }
    parsed = parse_reply(reply)
    if parsed is None:
        return {
            "verdict": None,
            "reward": float(reward_val),
            "quote_outcomes": [],
        }
    verdict = parsed["verdict"]
    if loop_detected(reply):
        verdict = "cannot_tell"
    passages = passages_of(row) if "context" in row else as_passages(row.get("passages") or ())
    claim = str(row.get("claim") or "")
    quote_outcomes: list[dict] = []
    for p in parsed["evidence_quotes"]:
        q = p["quote"]
        found = bool(quote_found(q, passages))
        if found:
            quote_outcomes.append(
                {
                    "found": True,
                    "rung": "rule_2",
                    "relevant": bool(quote_is_relevant(q, claim)) if verdict == "supported" else False,
                }
            )
        else:
            quote_outcomes.append(
                {
                    "found": False,
                    "rung": classify_quote_rung(q, passages),
                    "relevant": False,
                }
            )
    return {
        "verdict": verdict,
        "reward": float(reward_val),
        "quote_outcomes": quote_outcomes,
    }


def evaluation_summary(records: Sequence[dict], *, advantage_mode: str = "std") -> dict:
    """Compute per-evaluation diagnostic summary (counts and ratios only; never logs item text or quotes)."""
    total_rollouts = 0
    pred_supported = 0
    gold_supported_rollouts = 0
    cannot_tell_on_gold_supported = 0
    gold_supported_groups = 0
    gold_supported_groups_pos_ct_adv = 0
    false_accepts = 0
    false_rejects = 0
    waterfall: dict[str, int] = {r: 0 for r in (*WATERFALL_RUNGS, "unmatched")}
    relevant_quotes = 0
    total_found_supported_quotes = 0

    for grp in records:
        gold = str(grp.get("gold") or grp.get("label") or "")
        rollouts = list(grp.get("rollouts") or ())
        if rollouts and all("advantage" in r for r in rollouts):
            advs = [float(r["advantage"]) for r in rollouts]
        elif isinstance(grp.get("advantages"), Sequence) and len(grp["advantages"]) == len(rollouts):
            advs = [float(a) for a in grp["advantages"]]
        else:
            rewards = [float(r.get("reward", 0.0)) for r in rollouts]
            advs = advantages(rewards, mode=advantage_mode) if rollouts else []

        if gold == "supported" and rollouts:
            gold_supported_groups += 1
            if any(
                r.get("verdict") == "cannot_tell" and a > 0.0 for r, a in zip(rollouts, advs, strict=True)
            ):
                gold_supported_groups_pos_ct_adv += 1

        claim = str(grp.get("claim") or "")
        passages = (
            as_passages(grp.get("passages") or grp.get("context") or ())
            if ("passages" in grp or "context" in grp)
            else ()
        )

        for r in rollouts:
            total_rollouts += 1
            v = r.get("verdict")
            if v == "supported":
                pred_supported += 1
            if gold == "supported":
                gold_supported_rollouts += 1
                if v == "cannot_tell":
                    cannot_tell_on_gold_supported += 1
                elif v == "unsupported":
                    false_rejects += 1
            else:
                if v == "supported":
                    false_accepts += 1

            q_outcomes = r.get("quote_outcomes")
            if isinstance(q_outcomes, Sequence):
                for qo in q_outcomes:
                    if qo.get("found"):
                        if v == "supported":
                            total_found_supported_quotes += 1
                            if qo.get("relevant"):
                                relevant_quotes += 1
                    else:
                        rung = str(qo.get("rung") or "unmatched")
                        waterfall[rung] = waterfall.get(rung, 0) + 1
                continue

            raw_quotes = r.get("quotes")
            if raw_quotes is None and isinstance(r.get("evidence_quotes"), Sequence):
                raw_quotes = [p.get("quote", "") for p in r["evidence_quotes"] if isinstance(p, dict)]
            for q in raw_quotes or ():
                if quote_found(q, passages):
                    if v == "supported":
                        total_found_supported_quotes += 1
                        if quote_is_relevant(q, claim):
                            relevant_quotes += 1
                else:
                    rung = classify_quote_rung(q, passages)
                    waterfall[rung] = waterfall.get(rung, 0) + 1

    pred_rate = (pred_supported / total_rollouts) if total_rollouts > 0 else 0.0
    prior_rate = (gold_supported_rollouts / total_rollouts) if total_rollouts > 0 else 0.0
    ratio = (pred_rate / prior_rate) if prior_rate > 0 else None
    p_ct_given_sup = (
        (cannot_tell_on_gold_supported / gold_supported_rollouts) if gold_supported_rollouts > 0 else 0.0
    )
    pos_ct_share = (
        (gold_supported_groups_pos_ct_adv / gold_supported_groups) if gold_supported_groups > 0 else 0.0
    )
    rel_share = (relevant_quotes / total_found_supported_quotes) if total_found_supported_quotes > 0 else 0.0
    return {
        "supported_pred_to_prior_ratio": ratio,
        "p_cannot_tell_given_gold_supported": p_ct_given_sup,
        "gold_supported_groups_with_positive_cannot_tell_advantage": pos_ct_share,
        "false_accepts": false_accepts,
        "false_rejects": false_rejects,
        "miss_waterfall": waterfall,
        "quote_relevance": {
            "relevant": relevant_quotes,
            "total": total_found_supported_quotes,
            "share": rel_share,
        },
    }


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
    flags: dict | None = None,
    phase_logger: PhaseLogger | None = None,
    eval_summary_record: dict | None = None,
    rollout_only: bool = False,
    items_requested: int | None = None,
    items_completed: int | None = None,
    prepass_summary_record: dict | None = None,
) -> dict:
    """The pass decision, per path. A run that was cut, stopped or dry never passes.

    whole = the registered shape ran: no --steps/--prompts/--g cut, and not a dry rehearsal.
    pass  = whole, no stop, no OOM, every step completed, and the sampler's peak under the watchdog line.
    """

    completed = len(steps)
    whole = not cut and not dry and steps_requested == STEPS and not rollout_only
    step_s = [s["step_s"] for s in steps]
    tokens = sum(s["tokens"] for s in steps)
    rollout_s = sum(s["rollout_s"] for s in steps)
    fits = peak_mib is not None and peak_mib < VRAM_LINE_MIB
    complete = completed == steps_requested and stopped is None and not oom
    if phase_logger is not None and phase_logger.max_memory_allocated is not None:
        peak_alloc_torch = phase_logger.peak_alloc_mib()
    else:
        peak_alloc_torch = max((s.get("peak_alloc_mib") or 0 for s in steps), default=None) if steps else None
    if phase_logger is not None and phase_logger.max_memory_reserved is not None:
        peak_reserved_torch = phase_logger.peak_reserved_mib()
    else:
        reserved_vals = [s["peak_reserved_mib"] for s in steps if s.get("peak_reserved_mib") is not None]
        peak_reserved_torch = max(reserved_vals) if reserved_vals else None
    out = {
        "steps_requested": steps_requested,
        "steps_completed": completed,
        "peak_vram_mib_sampler": peak_mib,
        "sampler_samples": len(series),
        "vram_mib_sampler_series": list(series),
        "peak_vram_mib_torch": peak_alloc_torch,
        "peak_reserved_mib_torch": peak_reserved_torch,
        "fits": fits,
        "oom": oom,
        "rollout_tok_s": round(tokens / rollout_s, 2) if rollout_s > 0 else None,
        "step_s": {"p50": statistics.median(step_s), "max": max(step_s)} if step_s else None,
        "wall_s": round(sum(step_s), 2),
        "zero_variance_groups": sum(s["zero_variance_groups"] for s in steps),
        "active_groups": sum(s.get("active_groups", s["groups"] - s["zero_variance_groups"]) for s in steps),
        "groups": sum(s["groups"] for s in steps),
        "complete": complete and whole,
        "pass": bool(complete and whole and fits),
    }
    if flags:
        out["flags"] = dict(flags)
        if out["flags"].get("stress_prefix", 0) > 0:
            out["flags"]["stress_prefix_masks"] = "off"
    if rollout_only or (isinstance(out.get("flags"), dict) and out["flags"].get("rollout_only")):
        req_items = items_requested if items_requested is not None else steps_requested
        comp_items = items_completed if items_completed is not None else completed
        out["items_requested"] = req_items
        out["items_completed"] = comp_items
        out["ignored_flags"] = ["prompts", "steps"]
        out["prompts_ignored"] = True
        out["steps_ignored"] = True
        out["complete"] = bool(comp_items == req_items and stopped is None and not oom)
        out["pass"] = False
        out.setdefault("flags", {})["rollout_only"] = True
        out["flags"]["ignored_flags"] = ["prompts", "steps"]
        if prepass_summary_record is not None:
            out["prepass_summary"] = prepass_summary_record
    if eval_summary_record is not None:
        qr_rel = eval_summary_record.get("quote_relevance") or {}
        out["eval_summary"] = {
            "supported_pred_to_prior_ratio": eval_summary_record.get("supported_pred_to_prior_ratio"),
            "p_cannot_tell_given_gold_supported": eval_summary_record.get(
                "p_cannot_tell_given_gold_supported"
            ),
            "gold_supported_groups_with_positive_cannot_tell_advantage": eval_summary_record.get(
                "gold_supported_groups_with_positive_cannot_tell_advantage"
            ),
            "false_accepts": eval_summary_record.get("false_accepts"),
            "false_rejects": eval_summary_record.get("false_rejects"),
            "miss_waterfall": dict(eval_summary_record.get("miss_waterfall") or {}),
            "quote_relevance_share": qr_rel.get("share"),
        }
    return out


def verdict_line(
    result: dict, *, stopped: dict | None, cut: bool, dry: bool, rollout_only: bool = False
) -> str:
    """S2-OK only for a whole run that fits. S2-DRY: a rehearsal. S2-LIMITED: a cut. S2-STOPPED: a stop.
    S2-PREPASS: a finished rollout-only pre-pass.
    """
    if rollout_only or (isinstance(result.get("flags"), dict) and result["flags"].get("rollout_only")):
        if stopped:
            return "S2-STOPPED"
        req = result.get("items_requested", result.get("steps_requested"))
        comp = result.get("items_completed", result.get("steps_completed"))
        if (
            result.get("oom")
            or comp != req
            or (not dry and result.get("peak_vram_mib_sampler") is not None and not result.get("fits"))
        ):
            return "S2-STOPPED" if dry else "S2-FAIL"
        return "S2-PREPASS"
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
    """0 for S2-OK, S2-PREPASS and for a finished rehearsal; anything else is non-zero."""
    return 0 if line in ("S2-OK", "S2-DRY", "S2-PREPASS") else 1


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
    "llama-cli*",
    "llama-bench*",
    "uv*",
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


DOCTYPE_SYSTEM_ID = re.compile(r"nvsmi_device_v[0-9]+\.dtd")  # the rig's driver writes v13 (2026-11-03)


def check_report_markup(xml_text: str) -> None:
    """Refuse markup a real ``nvidia-smi -q -x`` report never needs, and name what was seen.

    Accepted: the XML declaration, comments anywhere, and **either no DOCTYPE or exactly** ``<!DOCTYPE
    nvidia_smi_log SYSTEM "nvsmi_device_v<digits>.dtd">`` (digits only, so a driver update from v13 to v14
    still passes) with no internal subset. Refused: a different root name in the DOCTYPE, a PUBLIC
    identifier, a different DTD file-name pattern (``v13a``, no digits, a path such as ``../x.dtd``), an
    internal subset (which is the only place an ENTITY declaration can sit), an entity reference, and any
    processing instruction other than the XML declaration.

    The external DTD is never opened or fetched: this uses expat (pyexpat) and ElementTree, and neither is
    given an external-entity handler, so a SYSTEM identifier is read as text and nothing else. A test
    proves it."""

    def refuse(why: str) -> None:
        raise SystemExit(f"refusing to start: nvidia_smi_garbage: {why}")

    def doctype(name: str, system_id: str | None, public_id: str | None, has_internal_subset: int) -> None:
        if has_internal_subset:
            refuse("the DOCTYPE has an internal subset (an ENTITY or any other declaration)")
        if name != "nvidia_smi_log":
            refuse(f"the DOCTYPE names {name!r}, not nvidia_smi_log")
        if public_id is not None:
            refuse("the DOCTYPE has a PUBLIC identifier")
        if system_id is None or not DOCTYPE_SYSTEM_ID.fullmatch(system_id):
            refuse(f"the DOCTYPE's DTD is {system_id!r}, not nvsmi_device_v<digits>.dtd")

    parser = expat.ParserCreate()
    parser.StartDoctypeDeclHandler = doctype
    parser.ProcessingInstructionHandler = lambda target, data: refuse(
        f"a processing instruction <?{target}?>"
    )
    parser.SkippedEntityHandler = lambda name, is_param: refuse(f"an entity reference &{name};")
    try:
        parser.Parse(xml_text, True)
    except expat.ExpatError as exc:
        raise SystemExit(
            f"refusing to start: nvidia_smi_garbage: the process list is not XML ({exc})"
        ) from exc


PROCESS_INFO_FIELDS = (
    "gpu_instance_id",
    "compute_instance_id",
    "pid",
    "type",
    "process_name",
    "used_memory",
)  # the elements the rig's real report carries in every <process_info> (26 of 26, 2026-11-03)


def _blank(text: str | None) -> bool:
    return not (text or "").strip()


def parse_gpu_processes(xml_text: str) -> list[dict]:
    """The processes from ``nvidia-smi -q -x``: ``[{"image": <file name>, "type": "C" | "G" | "C+G"}]``.

    That XML carries each process's type and name (the compute-apps query has no type and, on Windows, no
    usable memory figure). Output that is not that XML is a refusal, never an empty list.

    The ``<processes>`` subtree is walked against an allowlist, so nothing can hide in it. Exactly one
    ``<gpu>`` (at any depth) and exactly one ``<processes>`` (at any depth), directly under that
    ``<gpu>``, and every ``<process_info>`` in the report sits under it. Its only children are
    ``<process_info>``; each ``<process_info>``'s only children are the names in ``PROCESS_INFO_FIELDS``,
    each at most once, none with element children of its own, and ``type`` exactly once. Any other
    element, a repeat, a nested element, or non-whitespace text or tail anywhere in the subtree refuses,
    naming what was seen. The rest of the ``<gpu>`` block (hundreds of elements) is not constrained.
    ``type`` is read as C, G or C+G, case and padding ignored, and anything else refuses. Only a present,
    truly empty ``<processes/>`` (comments allowed) means "nothing on the card". The markup is checked
    first (``check_report_markup``).

    If the real rig ever returns a refused shape, it is amended by a card with the real fixture, never by
    relaxing this parser."""
    check_report_markup(xml_text)
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise SystemExit(
            f"refusing to start: nvidia_smi_garbage: the process list is not XML ({exc})"
        ) from exc

    def refuse(why: str) -> None:
        raise SystemExit(f"refusing to start: nvidia_smi_garbage: {why}")

    if root.tag != "nvidia_smi_log":
        refuse(f"unexpected root element {root.tag!r}")
    gpus = root.findall("gpu")
    if not gpus:
        refuse("no <gpu> element")
    every_gpu = len(list(root.iter("gpu")))
    if len(gpus) > 1 or every_gpu > 1:
        refuse(f"more than one <gpu> element ({every_gpu})")
    every_processes = len(list(root.iter("processes")))
    listings = gpus[0].findall("processes")
    if not listings:
        refuse("the <gpu> has no <processes> element")
    if len(listings) > 1 or every_processes > 1:
        refuse(f"more than one <processes> element ({every_processes})")
    listing = listings[0]
    parents = {child: parent for parent in root.iter() for child in parent}
    for info in root.iter("process_info"):
        if parents.get(info) is not listing:
            where = parents[info].tag if info in parents else "the root"
            refuse(f"a <process_info> sits under <{where}>, not under the one <processes>")
    if not _blank(listing.text):
        refuse(f"<processes> holds text {listing.text.strip()!r} outside any <process_info>")
    procs = []
    for info in listing:
        if info.tag != "process_info":
            refuse(f"<processes> holds a <{info.tag}> child, not process_info")
        if not _blank(info.tail):
            refuse(f"text {info.tail.strip()!r} after a </process_info>")
        if not _blank(info.text):
            refuse(f"a <process_info> holds text {info.text.strip()!r} outside its fields")
        fields: dict[str, ET.Element] = {}
        for field in info:
            if field.tag not in PROCESS_INFO_FIELDS:
                refuse(f"a <process_info> holds an unexpected <{field.tag}> element")
            if field.tag in fields:
                refuse(f"a <process_info> repeats <{field.tag}>")
            if len(field):
                refuse(f"a <process_info>'s <{field.tag}> has a nested <{field[0].tag}> element")
            if not _blank(field.tail):
                refuse(f"text {field.tail.strip()!r} after a <{field.tag}> in a <process_info>")
            fields[field.tag] = field
        if "type" not in fields:
            refuse("a <process_info> has no <type>")
        ptype = (fields["type"].text or "").strip().upper()
        if ptype not in ("C", "G", "C+G"):
            refuse(f"unknown process type {ptype!r}")
        name = fields["process_name"].text if "process_name" in fields else ""
        procs.append({"image": image_name(name or ""), "type": ptype})
    return procs


def parse_memory_used(text: str) -> int:
    """``memory.used`` in MiB from ``--query-gpu=memory.used --format=csv,noheader,nounits``.

    Exactly one line (the guard allows one GPU) of 1 to 9 ASCII digits (a bounded regex, not ``isdigit``,
    so ``int`` never sees a long string); anything else refuses by name. Rule (b) reads only this source;
    the XML's ``fb_memory_usage`` is not read."""
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    if not lines:
        raise SystemExit("refusing to start: nvidia_smi_garbage: no memory.used reading")
    if len(lines) != 1:
        raise SystemExit(
            f"refusing to start: nvidia_smi_garbage: {len(lines)} memory.used readings for one GPU"
        )
    if not re.fullmatch(r"[0-9]{1,9}", lines[0]):
        raise SystemExit(
            "refusing to start: nvidia_smi_garbage: "
            f"memory.used is {lines[0][:40]!r}, not a MiB integer of 1 to 9 digits"
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


# The canned desktop for --dry: 26 processes of type C+G, OS and shell image names only (the same multiset as
# the public fixture tests/fixtures/nvidia-smi-q-x-fixture-public-2026-11-03.xml; a test checks they agree).
# Never put a personal application's name in a public file.
DRY_DESKTOP_IMAGES = (
    "ShellHost.exe WindowsTerminal.exe explorer.exe CrossDeviceResume.exe TextInputHost.exe "
    "StartMenuExperienceHost.exe SearchHost.exe WidgetBoard.exe msedgewebview2.exe explorer.exe "
    "ApplicationFrameHost.exe msedgewebview2.exe ShellHost.exe SearchHost.exe StartMenuExperienceHost.exe "
    "WindowsTerminal.exe msedgewebview2.exe msedgewebview2.exe ShellExperienceHost.exe "
    "TextInputHost.exe ApplicationFrameHost.exe ShellExperienceHost.exe SystemSettings.exe "
    "CrossDeviceResume.exe TextInputHost.exe msedgewebview2.exe "
).split()
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
    dtype_census_record: dict | None = None,
) -> dict:
    out = {
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
    if dtype_census_record is not None:
        out["dtype_census"] = dtype_census_record
    return out


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
_ALLOC_CONF: str | None = None
_ALLOC_CONF_VERIFIED_TORCH: object | None = None


def configure_alloc_conf(mode: str | None) -> None:
    """Configure PYTORCH_CUDA_ALLOC_CONF to be applied before torch is imported in ``require_torch``."""
    global _ALLOC_CONF, _ALLOC_CONF_VERIFIED_TORCH
    if mode is not None and mode != "expandable":
        raise SystemExit(f"refusing to start: unknown --alloc-conf {mode!r}")
    _ALLOC_CONF = mode
    _ALLOC_CONF_VERIFIED_TORCH = None


def expandable_segments_supported(torch_mod) -> bool:
    """Verify ``expandable_segments:True`` via a 1 MiB CUDA allocation and allocator settings readback."""
    try:
        cuda = getattr(torch_mod, "cuda", None)
        is_avail = getattr(cuda, "is_available", None)
        if not callable(is_avail) or not is_avail():
            raise SystemExit("refusing to start: expandable segments check failed: CUDA is not available")
        probe = torch_mod.empty(
            1024 * 1024,
            dtype=getattr(torch_mod, "uint8", None),
            device="cuda",
        )
        del probe
        empty_cache = getattr(cuda, "empty_cache", None)
        if callable(empty_cache):
            empty_cache()
        settings = torch_mod._C._accelerator_getAllocatorSettings()
    except SystemExit:
        raise
    except Exception as exc:
        raise SystemExit(f"refusing to start: expandable segments check failed: {exc}") from exc
    if not isinstance(settings, str) or "expandable_segments:true" not in settings.lower():
        raise SystemExit(
            "refusing to start: expandable segments check failed: "
            f"allocator settings {settings!r} do not enable expandable_segments"
        )
    return True


def require_torch():
    """Import torch. Absent: a clear refusal. Present but broken (a DLL, an ABI clash): the real error."""
    global _ALLOC_CONF_VERIFIED_TORCH
    if _ALLOC_CONF == "expandable":
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    try:
        import torch
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise SystemExit(
                "torch is not installed in this Python; install it, or run the pure parts only"
            ) from exc
        raise
    if _ALLOC_CONF == "expandable" and _ALLOC_CONF_VERIFIED_TORCH is not torch:
        expandable_segments_supported(torch)
        _ALLOC_CONF_VERIFIED_TORCH = torch
    return torch


def dtype_census(model) -> dict:
    """Count parameters by dtype, split into trainable and non_trainable."""
    trainable: dict[str, int] = {}
    non_trainable: dict[str, int] = {}
    for _, param in model.named_parameters():
        key = str(param.dtype).removeprefix("torch.")
        bucket = trainable if param.requires_grad else non_trainable
        bucket[key] = bucket.get(key, 0) + 1
    return {"trainable": trainable, "non_trainable": non_trainable}


def restore_bf16(model) -> dict:
    """Cast non-trainable floating-point parameters (except Params4bit) back to bfloat16; keep LoRA fp32."""
    torch = require_torch()
    for _, param in model.named_parameters():
        if not param.requires_grad and param.is_floating_point() and param.__class__.__name__ != "Params4bit":
            param.data = param.data.to(torch.bfloat16)
    return dtype_census(model)


def cache_kv_tensors(cache) -> tuple[list, list]:
    """Extract (keys, values) tensor lists from a Hugging Face cache object."""
    if cache is None:
        return [], []
    if hasattr(cache, "layers"):
        keys = [layer.keys for layer in cache.layers if getattr(layer, "keys", None) is not None]
        values = [layer.values for layer in cache.layers if getattr(layer, "values", None) is not None]
        return keys, values
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        return list(cache.key_cache), list(cache.value_cache)
    if isinstance(cache, Sequence):
        return [pair[0] for pair in cache], [pair[1] for pair in cache]
    return [], []


def completion_logprobs(model, input_ids, attention_mask, prompt_len: int):
    """Log-probability of each completion token given what precedes it: (batch, completion_len)."""
    torch = require_torch()
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :].float()
    logp = torch.log_softmax(logits, dim=-1).gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    return logp[:, prompt_len - 1 :]


def completion_logprobs_microbatch(model, input_ids, attention_mask, prompt_len: int, microbatch: int):
    """Completion token log-probs in micro-batches of ``microbatch`` using ``logits_to_keep`` and
    ``gather - logsumexp`` (never a full ``log_softmax`` tensor)."""
    if microbatch < 1:
        raise SystemExit(f"--logprob-microbatch must be >= 1, got {microbatch}")
    torch = require_torch()
    batch, seq_len = input_ids.shape
    comp_len = seq_len - prompt_len
    keep = comp_len + 1
    chunks = []
    for start in range(0, batch, microbatch):
        end = min(start + microbatch, batch)
        mb_ids = input_ids[start:end]
        mb_att = attention_mask[start:end]
        out = model(input_ids=mb_ids, attention_mask=mb_att, logits_to_keep=keep)
        logits = out.logits[:, -keep:-1, :].float()
        labels = mb_ids[:, prompt_len:]
        token_logp = logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(logits, dim=-1)
        chunks.append(token_logp)
    return torch.cat(chunks, dim=0)


def _active_group_row_mask(adv, group_size: int | None = None):
    """Return a (batch, 1) float32 mask that is 1.0 for rows in non-zero-variance groups and 0.0 otherwise."""
    torch = require_torch()
    batch = int(adv.shape[0])
    g_sz = batch if group_size is None or group_size <= 0 else int(group_size)
    row_mask = torch.zeros((batch, 1), dtype=torch.float32, device=adv.device)
    for g_start in range(0, batch, g_sz):
        g_end = min(g_start + g_sz, batch)
        grp_adv = adv[g_start:g_end]
        if bool(torch.any(grp_adv != 0)):
            row_mask[g_start:g_end, 0] = 1.0
    return row_mask


def grpo_loss(
    logp,
    old_logp,
    ref_logp,
    adv,
    mask,
    *,
    beta: float = BETA,
    eps: float = EPSILON,
    zero_variance_mask: bool = False,
    group_size: int | None = None,
):
    """The clipped-ratio policy loss plus a k3 KL penalty to the reference, masked, averaged over tokens."""
    torch = require_torch()
    eff_mask = mask
    if zero_variance_mask:
        eff_mask = mask * _active_group_row_mask(adv, group_size)
        if float(eff_mask.sum().detach()) == 0.0:
            return logp.sum() * 0.0
    ratio = torch.exp(logp - old_logp)
    a = adv.unsqueeze(-1)
    policy = -torch.min(ratio * a, torch.clamp(ratio, 1 - eps, 1 + eps) * a)
    delta = ref_logp - logp
    kl = torch.exp(delta) - delta - 1
    return ((policy + beta * kl) * eff_mask).sum() / eff_mask.sum().clamp(min=1)


def grpo_microbatch_backward(
    model,
    input_ids,
    attention_mask,
    prompt_len: int,
    old_logp,
    ref_logp,
    adv,
    mask,
    *,
    microbatch: int,
    scale: float = 1.0,
    beta: float = BETA,
    eps: float = EPSILON,
    after_forward: Callable[[], None] | None = None,
    zero_variance_mask: bool = False,
    group_size: int | None = None,
):
    """Run training forward and backward over ``G`` sequences in micro-batches of ``microbatch`` with
    gradients accumulated, using ``logits_to_keep`` and ``gather - logsumexp``."""
    if microbatch < 1:
        raise SystemExit(f"--logprob-microbatch must be >= 1, got {microbatch}")
    torch = require_torch()
    batch, seq_len = input_ids.shape
    comp_len = seq_len - prompt_len
    keep = comp_len + 1
    eff_mask = mask * _active_group_row_mask(adv, group_size) if zero_variance_mask else mask
    if zero_variance_mask and float(eff_mask.sum().detach()) == 0.0:
        if after_forward is not None:
            after_forward()
        return torch.zeros((), dtype=torch.float32, device=input_ids.device)
    denom = eff_mask.sum().clamp(min=1)
    total_loss = torch.zeros((), dtype=torch.float32, device=input_ids.device)
    for start in range(0, batch, microbatch):
        end = min(start + microbatch, batch)
        mb_mask = eff_mask[start:end]
        if zero_variance_mask and float(mb_mask.sum().detach()) == 0.0:
            if end == batch and after_forward is not None:
                after_forward()
            continue
        mb_ids = input_ids[start:end]
        mb_att = attention_mask[start:end]
        out = model(input_ids=mb_ids, attention_mask=mb_att, logits_to_keep=keep)
        logits = out.logits[:, -keep:-1, :].float()
        labels = mb_ids[:, prompt_len:]
        mb_logp = logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(logits, dim=-1)
        mb_old = old_logp[start:end]
        mb_ref = ref_logp[start:end]
        mb_adv = adv[start:end].unsqueeze(-1)
        ratio = torch.exp(mb_logp - mb_old)
        policy = -torch.min(ratio * mb_adv, torch.clamp(ratio, 1 - eps, 1 + eps) * mb_adv)
        delta = mb_ref - mb_logp
        kl = torch.exp(delta) - delta - 1
        mb_loss = ((policy + beta * kl) * mb_mask).sum() / denom
        if end == batch and after_forward is not None:
            after_forward()
        (mb_loss * scale).backward()
        total_loss = total_loss + mb_loss.detach()
    return total_loss * scale


def _eos_token_ids(tok) -> set[int]:
    eos = getattr(tok, "eos_token_id", None)
    if isinstance(eos, int):
        return {eos}
    if isinstance(eos, Sequence):
        return {int(x) for x in eos}
    return set()


def hf_rollout(
    model,
    tok,
    prompt_text: str,
    g: int,
    max_new: int,
    temperature: float,
    *,
    pass_attention_mask: bool = False,
    min_new: int | None = None,
):
    """Path (a): Hugging Face generate, G samples of one prompt from the model that is being trained."""
    torch = require_torch()
    enc = tok(prompt_text, return_tensors="pt")
    prompt_ids = enc["input_ids"].to(model.device)
    repeated = prompt_ids.repeat(g, 1)
    gen_kwargs: dict = {
        "do_sample": True,
        "temperature": temperature,
        "max_new_tokens": max_new,
        "pad_token_id": tok.pad_token_id,
    }
    if min_new is not None:
        gen_kwargs["min_new_tokens"] = min_new
    if pass_attention_mask:
        gen_kwargs["attention_mask"] = torch.ones_like(repeated)
    with torch.no_grad():
        out = model.generate(repeated, **gen_kwargs)
    eos_ids = _eos_token_ids(tok)
    comps: list[list[int]] = []
    truncated: list[bool] = []
    for row in out[:, prompt_ids.shape[1] :]:
        ids = row.tolist()
        while ids and ids[-1] == tok.pad_token_id and tok.pad_token_id not in eos_ids:
            ids.pop()
        hit_cap = len(ids) >= max_new and (not eos_ids or not any(t in eos_ids for t in ids))
        comps.append(ids)
        truncated.append(hit_cap)
    return prompt_ids[0].tolist(), comps, truncated


PHASE_BOUNDARIES = (
    "pre_rollout",
    "post_rollout",
    "post_ref",
    "post_train_forward",
    "post_train_backward",
    "step_end",
)


class GroupLogger:
    """Per-group diagnostic JSONL logger (always on for GPU runs; a no-op on CPU and dry runs).

    Flushes one JSONL line per group with step, group, prompt_id, prompt_tokens, k_correct, g,
    reward_std, verdict_reward_std, truncations, mean_thinking_tokens, and zero_variance_label.
    Never records item text, quotes or claim text.
    """

    def __init__(
        self,
        path: Path | None,
        *,
        enabled: bool,
        cuda=None,
        fsync: Callable[[int], None] = os.fsync,
    ) -> None:
        self.path = path
        cuda_ready = (
            cuda is not None and callable(getattr(cuda, "is_available", None)) and bool(cuda.is_available())
        )
        self.enabled = bool(enabled and path is not None and cuda_ready)
        self.fsync = fsync
        self._fh = None
        if self.enabled and self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.path.open("w", encoding="utf-8", newline="\n")

    def record(
        self,
        *,
        step: int,
        group: int,
        prompt_id: str,
        prompt_tokens: int,
        k_correct: int,
        g: int,
        reward_std: float,
        verdict_reward_std: float,
        truncations: int,
        mean_thinking_tokens: float,
        zero_variance_label: str | None,
    ) -> None:
        if not self.enabled or self._fh is None:
            return
        line = {
            "step": int(step),
            "group": int(group),
            "prompt_id": str(prompt_id),
            "prompt_tokens": int(prompt_tokens),
            "k_correct": int(k_correct),
            "g": int(g),
            "reward_std": round(float(reward_std), 6),
            "verdict_reward_std": round(float(verdict_reward_std), 6),
            "truncations": int(truncations),
            "mean_thinking_tokens": round(float(mean_thinking_tokens), 2),
            "zero_variance_label": zero_variance_label,
        }
        self._fh.write(json.dumps(line) + "\n")
        self._fh.flush()
        self.fsync(self._fh.fileno())

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


class PhaseLogger:
    """Phase-boundary CUDA allocator logger (always on for GPU runs; a no-op on CPU and dry runs).

    Each boundary records allocator bytes, resets peak stats, and flushes + fsyncs the JSONL line so a kill
    mid-phase leaves the last completed boundary on disk. Never records process names, app names or paths.
    """

    def __init__(
        self,
        path: Path | None,
        *,
        enabled: bool,
        cuda=None,
        now: Callable[[], float] = time.monotonic,
        fsync: Callable[[int], None] = os.fsync,
    ) -> None:
        self.path = path
        cuda_ready = (
            cuda is not None and callable(getattr(cuda, "is_available", None)) and bool(cuda.is_available())
        )
        self.enabled = bool(enabled and path is not None and cuda_ready)
        self.cuda = cuda
        self.now = now
        self.fsync = fsync
        self.max_memory_allocated: int | None = None
        self.max_memory_reserved: int | None = None
        self._fh = None
        if self.enabled and self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.path.open("w", encoding="utf-8", newline="\n")

    def peak_alloc_mib(self) -> int | None:
        return int(self.max_memory_allocated / 2**20) if self.max_memory_allocated is not None else None

    def peak_reserved_mib(self) -> int | None:
        return int(self.max_memory_reserved / 2**20) if self.max_memory_reserved is not None else None

    def record(
        self,
        *,
        step: int,
        group: int,
        phase: str,
        prompt_tokens: int,
        completion_lengths: Sequence[int],
    ) -> None:
        if not self.enabled or self._fh is None or self.cuda is None:
            return
        alloc = int(self.cuda.memory_allocated())
        reserved = int(self.cuda.memory_reserved())
        max_alloc = int(self.cuda.max_memory_allocated())
        max_reserved = int(self.cuda.max_memory_reserved())
        self.max_memory_allocated = (
            max_alloc if self.max_memory_allocated is None else max(self.max_memory_allocated, max_alloc)
        )
        self.max_memory_reserved = (
            max_reserved if self.max_memory_reserved is None else max(self.max_memory_reserved, max_reserved)
        )
        self.cuda.reset_peak_memory_stats()
        lengths = list(completion_lengths)
        comp = {
            "min": min(lengths) if lengths else 0,
            "max": max(lengths) if lengths else 0,
            "sum": sum(lengths),
        }
        line = {
            "step": step,
            "group": group,
            "phase": phase,
            "monotonic_s": round(self.now(), 6),
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": comp,
            "memory_allocated": alloc,
            "memory_reserved": reserved,
            "max_memory_allocated": max_alloc,
            "max_memory_reserved": max_reserved,
        }
        self._fh.write(json.dumps(line) + "\n")
        self._fh.flush()
        self.fsync(self._fh.fileno())

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


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
def _call_reward_fn(reward_fn: Callable, text: str, row: dict, think_toks: int, truncated: bool) -> float:
    try:
        return float(reward_fn(text, row, think_toks, truncated=truncated))
    except TypeError:
        return float(reward_fn(text, row, think_toks))


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
    reward_fn: Callable,
    device_sync: Callable[[], None],
    peak_alloc_mib: Callable[[], int | None],
    on_step: Callable[[dict], None] | None = None,
    before_step: Callable[[int], None] | None = None,
    mode: Callable[[bool], None] | None = None,
    state: Callable[[], dict] | None = None,
    logprob_microbatch: int | None = None,
    empty_cache_phases: bool = False,
    empty_cache: Callable[[], None] | None = None,
    phase_logger: PhaseLogger | None = None,
    group_logger: GroupLogger | None = None,
    groups_per_update: int = GROUPS_PER_STEP,
    zero_variance_mask: bool = False,
    advantage_mode: str = "std",
    mask_truncated: bool = False,
    quote_reward: str = "v1",
    eval_records: list[dict] | None = None,
    on_eval: Callable[[dict], None] | None = None,
) -> tuple[list[dict], dict | None, bool]:
    """Run GRPO steps. Returns (step rows, a stop record or None, whether it ran out of memory).

    Each step is ``groups_per_update`` prompt groups of G samples, one group at a time (gradients accumulate),
    so peak memory is one group's. The reference is the same model with the adapter off.

    ``mode(False)`` puts the model in rollout mode (gradient checkpointing off, the KV cache on) and
    ``mode(True)`` in training mode (checkpointing on, cache off, as the plan's memory table assumes);
    ``state()`` reports whether checkpointing is on. Both states are recorded in every step row.
    """
    if groups_per_update < 1:
        raise SystemExit(f"--groups-per-update must be >= 1, got {groups_per_update}")

    torch = require_torch()
    if empty_cache is None:
        cuda_mod = getattr(torch, "cuda", None)
        empty_cache = getattr(cuda_mod, "empty_cache", lambda: None)
    eos_ids = _eos_token_ids(tok)
    rows: list[dict] = []
    oom_types = tuple(t for t in (getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None),) if t)
    for step in range(steps):
        try:
            started = time.time()
            if before_step:
                before_step(step)
            roll_s = train_s = 0.0
            tokens = zero_var = active_groups = 0
            rewards_all: list[float] = []
            optimiser.zero_grad(set_to_none=True)
            last_prompt_tokens = 0
            last_comp_lens: list[int] = []
            step_loss_sum = 0.0
            ckpt_train = ckpt_rollout = None
            for k in range(groups_per_update):
                row = prompts[(step * groups_per_update + k) % len(prompts)]
                prompt_text = build_prompt(row)
                if mode:
                    mode(False)
                ckpt_rollout = state()["grad_ckpt"] if state else None
                if phase_logger is not None:
                    phase_logger.record(
                        step=step,
                        group=k,
                        phase="pre_rollout",
                        prompt_tokens=len(tok.encode(prompt_text)),
                        completion_lengths=(),
                    )
                t0 = time.time()
                roll_out = rollout(prompt_text, g, max_new, TEMPERATURE)
                if len(roll_out) == 3:
                    prompt_ids, comps, truncs = roll_out
                    truncs = [bool(t) for t in truncs]
                else:
                    prompt_ids, comps = roll_out
                    truncs = [
                        len(c) >= max_new and (not eos_ids or not any(t in eos_ids for t in c)) for c in comps
                    ]
                device_sync()
                roll_s += time.time() - t0
                comp_lens = [len(c) for c in comps]
                last_prompt_tokens = len(prompt_ids)
                last_comp_lens = comp_lens
                if empty_cache_phases and empty_cache is not None:
                    empty_cache()
                if phase_logger is not None:
                    phase_logger.record(
                        step=step,
                        group=k,
                        phase="post_rollout",
                        prompt_tokens=len(prompt_ids),
                        completion_lengths=comp_lens,
                    )
                tokens += sum(comp_lens)
                rewards: list[float] = []
                think_counts: list[int] = []
                group_samples: list[dict] = []
                verdict_scores: list[float] = []
                rollout_outcomes: list[dict] = []
                for c, trunc in zip(comps, truncs, strict=True):
                    text = tok.decode(c, skip_special_tokens=True)
                    think, _ = split_reply(text)
                    tt = len(tok.encode(think)) if think else 0
                    think_counts.append(tt)
                    r_val = _call_reward_fn(reward_fn, text, row, tt, trunc)
                    rewards.append(r_val)
                    if eval_records is not None:
                        rollout_outcomes.append(
                            completion_eval_outcome(text, row, truncated=trunc, reward_val=r_val)
                        )
                    if group_logger is not None and group_logger.enabled:
                        det = reward_details(
                            text,
                            str(row.get("label", "supported")),
                            passages_of(row) if "context" in row else "",
                            tt,
                            truncated=trunc,
                            quote_reward=quote_reward,
                        )
                        group_samples.append(det)
                        verdict_scores.append(float(det["verdict_score"]))

                if mask_truncated:
                    active_rewards = [r for r, tr in zip(rewards, truncs, strict=True) if not tr]
                    is_zv = zero_variance(active_rewards)
                    active_adv = advantages(active_rewards, mode=advantage_mode)
                    adv_iter = iter(active_adv)
                    adv_list = [0.0 if tr else next(adv_iter) for tr in truncs]
                else:
                    is_zv = zero_variance(rewards)
                    adv_list = advantages(rewards, mode=advantage_mode)

                if eval_records is not None:
                    for ro_item, adv_val in zip(rollout_outcomes, adv_list, strict=True):
                        ro_item["advantage"] = float(adv_val)
                    eval_records.append(
                        {
                            "step": step,
                            "group": k,
                            "gold": str(row.get("label", "supported")),
                            "rollouts": rollout_outcomes,
                        }
                    )

                zero_var += int(is_zv)
                if not is_zv:
                    active_groups += 1
                rewards_all += rewards

                if group_logger is not None and group_logger.enabled:
                    zv_label = classify_zero_variance_group(group_samples) if is_zv else None
                    group_logger.record(
                        step=step,
                        group=k,
                        prompt_id=str(row.get("id", k)),
                        prompt_tokens=len(prompt_ids),
                        k_correct=sum(1 for s in group_samples if s.get("correct")),
                        g=len(comps),
                        reward_std=statistics.pstdev(rewards) if rewards else 0.0,
                        verdict_reward_std=statistics.pstdev(verdict_scores) if verdict_scores else 0.0,
                        truncations=sum(1 for tr in truncs if tr),
                        mean_thinking_tokens=statistics.fmean(think_counts) if think_counts else 0.0,
                        zero_variance_label=zv_label,
                    )

                adv = torch.tensor(adv_list, dtype=torch.float32)
                t1 = time.time()
                if mode:
                    mode(True)
                ckpt_train = state()["grad_ckpt"] if state else None
                width = max(comp_lens) or 1
                pad = tok.pad_token_id
                ids = torch.tensor([prompt_ids + c + [pad] * (width - len(c)) for c in comps])
                att = torch.tensor([[1] * (len(prompt_ids) + len(c)) + [0] * (width - len(c)) for c in comps])
                if mask_truncated:
                    mask = torch.tensor(
                        [
                            [0.0] * width if tr else [1.0] * len(c) + [0.0] * (width - len(c))
                            for c, tr in zip(comps, truncs, strict=True)
                        ]
                    )
                else:
                    mask = torch.tensor([[1.0] * len(c) + [0.0] * (width - len(c)) for c in comps])
                ids, att, mask, adv = (x.to(model.device) for x in (ids, att, mask, adv))
                with torch.no_grad():
                    if logprob_microbatch is not None:
                        old = completion_logprobs_microbatch(
                            model, ids, att, len(prompt_ids), logprob_microbatch
                        )
                        with ref_adapter_off():
                            ref = completion_logprobs_microbatch(
                                model, ids, att, len(prompt_ids), logprob_microbatch
                            )
                    else:
                        old = completion_logprobs(model, ids, att, len(prompt_ids))
                        with ref_adapter_off():
                            ref = completion_logprobs(model, ids, att, len(prompt_ids))
                if empty_cache_phases and empty_cache is not None:
                    empty_cache()
                if phase_logger is not None:
                    phase_logger.record(
                        step=step,
                        group=k,
                        phase="post_ref",
                        prompt_tokens=len(prompt_ids),
                        completion_lengths=comp_lens,
                    )
                grp_scale = 1.0 if zero_variance_mask else (1.0 / groups_per_update)
                if logprob_microbatch is not None:
                    grp_loss = grpo_microbatch_backward(
                        model,
                        ids,
                        att,
                        len(prompt_ids),
                        old,
                        ref,
                        adv,
                        mask,
                        microbatch=logprob_microbatch,
                        scale=grp_scale,
                        zero_variance_mask=zero_variance_mask,
                        group_size=g,
                        after_forward=(
                            (
                                lambda s=step, grp=k, pt=len(prompt_ids), cl=comp_lens: phase_logger.record(
                                    step=s,
                                    group=grp,
                                    phase="post_train_forward",
                                    prompt_tokens=pt,
                                    completion_lengths=cl,
                                )
                            )
                            if phase_logger is not None
                            else None
                        ),
                    )
                    step_loss_sum += float(grp_loss.detach())
                else:
                    if zero_variance_mask and is_zv:
                        grp_loss = torch.zeros((), dtype=torch.float32, device=ids.device)
                        if phase_logger is not None:
                            phase_logger.record(
                                step=step,
                                group=k,
                                phase="post_train_forward",
                                prompt_tokens=len(prompt_ids),
                                completion_lengths=comp_lens,
                            )
                    else:
                        logp = completion_logprobs(model, ids, att, len(prompt_ids))
                        grp_loss = (
                            grpo_loss(
                                logp,
                                old,
                                ref,
                                adv,
                                mask,
                                zero_variance_mask=zero_variance_mask,
                                group_size=g,
                            )
                            * grp_scale
                        )
                        if phase_logger is not None:
                            phase_logger.record(
                                step=step,
                                group=k,
                                phase="post_train_forward",
                                prompt_tokens=len(prompt_ids),
                                completion_lengths=comp_lens,
                            )
                        grp_loss.backward()
                    step_loss_sum += float(grp_loss.detach())
                device_sync()
                if phase_logger is not None:
                    phase_logger.record(
                        step=step,
                        group=k,
                        phase="post_train_backward",
                        prompt_tokens=len(prompt_ids),
                        completion_lengths=comp_lens,
                    )
                train_s += time.time() - t1
            t2 = time.time()
            if zero_variance_mask:
                if active_groups > 0:
                    for p in model.parameters():
                        if p.requires_grad and p.grad is not None:
                            p.grad.mul_(1.0 / active_groups)
                    optimiser.step()
                reported_loss = step_loss_sum / groups_per_update
            else:
                optimiser.step()
                reported_loss = step_loss_sum
            device_sync()
            if phase_logger is not None:
                phase_logger.record(
                    step=step,
                    group=groups_per_update - 1,
                    phase="step_end",
                    prompt_tokens=last_prompt_tokens,
                    completion_lengths=last_comp_lens,
                )
            train_s += time.time() - t2
            if not math.isfinite(reported_loss):
                return rows, {"step": step, "error": "non-finite loss"}, False
            rec = {
                "step": step,
                "groups": groups_per_update,
                "active_groups": active_groups,
                "zero_variance_groups": zero_var,
                "tokens": tokens,
                "rollout_s": round(roll_s, 4),
                "train_s": round(train_s, 4),
                "step_s": round(time.time() - started, 4),
                "mean_reward": round(statistics.fmean(rewards_all), 4),
                "loss": round(reported_loss, 6),
                "peak_alloc_mib": (
                    phase_logger.peak_alloc_mib()
                    if phase_logger is not None and phase_logger.max_memory_allocated is not None
                    else peak_alloc_mib()
                ),
                "grad_ckpt": ckpt_train,
                "grad_ckpt_rollout": ckpt_rollout,
            }
            rows.append(rec)
            if on_step:
                on_step(rec)
            if on_eval is not None and eval_records is not None:
                on_eval(evaluation_summary(eval_records, advantage_mode=advantage_mode))
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
    census = restore_bf16(model) if getattr(a, "bf16_restore", False) else None
    params = [p for p in model.parameters() if p.requires_grad]
    mode, state = make_mode(model)
    pass_att = bool(getattr(a, "attention_mask", False))
    return {
        "model": model,
        "tok": tok,
        "mode": mode,
        "state": state,
        "optimiser": torch.optim.AdamW(params, lr=LEARNING_RATE),
        "ref_adapter_off": model.disable_adapter,
        "rollout": lambda text, g, max_new, temp: hf_rollout(
            model, tok, text, g, max_new, temp, pass_attention_mask=pass_att
        ),
        "reward_fn": dry_reward,
        "sync": lambda: None,
        "peak_alloc_mib": lambda: None,
        "revision": None,
        "llama": None,
        "dtype_census": census,
        "dry_prepass_cap": DRY_PREPASS_ITEMS,
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
    census = restore_bf16(model) if getattr(a, "bf16_restore", False) else None
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
    pass_att = bool(getattr(a, "attention_mask", False))
    quote_rew = str(getattr(a, "quote_reward", "v1") or "v1")
    bundle = {
        "model": model,
        "tok": tok,
        "render": render,
        "mode": mode,
        "state": state,
        "optimiser": optimiser,
        "ref_adapter_off": model.disable_adapter,
        "reward_fn": lambda text, row, tt, truncated=False: reward(
            text,
            row["label"],
            passages_of(row),
            tt,
            truncated=truncated,
            quote_reward=quote_rew,
        ),
        "sync": torch.cuda.synchronize,
        "peak_alloc_mib": lambda: int(torch.cuda.max_memory_allocated() / 2**20),
        "revision": revision,
        "llama": None,
        "dtype_census": census,
    }
    if a.path == "hf":
        # generation runs with checkpointing off and the cache on, and training with it on: run_steps switches
        # the mode around every rollout and every training pass
        bundle["rollout"] = lambda text, g, max_new, temp: hf_rollout(
            model, tok, render(text), g, max_new, temp, pass_attention_mask=pass_att
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


# -- probe mode and prompt-lengths ---------------------------------------------------------------------------
def run_typeperf_gpu_memory() -> str:  # pragma: no cover - reads Windows PDH counters on the rig
    done = subprocess.run(
        [
            "typeperf",
            r"\GPU Adapter Memory(*)\Shared Usage",
            r"\GPU Adapter Memory(*)\Dedicated Usage",
            "-sc",
            "1",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
        stdin=subprocess.DEVNULL,
    )
    return done.stdout


def read_gpu_adapter_memory(runner: Callable[[], str] = run_typeperf_gpu_memory) -> dict:
    """Read Windows ``\\GPU Adapter Memory(*)\\Shared Usage`` and ``\\Dedicated Usage`` summed bytes.

    Instance names and host names are stripped and never recorded. If the counter cannot be read or parsed,
    returns ``null`` values with the error reason and never raises.
    """
    try:
        raw = runner()
        rows = [row for row in csv.reader(raw.splitlines()) if any(cell.strip() for cell in row)]
        if len(rows) < 2:
            raise ValueError("typeperf output had fewer than 2 CSV rows")
        header, values = rows[0], rows[1]
        if len(values) < len(header):
            raise ValueError("typeperf value row shorter than header")
        shared_total = 0.0
        dedicated_total = 0.0
        found_shared = found_dedicated = False
        for col_name, raw_val in zip(header[1:], values[1:]):
            name_low = col_name.strip().lower()
            val_str = raw_val.strip()
            if not val_str:
                continue
            if name_low.endswith("\\shared usage"):
                shared_total += float(val_str)
                found_shared = True
            elif name_low.endswith("\\dedicated usage"):
                dedicated_total += float(val_str)
                found_dedicated = True
        if not (found_shared and found_dedicated):
            raise ValueError("missing Shared Usage or Dedicated Usage columns")
        return {
            "shared_usage_bytes": int(round(shared_total)),
            "dedicated_usage_bytes": int(round(dedicated_total)),
            "error": None,
        }
    except Exception as exc:
        return {
            "shared_usage_bytes": None,
            "dedicated_usage_bytes": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def compute_prompt_lengths(prompts: Sequence[dict], tok) -> dict:
    """Tokenize the selected prompts (CPU only, no model, no GPU) and return count, min, median, p90, max."""
    lengths = sorted(len(tok.encode(build_prompt(r))) for r in prompts)
    if not lengths:
        return {"count": 0, "min": 0, "median": 0, "p90": 0, "max": 0}
    p90_idx = min(len(lengths) - 1, max(0, math.ceil(0.9 * len(lengths)) - 1))
    return {
        "count": len(lengths),
        "min": lengths[0],
        "median": statistics.median(lengths),
        "p90": lengths[p90_idx],
        "max": lengths[-1],
    }


def parse_probe_lengths(spec: str | Sequence[int]) -> list[int]:
    if isinstance(spec, str):
        items = [int(x.strip()) for x in spec.split(",") if x.strip()]
    else:
        items = [int(x) for x in spec]
    if not items or any(t < 1 for t in items):
        raise SystemExit(f"--probe-lengths must be comma-separated positive integers, got {spec!r}")
    return items


def _cuda_snapshot(cuda) -> dict:
    if cuda is None or (callable(getattr(cuda, "is_available", None)) and not cuda.is_available()):
        return {
            "memory_allocated": 0,
            "memory_reserved": 0,
            "max_memory_allocated": 0,
            "max_memory_reserved": 0,
        }
    return {
        "memory_allocated": int(cuda.memory_allocated()),
        "memory_reserved": int(cuda.memory_reserved()),
        "max_memory_allocated": int(cuda.max_memory_allocated()),
        "max_memory_reserved": int(cuda.max_memory_reserved()),
    }


def _model_static_info(model, kv_key_dtype: str | None) -> dict:
    base_lm = model.get_base_model() if hasattr(model, "get_base_model") else model
    inner = getattr(base_lm, "model", base_lm)
    in_emb = (
        base_lm.get_input_embeddings()
        if hasattr(base_lm, "get_input_embeddings")
        else getattr(inner, "embed_tokens", None)
    )
    out_emb = (
        base_lm.get_output_embeddings()
        if hasattr(base_lm, "get_output_embeddings")
        else getattr(base_lm, "lm_head", None)
    )
    embed_dtype = str(in_emb.weight.dtype).removeprefix("torch.") if in_emb is not None else None
    lm_head_dtype = str(out_emb.weight.dtype).removeprefix("torch.") if out_emb is not None else None
    cfg = getattr(base_lm, "config", None)
    use_cache = bool(getattr(cfg, "use_cache", False)) if cfg is not None else None
    attn_impl = getattr(cfg, "_attn_implementation", None) if cfg is not None else None
    if attn_impl is None and cfg is not None:
        attn_impl = "eager"
    return {
        "embed_tokens_dtype": embed_dtype,
        "lm_head_dtype": lm_head_dtype,
        "kv_cache_key_dtype": kv_key_dtype,
        "use_cache": use_cache,
        "attn_implementation": attn_impl,
    }


def run_probe_memory(
    a: argparse.Namespace,
    *,
    bundle: dict,
    prompts: list[dict],
    gpu: dict | None,
    guard: dict | None,
    counter_runner: Callable[[], str] = run_typeperf_gpu_memory,
) -> int:
    """Probe mode (``--probe-memory``): measures allocator and adapter memory across forced lengths."""
    torch = require_torch()
    model = bundle["model"]
    tok = bundle["tok"]
    mode = bundle.get("mode")
    sync = bundle["sync"]
    ref_adapter_off = bundle["ref_adapter_off"]
    cuda = bundle.get("cuda") or getattr(torch, "cuda", None)
    lengths = parse_probe_lengths(getattr(a, "probe_lengths", "16,64,256"))
    longest_row = max(prompts, key=lambda r: (len(tok.encode(build_prompt(r))), r["id"]))
    prompt_text = build_prompt(longest_row)
    enc = tok(prompt_text, return_tensors="pt")
    prompt_ids = enc["input_ids"].to(model.device)
    prompt_len = int(prompt_ids.shape[1])
    pass_att = bool(getattr(a, "attention_mask", False))
    microbatch = getattr(a, "logprob_microbatch", None)

    if mode:
        mode(False)
    kv_key_dtype: str | None = None
    length_records = []
    summary_per_t = []
    for t_len in lengths:
        if (
            cuda is not None
            and hasattr(cuda, "reset_peak_memory_stats")
            and (not callable(getattr(cuda, "is_available", None)) or cuda.is_available())
        ):
            cuda.reset_peak_memory_stats()
        before_gen = _cuda_snapshot(cuda)
        shared_before = read_gpu_adapter_memory(counter_runner)
        repeated = prompt_ids.repeat(a.g, 1)
        gen_kwargs: dict = {
            "do_sample": True,
            "temperature": TEMPERATURE,
            "min_new_tokens": t_len,
            "max_new_tokens": t_len,
            "pad_token_id": tok.pad_token_id,
            "return_dict_in_generate": True,
            "use_cache": True,
        }
        if pass_att:
            gen_kwargs["attention_mask"] = torch.ones_like(repeated)
        with torch.no_grad():
            gen_out = model.generate(repeated, **gen_kwargs)
        sync()
        after_gen = _cuda_snapshot(cuda)
        shared_after = read_gpu_adapter_memory(counter_runner)

        seqs = gen_out.sequences if hasattr(gen_out, "sequences") else gen_out
        if kv_key_dtype is None:
            keys, values = cache_kv_tensors(getattr(gen_out, "past_key_values", None))
            if keys:
                kv_key_dtype = str(keys[0].dtype).removeprefix("torch.")
            del keys, values
        if kv_key_dtype is None:
            with torch.no_grad():
                probe_out = model(input_ids=seqs[:1], use_cache=True)
            keys, values = cache_kv_tensors(getattr(probe_out, "past_key_values", None))
            if keys:
                kv_key_dtype = str(keys[0].dtype).removeprefix("torch.")
            del keys, values, probe_out

        att_full = torch.ones_like(seqs)
        with torch.no_grad(), ref_adapter_off():
            if microbatch is not None:
                ref_logp = completion_logprobs_microbatch(model, seqs, att_full, prompt_len, microbatch)
            else:
                ref_logp = completion_logprobs(model, seqs, att_full, prompt_len)
        sync()
        after_ref = _cuda_snapshot(cuda)

        del gen_out, seqs, att_full, ref_logp, repeated, gen_kwargs
        gc.collect()
        if cuda is not None and callable(getattr(cuda, "empty_cache", None)):
            cuda.empty_cache()
        after_release = _cuda_snapshot(cuda)

        alloc_delta_mib = (after_gen["memory_allocated"] - before_gen["memory_allocated"]) / 2**20
        slope_mib_per_tok = round(alloc_delta_mib / max(1, t_len), 6)
        res_minus_alloc_mib = round((after_gen["memory_reserved"] - after_gen["memory_allocated"]) / 2**20, 4)
        length_records.append(
            {
                "length": t_len,
                "phases": {
                    "before_generate": before_gen,
                    "after_generate": after_gen,
                    "after_ref_forward": after_ref,
                },
                "allocated_after_release": after_release["memory_allocated"],
                "shared_gpu_memory": {
                    "before_generate": shared_before,
                    "after_generate": shared_after,
                },
                "alloc_mib_per_token": slope_mib_per_tok,
                "reserved_minus_allocated_mib": res_minus_alloc_mib,
            }
        )
        summary_per_t.append(
            {
                "T": t_len,
                "alloc_mib_per_token": slope_mib_per_tok,
                "reserved_minus_allocated_mib": res_minus_alloc_mib,
                "allocated_after_release": after_release["memory_allocated"],
            }
        )

    probe_doc = {
        "test": "S2-probe",
        "path": a.path,
        "prompt_id": longest_row["id"],
        "prompt_tokens": prompt_len,
        "g": a.g,
        "flags": active_flags(a),
        "model_info": _model_static_info(model, kv_key_dtype),
        "dtype_census": bundle.get("dtype_census"),
        "gpu": {"name": gpu.get("name"), "driver": gpu.get("driver")} if isinstance(gpu, dict) else None,
        "exclusive_guard_result": guard.get("result") if isinstance(guard, dict) else None,
        "lengths": length_records,
    }
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"s2-{a.path}-probe.json").write_text(
        json.dumps(probe_doc, indent=1), encoding="utf-8", newline="\n"
    )
    print(json.dumps({"path": a.path, "mode": "probe_memory", "summary": summary_per_t}))
    return 0


# -- rollout-only pre-pass -----------------------------------------------------------------------------------
def item_seed(item_id: str, base_seed: int = PREPASS_BASE_SEED) -> int:
    """Stable 63-bit seed from (base_seed, item_id) so a resumed item is sampled identically."""
    digest = hashlib.sha256(f"{base_seed}:{item_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFF_FFFF_FFFF_FFFF


def script_sha256() -> str:
    return hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()


def ids_sha256(ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def build_prepass_header(
    *,
    prompt_ids: Sequence[str],
    gold_hashes: dict[str, str],
    g: int,
    max_new: int,
    quote_reward: str,
    seed: int = PREPASS_BASE_SEED,
    model_revision: str | None,
    script_hash: str | None = None,
) -> dict:
    return {
        "ids_sha256": ids_sha256(prompt_ids),
        "script_sha256": script_hash if script_hash is not None else script_sha256(),
        "gold_files_sha256": dict(gold_hashes),
        "g": int(g),
        "max_new": int(max_new),
        "quote_reward": str(quote_reward),
        "seed": int(seed),
        "model_revision": model_revision,
    }


def prepass_item_record(
    *,
    row: dict,
    prompt_ids: Sequence[int],
    comps: Sequence[Sequence[int]],
    truncs: Sequence[bool],
    tok,
    reward_fn: Callable,
    quote_reward: str = "v1",
) -> tuple[dict, list[dict]]:
    """Score one item's G completions and return (the text-free JSONL record, eval rollout outcomes)."""
    gold_verdict = str(row.get("label", "supported"))
    gold_file = str(row.get("gold_file") or "")
    passages = passages_of(row) if "context" in row else as_passages(row.get("passages") or ())
    rewards: list[float] = []
    verdict_counts = {"supported": 0, "unsupported": 0, "cannot_tell": 0, "malformed": 0}
    k_correct = 0
    quote_pass = 0
    trunc_count = 0
    rollout_outcomes: list[dict] = []

    for c, trunc in zip(comps, truncs, strict=True):
        is_trunc = bool(trunc)
        if is_trunc:
            trunc_count += 1
        text = tok.decode(c, skip_special_tokens=True)
        think, _ = split_reply(text)
        tt = len(tok.encode(think)) if think else 0
        r_val = _call_reward_fn(reward_fn, text, row, tt, is_trunc)
        rewards.append(r_val)
        det = reward_details(
            text,
            gold_verdict,
            passages,
            tt,
            truncated=is_trunc,
            quote_reward=quote_reward,
        )
        v = det.get("verdict")
        if v in VERDICTS:
            verdict_counts[v] += 1
        else:
            verdict_counts["malformed"] += 1
        if det.get("correct"):
            k_correct += 1
        if not is_trunc:
            parsed = parse_reply(text)
            if parsed is not None:
                parts = parsed["evidence_quotes"]
                if parts and all(quote_found(p["quote"], passages) for p in parts):
                    quote_pass += 1
        rollout_outcomes.append(completion_eval_outcome(text, row, truncated=is_trunc, reward_val=r_val))

    comp_lens = [len(c) for c in comps]
    rec = {
        "id": str(row["id"]),
        "gold_file": gold_file,
        "gold_verdict": gold_verdict,
        "g": len(comps),
        "k_correct": k_correct,
        "supported": verdict_counts["supported"],
        "unsupported": verdict_counts["unsupported"],
        "cannot_tell": verdict_counts["cannot_tell"],
        "malformed": verdict_counts["malformed"],
        "truncations": trunc_count,
        "quote_pass": quote_pass,
        "reward_mean": round(float(statistics.fmean(rewards)), 6) if rewards else 0.0,
        "reward_std": round(float(statistics.pstdev(rewards)), 6) if rewards else 0.0,
        "zero_variance": bool(zero_variance(rewards)),
        "prompt_tokens": len(prompt_ids),
        "completion_tokens_mean": round(float(statistics.fmean(comp_lens)), 2) if comp_lens else 0.0,
    }
    return rec, rollout_outcomes


def _k_correct_hist(records: Sequence[dict], g: int) -> dict[str, int]:
    hist = {str(k): 0 for k in range(g + 1)}
    for r in records:
        k_val = str(int(r["k_correct"]))
        hist[k_val] = hist.get(k_val, 0) + 1
    return hist


def prepass_summary(records: Sequence[dict], *, g: int) -> dict:
    """Summary of a rollout-only pre-pass: counts, histograms (0..G), and ratios only."""
    n = len(records)
    hist = _k_correct_hist(records, g)
    all_correct = sum(1 for r in records if int(r["k_correct"]) == g)
    zv_count = sum(1 for r in records if bool(r["zero_variance"]))
    by_file: dict[str, dict[str, int]] = {}
    for gf in GOLD_FILES:
        by_file[gf] = _k_correct_hist([r for r in records if r.get("gold_file") == gf], g)
    by_verdict: dict[str, dict[str, int]] = {}
    for gv in VERDICTS:
        by_verdict[gv] = _k_correct_hist([r for r in records if r.get("gold_verdict") == gv], g)
    total_truncs = sum(int(r["truncations"]) for r in records)
    gold_sup_completions = sum(int(r["g"]) for r in records if r.get("gold_verdict") == "supported")
    ct_on_gold_sup = sum(int(r["cannot_tell"]) for r in records if r.get("gold_verdict") == "supported")
    p_ct_given_sup = (ct_on_gold_sup / gold_sup_completions) if gold_sup_completions > 0 else 0.0
    return {
        "item_count": n,
        "g": int(g),
        "k_correct_histogram": hist,
        "all_correct_count": all_correct,
        "all_correct_share": (all_correct / n) if n > 0 else 0.0,
        "zero_variance_count": zv_count,
        "zero_variance_share": (zv_count / n) if n > 0 else 0.0,
        "by_gold_file": by_file,
        "by_gold_verdict": by_verdict,
        "total_truncations": total_truncs,
        "p_cannot_tell_given_gold_supported": p_ct_given_sup,
    }


def _prepare_prepass_resume(
    header_path: Path,
    jsonl_path: Path,
    expected_header: dict,
    *,
    fsync: Callable[[int], None] = os.fsync,
) -> list[dict]:
    """Verify or write the prepass header, truncate any trailing partial line, and return done records."""
    if header_path.exists():
        try:
            existing_header = json.loads(header_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise SystemExit(
                f"refusing to resume prepass: header {header_path} is not valid JSON ({exc})"
            ) from exc
        if not isinstance(existing_header, dict):
            raise SystemExit(f"refusing to resume prepass: header {header_path} is not a JSON object")
        mismatched = [
            k
            for k in sorted(set(expected_header) | set(existing_header))
            if existing_header.get(k) != expected_header.get(k)
        ]
        if mismatched:
            raise SystemExit(f"refusing to resume prepass: header mismatch on {', '.join(mismatched)}")
    elif jsonl_path.exists():
        raise SystemExit(f"refusing to resume prepass: {jsonl_path} exists without {header_path}")
    else:
        header_path.parent.mkdir(parents=True, exist_ok=True)
        with header_path.open("w", encoding="utf-8", newline="\n") as hfh:
            hfh.write(json.dumps(expected_header, indent=1) + "\n")
            hfh.flush()
            fsync(hfh.fileno())
        return []

    if not jsonl_path.exists():
        return []

    raw = jsonl_path.read_bytes()
    raw_lines = raw.splitlines(keepends=True)
    completed: list[dict] = []
    valid_bytes = 0
    for idx, line_bytes in enumerate(raw_lines):
        is_last = idx == len(raw_lines) - 1
        has_nl = line_bytes.endswith(b"\n")
        try:
            text_line = line_bytes.decode("utf-8").strip()
            obj = json.loads(text_line) if text_line else None
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            if is_last:
                break
            raise SystemExit(f"refusing to resume prepass: corrupt line {idx + 1} in {jsonl_path}") from exc
        if not has_nl or not isinstance(obj, dict) or not set(PREPASS_RECORD_KEYS).issubset(obj.keys()):
            if is_last:
                break
            raise SystemExit(
                f"refusing to resume prepass: incomplete record on line {idx + 1} in {jsonl_path}"
            )
        completed.append(obj)
        valid_bytes += len(line_bytes)

    if valid_bytes < len(raw):
        with jsonl_path.open("r+b") as tfh:
            tfh.truncate(valid_bytes)
            tfh.flush()
            fsync(tfh.fileno())
    return completed


def run_prepass(
    *,
    a: argparse.Namespace,
    prompts: list[dict],
    gold_hashes: dict[str, str],
    bundle: dict,
    phase_logger: PhaseLogger | None = None,
    eval_records: list[dict] | None = None,
    base_seed: int = PREPASS_BASE_SEED,
    stop_after_new: int | None = None,
    fsync: Callable[[int], None] = os.fsync,
) -> tuple[list[dict], list[dict], dict | None, bool]:
    """Run the rollout-only pre-pass over ``prompts`` with resume support.

    Returns (timing rows, completed prepass item records, stopped record or None, oom bool).
    Never calls optimiser, reference forward, training forward, or backward.
    """
    torch = require_torch()
    tok = bundle["tok"]
    rollout = bundle["rollout"]
    reward_fn = bundle["reward_fn"]
    device_sync = bundle["sync"]
    peak_alloc_mib = bundle["peak_alloc_mib"]
    before_step = bundle.get("before_step")
    mode = bundle.get("mode")
    state = bundle.get("state")
    empty_cache_phases = bool(getattr(a, "empty_cache_phases", False))
    cuda_mod = getattr(torch, "cuda", None)
    empty_cache = getattr(cuda_mod, "empty_cache", lambda: None)
    quote_rew = str(getattr(a, "quote_reward", "v1") or "v1")
    adv_mode = str(getattr(a, "advantage", "std") or "std")
    model_rev = (
        bundle.get("revision")
        if bundle.get("revision") is not None
        else str(getattr(a, "revision", STUDENT_REVISION))
    )
    expected_header = build_prepass_header(
        prompt_ids=[r["id"] for r in prompts],
        gold_hashes=gold_hashes,
        g=a.g,
        max_new=a.max_new,
        quote_reward=quote_rew,
        seed=base_seed,
        model_revision=model_rev,
    )
    a.out.mkdir(parents=True, exist_ok=True)
    header_path = a.out / f"s2-{a.path}-prepass-header.json"
    jsonl_path = a.out / f"s2-{a.path}-prepass.jsonl"
    summary_path = a.out / f"s2-{a.path}-prepass-summary.json"

    completed_records = _prepare_prepass_resume(header_path, jsonl_path, expected_header, fsync=fsync)
    done_ids = {str(r["id"]) for r in completed_records}

    if mode:
        mode(False)
    if before_step:
        before_step(0)

    eos_ids = _eos_token_ids(tok)
    oom_types = tuple(t for t in (getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None),) if t)
    step_rows: list[dict] = []
    new_count = 0

    with jsonl_path.open("a", encoding="utf-8", newline="\n") as jfh:
        for item_idx, row in enumerate(prompts):
            item_id = str(row["id"])
            if item_id in done_ids:
                continue
            if stop_after_new is not None and new_count >= stop_after_new:
                break
            try:
                started = time.time()
                torch.manual_seed(item_seed(item_id, base_seed))
                prompt_text = build_prompt(row)
                ckpt_rollout = state()["grad_ckpt"] if state else None
                if phase_logger is not None:
                    phase_logger.record(
                        step=0,
                        group=item_idx,
                        phase="pre_rollout",
                        prompt_tokens=len(tok.encode(prompt_text)),
                        completion_lengths=(),
                    )
                t0 = time.time()
                roll_out = rollout(prompt_text, a.g, a.max_new, TEMPERATURE)
                if len(roll_out) == 3:
                    prompt_ids, comps, truncs = roll_out
                    truncs = [bool(t) for t in truncs]
                else:
                    prompt_ids, comps = roll_out
                    truncs = [
                        len(c) >= a.max_new and (not eos_ids or not any(t in eos_ids for t in c))
                        for c in comps
                    ]
                device_sync()
                roll_s = time.time() - t0
                comp_lens = [len(c) for c in comps]
                if empty_cache_phases and empty_cache is not None:
                    empty_cache()
                if phase_logger is not None:
                    phase_logger.record(
                        step=0,
                        group=item_idx,
                        phase="post_rollout",
                        prompt_tokens=len(prompt_ids),
                        completion_lengths=comp_lens,
                    )
                rec, rollout_outcomes = prepass_item_record(
                    row=row,
                    prompt_ids=prompt_ids,
                    comps=comps,
                    truncs=truncs,
                    tok=tok,
                    reward_fn=reward_fn,
                    quote_reward=quote_rew,
                )
                if eval_records is not None:
                    rewards_list = [float(ro["reward"]) for ro in rollout_outcomes]
                    adv_list = advantages(rewards_list, mode=adv_mode) if rewards_list else []
                    for ro_item, adv_val in zip(rollout_outcomes, adv_list, strict=True):
                        ro_item["advantage"] = float(adv_val)
                    eval_records.append(
                        {
                            "step": 0,
                            "group": item_idx,
                            "gold": str(row.get("label", "supported")),
                            "rollouts": rollout_outcomes,
                        }
                    )
                jfh.write(json.dumps(rec) + "\n")
                jfh.flush()
                fsync(jfh.fileno())
                completed_records.append(rec)
                done_ids.add(item_id)
                new_count += 1
                is_zv = bool(rec["zero_variance"])
                step_rows.append(
                    {
                        "step": item_idx,
                        "groups": 1,
                        "active_groups": int(not is_zv),
                        "zero_variance_groups": int(is_zv),
                        "tokens": sum(comp_lens),
                        "rollout_s": round(roll_s, 4),
                        "train_s": 0.0,
                        "step_s": round(time.time() - started, 4),
                        "mean_reward": rec["reward_mean"],
                        "loss": 0.0,
                        "peak_alloc_mib": (
                            phase_logger.peak_alloc_mib()
                            if phase_logger is not None and phase_logger.max_memory_allocated is not None
                            else peak_alloc_mib()
                        ),
                        "grad_ckpt": None,
                        "grad_ckpt_rollout": ckpt_rollout,
                    }
                )
            except oom_types:
                return step_rows, completed_records, {"step": item_idx, "error": "out of memory"}, True
            except TimeoutError as exc:
                return step_rows, completed_records, {"step": item_idx, "error": f"timeout: {exc}"}, False

    if len(completed_records) == len(prompts):
        summary_doc = prepass_summary(completed_records, g=a.g)
        with summary_path.open("w", encoding="utf-8", newline="\n") as sfh:
            sfh.write(json.dumps(summary_doc, indent=1) + "\n")
            sfh.flush()
            fsync(sfh.fileno())
    return step_rows, completed_records, None, False


# -- the run -------------------------------------------------------------------------------------------------
def active_flags(a: argparse.Namespace) -> dict:
    """Return a dict of memory/probe/stress/loss/reward flags that are enabled on ``a``."""
    flags: dict = {}
    if getattr(a, "bf16_restore", False):
        flags["bf16_restore"] = True
    if getattr(a, "logprob_microbatch", None) is not None:
        flags["logprob_microbatch"] = int(a.logprob_microbatch)
    if getattr(a, "alloc_conf", None) is not None:
        flags["alloc_conf"] = a.alloc_conf
    if getattr(a, "empty_cache_phases", False):
        flags["empty_cache_phases"] = True
    if getattr(a, "attention_mask", False):
        flags["attention_mask"] = True
    if getattr(a, "stress_prefix", 0):
        flags["stress_prefix"] = int(a.stress_prefix)
        flags["stress_prefix_masks"] = "off"
    if getattr(a, "zero_variance_mask", False):
        flags["zero_variance_mask"] = True
    if getattr(a, "advantage", "std") != "std":
        flags["advantage"] = a.advantage
    if getattr(a, "groups_per_update", GROUPS_PER_STEP) != GROUPS_PER_STEP:
        flags["groups_per_update"] = int(a.groups_per_update)
    if getattr(a, "mask_truncated", False):
        flags["mask_truncated"] = True
    if getattr(a, "quote_reward", "v1") != "v1":
        flags["quote_reward"] = a.quote_reward
    if getattr(a, "rollout_only", False):
        flags["rollout_only"] = True
    return flags


def is_cut(a: argparse.Namespace) -> bool:
    """Whether the run is not the registered shape: --steps, --prompts, --g or --max-new changed."""
    if getattr(a, "rollout_only", False):
        return a.g != G or a.max_new != MAX_NEW_TOKENS
    return a.steps != STEPS or a.prompts != PROMPTS or a.g != G or a.max_new != MAX_NEW_TOKENS


def execute(
    a: argparse.Namespace,
    *,
    build: Callable[[argparse.Namespace], dict],
    rig=None,
    counter_runner: Callable[[], str] = run_typeperf_gpu_memory,
) -> int:
    """One S2 run: guards, prompts, steps, readout, receipt. ``build`` makes the model bundle; ``rig`` is the
    machine's side (the GPU query, the heartbeat, the sampler), None for a dry rehearsal."""
    if getattr(a, "alloc_conf", None) is not None:
        configure_alloc_conf(a.alloc_conf)
    if getattr(a, "logprob_microbatch", None) is not None and a.logprob_microbatch < 1:
        raise SystemExit(f"--logprob-microbatch must be >= 1, got {a.logprob_microbatch}")
    if getattr(a, "stress_prefix", 0) < 0:
        raise SystemExit(f"--stress-prefix must be >= 0, got {a.stress_prefix}")
    if getattr(a, "groups_per_update", GROUPS_PER_STEP) < 1:
        raise SystemExit(f"--groups-per-update must be >= 1, got {a.groups_per_update}")
    rollout_only = bool(getattr(a, "rollout_only", False))
    if rollout_only and getattr(a, "stress_prefix", 0) > 0:
        raise SystemExit("refusing to start: --rollout-only cannot be used with --stress-prefix > 0")

    rows, gold_hashes = read_gold(a.gold)
    prompts = select_prompts(rows, None if rollout_only else a.prompts)
    if getattr(a, "prompt_lengths", False) and not getattr(a, "probe_memory", False):
        if a.dry:
            tok = DryTokenizer()
        else:  # pragma: no cover - tokenizer load without dry
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(STUDENT, revision=a.revision)
        stats = compute_prompt_lengths(prompts, tok)
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / f"s2-{a.path}-prompt-lengths.json").write_text(
            json.dumps(stats, indent=1), encoding="utf-8", newline="\n"
        )
        print(json.dumps({"path": a.path, "mode": "prompt_lengths", **stats}))
        return 0

    cut = is_cut(a)
    gpu = None
    if a.dry:
        guard = check_exclusive(dry_smi, source="dry: a canned nvidia-smi output, never the real one")
    else:
        gpu = rig.preflight()
        guard = gpu.get("guard")
    bundle = build(a)
    if rollout_only and a.dry and bundle.get("dry_prepass_cap") is not None:
        prompts = prompts[: int(bundle["dry_prepass_cap"])]
    phase_logger: PhaseLogger | None = None
    group_logger: GroupLogger | None = None
    try:
        if getattr(a, "probe_memory", False):
            return run_probe_memory(
                a,
                bundle=bundle,
                prompts=prompts,
                gpu=gpu,
                guard=guard,
                counter_runner=counter_runner,
            )

        sampler = VramSampler(rig.vram_query) if rig is not None and not a.dry else None
        cuda_mod = bundle.get("cuda") or (
            getattr(require_torch(), "cuda", None) if rig is not None and not a.dry else None
        )
        phase_logger = PhaseLogger(
            a.out / f"s2-{a.path}-phases.jsonl",
            enabled=rig is not None and not a.dry,
            cuda=cuda_mod,
        )
        group_logger = GroupLogger(
            a.out / f"s2-{a.path}-groups.jsonl",
            enabled=rig is not None and not a.dry and not rollout_only,
            cuda=cuda_mod,
        )
        stress_n = int(getattr(a, "stress_prefix", 0) or 0)
        microbatch = getattr(a, "logprob_microbatch", None)
        empty_phases = bool(getattr(a, "empty_cache_phases", False))
        pass_att = bool(getattr(a, "attention_mask", False))
        groups_per_upd = int(getattr(a, "groups_per_update", GROUPS_PER_STEP) or GROUPS_PER_STEP)
        zv_mask = bool(getattr(a, "zero_variance_mask", False))
        adv_mode = str(getattr(a, "advantage", "std") or "std")
        mask_trunc = bool(getattr(a, "mask_truncated", False))
        quote_rew = str(getattr(a, "quote_reward", "v1") or "v1")
        eval_records: list[dict] = []
        prepass_records: list[dict] = []

        def write_eval_summary(summary_doc: dict) -> None:
            a.out.mkdir(parents=True, exist_ok=True)
            (a.out / f"s2-{a.path}-eval-summary.json").write_text(
                json.dumps(summary_doc, indent=1), encoding="utf-8", newline="\n"
            )

        def run_it() -> tuple[list[dict], dict | None, bool]:
            nonlocal prepass_records
            if rollout_only:
                header_path = a.out / f"s2-{a.path}-prepass-header.json"
                jsonl_path = a.out / f"s2-{a.path}-prepass.jsonl"
                exercise_dry_resume = (
                    a.dry
                    and bundle.get("dry_prepass_cap") is not None
                    and not header_path.exists()
                    and not jsonl_path.exists()
                    and len(prompts) >= 2
                )
                if exercise_dry_resume:
                    half = len(prompts) // 2
                    rows_p1, _, _, _ = run_prepass(
                        a=a,
                        prompts=prompts,
                        gold_hashes=gold_hashes,
                        bundle=bundle,
                        phase_logger=phase_logger,
                        eval_records=eval_records,
                        stop_after_new=half,
                    )
                    with jsonl_path.open("a", encoding="utf-8", newline="\n") as pfh:
                        pfh.write('{"id": "dry_partial_line"')
                        pfh.flush()
                    rows_p2, recs_p2, stop_p2, oom_p2 = run_prepass(
                        a=a,
                        prompts=prompts,
                        gold_hashes=gold_hashes,
                        bundle=bundle,
                        phase_logger=phase_logger,
                        eval_records=eval_records,
                    )
                    prepass_records = recs_p2
                    return rows_p1 + rows_p2, stop_p2, oom_p2
                rows_ro, recs_ro, stop_ro, oom_ro = run_prepass(
                    a=a,
                    prompts=prompts,
                    gold_hashes=gold_hashes,
                    bundle=bundle,
                    phase_logger=phase_logger,
                    eval_records=eval_records,
                )
                prepass_records = recs_ro
                return rows_ro, stop_ro, oom_ro
            if stress_n > 0:
                longest = max(
                    prompts,
                    key=lambda r: (len(bundle["tok"].encode(build_prompt(r))), r["id"]),
                )
                render_fn = bundle.get("render")

                def stress_rollout(text: str, g_cnt: int, max_n: int, temp: float):
                    return hf_rollout(
                        bundle["model"],
                        bundle["tok"],
                        render_fn(text) if callable(render_fn) else text,
                        g_cnt,
                        max_n,
                        temp,
                        pass_attention_mask=pass_att,
                        min_new=max_n,
                    )

                _, stress_stopped, stress_oom = run_steps(
                    model=bundle["model"],
                    ref_adapter_off=bundle["ref_adapter_off"],
                    tok=bundle["tok"],
                    optimiser=bundle["optimiser"],
                    prompts=[longest],
                    rollout=stress_rollout,
                    steps=stress_n,
                    g=a.g,
                    max_new=a.max_new,
                    reward_fn=bundle["reward_fn"],
                    device_sync=bundle["sync"],
                    peak_alloc_mib=bundle["peak_alloc_mib"],
                    before_step=bundle.get("before_step"),
                    mode=bundle.get("mode"),
                    state=bundle.get("state"),
                    logprob_microbatch=microbatch,
                    empty_cache_phases=empty_phases,
                    phase_logger=phase_logger,
                    groups_per_update=groups_per_upd,
                    zero_variance_mask=False,
                    advantage_mode=adv_mode,
                    mask_truncated=False,
                    quote_reward=quote_rew,
                )
                if stress_stopped is not None or stress_oom:
                    return [], stress_stopped, stress_oom
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
                logprob_microbatch=microbatch,
                empty_cache_phases=empty_phases,
                phase_logger=phase_logger,
                group_logger=group_logger,
                groups_per_update=groups_per_upd,
                zero_variance_mask=zv_mask,
                advantage_mode=adv_mode,
                mask_truncated=mask_trunc,
                quote_reward=quote_rew,
                eval_records=eval_records,
                on_eval=write_eval_summary,
            )

        if sampler is not None:
            with sampler:
                steps, stopped, oom = run_it()
        else:
            steps, stopped, oom = run_it()
        eval_summary_doc = evaluation_summary(eval_records, advantage_mode=adv_mode)
        write_eval_summary(eval_summary_doc)
        peak = sampler.peak_mib if sampler is not None else None
        flags_in_use = active_flags(a)
        prepass_summary_doc = (
            prepass_summary(prepass_records, g=a.g) if rollout_only and prepass_records else None
        )
        result = readout(
            steps,
            peak_mib=peak,
            steps_requested=len(prompts) if rollout_only else a.steps,
            stopped=stopped,
            oom=oom,
            cut=cut,
            dry=a.dry,
            series=sampler.series if sampler is not None else (),
            flags=flags_in_use or None,
            phase_logger=phase_logger,
            eval_summary_record=eval_summary_doc,
            rollout_only=rollout_only,
            items_requested=len(prompts) if rollout_only else None,
            items_completed=len(prepass_records) if rollout_only else None,
            prepass_summary_record=prepass_summary_doc,
        )
        line = verdict_line(result, stopped=stopped, cut=cut, dry=a.dry, rollout_only=rollout_only)
        env = environment(
            gpu=gpu,
            dtype="float32 (dry, CPU)" if a.dry else "bfloat16 compute on a 4-bit base",
            batch=a.g,
            exclusive=None if a.dry else True,
            cuda=rig.cuda_version() if rig is not None and not a.dry else None,
            guard=guard,
        )
        run_args = {
            "steps": a.steps,
            "prompts": a.prompts,
            "g": a.g,
            "max_new": a.max_new,
            "dry": a.dry,
            **flags_in_use,
        }
        if rollout_only:
            run_args["ignored_flags"] = ["prompts", "steps"]
        receipt = build_receipt(
            path=a.path,
            dry=a.dry,
            args=run_args,
            rnd_commit=a.rnd_commit,
            gold_hashes=gold_hashes,
            prompt_ids=[r["id"] for r in prompts],
            env=env,
            revision=bundle.get("revision"),
            llama=bundle.get("llama"),
            result=result,
            stopped=stopped,
            verdict=line,
            dtype_census_record=bundle.get("dtype_census"),
        )
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / f"s2-{a.path}.json").write_text(
            json.dumps(receipt, indent=1), encoding="utf-8", newline="\n"
        )
        if not rollout_only:
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
        summary_out = {"path": a.path, "stopped": stopped, **{k: result[k] for k in keys}}
        if rollout_only:
            summary_out["items_completed"] = result["items_completed"]
            summary_out["ignored_flags"] = result["ignored_flags"]
        print(json.dumps(summary_out))
        print(line)
        return exit_status(line)
    finally:
        if group_logger is not None:
            group_logger.close()
        if phase_logger is not None:
            phase_logger.close()
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
    ap.add_argument(
        "--bf16-restore",
        action="store_true",
        help="cast non-trainable floating-point parameters back to bfloat16 after prepare + PEFT",
    )
    ap.add_argument(
        "--logprob-microbatch",
        type=int,
        default=None,
        help="micro-batch size M >= 1 for reference and training passes with logits_to_keep",
    )
    ap.add_argument(
        "--alloc-conf",
        choices=("expandable",),
        default=None,
        help="set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True before torch is imported",
    )
    ap.add_argument(
        "--empty-cache-phases",
        action="store_true",
        help="call torch.cuda.empty_cache() after each rollout and after each reference pass",
    )
    ap.add_argument(
        "--attention-mask",
        action="store_true",
        help="pass an explicit all-ones attention_mask to generate",
    )
    ap.add_argument(
        "--probe-memory",
        action="store_true",
        help="replace the training loop with a memory probe across --probe-lengths",
    )
    ap.add_argument(
        "--probe-lengths",
        default="16,64,256",
        help="comma-separated generation lengths T for --probe-memory",
    )
    ap.add_argument(
        "--prompt-lengths",
        action="store_true",
        help="CPU only: tokenize the 32 selected prompts and write count, min, median, p90 and max",
    )
    ap.add_argument(
        "--stress-prefix",
        type=int,
        default=0,
        help=(
            "extra full steps before the registered steps on the longest prompt with min_new=max_new. "
            "The prefix always runs with both masks off, so it measures the full-width training forward."
        ),
    )
    ap.add_argument(
        "--zero-variance-mask",
        action="store_true",
        help="mask zero-variance groups out of the policy and KL loss; normalise over active groups only",
    )
    ap.add_argument(
        "--advantage",
        choices=("std", "dr-grpo"),
        default="std",
        help="advantage form: 'std' (default) or 'dr-grpo' (reward minus group mean, not divided by std)",
    )
    ap.add_argument(
        "--groups-per-update",
        type=int,
        default=GROUPS_PER_STEP,
        help="number of prompt groups accumulated per optimizer step (default 2)",
    )
    ap.add_argument(
        "--mask-truncated",
        action="store_true",
        help="leave truncated completions out of the loss and group advantage calculation instead of -1.0",
    )
    ap.add_argument(
        "--quote-reward",
        choices=("v1", "v2"),
        default="v1",
        help="quote reward schedule: 'v1' (default) or 'v2' (partial credit + span-disjoint anti-farming)",
    )
    ap.add_argument(
        "--rollout-only",
        action="store_true",
        help=(
            "rollout-only pre-pass over every tune item (sorted by id; --prompts and --steps are ignored): "
            "it trains nothing and writes counts per item."
        ),
    )
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
    if getattr(a, "alloc_conf", None) is not None:
        configure_alloc_conf(a.alloc_conf)
    if a.dry:
        return execute(apply_dry_defaults(a), build=dry_bundle)
    return execute(a, build=real_bundle, rig=Rig(a))


if __name__ == "__main__":
    sys.exit(main())
