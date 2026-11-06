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
import base64
import csv
import fnmatch
import gc
import gzip
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
import zlib
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any
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
DUAL_CLIP_C = 3.0
GUARD_WINDOW_UPDATES = 10
GUARD_WINDOW_GROUPS = 40
FROZEN_PIP_FREEZE_SHA256 = "27c98212c6cd9bd0303b089e76bed0f167cfeeec0eedd0cc9b0baf076c055324"
FROZEN_STACK_VERSIONS = {
    "torch": "2.14.1+cu130",
    "transformers": "5.19.0",
    "bitsandbytes": "0.50.2",
    "peft": "0.21.2",
    "accelerate": "1.15.0",
}
MAX_NEW_TOKENS = 2048
VRAM_LINE_MIB = 31_200  # the rig's watchdog line
SAMPLER_PERIOD_S = 1.0
HEARTBEAT_MAX_AGE_S = 60
READY_TIMEOUT_S = 300.0
READY_POLL_S = 1.0
HEARTBEAT_ENV = "WATCHDOG_HEARTBEAT"
PROMPT_SPLIT = "tune"
STRESS_POOLS = ("selected", "tune")
GOLD_FILES = ("grounded.jsonl", "prs/grounded-prs.jsonl", "diffs/reasoning-diffs.jsonl")
VERDICTS = ("supported", "unsupported", "cannot_tell")
DRY_PREPASS_ITEMS = 4
PREPASS_BASE_SEED = 0
STUDENT_FULL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
B2_REPLAY_SETTINGS: dict[str, Any] = {
    "path": "hf",
    "revision": STUDENT_FULL_REVISION,
    "bf16_restore": True,
    "logprob_microbatch": 2,
    "attention_mask": True,
    "alloc_conf": "expandable",
    "empty_cache_phases": True,
    "quote_reward": "v2",
    "g": G,
    "max_new": MAX_NEW_TOKENS,
}
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


def stress_prompt_pool(
    rows: list[dict],
    selected_prompts: Sequence[dict],
    stress_pool: str = "selected",
    *,
    split: str = PROMPT_SPLIT,
) -> list[dict]:
    """Prompt pool for the stress prefix: ``selected`` (first n tune rows) or ``tune`` (all tune rows)."""
    if stress_pool == "selected":
        return list(selected_prompts)
    if stress_pool == "tune":
        return select_prompts(rows, None, split=split)
    raise SystemExit(f"unknown --stress-pool {stress_pool!r}")


def select_stress_prompt(pool: Sequence[dict], tok) -> tuple[dict, int]:
    """Return (longest row, prompt token count) from ``pool``, breaking ties by larger claim id."""
    best_row: dict | None = None
    best_key: tuple[int, str] | None = None
    for r in pool:
        n_tok = len(tok.encode(build_prompt(r)))
        key = (n_tok, str(r["id"]))
        if best_key is None or key > best_key:
            best_row = r
            best_key = key
    assert best_row is not None and best_key is not None
    return best_row, best_key[0]


# -- the reward: the plan's mechanical reward, with offrig rule 2 and optional v2 schedule -------------------
try:
    from quote_rule import (  # noqa: F401
        MIN_LINE_QUOTE_CHARS,
        MIN_QUOTE_CHARS,
        QUOTE_RULE,
        WATERFALL_RUNGS,
        as_passages,
        classify_quote_rung,
        content_tokens,
        normalise,
        normalise_unmarked,
        quote_credit_v2,
        quote_found,
        quote_is_relevant,
        strip_markers,
    )
    from s2_draw import (  # noqa: F401
        DRAW_MODES,
        HOLDOUT_DEFAULT_FRACTION,
        HOLDOUT_DEFAULT_SEED,
        build_class_stratified_epoch,
        build_class_stratified_schedule,
        draw_prompts,
        select_prompts,
        split_stratified_holdout,
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
        content_tokens,
        normalise,
        normalise_unmarked,
        quote_credit_v2,
        quote_found,
        quote_is_relevant,
        strip_markers,
    )
    from s2_draw import (  # noqa: F401
        DRAW_MODES,
        HOLDOUT_DEFAULT_FRACTION,
        HOLDOUT_DEFAULT_SEED,
        build_class_stratified_epoch,
        build_class_stratified_schedule,
        draw_prompts,
        select_prompts,
        split_stratified_holdout,
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


def clamp_wrong_positive_advantages(
    adv_list: Sequence[float],
    correct_flags: Sequence[bool],
) -> list[float]:
    """Masking fallback: clamp wrong-label advantages to <= 0.0 after mean-centring, with no re-centring."""
    return [
        float(adv) if bool(is_corr) else min(float(adv), 0.0)
        for adv, is_corr in zip(adv_list, correct_flags, strict=True)
    ]


class CtCollapseGuard:
    """Two-class CT collapse guard (S1, S2, U1, U2) counted by distinct item over sampled-group windows.

    A window is ``window_groups`` sampled groups (default 40 = 10 updates of 4 groups, including
    zero-variance groups). Checked at the end of each window:
    - S1 (gold supported, window): >= 3 distinct gold-supported items in the window have any CT and
      >= 2 of those groups gave CT a positive advantage (only non-zero-variance groups have positive adv).
    - S2 (gold supported, cumulative): CT > 1.5% of gold-supported answers once >= 600 gold-supported
      answers have accrued, across >= 4 distinct gold-supported items.
    - U1 (gold unsupported, window): >= 4 distinct gold-unsupported items in the window have any CT and
      the window's correct rate on gold-unsupported is < 78%.
    - U2 (gold unsupported, easy items, window): >= 2 distinct gold-unsupported items with Pass-A k >= 6
      have any CT in one window.
    - No guard on gold cannot_tell items.
    """

    def __init__(
        self,
        *,
        window_groups: int = GUARD_WINDOW_GROUPS,
        pass_a_k: Mapping[str, int] | None = None,
    ) -> None:
        if window_groups < 1:
            raise SystemExit(f"--guard-window-groups must be >= 1, got {window_groups}")
        self.window_groups = int(window_groups)
        self.pass_a_k = {str(k): int(v) for k, v in pass_a_k.items()} if pass_a_k is not None else {}
        self.window_index = 0
        self.total_groups_sampled = 0
        self._window: list[dict[str, Any]] = []
        self.records: list[dict[str, Any]] = []
        self.cum_sup_answers = 0
        self.cum_sup_ct_answers = 0
        self.cum_sup_ct_items: set[str] = set()
        self.window_history: list[dict[str, Any]] = []

    def record_group(
        self,
        *,
        step: int,
        group: int,
        item_id: str,
        gold_verdict: str,
        verdicts: Sequence[str | None],
        correct_flags: Sequence[bool],
        advantages_list: Sequence[float],
        is_zero_variance: bool = False,
    ) -> dict[str, Any] | None:
        cid = str(item_id)
        gold = str(gold_verdict)
        v_list = list(verdicts)
        c_list = [bool(c) for c in correct_flags]
        a_list = [float(a) for a in advantages_list]
        ct_count = sum(1 for v in v_list if v == "cannot_tell")
        correct_count = sum(1 for c in c_list if c)
        ct_pos_adv = (not is_zero_variance) and any(
            v == "cannot_tell" and a > 0.0 for v, a in zip(v_list, a_list, strict=True)
        )
        if gold == "supported":
            self.cum_sup_answers += len(v_list)
            self.cum_sup_ct_answers += ct_count
            if ct_count > 0:
                self.cum_sup_ct_items.add(cid)

        self.total_groups_sampled += 1
        rec_entry = {
            "step": int(step),
            "group": int(group),
            "item_id": cid,
            "gold_verdict": gold,
            "n": len(v_list),
            "ct_count": ct_count,
            "correct_count": correct_count,
            "ct_pos_adv": bool(ct_pos_adv),
        }
        self.records.append(rec_entry)
        self._window.append(rec_entry)
        if len(self._window) < self.window_groups:
            return None
        return self._evaluate_window(step=int(step))

    def _evaluate_window(self, *, step: int) -> dict[str, Any] | None:
        self.window_index += 1
        win = list(self._window)
        self._window.clear()

        sup_groups = [g_rec for g_rec in win if g_rec["gold_verdict"] == "supported"]
        s1_ct_items = {g_rec["item_id"] for g_rec in sup_groups if g_rec["ct_count"] > 0}
        s1_pos_adv_groups = sum(1 for g_rec in sup_groups if g_rec["ct_count"] > 0 and g_rec["ct_pos_adv"])

        cum_sup_ct_rate = (
            (self.cum_sup_ct_answers / float(self.cum_sup_answers)) if self.cum_sup_answers > 0 else 0.0
        )

        unsup_groups = [g_rec for g_rec in win if g_rec["gold_verdict"] == "unsupported"]
        u1_ct_items = {g_rec["item_id"] for g_rec in unsup_groups if g_rec["ct_count"] > 0}
        u_total = sum(int(g_rec["n"]) for g_rec in unsup_groups)
        u_correct = sum(int(g_rec["correct_count"]) for g_rec in unsup_groups)
        u_correct_rate = (u_correct / float(u_total)) if u_total > 0 else 1.0

        u2_easy_ct_items = {
            g_rec["item_id"]
            for g_rec in unsup_groups
            if g_rec["ct_count"] > 0 and self.pass_a_k.get(g_rec["item_id"], -1) >= 6
        }

        triggered: list[str] = []
        if len(s1_ct_items) >= 3 and s1_pos_adv_groups >= 2:
            triggered.append("S1")
        if (
            self.cum_sup_answers >= 600
            and cum_sup_ct_rate > 0.015
            and len(self.cum_sup_ct_items) >= 4
        ):
            triggered.append("S2")
        if len(u1_ct_items) >= 4 and u_total > 0 and u_correct_rate < 0.78:
            triggered.append("U1")
        if len(u2_easy_ct_items) >= 2:
            triggered.append("U2")

        metrics = {
            "window_index": self.window_index,
            "total_groups_sampled": self.total_groups_sampled,
            "s1_distinct_ct_items": len(s1_ct_items),
            "s1_pos_adv_groups": s1_pos_adv_groups,
            "s2_cum_sup_answers": self.cum_sup_answers,
            "s2_cum_sup_ct_answers": self.cum_sup_ct_answers,
            "s2_cum_sup_ct_rate": round(cum_sup_ct_rate, 6),
            "s2_cum_distinct_ct_items": len(self.cum_sup_ct_items),
            "u1_distinct_ct_items": len(u1_ct_items),
            "u1_correct_rate": round(u_correct_rate, 6),
            "u2_distinct_easy_ct_items": len(u2_easy_ct_items),
            "triggered_rules": list(triggered),
        }
        self.window_history.append({"step": int(step), **metrics})
        if not triggered:
            return None
        return {
            "step": int(step),
            "reason": f"ct_guard_{triggered[0].lower()}",
            "triggered_rules": triggered,
            "window_index": self.window_index,
            "total_groups_sampled": self.total_groups_sampled,
            "guard_metrics": metrics,
        }

    def to_state(self) -> dict[str, Any]:
        """Everything a resumed run needs to carry the guard on: the open window and the S2 counters."""
        return {
            "window_groups": self.window_groups,
            "window_index": self.window_index,
            "total_groups_sampled": self.total_groups_sampled,
            "open_window": [dict(r) for r in self._window],
            "records": [dict(r) for r in self.records],
            "cum_sup_answers": self.cum_sup_answers,
            "cum_sup_ct_answers": self.cum_sup_ct_answers,
            "cum_sup_ct_items": sorted(self.cum_sup_ct_items),
            "window_history": [dict(w) for w in self.window_history],
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any], *, pass_a_k: Mapping[str, int] | None) -> CtCollapseGuard:
        guard = cls(window_groups=int(state["window_groups"]), pass_a_k=pass_a_k)
        guard.window_index = int(state["window_index"])
        guard.total_groups_sampled = int(state["total_groups_sampled"])
        guard._window = [dict(r) for r in state["open_window"]]
        guard.records = [dict(r) for r in state["records"]]
        guard.cum_sup_answers = int(state["cum_sup_answers"])
        guard.cum_sup_ct_answers = int(state["cum_sup_ct_answers"])
        guard.cum_sup_ct_items = {str(x) for x in state["cum_sup_ct_items"]}
        guard.window_history = [dict(w) for w in state["window_history"]]
        return guard


class TruncationHaltGuard:
    """The training run's truncation halt (an operational stop, not an outcome criterion).

    Fed each update's completion and truncation counts (every sample of every group, zero-variance groups
    included), it keeps a rolling window of the last ``window_updates`` updates. Once the window is full, a
    window rate (truncated / completions) above ``halt_rate`` halts the run at that update boundary. The
    descriptive flag (3x the starting rate) lives in the external chunk checker, not here.
    """

    def __init__(self, *, window_updates: int, halt_rate: float) -> None:
        if isinstance(window_updates, bool) or int(window_updates) < 1:
            raise SystemExit(f"--trunc-window-updates must be >= 1, got {window_updates!r}")
        rate = float(halt_rate)
        if not (0.0 < rate < 1.0):
            raise SystemExit(f"--trunc-halt-rate must be in (0, 1), got {halt_rate!r}")
        self.window_updates = int(window_updates)
        self.halt_rate = rate
        self.window: list[tuple[int, int, int]] = []  # (step, completions, truncated), oldest first
        self.history: list[dict[str, Any]] = []

    def record_update(self, *, step: int, completions: int, truncated: int) -> dict[str, Any] | None:
        self.window.append((int(step), int(completions), int(truncated)))
        if len(self.window) > self.window_updates:
            self.window.pop(0)
        n = sum(c for _, c, _ in self.window)
        t = sum(x for _, _, x in self.window)
        full = len(self.window) == self.window_updates
        rate = (t / n) if n > 0 else 0.0
        halt = bool(full and rate > self.halt_rate)
        entry = {
            "kind": "trunc_window",
            "step": int(step),
            "update_completions": int(completions),
            "update_truncated": int(truncated),
            "window_updates": len(self.window),
            "window_full": full,
            "window_completions": n,
            "window_truncated": t,
            "window_rate": round(rate, 6),
            "halt_rate": self.halt_rate,
            "halt": halt,
        }
        self.history.append(entry)
        if not halt:
            return None
        return {
            "step": int(step),
            "reason": "trunc_halt",
            "triggered_rules": ["TRUNC"],
            "window_updates": self.window_updates,
            "window_completions": n,
            "window_truncated": t,
            "window_rate": round(rate, 6),
            "halt_rate": self.halt_rate,
        }

    def to_state(self) -> dict[str, Any]:
        return {
            "window_updates": self.window_updates,
            "halt_rate": self.halt_rate,
            "window": [list(w) for w in self.window],
            "history": [dict(h) for h in self.history],
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> TruncationHaltGuard:
        guard = cls(window_updates=int(state["window_updates"]), halt_rate=float(state["halt_rate"]))
        guard.window = [(int(s), int(c), int(t)) for s, c, t in state["window"]]
        guard.history = [dict(h) for h in state["history"]]
        return guard


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
def quote_pass_role_text(quote_reward: str = "v1") -> str:
    return (
        "diagnostic: every evidence quote found in the passages; "
        f"not the registered reward (quote rule {quote_reward})"
    )


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
    quote_reward: str | None = None,
    mean_residual_by_k_band_record: dict[str, object] | None = None,
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
        q_rew = str(quote_reward or out["flags"].get("quote_reward") or "v1")
        if prepass_summary_record is not None:
            out["prepass_summary"] = prepass_summary_record
            if "quote_pass" in prepass_summary_record:
                out["quote_pass"] = prepass_summary_record["quote_pass"]
            out["quote_pass_role"] = prepass_summary_record.get(
                "quote_pass_role", quote_pass_role_text(q_rew)
            )
        else:
            out["quote_pass_role"] = quote_pass_role_text(q_rew)
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
    if mean_residual_by_k_band_record is not None:
        out["mean_residual_by_k_band"] = dict(mean_residual_by_k_band_record)
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
    """0 for S2-OK, S2-PREPASS, a finished rehearsal, and a training chunk or run that ended cleanly;
    2 for a training halt (S2-TRAIN-HALT-<rule>: stop and report); anything else is 1."""
    if line.startswith("S2-TRAIN-HALT-"):
        return TRAIN_EXIT_HALT
    return 0 if line in ("S2-OK", "S2-DRY", "S2-PREPASS", "S2-TRAIN-CHUNK-OK", "S2-TRAIN-COMPLETE") else 1


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


def check_frozen_env(
    pip_freeze_path: Path | None = None,
    *,
    expected_sha256: str | None = None,
    installed_versions: Mapping[str, str | None] | None = None,
    require_exact_versions: bool = False,
) -> dict:
    """Record frozen stack versions and check the frozen pip-freeze.txt sha256."""
    versions = (
        {k: installed_versions.get(k) for k in FROZEN_STACK_VERSIONS}
        if installed_versions is not None
        else {k: package_version(k) for k in FROZEN_STACK_VERSIONS}
    )
    actual_sha: str | None = None
    exp_sha = str(FROZEN_PIP_FREEZE_SHA256 if expected_sha256 is None else expected_sha256).strip().lower()
    eff_freeze_path = pip_freeze_path
    if eff_freeze_path is None:
        candidate_freeze = Path(sys.prefix) / "FROZEN-pip-freeze-2026-11-05.txt"
        if candidate_freeze.is_file():
            eff_freeze_path = candidate_freeze
    if require_exact_versions:
        for pkg, req_ver in FROZEN_STACK_VERSIONS.items():
            inst_ver = versions.get(pkg)
            if inst_ver != req_ver:
                raise SystemExit(
                    f"refusing to start (frozen_env_version_mismatch): "
                    f"installed {pkg}=={inst_ver!r}, expected {req_ver!r}"
                )
    if eff_freeze_path is None and require_exact_versions:
        raise SystemExit(
            "refusing to start (frozen_env_missing_file): "
            "--frozen-pip-freeze was not provided and "
            "FROZEN-pip-freeze-2026-11-05.txt is missing at sys.prefix"
        )
    if eff_freeze_path is not None:
        pf_path = Path(eff_freeze_path)
        if not pf_path.is_file():
            raise SystemExit(
                f"refusing to start (frozen_env_missing_file): pip-freeze file does not exist: {pf_path.name}"
            )
        raw_bytes = pf_path.read_bytes()
        actual_sha = hashlib.sha256(raw_bytes).hexdigest()
        if actual_sha != exp_sha:
            raise SystemExit(
                f"refusing to start (frozen_env_sha_mismatch): "
                f"expected pip-freeze sha256 {exp_sha}, got {actual_sha}"
            )
        freeze_versions: dict[str, str] = {}
        for raw_line in raw_bytes.decode("utf-8", errors="replace").splitlines():
            line = raw_line.strip()
            if "==" in line and not line.startswith("#"):
                pkg, _, ver = line.partition("==")
                freeze_versions[pkg.strip().lower()] = ver.strip()
        for pkg, req_ver in FROZEN_STACK_VERSIONS.items():
            if pkg in freeze_versions and freeze_versions[pkg] != req_ver:
                raise SystemExit(
                    f"refusing to start (frozen_env_version_mismatch): "
                    f"pip-freeze has {pkg}=={freeze_versions[pkg]}, expected {req_ver}"
                )
    return {
        "stack_versions": versions,
        "pip_freeze_sha256": actual_sha,
        "frozen_expected_sha256": exp_sha,
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
    save_all: bool | None = None,
    save_all_sha256: str | None = None,
    save_all_off_reason: str | None = None,
    checkpoint_sha256: str | None = None,
    checkpoint_files_sha256: Mapping[str, str] | None = None,
    adapter_sha256: str | None = None,
    adapter_files_sha256: Mapping[str, str] | None = None,
    stack_versions: Mapping[str, str | None] | None = None,
    pip_freeze_sha256: str | None = None,
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
        "stack_versions": (
            dict(stack_versions)
            if stack_versions is not None
            else {k: package_version(k) for k in FROZEN_STACK_VERSIONS}
        ),
        "pip_freeze_sha256": pip_freeze_sha256,
        "llama_cpp": llama,
        "pass_line": {"vram_line_mib": VRAM_LINE_MIB, "steps": STEPS, "oom": False},
        "stopped": stopped,
        "readout": result,
        "verdict": verdict,
    }
    if dtype_census_record is not None:
        out["dtype_census"] = dtype_census_record
    if save_all is not None:
        out["save_all"] = bool(save_all)
        out["save_all_sha256"] = save_all_sha256
        if not save_all and save_all_off_reason is not None:
            out["save_all_off_reason"] = save_all_off_reason
    if checkpoint_sha256 is not None:
        out["checkpoint_sha256"] = checkpoint_sha256
    if checkpoint_files_sha256 is not None:
        out["checkpoint_files_sha256"] = dict(checkpoint_files_sha256)
    if adapter_sha256 is not None:
        out["adapter_sha256"] = adapter_sha256
    if adapter_files_sha256 is not None:
        out["adapter_files_sha256"] = dict(adapter_files_sha256)
    for k in (
        "stress_pool",
        "stress_prompt_id",
        "stress_prompt_tokens",
        "stress_width",
        "draw_mode",
        "draw_n",
        "draw_seed",
        "drawn_ids_sha256",
        "prepass_sha256",
        "k_min",
        "k_max",
        "band_pool_size",
        "holdout_seed",
        "holdout_count",
        "holdout_ids_sha256",
        "train_ids_sha256",
        "stop_after_new",
    ):
        if k in args:
            out[k] = args[k]
    return out


def save_all_file_path(out_dir: Path, path: str = "hf") -> Path:
    """Path to the gzip-compressed JSONL completion archive beside the run's outputs."""
    return Path(out_dir) / f"s2-{path}-save-all.jsonl.gz"


def write_save_all(
    path: Path,
    records: Sequence[dict],
    *,
    fsync: Callable[[int], None] = os.fsync,
) -> str:
    """Write ``records`` as deterministic gzip-compressed JSONL (mtime=0) and return its sha256 hex digest."""
    raw_bytes = "".join(json.dumps(r) + "\n" for r in records).encode("utf-8")
    gz_bytes = gzip.compress(raw_bytes, mtime=0)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as fh:
        fh.write(gz_bytes)
        fh.flush()
        fsync(fh.fileno())
    return hashlib.sha256(gz_bytes).hexdigest()


def append_save_all(
    path: Path,
    records: Sequence[dict],
    *,
    fsync: Callable[[int], None] = os.fsync,
) -> None:
    """Append ``records`` as one deterministic gzip member (mtime=0) with flush + fsync."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        if not out_path.exists():
            gz_bytes = gzip.compress(b"", mtime=0)
            with out_path.open("ab") as fh:
                fh.write(gz_bytes)
                fh.flush()
                fsync(fh.fileno())
        return
    raw_bytes = "".join(json.dumps(r) + "\n" for r in records).encode("utf-8")
    gz_bytes = gzip.compress(raw_bytes, mtime=0)
    with out_path.open("ab") as fh:
        fh.write(gz_bytes)
        fh.flush()
        fsync(fh.fileno())


def dropped_archive_path(path: Path) -> Path:
    """Return the companion ``<archive>.dropped.jsonl.gz`` path for ``path``."""
    in_path = Path(path)
    name = in_path.name
    if name.endswith(".jsonl.gz"):
        base = name[: -len(".jsonl.gz")]
    elif name.endswith(".jsonl"):
        base = name[: -len(".jsonl")]
    else:
        base = name
    return in_path.with_name(f"{base}.dropped.jsonl.gz")


def _append_dropped_entries(
    archive_path: Path,
    entries: Sequence[dict],
    *,
    fsync: Callable[[int], None] = os.fsync,
) -> Path:
    """Append dropped record/byte entries to ``<archive>.dropped.jsonl.gz`` with flush + fsync."""
    drop_path = dropped_archive_path(archive_path)
    if entries:
        append_save_all(drop_path, entries, fsync=fsync)
    return drop_path


def _prepare_save_all_resume(
    path: Path,
    *,
    valid_record: Callable[[dict], bool] | None = None,
    record_key: Callable[[dict], Any] | None = None,
    fsync: Callable[[int], None] = os.fsync,
) -> list[dict]:
    """Read back a ``.jsonl.gz`` save-all archive and move dropped text to ``.dropped.jsonl.gz``."""
    in_path = Path(path)
    if not in_path.exists():
        return []
    raw = in_path.read_bytes()
    if not raw:
        return []
    kept_records: list[dict] = []
    seen_keys: set[Any] = set()
    clean_chunks: list[bytes] = []
    dropped_entries: list[dict] = []
    modified = False
    pos = 0
    total_len = len(raw)
    while pos < total_len:
        dobj = zlib.decompressobj(16 + zlib.MAX_WBITS)
        try:
            member_raw = dobj.decompress(raw[pos:])
        except zlib.error:
            dropped_entries.append(
                {
                    "reason": "torn_member",
                    "byte_offset": pos,
                    "raw_bytes_b64": base64.b64encode(raw[pos:]).decode("ascii"),
                }
            )
            modified = True
            break
        if not dobj.eof or (len(raw[pos:]) - len(dobj.unused_data)) <= 0:
            dropped_entries.append(
                {
                    "reason": "torn_member",
                    "byte_offset": pos,
                    "raw_bytes_b64": base64.b64encode(raw[pos:]).decode("ascii"),
                }
            )
            modified = True
            break
        consumed = len(raw[pos:]) - len(dobj.unused_data)
        member_bytes = raw[pos : pos + consumed]
        member_start = pos
        pos += consumed
        try:
            text = member_raw.decode("utf-8")
        except UnicodeDecodeError:
            dropped_entries.append(
                {
                    "reason": "corrupt_member",
                    "byte_offset": member_start,
                    "raw_bytes_b64": base64.b64encode(raw[member_start:]).decode("ascii"),
                }
            )
            modified = True
            break
        if text and not text.endswith("\n"):
            dropped_entries.append(
                {
                    "reason": "torn_member",
                    "byte_offset": member_start,
                    "raw_bytes_b64": base64.b64encode(raw[member_start:]).decode("ascii"),
                }
            )
            modified = True
            break
        member_records: list[dict] = []
        parse_ok = True
        for line in text.splitlines():
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                parse_ok = False
                break
            if not isinstance(obj, dict):
                parse_ok = False
                break
            member_records.append(obj)
        if not parse_ok:
            dropped_entries.append(
                {
                    "reason": "corrupt_member",
                    "byte_offset": member_start,
                    "raw_bytes_b64": base64.b64encode(raw[member_start:]).decode("ascii"),
                }
            )
            modified = True
            break
        filtered: list[dict] = []
        member_kept_all = True
        for rec in member_records:
            if valid_record is not None and not valid_record(rec):
                member_kept_all = False
                dropped_entries.append({**rec, "reason": "uncommitted_member", "record": rec})
                continue
            if record_key is not None:
                rk = record_key(rec)
                if rk in seen_keys:
                    member_kept_all = False
                    dropped_entries.append({**rec, "reason": "duplicate_record", "record": rec})
                    continue
                seen_keys.add(rk)
            filtered.append(rec)
        if not member_records:
            if not modified:
                clean_chunks.append(member_bytes)
            continue
        if member_kept_all and not modified:
            clean_chunks.append(member_bytes)
        else:
            modified = True
            if filtered:
                reencoded = "".join(json.dumps(r) + "\n" for r in filtered).encode("utf-8")
                clean_chunks.append(gzip.compress(reencoded, mtime=0))
        kept_records.extend(filtered)

    clean_bytes = b"".join(clean_chunks)
    if modified or len(clean_bytes) != len(raw):
        if dropped_entries:
            _append_dropped_entries(in_path, dropped_entries, fsync=fsync)
        with in_path.open("r+b") as fh:
            fh.seek(0)
            if clean_bytes:
                fh.write(clean_bytes)
            fh.truncate(len(clean_bytes))
            fh.flush()
            fsync(fh.fileno())
    return kept_records


def read_save_all(path: Path) -> list[dict]:
    """Read completion records from a ``.jsonl.gz`` (or plain ``.jsonl``) save-all archive."""
    in_path = Path(path)
    raw = in_path.read_bytes()
    if not raw:
        return []
    text = gzip.decompress(raw).decode("utf-8") if in_path.name.endswith(".gz") else raw.decode("utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def validate_save_all_args(a: argparse.Namespace) -> tuple[bool, str | None]:
    """Validate ``--save-all`` / ``--no-save-all`` and ``--save-all-off-reason``."""
    save_all_on = bool(getattr(a, "save_all", True))
    raw_reason = getattr(a, "save_all_off_reason", None)
    reason = raw_reason.strip() if isinstance(raw_reason, str) else None
    if not save_all_on:
        if not reason:
            raise SystemExit(
                "refusing to start (save_all_off_reason_required): "
                "--no-save-all requires a non-empty --save-all-off-reason"
            )
        return False, reason
    if raw_reason is not None:
        raise SystemExit(
            "refusing to start (save_all_off_reason_unused): "
            "--save-all-off-reason cannot be used unless --no-save-all is set"
        )
    return True, None


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
    dual_clip_c: float | None = None,
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
    surr = torch.min(ratio * a, torch.clamp(ratio, 1 - eps, 1 + eps) * a)
    if dual_clip_c is not None:
        surr = torch.where(a < 0, torch.max(surr, float(dual_clip_c) * a), surr)
    policy = -surr
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
    dual_clip_c: float | None = None,
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
        surr = torch.min(ratio * mb_adv, torch.clamp(ratio, 1 - eps, 1 + eps) * mb_adv)
        if dual_clip_c is not None:
            surr = torch.where(mb_adv < 0, torch.max(surr, float(dual_clip_c) * mb_adv), surr)
        policy = -surr
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
        correct_n: int | None = None,
        correct_negative_adv_n: int | None = None,
        truncated_n: int | None = None,
        outcome_var: float | None = None,
        correct_var: float | None = None,
        total_reward_var: float | None = None,
        k_prepass: int | None = None,
        residual: float | None = None,
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
        if correct_n is not None:
            line["correct_n"] = int(correct_n)
            line["correct_negative_adv_n"] = int(correct_negative_adv_n or 0)
            line["truncated_n"] = int(truncated_n if truncated_n is not None else truncations)
            line["outcome_var"] = round(float(outcome_var or 0.0), 6)
            line["correct_var"] = round(float(correct_var or 0.0), 6)
            line["total_reward_var"] = round(float(total_reward_var or 0.0), 6)
        if k_prepass is not None:
            line["k_prepass"] = int(k_prepass)
            line["residual"] = round(float(residual or 0.0), 6)
        self._fh.write(json.dumps(line) + "\n")
        self._fh.flush()
        self.fsync(self._fh.fileno())

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


SDPA_BACKEND_NAMES = ("flash", "efficient", "math", "cudnn", "unknown")


def detect_sdpa_backend(
    query,
    key,
    value,
    *,
    attn_mask=None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> tuple[str, str | None]:
    """Report the SDPA backend ('flash', 'efficient', 'math', 'cudnn', or ('unknown', reason)) for a call.

    Uses ``torch.nn.attention.SDPBackend``, ``torch.ops.aten._fused_sdp_choice``, and
    ``torch.backends.cuda.SDPAParams`` / ``can_use_*_attention`` without altering the attention configuration.
    """
    torch = require_torch()
    from torch.nn.attention import SDPBackend

    backend_map = {
        int(SDPBackend.FLASH_ATTENTION): "flash",
        int(SDPBackend.EFFICIENT_ATTENTION): "efficient",
        int(SDPBackend.MATH): "math",
        int(SDPBackend.CUDNN_ATTENTION): "cudnn",
    }
    choice_err: str | None = None
    try:
        code = int(
            torch.ops.aten._fused_sdp_choice(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=float(dropout_p),
                is_causal=bool(is_causal),
                scale=scale,
                enable_gqa=bool(enable_gqa),
            )
        )
        if code in backend_map:
            return backend_map[code], None
        choice_err = f"_fused_sdp_choice returned {code}"
    except Exception as exc:
        choice_err = f"{type(exc).__name__}: {exc}"

    try:
        cuda_backends = getattr(getattr(torch, "backends", None), "cuda", None)
        if cuda_backends is not None and getattr(query, "is_cuda", False):
            params = cuda_backends.SDPAParams(
                query,
                key,
                value,
                attn_mask,
                float(dropout_p),
                bool(is_causal),
                bool(enable_gqa),
            )
            if cuda_backends.flash_sdp_enabled() and cuda_backends.can_use_flash_attention(params, False):
                return "flash", None
            if cuda_backends.cudnn_sdp_enabled() and cuda_backends.can_use_cudnn_attention(params, False):
                return "cudnn", None
            if cuda_backends.mem_efficient_sdp_enabled() and cuda_backends.can_use_efficient_attention(
                params, False
            ):
                return "efficient", None
            if cuda_backends.math_sdp_enabled():
                return "math", None
    except Exception as exc:
        choice_err = f"{type(exc).__name__}: {exc}"

    return "unknown", (choice_err or "torch could not report an SDPA backend")


def _make_sdpa_function_mode(on_sdpa: Callable[[str, str | None], None]):
    """Build a ``torch.overrides.TorchFunctionMode`` that observes ``F.scaled_dot_product_attention``."""
    torch = require_torch()
    from torch.overrides import TorchFunctionMode

    sdpa_fn = torch.nn.functional.scaled_dot_product_attention

    class _SdpaObserverMode(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            out = func(*args, **kwargs)
            if func is sdpa_fn and len(args) >= 3:
                q, k, v = args[0], args[1], args[2]
                attn_mask = args[3] if len(args) > 3 else kwargs.get("attn_mask")
                dropout_p = args[4] if len(args) > 4 else kwargs.get("dropout_p", 0.0)
                is_causal = args[5] if len(args) > 5 else kwargs.get("is_causal", False)
                scale = kwargs.get("scale")
                enable_gqa = kwargs.get("enable_gqa", False)
                backend, reason = detect_sdpa_backend(
                    q,
                    k,
                    v,
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                    scale=scale,
                    enable_gqa=enable_gqa,
                )
                on_sdpa(backend, reason)
                if isinstance(out, torch.Tensor) and out.requires_grad:
                    out.register_hook(lambda _grad, b=backend, r=reason: on_sdpa(b, r))
            return out

    return _SdpaObserverMode()


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
        log_sdpa_backend: bool = False,
    ) -> None:
        self.path = path
        cuda_ready = (
            cuda is not None and callable(getattr(cuda, "is_available", None)) and bool(cuda.is_available())
        )
        self.enabled = bool(enabled and path is not None and cuda_ready)
        self.cuda = cuda
        self.now = now
        self.fsync = fsync
        self.log_sdpa_backend = bool(log_sdpa_backend)
        self.max_memory_allocated: int | None = None
        self.max_memory_reserved: int | None = None
        self._phase_sdpa_backends: list[str] = []
        self._phase_sdpa_reasons: list[str] = []
        self._sdpa_mode = None
        self._fh = None
        if self.enabled and self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.path.open("w", encoding="utf-8", newline="\n")
            if self.log_sdpa_backend:
                self._sdpa_mode = _make_sdpa_function_mode(self._note_sdpa)
                self._sdpa_mode.__enter__()

    def _note_sdpa(self, backend: str, reason: str | None = None) -> None:
        self._phase_sdpa_backends.append(backend)
        if reason:
            self._phase_sdpa_reasons.append(reason)

    def _consume_sdpa_backend(self) -> tuple[str, str | None]:
        backends = list(self._phase_sdpa_backends)
        reasons = list(self._phase_sdpa_reasons)
        self._phase_sdpa_backends.clear()
        self._phase_sdpa_reasons.clear()
        known = [b for b in backends if b in ("flash", "efficient", "math", "cudnn")]
        if known:
            if "math" in known:
                return "math", None
            return known[-1], None
        reason = reasons[-1] if reasons else "no_sdpa_call_in_phase"
        return "unknown", reason

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
        line: dict = {
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
        if self.log_sdpa_backend:
            sdpa_backend, sdpa_reason = self._consume_sdpa_backend()
            line["sdpa_backend"] = sdpa_backend
            if sdpa_backend == "unknown":
                line["sdpa_backend_reason"] = sdpa_reason or "no_sdpa_call_in_phase"
        self._fh.write(json.dumps(line) + "\n")
        self._fh.flush()
        self.fsync(self._fh.fileno())

    def close(self) -> None:
        if self._sdpa_mode is not None:
            self._sdpa_mode.__exit__(None, None, None)
            self._sdpa_mode = None
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


# -- training seeds (B3) -------------------------------------------------------------------------------------
ROLLOUT_SEED_RULE = (
    "torch.manual_seed(sha256('s2-train:<purpose>:<train_seed>:<step>:<group>')[:8] & 2**63-1) before each "
    "batched generate; sample i of a group is the i-th row of that seeded batch"
)


def train_rollout_seed(train_seed: int, step: int, group: int, *, purpose: str = "rollout") -> int:
    """The seed applied before one training rollout, fixed by (train seed, global step, group, purpose)."""
    key = f"s2-train:{purpose}:{int(train_seed)}:{int(step)}:{int(group)}"
    digest = hashlib.sha256(key.encode()).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFF_FFFF_FFFF_FFFF


def seed_everything(seed: int) -> None:
    """Seed python's random, numpy's global generator and torch (CPU and every CUDA device)."""
    import random

    random.seed(int(seed))
    try:
        import numpy as np

        np.random.seed(int(seed) % 2**32)
    except ModuleNotFoundError:  # pragma: no cover - numpy ships with torch
        pass
    require_torch().manual_seed(int(seed))


def capture_rng_state() -> tuple[dict[str, Any], dict[str, Any]]:
    """(tensors for torch.save, JSON-able python and numpy states) of every random generator in play."""
    import random

    torch = require_torch()
    cuda_mod = getattr(torch, "cuda", None)
    cuda_on = bool(cuda_mod is not None and cuda_mod.is_available())
    tensors = {
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": list(cuda_mod.get_rng_state_all()) if cuda_on else [],
    }
    py_ver, py_state, py_gauss = random.getstate()
    plain: dict[str, Any] = {"python": [py_ver, list(py_state), py_gauss], "numpy": None}
    try:
        import numpy as np

        name, keys, pos, has_gauss, cached = np.random.get_state()
        plain["numpy"] = [name, [int(k) for k in keys], int(pos), int(has_gauss), float(cached)]
    except ModuleNotFoundError:  # pragma: no cover - numpy ships with torch
        pass
    return tensors, plain


def restore_rng_state(tensors: Mapping[str, Any], plain: Mapping[str, Any]) -> None:
    import random

    torch = require_torch()
    torch.set_rng_state(tensors["torch_cpu"])
    cuda_states = list(tensors.get("torch_cuda") or [])
    cuda_mod = getattr(torch, "cuda", None)
    cuda_on = bool(cuda_mod is not None and cuda_mod.is_available())
    if cuda_states and not cuda_on:
        raise SystemExit(
            "refusing to resume (RESUME_RNG_DEVICE_MISMATCH): the saved state has CUDA generator states "
            "and this process has no CUDA device"
        )
    if cuda_states:  # pragma: no cover - needs the 5090
        cuda_mod.set_rng_state_all(cuda_states)
    py_ver, py_state, py_gauss = plain["python"]
    random.setstate((int(py_ver), tuple(int(x) for x in py_state), py_gauss))
    if plain.get("numpy") is not None:
        import numpy as np

        name, keys, pos, has_gauss, cached = plain["numpy"]
        np.random.set_state(
            (str(name), np.array(keys, dtype=np.uint32), int(pos), int(has_gauss), float(cached))
        )


# -- the critic arm (B4): mechanical tool hints, no learned model -------------------------------------------
CRITIC_HINT_SOURCE = (
    "mechanical: offrig quote rule 2 (quote_rule.quote_found) + lexical retrieval over the evidence"
)
CRITIC_NLI_STATUS = "off: smoke test S1 failed its pass line on both check types (receipts 2026-10-27, #172)"
CRITIC_RETRY_LINES = 2
CRITIC_LINE_CHARS = 200
CRITIC_PART_CHARS = 80


def critic_hint(claim: str, passages: Sequence[str], decoded: Sequence[tuple[str, bool]]) -> dict[str, Any]:
    """A tool-derived hint for an all-wrong group. It sees the claim, the evidence and the group's answers,
    never the gold label, so it cannot name the answer.

    Quote check: the first answer (in sample order) whose quote is not found in the evidence by offrig's rule
    is named by part. Retrieval: the evidence lines sharing the most content tokens with the claim.
    """
    quote_note: str
    quote_kind: str
    parsed_any = False
    unfound: tuple[int, str] | None = None
    for text, truncated in decoded:
        if truncated:
            continue
        parsed = parse_reply(text)
        if parsed is None:
            continue
        parsed_any = True
        for p_i, part in enumerate(parsed["evidence_quotes"], start=1):
            if not quote_found(str(part["quote"]), list(passages)):
                unfound = (p_i, str(part.get("part", "")))
                break
        if unfound is not None:
            break
    if unfound is not None:
        part_txt = " ".join(unfound[1].split())[:CRITIC_PART_CHARS]
        quote_kind = "quote_not_found"
        quote_note = f'The quote given for part {unfound[0]} ("{part_txt}") is not in the evidence.'
    elif parsed_any:
        quote_kind = "quotes_found"
        quote_note = (
            "Every quote in the earlier answers is in the evidence; check what those lines actually say."
        )
    else:
        quote_kind = "no_parseable_answer"
        quote_note = (
            "No earlier answer followed the reply contract; finish thinking and give the JSON object."
        )
    claim_toks = content_tokens(claim)
    scored: list[tuple[int, int, int, str]] = []
    for p_idx, passage in enumerate(passages):
        for l_idx, line in enumerate(str(passage).splitlines()):
            text_line = line.strip()
            if not text_line:
                continue
            overlap = len(content_tokens(text_line) & claim_toks)
            if overlap > 0:
                scored.append((-overlap, p_idx, l_idx, text_line[:CRITIC_LINE_CHARS]))
    lines = [s[3] for s in sorted(scored)[:CRITIC_RETRY_LINES]]
    hint = "Hint from a tool check of earlier attempts (it does not give the answer):\n- " + quote_note
    if lines:
        hint += "\n- Evidence lines that share the most terms with the claim:\n" + "\n".join(
            f"  {ln}" for ln in lines
        )
    return {"hint": hint, "quote_check": quote_kind, "retrieved_lines": len(lines)}


def critic_retry_prompt(row: dict, hint: str) -> str:
    """The retry's prompt: the unchanged task prompt with the hint appended."""
    return build_prompt(row) + "\n" + hint + "\n"


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
    sign_diagnostics: bool = False,
    prepass_baseline_k: dict[str, int] | None = None,
    group_residuals: list[tuple[str, int, float]] | None = None,
    save_all_records: list[dict] | None = None,
    save_all_path: Path | None = None,
    fsync: Callable[[int], None] = os.fsync,
    base_seed: int = 0,
    holdout_ids: Sequence[str] | set[str] | None = None,
    mask_wrong_positive_advantage: bool = False,
    dual_clip_c: float | None = None,
    ct_guard: CtCollapseGuard | None = None,
    start_step: int = 0,
    stop_after_updates: int | None = None,
    train_seed: int | None = None,
    trunc_guard: TruncationHaltGuard | None = None,
    guard_history_path: Path | None = None,
    arm: str | None = None,
    critic_archive_path: Path | None = None,
    critic_stats: dict[str, int] | None = None,
    progress: dict[str, Any] | None = None,
) -> tuple[list[dict], dict | None, bool]:
    """Run GRPO steps. Returns (step rows, a stop record or None, whether it ran out of memory).

    Training-run additions (all off by default, so the smoke run is unchanged):
    - ``start_step`` resumes the global step, so the draw position ``prompts[(step*gpu+k) % len]`` and the
      save-all step numbers carry on; ``stop_after_updates`` ends a chunk cleanly at an update boundary.
    - ``train_seed`` seeds every training rollout by (seed, step, group); save-all records the applied seed.
    - ``trunc_guard`` is the rolling truncation halt, checked at the update boundary beside the CT guard and
      sharing its halt path (the update's gradients are discarded).
    - ``guard_history_path`` appends every guard window (CT and truncation) to a crash-safe archive.
    - ``arm == "critic"`` adds a tool-hinted retry to each all-wrong group. Retries are scored on gold and
      archived at ``critic_archive_path``; they never enter the loss, the guards or the training save-all.
    - ``progress`` receives ``next_step``, and ``chunk_stop`` or ``halted``.

    Each step is ``groups_per_update`` prompt groups of G samples, one group at a time (gradients accumulate),
    so peak memory is one group's. The reference is the same model with the adapter off.

    ``mode(False)`` puts the model in rollout mode (gradient checkpointing off, the KV cache on) and
    ``mode(True)`` in training mode (checkpointing on, cache off, as the plan's memory table assumes);
    ``state()`` reports whether checkpointing is on. Both states are recorded in every step row.
    """
    if groups_per_update < 1:
        raise SystemExit(f"--groups-per-update must be >= 1, got {groups_per_update}")

    holdout_set = {str(cid) for cid in holdout_ids} if holdout_ids is not None else set()
    if holdout_set:
        overlap = sorted({str(r.get("id", "")) for r in prompts} & holdout_set)
        if overlap:
            raise SystemExit(
                f"refusing to train (holdout_item_in_training): "
                f"prompt {overlap[0]!r} is in the held-out evaluation set"
            )
    eff_dual_clip_c = (
        float(dual_clip_c)
        if dual_clip_c is not None
        else (DUAL_CLIP_C if mask_wrong_positive_advantage else None)
    )

    torch = require_torch()
    if empty_cache is None:
        cuda_mod = getattr(torch, "cuda", None)
        empty_cache = getattr(cuda_mod, "empty_cache", lambda: None)
    eos_ids = _eos_token_ids(tok)
    rows: list[dict] = []
    if save_all_path is not None:
        sa_p = Path(save_all_path)
        resumed_sa = _prepare_save_all_resume(
            sa_p,
            record_key=lambda r: (
                int(r.get("step", -1)),
                int(r.get("group", -1)),
                int(r.get("sample_index", -1)),
            ),
            fsync=fsync,
        )
        expected_total_groups = steps * groups_per_update
        resumed_groups = {
            (int(r.get("step", -1)), int(r.get("group", -1)))
            for r in resumed_sa
            if int(r.get("step", -1)) >= 0 and int(r.get("group", -1)) >= 0
        }
        if resumed_sa:
            if expected_total_groups > 0 and len(resumed_groups) >= expected_total_groups:
                raise SystemExit(
                    f"refusing to train (OUT_DIR_HAS_ARCHIVE, TRAINING_RESUME_UNSUPPORTED): "
                    f"complete training save-all archive {sa_p.name} already exists with "
                    f"{len(resumed_sa)} records across {len(resumed_groups)} groups"
                )
            raise SystemExit(
                f"refusing to train (TRAINING_RESUME_UNSUPPORTED, OUT_DIR_HAS_ARCHIVE): "
                f"training save-all archive {sa_p.name} already holds {len(resumed_sa)} records; "
                f"in-place resume is not supported: resume with --resume-from <chunk dir> and a fresh --out"
            )
        if save_all_records is not None:
            save_all_records.clear()
        if steps == 0 and not sa_p.exists():
            append_save_all(sa_p, (), fsync=fsync)
    if stop_after_updates is not None and int(stop_after_updates) < 1:
        raise SystemExit(f"--stop-after-updates must be >= 1, got {stop_after_updates!r}")
    is_critic = arm == "critic"
    if progress is not None:
        progress["next_step"] = int(start_step)

    ct_guard_written: dict[int, int] = {}
    if ct_guard is not None:
        ct_guard_written[id(ct_guard)] = len(ct_guard.window_history)

    def _append_history(entries: Sequence[dict]) -> None:
        if guard_history_path is not None and entries:
            append_save_all(guard_history_path, list(entries), fsync=fsync)

    oom_types = tuple(t for t in (getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None),) if t)
    for step in range(int(start_step), steps):
        try:
            started = time.time()
            if before_step:
                before_step(step)
            step_comp_n = step_trunc_n = 0
            step_hinted = step_retry_n = step_retry_correct = 0
            roll_s = train_s = 0.0
            tokens = zero_var = active_groups = 0
            rewards_all: list[float] = []
            optimiser.zero_grad(set_to_none=True)
            last_prompt_tokens = 0
            last_comp_lens: list[int] = []
            step_loss_sum = 0.0
            ckpt_train = ckpt_rollout = None
            step_correct_active_n = 0
            step_correct_zv_n = 0
            step_correct_neg_adv = 0
            active_outcome_var_sum = 0.0
            active_correct_var_sum = 0.0
            active_total_var_sum = 0.0
            step_guard_stop: dict[str, Any] | None = None
            need_details = (
                (group_logger is not None and group_logger.enabled)
                or sign_diagnostics
                or prepass_baseline_k is not None
                or mask_wrong_positive_advantage
                or ct_guard is not None
                or is_critic
            )
            for k in range(groups_per_update):
                row = prompts[(step * groups_per_update + k) % len(prompts)]
                if holdout_set and str(row.get("id", "")) in holdout_set:
                    raise SystemExit(
                        f"refusing to train (holdout_item_in_training): "
                        f"prompt {row.get('id')!r} is in the held-out evaluation set"
                    )
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
                applied_seed: int | None = None
                if train_seed is not None:
                    applied_seed = train_rollout_seed(train_seed, step, k)
                    torch.manual_seed(applied_seed)
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
                group_decoded: list[tuple[str, str, str, bool]] = []
                for c, trunc in zip(comps, truncs, strict=True):
                    text = tok.decode(c, skip_special_tokens=True)
                    think, answer = split_reply(text)
                    tt = len(tok.encode(think)) if think else 0
                    think_counts.append(tt)
                    r_val = _call_reward_fn(reward_fn, text, row, tt, trunc)
                    rewards.append(r_val)
                    if save_all_records is not None or save_all_path is not None or is_critic:
                        group_decoded.append((text, think, answer, bool(trunc)))
                    if eval_records is not None:
                        rollout_outcomes.append(
                            completion_eval_outcome(text, row, truncated=trunc, reward_val=r_val)
                        )
                    if need_details:
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

                unclamped_adv_list = list(adv_list)
                if mask_wrong_positive_advantage:
                    correct_flags = [bool(s.get("correct")) for s in group_samples]
                    adv_list = clamp_wrong_positive_advantages(adv_list, correct_flags)

                if ct_guard is not None:
                    guard_stop = ct_guard.record_group(
                        step=step,
                        group=k,
                        item_id=str(row.get("id", k)),
                        gold_verdict=str(row.get("label", "supported")),
                        verdicts=[s.get("verdict") for s in group_samples],
                        correct_flags=[bool(s.get("correct")) for s in group_samples],
                        advantages_list=unclamped_adv_list,
                        is_zero_variance=bool(is_zv),
                    )
                    if guard_stop is not None:
                        step_guard_stop = guard_stop
                    written = ct_guard_written.get(id(ct_guard), 0)
                    if guard_history_path is not None and len(ct_guard.window_history) > written:
                        _append_history(
                            [{"kind": "ct_guard_window", **w} for w in ct_guard.window_history[written:]]
                        )
                    ct_guard_written[id(ct_guard)] = len(ct_guard.window_history)

                if save_all_records is not None or save_all_path is not None:
                    grp_item_id = str(row.get("id", k))
                    # the seed actually applied to this group's rollout; None when the run is unseeded
                    grp_seed = applied_seed
                    grp_save_rows: list[dict] = []
                    for sample_idx, ((text, think, answer, is_trunc), r_val, adv_val) in enumerate(
                        zip(group_decoded, rewards, adv_list, strict=True)
                    ):
                        parsed_obj = None if is_trunc else parse_reply(text)
                        parse_status = (
                            "truncated" if is_trunc else ("malformed" if parsed_obj is None else "ok")
                        )
                        eff_verdict = (
                            None
                            if parsed_obj is None
                            else ("cannot_tell" if loop_detected(text) else parsed_obj["verdict"])
                        )
                        grp_save_rows.append(
                            {
                                "kind": "rollout",
                                "item_id": grp_item_id,
                                "seed": grp_seed,
                                "step": int(step),
                                "group": int(k),
                                "sample_index": int(sample_idx),
                                "text": text,
                                "thinking": think,
                                "answer": answer.strip(),
                                "parse_status": parse_status,
                                "verdict": eff_verdict,
                                "reward": round(float(r_val), 6),
                                "advantage": round(float(adv_val), 6),
                                "truncated": bool(is_trunc),
                                **({"arm": arm} if arm is not None else {}),
                            }
                        )
                    if save_all_path is not None:
                        append_save_all(save_all_path, grp_save_rows, fsync=fsync)
                    if save_all_records is not None:
                        save_all_records.extend(grp_save_rows)

                if is_critic and group_samples and not any(s.get("correct") for s in group_samples):
                    # the critic arm: one tool-hinted retry of an all-wrong group, scored on gold,
                    # never trained on
                    hint_rec = critic_hint(
                        str(row.get("claim", "")),
                        passages_of(row) if "context" in row else [],
                        [(d[0], d[3]) for d in group_decoded],
                    )
                    retry_seed = train_rollout_seed(
                        train_seed if train_seed is not None else 0, step, k, purpose="critic-retry"
                    )
                    torch.manual_seed(retry_seed)
                    t_retry = time.time()
                    retry_out = rollout(critic_retry_prompt(row, hint_rec["hint"]), g, max_new, TEMPERATURE)
                    device_sync()
                    roll_s += time.time() - t_retry
                    if len(retry_out) == 3:
                        _, retry_comps, retry_truncs = retry_out
                        retry_truncs = [bool(t) for t in retry_truncs]
                    else:
                        _, retry_comps = retry_out
                        retry_truncs = [
                            len(c) >= max_new and (not eos_ids or not any(t in eos_ids for t in c))
                            for c in retry_comps
                        ]
                    retry_rows: list[dict] = []
                    for r_idx, (rc, r_tr) in enumerate(zip(retry_comps, retry_truncs, strict=True)):
                        r_text = tok.decode(rc, skip_special_tokens=True)
                        r_think, r_answer = split_reply(r_text)
                        r_tt = len(tok.encode(r_think)) if r_think else 0
                        r_det = reward_details(
                            r_text,
                            str(row.get("label", "supported")),
                            passages_of(row) if "context" in row else "",
                            r_tt,
                            truncated=r_tr,
                            quote_reward=quote_reward,
                        )
                        r_correct = bool(r_det["correct"])
                        step_retry_correct += int(r_correct)
                        retry_rows.append(
                            {
                                "kind": "critic_retry",
                                "arm": "critic",
                                "item_id": str(row.get("id", k)),
                                "seed": int(retry_seed),
                                "step": int(step),
                                "group": int(k),
                                "sample_index": int(r_idx),
                                "hint": hint_rec["hint"],
                                "quote_check": hint_rec["quote_check"],
                                "text": r_text,
                                "thinking": r_think,
                                "answer": r_answer.strip(),
                                "verdict": r_det["verdict"],
                                "reward": round(_call_reward_fn(reward_fn, r_text, row, r_tt, r_tr), 6),
                                "correct": r_correct,
                                "truncated": bool(r_tr),
                                "in_gradient": False,
                            }
                        )
                    step_hinted += 1
                    step_retry_n += len(retry_rows)
                    if critic_archive_path is not None:
                        append_save_all(critic_archive_path, retry_rows, fsync=fsync)

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

                grp_k_correct = sum(1 for s in group_samples if s.get("correct")) if need_details else 0
                grp_trunc_n = sum(1 for tr in truncs if tr)
                step_trunc_n += grp_trunc_n
                step_comp_n += len(truncs)
                grp_correct_n: int | None = None
                grp_correct_neg_adv_n: int | None = None
                grp_outcome_var: float | None = None
                grp_correct_var: float | None = None
                grp_total_var: float | None = None
                if sign_diagnostics:
                    if mask_truncated:
                        diag_samples = [s for s, tr in zip(group_samples, truncs, strict=True) if not tr]
                        diag_outcomes = [v for v, tr in zip(verdict_scores, truncs, strict=True) if not tr]
                        diag_rewards = [r for r, tr in zip(rewards, truncs, strict=True) if not tr]
                        diag_advs = [a for a, tr in zip(adv_list, truncs, strict=True) if not tr]
                    else:
                        diag_samples = group_samples
                        diag_outcomes = verdict_scores
                        diag_rewards = rewards
                        diag_advs = adv_list
                    diag_correct_flags = [1.0 if s.get("correct") else 0.0 for s in diag_samples]
                    grp_correct_n = sum(1 for s in diag_samples if s.get("correct"))
                    grp_correct_neg_adv_n = sum(
                        1
                        for s, a_val in zip(diag_samples, diag_advs, strict=True)
                        if s.get("correct") and float(a_val) < 0.0
                    )
                    grp_outcome_var = float(statistics.pvariance(diag_outcomes)) if diag_outcomes else 0.0
                    grp_correct_var = (
                        float(statistics.pvariance(diag_correct_flags)) if diag_correct_flags else 0.0
                    )
                    grp_total_var = float(statistics.pvariance(diag_rewards)) if diag_rewards else 0.0
                    if is_zv:
                        step_correct_zv_n += grp_correct_n
                    else:
                        step_correct_active_n += grp_correct_n
                        step_correct_neg_adv += grp_correct_neg_adv_n
                        active_outcome_var_sum += grp_outcome_var
                        active_correct_var_sum += grp_correct_var
                        active_total_var_sum += grp_total_var

                grp_k_prepass: int | None = None
                grp_residual: float | None = None
                if prepass_baseline_k is not None:
                    pid_key = str(row.get("id", k))
                    if pid_key not in prepass_baseline_k:
                        raise SystemExit(
                            f"refusing to continue (prepass_baseline_missing_id): "
                            f"prompt {pid_key!r} has no baseline record"
                        )
                    grp_k_prepass = int(prepass_baseline_k[pid_key])
                    grp_residual = (grp_k_correct - grp_k_prepass) / float(len(comps))
                    if group_residuals is not None:
                        group_residuals.append((pid_key, grp_k_prepass, grp_residual))

                if group_logger is not None and group_logger.enabled:
                    zv_label = classify_zero_variance_group(group_samples) if is_zv else None
                    group_logger.record(
                        step=step,
                        group=k,
                        prompt_id=str(row.get("id", k)),
                        prompt_tokens=len(prompt_ids),
                        k_correct=grp_k_correct,
                        g=len(comps),
                        reward_std=statistics.pstdev(rewards) if rewards else 0.0,
                        verdict_reward_std=statistics.pstdev(verdict_scores) if verdict_scores else 0.0,
                        truncations=grp_trunc_n,
                        mean_thinking_tokens=statistics.fmean(think_counts) if think_counts else 0.0,
                        zero_variance_label=zv_label,
                        correct_n=grp_correct_n if sign_diagnostics else None,
                        correct_negative_adv_n=grp_correct_neg_adv_n if sign_diagnostics else None,
                        truncated_n=grp_trunc_n if sign_diagnostics else None,
                        outcome_var=grp_outcome_var if sign_diagnostics else None,
                        correct_var=grp_correct_var if sign_diagnostics else None,
                        total_reward_var=grp_total_var if sign_diagnostics else None,
                        k_prepass=grp_k_prepass,
                        residual=grp_residual,
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
                        dual_clip_c=eff_dual_clip_c,
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
                                dual_clip_c=eff_dual_clip_c,
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
            if trunc_guard is not None:
                trunc_stop = trunc_guard.record_update(
                    step=step, completions=step_comp_n, truncated=step_trunc_n
                )
                _append_history(trunc_guard.history[-1:])
                if trunc_stop is not None:
                    if step_guard_stop is None:
                        step_guard_stop = trunc_stop
                    else:
                        step_guard_stop = {**step_guard_stop, "also_triggered": trunc_stop}
            t2 = time.time()
            if step_guard_stop is not None:
                optimiser.zero_grad(set_to_none=True)
                if hasattr(model, "parameters"):
                    for p in model.parameters():
                        if getattr(p, "grad", None) is not None:
                            p.grad = None
                reported_loss = (
                    (step_loss_sum / groups_per_update) if zero_variance_mask else step_loss_sum
                )
            elif zero_variance_mask:
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
            rec: dict = {
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
            if sign_diagnostics:
                rec["step_correct_active_n"] = step_correct_active_n
                rec["step_correct_zv_n"] = step_correct_zv_n
                rec["correct_negative_adv_share"] = (
                    round(step_correct_neg_adv / step_correct_active_n, 6)
                    if step_correct_active_n > 0
                    else None
                )
                rec["outcome_to_total_var_ratio"] = (
                    round(active_outcome_var_sum / active_total_var_sum, 6)
                    if active_total_var_sum > 0.0
                    else None
                )
                rec["correct_to_total_var_ratio"] = (
                    round(active_correct_var_sum / active_total_var_sum, 6)
                    if active_total_var_sum > 0.0
                    else None
                )
            if trunc_guard is not None:
                rec["completions_n"] = step_comp_n
                rec["truncated_n"] = step_trunc_n
            if is_critic:
                rec["critic_hinted_groups"] = step_hinted
                rec["critic_retry_samples"] = step_retry_n
                rec["critic_retry_correct"] = step_retry_correct
                if critic_stats is not None:
                    critic_stats["hinted_groups"] = critic_stats.get("hinted_groups", 0) + step_hinted
                    critic_stats["retry_samples"] = critic_stats.get("retry_samples", 0) + step_retry_n
                    critic_stats["retry_correct"] = critic_stats.get("retry_correct", 0) + step_retry_correct
            rows.append(rec)
            if on_step:
                on_step(rec)
            if on_eval is not None and eval_records is not None:
                on_eval(evaluation_summary(eval_records, advantage_mode=advantage_mode))
            if step_guard_stop is not None:
                if progress is not None:
                    # the halted update was discarded: the model is the state after the previous update
                    progress["next_step"] = step
                    progress["halted"] = step_guard_stop
                return rows, step_guard_stop, False
            if progress is not None:
                progress["next_step"] = step + 1
            if (
                stop_after_updates is not None
                and (step + 1 - int(start_step)) >= int(stop_after_updates)
                and step + 1 < steps
            ):
                if progress is not None:
                    progress["chunk_stop"] = {
                        "reason": "stop_after_updates",
                        "updates": step + 1 - int(start_step),
                        "next_step": step + 1,
                    }
                break
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
    for_pass = bool(
        getattr(a, "rollout_only", False)
        or getattr(a, "logging_pass", False)
        or getattr(a, "postpass", False)
    )
    model, adapter_loaded, ckpt_files_sha, ckpt_sha = attach_or_load_adapter(
        LlamaForCausalLM(cfg),
        getattr(a, "checkpoint", None),
        for_pass=for_pass,
        init_seed=getattr(a, "train_seed", None),
    )
    census = restore_bf16(model) if getattr(a, "bf16_restore", False) else None
    params = [p for p in model.parameters() if p.requires_grad]
    optimiser = (
        torch.optim.AdamW(params, lr=LEARNING_RATE)
        if params
        else argparse.Namespace(zero_grad=lambda set_to_none=True: None, step=lambda: None)
    )
    mode, state = make_mode(model)
    pass_att = bool(getattr(a, "attention_mask", False))
    return {
        "model": model,
        "tok": tok,
        "mode": mode,
        "state": state,
        "optimiser": optimiser,
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
        "adapter_loaded": adapter_loaded,
        "checkpoint_sha256": ckpt_sha,
        "checkpoint_files_sha256": ckpt_files_sha,
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
    from peft import prepare_model_for_kbit_training
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
    for_pass = bool(
        getattr(a, "rollout_only", False)
        or getattr(a, "logging_pass", False)
        or getattr(a, "postpass", False)
    )
    model, adapter_loaded, ckpt_files_sha, ckpt_sha = attach_or_load_adapter(
        base,
        getattr(a, "checkpoint", None),
        for_pass=for_pass,
        init_seed=getattr(a, "train_seed", None),
    )
    census = restore_bf16(model) if getattr(a, "bf16_restore", False) else None
    model.print_trainable_parameters()
    torch.cuda.reset_peak_memory_stats()
    params = [p for p in model.parameters() if p.requires_grad]
    optimiser = (
        torch.optim.AdamW(params, lr=LEARNING_RATE)
        if params
        else argparse.Namespace(zero_grad=lambda set_to_none=True: None, step=lambda: None)
    )

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
        "adapter_loaded": adapter_loaded,
        "checkpoint_sha256": ckpt_sha,
        "checkpoint_files_sha256": ckpt_files_sha,
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


def compute_prompt_lengths(prompts: Sequence[dict], tok, *, stress_pool: str = "selected") -> dict:
    """Tokenize ``prompts`` (CPU only, no model, no GPU) and return length statistics and stress choice."""
    lengths = sorted(len(tok.encode(build_prompt(r))) for r in prompts)
    if not lengths:
        return {
            "count": 0,
            "min": 0,
            "median": 0,
            "p90": 0,
            "max": 0,
            "stress_pool": stress_pool,
            "stress_prompt_id": None,
            "stress_prompt_tokens": 0,
        }
    stress_row, stress_tokens = select_stress_prompt(prompts, tok)
    p90_idx = min(len(lengths) - 1, max(0, math.ceil(0.9 * len(lengths)) - 1))
    return {
        "count": len(lengths),
        "min": lengths[0],
        "median": statistics.median(lengths),
        "p90": lengths[p90_idx],
        "max": lengths[-1],
        "stress_pool": stress_pool,
        "stress_prompt_id": str(stress_row["id"]),
        "stress_prompt_tokens": int(stress_tokens),
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
    stack_versions: Mapping[str, str | None] | None = None,
    pip_freeze_sha256: str | None = None,
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

    eff_stack_versions = (
        dict(stack_versions)
        if stack_versions is not None
        else {k: package_version(k) for k in FROZEN_STACK_VERSIONS}
    )
    probe_doc = {
        "test": "S2-probe",
        "path": a.path,
        "prompt_id": longest_row["id"],
        "prompt_tokens": prompt_len,
        "g": a.g,
        "flags": active_flags(a),
        "stack_versions": eff_stack_versions,
        "pip_freeze_sha256": pip_freeze_sha256,
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


def k_band_name(k_correct: int) -> str:
    """Return the pre-registered Pass-A difficulty band ('[0,5]', '[6,7]', or '8') for ``k_correct``."""
    k_val = int(k_correct)
    if k_val <= 5:
        return "[0,5]"
    if k_val <= 7:
        return "[6,7]"
    return "8"


def checkpoint_file_shas(checkpoint_path: Path | None) -> dict[str, str] | None:
    """Return a sorted ``{rel_posix_path: sha256}`` map for every file in ``checkpoint_path``, or None."""
    if checkpoint_path is None:
        return None
    cp = Path(checkpoint_path)
    if not cp.exists():
        raise SystemExit(
            f"refusing to start (checkpoint_missing): checkpoint does not exist: {cp.name}"
        )
    if cp.is_file():
        return {cp.name: hashlib.sha256(cp.read_bytes()).hexdigest()}
    files = sorted(p for p in cp.rglob("*") if p.is_file())
    if not files:
        raise SystemExit(
            f"refusing to start (checkpoint_empty): checkpoint directory has no files: {cp.name}"
        )
    return {
        p.relative_to(cp).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in files
    }


def checkpoint_sha256(checkpoint_path: Path | None) -> str | None:
    """Return a deterministic sha256 digest of a checkpoint file or directory, or None when omitted."""
    file_shas = checkpoint_file_shas(checkpoint_path)
    if file_shas is None:
        return None
    cp = Path(checkpoint_path)
    if cp.is_file():
        return next(iter(file_shas.values()))
    entries = [f"{rel}:{sha}" for rel, sha in file_shas.items()]
    return hashlib.sha256("\n".join(entries).encode("utf-8")).hexdigest()


def attach_or_load_adapter(
    base,
    checkpoint_path: Path | None,
    *,
    for_pass: bool = False,
    adapter_name: str = "default",
    init_seed: int | None = None,
) -> tuple[Any, bool, dict[str, str] | None, str | None]:
    """Attach a fresh LoRA adapter or load an existing PEFT checkpoint into ``base``.

    ``init_seed`` (a training run's --train-seed) seeds every generator just before a fresh LoRA is made, so
    its random init (lora_A) is a function of the seed alone.
    """
    from peft import LoraConfig, PeftModel, get_peft_model

    if checkpoint_path is None:
        if init_seed is not None:
            seed_everything(init_seed)
        model = get_peft_model(
            base,
            LoraConfig(
                r=LORA_R,
                lora_alpha=LORA_ALPHA,
                target_modules=list(LORA_TARGETS),
                lora_dropout=0.0,
            ),
            adapter_name=adapter_name,
        )
        return model, False, None, None
    cp = Path(checkpoint_path)
    files_sha = checkpoint_file_shas(cp)
    combined_sha = checkpoint_sha256(cp)
    try:
        if isinstance(base, PeftModel):
            base.load_adapter(str(cp), adapter_name=adapter_name, is_trainable=not for_pass)
            model = base
        else:
            model = PeftModel.from_pretrained(
                base, str(cp), adapter_name=adapter_name, is_trainable=not for_pass
            )
    except Exception as exc:
        raise SystemExit(
            f"refusing to start (checkpoint_adapter_not_loaded): "
            f"failed to load adapter from {cp.name}: {exc}"
        ) from exc
    eff_name = str(adapter_name or getattr(model, "active_adapter", None) or "default")
    sha_by_adapter = dict(getattr(model, "_loaded_checkpoint_sha256_by_adapter", None) or {})
    if combined_sha is not None:
        sha_by_adapter[eff_name] = combined_sha
    model._loaded_checkpoint_sha256_by_adapter = sha_by_adapter
    files_by_adapter = dict(getattr(model, "_loaded_checkpoint_files_sha256_by_adapter", None) or {})
    if files_sha is not None:
        files_by_adapter[eff_name] = dict(files_sha)
    model._loaded_checkpoint_files_sha256_by_adapter = files_by_adapter
    model._loaded_checkpoint_sha256 = combined_sha
    model._loaded_checkpoint_files_sha256 = files_sha
    return model, True, files_sha, combined_sha


# -- the training run: arguments, header, chunk state, resume and verdict (B1, B2, B4, B6) -------------------
ARMS = ("mech", "critic")
TRAIN_HEADER_SCHEMA = "s2-train-header-v1"
TRAIN_STATE_DIR = "train-state"
TRAIN_STATE_FILES = ("trainable.pt", "optimizer.pt", "rng.pt")
TRAIN_ONLY_FLAGS = (
    ("--train-seed", "train_seed"),
    ("--stop-after-updates", "stop_after_updates"),
    ("--resume-from", "resume_from"),
    ("--trunc-window-updates", "trunc_window_updates"),
    ("--trunc-halt-rate", "trunc_halt_rate"),
)
# active_flags keys that change from chunk to chunk of one run; everything else must match on resume
CHUNK_LOCAL_FLAGS = ("stop_after_updates", "resume_from")
TRAIN_EXIT_HALT = 2


def is_training_run(a: argparse.Namespace) -> bool:
    """A training run is named by --arm, or by any training-only flag (which then requires --arm)."""
    return getattr(a, "arm", None) is not None or any(
        getattr(a, attr, None) is not None for _, attr in TRAIN_ONLY_FLAGS
    )


def validate_training_args(a: argparse.Namespace) -> bool:
    """Refuse a training run that does not name every required setting. Returns whether this is one.

    Required, with no defaults: --arm, --train-seed, --trunc-window-updates and --trunc-halt-rate.
    """
    if not is_training_run(a):
        return False
    missing = [
        flag
        for flag, attr in (
            ("--arm", "arm"),
            ("--train-seed", "train_seed"),
            ("--trunc-window-updates", "trunc_window_updates"),
            ("--trunc-halt-rate", "trunc_halt_rate"),
        )
        if getattr(a, attr, None) is None
    ]
    if missing:
        raise SystemExit(
            f"refusing to start (training_flags_required): a training run needs {', '.join(missing)} "
            f"(no defaults)"
        )
    if a.arm not in ARMS:
        raise SystemExit(f"refusing to start (unknown_arm): --arm must be one of {ARMS}, got {a.arm!r}")
    for flag in ("rollout_only", "logging_pass", "postpass", "probe_memory", "prompt_lengths"):
        if getattr(a, flag, False):
            raise SystemExit(
                f"refusing to start (training_flags_in_other_mode): training flags cannot be combined with "
                f"--{flag.replace('_', '-')}"
            )
    if getattr(a, "paired_estimators", None) is not None:
        raise SystemExit(
            "refusing to start (training_flags_in_other_mode): training flags cannot be combined with "
            "--paired-estimators"
        )
    if int(getattr(a, "stress_prefix", 0) or 0) > 0:
        raise SystemExit(
            "refusing to start (training_stress_prefix): a training run cannot take --stress-prefix "
            "(it would train on the stress prompt)"
        )
    if getattr(a, "path", "hf") != "hf":
        raise SystemExit(
            "refusing to start (training_path_hf_only): training runs use path (a), --path hf; llama.cpp "
            "rollouts are not seeded by --train-seed"
        )
    seed = a.train_seed
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise SystemExit(
            f"refusing to start (invalid_train_seed): --train-seed must be an int >= 0, got {seed!r}"
        )
    TruncationHaltGuard(window_updates=a.trunc_window_updates, halt_rate=a.trunc_halt_rate)  # validates both
    sau = getattr(a, "stop_after_updates", None)
    if sau is not None and (isinstance(sau, bool) or not isinstance(sau, int) or sau < 1):
        raise SystemExit(
            f"refusing to start (invalid_stop_after_updates): --stop-after-updates must be >= 1, got {sau!r}"
        )
    resume = getattr(a, "resume_from", None)
    if resume is not None:
        rp = Path(resume)
        if getattr(a, "checkpoint", None) is not None:
            raise SystemExit(
                "refusing to start (resume_with_checkpoint): --resume-from restores the run's own adapter; "
                "it cannot be combined with --checkpoint"
            )
        if not (rp / TRAIN_STATE_DIR / "state.json").is_file():
            raise SystemExit(
                f"refusing to resume (RESUME_STATE_MISSING): {rp.name}/{TRAIN_STATE_DIR}/state.json not found"
            )
        if rp.resolve() == Path(a.out).resolve():
            raise SystemExit(
                "refusing to resume (RESUME_SAME_DIR): each chunk writes to its own --out; "
                "--resume-from names the previous chunk's directory"
            )
    if (Path(a.out) / TRAIN_STATE_DIR).exists():
        raise SystemExit(
            f"refusing to start (OUT_DIR_HAS_TRAIN_STATE): "
            f"{Path(a.out).name}/{TRAIN_STATE_DIR} already exists"
        )
    return True


def aspire_commit(cwd: Path | None = None) -> str | None:
    """The aspire-si commit this script runs from (git HEAD of its checkout), or None outside a checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd or Path(__file__).resolve().parent),
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    sha = out.stdout.strip()
    return sha if re.fullmatch(r"[0-9a-f]{40}", sha) else None


def build_train_header(
    a: argparse.Namespace,
    *,
    flags: Mapping[str, Any],
    draw_record: Mapping[str, Any],
    prompt_ids: Sequence[str],
    gold_hashes: Mapping[str, str],
    revision_resolved: str | None,
    stack_versions: Mapping[str, Any] | None,
    pip_freeze_sha256: str | None,
    commit: str | None,
    checkpoint_sha: str | None,
) -> dict[str, Any]:
    """Everything that must be identical across the chunks of one run. Resume refuses on any difference."""
    settings = {
        "steps": int(a.steps),
        "prompts": int(a.prompts),
        "g": int(a.g),
        "max_new": int(a.max_new),
        "dry": bool(a.dry),
        **{k: v for k, v in flags.items() if k not in CHUNK_LOCAL_FLAGS},
    }
    return {
        "schema": TRAIN_HEADER_SCHEMA,
        "arm": a.arm,
        "train_seed": int(a.train_seed),
        "draw_seed": getattr(a, "draw_seed", None),
        "holdout_seed": draw_record.get("holdout_seed"),
        "holdout_ids_sha256": draw_record.get("holdout_ids_sha256"),
        "train_ids_sha256": draw_record.get("train_ids_sha256"),
        "drawn_ids_sha256": draw_record.get("drawn_ids_sha256"),
        "prompt_ids_sha256": ids_sha256(list(prompt_ids)),
        "aspire_commit": commit,
        "script_sha256": script_sha256(),
        "revision_requested": a.revision,
        "revision_resolved": revision_resolved,
        "stack_versions": dict(stack_versions) if stack_versions is not None else None,
        "pip_freeze_sha256": pip_freeze_sha256,
        "rnd_commit": a.rnd_commit,
        "gold_files_sha256": dict(gold_hashes),
        "start_policy": "base_fresh_lora" if checkpoint_sha is None else "checkpoint",
        "start_checkpoint_sha256": checkpoint_sha,
        "rollout_seed_rule": ROLLOUT_SEED_RULE,
        "scheduler": None,  # constant learning rate: the trainer has no LR scheduler
        "trunc_window_updates": int(a.trunc_window_updates),
        "trunc_halt_rate": float(a.trunc_halt_rate),
        "critic": (
            {"hint_source": CRITIC_HINT_SOURCE, "nli": CRITIC_NLI_STATUS, "retries_in_gradient": False}
            if a.arm == "critic"
            else None
        ),
        "settings": settings,
    }


def header_differences(saved: Mapping[str, Any], current: Mapping[str, Any]) -> list[str]:
    """Dotted keys whose values differ between two headers (settings compared key by key)."""
    diffs: list[str] = []
    for key in sorted(set(saved) | set(current)):
        if key == "settings":
            s_set, c_set = dict(saved.get(key) or {}), dict(current.get(key) or {})
            diffs.extend(
                f"settings.{k}" for k in sorted(set(s_set) | set(c_set)) if s_set.get(k) != c_set.get(k)
            )
        elif saved.get(key) != current.get(key):
            diffs.append(key)
    return diffs


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_train_state(
    out_dir: Path,
    *,
    model,
    optimiser,
    header: Mapping[str, Any],
    next_step: int,
    steps_total: int,
    chunk_index: int,
    ct_guard: CtCollapseGuard | None,
    trunc_guard: TruncationHaltGuard,
    save_all_chain: Sequence[Mapping[str, Any]],
    adapter_sha: str | None,
    halted: Mapping[str, Any] | None,
    resumed_from: Mapping[str, Any] | None,
) -> str:
    """Write ``<out>/train-state`` (LoRA tensors, AdamW state, RNG states, guards, step, save-all chain).

    It is written to ``train-state.partial`` and renamed into place, so a crash never leaves a half state; the
    returned sha256 is of ``state.json``, which lists every other file's sha256.
    """
    torch = require_torch()
    final_dir = Path(out_dir) / TRAIN_STATE_DIR
    tmp_dir = Path(out_dir) / f"{TRAIN_STATE_DIR}.partial"
    if tmp_dir.exists():
        for p in sorted(tmp_dir.iterdir()):
            p.unlink()
        tmp_dir.rmdir()
    tmp_dir.mkdir(parents=True)
    trainable = {
        name: p.detach().to("cpu").clone() for name, p in model.named_parameters() if p.requires_grad
    }
    torch.save(trainable, tmp_dir / "trainable.pt")
    torch.save(optimiser.state_dict(), tmp_dir / "optimizer.pt")
    rng_tensors, rng_plain = capture_rng_state()
    torch.save(rng_tensors, tmp_dir / "rng.pt")
    state = {
        "header": dict(header),
        "next_step": int(next_step),
        "steps_total": int(steps_total),
        "draw_position": int(next_step) * int(header["settings"].get("groups_per_update", GROUPS_PER_STEP)),
        "chunk_index": int(chunk_index),
        "complete": int(next_step) >= int(steps_total),
        "halted": dict(halted) if halted is not None else None,
        "resumed_from": dict(resumed_from) if resumed_from is not None else None,
        "ct_guard": ct_guard.to_state() if ct_guard is not None else None,
        "trunc_guard": trunc_guard.to_state(),
        "save_all_chain": [dict(c) for c in save_all_chain],
        "adapter_sha256": adapter_sha,
        "rng_plain": rng_plain,
        "trainable_names": sorted(trainable),
        "files_sha256": {name: _file_sha256(tmp_dir / name) for name in TRAIN_STATE_FILES},
    }
    with (tmp_dir / "state.json").open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(state, indent=1))
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp_dir, final_dir)
    return _file_sha256(final_dir / "state.json")


def load_train_state(resume_dir: Path, *, header: Mapping[str, Any], steps_total: int) -> dict[str, Any]:
    """Read and verify a chunk's train-state. Refuses on any header difference, a changed file, a halted or a
    finished run, or a previous save-all archive that no longer matches the sha256 the state recorded."""
    torch = require_torch()
    sdir = Path(resume_dir) / TRAIN_STATE_DIR
    state_path = sdir / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    for name, want in dict(state.get("files_sha256") or {}).items():
        fp = sdir / name
        if not fp.is_file() or _file_sha256(fp) != want:
            raise SystemExit(f"refusing to resume (RESUME_STATE_FILE_CHANGED): {name} is missing or changed")
    diffs = header_differences(state.get("header") or {}, header)
    if diffs:
        raise SystemExit(
            f"refusing to resume (RESUME_HEADER_MISMATCH): the run header differs in {', '.join(diffs)}"
        )
    if state.get("halted") is not None:
        raise SystemExit(
            f"refusing to resume (RESUME_FROM_HALTED): the run halted ({state['halted'].get('reason')}); "
            f"a halt is stop-and-report, not resumable"
        )
    if int(state["next_step"]) >= int(steps_total):
        raise SystemExit("refusing to resume (RESUME_RUN_COMPLETE): the run already completed every update")
    for link in state.get("save_all_chain") or []:
        if link.get("chunk_dir") == Path(resume_dir).name and link.get("sha256") is not None:
            archive = Path(resume_dir) / str(link["file"])
            if not archive.is_file() or _file_sha256(archive) != link["sha256"]:
                raise SystemExit(
                    "refusing to resume (RESUME_SAVE_ALL_CHANGED): the previous chunk's save-all archive is "
                    "missing or no longer matches the state"
                )
    state["_trainable"] = torch.load(sdir / "trainable.pt", map_location="cpu", weights_only=True)
    state["_optimizer"] = torch.load(sdir / "optimizer.pt", map_location="cpu", weights_only=True)
    state["_rng"] = torch.load(sdir / "rng.pt", map_location="cpu", weights_only=True)
    state["_state_sha256"] = _file_sha256(state_path)
    return state


def apply_train_state(state: Mapping[str, Any], *, model, optimiser) -> None:
    """Put the saved LoRA tensors and AdamW state into a freshly built model, then restore every RNG."""
    saved = state["_trainable"]
    live = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if sorted(live) != sorted(saved):
        raise SystemExit(
            "refusing to resume (RESUME_PARAMS_MISMATCH): the model's trainable parameters differ from the "
            "saved state's"
        )
    for name, p in live.items():
        src = saved[name]
        if tuple(src.shape) != tuple(p.shape) or src.dtype != p.dtype:
            raise SystemExit(f"refusing to resume (RESUME_PARAMS_MISMATCH): {name} shape or dtype differs")
        with require_torch().no_grad():
            p.copy_(src.to(p.device))
    optimiser.load_state_dict(state["_optimizer"])
    restore_rng_state(state["_rng"], state["rng_plain"])


def train_verdict_line(
    *,
    stopped: Mapping[str, Any] | None,
    oom: bool,
    next_step: int,
    steps_total: int,
    chunk_stop: Mapping[str, Any] | None,
    fits: bool | None,
    dry: bool,
) -> str:
    """The training run's verdict. S2-TRAIN-HALT-<rule>: a guard or truncation halt (stop and report).
    S2-TRAIN-COMPLETE: every update ran. S2-TRAIN-CHUNK-OK: a chunk ended cleanly at --stop-after-updates.
    S2-TRAIN-FAIL: anything else (an OOM, a non-finite loss, a timeout, a VRAM peak over the line)."""
    if oom:
        return "S2-TRAIN-FAIL"
    if stopped:
        rules = stopped.get("triggered_rules") or []
        if rules:
            return f"S2-TRAIN-HALT-{rules[0]}"
        return "S2-TRAIN-FAIL"
    if not dry and fits is False:
        return "S2-TRAIN-FAIL"
    if int(next_step) >= int(steps_total):
        return "S2-TRAIN-COMPLETE"
    if chunk_stop is not None:
        return "S2-TRAIN-CHUNK-OK"
    return "S2-TRAIN-FAIL"


def validate_replay_check_args(a: argparse.Namespace) -> tuple[Path | None, str | None, str | None]:
    """Validate ``--replay-check``, ``--replay-check-sha256``, and ``--replay-check-off-reason``.

    Returns ``(replay_check_path, replay_check_sha256, replay_check_off_reason)``.
    """
    logging_pass = bool(getattr(a, "logging_pass", False))
    rc_path_raw = getattr(a, "replay_check", None)
    rc_path = Path(rc_path_raw) if rc_path_raw is not None else None
    rc_sha_raw = getattr(a, "replay_check_sha256", None)
    rc_sha = str(rc_sha_raw).strip().lower() if rc_sha_raw and str(rc_sha_raw).strip() else None
    rc_off_raw = getattr(a, "replay_check_off_reason", None)
    rc_off_reason = str(rc_off_raw).strip() if rc_off_raw and str(rc_off_raw).strip() else None

    if rc_path is not None and rc_off_reason is not None:
        raise SystemExit(
            "refusing to start (replay_check_conflicting_args): "
            "--replay-check and --replay-check-off-reason cannot both be set"
        )
    if (rc_path is None) != (rc_sha is None):
        raise SystemExit(
            "refusing to start (replay_check_missing_arg): "
            "--replay-check and --replay-check-sha256 must be given together"
        )
    if logging_pass and rc_path is None and rc_off_reason is None:
        raise SystemExit(
            "refusing to start (logging_pass_requires_replay_check): "
            "--logging-pass requires --replay-check <b2.jsonl> --replay-check-sha256 <sha> "
            "unless a non-empty --replay-check-off-reason is provided"
        )
    return rc_path, rc_sha, rc_off_reason


def validate_pass_mode_args(a: argparse.Namespace) -> tuple[bool, bool, int, str | None]:
    """Validate ``--logging-pass``, ``--postpass``, ``--checkpoint``, and pass seeds.

    Returns ``(is_pass_mode, is_descriptive, effective_seed, checkpoint_sha256)``.
    """
    logging_pass = bool(getattr(a, "logging_pass", False))
    postpass = bool(getattr(a, "postpass", False))
    rollout_only = bool(getattr(a, "rollout_only", False))
    ckpt_path = getattr(a, "checkpoint", None)

    if logging_pass and postpass:
        raise SystemExit(
            "refusing to start (conflicting_pass_flags): --logging-pass and --postpass cannot be combined"
        )
    if logging_pass and ckpt_path is not None:
        raise SystemExit(
            "refusing to start (logging_pass_checkpoint_forbidden): "
            "--logging-pass is for the starting policy and cannot take --checkpoint"
        )
    ckpt_sha = checkpoint_sha256(ckpt_path)

    pass_seed_arg = getattr(a, "pass_seed", None)
    prepass_seed_arg = getattr(a, "prepass_seed", PREPASS_BASE_SEED)
    if logging_pass:
        if pass_seed_arg is not None and int(pass_seed_arg) != 1:
            raise SystemExit(
                f"refusing to start (logging_pass_seed_must_be_1): "
                f"--logging-pass requires post-pass seed 1, got {pass_seed_arg}"
            )
        if prepass_seed_arg not in (PREPASS_BASE_SEED, 1):
            raise SystemExit(
                f"refusing to start (logging_pass_seed_must_be_1): "
                f"--logging-pass requires post-pass seed 1, got {prepass_seed_arg}"
            )
        for setting_key, expected_val in B2_REPLAY_SETTINGS.items():
            actual_val = getattr(a, setting_key, None)
            if type(actual_val) is not type(expected_val) or actual_val != expected_val:
                raise SystemExit(
                    f"refusing to start (logging_pass_b2_settings_mismatch): "
                    f"--logging-pass requires B2 replay setting {setting_key}={expected_val!r}, "
                    f"got {actual_val!r}"
                )
        validate_replay_check_args(a)
        eff_seed = 1
        descriptive = True
    elif postpass or ckpt_path is not None:
        validate_replay_check_args(a)
        eff_seed = int(pass_seed_arg if pass_seed_arg is not None else prepass_seed_arg)
        if eff_seed in (0, 1):
            raise SystemExit(
                f"refusing to start (postpass_reserved_seed): "
                f"post-pass for a trained checkpoint refuses seed {eff_seed} "
                f"(seeds 0 and 1 are reserved for pre-pass and starting-policy logging pass)"
            )
        if postpass and ckpt_path is None:
            raise SystemExit(
                "refusing to start (postpass_requires_checkpoint): "
                "--postpass requires --checkpoint"
            )
        descriptive = False
    else:
        validate_replay_check_args(a)
        eff_seed = int(pass_seed_arg if pass_seed_arg is not None else prepass_seed_arg)
        descriptive = False

    is_pass_mode = bool(rollout_only or logging_pass or postpass)
    stop_after_new_raw = getattr(a, "stop_after_new", None)
    if stop_after_new_raw is not None:
        if (
            isinstance(stop_after_new_raw, bool)
            or not isinstance(stop_after_new_raw, int)
            or stop_after_new_raw < 1
        ):
            raise SystemExit(
                f"refusing to start (invalid_stop_after_new): "
                f"--stop-after-new must be >= 1, got {stop_after_new_raw!r}"
            )
        if not is_pass_mode:
            raise SystemExit(
                "refusing to start (unused_stop_after_new): "
                "--stop-after-new requires --rollout-only, --logging-pass, or --postpass"
            )
    return is_pass_mode, descriptive, eff_seed, ckpt_sha


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
    reproducible: bool = True,
    save_all: bool = True,
    descriptive: bool = False,
    checkpoint_sha: str | None = None,
    checkpoint_files_sha: Mapping[str, str] | None = None,
    pass_a_sha: str | None = None,
    stack_versions: Mapping[str, str | None] | None = None,
    pip_freeze_sha256: str | None = None,
    b2_replay_settings: Mapping[str, Any] | None = None,
    replay_check_sha256: str | None = None,
    replay_check_off_reason: str | None = None,
    stop_after_new: int | None = None,
) -> dict:
    hdr: dict = {
        "ids_sha256": ids_sha256(prompt_ids),
        "script_sha256": script_hash if script_hash is not None else script_sha256(),
        "gold_files_sha256": dict(gold_hashes),
        "g": int(g),
        "max_new": int(max_new),
        "quote_reward": str(quote_reward),
        "seed": int(seed),
        "model_revision": model_revision,
        "reproducible": bool(reproducible),
        "descriptive": bool(descriptive),
        "stack_versions": (
            dict(stack_versions)
            if stack_versions is not None
            else {k: package_version(k) for k in FROZEN_STACK_VERSIONS}
        ),
        "pip_freeze_sha256": pip_freeze_sha256,
    }
    if not save_all:
        hdr["save_all"] = False
    if descriptive:
        hdr["b2_replay_settings"] = dict(
            b2_replay_settings if b2_replay_settings is not None else B2_REPLAY_SETTINGS
        )
    if replay_check_sha256 is not None:
        hdr["replay_check_sha256"] = str(replay_check_sha256)
    if replay_check_off_reason is not None:
        hdr["replay_check_off_reason"] = str(replay_check_off_reason)
    if checkpoint_sha is not None:
        hdr["checkpoint_sha256"] = checkpoint_sha
    if checkpoint_files_sha is not None:
        hdr["checkpoint_files_sha256"] = dict(checkpoint_files_sha)
    if pass_a_sha is not None:
        hdr["pass_a_sha256"] = pass_a_sha
    if stop_after_new is not None:
        hdr["stop_after_new"] = int(stop_after_new)
    return hdr


def prepass_item_record(
    *,
    row: dict,
    prompt_ids: Sequence[int],
    comps: Sequence[Sequence[int]],
    truncs: Sequence[bool],
    tok,
    reward_fn: Callable,
    quote_reward: str = "v1",
    save_all_out: list[dict] | None = None,
    pass_id: str = "prepass_seed0",
    item_seed_val: int = 0,
    pass_seed: int = 0,
    descriptive: bool = False,
    pass_a_k: dict[str, int] | None = None,
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

    for sample_idx, (c, trunc) in enumerate(zip(comps, truncs, strict=True)):
        is_trunc = bool(trunc)
        if is_trunc:
            trunc_count += 1
        text = tok.decode(c, skip_special_tokens=True)
        think, answer = split_reply(text)
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
        if not is_trunc:
            if v in VERDICTS:
                verdict_counts[v] += 1
            else:
                verdict_counts["malformed"] += 1
        if det.get("correct"):
            k_correct += 1
        parsed = None if is_trunc else parse_reply(text)
        if parsed is not None:
            parts = parsed["evidence_quotes"]
            if parts and all(quote_found(p["quote"], passages) for p in parts):
                quote_pass += 1
        if save_all_out is not None:
            parse_status = "truncated" if is_trunc else ("malformed" if parsed is None else "ok")
            save_all_out.append(
                {
                    "kind": "pass",
                    "item_id": str(row["id"]),
                    "pass_id": str(pass_id),
                    "seed": int(item_seed_val),
                    "pass_seed": int(pass_seed),
                    "sample_index": int(sample_idx),
                    "text": text,
                    "thinking": think,
                    "answer": answer.strip(),
                    "parse_status": parse_status,
                    "verdict": v,
                    "correct": bool(det.get("correct")),
                    "reward": round(float(r_val), 6),
                    "truncated": bool(is_trunc),
                }
            )
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
    rec["descriptive"] = bool(descriptive)
    rec["pass_seed"] = int(pass_seed)
    item_id_key = str(row["id"])
    if pass_a_k is not None and item_id_key in pass_a_k:
        rec["pass_a_k"] = int(pass_a_k[item_id_key])
        rec["pass_a_band"] = k_band_name(rec["pass_a_k"])
    return rec, rollout_outcomes


def _k_correct_hist(records: Sequence[dict], g: int) -> dict[str, int]:
    hist = {str(k): 0 for k in range(g + 1)}
    for r in records:
        k_val = str(int(r["k_correct"]))
        hist[k_val] = hist.get(k_val, 0) + 1
    return hist


def item_confident_errors(rec: Mapping[str, Any]) -> int:
    """Count confident errors for one item record:
    - 'supported' on gold unsupported or gold cannot_tell, plus
    - 'unsupported' on gold supported.
    """
    gold = str(rec.get("gold_verdict", "supported"))
    if gold == "supported":
        return int(rec.get("unsupported", 0))
    return int(rec.get("supported", 0))


def build_confusion_matrix_3x3(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    """Build the 3x3 gold_verdict x predicted_verdict matrix plus malformed and truncations per gold class."""
    matrix: dict[str, dict[str, int]] = {
        gv: {
            "supported": 0,
            "unsupported": 0,
            "cannot_tell": 0,
            "malformed": 0,
            "truncations": 0,
            "total": 0,
        }
        for gv in VERDICTS
    }
    for r in records:
        gv = str(r.get("gold_verdict", "supported"))
        if gv not in matrix:
            continue
        row_m = matrix[gv]
        row_m["supported"] += int(r.get("supported", 0))
        row_m["unsupported"] += int(r.get("unsupported", 0))
        row_m["cannot_tell"] += int(r.get("cannot_tell", 0))
        row_m["malformed"] += int(r.get("malformed", 0))
        row_m["truncations"] += int(r.get("truncations", 0))
        row_m["total"] += int(r.get("g", 0))
    return matrix


def compute_hedge_flag(
    records: Sequence[Mapping[str, Any]],
    *,
    baseline_ct_share: float = 0.42,
    threshold: float | None = None,
) -> dict[str, Any]:
    """Compute the secondary hedge flag on gold cannot_tell items: CT's share of correct answers
    (cannot_tell / (cannot_tell + unsupported)), firing when it falls below half of baseline (~21%).
    """
    eff_threshold = (0.5 * float(baseline_ct_share)) if threshold is None else float(threshold)
    ct_recs = [r for r in records if str(r.get("gold_verdict")) == "cannot_tell"]
    ct_n = sum(int(r.get("cannot_tell", 0)) for r in ct_recs)
    unsup_n = sum(int(r.get("unsupported", 0)) for r in ct_recs)
    correct_total = ct_n + unsup_n
    share = (ct_n / float(correct_total)) if correct_total > 0 else 0.0
    return {
        "cannot_tell_count": ct_n,
        "unsupported_count": unsup_n,
        "correct_total": correct_total,
        "ct_share_of_correct": round(share, 6),
        "baseline_ct_share": round(float(baseline_ct_share), 6),
        "threshold": round(eff_threshold, 6),
        "triggered": bool(correct_total > 0 and share < eff_threshold),
    }


def _verify_pass_provenance(
    records: Sequence[Mapping[str, Any]],
    *,
    role: str,
    header: Mapping[str, Any] | None = None,
    expected_seed: int | None = None,
    forbid_seeds: Sequence[int] = (),
    require_seeds: bool = True,
) -> int | None:
    """Verify pass records and optional header are not descriptive and have the expected seed provenance."""
    if header is not None and (
        header.get("descriptive") is True
        or ("descriptive" in header and not isinstance(header["descriptive"], bool))
    ):
        raise SystemExit(
            f"refusing paired estimators (paired_estimator_descriptive): "
            f"{role} header has descriptive=true"
        )
    observed_seed: int | None = None
    if header is not None and "seed" in header and not isinstance(header["seed"], bool):
        observed_seed = int(header["seed"])
    for r in records:
        if r.get("descriptive") is True or (
            "descriptive" in r and not isinstance(r["descriptive"], bool)
        ):
            raise SystemExit(
                f"refusing paired estimators (paired_estimator_descriptive): "
                f"{role} record {r.get('id')!r} has descriptive=true"
            )
        rec_seed_raw = r.get("pass_seed", r.get("seed"))
        if rec_seed_raw is not None and not isinstance(rec_seed_raw, bool):
            rec_seed = int(rec_seed_raw)
            if observed_seed is None:
                observed_seed = rec_seed
            elif observed_seed != rec_seed:
                raise SystemExit(
                    f"refusing paired estimators (paired_estimator_wrong_seed): "
                    f"{role} has inconsistent seeds ({observed_seed} vs {rec_seed})"
                )
    if require_seeds and observed_seed is None:
        raise SystemExit(
            f"refusing paired estimators (paired_estimator_wrong_seed): "
            f"{role} is missing seed metadata"
        )
    if observed_seed is not None:
        if expected_seed is not None and observed_seed != int(expected_seed):
            raise SystemExit(
                f"refusing paired estimators (paired_estimator_wrong_seed): "
                f"{role} requires seed {int(expected_seed)}, got {observed_seed}"
            )
        if observed_seed in {int(s) for s in forbid_seeds}:
            raise SystemExit(
                f"refusing paired estimators (paired_estimator_wrong_seed): "
                f"{role} has reserved seed {observed_seed}"
            )
    return observed_seed


def compute_paired_arm_estimators(
    pass_a_records: Sequence[Mapping[str, Any]],
    pass_b_records: Sequence[Mapping[str, Any]],
    seed_postpasses: Sequence[tuple[Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]]]],
    *,
    holdout_ids: Sequence[str] | set[str] | None = None,
    baseline_ct_share: float = 0.42,
    pass_a_header: Mapping[str, Any] | None = None,
    pass_b_header: Mapping[str, Any] | None = None,
    postpass_headers: Sequence[tuple[Mapping[str, Any] | None, Mapping[str, Any] | None]] | None = None,
    require_seeds: bool = True,
) -> dict[str, Any]:
    """Compute paired item-level estimators across 2 post-passes per seed and multiple training seeds.

    Rules (Sections 11 & 13):
    - Per item, average the 2 post-passes first: ``post_mean = (post1 + post2) / 2``.
    - Then average ``post_mean`` across the training seeds paired by item (``n = items``).
    - Inside Pass-A bands (``[0,5]``, ``[6,7]``, ``8``): use ``post_mean - B``.
    - On unbanded sets (all items, held-out, trained, and by gold class): use ``post_mean - (A + B) / 2``.
    - Co-primary measures: ``k_correct`` and ``confident_errors`` (plus ``reward_mean`` as secondary).
    """
    if not seed_postpasses:
        raise ValueError("seed_postpasses must contain at least one (post1, post2) pair")
    _verify_pass_provenance(
        pass_a_records,
        role="Pass A",
        header=pass_a_header,
        expected_seed=0,
        require_seeds=require_seeds,
    )
    _verify_pass_provenance(
        pass_b_records,
        role="Pass B",
        header=pass_b_header,
        expected_seed=1,
        require_seeds=require_seeds,
    )
    for pair_idx, (post1, post2) in enumerate(seed_postpasses):
        hdr1, hdr2 = (
            postpass_headers[pair_idx]
            if postpass_headers is not None and pair_idx < len(postpass_headers)
            else (None, None)
        )
        s1 = _verify_pass_provenance(
            post1,
            role=f"seed_postpasses[{pair_idx}][0]",
            header=hdr1,
            forbid_seeds=(0, 1),
            require_seeds=require_seeds,
        )
        s2_seed = _verify_pass_provenance(
            post2,
            role=f"seed_postpasses[{pair_idx}][1]",
            header=hdr2,
            forbid_seeds=(0, 1),
            require_seeds=require_seeds,
        )
        if s1 is not None and s2_seed is not None and s1 == s2_seed:
            raise SystemExit(
                f"refusing paired estimators (paired_estimator_wrong_seed): "
                f"seed_postpasses[{pair_idx}] post1 and post2 have identical seed {s1}"
            )
    a_by_id = {str(r["id"]): r for r in pass_a_records}
    b_by_id = {str(r["id"]): r for r in pass_b_records}
    item_ids = sorted(a_by_id.keys())
    if set(b_by_id.keys()) != set(item_ids):
        raise ValueError("pass_a_records and pass_b_records must cover the exact same item IDs")

    holdout_set = {str(x) for x in holdout_ids} if holdout_ids is not None else set()

    # Per-seed, per-item post_mean = (post1 + post2) / 2
    per_item_k_seeds: dict[str, list[float]] = {cid: [] for cid in item_ids}
    per_item_ce_seeds: dict[str, list[float]] = {cid: [] for cid in item_ids}
    per_item_rew_seeds: dict[str, list[float]] = {cid: [] for cid in item_ids}
    flat_post_records: list[Mapping[str, Any]] = []

    for post1, post2 in seed_postpasses:
        p1_by_id = {str(r["id"]): r for r in post1}
        p2_by_id = {str(r["id"]): r for r in post2}
        flat_post_records.extend(post1)
        flat_post_records.extend(post2)
        for cid in item_ids:
            r1, r2 = p1_by_id[cid], p2_by_id[cid]
            per_item_k_seeds[cid].append(0.5 * (float(r1["k_correct"]) + float(r2["k_correct"])))
            per_item_ce_seeds[cid].append(
                0.5 * (float(item_confident_errors(r1)) + float(item_confident_errors(r2)))
            )
            per_item_rew_seeds[cid].append(
                0.5 * (float(r1.get("reward_mean", 0.0)) + float(r2.get("reward_mean", 0.0)))
            )

    post_mean_k = {cid: float(statistics.fmean(per_item_k_seeds[cid])) for cid in item_ids}
    post_mean_ce = {cid: float(statistics.fmean(per_item_ce_seeds[cid])) for cid in item_ids}
    post_mean_rew = {cid: float(statistics.fmean(per_item_rew_seeds[cid])) for cid in item_ids}

    def _subset_banded(ids_sub: Sequence[str]) -> dict[str, Any]:
        """Inside Pass-A bands: estimator is post_mean - B."""
        if not ids_sub:
            return {"n_items": 0, "k_diff": None, "confident_error_diff": None, "reward_diff": None}
        k_diffs = [post_mean_k[cid] - float(b_by_id[cid]["k_correct"]) for cid in ids_sub]
        ce_diffs = [post_mean_ce[cid] - float(item_confident_errors(b_by_id[cid])) for cid in ids_sub]
        rew_diffs = [post_mean_rew[cid] - float(b_by_id[cid].get("reward_mean", 0.0)) for cid in ids_sub]
        return {
            "n_items": len(ids_sub),
            "estimator": "post_mean - B",
            "k_post_mean": round(float(statistics.fmean(post_mean_k[cid] for cid in ids_sub)), 6),
            "k_diff": round(float(statistics.fmean(k_diffs)), 6),
            "confident_error_post_mean": round(
                float(statistics.fmean(post_mean_ce[cid] for cid in ids_sub)), 6
            ),
            "confident_error_diff": round(float(statistics.fmean(ce_diffs)), 6),
            "reward_diff": round(float(statistics.fmean(rew_diffs)), 6),
        }

    def _subset_unbanded(ids_sub: Sequence[str]) -> dict[str, Any]:
        """On unbanded sets: estimator is post_mean - (A + B) / 2."""
        if not ids_sub:
            return {"n_items": 0, "k_diff": None, "confident_error_diff": None, "reward_diff": None}
        k_diffs = [
            post_mean_k[cid] - 0.5 * (float(a_by_id[cid]["k_correct"]) + float(b_by_id[cid]["k_correct"]))
            for cid in ids_sub
        ]
        ce_diffs = [
            post_mean_ce[cid]
            - 0.5 * (float(item_confident_errors(a_by_id[cid])) + float(item_confident_errors(b_by_id[cid])))
            for cid in ids_sub
        ]
        rew_diffs = [
            post_mean_rew[cid]
            - 0.5
            * (
                float(a_by_id[cid].get("reward_mean", 0.0))
                + float(b_by_id[cid].get("reward_mean", 0.0))
            )
            for cid in ids_sub
        ]
        return {
            "n_items": len(ids_sub),
            "estimator": "post_mean - (A + B) / 2",
            "k_post_mean": round(float(statistics.fmean(post_mean_k[cid] for cid in ids_sub)), 6),
            "k_diff": round(float(statistics.fmean(k_diffs)), 6),
            "confident_error_post_mean": round(
                float(statistics.fmean(post_mean_ce[cid] for cid in ids_sub)), 6
            ),
            "confident_error_diff": round(float(statistics.fmean(ce_diffs)), 6),
            "reward_diff": round(float(statistics.fmean(rew_diffs)), 6),
        }

    by_band: dict[str, Any] = {}
    for band in ("[0,5]", "[6,7]", "8"):
        band_ids = [cid for cid in item_ids if k_band_name(int(a_by_id[cid]["k_correct"])) == band]
        by_band[band] = _subset_banded(band_ids)

    by_gold_class: dict[str, Any] = {}
    for gv in VERDICTS:
        gv_ids = [cid for cid in item_ids if str(a_by_id[cid].get("gold_verdict")) == gv]
        by_gold_class[gv] = _subset_unbanded(gv_ids)

    by_band_and_gold_class: dict[str, dict[str, Any]] = {}
    for band in ("[0,5]", "[6,7]", "8"):
        by_band_and_gold_class[band] = {}
        for gv in VERDICTS:
            cell_ids = [
                cid
                for cid in item_ids
                if k_band_name(int(a_by_id[cid]["k_correct"])) == band
                and str(a_by_id[cid].get("gold_verdict")) == gv
            ]
            by_band_and_gold_class[band][gv] = _subset_banded(cell_ids)

    hold_list = [cid for cid in item_ids if cid in holdout_set]
    train_list = [cid for cid in item_ids if cid not in holdout_set]

    return {
        "n_items": len(item_ids),
        "n_seeds": len(seed_postpasses),
        "unbanded": {
            "all_items": _subset_unbanded(item_ids),
            "holdout": _subset_unbanded(hold_list),
            "trained": _subset_unbanded(train_list),
        },
        "by_pass_a_band": by_band,
        "by_gold_class": by_gold_class,
        "by_band_and_gold_class": by_band_and_gold_class,
        "confusion_matrix_before": {
            "pass_a": build_confusion_matrix_3x3(pass_a_records),
            "pass_b": build_confusion_matrix_3x3(pass_b_records),
        },
        "confusion_matrix_after": build_confusion_matrix_3x3(flat_post_records),
        "hedge_flag_after": compute_hedge_flag(flat_post_records, baseline_ct_share=baseline_ct_share),
        "per_item_post_mean_k": {cid: round(post_mean_k[cid], 6) for cid in item_ids},
        "per_item_post_mean_confident_errors": {cid: round(post_mean_ce[cid], 6) for cid in item_ids},
    }


def _load_verified_pass_entry(entry: Mapping[str, Any], role: str) -> tuple[list[dict], dict | None]:
    """Load a pass JSONL + companion header from a paired-estimator spec entry, verifying sha256."""
    jsonl_val = entry.get("jsonl") or entry.get("path")
    sha_val = entry.get("sha256")
    if not jsonl_val or not sha_val:
        raise SystemExit(
            f"refusing paired estimators (paired_estimator_missing_arg): "
            f"{role} requires both 'jsonl' and 'sha256'"
        )
    jpath = Path(str(jsonl_val))
    if not jpath.is_file():
        raise SystemExit(
            f"refusing paired estimators (paired_estimator_missing_file): "
            f"{role} file does not exist: {jpath.name}"
        )
    raw_bytes = jpath.read_bytes()
    actual_sha = hashlib.sha256(raw_bytes).hexdigest()
    expected_sha = str(sha_val).strip().lower()
    if actual_sha != expected_sha:
        raise SystemExit(
            f"refusing paired estimators (paired_estimator_sha_mismatch): "
            f"{role} expected sha256 {expected_sha}, got {actual_sha}"
        )
    recs = [json.loads(line) for line in raw_bytes.decode("utf-8").splitlines() if line.strip()]
    hdr_val = entry.get("header")
    hdr_obj: dict | None = None
    candidates = (
        [Path(str(hdr_val))]
        if hdr_val
        else [
            jpath.with_name(jpath.stem + "-header.json"),
            jpath.parent / "s2-hf-prepass-header.json",
            jpath.parent / "s2-llamacpp-prepass-header.json",
        ]
    )
    for cpath in candidates:
        if cpath.is_file():
            loaded = json.loads(cpath.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                hdr_obj = loaded
                break
    return recs, hdr_obj


def run_paired_estimators_from_spec(spec_path: Path, out_dir: Path) -> dict[str, Any]:
    """Read Pass A, Pass B, and seed post-pass pairs from ``spec_path`` and write estimator report."""
    spath = Path(spec_path)
    if not spath.is_file():
        raise SystemExit(
            f"refusing paired estimators (paired_estimator_missing_spec): "
            f"spec file does not exist: {spath.name}"
        )
    spec = json.loads(spath.read_text(encoding="utf-8"))
    if not isinstance(spec, dict):
        raise SystemExit(
            "refusing paired estimators (paired_estimator_invalid_spec): spec must be a JSON object"
        )
    if "pass_a" not in spec or "pass_b" not in spec or "seed_postpasses" not in spec:
        raise SystemExit(
            "refusing paired estimators (paired_estimator_invalid_spec): "
            "spec requires 'pass_a', 'pass_b', and 'seed_postpasses'"
        )
    pa_recs, pa_hdr = _load_verified_pass_entry(spec["pass_a"], "pass_a")
    pb_recs, pb_hdr = _load_verified_pass_entry(spec["pass_b"], "pass_b")
    post_pairs: list[tuple[list[dict], list[dict]]] = []
    post_hdrs: list[tuple[dict | None, dict | None]] = []
    for idx, pair in enumerate(spec["seed_postpasses"]):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise SystemExit(
                f"refusing paired estimators (paired_estimator_invalid_spec): "
                f"seed_postpasses[{idx}] must be a 2-element list"
            )
        p1_recs, p1_hdr = _load_verified_pass_entry(pair[0], f"seed_postpasses[{idx}][0]")
        p2_recs, p2_hdr = _load_verified_pass_entry(pair[1], f"seed_postpasses[{idx}][1]")
        post_pairs.append((p1_recs, p2_recs))
        post_hdrs.append((p1_hdr, p2_hdr))
    doc = compute_paired_arm_estimators(
        pa_recs,
        pb_recs,
        post_pairs,
        holdout_ids=spec.get("holdout_ids"),
        baseline_ct_share=float(spec.get("baseline_ct_share", 0.42)),
        pass_a_header=pa_hdr,
        pass_b_header=pb_hdr,
        postpass_headers=post_hdrs,
        require_seeds=True,
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "s2-paired-estimators.json").write_text(
        json.dumps(doc, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    return doc


def prepass_summary(
    records: Sequence[dict],
    *,
    g: int,
    quote_reward: str = "v1",
    reproducible: bool = True,
    descriptive: bool = False,
    pass_a_k: dict[str, int] | None = None,
    checkpoint_sha: str | None = None,
    checkpoint_files_sha: Mapping[str, str] | None = None,
) -> dict:
    """Summary of a rollout-only pre-pass: counts, histograms (0..G), per-band truncations, and ratios."""
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
    truncs_by_band: dict[str, int] = {"[0,5]": 0, "[6,7]": 0, "8": 0}
    ce_by_band: dict[str, int] = {"[0,5]": 0, "[6,7]": 0, "8": 0}
    ce_by_verdict: dict[str, int] = {gv: 0 for gv in VERDICTS}
    total_ce = 0
    total_completions = sum(int(r.get("g", g)) for r in records)
    for r in records:
        rid = str(r["id"])
        k_ref = (
            int(pass_a_k[rid])
            if (pass_a_k is not None and rid in pass_a_k)
            else int(r.get("pass_a_k", r["k_correct"]))
        )
        band_key = k_band_name(k_ref)
        truncs_by_band[band_key] += int(r["truncations"])
        ce_val = item_confident_errors(r)
        total_ce += ce_val
        ce_by_band[band_key] += ce_val
        gv_key = str(r.get("gold_verdict", "supported"))
        if gv_key in ce_by_verdict:
            ce_by_verdict[gv_key] += ce_val
    total_quote_pass = sum(int(r.get("quote_pass", 0)) for r in records)
    gold_sup_completions = sum(int(r["g"]) for r in records if r.get("gold_verdict") == "supported")
    ct_on_gold_sup = sum(int(r["cannot_tell"]) for r in records if r.get("gold_verdict") == "supported")
    p_ct_given_sup = (ct_on_gold_sup / gold_sup_completions) if gold_sup_completions > 0 else 0.0
    out: dict = {
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
        "truncations_by_band": truncs_by_band,
        "confident_errors": total_ce,
        "confident_error_rate": (total_ce / total_completions) if total_completions > 0 else 0.0,
        "confident_errors_by_gold_verdict": ce_by_verdict,
        "confident_errors_by_band": ce_by_band,
        "confusion_matrix_3x3": build_confusion_matrix_3x3(records),
        "hedge_flag": compute_hedge_flag(records),
        "quote_pass": total_quote_pass,
        "quote_pass_role": quote_pass_role_text(quote_reward),
        "p_cannot_tell_given_gold_supported": p_ct_given_sup,
        "reproducible": bool(reproducible),
    }
    if descriptive:
        out["descriptive"] = True
    if checkpoint_sha is not None:
        out["checkpoint_sha256"] = checkpoint_sha
    if checkpoint_files_sha is not None:
        out["checkpoint_files_sha256"] = dict(checkpoint_files_sha)
    return out


def _prepare_prepass_resume(
    header_path: Path,
    jsonl_path: Path,
    expected_header: dict,
    *,
    path: str = "hf",
    summary_path: Path | None = None,
    replay_check_k: Mapping[str, int] | None = None,
    fsync: Callable[[int], None] = os.fsync,
) -> list[dict]:
    """Verify or write the prepass header, truncate any trailing partial line, and return done records."""
    if header_path.exists():
        if path == "llamacpp" or expected_header.get("reproducible") is False:
            raise SystemExit(
                "refusing to resume prepass: --path llamacpp samples are not seeded per item, "
                "so a resumed run is not reproducible"
            )
        try:
            existing_header = json.loads(header_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise SystemExit(
                f"refusing to resume prepass: header {header_path} is not valid JSON ({exc})"
            ) from exc
        if not isinstance(existing_header, dict):
            raise SystemExit(f"refusing to resume prepass: header {header_path} is not a JSON object")
        eff_summary_path = (
            Path(summary_path)
            if summary_path is not None
            else header_path.parent / f"s2-{path}-prepass-summary.json"
        )
        if eff_summary_path.exists():
            try:
                existing_summary = json.loads(eff_summary_path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise SystemExit(
                    f"refusing to resume prepass: summary {eff_summary_path} is not valid JSON ({exc})"
                ) from exc
            stopped_rec = existing_summary.get("stopped") if isinstance(existing_summary, dict) else None
            if isinstance(stopped_rec, dict) and stopped_rec.get("reason") == "LOGGING_PASS_REPLAY_MISMATCH":
                raise SystemExit(
                    f"refusing to resume prepass (LOGGING_PASS_REPLAY_MISMATCH): "
                    f"summary {eff_summary_path} has stopped.reason=LOGGING_PASS_REPLAY_MISMATCH "
                    f"({stopped_rec.get('error') or 'replay mismatch'})"
                )
        existing_cmp = dict(existing_header)
        if "reproducible" not in existing_cmp and expected_header.get("reproducible") is True:
            existing_cmp["reproducible"] = True
        if "descriptive" not in existing_cmp and expected_header.get("descriptive") is False:
            existing_cmp["descriptive"] = False
        for k_opt in ("stack_versions", "pip_freeze_sha256"):
            if k_opt not in existing_cmp and k_opt in expected_header:
                existing_cmp[k_opt] = expected_header[k_opt]
        mismatched = [
            k
            for k in sorted(set(expected_header) | set(existing_cmp))
            if k != "stop_after_new" and existing_cmp.get(k) != expected_header.get(k)
        ]
        if mismatched:
            raise SystemExit(f"refusing to resume prepass: header mismatch on {', '.join(mismatched)}")
        if existing_header.get("stop_after_new") != expected_header.get("stop_after_new"):
            updated_header = dict(existing_header)
            if "stop_after_new" in expected_header:
                updated_header["stop_after_new"] = expected_header["stop_after_new"]
            else:
                updated_header.pop("stop_after_new", None)
            with header_path.open("w", encoding="utf-8", newline="\n") as hfh:
                hfh.write(json.dumps(updated_header, indent=1) + "\n")
                hfh.flush()
                fsync(hfh.fileno())
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
        _append_dropped_entries(
            jsonl_path,
            [
                {
                    "reason": "torn_member",
                    "byte_offset": valid_bytes,
                    "raw_bytes_b64": base64.b64encode(raw[valid_bytes:]).decode("ascii"),
                }
            ],
            fsync=fsync,
        )
        with jsonl_path.open("r+b") as tfh:
            tfh.truncate(valid_bytes)
            tfh.flush()
            fsync(tfh.fileno())
    if replay_check_k is not None:
        for rec in completed:
            rec_id = str(rec["id"])
            if rec_id not in replay_check_k:
                raise SystemExit(
                    f"refusing to resume prepass (replay_check_missing_id): "
                    f"completed record {rec_id!r} has no replay-check entry"
                )
            b2_k = int(replay_check_k[rec_id])
            cur_k = int(rec["k_correct"])
            if cur_k != b2_k:
                raise SystemExit(
                    f"refusing to resume prepass (LOGGING_PASS_REPLAY_MISMATCH): "
                    f"completed record item_id={rec_id} k_correct={cur_k} b2_k_correct={b2_k}"
                )
    return completed


def run_prepass(
    *,
    a: argparse.Namespace,
    prompts: list[dict],
    gold_hashes: dict[str, str],
    bundle: dict,
    phase_logger: PhaseLogger | None = None,
    eval_records: list[dict] | None = None,
    base_seed: int | None = None,
    stop_after_new: int | None = None,
    descriptive: bool = False,
    checkpoint_sha: str | None = None,
    checkpoint_files_sha: Mapping[str, str] | None = None,
    pass_a_k: dict[str, int] | None = None,
    pass_a_sha: str | None = None,
    replay_check_k: Mapping[str, int] | None = None,
    replay_check_sha: str | None = None,
    replay_check_off_reason: str | None = None,
    stack_versions: Mapping[str, str | None] | None = None,
    pip_freeze_sha256: str | None = None,
    fsync: Callable[[int], None] = os.fsync,
) -> tuple[list[dict], list[dict], dict | None, bool]:
    """Run the rollout-only pre-pass over ``prompts`` with resume support.

    Returns (timing rows, completed prepass item records, stopped record or None, oom bool).
    Never calls optimiser, reference forward, training forward, or backward.
    """
    torch = require_torch()
    eff_seed = int(base_seed if base_seed is not None else getattr(a, "prepass_seed", PREPASS_BASE_SEED))
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
    path_name = str(getattr(a, "path", "hf") or "hf")
    reproducible = path_name != "llamacpp"
    save_all_on = bool(getattr(a, "save_all", True))
    default_pass_id = (
        f"logging_seed{eff_seed}"
        if descriptive
        else f"postpass_seed{eff_seed}"
        if (bool(getattr(a, "postpass", False)) or checkpoint_sha is not None)
        else f"prepass_seed{eff_seed}"
    )
    pass_id_str = str(getattr(a, "pass_id", None) or default_pass_id)
    eff_rc_sha = replay_check_sha
    eff_rc_off = replay_check_off_reason
    if eff_rc_sha is None and getattr(a, "replay_check_sha256", None):
        eff_rc_sha = str(a.replay_check_sha256).strip().lower()
    if eff_rc_off is None and getattr(a, "replay_check_off_reason", None):
        eff_rc_off = str(a.replay_check_off_reason).strip() or None
    eff_stop_after_new = (
        stop_after_new if stop_after_new is not None else getattr(a, "stop_after_new", None)
    )
    if eff_stop_after_new is not None:
        if (
            isinstance(eff_stop_after_new, bool)
            or not isinstance(eff_stop_after_new, int)
            or eff_stop_after_new < 1
        ):
            raise SystemExit(
                f"refusing to start (invalid_stop_after_new): "
                f"--stop-after-new must be >= 1, got {eff_stop_after_new!r}"
            )
    hdr_stop_after_new = getattr(a, "stop_after_new", None)
    if hdr_stop_after_new is not None:
        hdr_stop_after_new = int(hdr_stop_after_new)
    expected_header = build_prepass_header(
        prompt_ids=[r["id"] for r in prompts],
        gold_hashes=gold_hashes,
        g=a.g,
        max_new=a.max_new,
        quote_reward=quote_rew,
        seed=eff_seed,
        model_revision=model_rev,
        reproducible=reproducible,
        save_all=save_all_on,
        descriptive=descriptive,
        checkpoint_sha=checkpoint_sha,
        checkpoint_files_sha=checkpoint_files_sha,
        pass_a_sha=pass_a_sha,
        stack_versions=stack_versions,
        pip_freeze_sha256=pip_freeze_sha256,
        replay_check_sha256=eff_rc_sha,
        replay_check_off_reason=eff_rc_off,
        stop_after_new=hdr_stop_after_new,
    )
    a.out.mkdir(parents=True, exist_ok=True)
    header_path = a.out / f"s2-{a.path}-prepass-header.json"
    jsonl_path = a.out / f"s2-{a.path}-prepass.jsonl"
    summary_path = a.out / f"s2-{a.path}-prepass-summary.json"
    save_all_path = save_all_file_path(a.out, path_name)

    completed_records = _prepare_prepass_resume(
        header_path,
        jsonl_path,
        expected_header,
        path=path_name,
        summary_path=summary_path,
        replay_check_k=replay_check_k,
        fsync=fsync,
    )
    done_ids = {str(r["id"]) for r in completed_records}
    pass_save_records: list[dict] = []
    if save_all_on:
        pass_save_records = _prepare_save_all_resume(
            save_all_path,
            valid_record=lambda r: str(r.get("item_id")) in done_ids,
            record_key=lambda r: (str(r.get("item_id")), int(r.get("sample_index", -1))),
            fsync=fsync,
        )
        if not prompts and not save_all_path.exists():
            append_save_all(save_all_path, (), fsync=fsync)

    if mode:
        mode(False)
    if before_step:
        before_step(0)

    eos_ids = _eos_token_ids(tok)
    oom_types = tuple(t for t in (getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None),) if t)
    step_rows: list[dict] = []
    new_count = 0
    stopped_mismatch: dict | None = None

    with jsonl_path.open("a", encoding="utf-8", newline="\n") as jfh:
        for item_idx, row in enumerate(prompts):
            item_id = str(row["id"])
            if item_id in done_ids:
                continue
            if eff_stop_after_new is not None and new_count >= eff_stop_after_new:
                break
            try:
                started = time.time()
                seed_for_item = item_seed(item_id, eff_seed)
                torch.manual_seed(seed_for_item)
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
                item_save_records: list[dict] = []
                rec, rollout_outcomes = prepass_item_record(
                    row=row,
                    prompt_ids=prompt_ids,
                    comps=comps,
                    truncs=truncs,
                    tok=tok,
                    reward_fn=reward_fn,
                    quote_reward=quote_rew,
                    save_all_out=item_save_records if save_all_on else None,
                    pass_id=pass_id_str,
                    item_seed_val=seed_for_item,
                    pass_seed=eff_seed,
                    descriptive=descriptive,
                    pass_a_k=pass_a_k,
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
                if save_all_on:
                    append_save_all(save_all_path, item_save_records, fsync=fsync)
                    pass_save_records.extend(item_save_records)
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
                if replay_check_k is not None:
                    b2_k = int(replay_check_k[item_id])
                    cur_k = int(rec["k_correct"])
                    if cur_k != b2_k:
                        mismatch_msg = (
                            f"LOGGING_PASS_REPLAY_MISMATCH: item_id={item_id} "
                            f"k_correct={cur_k} b2_k_correct={b2_k}"
                        )
                        stopped_mismatch = {
                            "step": item_idx,
                            "reason": "LOGGING_PASS_REPLAY_MISMATCH",
                            "error": mismatch_msg,
                            "item_id": item_id,
                            "k_correct": cur_k,
                            "b2_k_correct": b2_k,
                        }
                        break
            except oom_types:
                return step_rows, completed_records, {"step": item_idx, "error": "out of memory"}, True
            except TimeoutError as exc:
                return (
                    step_rows,
                    completed_records,
                    {"step": item_idx, "error": f"timeout: {exc}"},
                    False,
                )

    if (
        len(completed_records) == len(prompts)
        or stopped_mismatch is not None
        or eff_stop_after_new is not None
    ):
        summary_doc = prepass_summary(
            completed_records,
            g=a.g,
            quote_reward=quote_rew,
            reproducible=reproducible,
            descriptive=descriptive,
            pass_a_k=pass_a_k,
            checkpoint_sha=checkpoint_sha,
            checkpoint_files_sha=checkpoint_files_sha,
        )
        if stopped_mismatch is not None:
            summary_doc["stopped"] = dict(stopped_mismatch)
        with summary_path.open("w", encoding="utf-8", newline="\n") as sfh:
            sfh.write(json.dumps(summary_doc, indent=1) + "\n")
            sfh.flush()
            fsync(sfh.fileno())
    return step_rows, completed_records, stopped_mismatch, False


def _check_header_not_descriptive(
    baseline_path: Path,
    *,
    expected_seeds: Sequence[int] = (0,),
) -> tuple[bool, bool]:
    """Refuse if any companion header beside ``baseline_path`` has ``descriptive: true`` or wrong seed."""
    allowed_seeds = {int(s) for s in expected_seeds}
    candidates = [
        baseline_path.with_name(baseline_path.stem + "-header.json"),
        baseline_path.parent / "s2-hf-prepass-header.json",
        baseline_path.parent / "s2-llamacpp-prepass-header.json",
    ]
    seen: set[Path] = set()
    header_has_seed = False
    header_has_descriptive = False
    for hdr_path in candidates:
        if hdr_path in seen or not hdr_path.is_file():
            continue
        seen.add(hdr_path)
        try:
            hdr_obj = json.loads(hdr_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(hdr_obj, dict):
            continue
        if hdr_obj.get("descriptive") is True or (
            "descriptive" in hdr_obj and not isinstance(hdr_obj["descriptive"], bool)
        ):
            raise SystemExit(
                "refusing to start (prepass_baseline_descriptive): "
                "pass header has descriptive=true; the starting-policy logging pass "
                "is never accepted as --prepass-baseline"
            )
        if "descriptive" in hdr_obj and hdr_obj["descriptive"] is False:
            header_has_descriptive = True
        if "seed" in hdr_obj:
            hdr_seed = hdr_obj.get("seed")
            if isinstance(hdr_seed, bool) or not isinstance(hdr_seed, int) or hdr_seed not in allowed_seeds:
                raise SystemExit(
                    f"refusing to start (prepass_baseline_wrong_seed): "
                    f"pass header has seed={hdr_seed!r}, expected one of {sorted(allowed_seeds)}"
                )
            header_has_seed = True
            if "ids_sha256" in hdr_obj and "descriptive" not in hdr_obj:
                header_has_descriptive = True
    return header_has_seed, header_has_descriptive


# -- the run -------------------------------------------------------------------------------------------------
def load_prepass_baseline(
    baseline_path: Path | None,
    baseline_sha256: str | None,
    required_ids: Sequence[str],
    *,
    g: int | None = G,
    expected_seeds: Sequence[int] = (0,),
) -> tuple[dict[str, int] | None, str | None]:
    """Load and SHA-check a pre-pass baseline JSONL. Never records item text or full file paths."""
    if baseline_path is None and baseline_sha256 is None:
        return None, None
    if baseline_path is None or not baseline_sha256:
        raise SystemExit(
            "refusing to start (prepass_baseline_missing_arg): "
            "--prepass-baseline and --prepass-baseline-sha256 must be given together"
        )
    path = Path(baseline_path)
    if not path.is_file():
        raise SystemExit(
            f"refusing to start (prepass_baseline_missing_file): baseline file does not exist: {path.name}"
        )
    allowed_seeds = {int(s) for s in expected_seeds}
    header_has_seed, header_has_descriptive = _check_header_not_descriptive(
        path, expected_seeds=expected_seeds
    )
    raw_bytes = path.read_bytes()
    actual_sha = hashlib.sha256(raw_bytes).hexdigest()
    expected_sha = str(baseline_sha256).strip().lower()
    if actual_sha != expected_sha:
        raise SystemExit(
            f"refusing to start (prepass_baseline_sha_mismatch): "
            f"expected sha256 {expected_sha}, got {actual_sha}"
        )
    expected_g = int(g) if g is not None else None
    k_by_id: dict[str, int] = {}
    for line_no, raw_line in enumerate(raw_bytes.decode("utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(
                f"refusing to start (prepass_baseline_corrupt_json): invalid JSON on line {line_no}"
            ) from exc
        rec_has_descriptive = False
        if isinstance(obj, dict) and "descriptive" in obj:
            if obj.get("descriptive") is True or not isinstance(obj.get("descriptive"), bool):
                raise SystemExit(
                    "refusing to start (prepass_baseline_descriptive): "
                    "record has descriptive=true; the starting-policy logging pass "
                    "is never accepted as --prepass-baseline"
                )
            if obj.get("descriptive") is False:
                rec_has_descriptive = True
        rec_has_seed = False
        if isinstance(obj, dict) and ("pass_seed" in obj or "seed" in obj):
            rec_seed = obj.get("pass_seed", obj.get("seed"))
            if isinstance(rec_seed, bool) or not isinstance(rec_seed, int) or rec_seed not in allowed_seeds:
                raise SystemExit(
                    f"refusing to start (prepass_baseline_wrong_seed): "
                    f"record on line {line_no} has seed={rec_seed!r}, expected one of {sorted(allowed_seeds)}"
                )
            rec_has_seed = True
        if not isinstance(obj, dict) or "id" not in obj or "k_correct" not in obj:
            raise SystemExit(
                f"refusing to start (prepass_baseline_invalid_record): "
                f"line {line_no} is missing 'id' or 'k_correct'"
            )
        rec_id = str(obj["id"])
        if rec_id in k_by_id:
            raise SystemExit(
                f"refusing to start (prepass_baseline_duplicate_id): "
                f"duplicate baseline record for id {rec_id!r}"
            )
        g_val = obj.get("g")
        if isinstance(g_val, bool) or not isinstance(g_val, int) or g_val < 1:
            raise SystemExit(
                f"refusing to start (prepass_baseline_g_mismatch): "
                f"baseline record for id {rec_id!r} has g={g_val!r}, expected {expected_g}"
            )
        if expected_g is None:
            expected_g = g_val
        elif g_val != expected_g:
            raise SystemExit(
                f"refusing to start (prepass_baseline_g_mismatch): "
                f"baseline record for id {rec_id!r} has g={g_val!r}, expected {expected_g}"
            )
        k_val = obj["k_correct"]
        if isinstance(k_val, bool) or not isinstance(k_val, int) or k_val < 0 or k_val > expected_g:
            raise SystemExit(
                f"refusing to start (prepass_baseline_k_out_of_range): "
                f"baseline record for id {rec_id!r} has k_correct={k_val!r} outside 0..{expected_g}"
            )
        if not (header_has_seed or rec_has_seed):
            raise SystemExit(
                f"refusing to start (prepass_baseline_missing_seed): "
                f"record on line {line_no} and companion header lack seed metadata"
            )
        if not (header_has_descriptive or rec_has_descriptive):
            raise SystemExit(
                f"refusing to start (prepass_baseline_missing_descriptive): "
                f"record on line {line_no} and companion header lack descriptive metadata"
            )
        k_by_id[rec_id] = k_val
    missing = [str(cid) for cid in required_ids if str(cid) not in k_by_id]
    if missing:
        raise SystemExit(
            f"refusing to start (prepass_baseline_missing_id): prompt {missing[0]!r} has no baseline record "
            f"({len(missing)} missing)"
        )
    return k_by_id, actual_sha


def load_replay_check_jsonl(
    replay_path: Path | None,
    replay_sha256: str | None,
    required_ids: Sequence[str],
    *,
    g: int | None = G,
) -> tuple[dict[str, int] | None, str | None]:
    """Load and SHA-check a B2 JSONL file for ``--replay-check`` during ``--logging-pass``."""
    if replay_path is None and replay_sha256 is None:
        return None, None
    if replay_path is None or not replay_sha256:
        raise SystemExit(
            "refusing to start (replay_check_missing_arg): "
            "--replay-check and --replay-check-sha256 must be given together"
        )
    path = Path(replay_path)
    if not path.is_file():
        raise SystemExit(
            f"refusing to start (replay_check_missing_file): replay-check file does not exist: {path.name}"
        )
    _check_header_not_descriptive(path, expected_seeds=(1,))
    raw_bytes = path.read_bytes()
    actual_sha = hashlib.sha256(raw_bytes).hexdigest()
    expected_sha = str(replay_sha256).strip().lower()
    if actual_sha != expected_sha:
        raise SystemExit(
            f"refusing to start (replay_check_sha_mismatch): "
            f"expected sha256 {expected_sha}, got {actual_sha}"
        )
    expected_g = int(g) if g is not None else None
    k_by_id: dict[str, int] = {}
    for line_no, raw_line in enumerate(raw_bytes.decode("utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(
                f"refusing to start (replay_check_corrupt_json): invalid JSON on line {line_no}"
            ) from exc
        if not isinstance(obj, dict) or "id" not in obj or "k_correct" not in obj:
            raise SystemExit(
                f"refusing to start (replay_check_invalid_record): "
                f"line {line_no} is missing 'id' or 'k_correct'"
            )
        if "descriptive" in obj and (
            obj.get("descriptive") is True or not isinstance(obj.get("descriptive"), bool)
        ):
            raise SystemExit(
                "refusing to start (replay_check_descriptive): "
                f"line {line_no} has descriptive=true; --replay-check requires non-descriptive Pass B2"
            )
        if "pass_seed" in obj or "seed" in obj:
            rec_seed = obj.get("pass_seed", obj.get("seed"))
            if isinstance(rec_seed, bool) or not isinstance(rec_seed, int) or rec_seed != 1:
                raise SystemExit(
                    f"refusing to start (replay_check_wrong_seed): "
                    f"record on line {line_no} has seed={rec_seed!r}, expected 1"
                )
        rec_id = str(obj["id"])
        if rec_id in k_by_id:
            raise SystemExit(
                f"refusing to start (replay_check_duplicate_id): duplicate record for id {rec_id!r}"
            )
        k_val = obj["k_correct"]
        if isinstance(k_val, bool) or not isinstance(k_val, int) or k_val < 0:
            raise SystemExit(
                f"refusing to start (replay_check_k_out_of_range): "
                f"record for id {rec_id!r} has invalid k_correct={k_val!r}"
            )
        if expected_g is not None and k_val > expected_g:
            raise SystemExit(
                f"refusing to start (replay_check_k_out_of_range): "
                f"record for id {rec_id!r} has k_correct={k_val!r} > g={expected_g}"
            )
        k_by_id[rec_id] = k_val
    missing = [str(cid) for cid in required_ids if str(cid) not in k_by_id]
    if missing:
        raise SystemExit(
            f"refusing to start (replay_check_missing_id): prompt {missing[0]!r} has no replay-check record "
            f"({len(missing)} missing)"
        )
    return k_by_id, actual_sha


def mean_residual_by_k_band(
    entries: Sequence[tuple[str, int, float]],
) -> dict[str, object]:
    """Mean of ``residual = (k_online - k_prepass) / g`` by ``k_prepass`` band, one residual per prompt id."""
    by_prompt: dict[str, tuple[int, list[float]]] = {}
    for pid_raw, k_pre_raw, res_raw in entries:
        pid, k_pre, res = str(pid_raw), int(k_pre_raw), float(res_raw)
        if pid not in by_prompt:
            by_prompt[pid] = (k_pre, [res])
        else:
            by_prompt[pid][1].append(res)
    bands: dict[str, list[float]] = {"[0,5]": [], "[6,7]": [], "8": []}
    for k_pre, res_list in by_prompt.values():
        prompt_mean_res = float(statistics.fmean(res_list))
        if 0 <= k_pre <= 5:
            bands["[0,5]"].append(prompt_mean_res)
        elif 6 <= k_pre <= 7:
            bands["[6,7]"].append(prompt_mean_res)
        elif k_pre == 8:
            bands["8"].append(prompt_mean_res)
    out: dict[str, object] = {
        band: (round(float(statistics.fmean(vals)), 6) if vals else None) for band, vals in bands.items()
    }
    out["n_prompts"] = {band: len(vals) for band, vals in bands.items()}
    out["prompt_visits_max"] = max((len(v[1]) for v in by_prompt.values()), default=0)
    return out


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
        flags["stress_pool"] = str(getattr(a, "stress_pool", "selected") or "selected")
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
    if any(getattr(a, attr, False) for attr in ("rollout_only", "logging_pass", "postpass")):
        flags["rollout_only"] = True
    if getattr(a, "logging_pass", False):
        flags["logging_pass"] = True
        flags["descriptive"] = True
        flags["prepass_seed"] = 1
    elif getattr(a, "postpass", False):
        flags["postpass"] = True
        if getattr(a, "pass_seed", None) is not None:
            flags["prepass_seed"] = int(a.pass_seed)
        elif int(getattr(a, "prepass_seed", PREPASS_BASE_SEED) or 0) != PREPASS_BASE_SEED:
            flags["prepass_seed"] = int(a.prepass_seed)
    elif int(getattr(a, "prepass_seed", PREPASS_BASE_SEED) or 0) != PREPASS_BASE_SEED:
        flags["prepass_seed"] = int(a.prepass_seed)
    elif getattr(a, "pass_seed", None) != PREPASS_BASE_SEED and getattr(a, "pass_seed", None) is not None:
        flags["prepass_seed"] = int(a.pass_seed)
    if getattr(a, "sign_diagnostics", False):
        flags["sign_diagnostics"] = True
    if getattr(a, "prepass_baseline_sha256", None):
        flags["prepass_baseline_sha256"] = str(a.prepass_baseline_sha256).strip().lower()
    if getattr(a, "pass_a_sha256", None):
        flags["pass_a_sha256"] = str(a.pass_a_sha256).strip().lower()
    if getattr(a, "replay_check_sha256", None) and str(getattr(a, "replay_check_sha256")).strip():
        flags["replay_check_sha256"] = str(a.replay_check_sha256).strip().lower()
    if getattr(a, "replay_check_off_reason", None) and str(getattr(a, "replay_check_off_reason")).strip():
        flags["replay_check_off_reason"] = str(a.replay_check_off_reason).strip()
    if getattr(a, "log_sdpa_backend", False):
        flags["log_sdpa_backend"] = True
    if not bool(getattr(a, "save_all", True)):
        flags["save_all"] = False
        flags["save_all_off_reason"] = (getattr(a, "save_all_off_reason", None) or "").strip()
    if getattr(a, "pass_id", None):
        flags["pass_id"] = str(a.pass_id)
    if getattr(a, "stop_after_new", None) is not None:
        flags["stop_after_new"] = int(a.stop_after_new)
    if getattr(a, "mask_wrong_positive_advantage", False):
        flags["mask_wrong_positive_advantage"] = True
        flags["dual_clip_c"] = float(getattr(a, "dual_clip_c", None) or DUAL_CLIP_C)
    elif getattr(a, "dual_clip_c", None) is not None:
        flags["dual_clip_c"] = float(a.dual_clip_c)
    if getattr(a, "ct_guard", False):
        flags["ct_guard"] = True
        flags["guard_window_groups"] = int(getattr(a, "guard_window_groups", GUARD_WINDOW_GROUPS))
    if getattr(a, "holdout", False):
        flags["holdout"] = True
        flags["holdout_fraction"] = float(getattr(a, "holdout_fraction", HOLDOUT_DEFAULT_FRACTION))
        if getattr(a, "holdout_seed", None) is not None:
            flags["holdout_seed"] = str(a.holdout_seed)
    elif getattr(a, "holdout_seed", None) is not None:
        flags["holdout_seed"] = str(a.holdout_seed)
    if getattr(a, "arm", None) is not None:
        flags["arm"] = str(a.arm)
    if getattr(a, "train_seed", None) is not None:
        flags["train_seed"] = int(a.train_seed)
    if getattr(a, "trunc_window_updates", None) is not None:
        flags["trunc_window_updates"] = int(a.trunc_window_updates)
    if getattr(a, "trunc_halt_rate", None) is not None:
        flags["trunc_halt_rate"] = float(a.trunc_halt_rate)
    if getattr(a, "stop_after_updates", None) is not None:
        flags["stop_after_updates"] = int(a.stop_after_updates)
    if getattr(a, "resume_from", None) is not None:
        flags["resume_from"] = Path(a.resume_from).name
    return flags


def is_cut(a: argparse.Namespace) -> bool:
    """Whether the run is not the registered shape: --steps, --prompts, --g or --max-new changed."""
    if any(getattr(a, attr, False) for attr in ("rollout_only", "logging_pass", "postpass")):
        return a.g != G or a.max_new != MAX_NEW_TOKENS
    expected_prompts = (
        int(a.steps) * int(getattr(a, "groups_per_update", GROUPS_PER_STEP) or GROUPS_PER_STEP)
        if str(getattr(a, "draw", "first") or "first") == "class_stratified"
        and not getattr(a, "_prompts_explicit", False)
        else PROMPTS
    )
    return a.steps != STEPS or a.prompts != expected_prompts or a.g != G or a.max_new != MAX_NEW_TOKENS


def execute(
    a: argparse.Namespace,
    *,
    build: Callable[[argparse.Namespace], dict],
    rig=None,
    counter_runner: Callable[[], str] = run_typeperf_gpu_memory,
) -> int:
    """One S2 run: guards, prompts, steps, readout, receipt. ``build`` makes the model bundle; ``rig`` is the
    machine's side (the GPU query, the heartbeat, the sampler), None for a dry rehearsal."""
    if getattr(a, "alloc_conf", None) is not None and not a.dry and build is real_bundle:
        configure_alloc_conf(a.alloc_conf)
    if getattr(a, "paired_estimators", None) is not None:
        run_paired_estimators_from_spec(Path(a.paired_estimators), a.out)
        return 0
    save_all_on, save_all_reason = validate_save_all_args(a)
    rollout_only, descriptive_pass, prepass_seed_val, ckpt_sha = validate_pass_mode_args(a)
    training_run = validate_training_args(a)
    ckpt_files_sha = checkpoint_file_shas(getattr(a, "checkpoint", None))
    frozen_pip_freeze = getattr(a, "frozen_pip_freeze", None)
    require_frozen = bool(
        getattr(a, "require_frozen_stack", False)
        or (not a.dry and (build is real_bundle or frozen_pip_freeze is not None))
    )
    if not a.dry or frozen_pip_freeze is not None or require_frozen:
        frozen_env = check_frozen_env(frozen_pip_freeze, require_exact_versions=require_frozen)
    else:
        frozen_env = {
            "stack_versions": {k: package_version(k) for k in FROZEN_STACK_VERSIONS},
            "pip_freeze_sha256": None,
        }
    if getattr(a, "logprob_microbatch", None) is not None and a.logprob_microbatch < 1:
        raise SystemExit(f"--logprob-microbatch must be >= 1, got {a.logprob_microbatch}")
    if getattr(a, "stress_prefix", 0) < 0:
        raise SystemExit(f"--stress-prefix must be >= 0, got {a.stress_prefix}")
    if getattr(a, "groups_per_update", GROUPS_PER_STEP) < 1:
        raise SystemExit(f"--groups-per-update must be >= 1, got {a.groups_per_update}")
    if rollout_only and getattr(a, "stress_prefix", 0) > 0:
        raise SystemExit("refusing to start: --rollout-only cannot be used with --stress-prefix > 0")
    draw_mode = str(getattr(a, "draw", "first") or "first")
    if rollout_only:
        ro_draw_flags: list[str] = []
        if draw_mode != "first":
            ro_draw_flags.append(f"--draw {draw_mode}")
        for flag, attr in (
            ("--draw-seed", "draw_seed"),
            ("--prepass-jsonl", "prepass_jsonl"),
            ("--prepass-sha256", "prepass_sha256"),
            ("--k-min", "k_min"),
            ("--k-max", "k_max"),
        ):
            if getattr(a, attr, None) is not None:
                ro_draw_flags.append(flag)
        if ro_draw_flags:
            joined_flags = ", ".join(ro_draw_flags)
            raise SystemExit(
                f"refusing to start (unused_draw_flags): --rollout-only cannot be used with {joined_flags}"
            )
    stress_pool = str(getattr(a, "stress_pool", "selected") or "selected")

    rows, gold_hashes = read_gold(a.gold)
    draw_record: dict = {}
    holdout_ids_list: list[str] = []
    if rollout_only:
        prompts = select_prompts(rows, None)
    else:
        prepass_jsonl = getattr(a, "prepass_jsonl", None)
        prepass_sha256 = getattr(a, "prepass_sha256", None)
        prepass_spec = (
            (prepass_jsonl, prepass_sha256)
            if (prepass_jsonl is not None or prepass_sha256 is not None)
            else None
        )
        hold_frac = float(getattr(a, "holdout_fraction", HOLDOUT_DEFAULT_FRACTION))
        hold_seed_raw = getattr(a, "holdout_seed", None)
        hold_seed_valid = (
            hold_seed_raw is not None
            and not isinstance(hold_seed_raw, bool)
            and str(hold_seed_raw).strip()
        )
        hold_seed = str(hold_seed_raw) if hold_seed_valid else None
        if draw_mode == "class_stratified":
            if prepass_spec is None and getattr(a, "pass_a_jsonl", None) is not None:
                prepass_spec = (
                    getattr(a, "pass_a_jsonl", None),
                    getattr(a, "pass_a_sha256", None),
                )
            total_groups_needed = int(a.steps) * int(
                getattr(a, "groups_per_update", GROUPS_PER_STEP) or GROUPS_PER_STEP
            )
            if not getattr(a, "_prompts_explicit", False):
                a.prompts = total_groups_needed
            elif int(a.prompts) < total_groups_needed:
                raise SystemExit(
                    f"refusing to start (prompts_smaller_than_total_groups): "
                    f"--draw class_stratified requires --prompts ({a.prompts}) >= "
                    f"steps * groups_per_update ({total_groups_needed})"
                )
        if getattr(a, "holdout", False):
            hold_spec = prepass_spec or (
                (getattr(a, "pass_a_jsonl", None), getattr(a, "pass_a_sha256", None))
                if getattr(a, "pass_a_jsonl", None) is not None
                else None
            )
            if hold_spec is None:
                raise SystemExit(
                    "refusing to start (holdout_requires_pass_a): "
                    "--holdout requires --pass-a-jsonl/--pass-a-sha256 or --prepass-jsonl/--prepass-sha256"
                )
            _, _, hold_meta = split_stratified_holdout(
                rows,
                hold_spec,
                holdout_fraction=hold_frac,
                holdout_seed=hold_seed,
            )
            holdout_ids_list = list(hold_meta["holdout_ids"])
            draw_record["holdout_seed"] = hold_meta["holdout_seed"]
            draw_record["holdout_count"] = hold_meta["holdout_count"]
            draw_record["holdout_ids_sha256"] = hold_meta["holdout_ids_sha256"]
            draw_record["train_ids_sha256"] = hold_meta["train_ids_sha256"]
        prompts, draw_rec = draw_prompts(
            rows,
            a.prompts,
            mode=draw_mode,
            seed=getattr(a, "draw_seed", None),
            prepass=prepass_spec,
            k_min=getattr(a, "k_min", None),
            k_max=getattr(a, "k_max", None),
            holdout_fraction=hold_frac,
            holdout_seed=hold_seed,
            holdout_ids=holdout_ids_list if holdout_ids_list else None,
        )
        if draw_mode != "first":
            draw_record.update(draw_rec)
        if draw_mode == "class_stratified" and prepass_spec is not None and not holdout_ids_list:
            _, _, hold_meta = split_stratified_holdout(
                rows,
                prepass_spec,
                holdout_fraction=hold_frac,
                holdout_seed=hold_seed,
            )
            holdout_ids_list = list(hold_meta["holdout_ids"])
            draw_record["holdout_seed"] = hold_meta["holdout_seed"]
    prepass_baseline_k, _ = load_prepass_baseline(
        getattr(a, "prepass_baseline", None),
        getattr(a, "prepass_baseline_sha256", None),
        [r["id"] for r in prompts],
        g=a.g,
    )
    pass_a_k, pass_a_sha = load_prepass_baseline(
        getattr(a, "pass_a_jsonl", None),
        getattr(a, "pass_a_sha256", None),
        [r["id"] for r in prompts],
        g=None,
    )
    rc_path_val, rc_sha_val, rc_off_val = validate_replay_check_args(a)
    replay_check_k, replay_check_sha = load_replay_check_jsonl(
        rc_path_val,
        rc_sha_val,
        [r["id"] for r in prompts],
        g=a.g,
    )
    if pass_a_k is None and not rollout_only and getattr(a, "prepass_jsonl", None) is not None:
        pass_a_k, pass_a_sha = load_prepass_baseline(
            getattr(a, "prepass_jsonl", None),
            getattr(a, "prepass_sha256", None),
            [r["id"] for r in prompts],
            g=None,
        )
    if getattr(a, "ct_guard", False) and not pass_a_k:
        raise SystemExit(
            "refusing to start (ct_guard_requires_pass_a_k): "
            "--ct-guard requires Pass-A k_correct "
            "(--pass-a-jsonl/--pass-a-sha256 or --prepass-jsonl/--prepass-sha256)"
        )
    if getattr(a, "prompt_lengths", False) and not getattr(a, "probe_memory", False):
        if a.dry:
            tok = DryTokenizer()
        else:  # pragma: no cover - tokenizer load without dry
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(STUDENT, revision=a.revision)
        length_pool = stress_prompt_pool(rows, prompts, stress_pool)
        stats = compute_prompt_lengths(length_pool, tok, stress_pool=stress_pool)
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
    if bool(getattr(a, "postpass", False)) or getattr(a, "checkpoint", None) is not None:
        if not bundle.get("adapter_loaded"):
            raise SystemExit(
                "refusing to start (checkpoint_adapter_not_loaded): "
                "--postpass / --checkpoint requires the adapter to be loaded into the model"
            )
    if rollout_only and a.dry and bundle.get("dry_prepass_cap") is not None:
        prompts = prompts[: int(bundle["dry_prepass_cap"])]
    ct_guard_obj = (
        CtCollapseGuard(
            window_groups=int(getattr(a, "guard_window_groups", GUARD_WINDOW_GROUPS)),
            pass_a_k=pass_a_k,
        )
        if getattr(a, "ct_guard", False)
        else None
    )
    trunc_guard_obj: TruncationHaltGuard | None = None
    train_header: dict[str, Any] | None = None
    resume_state: dict[str, Any] | None = None
    resumed_from_rec: dict[str, Any] | None = None
    start_step = 0
    chunk_index = 1
    train_progress: dict[str, Any] = {}
    critic_stats: dict[str, int] = {}
    if training_run:
        trunc_guard_obj = TruncationHaltGuard(
            window_updates=int(a.trunc_window_updates), halt_rate=float(a.trunc_halt_rate)
        )
        train_header = build_train_header(
            a,
            flags={**active_flags(a), **draw_record},
            draw_record=draw_record,
            prompt_ids=[r["id"] for r in prompts],
            gold_hashes=gold_hashes,
            revision_resolved=bundle.get("revision"),
            stack_versions=frozen_env.get("stack_versions"),
            pip_freeze_sha256=frozen_env.get("pip_freeze_sha256"),
            commit=aspire_commit(),
            checkpoint_sha=ckpt_sha,
        )
        if getattr(a, "resume_from", None) is not None:
            resume_state = load_train_state(
                Path(a.resume_from), header=train_header, steps_total=int(a.steps)
            )
            apply_train_state(resume_state, model=bundle["model"], optimiser=bundle["optimiser"])
            start_step = int(resume_state["next_step"])
            chunk_index = int(resume_state["chunk_index"]) + 1
            if ct_guard_obj is not None and resume_state.get("ct_guard") is not None:
                ct_guard_obj = CtCollapseGuard.from_state(resume_state["ct_guard"], pass_a_k=pass_a_k)
            trunc_guard_obj = TruncationHaltGuard.from_state(resume_state["trunc_guard"])
            resumed_from_rec = {
                "chunk_dir": Path(a.resume_from).name,
                "state_sha256": resume_state["_state_sha256"],
                "next_step": start_step,
                "chunk_index": int(resume_state["chunk_index"]),
            }
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / f"s2-{a.path}-train-header.json").write_text(
            json.dumps(
                {**train_header, "chunk_index": chunk_index, "resumed_from": resumed_from_rec}, indent=1
            ),
            encoding="utf-8",
            newline="\n",
        )
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
                stack_versions=frozen_env.get("stack_versions"),
                pip_freeze_sha256=frozen_env.get("pip_freeze_sha256"),
            )

        sampler = VramSampler(rig.vram_query) if rig is not None and not a.dry else None
        cuda_mod = bundle.get("cuda") or (
            getattr(require_torch(), "cuda", None) if rig is not None and not a.dry else None
        )
        log_sdpa = bool(getattr(a, "log_sdpa_backend", False))
        sign_diag = bool(getattr(a, "sign_diagnostics", False))
        phase_logger = PhaseLogger(
            a.out / f"s2-{a.path}-phases.jsonl",
            enabled=rig is not None and not a.dry,
            cuda=cuda_mod,
            log_sdpa_backend=log_sdpa,
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
        rollout_save_records: list[dict] = []
        group_residuals: list[tuple[str, int, float]] = []
        stress_meta: dict = {}

        def write_eval_summary(summary_doc: dict) -> None:
            a.out.mkdir(parents=True, exist_ok=True)
            (a.out / f"s2-{a.path}-eval-summary.json").write_text(
                json.dumps(summary_doc, indent=1), encoding="utf-8", newline="\n"
            )

        def run_it() -> tuple[list[dict], dict | None, bool]:
            nonlocal prepass_records, stress_meta
            if rollout_only:
                header_path = a.out / f"s2-{a.path}-prepass-header.json"
                jsonl_path = a.out / f"s2-{a.path}-prepass.jsonl"
                exercise_dry_resume = (
                    a.dry
                    and bundle.get("dry_prepass_cap") is not None
                    and getattr(a, "stop_after_new", None) is None
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
                        base_seed=prepass_seed_val,
                        stop_after_new=half,
                        descriptive=descriptive_pass,
                        checkpoint_sha=ckpt_sha,
                        checkpoint_files_sha=ckpt_files_sha,
                        pass_a_k=pass_a_k,
                        pass_a_sha=pass_a_sha,
                        replay_check_k=replay_check_k,
                        replay_check_sha=replay_check_sha,
                        replay_check_off_reason=rc_off_val,
                        stack_versions=frozen_env["stack_versions"],
                        pip_freeze_sha256=frozen_env["pip_freeze_sha256"],
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
                        base_seed=prepass_seed_val,
                        descriptive=descriptive_pass,
                        checkpoint_sha=ckpt_sha,
                        checkpoint_files_sha=ckpt_files_sha,
                        pass_a_k=pass_a_k,
                        pass_a_sha=pass_a_sha,
                        replay_check_k=replay_check_k,
                        replay_check_sha=replay_check_sha,
                        replay_check_off_reason=rc_off_val,
                        stack_versions=frozen_env["stack_versions"],
                        pip_freeze_sha256=frozen_env["pip_freeze_sha256"],
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
                    base_seed=prepass_seed_val,
                    descriptive=descriptive_pass,
                    checkpoint_sha=ckpt_sha,
                    checkpoint_files_sha=ckpt_files_sha,
                    pass_a_k=pass_a_k,
                    pass_a_sha=pass_a_sha,
                    replay_check_k=replay_check_k,
                    replay_check_sha=replay_check_sha,
                    replay_check_off_reason=rc_off_val,
                    stack_versions=frozen_env["stack_versions"],
                    pip_freeze_sha256=frozen_env["pip_freeze_sha256"],
                )
                prepass_records = recs_ro
                return rows_ro, stop_ro, oom_ro
            if stress_n > 0:
                stress_pool_rows = stress_prompt_pool(rows, prompts, stress_pool)
                longest, stress_prompt_tokens = select_stress_prompt(stress_pool_rows, bundle["tok"])
                stress_meta = {
                    "stress_pool": stress_pool,
                    "stress_prompt_id": str(longest["id"]),
                    "stress_prompt_tokens": int(stress_prompt_tokens),
                    "stress_width": int(stress_prompt_tokens + a.max_new),
                }
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
                sign_diagnostics=sign_diag,
                prepass_baseline_k=prepass_baseline_k,
                group_residuals=group_residuals if prepass_baseline_k is not None else None,
                save_all_records=rollout_save_records if save_all_on else None,
                save_all_path=save_all_file_path(a.out, a.path) if save_all_on else None,
                base_seed=int(getattr(a, "draw_seed", None) or 0),
                holdout_ids=holdout_ids_list or None,
                mask_wrong_positive_advantage=bool(getattr(a, "mask_wrong_positive_advantage", False)),
                dual_clip_c=getattr(a, "dual_clip_c", None),
                ct_guard=ct_guard_obj,
                start_step=start_step,
                stop_after_updates=getattr(a, "stop_after_updates", None) if training_run else None,
                train_seed=int(a.train_seed) if training_run else None,
                trunc_guard=trunc_guard_obj,
                guard_history_path=(
                    a.out / f"s2-{a.path}-guard-history.jsonl.gz"
                    if (ct_guard_obj is not None or trunc_guard_obj is not None)
                    else None
                ),
                arm=getattr(a, "arm", None) if training_run else None,
                critic_archive_path=(
                    a.out / f"s2-{a.path}-critic-retries.jsonl.gz"
                    if training_run and a.arm == "critic" and save_all_on
                    else None
                ),
                critic_stats=critic_stats,
                progress=train_progress,
            )

        if sampler is not None:
            with sampler:
                steps, stopped, oom = run_it()
        else:
            steps, stopped, oom = run_it()
        save_all_sha: str | None = None
        if save_all_on:
            sa_path = save_all_file_path(a.out, a.path)
            if not sa_path.exists():
                append_save_all(sa_path, ())
            save_all_sha = hashlib.sha256(sa_path.read_bytes()).hexdigest()
        eval_summary_doc = evaluation_summary(eval_records, advantage_mode=adv_mode)
        write_eval_summary(eval_summary_doc)
        peak = sampler.peak_mib if sampler is not None else None
        flags_in_use = {**active_flags(a), **stress_meta, **draw_record}
        prepass_summary_doc = (
            prepass_summary(
                prepass_records,
                g=a.g,
                quote_reward=quote_rew,
                reproducible=(a.path != "llamacpp"),
                descriptive=descriptive_pass,
                pass_a_k=pass_a_k,
                checkpoint_sha=ckpt_sha,
                checkpoint_files_sha=ckpt_files_sha,
            )
            if rollout_only and prepass_records
            else None
        )
        residual_band_doc = (
            mean_residual_by_k_band(group_residuals) if prepass_baseline_k is not None else None
        )
        if training_run:
            remaining = int(a.steps) - start_step
            sau = getattr(a, "stop_after_updates", None)
            requested_now = min(int(sau), remaining) if sau is not None else remaining
        result = readout(
            steps,
            peak_mib=peak,
            steps_requested=(
                len(prompts) if rollout_only else (requested_now if training_run else a.steps)
            ),
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
            quote_reward=quote_rew,
            mean_residual_by_k_band_record=residual_band_doc,
        )
        next_step = int(train_progress.get("next_step", start_step))
        if training_run:
            line = train_verdict_line(
                stopped=stopped,
                oom=oom,
                next_step=next_step,
                steps_total=int(a.steps),
                chunk_stop=train_progress.get("chunk_stop"),
                fits=result.get("fits"),
                dry=a.dry,
            )
        else:
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
        a.out.mkdir(parents=True, exist_ok=True)
        adapter_sha: str | None = None
        adapter_files_sha: dict[str, str] | None = None
        if not rollout_only and callable(getattr(bundle.get("model"), "save_pretrained", None)):
            adapter_dir = a.out / "adapter"
            adapter_dir.mkdir(parents=True, exist_ok=True)
            bundle["model"].save_pretrained(str(adapter_dir))
            adapter_files_sha = checkpoint_file_shas(adapter_dir)
            adapter_sha = checkpoint_sha256(adapter_dir)
        training_block: dict[str, Any] | None = None
        if training_run and train_header is not None and trunc_guard_obj is not None:
            sa_file = save_all_file_path(a.out, a.path)
            chain = list((resume_state or {}).get("save_all_chain") or [])
            chain.append(
                {
                    "chunk_dir": a.out.name,
                    "chunk_index": chunk_index,
                    "file": sa_file.name,
                    "sha256": save_all_sha,
                    "records": len(read_save_all(sa_file)) if save_all_on and sa_file.exists() else None,
                    "first_step": start_step,
                    "next_step": next_step,
                }
            )
            halted = train_progress.get("halted")
            state_ok = not oom and (stopped is None or bool(stopped.get("triggered_rules")))
            state_sha = (
                save_train_state(
                    a.out,
                    model=bundle["model"],
                    optimiser=bundle["optimiser"],
                    header=train_header,
                    next_step=next_step,
                    steps_total=int(a.steps),
                    chunk_index=chunk_index,
                    ct_guard=ct_guard_obj,
                    trunc_guard=trunc_guard_obj,
                    save_all_chain=chain,
                    adapter_sha=adapter_sha,
                    halted=halted,
                    resumed_from=resumed_from_rec,
                )
                if state_ok
                else None
            )
            gh_path = a.out / f"s2-{a.path}-guard-history.jsonl.gz"
            cr_path = a.out / f"s2-{a.path}-critic-retries.jsonl.gz"
            training_block = {
                "arm": a.arm,
                "train_seed": int(a.train_seed),
                "chunk_index": chunk_index,
                "start_step": start_step,
                "next_step": next_step,
                "steps_total": int(a.steps),
                "chunk_stop": train_progress.get("chunk_stop"),
                "halted": halted,
                "resumed_from": resumed_from_rec,
                "header_sha256": hashlib.sha256(
                    (a.out / f"s2-{a.path}-train-header.json").read_bytes()
                ).hexdigest(),
                "train_state_sha256": state_sha,
                "train_state_saved": state_sha is not None,
                "guard_history_sha256": _file_sha256(gh_path) if gh_path.exists() else None,
                "start_policy": train_header["start_policy"],
                "critic": (
                    {
                        **(train_header.get("critic") or {}),
                        "hinted_groups": critic_stats.get("hinted_groups", 0),
                        "retry_samples": critic_stats.get("retry_samples", 0),
                        "retry_correct": critic_stats.get("retry_correct", 0),
                        "signal_drops": 0,
                        "retries_sha256": _file_sha256(cr_path) if cr_path.exists() else None,
                    }
                    if a.arm == "critic"
                    else None
                ),
            }
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
            save_all=save_all_on,
            save_all_sha256=save_all_sha,
            save_all_off_reason=save_all_reason,
            checkpoint_sha256=ckpt_sha,
            checkpoint_files_sha256=ckpt_files_sha,
            adapter_sha256=adapter_sha,
            adapter_files_sha256=adapter_files_sha,
            stack_versions=frozen_env["stack_versions"],
            pip_freeze_sha256=frozen_env["pip_freeze_sha256"],
        )
        if training_block is not None:
            receipt["training"] = training_block
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
        if training_block is not None:
            summary_out["training"] = {
                k: training_block[k] for k in ("arm", "chunk_index", "start_step", "next_step", "steps_total")
            }
        print(json.dumps(summary_out))
        print(line)
        if isinstance(stopped, dict) and stopped.get("reason") == "LOGGING_PASS_REPLAY_MISMATCH":
            raise SystemExit(str(stopped.get("error") or "LOGGING_PASS_REPLAY_MISMATCH"))
        return exit_status(line)
    finally:
        if group_logger is not None:
            group_logger.close()
        if phase_logger is not None:
            phase_logger.close()
        close = bundle.get("close")
        if close:
            close()


class _TrackPromptsAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        setattr(namespace, "_prompts_explicit", True)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--path", choices=("hf", "llamacpp"), default="hf")
    ap.add_argument("--steps", type=int, default=STEPS, help="a value other than 20 makes the run S2-LIMITED")
    ap.add_argument(
        "--prompts",
        type=int,
        default=PROMPTS,
        action=_TrackPromptsAction,
        help="a value other than 32 makes the run S2-LIMITED",
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
        help="CPU only: tokenize the selected or tune prompts and write count, min, median, p90 and max",
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
        "--stress-pool",
        choices=STRESS_POOLS,
        default="selected",
        help="prompt pool for --stress-prefix and --prompt-lengths: 'selected' (default) or 'tune'",
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
            "it trains nothing and writes counts per item. "
            "quote_pass is diagnostic: every evidence quote found in the passages; "
            "not the registered reward (quote rule <v>)."
        ),
    )
    ap.add_argument(
        "--draw",
        choices=DRAW_MODES,
        default="first",
        help="prompt draw mode: 'first' (default, sorted prefix), 'uniform' (seeded sample), or 'kband'",
    )
    ap.add_argument(
        "--draw-seed",
        type=int,
        default=None,
        help="integer seed for --draw uniform or --draw kband",
    )
    ap.add_argument(
        "--prepass-jsonl",
        type=Path,
        default=None,
        help="pre-pass JSONL file for --draw kband",
    )
    ap.add_argument(
        "--prepass-sha256",
        default=None,
        help="expected sha256 hex digest of --prepass-jsonl for --draw kband",
    )
    ap.add_argument(
        "--k-min",
        type=int,
        default=None,
        help="minimum k_correct (inclusive) for --draw kband",
    )
    ap.add_argument(
        "--k-max",
        type=int,
        default=None,
        help="maximum k_correct (inclusive) for --draw kband",
    )
    ap.add_argument(
        "--prepass-seed",
        type=int,
        default=PREPASS_BASE_SEED,
        help="base integer seed for item_seed in --rollout-only (default 0)",
    )
    ap.add_argument(
        "--sign-diagnostics",
        action="store_true",
        help=(
            "record sign-disagreement and variance diagnostics in group and step logs:\n"
            "outcome_var: population variance of outcome (verdict + m/k quote match under v2);\n"
            "correct_var: population variance of the 0/1 correct-verdict indicator"
        ),
    )
    ap.add_argument(
        "--prepass-baseline",
        type=Path,
        default=None,
        help="independent pre-pass JSONL file for per-group k_prepass and residual=(k_online-k_prepass)/g",
    )
    ap.add_argument(
        "--prepass-baseline-sha256",
        default=None,
        help="expected sha256 hex digest of --prepass-baseline",
    )
    ap.add_argument(
        "--log-sdpa-backend",
        action="store_true",
        help="record sdpa_backend ('flash', 'efficient', 'math', 'cudnn', or 'unknown') in each phase row",
    )
    ap.add_argument(
        "--save-all",
        dest="save_all",
        action="store_true",
        default=True,
        help="write every pass completion and rollout to compressed JSONL beside outputs (default: on)",
    )
    ap.add_argument(
        "--no-save-all",
        dest="save_all",
        action="store_false",
        help="disable save-all completion logging (requires --save-all-off-reason)",
    )
    ap.add_argument(
        "--save-all-off-reason",
        default=None,
        help="required non-empty reason when --no-save-all is passed",
    )
    ap.add_argument(
        "--pass-id",
        default=None,
        help="optional pass identifier recorded in save-all completion rows",
    )
    ap.add_argument(
        "--stop-after-new",
        type=int,
        default=None,
        help=(
            "stop cleanly after N >= 1 newly completed items in --rollout-only, "
            "--logging-pass, or --postpass mode so summaries and receipts run"
        ),
    )
    ap.add_argument(
        "--logging-pass",
        action="store_true",
        help="run the starting-policy logging pass (seed 1, descriptive=true, never accepted as baseline)",
    )
    ap.add_argument(
        "--postpass",
        action="store_true",
        help="run a post-training evaluation pass over all tune items (refuses seeds 0 and 1)",
    )
    ap.add_argument(
        "--pass-seed",
        type=int,
        default=None,
        help="sampling seed for --logging-pass (must be 1) or --postpass (must not be 0 or 1)",
    )
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="LoRA checkpoint directory or file for --postpass evaluation",
    )
    ap.add_argument(
        "--pass-a-jsonl",
        type=Path,
        default=None,
        help="Pass-A pre-pass JSONL used to assign fixed [0,5], [6,7], 8 bands for truncation counts",
    )
    ap.add_argument(
        "--pass-a-sha256",
        default=None,
        help="expected sha256 hex digest of --pass-a-jsonl",
    )
    ap.add_argument(
        "--replay-check",
        type=Path,
        default=None,
        help="Pass-B2 pre-pass JSONL used to verify item-by-item k_correct replay during --logging-pass",
    )
    ap.add_argument(
        "--replay-check-sha256",
        default=None,
        help="expected sha256 hex digest of --replay-check",
    )
    ap.add_argument(
        "--replay-check-off-reason",
        default=None,
        help="explicit non-empty reason required when running --logging-pass without --replay-check",
    )
    ap.add_argument(
        "--frozen-pip-freeze",
        type=Path,
        default=None,
        help="optional path to frozen pip-freeze.txt verified against pre-registered sha256",
    )
    ap.add_argument(
        "--require-frozen-stack",
        action="store_true",
        help="verify installed torch, transformers, and bitsandbytes match FROZEN_STACK_VERSIONS",
    )
    ap.add_argument(
        "--holdout",
        action="store_true",
        help="hold out a stratified 20%% (Pass-A band x gold class) from all training draws",
    )
    ap.add_argument(
        "--holdout-fraction",
        type=float,
        default=HOLDOUT_DEFAULT_FRACTION,
        help="stratified holdout fraction (default 0.20)",
    )
    ap.add_argument(
        "--holdout-seed",
        default=None,
        help="deterministic seed string for the stratified holdout split (required when --holdout is set)",
    )
    ap.add_argument(
        "--paired-estimators",
        type=Path,
        default=None,
        help="JSON spec path to compute paired estimators across Pass A, Pass B, and post-pass pairs",
    )
    ap.add_argument(
        "--ct-guard",
        action="store_true",
        help="enable two-class CT collapse guard (S1, S2, U1, U2) over sampled-group windows",
    )
    ap.add_argument(
        "--guard-window-groups",
        type=int,
        default=GUARD_WINDOW_GROUPS,
        help="number of sampled groups per CT collapse guard window (default 40 = 10 updates)",
    )
    ap.add_argument(
        "--mask-wrong-positive-advantage",
        action="store_true",
        help="clamp advantages on wrong-label completions to <= 0 after mean-centring (no re-centring)",
    )
    ap.add_argument(
        "--dual-clip-c",
        type=float,
        default=None,
        help="dual-clip coefficient c on negative advantages (default 3.0 when masking wrong positive adv)",
    )
    ap.add_argument(
        "--arm",
        choices=ARMS,
        default=None,
        help=(
            "names a training run and its arm: 'mech' (mechanical reward only) or 'critic' (the same reward, "
            "plus a tool-hinted retry of each all-wrong group, scored and archived, never trained on). "
            "Required, with no default, whenever any training flag is given"
        ),
    )
    ap.add_argument(
        "--train-seed",
        type=int,
        default=None,
        help="training run: seeds the fresh LoRA's init and every training rollout (required, no default)",
    )
    ap.add_argument(
        "--stop-after-updates",
        type=int,
        default=None,
        help="training run: end this chunk cleanly after N >= 1 updates and save the train state",
    )
    ap.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="training run: continue from the previous chunk's directory (its train-state); a fresh --out",
    )
    ap.add_argument(
        "--trunc-window-updates",
        type=int,
        default=None,
        help="training run: rolling truncation window in updates (required, no default)",
    )
    ap.add_argument(
        "--trunc-halt-rate",
        type=float,
        default=None,
        help="training run: halt when the full window's truncated share exceeds this (required, no default)",
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
    if getattr(a, "alloc_conf", None) is not None and not a.dry:
        configure_alloc_conf(a.alloc_conf)
    if a.dry:
        return execute(apply_dry_defaults(a), build=dry_bundle)
    return execute(a, build=real_bundle, rig=Rig(a))


if __name__ == "__main__":
    sys.exit(main())
