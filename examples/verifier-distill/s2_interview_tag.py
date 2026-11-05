"""Two-judge descriptive tagging, judge controls, agreement stats, and blinded audit export for S2 Interview.

Implements the descriptive tagging layer of DESIGN-ct-interview-v2-2026-11-05.md:
- Two non-Qwen judges (refusing any judge whose name contains 'qwen', case-insensitive).
- Tags Turn 2 Q2's answer against the original Turn 1 <think> trace on:
  - missing_element_match: 'same' | 'different' | 'none'
  - reason_type: 'lacks' | 'conflicts' | 'generic_or_none'
- Requires explicit ``turn1_thinking`` on every record (never falls back to Turn 2's own thinking).
- Positive and negative judge controls:
  - 30 shuffled pairs (expect 'different')
  - 30 verbatim positives (expect 'same')
  - 15 empty-thinking controls with truly empty thinking (expect 'none')
  - Gated at >= 90% (0.90) accuracy on each control set.
- Inter-judge agreement: raw agreement, Cohen's kappa, and Gwet's AC1.
- 30-item blinded human audit export (10 disagreements + 20 stratified by label, model identifiers
  replaced with blind codes and separate key file).
- Hash-only ``s2-interview-tags.jsonl`` (ids, tags, token counts, and sha256 hashes only; no raw text).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import statistics
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

MISSING_ELEMENT_LABELS = ("same", "different", "none")
REASON_TYPE_LABELS = ("lacks", "conflicts", "generic_or_none")
CONTROL_GATE_MIN_ACCURACY = 0.90
TAG_RECORD_KEYS = (
    "item_id",
    "policy",
    "question_id",
    "original_turn1_thinking_tokens",
    "turn1_thinking_sha256",
    "q2_answer_sha256",
    "judge_a_missing_element_match",
    "judge_b_missing_element_match",
    "consensus_missing_element_match",
    "judge_a_reason_type",
    "judge_b_reason_type",
    "consensus_reason_type",
)


def validate_judge_name(judge_name: str) -> str:
    """Refuse any judge whose name/identifier is empty or contains 'qwen' (case-insensitive)."""
    cleaned = str(judge_name or "").strip()
    if not cleaned:
        raise SystemExit("refusing judge (judge_name_empty): judge name must be non-empty")
    if "qwen" in cleaned.lower():
        raise SystemExit(
            f"refusing judge (judge_qwen_family_forbidden): "
            f"judge {cleaned!r} belongs to the Qwen family; interview tagging requires non-Qwen judges"
        )
    return cleaned


def require_two_non_qwen_judges(judge_a_name: str, judge_b_name: str) -> tuple[str, str]:
    """Validate two distinct non-Qwen judge identifiers."""
    ja = validate_judge_name(judge_a_name)
    jb = validate_judge_name(judge_b_name)
    if ja.lower() == jb.lower():
        raise SystemExit(
            f"refusing judges (judges_must_be_distinct): judge_a and judge_b must be distinct, got {ja!r}"
        )
    return ja, jb


def validate_tag_record(tag: Mapping[str, Any]) -> dict[str, str]:
    """Validate that a judge tag dict contains valid ``missing_element_match`` and ``reason_type`` labels."""
    mem = str(tag.get("missing_element_match", "")).strip().lower()
    rt = str(tag.get("reason_type", "")).strip().lower()
    if mem not in MISSING_ELEMENT_LABELS:
        raise ValueError(
            f"Invalid missing_element_match {mem!r}; expected one of {MISSING_ELEMENT_LABELS}"
        )
    if rt not in REASON_TYPE_LABELS:
        raise ValueError(f"Invalid reason_type {rt!r}; expected one of {REASON_TYPE_LABELS}")
    return {"missing_element_match": mem, "reason_type": rt}


# -- Positive and negative judge controls (30 shuffled, 30 verbatim, 15 empty-thinking) ----------------------
def build_tagging_prompt(turn1_thinking: str, q2_answer: str) -> str:
    """Build the judge prompt comparing the original Turn 1 ``<think>`` trace against Turn 2 Q2's answer."""
    return (
        "Compare the verifier's original Turn 1 <think> trace against its Turn 2 Q2 answer "
        "(which states why the evidence was insufficient and names MISSING_FACT).\n\n"
        f"Original Turn 1 <think>:\n{turn1_thinking}\n\n"
        f"Turn 2 Q2 answer:\n{q2_answer}\n\n"
        "Respond with JSON: "
        '{"missing_element_match": "same" | "different" | "none", '
        '"reason_type": "lacks" | "conflicts" | "generic_or_none"}'
    )


def extract_interview_q2_records(
    interview_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Select non-cold Turn 2 ``Q2`` records for two-judge tagging against ``turn1_thinking``."""
    has_qid = any("question_id" in r for r in interview_records)
    selected: list[dict[str, Any]] = []
    for r in interview_records:
        if r.get("cold_control") is True or str(r.get("policy", "")).startswith("cold_control"):
            continue
        if has_qid and r.get("question_id") != "Q2":
            continue
        rec_copy = dict(r)
        if "q2_answer" not in rec_copy:
            rec_copy["q2_answer"] = str(rec_copy.get("answer", rec_copy.get("q1_answer", "")))
        selected.append(rec_copy)
    if has_qid and not selected and interview_records:
        raise SystemExit(
            "refusing interview tagging (missing_q2_records): "
            "interview tagging requires Q2 turn records, but none were found"
        )
    return selected


def build_judge_controls(
    items: Sequence[Mapping[str, Any]],
    *,
    seed: int = 20261105,
    n_shuffled: int = 30,
    n_verbatim: int = 30,
    n_empty: int = 15,
) -> dict[str, list[dict[str, Any]]]:
    """Construct the pre-registered positive and negative judge control sets:
    - ``shuffled`` (30): Q2 answer paired with a different item's Turn 1 ``<think>`` -> ``different``.
    - ``verbatim`` (30): Turn 1 ``<think>`` missing-element sentence + Q2 paraphrase -> ``same``.
    - ``empty_thinking`` (15): truly empty ``<think>`` trace -> expect ``none``.
    """
    rng = random.Random(seed)
    pool: list[dict[str, Any]] = []
    for x in items:
        t_val = str(x.get("turn1_thinking", x.get("thinking", ""))).strip()
        a_val = str(x.get("q2_answer", x.get("q1_answer", x.get("answer", "")))).strip()
        if t_val and a_val:
            pool.append(
                {
                    "item_id": x.get("item_id"),
                    "thinking": t_val,
                    "q2_answer": a_val,
                    "q1_answer": a_val,
                }
            )
    if len(pool) < 2:
        # Synthesize distinct base items if fewer than 2 items are provided
        pool = [
            {
                "item_id": f"ctrl_src_{i:03d}",
                "thinking": f"The passage does not state the release date for module {i} in version {i}.",
                "q2_answer": f"The evidence lacks the specific release date for module {i}.",
                "q1_answer": f"The evidence lacks the specific release date for module {i}.",
            }
            for i in range(max(n_shuffled, n_verbatim, 4))
        ]

    shuffled_controls: list[dict[str, Any]] = []
    for idx in range(n_shuffled):
        src_a = pool[idx % len(pool)]
        other_candidates = [
            p
            for p in pool
            if p.get("item_id") != src_a.get("item_id") and p.get("thinking") != src_a.get("thinking")
        ]
        src_b = rng.choice(other_candidates) if other_candidates else {
            "item_id": f"other_{idx}",
            "thinking": f"Unrelated thinking trace about topic {idx + 1000}.",
        }
        ans_a = str(src_a["q2_answer"])
        shuffled_controls.append(
            {
                "control_id": f"ctrl_shuffled_{idx:02d}",
                "control_type": "shuffled",
                "thinking": str(src_b["thinking"]),
                "q2_answer": ans_a,
                "q1_answer": ans_a,
                "expected_missing_element_match": "different",
            }
        )

    verbatim_controls: list[dict[str, Any]] = []
    for idx in range(n_verbatim):
        src = pool[idx % len(pool)]
        think_txt = str(src["thinking"]).strip()
        verb_ans = f"As noted in the reasoning, {think_txt}\nMISSING_FACT: {think_txt}"
        verbatim_controls.append(
            {
                "control_id": f"ctrl_verbatim_{idx:02d}",
                "control_type": "verbatim",
                "thinking": think_txt,
                "q2_answer": verb_ans,
                "q1_answer": verb_ans,
                "expected_missing_element_match": "same",
            }
        )

    empty_controls: list[dict[str, Any]] = []
    for idx in range(n_empty):
        src = pool[idx % len(pool)]
        ans_src = str(src["q2_answer"])
        empty_controls.append(
            {
                "control_id": f"ctrl_empty_{idx:02d}",
                "control_type": "empty_thinking",
                "thinking": "",
                "q2_answer": ans_src,
                "q1_answer": ans_src,
                "expected_missing_element_match": "none",
            }
        )

    return {
        "shuffled": shuffled_controls,
        "verbatim": verbatim_controls,
        "empty_thinking": empty_controls,
    }


def evaluate_judge_controls(
    controls: Mapping[str, Sequence[Mapping[str, Any]]],
    judge_fn: Callable[[str, str], Mapping[str, Any]],
    *,
    min_accuracy: float = CONTROL_GATE_MIN_ACCURACY,
    enforce_gate: bool = True,
) -> dict[str, Any]:
    """Run ``judge_fn(thinking, q2_answer)`` over the 3 control sets and gate at ``>= min_accuracy``."""
    accuracies: dict[str, float] = {}
    counts: dict[str, dict[str, int]] = {}
    failed_sets: list[str] = []

    for set_name in ("shuffled", "verbatim", "empty_thinking"):
        ctrl_list = list(controls.get(set_name, ()))
        if not ctrl_list:
            raise ValueError(f"Control set {set_name!r} is empty")
        correct = 0
        for item in ctrl_list:
            ans_txt = str(item.get("q2_answer", item.get("q1_answer", "")))
            pred = validate_tag_record(judge_fn(str(item["thinking"]), ans_txt))
            if pred["missing_element_match"] == item["expected_missing_element_match"]:
                correct += 1
        acc = correct / float(len(ctrl_list))
        accuracies[set_name] = round(acc, 6)
        counts[set_name] = {"correct": correct, "total": len(ctrl_list)}
        if acc < float(min_accuracy):
            failed_sets.append(f"{set_name}={acc:.3f}<{min_accuracy:.2f}")

    passed = len(failed_sets) == 0
    if enforce_gate and not passed:
        raise SystemExit(
            f"refusing judge tags (judge_control_gate_failed): "
            f"judge failed >= {min_accuracy:.0%} control gate on {', '.join(failed_sets)}"
        )
    return {
        "passed": passed,
        "min_accuracy": float(min_accuracy),
        "accuracies": accuracies,
        "counts": counts,
        "failed_sets": failed_sets,
    }


# -- Inter-judge agreement statistics (raw agreement, Cohen's kappa, Gwet's AC1) -----------------------------
def compute_agreement_stats(
    labels_a: Sequence[str],
    labels_b: Sequence[str],
    *,
    categories: Sequence[str],
) -> dict[str, Any]:
    """Compute raw agreement, Cohen's kappa, and Gwet's AC1 across ``categories``."""
    if len(labels_a) != len(labels_b):
        raise ValueError("labels_a and labels_b must have the same length")
    n = len(labels_a)
    cats = list(categories)
    k_cats = len(cats)
    if n == 0 or k_cats < 2:
        return {"n": n, "raw_agreement": None, "cohens_kappa": None, "gwets_ac1": None}

    agree_n = sum(1 for a, b in zip(labels_a, labels_b, strict=True) if a == b)
    p_a = agree_n / float(n)

    counts_a = {c: 0 for c in cats}
    counts_b = {c: 0 for c in cats}
    for a, b in zip(labels_a, labels_b, strict=True):
        if a in counts_a:
            counts_a[a] += 1
        if b in counts_b:
            counts_b[b] += 1

    # Cohen's kappa chance agreement: sum_k p_{k,a} * p_{k,b}
    p_e_kappa = sum((counts_a[c] / float(n)) * (counts_b[c] / float(n)) for c in cats)
    if abs(1.0 - p_e_kappa) < 1e-12:
        kappa = 1.0 if abs(1.0 - p_a) < 1e-12 else 0.0
    else:
        kappa = (p_a - p_e_kappa) / (1.0 - p_e_kappa)

    # Gwet's AC1 chance agreement: (1 / (K - 1)) * sum_k pi_k * (1 - pi_k)
    pi_by_c = {c: 0.5 * ((counts_a[c] + counts_b[c]) / float(n)) for c in cats}
    p_e_ac1 = sum(pi_by_c[c] * (1.0 - pi_by_c[c]) for c in cats) / float(k_cats - 1)
    if abs(1.0 - p_e_ac1) < 1e-12:
        ac1 = 1.0 if abs(1.0 - p_a) < 1e-12 else 0.0
    else:
        ac1 = (p_a - p_e_ac1) / (1.0 - p_e_ac1)

    return {
        "n": n,
        "raw_agreement": round(p_a, 6),
        "cohens_kappa": round(kappa, 6),
        "gwets_ac1": round(ac1, 6),
        "disagreements": n - agree_n,
    }


# -- Thinking length confound log ----------------------------------------------------------------------------
def compute_thinking_length_confound(
    tagged_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Log original Turn 1 thinking token length against ``missing_element_match`` ('same' vs not 'same')."""
    if not tagged_records:
        return {"n": 0, "mean_thinking_tokens_by_label": {}, "pearson_r_same_vs_tokens": None}

    by_label: dict[str, list[int]] = {lbl: [] for lbl in MISSING_ELEMENT_LABELS}
    xs_tokens: list[float] = []
    ys_same: list[float] = []
    for r in tagged_records:
        toks = int(r.get("original_turn1_thinking_tokens", 0))
        lbl = str(
            r.get("consensus_missing_element_match")
            or r.get("judge_a_missing_element_match")
            or "none"
        )
        if lbl in by_label:
            by_label[lbl].append(toks)
        xs_tokens.append(float(toks))
        ys_same.append(1.0 if lbl == "same" else 0.0)

    mean_by_label = {
        lbl: (round(float(statistics.fmean(vals)), 2) if vals else None)
        for lbl, vals in by_label.items()
    }
    pearson_r: float | None = None
    if len(xs_tokens) >= 2 and len(set(xs_tokens)) > 1 and len(set(ys_same)) > 1:
        mx = statistics.fmean(xs_tokens)
        my = statistics.fmean(ys_same)
        num = sum((x - mx) * (y - my) for x, y in zip(xs_tokens, ys_same, strict=True))
        den = math.sqrt(sum((x - mx) ** 2 for x in xs_tokens) * sum((y - my) ** 2 for y in ys_same))
        if den > 0:
            pearson_r = round(num / den, 6)
    return {
        "n": len(tagged_records),
        "mean_thinking_tokens_by_label": mean_by_label,
        "pearson_r_same_vs_tokens": pearson_r,
    }


# -- 30-item blinded human audit export ----------------------------------------------------------------------
_FORBIDDEN_AUDIT_KEYS = (
    "policy",
    "model",
    "adapter",
    "checkpoint",
    "pass_id",
    "judge_a_name",
    "judge_b_name",
)


def export_blinded_human_audit(
    tagged_records: Sequence[Mapping[str, Any]],
    out_dir: Path,
    *,
    target_total: int = 30,
    target_disagreements: int = 10,
    seed: int = 20261105,
) -> dict[str, Any]:
    """Export a 30-item blinded human audit sample (10 judge disagreements + 20 stratified by label)
    with all model/policy/judge identifiers stripped and replaced by blind codes.
    """
    rng = random.Random(seed)
    pool = [dict(r) for r in tagged_records]

    disagreements = [
        r
        for r in pool
        if r.get("judge_a_missing_element_match") != r.get("judge_b_missing_element_match")
        or r.get("judge_a_reason_type") != r.get("judge_b_reason_type")
    ]
    agreements = [r for r in pool if r not in disagreements]

    rng.shuffle(disagreements)
    picked_dis = disagreements[: min(target_disagreements, len(disagreements))]
    needed_strat = max(0, min(target_total, len(pool)) - len(picked_dis))

    by_lbl: dict[str, list[dict[str, Any]]] = {lbl: [] for lbl in MISSING_ELEMENT_LABELS}
    for r in agreements:
        lbl = str(r.get("judge_a_missing_element_match", "none"))
        by_lbl.setdefault(lbl, []).append(r)
    for lbl in by_lbl:
        rng.shuffle(by_lbl[lbl])

    picked_strat: list[dict[str, Any]] = []
    active_lbls = [lbl for lbl in MISSING_ELEMENT_LABELS if by_lbl.get(lbl)]
    idx = 0
    while len(picked_strat) < needed_strat and any(by_lbl.values()):
        lbl = active_lbls[idx % len(active_lbls)]
        if by_lbl[lbl]:
            picked_strat.append(by_lbl[lbl].pop(0))
        idx += 1
        active_lbls = [k for k in MISSING_ELEMENT_LABELS if by_lbl.get(k)]
        if not active_lbls:
            break

    if len(picked_dis) + len(picked_strat) < min(target_total, len(pool)):
        rem_dis = disagreements[len(picked_dis) :]
        extra_needed = min(target_total, len(pool)) - (len(picked_dis) + len(picked_strat))
        picked_strat.extend(rem_dis[:extra_needed])

    combined = picked_dis + picked_strat
    rng.shuffle(combined)

    blinded_rows: list[dict[str, Any]] = []
    key_mapping: list[dict[str, Any]] = []
    for audit_idx, rec in enumerate(combined, start=1):
        audit_id = f"AUDIT_{audit_idx:03d}"
        raw_id = str(rec.get("item_id", f"item_{audit_idx}"))
        raw_pol = str(rec.get("policy", "unknown"))
        blind_hash = hashlib.sha256(f"{seed}:{audit_id}:{raw_id}:{raw_pol}".encode()).hexdigest()[:12]
        blind_code = f"BLIND_{blind_hash}"

        if "turn1_thinking" not in rec or rec.get("turn1_thinking") is None:
            raise SystemExit(
                f"refusing blinded audit export (missing_turn1_thinking): "
                f"item {raw_id!r} is missing required turn1_thinking"
            )
        q2_txt = str(rec.get("q2_answer", rec.get("q1_answer", rec.get("answer", ""))))
        blinded_row = {
            "audit_id": audit_id,
            "blind_source_code": blind_code,
            "thinking": str(rec["turn1_thinking"]),
            "q2_answer": q2_txt,
            "original_turn1_thinking_tokens": int(rec.get("original_turn1_thinking_tokens", 0)),
        }
        assert not any(k in blinded_row for k in _FORBIDDEN_AUDIT_KEYS)
        blinded_rows.append(blinded_row)
        key_mapping.append(
            {
                "audit_id": audit_id,
                "blind_source_code": blind_code,
                "item_id": raw_id,
                "policy": raw_pol,
                "is_disagreement": bool(
                    rec.get("judge_a_missing_element_match") != rec.get("judge_b_missing_element_match")
                    or rec.get("judge_a_reason_type") != rec.get("judge_b_reason_type")
                ),
                "judge_a_missing_element_match": rec.get("judge_a_missing_element_match"),
                "judge_b_missing_element_match": rec.get("judge_b_missing_element_match"),
                "judge_a_reason_type": rec.get("judge_a_reason_type"),
                "judge_b_reason_type": rec.get("judge_b_reason_type"),
            }
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_path = out_dir / "s2-interview-blinded-audit.jsonl"
    key_path = out_dir / "s2-interview-blinded-audit-key.json"
    with audit_path.open("w", encoding="utf-8", newline="\n") as afh:
        afh.write("".join(json.dumps(r) + "\n" for r in blinded_rows))
        afh.flush()
        os.fsync(afh.fileno())
    with key_path.open("w", encoding="utf-8", newline="\n") as kfh:
        key_doc = {"seed": seed, "count": len(key_mapping), "items": key_mapping}
        kfh.write(json.dumps(key_doc, indent=1) + "\n")
        kfh.flush()
        os.fsync(kfh.fileno())
    return {
        "audit_count": len(blinded_rows),
        "disagreements_included": sum(1 for k in key_mapping if k["is_disagreement"]),
        "stratified_agreements_included": sum(1 for k in key_mapping if not k["is_disagreement"]),
        "audit_path": audit_path.name,
        "key_path": key_path.name,
    }


def _prepare_tags_resume(
    tags_path: Path,
    *,
    fsync: Callable[[int], None] = os.fsync,
) -> list[dict[str, Any]]:
    """Read back completed tag rows from ``s2-interview-tags.jsonl`` and truncate partial/duplicate lines."""
    if not tags_path.exists():
        return []
    raw = tags_path.read_bytes()
    if not raw:
        return []
    raw_lines = raw.splitlines(keepends=True)
    completed: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()
    clean_chunks: list[bytes] = []
    modified = False
    for idx, line_bytes in enumerate(raw_lines):
        is_last = idx == len(raw_lines) - 1
        has_nl = line_bytes.endswith(b"\n")
        try:
            text_line = line_bytes.decode("utf-8").strip()
            obj = json.loads(text_line) if text_line else None
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            if is_last:
                modified = True
                break
            raise SystemExit(
                f"refusing to resume interview tagging: corrupt line {idx + 1} in {tags_path}"
            ) from exc
        if not has_nl or not isinstance(obj, dict) or not set(TAG_RECORD_KEYS).issubset(obj.keys()):
            if is_last:
                modified = True
                break
            raise SystemExit(
                f"refusing to resume interview tagging: incomplete record on line {idx + 1} in {tags_path}"
            )
        rk = (str(obj.get("item_id", "")), str(obj.get("policy", "baseline")))
        if rk in seen_keys:
            modified = True
            continue
        seen_keys.add(rk)
        completed.append(obj)
        clean_chunks.append(line_bytes)

    clean_bytes = b"".join(clean_chunks)
    if modified or len(clean_bytes) != len(raw):
        with tags_path.open("r+b") as tfh:
            tfh.seek(0)
            if clean_bytes:
                tfh.write(clean_bytes)
            tfh.truncate(len(clean_bytes))
            tfh.flush()
            fsync(tfh.fileno())
    return completed


# -- Full two-judge tagging pipeline -------------------------------------------------------------------------
def run_two_judge_tagging(
    *,
    interview_q2_records: Sequence[Mapping[str, Any]] | None = None,
    interview_q1_records: Sequence[Mapping[str, Any]] | None = None,
    judge_a_name: str,
    judge_b_name: str,
    judge_a_fn: Callable[[str, str], Mapping[str, Any]],
    judge_b_fn: Callable[[str, str], Mapping[str, Any]],
    out_dir: Path,
    controls: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    min_control_accuracy: float = CONTROL_GATE_MIN_ACCURACY,
    audit_seed: int = 20261105,
    fsync: Callable[[int], None] = os.fsync,
) -> dict[str, Any]:
    """Validate two non-Qwen judges on the 3 control sets (>=90% gate), tag all Q2 records against
    ``turn1_thinking`` (requiring ``turn1_thinking`` with no fallback to Turn 2's thinking),
    compute raw agreement + Cohen's kappa + Gwet's AC1, log thinking-length confound, export
    the 30-item blinded human audit sample, and write hash-only ``s2-interview-tags.jsonl``.
    """
    raw_input = interview_q2_records if interview_q2_records is not None else interview_q1_records
    if raw_input is None:
        raise ValueError("interview_q2_records is required")
    q2_records = extract_interview_q2_records(raw_input)

    for rec in q2_records:
        if "turn1_thinking" not in rec or rec.get("turn1_thinking") is None:
            raise SystemExit(
                f"refusing interview tagging (missing_turn1_thinking): "
                f"record for item {rec.get('item_id')!r} is missing required 'turn1_thinking'; "
                f"fallback to Turn 2 thinking is forbidden"
            )

    ja_name, jb_name = require_two_non_qwen_judges(judge_a_name, judge_b_name)
    ctrl_sets = (
        controls
        if controls is not None
        else build_judge_controls(q2_records, seed=audit_seed)
    )

    ctrl_res_a = evaluate_judge_controls(
        ctrl_sets, judge_a_fn, min_accuracy=min_control_accuracy, enforce_gate=True
    )
    ctrl_res_b = evaluate_judge_controls(
        ctrl_sets, judge_b_fn, min_accuracy=min_control_accuracy, enforce_gate=True
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tags_path = out_dir / "s2-interview-tags.jsonl"
    resumed_tags = _prepare_tags_resume(tags_path, fsync=fsync)
    resumed_by_key: dict[tuple[str, str], dict[str, Any]] = {
        (str(r.get("item_id", "")), str(r.get("policy", "baseline"))): r for r in resumed_tags
    }

    tagged_records: list[dict[str, Any]] = []
    audit_source_records: list[dict[str, Any]] = []
    mem_a: list[str] = []
    mem_b: list[str] = []
    rt_a: list[str] = []
    rt_b: list[str] = []

    with tags_path.open("a", encoding="utf-8", newline="\n") as tfh:
        for rec in q2_records:
            item_key = (str(rec.get("item_id", "")), str(rec.get("policy", "baseline")))
            t1_think = str(rec["turn1_thinking"])
            q2_ans = str(rec.get("q2_answer", rec.get("answer", rec.get("q1_answer", ""))))
            if item_key in resumed_by_key:
                tag_row = resumed_by_key[item_key]
                ma = str(tag_row["judge_a_missing_element_match"])
                mb = str(tag_row["judge_b_missing_element_match"])
                ra = str(tag_row["judge_a_reason_type"])
                rb = str(tag_row["judge_b_reason_type"])
            else:
                tag_a = validate_tag_record(judge_a_fn(t1_think, q2_ans))
                tag_b = validate_tag_record(judge_b_fn(t1_think, q2_ans))
                ma, mb = tag_a["missing_element_match"], tag_b["missing_element_match"]
                ra, rb = tag_a["reason_type"], tag_b["reason_type"]
                consensus_mem = ma if ma == mb else "disagreement"
                consensus_rt = ra if ra == rb else "disagreement"
                tag_row = {
                    "item_id": item_key[0],
                    "policy": item_key[1],
                    "question_id": "Q2",
                    "original_turn1_thinking_tokens": int(rec.get("original_turn1_thinking_tokens", 0)),
                    "turn1_thinking_sha256": hashlib.sha256(t1_think.encode("utf-8")).hexdigest(),
                    "q2_answer_sha256": hashlib.sha256(q2_ans.encode("utf-8")).hexdigest(),
                    "judge_a_missing_element_match": ma,
                    "judge_b_missing_element_match": mb,
                    "consensus_missing_element_match": consensus_mem,
                    "judge_a_reason_type": ra,
                    "judge_b_reason_type": rb,
                    "consensus_reason_type": consensus_rt,
                }
                tfh.write(json.dumps(tag_row) + "\n")
                tfh.flush()
                fsync(tfh.fileno())
                resumed_by_key[item_key] = tag_row

            mem_a.append(ma)
            mem_b.append(mb)
            rt_a.append(ra)
            rt_b.append(rb)
            tagged_records.append(tag_row)
            audit_source_records.append(
                {
                    **tag_row,
                    "turn1_thinking": t1_think,
                    "q2_answer": q2_ans,
                }
            )
        if not q2_records:
            tfh.flush()
            fsync(tfh.fileno())

    agreement_mem = compute_agreement_stats(mem_a, mem_b, categories=MISSING_ELEMENT_LABELS)
    agreement_rt = compute_agreement_stats(rt_a, rt_b, categories=REASON_TYPE_LABELS)
    confound_log = compute_thinking_length_confound(tagged_records)
    audit_meta = export_blinded_human_audit(audit_source_records, out_dir, seed=audit_seed)

    summary = {
        "judge_a": ja_name,
        "judge_b": jb_name,
        "tagged_question_id": "Q2",
        "controls_judge_a": ctrl_res_a,
        "controls_judge_b": ctrl_res_b,
        "agreement_missing_element_match": agreement_mem,
        "agreement_reason_type": agreement_rt,
        "thinking_length_confound": confound_log,
        "blinded_human_audit": audit_meta,
    }
    (out_dir / "s2-interview-tagging-summary.json").write_text(
        json.dumps(summary, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    return summary
