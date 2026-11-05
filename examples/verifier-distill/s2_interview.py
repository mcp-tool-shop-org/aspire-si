"""Post-hoc 'cannot_tell' (CT) Interview Runner for Stage 2 Verifier Distillation.

Implements the pre-registered interview protocol (DESIGN-ct-interview-v2-2026-11-05.md):
- Cohort selection from saved passes (every item with >= 1 CT in starting policy or checkpoint,
  selecting the first CT answer in sample order, recording per-item ct_rate, majority_ct, and cohort tag).
- Turn 1 + Turn 2 chat template construction (verifying Qwen3's chat template strips Turn 1
  <think>...</think>, logging original thinking token length as a confound check, plus Cold Control).
- Fixed question order and exact wording (Q1 -> Q4 -> Q2 -> Q3, with Q3 {A, B} 50/50 counterbalanced).
- Model-card Qwen3 thinking sampling (temp=0.6, top_p=0.95, top_k=20, min_p=0.0) with two-stage thinking cap
  (256 thinking tokens -> force '\\n</think>\\n\\n' and think_truncated=True if unclosed -> 200 answer tokens)
  and 4-gram repetition rate logging.
- Mechanical measures (Part match with whole-claim-copy guard, Quote validity & consistency,
  Counterfactual check from Q2 MISSING_FACT, and Q3 Lean balanced accuracy / gold-CT proportions).
- Adapter swapping on a single loaded base model and keep-all compressed JSONL output with sha256 in receipt.
"""

from __future__ import annotations

import hashlib
import json
import re
import statistics
import sys
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from quote_rule import as_passages, normalise, quote_found  # noqa: E402
from s2_grpo_smoke import (  # noqa: E402
    build_prompt,
    loop_detected,
    parse_reply,
    passages_of,
    split_reply,
    write_save_all,
)

QWEN3_THINK_TOKEN_ID = 151667
QWEN3_END_THINK_TOKEN_ID = 151668
QWEN3_EOS_TOKEN_ID = 151645
QWEN3_PAD_TOKEN_ID = 151643

THINKING_CAP_TOKENS = 256
ANSWER_CAP_TOKENS = 200

INTERVIEW_SAMPLING = {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
}

QUESTION_ORDER = ("Q1", "Q4", "Q2", "Q3")

INTERVIEW_QUESTIONS = {
    "Q1": (
        "Which part of the claim could you not decide from the evidence? "
        "Copy the exact words of that part from the claim."
    ),
    "Q4": (
        "Quote the single most relevant sentence or phrase from the evidence for that part, "
        "or write NONE if the evidence does not touch on it at all."
    ),
    "Q2": (
        "Did the evidence lack information about that part, or did the evidence conflict with itself? "
        "Name the one specific missing fact or conflict, then end with one line: "
        "MISSING_FACT: <one sentence that, if added to the evidence, would decide this part>."
    ),
    "Q3": (
        "If you had to choose between {A} and {B} right now without seeing more evidence, "
        "which would you lean toward, and why in one sentence?"
    ),
}

COLD_CONTROL_QUESTIONS = {
    "Q1": INTERVIEW_QUESTIONS["Q1"],
    "Q2": INTERVIEW_QUESTIONS["Q2"],
}


def interview_seed(item_id: str, question_id: str, base_seed: int = 0) -> int:
    """Deterministic 63-bit seed per (item_id, question_id, base_seed)."""
    digest = hashlib.sha256(f"interview:{base_seed}:{item_id}:{question_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFF_FFFF_FFFF_FFFF


def q3_choice_order(item_id: str) -> tuple[str, str]:
    """Counterbalance {A, B} in Q3 50/50 by item_id sha256 parity."""
    digest = hashlib.sha256(str(item_id).encode()).digest()
    if digest[0] % 2 == 0:
        return ("supported", "unsupported")
    return ("unsupported", "supported")


def format_interview_question(question_id: str, item_id: str, *, cold_control: bool = False) -> str:
    """Return the exact pre-registered question wording for ``question_id``."""
    if cold_control:
        if question_id not in COLD_CONTROL_QUESTIONS:
            raise ValueError(f"Cold control only supports Q1 and Q2, got {question_id!r}")
        return COLD_CONTROL_QUESTIONS[question_id]
    if question_id not in INTERVIEW_QUESTIONS:
        raise ValueError(f"Unknown interview question_id {question_id!r}")
    if question_id == "Q3":
        choice_a, choice_b = q3_choice_order(item_id)
        return INTERVIEW_QUESTIONS["Q3"].format(A=choice_a, B=choice_b)
    return INTERVIEW_QUESTIONS[question_id]


# -- Cohort selection from saved passes ----------------------------------------------------------------------
def _group_pass_completions_by_item(pass_records: Sequence[Mapping[str, Any]]) -> dict[str, list[dict]]:
    by_item: dict[str, list[dict]] = {}
    for rec in pass_records:
        cid = str(rec["item_id"])
        by_item.setdefault(cid, []).append(dict(rec))
    for cid in by_item:
        by_item[cid].sort(key=lambda r: int(r.get("sample_index", 0)))
    return by_item


def _first_ct_completion(samples: Sequence[Mapping[str, Any]]) -> dict | None:
    """Return the first completion in sample order whose verdict is 'cannot_tell', or None."""
    for s in samples:
        if s.get("verdict") == "cannot_tell":
            return dict(s)
    return None


def select_interview_cohort(
    baseline_pass_records: Sequence[Mapping[str, Any]],
    checkpoint_pass_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Select every item with >= 1 CT completion in starting policy (logging pass seed 1) or checkpoint
    (post-pass seed 1). Selects the first CT answer in sample order and records ct_rate, majority_ct,
    and cohort tag ('both_CT', 'baseline_only_CT', 'checkpoint_only_CT').
    """
    base_by_item = _group_pass_completions_by_item(baseline_pass_records)
    ckpt_by_item = _group_pass_completions_by_item(checkpoint_pass_records)
    all_ids = sorted(set(base_by_item.keys()) | set(ckpt_by_item.keys()))

    cohort: list[dict[str, Any]] = []
    for cid in all_ids:
        b_samples = base_by_item.get(cid, [])
        c_samples = ckpt_by_item.get(cid, [])
        b_k_ct = sum(1 for s in b_samples if s.get("verdict") == "cannot_tell")
        c_k_ct = sum(1 for s in c_samples if s.get("verdict") == "cannot_tell")
        if b_k_ct == 0 and c_k_ct == 0:
            continue

        b_total = len(b_samples) or 1
        c_total = len(c_samples) or 1
        b_first = _first_ct_completion(b_samples)
        c_first = _first_ct_completion(c_samples)

        if b_k_ct > 0 and c_k_ct > 0:
            tag = "both_CT"
        elif b_k_ct > 0:
            tag = "baseline_only_CT"
        else:
            tag = "checkpoint_only_CT"

        cohort.append(
            {
                "item_id": cid,
                "cohort": tag,
                "baseline_k_ct": b_k_ct,
                "baseline_g": len(b_samples),
                "baseline_ct_rate": round(b_k_ct / float(b_total), 6),
                "baseline_majority_ct": bool(len(b_samples) > 0 and 2 * b_k_ct >= b_total),
                "baseline_first_ct": b_first,
                "checkpoint_k_ct": c_k_ct,
                "checkpoint_g": len(c_samples),
                "checkpoint_ct_rate": round(c_k_ct / float(c_total), 6),
                "checkpoint_majority_ct": bool(len(c_samples) > 0 and 2 * c_k_ct >= c_total),
                "checkpoint_first_ct": c_first,
            }
        )
    return cohort


def summarize_cohort_resolution(
    cohort: Sequence[Mapping[str, Any]],
    *,
    checkpoint_verdicts_by_item: Mapping[str, str] | None = None,
    baseline_q3_lean_by_item: Mapping[str, str | None] | None = None,
    both_ct_paired_outcomes: Mapping[str, tuple[bool, bool]] | None = None,
) -> dict[str, Any]:
    """Cross-tab resolved ('baseline_only_CT') items against starting policy's Q3 lean and compute
    paired McNemar +/- counts on 'both_CT' items.
    """
    counts = {"both_CT": 0, "baseline_only_CT": 0, "checkpoint_only_CT": 0}
    cross_tab: dict[str, dict[str, int]] = {
        "supported": {"supported": 0, "unsupported": 0, "other": 0},
        "unsupported": {"supported": 0, "unsupported": 0, "other": 0},
        "unparsed": {"supported": 0, "unsupported": 0, "other": 0},
    }
    for item in cohort:
        cid = str(item["item_id"])
        tag = str(item["cohort"])
        if tag in counts:
            counts[tag] += 1
        if tag == "baseline_only_CT" and checkpoint_verdicts_by_item is not None:
            lean = (
                baseline_q3_lean_by_item.get(cid)
                if baseline_q3_lean_by_item is not None
                else None
            )
            lean_key = lean if lean in ("supported", "unsupported") else "unparsed"
            ckpt_v = checkpoint_verdicts_by_item.get(cid, "other")
            v_key = ckpt_v if ckpt_v in ("supported", "unsupported") else "other"
            cross_tab[lean_key][v_key] += 1

    # Paired McNemar counts on both_CT:
    # b = baseline True & checkpoint False; c = baseline False & checkpoint True
    mcnemar_b = 0
    mcnemar_c = 0
    mcnemar_both_true = 0
    mcnemar_both_false = 0
    if both_ct_paired_outcomes is not None:
        for item in cohort:
            if item["cohort"] != "both_CT":
                continue
            cid = str(item["item_id"])
            if cid not in both_ct_paired_outcomes:
                continue
            b_ok, c_ok = both_ct_paired_outcomes[cid]
            if b_ok and c_ok:
                mcnemar_both_true += 1
            elif b_ok and not c_ok:
                mcnemar_b += 1
            elif not b_ok and c_ok:
                mcnemar_c += 1
            else:
                mcnemar_both_false += 1

    return {
        "cohort_counts": counts,
        "resolved_cross_tab_lean_vs_checkpoint": cross_tab,
        "both_ct_mcnemar": {
            "both_true": mcnemar_both_true,
            "baseline_only_true_minus": mcnemar_b,
            "checkpoint_only_true_plus": mcnemar_c,
            "both_false": mcnemar_both_false,
        },
    }


# -- Turn 1 + Turn 2 chat template construction --------------------------------------------------------------
def build_interview_messages(
    row: Mapping[str, Any],
    turn1_completion_text: str,
    question_id: str,
) -> list[dict[str, str]]:
    """Build [user_1, assistant_1, user_2] for Turn 2 probing."""
    cid = str(row["id"])
    user_1 = build_prompt(dict(row))
    user_2 = format_interview_question(question_id, cid, cold_control=False)
    return [
        {"role": "user", "content": user_1},
        {"role": "assistant", "content": turn1_completion_text},
        {"role": "user", "content": user_2},
    ]


def build_cold_control_messages(
    row: Mapping[str, Any],
    question_id: str,
) -> list[dict[str, str]]:
    """Build [user_cold] with task + evidence + claim + cold question (no Turn 1 JSON contract or answer)."""
    cid = str(row["id"])
    passages = passages_of(dict(row))
    if isinstance(passages, Sequence) and not isinstance(passages, str):
        evidence_text = "\n".join(str(p) for p in passages)
    else:
        evidence_text = str(passages)
    claim_text = str(row.get("claim", ""))
    cold_q = format_interview_question(question_id, cid, cold_control=True)
    user_cold = (
        "Read the evidence and the claim below, then answer the question.\n\n"
        f"Evidence:\n{evidence_text}\n\n"
        f"Claim: {claim_text}\n\n"
        f"Question: {cold_q}"
    )
    return [{"role": "user", "content": user_cold}]


def render_interview_prompt(
    tok,
    messages: Sequence[Mapping[str, str]],
) -> tuple[str, list[int]]:
    """Render ``messages`` via Qwen3's ``apply_chat_template(..., enable_thinking=True)``."""
    rendered_text = tok.apply_chat_template(
        list(messages),
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=True,
    )
    prompt_ids = tok.encode(rendered_text)
    return rendered_text, list(prompt_ids)


def verify_turn1_thinking_stripped(
    rendered_prompt_text: str,
    rendered_prompt_ids: Sequence[int],
    original_thinking: str,
) -> bool:
    """Assert that Turn 1's ``<think>...</think>`` block was stripped from the rendered Turn 2 prompt."""
    # In Qwen3 chat template with enable_thinking=True and add_generation_prompt=True,
    # the generation prompt at the very end may open `<think>\n` for Turn 2, OR not contain `</think>` at all.
    # Specifically, `</think>` (token 151668) must NEVER appear in the rendered prompt before generation,
    # and the non-empty Turn 1 thinking text must not appear in the rendered prompt.
    if QWEN3_END_THINK_TOKEN_ID in rendered_prompt_ids or "</think>" in rendered_prompt_text:
        raise AssertionError("Turn 1 </think> tag leaked into rendered Turn 2 prompt")
    clean_think = original_thinking.strip()
    if clean_think and len(clean_think) >= 8 and clean_think in rendered_prompt_text:
        raise AssertionError("Turn 1 thinking text leaked into rendered Turn 2 prompt")
    return True


# -- Two-stage thinking generation & n-gram repetition -------------------------------------------------------
def compute_ngram_repetition_rate(text: str, n: int = 4) -> float:
    """Return word n-gram repetition rate in ``text``: ``1 - unique_ngrams / total_ngrams``."""
    words = text.lower().split()
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    return round(1.0 - (len(set(ngrams)) / float(len(ngrams))), 6)


def generate_two_stage_thinking(
    model,
    tok,
    prompt_ids: Sequence[int],
    *,
    seed: int,
    thinking_cap: int = THINKING_CAP_TOKENS,
    answer_cap: int = ANSWER_CAP_TOKENS,
    sampling: Mapping[str, Any] = INTERVIEW_SAMPLING,
) -> dict[str, Any]:
    """Generate a Turn 2 interview response with the two-stage thinking cap (256 + 200 tokens).

    Stage 1 generates up to ``thinking_cap`` tokens. If ``</think>`` is not produced within ``thinking_cap``
    tokens, appends ``\\n</think>\\n\\n``, marks ``think_truncated = True``, and runs Stage 2 for up to
    ``answer_cap`` answer tokens.
    """
    import torch

    torch.manual_seed(int(seed))
    device = getattr(model, "device", torch.device("cpu"))
    input_tensor = torch.tensor([list(prompt_ids)], dtype=torch.long, device=device)
    prompt_len = int(input_tensor.shape[1])

    eos_id = getattr(tok, "eos_token_id", QWEN3_EOS_TOKEN_ID)
    stage1_eos_ids = (
        [int(eos_id), QWEN3_END_THINK_TOKEN_ID]
        if int(eos_id) != QWEN3_END_THINK_TOKEN_ID
        else [QWEN3_END_THINK_TOKEN_ID]
    )
    gen_kwargs_stage1: dict[str, Any] = {
        "do_sample": True,
        "temperature": float(sampling.get("temperature", 0.6)),
        "top_p": float(sampling.get("top_p", 0.95)),
        "top_k": int(sampling.get("top_k", 20)),
        "pad_token_id": getattr(tok, "pad_token_id", QWEN3_PAD_TOKEN_ID),
        "eos_token_id": stage1_eos_ids,
    }
    gen_kwargs_stage2: dict[str, Any] = {
        **gen_kwargs_stage1,
        "eos_token_id": eos_id,
    }

    with torch.no_grad():
        stage1_out = model.generate(
            input_tensor,
            max_new_tokens=int(thinking_cap),
            attention_mask=torch.ones_like(input_tensor),
            **gen_kwargs_stage1,
        )
    stage1_seqs = stage1_out.sequences if hasattr(stage1_out, "sequences") else stage1_out
    stage1_new_ids = [int(x) for x in stage1_seqs[0, prompt_len:].tolist()]
    stage1_text = tok.decode(stage1_new_ids, skip_special_tokens=False)

    closed_in_stage1 = (
        QWEN3_END_THINK_TOKEN_ID in stage1_new_ids or "</think>" in stage1_text
    )

    if not closed_in_stage1:
        think_truncated = True
        close_ids = list(tok.encode("\n</think>\n\n"))
        stage2_prefix_ids = list(prompt_ids) + stage1_new_ids + close_ids
        stage2_input = torch.tensor([stage2_prefix_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            stage2_out = model.generate(
                stage2_input,
                max_new_tokens=int(answer_cap),
                attention_mask=torch.ones_like(stage2_input),
                **gen_kwargs_stage2,
            )
        stage2_seqs = stage2_out.sequences if hasattr(stage2_out, "sequences") else stage2_out
        answer_new_ids = [int(x) for x in stage2_seqs[0, len(stage2_prefix_ids) :].tolist()][
            : int(answer_cap)
        ]
        full_new_ids = stage1_new_ids + close_ids + answer_new_ids
        thinking_tok_count = len(stage1_new_ids)
        answer_tok_count = len(answer_new_ids)
    else:
        think_truncated = False
        if QWEN3_END_THINK_TOKEN_ID in stage1_new_ids:
            end_idx = stage1_new_ids.index(QWEN3_END_THINK_TOKEN_ID)
            thinking_ids_part = stage1_new_ids[:end_idx]
            after_think_ids = stage1_new_ids[end_idx + 1 : end_idx + 1 + int(answer_cap)]
            stage1_prefix_through_think = stage1_new_ids[: end_idx + 1]
        else:
            think_str, ans_str = split_reply(stage1_text)
            thinking_ids_part = list(tok.encode(think_str)) if think_str else []
            after_think_ids = (list(tok.encode(ans_str)) if ans_str else [])[: int(answer_cap)]
            stage1_prefix_through_think = thinking_ids_part + [QWEN3_END_THINK_TOKEN_ID]

        hit_eos = eos_id in after_think_ids
        if not hit_eos and len(after_think_ids) < int(answer_cap):
            rem_answer = int(answer_cap) - len(after_think_ids)
            stage2_prefix_ids = list(prompt_ids) + stage1_prefix_through_think + after_think_ids
            stage2_input = torch.tensor([stage2_prefix_ids], dtype=torch.long, device=device)
            with torch.no_grad():
                stage2_out = model.generate(
                    stage2_input,
                    max_new_tokens=rem_answer,
                    attention_mask=torch.ones_like(stage2_input),
                    **gen_kwargs_stage2,
                )
            stage2_seqs = stage2_out.sequences if hasattr(stage2_out, "sequences") else stage2_out
            extra_ans_ids = [int(x) for x in stage2_seqs[0, len(stage2_prefix_ids) :].tolist()][
                :rem_answer
            ]
            after_think_ids = (after_think_ids + extra_ans_ids)[: int(answer_cap)]
        full_new_ids = stage1_prefix_through_think + after_think_ids

        thinking_tok_count = len(thinking_ids_part)
        answer_tok_count = len(after_think_ids)

    full_text = tok.decode(full_new_ids, skip_special_tokens=True)
    thinking_text, answer_text = split_reply(full_text)
    answer_clean = answer_text.strip()
    return {
        "seed": int(seed),
        "text": full_text,
        "thinking": thinking_text,
        "answer": answer_clean,
        "thinking_tokens": int(thinking_tok_count),
        "answer_tokens": int(answer_tok_count),
        "think_truncated": bool(think_truncated),
        "ngram_4_rep_rate": compute_ngram_repetition_rate(full_text, n=4),
    }


# -- Mechanical measures (no judge model) --------------------------------------------------------------------
def _normalised_tokens(text: str) -> list[str]:
    norm = normalise(text)
    return [tok for tok in norm.split() if tok]


def _token_f1_and_containment(q_text: str, ref_text: str) -> tuple[float, float]:
    """Return ``(token_f1, containment)`` where containment is ``|Q1 ∩ JSON| / |Q1|``."""
    q_toks = _normalised_tokens(q_text)
    r_toks = _normalised_tokens(ref_text)
    if not q_toks or not r_toks:
        return 0.0, 0.0
    q_set = set(q_toks)
    r_set = set(r_toks)
    inter = len(q_set & r_set)
    if inter == 0:
        return 0.0, 0.0
    containment = inter / float(len(q_set))
    recall = inter / float(len(r_set))
    f1 = (2.0 * containment * recall) / (containment + recall)
    return round(f1, 6), round(containment, 6)


def _parse_turn_json(text: str) -> dict | None:
    """Parse Turn 1 / counterfactual JSON reply whether or not it includes a ``</think>`` prefix."""
    parsed = parse_reply(text)
    if parsed is None and "</think>" not in text:
        parsed = parse_reply(f"</think>{text}")
    return parsed


class _CtPartsList(list):
    """List of extracted part strings carrying ``ct_part_fallback`` metadata."""

    ct_part_fallback: bool = False


def extract_ct_parts_from_json(turn1_text: str) -> list[str]:
    """Extract the ``part`` strings marked ``cannot_tell`` (or all parts with ``ct_part_fallback=True``)."""
    parsed = _parse_turn_json(turn1_text)
    if parsed is None:
        empty = _CtPartsList()
        empty.ct_part_fallback = True
        return empty
    parts = parsed.get("evidence_quotes") or []
    ct_parts = _CtPartsList(
        str(p.get("part", "")).strip()
        for p in parts
        if isinstance(p, Mapping) and p.get("verdict") == "cannot_tell" and str(p.get("part", "")).strip()
    )
    if ct_parts:
        ct_parts.ct_part_fallback = False
        return ct_parts
    fallback_parts = _CtPartsList(
        str(p.get("part", "")).strip()
        for p in parts
        if isinstance(p, Mapping) and str(p.get("part", "")).strip()
    )
    fallback_parts.ct_part_fallback = True
    return fallback_parts


def score_part_match(
    q1_answer: str,
    json_ct_parts: Sequence[str],
    claim_text: str,
    *,
    ct_part_fallback: bool | None = None,
) -> dict[str, Any]:
    """Score Q1's named claim part against the original JSON's ``cannot_tell`` part(s).

    Reports:
    - ``token_f1`` and ``containment`` (``|Q1 ∩ JSON| / |Q1|``), best match across ``json_ct_parts``.
    - ``unspecific_full_claim_copy`` (and ``whole_claim_copy`` alias): True when Q1 covers > 85% of claim
      tokens while JSON part covers < 70%.
    - ``ct_part_fallback``: True when no ``evidence_quotes`` entry had ``verdict == "cannot_tell"``.
    - Separate threshold flags for ``token_f1`` and ``containment`` (plus ``max_overlap``) at ``0.3``,
      ``0.5``, ``0.7`` (gated off when ``unspecific_full_claim_copy`` is True).
    """
    claim_toks = set(_normalised_tokens(claim_text))
    q1_toks = set(_normalised_tokens(q1_answer))
    is_fallback = (
        bool(getattr(json_ct_parts, "ct_part_fallback", False))
        if ct_part_fallback is None
        else bool(ct_part_fallback)
    )

    best_f1 = 0.0
    best_containment = 0.0
    best_part = ""
    for part_str in json_ct_parts:
        f1, cont = _token_f1_and_containment(q1_answer, part_str)
        if (f1, cont) >= (best_f1, best_containment):
            best_f1 = f1
            best_containment = cont
            best_part = part_str

    part_toks = set(_normalised_tokens(best_part)) if best_part else set()
    q1_claim_cov = (len(q1_toks & claim_toks) / float(len(claim_toks))) if claim_toks else 0.0
    part_claim_cov = (len(part_toks & claim_toks) / float(len(claim_toks))) if claim_toks else 0.0
    unspecific_full_claim_copy = bool(claim_toks and q1_claim_cov > 0.85 and part_claim_cov < 0.70)

    eff_f1 = 0.0 if unspecific_full_claim_copy else best_f1
    eff_cont = 0.0 if unspecific_full_claim_copy else best_containment
    eff_score = max(eff_f1, eff_cont)
    return {
        "token_f1": best_f1,
        "containment": best_containment,
        "max_overlap": round(max(best_f1, best_containment), 6),
        "q1_claim_coverage": round(q1_claim_cov, 6),
        "json_part_claim_coverage": round(part_claim_cov, 6),
        "unspecific_full_claim_copy": unspecific_full_claim_copy,
        "whole_claim_copy": unspecific_full_claim_copy,
        "ct_part_fallback": is_fallback,
        "f1_at_0_3": bool(not unspecific_full_claim_copy and eff_f1 >= 0.3),
        "f1_at_0_5": bool(not unspecific_full_claim_copy and eff_f1 >= 0.5),
        "f1_at_0_7": bool(not unspecific_full_claim_copy and eff_f1 >= 0.7),
        "containment_at_0_3": bool(not unspecific_full_claim_copy and eff_cont >= 0.3),
        "containment_at_0_5": bool(not unspecific_full_claim_copy and eff_cont >= 0.5),
        "containment_at_0_7": bool(not unspecific_full_claim_copy and eff_cont >= 0.7),
        "match_at_0_3": bool(not unspecific_full_claim_copy and eff_score >= 0.3),
        "match_at_0_5": bool(not unspecific_full_claim_copy and eff_score >= 0.5),
        "match_at_0_7": bool(not unspecific_full_claim_copy and eff_score >= 0.7),
    }


def score_quote_validity_and_consistency(
    q4_answer: str,
    evidence: str | Sequence[str | dict],
    turn1_text: str,
) -> dict[str, Any]:
    """Score Q4 quote validity (NONE or valid substring of evidence under S2 quote rule) and
    quote consistency (token overlap between Q4 and original JSON quotes).
    """
    ans_clean = q4_answer.strip().strip('"').strip("'").rstrip(".").strip()
    is_none = ans_clean.upper() == "NONE"
    passages = as_passages(evidence)
    in_evidence = False if is_none else bool(quote_found(ans_clean, passages))
    valid = bool(is_none or in_evidence)

    parsed = _parse_turn_json(turn1_text)
    json_quotes = [
        str(p.get("quote", "")).strip()
        for p in (parsed.get("evidence_quotes") or [] if parsed else [])
        if isinstance(p, Mapping) and str(p.get("quote", "")).strip()
    ]
    best_f1 = 0.0
    best_cont = 0.0
    if not is_none and json_quotes:
        for jq in json_quotes:
            f1, cont = _token_f1_and_containment(ans_clean, jq)
            if (cont, f1) > (best_cont, best_f1):
                best_f1, best_cont = f1, cont
    elif is_none and not json_quotes:
        best_f1 = best_cont = 1.0

    return {
        "is_none": is_none,
        "in_evidence": in_evidence,
        "quote_valid": valid,
        "quote_consistency_f1": best_f1,
        "quote_consistency_containment": best_cont,
    }


_MISSING_FACT_RE = re.compile(r"MISSING_FACT\s*:\s*(.+)", re.IGNORECASE)


def extract_missing_fact(q2_answer: str) -> str | None:
    """Extract ``<one sentence>`` from the final ``MISSING_FACT: <one sentence>`` line in Q2's answer."""
    for line in reversed(q2_answer.splitlines()):
        m = _MISSING_FACT_RE.search(line.strip())
        if m:
            fact = m.group(1).strip()
            if fact:
                return fact
    return None


def append_counterfactual_fact(row: Mapping[str, Any], missing_fact: str) -> dict[str, Any]:
    """Return a shallow copy of ``row`` with ``missing_fact`` appended to its evidence context."""
    new_row = dict(row)
    ctx = [dict(c) for c in (row.get("context") or [])]
    ctx.append({"source": "counterfactual_missing_fact", "text": str(missing_fact).strip()})
    new_row["context"] = ctx
    return new_row


def run_counterfactual_check(
    row: Mapping[str, Any],
    q2_answer: str,
    verifier_runner: Callable[[dict[str, Any], int], str],
    *,
    base_seed: int = 0,
) -> dict[str, Any]:
    """Extract ``MISSING_FACT: <one sentence>`` from Q2, append it to the item's evidence, rerun the normal
    verification task once with fixed seed, and record whether the verdict leaves ``cannot_tell``.
    """
    fact = extract_missing_fact(q2_answer)
    cid = str(row["id"])
    cf_seed = interview_seed(cid, "CF_VERIFY", base_seed=base_seed)
    if not fact:
        return {
            "format_ok": False,
            "missing_fact": None,
            "seed": cf_seed,
            "reply_text": None,
            "new_verdict": None,
            "left_cannot_tell": False,
            "direction": "unparsed_q2",
        }
    cf_row = append_counterfactual_fact(row, fact)
    reply_text = verifier_runner(cf_row, cf_seed)
    parsed = _parse_turn_json(reply_text)
    if parsed is None:
        new_verdict = None
        direction = "malformed"
        left_ct = False
    else:
        new_verdict = "cannot_tell" if loop_detected(reply_text) else str(parsed["verdict"])
        direction = new_verdict
        left_ct = new_verdict in ("supported", "unsupported")
    return {
        "format_ok": True,
        "missing_fact": fact,
        "seed": cf_seed,
        "reply_text": reply_text,
        "new_verdict": new_verdict,
        "left_cannot_tell": bool(left_ct),
        "direction": direction,
    }


_LEAN_RE = re.compile(
    r"LEAN\s*:\s*(supported|unsupported)\b(?:\s*(?:—|--|-|:)\s*(.+))?",
    re.IGNORECASE,
)
_PROSE_LEAN_RE = re.compile(
    r"(?:lean\s+(?:toward|towards)|choose|pick|bet\s+on|favor|favour|select)\s+[`'\"*]*(supported|unsupported)\b",
    re.IGNORECASE,
)
_WORD_LEAN_RE = re.compile(r"\b(supported|unsupported)\b", re.IGNORECASE)


def parse_q3_lean(q3_answer: str) -> dict[str, Any]:
    """Parse ``supported`` or ``unsupported`` lean from Q3's answer (either ``LEAN:`` or natural prose)."""
    for line in q3_answer.splitlines():
        m = _LEAN_RE.search(line.strip())
        if m:
            choice = m.group(1).strip().lower()
            rationale = (m.group(2) or "").strip()
            return {
                "format_ok": True,
                "lean": choice,
                "rationale": rationale,
            }
    m_prose = _PROSE_LEAN_RE.search(q3_answer)
    if m_prose:
        return {
            "format_ok": True,
            "lean": m_prose.group(1).strip().lower(),
            "rationale": q3_answer.strip(),
        }
    m_word = _WORD_LEAN_RE.search(q3_answer)
    if m_word:
        return {
            "format_ok": True,
            "lean": m_word.group(1).strip().lower(),
            "rationale": q3_answer.strip(),
        }
    return {"format_ok": False, "lean": None, "rationale": ""}


def score_q3_lean_item(q3_answer: str, gold_verdict: str, item_id: str) -> dict[str, Any]:
    """Score Q3 forced lean on a single item.

    On decidable items (gold supported or unsupported), scores whether the forced choice matches gold.
    On gold cannot_tell items, records the choice without scoring accuracy (``accuracy_scored = False``).
    """
    parsed = parse_q3_lean(q3_answer)
    lean = parsed["lean"]
    gold = str(gold_verdict)
    order = q3_choice_order(item_id)
    if gold in ("supported", "unsupported"):
        correct = bool(lean == gold) if lean is not None else False
        return {
            **parsed,
            "choice_order": list(order),
            "gold_verdict": gold,
            "accuracy_scored": True,
            "correct": correct,
        }
    return {
        **parsed,
        "choice_order": list(order),
        "gold_verdict": gold,
        "accuracy_scored": False,
        "correct": None,
    }


def summarize_q3_lean(lean_records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute balanced accuracy on decidable items (gold supported & unsupported, counting unparsed
    answers as wrong in the recall denominators) and choice proportions on gold cannot_tell items.
    """
    sup_recs = [r for r in lean_records if r.get("gold_verdict") == "supported"]
    unsup_recs = [r for r in lean_records if r.get("gold_verdict") == "unsupported"]
    ct_recs = [r for r in lean_records if r.get("gold_verdict") == "cannot_tell"]
    unparsed_total = sum(1 for r in lean_records if r.get("lean") is None)
    sup_unparsed = sum(1 for r in sup_recs if r.get("lean") is None)
    unsup_unparsed = sum(1 for r in unsup_recs if r.get("lean") is None)
    ct_unparsed = sum(1 for r in ct_recs if r.get("lean") is None)

    sup_acc = (
        sum(1 for r in sup_recs if r.get("lean") == "supported") / float(len(sup_recs))
        if sup_recs
        else None
    )
    unsup_acc = (
        sum(1 for r in unsup_recs if r.get("lean") == "unsupported") / float(len(unsup_recs))
        if unsup_recs
        else None
    )
    if sup_acc is not None and unsup_acc is not None:
        bal_acc: float | None = round(0.5 * (sup_acc + unsup_acc), 6)
    elif sup_acc is not None:
        bal_acc = round(sup_acc, 6)
    elif unsup_acc is not None:
        bal_acc = round(unsup_acc, 6)
    else:
        bal_acc = None

    ct_sup_n = sum(1 for r in ct_recs if r.get("lean") == "supported")
    ct_unsup_n = sum(1 for r in ct_recs if r.get("lean") == "unsupported")
    ct_total = len(ct_recs)
    return {
        "unparsed_count": unparsed_total,
        "decidable_supported_n": len(sup_recs),
        "decidable_supported_unparsed": sup_unparsed,
        "decidable_supported_acc": round(sup_acc, 6) if sup_acc is not None else None,
        "decidable_unsupported_n": len(unsup_recs),
        "decidable_unsupported_unparsed": unsup_unparsed,
        "decidable_unsupported_acc": round(unsup_acc, 6) if unsup_acc is not None else None,
        "balanced_accuracy": bal_acc,
        "gold_cannot_tell": {
            "n": ct_total,
            "unparsed_count": ct_unparsed,
            "accuracy_scored": False,
            "supported_share": round(ct_sup_n / float(ct_total), 6) if ct_total > 0 else None,
            "unsupported_share": round(ct_unsup_n / float(ct_total), 6) if ct_total > 0 else None,
            "unparsed_share": round(ct_unparsed / float(ct_total), 6) if ct_total > 0 else None,
        },
    }


# -- Adapter swapping on a single loaded base model ----------------------------------------------------------
class InterviewModelRunner:
    """Runs starting policy (adapter disabled), checkpoint (LoRA adapter enabled), and cold control on one
    loaded base model without reloading weights.
    """

    def __init__(self, model, tok) -> None:
        self.model = model
        self.tok = tok
        self.active_policy = "baseline"

    def _policy_context(self, policy: str):
        eff_policy = policy.removeprefix("cold_control:") if policy.startswith("cold_control:") else policy
        if eff_policy in ("baseline", "cold_control"):
            disable_fn = getattr(self.model, "disable_adapter", None)
            if callable(disable_fn):
                return disable_fn()
            return nullcontext()
        if eff_policy.startswith("checkpoint"):
            set_adapter_fn = getattr(self.model, "set_adapter", None)
            if callable(set_adapter_fn) and ":" in eff_policy:
                _, _, adapter_name = eff_policy.partition(":")
                if adapter_name:
                    set_adapter_fn(adapter_name)
            return nullcontext()
        raise ValueError(f"Unknown interview policy {policy!r}")

    def ask_question(
        self,
        *,
        row: Mapping[str, Any],
        turn1_completion_text: str | None,
        question_id: str,
        policy: str = "baseline",
        cold_control: bool = False,
        base_seed: int = 0,
        thinking_cap: int = THINKING_CAP_TOKENS,
        answer_cap: int = ANSWER_CAP_TOKENS,
    ) -> dict[str, Any]:
        cid = str(row["id"])
        cold = bool(cold_control or policy == "cold_control" or policy.startswith("cold_control:"))
        if cold:
            messages = build_cold_control_messages(row, question_id)
            orig_think: str | None = None
            orig_think_tokens = 0
        else:
            if turn1_completion_text is None:
                raise ValueError("turn1_completion_text is required when policy is not 'cold_control'")
            orig_think_raw, _ = split_reply(turn1_completion_text)
            orig_think = orig_think_raw or ""
            orig_think_tokens = len(self.tok.encode(orig_think)) if orig_think else 0
            messages = build_interview_messages(row, turn1_completion_text, question_id)

        rendered_text, prompt_ids = render_interview_prompt(self.tok, messages)
        if not cold and turn1_completion_text is not None:
            verify_turn1_thinking_stripped(rendered_text, prompt_ids, orig_think)

        q_seed = interview_seed(cid, question_id, base_seed=base_seed)
        with self._policy_context(policy):
            gen_res = generate_two_stage_thinking(
                self.model,
                self.tok,
                prompt_ids,
                seed=q_seed,
                thinking_cap=thinking_cap,
                answer_cap=answer_cap,
            )
        return {
            "item_id": cid,
            "policy": policy,
            "question_id": question_id,
            "cold_control": cold,
            "prompt_text": rendered_text,
            "prompt_tokens": len(prompt_ids),
            "turn1_thinking": orig_think,
            "original_turn1_thinking_tokens": int(orig_think_tokens),
            **gen_res,
        }

    def run_verifier(
        self,
        row: Mapping[str, Any],
        seed: int,
        *,
        policy: str = "baseline",
        thinking_cap: int = THINKING_CAP_TOKENS,
        answer_cap: int = ANSWER_CAP_TOKENS,
    ) -> str:
        """Run the normal verification task under ``policy`` with ``seed`` and return raw text."""
        messages = [{"role": "user", "content": build_prompt(dict(row))}]
        _, prompt_ids = render_interview_prompt(self.tok, messages)
        with self._policy_context(policy):
            gen_res = generate_two_stage_thinking(
                self.model,
                self.tok,
                prompt_ids,
                seed=seed,
                thinking_cap=thinking_cap,
                answer_cap=answer_cap,
            )
        return str(gen_res["raw_text"])


# -- CPU Qwen3 stub for fast offline unit testing ------------------------------------------------------------
_THINK_TAG_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


class Qwen3ChatStubTokenizer:
    """Offline tokenizer implementing Qwen3's chat template (stripping <think>...</think> from prior
    assistant turns) and special token IDs (<think>=151667, </think>=151668, eos=151645, pad=151643).
    """

    pad_token_id = QWEN3_PAD_TOKEN_ID
    eos_token_id = QWEN3_EOS_TOKEN_ID
    vocab_size = 152000

    def apply_chat_template(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        add_generation_prompt: bool = True,
        tokenize: bool = False,
        enable_thinking: bool = True,
    ) -> str | list[int]:
        parts: list[str] = []
        for idx, msg in enumerate(messages):
            role = str(msg["role"])
            content = str(msg["content"])
            if role == "assistant" and idx < len(messages) - 1:
                content = _THINK_TAG_RE.sub("", content).strip()
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        if add_generation_prompt:
            if enable_thinking:
                parts.append("<|im_start|>assistant\n<think>\n")
            else:
                parts.append("<|im_start|>assistant\n")
        rendered = "\n".join(parts)
        return self.encode(rendered) if tokenize else rendered

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        pos = 0
        while pos < len(text):
            if text.startswith("<think>", pos):
                out.append(QWEN3_THINK_TOKEN_ID)
                pos += len("<think>")
            elif text.startswith("</think>", pos):
                out.append(QWEN3_END_THINK_TOKEN_ID)
                pos += len("</think>")
            else:
                out.append((ord(text[pos]) % 250) + 10)
                pos += 1
        return out

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        chars: list[str] = []
        for tok_id in ids:
            tid = int(tok_id)
            if tid in (self.pad_token_id, self.eos_token_id) and skip_special_tokens:
                continue
            if tid == QWEN3_THINK_TOKEN_ID:
                chars.append("<think>")
            elif tid == QWEN3_END_THINK_TOKEN_ID:
                chars.append("</think>")
            elif 10 <= tid < 260:
                chars.append(chr(tid - 10))
        return "".join(chars)


def build_cpu_qwen3_stub():
    """Build a tiny in-memory ``Qwen3ForCausalLM`` on CPU with ``Qwen3ChatStubTokenizer`` (no download)."""
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    tok = Qwen3ChatStubTokenizer()
    cfg = Qwen3Config(
        vocab_size=tok.vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=2048,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )
    model = Qwen3ForCausalLM(cfg)
    model.eval()
    return model, tok


# -- End-to-end interview runner & keep-all archive ----------------------------------------------------------
def run_interview_for_cohort(
    *,
    cohort: Sequence[Mapping[str, Any]],
    rows_by_id: Mapping[str, Mapping[str, Any]],
    runner: InterviewModelRunner,
    out_dir: Path,
    base_seed: int = 0,
    thinking_cap: int = THINKING_CAP_TOKENS,
    answer_cap: int = ANSWER_CAP_TOKENS,
    verifier_runner: Callable[[dict[str, Any], int], str] | None = None,
    include_cold_control: bool = True,
) -> dict[str, Any]:
    """Run Q1 -> Q4 -> Q2 -> Q3 (plus Cold Control Q1 & Q2) across ``cohort`` and write keep-all archive."""
    all_records: list[dict[str, Any]] = []
    q3_by_policy: dict[str, list[dict[str, Any]]] = {"baseline": [], "checkpoint": []}
    part_scores_by_policy: dict[str, list[dict[str, Any]]] = {"baseline": [], "checkpoint": []}
    cold_part_scores_by_policy: dict[str, list[dict[str, Any]]] = {"baseline": [], "checkpoint": []}
    cf_by_policy: dict[str, list[dict[str, Any]]] = {"baseline": [], "checkpoint": []}

    for item in cohort:
        cid = str(item["item_id"])
        row = rows_by_id[cid]
        claim_text = str(row.get("claim", ""))
        gold_verdict = str(row.get("label", "supported"))
        passages = passages_of(dict(row))

        policies_to_run: list[tuple[str, dict | None]] = []
        if item.get("baseline_first_ct") is not None:
            policies_to_run.append(("baseline", item["baseline_first_ct"]))
        if item.get("checkpoint_first_ct") is not None:
            policies_to_run.append(("checkpoint", item["checkpoint_first_ct"]))

        for pol_name, first_ct in policies_to_run:
            assert first_ct is not None
            turn1_text = str(first_ct["text"])
            ct_parts = extract_ct_parts_from_json(turn1_text)
            pol_ct_rate = float(item.get(f"{pol_name}_ct_rate", 0.0))
            pol_majority_ct = bool(item.get(f"{pol_name}_majority_ct", False))
            for q_id in QUESTION_ORDER:
                q_rec = runner.ask_question(
                    row=row,
                    turn1_completion_text=turn1_text,
                    question_id=q_id,
                    policy=pol_name,
                    base_seed=base_seed,
                    thinking_cap=thinking_cap,
                    answer_cap=answer_cap,
                )
                q_rec["cohort"] = item["cohort"]
                q_rec["turn1_sample_index"] = int(first_ct.get("sample_index", 0))
                q_rec["ct_rate"] = pol_ct_rate
                q_rec["majority_ct"] = pol_majority_ct
                q_rec["baseline_ct_rate"] = float(item.get("baseline_ct_rate", 0.0))
                q_rec["baseline_majority_ct"] = bool(item.get("baseline_majority_ct", False))
                q_rec["checkpoint_ct_rate"] = float(item.get("checkpoint_ct_rate", 0.0))
                q_rec["checkpoint_majority_ct"] = bool(item.get("checkpoint_majority_ct", False))
                if q_id == "Q1":
                    pm = score_part_match(q_rec["answer"], ct_parts, claim_text)
                    q_rec["mechanical_score"] = pm
                    part_scores_by_policy[pol_name].append(pm)
                elif q_id == "Q4":
                    q_rec["mechanical_score"] = score_quote_validity_and_consistency(
                        q_rec["answer"], passages, turn1_text
                    )
                elif q_id == "Q2":
                    pol_verifier_runner = (
                        verifier_runner
                        if verifier_runner is not None
                        else (
                            lambda cf_row, cf_seed, _pol=pol_name: runner.run_verifier(
                                cf_row,
                                cf_seed,
                                policy=_pol,
                                thinking_cap=thinking_cap,
                                answer_cap=answer_cap,
                            )
                        )
                    )
                    cf_sc = run_counterfactual_check(
                        row, q_rec["answer"], pol_verifier_runner, base_seed=base_seed
                    )
                    q_rec["mechanical_score"] = cf_sc
                    cf_by_policy[pol_name].append(cf_sc)
                elif q_id == "Q3":
                    lean_sc = score_q3_lean_item(q_rec["answer"], gold_verdict, cid)
                    q_rec["mechanical_score"] = lean_sc
                    q3_by_policy[pol_name].append(lean_sc)
                all_records.append(q_rec)

            if include_cold_control:
                for cold_q_id in ("Q1", "Q2"):
                    c_rec = runner.ask_question(
                        row=row,
                        turn1_completion_text=None,
                        question_id=cold_q_id,
                        policy=f"cold_control:{pol_name}",
                        cold_control=True,
                        base_seed=base_seed,
                        thinking_cap=thinking_cap,
                        answer_cap=answer_cap,
                    )
                    c_rec["cohort"] = item["cohort"]
                    c_rec["ct_rate"] = pol_ct_rate
                    c_rec["majority_ct"] = pol_majority_ct
                    c_rec["baseline_ct_rate"] = float(item.get("baseline_ct_rate", 0.0))
                    c_rec["baseline_majority_ct"] = bool(item.get("baseline_majority_ct", False))
                    c_rec["checkpoint_ct_rate"] = float(item.get("checkpoint_ct_rate", 0.0))
                    c_rec["checkpoint_majority_ct"] = bool(item.get("checkpoint_majority_ct", False))
                    if cold_q_id == "Q1":
                        cold_pm = score_part_match(c_rec["answer"], ct_parts, claim_text)
                        c_rec["mechanical_score"] = cold_pm
                        cold_part_scores_by_policy[pol_name].append(cold_pm)
                    all_records.append(c_rec)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_all_path = out_dir / "s2-interview-save-all.jsonl.gz"
    save_all_sha = write_save_all(save_all_path, all_records)

    def _summarize_part_scores(scores: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not scores:
            return {"n": 0, "ct_part_fallback_count": 0, "unspecific_full_claim_copy_count": 0}
        primary = [s for s in scores if not s.get("ct_part_fallback", False)]
        scored = primary if primary else list(scores)
        n_scored = float(len(scored))
        f1s = [float(s["token_f1"]) for s in scored]
        conts = [float(s["containment"]) for s in scored]
        overlaps = [float(s["max_overlap"]) for s in scored]
        return {
            "n": len(scores),
            "primary_n": len(primary),
            "ct_part_fallback_count": sum(1 for s in scores if s.get("ct_part_fallback", False)),
            "unspecific_full_claim_copy_count": sum(
                1 for s in scores if s.get("unspecific_full_claim_copy", False)
            ),
            "whole_claim_copy_count": sum(1 for s in scores if s.get("whole_claim_copy", False)),
            "mean_token_f1": round(float(statistics.fmean(f1s)), 6),
            "median_token_f1": round(float(statistics.median(f1s)), 6),
            "f1_rate_0_3": round(sum(1 for s in scored if s.get("f1_at_0_3")) / n_scored, 6),
            "f1_rate_0_5": round(sum(1 for s in scored if s.get("f1_at_0_5")) / n_scored, 6),
            "f1_rate_0_7": round(sum(1 for s in scored if s.get("f1_at_0_7")) / n_scored, 6),
            "mean_containment": round(float(statistics.fmean(conts)), 6),
            "median_containment": round(float(statistics.median(conts)), 6),
            "containment_rate_0_3": round(
                sum(1 for s in scored if s.get("containment_at_0_3")) / n_scored, 6
            ),
            "containment_rate_0_5": round(
                sum(1 for s in scored if s.get("containment_at_0_5")) / n_scored, 6
            ),
            "containment_rate_0_7": round(
                sum(1 for s in scored if s.get("containment_at_0_7")) / n_scored, 6
            ),
            "mean_overlap": round(float(statistics.fmean(overlaps)), 6),
            "median_overlap": round(float(statistics.median(overlaps)), 6),
            "match_rate_0_3": round(sum(1 for s in scored if s["match_at_0_3"]) / n_scored, 6),
            "match_rate_0_5": round(sum(1 for s in scored if s["match_at_0_5"]) / n_scored, 6),
            "match_rate_0_7": round(sum(1 for s in scored if s["match_at_0_7"]) / n_scored, 6),
        }

    def _summarize_counterfactual(cf_scores: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not cf_scores:
            return {
                "n": 0,
                "format_ok_count": 0,
                "left_cannot_tell_count": 0,
                "left_cannot_tell_rate": 0.0,
                "direction_counts": {},
            }
        dir_counts: dict[str, int] = {}
        for s in cf_scores:
            d = str(s.get("direction", "unknown"))
            dir_counts[d] = dir_counts.get(d, 0) + 1
        left_n = sum(1 for s in cf_scores if s.get("left_cannot_tell"))
        return {
            "n": len(cf_scores),
            "format_ok_count": sum(1 for s in cf_scores if s.get("format_ok")),
            "left_cannot_tell_count": left_n,
            "left_cannot_tell_rate": round(left_n / float(len(cf_scores)), 6),
            "direction_counts": dir_counts,
        }

    receipt = {
        "test": "S2-CT-INTERVIEW",
        "cohort_size": len(cohort),
        "total_turns_recorded": len(all_records),
        "save_all_path": save_all_path.name,
        "save_all_sha256": save_all_sha,
        "part_match_summary": {
            "baseline": _summarize_part_scores(part_scores_by_policy["baseline"]),
            "checkpoint": _summarize_part_scores(part_scores_by_policy["checkpoint"]),
        },
        "cold_control_part_match_summary": {
            "baseline": _summarize_part_scores(cold_part_scores_by_policy["baseline"]),
            "checkpoint": _summarize_part_scores(cold_part_scores_by_policy["checkpoint"]),
        },
        "counterfactual_summary": {
            "baseline": _summarize_counterfactual(cf_by_policy["baseline"]),
            "checkpoint": _summarize_counterfactual(cf_by_policy["checkpoint"]),
        },
        "q3_lean_summary": {
            "baseline": summarize_q3_lean(q3_by_policy["baseline"]),
            "checkpoint": summarize_q3_lean(q3_by_policy["checkpoint"]),
        },
    }
    (out_dir / "s2-interview-receipt.json").write_text(
        json.dumps(receipt, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    return receipt
