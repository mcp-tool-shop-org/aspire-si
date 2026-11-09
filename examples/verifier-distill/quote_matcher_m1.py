"""Quote matcher M1 (pre-registered in PREREG-matcher-audit-2026-11-07.md).

Tiers:
- Tier 0: existing quote_found from quote_rule.py, called unchanged.
- Tier 1: canonical normalisation (NFKC, strip U+200B-D/U+FEFF/U+00AD, join hyphen+newline breaks,
  collapse whitespace to single space, map typographic quotes to ' and " and dash variants
  U+2010-2015 and U+2212 to -, casefold, strip leading/trailing punctuation from the quote only
  while preserving digits, '.' between digits, '%', '<', '>', and '-' before a digit).
- Tier 2: bounded ellipsis matching (<= 2 ellipses per quote, >= 3 words and >= 2 content_tokens
  per fragment, every fragment matches Tier 1 in the same passage in order without overlap,
  each gap <= 250 normalised characters with no blank-line paragraph break, and for gaps < 40
  characters no negation token from the pre-registered list unless present in the adjacent fragments).
- Adaptive short rule: for quotes shorter than MIN_QUOTE_CHARS after normalisation, accept when
  (a) it matches a whole source line (as in Tier 0), or
  (b) it has 2-4 words, contains a digit or an internal capitalised token in the original quote,
      and occurs at most twice in the normalised passage.
- Never fuzzy. Never matched against the claim or prompt.
"""

from __future__ import annotations

import re
import sys
import unicodedata
from collections.abc import Sequence
from pathlib import Path

WORKTREE_VERIFIER_DIR = Path("E:/AI/_worktrees/aspire-s2-train-ab0e5546/examples/verifier-distill")
if str(WORKTREE_VERIFIER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKTREE_VERIFIER_DIR))

from quote_rule import (  # noqa: E402
    MIN_LINE_QUOTE_CHARS,
    MIN_QUOTE_CHARS,
    _find_quote_spans,
    _spans_overlap_25pct,
    as_passages,
    content_tokens,
    normalise,
    quote_found,
    rust_lines,
    strip_markers,
)

# Pre-registered thresholds (PREREG-matcher-audit-2026-11-07.md)
MAX_ELLIPSES: int = 2
MIN_FRAGMENT_WORDS: int = 3
MIN_FRAGMENT_CONTENT_TOKENS: int = 2
MAX_GAP_CHARS: int = 250
NEGATION_GAP_THRESHOLD: int = 40
SHORT_MIN_WORDS: int = 2
SHORT_MAX_WORDS: int = 4
SHORT_MAX_OCCURRENCES: int = 2

NEGATION_TOKENS: frozenset[str] = frozenset(
    {
        "not",
        "no",
        "never",
        "neither",
        "nor",
        "none",
        "cannot",
        "n't",
        "without",
        "except",
        "unless",
        "false",
        "fails",
        "failed",
        "unlikely",
    }
)

_ZW_STRIP_TABLE = str.maketrans(
    "",
    "",
    "\u200b\u200c\u200d\ufeff\u00ad",
)

_QUOTE_DASH_MAP = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        "\u2039": "'",
        "\u203a": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u201f": '"',
        "\u00ab": '"',
        "\u00bb": '"',
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2015": "-",
        "\u2212": "-",
    }
)

_HYPHEN_NEWLINE_RE = re.compile(r"-\r?\n\s*")
_ELLIPSIS_RE = re.compile(r"\[\.\.\.\]|\[\u2026\]|\.\s\.\s\.|\.\.\.|\u2026")
_ELLIPSIS_SENTINEL = "\u0001"
_PARA_BREAK_RE = re.compile(r"\r?\n[ \t]*\r?\n")
_PARA_SENTINEL = "\u0000"
_WORD_RE = re.compile(r"[A-Za-z0-9_]+")
_NEG_TOKEN_SCAN_RE = re.compile(r"n't\b|\b[a-z0-9_']+\b")


def _is_protected_edge_char(s: str, idx: int) -> bool:
    """Digits, '.' between digits, '%', '<', '>' and '-' before a digit are never removed."""
    ch = s[idx]
    if ch.isalnum() or ch in "%<>":
        return True
    if ch == "-" and idx + 1 < len(s) and s[idx + 1].isdigit():
        return True
    if ch == "." and 0 < idx < len(s) - 1 and s[idx - 1].isdigit() and s[idx + 1].isdigit():
        return True
    return False


def _strip_quote_outer_punct(s: str) -> str:
    """Strip leading and trailing punctuation and whitespace from the quote only."""
    start = 0
    end = len(s)
    while start < end and not _is_protected_edge_char(s, start):
        ch = s[start]
        if unicodedata.category(ch).startswith(("P", "S")) or ch.isspace():
            start += 1
        else:
            break
    while end > start and not _is_protected_edge_char(s, end - 1):
        ch = s[end - 1]
        if unicodedata.category(ch).startswith(("P", "S")) or ch.isspace():
            end -= 1
        else:
            break
    return s[start:end]


def normalise_tier1(text: str, *, is_quote: bool = False) -> str:
    """Tier 1 canonical normalisation."""
    s = unicodedata.normalize("NFKC", text)
    s = s.translate(_ZW_STRIP_TABLE)
    s = _HYPHEN_NEWLINE_RE.sub("", s)
    s = " ".join(s.split())
    s = s.translate(_QUOTE_DASH_MAP)
    s = s.casefold()
    if is_quote:
        s = _strip_quote_outer_punct(s)
    return s


def _unmarked_text(p: str) -> str:
    return "\n".join(strip_markers(line) for line in rust_lines(p))


def _passage_variants_tier1(p: str) -> tuple[str, str]:
    return normalise_tier1(p, is_quote=False), normalise_tier1(_unmarked_text(p), is_quote=False)


def _normalise_tier1_keep_para(text: str) -> str:
    parts = _PARA_BREAK_RE.split(text)
    return f" {_PARA_SENTINEL} ".join(normalise_tier1(seg, is_quote=False) for seg in parts)


def _passage_variants_with_para(p: str) -> tuple[str, str]:
    return _normalise_tier1_keep_para(p), _normalise_tier1_keep_para(_unmarked_text(p))


def _extract_negation_tokens(text: str) -> set[str]:
    found: set[str] = set()
    for m in _NEG_TOKEN_SCAN_RE.finditer(text.lower()):
        tok = m.group(0)
        if tok in NEGATION_TOKENS:
            found.add(tok)
        elif tok.endswith("n't"):
            found.add("n't")
            if tok == "can't":
                found.add("cannot")
    return found


def _has_internal_capitalised_token(original_quote: str) -> bool:
    """True when the original quote has an internal capitalised token (e.g. token after the first
    starts with uppercase, or any token has uppercase after its first character).
    """
    tokens = _WORD_RE.findall(original_quote)
    if not tokens:
        return False
    for tok in tokens[1:]:
        if any(ch.isupper() for ch in tok):
            return True
    for tok in tokens:
        if any(ch.isupper() for ch in tok[1:]):
            return True
    return False


def _match_tier2_in_passage(
    norm_fragments: list[str],
    passage_with_para: str,
    *,
    max_gap: int = MAX_GAP_CHARS,
    neg_gap_limit: int = NEGATION_GAP_THRESHOLD,
    enable_neg_guard: bool = True,
) -> tuple[int, int] | None:
    """Try to match all fragments in order without overlap in a single passage string
    that carries _PARA_SENTINEL markers at blank-line breaks.
    Returns (start_pos, end_pos) in passage_with_para (excluding sentinels) or None.
    """

    def backtrack(frag_idx: int, search_from: int, first_start: int) -> tuple[int, int] | None:
        if frag_idx == len(norm_fragments):
            clean_prefix = passage_with_para[:first_start].replace(f" {_PARA_SENTINEL} ", " ").replace(_PARA_SENTINEL, "")
            clean_full = passage_with_para[:search_from].replace(f" {_PARA_SENTINEL} ", " ").replace(_PARA_SENTINEL, "")
            return len(clean_prefix), len(clean_full)

        frag = norm_fragments[frag_idx]
        pos = passage_with_para.find(frag, search_from)
        while pos != -1:
            if frag_idx > 0:
                raw_gap = passage_with_para[search_from:pos]
                if _PARA_SENTINEL in raw_gap:
                    pos = passage_with_para.find(frag, pos + 1)
                    continue
                if len(raw_gap) > max_gap:
                    break
                if enable_neg_guard and len(raw_gap) < neg_gap_limit:
                    gap_negs = _extract_negation_tokens(raw_gap)
                    if gap_negs:
                        adj_negs = _extract_negation_tokens(norm_fragments[frag_idx - 1] + " " + frag)
                        if gap_negs - adj_negs:
                            pos = passage_with_para.find(frag, pos + 1)
                            continue
            res = backtrack(
                frag_idx + 1,
                pos + len(frag),
                pos if frag_idx == 0 else first_start,
            )
            if res is not None:
                return res
            pos = passage_with_para.find(frag, pos + 1)
        return None

    return backtrack(0, 0, 0)


def quote_match_tier_m1(
    quote: str,
    passages: str | Sequence[str | dict],
    *,
    _max_gap: int = MAX_GAP_CHARS,
    _enable_neg_guard: bool = True,
) -> tuple[bool, int | str | None, tuple[int, int, int] | None]:
    """Return (accepted, tier, span) where tier is 0, 1, 2, 'short', or None."""
    ps = as_passages(passages)
    if not quote or not ps:
        return False, None, None

    # Tier 0: existing quote_found unchanged
    if quote_found(quote, ps):
        q0 = normalise(quote)
        spans = _find_quote_spans(q0, ps)
        return True, 0, (spans[0] if spans else (0, 0, len(q0)))

    q1 = normalise_tier1(quote, is_quote=True)

    # Tier 1: canonical normalisation on quotes >= MIN_QUOTE_CHARS
    if len(q1) >= MIN_QUOTE_CHARS:
        for p_idx, p in enumerate(ps):
            for cand_p in _passage_variants_tier1(p):
                pos = cand_p.find(q1)
                if pos != -1:
                    return True, 1, (p_idx, pos, pos + len(q1))

    # Tier 2: bounded ellipsis
    q_sentinel = _ELLIPSIS_RE.sub(_ELLIPSIS_SENTINEL, quote)
    ell_count = q_sentinel.count(_ELLIPSIS_SENTINEL)
    if 1 <= ell_count <= MAX_ELLIPSES:
        raw_frags = [f for f in q_sentinel.split(_ELLIPSIS_SENTINEL) if f.strip()]
        if len(raw_frags) >= 2:
            norm_frags: list[str] = []
            frags_valid = True
            for rf in raw_frags:
                nf = normalise_tier1(rf, is_quote=True)
                words = _WORD_RE.findall(nf)
                if len(words) < MIN_FRAGMENT_WORDS or len(content_tokens(nf)) < MIN_FRAGMENT_CONTENT_TOKENS:
                    frags_valid = False
                    break
                norm_frags.append(nf)
            if frags_valid:
                for p_idx, p in enumerate(ps):
                    for cand_pb in _passage_variants_with_para(p):
                        span_bounds = _match_tier2_in_passage(
                            norm_frags,
                            cand_pb,
                            max_gap=_max_gap,
                            enable_neg_guard=_enable_neg_guard,
                        )
                        if span_bounds is not None:
                            return True, 2, (p_idx, span_bounds[0], span_bounds[1])

    # Adaptive minimum (replaces only the short-quote branch: len(q1) < MIN_QUOTE_CHARS)
    if 0 < len(q1) < MIN_QUOTE_CHARS:
        # (a) matches a whole source line (as now)
        if len(q1) >= MIN_LINE_QUOTE_CHARS and any(ch.isalnum() for ch in q1):
            for p_idx, p in enumerate(ps):
                offset = 0
                for line in rust_lines(p):
                    l_norm = normalise_tier1(line, is_quote=True)
                    l_unm = normalise_tier1(strip_markers(line), is_quote=True)
                    if l_norm == q1 or l_unm == q1:
                        return True, "short", (p_idx, offset, offset + len(q1))
                    offset += max(len(l_norm), 1) + 1

        # (b) 2-4 words, contains digit or internal capitalised token in original quote, <= 2 occurrences in passage
        words = _WORD_RE.findall(q1)
        if SHORT_MIN_WORDS <= len(words) <= SHORT_MAX_WORDS:
            has_digit = any(ch.isdigit() for ch in quote)
            has_cap = _has_internal_capitalised_token(quote)
            if has_digit or has_cap:
                for p_idx, p in enumerate(ps):
                    for cand_p in _passage_variants_tier1(p):
                        cnt = cand_p.count(q1)
                        if 1 <= cnt <= SHORT_MAX_OCCURRENCES:
                            pos = cand_p.find(q1)
                            return True, "short", (p_idx, pos, pos + len(q1))

    return False, None, None


def quote_tier_m1(quote: str, passages: str | Sequence[str | dict]) -> int | str | None:
    """Return which tier accepted the quote: 0, 1, 2, 'short', or None if rejected."""
    _, tier, _ = quote_match_tier_m1(quote, passages)
    return tier


def quote_found_m1(quote: str, passages: str | Sequence[str | dict]) -> bool:
    """Pre-registered candidate matcher M1."""
    accepted, _, _ = quote_match_tier_m1(quote, passages)
    return accepted


def find_quote_spans_m1(quote: str, passages: Sequence[str]) -> list[tuple[int, int, int]]:
    """Return candidate (passage_idx, start, end) spans in normalised passage coordinates under M1."""
    accepted, tier, span = quote_match_tier_m1(quote, passages)
    if not accepted:
        return []
    if tier == 0:
        return _find_quote_spans(normalise(quote), passages)
    if tier == 1 or tier == "short":
        q1 = normalise_tier1(quote, is_quote=True)
        spans: list[tuple[int, int, int]] = []
        for p_idx, p in enumerate(passages):
            for cand_p in _passage_variants_tier1(p):
                pos = cand_p.find(q1)
                while pos != -1:
                    spans.append((p_idx, pos, pos + len(q1)))
                    pos = cand_p.find(q1, pos + 1)
                if spans and spans[-1][0] == p_idx:
                    break
        if spans:
            return spans
    return [span] if span is not None else []


def quote_credit_m1(
    quotes: Sequence[str],
    evidence: str | Sequence[str | dict],
    *,
    verdict: str,
    correct: bool,
) -> float:
    """Compute quote_credit_v2 using M1 as the quote matcher."""
    if not correct:
        return 0.0
    passages = as_passages(evidence)
    seen_quotes: set[str] = set()
    credited_spans: list[tuple[int, int, int]] = []
    for raw_q in quotes:
        q = normalise(raw_q)
        if not q or q in seen_quotes:
            continue
        seen_quotes.add(q)
        if not quote_found_m1(raw_q, passages):
            continue
        if sum(1 for ch in q if ch.isalnum()) < 8:
            continue
        if verdict == "unsupported" and len(q) < 20:
            continue
        candidates = find_quote_spans_m1(raw_q, passages)
        chosen: tuple[int, int, int] | None = None
        for cand in candidates:
            if not any(_spans_overlap_25pct(cand, prev) for prev in credited_spans):
                chosen = cand
                break
        if chosen is not None:
            credited_spans.append(chosen)
    return min(0.05 * len(credited_spans), 0.15)
