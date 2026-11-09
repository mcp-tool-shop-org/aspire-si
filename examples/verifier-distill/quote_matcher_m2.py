"""Quote matcher arms M2a and M2b (pre-registered in PREREG-matcher-audit-v2-2026-11-07.md).

Builds on quote_matcher_m1 (imported, not forked):
- M2a = M1 + bracket/punctuation whitespace removal immediately before or after
  ( ) [ ] { } , ; : and ., applied identically to quote and passage after M1 Tier 1
  canonical normalisation, with two exemptions:
    (1) inside string literals: a span from ' or " to the next matching unescaped quote
        of the same type, on the same line, is copied verbatim with its whitespace untouched;
    (2) a . or , with a digit on both sides of the whitespace (e.g. 1 .5 vs 1.5, 1, 000 vs 1,000).
  Operators (+ - * / = < > ! & | ^ % ~) are never in the whitespace-removal set.
  Tier 2 (ellipsis) and the adaptive short rule are as in M1, using the M2a canonical form.
- M2b = M2a + '/' unification: map ' to " in the quote and passage, applied after the M2a
  canonical form (with literal boundaries detected before unification). Nothing is removed;
  a backslash-escaped quote stays a backslash followed by ".
"""

from __future__ import annotations

import re
import sys
import unicodedata
from collections.abc import Callable, Sequence
from pathlib import Path

REPO_ROOT = Path("E:/AI/rnd-local/projects/aspire-si-s2")
WORKTREE_VERIFIER_DIR = Path("E:/AI/_worktrees/aspire-s2-train-ab0e5546/examples/verifier-distill")
for _p in (str(REPO_ROOT), str(WORKTREE_VERIFIER_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import quote_matcher_m1 as m1  # noqa: E402
from quote_rule import (  # noqa: E402
    MIN_LINE_QUOTE_CHARS,
    MIN_QUOTE_CHARS,
    _spans_overlap_25pct,
    as_passages,
    content_tokens,
    normalise,
    rust_lines,
    strip_markers,
)

# Pre-registered M2a punctuation set (PREREG-matcher-audit-v2-2026-11-07.md):
# ( ) [ ] { } , ; : and .
# Operators (+ - * / = < > ! & | ^ % ~) are excluded.
M2A_PUNCT_SET: frozenset[str] = frozenset({"(", ")", "[", "]", "{", "}", ",", ";", ":", "."})

# Flags used by self-mutation verification tests (always True in production)
_ENABLE_LITERAL_EXEMPTION: bool = True
_ENABLE_DIGIT_FLANK_EXEMPTION: bool = True

_HORIZ_WS_RE = re.compile(r"[^\S\r\n]+")
_LINE_BREAK_WS_RE = re.compile(r" *\r?\n *")


def find_same_line_literal_spans(text: str) -> list[tuple[int, int]]:
    """Return non-overlapping [start, end) spans of string literals on the same line.

    A literal span runs from an unescaped ' or " to the next matching unescaped quote
    of the same type on the same line (never crossing '\\n' or '\\r').
    """
    spans: list[tuple[int, int]] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "\\":
            i += 2
            continue
        if ch in ("'", '"'):
            qch = ch
            j = i + 1
            found = -1
            while j < n:
                if text[j] in ("\n", "\r"):
                    break
                if text[j] == "\\":
                    j += 2
                    continue
                if text[j] == qch:
                    found = j
                    break
                j += 1
            if found != -1:
                spans.append((i, found + 1))
                i = found + 1
                continue
        i += 1
    return spans


def _strip_punct_ws_outside_literals(
    seg: str,
    *,
    punct_set: frozenset[str] = M2A_PUNCT_SET,
    digit_flank_exempt: bool = True,
) -> str:
    """Remove whitespace immediately before or after characters in punct_set,
    except around a '.' or ',' that has a digit on both sides of the whitespace.
    """
    n = len(seg)

    def is_exempt_dot_or_comma(idx: int) -> bool:
        if not digit_flank_exempt:
            return False
        if seg[idx] not in (".", ","):
            return False
        l_idx = idx - 1
        while l_idx >= 0 and seg[l_idx].isspace():
            l_idx -= 1
        r_idx = idx + 1
        while r_idx < n and seg[r_idx].isspace():
            r_idx += 1
        return (l_idx >= 0 and seg[l_idx].isdigit()) and (r_idx < n and seg[r_idx].isdigit())

    out: list[str] = []
    i = 0
    while i < n:
        if seg[i].isspace():
            j = i
            while j < n and seg[j].isspace():
                j += 1
            left_idx = i - 1
            right_idx = j
            remove = False
            if left_idx >= 0 and seg[left_idx] in punct_set and not is_exempt_dot_or_comma(left_idx):
                remove = True
            if right_idx < n and seg[right_idx] in punct_set and not is_exempt_dot_or_comma(right_idx):
                remove = True
            if not remove:
                out.append(" ")
            i = j
        else:
            out.append(seg[i])
            i += 1
    return "".join(out)


def normalise_m2a(
    text: str,
    *,
    is_quote: bool = False,
    _literal_exempt: bool | None = None,
    _digit_flank_exempt: bool | None = None,
    _punct_set: frozenset[str] | None = None,
) -> str:
    """M2a canonical normalisation: M1 Tier 1 canonical form + bracket/punctuation whitespace
    removal outside same-line string literals and digit-flanked '.' / ','.
    """
    lit_exempt = _ENABLE_LITERAL_EXEMPTION if _literal_exempt is None else _literal_exempt
    dig_exempt = _ENABLE_DIGIT_FLANK_EXEMPTION if _digit_flank_exempt is None else _digit_flank_exempt
    pset = M2A_PUNCT_SET if _punct_set is None else _punct_set

    s = unicodedata.normalize("NFKC", text)
    s = s.translate(m1._ZW_STRIP_TABLE)
    s = m1._HYPHEN_NEWLINE_RE.sub("", s)
    s = s.translate(m1._QUOTE_DASH_MAP)
    s = s.casefold()
    # Collapse horizontal whitespace within each line while keeping '\n' so literal spans
    # are detected strictly on the same line before lines are joined into single spaces.
    s = _HORIZ_WS_RE.sub(" ", s)
    s = _LINE_BREAK_WS_RE.sub("\n", s).strip()

    spans = find_same_line_literal_spans(s) if lit_exempt else []
    s_flat = s.replace("\n", " ")

    out: list[str] = []
    last = 0
    for st, en in spans:
        if st > last:
            out.append(
                _strip_punct_ws_outside_literals(
                    s_flat[last:st],
                    punct_set=pset,
                    digit_flank_exempt=dig_exempt,
                )
            )
        out.append(s_flat[st:en])
        last = en
    if last < len(s_flat):
        out.append(
            _strip_punct_ws_outside_literals(
                s_flat[last:],
                punct_set=pset,
                digit_flank_exempt=dig_exempt,
            )
        )

    res = "".join(out)
    if is_quote:
        res = m1._strip_quote_outer_punct(res)
    return res


def normalise_m2b(
    text: str,
    *,
    is_quote: bool = False,
    _literal_exempt: bool | None = None,
    _digit_flank_exempt: bool | None = None,
    _punct_set: frozenset[str] | None = None,
) -> str:
    """M2b canonical normalisation: map ' to " after the M2a canonical form
    (with literal boundaries detected before the unification).
    """
    s_m2a = normalise_m2a(
        text,
        is_quote=is_quote,
        _literal_exempt=_literal_exempt,
        _digit_flank_exempt=_digit_flank_exempt,
        _punct_set=_punct_set,
    )
    return s_m2a.replace("'", '"')


def _passage_variants_with_norm(
    p: str,
    norm_fn: Callable[..., str],
) -> tuple[str, str]:
    return norm_fn(p, is_quote=False), norm_fn(m1._unmarked_text(p), is_quote=False)


def _normalise_keep_para_with_norm(
    text: str,
    norm_fn: Callable[..., str],
) -> str:
    parts = m1._PARA_BREAK_RE.split(text)
    return f" {m1._PARA_SENTINEL} ".join(norm_fn(seg, is_quote=False) for seg in parts)


def _passage_variants_with_para_norm(
    p: str,
    norm_fn: Callable[..., str],
) -> tuple[str, str]:
    return (
        _normalise_keep_para_with_norm(p, norm_fn),
        _normalise_keep_para_with_norm(m1._unmarked_text(p), norm_fn),
    )


def _match_tiers_with_norm(
    quote: str,
    ps: list[str],
    norm_fn: Callable[..., str],
    tier_prefix: str,
) -> tuple[bool, str | None, tuple[int, int, int] | None]:
    q1 = norm_fn(quote, is_quote=True)

    # Tier 1 with candidate canonical form on quotes >= MIN_QUOTE_CHARS
    if len(q1) >= MIN_QUOTE_CHARS:
        for p_idx, p in enumerate(ps):
            for cand_p in _passage_variants_with_norm(p, norm_fn):
                pos = cand_p.find(q1)
                if pos != -1:
                    return True, f"{tier_prefix}_1", (p_idx, pos, pos + len(q1))

    # Tier 2: bounded ellipsis using candidate canonical form
    q_sentinel = m1._ELLIPSIS_RE.sub(m1._ELLIPSIS_SENTINEL, quote)
    ell_count = q_sentinel.count(m1._ELLIPSIS_SENTINEL)
    if 1 <= ell_count <= m1.MAX_ELLIPSES:
        raw_frags = [f for f in q_sentinel.split(m1._ELLIPSIS_SENTINEL) if f.strip()]
        if len(raw_frags) >= 2:
            norm_frags: list[str] = []
            frags_valid = True
            for rf in raw_frags:
                nf = norm_fn(rf, is_quote=True)
                words = m1._WORD_RE.findall(nf)
                if len(words) < m1.MIN_FRAGMENT_WORDS or len(content_tokens(nf)) < m1.MIN_FRAGMENT_CONTENT_TOKENS:
                    frags_valid = False
                    break
                norm_frags.append(nf)
            if frags_valid:
                for p_idx, p in enumerate(ps):
                    for cand_pb in _passage_variants_with_para_norm(p, norm_fn):
                        span_bounds = m1._match_tier2_in_passage(norm_frags, cand_pb)
                        if span_bounds is not None:
                            return True, f"{tier_prefix}_2", (p_idx, span_bounds[0], span_bounds[1])

    # Adaptive short rule using candidate canonical form
    if 0 < len(q1) < MIN_QUOTE_CHARS:
        if len(q1) >= MIN_LINE_QUOTE_CHARS and any(ch.isalnum() for ch in q1):
            for p_idx, p in enumerate(ps):
                offset = 0
                for line in rust_lines(p):
                    l_norm = norm_fn(line, is_quote=True)
                    l_unm = norm_fn(strip_markers(line), is_quote=True)
                    if l_norm == q1 or l_unm == q1:
                        return True, f"{tier_prefix}_short", (p_idx, offset, offset + len(q1))
                    offset += max(len(l_norm), 1) + 1

        words = m1._WORD_RE.findall(q1)
        if m1.SHORT_MIN_WORDS <= len(words) <= m1.SHORT_MAX_WORDS:
            has_digit = any(ch.isdigit() for ch in quote)
            has_cap = m1._has_internal_capitalised_token(quote)
            if has_digit or has_cap:
                for p_idx, p in enumerate(ps):
                    for cand_p in _passage_variants_with_norm(p, norm_fn):
                        cnt = cand_p.count(q1)
                        if 1 <= cnt <= m1.SHORT_MAX_OCCURRENCES:
                            pos = cand_p.find(q1)
                            return True, f"{tier_prefix}_short", (p_idx, pos, pos + len(q1))

    return False, None, None


def quote_match_tier_m2a(
    quote: str,
    passages: str | Sequence[str | dict],
) -> tuple[bool, int | str | None, tuple[int, int, int] | None]:
    """Return (accepted, tier, span) for arm M2a.

    Accepting tiers:
      0, 1, 2, 'short' (when accepted already by M1), or
      'm2a_1', 'm2a_2', 'm2a_short' (when accepted via M2a bracket/punctuation whitespace).
    """
    ps = as_passages(passages)
    if not quote or not ps:
        return False, None, None

    ok_m1, tier_m1, span_m1 = m1.quote_match_tier_m1(quote, ps)
    if ok_m1:
        return True, tier_m1, span_m1

    return _match_tiers_with_norm(quote, ps, normalise_m2a, "m2a")


def quote_match_tier_m2b(
    quote: str,
    passages: str | Sequence[str | dict],
) -> tuple[bool, int | str | None, tuple[int, int, int] | None]:
    """Return (accepted, tier, span) for arm M2b.

    Accepting tiers:
      0, 1, 2, 'short', 'm2a_1', 'm2a_2', 'm2a_short' (when accepted already by M2a), or
      'm2b_1', 'm2b_2', 'm2b_short' (when accepted via M2b '/' unification).
    """
    ps = as_passages(passages)
    if not quote or not ps:
        return False, None, None

    ok_m2a, tier_m2a, span_m2a = quote_match_tier_m2a(quote, ps)
    if ok_m2a:
        return True, tier_m2a, span_m2a

    return _match_tiers_with_norm(quote, ps, normalise_m2b, "m2b")


def quote_tier_m2a(quote: str, passages: str | Sequence[str | dict]) -> int | str | None:
    """Return the accepting tier name for M2a, or None if rejected."""
    _, tier, _ = quote_match_tier_m2a(quote, passages)
    return tier


def quote_tier_m2b(quote: str, passages: str | Sequence[str | dict]) -> int | str | None:
    """Return the accepting tier name for M2b, or None if rejected."""
    _, tier, _ = quote_match_tier_m2b(quote, passages)
    return tier


def quote_tier_m2(
    quote: str,
    passages: str | Sequence[str | dict],
    *,
    arm: str = "m2b",
) -> int | str | None:
    """Helper that names the accepting tier for the specified arm ('m2a' or 'm2b')."""
    if arm.lower() == "m2a":
        return quote_tier_m2a(quote, passages)
    if arm.lower() == "m2b":
        return quote_tier_m2b(quote, passages)
    raise ValueError(f"Unknown arm: {arm!r}")


def quote_found_m2a(quote: str, passages: str | Sequence[str | dict]) -> bool:
    """Pre-registered candidate matcher arm M2a."""
    accepted, _, _ = quote_match_tier_m2a(quote, passages)
    return accepted


def quote_found_m2b(quote: str, passages: str | Sequence[str | dict]) -> bool:
    """Pre-registered candidate matcher arm M2b."""
    accepted, _, _ = quote_match_tier_m2b(quote, passages)
    return accepted


def _find_quote_spans_for_arm(
    quote: str,
    passages: Sequence[str],
    match_fn: Callable[..., tuple[bool, int | str | None, tuple[int, int, int] | None]],
    norm_fn: Callable[..., str],
) -> list[tuple[int, int, int]]:
    accepted, tier, span = match_fn(quote, passages)
    if not accepted:
        return []
    if tier in (0, 1, 2, "short"):
        return m1.find_quote_spans_m1(quote, passages)
    t_str = str(tier)
    if t_str.endswith("_1") or t_str.endswith("_short"):
        q1 = norm_fn(quote, is_quote=True)
        spans: list[tuple[int, int, int]] = []
        for p_idx, p in enumerate(passages):
            for cand_p in _passage_variants_with_norm(p, norm_fn):
                pos = cand_p.find(q1)
                while pos != -1:
                    spans.append((p_idx, pos, pos + len(q1)))
                    pos = cand_p.find(q1, pos + 1)
                if spans and spans[-1][0] == p_idx:
                    break
        if spans:
            return spans
    return [span] if span is not None else []


def find_quote_spans_m2a(quote: str, passages: Sequence[str]) -> list[tuple[int, int, int]]:
    """Return candidate (passage_idx, start, end) spans under M2a."""
    return _find_quote_spans_for_arm(quote, passages, quote_match_tier_m2a, normalise_m2a)


def find_quote_spans_m2b(quote: str, passages: Sequence[str]) -> list[tuple[int, int, int]]:
    """Return candidate (passage_idx, start, end) spans under M2b."""
    return _find_quote_spans_for_arm(quote, passages, quote_match_tier_m2b, normalise_m2b)


def _quote_credit_for_arm(
    quotes: Sequence[str],
    evidence: str | Sequence[str | dict],
    *,
    verdict: str,
    correct: bool,
    found_fn: Callable[[str, list[str]], bool],
    spans_fn: Callable[[str, list[str]], list[tuple[int, int, int]]],
) -> float:
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
        if not found_fn(raw_q, passages):
            continue
        if sum(1 for ch in q if ch.isalnum()) < 8:
            continue
        if verdict == "unsupported" and len(q) < 20:
            continue
        candidates = spans_fn(raw_q, passages)
        chosen: tuple[int, int, int] | None = None
        for cand in candidates:
            if not any(_spans_overlap_25pct(cand, prev) for prev in credited_spans):
                chosen = cand
                break
        if chosen is not None:
            credited_spans.append(chosen)
    return min(0.05 * len(credited_spans), 0.15)


def quote_credit_m2a(
    quotes: Sequence[str],
    evidence: str | Sequence[str | dict],
    *,
    verdict: str,
    correct: bool,
) -> float:
    """Compute quote_credit_v2 using M2a as the quote matcher."""
    return _quote_credit_for_arm(
        quotes,
        evidence,
        verdict=verdict,
        correct=correct,
        found_fn=quote_found_m2a,
        spans_fn=find_quote_spans_m2a,
    )


def quote_credit_m2b(
    quotes: Sequence[str],
    evidence: str | Sequence[str | dict],
    *,
    verdict: str,
    correct: bool,
) -> float:
    """Compute quote_credit_v2 using M2b as the quote matcher."""
    return _quote_credit_for_arm(
        quotes,
        evidence,
        verdict=verdict,
        correct=correct,
        found_fn=quote_found_m2b,
        spans_fn=find_quote_spans_m2b,
    )
