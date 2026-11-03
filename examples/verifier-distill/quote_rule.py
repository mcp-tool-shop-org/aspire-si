"""Offrig quote rule 2 port (crates/offrig-core/src/verify.rs:487-564), span-disjoint quote credit v2,
the 7-rung miss waterfall, and quote-to-claim relevance measurement.

offrig is the authority for what counts as a found quote:
- QUOTE_RULE = 2
- MIN_QUOTE_CHARS = 12
- MIN_LINE_QUOTE_CHARS = 5
"""

from __future__ import annotations

import re
from collections.abc import Sequence

MIN_QUOTE_CHARS: int = 12
MIN_LINE_QUOTE_CHARS: int = 5
QUOTE_RULE: int = 2

_CURLY_QUOTES = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u201f": '"',
    }
)

# Explicit Unicode White_Space property set matching Rust's char::is_whitespace.
# Note: \x1c-\x1f (FS, GS, RS, US) are NOT in Unicode White_Space (nor Rust char::is_whitespace),
# even though Python's str.isspace() and str.split() treat them as whitespace.
RUST_WHITESPACE_CHARS: frozenset[str] = frozenset(
    {
        "\t",  # U+0009
        "\n",  # U+000A
        "\x0b",  # U+000B
        "\x0c",  # U+000C
        "\r",  # U+000D
        " ",  # U+0020
        "\x85",  # U+0085
        "\xa0",  # U+00A0
        "\u1680",  # U+1680
        "\u2000",  # U+2000
        "\u2001",  # U+2001
        "\u2002",  # U+2002
        "\u2003",  # U+2003
        "\u2004",  # U+2004
        "\u2005",  # U+2005
        "\u2006",  # U+2006
        "\u2007",  # U+2007
        "\u2008",  # U+2008
        "\u2009",  # U+2009
        "\u200a",  # U+200A
        "\u2028",  # U+2028
        "\u2029",  # U+2029
        "\u202f",  # U+202F
        "\u205f",  # U+205F
        "\u3000",  # U+3000
    }
)
_RUST_WHITESPACE_STR: str = "".join(sorted(RUST_WHITESPACE_CHARS))
_RUST_WS_RUN_RE = re.compile("[" + re.escape(_RUST_WHITESPACE_STR) + "]+")

# Explicit line-break set matching Rust's str::lines(): split on '\n' and strip a single trailing '\r'
# (i.e. '\n' and '\r\n' only; never standalone '\r', '\x0b', '\x0c', '\x85', '\u2028', or '\u2029').
RUST_LINE_BREAK_CHARS: frozenset[str] = frozenset({"\n"})

_COMMENT_MARKERS: tuple[str, ...] = ("///", "//!", "//", "#", "*", "--")

WATERFALL_RUNGS: tuple[str, ...] = (
    "rule_2",
    "no_minimum",
    "case",
    "punctuation",
    "ellipsis_split",
    "joined_passages",
    "fuzzy",
)

_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "but",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "not",
        "of",
        "on",
        "or",
        "so",
        "such",
        "than",
        "that",
        "the",
        "their",
        "then",
        "there",
        "these",
        "they",
        "this",
        "to",
        "was",
        "were",
        "when",
        "where",
        "which",
        "who",
        "will",
        "with",
        "would",
    }
)


def rust_lines(text: str) -> list[str]:
    """Split text into lines matching Rust's str::lines() exactly.

    Splits on '\\n' only (dropping a single trailing '\\n' at the end of the string) and strips
    a single trailing '\\r' from each resulting line. Standalone '\\r', '\\x0b', '\\x0c',
    '\\x85', '\\u2028', and '\\u2029' do not split lines.
    """
    if not text:
        return []
    if text.endswith("\n"):
        text = text[:-1]
    return [part[:-1] if part.endswith("\r") else part for part in text.split("\n")]


def _rust_split_whitespace(text: str) -> list[str]:
    """Split on runs of Unicode White_Space matching Rust's str::split_whitespace()."""
    stripped = text.strip(_RUST_WHITESPACE_STR)
    if not stripped:
        return []
    return _RUST_WS_RUN_RE.split(stripped)


def normalise(s: str) -> str:
    """Collapse runs of whitespace to single spaces, trim, and straighten curly quotes (verify.rs:487-497)."""
    straight = s.translate(_CURLY_QUOTES)
    return " ".join(_rust_split_whitespace(straight))


def strip_markers(line: str) -> str:
    """Strip one leading diff marker (+ or -, not ++ or --) and one leading comment marker."""
    t = line.lstrip(_RUST_WHITESPACE_STR)
    if t and t[0] in ("+", "-"):
        rest = t[1:]
        if not (rest and rest[0] in ("+", "-")):
            t = rest.lstrip(_RUST_WHITESPACE_STR)
    for m in _COMMENT_MARKERS:
        if t.startswith(m):
            return t[len(m) :].lstrip(_RUST_WHITESPACE_STR)
    return t


def normalise_unmarked(text: str) -> str:
    """Strip leading diff/comment markers from each line, then normalise (verify.rs:532-540)."""
    return normalise("\n".join(strip_markers(line) for line in rust_lines(text)))


def as_passages(evidence: str | Sequence[str | dict]) -> list[str]:
    """Convert evidence (passage strings, context dicts, or a single string) to passage strings."""
    if isinstance(evidence, str):
        return [evidence]
    out: list[str] = []
    for item in evidence:
        if isinstance(item, dict):
            out.append(str(item.get("text", "")))
        else:
            out.append(str(item))
    return out


def quote_found(quote: str, evidence: str | Sequence[str | dict]) -> bool:
    """Check whether quote appears in any passage of evidence under offrig rule 2 (verify.rs:546-564)."""
    q = normalise(quote)
    length = len(q)
    if length == 0:
        return False
    passages = as_passages(evidence)
    if length >= MIN_QUOTE_CHARS:
        return any(q in normalise(p) or q in normalise_unmarked(p) for p in passages)
    return (
        length >= MIN_LINE_QUOTE_CHARS
        and any(ch.isalnum() for ch in q)
        and any(
            any(normalise(line) == q or normalise(strip_markers(line)) == q for line in rust_lines(p))
            for p in passages
        )
    )


def _find_quote_spans(q: str, passages: Sequence[str]) -> list[tuple[int, int, int]]:
    """Return candidate (passage_idx, start, end) spans in normalised passage coordinates for quote q."""
    spans: list[tuple[int, int, int]] = []
    q_len = len(q)
    if q_len >= MIN_QUOTE_CHARS:
        for p_idx, p in enumerate(passages):
            for text_norm in (normalise(p), normalise_unmarked(p)):
                pos = text_norm.find(q)
                while pos != -1:
                    spans.append((p_idx, pos, pos + q_len))
                    pos = text_norm.find(q, pos + 1)
                if spans and spans[-1][0] == p_idx:
                    break
        return spans

    for p_idx, p in enumerate(passages):
        offset = 0
        for line in rust_lines(p):
            l_norm = normalise(line)
            l_unmarked = normalise(strip_markers(line))
            if l_norm == q or l_unmarked == q:
                spans.append((p_idx, offset, offset + q_len))
            offset += max(len(l_norm), 1) + 1
    return spans


def _spans_overlap_25pct(a: tuple[int, int, int], b: tuple[int, int, int]) -> bool:
    """True when two spans in the same passage overlap by 25% or more of the shorter span."""
    if a[0] != b[0]:
        return False
    inter = max(0, min(a[2], b[2]) - max(a[1], b[1]))
    if inter <= 0:
        return False
    shorter = max(1, min(a[2] - a[1], b[2] - b[1]))
    return (inter / shorter) >= 0.25


def quote_credit_v2(
    quotes: Sequence[str],
    evidence: str | Sequence[str | dict],
    *,
    verdict: str,
    correct: bool,
) -> float:
    """Compute anti-farming quote credit under DESIGN-quote-rule-2026-11-03.md §3.

    - +0.05 per span-disjoint found quote, only when the claim verdict is correct;
    - capped at +0.15 (inside the +0.3 positive-extras cap);
    - a quote overlapping an already-credited span by >= 25% in the same passage earns nothing;
    - duplicate quotes earn once;
    - density: at least 8 alphanumeric characters;
    - no credit on 'unsupported' unless the quote is at least 20 normalised characters.
    """
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
        if not quote_found(raw_q, passages):
            continue
        if sum(1 for ch in q if ch.isalnum()) < 8:
            continue
        if verdict == "unsupported" and len(q) < 20:
            continue
        candidates = _find_quote_spans(q, passages)
        chosen: tuple[int, int, int] | None = None
        for cand in candidates:
            if not any(_spans_overlap_25pct(cand, prev) for prev in credited_spans):
                chosen = cand
                break
        if chosen is not None:
            credited_spans.append(chosen)
    return min(0.05 * len(credited_spans), 0.15)


def _strip_punct(text: str) -> str:
    """Lowercase and strip punctuation (preserving words/numbers), except '...' which stays distinct."""
    if "..." in text or "\u2026" in text:
        text = text.replace("\u2026", "...")
        parts = [re.sub(r"[^\w\s]+", " ", seg.lower()) for seg in text.split("...")]
        return " ... ".join(" ".join(_rust_split_whitespace(p)) for p in parts)
    cleaned = re.sub(r"[^\w\s]+", " ", text.lower())
    return " ".join(_rust_split_whitespace(cleaned))


def _word_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", text.lower())


def classify_quote_rung(quote: str, evidence: str | Sequence[str | dict]) -> str:
    """Tag a quote with the first rung of the 7-rung waterfall that matches, or 'unmatched'."""
    passages = as_passages(evidence)
    if quote_found(quote, passages):
        return "rule_2"
    q = normalise(quote)
    if not q:
        return "unmatched"

    # Rung 2: no minimum length
    if any(q in normalise(p) or q in normalise_unmarked(p) for p in passages):
        return "no_minimum"

    # Rung 3: case-insensitive
    q_low = q.lower()
    if any(q_low in normalise(p).lower() or q_low in normalise_unmarked(p).lower() for p in passages):
        return "case"

    # Rung 4: punctuation
    q_punct = _strip_punct(q)
    if q_punct and "..." not in q_punct:
        if any(
            q_punct in _strip_punct(p) or q_punct in _strip_punct(normalise_unmarked(p)) for p in passages
        ):
            return "punctuation"

    # Rung 5: ellipsis split
    if "..." in q or "\u2026" in q:
        segs = [normalise(s).lower() for s in re.split(r"(?:\.\.\.|\u2026)+", q) if normalise(s)]
        if len(segs) >= 2 and any(
            all(seg in normalise(p).lower() or seg in normalise_unmarked(p).lower() for seg in segs)
            for p in passages
        ):
            return "ellipsis_split"

    # Rung 6: joined passages
    joined_norm = " ".join(normalise(p) for p in passages).lower()
    joined_unmarked = " ".join(normalise_unmarked(p) for p in passages).lower()
    if q_low in joined_norm or q_low in joined_unmarked:
        return "joined_passages"

    # Rung 7: fuzzy match (token Jaccard >= 0.70 against a window in any passage)
    q_toks = _word_tokens(q)
    q_set = set(q_toks)
    if q_set:
        win_sizes = {max(1, len(q_toks) - 1), max(1, len(q_toks)), len(q_toks) + 1}
        for p in passages:
            p_toks = _word_tokens(p)
            if not p_toks:
                continue
            for w in win_sizes:
                if w >= len(p_toks):
                    w_set = set(p_toks)
                    if len(q_set & w_set) / len(q_set | w_set) >= 0.70:
                        return "fuzzy"
                else:
                    for i in range(len(p_toks) - w + 1):
                        w_set = set(p_toks[i : i + w])
                        if len(q_set & w_set) / len(q_set | w_set) >= 0.70:
                            return "fuzzy"

    return "unmatched"


def content_tokens(text: str) -> set[str]:
    """Extract non-stopword content tokens, numbers, or identifiers from text."""
    raw = re.findall(r"[A-Za-z0-9_]+", text)
    out: set[str] = set()
    for tok in raw:
        low = tok.lower()
        is_num = any(ch.isdigit() for ch in tok)
        is_ident = "_" in tok or (any(ch.isupper() for ch in tok[1:]) and any(ch.islower() for ch in tok))
        if is_num or is_ident or (len(low) >= 2 and low not in _STOPWORDS):
            out.add(low)
    return out


def quote_is_relevant(quote: str, claim: str) -> bool:
    """True when quote shares at least one non-stopword content token, number, or identifier with claim."""
    q_toks = content_tokens(quote)
    c_toks = content_tokens(claim)
    return bool(q_toks and c_toks and (q_toks & c_toks))
