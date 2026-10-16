"""The Stage 1 general-capability panel scorer (design §4).

The runner pins this file's bytes and refuses a sitting when they change. This module therefore
does not record its own hash.

Truncation and an unparsed answer are independent counts. A capped row whose answer side does
not parse is in both. A capped row is never right, even when the answer side matches the gold.
An unclosed think tag leaves the answer empty (`harness.split_thinking`), and an empty answer
is unparsed. Only the answer side is parsed.
"""

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _evaldir import EVAL_DIR  # noqa: E402

sys.path.insert(0, str(EVAL_DIR))
import harness  # noqa: E402

GSM_LINE = re.compile(r"^ANSWER:\s*(-?\d[\d,]*)\s*$")
WIN_LINE = re.compile(r"^ANSWER:\s*([AB])\s*$")
SOURCES = ("gsm8k", "winogrande")


def pattern_for(source: str):
    if source == "gsm8k":
        return GSM_LINE
    if source == "winogrande":
        return WIN_LINE
    raise ValueError(f"unknown panel source: {source}")


def canonical(value: str, source: str) -> str:
    """The comparable form: a comma-free integer, or the letter A or B."""
    if source == "gsm8k":
        return str(int(value.replace(",", "")))
    return value


def last_match(answer: str, source: str) -> str | None:
    """The last answer-side line that matches the source's anchored pattern, or None."""
    found = None
    rx = pattern_for(source)
    for line in answer.splitlines():
        match = rx.match(line)
        if match:
            found = canonical(match.group(1), source)
    return found


def strict_one_line(answer: str, source: str) -> bool:
    """The answer side, stripped, is one line and that line matches. Descriptive only."""
    lines = answer.strip().splitlines()
    return len(lines) == 1 and pattern_for(source).match(lines[0]) is not None


def score_generation(generation: dict, gold: str, source: str) -> dict:
    """Score one generation. `right` requires a match and a generation that was not truncated.

    `truncated` is the harness flag, unread as text. `empty` is an empty answer side, including
    a think tag that never closed. `unparsed` is an empty answer side or no matching line.
    """
    _thinking, answer, closed = harness.split_thinking(generation.get("text") or "")
    truncated = bool(generation.get("truncated"))
    empty = answer == ""
    parsed = None if empty else last_match(answer, source)
    matched = parsed is not None and parsed == canonical(str(gold), source)
    return {"parsed": parsed, "right": matched and not truncated, "truncated": truncated,
            "empty": empty, "unparsed": empty or parsed is None,
            "strict_one_line": (not empty) and strict_one_line(answer, source),
            "thinking_closed": closed}


def gold_reply(gold: str) -> str:
    """A closed reply whose answer side is the gold rendered as one ANSWER line."""
    return f"<think>\nchecked\n</think>\nANSWER: {gold}"
