"""The Stage 1 stuck rule (eval plan amendment a7b336b, R&D's ruling of 2026-10-10), frozen before round 1.

An answer is stuck when any of these holds:
  (A) the same 8-word span appears 3 or more times in the thinking;
  (B) the generation reached the 16,384-token cap;
  (C) the thinking is long AND re-reads above the baseline:
      long      = thinking tokens > max(the tier's frozen p95, 2 x the item's own target thinking tokens);
      re-reads  = the re-read rate (re-reads per 1,000 thinking tokens) > the frozen baseline (9.28).
A material line is quoted where any 5 consecutive words of it appear in the trace (the whole line if shorter, at least
3 words), after lenient normalising (case, markup, quotes, punctuation); a window shared by two or more material lines
doesn't count; hits on one line within 5 words of each other are one quote. A re-read is a quote of a line already quoted earlier in the same trace. Length over the p95 and over
2x the target are recorded, never triggers on their own. Detection is passive: it reads a finished answer.
A "no new line quoted in the final 30%" branch was dropped before freezing: it held for 98 of 102 untrained traces and
24 of 24 ROLE targets, so it carried no information.
"""

import json
import re
from pathlib import Path

RULE_FILE = Path(__file__).resolve().parents[1] / "stage1-stuck-rule.json"
SPAN = 8
CAP = 16384
QUOTE_WORDS = 5  # a material line counts as quoted where any 5 of its consecutive words appear
MIN_WORDS = 3


def lenient(text: str) -> str:
    t = text.lower().replace("**", "").replace("__", "").replace("`", "")
    t = re.sub(r"[\"'“”‘’]", "", t)
    return " ".join(re.findall(r"[a-z0-9.%£$€-]+", t))


def _windows(material: str) -> list[set]:
    """Per material line, its QUOTE_WORDS-word windows (the whole line when shorter, if at least MIN_WORDS)."""
    per = []
    for line in material.splitlines():
        w = lenient(line).split()
        if len(w) < MIN_WORDS:
            continue
        n = min(QUOTE_WORDS, len(w))
        win = {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}
        if win and win not in per:
            per.append(win)
    # a window shared by two or more lines ("leaves the north gate at") doesn't say which line was quoted
    seen: dict = {}
    for win in per:
        for x in win:
            seen[x] = seen.get(x, 0) + 1
    return [u for u in ({x for x in win if seen[x] == 1} for win in per) if u]


def quotes(thinking: str, material: str) -> tuple[list, int]:
    """(first-quote positions as fractions of the trace, re-read count) for the material's lines. A line is quoted
    where any of its windows appears; quotes of one line within QUOTE_WORDS words of each other are one quote."""
    tw = lenient(thinking).split()
    if not tw:
        return [], 0
    firsts, rereads = [], 0
    for win in _windows(material):
        n = len(next(iter(win)).split())
        hits = [i for i in range(len(tw) - n + 1) if " ".join(tw[i:i + n]) in win]
        quotes_ = []
        for h in hits:
            if not quotes_ or h - quotes_[-1][1] > QUOTE_WORDS:
                quotes_.append([h, h + n])
            else:
                quotes_[-1][1] = h + n
        if quotes_:
            firsts.append(quotes_[0][0] / len(tw))
            rereads += len(quotes_) - 1
    return firsts, rereads


def repeated_span(thinking: str) -> bool:
    w = re.findall(r"[a-z0-9']+", thinking.lower())
    seen: dict = {}
    for i in range(len(w) - SPAN + 1):
        k = " ".join(w[i:i + SPAN])
        seen[k] = seen.get(k, 0) + 1
        if seen[k] >= 3:
            return True
    return False


def load_rule(path=RULE_FILE) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def judge(thinking: str, material: str, tier: int, target_thinking_tokens: int, thinking_tokens: int,
          total_tokens: int, rule: dict) -> dict:
    """The stuck verdict and every marker, for one finished answer."""
    p95 = rule["p95_by_tier"].get(str(tier), rule["p95_by_tier"]["7+"])
    firsts, rereads = quotes(thinking, material)
    rate = 1000.0 * rereads / max(1, thinking_tokens)
    long_ = thinking_tokens > max(p95, 2 * target_thinking_tokens)
    a = repeated_span(thinking)
    b = total_tokens >= CAP
    c = long_ and rate > rule["reread_rate_p95"]
    return {"stuck": bool(a or b or c), "loop_span": a, "cap": b, "long_not_progressing": c,
            "over_p95": thinking_tokens > p95, "over_2x_target": thinking_tokens > 2 * target_thinking_tokens,
            "reread_rate": round(rate, 2), "rereads": rereads, "lines_quoted": len(firsts)}
