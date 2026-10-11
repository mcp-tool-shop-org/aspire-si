"""The Stage 1 stuck rule (eval plan amendments a7b336b and 4bf6384; R&D's rulings of 2026-10-10), frozen before
round 1.

An answer is stuck when any of these holds:
  (A) the same 8-word span appears 3 or more times in the thinking;
  (B) the generation reached the 16,384-token cap;
  (C) the thinking is long AND not progressing:
      long            = thinking tokens > max(the tier's frozen p95, 2 x the item's own target thinking tokens);
      not progressing = the re-read rate > its frozen baseline (8.09 per 1,000 thinking tokens), OR at least one
                        burst: 3 or more hedges within 200 thinking tokens (W and k fixed a priori).
A material line is quoted where any 5 consecutive words of it appear in the trace (the whole line if shorter, at least
3 words), after lenient normalising (case, markup, quotes, punctuation); a window shared by two or more material lines
doesn't count; hits on one line within 5 words of each other are one quote. A re-read is a quote of a line already quoted earlier in the same trace. Length over the p95 and over
2x the target are recorded, never triggers on their own. Detection is passive: it reads a finished answer.
A hedge is a phrase from HEDGES (fixed before counting), matched case-insensitively at word boundaries, except inside
a taught self-check ("If ..., then I ..."), a planned-step "check", or a quoted material line.
The hedge rate (p95 8.32 per 1,000) was measured and then replaced as a trigger by the burst clause; it is recorded
as a measure on every trace, with the per-phrase counts, positions, gaps and bursts (hedge_profile, burst_profile).
A "no new line quoted in the final 30%" branch was dropped before freezing: it held for 98 of 102 untrained traces and
48 of 48 ROLE D targets, so it carried no information.
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


# The hedge list, fixed by R&D before any count was seen. Not pruned after measuring.
HEDGES = ["wait", "hmm", "actually", "hold on",
          "let me re-check", "let me recheck", "let me check again", "let me reconsider", "let me re-read",
          "let me reread", "on second thought", "i'm not sure", "i am not sure", "maybe i", "or maybe",
          "did i", "am i missing", "that doesn't seem right", "that can't be right", "let me start over",
          "going back", "again,"]
HEDGE_RX = re.compile(r"(?<![\w'])(" + "|".join(re.escape(h) for h in HEDGES) + r")(?![\w'])", re.I)
SELF_CHECK = re.compile(r"^\s*if\b.*\bthen i\b", re.I)          # a taught self-check frame
PLANNED_CHECK = re.compile(r"\bi (?:will |'ll )?check\b", re.I)  # "I check line 3": a planned step


def _material_windows(material: str) -> set:
    """Every 5-word window (or whole short line, at least 3 words) of every material line, shared or not."""
    out = set()
    for line in material.splitlines():
        w = lenient(line).split()
        if len(w) >= MIN_WORDS:
            n = min(QUOTE_WORDS, len(w))
            out |= {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}
    return out


def _hedge_hits(thinking: str, material: str = "") -> list:
    """[(phrase, character position)] for the hedges in a trace, outside self-check sentences, planned-step checks
    and quoted material lines. A hit is inside a quoted material line when a material window containing its first
    word appears around it."""
    wins = _material_windows(material)
    out = []
    offset = 0
    for line in thinking.replace("’", "'").split("\n"):
        cursor = 0
        for sent in re.split(r"(?<=[.!?])\s+", line):
            start = line.find(sent, cursor)
            cursor = start + len(sent)
            if SELF_CHECK.match(sent):
                continue
            for m in HEDGE_RX.finditer(sent):
                if PLANNED_CHECK.search(m.group(0)):
                    continue
                around = " " + lenient(sent[max(0, m.start() - 80):m.end() + 80]) + " "
                first = lenient(m.group(0)).split()[:1]
                if first and any(f" {w} " in around for w in wins if first[0] in w.split()):
                    continue
                out.append((m.group(0).lower(), offset + start + m.start()))
        offset += len(line) + 1
    return out


def hedges(thinking: str, material: str = "") -> int:
    return len(_hedge_hits(thinking, material))


def hedge_profile(thinking: str, material: str = "", thinking_tokens: int = 0, count=None) -> dict:
    """The recorded measure, on every trace (trigger or not): counts per phrase, the rate per 1,000 thinking tokens,
    first and last hedge positions as fractions of the trace's thinking tokens, and end clustering (the share of
    hedges in the final 30%; None when there are fewer than 2). `count(text)` gives token counts (the student's
    tokenizer); without it, positions fall back to characters."""
    hits = _hedge_hits(thinking, material)
    per: dict = {}
    for k, _ in hits:
        per[k] = per.get(k, 0) + 1
    if count is not None:
        total = max(1, thinking_tokens or count(thinking))
        pos = [count(thinking[:x]) / total for _, x in hits]
    else:
        total = max(1, len(thinking))
        pos = [x / total for _, x in hits]
    return {"hedges": len(hits), "hedge_rate": round(1000.0 * len(hits) / max(1, thinking_tokens), 2),
            "per_phrase": dict(sorted(per.items())),
            "first": round(pos[0], 3) if pos else None, "last": round(pos[-1], 3) if pos else None,
            "end_clustering": round(sum(1 for x in pos if x >= 0.7) / len(pos), 3) if len(pos) >= 2 else None}


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
          total_tokens: int, rule: dict, count=None) -> dict:
    """The stuck verdict and every marker, for one finished answer. `count(text)` gives token counts (the student's
    tokenizer) for burst positions; without it, words stand in for tokens."""
    p95 = rule["p95_by_tier"].get(str(tier), rule["p95_by_tier"]["7+"])
    firsts, rereads = quotes(thinking, material)
    rate = 1000.0 * rereads / max(1, thinking_tokens)
    long_ = thinking_tokens > max(p95, 2 * target_thinking_tokens)
    a = repeated_span(thinking)
    b = total_tokens >= CAP
    count = count or (lambda x: len(x.split()))
    pos = hedge_token_positions(thinking, material, count)
    burst = burst_profile(pos, thinking_tokens, rule["burst"]["window"], rule["burst"]["k"])
    hrate = 1000.0 * len(pos) / max(1, thinking_tokens)
    c = long_ and (rate > rule["reread_rate_p95"] or bool(burst["bursts"]))
    return {"stuck": bool(a or b or c), "loop_span": a, "cap": b, "long_not_progressing": c,
            "over_p95": thinking_tokens > p95, "over_2x_target": thinking_tokens > 2 * target_thinking_tokens,
            "reread_rate": round(rate, 2), "rereads": rereads, "lines_quoted": len(firsts),
            "hedge_rate": round(hrate, 2), "hedges": len(pos), "bursts": len(burst["bursts"]),
            "max_in_window": burst["max_in_window"], "burst_before_answer": burst["burst_before_answer"]}


def hedge_token_positions(thinking: str, material: str, count) -> list:
    """Thinking-token offsets of the hedges (count(text) gives a token count)."""
    return [count(thinking[:x]) for _, x in _hedge_hits(thinking, material)]


def burst_profile(positions: list, thinking_tokens: int, window: int, k: int, before_answer: float = 0.10) -> dict:
    """Bursts: k or more hedges within `window` thinking tokens. Gaps between consecutive hedges, the most hedges in
    any window, the bursts (start and end token), and whether a burst ends in the final `before_answer` share of the
    trace, right before the answer."""
    pos = sorted(positions)
    gaps = [b - a for a, b in zip(pos, pos[1:])]
    best, j = 0, 0
    for i in range(len(pos)):
        while pos[i] - pos[j] >= window:
            j += 1
        best = max(best, i - j + 1)
    bursts = []
    for i in range(len(pos) - k + 1):
        if pos[i + k - 1] - pos[i] < window:
            if bursts and pos[i] <= bursts[-1][1]:
                bursts[-1][1] = pos[i + k - 1]
            else:
                bursts.append([pos[i], pos[i + k - 1]])
    end = max(1, thinking_tokens)
    return {"gaps": gaps, "max_in_window": best, "bursts": bursts,
            "burst_before_answer": any(b[1] >= (1 - before_answer) * end for b in bursts)}
