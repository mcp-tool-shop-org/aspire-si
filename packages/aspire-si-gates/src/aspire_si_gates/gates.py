"""The Stage 1 data gates: schema, per-kind parts, amendments, diversity, floors, the leakage guard against
given held-out sets, and the batch and freeze reports. Standard library only.

validate(rows) = check_item on every row, then check_batch on the whole set. check_against(row, accepted)
gives the incremental refusals a generator can hand back to its writer.
"""

import hashlib
import json
import re
import sys
from pathlib import Path

from .answer_format import parse_answer
import math

from .constants import LESSON_HEADING, LESSONS, PRE_AMENDMENT_LESSONS, ROLE_WORDS, TRAIT_NAMES  # noqa: F401

KINDS = {"A", "B", "C", "D", "E"}
VERDICTS = {"supported", "unsupported", "cannot_tell"}
FLAWS = {"rubber_stamp", "guess", "loop", "nitpick", "fix", "obeyed_material", "caved", "reject_by_default", "none"}
ESC = re.compile(r"^(no|yes \((medical|legal|safety|money)[^)]*\))$")
REQUIRED = {"id": str, "lesson": str, "trait": str, "kind": str, "side": str, "content_kind": str, "tier": int,
            "turns": list, "sources": list, "invented": bool, "key_basis": str}


class DataError(ValueError):
    def __init__(self, message: str = "", *, code: str | None = None, hint: str | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint


FLAW_STEMS = {
    "guess": re.compile(r"\bguess(?:ed|es|ing)?\b", re.I),
    "rubber_stamp": re.compile(r"\brubber[- ]?stamp(?:ed|s|ing)?\b", re.I),
    "loop": re.compile(r"\bloop(?:ed|s|ing)?\b", re.I),
    "nitpick": re.compile(r"\bnit[- ]?pick(?:ed|s|ing|y)?\b", re.I),
    "fix": re.compile(r"\bfix(?:ed|es|ing)?\b", re.I),
    "obeyed_material": re.compile(r"\bobey(?:ed|s|ing)?\b", re.I),
    "caved": re.compile(r"\bcav(?:e|ed|es|ing)\b", re.I),
    "reject_by_default": re.compile(r"\breject(?:ed|s|ing|ion)?[- ]?(?:by[- ]?default)?\b", re.I),
}
TRACE_LEAK_RE = re.compile(
    r"\b(?:guess(?:ed|es|ing)?|rubber[- ]?stamp(?:ed|s|ing)?|loop(?:ed|s|ing)?|nit[- ]?pick(?:ed|s|ing|y)?|fix(?:ed|es|ing)?|obey(?:ed|s|ing)?|cav(?:e|ed|es|ing)|reject(?:ed|s|ing|ion)?[- ]?(?:by[- ]?default)?)\b",
    re.I,
)
_QUOTED_TRACE_SPAN_RE = re.compile(r'"([^"]{4,})"|(?<!\w)\'([^\']{4,})\'(?!\w)|“([^”]{4,})”|‘([^’]{4,})’')

# Frozen Stage 1 items written before the trace-leak gate (1.7.6), exempt by id in check_batch so the frozen set
# (FROZEN.md e7b69464) validates unchanged.
TRACE_LEAK_EXEMPT = {
    "L1-C-05": "batch 8 (frozen Stage 1): L1 trace says 'They obeyed the stamp sentence.'",
    "L1-C-06": "batch 8 (frozen Stage 1): L1 trace says 'They guessed two pounds.'",
}


REPO_CITATION = re.compile(r"\b[\w.-]+/[\w.-]+@[0-9a-f]{7,40}\b|github\.com/")
DRAWN = [re.compile(r"[├└│]|\bROOT\b"), re.compile(r"(?m)^\s*\|.*\|\s*$"), re.compile(r"(?m)^\s*\d+[.)]\s")]


TIME_WORDS = re.compile(r"\b(today|now|currently|current|at the moment|these days|right now|this (week|month|year)|"
                        r"latest|recently|yesterday|tomorrow)\b", re.I)
DATED = re.compile(r"\b(19|20)\d\d\b|\b\d{1,2} (jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", re.I)


def _statement(user: str) -> str:
    m = re.search(r"statement\s*:\s*(.+)", user, re.I)
    return m.group(1).strip() if m else ""


def _sentences(text: str) -> int:
    return len(re.findall(r"[.!?](?:\s|$)", text))


def _source_ok(src) -> tuple[bool, bool]:
    """(well-formed, real). A real source names a repo and commit; an invented part says so."""
    if isinstance(src, str):
        return bool(src.strip()), bool(re.search(r"@[0-9a-f]{7,40}", src)) and not src.lower().startswith("invented")
    if isinstance(src, dict):
        if src.get("invented"):
            return True, False
        return bool(src.get("repo") and src.get("commit")), bool(src.get("repo") and src.get("commit"))
    return False, False


def _shingles(text: str, n: int = 8) -> set:
    w = re.findall(r"[a-z']+", text.lower())
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def _key_points(k) -> list[str]:
    if isinstance(k, dict):
        return [norm(str(x)).lower() for x in (k.get("key_points") or [k.get("verdict", "")]) if x]
    if isinstance(k, list):
        return [norm(str(x)).lower() for x in k if x]
    return [norm(str(k)).lower()] if k else []


def ws(x: str) -> str:
    """Whitespace collapsed, markup kept: a quoted line may wrap, but `**` must stay as printed."""
    return re.sub(r"\s+", " ", x).strip()


def norm(x: str) -> str:
    return re.sub(r"\s+", " ", x.replace("**", "").replace("__", "")).strip()


HEADING = re.compile(r"^\s*(#{1,6})\s")
VERSIONED = re.compile(r"\bv?\d+\.\d+(?:\.\d+)?\b|\b(?:19|20)\d\d-\d\d-\d\d\b|"
                       r"\b\d{1,2} (?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* (?:19|20)\d\d\b", re.I)
MIN_MULTI_LINE_SHARE = 0.25


def _version_heading_above(lines: list[str], idx: int):
    """The version or date heading a material line sits under, or None. A plain heading at the same level or
    higher closes the version section above it."""
    current = None  # (level, line)
    for line in lines[:idx]:
        m = HEADING.match(line)
        if not m:
            continue
        level = len(m.group(1))
        if VERSIONED.search(line):
            current = (level, line.strip())
        elif current and level <= current[0]:
            current = None
    return current[1] if current else None


def _check_quoting(where, keys, material, content_kind):
    """Amendment 1, rules 2 and 3: a table row is quoted whole; in a changelog, a line under a version or date
    heading carries that heading as its own DECIDING line."""
    lines = (material or "").splitlines()
    for d in keys["deciding"]:
        wd = ws(d)
        rows = [ws(x) for x in lines if x.strip().startswith("|") and wd in ws(x)]
        prose = [x for x in lines if not x.strip().startswith("|") and wd in ws(x)]
        if rows and not prose and wd not in rows:
            raise DataError(f"{where}: quote the whole table row, not part of it: {d!r} is inside {rows[0]!r}")
    if content_kind != "changelog":
        return
    given = {ws(x) for x in keys["deciding"]}
    for d in keys["deciding"]:
        if HEADING.match(d):
            continue
        idx = next((i for i, x in enumerate(lines) if ws(d) in ws(x)), None)
        head = _version_heading_above(lines, idx) if idx is not None else None
        if head and ws(head) not in given:
            raise DataError(f"{where}: {d!r} sits under the version heading {head!r}; quote the heading as its "
                            f"own DECIDING line")


def check_multi_line_share(rows: list[dict], skip=PRE_AMENDMENT_LESSONS) -> None:
    """Amendment 1, rule 4: per lesson, at least a quarter of the verdict-bearing items (supported or
    unsupported) have 2 to 4 DECIDING lines."""
    per: dict = {}
    for r in rows:
        if r.get("lesson") in skip:
            continue
        keys = r if r.get("kind") in ("C", "D") else (r.get("quiz") if r.get("kind") == "E" else None)
        if not isinstance(keys, dict) or keys.get("verdict") not in ("supported", "unsupported"):
            continue
        n = per.setdefault(r["lesson"], [0, 0])
        n[0] += 1
        n[1] += 2 <= len(keys.get("deciding") or []) <= 4
    for lesson, (total, multi) in sorted(per.items()):
        if multi < MIN_MULTI_LINE_SHARE * total:
            raise DataError(f"{lesson}: {multi} of {total} verdict-bearing items ({multi / total:.0%}) have 2-4 "
                            f"DECIDING lines; at least {MIN_MULTI_LINE_SHARE:.0%} are needed")


MIN_VERDICT_SHARE = 0.25  # amendment 2, point 1: cannot_tell and supported each at least a quarter


def _verdict_keys(r: dict):
    """The final-turn keys of a verdict-bearing item (C, D, or a verdict-type E), or None."""
    if r.get("kind") in ("C", "D"):
        return r
    q = r.get("quiz")
    return q if r.get("kind") == "E" and isinstance(q, dict) and q.get("type") == "verdict" else None


def _turn_verdicts(where, r: dict) -> list:
    """Each turn's verdict for a multi-turn kind D item: turn_keys where given, else the reply's own VERDICT line.
    Amendment 2: every turn is a complete answer with its own three lines."""
    out = []
    for i, t in enumerate(r["turns"], 1):
        p = parse_answer(t["reply"])
        if not (p["verdict"] and p["deciding"] is not None and p["escalate"] is not None):
            raise DataError(f"{where} turn {i}: a multi-turn kind D reply needs its own three lines "
                            f"(VERDICT, DECIDING, ESCALATE)")
        keyed = (r.get("turn_keys") or {}).get(str(i)) if i < len(r["turns"]) else r
        out.append(keyed["verdict"] if keyed else p["verdict"])
    return out


def check_floors(rows: list[dict], skip=PRE_AMENDMENT_LESSONS) -> None:
    """Amendment 2, per lesson: (e) cannot_tell and supported each at least 25% of the verdict-bearing items;
    (f) at least one ESCALATE yes (...) item and one cannot_tell item with ESCALATE no; (g) at least two
    multi-turn kind D items, one whose last verdict differs from its first and one whose verdict stands."""
    per: dict = {}
    for n, r in enumerate(rows, 1):
        if r.get("lesson") in skip:
            continue
        keys = _verdict_keys(r)
        if not isinstance(keys, dict):
            continue
        st = per.setdefault(r["lesson"], {"n": 0, "cannot_tell": 0, "supported": 0, "esc_yes": 0,
                                          "ct_esc_no": 0, "multi": 0, "changed": 0, "stood": 0})
        st["n"] += 1
        v = keys.get("verdict")
        st["cannot_tell"] += v == "cannot_tell"
        st["supported"] += v == "supported"
        esc = keys.get("escalate") or ""
        st["esc_yes"] += esc.startswith("yes (")
        st["ct_esc_no"] += v == "cannot_tell" and esc == "no"
        if r["kind"] == "D" and len(r["turns"]) >= 2:
            vs = _turn_verdicts(f"line {n} ({r.get('id', '?')})", r)
            st["multi"] += 1
            st["changed"] += vs[-1] != vs[0]
            st["stood"] += vs[-1] == vs[0]
    for lesson, st in sorted(per.items()):
        n = st["n"]
        for v in ("cannot_tell", "supported"):
            if st[v] < MIN_VERDICT_SHARE * n:
                raise DataError(f"{lesson}: {st[v]} of {n} verdict-bearing items ({st[v] / n:.0%}) are keyed {v}; "
                                f"at least {MIN_VERDICT_SHARE:.0%} are needed")
        if st["esc_yes"] < 1:
            raise DataError(f"{lesson}: 0 of {n} verdict-bearing items are keyed ESCALATE yes (...); at least 1 is needed")
        if st["ct_esc_no"] < 1:
            raise DataError(f"{lesson}: 0 of {st['cannot_tell']} cannot_tell items are keyed ESCALATE no; "
                            f"at least 1 is needed")
        if st["multi"] < 2 or not st["changed"] or not st["stood"]:
            raise DataError(f"{lesson}: {st['multi']} multi-turn kind D items ({st['changed']} with a changed "
                            f"verdict, {st['stood']} where it stands); at least 2 are needed, with at least 1 of each")


def check_pairs(rows: list[dict], skip=PRE_AMENDMENT_LESSONS, min_pairs: int = 2) -> None:
    """Amendment 2, contrast pairs: items sharing a `pair` id are exactly one cannot_tell and one unsupported
    verdict-bearing item of one lesson (the same statement against silent and against contradicting material);
    every lesson has at least `min_pairs` of them."""
    pairs: dict = {}
    for r in rows:
        if r.get("lesson") in skip or r.get("pair") is None:
            continue
        if not isinstance(r["pair"], str) or not r["pair"].strip():
            raise DataError(f"{r.get('id', '?')}: pair must be a non-empty string id")
        keys = _verdict_keys(r)
        if not isinstance(keys, dict):
            raise DataError(f"{r.get('id', '?')}: pair {r['pair']!r} is on an item without a verdict")
        pairs.setdefault((r["lesson"], r["pair"]), []).append((r["id"], keys.get("verdict")))
    count: dict = {}
    for (lesson, pid), members in sorted(pairs.items()):
        if sorted(v for _, v in members) != ["cannot_tell", "unsupported"]:
            raise DataError(f"{lesson}: pair {pid!r} is {members}; it needs exactly one cannot_tell and one "
                            f"unsupported item")
        count[lesson] = count.get(lesson, 0) + 1
    lessons = sorted({r["lesson"] for r in rows if r.get("lesson") not in skip and _verdict_keys(r)})
    for lesson in lessons:
        if count.get(lesson, 0) < min_pairs:
            raise DataError(f"{lesson}: {count.get(lesson, 0)} contrast pairs (shared `pair` id); at least "
                            f"{min_pairs} are needed")


# ---------------------------------------------------------------- the diversity gate (after batch 3)
THREE_LINE = re.compile(r"^\s*(VERDICT|DECIDING|ESCALATE)\s*:", re.I)
SPAN_WORDS = 8


def _content_lines(r: dict) -> str:
    """Every material line of an item (its material and every turn's new_material), whitespace-normalised."""
    return ws((r.get("material") or "") + "\n" + "\n".join(
        x for t in r["turns"] for x in t.get("new_material", []))).lower()


def _spans(r: dict) -> set:
    """8-word spans of an item's user turns, thinking and replies, outside material lines, the Statement line
    and the three-line block. A bracketed quote of the material counts as one token, so a skeleton sentence
    with different quotes in it is still the same span."""
    content = _content_lines(r)
    out = set()
    for t in r["turns"]:
        for text in (t["user"], t["thinking"], t["reply"]):
            for line in text.splitlines():
                bare = ws(line).replace('\\"', '"').strip("\"'“” ").lower()
                if not bare or THREE_LINE.match(line) or STATEMENT_RE.match(line) or bare in content:
                    continue
                out |= line_spans(line)
    return out


def line_spans(line: str) -> set:
    """The 8-word spans of one line, tokenized as the diversity gate tokenizes: lower case, digits kept, and a
    bracketed quote as the one token [q]. _spans applies it to an item's lines; a trace's lines go through it too."""
    words = re.findall(r"[a-z0-9']+|\[q\]", re.sub(r"\[[^\]]*\]", " [q] ", line.lower()))
    return {" ".join(words[i:i + SPAN_WORDS]) for i in range(len(words) - SPAN_WORDS + 1)}


STATEMENT_RE = re.compile(r"^\s*statement\s*:", re.I)


def shared_spans(rows: list[dict], skip=PRE_AMENDMENT_LESSONS, max_items: int = 2) -> dict:
    """{(lesson, span): [ids]} for every 8-word span in more than max_items items of a lesson."""
    seen: dict = {}
    for r in rows:
        if r.get("lesson") in skip:
            continue
        for sp in _spans(r):
            seen.setdefault((r["lesson"], sp), []).append(r["id"])
    return {k: v for k, v in seen.items() if len(v) > max_items}


def check_spans(rows: list[dict], skip=PRE_AMENDMENT_LESSONS) -> None:
    st = shared_spans(rows, skip)
    if st:
        (lesson, span), ids = min(st.items(), key=lambda kv: (-len(kv[1]), kv[0]))
        raise DataError(f"{lesson}: the span {span!r} appears in {len(ids)} items ({', '.join(ids[:8])}); "
                        f"{len(st)} shared 8-word span(s) in all; none may appear in 3 or more items")


# Framing words that name a kind of material, and the content kinds each clearly does NOT fit. Conservative: only
# clear mismatches are refused (a board over code, rules over a changelog); ambiguous words ("note", "file",
# "entry", "text", "record") are not checked.
PHYSICAL = {"board", "card", "slip", "panel", "sign", "poster", "sheet", "menu", "leaflet", "flyer", "rules",
            "notice", "label", "timetable", "memo", "minutes", "plaque", "tag", "ticket"}
CODE_WORDS = {"code", "snippet", "function", "script", "module", "config"}
CHANGELOG_WORDS = {"changelog", "release notes", "release note"}
README_WORDS = {"readme"}
NOT_FIT = {
    "code": PHYSICAL | CHANGELOG_WORDS,
    "changelog": PHYSICAL | CODE_WORDS,
    "design-note": PHYSICAL - {"memo"},
}
for _k in ("notice", "policy", "label", "timetable", "record"):
    NOT_FIT[_k] = CODE_WORDS | CHANGELOG_WORDS | README_WORDS


def framing_mismatch(r: dict) -> str:
    """The framing word that clearly doesn't fit the item's content_kind, or ''."""
    if r.get("kind") not in ("C", "D"):
        return ""
    user = r["turns"][0]["user"]
    at = user.find("Material:")
    framing = user[:at].lower() if at > 0 else ""
    bad = NOT_FIT.get(r.get("content_kind", ""), set())
    for w in sorted(bad, key=len, reverse=True):
        if re.search(rf"\b{re.escape(w)}s?\b", framing):
            return w
    return ""


def check_framing_fits(rows: list[dict], skip=PRE_AMENDMENT_LESSONS) -> None:
    bad = [(r["id"], framing_mismatch(r), r.get("content_kind")) for r in rows
           if r.get("lesson") not in skip and framing_mismatch(r)]
    if bad:
        i, w, k = bad[0]
        raise DataError(f"{i}: the framing calls the material a {w!r} but its content_kind is {k!r}; "
                        f"{len(bad)} mismatched framing(s) in all")


def reused_material(rows: list[dict], skip=PRE_AMENDMENT_LESSONS) -> list:
    """(lesson, ids) for each material used by more than one item of a lesson, unless by exactly two items that
    share a `pair` id."""
    seen: dict = {}
    for r in rows:
        m = ws(r.get("material") or "").lower()
        if r.get("lesson") in skip or not m:
            continue
        seen.setdefault((r["lesson"], m), []).append(r)
    bad = []
    for (lesson, _), items in seen.items():
        pairs = {it.get("pair") for it in items}
        if len(items) == 1 or (len(items) == 2 and len(pairs) == 1 and None not in pairs):
            continue
        bad.append((lesson, [it["id"] for it in items]))
    return bad


def check_material_reuse(rows: list[dict], skip=PRE_AMENDMENT_LESSONS) -> None:
    bad = reused_material(rows, skip)
    if bad:
        lesson, ids = max(bad, key=lambda x: len(x[1]))
        raise DataError(f"{lesson}: the same material in {len(ids)} items ({', '.join(ids[:6])}); a material may "
                        f"appear in at most 2 items, and only as a contrast pair; {len(bad)} reused material(s) in all")


COUNT_WORDS = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
ORDER_CUE = re.compile(r"\b(left of|right of|above|below|ahead of|behind|before|after|less than|more than|"
                       r"\w+er than|higher than|lower than|purer than|faster than|older than|younger than)\b", re.I)
GATE_CUE = re.compile(r"\bonly after\b|\bonly before\b", re.I)
NOT_CUE = re.compile(r"\b(not|never)\b", re.I)
ONE_EACH = re.compile(r"\b(one each|exactly one|one per)\b", re.I)


def puzzle_signature(r: dict) -> tuple:
    """A non-verification B puzzle's structure: (entity count, constraint form). The count is the first count word
    or numeral before a plural noun (else the number of distinct capitalised names); the form is which cues the
    clue sentences use (order, gate, negation, one-each) and whether the order clues chain (one clue's object is
    the next clue's subject)."""
    text = r["turns"][0]["user"]
    m = re.search(r"\b(two|three|four|five|six|seven|eight|nine|ten|\d{1,2})\b(?:\s+[a-z-]+){0,2}?\s+[a-z]+s\b",
                  text, re.I)
    if m:
        g = m.group(1).lower()
        n = int(g) if g.isdigit() else COUNT_WORDS[g]
    else:
        n = len(set(re.findall(r"(?<![.!?]\s)(?<!^)\b[A-Z][a-z]+\b", text)))
    sents = [x for x in re.split(r"(?<=[.!?])\s+|\n", text) if x.strip()]
    order = []
    for x in sents:
        mo = ORDER_CUE.search(x)
        if mo and not GATE_CUE.search(x) and "?" not in x:
            order.append((set(re.findall(r"[a-z]{3,}", x[:mo.start()].lower())),
                          set(re.findall(r"[a-z]{3,}", x[mo.end():].lower()))))
    stop = {"the", "and", "than", "one", "its", "has", "have", "you", "may", "are", "is"}
    chain = any((order[i][1] - stop) & (order[i + 1][0] - stop) for i in range(len(order) - 1))
    form = tuple(name for name, ok in (("order", bool(order)), ("chain", chain),
                                       ("gate", bool(GATE_CUE.search(text))),
                                       ("negation", bool(NOT_CUE.search(text))),
                                       ("one-each", bool(ONE_EACH.search(text)))) if ok)
    return n, form


def _is_puzzle(r: dict) -> bool:
    """A non-verification B puzzle: tagged "non-verification" (R&D's T18 ruling: claim-keyed puzzles keep their
    three lines and are tagged), or, for items written before the tag, a B item whose reply has no VERDICT line."""
    if r.get("kind") != "B":
        return False
    return "non-verification" in (r.get("tags") or []) or \
        not re.search(r"(?m)^\s*VERDICT\s*:", r["turns"][-1]["reply"])


# The signature from a puzzle's sidecar model (R&D, 2026-10-11): the entity count and the multiset of fact rule
# types, so counts matter. `after` is folded into `before` (the same relation written backwards); is_not, not_adjacent
# and not_in stay apart (elimination and placement are different reasoning). The regex signature above is the
# fallback for an item without a sidecar model.
RULE_FOLD = {"after": "before"}


def model_signature(model: dict) -> tuple:
    rules = sorted(RULE_FOLD.get(f["rule"][0], f["rule"][0]) for f in model.get("facts") or [])
    return ("model", len(model.get("entities") or []), tuple(rules))


def describe_signature(sig: tuple) -> str:
    if sig and sig[0] == "model":
        return f"{sig[1]} entities, rules {' + '.join(sig[2]) or 'none'}"
    return f"{sig[0]} entities, {' + '.join(sig[1]) or 'no cues'}"


def row_sha256(line: str) -> str:
    """role-set's stamp (puzzle_check.row_sha256): the sha256 of the item's line as committed, CR/LF stripped."""
    return hashlib.sha256(line.rstrip("\r\n").encode("utf-8")).hexdigest()


def load_models(data_path) -> dict:
    """{id: model} from the `<name>.models.jsonl` sidecar beside a data file, each line's row_sha256 checked against
    its row as committed. A mismatch, or a sidecar line naming an item the file doesn't have, is refused: the model
    (and its puzzle-check) would describe a different row. No sidecar file: {} (the regex fallback applies)."""
    data_path = Path(data_path)
    side = data_path.with_name(data_path.name[:-len(".jsonl")] + ".models.jsonl")
    if not side.exists():
        return {}
    shas = {json.loads(x)["id"]: row_sha256(x)
            for x in data_path.read_text(encoding="utf-8").splitlines() if x.strip()}
    out = {}
    for x in side.read_text(encoding="utf-8").splitlines():
        if not x.strip():
            continue
        m = json.loads(x)
        if m["id"] not in shas:
            raise DataError(f"{side.name}: a model for {m['id']}, which {data_path.name} doesn't have")
        if m.get("row_sha256") != shas[m["id"]]:
            raise DataError(f"{side.name}: {m['id']}'s row_sha256 doesn't match its row as committed (the row changed "
                            "after its model was written; re-run role-set puzzle-check --stamp after review)")
        if isinstance(m.get("model"), dict):
            out[m["id"]] = m["model"]
    return out


def puzzle_sig(r: dict, models: dict | None = None) -> tuple:
    """A puzzle's structure signature: the model signature where the item has a sidecar model, else the regex
    signature. The one definition the batch gate (puzzle_repeats) and the one-at-a-time check (check_against)
    share, so the two can't disagree (R&D, 2026-10-15)."""
    m = (models or {}).get(r.get("id"))
    return model_signature(m) if m else puzzle_signature(r)


def puzzle_repeats(rows: list[dict], skip=PRE_AMENDMENT_LESSONS, models: dict | None = None) -> dict:
    """{signature: [(lesson, id)]} for the non-verification B puzzles, signed by puzzle_sig."""
    seen: dict = {}
    for r in rows:
        if r.get("lesson") not in skip and _is_puzzle(r):
            seen.setdefault(puzzle_sig(r, models), []).append((r["lesson"], r["id"]))
    return seen


def check_puzzles(rows: list[dict], skip=PRE_AMENDMENT_LESSONS, per_batch: int | None = 2,
                  models: dict | None = None) -> None:
    """No two puzzles of one lesson share a structure signature; across the rows given (a batch), at most
    per_batch share one. per_batch=None leaves out the batch cap, for a set made of several batches."""
    for sig, items in puzzle_repeats(rows, skip, models).items():
        by_lesson: dict = {}
        for lesson, i in items:
            by_lesson.setdefault(lesson, []).append(i)
        dup = next(((l, ids) for l, ids in by_lesson.items() if len(ids) > 1), None)
        if dup:
            raise DataError(f"{dup[0]}: puzzles {', '.join(dup[1])} share the structure "
                            f"{describe_signature(sig)}; vary the puzzle shape")
        if per_batch is not None and len(items) > per_batch:
            raise DataError(f"{len(items)} puzzles share the structure {describe_signature(sig)} "
                            f"({', '.join(i for _, i in items)}); at most {per_batch} per batch")


def framing_of(r: dict) -> str:
    """A C or D item's framing: its first user turn before "Material:", case and whitespace normalised."""
    if r.get("kind") not in ("C", "D"):
        return ""
    user = r["turns"][0]["user"]
    at = user.find("Material:")
    return re.sub(r"\s+", " ", user[:at]).strip().lower() if at > 0 else ""


def framing_repeats(rows: list[dict], skip=PRE_AMENDMENT_LESSONS) -> dict:
    seen: dict = {}
    for r in rows:
        f = framing_of(r) if r.get("lesson") not in skip else ""
        if f:
            seen.setdefault(f, []).append(r["id"])
    return {f: ids for f, ids in seen.items() if len(ids) > 1}


# Articles, prepositions, pronouns, auxiliaries and conjunctions: a phrase made only of these is exempt.
FUNCTION_WORDS = set("""
a an the this that these those each every some any no
of in on at to for from by with about into onto over under after before between through during without within
along across against among around behind beside up down out off near than as per via
i me my mine we us our you your he him his she her it its they them their what which who whom whose
is are was were be been being am do does did done have has had having will would shall should can could may
might must and or but if so nor yet then not
""".split())
# The nouns of the task itself: a phrase made only of these and function words ("the claim", "the material") is
# the subject matter, not a stamp.
TASK_WORDS = {"claim", "claims", "statement", "statements", "material", "line", "lines", "verdict", "text",
              "deciding", "escalate", "answer"}
MAX_PHRASE_SHARE = 0.40
MAX_FORMULA_SHARE = 0.25
DETERMINERS = {"the", "a", "an", "this", "that", "these", "those", "each", "every", "its", "their", "his", "her",
               "my", "our", "your"}


_CLOSING_LABELS = {("supported",), ("unsupported",), ("cannot", "tell")}


def _is_closing_label(line: str) -> bool:
    """True only when the line is the verdict label and nothing else. The same words inside a sentence stay."""
    return tuple(re.findall(r"[a-z']+", line.lower())) in _CLOSING_LABELS


def _target_lines(r: dict):
    """(turn index, line) for each line of the assistant turns, outside material lines, the Statement line,
    the three lines and the closing label; bracketed quotes collapsed to one token."""
    content = _content_lines(r)
    for i, t in enumerate(r["turns"]):
        reply_lines = t["reply"].splitlines()
        label_at = next((j for j in range(len(reply_lines) - 1, -1, -1) if reply_lines[j].strip()), None)
        if label_at is None or not _is_closing_label(reply_lines[label_at]):
            label_at = None
        for is_reply, lines in ((False, t["thinking"].splitlines()), (True, reply_lines)):
            for j, line in enumerate(lines):
                if is_reply and j == label_at:
                    continue
                bare = ws(line).replace('\\"', '"').strip("\"'“” ").lower()
                if not bare or THREE_LINE.match(line) or STATEMENT_RE.match(line) or bare in content:
                    continue
                yield i, re.sub(r"\[[^\]]*\]", " [q] ", line.lower())


def frequent_phrases(rows: list[dict], skip=PRE_AMENDMENT_LESSONS, max_share: float = MAX_PHRASE_SHARE) -> dict:
    """{(lesson, phrase): (turns with it, turns in lesson)} for 2-7 word phrases (with at least one word that is
    not a function word) in more than max_share of a lesson's assistant turns. Only the longest such phrase of
    an overlapping family is kept."""
    per: dict = {}
    turns: dict = {}
    for r in rows:
        if r.get("lesson") in skip:
            continue
        lesson = r["lesson"]
        grams_by_turn: dict = {}
        for i, line in _target_lines(r):
            w = re.findall(r"[a-z0-9']+|\[q\]", line)
            g = grams_by_turn.setdefault(i, set())
            for n in range(2, 8):
                for k in range(len(w) - n + 1):
                    ph = w[k:k + n]
                    if any(x not in FUNCTION_WORDS and x not in TASK_WORDS and x != "[q]" for x in ph):
                        g.add(" ".join(ph))
        turns[lesson] = turns.get(lesson, 0) + len(r["turns"])
        for g in grams_by_turn.values():
            for ph in g:
                per[(lesson, ph)] = per.get((lesson, ph), 0) + 1
    hits = {k: (v, turns[k[0]]) for k, v in per.items() if v > max_share * turns[k[0]]}
    return {k: v for k, v in hits.items()
            if not any(o[0] == k[0] and o[1] != k[1] and f" {k[1]} " in f" {o[1]} " and hits[o][0] == v[0]
                       for o in hits)}


def check_phrases(rows: list[dict], skip=PRE_AMENDMENT_LESSONS) -> None:
    fp = frequent_phrases(rows, skip)
    if fp:
        (lesson, ph), (n, total) = min(fp.items(), key=lambda kv: (-kv[1][0] / kv[1][1], kv[0]))
        raise DataError(f"{lesson}: the phrase {ph!r} is in {n} of {total} turns ({n / total:.0%}); at most "
                        f"{MAX_PHRASE_SHARE:.0%}; {len(fp)} over-used phrase(s) in all")


def _template(sentence: str) -> str:
    """A sentence with numerals as N and each determiner's noun phrase (up to two words after it) as X, and a
    leading list number dropped: "2. Step 2 is the decisive step." -> "step n is the x"."""
    w = re.findall(r"[a-z0-9']+|\[q\]", re.sub(r"^\s*\d+[.)]\s*", "", sentence.lower()))
    out: list[str] = []
    skip = 0
    for k, x in enumerate(w):
        if skip:
            skip -= 1
            continue
        if x.isdigit():
            out.append("n")
        elif x in DETERMINERS:
            out += [x, "x"]
            skip = min(2, sum(1 for y in w[k + 1:k + 3] if y not in FUNCTION_WORDS and not y.isdigit()))
            skip = skip if skip else 0
        else:
            out.append(x)
    return " ".join(out)


def formulas(rows: list[dict], skip=PRE_AMENDMENT_LESSONS, max_share: float = MAX_FORMULA_SHARE) -> dict:
    """{(lesson, template): (B items with it, B items in lesson)} for sentence templates of 4+ tokens in more than
    max_share of a lesson's kind B items."""
    per: dict = {}
    count: dict = {}
    for r in rows:
        if r.get("lesson") in skip or r.get("kind") != "B":
            continue
        count[r["lesson"]] = count.get(r["lesson"], 0) + 1
        temps = set()
        for _, line in _target_lines(r):
            for sent in re.split(r"(?<=[.!?:])\s+", line):
                t = _template(sent).split()
                if len(t) >= 4:
                    temps.add(" ".join(t))
                # the opener: a grafted formula keeps its first words while the rest varies ("Step N is …")
                opener = t[:2] if "n" in t[:2] else t[:3]
                if len(opener) >= 2 and any(x not in FUNCTION_WORDS and x not in ("n", "x", "[q]") for x in opener):
                    temps.add(" ".join(opener) + " …")
        for t in temps:
            per[(r["lesson"], t)] = per.get((r["lesson"], t), 0) + 1
    return {k: (v, count[k[0]]) for k, v in per.items() if v > max_share * count[k[0]] and v >= 2}  # a formula repeats


def check_formulas(rows: list[dict], skip=PRE_AMENDMENT_LESSONS) -> None:
    fm = formulas(rows, skip)
    if fm:
        (lesson, t), (n, total) = min(fm.items(), key=lambda kv: (-kv[1][0], kv[0]))
        raise DataError(f"{lesson}: the sentence template {t!r} is in {n} of {total} kind B items; at most "
                        f"{MAX_FORMULA_SHARE:.0%}; {len(fm)} formula(s) in all")


# ---------------------------------------------------------------- kind A lesson drift (a figure, never a refusal)
# Term overlap with the trait lessons was probed on the pilot, batch 2 and batch 3 (280 items): an item's own
# lesson was strictly nearest for only 15%, mostly by design (C teaches through another trait's failure; B and D
# talk about the material). So it is reported for kind A only, as a reviewer's ranking, and refuses nothing.
_DRIFT_STOP = {"verifier", "verifiers", "trait", "also", "just", "only", "one", "two"}


def trait_sections(markdown: str) -> dict:
    """{lesson: text} from the trait-lesson file, split at its numbered headings."""
    heads = list(LESSON_HEADING.finditer(markdown))
    out = {}
    for i, m in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(markdown)
        out[f"T{int(m.group(1)):02d}"] = markdown[m.start():end].split("\n## What a verifier doesn't do")[0]
    return out


def _drift_terms(text: str) -> list:
    out = []
    for w in re.findall(r"[a-z][a-z']+", text.lower()):
        w = w.strip("'")
        if len(w) < 3 or w in FUNCTION_WORDS or w in TASK_WORDS or w in _DRIFT_STOP:
            continue
        for suf in ("ing", "ed", "es", "s", "ly"):
            if w.endswith(suf) and len(w) - len(suf) >= 4:
                w = w[: -len(suf)]
                break
        out.append(w)
    return out


def _tfidf(counts: dict, idf: dict, default: float) -> dict:
    v = {w: (1 + math.log(n)) * idf.get(w, default) for w, n in counts.items()}
    norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
    return {w: x / norm for w, x in v.items()}


def _count(terms: list) -> dict:
    c: dict = {}
    for t in terms:
        c[t] = c.get(t, 0) + 1
    return c


def a_lesson_ranks(rows: list[dict], lessons: dict) -> dict:
    """For each kind A item: the rank (1 = nearest) of its own lesson among the trait lessons by TF-IDF cosine of
    its passage and reply, and the nearest lesson. A figure for reviewers, never a refusal."""
    docs = {k: _count(_drift_terms(v)) for k, v in lessons.items()}
    n = len(docs)
    df: dict = {}
    for c in docs.values():
        for w in c:
            df[w] = df.get(w, 0) + 1
    idf = {w: math.log((1 + n) / (1 + d)) + 1 for w, d in df.items()}
    default = math.log(1 + n) + 1
    vecs = {k: _tfidf(c, idf, default) for k, c in docs.items()}
    out = {}
    for r in rows:
        if r.get("kind") != "A" or r.get("lesson") not in vecs:
            continue
        text = "\n".join(t["user"] + "\n" + t["reply"] for t in r["turns"])
        v = _tfidf(_count(_drift_terms(text)), idf, default)
        sims = sorted(((sum(v.get(w, 0.0) * x for w, x in lv.items()), k) for k, lv in vecs.items()),
                      key=lambda p: (-p[0], p[1]))
        order = [k for _, k in sims]
        out[r["id"]] = {"rank": order.index(r["lesson"]) + 1, "nearest": order[0]}
    return out


def batch_report(rows: list[dict], skip=PRE_AMENDMENT_LESSONS, lessons=None) -> dict:
    """Every diversity figure for a set of rows, refusing nothing (for review and for briefs). With `lessons`
    ({lesson: text}), it adds the kind A lesson-drift ranking."""
    rep = {"shared_spans": len(shared_spans(rows, skip)),
            "framing_mismatches": sum(bool(framing_mismatch(r)) for r in rows if r.get("lesson") not in skip),
            "reused_materials": len(reused_material(rows, skip)),
            "puzzle_signatures": {f"{k[0]} / {'+'.join(k[1])}": [i for _, i in v]
                                  for k, v in puzzle_repeats(rows, skip).items()},
            "repeated_framings": len(framing_repeats(rows, skip)),
            "closed_form_letters": closed_form_letters(rows)}
    if lessons:
        ranks = a_lesson_ranks(rows, lessons)
        rep["a_lesson_rank"] = {
            "own_nearest": sum(1 for x in ranks.values() if x["rank"] == 1),
            "own_in_top3": sum(1 for x in ranks.values() if x["rank"] <= 3),
            "items": len(ranks),
            "outside_top3": sorted([i, x["rank"], x["nearest"]] for i, x in ranks.items() if x["rank"] > 3)}
    return rep


def closed_form_letters(rows: list[dict]) -> dict:
    """{lesson: {letter: count}} of the keyed letters of closed-form written variants (amendment 5: vary the
    keyed letter across a lesson's items). A figure for review, refusing nothing."""
    out: dict = {}
    for r in rows:
        v = r.get("variant") or {}
        if r.get("kind") == "E" and isinstance(v.get("key"), dict) and "answer" in v["key"]:
            d = out.setdefault(r.get("lesson"), {})
            d[v["key"]["answer"]] = d.get(v["key"]["answer"], 0) + 1
    return {k: dict(sorted(v.items())) for k, v in sorted(out.items())}


def freeze_report(rows: list[dict], exempt=PRE_AMENDMENT_LESSONS) -> dict:
    """The whole-dataset check before the freeze, over every file together. Across all lessons: cannot_tell and
    supported each at least 25% of verdict-bearing items, and the multi-line share (2-4 DECIDING lines among
    supported/unsupported) at least 25%; every lesson has an ESCALATE yes item (the exempt lessons are reported,
    not refused). Returns the figures; raises DataError listing every shortfall."""
    n = ct = sup = vb = multi = 0
    esc: dict = {}
    for r in rows:
        keys = _verdict_keys(r)
        if not isinstance(keys, dict):
            continue
        v = keys.get("verdict")
        n += 1
        ct += v == "cannot_tell"
        sup += v == "supported"
        if v in ("supported", "unsupported"):
            vb += 1
            multi += 2 <= len(keys.get("deciding") or []) <= 4
        esc[r["lesson"]] = esc.get(r["lesson"], 0) + (keys.get("escalate") or "").startswith("yes (")
    rep = {"verdict_bearing": n, "cannot_tell": ct, "supported": sup, "supported_or_unsupported": vb,
           "multi_line": multi, "escalate_yes_per_lesson": dict(sorted(esc.items())),
           "lessons_without_escalate_yes": sorted(k for k, v in esc.items() if not v)}
    short = []
    for name, k, total in (("cannot_tell", ct, n), ("supported", sup, n), ("multi-line share", multi, vb)):
        if total and k < MIN_VERDICT_SHARE * total:
            need = -(-int(MIN_VERDICT_SHARE * 100) * total // 100) - k
            short.append(f"{name} {k} of {total} ({k / total:.0%}); {need} more needed for 25%")
    refused = [x for x in rep["lessons_without_escalate_yes"] if x not in exempt]
    if refused:
        short.append(f"no ESCALATE yes item in {', '.join(refused)}")
    rep["exempt_without_escalate_yes"] = [x for x in rep["lessons_without_escalate_yes"] if x in exempt]
    repeats = framing_repeats(rows, exempt)
    rep["repeated_framings"] = len(repeats)
    if repeats:
        f, ids = next(iter(repeats.items()))
        short.append(f"{len(repeats)} framing line(s) repeated across files, e.g. {f!r} in {', '.join(ids[:5])}")
    rep["shortfalls"] = short
    if short:
        raise DataError("freeze: " + "; ".join(short) + f" (figures: {json.dumps(rep)})")
    return rep


def role_words_in_prompt(r: dict) -> list[str]:
    """Banned words in a C, D or verdict-type E item's user turns, outside the material, new material and the
    statement line (those are the content being checked, shared by both arms as they are)."""
    if not (r.get("kind") in ("C", "D") or (r.get("kind") == "E" and (r.get("quiz") or {}).get("type") == "verdict")):
        return []
    content = {x.strip() for x in (r.get("material") or "").splitlines() if x.strip()}
    found = []
    prompts = [(t["user"], t.get("new_material", [])) for t in r["turns"]]
    if r.get("kind") == "E" and isinstance(r.get("variant"), dict):
        prompts.append((r["variant"].get("user") or "", []))  # the re-asked variant is a prompt too
    for user, new in prompts:
        content |= {x.strip() for x in new if x.strip()}
        for line in user.splitlines():
            bare = ws(line).replace('\\"', '"').strip("\"'“” ")
            if line.strip() in content or re.match(r"\s*statement\s*:", line, re.I) or \
                    (bare and bare in ws("\n".join(content))):  # a material line, possibly quoted
                continue
            found += [m.group(0) for m in ROLE_WORDS.finditer(line)]
    return found


def _check_keys(where, keys, material):
    if keys.get("verdict") not in VERDICTS:
        raise DataError(f"{where}: verdict {keys.get('verdict')!r}")
    deciding = keys.get("deciding")
    if not isinstance(deciding, list) or len(deciding) > 4:
        raise DataError(f"{where}: deciding must be a list of at most 4 lines")
    if (keys["verdict"] == "cannot_tell") != (not deciding):
        raise DataError(f"{where}: cannot_tell exactly when deciding is empty")
    for d in deciding:
        if not isinstance(d, str) or not d.strip() or ws(d) not in ws(material or ""):
            raise DataError(f"{where}: deciding line not verbatim in the material (markup such as ** kept): {d!r}")
    if not isinstance(keys.get("escalate"), str) or not ESC.match(keys["escalate"]):
        raise DataError(f"{where}: escalate {keys.get('escalate')!r} (only no, or yes (medical|legal|safety|money ...))")


def _reply_matches(where, reply, keys):
    p = parse_answer(reply)
    if p["verdict"] != keys["verdict"]:
        raise DataError(f"{where}: reply's VERDICT {p['verdict']!r} != key {keys['verdict']!r}")
    raw = (p["deciding"] or "").strip()
    parts = [x.strip() for x in raw.split(" | ") if x.strip()]
    if len(parts) > 1 and any(x.upper() == "NONE" for x in parts):
        raise DataError(f"{where}: reply's DECIDING puts NONE beside other lines; NONE stands alone, for cannot_tell only")
    last_key = keys["deciding"][-1].strip() if keys["deciding"] else ""
    if raw.endswith("|") and not last_key.endswith("|"):
        raise DataError(f"{where}: reply's DECIDING ends with a trailing separator \"|\"")
    given = "" if raw.upper() == "NONE" else ws(raw)
    if given != ws(" | ".join(keys["deciding"])):
        raise DataError(f"{where}: reply's DECIDING differs from the key (exact, markup kept)")
    if (p["escalate"] or "").strip() != keys["escalate"]:
        raise DataError(f"{where}: reply's ESCALATE {p['escalate']!r} != key {keys['escalate']!r}")


# Every non-empty line of an item's `material` appears in its first user turn (R&D, 2026-10-11: batch 6's rebuilt
# verdict B items kept an older item's material, and leakage and source checks read that field). Two older items
# are exempt by id, each with its reason (R&D's ruling).
MATERIAL_IN_TURN_EXEMPT = {
    "T01-C-02": "batch 2: the turn quotes the material with escaped quotation marks; the lines are the same text",
    "T04-B-04": "pilot: the material is written as sentences, the turn shows the same rules as bullets",
}


def _is_apply(r: dict) -> bool:
    return r.get("split") == "apply" or "-AP-" in str(r.get("id", ""))


def check_item(r: dict, n: int = 1, pre_amendment=PRE_AMENDMENT_LESSONS, seen=None, closed_form=None) -> None:
    """Every check on one row that needs no other row. `n` is its line number in the file (for messages);
    `seen`, when given, is the set of ids before it (the duplicate-id check)."""
    where = f"line {n} ({r.get('id', '?')})"
    if _is_apply(r):  # Argon's finding 7.2
        raise DataError(f"{where}: an apply item (split apply or an -AP- id) is never trained")
    if r.get("split") == "star":
        raise DataError(f"{where}: a STaR row is checked by validate_training_set's star check, not check_item")
    _check_item_body(r, where, pre_amendment, seen, closed_form)


# The apply items' twin fields, by lesson (aspire-si#127, #137; R&D's rulings on #128): T10 carries a keyed companion
# claim, T09 whether its lines can be reversed, and T01, T04 and T14 whether dropping the deciding lines leaves the
# material silent on the statement.
SILENT_TWIN_LESSONS = ("T01", "T04", "T14")


def check_apply_item(r: dict, n: int = 1, pre_amendment=PRE_AMENDMENT_LESSONS, seen=None, closed_form=None) -> None:
    """An apply item: asked once after each session and never trained (aspire-si#127). The same content and shape
    checks check_item runs (one shared body, so the two can't drift), without its never-trained refusal, plus the apply
    rules. A row that isn't an apply item is refused, so this can never wave a training row through; check_item and
    validate_training_set still refuse apply rows (R&D ruling, 2026-10-17)."""
    where = f"line {n} ({r.get('id', '?')})"
    if r.get("split") != "apply" or "-AP-" not in str(r.get("id", "")):
        raise DataError(f"{where}: check_apply_item takes only apply items (split apply and an -AP- id); training rows "
                        "go through check_item")
    if r.get("kind") not in ("C", "D"):
        raise DataError(f"{where}: an apply item is kind C or D, not {r.get('kind')!r}")
    if not isinstance(r.get("turns"), list) or len(r["turns"]) != 1 or r.get("turn_keys"):
        raise DataError(f"{where}: an apply item is one turn: the model is asked its first user turn once")
    if r.get("pair"):
        raise DataError(f"{where}: an apply item carries no contrast pair")
    lesson, verdict = r.get("lesson"), r.get("verdict")
    if lesson == "T10":
        twin = r.get("r10_twin")
        if not isinstance(twin, dict) or not isinstance(twin.get("statement"), str) or not twin["statement"].strip() \
                or twin.get("verdict") not in VERDICTS:
            raise DataError(f"{where}: a T10 apply item needs r10_twin {{statement, verdict}}")
        if twin["verdict"] == verdict:
            raise DataError(f"{where}: r10_twin's verdict must differ from the item's ({verdict})")
        lines = [ln for ln in r["turns"][0].get("user", "").split(chr(10)) if re.match(r"\s*Statement:", ln)]
        if len(lines) != 1:  # the twin swaps that one line (Argon's finding 6.4)
            raise DataError(f"{where}: a T10 apply item needs exactly one Statement line, not {len(lines)}")
    if lesson == "T09" and not isinstance(r.get("order_free"), bool):
        raise DataError(f"{where}: a T09 apply item needs order_free (true or false)")
    if lesson in SILENT_TWIN_LESSONS:
        # As the apply sitting's own schema check has it (docs/runs/stage1-train/apply.py): a decided item needs the
        # flag; a cannot_tell item has no deciding lines, so it may leave the flag out but never set it true.
        silent = r.get("silent_without_deciding")
        if verdict in ("supported", "unsupported") and not isinstance(silent, bool):
            raise DataError(f"{where}: a decided {lesson} apply item needs silent_without_deciding (true or false)")
        if silent is True and verdict == "cannot_tell":
            raise DataError(f"{where}: silent_without_deciding is never true on a cannot_tell item; it has no deciding "
                            "lines to drop")
    _check_item_body(r, where, pre_amendment, seen, closed_form)


def _check_item_body(r: dict, where: str, pre_amendment, seen, closed_form) -> None:
    """The content and shape checks every item gets, training or apply."""
    for k, typ in REQUIRED.items():
        if not isinstance(r.get(k), typ):
            raise DataError(f"{where}: field {k} missing or not {typ.__name__}")
    if seen is not None:
        if r["id"] in seen:
            raise DataError(f"{where}: duplicate id")
        seen.add(r["id"])
    if r["lesson"] not in LESSONS or r["kind"] not in KINDS:
        raise DataError(f"{where}: unknown lesson {r['lesson']} or kind {r['kind']}")
    if r["tier"] < 1 or r["key_basis"] not in ("construction", "source") or not r["turns"]:
        raise DataError(f"{where}: tier, key_basis or turns")
    material = r.get("material") or ""
    if r["id"] not in MATERIAL_IN_TURN_EXEMPT:
        shown = ws(r["turns"][0]["user"])
        missing = [line.strip() for line in material.splitlines() if line.strip() and ws(line) not in shown]
        if missing:
            raise DataError(f"{where}: material line not in the first user turn: {missing[0][:80]!r} (the material "
                            "field must be what the turn shows)")
    available = []  # material available at each turn: the item's material plus new lines brought so far
    for i, t in enumerate(r["turns"], 1):
        for k in ("user", "thinking", "reply"):
            if not isinstance(t.get(k), str):
                raise DataError(f"{where} turn {i}: field {k} missing")
        if not t["reply"].strip() or "<think>" in t["thinking"] + t["reply"] or "</think>" in t["thinking"] + t["reply"]:
            raise DataError(f"{where} turn {i}: empty reply or <think> tags")
        for nm in t.get("new_material", []):
            if nm not in t["user"]:
                raise DataError(f"{where} turn {i}: new_material not in the user message")
        material = material + "\n" + "\n".join(t.get("new_material", []))
        available.append(material)
    last = r["turns"][-1]["reply"]
    kind = r["kind"]
    first_user = r["turns"][0]["user"]
    # provenance on every item: real material cited, invented parts marked (the handoff's table)
    if not r["sources"]:
        raise DataError(f"{where}: sources is empty (cite real material; mark invented parts as invented)")
    checked = [_source_ok(x) for x in r["sources"]]
    if not all(ok for ok, _ in checked):
        raise DataError(f"{where}: a sources entry is neither repo@commit nor marked invented")
    has_real = any(real for _, real in checked)
    stmt = _statement(first_user)
    if kind in ("C", "D", "E") and stmt and TIME_WORDS.search(stmt) and not DATED.search(r.get("material") or ""):
        raise DataError(f"{where}: undated time word in the statement ({TIME_WORDS.search(stmt).group(0)!r})")
    if kind == "A":
        question_at = first_user.rfind("?")
        before = first_user[:first_user.rfind(".", 0, question_at) + 1] if question_at > 0 else first_user
        if _sentences(before) < 2:
            raise DataError(f"{where}: kind A needs a teaching passage of at least 2 sentences before the question")
        names = TRAIT_NAMES.get(r["lesson"], [])
        if names and not any(n in first_user.lower() for n in names):
            raise DataError(f"{where}: kind A's passage doesn't name its trait ({r['lesson']})")
        if not re.match(r"^\W*(as a verifier,? i\b|i\b|i'm\b|i've\b|my\b)", last.strip().lower()):
            raise DataError(f"{where}: kind A's reply isn't in the first person")
        if re.search(r"(?m)^\s*VERDICT\s*:", last):
            raise DataError(f"{where}: kind A's reply has a VERDICT line")
    if kind == "B":
        drawn = "\n".join(t["thinking"] + "\n" + t["reply"] for t in r["turns"])
        if not any(rx.search(drawn) for rx in DRAWN):
            raise DataError(f"{where}: kind B draws no pattern (tree glyphs, a grid table or numbered elimination)")
    if kind == "C":
        if not re.search(r"\b(verdict|supported|unsupported|cannot_tell|can't tell)\b", first_user, re.I):
            raise DataError(f"{where}: kind C's user turn has no checker verdict")
        if REPO_CITATION.search((r.get("material") or "") + "\n" + first_user) and not has_real:
            raise DataError(f"{where}: kind C cites a repository but sources has no real (non-invented) entry")
    if kind == "A" and not r.get("key_points"):
        raise DataError(f"{where}: kind A needs key_points")
    if kind == "B" and not r.get("answer"):
        raise DataError(f"{where}: kind B needs answer")
    if r["lesson"] not in pre_amendment and role_words_in_prompt(r):
        raise DataError(f"{where}: the prompt names the role, a trait or a pattern ({role_words_in_prompt(r)[0]!r}) "
                        f"outside the material and the statement; prompts are role-neutral")
    if kind in ("C", "D"):
        if kind == "C" and r.get("planted_flaw") not in FLAWS:
            raise DataError(f"{where}: planted_flaw {r.get('planted_flaw')!r}")
        if kind == "D" and r.get("material") is None:
            raise DataError(f"{where}: kind D needs material")
        _check_keys(where, r, available[-1])
        _reply_matches(where, last, r)
        if r["lesson"] not in pre_amendment:
            _check_quoting(where, r, available[-1], r["content_kind"])
        for idx, keys in (r.get("turn_keys") or {}).items():
            i = int(idx)
            _check_keys(f"{where} turn {i}", keys, available[i - 1])
            _reply_matches(f"{where} turn {i}", r["turns"][i - 1]["reply"], keys)
            if r["lesson"] not in pre_amendment:
                _check_quoting(f"{where} turn {i}", keys, available[i - 1], r["content_kind"])
    if kind == "E":
        q, v = r.get("quiz"), r.get("variant")
        if not isinstance(q, dict) or q.get("type") not in ("verdict", "written"):
            raise DataError(f"{where}: quiz must be {{type: verdict|written, ...}}")
        if q["type"] == "verdict":
            _check_keys(where, q, available[-1])
            _reply_matches(where, last, q)
            if r["lesson"] not in pre_amendment:
                _check_quoting(where, q, available[-1], r["content_kind"])
        elif not q.get("key_points"):
            raise DataError(f"{where}: written quiz needs key_points")
        if not isinstance(v, dict) or not v.get("user") or v["user"] == r["turns"][0]["user"] or "key" not in v:
            raise DataError(f"{where}: variant must be a re-worded {{user, key}}")
        original = _key_points(q)
        if q["type"] == "written" and isinstance(v["key"], dict) and "answer" in v["key"]:
            _check_closed_form(where, v, original, r, closed_form_settings(closed_form))
            return
        variant = " ".join(_key_points(v["key"]))
        if not any(k and (k in variant or variant in k) for k in original):
            raise DataError(f"{where}: the variant's key shares no key point with the original")


OPTION_RX = re.compile(r"^\s*\(?([A-D])[.)]\s+(\S.*)$")
ANSWER_RX = re.compile(r"^\s*ANSWER:\s*\(?([A-D])\)?\s*[.]?\s*$", re.M)


def closed_form_options(user: str) -> dict:
    """{letter: option text} for a closed-form variant: lines that start "A." or "A)" (or "(A)"), A to D."""
    out = {}
    for line in user.splitlines():
        m = OPTION_RX.match(line)
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


def parse_choice(answer: str) -> str | None:
    """The letter on the last ANSWER line of a reply to a closed-form variant, or None."""
    found = ANSWER_RX.findall(answer or "")
    return found[-1] if found else None


# The closed-form settings (the maintainer: the configurable option wins; the defaults are R&D's ruling (b),
# 2026-10-11). carry: how the keyed option carries the item's key point:
#   "verbatim"   the keyed option contains a key point, or is contained in one, after normalising (amendment 5 as
#                first written); no distractor does;
#   "paraphrase" the keyed option holds at least `share` of a key point's content words (lightly stemmed, function
#                and task words dropped; the keyed share is its best over the item's key points); every distractor's
#                share on every key point is strictly below the keyed share minus `distractor_margin` (no closeness
#                inversion) and below `distractor_max` (a near-copy guard); and no option shares a `span`-word span
#                with a trained turn (the item's own here; any item's in check_closed_form_spans);
#   "either"     the keyed option passes verbatim or paraphrase; distractors fail both; the span rule applies to
#                every option except a keyed option that passes verbatim.
# Provenance (R&D, 2026-10-11): share 0.5 and distractor_max 0.8 from batch 5's measured shares; distractor_margin
# 0.15 from batch 5's paraphrased round (6a01c18): smallest keyed-to-top-distractor gap 0.21, median 0.50, set below
# the minimum to leave headroom; re-measured after batch 6 and raised if the minimum gap stays above 0.2.
# distractor_margin 0.20 (was 0.15): R&D ruling 2026-10-15, from the batch 5+6 re-measure at 0c181e4. Over 24
# closed-form written-E items, minimum gap 0.214 (T09-E-07), batch 6 minimum 0.223, median 0.437. Re-measured at the
# batch 8 freeze. There is
# no floor on distractor share: on-topic near-misses can share none of the point's words (off-topic stays a
# key-check judgement).
CLOSED_FORM_DEFAULT = {"carry": "paraphrase", "share": 0.5, "distractor_max": 0.8, "distractor_margin": 0.2,
                       "span": 8, "exempt_keyed_key_points": False}  # R&D: 2026-10-11 from batch 5's shares; margin 2026-10-15
CARRY_MODES = ("verbatim", "paraphrase", "either")


def closed_form_settings(cfg=None) -> dict:
    out = {**CLOSED_FORM_DEFAULT, **(cfg or {})}
    if out["carry"] not in CARRY_MODES or not (0 < out["share"] <= 1) or not (0 < out["distractor_max"] <= 1) \
            or not (0 <= out["distractor_margin"] < 1) or int(out["span"]) < 2:
        raise ValueError(f"closed-form settings out of range: {out}")
    out["span"] = int(out["span"])
    return out


SUFFIXES = ("ing", "ed", "es", "s", "ly")


def content_stems(text: str) -> set:
    """Content words, lightly stemmed: function words and the task's own nouns dropped, one common suffix cut."""
    out = set()
    for w in re.findall(r"[a-z0-9']+", text.lower()):
        if w in FUNCTION_WORDS or w in TASK_WORDS:
            continue
        for suf in SUFFIXES:
            if w.endswith(suf) and len(w) - len(suf) >= 3:
                w = w[:-len(suf)]
                break
        out.add(w)
    return out


def key_point_share(key_point: str, option: str) -> float:
    k = content_stems(key_point)
    return len(k & content_stems(option)) / len(k) if k else 0.0


def word_spans(text: str, n: int) -> set:
    w = re.findall(r"[a-z0-9']+", text.lower())
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def trained_turn_spans(rows, n: int) -> set:
    out: set = set()
    for r in rows:
        for t in r.get("turns") or []:
            for k in ("user", "thinking", "reply"):
                out |= word_spans(t.get(k) or "", n)
    return out


def _check_closed_form(where: str, v: dict, original: list, item: dict | None = None, cfg: dict | None = None):
    """A written E item's closed-form variant (R&D, 2026-10-11): 3 or 4 options labelled A, B, C(, D) on their
    own lines, an instruction to answer with an ANSWER line, key {"answer": "<letter>"}, exactly one option
    carrying a key point of the original (as `cfg["carry"]` sets), and nothing else in the variant (no material:
    it stands alone)."""
    cfg = closed_form_settings(cfg)
    extra = set(v) - {"user", "key"}
    if extra:
        raise DataError(f"{where}: a closed-form variant is {{user, key}} only (it stands alone), not {sorted(extra)}")
    if set(v["key"]) != {"answer"}:
        raise DataError(f"{where}: a closed-form key is {{\"answer\": \"<letter>\"}} only")
    opts = closed_form_options(v["user"])
    letters = sorted(opts)
    if len(opts) not in (3, 4) or letters != list("ABCD"[:len(opts)]):
        raise DataError(f"{where}: a closed-form variant needs 3 or 4 options labelled A, B, C(, D), got {letters}")
    if "ANSWER:" not in v["user"]:
        raise DataError(f"{where}: a closed-form variant asks for the letter on an ANSWER: line")
    texts = [norm(t).lower() for t in opts.values()]
    if len(set(texts)) != len(texts):
        raise DataError(f"{where}: two closed-form options are the same after normalising")
    key = v["key"]["answer"]
    if key not in opts:
        raise DataError(f"{where}: the closed-form key {key!r} isn't one of the options {letters}")

    def verbatim(t):
        t = norm(t).lower()
        return any(k and (k in t or t in k) for k in original)

    def share(t):
        return max((key_point_share(k, t) for k in original), default=0.0)

    mode = cfg["carry"]
    keyed_verbatim, keyed_share = verbatim(opts[key]), share(opts[key])
    keyed_ok = {"verbatim": keyed_verbatim, "paraphrase": keyed_share >= cfg["share"],
                "either": keyed_verbatim or keyed_share >= cfg["share"]}[mode]
    if not keyed_ok:
        raise DataError(f"{where}: the keyed option doesn't carry a key point of the original "
                        f"(carry {mode}: verbatim {keyed_verbatim}, content-word share {keyed_share:.2f}, "
                        f"needed {cfg['share']})")
    others = []
    for x in letters:
        if x == key:
            continue
        d = share(opts[x])
        if (mode in ("verbatim", "either") and verbatim(opts[x])) or \
                (mode in ("paraphrase", "either") and (d >= cfg["distractor_max"]
                                                       or d >= keyed_share - cfg["distractor_margin"])):
            others.append(x)
    if others:
        raise DataError(f"{where}: options {others} also carry a key point of the original (carry {mode}: a "
                        f"distractor's share must be under {cfg['distractor_max']} and under the keyed share "
                        f"{keyed_share:.2f} minus {cfg['distractor_margin']}): exactly one is keyed")
    if mode != "verbatim" and item is not None:
        trained = trained_turn_spans([item], cfg["span"])
        hit = [x for x in letters if not (mode == "either" and x == key and keyed_verbatim)
               and word_spans(opts[x], cfg["span"]) & trained]
        if hit:
            raise DataError(f"{where}: options {hit} share a {cfg['span']}-word span with the item's trained turn")


def closed_form_figures(rows: list[dict]) -> list:
    """Per closed-form item: the keyed option's share (its best over the key points) and the highest distractor
    share, for the diagnostics; R&D sets distractor_margin from their distribution."""
    out = []
    for r in rows:
        v = r.get("variant") or {}
        if r.get("kind") != "E" or not isinstance(v.get("key"), dict) or "answer" not in v["key"]:
            continue
        kps = (r.get("quiz") or {}).get("key_points") or []
        shares = {x: max((key_point_share(k, t) for k in kps), default=0.0)
                  for x, t in closed_form_options(v["user"]).items()}
        key = v["key"]["answer"]
        out.append({"id": r["id"], "keyed_share": round(shares.get(key, 0.0), 3),
                    "top_distractor_share": round(max((s_ for x, s_ in shares.items() if x != key), default=0.0), 3)})
    return out


def variant_checked_spans(r: dict, v: dict, cfg: dict) -> set:
    """The spans of a variant that the spent check compares with trained turns (the same rule as quiz.py's): the
    whole user text, minus the item's own material and new_material; with exempt_keyed_key_points on, a
    closed-form keyed option is checked without the item's own key points (the stem and distractors in full)."""
    n = cfg["span"]
    text = v.get("user") or ""
    if cfg["exempt_keyed_key_points"] and isinstance(v.get("key"), dict) and "answer" in v["key"]:
        letter = v["key"]["answer"]
        keyed = closed_form_options(text).get(letter, "")
        rest = "\n".join(line for line in text.splitlines()
                         if not (line.lstrip().lstrip("(").startswith(letter) and keyed and keyed in line))
        kp_spans: set = set()
        for kp in (r.get("quiz") or {}).get("key_points") or []:
            kp_spans |= word_spans(kp, n)
        out = word_spans(rest, n) | (word_spans(keyed, n) - kp_spans)
    else:
        out = word_spans(text, n)
    material = word_spans(r.get("material") or "", n)
    for t in r.get("turns") or []:
        nm = t.get("new_material") or ""
        material |= word_spans("\n".join(nm) if isinstance(nm, list) else nm, n)
    return out - material


def check_closed_form_spans(rows: list[dict], cfg=None) -> None:
    """The quiz's spent check, run at authoring (R&D on #105): every quizzable kind-E variant (verdict, or written
    in closed form) must share no `span`-word span with any trained turn of any item, outside the item's own
    material (and, if exempt_keyed_key_points is on, the item's key points inside its keyed option). A variant
    that fails would be spent at its first quiz, so it is refused here. Free-text written variants aren't quizzed
    and are left out."""
    cfg = closed_form_settings(cfg)
    trained = trained_turn_spans(rows, cfg["span"])
    for r in rows:
        v = r.get("variant") or {}
        if r.get("kind") != "E" or not v.get("user"):
            continue
        q = r.get("quiz") or {}
        closed = isinstance(v.get("key"), dict) and "answer" in v["key"]
        if q.get("type") == "written" and not closed:
            continue
        if variant_checked_spans(r, v, cfg) & trained:
            raise DataError(f"{r['id']}: its variant shares a {cfg['span']}-word span with a trained turn, so it "
                            "would be spent at its first quiz")


def _checker_trace(r: dict) -> str:
    if isinstance(r.get("checker_trace"), str):
        return r["checker_trace"]
    turns = r.get("turns") or []
    user = turns[0].get("user", "") if turns and isinstance(turns[0], dict) else ""
    m = re.search(r"(?m)^\s*statement\s*:\s*.+$", user, re.I)
    rest = user[m.end():] if m else ""
    rest = re.sub(r"(?m)^\s*Checker verdict:.*$", "", rest, flags=re.I)
    return rest.strip()


def _strip_quoted_material_or_statement(trace: str, material: str, statement: str) -> str:
    norm_mat = ws(material).lower()
    norm_stmt = ws(statement).lower()

    def _repl(m: re.Match) -> str:
        inner = next((g for g in m.groups() if g is not None), "")
        norm_inner = ws(inner).lower()
        if len(norm_inner) >= 4 and (norm_inner in norm_mat or norm_inner in norm_stmt):
            return " "
        return m.group(0)

    return _QUOTED_TRACE_SPAN_RE.sub(_repl, trace)


def check_trace_leaks_label(row: dict) -> None:
    """Refuse a kind C row whose checker trace names its own planted flaw, or any flaw stem outside the carve-out."""
    if row.get("kind") != "C":
        return
    trace = _checker_trace(row)
    if not trace:
        return
    turns = row.get("turns") or []
    first_user = turns[0].get("user", "") if turns and isinstance(turns[0], dict) else ""
    stmt = row.get("statement") if isinstance(row.get("statement"), str) else _statement(first_user)
    material = (row.get("material") or "") + "\n" + "\n".join(
        x for t in turns if isinstance(t, dict) for x in (t.get("new_material") or [])
    )
    cleaned = _strip_quoted_material_or_statement(trace, material, stmt)
    own = row.get("planted_flaw")
    mat_and_stmt = f"{material}\n{stmt}"
    for m in TRACE_LEAK_RE.finditer(cleaned):
        word = m.group(0).strip(" -")
        flaw = next((f for f, rx in FLAW_STEMS.items() if rx.fullmatch(word)), None)
        if flaw is None:  # pragma: no cover
            continue
        if flaw != own and FLAW_STEMS[flaw].search(mat_and_stmt):
            continue
        rid = row.get("id", "?")
        code = "TRACE_LEAKS_LABEL"
        hint = "show the flaw in the checker's reasoning without naming a flaw label or stem"
        msg = f"{rid}: {code}: checker trace leaks flaw label {word!r} (stem '{flaw}'); hint: {hint}"
        raise DataError(msg, code=code, hint=hint)


def check_trace_leaks(rows: list[dict], skip=PRE_AMENDMENT_LESSONS, exempt=TRACE_LEAK_EXEMPT) -> None:
    for r in rows:
        if r.get("lesson") in skip or r.get("id") in exempt:
            continue
        check_trace_leaks_label(r)


def check_batch(rows: list[dict], pre_amendment=PRE_AMENDMENT_LESSONS, batch_caps: bool = True,
                closed_form=None, models: dict | None = None) -> None:
    """Every check that needs the whole set: stamps, frames, shares, floors, pairs and the diversity gate.
    batch_caps=False leaves out the checks that are per batch by design (the puzzle-signature cap), for a set
    made of several batches; everything per lesson still runs."""
    check_stamps(rows)
    check_frames(rows)
    check_multi_line_share(rows, skip=pre_amendment)
    check_floors(rows, skip=pre_amendment)
    check_pairs(rows, skip=pre_amendment)
    check_spans(rows, skip=pre_amendment)
    check_framing_fits(rows, skip=pre_amendment)
    check_material_reuse(rows, skip=pre_amendment)
    check_puzzles(rows, skip=pre_amendment, per_batch=2 if batch_caps else None, models=models)
    check_phrases(rows, skip=pre_amendment)
    check_formulas(rows, skip=pre_amendment)
    check_closed_form_spans(rows, closed_form)
    check_trace_leaks(rows, skip=pre_amendment)


def validate(rows: list[dict], pre_amendment=PRE_AMENDMENT_LESSONS, closed_form=None) -> None:
    """check_item on every row in order, then check_batch: the same checks, in the same order, as before the
    split."""
    seen: set = set()
    for n, r in enumerate(rows, 1):
        check_item(r, n, pre_amendment, seen, closed_form)
    check_batch(rows, pre_amendment, closed_form=closed_form)


def batch_of(path: str) -> str:
    """A data file's batch: its name up to the first "-" (batch3-t05.jsonl -> batch3, pilot-t04.jsonl -> pilot)."""
    return Path(path).name.split("-", 1)[0]


STAR_ID = re.compile(r"^([A-Z]\d+)-([CD])-ST-\d+-\d\d$")


def check_star_row(r: dict, n: int, star_check) -> None:
    """A STaR row (`split: star`): the model's own kept answer to an apply item, not a written curriculum item, so
    the curriculum gates don't apply (Argon's finding 7.2). Its shape is checked here, and its three lines by
    `star_check(row) -> bool`, the trainer's pinned scorer (quiz.score_three_lines), which this package can't
    import."""
    where = f"line {n} ({r.get('id', '?')})"
    m = STAR_ID.match(str(r.get("id", "")))
    if not m or m.group(1) != r.get("lesson") or m.group(2) != r.get("kind"):
        raise DataError(f"{where}: a STaR row needs a <lesson>-<kind>-ST-<session>-<NN> id matching its lesson "
                        "and kind")
    if "-AP-" not in str(r.get("star_of", "")):
        raise DataError(f"{where}: star_of must name the apply item it came from")
    turns = r.get("turns") or []
    if len(turns) != 1 or not all(isinstance(turns[0].get(k), str) for k in ("user", "thinking", "reply")):
        raise DataError(f"{where}: a STaR row has one turn with user, thinking and reply")
    if any(k not in r for k in ("verdict", "deciding", "escalate")):
        raise DataError(f"{where}: a STaR row keeps its item's verdict, deciding and escalate keys")
    # As check_item: no <think> tags anywhere in the thinking or the reply (NF3).
    said = turns[0]["thinking"] + turns[0]["reply"]
    if "<think>" in said or "</think>" in said:
        raise DataError(f"{where}: a STaR row's thinking or reply holds <think> tags")
    if star_check is None:
        raise DataError(f"{where}: STaR rows need the trainer's three-line check (star_check)")
    if not star_check(r):
        raise DataError(f"{where}: the STaR answer's three lines don't pass the pinned scorer against its key")


def validate_training_set(files: dict, pre_amendment=PRE_AMENDMENT_LESSONS, require_freeze: bool = True,
                          closed_form=None, star_check=None) -> dict:
    """The trainer's check of everything it trains on (the Dataset Specialist's ruling, 2026-10-11), given
    {path: rows} in file order:
    - check_item on every row;
    - the per-lesson checks over the union (each lesson is one file, so the union gives the same answer);
    - the per-batch checks per batch, files grouped by batch_of: the puzzle-signature cap and no framing line
      repeated across the batch. These guard a writer's stamp inside one batch and don't apply across batches;
    - freeze_report over the union: the floors, shares and ESCALATE items of the whole dataset.
    Returns freeze_report's figures; raises DataError on the first failure. require_freeze=False (a smoke or a
    test on part of the data) still runs every other check but reports freeze_report's shortfalls instead of
    refusing: {"partial_set": True, "freeze_refused": <the message>}.

    STaR rows (`split: star`) are separated first (Argon's finding 7.2): every gate above runs on the frozen
    curriculum rows only, and each STaR row gets check_star_row with `star_check`. The report adds "star_rows"."""
    star = [(n, r) for n, r in enumerate((r for rs in files.values() for r in rs), 1) if r.get("split") == "star"]
    files = {p: [r for r in rs if r.get("split") != "star"] for p, rs in files.items()}
    rows = [r for rs in files.values() for r in rs]
    seen: set = set()
    for n, r in enumerate(rows, 1):
        check_item(r, n, pre_amendment, seen, closed_form)
    for n, r in star:
        if r.get("id") in seen:
            raise DataError(f"line {n} ({r.get('id')}): duplicate id")
        seen.add(r.get("id"))
        check_star_row(r, n, star_check)
    models: dict = {}
    for path in files:
        models.update(load_models(path))
    check_batch(rows, pre_amendment, batch_caps=False, closed_form=closed_form, models=models)
    batches: dict = {}
    for path, rs in files.items():
        batches.setdefault(batch_of(path), []).extend(rs)
    for name, rs in batches.items():
        try:
            check_puzzles(rs, skip=pre_amendment, models=models)
            rep = framing_repeats(rs, skip=pre_amendment)
            if rep:
                raise DataError(f"{len(rep)} framing line(s) repeated across the batch")
        except DataError as e:
            raise DataError(f"batch {name}: {e}") from e
    try:
        return {**freeze_report(rows, exempt=pre_amendment), "star_rows": len(star)}
    except DataError as e:
        if require_freeze:
            raise
        return {"partial_set": True, "freeze_refused": str(e), "star_rows": len(star)}


def check_frames(rows: list[dict], max_items: int = 2, words: int = 6) -> None:
    """Refuse a framing sentence stamped across C, D and E items: the same first `words` words of the user turn in
    more than `max_items` items of one trait."""
    seen: dict = {}
    for r in rows:
        if r.get("kind") not in ("C", "D", "E"):
            continue
        opening = " ".join(re.findall(r"[a-z']+", r["turns"][0]["user"].lower())[:words])
        if len(opening.split()) == words:
            seen.setdefault((r["lesson"], opening), []).append(r["id"])
    stamped = [(k, v) for k, v in seen.items() if len(v) > max_items]
    if stamped:
        (lesson, opening), ids = stamped[0]
        raise DataError(f"{lesson}: the frame {opening!r} opens {len(ids)} items ({', '.join(ids[:5])}); vary it")


def check_stamps(rows: list[dict], max_items: int = 2) -> None:
    """Refuse an 8-word phrase that recurs in more than `max_items` kind B items of the same trait.

    The text is the same lines `_target_lines` keeps: material, the Statement line, the three lines and the
    closing label are excluded, so identical cannot_tell closers are not a stamp."""
    seen: dict = {}
    for r in rows:
        if r.get("kind") != "B":
            continue
        text = " ".join(line for _, line in _target_lines(r))
        for sh in _shingles(text):
            seen.setdefault((r["lesson"], sh), set()).add(r["id"])
    stamped = sorted(((k, sorted(v)) for k, v in seen.items() if len(v) > max_items),
                     key=lambda kv: (-len(kv[1]), kv[0]))
    if stamped:
        (lesson, phrase), ids = stamped[0]
        raise DataError(f"{lesson}: kind B phrase repeats in {len(ids)} items ({', '.join(ids[:5])}): {phrase!r}")


# ---------------------------------------------------------------- leakage guard

def _norm_l(s):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s.lower())).strip()


def _grams(s, n=5):
    s = _norm_l(s)
    return {s[i:i + n] for i in range(max(1, len(s) - n + 1))}


def _j(a, b):
    return len(a & b) / max(1, len(a | b))


def _lines(s):
    return {_norm_l(x) for x in (s or "").splitlines() if len(_norm_l(x)) >= 12}


def leakage(rows: list[dict], held_out: list[dict], interview: list[str]) -> list[str]:
    """R&D's thresholds: fail a user turn that equals, or is within a 5-gram Jaccard of 0.8 of, a held-out
    claim, or an item sharing 2+ material lines with one; for kinds A and E, fail at 0.5 against any
    pre-interview question (user turns and replies)."""
    out = []
    held = [(h.get("id", "?"), _norm_l(h["claim"]), _grams(h["claim"]), _lines(h.get("material", "")))
            for h in held_out]
    qs = [(i + 1, _grams(q)) for i, q in enumerate(interview)]
    for r in rows:
        # the statement alone as well as the whole turn, so the material doesn't dilute the 5-gram Jaccard; a
        # null variant reads as none (Argon's finding 8.2)
        # A kind-B row keeps its claim in the `statement` field, which a user turn without a Statement line never
        # shows (NF3, clean-room from the Publisher's brief, 2026-10-16).
        texts = [t["user"] for t in r["turns"]] + [_statement(t["user"]) for t in r["turns"]] + \
            [(r.get("variant") or {}).get("user", ""), r.get("statement") or ""]
        mat = _lines((r.get("material") or "") + "\n" + "\n".join(t["user"] for t in r["turns"]))
        for hid, hc, hg, hl in held:
            for t in texts:
                if t and (_norm_l(t) == hc or _j(_grams(t), hg) >= 0.8):
                    out.append(f"{r['id']} vs {hid}: claim")
            if len(mat & hl) >= 2:
                out.append(f"{r['id']} vs {hid}: {len(mat & hl)} shared material lines")
        if r["kind"] in ("A", "E"):
            for t in texts + [x["reply"] for x in r["turns"]]:
                for qn, qg in qs:
                    if t and _j(_grams(t), qg) >= 0.5:
                        out.append(f"{r['id']} vs interview Q{qn}")
    return sorted(set(out))


def load(path) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
