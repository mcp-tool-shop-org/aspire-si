"""Stage-1 training data (the role-shaping curriculum): validate against docs/runs/stage1-training-data-schema.md,
guard against leakage into the held-out sets and the pre-interview, and turn items into training sequences.

Every assistant turn is a target, whole: `<think>{thinking}</think>` then the reply. One sequence per assistant
turn; the prefix is the conversation so far, with earlier assistant turns shown by their reply only (as the
chat template renders history at inference). Loss falls on the target only.
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _evaldir import EVAL_DIR  # noqa: E402

sys.path.insert(0, str(EVAL_DIR))
import harness  # noqa: E402  (parse_answer: the same three-line parser the evaluation uses)

LESSONS = {"L1", "L2", "L3"} | {f"T{i:02d}" for i in range(1, 19)}
KINDS = {"A", "B", "C", "D", "E"}
VERDICTS = {"supported", "unsupported", "cannot_tell"}
FLAWS = {"rubber_stamp", "guess", "loop", "nitpick", "fix", "obeyed_material", "caved", "reject_by_default", "none"}
ESC = re.compile(r"^(no|yes \((medical|legal|safety|money)[^)]*\))$")
REQUIRED = {"id": str, "lesson": str, "trait": str, "kind": str, "side": str, "content_kind": str, "tier": int,
            "turns": list, "sources": list, "invented": bool, "key_basis": str}


class DataError(ValueError):
    pass


# Names a Kind A teaching passage must use for its trait (lower case; any one is enough).
TRAIT_NAMES = {
    "L1": ["verifier"], "L2": ["think", "pattern"], "L3": ["expected", "rely on", "relying on"],
    "T01": ["skeptic"], "T02": ["to the source", "go to the source", "goes to the source"],
    "T03": ["precise", "precision"], "T04": ["honest about uncertainty", "uncertain"],
    "T05": ["independen"], "T06": ["break things down", "breaks things down", "breaking things down",
                                   "break it down", "decompos"],
    "T07": ["proportion"], "T08": ["your limits", "its limits", "know your limits", "knowing its limits"],
    "T09": ["consistent", "consistency"], "T10": ["literal"],
    "T11": ["what a mistake costs", "cost of a mistake", "mistake costs", "costs of mistakes"],
    "T12": ["version", "date"], "T13": ["agree just to please", "to please", "rubber-stamp", "rubber stamp"],
    "T14": ["fill gaps", "filling gaps", "fill the gap", "fill in gaps"],
    "T15": ["round in circles", "going in circles", "loop"], "T16": ["nitpick"],
    "T17": ["take orders from the material", "orders from the material", "instructions in the material",
            "instructions inside the material"],
    "T18": ["fix things", "fixing", "don't fix", "doesn't fix"],
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


# Amendment 1 (quoting the deciding lines) binds from batch 3 on. The pilot (T04) and batch 2 (T01, T14, T16)
# were written before it and are not redone, so the whole-row, version-heading and multi-line-share checks skip
# these lessons by name. The separator and NONE check applies to every lesson.
PRE_AMENDMENT_LESSONS = frozenset({"T04", "T01", "T14", "T16"})
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
        p = harness.parse_answer(t["reply"])
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
    rep["shortfalls"] = short
    if short:
        raise DataError("freeze: " + "; ".join(short) + f" (figures: {json.dumps(rep)})")
    return rep


# Fixed banned list (R&D, 2026-10-10): the role, the 18 trait names and their short forms, and Lesson 2's
# pattern names. "Checker" is allowed: in kind C it names the other party, not the role.
ROLE_WORDS = re.compile(
    r"\b(verifiers?|sceptic\w*|skeptic\w*|goe?s? to the source|going to the source|precise|precision|"
    r"honest about uncertainty|independen\w*|breaks? things down|breaking things down|decompos\w*|"
    r"in proportion|proportionate|knows? (?:its|your|my) limits|consistent|consistency|literal\w*|"
    r"what a mistake costs|agrees? just to please|fill(?:s|ing)? (?:the )?gaps?|round in circles|nitpick\w*|"
    r"takes? orders from the material|fix(?:es|ing)? things|"
    r"trees?|prun\w*|falsif\w*|eliminat\w*|grids?|constraint grids?)\b", re.I)


def role_words_in_prompt(r: dict) -> list[str]:
    """Banned words in a C, D or verdict-type E item's user turns, outside the material, new material and the
    statement line (those are the content being checked, shared by both arms as they are)."""
    if not (r.get("kind") in ("C", "D") or (r.get("kind") == "E" and (r.get("quiz") or {}).get("type") == "verdict")):
        return []
    content = {x.strip() for x in (r.get("material") or "").splitlines() if x.strip()}
    found = []
    for t in r["turns"]:
        content |= {x.strip() for x in t.get("new_material", []) if x.strip()}
        for line in t["user"].splitlines():
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
    p = harness.parse_answer(reply)
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


def validate(rows: list[dict], pre_amendment=PRE_AMENDMENT_LESSONS) -> None:
    seen = set()
    for n, r in enumerate(rows, 1):
        where = f"line {n} ({r.get('id', '?')})"
        for k, typ in REQUIRED.items():
            if not isinstance(r.get(k), typ):
                raise DataError(f"{where}: field {k} missing or not {typ.__name__}")
        if r["id"] in seen:
            raise DataError(f"{where}: duplicate id")
        seen.add(r["id"])
        if r["lesson"] not in LESSONS or r["kind"] not in KINDS:
            raise DataError(f"{where}: unknown lesson {r['lesson']} or kind {r['kind']}")
        if r["tier"] < 1 or r["key_basis"] not in ("construction", "source") or not r["turns"]:
            raise DataError(f"{where}: tier, key_basis or turns")
        material = r.get("material") or ""
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
            variant = " ".join(_key_points(v["key"]))
            if not any(k and (k in variant or variant in k) for k in original):
                raise DataError(f"{where}: the variant's key shares no key point with the original")
    check_stamps(rows)
    check_frames(rows)
    check_multi_line_share(rows, skip=pre_amendment)
    check_floors(rows, skip=pre_amendment)
    check_pairs(rows, skip=pre_amendment)


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
    """Refuse an 8-word phrase that recurs in more than `max_items` kind B items of the same trait."""
    seen: dict = {}
    for r in rows:
        if r.get("kind") != "B":
            continue
        text = " ".join(t["thinking"] + " " + t["reply"] for t in r["turns"])
        for sh in _shingles(text):
            seen.setdefault((r["lesson"], sh), set()).add(r["id"])
    stamped = [(k, sorted(v)) for k, v in seen.items() if len(v) > max_items]
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
        texts = [t["user"] for t in r["turns"]] + [r.get("variant", {}).get("user", "")]
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


# ---------------------------------------------------------------- rendering

def sequences(r: dict, tok) -> list[dict]:
    """[{prompt_ids, target_ids}] for each assistant turn of one item."""
    msgs, out = [], []
    for t in r["turns"]:
        msgs.append({"role": "user", "content": t["user"]})
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        think = t["thinking"].strip()
        target = (f"<think>\n{think}\n</think>\n\n" if think else "<think>\n\n</think>\n\n") + t["reply"].strip() \
            + "<|im_end|>"
        out.append({"prompt_ids": tok(prompt, add_special_tokens=False)["input_ids"],
                    "target_ids": tok(target, add_special_tokens=False)["input_ids"]})
        msgs.append({"role": "assistant", "content": t["reply"].strip()})
    return out


def load(path) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


if __name__ == "__main__":
    # python data.py --freeze <file.jsonl> [<file.jsonl> ...]: the whole-dataset check, over all files together
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", nargs="+", required=True, metavar="FILE")
    a = ap.parse_args()
    everything = [r for f in a.freeze for r in load(f)]
    try:
        print(json.dumps(freeze_report(everything), indent=1))
        print("freeze: clean")
    except DataError as e:
        print(e)
        sys.exit(1)
