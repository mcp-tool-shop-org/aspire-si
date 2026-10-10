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


def norm(x: str) -> str:
    return re.sub(r"\s+", " ", x.replace("**", "").replace("__", "")).strip()


def _check_keys(where, keys, material):
    if keys.get("verdict") not in VERDICTS:
        raise DataError(f"{where}: verdict {keys.get('verdict')!r}")
    deciding = keys.get("deciding")
    if not isinstance(deciding, list) or len(deciding) > 4:
        raise DataError(f"{where}: deciding must be a list of at most 4 lines")
    if (keys["verdict"] == "cannot_tell") != (not deciding):
        raise DataError(f"{where}: cannot_tell exactly when deciding is empty")
    for d in deciding:
        if not isinstance(d, str) or not d.strip() or d.strip() not in (material or ""):
            raise DataError(f"{where}: deciding line not verbatim in the material (markup such as ** kept): {d!r}")
    if not isinstance(keys.get("escalate"), str) or not ESC.match(keys["escalate"]):
        raise DataError(f"{where}: escalate {keys.get('escalate')!r} (only no, or yes (medical|legal|safety|money ...))")


def _reply_matches(where, reply, keys):
    p = harness.parse_answer(reply)
    if p["verdict"] != keys["verdict"]:
        raise DataError(f"{where}: reply's VERDICT {p['verdict']!r} != key {keys['verdict']!r}")
    given = [] if not p["deciding"] or p["deciding"].strip().upper() == "NONE" else \
        [x.strip() for x in p["deciding"].split(" | ") if x.strip()]
    if [x.strip() for x in given] != [x.strip() for x in keys["deciding"]]:
        raise DataError(f"{where}: reply's DECIDING differs from the key (exact, markup kept)")
    if (p["escalate"] or "").strip() != keys["escalate"]:
        raise DataError(f"{where}: reply's ESCALATE {p['escalate']!r} != key {keys['escalate']!r}")


def validate(rows: list[dict]) -> None:
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
            if not re.search(r"verdict", first_user, re.I):
                raise DataError(f"{where}: kind C's user turn has no checker verdict")
            if REPO_CITATION.search((r.get("material") or "") + "\n" + first_user) and not has_real:
                raise DataError(f"{where}: kind C cites a repository but sources has no real (non-invented) entry")
        if kind == "A" and not r.get("key_points"):
            raise DataError(f"{where}: kind A needs key_points")
        if kind == "B" and not r.get("answer"):
            raise DataError(f"{where}: kind B needs answer")
        if kind in ("C", "D"):
            if kind == "C" and r.get("planted_flaw") not in FLAWS:
                raise DataError(f"{where}: planted_flaw {r.get('planted_flaw')!r}")
            if kind == "D" and r.get("material") is None:
                raise DataError(f"{where}: kind D needs material")
            _check_keys(where, r, available[-1])
            _reply_matches(where, last, r)
            for idx, keys in (r.get("turn_keys") or {}).items():
                i = int(idx)
                _check_keys(f"{where} turn {i}", keys, available[i - 1])
                _reply_matches(f"{where} turn {i}", r["turns"][i - 1]["reply"], keys)
        if kind == "E":
            q, v = r.get("quiz"), r.get("variant")
            if not isinstance(q, dict) or q.get("type") not in ("verdict", "written"):
                raise DataError(f"{where}: quiz must be {{type: verdict|written, ...}}")
            if q["type"] == "verdict":
                _check_keys(where, q, available[-1])
                _reply_matches(where, last, q)
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
