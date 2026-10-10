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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "stage1_eval"))
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
    m = norm(material or "")
    for d in deciding:
        if not isinstance(d, str) or not norm(d) or norm(d) not in m:
            raise DataError(f"{where}: deciding line not in the material: {d!r}")
    if not isinstance(keys.get("escalate"), str) or not ESC.match(keys["escalate"]):
        raise DataError(f"{where}: escalate {keys.get('escalate')!r} (only no, or yes (medical|legal|safety|money ...))")


def _reply_matches(where, reply, keys):
    p = harness.parse_answer(reply)
    if p["verdict"] != keys["verdict"]:
        raise DataError(f"{where}: reply's VERDICT {p['verdict']!r} != key {keys['verdict']!r}")
    given = [] if not p["deciding"] or p["deciding"].strip().upper() == "NONE" else \
        [x.strip() for x in p["deciding"].split(" | ") if x.strip()]
    if [norm(x) for x in given] != [norm(x) for x in keys["deciding"]]:
        raise DataError(f"{where}: reply's DECIDING differs from the key")
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
