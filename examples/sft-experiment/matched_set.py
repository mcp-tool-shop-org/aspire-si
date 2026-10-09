"""Addendum 4, step 2 of the 2026-10-08 Auditor plan: the two-sided matched set, P-matched.

Errors and meaning-preserving rewrites are brought toward each other in surprisal, instead of
dragging the paraphrases alone toward the errors (Zellers et al. 2018, 2019; Le Bras et al. 2020;
Wu et al. 2021). It is a search in the shape of mcp-stress-test's fuzz_until_evasion: generate,
score, keep or retry.

On the 78 confirmation strong answers:
- **propose:** the planter (Qwen2.5-32B-Instruct Q4_K_M on llama-server) writes candidate errors
  and candidate rewrites, one per call, five per answer per side and round. Each later candidate is
  told which sentences and kinds the earlier ones used. Both sides target 2-8 changed words.
- **confirm:**
  - errors: gemma4:31b (thinking) on every candidate. muse-glimmer (thinking, plain text) on every
    candidate gemma calls "wrong", plus a seed-0 random 15% of the rest, which measures gemma's
    false negatives. An error is kept only if both say "wrong" and "grammatical".
  - rewrites: mistral-small:24b and gemma4:31b (thinking). Kept only if both say "same" with no
    claim changed, and gemma says "grammatical".
- **score:** Phi-3-mini's tail delta for every kept candidate (typicality.py).
- **pair:** per answer, one kept error and one kept rewrite with tail deltas within 2 nats and edit
  sizes within a ratio of 1.5. Prefer the least surprising error and the most surprising rewrite.
  Unmatched answers get up to two more rounds.

`grammar` also runs on the existing kept word swaps: a CPU rule pre-screen, then gemma on every
flagged pair plus a seed-0 random 10% of the unflagged ones, which measures the pre-screen's misses.

Every model call is one item, gives reasoning first, and is stored whole. A reply cut off at its
token limit is "truncated", never a verdict.
"""

from __future__ import annotations

import argparse
import difflib
import json
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from clean_dataset import char_similarity  # noqa: E402
from critic_heads import load_pairs  # noqa: E402
from lib import SELF_FLAG_MARKERS, _restore_latex, extract_json_object, refuse_cloud  # noqa: E402

TARGET_WORDS = "2-8"
CALIPER_NATS = 2.0
SIZE_RATIO = 1.5
CANDIDATES = 5
ROUNDS = 3
MUSE_SAMPLE = 0.15
GRAMMAR_SAMPLE = 0.10
MIN_SIMILARITY = 0.85

PLANTER_SYSTEM = "You are an expert writer helping to build a careful test set. You follow the brief exactly."
JUDGE_SYSTEM = "You are a careful expert reviewer. You reason before you decide, and you say so when unsure."

PURPOSE = """## Why this exists

We are studying "critics": small models that score answers. To test whether a critic detects
errors, or merely notices that an answer was edited, we need two kinds of edited answers that look
alike in every way except one:
- an answer with exactly one subtle error that a careful expert would catch;
- an answer with a rewording that changes nothing it claims.
Both kinds change one sentence by about {words} words, read naturally, and never announce
themselves. Your job now is to write ONE item of the kind described below."""

ERROR_BRIEF = """## Your task: one subtle error

Change one sentence of the answer so that it becomes wrong, in fact or in logic, in the context of
the whole answer. Change about {words} words.

A good error:
- is really wrong: a wrong number or unit, a reversed cause or direction, a wrong condition, a
  term swapped for one with a different technical meaning, or a step in the wrong order;
- is plausible: same register, same confidence, grammatical, idiomatic;
- doesn't contradict another sentence word for word (that makes it trivial);
- adds no hedge, comment or marker ("note:", "(incorrect)", "actually").

## Worked examples (from other material)

Good. Original: "Water boils at a lower temperature at high altitude because the air pressure is
lower." Edited: "Water boils at a higher temperature at high altitude because the air pressure is
lower." Why good: the direction is reversed, it reads naturally, and only an expert reader of the
whole sentence catches it.

Good. Original: "Merge sort runs in O(n log n) time in the worst case." Edited: "Merge sort runs in
O(n log n) time only in the average case." Why good: a wrong condition, a plausible-sounding
technical claim, 4 words changed.

Bad (too obvious). Original: "The French Revolution began in 1789." Edited: "The French Revolution
began in 1989." Why bad: anyone notices a two-century slip; it tests nothing.

Bad (not actually wrong). Original: "Tides are caused mainly by the Moon's gravity." Edited: "Tides
are driven mainly by the Moon's gravity." Why bad: this is a paraphrase, not an error.

Bad (flags itself). Original: "Compound interest grows faster than simple interest." Edited:
"Compound interest (incorrectly) grows slower than simple interest." Why bad: the marker gives the
error away.
"""

REWRITE_BRIEF = """## Your task: one meaning-preserving rewording

Reword one phrase of one sentence, about {words} words, so that the sentence says exactly the same
thing. Not a single word: a phrase.

A good rewording:
- keeps every claim exactly: nothing added, removed, strengthened or weakened;
- changes no number, unit, name, quantifier ("all", "most"), modal ("can", "must"), negation or
  technical term;
- is grammatical and natural, in the same register as the answer.

## Worked examples (from other material)

Good. Original: "Plants absorb carbon dioxide through small pores on their leaves." Edited: "Plants
take in carbon dioxide through tiny openings on their leaves." Why good: two phrases reworded,
identical meaning, natural.

Good. Original: "Blood pressure is measured as two numbers, the higher one during a heartbeat."
Edited: "Blood pressure is recorded as two numbers, the higher one while the heart beats." Why
good: same facts, a different phrasing of the same moment.

Bad (one word only). Original: "The bridge closed due to high winds." Edited: "The bridge closed
owing to high winds." Why bad: a single-word swap; we need a phrase.

Bad (meaning shift). Original: "Interest rates usually slow inflation." Edited: "Interest rates
reliably slow inflation." Why bad: "reliably" strengthens the claim.

Bad (hedge added). Original: "The algorithm sorts in place." Edited: "The algorithm, it seems,
sorts in place." Why bad: adds uncertainty the original didn't have.
"""

ITEM = """## The answer

Question: {prompt}

Answer:
{answer}
{avoid}
## Your reply

Reason first, then give the change. Copy the sentence exactly as it appears in the answer. Reply
with a JSON object only:
{{"reasoning": "<why this change fits the brief>", "sentence": "<the original sentence, copied
exactly>", "edited": "<the same sentence with your change>"{kind_field}}}"""

ERROR_CHECK = """## Why this check exists

We are building a test set of answers that each contain one subtle planted error. A candidate is
useful only if the edited answer is really wrong where it differs from the original, and still
reads as natural, grammatical prose. Judge honestly; "unsure" is a useful answer.

## The item

Question: {prompt}

Original answer (in full):
{strong}

Edited answer (in full):
{flawed}

The only difference is in this sentence.
Original sentence: {original}
Edited sentence: {edited}

## Your reply

First reason it through: what changed, and whether the edited version is wrong in fact or logic,
in the context of the whole answer. Then:
- "verdict": "wrong" (the edited answer is wrong where it differs), "not wrong", or "unsure";
- "grammatical": false only if a careful copy editor would mark the edited sentence as a grammar
  or usage error (a wrong preposition or verb pattern, broken agreement, a wrong article, a word
  that can't be used this way). A less natural or less common word choice alone is NOT an error;
- "idiomatic": true if the edited sentence reads as natural, idiomatic English (recorded only)."""

MUSE_FORMAT = """

Write your reasoning, then end with exactly three lines:
VERDICT: wrong | not wrong | unsure
GRAMMATICAL: yes | no
IDIOMATIC: yes | no

Say "GRAMMATICAL: no" only for a grammar or usage error a careful copy editor would mark; a less
natural word choice alone is not one. IDIOMATIC is recorded only."""

GRAMMAR_CHECK = """## The item

A sentence in an answer was reworded. Would a careful copy editor mark the reworded sentence as a
grammar or usage error: a wrong preposition or verb pattern ("upholds to", "results to"), broken
agreement, a wrong article, or a word that can't be used this way? A less natural or less common
word choice alone is NOT an error. Judge only grammar and usage, not meaning or truth.

Question: {prompt}

Answer (with the reworded sentence in place):
{flawed}

Original sentence: {original}
Reworded sentence: {edited}

Reason first, then give "grammatical" (false only for a grammar or usage error) and "idiomatic"
(true if it reads as natural, idiomatic English; recorded, never used to drop anything)."""


def schema(fields: dict) -> dict:
    """A JSON schema with "reasoning" first and required, then `fields`, and nothing else."""
    return {
        "type": "object",
        "properties": {"reasoning": {"type": "string"}, **fields},
        "required": ["reasoning", *fields],
        "additionalProperties": False,
    }


ERROR_SCHEMA = schema(
    {
        "verdict": {"type": "string", "enum": ["wrong", "not wrong", "unsure"]},
        "grammatical": {"type": "boolean"},
        "idiomatic": {"type": "boolean"},
    }
)
GRAMMAR_SCHEMA = schema({"grammatical": {"type": "boolean"}, "idiomatic": {"type": "boolean"}})
# The grammar question's version. v1 asked "grammatical and idiomatic" and so dropped merely
# less-idiomatic swaps, which pushes paraphrases toward typical wording. v2 gates on grammar or usage
# errors only and records idiom (2026-10-08, before any head read).
GRAMMAR_QUESTION = "v2: grammar or usage error only; idiom recorded"


def planter_request(side: str, item: dict, earlier: list[dict]) -> str:
    """The planter's message for one candidate. `earlier` are this answer's earlier candidates."""
    avoid = ""
    if earlier:
        used = "\n".join(
            f'- "{c["sentence"]}"' + (f" ({c['kind']})" if c.get("kind") else "") for c in earlier
        )
        avoid = (
            "\nEarlier items for this answer already changed these sentences"
            + (" (with these kinds of error)" if side == "error" else "")
            + f":\n{used}\nChoose a different sentence"
            + (" or a different kind of error." if side == "error" else ".")
            + "\n"
        )
    brief = ERROR_BRIEF if side == "error" else REWRITE_BRIEF
    kind_field = ', "kind": "<the kind of error>"' if side == "error" else ""
    return (
        PURPOSE.format(words=TARGET_WORDS)
        + "\n\n"
        + brief.format(words=TARGET_WORDS)
        + "\n"
        + ITEM.format(prompt=item["prompt"], answer=item["strong"], avoid=avoid, kind_field=kind_field)
    )


def apply_candidate(reply: str, answer: str) -> tuple[dict | None, str]:
    """The candidate applied by code, or None and why: no JSON, sentence not found, unchanged,
    self-flagging, or too different (character similarity below 0.85)."""
    data = extract_json_object(reply)
    if not data or not all(isinstance(data.get(k), str) for k in ("sentence", "edited")):
        return None, "no JSON candidate"
    for original, edited in (
        (data["sentence"], data["edited"]),
        (_restore_latex(data["sentence"]), _restore_latex(data["edited"])),
    ):
        original, edited = original.strip(), edited.strip()
        if not original or original not in answer:
            continue
        if edited == original:
            return None, "unchanged"
        flags = [m for m in SELF_FLAG_MARKERS if edited.lower().count(m) > original.lower().count(m)]
        if flags:
            return None, "self-flagging"
        flawed = answer.replace(original, edited, 1)
        if char_similarity(answer, flawed) < MIN_SIMILARITY:
            return None, "too different"
        chars, words = edit_size(answer, flawed)
        return {
            "reasoning": data.get("reasoning", ""),
            "sentence": original,
            "edited_sentence": edited,
            "kind": data.get("kind", ""),
            "flawed": flawed,
            "edit_chars": chars,
            "edit_words": words,
        }, "ok"
    return None, "sentence not in the answer"


def edit_size(a: str, b: str) -> tuple[int, int]:
    """(characters changed, words changed): difflib, autojunk off, the larger side of each
    non-equal opcode, on characters and on whitespace-split words."""

    def changed(x, y):
        ops = difflib.SequenceMatcher(None, x, y, autojunk=False).get_opcodes()
        return sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in ops if op != "equal")

    return changed(a, b), changed(a.split(), b.split())


def parse_muse(text: str) -> dict:
    """muse-glimmer replies in plain text (format constraints empty its replies), ending in two
    lines: VERDICT and GRAMMATICAL. A trailing <|eot|> is stripped."""
    text = text.replace("<|eot|>", "").strip()
    verdict = re.findall(r"^\s*VERDICT:\s*(wrong|not wrong|unsure)\s*$", text, flags=re.I | re.M)
    grammar = re.findall(r"^\s*GRAMMATICAL:\s*(yes|no)\s*$", text, flags=re.I | re.M)
    idiom = re.findall(r"^\s*IDIOMATIC:\s*(yes|no)\s*$", text, flags=re.I | re.M)
    if not verdict or not grammar:
        return {"outcome": "unparsed", "reasoning": text}
    return {
        "outcome": verdict[-1].lower(),
        "grammatical": grammar[-1].lower() == "yes",
        "idiomatic": (idiom[-1].lower() == "yes") if idiom else None,
        "reasoning": text.rsplit("VERDICT:", 1)[0].strip(),
    }


def error_kept(gemma: dict, muse: dict | None) -> bool:
    return (
        gemma.get("outcome") == "wrong"
        and gemma.get("grammatical") is True
        and muse is not None
        and muse.get("outcome") == "wrong"
        and muse.get("grammatical") is True
    )


def rewrite_kept(mistral: dict, gemma: dict) -> bool:
    return (
        mistral.get("outcome") == "same"
        and mistral.get("claim_changed") is False
        and gemma.get("outcome") == "same"
        and gemma.get("claim_changed") is False
        and gemma.get("grammatical") is True
    )


def seeded_sample(ids: list[str], fraction: float, seed: int = 0) -> set[str]:
    """A seeded random `fraction` of `ids` (at least one when there are any), independent of order."""
    pool = sorted(ids)
    random.Random(seed).shuffle(pool)
    return set(pool[: max(1, round(fraction * len(pool)))] if pool else [])


def choose_pair(
    errors: list[dict], rewrites: list[dict], caliper: float = CALIPER_NATS, ratio: float = SIZE_RATIO
):
    """One kept error and one kept rewrite within `caliper` nats of tail delta and within `ratio`
    of edit size (characters, larger / smaller), preferring the least surprising error (highest
    tail), then the most surprising rewrite (lowest tail). None if no pair fits."""
    for e in sorted(errors, key=lambda c: -c["tail"]):
        for r in sorted(rewrites, key=lambda c: c["tail"]):
            big, small = max(e["edit_chars"], r["edit_chars"]), min(e["edit_chars"], r["edit_chars"])
            if abs(e["tail"] - r["tail"]) <= caliper and small > 0 and big / small <= ratio:
                return e, r
    return None


# Grammar pre-screen: patterns an edit may newly introduce.
_RULES = (
    ("result to", re.compile(r"\bresult(?:s|ed|ing)? (?:\w+ )?to\b", re.I)),
    ("comprise of", re.compile(r"\bcompris(?:e|es|ed|ing) of\b", re.I)),
    ("doubled word", re.compile(r"\b(\w+) \1\b", re.I)),
    ("a before vowel sound", re.compile(r"\ba (?=[aeio]\w)", re.I)),
    ("an before consonant sound", re.compile(r"\ban (?=[bcdfgjklmnpqrstvwxyz]\w)", re.I)),
)


def prescreen(original: str, edited: str) -> list[str]:
    """Rule flags the edit newly introduces (present in the edited sentence, absent or rarer in the
    original). Words beginning "uni"/"eu"/"one"/"use"/"hour"/"hon" are left to the model."""
    flags = []
    for name, pattern in _RULES:

        def hits(text):
            return [
                m
                for m in pattern.finditer(text)
                if not re.match(
                    r"an? (uni|eu|one|use|usu|hour|hon|heir)", text[m.start() : m.start() + 8], re.I
                )
            ]

        if len(hits(edited)) > len(hits(original)):
            flags.append(name)
    return flags


OLLAMA = "http://127.0.0.1:11434"


def ollama_chat(
    model: str, system: str, user: str, fmt, think, num_predict: int, num_ctx: int, url: str = OLLAMA
) -> dict:  # pragma: no cover
    import urllib.request

    body = {
        "model": refuse_cloud(model),
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": num_predict, "num_ctx": num_ctx},
    }
    if fmt is not None:
        body["format"] = fmt
    if think is not None:
        body["think"] = think
    req = urllib.request.Request(
        url.rstrip("/") + "/api/chat",
        data=json.dumps(body).encode("utf-8"),
        headers={"content-type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        response = json.loads(r.read())
    response["_seconds"] = round(time.time() - t0, 1)
    return response


def judged(response: dict, fields: tuple[str, ...]) -> dict:
    """A schema-constrained Ollama reply as an outcome. "truncated" when cut off at num_predict."""
    message = response.get("message", {})
    out = {
        "thinking_chars": len(message.get("thinking") or ""),
        "done_reason": response.get("done_reason"),
        "seconds": response.get("_seconds"),
    }
    if response.get("done_reason") == "length":
        return out | {"outcome": "truncated"}
    data = extract_json_object(message.get("content") or "")
    if not data or any(f not in data for f in fields):
        return out | {"outcome": "unparsed"}
    return out | {k: data[k] for k in data} | {"outcome": data.get("verdict", "ok")}


def unload(model: str, url: str = OLLAMA) -> None:  # pragma: no cover
    import urllib.request

    req = urllib.request.Request(
        url.rstrip("/") + "/api/generate",
        data=json.dumps({"model": model, "keep_alive": 0}).encode("utf-8"),
        headers={"content-type": "application/json"},
    )
    urllib.request.urlopen(req, timeout=120).read()


GRAMMAR_ASK = (
    '\nAlso give "grammatical" (false only if a careful copy editor would mark the edited sentence as'
    ' a grammar or usage error; a less natural word choice alone is not one) and "idiomatic"'
    " (recorded only)."
)
THINKING = {"num_predict": 16000, "num_ctx": 24576}
MISTRAL = {"num_predict": 2048, "num_ctx": 8192}


def _load_state(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _save(path: Path, state: dict) -> None:
    path.write_text(json.dumps(state, indent=1, ensure_ascii=False), encoding="utf-8")


def strong_answers(pairs: list[dict]) -> list[dict]:
    seen = {}
    for p in pairs:
        seen.setdefault(
            p["prompt_id"], {"prompt_id": p["prompt_id"], "prompt": p["prompt"], "strong": p["strong"]}
        )
    return list(seen.values())


def stage_propose(
    items, out: Path, side: str, url: str, round_no: int, only: set | None
) -> None:  # pragma: no cover
    from lib import Chat, ServerBackend

    path = out / f"candidates-{side}.json"
    state = _load_state(path)
    backend = ServerBackend(url)
    todo = [it for it in items if only is None or it["prompt_id"] in only]
    for slot in range(CANDIDATES):
        chats, which = [], []
        for it in todo:
            earlier = [c for c in state.get(it["prompt_id"], []) if c.get("sentence")]
            done = [c for c in state.get(it["prompt_id"], []) if c["round"] == round_no and c["slot"] == slot]
            if done:
                continue
            chats.append(Chat(PLANTER_SYSTEM, [("user", planter_request(side, it, earlier))]))
            which.append(it)
        if not chats:
            continue
        replies = backend.generate(chats, 1536, 0.7)
        for it, reply, cut in zip(which, replies, backend.last_truncated):
            cand, why = (None, "truncated") if cut else apply_candidate(reply, it["strong"])
            entry = {"round": round_no, "slot": slot, "status": why, "raw": reply}
            if cand:
                entry |= cand | {"id": f"{it['prompt_id']}-{side[0]}{round_no}{slot}"}
            state.setdefault(it["prompt_id"], []).append(entry)
        _save(path, state)
        print(side, "round", round_no, "slot", slot, len(chats), flush=True)


def stage_confirm(items, out: Path, side: str, judge: str) -> None:  # pragma: no cover
    from recheck import REQUEST, SCHEMA, SYSTEM

    by = {it["prompt_id"]: it for it in items}
    cands = _load_state(out / f"candidates-{side}.json")
    vpath = out / f"verdicts-{side}-{judge.replace(':', '_')}.json"
    verdicts = _load_state(vpath)
    gemma = (
        _load_state(out / "verdicts-error-gemma4_31b.json")
        if (side == "error" and judge.startswith("muse"))
        else None
    )
    if gemma is not None:
        not_wrong = [cid for cid, v in gemma.items() if v.get("outcome") != "wrong"]
        sample = seeded_sample(not_wrong, MUSE_SAMPLE)
    try:
        for pid, cs in cands.items():
            for c in cs:
                cid = c.get("id")
                if not cid or cid in verdicts:
                    continue
                if gemma is not None and not (gemma.get(cid, {}).get("outcome") == "wrong" or cid in sample):
                    continue
                it = by[pid]
                if side == "error":
                    user = ERROR_CHECK.format(
                        prompt=it["prompt"],
                        strong=it["strong"],
                        flawed=c["flawed"],
                        original=c["sentence"],
                        edited=c["edited_sentence"],
                    )
                    if judge.startswith("muse"):
                        r = ollama_chat(judge, JUDGE_SYSTEM, user + MUSE_FORMAT, None, True, **THINKING)
                        v = (
                            {"outcome": "truncated"}
                            if r.get("done_reason") == "length"
                            else parse_muse(r.get("message", {}).get("content") or "")
                        )
                        v |= {
                            "seconds": r["_seconds"],
                            "thinking_chars": len(r.get("message", {}).get("thinking") or ""),
                        }
                        v["sampled_not_wrong"] = cid in sample
                    else:
                        v = judged(
                            ollama_chat(judge, JUDGE_SYSTEM, user, ERROR_SCHEMA, True, **THINKING),
                            ("verdict", "grammatical"),
                        )
                else:
                    user = REQUEST.format(
                        prompt=it["prompt"],
                        strong=it["strong"],
                        flawed=c["flawed"],
                        original=c["sentence"],
                        edited=c["edited_sentence"],
                    )
                    if judge.startswith("gemma"):
                        fmt = dict(
                            SCHEMA,
                            properties={
                                **SCHEMA["properties"],
                                "grammatical": {"type": "boolean"},
                                "idiomatic": {"type": "boolean"},
                            },
                            required=[*SCHEMA["required"], "grammatical", "idiomatic"],
                        )
                        user += GRAMMAR_ASK
                        v = judged(
                            ollama_chat(judge, SYSTEM, user, fmt, True, **THINKING),
                            ("verdict", "claim_changed", "grammatical"),
                        )
                    else:
                        v = judged(
                            ollama_chat(judge, SYSTEM, user, SCHEMA, None, **MISTRAL),
                            ("verdict", "claim_changed"),
                        )
                verdicts[cid] = v
                _save(vpath, verdicts)
            print(side, judge, pid, len(verdicts), flush=True)
    finally:
        unload(judge)


def stage_score(items, out: Path) -> None:  # pragma: no cover
    from typicality import score

    by = {it["prompt_id"]: it for it in items}
    sets = {}
    for side in ("error", "rewrite"):
        cands = _load_state(out / f"candidates-{side}.json")
        sets[side] = [
            {
                "pair_id": c["id"],
                "prompt": by[pid]["prompt"],
                "strong": by[pid]["strong"],
                "flawed": c["flawed"],
            }
            for pid, cs in cands.items()
            for c in cs
            if c.get("id")
        ]
    _save(out / "tail-phi3.json", score(sets))


def escalate(summaries: dict) -> bool:
    """The committed rule is for the whole pass: a miss in any set's sample sends every pair of
    every set to gemma. A per-set reading would let a tiny sample (P-second's 4) pass on nothing."""
    return any(not s["prescreen_stands"] for s in summaries.values())


def stage_grammar(sets: dict[str, list[dict]], kept: dict[str, set], out: Path) -> None:  # pragma: no cover
    """The word swaps' grammar pass: the CPU pre-screen on every kept pair, then gemma4 (thinking) on
    every flagged pair plus a seed-0 random 10% of the unflagged ones (the pre-screen's miss rate).
    If any set's sample shows a miss, gemma then checks every pair of every set."""
    from skeptic_pairs import sentences

    path = out / "grammar-wordswaps.json"
    state = _load_state(path)
    judge = "gemma4:31b"
    plan = {}
    for name, pairs in sets.items():
        rows = state.setdefault(name, {})
        mine = [p for p in pairs if p["pair_id"] in kept[name]]
        for p in mine:
            rows.setdefault(p["pair_id"], {"flags": prescreen(*sentences(p))})
        unflagged = [pid for pid, r in rows.items() if not r["flags"]]
        plan[name] = (rows, mine, seeded_sample(unflagged, GRAMMAR_SAMPLE))

    def ask(r: dict, p: dict, sample: set) -> None:
        original, edited = sentences(p)
        user = GRAMMAR_CHECK.format(prompt=p["prompt"], flawed=p["flawed"], original=original, edited=edited)
        if "gemma" in r:
            r["gemma_v1"] = r.pop("gemma")  # the v1 ("and idiomatic") verdict, kept for the record
        r["gemma"] = judged(
            ollama_chat(judge, JUDGE_SYSTEM, user, GRAMMAR_SCHEMA, True, **THINKING),
            ("grammatical", "idiomatic"),
        ) | {"question": GRAMMAR_QUESTION}
        r["sampled_unflagged"] = p["pair_id"] in sample
        _save(path, state)

    try:
        for name, (rows, mine, sample) in plan.items():
            for p in mine:
                r = rows[p["pair_id"]]
                if not _v2(r) and (r["flags"] or p["pair_id"] in sample):
                    ask(r, p, sample)
            print("grammar sample", name, flush=True)
        if escalate({name: grammar_summary(rows) for name, (rows, _, _) in plan.items()}):
            for name, (rows, mine, sample) in plan.items():
                for p in mine:
                    if not _v2(rows[p["pair_id"]]):
                        ask(rows[p["pair_id"]], p, sample)
                print("grammar all", name, len(rows), flush=True)
    finally:
        unload(judge)


def _v2(r: dict) -> dict:
    """The row's current-question (v2) gemma verdict, or {} if it has none."""
    g = r.get("gemma", {})
    return g if g.get("question") == GRAMMAR_QUESTION else {}


def grammar_summary(rows: dict) -> dict:
    """Per set, from v2 verdicts only: flagged, the pre-screen's measured misses on the unflagged
    sample (with the rule-of-three upper bound, 3 / n, when none are found), the pairs to drop
    (grammar or usage errors), and idiom recorded but not gated."""
    flagged = [pid for pid, r in rows.items() if r["flags"]]
    sampled = [r for r in rows.values() if r.get("sampled_unflagged") and _v2(r)]
    misses = sum(1 for r in sampled if _v2(r).get("grammatical") is False)
    drop = [pid for pid, r in rows.items() if _v2(r).get("grammatical") is False]
    judged_rows = [r for r in rows.values() if _v2(r)]
    unresolved = [pid for pid, r in rows.items() if _v2(r).get("outcome") in ("truncated", "unparsed")]
    n = len(sampled)
    return {
        "question": GRAMMAR_QUESTION,
        "pairs": len(rows),
        "judged_by_gemma": len(judged_rows),
        "flagged": len(flagged),
        "flagged_confirmed_ungrammatical": sum(1 for pid in flagged if pid in drop),
        "unflagged_sampled": n,
        "misses_in_sample": misses,
        "miss_rate_upper_bound_95": (3 / n if n else None) if misses == 0 else None,
        "prescreen_stands": misses == 0,
        "not_idiomatic_recorded_only": sum(1 for r in judged_rows if r["gemma"].get("idiomatic") is False),
        # The sensitivity row (R&D review): what a grammar-AND-idiom gate would drop, so the stricter
        # gate can be re-applied later and its effect on the conclusions seen.
        "drop_if_idiom_gated": [
            pid
            for pid, r in rows.items()
            if _v2(r).get("grammatical") is False or _v2(r).get("idiomatic") is False
        ],
        "drop": drop,
        "unresolved": unresolved,
    }


def kept_candidates(out: Path) -> dict[str, dict[str, list[dict]]]:
    """Per answer, the kept errors and kept rewrites with their Phi-3 tail delta."""
    tails = _load_state(out / "tail-phi3.json")
    v = {
        n: _load_state(out / f"verdicts-{n}.json")
        for n in (
            "error-gemma4_31b",
            "error-muse-glimmer_latest",
            "rewrite-mistral-small_24b",
            "rewrite-gemma4_31b",
        )
    }
    kept: dict[str, dict[str, list[dict]]] = {}
    for side in ("error", "rewrite"):
        for pid, cs in _load_state(out / f"candidates-{side}.json").items():
            for c in cs:
                cid = c.get("id")
                if not cid or cid not in tails.get(side, {}):
                    continue
                ok = (
                    error_kept(v["error-gemma4_31b"].get(cid, {}), v["error-muse-glimmer_latest"].get(cid))
                    if side == "error"
                    else rewrite_kept(
                        v["rewrite-mistral-small_24b"].get(cid, {}), v["rewrite-gemma4_31b"].get(cid, {})
                    )
                )
                if ok:
                    kept.setdefault(pid, {"error": [], "rewrite": []})[side].append(
                        c | {"tail": tails[side][cid]["tail"]}
                    )
    return kept


def main() -> None:  # pragma: no cover - needs the GPU, llama-server or Ollama
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("stage", choices=["propose", "confirm", "score", "pair", "grammar"])
    parser.add_argument("--wordswaps", action="append", default=[], help="grammar: NAME=pairs.json")
    parser.add_argument("--kept", type=Path, help="grammar: recheck.json (its kept_ids per set)")
    parser.add_argument("--pairs", type=Path, required=True, help="the confirm set (its strong answers)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--side", choices=["error", "rewrite"])
    parser.add_argument("--url", default="http://127.0.0.1:8010")
    parser.add_argument("--judge")
    parser.add_argument("--round", type=int, default=1)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    items = strong_answers(load_pairs(args.pairs))
    if args.stage == "grammar":
        sets = {n: load_pairs(Path(p)) for n, p in (x.split("=", 1) for x in args.wordswaps)}
        rk = json.loads(args.kept.read_text(encoding="utf-8"))
        stage_grammar(sets, {n: set(rk[n]["kept_ids"]) for n in sets}, args.out)
        rows = _load_state(args.out / "grammar-wordswaps.json")
        summary = {n: grammar_summary(r) for n, r in rows.items()}
        _save(args.out / "grammar-summary.json", summary)
        print(
            json.dumps({n: {k: v for k, v in r.items() if k != "drop"} for n, r in summary.items()}, indent=1)
        )
    elif args.stage == "propose":
        only = None
        if args.round > 1:
            only = set(_load_state(args.out / "pairing.json").get("unmatched", []))
        stage_propose(items, args.out, args.side, args.url, args.round, only)
    elif args.stage == "confirm":
        stage_confirm(items, args.out, args.side, args.judge)
    elif args.stage == "score":
        stage_score(items, args.out)
    else:
        kept = kept_candidates(args.out)
        matched, unmatched = {}, []
        for it in items:
            k = kept.get(it["prompt_id"], {"error": [], "rewrite": []})
            pick = choose_pair(k["error"], k["rewrite"])
            if pick:
                matched[it["prompt_id"]] = {"error": pick[0]["id"], "rewrite": pick[1]["id"]}
            else:
                unmatched.append(it["prompt_id"])
        _save(args.out / "pairing.json", {"matched": matched, "unmatched": unmatched})
        print("matched", len(matched), "unmatched", len(unmatched))
    print("STAGE-OK", args.stage)


if __name__ == "__main__":
    main()
