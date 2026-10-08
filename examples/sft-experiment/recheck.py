"""Addendum 4 of the 2026-10-08 Auditor plan: the context-rich meaning re-check of the word swaps.

The first meaning check was starved: it saw two sentences plus one of context, with thinking off.
This re-check gives each judge the purpose, the criteria, five worked examples from outside every
set, both full answers and the changed sentence. It takes one item per call, asks for reasoning
before the verdict, and constrains the reply with a JSON schema. Reasoning is the schema's first
required property, and the verdict fields are enums.

Two judges from two families, run one after the other (one model on the card at a time):
  - mistral-small:24b: no thinking mode (Ollama's /api/show), so it reasons in the reply;
  - gemma4:31b: thinking on, with room for it.

A swap is kept only when both judges say verdict "same" with claim_changed false. "unsure" is its
own verdict, counted and dropped. A reply cut off at num_predict (done_reason "length") is logged
as "truncated", never as a verdict. Every pair is judged, including those the first check dropped,
so the two checks can be compared item by item.

Usage:
  python recheck.py judge --judge mistral-small:24b --pairs NAME=PATH ... --out DIR
  python recheck.py combine --out DIR --pairs NAME=PATH ... --old NAME=verify.json ...
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from critic_heads import load_pairs  # noqa: E402
from lib import extract_json_object, refuse_cloud  # noqa: E402
from skeptic_pairs import sentences  # noqa: E402

SYSTEM = "You are a careful editor checking whether two versions of an answer say exactly the same thing."

REQUEST = """## Why this check exists

We are studying "critics": small models that score answers. To test whether a critic really detects
errors, or merely notices that an answer was edited, we need pairs of answers that differ by an edit
that changes NOTHING about what the answer claims: a word or phrase replaced by one that means the
same thing in this context. If such an edit secretly changes a claim, the pair is useless, because a
critic that flags it would be right to. Your job is to catch any edit that changes meaning, however
slightly, and to say so honestly when you can't tell.

## What counts

The meaning has CHANGED if the edit does any of these, in the context of the whole answer:
- adds, removes, strengthens or weakens a claim ("often" to "always", "can" to "will", "some" to "most");
- changes a number, unit, quantity, name, date, condition, cause, or the order of steps;
- swaps a term for a near-synonym with a different technical sense ("efficient" vs "effective",
  "accuracy" vs "precision", "weather" vs "climate");
- makes a sentence ungrammatical or unclear enough that a reader could take it differently.

The meaning is the SAME if a reader who knows the subject would draw exactly the same conclusions
from both versions. Differences of style, register or word choice alone don't count.

If you genuinely can't decide, say "unsure". Don't force a verdict: an honest "unsure" is useful.

## Worked examples (from other material, not from the answers you will judge)

Example 1.
Original sentence: "Plants use sunlight to turn carbon dioxide and water into glucose."
Edited sentence: "Plants use sunlight to convert carbon dioxide and water into glucose."
Reasoning: "turn ... into" and "convert ... into" mean the same here; no claim is added or changed.
Verdict: same; claim_changed false.

Example 2.
Original sentence: "Raising interest rates usually slows inflation."
Edited sentence: "Raising interest rates reliably slows inflation."
Reasoning: "usually" allows exceptions; "reliably" claims it nearly always works. The claim is
strengthened.
Verdict: changed; claim_changed true.

Example 3.
Original sentence: "The new process is more efficient, using less energy per unit."
Edited sentence: "The new process is more effective, using less energy per unit."
Reasoning: "efficient" is about resources per output; "effective" is about achieving the goal. In
this context they are different claims, even though the rest of the sentence hints at efficiency.
Verdict: changed; claim_changed true.

Example 4.
Original sentence: "Before you begin, back up your files."
Edited sentence: "Before you start, back up your files."
Reasoning: "begin" and "start" are interchangeable here.
Verdict: same; claim_changed false.

Example 5.
Original sentence: "The bridge was closed due to high winds."
Edited sentence: "The bridge was closed owing to high winds."
Reasoning: "due to" and "owing to" both give the cause; identical meaning.
Verdict: same; claim_changed false.

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

First reason it through in "reasoning": what exactly changed, what each version claims in the
context of the whole answer, and whether a knowledgeable reader would conclude anything different.
Then give "verdict" ("same", "changed" or "unsure") and "claim_changed" (true or false)."""

SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "verdict": {"type": "string", "enum": ["same", "changed", "unsure"]},
        "claim_changed": {"type": "boolean"},
    },
    "required": ["reasoning", "verdict", "claim_changed"],
    "additionalProperties": False,
}

# Per judge: whether it thinks, and the room it gets (R&D's pilot: gemma4 with thinking truncated
# every time at 6000 tokens).
JUDGES = {
    "mistral-small:24b": {"think": None, "num_predict": 2048, "num_ctx": 8192},
    "gemma4:31b": {"think": True, "num_predict": 12000, "num_ctx": 16384},
}


def request_body(judge: str, pair: dict) -> dict:
    original, edited = sentences(pair)
    content = REQUEST.format(
        prompt=pair["prompt"], strong=pair["strong"], flawed=pair["flawed"], original=original, edited=edited
    )
    cfg = JUDGES[judge]
    body = {
        "model": refuse_cloud(judge),
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}],
        "stream": False,
        "format": SCHEMA,
        "options": {"temperature": 0.0, "num_predict": cfg["num_predict"], "num_ctx": cfg["num_ctx"]},
    }
    if cfg["think"] is not None:
        body["think"] = cfg["think"]
    return body


def parse(response: dict) -> dict:
    """One judge's outcome: a verdict ("same", "changed", "unsure"), or "truncated" / "unparsed"."""
    message = response.get("message", {})
    out = {"thinking_chars": len(message.get("thinking") or ""), "done_reason": response.get("done_reason")}
    if response.get("done_reason") == "length":
        return out | {"outcome": "truncated"}
    data = extract_json_object(message.get("content") or "")
    if (
        not data
        or data.get("verdict") not in ("same", "changed", "unsure")
        or not isinstance(data.get("claim_changed"), bool)
    ):
        return out | {"outcome": "unparsed"}
    return out | {
        "outcome": data["verdict"],
        "claim_changed": data["claim_changed"],
        "reasoning": data.get("reasoning", ""),
    }


def keeps(outcome: dict) -> bool:
    return outcome.get("outcome") == "same" and outcome.get("claim_changed") is False


def combine(verdicts: dict[str, dict[str, dict]], pair_ids: list[str], old_dropped: set) -> dict:
    """verdicts: judge -> pair_id -> parsed outcome. The kept ids (both judges keep), each judge's
    outcome counts, the 2x2 agreement table, and the comparison with the first check."""
    a, b = list(verdicts)
    table = {"both_keep": 0, f"only_{a}": 0, f"only_{b}": 0, "neither": 0}
    counts = {j: {} for j in verdicts}
    kept = []
    for pid in pair_ids:
        ka, kb = keeps(verdicts[a][pid]), keeps(verdicts[b][pid])
        for j in verdicts:
            o = verdicts[j][pid]["outcome"]
            counts[j][o] = counts[j].get(o, 0) + 1
        key = "both_keep" if ka and kb else f"only_{a}" if ka else f"only_{b}" if kb else "neither"
        table[key] += 1
        if ka and kb:
            kept.append(pid)
    kept_set = set(kept)
    old_kept = set(pair_ids) - old_dropped
    return {
        "pairs": len(pair_ids),
        "kept": len(kept),
        "kept_ids": kept,
        "dropped_ids": [p for p in pair_ids if p not in kept_set],
        "judges_disagree": table[f"only_{a}"] + table[f"only_{b}"],
        "agreement": table,
        "outcomes": counts,
        "against_first_check": {
            "kept_by_both_checks": len(kept_set & old_kept),
            "kept_now_dropped_before": len(kept_set - old_kept),
            "dropped_now_kept_before": len(old_kept - kept_set),
            "dropped_by_both": len(set(pair_ids) - kept_set - old_kept),
        },
    }


def judge_all(judge: str, sets: dict[str, list[dict]], out: Path) -> None:  # pragma: no cover - Ollama
    import urllib.request

    def post(path, body, timeout=1800):
        req = urllib.request.Request(
            "http://127.0.0.1:11434" + path,
            data=json.dumps(body).encode("utf-8"),
            headers={"content-type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())

    caps = post("/api/show", {"model": judge}).get("capabilities", [])
    if JUDGES[judge]["think"] and "thinking" not in caps:
        raise SystemExit(f"{judge} has no thinking capability: {caps}")
    tags = post_get("http://127.0.0.1:11434/api/tags")
    digest = next(m["digest"][:12] for m in tags["models"] if m["name"] == judge)
    path = out / f"verdicts-{judge.replace(':', '_')}.json"
    done = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"verdicts": {}}
    result = {"judge": judge, "digest": digest, "capabilities": caps, "settings": JUDGES[judge]}
    verdicts = done["verdicts"]
    try:
        for name, pairs in sets.items():
            verdicts.setdefault(name, {})
            for p in pairs:
                if p["pair_id"] in verdicts[name]:
                    continue  # resumable: a finished item is never asked again
                t0 = time.time()
                parsed = parse(post("/api/chat", request_body(judge, p)))
                verdicts[name][p["pair_id"]] = parsed | {"seconds": round(time.time() - t0, 1)}
                path.write_text(json.dumps(result | {"verdicts": verdicts}, indent=1), encoding="utf-8")
            print(name, len(verdicts[name]), flush=True)
    finally:
        post("/api/generate", {"model": judge, "keep_alive": 0})


def post_get(url: str) -> dict:  # pragma: no cover - Ollama
    import urllib.request

    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read())


def main() -> None:  # pragma: no cover - needs the local Ollama daemon and the run data
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("judge")
    j.add_argument("--judge", choices=sorted(JUDGES), required=True)
    j.add_argument("--pairs", action="append", required=True, help="NAME=PATH")
    j.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("combine")
    c.add_argument("--pairs", action="append", required=True, help="NAME=PATH")
    c.add_argument("--old", action="append", required=True, help="NAME=first-check verify.json")
    c.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sets = {n: load_pairs(Path(p)) for n, p in (s.split("=", 1) for s in args.pairs)}
    if args.cmd == "judge":
        judge_all(args.judge, sets, args.out)
        print("RECHECK-JUDGE-OK", args.judge)
        return
    old = {}
    for n, p in (s.split("=", 1) for s in args.old):
        v = json.loads(Path(p).read_text(encoding="utf-8"))
        old[n] = set(v["flagged_changed_meaning"]) | set(v["unparsed"])
    verdicts = {
        jd: json.loads((args.out / f"verdicts-{jd.replace(':', '_')}.json").read_text(encoding="utf-8"))[
            "verdicts"
        ]
        for jd in JUDGES
    }
    result = {}
    for name, pairs in sets.items():
        ids = [p["pair_id"] for p in pairs]
        result[name] = combine({jd: verdicts[jd][name] for jd in JUDGES}, ids, old.get(name, set()))
    (args.out / "recheck.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(
        json.dumps(
            {n: {k: v for k, v in r.items() if not k.endswith("_ids")} for n, r in result.items()}, indent=1
        )
    )
    print("RECHECK-OK")


if __name__ == "__main__":
    main()
