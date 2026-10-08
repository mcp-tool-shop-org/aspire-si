"""Paraphrase pairs for the Skeptic control (docs/runs/2026-10-08-auditor-plan.md, addendum 2).

Every planted error pair so far has its error in the edited copy, so a head could score well just by
noticing which copy was edited. A paraphrase pair holds a strong answer and the same answer with one
sentence reworded, meaning unchanged and no error, planted the way errors were (a JSON sentence edit
applied by code, retried at rising temperature, the same filters). In the written file the original
sits in the "strong" slot and the paraphrased copy in the "flawed" slot, so a head is read exactly as
on error pairs: a "win" means it ranks the edited copy as the worse or flawed one.

  plant     two paraphrases per distinct strong answer of a pairs file, on different sentences, by
            the model that planted that file's errors: llama-server (Qwen2.5-32B Q4) or a local
            Ollama model (gemma4:31b, thinking off, cloud models refused); edit statistics beside
            the matched error pairs'.
  verify    a local judge of another family reads each original and reworded sentence and says
            whether the meaning or any fact changed; the rate is reported, and pairs it flags are
            listed (the readout leaves them out).

Usage:
  python skeptic_pairs.py plant --pairs fresh/confirm_set.json --url http://127.0.0.1:8010
      --model "Qwen2.5-32B-Instruct Q4_K_M" --out skeptic/pconfirm         (Qwen-planted errors)
  python skeptic_pairs.py plant --pairs second/second_set.json --out skeptic/psecond   (gemma4:31b)
  python skeptic_pairs.py verify --out skeptic/pconfirm --judge mistral-small:24b
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from build_dataset import plant_errors  # noqa: E402
from critic_heads import load_pairs  # noqa: E402
from fresh_pairs import changed_span  # noqa: E402
from lib import PARAPHRASE_REQUEST, Backend, Chat, extract_json_object, refuse_cloud  # noqa: E402
from second_planter import edit_stats  # noqa: E402

FIRST = "Choose any one sentence."


def avoid(sentence: str) -> str:
    """The instruction for a second rewording: any sentence but the one already reworded."""
    return f'Choose a different sentence from this one, which must stay exactly as it is: "{sentence}".'


VERIFY_REQUEST = """A sentence from an answer was reworded. It is shown before and after, with one sentence of
context on each side.

Context before: {before}

Original: {original}

Reworded: {edited}

Context after: {after}

1. Does the reworded sentence say exactly the same thing as the original, in this context?
2. Does the rewording add, remove, strengthen or weaken any claim (for example "most" to "all",
   "can" to "will", or a changed number, condition or name)?

Small differences of wording or style don't count.

Reply with a JSON object only:
{{"same_meaning": true or false, "claim_changed": true or false, "reason": "<a few words>"}}"""

# The lexical guard (R&D review of the size gate): swaps that touch these words or token kinds are
# the ones that flip a claim, and a 24B judge lets some of them through, so they are rejected by rule
# before any model check.
QUANTIFIERS = {"all", "every", "each", "most", "many", "some", "few", "several", "none", "no", "any"}
FREQUENCY = {"always", "usually", "often", "sometimes", "rarely", "never", "seldom", "frequently"}
MODALS = {"can", "could", "may", "might", "must", "should", "will", "would", "shall"}
NEGATION = {"not", "no", "without", "nor", "neither", "never"}
COMPARATIVES = {"more", "less", "most", "least", "fewer", "better", "worse", "best", "worst"}
NUMBER_WORDS = {
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "hundred",
    "thousand",
    "million",
    "billion",
    "half",
    "double",
    "twice",
    "triple",
    "single",
    "dozen",
    "first",
    "second",
    "third",
    "once",
}
UNITS = {
    "percent",
    "%",
    "km",
    "m",
    "cm",
    "mm",
    "kg",
    "g",
    "mg",
    "s",
    "ms",
    "hz",
    "khz",
    "mhz",
    "ghz",
    "mph",
    "kph",
    "degrees",
    "celsius",
    "fahrenheit",
    "kelvin",
    "c",
    "f",
    "k",
    "b",
    "kb",
    "mb",
    "gb",
    "tb",
    "byte",
    "bytes",
    "bit",
    "bits",
    "meter",
    "meters",
    "metre",
    "metres",
    "kilometer",
    "kilometers",
    "second",
    "seconds",
    "minute",
    "minutes",
    "hour",
    "hours",
    "day",
    "days",
    "year",
    "years",
    "watt",
    "watts",
    "volt",
    "volts",
    "joule",
    "joules",
    "newton",
    "newtons",
    "liter",
    "liters",
    "litre",
    "litres",
    "pound",
    "pounds",
    "dollar",
    "dollars",
    "$",
}
_GUARD_CLASSES = (
    ("quantifier", QUANTIFIERS),
    ("frequency", FREQUENCY),
    ("modal", MODALS),
    ("negation", NEGATION),
    ("comparative", COMPARATIVES),
    ("number word", NUMBER_WORDS),
    ("unit", UNITS),
)


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+(?:'\w+)?|[^\w\s]", text)


def lexical_guard(pair: dict) -> str | None:
    """Why the swap may flip a claim, or None. Looks at the changed tokens on either side."""
    a, b = _tokens(pair["strong"]), _tokens(pair["flawed"])
    changed: list[tuple[str, list[str], int]] = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op == "equal":
            continue
        changed += [(a[i], a, i) for i in range(i1, i2)] + [(b[j], b, j) for j in range(j1, j2)]
    words = [t.lower() for t, _, _ in changed]
    for tok, seq, i in changed:
        low = tok.lower()
        if "n't" in low:
            return "negation"
        for name, vocabulary in _GUARD_CLASSES:
            if low in vocabulary:
                return name
        if any(ch.isdigit() for ch in tok):
            return "number"
        if tok[:1].isupper() and i > 0 and seq[i - 1] not in {".", "!", "?", ":", '"', "\n"}:
            return "named entity"
        for other in words:
            if other != low and (low in (other + "er", other + "est", other + "r", other + "st")):
                return "comparative"
    return None


def strong_items(pairs: list[dict]) -> list[dict]:
    """One {prompt_id, topic, prompt, strong, error_chars} per distinct strong answer, in first-seen
    order. error_chars is the median edit size of that answer's own error pairs (the size gate)."""
    seen: dict = {}
    sizes: dict = {}
    for p in pairs:
        key = (p["prompt_id"], p["strong"])
        sizes.setdefault(key, []).append(edit_chars(p))
        if key in seen:
            continue
        seen[key] = {
            "prompt_id": p["prompt_id"],
            "topic": p.get("topic", ""),
            "prompt": p["prompt"],
            "strong": p["strong"],
        }
    return [item | {"error_chars": statistics.median(sizes[key])} for key, item in seen.items()]


def size_cap(item: dict, floor: int = 8) -> int:
    """The size gate for one answer: twice its own error edit's size, at least `floor` characters."""
    return max(floor, int(2 * item.get("error_chars", 0)))


def _pairs(items, edited, log, slots, tag: str, attempt_round: int) -> list[dict]:
    out = []
    for i, kind in slots:
        if (i, kind) in edited:
            it = items[i]
            out.append(
                {
                    "pair_id": f"{it['prompt_id']}-{tag}",
                    "prompt_id": it["prompt_id"],
                    "topic": it["topic"],
                    "prompt": it["prompt"],
                    "flaw_kind": "none (paraphrase)",
                    "method": "paraphrase",
                    "attempts": len(log[(i, kind)]),
                    "attempt_round": attempt_round,
                    "strong": it["strong"],
                    "flawed": edited[(i, kind)],
                    "strong_truncated": False,
                    "flawed_truncated": False,
                    "edit": changed_span(it["strong"], edited[(i, kind)]),
                }
            )
    return out


def overlaps(a: dict, b: dict) -> bool:
    """Whether two paraphrases of the same answer touch the same sentence."""
    sa, sb = sentences(a)[0], sentences(b)[0]
    return bool(sa) and sa == sb


def edit_chars(pair: dict) -> int:
    """Characters changed (difflib, autojunk off: the larger side of each non-equal opcode)."""
    ops = [
        o
        for o in difflib.SequenceMatcher(None, pair["strong"], pair["flawed"], autojunk=False).get_opcodes()
        if o[0] != "equal"
    ]
    return sum(max(o[2] - o[1], o[4] - o[3]) for o in ops)


def _round(backend, system, items, held, strong, slots, tag, attempt_round, max_chars, outcomes):
    """One planting pass over `slots`. A pair over its size cap, or one the lexical guard stops, is
    rejected like a failed filter; each reason is counted."""
    edited, log = plant_errors(backend, system, held, strong, slots, 1, PARAPHRASE_REQUEST)
    for tries in log.values():
        for why in tries:
            key = (
                why.split(":")[0].split(" ")[0]
                if why.startswith(("similarity", "length"))
                else why.split(":")[0]
            )
            outcomes[key] = outcomes.get(key, 0) + 1
    kept = []
    for p in _pairs(items, edited, log, slots, tag, attempt_round):
        it = next(x for x in items if x["prompt_id"] == p["prompt_id"])
        cap = max_chars if max_chars is not None else size_cap(it)
        if edit_chars(p) > cap:
            outcomes["over the size gate"] = outcomes.get("over the size gate", 0) + 1
            continue
        why = lexical_guard(p)
        if why:
            outcomes[f"lexical guard: {why}"] = outcomes.get(f"lexical guard: {why}", 0) + 1
            continue
        kept.append(p)
    return kept


def plant(
    backend: Backend,
    system: str,
    items: list[dict],
    rounds: int = 7,
    per_answer: int = 2,
    max_chars: int | None = None,
) -> tuple[list[dict], dict]:
    """Up to `per_answer` paraphrases per strong answer, each on a different sentence, within its
    size cap (twice the answer's own error edit, at least 8 characters, or `max_chars` if given) and
    past the lexical guard; the pairs (original as "strong", reworded as "flawed"). Each paraphrase
    gets up to `rounds` attempts; the round that produced it is recorded."""
    held = [(it["topic"], it["prompt"]) for it in items]
    strong = [it["strong"] for it in items]
    outcomes: dict[str, int] = {}
    first: dict = {}
    for r in range(1, rounds + 1):
        todo = [(i, FIRST) for i, it in enumerate(items) if it["prompt_id"] not in first]
        if not todo:
            break
        for p in _round(backend, system, items, held, strong, todo, "p1", r, max_chars, outcomes):
            first.setdefault(p["prompt_id"], p)
    pairs = list(first.values())
    second: dict = {}
    dropped_same_sentence = 0
    if per_answer > 1:
        index = {it["prompt_id"]: i for i, it in enumerate(items)}
        for r in range(1, rounds + 1):
            todo = [(index[pid], avoid(sentences(p)[0])) for pid, p in first.items() if pid not in second]
            if not todo:
                break
            for p in _round(backend, system, items, held, strong, todo, "p2", r, max_chars, outcomes):
                if overlaps(p, first[p["prompt_id"]]):
                    dropped_same_sentence += 1
                    continue
                second.setdefault(p["prompt_id"], p)
        pairs += list(second.values())
    attempts = sum(outcomes.values())
    guard = sum(v for k, v in outcomes.items() if k.startswith("lexical guard"))
    return pairs, {
        "strong_answers": len(items),
        "planted": len(pairs),
        "answers_with_a_pair": len(first),
        "answers_with_no_pair": len(items) - len(first),
        "answers_with_two_pairs": len(second),
        "dropped_second_on_same_sentence": dropped_same_sentence,
        "size_gate": "per answer: 2x its own error edit, at least 8 characters"
        if max_chars is None
        else max_chars,
        "size_gate_rejection_rate": outcomes.get("over the size gate", 0) / attempts if attempts else 0.0,
        "lexical_guard_rejection_rate": guard / attempts if attempts else 0.0,
        "kept_on_first_round": sum(p["attempt_round"] == 1 for p in pairs),
        "attempt_outcomes": outcomes,
    }


def quantiles(values: list[float]) -> list[float] | None:
    """The 25th, 50th and 75th percentiles."""
    if not values:
        return None
    v = sorted(values)
    return [v[int(q * (len(v) - 1))] for q in (0.25, 0.5, 0.75)]


def sentence_of(text: str, start: int, end: int) -> str:
    """The sentence of `text` that holds [start, end): from the previous sentence end to the next."""
    return text[_sentence_bounds(text, start, end)[0] : _sentence_bounds(text, start, end)[1]].strip()


def _sentence_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    lo = max(text.rfind(c, 0, start) for c in ".!?\n") + 1
    hi_candidates = [i for i in (text.find(c, end) for c in ".!?\n") if i >= 0]
    hi = (min(hi_candidates) + 1) if hi_candidates else len(text)
    return lo, hi


def sentences(pair: dict) -> tuple[str, str]:
    """The original and reworded sentence of a paraphrase pair."""
    ops = [
        o
        for o in difflib.SequenceMatcher(None, pair["strong"], pair["flawed"], autojunk=False).get_opcodes()
        if o[0] != "equal"
    ]
    if not ops:
        return "", ""
    return (
        sentence_of(pair["strong"], ops[0][1], ops[-1][2]),
        sentence_of(pair["flawed"], ops[0][3], ops[-1][4]),
    )


def context(pair: dict) -> tuple[str, str]:
    """One sentence of the original answer before and after the reworded sentence."""
    ops = [
        o
        for o in difflib.SequenceMatcher(None, pair["strong"], pair["flawed"], autojunk=False).get_opcodes()
        if o[0] != "equal"
    ]
    if not ops:
        return "", ""
    text = pair["strong"]
    lo, hi = _sentence_bounds(text, ops[0][1], ops[-1][2])
    before = text[_sentence_bounds(text, max(lo - 2, 0), max(lo - 2, 0))[0] : lo].strip() if lo > 1 else ""
    after = text[hi : _sentence_bounds(text, hi + 1, hi + 1)[1]].strip() if hi < len(text) - 1 else ""
    return before, after


def verify(backend: Backend, system: str, pairs: list[dict]) -> dict:
    """A local judge's verdict on each pair, with a sentence of context either side. A pair is kept
    only on "same meaning: true" and "claim changed: false"; anything else, unparsed included, is
    dropped."""
    chats = []
    for p in pairs:
        original, edited = sentences(p)
        before, after = context(p)
        chats.append(
            Chat(
                system,
                [
                    (
                        "user",
                        VERIFY_REQUEST.format(before=before, original=original, edited=edited, after=after),
                    )
                ],
            )
        )
    replies = backend.generate(chats, 200, 0.0)
    verdicts = []
    for p, reply in zip(pairs, replies):
        data = extract_json_object(reply) or {}
        same, changed = data.get("same_meaning"), data.get("claim_changed")
        verdicts.append(
            {
                "pair_id": p["pair_id"],
                "same_meaning": same if isinstance(same, bool) else None,
                "claim_changed": changed if isinstance(changed, bool) else None,
                "reason": data.get("reason", ""),
            }
        )
    unparsed = [v["pair_id"] for v in verdicts if v["same_meaning"] is None or v["claim_changed"] is None]
    flagged = [
        v["pair_id"]
        for v in verdicts
        if v["pair_id"] not in unparsed and not (v["same_meaning"] and not v["claim_changed"])
    ]
    return {
        "pairs": len(pairs),
        "kept_meaning": len(pairs) - len(flagged) - len(unparsed),
        "flagged_changed_meaning": flagged,
        "unparsed": unparsed,
        "changed_rate": len(flagged) / max(len(pairs), 1),
        "verdicts": verdicts,
    }


def main() -> None:  # pragma: no cover - needs the local Ollama daemon
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="stage", required=True)
    a = sub.add_parser("plant")
    a.add_argument(
        "--pairs", type=Path, required=True, help="the error pairs whose strong answers are reworded"
    )
    a.add_argument(
        "--model", default="gemma4:31b", help="an Ollama model, or the served model's name with --url"
    )
    a.add_argument("--url", help="an OpenAI-compatible server (llama-server) instead of Ollama")
    a.add_argument("--teacher", default="Qwen/Qwen2.5-32B-Instruct", help="system prompt for --url")
    a.add_argument("--per-answer", type=int, default=2)
    a.add_argument("--out", type=Path, required=True)
    b = sub.add_parser("verify")
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--judge", default="mistral-small:24b", help="a local model of another family")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from lib import OllamaBackend, teacher_system_prompt

    if args.stage == "plant":
        source = load_pairs(args.pairs)
        # The size gate is per answer: twice that answer's own error edit, at least 8 characters.
        max_chars = None
        if args.url:
            from lib import ServerBackend

            backend, model_id, system = (
                ServerBackend(args.url),
                args.model,
                teacher_system_prompt(args.teacher),
            )
            pairs, report = plant(
                backend, system, strong_items(source), per_answer=args.per_answer, max_chars=max_chars
            )
        else:
            refuse_cloud(args.model)
            backend = OllamaBackend(args.model)
            model_id = backend.model_id()
            try:
                pairs, report = plant(
                    backend,
                    teacher_system_prompt(args.model),
                    strong_items(source),
                    per_answer=args.per_answer,
                    max_chars=max_chars,
                )
            finally:
                backend.unload()
        planted_ids = {p["prompt_id"] for p in pairs}
        matched = [p for p in source if p["prompt_id"] in planted_ids]
        report |= {
            "planted_by": args.model,
            "planter_model_id": model_id,
            "provenance": f"Paraphrases planted by {args.model} (local Ollama {model_id}); "
            f"strong answers from {args.pairs.name}.",
            "edit_stats": edit_stats(pairs),
            "matched_error_edit_stats": edit_stats(matched),
        }
        report["size_quantiles_paraphrase"] = quantiles([edit_chars(p) for p in pairs])
        report["size_quantiles_matched_errors"] = quantiles([edit_chars(p) for p in matched])
        sizes = (
            report["edit_stats"]["median_chars_changed"],
            report["matched_error_edit_stats"]["median_chars_changed"],
        )
        report["edit_size_ratio"] = sizes[0] / sizes[1] if sizes[1] else None
        report["edit_size_caveat"] = bool(report["edit_size_ratio"]) and not (
            0.5 <= report["edit_size_ratio"] <= 2
        )
        (args.out / "pairs.json").write_text(
            json.dumps(pairs, indent=1, ensure_ascii=False), encoding="utf-8"
        )
        (args.out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(json.dumps(report, indent=1))
        print("PLANT-OK", args.out)
        return

    refuse_cloud(args.judge)
    pairs = json.loads((args.out / "pairs.json").read_text(encoding="utf-8"))
    backend = OllamaBackend(args.judge)
    model_id = backend.model_id()
    try:
        result = verify(backend, "You check whether two sentences say the same thing.", pairs)
    finally:
        backend.unload()
    result |= {"judge": args.judge, "judge_model_id": model_id}
    (args.out / "verify.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "verdicts"}, indent=1))
    print("VERIFY-OK", args.out)


if __name__ == "__main__":
    main()
