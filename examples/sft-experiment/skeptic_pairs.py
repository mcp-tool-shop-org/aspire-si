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
import json
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


VERIFY_REQUEST = """Two versions of one sentence from an answer are shown. The second was reworded.

Original: {original}

Reworded: {edited}

Does the reworded sentence say exactly the same thing as the original, with every fact, number,
condition and step of reasoning unchanged? Small differences of wording or style don't count.

Reply with a JSON object only: {{"same_meaning": true or false, "reason": "<a few words>"}}"""


def strong_items(pairs: list[dict]) -> list[dict]:
    """One {prompt_id, topic, prompt, strong} per distinct strong answer, in first-seen order."""
    seen, items = set(), []
    for p in pairs:
        key = (p["prompt_id"], p["strong"])
        if key in seen:
            continue
        seen.add(key)
        items.append(
            {
                "prompt_id": p["prompt_id"],
                "topic": p.get("topic", ""),
                "prompt": p["prompt"],
                "strong": p["strong"],
            }
        )
    return items


def _pairs(items, edited, log, slots, tag: str) -> list[dict]:
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
    import difflib

    ops = [
        o
        for o in difflib.SequenceMatcher(None, pair["strong"], pair["flawed"], autojunk=False).get_opcodes()
        if o[0] != "equal"
    ]
    return sum(max(o[2] - o[1], o[4] - o[3]) for o in ops)


def _round(backend, system, items, held, strong, slots, tag, max_chars, outcomes):
    """One planting pass over `slots`; pairs over `max_chars` are rejected like a failed filter."""
    edited, log = plant_errors(backend, system, held, strong, slots, 1, PARAPHRASE_REQUEST)
    for tries in log.values():
        for why in tries:
            key = (
                why.split(":")[0].split(" ")[0]
                if why.startswith(("similarity", "length"))
                else why.split(":")[0]
            )
            outcomes[key] = outcomes.get(key, 0) + 1
    kept, too_big = [], 0
    for p in _pairs(items, edited, log, slots, tag):
        if max_chars is not None and edit_chars(p) > max_chars:
            too_big += 1
            continue
        kept.append(p)
    outcomes["over the size gate"] = outcomes.get("over the size gate", 0) + too_big
    return kept


def plant(
    backend: Backend,
    system: str,
    items: list[dict],
    rounds: int = 7,
    per_answer: int = 2,
    max_chars: int | None = None,
) -> tuple[list[dict], dict]:
    """Up to `per_answer` paraphrases per strong answer, each on a different sentence and no larger
    than `max_chars` changed characters (the size gate); the pairs (original as "strong", reworded as
    "flawed"). Each answer gets up to `rounds` attempts per paraphrase; an attempt that fails the
    filters or the size gate is retried."""
    held = [(it["topic"], it["prompt"]) for it in items]
    strong = [it["strong"] for it in items]
    outcomes: dict[str, int] = {}
    first: dict = {}
    for _ in range(rounds):
        todo = [(i, FIRST) for i, it in enumerate(items) if it["prompt_id"] not in first]
        if not todo:
            break
        for p in _round(backend, system, items, held, strong, todo, "p1", max_chars, outcomes):
            first.setdefault(p["prompt_id"], p)
    pairs = list(first.values())
    second: dict = {}
    dropped_same_sentence = 0
    if per_answer > 1:
        index = {it["prompt_id"]: i for i, it in enumerate(items)}
        for _ in range(rounds):
            todo = [(index[pid], avoid(sentences(p)[0])) for pid, p in first.items() if pid not in second]
            if not todo:
                break
            for p in _round(backend, system, items, held, strong, todo, "p2", max_chars, outcomes):
                if overlaps(p, first[p["prompt_id"]]):
                    dropped_same_sentence += 1
                    continue
                second.setdefault(p["prompt_id"], p)
        pairs += list(second.values())
    attempts = sum(outcomes.values())
    return pairs, {
        "strong_answers": len(items),
        "planted": len(pairs),
        "answers_with_a_pair": len(first),
        "answers_with_no_pair": len(items) - len(first),
        "answers_with_two_pairs": len(second),
        "dropped_second_on_same_sentence": dropped_same_sentence,
        "size_gate_chars": max_chars,
        "size_gate_rejection_rate": outcomes.get("over the size gate", 0) / attempts if attempts else 0.0,
        "attempt_outcomes": outcomes,
    }


def sentence_of(text: str, start: int, end: int) -> str:
    """The sentence of `text` that holds [start, end): from the previous sentence end to the next."""
    lo = max(text.rfind(c, 0, start) for c in ".!?\n") + 1
    hi_candidates = [i for i in (text.find(c, end) for c in ".!?\n") if i >= 0]
    hi = (min(hi_candidates) + 1) if hi_candidates else len(text)
    return text[lo:hi].strip()


def sentences(pair: dict) -> tuple[str, str]:
    """The original and reworded sentence of a paraphrase pair."""
    import difflib

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


def verify(backend: Backend, system: str, pairs: list[dict]) -> dict:
    """A local judge's verdict on each pair: did the rewording keep the meaning?"""
    chats = []
    for p in pairs:
        original, edited = sentences(p)
        chats.append(Chat(system, [("user", VERIFY_REQUEST.format(original=original, edited=edited))]))
    replies = backend.generate(chats, 200, 0.0)
    verdicts = []
    for p, reply in zip(pairs, replies):
        data = extract_json_object(reply) or {}
        same = data.get("same_meaning")
        verdicts.append(
            {
                "pair_id": p["pair_id"],
                "same_meaning": same if isinstance(same, bool) else None,
                "reason": data.get("reason", ""),
            }
        )
    flagged = [v["pair_id"] for v in verdicts if v["same_meaning"] is False]
    unparsed = [v["pair_id"] for v in verdicts if v["same_meaning"] is None]
    return {
        "pairs": len(pairs),
        "kept_meaning": sum(v["same_meaning"] is True for v in verdicts),
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
        # The size gate: at most twice the median edit of this file's error pairs (plan, addendum 2).
        max_chars = 2 * edit_stats(source)["median_chars_changed"]
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
