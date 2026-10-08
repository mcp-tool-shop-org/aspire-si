"""Find the judge pairs the teacher can separate when it compares the two answers directly.

The teacher's whole-number scores tie most pairs (89 of 127 in the 2026-10-07 judge set), so a
subset defined by absolute scores is small. Here the teacher is shown both answers to the same
question and asked which is better, once with the strong answer first and once with it second.
A pair is *teacher-separable* when the teacher picks the strong answer both times; asking in both
orders cancels a preference for whichever answer comes first.

This reads the teacher's text choice only. judge_logprob.py asks the same question and reads its
A/B log-probabilities, which judge_kev.order_averaged averages over the two orders; read that way
the teacher separates most pairs (2026-10-08 Kev confirmation report).

Writes --out (pairwise.json): per pair, both picks and whether it is separable, the list of
separable pair_ids, and counts (pairs, prompts, how often the first answer was picked).

Usage on the pod: python pairwise_teacher.py --judge-set data/judge_set.json --out results/pairwise.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from lib import Backend, Chat, teacher_system_prompt  # noqa: E402

PAIRWISE_REQUEST = """Here is a question and two answers to it.

Question: {prompt}

Answer A:
{a}

Answer B:
{b}

The two answers may differ in only a sentence. Which answer is more correct? Judge correctness
of facts, numbers and reasoning, not style or length. Reply with one letter only: A or B."""


def parse_choice(reply: str) -> str | None:
    """The letter the teacher chose: the first standalone A or B, or None if there is none."""
    match = re.search(r"\b([AB])\b", reply.strip())
    return match.group(1) if match else None


def compare(backend: Backend, teacher: str, pairs: list[dict]) -> dict:
    system = teacher_system_prompt(teacher)
    chats = []
    for p in pairs:
        chats.append(
            Chat(
                system, [("user", PAIRWISE_REQUEST.format(prompt=p["prompt"], a=p["strong"], b=p["flawed"]))]
            )
        )
        chats.append(
            Chat(
                system, [("user", PAIRWISE_REQUEST.format(prompt=p["prompt"], a=p["flawed"], b=p["strong"]))]
            )
        )
    replies = backend.generate(chats, 8, 0.0)
    rows = []
    for k, p in enumerate(pairs):
        first, second = parse_choice(replies[2 * k]), parse_choice(replies[2 * k + 1])
        rows.append(
            {
                "pair_id": p.get("pair_id", k),
                "prompt_id": p.get("prompt_id", p["prompt"]),
                "strong_first": first,  # "A" picks the strong answer
                "strong_second": second,  # "B" picks the strong answer
                "separable": first == "A" and second == "B",
            }
        )
    picks = [r[key] for r in rows for key in ("strong_first", "strong_second")]
    separable = [r for r in rows if r["separable"]]
    return {
        "pairs": len(rows),
        "separable_pairs": len(separable),
        "separable_prompts": len({r["prompt_id"] for r in separable}),
        "separable_ids": [r["pair_id"] for r in separable],
        "first_answer_rate": picks.count("A") / max(len(picks), 1),
        "unparsed": picks.count(None),
        "rows": rows,
    }


def main() -> None:  # pragma: no cover - runs on the pod
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--judge-set", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--teacher", default="Qwen/Qwen2.5-32B-Instruct")
    args = parser.parse_args()

    from lib import VllmBackend

    pairs = json.loads(args.judge_set.read_text(encoding="utf-8"))
    result = compare(VllmBackend(args.teacher), args.teacher, pairs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("rows", "separable_ids")}, indent=1))
    print("PAIRWISE-OK", args.out)


if __name__ == "__main__":
    main()
