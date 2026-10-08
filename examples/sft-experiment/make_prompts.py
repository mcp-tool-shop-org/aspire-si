"""Draw N ASPIRE training prompts from the experiment's training questions, stratified by topic.

Step 2 of the 2026-10-08 plan trains critics on 128 prompts instead of the control's 32. The
prompts come from the 400 training questions build_dataset.py split off with seed 42, so they are
disjoint from the 64 held-out questions behind the judge set. They were already deduplicated
against the 32 control prompts by word overlap and embeddings. Topics are drawn round-robin, so
each of the 12 topic areas contributes about N/12 prompts.

Writes a JSON list of prompt strings, the format `aspire train --prompts` reads.

Usage: python make_prompts.py --questions data/questions.json --n 128 --out prompts-128.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from lib import split_held_out  # noqa: E402


def draw(
    kept: list[tuple[str, str]], n: int, seed: int = 0, held_out: int = 64, train_cap: int = 400
) -> list[str]:
    """N training prompts, round-robin over topics, from the same split build_dataset.py made."""
    train, _ = split_held_out(kept, held_out, train_cap, 42)
    by_topic: dict[str, list[str]] = {}
    for topic, prompt in train:
        by_topic.setdefault(topic, []).append(prompt)
    rng = random.Random(seed)
    for prompts in by_topic.values():
        rng.shuffle(prompts)
    topics = sorted(by_topic)
    picked: list[str] = []
    i = 0
    while len(picked) < n and any(by_topic[t] for t in topics):
        t = topics[i % len(topics)]
        if by_topic[t]:
            picked.append(by_topic[t].pop())
        i += 1
    return picked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--questions", type=Path, required=True, help="build_dataset.py's questions.json")
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    kept = [tuple(q) for q in json.loads(args.questions.read_text(encoding="utf-8"))["kept"]]
    prompts = draw(kept, args.n, args.seed)
    args.out.write_text(
        json.dumps(prompts, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"{len(prompts)} prompts -> {args.out}")


if __name__ == "__main__":
    main()
