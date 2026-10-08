"""Fresh planted-error pairs from the training prompts' strong answers, split for two uses.

The 2026-10-08 Kev confirmation plan needs pairs no judge has been read on. Each of the 400
training prompts already has a strong teacher answer (the `answer` rows of train.jsonl); none of
them is a held-out prompt behind the judge set. Errors are planted as build_dataset.py plants
them: one JSON sentence edit per flaw kind, applied by code, retried at rising temperature, and
kept only when the pair passes clean_dataset.py's filters.

The prompts are split by topic, round-robin, into `--confirm` confirmation prompts and the rest:
  - confirm_set.json: judge_set.json's format, for judge_kev.py and judge_logprob.py;
  - train_pairs.jsonl: {pair_id, prompt_id, topic, prompt, strong, flawed, flaw_kind, edit} rows for
    the R&D session's Kev fine-tune; `edit` is the changed span of the one edited sentence;
  - split.json: the prompt ids on each side;
  - report.json: counts, attempt outcomes and the planting model.

Usage (llama-server on 127.0.0.1:8010):
  python fresh_pairs.py --train data/train.jsonl --planted-by "Qwen2.5-32B-Instruct Q4_K_M" --out fresh
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from build_dataset import plant_errors  # noqa: E402
from lib import FLAW_KINDS, Backend, read_jsonl, teacher_system_prompt, write_jsonl  # noqa: E402


def strong_answers(rows: list[dict]) -> list[dict]:
    """One {prompt_id, topic, prompt, strong} per untruncated `answer` row, in file order."""
    out = []
    for row in rows:
        if row["kind"] != "answer" or row.get("truncated"):
            continue
        user = [m["content"] for m in row["messages"] if m["role"] == "user"]
        reply = [m["content"] for m in row["messages"] if m["role"] == "assistant"]
        out.append(
            {"prompt_id": f"t{len(out)}", "topic": row["topic"], "prompt": user[0], "strong": reply[0]}
        )
    return out


def split(items: list[dict], confirm: int, seed: int = 0) -> tuple[list[str], list[str]]:
    """Confirmation and training prompt ids: round-robin over topics, so each topic gives about
    confirm/len(topics) confirmation prompts."""
    by_topic: dict[str, list[str]] = {}
    for item in items:
        by_topic.setdefault(item["topic"], []).append(item["prompt_id"])
    rng = random.Random(seed)
    for ids in by_topic.values():
        rng.shuffle(ids)
    topics = sorted(by_topic)
    picked: list[str] = []
    i = 0
    while len(picked) < confirm and any(by_topic[t] for t in topics):
        t = topics[i % len(topics)]
        if by_topic[t]:
            picked.append(by_topic[t].pop())
        i += 1
    chosen = set(picked)
    return picked, [item["prompt_id"] for item in items if item["prompt_id"] not in chosen]


def changed_span(strong: str, flawed: str) -> dict[str, str]:
    """The part of the text that differs, after trimming the common start and end."""
    start = 0
    while start < min(len(strong), len(flawed)) and strong[start] == flawed[start]:
        start += 1
    end = 0
    while end < min(len(strong), len(flawed)) - start and strong[-1 - end] == flawed[-1 - end]:
        end += 1
    return {"original": strong[start : len(strong) - end], "edited": flawed[start : len(flawed) - end]}


def build(
    backend: Backend, teacher: str, items: list[dict], confirm: int, attempts: int = 5, planted_by: str = ""
) -> dict:
    """Plant one error of each kind per prompt, then split the kept pairs by prompt. `teacher` sets
    the system prompt, as on the pod; `planted_by` records the weights that actually served it."""
    held = [(item["topic"], item["prompt"]) for item in items]
    strong = [item["strong"] for item in items]
    slots = [(i, kind) for i in range(len(items)) for kind in FLAW_KINDS]
    flawed, log = plant_errors(backend, teacher_system_prompt(teacher), held, strong, slots, attempts)
    confirm_ids, train_ids = split(items, confirm)
    side = {pid: "confirm" for pid in confirm_ids} | {pid: "train" for pid in train_ids}
    pairs: dict[str, list[dict]] = {"confirm": [], "train": []}
    for i, kind in slots:
        if (i, kind) not in flawed:
            continue
        item = items[i]
        pairs[side[item["prompt_id"]]].append(
            {
                "pair_id": f"{item['prompt_id']}-{FLAW_KINDS.index(kind)}",
                "prompt_id": item["prompt_id"],
                "topic": item["topic"],
                "prompt": item["prompt"],
                "flaw_kind": kind,
                "method": "edit",
                "attempts": len(log[(i, kind)]),
                "strong": strong[i],
                "flawed": flawed[(i, kind)],
                "strong_truncated": False,
                "flawed_truncated": False,
                "edit": changed_span(strong[i], flawed[(i, kind)]),
            }
        )
    outcomes: dict[str, int] = {}
    for tries in log.values():
        for why in tries:
            key = (
                why.split(":")[0].split(" ")[0]
                if why.startswith(("similarity", "length"))
                else why.split(":")[0]
            )
            outcomes[key] = outcomes.get(key, 0) + 1
    return {
        "pairs": pairs,
        "split": {"confirm": confirm_ids, "train": train_ids},
        "report": {
            "planted_by": planted_by or teacher,
            "prompts": len(items),
            "slots": len(slots),
            "planted": len(flawed),
            "attempt_outcomes": outcomes,
            "confirm": {
                "prompts": len(confirm_ids),
                "pairs": len(pairs["confirm"]),
                "prompts_with_pairs": len({p["prompt_id"] for p in pairs["confirm"]}),
            },
            "train": {
                "prompts": len(train_ids),
                "pairs": len(pairs["train"]),
                "prompts_with_pairs": len({p["prompt_id"] for p in pairs["train"]}),
            },
        },
    }


def write(result: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "confirm_set.json").write_text(
        json.dumps(result["pairs"]["confirm"], indent=1, ensure_ascii=False), encoding="utf-8"
    )
    write_jsonl(
        out / "train_pairs.jsonl",
        [
            {
                k: p[k]
                for k in ("pair_id", "prompt_id", "topic", "prompt", "strong", "flawed", "flaw_kind", "edit")
            }
            for p in result["pairs"]["train"]
        ],
    )
    (out / "split.json").write_text(json.dumps(result["split"], indent=1), encoding="utf-8")
    (out / "report.json").write_text(json.dumps(result["report"], indent=1), encoding="utf-8")


def main() -> None:  # pragma: no cover - needs a running server
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train", type=Path, required=True, help="build_dataset.py's train.jsonl")
    parser.add_argument("--teacher", default="Qwen/Qwen2.5-32B-Instruct", help="sets the system prompt")
    parser.add_argument("--planted-by", required=True, help="the served weights, as recorded")
    parser.add_argument("--confirm", type=int, default=80)
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--url", default="http://127.0.0.1:8010")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, help="only the first N prompts (a smoke test)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from lib import ServerBackend

    items = strong_answers(read_jsonl(args.train))[: args.limit]
    result = build(
        ServerBackend(args.url, args.workers),
        args.teacher,
        items,
        args.confirm,
        args.attempts,
        args.planted_by,
    )
    write(result, args.out)
    print(json.dumps(result["report"], indent=1))
    print("FRESH-OK", args.out)


if __name__ == "__main__":
    main()
