"""A small evaluation set whose errors are planted by a different model family (the Auditor plan).

Every earlier planted pair came from Qwen2.5-32B-Instruct. A critic that keys on how that model
edits would pass all of those sets, so this set changes the planter and nothing else:

  answers  strong answers for the questions no other set uses (the "spare" prompts left after the
           seed-42 train/held-out split), written by the Qwen2.5-32B teacher in the student's chat
           format, as for the training set (llama-server, Q4);
  plant    one JSON sentence edit per flaw kind, planted by a local non-Qwen model through the
           Ollama daemon (gemma4:31b), with build_dataset.py's request, retries and filters.

The two stages need different servers, so they run separately; one model on the GPU at a time.
`plant` refuses any model name containing "cloud" (the standing rule: no Ollama Cloud) and records
the local model's ID. report.json also holds the edit statistics (characters changed, length
change, position in the answer) beside the same statistics for a reference set, so a set that is
easier or harder for reasons other than the planter shows.

Usage:
  python second_planter.py answers --questions data/questions.json --out second   (llama-server :8010)
  python second_planter.py plant --model gemma4:31b --out second --reference fresh/confirm_set.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from build_dataset import plant_errors  # noqa: E402
from fresh_pairs import changed_span  # noqa: E402
from lib import (  # noqa: E402
    FLAW_KINDS,
    STUDENT_SYSTEM,
    Backend,
    Chat,
    refuse_cloud,
    split_held_out,
    teacher_system_prompt,
)


def spare_prompts(
    kept: list[tuple[str, str]], held_out: int = 64, train_cap: int = 400
) -> list[tuple[str, str]]:
    """The (topic, prompt) questions that neither the training split nor the held-out split uses."""
    train, held = split_held_out(kept, held_out, train_cap, 42)
    used = {p for _, p in train} | {p for _, p in held}
    return [(t, p) for t, p in kept if p not in used]


def write_answers(backend: Backend, prompts: list[tuple[str, str]], answer_tokens: int = 1200) -> list[dict]:
    """One strong answer per prompt in the student's chat format; answers cut at the cap are dropped."""
    replies = backend.generate([Chat(STUDENT_SYSTEM, [("user", p)]) for _, p in prompts], answer_tokens, 0.7)
    cut = backend.last_truncated
    return [
        {"prompt_id": f"s{i}", "topic": t, "prompt": p, "strong": r.strip()}
        for i, ((t, p), r, c) in enumerate(zip(prompts, replies, cut))
        if not c and r.strip()
    ]


def edit_stats(pairs: list[dict]) -> dict:
    """Edit size and position, measured as the R&D session measures them, so the numbers compare.

    difflib at character level, autojunk off (on, it treats common characters as junk in texts over
    200 characters and the spans come out wrong): the characters changed sum the larger side of each
    non-equal opcode; the position is where the first differing character falls, as a fraction of
    the strong answer."""
    import difflib

    position, size = [], []
    for p in pairs:
        sm = difflib.SequenceMatcher(None, p["strong"], p["flawed"], autojunk=False)
        ops = [o for o in sm.get_opcodes() if o[0] != "equal"]
        if ops:
            position.append(ops[0][1] / max(1, len(p["strong"])))
            size.append(sum(max(o[2] - o[1], o[4] - o[3]) for o in ops))
    diff = [len(p["flawed"]) - len(p["strong"]) for p in pairs]
    n = max(len(pairs), 1)
    return {
        "pairs": len(pairs),
        "median_chars_changed": statistics.median(size) if size else None,
        "flawed_longer": sum(d > 0 for d in diff) / n,
        "same_length": sum(d == 0 for d in diff) / n,
        "median_length_change": statistics.median(diff) if diff else None,
        "median_position": statistics.median(position) if position else None,
    }


def plant(backend: Backend, planter: str, items: list[dict], attempts: int = 5) -> tuple[list[dict], dict]:
    """Plant one error of each kind in each strong answer; the pairs and the attempt outcomes."""
    held = [(it["topic"], it["prompt"]) for it in items]
    strong = [it["strong"] for it in items]
    slots = [(i, kind) for i in range(len(items)) for kind in FLAW_KINDS]
    flawed, log = plant_errors(backend, teacher_system_prompt(planter), held, strong, slots, attempts)
    pairs = []
    for i, kind in slots:
        if (i, kind) in flawed:
            it = items[i]
            pairs.append(
                {
                    "pair_id": f"{it['prompt_id']}-{FLAW_KINDS.index(kind)}",
                    "prompt_id": it["prompt_id"],
                    "topic": it["topic"],
                    "prompt": it["prompt"],
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
    return pairs, {"slots": len(slots), "planted": len(pairs), "attempt_outcomes": outcomes}


def main() -> None:  # pragma: no cover - needs local servers
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="stage", required=True)
    a = sub.add_parser("answers", help="strong answers from the Qwen2.5-32B teacher on llama-server")
    a.add_argument("--questions", type=Path, required=True)
    a.add_argument("--url", default="http://127.0.0.1:8010")
    a.add_argument("--out", type=Path, required=True)
    b = sub.add_parser("plant", help="errors planted by a local non-Qwen model through Ollama")
    b.add_argument("--model", default="gemma4:31b")
    b.add_argument("--reference", type=Path, help="a Qwen-planted set, for the edit statistics beside these")
    b.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.stage == "answers":
        from lib import ServerBackend

        kept = [tuple(q) for q in json.loads(args.questions.read_text(encoding="utf-8"))["kept"]]
        prompts = spare_prompts(kept)
        items = write_answers(ServerBackend(args.url), prompts)
        (args.out / "answers.json").write_text(
            json.dumps(items, indent=1, ensure_ascii=False), encoding="utf-8"
        )
        print(f"{len(items)} of {len(prompts)} spare prompts answered -> {args.out / 'answers.json'}")
        print("ANSWERS-OK")
        return

    from lib import OllamaBackend

    refuse_cloud(args.model)
    items = json.loads((args.out / "answers.json").read_text(encoding="utf-8"))
    backend = OllamaBackend(args.model)
    model_id = backend.model_id()
    try:
        pairs, report = plant(backend, args.model, items)
    finally:
        backend.unload()
    report |= {
        "planted_by": args.model,
        "planter_model_id": model_id,
        "answers_by": "Qwen2.5-32B-Instruct Q4_K_M (official GGUF, llama.cpp)",
        "prompts": len(items),
        "prompts_with_pairs": len({p["prompt_id"] for p in pairs}),
        "edit_stats": edit_stats(pairs),
    }
    if args.reference:
        report["reference_edit_stats"] = edit_stats(json.loads(args.reference.read_text(encoding="utf-8")))
    (args.out / "second_set.json").write_text(
        json.dumps(pairs, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    (args.out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    print("PLANT-OK", args.out)


if __name__ == "__main__":
    main()
