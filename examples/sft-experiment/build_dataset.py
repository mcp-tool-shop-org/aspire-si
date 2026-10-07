"""Plan A of the fine-tune-then-ASPIRE experiment: build the dataset, held-out prompts and judge set.

Everything is written by the teacher (Qwen2.5-32B-Instruct, Apache-2.0). Nothing comes from the 32
control prompts or their cached answers: new prompts near any of them are dropped.

  1. Questions: the teacher writes questions per topic area (it never sees the evaluation prompts).
     Near-duplicates and anything close to an evaluation prompt are dropped, by content-word
     overlap or by embedding cosine (BAAI/bge-small-en-v1.5, MIT), which catches paraphrases.
  2. Split: `--held-out` prompts (stratified by topic) are set aside and never trained on.
  3. Training examples: per prompt, the teacher answers in the student's chat format, challenges
     that answer as an ASPIRE teacher would, and writes a strong revision. Each answer and
     revision is scored with ASPIRE's own scoring request; only those at `--min-score` or above
     are kept. That is up to two examples per prompt: [prompt -> answer] and
     [prompt -> answer -> challenge -> revision].
  4. Judge set: per held-out prompt, a strong answer and `--flaws-per-prompt` minimal edits of
     it, each changing one fact, number or reasoning step. The teacher names one sentence and its
     edited form as JSON and the code applies it, so every pair differs only there. An edit that
     is not found, unchanged, self-flagging or fails clean_dataset.py's filters is asked for again
     (up to --edit-attempts times, at rising temperature). The planted error is the label; the
     teacher scores both sides only as a reference.
  5. Noise floor: the held-out strong answers are scored a second time.

Every generated answer records whether it stopped at the token cap (`truncated`), so
clean_dataset.py can drop it. `--questions` reuses an earlier run's questions.json, which with
the same seed gives the same training and held-out prompts. `--judge-only --from <dir>` keeps an
earlier run's training data, strong answers and passing pairs, and plants errors only where a pair
is missing.

Writes to --out: questions.json, train.jsonl, held_out.json, judge_set.json, noise.json and
report.json (counts, drop reasons, parse rates, score spreads).

Usage on the pod:  python build_dataset.py --teacher Qwen/Qwen2.5-32B-Instruct
                     --eval-prompts ../local-run/prompts.json --out data
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from clean_dataset import pair_problems  # noqa: E402
from lib import (  # noqa: E402
    EDIT_REQUEST,
    FLAW_KINDS,
    STUDENT_SYSTEM,
    TOPICS,
    Backend,
    Chat,
    apply_edit,
    challenge_request,
    challenge_types,
    dedupe,
    pairwise_accuracy,
    parse_question_list,
    question_request,
    score_of,
    scoring_request,
    split_held_out,
    teacher_system_prompt,
    write_jsonl,
)

EDIT_TEMPERATURES = (0.7, 0.8, 0.9, 1.0, 1.0)


def plant_errors(
    backend: Backend,
    system: str,
    held: list[tuple[str, str]],
    strong: list[str],
    slots: list[tuple[int, str]],
    attempts: int = 5,
) -> tuple[dict[tuple[int, str], str], dict[tuple[int, str], list[str]]]:
    """Flawed answers for (prompt index, flaw kind) slots, and every attempt's outcome per slot."""
    flawed: dict[tuple[int, str], str] = {}
    log: dict[tuple[int, str], list[str]] = {slot: [] for slot in slots}
    pending = list(slots)
    for attempt in range(attempts):
        if not pending:
            break
        replies = backend.generate(
            [
                Chat(system, [("user", EDIT_REQUEST.format(prompt=held[i][1], answer=strong[i], kind=kind))])
                for i, kind in pending
            ],
            600,
            EDIT_TEMPERATURES[min(attempt, len(EDIT_TEMPERATURES) - 1)],
        )
        retry = []
        for (i, kind), reply in zip(pending, replies):
            edited, why = apply_edit(reply, strong[i])
            if edited is not None:
                problems, _ = pair_problems(
                    {
                        "strong": strong[i],
                        "flawed": edited,
                        "strong_truncated": False,
                        "flawed_truncated": False,
                    }
                )
                why = "; ".join(problems) if problems else "ok"
            log[(i, kind)].append(why)
            if why == "ok":
                flawed[(i, kind)] = edited
            else:
                retry.append((i, kind))
        pending = retry
    return flawed, log


def judge_report(judge: list[dict], log: dict, prompts: int) -> dict:
    pair_strong = [j["teacher_strong"] for j in judge]
    pair_flawed = [j["teacher_flawed"] for j in judge]
    outcomes: dict[str, int] = {}
    for tries in log.values():
        for why in tries:
            first = why.split("; ")[0]
            key = first.split(" ")[0] if first.startswith(("similarity", "length")) else first.split(":")[0]
            outcomes[key] = outcomes.get(key, 0) + 1
    return {
        "prompts": prompts,
        "pairs": len(judge),
        "label": "the planted error: the strong answer is always the better one",
        "teacher_reference_accuracy": pairwise_accuracy(pair_strong, pair_flawed) if judge else None,
        "teacher_prefers_strong": sum(a > b for a, b in zip(pair_strong, pair_flawed)) / max(len(judge), 1),
        "teacher_strong_mean": statistics.fmean(pair_strong) if judge else None,
        "teacher_flawed_mean": statistics.fmean(pair_flawed) if judge else None,
        "teacher_detectable_pairs": sum(bool(j["teacher_detects"]) for j in judge),
        "teacher_detectable_prompts": len({j["prompt_id"] for j in judge if j["teacher_detects"]}),
        "slots_without_an_edit": sum(1 for tries in log.values() if tries and tries[-1] != "ok"),
        "edit_attempt_outcomes": outcomes,
    }


def build(
    backend: Backend,
    teacher: str,
    eval_prompts: list[str],
    out: Path,
    questions_per_topic: int = 45,
    held_out: int = 64,
    train_cap: int = 400,
    min_score: float = 8.0,
    seed: int = 42,
    embed=None,
    questions: dict | None = None,
    answer_tokens: int = 1200,
    flaws_per_prompt: int = 2,
    edit_attempts: int = 5,
) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    system = teacher_system_prompt(teacher)
    report: dict = {"teacher": teacher, "seed": seed}

    # 1. Questions, in batches of 15 per request so the lists stay well-formed, unless reused.
    if questions is not None:
        kept = [tuple(q) for q in questions["kept"]]
        dropped = questions["dropped"]
        report["questions"] = {"reused": True, "kept": len(kept)}
    else:
        chats, topics = [], []
        for topic, description in TOPICS.items():
            for _ in range(max(1, questions_per_topic // 15)):
                chats.append(Chat(system, [("user", question_request(description, 15))]))
                topics.append(topic)
        replies = backend.generate(chats, 2048, 0.9)
        candidates = [(t, q) for t, r in zip(topics, replies) for q in parse_question_list(r)]
        kept, dropped = dedupe(candidates, eval_prompts, embed=embed)
        report["questions"] = {
            "requested": questions_per_topic * len(TOPICS),
            "parsed": len(candidates),
            "kept": len(kept),
            "dropped_near_eval": sum(d["reason"] == "near an evaluation prompt" for d in dropped),
            "dropped_duplicate": sum(d["reason"] == "duplicate" for d in dropped),
        }
    (out / "questions.json").write_text(
        json.dumps({"kept": kept, "dropped": dropped}, indent=1, ensure_ascii=False), encoding="utf-8"
    )

    # 2. Split.
    train, held = split_held_out(kept, held_out, train_cap, seed)
    report["split"] = {"train": len(train), "held_out": len(held)}

    # 3. Answers, challenges, revisions, scores.
    student = lambda turns: Chat(STUDENT_SYSTEM, turns)  # noqa: E731
    answers = backend.generate([student([("user", p)]) for _, p in train], answer_tokens, 0.7)
    answers_cut = backend.last_truncated
    kind = challenge_types(seed)
    kinds = [kind() for _ in train]
    challenges = backend.generate(
        [Chat(system, [("user", challenge_request(p, a, k))]) for (_, p), a, k in zip(train, answers, kinds)],
        256,
        0.7,
    )
    challenges = [c.strip() for c in challenges]
    revisions = backend.generate(
        [
            student([("user", p), ("assistant", a), ("user", c)])
            for (_, p), a, c in zip(train, answers, challenges)
        ],
        answer_tokens,
        0.7,
    )
    revisions_cut = backend.last_truncated
    answer_scores = backend.generate(
        [Chat(system, [("user", scoring_request(p, a))]) for (_, p), a in zip(train, answers)], 1536, 0.3
    )
    revision_scores = backend.generate(
        [
            Chat(
                system,
                [("user", scoring_request(p, r, f"\n\nDialogue history:\nChallenge: {c}\nStudent: {r}\n\n"))],
            )
            for (_, p), r, c in zip(train, revisions, challenges)
        ],
        1536,
        0.3,
    )
    rows, parses, all_scores = [], [], []
    for (topic, prompt), a, k, c, r, sa, sr, cut_a, cut_r in zip(
        train,
        answers,
        kinds,
        challenges,
        revisions,
        answer_scores,
        revision_scores,
        answers_cut,
        revisions_cut,
    ):
        score_a, parse_a, _ = score_of(sa, teacher)
        score_r, parse_r, _ = score_of(sr, teacher)
        parses += [parse_a, parse_r]
        all_scores += [score_a, score_r]
        if score_a >= min_score and a.strip():
            rows.append(
                {
                    "topic": topic,
                    "kind": "answer",
                    "score": score_a,
                    "truncated": cut_a,
                    "messages": [
                        {"role": "system", "content": STUDENT_SYSTEM},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": a.strip()},
                    ],
                }
            )
        if score_a >= min_score and score_r >= min_score and r.strip() and c:
            rows.append(
                {
                    "topic": topic,
                    "kind": "revision",
                    "challenge_type": k,
                    "score": score_r,
                    "truncated": cut_a or cut_r,
                    "messages": [
                        {"role": "system", "content": STUDENT_SYSTEM},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": a.strip()},
                        {"role": "user", "content": c},
                        {"role": "assistant", "content": r.strip()},
                    ],
                }
            )
    write_jsonl(out / "train.jsonl", rows)
    report["train_examples"] = {
        "kept": len(rows),
        "answers": sum(r["kind"] == "answer" for r in rows),
        "revisions": sum(r["kind"] == "revision" for r in rows),
        "parse_json_rate": parses.count("json") / max(len(parses), 1),
        "score_mean": statistics.fmean(all_scores) if all_scores else None,
        "score_sd": statistics.pstdev(all_scores) if len(all_scores) > 1 else None,
        "kept_rate": len(rows) / max(2 * len(train), 1),
    }

    # 4. Judge set on the held-out prompts: one strong answer per prompt, errors planted in it.
    strong = backend.generate([student([("user", p)]) for _, p in held], answer_tokens, 0.7)
    strong = [x.strip() for x in strong]
    strong_cut = backend.last_truncated
    n = len(held)
    slots = [(i, kind) for i in range(n) if not strong_cut[i] for kind in FLAW_KINDS[:flaws_per_prompt]]
    flawed, log = plant_errors(backend, system, held, strong, slots, edit_attempts)
    made = [slot for slot in slots if slot in flawed]
    score_requests = [Chat(system, [("user", scoring_request(p, x))]) for (_, p), x in zip(held, strong)]
    scored = backend.generate(
        score_requests
        + [Chat(system, [("user", scoring_request(held[i][1], flawed[(i, k)]))]) for i, k in made],
        1536,
        0.3,
    )
    strong_scores = [score_of(x, teacher)[0] for x in scored[:n]]
    flawed_scores = [score_of(x, teacher)[0] for x in scored[n:]]
    judge = [
        {
            "pair_id": k,
            "prompt_id": i,
            "topic": held[i][0],
            "prompt": held[i][1],
            "flaw_kind": kind,
            "method": "sentence-edit",
            "attempts": len(log[(i, kind)]),
            "strong": strong[i],
            "flawed": flawed[(i, kind)],
            "strong_truncated": False,
            "flawed_truncated": False,
            "teacher_strong": strong_scores[i],
            "teacher_flawed": fs,
            "teacher_detects": strong_scores[i] > fs,
        }
        for k, ((i, kind), fs) in enumerate(zip(made, flawed_scores))
    ]
    (out / "held_out.json").write_text(
        json.dumps([{"topic": t, "prompt": p} for t, p in held], indent=1, ensure_ascii=False),
        encoding="utf-8",
    )
    (out / "judge_set.json").write_text(json.dumps(judge, indent=1, ensure_ascii=False), encoding="utf-8")
    report["judge_set"] = judge_report(judge, log, n) | {"strong_truncated": sum(strong_cut)}

    # 5. Noise floor: the strong answers scored again.
    again = backend.generate(score_requests[:n], 1536, 0.3)
    again_scores = [score_of(s, teacher)[0] for s in again]
    diffs = [abs(a - b) for a, b in zip(strong_scores, again_scores)]
    (out / "noise.json").write_text(
        json.dumps({"first": strong_scores, "second": again_scores}, indent=1), encoding="utf-8"
    )
    report["noise_floor"] = {
        "mean_abs_diff": statistics.fmean(diffs) if diffs else None,
        "max_abs_diff": max(diffs) if diffs else None,
        "identical_rate": sum(d == 0 for d in diffs) / max(len(diffs), 1),
    }
    (out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


def judge_only(
    backend: Backend,
    teacher: str,
    source: Path,
    out: Path,
    flaws_per_prompt: int = 2,
    edit_attempts: int = 5,
) -> dict:
    """Rebuild only the judge set of an earlier run: keep its strong answers and passing pairs,
    plant errors where a (prompt, flaw kind) pair is missing, and copy everything else."""
    out.mkdir(parents=True, exist_ok=True)
    system = teacher_system_prompt(teacher)
    held = [
        (h["topic"], h["prompt"]) for h in json.loads((source / "held_out.json").read_text(encoding="utf-8"))
    ]
    old = json.loads((source / "judge_set.json").read_text(encoding="utf-8"))
    index = {p: i for i, (_, p) in enumerate(held)}
    strong: dict[int, str] = {}
    strong_score: dict[int, float] = {}
    strong_cut: dict[int, bool] = {}
    kept: dict[tuple[int, str], dict] = {}
    for pair in old:
        i = index[pair["prompt"]]
        strong[i], strong_score[i] = pair["strong"], pair["teacher_strong"]
        strong_cut[i] = bool(pair.get("strong_truncated", False))
        problems, _ = pair_problems(pair)
        if not problems and pair.get("flaw_kind") in FLAW_KINDS[:flaws_per_prompt]:
            kept[(i, pair["flaw_kind"])] = pair | {"method": pair.get("method", "rewrite")}
    missing_strong = [i for i in range(len(held)) if i not in strong]
    if missing_strong:
        raise SystemExit(f"{source}/judge_set.json has no strong answer for prompts {missing_strong}")
    texts = [strong[i] for i in range(len(held))]
    slots = [
        (i, kind)
        for i in range(len(held))
        if not strong_cut[i]
        for kind in FLAW_KINDS[:flaws_per_prompt]
        if (i, kind) not in kept
    ]
    flawed, log = plant_errors(backend, system, held, texts, slots, edit_attempts)
    made = [slot for slot in slots if slot in flawed]
    scored = backend.generate(
        [Chat(system, [("user", scoring_request(held[i][1], flawed[(i, k)]))]) for i, k in made], 1536, 0.3
    )
    new = {
        (i, kind): {
            "prompt_id": i,
            "topic": held[i][0],
            "prompt": held[i][1],
            "flaw_kind": kind,
            "method": "sentence-edit",
            "attempts": len(log[(i, kind)]),
            "strong": strong[i],
            "flawed": flawed[(i, kind)],
            "strong_truncated": False,
            "flawed_truncated": False,
            "teacher_strong": strong_score[i],
            "teacher_flawed": score_of(reply, teacher)[0],
        }
        for (i, kind), reply in zip(made, scored)
    }
    order = {kind: k for k, kind in enumerate(FLAW_KINDS)}
    pairs = sorted({**kept, **new}.items(), key=lambda item: (item[0][0], order[item[0][1]]))
    judge = [
        pair
        | {"pair_id": k, "prompt_id": i, "teacher_detects": pair["teacher_strong"] > pair["teacher_flawed"]}
        for k, ((i, _), pair) in enumerate(pairs)
    ]
    (out / "judge_set.json").write_text(json.dumps(judge, indent=1, ensure_ascii=False), encoding="utf-8")
    for name in ("train.jsonl", "held_out.json", "noise.json", "questions.json"):
        if (source / name).exists() and source.resolve() != out.resolve():
            shutil.copyfile(source / name, out / name)
    report = json.loads((source / "report.json").read_text(encoding="utf-8"))
    report["judge_set"] = judge_report(judge, log, len(held)) | {
        "kept_from_source": len(kept),
        "planted": len(new),
        "source": str(source),
    }
    (out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


def main() -> None:  # pragma: no cover - runs on the pod
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--teacher", default="Qwen/Qwen2.5-32B-Instruct")
    parser.add_argument("--eval-prompts", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--questions-per-topic", type=int, default=45)
    parser.add_argument("--held-out", type=int, default=64)
    parser.add_argument("--train-cap", type=int, default=400)
    parser.add_argument("--min-score", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--questions", type=Path, help="reuse this questions.json (same seed, same split)")
    parser.add_argument("--answer-tokens", type=int, default=1200)
    parser.add_argument("--flaws-per-prompt", type=int, default=2, choices=[1, 2])
    parser.add_argument("--edit-attempts", type=int, default=5)
    parser.add_argument("--judge-only", action="store_true", help="rebuild only the judge set of --from")
    parser.add_argument("--from", dest="source", type=Path, help="the earlier run's directory (--judge-only)")
    args = parser.parse_args()

    from lib import Embedder, VllmBackend

    if args.judge_only:
        if not args.source:
            parser.error("--judge-only needs --from")
        report = judge_only(
            VllmBackend(args.teacher, seed=args.seed),
            args.teacher,
            args.source,
            args.out,
            args.flaws_per_prompt,
            args.edit_attempts,
        )
        print(json.dumps(report["judge_set"], indent=1))
        print("DATASET-OK")
        return
    if not args.eval_prompts:
        parser.error("--eval-prompts is required to build a dataset")

    questions = json.loads(args.questions.read_text(encoding="utf-8")) if args.questions else None
    embed = None if questions else Embedder()  # on the CPU: the GPU is the teacher's
    backend = VllmBackend(args.teacher, seed=args.seed)
    eval_prompts = json.loads(args.eval_prompts.read_text(encoding="utf-8"))
    report = build(
        backend,
        args.teacher,
        eval_prompts,
        args.out,
        args.questions_per_topic,
        args.held_out,
        args.train_cap,
        args.min_score,
        args.seed,
        embed,
        questions,
        args.answer_tokens,
        args.flaws_per_prompt,
        args.edit_attempts,
    )
    print(json.dumps(report, indent=1))
    print("DATASET-OK")


if __name__ == "__main__":
    main()
