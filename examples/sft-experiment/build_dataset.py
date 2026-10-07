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
  4. Judge set: per held-out prompt, a strong answer and a flawed rewrite with one substantive
     error, both scored.
  5. Noise floor: the held-out strong answers are scored a second time.

Writes to --out: questions.json, train.jsonl, held_out.json, judge_set.json, noise.json and
report.json (counts, drop reasons, parse rates, score spreads).

Usage on the pod:  python build_dataset.py --teacher Qwen/Qwen2.5-32B-Instruct
                     --eval-prompts ../local-run/prompts.json --out data
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from lib import (  # noqa: E402
    FLAWED_REQUEST,
    STUDENT_SYSTEM,
    TOPICS,
    Backend,
    Chat,
    challenge_request,
    challenge_types,
    dedupe,
    parse_question_list,
    question_request,
    score_of,
    scoring_request,
    split_held_out,
    teacher_system_prompt,
    write_jsonl,
)


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
) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    system = teacher_system_prompt(teacher)
    report: dict = {"teacher": teacher, "seed": seed}

    # 1. Questions, in batches of 15 per request so the lists stay well-formed.
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
    answers = backend.generate([student([("user", p)]) for _, p in train], 700, 0.7)
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
        700,
        0.7,
    )
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
    for (topic, prompt), a, k, c, r, sa, sr in zip(
        train, answers, kinds, challenges, revisions, answer_scores, revision_scores
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

    # 4. Judge set on the held-out prompts.
    strong = backend.generate([student([("user", p)]) for _, p in held], 700, 0.7)
    flawed = backend.generate(
        [
            Chat(system, [("user", FLAWED_REQUEST.format(prompt=p, answer=s))])
            for (_, p), s in zip(held, strong)
        ],
        900,
        0.7,
    )
    score_requests = [Chat(system, [("user", scoring_request(p, x))]) for (_, p), x in zip(held, strong)]
    score_requests += [Chat(system, [("user", scoring_request(p, x))]) for (_, p), x in zip(held, flawed)]
    scored = backend.generate(score_requests, 1536, 0.3)
    n = len(held)
    strong_scores = [score_of(s, teacher)[0] for s in scored[:n]]
    flawed_scores = [score_of(s, teacher)[0] for s in scored[n:]]
    judge = [
        {
            "topic": t,
            "prompt": p,
            "strong": s.strip(),
            "flawed": f.strip(),
            "teacher_strong": ss,
            "teacher_flawed": fs,
        }
        for (t, p), s, f, ss, fs in zip(held, strong, flawed, strong_scores, flawed_scores)
    ]
    (out / "held_out.json").write_text(
        json.dumps([{"topic": t, "prompt": p} for t, p in held], indent=1, ensure_ascii=False),
        encoding="utf-8",
    )
    (out / "judge_set.json").write_text(json.dumps(judge, indent=1, ensure_ascii=False), encoding="utf-8")
    report["judge_set"] = {
        "pairs": n,
        "teacher_prefers_strong": sum(s > f for s, f in zip(strong_scores, flawed_scores)) / max(n, 1),
        "teacher_strong_mean": statistics.fmean(strong_scores) if n else None,
        "teacher_flawed_mean": statistics.fmean(flawed_scores) if n else None,
    }

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


def main() -> None:  # pragma: no cover - runs on the pod
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--teacher", default="Qwen/Qwen2.5-32B-Instruct")
    parser.add_argument("--eval-prompts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--questions-per-topic", type=int, default=45)
    parser.add_argument("--held-out", type=int, default=64)
    parser.add_argument("--train-cap", type=int, default=400)
    parser.add_argument("--min-score", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from lib import Embedder, VllmBackend

    embed = Embedder()  # on the CPU: the GPU is the teacher's
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
    )
    print(json.dumps(report, indent=1))
    print("DATASET-OK")


if __name__ == "__main__":
    main()
