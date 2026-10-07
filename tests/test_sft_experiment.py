"""Tests for the fine-tune-then-ASPIRE experiment scripts (examples/sft-experiment)."""

import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

EXPERIMENT = Path(__file__).resolve().parent.parent / "examples" / "sft-experiment"
sys.path.insert(0, str(EXPERIMENT))

import build_dataset  # noqa: E402
import eval_heldout  # noqa: E402
import judge_eval  # noqa: E402
import lib  # noqa: E402
import probe_models  # noqa: E402
import sft  # noqa: E402

EVAL_PROMPTS = [
    "Why does ice float on water?",
    "Explain the Monty Hall problem and why the answer surprises people.",
    "What is a race condition? Give a short example.",
]


class TestSimilarity:
    def test_shared_phrasing_is_not_similarity(self):
        a = "In applied ethics, how would you explain volcanoes to a skeptic?"
        b = "In applied ethics, how would you explain bridges to a skeptic?"
        assert lib.similarity(a, b) < 0.7

    def test_same_text_is_one_and_unrelated_is_low(self):
        assert lib.similarity("Why does ice float?", "Why does ice float?") == pytest.approx(1.0)
        assert lib.similarity("Why does ice float on water?", "What causes inflation in an economy?") < 0.3

    def test_paraphrase_of_an_eval_prompt_is_dropped_and_duplicates_too(self):
        candidates = [
            ("physics", "Why does ice float on top of water?"),  # near an eval prompt
            ("physics", "How do rainbows form after a storm?"),
            ("physics", "How do rainbows form after a storm??"),  # duplicate
            ("ethics", "When is it acceptable to tell a white lie?"),
        ]
        kept, dropped = lib.dedupe(candidates, EVAL_PROMPTS)
        assert [p for _, p in kept] == [
            "How do rainbows form after a storm?",
            "When is it acceptable to tell a white lie?",
        ]
        assert sorted(d["reason"] for d in dropped) == ["duplicate", "near an evaluation prompt"]

    def test_embedding_check_catches_paraphrases_that_share_no_words(self):
        prompts = ["How does TLS protect passwords sent to a website?", "Why do volcanoes erupt?"]
        vectors = {
            prompts[0]: [1.0, 0.0],
            prompts[1]: [0.0, 1.0],
            "How does HTTPS keep a password safe?": [0.99, 0.141],
        }

        def embed(texts):
            return np.array([vectors[t] for t in texts])

        kept, dropped = lib.dedupe(
            [("cs", p) for p in prompts], ["How does HTTPS keep a password safe?"], embed=embed
        )
        assert [p for _, p in kept] == ["Why do volcanoes erupt?"]
        assert dropped[0]["reason"] == "near an evaluation prompt" and dropped[0]["cosine"] > 0.85


class TestSplit:
    def test_held_out_is_stratified_disjoint_and_capped(self):
        pairs = [(t, f"{t} question {i}") for t in ("a", "b", "c", "d") for i in range(20)]
        train, held = lib.split_held_out(pairs, held_out=8, train_cap=50, seed=1)
        assert len(held) == 8 and len(train) == 50
        assert {t for t, _ in held} == {"a", "b", "c", "d"}
        assert not {p for _, p in held} & {p for _, p in train}
        assert lib.split_held_out(pairs, 8, 50, seed=1) == (train, held), "deterministic"


class TestQuestionParsing:
    def test_lists_objects_and_garbage(self):
        assert lib.parse_question_list('Sure: ["Why is the sky blue at noon?", "x"]') == [
            "Why is the sky blue at noon?"
        ]
        assert lib.parse_question_list('{"questions": ["How does a heat pump move heat?"]}') == [
            "How does a heat pump move heat?"
        ]
        assert lib.parse_question_list("no list here") == []


def fake_teacher(chat: lib.Chat) -> str:
    """Plays the teacher and the student for build(): every request type gets a canned answer."""
    user = chat.turns[-1][1]
    if user.startswith("Write 15 distinct questions"):
        topic = re.search(r"about (.*?):", user).group(1)
        subjects = [
            "volcanoes",
            "bridges",
            "compilers",
            "glaciers",
            "auctions",
            "vaccines",
            "satellites",
            "orchestras",
            "tariffs",
            "lighthouses",
            "spreadsheets",
            "beehives",
            "elections",
            "telescopes",
            "bakeries",
        ]
        questions = [f"In {topic}, how would you explain {s} to a skeptic?" for s in subjects]
        if "everyday physics" in topic:
            questions[0] = "Why does ice float on water?"  # an eval prompt: must be dropped
        return json.dumps(questions)
    if user.startswith("Generate a"):
        return "  But what about the edge case?  "
    if "Rewrite the answer" in user:
        return "A confident but wrong answer."
    if user.startswith("Evaluate this student response"):
        score = 3 if "wrong answer" in user else 9
        return json.dumps(
            {"overall_score": score, "dimension_scores": [{"dimension": "clarity", "score": score}]}
        )
    if chat.system == lib.STUDENT_SYSTEM:
        return "A strong revision." if len(chat.turns) > 1 else "A strong answer."
    raise AssertionError(f"unexpected request: {user[:60]}")


class TestBuildDataset:
    @pytest.fixture
    def built(self, tmp_path):
        backend = lib.FakeBackend(fake_teacher)
        report = build_dataset.build(
            backend,
            "Qwen/Qwen2.5-32B-Instruct",
            EVAL_PROMPTS,
            tmp_path,
            questions_per_topic=15,
            held_out=12,
            train_cap=60,
        )
        return report, tmp_path, backend

    def test_no_evaluation_prompt_reaches_training_or_held_out(self, built):
        report, out, _ = built
        train = lib.read_jsonl(out / "train.jsonl")
        held = json.loads((out / "held_out.json").read_text(encoding="utf-8"))
        train_prompts = {r["messages"][1]["content"] for r in train}
        held_prompts = {h["prompt"] for h in held}
        assert not (train_prompts | held_prompts) & set(EVAL_PROMPTS)
        assert not train_prompts & held_prompts
        assert report["questions"]["dropped_near_eval"] >= 1

    def test_examples_are_chats_in_the_students_format_and_scored(self, built):
        report, out, _ = built
        rows = lib.read_jsonl(out / "train.jsonl")
        assert report["split"] == {"train": 60, "held_out": 12}
        assert report["train_examples"]["answers"] == 60 and report["train_examples"]["revisions"] == 60
        answer = next(r for r in rows if r["kind"] == "answer")
        assert [m["role"] for m in answer["messages"]] == ["system", "user", "assistant"]
        assert answer["messages"][0]["content"] == lib.STUDENT_SYSTEM
        revision = next(r for r in rows if r["kind"] == "revision")
        assert [m["role"] for m in revision["messages"]] == [
            "system",
            "user",
            "assistant",
            "user",
            "assistant",
        ]
        assert revision["messages"][3]["content"] == "But what about the edge case?"
        assert report["train_examples"]["parse_json_rate"] == 1.0

    def test_judge_set_and_noise_floor(self, built):
        report, out, backend = built
        judge = json.loads((out / "judge_set.json").read_text(encoding="utf-8"))
        assert len(judge) == 12 and judge[0]["teacher_strong"] == 9 and judge[0]["teacher_flawed"] == 3
        assert report["judge_set"]["teacher_prefers_strong"] == 1.0
        assert report["noise_floor"]["mean_abs_diff"] == 0.0
        # scoring uses aspire-si's own request and the control teacher's system prompt
        scoring = [c for c in backend.calls if c.turns[-1][1].startswith("Evaluate this student response")]
        assert scoring and all(
            c.system == lib.teacher_system_prompt("Qwen/Qwen2.5-32B-Instruct") for c in scoring
        )
        assert "local:Qwen2.5-32B-Instruct" in scoring[0].system

    def test_low_scores_are_filtered(self, tmp_path):
        def harsh(chat):
            reply = fake_teacher(chat)
            return (
                reply.replace('"overall_score": 9', '"overall_score": 6')
                if "overall_score" in reply
                else reply
            )

        report = build_dataset.build(
            lib.FakeBackend(harsh),
            "t/m",
            EVAL_PROMPTS,
            tmp_path,
            questions_per_topic=15,
            held_out=12,
            train_cap=20,
        )
        assert report["train_examples"]["kept"] == 0


class FakeTokenizer:
    """A chat template with one token per character: easy to check spans against."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)
        if add_generation_prompt:
            text += "<assistant>"
        return {"input_ids": [ord(c) for c in text]} if tokenize else text


class TestSftTokenizing:
    def test_labels_cover_only_assistant_turns(self):
        messages = [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "C"},
            {"role": "assistant", "content": "A2"},
        ]
        example = sft.tokenize_example(FakeTokenizer(), messages, max_length=10_000)
        labelled = "".join(
            chr(i) for i, label in zip(example["input_ids"], example["labels"]) if label != sft.IGNORE
        )
        assert labelled == "A1</assistant>A2</assistant>"

    def test_truncation_keeps_the_end(self):
        messages = [{"role": "user", "content": "Q" * 50}, {"role": "assistant", "content": "answer"}]
        example = sft.tokenize_example(FakeTokenizer(), messages, max_length=20)
        assert len(example["input_ids"]) == 20
        assert "".join(chr(i) for i in example["input_ids"]).endswith("answer</assistant>")

    def test_collate_pads_and_masks(self):
        batch = sft.collate(
            [{"input_ids": [1, 2, 3], "labels": [-100, 2, 3]}, {"input_ids": [4], "labels": [4]}], pad_id=0
        )
        assert batch["input_ids"].tolist() == [[1, 2, 3], [4, 0, 0]]
        assert batch["labels"].tolist() == [[-100, 2, 3], [4, -100, -100]]
        assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]

    def test_defaults_match_the_control_lora(self):
        args = sft.parse_args(["--data", "d.jsonl", "--out", "o"])
        assert (args.lora_r, args.lora_alpha, args.lora_dropout, args.seed) == (16, 32, 0.05, 42)
        assert args.student == "Qwen/Qwen2.5-1.5B-Instruct"


class TestStatistics:
    def test_pairwise_accuracy_and_interval(self):
        assert lib.pairwise_accuracy([5, 5, 5, 5], [1, 1, 5, 9]) == pytest.approx(0.625)
        lo, hi = lib.bootstrap_ci([5] * 40, [1] * 40)
        assert (lo, hi) == (1.0, 1.0)
        lo, hi = lib.bootstrap_ci([5, 1] * 20, [1, 5] * 20)
        assert lo < 0.5 < hi

    def test_mean_interval(self):
        mean, lo, hi = lib.mean_ci([7.0] * 10)
        assert (mean, lo, hi) == (7.0, 7.0, 7.0)


class TestEvalAndJudge:
    def test_scores_per_entry_with_ascii_parse(self, tmp_path):
        (tmp_path / "answers-base.json").write_text(
            json.dumps({"name": "base", "answers": [{"prompt": "p", "answer": "a"}] * 4}), encoding="utf-8"
        )
        backend = lib.FakeBackend(lambda chat: '{"overall_score": 6}')
        result = eval_heldout.score_answers(
            backend, "Qwen/Qwen2.5-32B-Instruct", [tmp_path / "answers-base.json"]
        )
        assert result["base"]["mean"] == 6.0 and result["base"]["json_rate"] == 1.0

    def test_judge_evaluate_counts_strong_wins(self):
        scorer = SimpleNamespace(
            score_many=lambda prompts, responses: [9.0 if "good" in r else 2.0 for r in responses]
        )
        pairs = [{"prompt": "p", "strong": "good", "flawed": "bad"}] * 10
        result = judge_eval.evaluate(scorer, pairs)
        assert result["accuracy"] == 1.0 and result["mean_gap"] == 7.0


class TestProbe:
    def test_entries_parse(self):
        assert probe_models.parse_entry("sft=sft/merged") == ("sft", "sft/merged", None)
        assert probe_models.parse_entry("a1=sft/merged+out/checkpoint-1/student") == (
            "a1",
            "sft/merged",
            "out/checkpoint-1/student",
        )
        with pytest.raises(SystemExit):
            probe_models.parse_entry("nomodel")

    def test_pairs_come_sorted_with_scores(self, tmp_path):
        for i, prompt in enumerate(["b prompt", "a prompt"]):
            (tmp_path / f"{i}.json").write_text(
                json.dumps(
                    {
                        "prompt": prompt,
                        "final_response": f"r{i}",
                        "final_evaluation": {
                            "overall_score": 7,
                            "dimension_scores": [{"dimension": "clarity", "score": 7}],
                            "metadata": {"teacher_scores": {"q": 7, "g": 3}},
                        },
                    }
                ),
                encoding="utf-8",
            )
        pairs, dims, teachers = probe_models.load_pairs(tmp_path)
        assert pairs == [("a prompt", "r1"), ("b prompt", "r0")]
        assert dims[0] == {"clarity": 7.0} and teachers[0] == {"q": 7.0, "g": 3.0}

    def test_drift_numbers(self):
        rng = np.random.default_rng(0)
        base = rng.standard_normal((6, 8))
        shift = np.ones(8)
        states = np.stack([base, base + 0.1 * shift, base + 0.2 * shift])
        summary = probe_models.drift_summary(states, ["base", "c1", "c2"], 0)
        assert summary["per_entry"][0]["mean_drift_norm"] == 0.0
        assert summary["per_entry"][2]["mean_drift_norm"] > summary["per_entry"][1]["mean_drift_norm"]
        assert summary["per_entry"][2]["shared_direction"] == pytest.approx(1.0)
        scores = base @ shift  # position along the drift axis predicts the score exactly
        result = probe_models.alignment(states, 0, 2, scores, nulls=50)
        assert result["r"] == pytest.approx(1.0)
