"""Tests for the fine-tune-then-ASPIRE experiment scripts (examples/sft-experiment)."""

import json
import math
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

EXPERIMENT = Path(__file__).resolve().parent.parent / "examples" / "sft-experiment"
sys.path.insert(0, str(EXPERIMENT))

import build_dataset  # noqa: E402
import clean_dataset  # noqa: E402
import eval_heldout  # noqa: E402
import fresh_pairs  # noqa: E402
import host_check  # noqa: E402
import judge_eval  # noqa: E402
import judge_kev  # noqa: E402
import judge_logprob  # noqa: E402
import lib  # noqa: E402
import make_prompts  # noqa: E402
import pairwise_teacher  # noqa: E402
import probe_models  # noqa: E402
import seed_configs  # noqa: E402
import sft  # noqa: E402
import yaml  # noqa: E402

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


STRONG_ANSWER = (
    "A strong answer: at sea level water boils at 100 degrees Celsius, so the kettle switches off."
)
EDIT_HEAD = "Here is a question and a strong answer"


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
    if user.startswith("Here is a question and a strong answer"):
        return json.dumps({"original": "boils at 100 degrees", "edited": "boils at 90 degrees"})
    if user.startswith("Evaluate this student response"):
        score = 3 if "90 degrees" in user else 9
        return json.dumps(
            {"overall_score": score, "dimension_scores": [{"dimension": "clarity", "score": score}]}
        )
    if chat.system == lib.STUDENT_SYSTEM:
        return "A strong revision." if len(chat.turns) > 1 else STRONG_ANSWER
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
        # two minimal edits per held-out prompt, one per flaw kind, labelled by prompt
        assert len(judge) == 24 and judge[0]["teacher_strong"] == 9 and judge[0]["teacher_flawed"] == 3
        assert [j["prompt_id"] for j in judge[:4]] == [0, 0, 1, 1]
        assert {j["flaw_kind"] for j in judge} == set(lib.FLAW_KINDS)
        assert judge[0]["strong_truncated"] is False and judge[0]["flawed_truncated"] is False
        assert report["judge_set"]["teacher_prefers_strong"] == 1.0
        assert all(j["teacher_detects"] for j in judge)
        assert report["judge_set"]["teacher_detectable_pairs"] == 24
        assert report["judge_set"]["prompts"] == 12 and report["judge_set"]["pairs"] == 24
        assert judge[0]["flawed"] == STRONG_ANSWER.replace("100", "90")
        assert judge[0]["method"] == "sentence-edit" and judge[0]["attempts"] == 1
        edits = [c for c in backend.calls if c.turns[-1][1].startswith(EDIT_HEAD)]
        assert len(edits) == 24 and f"Strong answer: {STRONG_ANSWER}" in edits[0].turns[-1][1]
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

    def test_reused_questions_give_the_same_split_without_asking_again(self, built, tmp_path):
        _, first, _ = built
        backend = lib.FakeBackend(fake_teacher)
        questions = json.loads((first / "questions.json").read_text(encoding="utf-8"))
        again = tmp_path / "again"
        report = build_dataset.build(
            backend,
            "Qwen/Qwen2.5-32B-Instruct",
            EVAL_PROMPTS,
            again,
            held_out=12,
            train_cap=60,
            questions=questions,
        )
        assert report["questions"] == {"reused": True, "kept": len(questions["kept"])}
        assert not [c for c in backend.calls if c.turns[-1][1].startswith("Write 15 distinct questions")]
        held = lambda d: (d / "held_out.json").read_text(encoding="utf-8")  # noqa: E731
        assert held(again) == held(first)
        prompts = lambda d: [r["messages"][1]["content"] for r in lib.read_jsonl(d / "train.jsonl")]  # noqa: E731
        assert prompts(again) == prompts(first)

    def test_answers_at_the_cap_are_flagged_truncated(self, tmp_path):
        def verbose(chat):
            if chat.system == lib.STUDENT_SYSTEM and len(chat.turns) == 1:
                return "x" * 50
            return fake_teacher(chat)

        build_dataset.build(
            lib.FakeBackend(verbose),
            "Qwen/Qwen2.5-32B-Instruct",
            EVAL_PROMPTS,
            tmp_path,
            questions_per_topic=15,
            held_out=12,
            train_cap=20,
            answer_tokens=40,
        )
        rows = lib.read_jsonl(tmp_path / "train.jsonl")
        assert rows and all(r["truncated"] for r in rows)  # a revision whose first answer was cut, too
        # no errors are planted in a truncated strong answer: the pair could never be used
        assert json.loads((tmp_path / "judge_set.json").read_text(encoding="utf-8")) == []
        report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
        assert report["judge_set"]["strong_truncated"] == 12

    def test_an_unusable_edit_is_asked_for_again(self, tmp_path):
        seen = set()

        def stubborn(chat):
            user = chat.turns[-1][1]
            if user.startswith(EDIT_HEAD) and user not in seen:
                seen.add(user)  # the first attempt per slot copies the sentence unchanged
                return json.dumps({"original": "boils at 100 degrees", "edited": "boils at 100 degrees"})
            return fake_teacher(chat)

        report = build_dataset.build(
            lib.FakeBackend(stubborn),
            "Qwen/Qwen2.5-32B-Instruct",
            EVAL_PROMPTS,
            tmp_path,
            questions_per_topic=15,
            held_out=4,
            train_cap=8,
        )
        judge = json.loads((tmp_path / "judge_set.json").read_text(encoding="utf-8"))
        assert len(judge) == 8 and all(j["attempts"] == 2 for j in judge)
        assert report["judge_set"]["edit_attempt_outcomes"] == {"edit unchanged": 8, "ok": 8}
        assert report["judge_set"]["slots_without_an_edit"] == 0

    def test_judge_only_keeps_passing_pairs_and_fills_the_rest(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        held = [{"topic": "t", "prompt": "q0"}, {"topic": "t", "prompt": "q1"}]
        fact, step = lib.FLAW_KINDS
        good = STRONG_ANSWER.replace("Celsius", "Kelvin")
        old = [
            {
                "prompt": "q0",
                "flaw_kind": fact,
                "strong": STRONG_ANSWER,
                "flawed": good,
                "teacher_strong": 9.0,
                "teacher_flawed": 9.0,
                "strong_truncated": False,
                "flawed_truncated": False,
            },
            {
                "prompt": "q0",
                "flaw_kind": step,
                "strong": STRONG_ANSWER,
                "flawed": STRONG_ANSWER,
                "teacher_strong": 9.0,
                "teacher_flawed": 9.0,
                "strong_truncated": False,
                "flawed_truncated": False,
            },
            {
                "prompt": "q1",
                "flaw_kind": fact,
                "strong": STRONG_ANSWER,
                "flawed": STRONG_ANSWER,
                "teacher_strong": 8.0,
                "teacher_flawed": 8.0,
                "strong_truncated": False,
                "flawed_truncated": False,
            },
        ]  # q1's second kind is missing entirely
        (src / "judge_set.json").write_text(json.dumps(old), encoding="utf-8")
        (src / "held_out.json").write_text(json.dumps(held), encoding="utf-8")
        (src / "report.json").write_text(json.dumps({"train_examples": {"kept": 1}}), encoding="utf-8")
        lib.write_jsonl(src / "train.jsonl", [{"kind": "answer"}])
        backend = lib.FakeBackend(fake_teacher)
        report = build_dataset.judge_only(backend, "Qwen/Qwen2.5-32B-Instruct", src, tmp_path / "out")
        judge = json.loads((tmp_path / "out" / "judge_set.json").read_text(encoding="utf-8"))
        assert [(j["pair_id"], j["prompt_id"], j["flaw_kind"], j["method"]) for j in judge] == [
            (0, 0, fact, "rewrite"),
            (1, 0, step, "sentence-edit"),
            (2, 1, fact, "sentence-edit"),
            (3, 1, step, "sentence-edit"),
        ]
        assert (
            judge[0]["flawed"] == good
            and judge[3]["teacher_strong"] == 8.0
            and judge[3]["teacher_flawed"] == 3
        )
        assert [j["teacher_detects"] for j in judge] == [False, True, True, True]
        assert report["judge_set"]["kept_from_source"] == 1 and report["judge_set"]["planted"] == 3
        assert report["train_examples"] == {"kept": 1}
        assert lib.read_jsonl(tmp_path / "out" / "train.jsonl") == [{"kind": "answer"}]
        # only the missing slots were asked for, and only the new pairs were scored
        assert sum(c.turns[-1][1].startswith(EDIT_HEAD) for c in backend.calls) == 3
        assert sum(c.turns[-1][1].startswith("Evaluate") for c in backend.calls) == 3


class TestApplyEdit:
    ANSWER = "First, $\\frac{1}{2}$ of the doors. Then the host opens one. So switching wins 2/3."

    def edit(self, original, edited):
        return lib.apply_edit(json.dumps({"original": original, "edited": edited}), self.ANSWER)

    def test_applies_one_sentence(self):
        text, why = self.edit("So switching wins 2/3.", "So switching wins 1/2.")
        assert why == "ok" and text == self.ANSWER.replace("2/3.", "1/2.")

    def test_rejections(self):
        assert self.edit("A sentence that is not there.", "x")[1] == "original not in the answer"
        assert self.edit("Then the host opens one.", "Then the host opens one.")[1] == "edit unchanged"
        assert self.edit("Then the host opens one.", "Then the host opens one (this is the error).")[1] == (
            "self-flagging: error, (this is"
        )
        assert lib.apply_edit("I changed nothing.", self.ANSWER) == (None, "no JSON edit")

    def test_latex_backslashes_survive_json(self):
        # the model writes \frac unescaped; JSON reads \f as a form feed
        reply = '{"original": "First, $\\frac{1}{2}$ of the doors.", "edited": "First, $\\frac{1}{3}$ of the doors."}'
        text, why = lib.apply_edit(reply, self.ANSWER)
        assert why == "ok" and "{1}{3}" in text


STRONG = (
    "The probability is computed step by step. First, the chance of blue is 15/25 = 3/5. "
    "Then, with one blue removed, the chance of red is 10/24 = 5/12. Multiplying gives 1/4. "
    "So the answer is 1/4, and the order of the draws matters for the intermediate steps."
)


def judge_pair(pair_id, flawed, strong=STRONG, **flags):
    return {
        "pair_id": pair_id,
        "prompt_id": pair_id // 2,
        "prompt": "p",
        "strong": strong,
        "flawed": flawed,
    } | flags


class TestCleanDataset:
    @pytest.fixture
    def data(self, tmp_path):
        d = tmp_path / "raw"
        d.mkdir()

        def msg(text):
            return [
                {"role": "system", "content": "s"},
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": text},
            ]

        lib.write_jsonl(
            d / "train.jsonl",
            [
                {"kind": "answer", "truncated": False, "messages": msg("Done.")},
                {"kind": "answer", "truncated": True, "messages": msg("Cut off in the")},
                {"kind": "revision", "messages": msg("No flag, ends cleanly.")},
                {"kind": "revision", "messages": msg("No flag, cut off in the")},
            ],
        )
        pairs = [
            judge_pair(0, STRONG.replace("1/4", "1/5"), strong_truncated=False, flawed_truncated=False),
            judge_pair(1, STRONG.replace("5/12", "7/12")),
            judge_pair(2, "A paraphrase that says something else entirely and is much shorter."),
            judge_pair(3, STRONG.replace("1/4.", "1/5 (this is the error).")),
            judge_pair(4, STRONG, strong_truncated=False, flawed_truncated=False),
            judge_pair(5, STRONG.replace("1/4", "1/5"), strong_truncated=True, flawed_truncated=False),
            judge_pair(6, STRONG + " Also, the draws should be independent in this case, so ignore it."),
        ]
        (d / "judge_set.json").write_text(json.dumps(pairs), encoding="utf-8")
        (d / "held_out.json").write_text("[]", encoding="utf-8")
        (d / "report.json").write_text(json.dumps({"teacher": "t"}), encoding="utf-8")
        return d

    def test_drops_truncated_examples_by_flag_or_ending(self, data, tmp_path):
        section = clean_dataset.clean(data, tmp_path / "out", min_pairs=2)
        kept = lib.read_jsonl(tmp_path / "out" / "train.jsonl")
        assert [r["messages"][-1]["content"] for r in kept] == ["Done.", "No flag, ends cleanly."]
        assert section["train"]["dropped_truncated"] == 2
        assert [d["index"] for d in section["dropped_examples"]] == [1, 3]

    def test_keeps_only_minimal_untruncated_unflagged_edits(self, data, tmp_path):
        section = clean_dataset.clean(data, tmp_path / "out", min_pairs=2)
        kept = json.loads((tmp_path / "out" / "judge_set.json").read_text(encoding="utf-8"))
        assert [p["pair_id"] for p in kept] == [0, 1]
        reasons = {d["pair_id"]: " | ".join(d["reasons"]) for d in section["dropped_pairs"]}
        assert "similarity" in reasons[2] and "length" in reasons[2]
        assert "self-flagging: error, (this is" in reasons[3]
        assert reasons[4] == "no edit"
        assert reasons[5] == "strong truncated"
        assert "self-flagging: should be" in reasons[6] and "length" in reasons[6]
        assert section["judge"]["enough_pairs"] and section["judge"]["prompts_kept"] == 1
        report = json.loads((tmp_path / "out" / "report.json").read_text(encoding="utf-8"))
        assert report["teacher"] == "t" and report["clean"]["judge"]["kept"] == 2
        assert (tmp_path / "out" / "held_out.json").exists()

    def test_too_few_pairs_is_reported(self, data, tmp_path):
        section = clean_dataset.clean(data, tmp_path / "out")
        assert section["judge"]["min_pairs"] == 96 and not section["judge"]["enough_pairs"]

    def test_a_marker_already_in_the_strong_answer_is_not_a_flag(self):
        strong = "Food should be cooked to 165F. " * 3
        assert clean_dataset.self_flags(strong, strong.replace("165F", "150F")) == []


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

    def test_grouped_interval_resamples_prompts_not_pairs(self):
        strong, flawed = [5] * 10 + [1] * 10, [1] * 20
        lo, _ = lib.bootstrap_ci(strong, flawed)
        assert lo > 0.6  # twenty pairs look like strong evidence...
        lo, hi = lib.bootstrap_ci(strong, flawed, groups=["a"] * 10 + ["b"] * 10)
        assert (lo, hi) == (0.5, 1.0)  # ...but they are two prompts

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
        assert result["base"]["complete_rate"] is None  # answers written before the flag existed

    def test_complete_rate_and_cap_come_from_the_answers_file(self, tmp_path):
        answers = [{"prompt": "p", "answer": "a", "truncated": k == 0} for k in range(4)]
        (tmp_path / "answers-sft.json").write_text(
            json.dumps({"name": "sft", "max_new_tokens": 1024, "answers": answers}), encoding="utf-8"
        )
        backend = lib.FakeBackend(lambda chat: '{"overall_score": 7}')
        result = eval_heldout.score_answers(backend, "t/m", [tmp_path / "answers-sft.json"])
        assert result["sft"]["complete_rate"] == 0.75 and result["sft"]["max_new_tokens"] == 1024

    def test_an_answer_ended_only_if_it_produced_an_end_token(self):
        assert eval_heldout.ended([5, 6, 2, 0, 0], {2, 0})
        assert not eval_heldout.ended([5, 6, 7, 8], {2, 0})

    def test_judge_evaluate_counts_strong_wins(self):
        scorer = SimpleNamespace(
            score_many=lambda prompts, responses: [9.0 if "good" in r else 2.0 for r in responses]
        )
        pairs = [{"prompt": "p", "strong": "good", "flawed": "bad"}] * 10
        result = judge_eval.evaluate(scorer, pairs)
        assert result["accuracy"] == 1.0 and result["mean_gap"] == 7.0
        assert judge_eval.prompt_groups([{"prompt": "p", "prompt_id": 3}, {"prompt": "q"}]) == [3, "q"]

    def test_judge_reports_the_teacher_detectable_subset(self):
        def score(prompts, responses):
            return [9.0 if "good" in r else 5.0 if "subtle" in r else 2.0 for r in responses]

        scorer = SimpleNamespace(score_many=score)
        seen = [
            {"prompt": f"p{k}", "prompt_id": k, "strong": "good", "flawed": "bad", "teacher_detects": True}
            for k in range(4)
        ]
        missed = [
            {
                "prompt": f"q{k}",
                "prompt_id": 10 + k,
                "strong": "subtle",
                "flawed": "subtle!",
                "teacher_detects": False,
            }
            for k in range(4)
        ]
        result = judge_eval.evaluate(scorer, seen + missed)
        assert result["accuracy"] == 0.75 and result["pairs"] == 8
        subset = result["teacher_detectable"]
        assert subset["accuracy"] == 1.0 and (subset["pairs"], subset["prompts"]) == (4, 4)
        assert judge_eval.evaluate(scorer, missed)["teacher_detectable"] is None


class TestSeparableSubset:
    def pairs(self):
        return [
            {"pair_id": k, "prompt_id": k // 2, "prompt": f"p{k // 2}", "strong": "s", "flawed": "f"}
            for k in range(6)
        ]

    def test_pairwise_both_orders_must_pick_the_strong_answer(self):
        replies = iter(["A", "B", "A", "A", "B", "B", "  A.", "Answer B", "A", "B", "maybe", "B"])
        backend = lib.FakeBackend(lambda chat: next(replies))
        result = pairwise_teacher.compare(backend, "Qwen/Qwen2.5-32B-Instruct", self.pairs())
        assert [r["separable"] for r in result["rows"]] == [True, False, False, True, True, False]
        assert result["separable_ids"] == [0, 3, 4] and result["separable_prompts"] == 3
        assert result["unparsed"] == 1 and result["first_answer_rate"] == pytest.approx(5 / 12)
        first = backend.calls[0].turns[-1][1]
        assert first.index("Answer A:\ns") < first.index("Answer B:\nf")  # strong first, then swapped
        assert backend.calls[1].turns[-1][1].index("Answer A:\nf") > 0

    def test_parse_choice(self):
        assert pairwise_teacher.parse_choice("B") == "B"
        assert pairwise_teacher.parse_choice("Answer A is more correct.") == "A"
        assert pairwise_teacher.parse_choice("Neither") is None

    def test_add_separable_re_reads_recorded_scores(self):
        pairs = self.pairs()
        result = {"critics": {"c": judge_eval.from_scores([9.0, 1, 9, 1, 9, 9], [1.0, 9, 1, 9, 1, 1], pairs)}}
        judge_eval.add_separable(result, pairs, [0, 2, 4])
        sep = result["critics"]["c"]["teacher_separable"]
        assert sep["accuracy"] == 1.0 and (sep["pairs"], sep["prompts"]) == (3, 3)
        assert result["teacher_separable"]["pairs"] == 3
        judge_eval.add_separable(result, pairs, [])
        assert result["critics"]["c"]["teacher_separable"] is None


class TestNextRuns:
    def test_seed_configs_differ_only_where_they_must(self, tmp_path):
        written = seed_configs.seed_configs(43, "/job/sft-s43/merged", tmp_path)
        assert sorted(written) == [
            "control-composite-s43",
            "control-local-s43",
            "sft-composite-s43",
            "sft-local-s43",
        ]
        load = lambda name: yaml.safe_load(written[name].read_text(encoding="utf-8"))  # noqa: E731
        ctl, treated = load("control-composite-s43"), load("sft-composite-s43")
        assert ctl["seed"] == treated["seed"] == 43
        assert ctl["training"]["output_dir"] == "outputs/control-composite-s43"
        assert treated["student"]["model_name_or_path"] == "/job/sft-s43/merged"
        assert ctl["student"]["model_name_or_path"] == "Qwen/Qwen2.5-1.5B-Instruct"
        for config in (ctl, treated):
            del config["seed"], config["experiment_name"], config["training"]["output_dir"]
            del config["student"]["model_name_or_path"]
        assert ctl == treated  # teacher, schedule and everything else as in the 2026-10-06 control
        assert load("control-local-s43")["teacher"]["default_teacher"] == "local"

    def test_prompt_config_is_control_local_with_its_own_folder(self, tmp_path):
        path = seed_configs.prompt_config(43, 128, tmp_path)
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert path.name == "control-local-p128-s43.yaml" and config["seed"] == 43
        assert config["training"]["output_dir"] == "outputs/control-local-p128-s43"
        assert config["teacher"]["default_teacher"] == "local"
        assert config["student"]["model_name_or_path"] == "Qwen/Qwen2.5-1.5B-Instruct"

    def test_prompts_are_drawn_from_training_questions_evenly_by_topic(self):
        kept = [(f"topic-{t}", f"question {t}-{i}") for t in range(4) for i in range(40)]
        train, held = lib.split_held_out(kept, 64, 400, 42)
        picked = make_prompts.draw(kept, 20)
        assert len(picked) == len(set(picked)) == 20
        assert set(picked) <= {p for _, p in train} and not set(picked) & {p for _, p in held}
        counts = {t: sum(p.startswith(f"question {t}-") for p in picked) for t in range(4)}
        assert set(counts.values()) == {5}
        assert make_prompts.draw(kept, 20) == picked  # deterministic

    def test_sft_merge_only_needs_no_data(self):
        args = sft.parse_args(["--merge-only", "sft/epoch-2", "--out", "sft"])
        assert args.merge_only == Path("sft/epoch-2") and args.data is None
        with pytest.raises(SystemExit):
            sft.parse_args(["--out", "sft"])


class TestHostCheck:
    # nvidia-smi header lines from the two kinds of host the job profile has given us
    OLD = "| NVIDIA-SMI 570.195.03   Driver Version: 570.195.03   CUDA Version: 12.8     |"
    NEW = "| NVIDIA-SMI 580.82.07    Driver Version: 580.82.07    CUDA Version: 13.0     |"

    def test_the_a100_with_a_cuda_12_8_driver_is_refused(self):
        found = host_check.problems(self.OLD, "NVIDIA A100-SXM4-80GB, 81920 MiB", (13, 0), 90)
        assert found == [
            "the driver supports CUDA 12.8; this experiment needs 13.0",
            "the largest GPU (NVIDIA A100-SXM4-80GB) has 80 GB; this needs 90 GB",
        ]

    def test_the_rtx_pro_6000_passes(self):
        query = "NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887 MiB"
        assert host_check.problems(self.NEW, query, (13, 0), 90) == []
        assert host_check.gpus(query) == [("NVIDIA RTX PRO 6000 Blackwell Server Edition", 97887 / 1024)]

    def test_no_gpu_or_no_version_is_refused(self):
        assert host_check.problems("", "", (13, 0), 0) == [
            "nvidia-smi reported no CUDA version",
            "nvidia-smi reported no GPU",
        ]
        assert host_check.driver_cuda(self.NEW) == (13, 0)

    def test_slow_transfers_are_refused_with_the_rate(self):
        # seed 43's first pod: 26 GB in 31 minutes, about 14 MB/s
        assert host_check.rate_problem("downloads", 26_000_000_000, 1860, 100) == (
            "downloads ran at 14 MB/s (26.00 GB in 1860 s); this needs 100 MB/s"
        )
        assert host_check.rate_problem("downloads", 3_900_000_000, 20, 100) is None

    def test_the_download_floor_comes_from_the_plans_budget(self):
        # 66 GB in 45 minutes needs about 24 MB/s
        assert host_check.required_mbps(66, 45) == pytest.approx(24.44, abs=0.01)

    def test_the_2026_10_08_pod_would_have_been_refused(self):
        # plan 17: the 32B arrived at about 11 MB/s; a 3.9 GB shard at that rate
        nbytes, seconds = 3_900_000_000, 355
        assert host_check.projected_minutes(66, nbytes, seconds) == pytest.approx(100, abs=1)
        problem = host_check.rate_problem("the download", nbytes, seconds, host_check.required_mbps(66, 45))
        assert problem == "the download ran at 11 MB/s (3.90 GB in 355 s); this needs 24 MB/s"

    def test_write_rate_writes_flushes_and_cleans_up(self, tmp_path):
        nbytes, seconds = host_check.write_rate(tmp_path / "hf", megabytes=4)
        assert nbytes == 4 << 20 and seconds >= 0  # a 4 MiB write can be under the clock tick
        assert list((tmp_path / "hf").iterdir()) == []

    def test_newer_drivers_print_the_umd_version(self):
        header = "| NVIDIA-SMI 617.14     KMD Version: 617.14     CUDA UMD Version: 13.4     |"
        assert host_check.driver_cuda(header) == (13, 4)


class TestKevJudge:
    PAIRS = [
        {
            "pair_id": k,
            "prompt_id": k // 2,
            "prompt": f"q{k // 2}",
            "strong": f"good {k}",
            "flawed": f"bad {k}",
        }
        for k in range(4)
    ]

    @staticmethod
    def kev(prefer):
        """A fake Kev: picks the option `prefer` returns, with 0.8 of the probability."""

        def ask(body):
            options = body["questions"]["better"]["criteria"]
            pick = prefer(options)
            other = "B" if pick == "A" else "A"
            return {
                "answers": {
                    "better": {"type": "choice", "choice": pick, "probabilities": {pick: 0.8, other: 0.2}}
                }
            }

        return ask

    def test_request_puts_the_answers_in_the_options(self):
        body = judge_kev.request("Why is the sky blue?", "first answer", "second answer")
        question = body["questions"]["better"]
        assert body["state"] == "Why is the sky blue?" and question["type"] == "choice"
        assert question["criteria"] == {"A": "first answer", "B": "second answer"}

    def test_a_judge_that_reads_the_answers_is_accurate_and_unbiased(self):
        result = judge_kev.judge(self.kev(lambda o: "A" if o["A"].startswith("good") else "B"), self.PAIRS)
        assert (result["accuracy"], result["a_rate"]) == (1.0, 0.5)
        assert result["mean_margin"] == pytest.approx(0.6)
        assert result["separable_ids"] == [0, 1, 2, 3] and result["separable_prompts"] == 2
        assert judge_kev.reading(result) == "usable reference judge"

    def test_a_judge_that_always_says_a_is_half_right_and_biased(self):
        result = judge_kev.judge(self.kev(lambda o: "A"), self.PAIRS)
        assert (result["accuracy"], result["a_rate"], result["separable_pairs"]) == (0.5, 1.0, 0)
        assert result["mean_margin"] == pytest.approx(0.0)
        assert judge_kev.reading(result) == "not a useful judge of single planted errors"

    def test_reading_flags_an_accurate_but_biased_judge(self):
        assert (
            judge_kev.reading({"accuracy": 0.8, "a_rate": 0.7})
            == "accurate but position-biased: report both orders"
        )


class TestOrderAveraged:
    @staticmethod
    def row(prompt_id, a_strong_first, a_strong_second):
        """A pair from p(A) in each order: strong first, then strong second."""
        return {
            "prompt_id": prompt_id,
            "strong_first": {
                "choice": "A" if a_strong_first >= 0.5 else "B",
                "p_strong": a_strong_first,
                "p_flawed": 1 - a_strong_first,
            },
            "strong_second": {
                "choice": "A" if a_strong_second >= 0.5 else "B",
                "p_strong": 1 - a_strong_second,
                "p_flawed": a_strong_second,
            },
        }

    def test_a_biased_judge_with_a_steady_preference_is_favoured_on_every_pair(self):
        # Always says A, but leans further to A when the strong answer is A.
        rows = [self.row(k // 2, 0.9, 0.8) for k in range(6)]
        result = judge_kev.order_averaged(rows)
        assert result["accuracy"] == 1.0 and result["favoured_pairs"] == 6 and result["prompts"] == 3
        assert result["both_orders_strong"] == 0
        assert result["mean_abs_gap"] == pytest.approx(0.1)
        assert judge_kev.confirmation_reading(result["accuracy"], result["ci"][0]) == "confirmed"

    def test_ties_count_half_and_the_bar_is_085(self):
        rows = [self.row(0, 0.9, 0.9), self.row(1, 0.9, 0.7)]
        result = judge_kev.order_averaged(rows)
        assert result["accuracy"] == 0.75 and result["favoured_pairs"] == 1
        assert judge_kev.confirmation_reading(0.85, 0.75) == "confirmed"
        assert judge_kev.confirmation_reading(0.849, 0.8) == "not confirmed"
        assert judge_kev.confirmation_reading(0.9, 0.749) == "not confirmed"

    def test_probabilities_are_normalised_over_the_two_options(self):
        row = {
            "prompt_id": 0,
            "strong_first": {"choice": "A", "p_strong": 0.375, "p_flawed": 0.125},
            "strong_second": {"choice": "A", "p_strong": 0.125, "p_flawed": 0.375},
        }
        assert judge_kev.order_averaged([row])["accuracy"] == 0.5

    def test_the_interval_resamples_whole_prompts(self):
        rows = [self.row(0, 0.9, 0.8)] * 5 + [self.row(1, 0.8, 0.9)]
        lo, hi = judge_kev.order_averaged(rows)["ci"]
        assert lo < 0.2 and hi == 1.0


class TestLogprobJudge:
    PAIRS = TestKevJudge.PAIRS

    @staticmethod
    def server(lean):
        """A fake llama-server: p(A) = lean(request text), returned as top log-probabilities."""

        def ask(body):
            p_a = lean(body["messages"][1]["content"])
            top = [
                {"token": "A", "logprob": math.log(p_a)},
                {"token": " B", "logprob": math.log(1 - p_a)},
                {"token": "The", "logprob": math.log(1e-4)},
            ]
            return {"choices": [{"logprobs": {"content": [{"top_logprobs": top}]}}]}

        return ask

    def test_reads_letters_from_the_top_logprobs(self):
        response = self.server(lambda text: 0.7)({"messages": [{}, {"content": ""}]})
        assert judge_logprob.letter_probs(response) == pytest.approx((0.7, 0.3))

    def test_a_judge_that_always_says_a_can_still_prefer_the_strong_answer(self):
        def lean(text):
            return 0.95 if "Answer A:\ngood" in text else 0.85

        result = judge_logprob.judge(self.server(lean), self.PAIRS)
        assert (result["accuracy"], result["a_rate"]) == (0.5, 1.0)
        assert result["order_averaged"]["accuracy"] == 1.0
        assert result["order_averaged"]["reading"] == "confirmed"

    def test_no_letter_in_the_top_logprobs_is_a_tie(self):
        top = [{"token": "x", "logprob": 0.0}]
        response = {"choices": [{"logprobs": {"content": [{"top_logprobs": top}]}}]}
        assert judge_logprob.letter_probs(response) == (0.5, 0.5)


class TestFreshPairs:
    @staticmethod
    def train_rows():
        rows = []
        for k in range(12):
            topic = ["alpha", "beta", "gamma"][k % 3]
            answer = f"Sentence one about {k}. The value is {k}. Sentence three closes {k}."
            for kind in ("answer", "revision"):
                rows.append(
                    {
                        "topic": topic,
                        "kind": kind,
                        "truncated": k == 11,
                        "messages": [
                            {"role": "system", "content": "s"},
                            {"role": "user", "content": f"Question {k}?"},
                            {"role": "assistant", "content": answer},
                        ],
                    }
                )
        return rows

    def test_strong_answers_keep_untruncated_answer_rows(self):
        items = fresh_pairs.strong_answers(self.train_rows())
        assert len(items) == 11 and items[0]["prompt_id"] == "t0" and items[3]["prompt"] == "Question 3?"

    def test_split_is_disjoint_and_balanced_by_topic(self):
        items = fresh_pairs.strong_answers(self.train_rows())
        confirm, train = fresh_pairs.split(items, 3)
        assert len(confirm) == 3 and not set(confirm) & set(train)
        assert sorted(confirm + train) == sorted(i["prompt_id"] for i in items)
        topics = {i["prompt_id"]: i["topic"] for i in items}
        assert sorted(topics[c] for c in confirm) == ["alpha", "beta", "gamma"]

    def test_changed_span_is_the_edit(self):
        span = fresh_pairs.changed_span("The value is 7. Done.", "The value is 9. Done.")
        assert span == {"original": "7", "edited": "9"}

    def test_build_plants_and_splits_pairs(self, tmp_path):
        def answer(chat):
            value = re.search(r"The value is (\d+)\.", chat.turns[0][1]).group(1)
            return json.dumps(
                {"original": f"The value is {value}.", "edited": f"The value is {int(value) + 1}."}
            )

        items = fresh_pairs.strong_answers(self.train_rows())
        backend = lib.FakeBackend(answer)
        result = fresh_pairs.build(backend, "Qwen/Qwen2.5-32B-Instruct", items, 3, planted_by="q4")
        report = result["report"]
        assert report["planted_by"] == "q4" and report["slots"] == 22
        assert report["confirm"]["pairs"] + report["train"]["pairs"] == report["planted"] > 0
        confirm_ids = set(result["split"]["confirm"])
        assert all(p["prompt_id"] in confirm_ids for p in result["pairs"]["confirm"])
        assert not any(p["prompt_id"] in confirm_ids for p in result["pairs"]["train"])
        fresh_pairs.write(result, tmp_path)
        train = lib.read_jsonl(tmp_path / "train_pairs.jsonl")
        keys = {"pair_id", "prompt_id", "topic", "prompt", "strong", "flawed", "flaw_kind", "edit"}
        assert train and set(train[0]) == keys
        confirm = json.loads((tmp_path / "confirm_set.json").read_text(encoding="utf-8"))
        assert all(c["strong"] != c["flawed"] for c in confirm)


class TestProbe:
    def test_drift_exports_record_the_models_training_seed(self, tmp_path):
        rng = np.random.default_rng(0)
        names = ["base", "sft", "sft-aspire-1"]
        states = rng.standard_normal((3, 6, 8))
        pairs = [(f"q{i}", f"a{i}") for i in range(6)]
        dims = [{"correctness": 7.0 + i % 3} for i in range(6)]
        teachers = [{"local:Qwen2.5-32B-Instruct": 7.0 + i % 3} for i in range(6)]
        summary = probe_models.write_exports(
            states, names, pairs, dims, teachers, ["base", "sft"], tmp_path, 43
        )
        assert summary["seed"] == 43
        for ref, checkpoints in (("base", 2), ("sft", 1)):
            meta = json.loads((tmp_path / f"drift-from-{ref}.geometry.json").read_text(encoding="utf-8"))[
                "run_metadata"
            ]
            assert (meta["seed"], meta["run_id"], meta["checkpoints"]) == (
                43,
                f"drift-from-{ref}",
                checkpoints,
            )
            assert (meta["step_axis"], meta["scalar_source"]) == ("checkpoint_by_item", "fixed_per_item")

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
