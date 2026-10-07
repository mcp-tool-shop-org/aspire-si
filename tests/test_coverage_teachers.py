"""Coverage-closing tests for aspire.teachers (base, composite, claude, openai, local)."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch

from aspire.teachers.base import (
    MAX_PROMPT_LENGTH,
    MAX_RESPONSE_LENGTH,
    BaseTeacher,
    ChallengeType,
    DialogueHistory,
    DialogueTurn,
    DimensionScore,
    EvaluationDimension,
    InputValidationError,
    TeacherChallenge,
    TeacherEvaluation,
)
from aspire.teachers.claude import ClaudeTeacher, ClaudeTeacherError
from aspire.teachers.composite import CompositeTeacher, CurriculumCompositeTeacher
from aspire.teachers.local import LocalTeacher
from aspire.teachers.openai import OpenAITeacher, OpenAITeacherError

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class FakeTeacher(BaseTeacher):
    """Deterministic teacher that records which method was called."""

    def __init__(self, name="fake", score=5.0, challenges=None, dims=None, **kw):
        super().__init__(
            name=name,
            preferred_challenges=challenges,
            evaluation_dimensions=dims,
            **kw,
        )
        self.score = score
        self.challenge_calls = []
        self.evaluate_calls = []

    async def challenge(self, prompt, student_response, dialogue_history=None, challenge_type=None):
        self.challenge_calls.append(challenge_type)
        return TeacherChallenge(
            challenge_type=challenge_type or ChallengeType.SOCRATIC, content=f"{self.name}-challenge"
        )

    async def evaluate(self, prompt, student_response, dialogue_history=None, generate_improved=True):
        self.evaluate_calls.append(generate_improved)
        return TeacherEvaluation(
            overall_score=self.score,
            dimension_scores=[
                DimensionScore(EvaluationDimension.CLARITY, self.score, "clear"),
                DimensionScore(EvaluationDimension.REASONING, self.score + 1, "reasoned"),
            ],
            reasoning=f"{self.name}-reason",
            improved_response=f"{self.name}-improved" if generate_improved else None,
            strengths=[f"{self.name}-s", "shared-s"],
            weaknesses=["shared-w"],
            suggestions=["shared-sug"],
        )


# ---------------------------------------------------------------------------
# BaseTeacher
# ---------------------------------------------------------------------------


class _AbstractBodies(BaseTeacher):
    async def challenge(self, prompt, student_response, dialogue_history=None, challenge_type=None):
        return await super().challenge(prompt, student_response, dialogue_history, challenge_type)

    async def evaluate(self, prompt, student_response, dialogue_history=None, generate_improved=True):
        return await super().evaluate(prompt, student_response, dialogue_history, generate_improved)


class TestBaseTeacherValidation:
    async def test_abstract_bodies_return_none(self):
        t = _AbstractBodies()
        assert await t.challenge("p", "r") is None
        assert await t.evaluate("p", "r") is None

    def test_non_string_prompt_rejected(self):
        with pytest.raises(InputValidationError, match="Prompt must be a string, got int"):
            FakeTeacher()._validate_input(prompt=123)

    def test_overlong_prompt_rejected_but_limit_accepted(self):
        t = FakeTeacher()
        t._validate_input(prompt="x" * MAX_PROMPT_LENGTH)  # exactly at limit is fine
        with pytest.raises(InputValidationError, match="Prompt exceeds maximum length"):
            t._validate_input(prompt="x" * (MAX_PROMPT_LENGTH + 1))

    def test_non_string_response_rejected(self):
        with pytest.raises(InputValidationError, match="Student response must be a string, got list"):
            FakeTeacher()._validate_input(student_response=["a"])

    def test_overlong_response_rejected_but_limit_accepted(self):
        t = FakeTeacher()
        t._validate_input(student_response="x" * MAX_RESPONSE_LENGTH)
        with pytest.raises(InputValidationError, match="Student response exceeds maximum length"):
            t._validate_input(student_response="x" * (MAX_RESPONSE_LENGTH + 1))

    def test_none_inputs_are_skipped(self):
        FakeTeacher()._validate_input()

    def test_input_validation_error_is_value_error_with_code(self):
        err = InputValidationError("bad")
        assert isinstance(err, ValueError)
        assert err.code == "ASPIRE_INVALID_INPUT"


class TestBaseTeacherRunDialogue:
    async def test_run_dialogue_records_turns_and_final_eval(self):
        teacher = FakeTeacher(score=7.0)
        seen = []

        async def student(prompt, challenge, dialogue_history):
            seen.append(challenge)
            return f"reply{len(seen)}"

        history = await teacher.run_dialogue("p", "init", student, max_turns=2, evaluate_each_turn=True)
        assert [t.student_response for t in history.turns] == ["reply1", "reply2"]
        assert [t.turn_number for t in history.turns] == [1, 2]
        assert all(t.evaluation is not None for t in history.turns)
        assert history.final_evaluation.improved_response == "fake-improved"
        # per-turn evals skip the improved response, the final one asks for it
        assert teacher.evaluate_calls == [False, False, True]
        assert history.get_trajectory_scores() == [7.0, 7.0]

    async def test_run_dialogue_without_per_turn_eval(self):
        teacher = FakeTeacher()

        async def student(prompt, challenge, dialogue_history):
            return "r"

        history = await teacher.run_dialogue("p", "init", student, max_turns=1)
        assert history.turns[0].evaluation is None
        assert history.get_trajectory_scores() == []
        assert history.num_turns == 1

    def test_select_challenge_type_uses_preferred(self):
        t = FakeTeacher(challenges=[ChallengeType.ETHICAL])
        assert t.select_challenge_type() is ChallengeType.ETHICAL

    def test_system_prompt_mentions_name_and_description(self):
        t = FakeTeacher(name="Prof")
        assert "Prof" in t.get_system_prompt()
        assert t.description in t.get_system_prompt()

    def test_evaluation_passed_threshold(self):
        assert TeacherEvaluation(6.0, [], "r").passed
        assert not TeacherEvaluation(5.99, [], "r").passed

    def test_evaluation_to_dict(self):
        ev = TeacherEvaluation(
            8.0, [DimensionScore(EvaluationDimension.CLARITY, 8.0, "e")], "r", strengths=["s"]
        )
        d = ev.to_dict()
        assert d["dimension_scores"] == [{"dimension": "clarity", "score": 8.0, "explanation": "e"}]
        assert d["strengths"] == ["s"]
        assert json.dumps(d)  # serializable


# ---------------------------------------------------------------------------
# CompositeTeacher
# ---------------------------------------------------------------------------


class TestCompositeConstruction:
    def test_empty_teachers_rejected(self):
        with pytest.raises(ValueError, match="at least one teacher"):
            CompositeTeacher([])

    def test_weight_length_mismatch_rejected(self):
        with pytest.raises(ValueError, match="Weights must match"):
            CompositeTeacher([FakeTeacher()], weights=[0.5, 0.5])

    def test_default_weights_uniform_and_merged_attributes(self):
        a = FakeTeacher("a", challenges=[ChallengeType.ETHICAL], dims=[EvaluationDimension.CLARITY])
        b = FakeTeacher("b", challenges=[ChallengeType.CREATIVE], dims=[EvaluationDimension.EMPATHY])
        comp = CompositeTeacher([a, b])
        assert comp.weights == [0.5, 0.5]
        assert set(comp.preferred_challenges) == {ChallengeType.ETHICAL, ChallengeType.CREATIVE}
        assert set(comp.evaluation_dimensions) == {EvaluationDimension.CLARITY, EvaluationDimension.EMPATHY}

    def test_system_prompt_lists_teachers(self):
        comp = CompositeTeacher([FakeTeacher("alpha"), FakeTeacher("beta")])
        prompt = comp.get_system_prompt()
        assert "alpha" in prompt and "beta" in prompt


class TestCompositeChallengeStrategies:
    async def test_rotate_cycles_through_teachers(self):
        a, b = FakeTeacher("a"), FakeTeacher("b")
        comp = CompositeTeacher([a, b], strategy="rotate")
        names = [(await comp.challenge("p", "r")).content for _ in range(3)]
        assert names == ["a-challenge", "b-challenge", "a-challenge"]

    async def test_specialize_picks_teacher_with_matching_challenge(self):
        a = FakeTeacher("a", challenges=[ChallengeType.ETHICAL])
        b = FakeTeacher("b", challenges=[ChallengeType.CREATIVE])
        comp = CompositeTeacher([a, b], strategy="specialize")
        result = await comp.challenge("p", "r", challenge_type=ChallengeType.CREATIVE)
        assert result.content == "b-challenge"
        assert b.challenge_calls == [ChallengeType.CREATIVE]
        assert a.challenge_calls == []

    async def test_specialize_auto_selects_challenge_type_when_none(self):
        a = FakeTeacher("a", challenges=[ChallengeType.ETHICAL])
        comp = CompositeTeacher([a], strategy="specialize")
        await comp.challenge("p", "r")
        assert a.challenge_calls == [ChallengeType.ETHICAL]

    async def test_specialize_falls_back_to_first_teacher(self):
        a = FakeTeacher("a", challenges=[ChallengeType.ETHICAL])
        b = FakeTeacher("b", challenges=[ChallengeType.CREATIVE])
        comp = CompositeTeacher([a, b], strategy="specialize")
        # Force a challenge type that no teacher prefers.
        a.preferred_challenges = [ChallengeType.ETHICAL]
        b.preferred_challenges = [ChallengeType.CREATIVE]
        result = await comp.challenge("p", "r", challenge_type=ChallengeType.EDGE_CASE)
        assert result.content == "a-challenge"

    async def test_random_uses_weighted_choice(self):
        a, b = FakeTeacher("a"), FakeTeacher("b")
        comp = CompositeTeacher([a, b], strategy="random", weights=[0.1, 0.9])
        with patch("aspire.teachers.composite.random.choices", return_value=[b]) as choices:
            result = await comp.challenge("p", "r")
        choices.assert_called_once_with([a, b], weights=[0.1, 0.9], k=1)
        assert result.content == "b-challenge"

    async def test_unknown_strategy_defaults_to_first_teacher(self):
        a, b = FakeTeacher("a"), FakeTeacher("b")
        comp = CompositeTeacher([a, b], strategy="vote")
        assert (await comp.challenge("p", "r")).content == "a-challenge"


class TestCompositeEvaluationStrategies:
    async def test_rotate_evaluate_uses_current_teacher_without_advancing(self):
        a, b = FakeTeacher("a", score=3.0), FakeTeacher("b", score=9.0)
        comp = CompositeTeacher([a, b], strategy="rotate")
        assert (await comp.evaluate("p", "r")).overall_score == 3.0
        await comp.challenge("p", "r")  # advances rotation
        assert (await comp.evaluate("p", "r")).overall_score == 9.0

    async def test_unknown_strategy_evaluates_with_first_teacher(self):
        a, b = FakeTeacher("a", score=3.0), FakeTeacher("b", score=9.0)
        comp = CompositeTeacher([a, b], strategy="specialize")
        assert (await comp.evaluate("p", "r")).overall_score == 3.0
        assert b.evaluate_calls == []

    async def test_vote_weighted_average_and_dimension_merge(self):
        a, b = FakeTeacher("a", score=4.0), FakeTeacher("b", score=8.0)
        comp = CompositeTeacher([a, b], strategy="vote", weights=[1.0, 3.0])
        ev = await comp.evaluate("p", "r")
        assert ev.overall_score == pytest.approx((4.0 * 1 + 8.0 * 3) / 4)
        dims = {d.dimension: d for d in ev.dimension_scores}
        assert dims[EvaluationDimension.CLARITY].score == pytest.approx(7.0)
        assert dims[EvaluationDimension.REASONING].score == pytest.approx(8.0)
        assert dims[EvaluationDimension.CLARITY].explanation == "Combined from 2 teachers"
        # best (highest scoring) teacher supplies the improved response
        assert ev.improved_response == "b-improved"
        assert ev.metadata["strategy"] == "vote"
        assert ev.metadata["teacher_scores"] == {"a": 4.0, "b": 8.0}
        # feedback is de-duplicated across teachers
        assert sorted(ev.strengths) == ["a-s", "b-s", "shared-s"]
        assert ev.weaknesses == ["shared-w"]
        assert "[a]: a-reason" in ev.reasoning and "[b]: b-reason" in ev.reasoning

    async def test_vote_without_improved_response(self):
        comp = CompositeTeacher([FakeTeacher("a"), FakeTeacher("b")], strategy="vote")
        ev = await comp.evaluate("p", "r", generate_improved=False)
        assert ev.improved_response is None

    async def test_debate_delegates_to_vote_combination(self):
        a, b = FakeTeacher("a", score=2.0), FakeTeacher("b", score=6.0)
        comp = CompositeTeacher([a, b], strategy="debate")
        ev = await comp.evaluate("p", "r")
        assert ev.overall_score == pytest.approx(4.0)
        assert ev.metadata["strategy"] == "vote"


class TestCurriculumComposite:
    def test_stage_switching_updates_weights(self):
        a, b = FakeTeacher("a"), FakeTeacher("b")
        comp = CurriculumCompositeTeacher(
            [a, b],
            stage_weights={"foundation": [0.9, 0.1], "advanced": [0.2, 0.8]},
        )
        assert comp.weights == [0.9, 0.1]
        comp.set_stage("advanced")
        assert comp.current_stage == "advanced"
        assert comp.weights == [0.2, 0.8]

    def test_unknown_stage_is_ignored(self):
        comp = CurriculumCompositeTeacher(
            [FakeTeacher("a"), FakeTeacher("b")],
            stage_weights={"foundation": [0.9, 0.1]},
        )
        comp.set_stage("nonexistent")
        assert comp.current_stage == "foundation"
        assert comp.weights == [0.9, 0.1]

    def test_initial_stage_missing_from_weights_keeps_uniform_default(self):
        comp = CurriculumCompositeTeacher(
            [FakeTeacher("a"), FakeTeacher("b")],
            stage_weights={"advanced": [0.2, 0.8]},
            current_stage="foundation",
        )
        assert comp.weights == [0.5, 0.5]

    async def test_curriculum_weights_affect_vote(self):
        a, b = FakeTeacher("a", score=0.0), FakeTeacher("b", score=10.0)
        comp = CurriculumCompositeTeacher([a, b], stage_weights={"s1": [1.0, 0.0], "s2": [0.0, 1.0]}, current_stage="s1")
        assert (await comp.evaluate("p", "r")).overall_score == 0.0
        comp.set_stage("s2")
        assert (await comp.evaluate("p", "r")).overall_score == 10.0


# ---------------------------------------------------------------------------
# ClaudeTeacher
# ---------------------------------------------------------------------------


def _claude_response(text):
    return SimpleNamespace(content=[SimpleNamespace(text=text)])


@pytest.fixture
def claude_teacher(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with patch("aspire.teachers.claude.anthropic.AsyncAnthropic") as cls:
        client = MagicMock()
        client.messages.create = AsyncMock()
        cls.return_value = client
        teacher = ClaudeTeacher()
        yield teacher, client


class TestClaudeTeacher:
    def test_missing_key_raises_with_variable_name(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(ClaudeTeacherError, match="ANTHROPIC_API_KEY"):
            ClaudeTeacher()

    async def test_challenge_parses_plain_fence(self, claude_teacher):
        teacher, client = claude_teacher
        payload = {"challenge": "Why?", "context": "ctx", "difficulty": 0.8}
        client.messages.create.return_value = _claude_response(f"```\n{json.dumps(payload)}\n```")
        ch = await teacher.challenge("p", "r", challenge_type=ChallengeType.SOCRATIC)
        assert (ch.content, ch.context, ch.difficulty) == ("Why?", "ctx", 0.8)

    async def test_challenge_parses_json_fence_and_includes_history(self, claude_teacher):
        teacher, client = claude_teacher
        client.messages.create.return_value = _claude_response('```json\n{"challenge": "Hmm?"}\n```')
        history = DialogueHistory("p", "r")
        history.add_turn(DialogueTurn(1, TeacherChallenge(ChallengeType.EDGE_CASE, "earlier-q"), "earlier-a"))
        ch = await teacher.challenge("p", "r", history, ChallengeType.EDGE_CASE)
        assert ch.content == "Hmm?" and ch.difficulty == 0.5 and ch.context is None
        sent = client.messages.create.call_args.kwargs["messages"][0]["content"]
        assert "earlier-q" in sent and "earlier-a" in sent

    async def test_challenge_selects_type_when_not_given(self, claude_teacher):
        teacher, client = claude_teacher
        client.messages.create.return_value = _claude_response('{"challenge": "q"}')
        ch = await teacher.challenge("p", "r")
        assert ch.challenge_type in teacher.preferred_challenges

    async def test_challenge_falls_back_to_raw_text_on_bad_json(self, claude_teacher):
        teacher, client = claude_teacher
        client.messages.create.return_value = _claude_response("just a question?")
        ch = await teacher.challenge("p", "r", challenge_type=ChallengeType.CREATIVE)
        assert ch.content == "just a question?"
        assert ch.challenge_type is ChallengeType.CREATIVE

    async def test_challenge_falls_back_when_key_missing(self, claude_teacher):
        teacher, client = claude_teacher
        client.messages.create.return_value = _claude_response('{"other": 1}')
        ch = await teacher.challenge("p", "r", challenge_type=ChallengeType.CREATIVE)
        assert ch.content == '{"other": 1}'

    async def test_challenge_validates_input_before_calling_api(self, claude_teacher):
        teacher, client = claude_teacher
        with pytest.raises(InputValidationError):
            await teacher.challenge(123, "r")
        client.messages.create.assert_not_called()

    async def test_evaluate_parses_plain_fence_and_skips_bad_dimensions(self, claude_teacher):
        teacher, client = claude_teacher
        payload = {
            "overall_score": 8.5,
            "dimension_scores": [
                {"dimension": "clarity", "score": 9.0, "explanation": "good"},
                {"dimension": "not_a_dimension", "score": 1.0},
                {"score": 2.0},
            ],
            "reasoning": "ok",
            "strengths": ["s"],
            "improved_response": "better",
        }
        client.messages.create.return_value = _claude_response(f"```\n{json.dumps(payload)}\n```")
        ev = await teacher.evaluate("p", "r")
        assert ev.overall_score == 8.5
        assert [d.dimension for d in ev.dimension_scores] == [EvaluationDimension.CLARITY]
        assert ev.improved_response == "better" and ev.strengths == ["s"]
        assert ev.weaknesses == [] and ev.suggestions == []

    async def test_evaluate_parses_json_fence_with_history(self, claude_teacher):
        teacher, client = claude_teacher
        client.messages.create.return_value = _claude_response('```json\n{"overall_score": 6}\n```')
        history = DialogueHistory("p", "r")
        history.add_turn(DialogueTurn(1, TeacherChallenge(ChallengeType.EDGE_CASE, "hq"), "ha"))
        ev = await teacher.evaluate("p", "r", history, generate_improved=False)
        assert ev.overall_score == 6 and ev.passed
        kwargs = client.messages.create.call_args.kwargs
        # The default model (Sonnet 5.5) rejects a non-default temperature, so none is sent.
        assert "temperature" not in kwargs
        assert kwargs["max_tokens"] == teacher.max_tokens * 2
        assert "hq" in kwargs["messages"][0]["content"]

    async def test_older_models_still_get_the_evaluation_temperature(self, claude_teacher):
        teacher, client = claude_teacher
        teacher.model = "claude-sonnet-4-6"
        client.messages.create.return_value = _claude_response('{"overall_score": 6}')
        await teacher.evaluate("p", "r", generate_improved=False)
        assert client.messages.create.call_args.kwargs["temperature"] == 0.3
        await teacher.challenge("p", "r")
        assert client.messages.create.call_args.kwargs["temperature"] == teacher.temperature

    @pytest.mark.parametrize(
        ("model", "accepts"),
        [
            ("claude-sonnet-5-5", False),
            ("claude-sonnet-5", False),
            ("claude-opus-5-5", False),
            ("claude-opus-4-8", False),
            ("claude-opus-4-7", False),
            ("claude-fable-5-1", False),
            ("claude-opus-4-6", True),
            ("claude-sonnet-4-6", True),
            ("claude-haiku-4-5", True),
            ("claude-sonnet-4-20250514", True),
        ],
    )
    def test_which_models_take_a_temperature(self, model, accepts):
        from aspire.teachers.claude import accepts_temperature

        assert accepts_temperature(model) is accepts

    async def test_thinking_blocks_before_the_answer_are_skipped(self, claude_teacher):
        teacher, client = claude_teacher
        client.messages.create.return_value = SimpleNamespace(
            stop_reason="end_turn",
            content=[
                SimpleNamespace(type="thinking", thinking=""),
                SimpleNamespace(type="text", text='{"overall_score": 8}'),
            ],
        )
        ev = await teacher.evaluate("p", "r", generate_improved=False)
        assert ev.overall_score == 8

    async def test_a_refusal_raises_a_teacher_error(self, claude_teacher):
        from aspire.teachers.claude import ClaudeTeacherError

        teacher, client = claude_teacher
        client.messages.create.return_value = SimpleNamespace(
            stop_reason="refusal",
            stop_details=SimpleNamespace(category="cyber"),
            content=[],
        )
        with pytest.raises(ClaudeTeacherError, match=r"declined.*\(cyber\)"):
            await teacher.evaluate("p", "r")

    async def test_evaluate_falls_back_to_neutral_score_on_garbage(self, claude_teacher):
        teacher, client = claude_teacher
        client.messages.create.return_value = _claude_response("not json at all")
        ev = await teacher.evaluate("p", "r")
        assert ev.overall_score == 5.0
        assert ev.reasoning == "not json at all"
        assert ev.dimension_scores == []

    async def test_evaluate_falls_back_when_overall_score_missing(self, claude_teacher):
        teacher, client = claude_teacher
        client.messages.create.return_value = _claude_response('{"reasoning": "x"}')
        ev = await teacher.evaluate("p", "r")
        assert ev.overall_score == 5.0

    def test_every_challenge_type_has_a_specific_description(self, claude_teacher):
        teacher, _ = claude_teacher
        generic = "Ask a challenging follow-up question"
        for ct in ChallengeType:
            assert teacher._get_challenge_description(ct) != generic


# ---------------------------------------------------------------------------
# OpenAITeacher
# ---------------------------------------------------------------------------


def _openai_response(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


@pytest.fixture
def openai_teacher(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    with patch("aspire.teachers.openai.AsyncOpenAI") as cls:
        client = MagicMock()
        client.chat.completions.create = AsyncMock()
        cls.return_value = client
        yield OpenAITeacher(), client


class TestOpenAITeacher:
    def test_missing_key_raises_with_variable_name(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(OpenAITeacherError, match="OPENAI_API_KEY"):
            OpenAITeacher()

    async def test_challenge_json_with_history(self, openai_teacher):
        teacher, client = openai_teacher
        client.chat.completions.create.return_value = _openai_response(
            json.dumps({"challenge": "Prove it", "context": "c", "difficulty": 0.9})
        )
        history = DialogueHistory("p", "r")
        history.add_turn(DialogueTurn(1, TeacherChallenge(ChallengeType.EDGE_CASE, "oq"), "oa"))
        ch = await teacher.challenge("p", "r", history, ChallengeType.STEELMAN)
        assert (ch.content, ch.context, ch.difficulty) == ("Prove it", "c", 0.9)
        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["response_format"] == {"type": "json_object"}
        assert "oq" in kwargs["messages"][1]["content"]

    async def test_challenge_falls_back_to_raw_content(self, openai_teacher):
        teacher, client = openai_teacher
        client.chat.completions.create.return_value = _openai_response("plain question")
        ch = await teacher.challenge("p", "r", challenge_type=ChallengeType.ETHICAL)
        assert ch.content == "plain question" and ch.difficulty == 0.5

    async def test_challenge_falls_back_on_missing_key(self, openai_teacher):
        teacher, client = openai_teacher
        client.chat.completions.create.return_value = _openai_response("{}")
        ch = await teacher.challenge("p", "r")
        assert ch.content == "{}"

    async def test_challenge_selects_type_when_not_given(self, openai_teacher):
        teacher, client = openai_teacher
        client.chat.completions.create.return_value = _openai_response('{"challenge": "q"}')
        ch = await teacher.challenge("p", "r")
        assert ch.challenge_type in teacher.preferred_challenges

    async def test_evaluate_parses_dimensions_and_skips_invalid(self, openai_teacher):
        teacher, client = openai_teacher
        payload = {
            "overall_score": 7.0,
            "dimension_scores": [
                {"dimension": "reasoning", "score": 7.5},
                {"dimension": "bogus", "score": 1},
                {"dimension": "clarity"},  # missing score -> skipped
            ],
            "reasoning": "r",
            "improved_response": "imp",
        }
        client.chat.completions.create.return_value = _openai_response(json.dumps(payload))
        history = DialogueHistory("p", "r")
        history.add_turn(DialogueTurn(1, TeacherChallenge(ChallengeType.EDGE_CASE, "oq"), "oa"))
        ev = await teacher.evaluate("p", "r", history)
        assert ev.overall_score == 7.0
        assert [(d.dimension, d.score, d.explanation) for d in ev.dimension_scores] == [
            (EvaluationDimension.REASONING, 7.5, "")
        ]
        assert ev.improved_response == "imp"
        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["temperature"] == 0.3
        assert "oq" in kwargs["messages"][1]["content"]

    async def test_evaluate_falls_back_on_invalid_json(self, openai_teacher):
        teacher, client = openai_teacher
        client.chat.completions.create.return_value = _openai_response("oops")
        ev = await teacher.evaluate("p", "r")
        assert (ev.overall_score, ev.reasoning, ev.dimension_scores) == (5.0, "oops", [])

    async def test_evaluate_falls_back_on_missing_overall_score(self, openai_teacher):
        teacher, client = openai_teacher
        client.chat.completions.create.return_value = _openai_response('{"reasoning": "x"}')
        ev = await teacher.evaluate("p", "r", generate_improved=False)
        assert ev.overall_score == 5.0

    async def test_validation_rejects_before_api_call(self, openai_teacher):
        teacher, client = openai_teacher
        with pytest.raises(InputValidationError):
            await teacher.evaluate("p", 5)
        with pytest.raises(InputValidationError):
            await teacher.challenge("p", 5)
        client.chat.completions.create.assert_not_called()


# ---------------------------------------------------------------------------
# LocalTeacher
# ---------------------------------------------------------------------------


class _Batch(dict):
    def to(self, device):
        self.device = device
        return self


class _FakeTok:
    pad_token = None
    eos_token = "<eos>"
    pad_token_id = 0

    def __init__(self, decoded="generated text"):
        self.decoded = decoded
        self.prompts = []

    def __call__(self, text, **kwargs):
        self.prompts.append(text)
        return _Batch(input_ids=torch.ones(1, 4, dtype=torch.long))

    def decode(self, ids, skip_special_tokens=True):
        self.last_ids = ids
        return self.decoded


@pytest.fixture
def local_teacher_factory():
    created = {}

    def make(**kwargs):
        with patch("aspire.teachers.local.AutoTokenizer") as tok_cls, patch(
            "aspire.teachers.local.AutoModelForCausalLM"
        ) as model_cls, patch("aspire.teachers.local.BitsAndBytesConfig") as bnb_cls:
            tok = _FakeTok()
            tok_cls.from_pretrained.return_value = tok
            model = MagicMock()
            model.generate.return_value = torch.arange(10).unsqueeze(0)
            model_cls.from_pretrained.return_value = model
            created.update(tok=tok, model=model, bnb=bnb_cls, model_cls=model_cls)
            kwargs.setdefault("device", "cpu")
            return LocalTeacher("some/model", **kwargs)

    return make, created


class TestLocalTeacher:
    def test_default_4bit_quantization_and_pad_fallback(self, local_teacher_factory):
        make, created = local_teacher_factory
        teacher = make()
        assert teacher.tokenizer.pad_token == "<eos>"
        assert created["bnb"].call_args.kwargs["load_in_4bit"] is True
        qcfg = created["model_cls"].from_pretrained.call_args.kwargs["quantization_config"]
        assert qcfg is created["bnb"].return_value
        created["model"].eval.assert_called_once()

    def test_existing_pad_token_is_kept(self, local_teacher_factory):
        make, created = local_teacher_factory
        with patch.object(_FakeTok, "pad_token", "<pad>"):
            teacher = make()
            assert teacher.tokenizer.pad_token == "<pad>"

    def test_8bit_when_4bit_disabled(self, local_teacher_factory):
        make, created = local_teacher_factory
        make(load_in_4bit=False, load_in_8bit=True)
        created["bnb"].assert_called_once_with(load_in_8bit=True)

    def test_no_quantization_when_both_disabled(self, local_teacher_factory):
        make, created = local_teacher_factory
        make(load_in_4bit=False, load_in_8bit=False)
        created["bnb"].assert_not_called()
        assert created["model_cls"].from_pretrained.call_args.kwargs["quantization_config"] is None

    async def test_challenge_decodes_only_new_tokens(self, local_teacher_factory):
        make, created = local_teacher_factory
        teacher = make()
        created["tok"].decoded = "  Why do you think so?  "
        history = DialogueHistory("p", "r")
        history.add_turn(DialogueTurn(1, TeacherChallenge(ChallengeType.EDGE_CASE, "oldq"), "olda"))
        ch = await teacher.challenge("p", "r", history, ChallengeType.PRACTICAL)
        assert ch.content == "Why do you think so?"
        assert ch.challenge_type is ChallengeType.PRACTICAL
        assert "oldq" in created["tok"].prompts[0]
        # prompt had 4 tokens; only tokens after index 4 are decoded
        assert created["tok"].last_ids.tolist() == [4, 5, 6, 7, 8, 9]

    async def test_challenge_auto_selects_type(self, local_teacher_factory):
        make, created = local_teacher_factory
        teacher = make()
        ch = await teacher.challenge("p", "r")
        assert ch.challenge_type in teacher.preferred_challenges

    async def test_evaluate_extracts_score_and_improved(self, local_teacher_factory):
        make, created = local_teacher_factory
        teacher = make()
        created["tok"].decoded = "Score: 8.5\nGood.\nImproved version: a better answer"
        history = DialogueHistory("p", "r")
        history.add_turn(DialogueTurn(1, TeacherChallenge(ChallengeType.EDGE_CASE, "hq"), "ha"))
        ev = await teacher.evaluate("p", "r", history)
        assert ev.overall_score == 8.5
        assert ev.improved_response == "a better answer"
        assert ev.dimension_scores == []
        assert "hq" in created["tok"].prompts[0]
        assert "improved version" in created["tok"].prompts[0]

    async def test_evaluate_without_improved(self, local_teacher_factory):
        make, created = local_teacher_factory
        teacher = make()
        created["tok"].decoded = "Score: 3\nImproved: ignored"
        ev = await teacher.evaluate("p", "r", generate_improved=False)
        assert ev.improved_response is None
        assert ev.overall_score == 3.0
        assert "improved version" not in created["tok"].prompts[0]

    def test_extract_score_patterns_and_clamping(self, local_teacher_factory):
        make, _ = local_teacher_factory
        t = make()
        assert t._extract_score("score 7") == 7.0
        assert t._extract_score("I give it 6.5/10 overall") == 6.5
        assert t._extract_score("9 out of ten") == 9.0
        assert t._extract_score("Score: 42") == 10.0
        assert t._extract_score("no numbers here") is None

    def test_extract_improved_markers(self, local_teacher_factory):
        make, _ = local_teacher_factory
        t = make()
        assert t._extract_improved("blah\nBetter Response: new text ") == "new text"
        assert t._extract_improved("IMPROVED: x") == "x"
        assert t._extract_improved("nothing here") is None
