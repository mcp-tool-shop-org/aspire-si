"""
Tests for the ASPIRE teachers module.

Tests cover:
- Data structures: ChallengeType, EvaluationDimension, TeacherChallenge, etc.
- BaseTeacher abstract interface
- TeacherRegistry and registration system
- Persona teachers: Socratic, Scientific, Creative, Adversarial, Compassionate
- CompositeTeacher and CurriculumCompositeTeacher
"""

import os
from unittest.mock import AsyncMock, patch

import pytest

from aspire.teachers.base import (
    BaseTeacher,
    ChallengeType,
    DialogueHistory,
    DialogueTurn,
    DimensionScore,
    EvaluationDimension,
    TeacherChallenge,
    TeacherEvaluation,
)
from aspire.teachers.composite import CompositeTeacher, CurriculumCompositeTeacher
from aspire.teachers.personas import (
    AdversarialTeacher,
    CompassionateTeacher,
    CreativeTeacher,
    ScientificTeacher,
    SocraticTeacher,
    create_persona_teacher,
)
from aspire.teachers.registry import (
    TeacherRegistry,
    get_teacher,
    register_teacher,
)

# =============================================================================
# Data Structure Tests
# =============================================================================


class TestChallengeTypeEnum:
    """Tests for ChallengeType enum."""

    def test_challenge_type_all_values_accessible(self):
        """All ChallengeType values are accessible."""
        assert ChallengeType.PROBE_REASONING == "probe_reasoning"
        assert ChallengeType.EDGE_CASE == "edge_case"
        assert ChallengeType.DEVILS_ADVOCATE == "devils_advocate"
        assert ChallengeType.SOCRATIC == "socratic"
        assert ChallengeType.CLARIFICATION == "clarification"
        assert ChallengeType.EXTENSION == "extension"
        assert ChallengeType.CONTRADICTION == "contradiction"
        assert ChallengeType.STEELMAN == "steelman"
        assert ChallengeType.EMOTIONAL == "emotional"
        assert ChallengeType.PRACTICAL == "practical"
        assert ChallengeType.ETHICAL == "ethical"
        assert ChallengeType.CREATIVE == "creative"

    def test_challenge_type_is_str_enum(self):
        """ChallengeType values are strings."""
        for ct in ChallengeType:
            assert isinstance(ct.value, str)
            assert isinstance(ct, str)


class TestEvaluationDimensionEnum:
    """Tests for EvaluationDimension enum."""

    def test_evaluation_dimension_all_values_accessible(self):
        """All EvaluationDimension values are accessible."""
        assert EvaluationDimension.CORRECTNESS == "correctness"
        assert EvaluationDimension.REASONING == "reasoning"
        assert EvaluationDimension.NUANCE == "nuance"
        assert EvaluationDimension.ADAPTABILITY == "adaptability"
        assert EvaluationDimension.CLARITY == "clarity"
        assert EvaluationDimension.INTELLECTUAL_HONESTY == "intellectual_honesty"
        assert EvaluationDimension.CREATIVITY == "creativity"
        assert EvaluationDimension.EMPATHY == "empathy"
        assert EvaluationDimension.PRACTICALITY == "practicality"

    def test_evaluation_dimension_is_str_enum(self):
        """EvaluationDimension values are strings."""
        for ed in EvaluationDimension:
            assert isinstance(ed.value, str)
            assert isinstance(ed, str)


class TestTeacherChallenge:
    """Tests for TeacherChallenge dataclass."""

    def test_teacher_challenge_creation(self):
        """TeacherChallenge can be created with required fields."""
        challenge = TeacherChallenge(
            challenge_type=ChallengeType.PROBE_REASONING,
            content="Why do you think that?",
        )
        assert challenge.challenge_type == ChallengeType.PROBE_REASONING
        assert challenge.content == "Why do you think that?"

    def test_teacher_challenge_defaults(self):
        """TeacherChallenge has expected defaults."""
        challenge = TeacherChallenge(
            challenge_type=ChallengeType.EDGE_CASE,
            content="What about edge case X?",
        )
        assert challenge.context is None
        assert challenge.difficulty == 0.5
        assert challenge.metadata == {}

    def test_teacher_challenge_with_all_fields(self):
        """TeacherChallenge can be created with all fields."""
        challenge = TeacherChallenge(
            challenge_type=ChallengeType.SOCRATIC,
            content="What assumption are you making?",
            context="Testing assumptions",
            difficulty=0.8,
            metadata={"turn": 1},
        )
        assert challenge.context == "Testing assumptions"
        assert challenge.difficulty == 0.8
        assert challenge.metadata == {"turn": 1}


class TestDimensionScore:
    """Tests for DimensionScore dataclass."""

    def test_dimension_score_creation(self):
        """DimensionScore can be created with all fields."""
        score = DimensionScore(
            dimension=EvaluationDimension.CORRECTNESS,
            score=8.5,
            explanation="Answer is factually accurate",
        )
        assert score.dimension == EvaluationDimension.CORRECTNESS
        assert score.score == 8.5
        assert score.explanation == "Answer is factually accurate"


class TestTeacherEvaluation:
    """Tests for TeacherEvaluation dataclass."""

    def test_teacher_evaluation_creation(self):
        """TeacherEvaluation can be created with required fields."""
        dim_scores = [
            DimensionScore(EvaluationDimension.CORRECTNESS, 7.5, "Good accuracy")
        ]
        evaluation = TeacherEvaluation(
            overall_score=7.5,
            dimension_scores=dim_scores,
            reasoning="Solid response overall",
        )
        assert evaluation.overall_score == 7.5
        assert len(evaluation.dimension_scores) == 1
        assert evaluation.reasoning == "Solid response overall"

    def test_teacher_evaluation_defaults(self):
        """TeacherEvaluation has expected defaults."""
        evaluation = TeacherEvaluation(
            overall_score=6.0,
            dimension_scores=[],
            reasoning="Acceptable",
        )
        assert evaluation.improved_response is None
        assert evaluation.strengths == []
        assert evaluation.weaknesses == []
        assert evaluation.suggestions == []
        assert evaluation.metadata == {}

    def test_teacher_evaluation_passed_above_threshold(self):
        """TeacherEvaluation.passed returns True for score >= 6.0."""
        evaluation = TeacherEvaluation(
            overall_score=6.0,
            dimension_scores=[],
            reasoning="Passing",
        )
        assert evaluation.passed is True

        evaluation_high = TeacherEvaluation(
            overall_score=9.5,
            dimension_scores=[],
            reasoning="Excellent",
        )
        assert evaluation_high.passed is True

    def test_teacher_evaluation_passed_below_threshold(self):
        """TeacherEvaluation.passed returns False for score < 6.0."""
        evaluation = TeacherEvaluation(
            overall_score=5.9,
            dimension_scores=[],
            reasoning="Needs improvement",
        )
        assert evaluation.passed is False

        evaluation_low = TeacherEvaluation(
            overall_score=2.0,
            dimension_scores=[],
            reasoning="Poor",
        )
        assert evaluation_low.passed is False

    def test_teacher_evaluation_to_dict(self):
        """TeacherEvaluation.to_dict serializes all fields correctly."""
        dim_scores = [
            DimensionScore(EvaluationDimension.CORRECTNESS, 8.0, "Good"),
            DimensionScore(EvaluationDimension.REASONING, 7.5, "Clear"),
        ]
        evaluation = TeacherEvaluation(
            overall_score=7.75,
            dimension_scores=dim_scores,
            reasoning="Well-reasoned response",
            improved_response="An even better answer would be...",
            strengths=["Clear structure", "Good examples"],
            weaknesses=["Missing edge case"],
            suggestions=["Consider X"],
            metadata={"teacher": "test"},
        )

        result = evaluation.to_dict()

        assert result["overall_score"] == 7.75
        assert result["reasoning"] == "Well-reasoned response"
        assert result["improved_response"] == "An even better answer would be..."
        assert result["strengths"] == ["Clear structure", "Good examples"]
        assert result["weaknesses"] == ["Missing edge case"]
        assert result["suggestions"] == ["Consider X"]
        assert result["metadata"] == {"teacher": "test"}

        # Check dimension_scores are converted properly
        assert len(result["dimension_scores"]) == 2
        assert result["dimension_scores"][0]["dimension"] == "correctness"
        assert result["dimension_scores"][0]["score"] == 8.0
        assert result["dimension_scores"][0]["explanation"] == "Good"


class TestDialogueTurn:
    """Tests for DialogueTurn dataclass."""

    def test_dialogue_turn_creation(self):
        """DialogueTurn can be created with required fields."""
        challenge = TeacherChallenge(
            challenge_type=ChallengeType.PROBE_REASONING,
            content="Why?",
        )
        turn = DialogueTurn(
            turn_number=1,
            challenge=challenge,
            student_response="Because of X",
        )
        assert turn.turn_number == 1
        assert turn.challenge == challenge
        assert turn.student_response == "Because of X"

    def test_dialogue_turn_evaluation_optional(self):
        """DialogueTurn.evaluation is optional."""
        challenge = TeacherChallenge(ChallengeType.EDGE_CASE, "What if?")
        turn = DialogueTurn(
            turn_number=2,
            challenge=challenge,
            student_response="Then Y would happen",
        )
        assert turn.evaluation is None
        assert turn.timestamp is None

    def test_dialogue_turn_with_evaluation(self):
        """DialogueTurn can include evaluation."""
        challenge = TeacherChallenge(ChallengeType.CLARIFICATION, "Clarify?")
        evaluation = TeacherEvaluation(6.5, [], "Good clarification")
        turn = DialogueTurn(
            turn_number=3,
            challenge=challenge,
            student_response="I mean Z",
            evaluation=evaluation,
            timestamp=1234567890.0,
        )
        assert turn.evaluation == evaluation
        assert turn.timestamp == 1234567890.0


class TestDialogueHistory:
    """Tests for DialogueHistory dataclass."""

    def test_dialogue_history_creation(self):
        """DialogueHistory can be created with required fields."""
        history = DialogueHistory(
            prompt="Explain quantum computing",
            initial_response="Quantum computing uses qubits...",
        )
        assert history.prompt == "Explain quantum computing"
        assert history.initial_response == "Quantum computing uses qubits..."

    def test_dialogue_history_turns_empty_by_default(self):
        """DialogueHistory.turns is empty by default."""
        history = DialogueHistory(prompt="Test", initial_response="Response")
        assert history.turns == []
        assert history.final_evaluation is None

    def test_dialogue_history_add_turn(self):
        """DialogueHistory.add_turn appends turn to turns list."""
        history = DialogueHistory(prompt="Test", initial_response="Response")
        challenge = TeacherChallenge(ChallengeType.PROBE_REASONING, "Why?")
        turn = DialogueTurn(turn_number=1, challenge=challenge, student_response="Because")

        history.add_turn(turn)

        assert len(history.turns) == 1
        assert history.turns[0] == turn

    def test_dialogue_history_num_turns(self):
        """DialogueHistory.num_turns returns correct count."""
        history = DialogueHistory(prompt="Test", initial_response="Response")
        assert history.num_turns == 0

        for i in range(3):
            challenge = TeacherChallenge(ChallengeType.EDGE_CASE, f"Case {i}")
            turn = DialogueTurn(turn_number=i + 1, challenge=challenge, student_response=f"R{i}")
            history.add_turn(turn)

        assert history.num_turns == 3

    def test_dialogue_history_get_trajectory_scores(self):
        """DialogueHistory.get_trajectory_scores returns scores from evaluated turns."""
        history = DialogueHistory(prompt="Test", initial_response="Response")

        # Add 3 turns with evaluations
        for i, score in enumerate([5.0, 6.5, 8.0]):
            challenge = TeacherChallenge(ChallengeType.SOCRATIC, f"Q{i}")
            evaluation = TeacherEvaluation(score, [], f"Score {score}")
            turn = DialogueTurn(
                turn_number=i + 1,
                challenge=challenge,
                student_response=f"A{i}",
                evaluation=evaluation,
            )
            history.add_turn(turn)

        scores = history.get_trajectory_scores()
        assert scores == [5.0, 6.5, 8.0]

    def test_dialogue_history_get_trajectory_scores_skips_unevaluated(self):
        """get_trajectory_scores skips turns without evaluation."""
        history = DialogueHistory(prompt="Test", initial_response="Response")

        # Turn 1: with evaluation
        c1 = TeacherChallenge(ChallengeType.PROBE_REASONING, "Q1")
        t1 = DialogueTurn(
            turn_number=1,
            challenge=c1,
            student_response="A1",
            evaluation=TeacherEvaluation(7.0, [], "Good"),
        )
        history.add_turn(t1)

        # Turn 2: without evaluation
        c2 = TeacherChallenge(ChallengeType.EDGE_CASE, "Q2")
        t2 = DialogueTurn(turn_number=2, challenge=c2, student_response="A2")
        history.add_turn(t2)

        # Turn 3: with evaluation
        c3 = TeacherChallenge(ChallengeType.STEELMAN, "Q3")
        t3 = DialogueTurn(
            turn_number=3,
            challenge=c3,
            student_response="A3",
            evaluation=TeacherEvaluation(8.5, [], "Better"),
        )
        history.add_turn(t3)

        scores = history.get_trajectory_scores()
        assert scores == [7.0, 8.5]


# =============================================================================
# BaseTeacher Tests
# =============================================================================


class ConcreteTestTeacher(BaseTeacher):
    """Concrete implementation for testing BaseTeacher."""

    async def challenge(
        self,
        prompt: str,
        student_response: str,
        dialogue_history: DialogueHistory | None = None,
        challenge_type: ChallengeType | None = None,
    ) -> TeacherChallenge:
        ct = challenge_type or ChallengeType.PROBE_REASONING
        return TeacherChallenge(
            challenge_type=ct,
            content=f"Challenge for: {student_response[:20]}",
        )

    async def evaluate(
        self,
        prompt: str,
        student_response: str,
        dialogue_history: DialogueHistory | None = None,
        generate_improved: bool = True,
    ) -> TeacherEvaluation:
        improved = "Improved version" if generate_improved else None
        return TeacherEvaluation(
            overall_score=7.0,
            dimension_scores=[],
            reasoning="Test evaluation",
            improved_response=improved,
        )


class TestBaseTeacher:
    """Tests for BaseTeacher abstract class."""

    def test_base_teacher_is_abstract(self):
        """BaseTeacher cannot be instantiated directly."""
        with pytest.raises(TypeError, match="abstract"):
            BaseTeacher()

    def test_base_teacher_init(self):
        """BaseTeacher.__init__ stores all parameters."""
        teacher = ConcreteTestTeacher(
            name="Test Teacher",
            description="A test teacher",
            temperature=0.5,
            max_tokens=512,
        )
        assert teacher.name == "Test Teacher"
        assert teacher.description == "A test teacher"
        assert teacher.temperature == 0.5
        assert teacher.max_tokens == 512

    def test_base_teacher_init_defaults(self):
        """BaseTeacher has expected default values."""
        teacher = ConcreteTestTeacher()
        assert teacher.name == "BaseTeacher"
        assert teacher.description == "A teacher for ASPIRE training"
        assert teacher.temperature == 0.7
        assert teacher.max_tokens == 1024

    def test_base_teacher_evaluation_dimensions_default(self):
        """evaluation_dimensions defaults to all dimensions."""
        teacher = ConcreteTestTeacher()
        assert set(teacher.evaluation_dimensions) == set(EvaluationDimension)

    def test_base_teacher_preferred_challenges_default(self):
        """preferred_challenges defaults to all challenge types."""
        teacher = ConcreteTestTeacher()
        assert set(teacher.preferred_challenges) == set(ChallengeType)

    def test_base_teacher_custom_dimensions(self):
        """BaseTeacher accepts custom evaluation_dimensions."""
        custom_dims = [EvaluationDimension.CORRECTNESS, EvaluationDimension.REASONING]
        teacher = ConcreteTestTeacher(evaluation_dimensions=custom_dims)
        assert teacher.evaluation_dimensions == custom_dims

    def test_base_teacher_custom_challenges(self):
        """BaseTeacher accepts custom preferred_challenges."""
        custom_challenges = [ChallengeType.SOCRATIC, ChallengeType.PROBE_REASONING]
        teacher = ConcreteTestTeacher(preferred_challenges=custom_challenges)
        assert teacher.preferred_challenges == custom_challenges

    @pytest.mark.asyncio
    async def test_base_teacher_run_dialogue(self):
        """run_dialogue executes multi-turn dialogue."""
        teacher = ConcreteTestTeacher()

        async def mock_generate(prompt, challenge, dialogue_history):
            return f"Student reply to: {challenge[:20]}"

        history = await teacher.run_dialogue(
            prompt="Test prompt",
            student_response="Initial response",
            student_generate_fn=mock_generate,
            max_turns=3,
        )

        assert history.num_turns == 3
        assert history.final_evaluation is not None
        assert history.final_evaluation.improved_response == "Improved version"

    @pytest.mark.asyncio
    async def test_base_teacher_run_dialogue_with_turn_eval(self):
        """run_dialogue evaluates each turn when evaluate_each_turn=True."""
        teacher = ConcreteTestTeacher()

        async def mock_generate(prompt, challenge, dialogue_history):
            return "Student reply"

        history = await teacher.run_dialogue(
            prompt="Test prompt",
            student_response="Initial response",
            student_generate_fn=mock_generate,
            max_turns=2,
            evaluate_each_turn=True,
        )

        assert history.num_turns == 2
        for turn in history.turns:
            assert turn.evaluation is not None
            assert turn.evaluation.overall_score == 7.0

    def test_base_teacher_select_challenge_type(self):
        """select_challenge_type returns one of preferred_challenges."""
        teacher = ConcreteTestTeacher(
            preferred_challenges=[ChallengeType.SOCRATIC, ChallengeType.EDGE_CASE]
        )
        for _ in range(10):
            selected = teacher.select_challenge_type()
            assert selected in teacher.preferred_challenges

    def test_base_teacher_get_system_prompt(self):
        """get_system_prompt includes name and description."""
        teacher = ConcreteTestTeacher(
            name="Test Teacher",
            description="A helpful test teacher",
        )
        prompt = teacher.get_system_prompt()
        assert "Test Teacher" in prompt
        assert "A helpful test teacher" in prompt

    def test_base_teacher_repr(self):
        """__repr__ returns expected format."""
        teacher = ConcreteTestTeacher(name="Test Teacher")
        assert repr(teacher) == "ConcreteTestTeacher(name='Test Teacher')"


# =============================================================================
# TeacherRegistry Tests
# =============================================================================


class TestTeacherRegistry:
    """Tests for TeacherRegistry class."""

    def test_registry_register(self):
        """TeacherRegistry.register stores teacher class."""
        # Use unique name to avoid conflicts
        TeacherRegistry.register("test_unique_123", ConcreteTestTeacher)
        assert TeacherRegistry.get("test_unique_123") == ConcreteTestTeacher

    def test_registry_register_lowercase(self):
        """TeacherRegistry.register stores names as lowercase."""
        TeacherRegistry.register("TestCamelCase456", ConcreteTestTeacher)
        assert TeacherRegistry.get("testcamelcase456") == ConcreteTestTeacher

    def test_registry_get(self):
        """TeacherRegistry.get retrieves registered class."""
        TeacherRegistry.register("test_get_789", ConcreteTestTeacher)
        retrieved = TeacherRegistry.get("test_get_789")
        assert retrieved == ConcreteTestTeacher

    def test_registry_get_case_insensitive(self):
        """TeacherRegistry.get is case-insensitive."""
        TeacherRegistry.register("TestCase101", ConcreteTestTeacher)
        assert TeacherRegistry.get("TESTCASE101") == ConcreteTestTeacher
        assert TeacherRegistry.get("testcase101") == ConcreteTestTeacher

    def test_registry_get_missing(self):
        """TeacherRegistry.get returns None for unknown names."""
        result = TeacherRegistry.get("nonexistent_teacher_xyz")
        assert result is None

    def test_registry_list(self):
        """TeacherRegistry.list returns all registered names."""
        TeacherRegistry.register("list_test_a", ConcreteTestTeacher)
        TeacherRegistry.register("list_test_b", ConcreteTestTeacher)
        names = TeacherRegistry.list()
        assert "list_test_a" in names
        assert "list_test_b" in names

    def test_registry_create(self):
        """TeacherRegistry.create instantiates teacher with kwargs."""
        TeacherRegistry.register("create_test_202", ConcreteTestTeacher)
        instance = TeacherRegistry.create("create_test_202", temperature=0.5)
        assert isinstance(instance, ConcreteTestTeacher)
        assert instance.temperature == 0.5

    def test_registry_create_unknown(self):
        """TeacherRegistry.create raises ValueError for unknown name."""
        with pytest.raises(ValueError, match="Unknown teacher.*Available"):
            TeacherRegistry.create("unknown_teacher_xyz")


class TestRegisterTeacherDecorator:
    """Tests for @register_teacher decorator."""

    def test_register_teacher_decorator(self):
        """@register_teacher decorator registers class."""

        @register_teacher("decorated_test_303")
        class DecoratedTeacher(ConcreteTestTeacher):
            pass

        assert TeacherRegistry.get("decorated_test_303") == DecoratedTeacher


class TestGetTeacherFunction:
    """Tests for get_teacher function."""

    def test_get_teacher_function(self):
        """get_teacher returns instance by name."""
        TeacherRegistry.register("func_test_404", ConcreteTestTeacher)
        instance = get_teacher("func_test_404", max_tokens=256)
        assert isinstance(instance, ConcreteTestTeacher)
        assert instance.max_tokens == 256


class TestBuiltinTeachers:
    """Tests for built-in teacher registrations."""

    def test_claude_teacher_registered(self):
        """'claude' teacher is registered."""
        assert TeacherRegistry.get("claude") is not None

    def test_openai_teacher_registered(self):
        """'openai' teacher is registered."""
        assert TeacherRegistry.get("openai") is not None

    def test_gpt4_alias_registered(self):
        """'gpt4' is an alias for OpenAI teacher."""
        assert TeacherRegistry.get("gpt4") is not None
        assert TeacherRegistry.get("gpt4") == TeacherRegistry.get("openai")

    def test_local_teacher_registered(self):
        """'local' teacher is registered."""
        assert TeacherRegistry.get("local") is not None

    def test_persona_teachers_registered(self):
        """All persona teachers are registered."""
        personas = ["socratic", "scientific", "creative", "adversarial", "compassionate"]
        for persona in personas:
            assert TeacherRegistry.get(persona) is not None, f"{persona} not registered"

    def test_persona_aliases_registered(self):
        """Persona aliases are registered."""
        aliases = {
            "socrates": "socratic",
            "scientist": "scientific",
            "innovator": "creative",
            "challenger": "adversarial",
            "guide": "compassionate",
        }
        for alias, main in aliases.items():
            assert TeacherRegistry.get(alias) == TeacherRegistry.get(main)


# =============================================================================
# Persona Teachers Tests
# =============================================================================


class TestSocraticTeacher:
    """Tests for SocraticTeacher."""

    def test_socratic_teacher_init(self):
        """SocraticTeacher initializes with correct name."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            teacher = SocraticTeacher()
            assert teacher.name == "Socrates"

    def test_socratic_teacher_preferred_challenges(self):
        """SocraticTeacher has SOCRATIC in preferred_challenges."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            teacher = SocraticTeacher()
            assert ChallengeType.SOCRATIC in teacher.preferred_challenges
            assert ChallengeType.PROBE_REASONING in teacher.preferred_challenges
            assert ChallengeType.CLARIFICATION in teacher.preferred_challenges
            assert ChallengeType.STEELMAN in teacher.preferred_challenges

    def test_socratic_teacher_system_prompt(self):
        """SocraticTeacher system prompt emphasizes questions."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            teacher = SocraticTeacher()
            prompt = teacher.get_system_prompt()
            assert "questions" in prompt.lower() or "question" in prompt.lower()
            assert "NEVER give answers directly" in prompt or "never" in prompt.lower()


class TestScientificTeacher:
    """Tests for ScientificTeacher."""

    def test_scientific_teacher_init(self):
        """ScientificTeacher initializes with correct name."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            teacher = ScientificTeacher()
            assert teacher.name == "Dr. Empirica"

    def test_scientific_teacher_preferred_challenges(self):
        """ScientificTeacher has expected challenge types."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            teacher = ScientificTeacher()
            assert ChallengeType.EDGE_CASE in teacher.preferred_challenges
            assert ChallengeType.PROBE_REASONING in teacher.preferred_challenges
            assert ChallengeType.CONTRADICTION in teacher.preferred_challenges
            assert ChallengeType.PRACTICAL in teacher.preferred_challenges

    def test_scientific_teacher_system_prompt(self):
        """ScientificTeacher system prompt emphasizes evidence."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            teacher = ScientificTeacher()
            prompt = teacher.get_system_prompt()
            assert "evidence" in prompt.lower()
            assert "falsif" in prompt.lower()  # falsifiable/falsifiability


class TestCreativeTeacher:
    """Tests for CreativeTeacher."""

    def test_creative_teacher_init(self):
        """CreativeTeacher initializes with correct name."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            teacher = CreativeTeacher()
            assert teacher.name == "The Innovator"

    def test_creative_teacher_preferred_challenges(self):
        """CreativeTeacher has CREATIVE in preferred_challenges."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            teacher = CreativeTeacher()
            assert ChallengeType.CREATIVE in teacher.preferred_challenges
            assert ChallengeType.EXTENSION in teacher.preferred_challenges

    def test_creative_teacher_system_prompt(self):
        """CreativeTeacher system prompt emphasizes novel ideas."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            teacher = CreativeTeacher()
            prompt = teacher.get_system_prompt()
            assert "novel" in prompt.lower() or "creative" in prompt.lower()
            assert "what else" in prompt.lower() or "what if" in prompt.lower()


class TestAdversarialTeacher:
    """Tests for AdversarialTeacher."""

    def test_adversarial_teacher_init(self):
        """AdversarialTeacher initializes with correct name."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            teacher = AdversarialTeacher()
            assert teacher.name == "The Challenger"

    def test_adversarial_teacher_preferred_challenges(self):
        """AdversarialTeacher has DEVILS_ADVOCATE and STEELMAN."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            teacher = AdversarialTeacher()
            assert ChallengeType.DEVILS_ADVOCATE in teacher.preferred_challenges
            assert ChallengeType.STEELMAN in teacher.preferred_challenges
            assert ChallengeType.CONTRADICTION in teacher.preferred_challenges
            assert ChallengeType.EDGE_CASE in teacher.preferred_challenges

    def test_adversarial_teacher_system_prompt(self):
        """AdversarialTeacher system prompt emphasizes opposition."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            teacher = AdversarialTeacher()
            prompt = teacher.get_system_prompt()
            assert "opposite" in prompt.lower()
            assert "weakness" in prompt.lower()


class TestCompassionateTeacher:
    """Tests for CompassionateTeacher."""

    def test_compassionate_teacher_init(self):
        """CompassionateTeacher initializes with correct name."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            teacher = CompassionateTeacher()
            assert teacher.name == "The Guide"

    def test_compassionate_teacher_preferred_challenges(self):
        """CompassionateTeacher has EMOTIONAL and ETHICAL."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            teacher = CompassionateTeacher()
            assert ChallengeType.EMOTIONAL in teacher.preferred_challenges
            assert ChallengeType.ETHICAL in teacher.preferred_challenges
            assert ChallengeType.PRACTICAL in teacher.preferred_challenges
            assert ChallengeType.CLARIFICATION in teacher.preferred_challenges

    def test_compassionate_teacher_system_prompt(self):
        """CompassionateTeacher system prompt emphasizes human impact."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            teacher = CompassionateTeacher()
            prompt = teacher.get_system_prompt()
            assert "human impact" in prompt.lower() or "human" in prompt.lower()
            assert "encourage" in prompt.lower() or "support" in prompt.lower()


class TestCreatePersonaTeacher:
    """Tests for create_persona_teacher factory function."""

    def test_create_persona_teacher_socratic(self):
        """create_persona_teacher returns SocraticTeacher."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            teacher = create_persona_teacher("socratic")
            assert isinstance(teacher, SocraticTeacher)

    def test_create_persona_teacher_scientific(self):
        """create_persona_teacher returns ScientificTeacher."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            teacher = create_persona_teacher("scientific")
            assert isinstance(teacher, ScientificTeacher)

    def test_create_persona_teacher_creative(self):
        """create_persona_teacher returns CreativeTeacher."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            teacher = create_persona_teacher("creative")
            assert isinstance(teacher, CreativeTeacher)

    def test_create_persona_teacher_adversarial(self):
        """create_persona_teacher returns AdversarialTeacher."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            teacher = create_persona_teacher("adversarial")
            assert isinstance(teacher, AdversarialTeacher)

    def test_create_persona_teacher_compassionate(self):
        """create_persona_teacher returns CompassionateTeacher."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            teacher = create_persona_teacher("compassionate")
            assert isinstance(teacher, CompassionateTeacher)

    def test_create_persona_teacher_case_insensitive(self):
        """create_persona_teacher is case-insensitive."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            teacher = create_persona_teacher("SOCRATIC")
            assert isinstance(teacher, SocraticTeacher)

    def test_create_persona_teacher_invalid(self):
        """create_persona_teacher raises ValueError for invalid persona."""
        with pytest.raises(ValueError, match="Unknown persona.*Available"):
            create_persona_teacher("invalid_persona")


# =============================================================================
# CompositeTeacher Tests
# =============================================================================


class TestCompositeTeacher:
    """Tests for CompositeTeacher."""

    def test_composite_teacher_init(self):
        """CompositeTeacher initializes with teacher list."""
        t1 = ConcreteTestTeacher(name="Teacher1")
        t2 = ConcreteTestTeacher(name="Teacher2")
        composite = CompositeTeacher([t1, t2])
        assert len(composite.teachers) == 2
        assert composite.teachers[0] == t1
        assert composite.teachers[1] == t2

    def test_composite_teacher_init_default_weights(self):
        """CompositeTeacher defaults to equal weights."""
        t1 = ConcreteTestTeacher()
        t2 = ConcreteTestTeacher()
        composite = CompositeTeacher([t1, t2])
        assert composite.weights == [0.5, 0.5]

    def test_composite_teacher_init_custom_weights(self):
        """CompositeTeacher accepts custom weights."""
        t1 = ConcreteTestTeacher()
        t2 = ConcreteTestTeacher()
        composite = CompositeTeacher([t1, t2], weights=[0.7, 0.3])
        assert composite.weights == [0.7, 0.3]

    def test_composite_teacher_init_empty_raises(self):
        """CompositeTeacher raises ValueError for empty teacher list."""
        with pytest.raises(ValueError, match="at least one teacher"):
            CompositeTeacher([])

    def test_composite_teacher_init_mismatched_weights_raises(self):
        """CompositeTeacher raises ValueError when weights don't match teachers."""
        t1 = ConcreteTestTeacher()
        t2 = ConcreteTestTeacher()
        with pytest.raises(ValueError, match="Weights must match"):
            CompositeTeacher([t1, t2], weights=[0.5])

    def test_composite_teacher_merges_challenges(self):
        """CompositeTeacher merges preferred_challenges from all teachers."""
        t1 = ConcreteTestTeacher(
            preferred_challenges=[ChallengeType.SOCRATIC, ChallengeType.EDGE_CASE]
        )
        t2 = ConcreteTestTeacher(
            preferred_challenges=[ChallengeType.EDGE_CASE, ChallengeType.CREATIVE]
        )
        composite = CompositeTeacher([t1, t2])

        # Should have union of all challenges
        assert ChallengeType.SOCRATIC in composite.preferred_challenges
        assert ChallengeType.EDGE_CASE in composite.preferred_challenges
        assert ChallengeType.CREATIVE in composite.preferred_challenges

    def test_composite_teacher_merges_dimensions(self):
        """CompositeTeacher merges evaluation_dimensions from all teachers."""
        t1 = ConcreteTestTeacher(
            evaluation_dimensions=[EvaluationDimension.CORRECTNESS, EvaluationDimension.REASONING]
        )
        t2 = ConcreteTestTeacher(
            evaluation_dimensions=[EvaluationDimension.REASONING, EvaluationDimension.CREATIVITY]
        )
        composite = CompositeTeacher([t1, t2])

        # Should have union of all dimensions
        assert EvaluationDimension.CORRECTNESS in composite.evaluation_dimensions
        assert EvaluationDimension.REASONING in composite.evaluation_dimensions
        assert EvaluationDimension.CREATIVITY in composite.evaluation_dimensions

    @pytest.mark.asyncio
    async def test_composite_challenge_rotate(self):
        """CompositeTeacher with rotate strategy cycles through teachers."""
        t1 = ConcreteTestTeacher(name="T1")
        t2 = ConcreteTestTeacher(name="T2")
        t3 = ConcreteTestTeacher(name="T3")

        t1.challenge = AsyncMock(return_value=TeacherChallenge(ChallengeType.SOCRATIC, "Q1"))
        t2.challenge = AsyncMock(return_value=TeacherChallenge(ChallengeType.EDGE_CASE, "Q2"))
        t3.challenge = AsyncMock(return_value=TeacherChallenge(ChallengeType.CREATIVE, "Q3"))

        composite = CompositeTeacher([t1, t2, t3], strategy="rotate")

        # First call uses t1
        c1 = await composite.challenge("prompt", "response")
        assert c1.content == "Q1"
        t1.challenge.assert_called_once()

        # Second call uses t2
        c2 = await composite.challenge("prompt", "response")
        assert c2.content == "Q2"
        t2.challenge.assert_called_once()

        # Third call uses t3
        c3 = await composite.challenge("prompt", "response")
        assert c3.content == "Q3"
        t3.challenge.assert_called_once()

        # Fourth call cycles back to t1
        c4 = await composite.challenge("prompt", "response")
        assert c4.content == "Q1"
        assert t1.challenge.call_count == 2

    @pytest.mark.asyncio
    async def test_composite_challenge_specialize(self):
        """CompositeTeacher with specialize strategy picks matching teacher."""
        t1 = ConcreteTestTeacher(
            name="Socratic",
            preferred_challenges=[ChallengeType.SOCRATIC],
        )
        t2 = ConcreteTestTeacher(
            name="Creative",
            preferred_challenges=[ChallengeType.CREATIVE],
        )

        t1.challenge = AsyncMock(return_value=TeacherChallenge(ChallengeType.SOCRATIC, "Socratic Q"))
        t2.challenge = AsyncMock(return_value=TeacherChallenge(ChallengeType.CREATIVE, "Creative Q"))

        composite = CompositeTeacher([t1, t2], strategy="specialize")

        # Ask for creative challenge - should use t2
        challenge = await composite.challenge(
            "prompt", "response", challenge_type=ChallengeType.CREATIVE
        )
        assert challenge.content == "Creative Q"
        t2.challenge.assert_called_once()
        t1.challenge.assert_not_called()

    @pytest.mark.asyncio
    async def test_composite_challenge_random(self):
        """CompositeTeacher with random strategy uses weighted selection."""
        t1 = ConcreteTestTeacher(name="T1")
        t2 = ConcreteTestTeacher(name="T2")

        t1.challenge = AsyncMock(return_value=TeacherChallenge(ChallengeType.SOCRATIC, "Q1"))
        t2.challenge = AsyncMock(return_value=TeacherChallenge(ChallengeType.EDGE_CASE, "Q2"))

        composite = CompositeTeacher([t1, t2], strategy="random", weights=[0.9, 0.1])

        with patch("random.choices", return_value=[t2]):
            challenge = await composite.challenge("prompt", "response")
            assert challenge.content == "Q2"
            t2.challenge.assert_called_once()

    @pytest.mark.asyncio
    async def test_composite_evaluate_vote(self):
        """CompositeTeacher with vote strategy combines weighted scores."""
        t1 = ConcreteTestTeacher(name="T1")
        t2 = ConcreteTestTeacher(name="T2")

        eval1 = TeacherEvaluation(
            overall_score=8.0,
            dimension_scores=[
                DimensionScore(EvaluationDimension.CORRECTNESS, 8.0, "Good")
            ],
            reasoning="T1 reasoning",
            improved_response="Improved by T1",
            strengths=["Clear"],
            weaknesses=["Verbose"],
            suggestions=["Be concise"],
        )
        eval2 = TeacherEvaluation(
            overall_score=6.0,
            dimension_scores=[
                DimensionScore(EvaluationDimension.CORRECTNESS, 6.0, "OK")
            ],
            reasoning="T2 reasoning",
            improved_response="Improved by T2",
            strengths=["Detailed"],
            weaknesses=["Complex"],
            suggestions=["Simplify"],
        )

        t1.evaluate = AsyncMock(return_value=eval1)
        t2.evaluate = AsyncMock(return_value=eval2)

        composite = CompositeTeacher([t1, t2], strategy="vote", weights=[0.5, 0.5])

        result = await composite.evaluate("prompt", "response")

        # Weighted average: (8.0 * 0.5 + 6.0 * 0.5) / 1.0 = 7.0
        assert result.overall_score == 7.0
        assert "[T1]" in result.reasoning
        assert "[T2]" in result.reasoning

    @pytest.mark.asyncio
    async def test_composite_evaluate_vote_combines_dimensions(self):
        """Vote strategy combines dimension scores from all teachers."""
        t1 = ConcreteTestTeacher(name="T1")
        t2 = ConcreteTestTeacher(name="T2")

        eval1 = TeacherEvaluation(
            overall_score=8.0,
            dimension_scores=[
                DimensionScore(EvaluationDimension.CORRECTNESS, 8.0, "Good"),
                DimensionScore(EvaluationDimension.REASONING, 7.0, "Solid"),
            ],
            reasoning="T1",
        )
        eval2 = TeacherEvaluation(
            overall_score=6.0,
            dimension_scores=[
                DimensionScore(EvaluationDimension.CORRECTNESS, 6.0, "OK"),
            ],
            reasoning="T2",
        )

        t1.evaluate = AsyncMock(return_value=eval1)
        t2.evaluate = AsyncMock(return_value=eval2)

        composite = CompositeTeacher([t1, t2], strategy="vote", weights=[0.5, 0.5])

        result = await composite.evaluate("prompt", "response")

        # Should have combined dimension scores
        dim_map = {ds.dimension: ds.score for ds in result.dimension_scores}
        # CORRECTNESS: (8.0 * 0.5 + 6.0 * 0.5) / 1.0 = 7.0
        assert dim_map[EvaluationDimension.CORRECTNESS] == 7.0
        # REASONING only from T1: 7.0
        assert dim_map[EvaluationDimension.REASONING] == 7.0

    @pytest.mark.asyncio
    async def test_composite_evaluate_vote_combines_feedback(self):
        """Vote strategy merges strengths, weaknesses, suggestions."""
        t1 = ConcreteTestTeacher(name="T1")
        t2 = ConcreteTestTeacher(name="T2")

        eval1 = TeacherEvaluation(
            overall_score=7.0,
            dimension_scores=[],
            reasoning="T1 feedback",
            strengths=["A", "B"],
            weaknesses=["X"],
            suggestions=["Do Y"],
        )
        eval2 = TeacherEvaluation(
            overall_score=7.0,
            dimension_scores=[],
            reasoning="T2 feedback",
            strengths=["B", "C"],
            weaknesses=["Z"],
            suggestions=["Do W"],
        )

        t1.evaluate = AsyncMock(return_value=eval1)
        t2.evaluate = AsyncMock(return_value=eval2)

        composite = CompositeTeacher([t1, t2], strategy="vote")

        result = await composite.evaluate("prompt", "response")

        # Should have merged (deduped) lists
        assert "A" in result.strengths
        assert "B" in result.strengths
        assert "C" in result.strengths
        assert "X" in result.weaknesses
        assert "Z" in result.weaknesses
        assert "Do Y" in result.suggestions
        assert "Do W" in result.suggestions
        # All reasoning combined
        assert "T1 feedback" in result.reasoning
        assert "T2 feedback" in result.reasoning

    @pytest.mark.asyncio
    async def test_composite_evaluate_vote_picks_best_improved(self):
        """Vote strategy picks improved_response from highest-scoring teacher."""
        t1 = ConcreteTestTeacher(name="T1")
        t2 = ConcreteTestTeacher(name="T2")

        eval1 = TeacherEvaluation(
            overall_score=6.0,
            dimension_scores=[],
            reasoning="Lower",
            improved_response="T1 improved",
        )
        eval2 = TeacherEvaluation(
            overall_score=9.0,
            dimension_scores=[],
            reasoning="Higher",
            improved_response="T2 improved",
        )

        t1.evaluate = AsyncMock(return_value=eval1)
        t2.evaluate = AsyncMock(return_value=eval2)

        composite = CompositeTeacher([t1, t2], strategy="vote")

        result = await composite.evaluate("prompt", "response", generate_improved=True)

        # Should pick improved from T2 (higher score)
        assert result.improved_response == "T2 improved"

    @pytest.mark.asyncio
    async def test_composite_evaluate_rotate(self):
        """CompositeTeacher with rotate uses current teacher for evaluation."""
        t1 = ConcreteTestTeacher(name="T1")
        t2 = ConcreteTestTeacher(name="T2")

        eval1 = TeacherEvaluation(7.0, [], "T1 eval")
        eval2 = TeacherEvaluation(8.0, [], "T2 eval")

        t1.evaluate = AsyncMock(return_value=eval1)
        t2.evaluate = AsyncMock(return_value=eval2)

        composite = CompositeTeacher([t1, t2], strategy="rotate")

        # Current index is 0, so should use t1
        result = await composite.evaluate("prompt", "response")
        assert result.reasoning == "T1 eval"
        t1.evaluate.assert_called_once()
        t2.evaluate.assert_not_called()

    def test_composite_get_system_prompt(self):
        """CompositeTeacher system prompt lists all teacher names."""
        t1 = ConcreteTestTeacher(name="Teacher Alpha")
        t2 = ConcreteTestTeacher(name="Teacher Beta")
        composite = CompositeTeacher([t1, t2])

        prompt = composite.get_system_prompt()
        assert "Teacher Alpha" in prompt
        assert "Teacher Beta" in prompt


class TestCurriculumCompositeTeacher:
    """Tests for CurriculumCompositeTeacher."""

    def test_curriculum_composite_teacher_init(self):
        """CurriculumCompositeTeacher initializes with stage_weights."""
        t1 = ConcreteTestTeacher(name="T1")
        t2 = ConcreteTestTeacher(name="T2")

        stage_weights = {
            "foundation": [0.8, 0.2],
            "advanced": [0.3, 0.7],
        }

        curriculum = CurriculumCompositeTeacher(
            teachers=[t1, t2],
            stage_weights=stage_weights,
            current_stage="foundation",
        )

        assert curriculum.current_stage == "foundation"
        assert curriculum.weights == [0.8, 0.2]

    def test_curriculum_composite_set_stage(self):
        """set_stage updates weights correctly."""
        t1 = ConcreteTestTeacher()
        t2 = ConcreteTestTeacher()

        stage_weights = {
            "foundation": [0.9, 0.1],
            "intermediate": [0.5, 0.5],
            "advanced": [0.2, 0.8],
        }

        curriculum = CurriculumCompositeTeacher(
            teachers=[t1, t2],
            stage_weights=stage_weights,
            current_stage="foundation",
        )

        assert curriculum.weights == [0.9, 0.1]

        curriculum.set_stage("advanced")
        assert curriculum.current_stage == "advanced"
        assert curriculum.weights == [0.2, 0.8]

        curriculum.set_stage("intermediate")
        assert curriculum.current_stage == "intermediate"
        assert curriculum.weights == [0.5, 0.5]

    def test_curriculum_composite_unknown_stage(self):
        """set_stage with unknown stage keeps both stage and weights unchanged."""
        t1 = ConcreteTestTeacher()
        t2 = ConcreteTestTeacher()

        stage_weights = {
            "foundation": [0.8, 0.2],
        }

        curriculum = CurriculumCompositeTeacher(
            teachers=[t1, t2],
            stage_weights=stage_weights,
            current_stage="foundation",
        )

        original_weights = curriculum.weights.copy()
        original_stage = curriculum.current_stage
        curriculum.set_stage("unknown_stage")

        # Both stage and weights should remain unchanged (silently ignored)
        assert curriculum.weights == original_weights
        assert curriculum.current_stage == original_stage


# =============================================================================
# Integration-style Tests (with mocked API calls)
# =============================================================================


class TestTeacherIntegration:
    """Integration tests for teacher workflows."""

    @pytest.mark.asyncio
    async def test_full_dialogue_workflow(self):
        """Test complete dialogue workflow with mocked teacher."""
        teacher = ConcreteTestTeacher(name="Test Teacher")

        call_count = 0

        async def mock_student_generate(prompt, challenge, dialogue_history):
            nonlocal call_count
            call_count += 1
            return f"Response {call_count} to challenge"

        history = await teacher.run_dialogue(
            prompt="Explain recursion",
            student_response="Recursion is when a function calls itself",
            student_generate_fn=mock_student_generate,
            max_turns=3,
            evaluate_each_turn=True,
        )

        # Verify dialogue structure
        assert history.prompt == "Explain recursion"
        assert history.initial_response == "Recursion is when a function calls itself"
        assert history.num_turns == 3

        # Verify each turn has evaluation (since evaluate_each_turn=True)
        for i, turn in enumerate(history.turns):
            assert turn.turn_number == i + 1
            assert turn.evaluation is not None
            assert f"Response {i + 1}" in turn.student_response

        # Verify final evaluation
        assert history.final_evaluation is not None
        assert history.final_evaluation.improved_response is not None

    @pytest.mark.asyncio
    async def test_composite_teacher_full_workflow(self):
        """Test composite teacher with multiple teachers."""
        t1 = ConcreteTestTeacher(name="Questioner")
        t2 = ConcreteTestTeacher(name="Critic")

        # Override their methods to return different results
        t1.evaluate = AsyncMock(
            return_value=TeacherEvaluation(
                overall_score=8.0,
                dimension_scores=[DimensionScore(EvaluationDimension.REASONING, 8.0, "Good")],
                reasoning="Solid reasoning",
                strengths=["Clear logic"],
            )
        )
        t2.evaluate = AsyncMock(
            return_value=TeacherEvaluation(
                overall_score=6.0,
                dimension_scores=[DimensionScore(EvaluationDimension.NUANCE, 6.0, "Could improve")],
                reasoning="Missing nuance",
                weaknesses=["Too simple"],
            )
        )

        composite = CompositeTeacher([t1, t2], strategy="vote")

        result = await composite.evaluate(
            prompt="Test prompt",
            student_response="Test response",
        )

        # Verify weighted combination
        assert result.overall_score == 7.0  # (8 + 6) / 2
        assert "Clear logic" in result.strengths
        assert "Too simple" in result.weaknesses


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
