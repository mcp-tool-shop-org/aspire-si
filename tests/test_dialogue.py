"""
Comprehensive tests for ASPIRE dialogue system.

Tests for:
- GeneratedDialogue: Data structure for dialogue outputs
- DialogueGenerator: Generates adversarial dialogues between student and teacher
- DialogueFormatter: Formats dialogues for different training objectives
- DialogueManager: Manages dialogue generation, caching, and retrieval

Windows compatibility notes:
- Use num_workers=0 in DataLoader tests
- Use if __name__ == "__main__": freeze_support() pattern
- Mock heavy model loading to keep tests fast
"""

from __future__ import annotations

import hashlib
import json
from multiprocessing import freeze_support
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch

from aspire.dialogue.formatter import DialogueFormatter, FormattedDialogue
from aspire.dialogue.generator import DialogueGenerator, GeneratedDialogue
from aspire.dialogue.manager import DialogueManager
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

# ============================================================================
# Test Fixtures
# ============================================================================


class MockTokenizerOutput:
    """Mock tokenizer output that supports .to() method."""

    def __init__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self._data = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def to(self, device):
        """Move tensors to device."""
        return self

    def __getitem__(self, key):
        return self._data[key]

    def keys(self):
        return self._data.keys()


@pytest.fixture
def mock_tokenizer():
    """Mock HuggingFace tokenizer."""
    tokenizer = MagicMock()
    tokenizer.pad_token = "<pad>"
    tokenizer.eos_token = "</s>"
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 1

    def mock_call(text, **kwargs):
        if isinstance(text, str):
            batch_size = 1
        else:
            batch_size = len(text)

        max_length = kwargs.get("max_length", 512)
        input_ids = torch.randint(0, 1000, (batch_size, max_length))
        attention_mask = torch.ones(batch_size, max_length, dtype=torch.long)
        return MockTokenizerOutput(input_ids, attention_mask)

    tokenizer.return_value = mock_call("test")
    tokenizer.side_effect = mock_call
    tokenizer.__call__ = mock_call
    tokenizer.decode = MagicMock(return_value="decoded response text")

    return tokenizer


@pytest.fixture
def mock_student_model():
    """Mock student model."""
    model = MagicMock()
    model.config = MagicMock()
    model.config.hidden_size = 768

    # Mock generate method
    def mock_generate(**kwargs):
        input_length = kwargs.get("input_ids", torch.zeros(1, 10)).shape[1]
        # Return input + some generated tokens
        return torch.randint(0, 1000, (1, input_length + 50))

    model.generate = MagicMock(side_effect=mock_generate)

    # Mock forward for hidden states
    def mock_forward(**kwargs):
        batch_size = kwargs.get("input_ids", torch.zeros(1, 10)).shape[0]
        seq_len = kwargs.get("input_ids", torch.zeros(1, 10)).shape[1]

        output = MagicMock()
        output.hidden_states = (
            torch.randn(batch_size, seq_len, 768),
            torch.randn(batch_size, seq_len, 768),
        )
        return output

    model.side_effect = mock_forward
    model.__call__ = mock_forward
    model.to = MagicMock(return_value=model)
    model.eval = MagicMock(return_value=model)

    return model


@pytest.fixture
def mock_teacher():
    """Mock teacher implementation."""
    teacher = MagicMock(spec=BaseTeacher)
    teacher.name = "MockTeacher"

    # Mock challenge method
    async def mock_challenge(**kwargs):
        return TeacherChallenge(
            challenge_type=ChallengeType.PROBE_REASONING,
            content="Why do you think that?",
            context="Testing reasoning",
            difficulty=0.5,
        )

    teacher.challenge = AsyncMock(side_effect=mock_challenge)

    # Mock evaluate method
    async def mock_evaluate(**kwargs):
        generate_improved = kwargs.get("generate_improved", True)
        return TeacherEvaluation(
            overall_score=7.5,
            dimension_scores=[
                DimensionScore(
                    dimension=EvaluationDimension.REASONING,
                    score=7.5,
                    explanation="Good reasoning",
                )
            ],
            reasoning="Overall good response with room for improvement",
            improved_response="An improved version of the response"
            if generate_improved
            else None,
            strengths=["Clear explanation"],
            weaknesses=["Could add more detail"],
            suggestions=["Consider edge cases"],
        )

    teacher.evaluate = AsyncMock(side_effect=mock_evaluate)

    return teacher


@pytest.fixture
def sample_dialogue_history():
    """Create a sample DialogueHistory."""
    history = DialogueHistory(
        prompt="What is machine learning?",
        initial_response="Machine learning is a subset of AI.",
    )

    turn1 = DialogueTurn(
        turn_number=1,
        challenge=TeacherChallenge(
            challenge_type=ChallengeType.PROBE_REASONING,
            content="Can you explain how it differs from traditional programming?",
        ),
        student_response="Unlike traditional programming where rules are explicit, ML learns patterns from data.",
        evaluation=TeacherEvaluation(
            overall_score=6.5,
            dimension_scores=[],
            reasoning="Basic explanation",
        ),
    )
    history.add_turn(turn1)

    turn2 = DialogueTurn(
        turn_number=2,
        challenge=TeacherChallenge(
            challenge_type=ChallengeType.EDGE_CASE,
            content="What about when data is scarce?",
        ),
        student_response="When data is scarce, techniques like transfer learning can help.",
    )
    history.add_turn(turn2)

    return history


@pytest.fixture
def sample_generated_dialogue(sample_dialogue_history):
    """Create a sample GeneratedDialogue."""
    final_eval = TeacherEvaluation(
        overall_score=8.0,
        dimension_scores=[
            DimensionScore(
                dimension=EvaluationDimension.REASONING,
                score=8.0,
                explanation="Strong reasoning demonstrated",
            )
        ],
        reasoning="Good progression throughout the dialogue",
        improved_response="Machine learning is a paradigm where systems learn patterns from data...",
        strengths=["Adapted well to challenges"],
        weaknesses=["Could use more concrete examples"],
        suggestions=["Add real-world applications"],
    )

    sample_dialogue_history.final_evaluation = final_eval

    return GeneratedDialogue(
        prompt="What is machine learning?",
        initial_response="Machine learning is a subset of AI.",
        history=sample_dialogue_history,
        final_evaluation=final_eval,
        turn_evaluations=[
            sample_dialogue_history.turns[0].evaluation,
            None,  # Second turn has no evaluation
        ],
        metadata={"num_turns": 2, "teacher": "TestTeacher"},
    )


# ============================================================================
# GeneratedDialogue Tests
# ============================================================================


class TestGeneratedDialogue:
    """Tests for GeneratedDialogue dataclass."""

    def test_generated_dialogue_creation(self, sample_dialogue_history):
        """GeneratedDialogue can be created with all fields."""
        final_eval = TeacherEvaluation(
            overall_score=7.5,
            dimension_scores=[],
            reasoning="Good response",
        )

        dialogue = GeneratedDialogue(
            prompt="Test prompt",
            initial_response="Test response",
            history=sample_dialogue_history,
            final_evaluation=final_eval,
            turn_evaluations=[None, None],
            metadata={"key": "value"},
        )

        assert dialogue.prompt == "Test prompt"
        assert dialogue.initial_response == "Test response"
        assert dialogue.history is sample_dialogue_history
        assert dialogue.final_evaluation is final_eval

    def test_generated_dialogue_access_attributes(self, sample_generated_dialogue):
        """Verify all attributes are accessible."""
        dialogue = sample_generated_dialogue

        assert dialogue.prompt == "What is machine learning?"
        assert dialogue.initial_response == "Machine learning is a subset of AI."
        assert dialogue.history.num_turns == 2
        assert dialogue.final_evaluation.overall_score == 8.0

    def test_generated_dialogue_metadata(self, sample_generated_dialogue):
        """Verify metadata dict stores arbitrary data."""
        dialogue = sample_generated_dialogue

        assert isinstance(dialogue.metadata, dict)
        assert dialogue.metadata["num_turns"] == 2
        assert dialogue.metadata["teacher"] == "TestTeacher"


# ============================================================================
# DialogueGenerator Tests
# ============================================================================


class TestDialogueGenerator:
    """Tests for DialogueGenerator class."""

    def test_generator_init(self, mock_student_model, mock_tokenizer, mock_teacher):
        """DialogueGenerator initializes with all required components."""
        generator = DialogueGenerator(
            student_model=mock_student_model,
            student_tokenizer=mock_tokenizer,
            teacher=mock_teacher,
            device="cpu",
        )

        assert generator.student_model is mock_student_model
        assert generator.student_tokenizer is mock_tokenizer
        assert generator.teacher is mock_teacher

    def test_generator_init_defaults(
        self, mock_student_model, mock_tokenizer, mock_teacher
    ):
        """DialogueGenerator has correct default values."""
        generator = DialogueGenerator(
            student_model=mock_student_model,
            student_tokenizer=mock_tokenizer,
            teacher=mock_teacher,
            device="cpu",
        )

        assert generator.max_turns == 3
        assert generator.evaluate_each_turn is True
        assert generator.student_max_length == 512
        assert generator.student_temperature == 0.7

    def test_generator_init_sets_pad_token(
        self, mock_student_model, mock_teacher
    ):
        """Generator sets pad_token to eos_token if None."""
        tokenizer = MagicMock()
        tokenizer.pad_token = None
        tokenizer.eos_token = "</s>"

        generator = DialogueGenerator(
            student_model=mock_student_model,
            student_tokenizer=tokenizer,
            teacher=mock_teacher,
            device="cpu",
        )

        assert tokenizer.pad_token == "</s>"

    @pytest.mark.asyncio
    async def test_generator_generate_dialogue_creates_initial_response(
        self, mock_student_model, mock_tokenizer, mock_teacher
    ):
        """generate_dialogue generates initial response if not provided."""
        generator = DialogueGenerator(
            student_model=mock_student_model,
            student_tokenizer=mock_tokenizer,
            teacher=mock_teacher,
            max_turns=1,
            device="cpu",
        )

        dialogue = await generator.generate_dialogue("What is AI?")

        # Model generate should have been called for initial response
        assert mock_student_model.generate.called
        assert dialogue.prompt == "What is AI?"

    @pytest.mark.asyncio
    async def test_generator_generate_dialogue_uses_provided_initial(
        self, mock_student_model, mock_tokenizer, mock_teacher
    ):
        """generate_dialogue uses provided initial_response."""
        generator = DialogueGenerator(
            student_model=mock_student_model,
            student_tokenizer=mock_tokenizer,
            teacher=mock_teacher,
            max_turns=1,
            device="cpu",
        )

        dialogue = await generator.generate_dialogue(
            "What is AI?",
            initial_response="AI stands for Artificial Intelligence.",
        )

        assert dialogue.initial_response == "AI stands for Artificial Intelligence."

    @pytest.mark.asyncio
    async def test_generator_generate_dialogue_runs_turns(
        self, mock_student_model, mock_tokenizer, mock_teacher
    ):
        """generate_dialogue runs max_turns dialogue turns."""
        generator = DialogueGenerator(
            student_model=mock_student_model,
            student_tokenizer=mock_tokenizer,
            teacher=mock_teacher,
            max_turns=3,
            device="cpu",
        )

        dialogue = await generator.generate_dialogue(
            "What is AI?",
            initial_response="Initial response",
        )

        assert dialogue.history.num_turns == 3
        assert mock_teacher.challenge.call_count == 3

    @pytest.mark.asyncio
    async def test_generator_generate_dialogue_per_turn_eval(
        self, mock_student_model, mock_tokenizer, mock_teacher
    ):
        """evaluate_each_turn=True calls evaluate each turn."""
        generator = DialogueGenerator(
            student_model=mock_student_model,
            student_tokenizer=mock_tokenizer,
            teacher=mock_teacher,
            max_turns=2,
            evaluate_each_turn=True,
            device="cpu",
        )

        dialogue = await generator.generate_dialogue(
            "What is AI?",
            initial_response="Initial response",
        )

        # 2 per-turn evals + 1 final eval = 3 calls
        assert mock_teacher.evaluate.call_count == 3
        assert len(dialogue.turn_evaluations) == 2

    @pytest.mark.asyncio
    async def test_generator_generate_dialogue_skip_per_turn_eval(
        self, mock_student_model, mock_tokenizer, mock_teacher
    ):
        """evaluate_each_turn=False only calls final evaluation."""
        generator = DialogueGenerator(
            student_model=mock_student_model,
            student_tokenizer=mock_tokenizer,
            teacher=mock_teacher,
            max_turns=2,
            evaluate_each_turn=False,
            device="cpu",
        )

        dialogue = await generator.generate_dialogue(
            "What is AI?",
            initial_response="Initial response",
        )

        # Only final eval
        assert mock_teacher.evaluate.call_count == 1
        assert all(e is None for e in dialogue.turn_evaluations)

    @pytest.mark.asyncio
    async def test_generator_generate_dialogue_final_evaluation(
        self, mock_student_model, mock_tokenizer, mock_teacher
    ):
        """Final evaluation is generated with generate_improved=True."""
        generator = DialogueGenerator(
            student_model=mock_student_model,
            student_tokenizer=mock_tokenizer,
            teacher=mock_teacher,
            max_turns=1,
            device="cpu",
        )

        dialogue = await generator.generate_dialogue(
            "What is AI?",
            initial_response="Initial response",
        )

        # Check final eval call had generate_improved=True
        final_call = mock_teacher.evaluate.call_args_list[-1]
        assert final_call.kwargs.get("generate_improved", True) is True
        assert dialogue.final_evaluation is not None
        assert dialogue.history.final_evaluation is dialogue.final_evaluation

    @pytest.mark.asyncio
    async def test_generator_generate_dialogue_returns_metadata(
        self, mock_student_model, mock_tokenizer, mock_teacher
    ):
        """Metadata includes num_turns and teacher name."""
        generator = DialogueGenerator(
            student_model=mock_student_model,
            student_tokenizer=mock_tokenizer,
            teacher=mock_teacher,
            max_turns=2,
            device="cpu",
        )

        dialogue = await generator.generate_dialogue(
            "What is AI?",
            initial_response="Initial response",
        )

        assert dialogue.metadata["num_turns"] == 2
        assert dialogue.metadata["teacher"] == "MockTeacher"

    def test_generator_generate_student_response(
        self, mock_student_model, mock_tokenizer, mock_teacher
    ):
        """_generate_student_response calls model correctly."""
        generator = DialogueGenerator(
            student_model=mock_student_model,
            student_tokenizer=mock_tokenizer,
            teacher=mock_teacher,
            device="cpu",
        )

        response = generator._generate_student_response("Test prompt")

        # Model should be in eval mode
        mock_student_model.eval.assert_called()
        # Generate should be called
        mock_student_model.generate.assert_called()
        # Tokenizer decode called on new tokens
        mock_tokenizer.decode.assert_called()

    def test_generator_format_student_input_basic(
        self, mock_student_model, mock_tokenizer, mock_teacher
    ):
        """_format_student_input formats basic prompt."""
        generator = DialogueGenerator(
            student_model=mock_student_model,
            student_tokenizer=mock_tokenizer,
            teacher=mock_teacher,
            device="cpu",
        )

        formatted = generator._format_student_input("What is AI?")

        assert "Task: What is AI?" in formatted

    def test_generator_format_student_input_with_challenge(
        self, mock_student_model, mock_tokenizer, mock_teacher
    ):
        """_format_student_input includes challenge and response prompt."""
        generator = DialogueGenerator(
            student_model=mock_student_model,
            student_tokenizer=mock_tokenizer,
            teacher=mock_teacher,
            device="cpu",
        )

        formatted = generator._format_student_input(
            "What is AI?",
            challenge="Why do you think that?",
        )

        assert "Challenge: Why do you think that?" in formatted
        assert "Your response:" in formatted

    def test_generator_format_student_input_with_history(
        self, mock_student_model, mock_tokenizer, mock_teacher, sample_dialogue_history
    ):
        """_format_student_input includes history turns."""
        generator = DialogueGenerator(
            student_model=mock_student_model,
            student_tokenizer=mock_tokenizer,
            teacher=mock_teacher,
            device="cpu",
        )

        formatted = generator._format_student_input(
            "What is machine learning?",
            challenge="New challenge",
            history=sample_dialogue_history,
        )

        assert "Your initial response:" in formatted
        # Previous turns should be included
        assert "differs from traditional programming" in formatted

    @pytest.mark.asyncio
    async def test_generator_generate_batch(
        self, mock_student_model, mock_tokenizer, mock_teacher
    ):
        """generate_batch processes multiple prompts."""
        generator = DialogueGenerator(
            student_model=mock_student_model,
            student_tokenizer=mock_tokenizer,
            teacher=mock_teacher,
            max_turns=1,
            device="cpu",
        )

        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        dialogues = await generator.generate_batch(prompts)

        assert len(dialogues) == 3
        # Each prompt should have had a dialogue generated
        for dialogue, prompt in zip(dialogues, prompts):
            assert dialogue.prompt == prompt

    @pytest.mark.asyncio
    async def test_generator_generate_batch_concurrency(
        self, mock_student_model, mock_tokenizer, mock_teacher
    ):
        """generate_batch respects max_concurrent."""
        generator = DialogueGenerator(
            student_model=mock_student_model,
            student_tokenizer=mock_tokenizer,
            teacher=mock_teacher,
            max_turns=1,
            device="cpu",
        )

        prompts = ["P1", "P2", "P3", "P4", "P5"]
        dialogues = await generator.generate_batch(prompts, max_concurrent=2)

        assert len(dialogues) == 5

    def test_generator_get_student_hidden_states(
        self, mock_student_model, mock_tokenizer, mock_teacher
    ):
        """get_student_hidden_states returns last layer hidden states."""
        generator = DialogueGenerator(
            student_model=mock_student_model,
            student_tokenizer=mock_tokenizer,
            teacher=mock_teacher,
            device="cpu",
        )

        hidden = generator.get_student_hidden_states("Test text")

        # Model should be in eval mode
        mock_student_model.eval.assert_called()
        # Should return tensor from hidden_states
        assert isinstance(hidden, torch.Tensor)


# ============================================================================
# DialogueFormatter Tests
# ============================================================================


class TestDialogueFormatter:
    """Tests for DialogueFormatter class."""

    def test_formatter_init_defaults(self):
        """DialogueFormatter has correct defaults."""
        formatter = DialogueFormatter()

        assert formatter.format_type == "chat"
        assert "helpful assistant" in formatter.system_message.lower()
        assert formatter.include_reasoning is False

    def test_formatter_init_custom(self):
        """DialogueFormatter accepts custom parameters."""
        formatter = DialogueFormatter(
            format_type="instruction",
            system_message="Custom system message",
            include_reasoning=True,
        )

        assert formatter.format_type == "instruction"
        assert formatter.system_message == "Custom system message"
        assert formatter.include_reasoning is True

    def test_formatter_format_dialogue_standard(self, sample_generated_dialogue):
        """Standard format returns prompt as input, improved as target."""
        formatter = DialogueFormatter(format_type="standard")

        formatted = formatter.format_dialogue(
            sample_generated_dialogue,
            use_improved_as_target=True,
        )

        assert isinstance(formatted, FormattedDialogue)
        assert formatted.input_text == sample_generated_dialogue.prompt
        assert (
            formatted.target_text
            == sample_generated_dialogue.final_evaluation.improved_response
        )

    def test_formatter_format_dialogue_standard_no_improved(
        self, sample_generated_dialogue
    ):
        """use_improved_as_target=False uses last student response."""
        formatter = DialogueFormatter(format_type="standard")

        formatted = formatter.format_dialogue(
            sample_generated_dialogue,
            use_improved_as_target=False,
        )

        # Should use last turn's student response
        last_response = sample_generated_dialogue.history.turns[-1].student_response
        assert formatted.target_text == last_response

    def test_formatter_format_dialogue_standard_fallback(self, sample_dialogue_history):
        """Falls back to initial_response when no improved and no turns."""
        # Create dialogue with no improved response and no turns
        history = DialogueHistory(
            prompt="Test prompt",
            initial_response="Initial response only",
        )
        final_eval = TeacherEvaluation(
            overall_score=5.0,
            dimension_scores=[],
            reasoning="Basic",
            improved_response=None,  # No improved
        )
        history.final_evaluation = final_eval

        dialogue = GeneratedDialogue(
            prompt="Test prompt",
            initial_response="Initial response only",
            history=history,
            final_evaluation=final_eval,
            turn_evaluations=[],
            metadata={},
        )

        formatter = DialogueFormatter(format_type="standard")
        formatted = formatter.format_dialogue(dialogue, use_improved_as_target=True)

        # Should fall back to initial_response since no improved and no turns
        assert formatted.target_text == "Initial response only"

    def test_formatter_format_dialogue_chat(self, sample_generated_dialogue):
        """Chat format includes role tags."""
        formatter = DialogueFormatter(format_type="chat")

        formatted = formatter.format_dialogue(sample_generated_dialogue)

        assert "<|system|>" in formatted.full_conversation
        assert "<|user|>" in formatted.full_conversation
        assert "<|assistant|>" in formatted.full_conversation
        assert "<|end|>" in formatted.full_conversation

    def test_formatter_format_dialogue_chat_with_turns(self, sample_generated_dialogue):
        """Chat format includes all dialogue turns."""
        formatter = DialogueFormatter(format_type="chat")

        formatted = formatter.format_dialogue(sample_generated_dialogue)

        # Check that turns are included
        assert sample_generated_dialogue.initial_response in formatted.full_conversation
        for turn in sample_generated_dialogue.history.turns:
            assert turn.challenge.content in formatted.full_conversation
            assert turn.student_response in formatted.full_conversation

    def test_formatter_format_dialogue_instruction(self, sample_generated_dialogue):
        """Instruction format uses ### markers."""
        formatter = DialogueFormatter(format_type="instruction")

        formatted = formatter.format_dialogue(sample_generated_dialogue)

        assert "### Instruction:" in formatted.input_text
        assert "### Response:" in formatted.input_text

    def test_formatter_format_dialogue_instruction_with_reasoning(
        self, sample_generated_dialogue
    ):
        """include_reasoning=True adds feedback section."""
        formatter = DialogueFormatter(
            format_type="instruction",
            include_reasoning=True,
        )

        formatted = formatter.format_dialogue(sample_generated_dialogue)

        assert "### Feedback" in formatted.input_text

    def test_formatter_format_dialogue_invalid_type(self, sample_generated_dialogue):
        """Invalid format_type raises ValueError."""
        formatter = DialogueFormatter()
        formatter.format_type = "invalid"

        with pytest.raises(ValueError, match="Unknown format type"):
            formatter.format_dialogue(sample_generated_dialogue)

    def test_formatter_formatted_dialogue_structure(self, sample_generated_dialogue):
        """FormattedDialogue has all required fields."""
        formatter = DialogueFormatter(format_type="standard")

        formatted = formatter.format_dialogue(sample_generated_dialogue)

        assert hasattr(formatted, "input_text")
        assert hasattr(formatted, "target_text")
        assert hasattr(formatted, "full_conversation")
        assert hasattr(formatted, "score")
        assert hasattr(formatted, "improved_response")
        assert hasattr(formatted, "num_turns")

        assert formatted.score == sample_generated_dialogue.final_evaluation.overall_score
        assert formatted.num_turns == len(sample_generated_dialogue.history.turns)

    def test_formatter_build_full_conversation(self, sample_generated_dialogue):
        """_build_full_conversation includes all dialogue components."""
        formatter = DialogueFormatter(format_type="standard")

        full_conv = formatter._build_full_conversation(sample_generated_dialogue)

        assert f"Prompt: {sample_generated_dialogue.prompt}" in full_conv
        assert f"Initial: {sample_generated_dialogue.initial_response}" in full_conv

        for turn in sample_generated_dialogue.history.turns:
            assert f"Challenge: {turn.challenge.content}" in full_conv
            assert f"Response: {turn.student_response}" in full_conv

        assert (
            f"Improved: {sample_generated_dialogue.final_evaluation.improved_response}"
            in full_conv
        )

    def test_formatter_format_for_critic(self, sample_generated_dialogue):
        """format_for_critic creates proper critic input."""
        formatter = DialogueFormatter()

        critic_input = formatter.format_for_critic(sample_generated_dialogue)

        assert f"Task: {sample_generated_dialogue.prompt}" in critic_input
        assert "Response:" in critic_input

    def test_formatter_format_for_critic_with_response(self, sample_generated_dialogue):
        """format_for_critic uses custom response_to_evaluate."""
        formatter = DialogueFormatter()

        critic_input = formatter.format_for_critic(
            sample_generated_dialogue,
            response_to_evaluate="Custom response to evaluate",
        )

        assert "Response: Custom response to evaluate" in critic_input

    def test_formatter_format_for_critic_with_context(self, sample_generated_dialogue):
        """format_for_critic includes dialogue context."""
        formatter = DialogueFormatter()

        critic_input = formatter.format_for_critic(sample_generated_dialogue)

        # Should have context section since there are turns
        assert "Dialogue context:" in critic_input


# ============================================================================
# DialogueManager Tests
# ============================================================================


class TestDialogueManager:
    """Tests for DialogueManager class."""

    @pytest.fixture
    def mock_generator(self, mock_student_model, mock_tokenizer, mock_teacher):
        """Create mock DialogueGenerator."""
        generator = MagicMock(spec=DialogueGenerator)
        generator.teacher = mock_teacher

        async def mock_generate(prompt, **kwargs):
            history = DialogueHistory(prompt=prompt, initial_response="Generated response")
            final_eval = TeacherEvaluation(
                overall_score=7.0,
                dimension_scores=[],
                reasoning="Good",
                improved_response="Improved version",
            )
            history.final_evaluation = final_eval
            return GeneratedDialogue(
                prompt=prompt,
                initial_response="Generated response",
                history=history,
                final_evaluation=final_eval,
                turn_evaluations=[],
                metadata={"num_turns": 0, "teacher": "MockTeacher"},
            )

        generator.generate_dialogue = AsyncMock(side_effect=mock_generate)

        async def mock_batch(prompts, **kwargs):
            results = []
            for prompt in prompts:
                results.append(await mock_generate(prompt))
            return results

        generator.generate_batch = AsyncMock(side_effect=mock_batch)

        return generator

    def test_manager_init(self, mock_generator, tmp_path):
        """DialogueManager initializes with generator and cache_dir."""
        manager = DialogueManager(
            generator=mock_generator,
            cache_dir=tmp_path / "cache",
            use_cache=True,
        )

        assert manager.generator is mock_generator
        assert manager.cache_dir == tmp_path / "cache"
        assert manager.use_cache is True
        assert manager.cache_dir.exists()

    def test_manager_init_no_cache(self, mock_generator, tmp_path):
        """use_cache=False doesn't create cache_dir."""
        cache_path = tmp_path / "no_cache"

        manager = DialogueManager(
            generator=mock_generator,
            cache_dir=cache_path,
            use_cache=False,
        )

        assert manager.use_cache is False
        assert not cache_path.exists()

    def test_manager_get_cache_key(self, mock_generator, tmp_path):
        """_get_cache_key includes teacher name and returns md5 hash."""
        manager = DialogueManager(
            generator=mock_generator,
            cache_dir=tmp_path / "cache",
        )

        key = manager._get_cache_key("Test prompt")

        # Should be md5 hash of "TeacherName:prompt"
        expected = hashlib.md5(b"MockTeacher:Test prompt").hexdigest()
        assert key == expected

    def test_manager_get_cache_path(self, mock_generator, tmp_path):
        """_get_cache_path returns cache_dir/{hash}.json."""
        manager = DialogueManager(
            generator=mock_generator,
            cache_dir=tmp_path / "cache",
        )

        path = manager._get_cache_path("Test prompt")

        assert path.parent == manager.cache_dir
        assert path.suffix == ".json"

    def test_manager_save_to_cache(self, mock_generator, tmp_path, sample_generated_dialogue):
        """_save_to_cache creates JSON file with dialogue data."""
        manager = DialogueManager(
            generator=mock_generator,
            cache_dir=tmp_path / "cache",
        )

        manager._save_to_cache(sample_generated_dialogue)

        cache_path = manager._get_cache_path(sample_generated_dialogue.prompt)
        assert cache_path.exists()

        with open(cache_path) as f:
            data = json.load(f)

        assert data["prompt"] == sample_generated_dialogue.prompt
        assert data["initial_response"] == sample_generated_dialogue.initial_response
        assert "final_evaluation" in data
        assert "turns" in data

    def test_manager_load_from_cache_exists(
        self, mock_generator, tmp_path, sample_generated_dialogue
    ):
        """_load_from_cache returns GeneratedDialogue from cache."""
        manager = DialogueManager(
            generator=mock_generator,
            cache_dir=tmp_path / "cache",
        )

        # Save first
        manager._save_to_cache(sample_generated_dialogue)

        # Load
        loaded = manager._load_from_cache(sample_generated_dialogue.prompt)

        assert loaded is not None
        assert loaded.prompt == sample_generated_dialogue.prompt
        assert loaded.initial_response == sample_generated_dialogue.initial_response
        assert (
            loaded.final_evaluation.overall_score
            == sample_generated_dialogue.final_evaluation.overall_score
        )

    def test_manager_load_from_cache_missing(self, mock_generator, tmp_path):
        """_load_from_cache returns None for non-existent prompt."""
        manager = DialogueManager(
            generator=mock_generator,
            cache_dir=tmp_path / "cache",
        )

        loaded = manager._load_from_cache("nonexistent prompt")

        assert loaded is None

    def test_manager_load_from_cache_corrupt(self, mock_generator, tmp_path):
        """_load_from_cache returns None for corrupt JSON."""
        manager = DialogueManager(
            generator=mock_generator,
            cache_dir=tmp_path / "cache",
        )

        # Write corrupt JSON
        cache_path = manager._get_cache_path("corrupt prompt")
        cache_path.write_text("{ invalid json }")

        loaded = manager._load_from_cache("corrupt prompt")

        assert loaded is None

    @pytest.mark.asyncio
    async def test_manager_get_dialogue_from_cache(
        self, mock_generator, tmp_path, sample_generated_dialogue
    ):
        """get_dialogue uses cache when available."""
        manager = DialogueManager(
            generator=mock_generator,
            cache_dir=tmp_path / "cache",
        )

        # Pre-populate cache
        manager._save_to_cache(sample_generated_dialogue)

        # Get dialogue
        dialogue = await manager.get_dialogue(sample_generated_dialogue.prompt)

        # Generator should NOT have been called
        mock_generator.generate_dialogue.assert_not_called()
        assert dialogue.prompt == sample_generated_dialogue.prompt

    @pytest.mark.asyncio
    async def test_manager_get_dialogue_generates_if_missing(
        self, mock_generator, tmp_path
    ):
        """get_dialogue generates and caches when not in cache."""
        manager = DialogueManager(
            generator=mock_generator,
            cache_dir=tmp_path / "cache",
        )

        dialogue = await manager.get_dialogue("New prompt")

        mock_generator.generate_dialogue.assert_called_once()
        assert dialogue.prompt == "New prompt"

        # Should be cached now
        cache_path = manager._get_cache_path("New prompt")
        assert cache_path.exists()

    @pytest.mark.asyncio
    async def test_manager_get_dialogue_force_regenerate(
        self, mock_generator, tmp_path, sample_generated_dialogue
    ):
        """force_regenerate=True ignores cache."""
        manager = DialogueManager(
            generator=mock_generator,
            cache_dir=tmp_path / "cache",
        )

        # Pre-populate cache
        manager._save_to_cache(sample_generated_dialogue)

        # Get with force_regenerate
        dialogue = await manager.get_dialogue(
            sample_generated_dialogue.prompt,
            force_regenerate=True,
        )

        # Generator SHOULD have been called despite cache
        mock_generator.generate_dialogue.assert_called_once()

    @pytest.mark.asyncio
    async def test_manager_get_dialogues_batch(self, mock_generator, tmp_path):
        """get_dialogues processes multiple prompts."""
        manager = DialogueManager(
            generator=mock_generator,
            cache_dir=tmp_path / "cache",
        )

        prompts = ["P1", "P2", "P3"]
        dialogues = await manager.get_dialogues(prompts)

        assert len(dialogues) == 3
        for i, dialogue in enumerate(dialogues):
            assert dialogue.prompt == prompts[i]

    @pytest.mark.asyncio
    async def test_manager_get_dialogues_partial_cache(
        self, mock_generator, tmp_path, sample_generated_dialogue
    ):
        """get_dialogues uses cache for some, generates others."""
        manager = DialogueManager(
            generator=mock_generator,
            cache_dir=tmp_path / "cache",
        )

        # Cache P1 only
        cached_dialogue = GeneratedDialogue(
            prompt="P1",
            initial_response="Cached response",
            history=DialogueHistory(prompt="P1", initial_response="Cached response"),
            final_evaluation=TeacherEvaluation(
                overall_score=9.0,
                dimension_scores=[],
                reasoning="Excellent",
            ),
            turn_evaluations=[],
            metadata={},
        )
        manager._save_to_cache(cached_dialogue)

        dialogues = await manager.get_dialogues(["P1", "P2"])

        # P1 from cache, P2 generated
        assert dialogues[0].initial_response == "Cached response"
        assert dialogues[1].initial_response == "Generated response"

        # P2 should now be cached
        assert manager._get_cache_path("P2").exists()

    def test_manager_iterate_cached(self, mock_generator, tmp_path):
        """iterate_cached yields all cached dialogues."""
        manager = DialogueManager(
            generator=mock_generator,
            cache_dir=tmp_path / "cache",
        )

        # Create multiple cached dialogues
        for i in range(3):
            history = DialogueHistory(
                prompt=f"Prompt {i}",
                initial_response=f"Response {i}",
            )
            eval = TeacherEvaluation(
                overall_score=float(i + 5),
                dimension_scores=[],
                reasoning="Test",
            )
            history.final_evaluation = eval
            dialogue = GeneratedDialogue(
                prompt=f"Prompt {i}",
                initial_response=f"Response {i}",
                history=history,
                final_evaluation=eval,
                turn_evaluations=[],
                metadata={},
            )
            manager._save_to_cache(dialogue)

        cached = list(manager.iterate_cached())

        assert len(cached) == 3

    def test_manager_iterate_cached_empty(self, mock_generator, tmp_path):
        """iterate_cached returns empty for empty cache."""
        manager = DialogueManager(
            generator=mock_generator,
            cache_dir=tmp_path / "cache",
        )

        cached = list(manager.iterate_cached())

        assert len(cached) == 0

    def test_manager_iterate_cached_skips_corrupt(self, mock_generator, tmp_path):
        """iterate_cached skips corrupt JSON files."""
        manager = DialogueManager(
            generator=mock_generator,
            cache_dir=tmp_path / "cache",
        )

        # Create one good and one corrupt
        history = DialogueHistory(prompt="Good", initial_response="Response")
        eval = TeacherEvaluation(overall_score=7.0, dimension_scores=[], reasoning="OK")
        history.final_evaluation = eval
        good = GeneratedDialogue(
            prompt="Good",
            initial_response="Response",
            history=history,
            final_evaluation=eval,
            turn_evaluations=[],
            metadata={},
        )
        manager._save_to_cache(good)

        # Add corrupt file
        corrupt_path = manager.cache_dir / "corrupt.json"
        corrupt_path.write_text("{ not valid json")

        cached = list(manager.iterate_cached())

        # Should only get the good one
        assert len(cached) == 1
        assert cached[0].prompt == "Good"

    def test_manager_clear_cache(self, mock_generator, tmp_path):
        """clear_cache removes all cached files and returns count."""
        manager = DialogueManager(
            generator=mock_generator,
            cache_dir=tmp_path / "cache",
        )

        # Create cached dialogues
        for i in range(5):
            history = DialogueHistory(prompt=f"P{i}", initial_response=f"R{i}")
            eval = TeacherEvaluation(overall_score=7.0, dimension_scores=[], reasoning="OK")
            history.final_evaluation = eval
            dialogue = GeneratedDialogue(
                prompt=f"P{i}",
                initial_response=f"R{i}",
                history=history,
                final_evaluation=eval,
                turn_evaluations=[],
                metadata={},
            )
            manager._save_to_cache(dialogue)

        count = manager.clear_cache()

        assert count == 5
        assert list(manager.cache_dir.glob("*.json")) == []

    def test_manager_cache_stats(self, mock_generator, tmp_path):
        """cache_stats returns count and size."""
        manager = DialogueManager(
            generator=mock_generator,
            cache_dir=tmp_path / "cache",
        )

        # Create cached dialogues
        for i in range(3):
            history = DialogueHistory(prompt=f"Prompt {i}", initial_response=f"Response {i}")
            eval = TeacherEvaluation(overall_score=7.0, dimension_scores=[], reasoning="OK")
            history.final_evaluation = eval
            dialogue = GeneratedDialogue(
                prompt=f"Prompt {i}",
                initial_response=f"Response {i}",
                history=history,
                final_evaluation=eval,
                turn_evaluations=[],
                metadata={},
            )
            manager._save_to_cache(dialogue)

        stats = manager.cache_stats()

        assert stats["count"] == 3
        assert stats["size_bytes"] > 0
        assert "size_mb" in stats

    def test_manager_cache_stats_empty(self, mock_generator, tmp_path):
        """cache_stats returns zeros for empty cache."""
        manager = DialogueManager(
            generator=mock_generator,
            cache_dir=tmp_path / "cache",
        )

        stats = manager.cache_stats()

        assert stats["count"] == 0
        assert stats["size_bytes"] == 0


# ============================================================================
# Integration Tests
# ============================================================================


class TestDialogueIntegration:
    """Integration tests for dialogue components working together."""

    @pytest.mark.asyncio
    async def test_full_dialogue_flow(
        self, mock_student_model, mock_tokenizer, mock_teacher, tmp_path
    ):
        """Test complete flow: generate -> format -> cache."""
        generator = DialogueGenerator(
            student_model=mock_student_model,
            student_tokenizer=mock_tokenizer,
            teacher=mock_teacher,
            max_turns=2,
            device="cpu",
        )

        manager = DialogueManager(
            generator=generator,
            cache_dir=tmp_path / "cache",
        )

        formatter = DialogueFormatter(format_type="chat")

        # Generate dialogue
        dialogue = await manager.get_dialogue("What is AI?")

        # Format it
        formatted = formatter.format_dialogue(dialogue)

        # Verify chain worked
        assert dialogue.prompt == "What is AI?"
        assert formatted.input_text is not None
        assert formatted.target_text is not None

        # Verify cached
        stats = manager.cache_stats()
        assert stats["count"] == 1

    def test_formatter_handles_all_dialogue_states(self):
        """Formatter handles dialogues with various states."""
        formatter = DialogueFormatter(format_type="standard")

        # Minimal dialogue (no turns)
        minimal_history = DialogueHistory(prompt="P", initial_response="R")
        minimal_eval = TeacherEvaluation(
            overall_score=5.0,
            dimension_scores=[],
            reasoning="Basic",
            improved_response=None,
        )
        minimal_history.final_evaluation = minimal_eval

        minimal = GeneratedDialogue(
            prompt="P",
            initial_response="R",
            history=minimal_history,
            final_evaluation=minimal_eval,
            turn_evaluations=[],
            metadata={},
        )

        formatted = formatter.format_dialogue(minimal)
        assert formatted.target_text == "R"  # Falls back to initial

        # Dialogue with improved response
        improved_history = DialogueHistory(prompt="P2", initial_response="R2")
        improved_eval = TeacherEvaluation(
            overall_score=8.0,
            dimension_scores=[],
            reasoning="Good",
            improved_response="Improved R2",
        )
        improved_history.final_evaluation = improved_eval

        improved = GeneratedDialogue(
            prompt="P2",
            initial_response="R2",
            history=improved_history,
            final_evaluation=improved_eval,
            turn_evaluations=[],
            metadata={},
        )

        formatted = formatter.format_dialogue(improved)
        assert formatted.target_text == "Improved R2"


# ============================================================================
# Windows Compatibility Entry Point
# ============================================================================


if __name__ == "__main__":
    freeze_support()
    pytest.main([__file__, "-v"])
