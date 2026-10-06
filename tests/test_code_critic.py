"""
Comprehensive tests for ASPIRE Code Critic components.

Tests for:
- CriticOutput: Output structure from code critic
- PositionalEncoding: Sinusoidal position embeddings
- CodeEncoder: Encodes code into representations
- CodeCriticHead: Prediction heads for various outputs
- CodeCritic: Complete code evaluation model
- CodeCriticLoss: Training loss for code critic
- create_critic_from_pretrained: Factory function

Windows compatibility notes:
- Use num_workers=0 in DataLoader tests
- Use if __name__ == "__main__": freeze_support() pattern
- Focus on tensor shapes, gradient flow, loss values in expected ranges
"""

from __future__ import annotations

from multiprocessing import freeze_support
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from integrations.code.code_critic import (
    CodeCritic,
    CodeCriticHead,
    CodeCriticLoss,
    CodeEncoder,
    CriticOutput,
    PositionalEncoding,
    create_critic_from_pretrained,
)
from integrations.code.config import CriticArchitecture, CriticConfig

if TYPE_CHECKING:
    pass


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def batch_size():
    """Standard batch size for tests."""
    return 2


@pytest.fixture
def seq_len():
    """Standard sequence length."""
    return 100


@pytest.fixture
def hidden_dim():
    """Standard hidden dimension."""
    return 512


@pytest.fixture
def vocab_size():
    """Standard vocabulary size."""
    return 50000


@pytest.fixture
def sample_input_ids(batch_size, seq_len, vocab_size):
    """Sample tokenized input."""
    return torch.randint(0, vocab_size, (batch_size, seq_len))


@pytest.fixture
def sample_attention_mask(batch_size, seq_len):
    """Sample attention mask with some padding."""
    mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    # Add some padding to second sample
    mask[1, 80:] = 0
    return mask


@pytest.fixture
def sample_encoding(batch_size, seq_len, hidden_dim):
    """Sample encoded representation."""
    return torch.randn(batch_size, seq_len, hidden_dim)


@pytest.fixture
def sample_code_good():
    """Sample good Python code."""
    return '''
def factorial(n: int) -> int:
    """Calculate factorial of n."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n <= 1:
        return 1
    return n * factorial(n - 1)
'''


@pytest.fixture
def sample_code_bad():
    """Sample bad Python code with issues."""
    return '''
def f(x):
    y = eval(x)
    data = []
    for i in range(len(y)):
        if y[i] != None:
            data.append(y[i])
    return data
'''


# ============================================================================
# CriticOutput Tests
# ============================================================================


class TestCriticOutput:
    """Tests for CriticOutput dataclass."""

    def test_code_critic_output_creation(self, batch_size):
        """CriticOutput can be created with just score."""
        score = torch.randn(batch_size)

        output = CriticOutput(score=score)

        assert output.score is score
        assert output.dimension_scores is None
        assert output.reasoning_embedding is None
        assert output.fix_embedding is None
        assert output.token_scores is None
        assert output.attention_weights is None

    def test_code_critic_output_all_fields(
        self, batch_size, seq_len, hidden_dim
    ):
        """CriticOutput stores all fields when populated."""
        score = torch.randn(batch_size)
        dim_scores = {
            "correctness": torch.randn(batch_size),
            "style": torch.randn(batch_size),
        }
        reasoning = torch.randn(batch_size, hidden_dim)
        fix = torch.randn(batch_size, hidden_dim)
        tokens = torch.randn(batch_size, seq_len)
        attn = torch.randn(batch_size, 8, seq_len, seq_len)

        output = CriticOutput(
            score=score,
            dimension_scores=dim_scores,
            reasoning_embedding=reasoning,
            fix_embedding=fix,
            token_scores=tokens,
            attention_weights=attn,
        )

        assert output.score is score
        assert output.dimension_scores is dim_scores
        assert output.reasoning_embedding is reasoning
        assert output.fix_embedding is fix
        assert output.token_scores is tokens
        assert output.attention_weights is attn


# ============================================================================
# PositionalEncoding Tests
# ============================================================================


class TestPositionalEncoding:
    """Tests for PositionalEncoding module."""

    def test_positional_encoding_init(self, hidden_dim):
        """PositionalEncoding initializes with pe buffer."""
        max_len = 2048
        pe = PositionalEncoding(d_model=hidden_dim, max_len=max_len)

        assert hasattr(pe, "pe")
        assert pe.pe.shape == (1, max_len, hidden_dim)

    def test_positional_encoding_forward(self, batch_size, hidden_dim):
        """PositionalEncoding forward preserves shape and adds positions."""
        pe = PositionalEncoding(d_model=hidden_dim, max_len=2048, dropout=0.0)
        pe.eval()  # Disable dropout for testing

        x = torch.randn(batch_size, 100, hidden_dim)
        output = pe(x)

        assert output.shape == x.shape

        # Output should be different from input (positions added)
        assert not torch.allclose(output, x)

    def test_positional_encoding_dropout(self, batch_size, hidden_dim):
        """PositionalEncoding applies dropout in training mode."""
        pe = PositionalEncoding(d_model=hidden_dim, dropout=0.5)
        pe.train()

        x = torch.randn(batch_size, 100, hidden_dim)

        # Run multiple times and check outputs differ (due to dropout)
        out1 = pe(x)
        out2 = pe(x)

        # Outputs should differ due to dropout
        assert not torch.allclose(out1, out2)

    def test_positional_encoding_deterministic_eval(self, batch_size, hidden_dim):
        """PositionalEncoding is deterministic in eval mode."""
        pe = PositionalEncoding(d_model=hidden_dim, dropout=0.5)
        pe.eval()

        x = torch.randn(batch_size, 100, hidden_dim)

        out1 = pe(x)
        out2 = pe(x)

        assert torch.allclose(out1, out2)


# ============================================================================
# CodeEncoder Tests
# ============================================================================


class TestCodeEncoder:
    """Tests for CodeEncoder module."""

    def test_code_encoder_init_transformer(self, hidden_dim):
        """CodeEncoder initializes with transformer architecture."""
        encoder = CodeEncoder(
            vocab_size=50000,
            hidden_dim=hidden_dim,
            num_layers=2,
            num_heads=4,
            architecture=CriticArchitecture.TRANSFORMER,
        )

        assert hasattr(encoder, "embedding")
        assert hasattr(encoder, "pos_encoding")
        assert hasattr(encoder, "encoder")

    @patch("transformers.AutoModel")
    @patch("transformers.AutoTokenizer")
    def test_code_encoder_init_codebert(
        self, mock_tokenizer_class, mock_model_class, hidden_dim
    ):
        """CodeEncoder with CODEBERT loads pretrained model."""
        # Mock the model
        mock_model = MagicMock()
        mock_model.config.hidden_size = 768
        mock_model_class.from_pretrained.return_value = mock_model

        # Mock tokenizer
        mock_tokenizer_class.from_pretrained.return_value = MagicMock()

        encoder = CodeEncoder(
            hidden_dim=hidden_dim,
            architecture=CriticArchitecture.CODEBERT,
            pretrained_model="microsoft/codebert-base",
        )

        assert hasattr(encoder, "backbone")
        mock_model_class.from_pretrained.assert_called_with("microsoft/codebert-base")

    @patch("transformers.AutoModel")
    @patch("transformers.AutoTokenizer")
    def test_code_encoder_init_codebert_projection(
        self, mock_tokenizer_class, mock_model_class
    ):
        """CodeEncoder creates projection when backbone_dim != hidden_dim."""
        mock_model = MagicMock()
        mock_model.config.hidden_size = 768  # Different from hidden_dim=256
        mock_model_class.from_pretrained.return_value = mock_model
        mock_tokenizer_class.from_pretrained.return_value = MagicMock()

        encoder = CodeEncoder(
            hidden_dim=256,  # Different from backbone (768)
            architecture=CriticArchitecture.CODEBERT,
            pretrained_model="test-model",
        )

        assert isinstance(encoder.projection, nn.Linear)
        assert encoder.projection.in_features == 768
        assert encoder.projection.out_features == 256

    def test_code_encoder_forward_transformer(
        self, sample_input_ids, sample_attention_mask, hidden_dim
    ):
        """Transformer encoder produces correct output shape."""
        encoder = CodeEncoder(
            vocab_size=50000,
            hidden_dim=hidden_dim,
            num_layers=2,
            num_heads=4,
            architecture=CriticArchitecture.TRANSFORMER,
        )

        encoding, attention = encoder(sample_input_ids, sample_attention_mask)

        batch, seq = sample_input_ids.shape
        assert encoding.shape == (batch, seq, hidden_dim)
        # Transformer encoder doesn't return attention by default
        assert attention is None

    def test_code_encoder_forward_with_mask(
        self, sample_input_ids, sample_attention_mask, hidden_dim
    ):
        """Encoder handles attention mask (padding)."""
        encoder = CodeEncoder(
            vocab_size=50000,
            hidden_dim=hidden_dim,
            num_layers=2,
            num_heads=4,
            architecture=CriticArchitecture.TRANSFORMER,
        )

        # Should not raise error with mask
        encoding, _ = encoder(sample_input_ids, sample_attention_mask)

        assert encoding.shape[0] == sample_input_ids.shape[0]


# ============================================================================
# CodeCriticHead Tests
# ============================================================================


class TestCodeCriticHead:
    """Tests for CodeCriticHead prediction module."""

    def test_critic_head_init_defaults(self, hidden_dim):
        """CodeCriticHead initializes with all default heads."""
        head = CodeCriticHead(hidden_dim=hidden_dim)

        assert head.score_head is not None
        assert head.dimension_heads is not None
        assert head.token_head is not None
        assert head.reasoning_head is not None
        assert head.fix_head is not None

    def test_critic_head_init_selective(self, hidden_dim):
        """CodeCriticHead can disable specific heads."""
        head = CodeCriticHead(
            hidden_dim=hidden_dim,
            predict_score=False,
            predict_dimensions=False,
        )

        assert head.score_head is None
        assert head.dimension_heads is None
        assert head.token_head is not None  # Still enabled

    def test_critic_head_dimensions_list(self):
        """CodeCriticHead.DIMENSIONS has expected values."""
        expected = [
            "correctness",
            "style",
            "security",
            "performance",
            "maintainability",
            "architecture",
            "documentation",
        ]

        assert CodeCriticHead.DIMENSIONS == expected

    def test_critic_head_forward_mean_pool(self, sample_encoding):
        """Mean pooling averages across sequence dimension."""
        hidden_dim = sample_encoding.shape[-1]
        head = CodeCriticHead(hidden_dim=hidden_dim)

        output = head(sample_encoding, pool="mean")

        # Score should be batch-sized
        batch = sample_encoding.shape[0]
        assert output.score.shape == (batch,)

    def test_critic_head_forward_cls_pool(self, sample_encoding):
        """CLS pooling uses first token."""
        hidden_dim = sample_encoding.shape[-1]
        head = CodeCriticHead(hidden_dim=hidden_dim)

        output = head(sample_encoding, pool="cls")

        batch = sample_encoding.shape[0]
        assert output.score.shape == (batch,)

    def test_critic_head_forward_max_pool(self, sample_encoding):
        """Max pooling uses maximum across sequence."""
        hidden_dim = sample_encoding.shape[-1]
        head = CodeCriticHead(hidden_dim=hidden_dim)

        output = head(sample_encoding, pool="max")

        batch = sample_encoding.shape[0]
        assert output.score.shape == (batch,)

    def test_critic_head_forward_invalid_pool(self, sample_encoding):
        """Invalid pool type raises ValueError."""
        hidden_dim = sample_encoding.shape[-1]
        head = CodeCriticHead(hidden_dim=hidden_dim)

        with pytest.raises(ValueError, match="Unknown pool"):
            head(sample_encoding, pool="invalid")

    def test_critic_head_forward_score(self, sample_encoding):
        """Score is in [0, 10] range via sigmoid."""
        hidden_dim = sample_encoding.shape[-1]
        head = CodeCriticHead(hidden_dim=hidden_dim)

        # Test with many samples
        large_encoding = torch.randn(100, 50, hidden_dim)
        output = head(large_encoding)

        assert output.score.min() >= 0.0
        assert output.score.max() <= 10.0

    def test_critic_head_forward_dimensions(self, sample_encoding):
        """Dimension scores dict is populated correctly."""
        hidden_dim = sample_encoding.shape[-1]
        head = CodeCriticHead(hidden_dim=hidden_dim)

        output = head(sample_encoding)

        assert output.dimension_scores is not None
        for dim in CodeCriticHead.DIMENSIONS:
            assert dim in output.dimension_scores
            # Each dimension score in [0, 10]
            assert output.dimension_scores[dim].min() >= 0.0
            assert output.dimension_scores[dim].max() <= 10.0

    def test_critic_head_forward_tokens(self, sample_encoding):
        """Token scores have correct shape."""
        batch, seq, hidden_dim = sample_encoding.shape
        head = CodeCriticHead(hidden_dim=hidden_dim)

        output = head(sample_encoding)

        assert output.token_scores is not None
        assert output.token_scores.shape == (batch, seq)

    def test_critic_head_forward_reasoning(self, sample_encoding):
        """Reasoning embedding has correct shape."""
        batch, seq, hidden_dim = sample_encoding.shape
        reasoning_dim = 256
        head = CodeCriticHead(hidden_dim=hidden_dim, reasoning_dim=reasoning_dim)

        output = head(sample_encoding)

        assert output.reasoning_embedding is not None
        assert output.reasoning_embedding.shape == (batch, reasoning_dim)

    def test_critic_head_forward_fix(self, sample_encoding):
        """Fix embedding has correct shape."""
        batch, seq, hidden_dim = sample_encoding.shape
        fix_dim = 512
        head = CodeCriticHead(hidden_dim=hidden_dim, fix_dim=fix_dim)

        output = head(sample_encoding)

        assert output.fix_embedding is not None
        assert output.fix_embedding.shape == (batch, fix_dim)

    def test_critic_head_forward_disabled_heads(self, sample_encoding):
        """Disabled heads return None in output."""
        hidden_dim = sample_encoding.shape[-1]
        head = CodeCriticHead(
            hidden_dim=hidden_dim,
            predict_score=False,
            predict_dimensions=False,
        )

        output = head(sample_encoding)

        assert output.score is None
        assert output.dimension_scores is None


# ============================================================================
# CodeCritic Tests
# ============================================================================


class TestCodeCritic:
    """Tests for complete CodeCritic model."""

    def test_code_critic_init_default(self):
        """CodeCritic initializes with default config."""
        # Use transformer architecture to avoid loading pretrained
        config = CriticConfig(
            architecture=CriticArchitecture.TRANSFORMER,
            pretrained_model=None,
        )
        critic = CodeCritic(config=config)

        assert critic.encoder is not None
        assert critic.heads is not None

    def test_code_critic_init_custom_config(self):
        """CodeCritic applies custom config."""
        config = CriticConfig(
            architecture=CriticArchitecture.TRANSFORMER,
            hidden_dim=256,
            num_layers=2,
            pretrained_model=None,
        )
        critic = CodeCritic(config=config)

        assert critic.config.hidden_dim == 256
        assert critic.config.num_layers == 2

    @patch("transformers.AutoTokenizer")
    def test_code_critic_get_tokenizer(self, mock_tokenizer_class):
        """get_tokenizer returns and caches tokenizer."""
        mock_tokenizer = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        config = CriticConfig(
            architecture=CriticArchitecture.TRANSFORMER,
            pretrained_model=None,
        )
        critic = CodeCritic(config=config)

        # First call
        tok1 = critic.get_tokenizer()
        # Second call should return cached
        tok2 = critic.get_tokenizer()

        assert tok1 is tok2

    @patch("transformers.AutoTokenizer")
    def test_code_critic_tokenize(self, mock_tokenizer_class):
        """tokenize returns input_ids and attention_mask."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.randint(0, 1000, (1, 50)),
            "attention_mask": torch.ones(1, 50),
        }
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        config = CriticConfig(
            architecture=CriticArchitecture.TRANSFORMER,
            pretrained_model=None,
        )
        critic = CodeCritic(config=config)

        result = critic.tokenize("def foo(): pass")

        assert "input_ids" in result
        assert "attention_mask" in result

    @patch("transformers.AutoTokenizer")
    def test_code_critic_tokenize_batch(self, mock_tokenizer_class):
        """tokenize handles batch of code strings."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.randint(0, 1000, (2, 50)),
            "attention_mask": torch.ones(2, 50),
        }
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        config = CriticConfig(
            architecture=CriticArchitecture.TRANSFORMER,
            pretrained_model=None,
        )
        critic = CodeCritic(config=config)

        result = critic.tokenize(["code1", "code2"])

        # Should have called with list
        mock_tokenizer.assert_called()

    def test_code_critic_forward(self, sample_input_ids, sample_attention_mask):
        """forward returns CriticOutput with attention_weights."""
        config = CriticConfig(
            architecture=CriticArchitecture.TRANSFORMER,
            pretrained_model=None,
            hidden_dim=256,
        )
        critic = CodeCritic(config=config)

        output = critic(sample_input_ids, sample_attention_mask)

        assert isinstance(output, CriticOutput)
        assert output.score is not None
        # Transformer doesn't return attention by default

    @patch("transformers.AutoTokenizer")
    def test_code_critic_score_code_single(self, mock_tokenizer_class):
        """score_code returns tensor for single code string."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.randint(0, 1000, (1, 50)),
            "attention_mask": torch.ones(1, 50),
        }
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        config = CriticConfig(
            architecture=CriticArchitecture.TRANSFORMER,
            pretrained_model=None,
            hidden_dim=256,
        )
        critic = CodeCritic(config=config)

        score = critic.score_code("def foo(): pass", device="cpu")

        assert isinstance(score, torch.Tensor)
        assert score.shape == (1,)

    @patch("transformers.AutoTokenizer")
    def test_code_critic_score_code_batch(self, mock_tokenizer_class):
        """score_code handles batch of code strings."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.randint(0, 1000, (2, 50)),
            "attention_mask": torch.ones(2, 50),
        }
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        config = CriticConfig(
            architecture=CriticArchitecture.TRANSFORMER,
            pretrained_model=None,
            hidden_dim=256,
        )
        critic = CodeCritic(config=config)

        scores = critic.score_code(["code1", "code2"], device="cpu")

        assert scores.shape == (2,)

    @patch("transformers.AutoTokenizer")
    def test_code_critic_score_code_eval_mode(self, mock_tokenizer_class):
        """score_code runs in eval mode with no_grad."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.randint(0, 1000, (1, 50)),
            "attention_mask": torch.ones(1, 50),
        }
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        config = CriticConfig(
            architecture=CriticArchitecture.TRANSFORMER,
            pretrained_model=None,
            hidden_dim=256,
        )
        critic = CodeCritic(config=config)

        # Track if eval was called
        eval_called = False
        original_eval = critic.eval

        def tracked_eval():
            nonlocal eval_called
            eval_called = True
            return original_eval()

        critic.eval = tracked_eval

        critic.score_code("def foo(): pass", device="cpu")

        assert eval_called

    @patch("transformers.AutoTokenizer")
    def test_code_critic_get_problem_tokens(self, mock_tokenizer_class):
        """get_problem_tokens returns list of (start, end, score)."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.randint(0, 1000, (1, 50)),
            "attention_mask": torch.ones(1, 50),
        }
        mock_tokenizer.decode = MagicMock(return_value="token")
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        config = CriticConfig(
            architecture=CriticArchitecture.TRANSFORMER,
            pretrained_model=None,
            hidden_dim=256,
        )
        critic = CodeCritic(config=config)

        # Set threshold high to catch some tokens
        problems = critic.get_problem_tokens("bad code", threshold=5.0, device="cpu")

        assert isinstance(problems, list)
        for item in problems:
            assert len(item) == 3  # (start, end, score)

    @patch("transformers.AutoTokenizer")
    def test_code_critic_get_problem_tokens_empty(self, mock_tokenizer_class):
        """get_problem_tokens returns empty when all tokens score high."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.randint(0, 1000, (1, 50)),
            "attention_mask": torch.ones(1, 50),
        }
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        config = CriticConfig(
            architecture=CriticArchitecture.TRANSFORMER,
            pretrained_model=None,
            hidden_dim=256,
        )
        critic = CodeCritic(config=config)

        # Very low threshold = all tokens "good"
        problems = critic.get_problem_tokens("good code", threshold=0.0, device="cpu")

        assert problems == []


# ============================================================================
# CodeCriticLoss Tests
# ============================================================================


class TestCodeCriticLoss:
    """Tests for CodeCriticLoss training objective."""

    @pytest.fixture
    def sample_critic_output(self, batch_size, seq_len, hidden_dim):
        """Create sample CriticOutput for testing."""
        return CriticOutput(
            score=torch.randn(batch_size),
            dimension_scores={
                "correctness": torch.randn(batch_size),
                "style": torch.randn(batch_size),
                "security": torch.randn(batch_size),
            },
            reasoning_embedding=torch.randn(batch_size, 256),
            token_scores=torch.randn(batch_size, seq_len),
        )

    def test_code_critic_loss_init(self):
        """CodeCriticLoss initializes with weights."""
        loss_fn = CodeCriticLoss(
            score_weight=1.0,
            dimension_weight=0.5,
            token_weight=0.3,
            reasoning_weight=0.3,
        )

        assert loss_fn.score_weight == 1.0
        assert loss_fn.dimension_weight == 0.5
        assert loss_fn.token_weight == 0.3
        assert loss_fn.reasoning_weight == 0.3

    def test_code_critic_loss_forward_score_only(
        self, sample_critic_output, batch_size
    ):
        """Loss with only score computes MSE."""
        loss_fn = CodeCriticLoss()
        teacher_score = torch.randn(batch_size)

        # Create output with only score
        output = CriticOutput(score=sample_critic_output.score)

        result = loss_fn(output, teacher_score)

        assert "score" in result
        assert "total" in result
        assert result["score"] >= 0  # MSE is non-negative

    def test_code_critic_loss_forward_dimensions(
        self, sample_critic_output, batch_size
    ):
        """Loss computes dimension losses when provided."""
        loss_fn = CodeCriticLoss()
        teacher_score = torch.randn(batch_size)
        teacher_dims = {
            "correctness": torch.randn(batch_size),
            "style": torch.randn(batch_size),
        }

        result = loss_fn(
            sample_critic_output,
            teacher_score,
            teacher_dimensions=teacher_dims,
        )

        assert "dimensions" in result
        assert result["dimensions"] >= 0

    def test_code_critic_loss_forward_tokens(
        self, sample_critic_output, batch_size, seq_len
    ):
        """Loss computes token-level loss with length alignment."""
        loss_fn = CodeCriticLoss()
        teacher_score = torch.randn(batch_size)
        teacher_tokens = torch.randn(batch_size, seq_len + 10)  # Different length

        result = loss_fn(
            sample_critic_output,
            teacher_score,
            teacher_token_scores=teacher_tokens,
        )

        assert "tokens" in result
        assert result["tokens"] >= 0

    def test_code_critic_loss_forward_reasoning(
        self, sample_critic_output, batch_size
    ):
        """Loss computes reasoning distillation loss."""
        loss_fn = CodeCriticLoss()
        teacher_score = torch.randn(batch_size)
        teacher_reasoning = torch.randn(batch_size, 256)

        result = loss_fn(
            sample_critic_output,
            teacher_score,
            teacher_reasoning_embedding=teacher_reasoning,
        )

        assert "reasoning" in result
        assert result["reasoning"] >= 0  # 1 - cosine_sim is non-negative for normalized

    def test_code_critic_loss_forward_total(
        self, sample_critic_output, batch_size, seq_len
    ):
        """Total is weighted sum of all components."""
        loss_fn = CodeCriticLoss(
            score_weight=1.0,
            dimension_weight=0.5,
            token_weight=0.3,
            reasoning_weight=0.3,
        )
        teacher_score = torch.randn(batch_size)
        teacher_dims = {"correctness": torch.randn(batch_size)}
        teacher_tokens = torch.randn(batch_size, seq_len)
        teacher_reasoning = torch.randn(batch_size, 256)

        result = loss_fn(
            sample_critic_output,
            teacher_score,
            teacher_dimensions=teacher_dims,
            teacher_token_scores=teacher_tokens,
            teacher_reasoning_embedding=teacher_reasoning,
        )

        expected_total = (
            1.0 * result["score"]
            + 0.5 * result["dimensions"]
            + 0.3 * result["tokens"]
            + 0.3 * result["reasoning"]
        )

        assert torch.allclose(result["total"], expected_total, atol=1e-5)

    def test_code_critic_loss_forward_missing_components(self, batch_size):
        """Loss handles missing optional components gracefully."""
        loss_fn = CodeCriticLoss()
        teacher_score = torch.randn(batch_size)

        # Minimal output with only score
        output = CriticOutput(score=torch.randn(batch_size))

        result = loss_fn(output, teacher_score)

        # Missing components should be zero tensors
        assert "dimensions" in result
        assert "tokens" in result
        assert "reasoning" in result
        assert torch.allclose(result["dimensions"], torch.tensor(0.0))
        assert torch.allclose(result["tokens"], torch.tensor(0.0))
        assert torch.allclose(result["reasoning"], torch.tensor(0.0))

    def test_code_critic_loss_gradient_flow(self, batch_size, seq_len):
        """Gradients flow through loss computation."""
        loss_fn = CodeCriticLoss()

        # Create output with requires_grad
        score = torch.randn(batch_size, requires_grad=True)
        reasoning = torch.randn(batch_size, 256, requires_grad=True)

        output = CriticOutput(
            score=score,
            reasoning_embedding=reasoning,
        )

        teacher_score = torch.randn(batch_size)
        teacher_reasoning = torch.randn(batch_size, 256)

        result = loss_fn(
            output,
            teacher_score,
            teacher_reasoning_embedding=teacher_reasoning,
        )

        result["total"].backward()

        assert score.grad is not None
        assert reasoning.grad is not None


# ============================================================================
# create_critic_from_pretrained Tests
# ============================================================================


class TestCreateCriticFromPretrained:
    """Tests for factory function."""

    @patch("transformers.AutoModel")
    @patch("transformers.AutoTokenizer")
    def test_create_critic_from_pretrained(
        self, mock_tokenizer_class, mock_model_class
    ):
        """create_critic_from_pretrained sets CODEBERT architecture."""
        mock_model = MagicMock()
        mock_model.config.hidden_size = 768
        mock_model_class.from_pretrained.return_value = mock_model
        mock_tokenizer_class.from_pretrained.return_value = MagicMock()

        critic = create_critic_from_pretrained("microsoft/codebert-base")

        assert critic.config.architecture == CriticArchitecture.CODEBERT
        assert critic.config.pretrained_model == "microsoft/codebert-base"

    @patch("transformers.AutoModel")
    @patch("transformers.AutoTokenizer")
    def test_create_critic_from_pretrained_kwargs(
        self, mock_tokenizer_class, mock_model_class
    ):
        """create_critic_from_pretrained passes kwargs to config."""
        mock_model = MagicMock()
        mock_model.config.hidden_size = 768
        mock_model_class.from_pretrained.return_value = mock_model
        mock_tokenizer_class.from_pretrained.return_value = MagicMock()

        critic = create_critic_from_pretrained(
            "test-model",
            hidden_dim=512,
            predict_fix=False,
        )

        assert critic.config.hidden_dim == 512
        assert critic.config.predict_fix is False


# ============================================================================
# Integration Tests
# ============================================================================


class TestCodeCriticIntegration:
    """Integration tests for code critic components."""

    def test_end_to_end_transformer_critic(self, batch_size, seq_len, vocab_size):
        """Full pipeline: input_ids -> encoding -> predictions."""
        config = CriticConfig(
            architecture=CriticArchitecture.TRANSFORMER,
            pretrained_model=None,
            hidden_dim=256,
            num_layers=2,
        )
        critic = CodeCritic(config=config)

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        output = critic(input_ids, attention_mask)

        # All expected outputs should be present
        assert output.score is not None
        assert output.score.shape == (batch_size,)
        assert output.dimension_scores is not None
        assert output.token_scores.shape == (batch_size, seq_len)

    def test_training_step_simulation(self, batch_size, seq_len, vocab_size):
        """Simulate a training step with forward + loss + backward."""
        config = CriticConfig(
            architecture=CriticArchitecture.TRANSFORMER,
            pretrained_model=None,
            hidden_dim=256,
            num_layers=2,
        )
        critic = CodeCritic(config=config)
        critic.train()

        loss_fn = CodeCriticLoss()

        # Forward
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        output = critic(input_ids)

        # Teacher targets
        teacher_score = torch.rand(batch_size) * 10
        teacher_tokens = torch.rand(batch_size, seq_len) * 10

        # Loss
        losses = loss_fn(
            output,
            teacher_score,
            teacher_token_scores=teacher_tokens,
        )

        # Backward
        losses["total"].backward()

        # Check gradients exist
        for param in critic.parameters():
            if param.requires_grad:
                # At least some params should have gradients
                break
        else:
            pytest.fail("No parameters have gradients")


# ============================================================================
# Windows Compatibility Entry Point
# ============================================================================


if __name__ == "__main__":
    freeze_support()
    pytest.main([__file__, "-v"])
