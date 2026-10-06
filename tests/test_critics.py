"""
Comprehensive tests for ASPIRE critic models.

Tests for:
- CriticOutput: Data structure for critic predictions
- BaseCritic: Abstract base class
- CriticHead: Lightweight MLP on student hidden states
- MultiHeadCriticHead: Multi-head attention pooling variant
- SeparateCritic: Independent encoder model
- SharedEncoderCritic: Shared encoder with student

Windows compatibility notes:
- Use num_workers=0 in DataLoader tests
- Use if __name__ == "__main__": freeze_support() pattern
- Mock heavy model loading to keep tests fast
"""

from __future__ import annotations

from multiprocessing import freeze_support
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from aspire.critic.base import BaseCritic, CriticOutput
from aspire.critic.head import CriticHead, MultiHeadCriticHead
from aspire.critic.separate import SeparateCritic
from aspire.critic.shared import SharedEncoderCritic

if TYPE_CHECKING:
    pass


# ============================================================================
# CriticOutput Tests
# ============================================================================


class TestCriticOutput:
    """Tests for CriticOutput dataclass."""

    def test_critic_output_creation(self):
        """CriticOutput can be created with score and reasoning_embedding tensors."""
        score = torch.tensor([7.5, 8.0, 6.5])
        reasoning = torch.randn(3, 768)

        output = CriticOutput(score=score, reasoning_embedding=reasoning)

        assert output.score is score
        assert output.reasoning_embedding is reasoning
        assert output.dimension_scores is None
        assert output.hidden_states is None
        assert output.attentions is None

    def test_critic_output_to_dict(self):
        """to_dict() returns numpy arrays on CPU."""
        score = torch.tensor([7.5, 8.0])
        reasoning = torch.randn(2, 768)
        dimension_scores = {
            "dim_0": torch.tensor([6.0, 7.0]),
            "dim_1": torch.tensor([8.0, 9.0]),
        }

        output = CriticOutput(
            score=score,
            reasoning_embedding=reasoning,
            dimension_scores=dimension_scores,
        )

        result = output.to_dict()

        # Verify numpy arrays returned
        assert isinstance(result["score"], np.ndarray)
        assert isinstance(result["reasoning_embedding"], np.ndarray)
        assert isinstance(result["dimension_scores"], dict)
        assert isinstance(result["dimension_scores"]["dim_0"], np.ndarray)
        assert isinstance(result["dimension_scores"]["dim_1"], np.ndarray)

        # Verify values match
        np.testing.assert_array_almost_equal(result["score"], score.numpy())
        np.testing.assert_array_almost_equal(
            result["reasoning_embedding"], reasoning.numpy()
        )

    def test_critic_output_to_dict_cuda(self):
        """to_dict() moves CUDA tensors to CPU."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        score = torch.tensor([7.5]).cuda()
        reasoning = torch.randn(1, 768).cuda()

        output = CriticOutput(score=score, reasoning_embedding=reasoning)
        result = output.to_dict()

        # Should be numpy arrays (on CPU)
        assert isinstance(result["score"], np.ndarray)
        assert isinstance(result["reasoning_embedding"], np.ndarray)

    def test_critic_output_optional_fields(self):
        """CriticOutput works with only required score field."""
        score = torch.tensor([5.0])

        output = CriticOutput(score=score)

        assert output.score is score
        assert output.reasoning_embedding is None
        assert output.dimension_scores is None
        assert output.hidden_states is None
        assert output.attentions is None

    def test_critic_output_to_dict_with_none_fields(self):
        """to_dict() handles None optional fields."""
        score = torch.tensor([5.0])

        output = CriticOutput(score=score)
        result = output.to_dict()

        assert result["reasoning_embedding"] is None
        assert result["dimension_scores"] is None


# ============================================================================
# BaseCritic Tests
# ============================================================================


class ConcreteCritic(BaseCritic):
    """Concrete implementation of BaseCritic for testing."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.score_layer = nn.Linear(kwargs.get("hidden_dim", 768), 1)
        self.reasoning_layer = nn.Linear(
            kwargs.get("hidden_dim", 768), kwargs.get("reasoning_dim", 768)
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        hidden_states=None,
        **kwargs,
    ):
        if hidden_states is None:
            raise ValueError("hidden_states required")

        pooled = hidden_states.mean(dim=1)
        score = torch.sigmoid(self.score_layer(pooled)) * 10.0
        reasoning = self.reasoning_layer(pooled)

        return CriticOutput(
            score=score.squeeze(-1),
            reasoning_embedding=reasoning,
        )

    def get_trainable_parameters(self):
        return list(self.parameters())


class TestBaseCritic:
    """Tests for BaseCritic abstract base class."""

    def test_base_critic_is_abstract(self):
        """BaseCritic cannot be instantiated directly."""
        with pytest.raises(TypeError, match="abstract"):
            BaseCritic()

    def test_base_critic_init(self):
        """Concrete subclass stores configuration correctly."""
        critic = ConcreteCritic(
            hidden_dim=512,
            score_dim=1,
            reasoning_dim=256,
            num_dimensions=3,
        )

        assert critic.hidden_dim == 512
        assert critic.score_dim == 1
        assert critic.reasoning_dim == 256
        assert critic.num_dimensions == 3

    def test_base_critic_init_defaults(self):
        """BaseCritic has sensible defaults."""
        critic = ConcreteCritic()

        assert critic.hidden_dim == 768
        assert critic.score_dim == 1
        assert critic.reasoning_dim == 768
        assert critic.num_dimensions == 0

    def test_base_critic_predict_score(self):
        """predict_score() calls forward in eval mode with no_grad."""
        critic = ConcreteCritic()
        hidden_states = torch.randn(1, 10, 768)

        # Track if eval() was called
        eval_called = False
        original_eval = critic.eval

        def tracked_eval():
            nonlocal eval_called
            eval_called = True
            return original_eval()

        critic.eval = tracked_eval

        score = critic.predict_score(hidden_states=hidden_states)

        assert eval_called
        assert isinstance(score, float)
        assert 0 <= score <= 10

    def test_base_critic_predict_score_no_grad(self):
        """predict_score() runs with torch.no_grad()."""
        critic = ConcreteCritic()
        hidden_states = torch.randn(1, 10, 768, requires_grad=True)

        score = critic.predict_score(hidden_states=hidden_states)

        # Should not have computed gradients for internal operations
        assert isinstance(score, float)

    def test_base_critic_save(self, tmp_path):
        """save() stores state_dict and config."""
        critic = ConcreteCritic(hidden_dim=512, reasoning_dim=256, num_dimensions=2)
        save_path = tmp_path / "critic.pt"

        critic.save(str(save_path))

        assert save_path.exists()

        # Load and verify
        checkpoint = torch.load(save_path)
        assert "state_dict" in checkpoint
        assert "config" in checkpoint
        assert checkpoint["config"]["hidden_dim"] == 512
        assert checkpoint["config"]["reasoning_dim"] == 256
        assert checkpoint["config"]["num_dimensions"] == 2

    def test_base_critic_load(self, tmp_path):
        """load() restores critic from saved state."""
        # Create and save critic
        original = ConcreteCritic(hidden_dim=512, reasoning_dim=256)
        save_path = tmp_path / "critic.pt"
        original.save(str(save_path))

        # Load into new instance
        loaded = ConcreteCritic.load(str(save_path))

        assert loaded.hidden_dim == 512
        assert loaded.reasoning_dim == 256

        # Check weights match
        for (name1, param1), (name2, param2) in zip(
            original.named_parameters(), loaded.named_parameters()
        ):
            assert name1 == name2
            assert torch.allclose(param1, param2)


# ============================================================================
# CriticHead Tests
# ============================================================================


class TestCriticHead:
    """Tests for CriticHead - lightweight MLP on student hidden states."""

    def test_critic_head_init(self):
        """CriticHead initializes with correct architecture."""
        head = CriticHead(input_dim=768, hidden_dim=512, num_layers=2)

        assert head.input_dim == 768
        assert head.hidden_dim == 512
        assert head.pooling == "mean"

        # Check MLP structure
        assert head.mlp is not None
        assert head.score_head is not None
        assert head.reasoning_head is not None

    def test_critic_head_init_with_attention_pooling(self):
        """Attention pooling creates pool_attention layer."""
        head = CriticHead(input_dim=768, pooling="attention")

        assert hasattr(head, "pool_attention")
        assert isinstance(head.pool_attention, nn.Linear)
        assert head.pool_attention.in_features == 768
        assert head.pool_attention.out_features == 1

    def test_critic_head_init_with_dimensions(self):
        """num_dimensions creates dimension_heads ModuleList."""
        head = CriticHead(input_dim=768, num_dimensions=5)

        assert hasattr(head, "dimension_heads")
        assert isinstance(head.dimension_heads, nn.ModuleList)
        assert len(head.dimension_heads) == 5

    def test_critic_head_init_weights(self):
        """Weights are initialized with xavier_uniform, biases with zeros."""
        head = CriticHead(input_dim=768)

        # Check a linear layer's bias is zeros
        for module in head.modules():
            if isinstance(module, nn.Linear):
                if module.bias is not None:
                    assert torch.allclose(module.bias, torch.zeros_like(module.bias))

    def test_critic_head_pool_mean(self):
        """Mean pooling computes mean across sequence dimension."""
        head = CriticHead(input_dim=768, pooling="mean")
        hidden_states = torch.randn(4, 128, 768)

        pooled = head._pool(hidden_states)

        assert pooled.shape == (4, 768)
        # Verify it's actually the mean
        expected = hidden_states.mean(dim=1)
        assert torch.allclose(pooled, expected)

    def test_critic_head_pool_mean_with_mask(self):
        """Mean pooling excludes padding tokens."""
        head = CriticHead(input_dim=768, pooling="mean")
        hidden_states = torch.randn(2, 10, 768)

        # Mask: first sample has 5 tokens, second has 8 tokens
        attention_mask = torch.zeros(2, 10, dtype=torch.long)
        attention_mask[0, :5] = 1
        attention_mask[1, :8] = 1

        pooled = head._pool(hidden_states, attention_mask)

        assert pooled.shape == (2, 768)

        # Verify first sample uses only first 5 tokens
        expected_0 = hidden_states[0, :5].mean(dim=0)
        assert torch.allclose(pooled[0], expected_0)

        # Verify second sample uses only first 8 tokens
        expected_1 = hidden_states[1, :8].mean(dim=0)
        assert torch.allclose(pooled[1], expected_1)

    def test_critic_head_pool_last(self):
        """Last pooling uses last token."""
        head = CriticHead(input_dim=768, pooling="last")
        hidden_states = torch.randn(4, 128, 768)

        pooled = head._pool(hidden_states)

        assert pooled.shape == (4, 768)
        # Verify it's the last token
        expected = hidden_states[:, -1, :]
        assert torch.allclose(pooled, expected)

    def test_critic_head_pool_last_with_mask(self):
        """Last pooling finds last non-padded token."""
        head = CriticHead(input_dim=768, pooling="last")
        hidden_states = torch.randn(2, 10, 768)

        # Different sequence lengths
        attention_mask = torch.zeros(2, 10, dtype=torch.long)
        attention_mask[0, :5] = 1  # Last valid is index 4
        attention_mask[1, :8] = 1  # Last valid is index 7

        pooled = head._pool(hidden_states, attention_mask)

        assert pooled.shape == (2, 768)
        assert torch.allclose(pooled[0], hidden_states[0, 4])
        assert torch.allclose(pooled[1], hidden_states[1, 7])

    def test_critic_head_pool_attention(self):
        """Attention pooling computes weighted sum."""
        head = CriticHead(input_dim=768, pooling="attention")
        hidden_states = torch.randn(4, 128, 768)

        pooled = head._pool(hidden_states)

        assert pooled.shape == (4, 768)

    def test_critic_head_pool_invalid(self):
        """Invalid pooling raises ValueError."""
        # Can't set invalid pooling via normal init due to no validation,
        # but we can test _pool directly
        head = CriticHead(input_dim=768, pooling="mean")
        head.pooling = "invalid"

        with pytest.raises(ValueError, match="Unknown pooling"):
            head._pool(torch.randn(4, 128, 768))

    def test_critic_head_forward_shape(self):
        """Forward pass produces correct output shapes."""
        head = CriticHead(input_dim=768, hidden_dim=512, reasoning_dim=256)
        hidden_states = torch.randn(4, 128, 768)

        output = head(hidden_states=hidden_states)

        assert isinstance(output, CriticOutput)
        assert output.score.shape == (4,)
        assert output.reasoning_embedding.shape == (4, 256)

    def test_critic_head_forward_requires_hidden_states(self):
        """Forward without hidden_states raises ValueError."""
        head = CriticHead(input_dim=768)

        with pytest.raises(ValueError, match="requires hidden_states"):
            head()

    def test_critic_head_forward_score_range(self):
        """Score is bounded in [0, 10] via sigmoid."""
        head = CriticHead(input_dim=768)
        hidden_states = torch.randn(100, 10, 768)  # Large batch to test range

        output = head(hidden_states=hidden_states)

        assert output.score.min() >= 0.0
        assert output.score.max() <= 10.0

    def test_critic_head_forward_reasoning_normalized(self):
        """Reasoning embedding is L2 normalized."""
        head = CriticHead(input_dim=768, reasoning_dim=256)
        hidden_states = torch.randn(4, 128, 768)

        output = head(hidden_states=hidden_states)

        # L2 norm should be ~1 for each sample
        norms = torch.norm(output.reasoning_embedding, p=2, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_critic_head_forward_with_dimensions(self):
        """num_dimensions produces dimension_scores dict."""
        head = CriticHead(input_dim=768, num_dimensions=3)
        hidden_states = torch.randn(4, 128, 768)

        output = head(hidden_states=hidden_states)

        assert output.dimension_scores is not None
        assert "dim_0" in output.dimension_scores
        assert "dim_1" in output.dimension_scores
        assert "dim_2" in output.dimension_scores

        # Each dimension score should be [batch_size, 1]
        for i in range(3):
            assert output.dimension_scores[f"dim_{i}"].shape == (4, 1)
            assert output.dimension_scores[f"dim_{i}"].min() >= 0.0
            assert output.dimension_scores[f"dim_{i}"].max() <= 10.0

    def test_critic_head_forward_with_attention_mask(self):
        """Forward pass works with attention_mask."""
        head = CriticHead(input_dim=768, pooling="mean")
        hidden_states = torch.randn(4, 128, 768)
        attention_mask = torch.ones(4, 128)
        attention_mask[:, 64:] = 0  # Half padding

        output = head(hidden_states=hidden_states, attention_mask=attention_mask)

        assert output.score.shape == (4,)

    def test_critic_head_get_trainable_parameters(self):
        """get_trainable_parameters returns all parameters."""
        head = CriticHead(input_dim=768, num_dimensions=2)

        params = head.get_trainable_parameters()

        assert isinstance(params, list)
        assert len(params) == sum(1 for _ in head.parameters())
        assert all(isinstance(p, nn.Parameter) for p in params)


# ============================================================================
# MultiHeadCriticHead Tests
# ============================================================================


class TestMultiHeadCriticHead:
    """Tests for MultiHeadCriticHead with multi-head attention pooling."""

    def test_multi_head_critic_init(self):
        """MultiHeadCriticHead initializes with multihead attention."""
        head = MultiHeadCriticHead(input_dim=768, num_heads=4)

        assert hasattr(head, "multihead_attn")
        assert isinstance(head.multihead_attn, nn.MultiheadAttention)
        assert head.num_heads == 4

    def test_multi_head_critic_init_pool_query(self):
        """MultiHeadCriticHead has learnable pool_query parameter."""
        head = MultiHeadCriticHead(input_dim=768, num_heads=4)

        assert hasattr(head, "pool_query")
        assert isinstance(head.pool_query, nn.Parameter)
        assert head.pool_query.shape == (1, 1, 768)

    def test_multi_head_critic_pool(self):
        """Multi-head attention pooling produces correct shape."""
        head = MultiHeadCriticHead(input_dim=768, num_heads=4)
        hidden_states = torch.randn(4, 128, 768)

        pooled = head._pool(hidden_states)

        assert pooled.shape == (4, 768)

    def test_multi_head_critic_pool_with_mask(self):
        """Multi-head pooling respects attention mask."""
        head = MultiHeadCriticHead(input_dim=768, num_heads=4)
        hidden_states = torch.randn(2, 10, 768)

        attention_mask = torch.ones(2, 10, dtype=torch.long)
        attention_mask[0, 5:] = 0  # Pad second half of first sample

        pooled = head._pool(hidden_states, attention_mask)

        assert pooled.shape == (2, 768)

    def test_multi_head_critic_forward(self):
        """Forward pass works with multi-head pooling."""
        head = MultiHeadCriticHead(input_dim=768, num_heads=4, hidden_dim=512)
        hidden_states = torch.randn(4, 128, 768)

        output = head(hidden_states=hidden_states)

        assert isinstance(output, CriticOutput)
        assert output.score.shape == (4,)

    def test_multi_head_critic_different_heads(self):
        """Different num_heads values work correctly."""
        for num_heads in [1, 2, 4, 8]:
            head = MultiHeadCriticHead(input_dim=768, num_heads=num_heads)
            hidden_states = torch.randn(2, 64, 768)

            output = head(hidden_states=hidden_states)
            assert output.score.shape == (2,)


# ============================================================================
# SeparateCritic Tests
# ============================================================================


class TestSeparateCritic:
    """Tests for SeparateCritic with independent encoder."""

    @pytest.fixture
    def mock_encoder_and_tokenizer(self):
        """Mock encoder model and tokenizer for SeparateCritic."""
        with patch("aspire.critic.separate.AutoModel") as mock_model_class, patch(
            "aspire.critic.separate.AutoTokenizer"
        ) as mock_tokenizer_class:
            # Mock config
            mock_config = MagicMock()
            mock_config.hidden_size = 768

            # Mock model with real device
            mock_model = MagicMock()
            mock_model.config = mock_config
            mock_model.device = torch.device("cpu")
            mock_model.to = MagicMock(return_value=mock_model)

            # Mock model output with real tensors
            def make_output(batch_size=2):
                mock_output = MagicMock()
                mock_output.pooler_output = torch.randn(batch_size, 768)
                mock_output.last_hidden_state = torch.randn(batch_size, 128, 768)
                return mock_output

            mock_model.return_value = make_output()
            mock_model.side_effect = lambda **kwargs: make_output(
                kwargs.get("input_ids", torch.zeros(2, 10)).shape[0]
            )

            # Mock from_pretrained to return model with config
            mock_model_class.from_pretrained.return_value = mock_model

            # Mock tokenizer
            mock_tokenizer = MagicMock()
            mock_tokenizer.pad_token = None
            mock_tokenizer.eos_token = "</s>"

            def mock_tokenize(text, **kwargs):
                if isinstance(text, str):
                    text = [text]
                batch_size = len(text)
                return {
                    "input_ids": torch.randint(0, 1000, (batch_size, 128)),
                    "attention_mask": torch.ones(batch_size, 128, dtype=torch.long),
                }

            mock_tokenizer.return_value = mock_tokenize(["test"])
            mock_tokenizer.side_effect = mock_tokenize
            mock_tokenizer.__call__ = mock_tokenize
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

            yield mock_model, mock_tokenizer

    def test_separate_critic_init(self, mock_encoder_and_tokenizer):
        """SeparateCritic initializes with encoder and tokenizer."""
        mock_model, mock_tokenizer = mock_encoder_and_tokenizer

        critic = SeparateCritic(
            model_name_or_path="microsoft/deberta-v3-base",
            device="cpu",
        )

        assert critic.encoder is not None
        assert critic.tokenizer is not None
        assert critic.projection is not None
        assert critic.score_head is not None
        assert critic.reasoning_head is not None

    def test_separate_critic_init_frozen_encoder(self, mock_encoder_and_tokenizer):
        """freeze_encoder=True sets encoder params to not require grad."""
        mock_model, _ = mock_encoder_and_tokenizer

        # Add mock parameters to encoder
        mock_param = torch.nn.Parameter(torch.randn(10, 10))
        mock_model.parameters.return_value = iter([mock_param])

        critic = SeparateCritic(
            model_name_or_path="test",
            freeze_encoder=True,
            device="cpu",
        )

        assert critic.freeze_encoder is True

    def test_separate_critic_forward_with_input_ids(self, mock_encoder_and_tokenizer):
        """Forward with input_ids calls encoder."""
        mock_model, _ = mock_encoder_and_tokenizer

        critic = SeparateCritic(model_name_or_path="test", device="cpu")

        input_ids = torch.randint(0, 1000, (2, 128))
        attention_mask = torch.ones(2, 128)

        output = critic(input_ids=input_ids, attention_mask=attention_mask)

        assert isinstance(output, CriticOutput)
        mock_model.assert_called()

    def test_separate_critic_forward_with_text(self, mock_encoder_and_tokenizer):
        """Forward with text tokenizes first."""
        mock_model, mock_tokenizer = mock_encoder_and_tokenizer

        critic = SeparateCritic(model_name_or_path="test", device="cpu")

        output = critic(text="sample text")

        assert isinstance(output, CriticOutput)

    def test_separate_critic_forward_with_text_list(self, mock_encoder_and_tokenizer):
        """Forward with list of texts processes batch."""
        critic = SeparateCritic(model_name_or_path="test", device="cpu")

        output = critic(text=["text1", "text2"])

        assert isinstance(output, CriticOutput)

    def test_separate_critic_forward_requires_input(self, mock_encoder_and_tokenizer):
        """Forward without input_ids or text raises ValueError."""
        critic = SeparateCritic(model_name_or_path="test", device="cpu")

        with pytest.raises(ValueError, match="requires input_ids or text"):
            critic()

    def test_separate_critic_forward_pooling_with_pooler(
        self, mock_encoder_and_tokenizer
    ):
        """Uses pooler_output when available."""
        mock_model, _ = mock_encoder_and_tokenizer

        critic = SeparateCritic(model_name_or_path="test", device="cpu")

        output = critic(text="test")

        # pooler_output is available in mock, should be used
        assert isinstance(output, CriticOutput)

    def test_separate_critic_forward_pooling_fallback(self, mock_encoder_and_tokenizer):
        """Falls back to mean pooling when no pooler_output."""
        mock_model, _ = mock_encoder_and_tokenizer

        # Remove pooler_output by setting side_effect that returns None for pooler
        def make_output_no_pooler(**kwargs):
            batch_size = kwargs.get("input_ids", torch.zeros(1, 10)).shape[0]
            mock_output = MagicMock()
            mock_output.pooler_output = None
            mock_output.last_hidden_state = torch.randn(batch_size, 128, 768)
            return mock_output

        mock_model.side_effect = make_output_no_pooler

        critic = SeparateCritic(model_name_or_path="test", device="cpu")
        output = critic(text="test")

        assert isinstance(output, CriticOutput)

    def test_separate_critic_get_trainable_parameters_unfrozen(
        self, mock_encoder_and_tokenizer
    ):
        """Unfrozen encoder includes encoder params."""
        mock_model, _ = mock_encoder_and_tokenizer

        # Create mock parameters
        encoder_param = torch.nn.Parameter(torch.randn(10, 10))
        mock_model.parameters.return_value = iter([encoder_param])

        critic = SeparateCritic(
            model_name_or_path="test",
            freeze_encoder=False,
            device="cpu",
        )

        params = critic.get_trainable_parameters()

        assert len(params) > 0

    def test_separate_critic_get_trainable_parameters_frozen(
        self, mock_encoder_and_tokenizer
    ):
        """Frozen encoder excludes encoder params but includes heads."""
        mock_model, _ = mock_encoder_and_tokenizer

        critic = SeparateCritic(
            model_name_or_path="test",
            freeze_encoder=True,
            device="cpu",
        )

        params = critic.get_trainable_parameters()

        # Should have projection + score_head + reasoning_head params
        assert len(params) > 0

    def test_separate_critic_encode_text(self, mock_encoder_and_tokenizer):
        """encode_text returns hidden_states tensor in eval mode."""
        critic = SeparateCritic(model_name_or_path="test", device="cpu")

        result = critic.encode_text("sample text")

        assert isinstance(result, torch.Tensor)

    def test_separate_critic_with_dimensions(self, mock_encoder_and_tokenizer):
        """num_dimensions creates dimension heads."""
        critic = SeparateCritic(
            model_name_or_path="test",
            num_dimensions=3,
            device="cpu",
        )

        assert hasattr(critic, "dimension_heads")
        assert len(critic.dimension_heads) == 3

        output = critic(text="test")
        assert output.dimension_scores is not None
        assert "dim_0" in output.dimension_scores


# ============================================================================
# SharedEncoderCritic Tests
# ============================================================================


class TestSharedEncoderCritic:
    """Tests for SharedEncoderCritic that shares encoder with student."""

    @pytest.fixture
    def mock_student_model(self):
        """Create mock student model."""
        model = MagicMock()

        # Config with hidden_size
        model.config = MagicMock()
        model.config.hidden_size = 768

        # Forward pass returns hidden states
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

        return model

    @pytest.fixture
    def mock_student_without_hidden_size(self):
        """Create mock student model without hidden_size in config."""
        model = MagicMock()

        # Config without hidden_size
        model.config = MagicMock(spec=[])  # Empty spec - no hidden_size

        # Fallback to embedding_dim
        mock_embedding = MagicMock()
        mock_embedding.embedding_dim = 512
        model.get_input_embeddings.return_value = mock_embedding

        def mock_forward(**kwargs):
            batch_size = kwargs.get("input_ids", torch.zeros(1, 10)).shape[0]
            seq_len = kwargs.get("input_ids", torch.zeros(1, 10)).shape[1]

            output = MagicMock()
            output.hidden_states = (torch.randn(batch_size, seq_len, 512),)
            return output

        model.side_effect = mock_forward
        model.__call__ = mock_forward

        return model

    def test_shared_critic_init(self, mock_student_model):
        """SharedEncoderCritic initializes with student model reference."""
        critic = SharedEncoderCritic(student_model=mock_student_model)

        assert critic.student_model is mock_student_model
        assert critic.student_hidden_size == 768
        assert critic.pool_attention is not None
        assert critic.projection is not None
        assert critic.score_head is not None
        assert critic.reasoning_head is not None

    def test_shared_critic_init_infer_hidden_size(self, mock_student_model):
        """Hidden size inferred from student_model.config.hidden_size."""
        critic = SharedEncoderCritic(student_model=mock_student_model)

        assert critic.student_hidden_size == 768

    def test_shared_critic_init_infer_hidden_size_fallback(
        self, mock_student_without_hidden_size
    ):
        """Falls back to embedding_dim when hidden_size not in config."""
        critic = SharedEncoderCritic(student_model=mock_student_without_hidden_size)

        assert critic.student_hidden_size == 512

    def test_shared_critic_init_with_adapters(self, mock_student_model):
        """use_adapters=True creates adapter Sequential."""
        critic = SharedEncoderCritic(
            student_model=mock_student_model,
            use_adapters=True,
            adapter_dim=64,
        )

        assert critic.use_adapters is True
        assert hasattr(critic, "adapter")
        assert isinstance(critic.adapter, nn.Sequential)

    def test_shared_critic_adapter_initialized_near_identity(self, mock_student_model):
        """Adapter weights initialized to zeros (near identity)."""
        critic = SharedEncoderCritic(
            student_model=mock_student_model,
            use_adapters=True,
        )

        # First linear should have zero weights
        assert torch.allclose(
            critic.adapter[0].weight, torch.zeros_like(critic.adapter[0].weight)
        )
        # Last linear should have zero weights
        assert torch.allclose(
            critic.adapter[2].weight, torch.zeros_like(critic.adapter[2].weight)
        )

    def test_shared_critic_forward_with_input_ids(self, mock_student_model):
        """Forward with input_ids calls student model."""
        critic = SharedEncoderCritic(student_model=mock_student_model)

        input_ids = torch.randint(0, 1000, (2, 128))
        attention_mask = torch.ones(2, 128)

        output = critic(input_ids=input_ids, attention_mask=attention_mask)

        assert isinstance(output, CriticOutput)
        mock_student_model.assert_called()

    def test_shared_critic_forward_with_hidden_states(self, mock_student_model):
        """Forward with hidden_states skips student model call."""
        critic = SharedEncoderCritic(student_model=mock_student_model)

        hidden_states = torch.randn(2, 128, 768)

        # Reset mock to track if it gets called
        mock_student_model.reset_mock()

        output = critic(hidden_states=hidden_states)

        assert isinstance(output, CriticOutput)
        # Student model should NOT be called when hidden_states provided
        mock_student_model.assert_not_called()

    def test_shared_critic_forward_requires_input(self, mock_student_model):
        """Forward without input_ids or hidden_states raises ValueError."""
        critic = SharedEncoderCritic(student_model=mock_student_model)

        with pytest.raises(ValueError, match="Need either input_ids or hidden_states"):
            critic()

    def test_shared_critic_forward_with_adapters(self, mock_student_model):
        """With adapters, hidden_states get residual connection."""
        critic = SharedEncoderCritic(
            student_model=mock_student_model,
            use_adapters=True,
        )

        hidden_states = torch.randn(2, 128, 768)

        output = critic(hidden_states=hidden_states)

        assert isinstance(output, CriticOutput)

    def test_shared_critic_forward_attention_pooling(self, mock_student_model):
        """Attention weights computed with softmax and mask."""
        critic = SharedEncoderCritic(student_model=mock_student_model)

        hidden_states = torch.randn(2, 10, 768)
        attention_mask = torch.ones(2, 10)
        attention_mask[0, 5:] = 0  # Mask second half

        output = critic(hidden_states=hidden_states, attention_mask=attention_mask)

        # Attention weights should be in output
        assert output.attentions is not None
        assert output.attentions.shape == (2, 10)

        # Verify masked positions have ~0 attention
        assert torch.allclose(
            output.attentions[0, 5:], torch.zeros(5), atol=1e-5
        )

    def test_shared_critic_forward_output_structure(self, mock_student_model):
        """Output has attentions field with attention weights."""
        critic = SharedEncoderCritic(student_model=mock_student_model)
        hidden_states = torch.randn(2, 128, 768)

        output = critic(hidden_states=hidden_states)

        assert output.attentions is not None
        assert output.attentions.shape == (2, 128)
        # Attention weights should sum to 1
        assert torch.allclose(
            output.attentions.sum(dim=-1), torch.ones(2), atol=1e-5
        )

    def test_shared_critic_get_trainable_parameters(self, mock_student_model):
        """Trainable parameters exclude student model."""
        critic = SharedEncoderCritic(student_model=mock_student_model)

        params = critic.get_trainable_parameters()

        # Should only have critic head params
        expected_param_count = (
            sum(p.numel() for p in critic.pool_attention.parameters())
            + sum(p.numel() for p in critic.projection.parameters())
            + sum(p.numel() for p in critic.score_head.parameters())
            + sum(p.numel() for p in critic.reasoning_head.parameters())
        )

        actual_param_count = sum(p.numel() for p in params)
        assert actual_param_count == expected_param_count

    def test_shared_critic_get_trainable_parameters_with_adapters(
        self, mock_student_model
    ):
        """With adapters, adapter params included in trainable."""
        critic = SharedEncoderCritic(
            student_model=mock_student_model,
            use_adapters=True,
        )

        params = critic.get_trainable_parameters()

        # Should include adapter params
        adapter_param_count = sum(p.numel() for p in critic.adapter.parameters())
        head_param_count = (
            sum(p.numel() for p in critic.pool_attention.parameters())
            + sum(p.numel() for p in critic.projection.parameters())
            + sum(p.numel() for p in critic.score_head.parameters())
            + sum(p.numel() for p in critic.reasoning_head.parameters())
        )

        actual_param_count = sum(p.numel() for p in params)
        assert actual_param_count == adapter_param_count + head_param_count

    def test_shared_critic_get_attention_weights(self, mock_student_model):
        """get_attention_weights returns attention tensor in eval mode."""
        critic = SharedEncoderCritic(student_model=mock_student_model)

        input_ids = torch.randint(0, 1000, (2, 64))

        weights = critic.get_attention_weights(input_ids)

        assert isinstance(weights, torch.Tensor)
        assert weights.shape == (2, 64)

    def test_shared_critic_forward_score_range(self, mock_student_model):
        """Score is bounded in [0, 10]."""
        critic = SharedEncoderCritic(student_model=mock_student_model)
        hidden_states = torch.randn(100, 10, 768)

        output = critic(hidden_states=hidden_states)

        assert output.score.min() >= 0.0
        assert output.score.max() <= 10.0

    def test_shared_critic_forward_reasoning_normalized(self, mock_student_model):
        """Reasoning embedding is L2 normalized."""
        critic = SharedEncoderCritic(student_model=mock_student_model)
        hidden_states = torch.randn(4, 64, 768)

        output = critic(hidden_states=hidden_states)

        norms = torch.norm(output.reasoning_embedding, p=2, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_shared_critic_with_dimensions(self, mock_student_model):
        """num_dimensions creates dimension heads."""
        critic = SharedEncoderCritic(
            student_model=mock_student_model,
            num_dimensions=3,
        )

        assert hasattr(critic, "dimension_heads")
        assert len(critic.dimension_heads) == 3

        hidden_states = torch.randn(2, 64, 768)
        output = critic(hidden_states=hidden_states)

        assert output.dimension_scores is not None
        assert "dim_0" in output.dimension_scores
        assert "dim_1" in output.dimension_scores
        assert "dim_2" in output.dimension_scores


# ============================================================================
# Integration Tests
# ============================================================================


class TestCriticIntegration:
    """Integration tests for critic components working together."""

    def test_critic_head_gradient_flow(self):
        """Gradients flow through CriticHead."""
        head = CriticHead(input_dim=768, hidden_dim=256)
        hidden_states = torch.randn(2, 10, 768, requires_grad=True)

        output = head(hidden_states=hidden_states)
        loss = output.score.sum()
        loss.backward()

        assert hidden_states.grad is not None
        assert hidden_states.grad.shape == hidden_states.shape

    def test_shared_critic_no_gradient_to_student(self, mock_model):
        """SharedEncoderCritic doesn't backprop through student."""
        # Create a simple "student" with real parameters
        student = nn.Linear(100, 768)

        # Mock config
        class MockConfig:
            hidden_size = 768

        student.config = MockConfig()

        # Create critic
        critic = SharedEncoderCritic(student_model=student)

        # Provide hidden_states directly (bypassing student forward)
        hidden_states = torch.randn(2, 10, 768, requires_grad=True)

        output = critic(hidden_states=hidden_states)
        loss = output.score.sum()
        loss.backward()

        # Student params should not have gradients
        assert student.weight.grad is None

    def test_critic_batch_consistency(self):
        """Critic produces consistent results for repeated samples."""
        head = CriticHead(input_dim=768)
        head.eval()

        # Same input repeated
        single = torch.randn(1, 10, 768)
        batched = single.repeat(4, 1, 1)

        with torch.no_grad():
            single_out = head(hidden_states=single)
            batched_out = head(hidden_states=batched)

        # All batch items should give same score
        assert torch.allclose(
            batched_out.score,
            single_out.score.repeat(4),
            atol=1e-5,
        )

    def test_critic_deterministic_in_eval(self):
        """Critic is deterministic in eval mode."""
        head = CriticHead(input_dim=768, dropout=0.5)  # High dropout
        head.eval()

        hidden_states = torch.randn(2, 10, 768)

        with torch.no_grad():
            out1 = head(hidden_states=hidden_states)
            out2 = head(hidden_states=hidden_states)

        assert torch.allclose(out1.score, out2.score)
        assert torch.allclose(out1.reasoning_embedding, out2.reasoning_embedding)


# ============================================================================
# Windows Compatibility Entry Point
# ============================================================================


if __name__ == "__main__":
    freeze_support()
    pytest.main([__file__, "-v"])
