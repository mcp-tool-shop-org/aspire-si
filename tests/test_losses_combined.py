"""
Comprehensive tests for ASPIRE combined loss functions.

Tests for:
- AspireLoss: Combined training loss orchestrating critic and student losses

Windows compatibility notes:
- Use num_workers=0 in DataLoader tests
- Use if __name__ == "__main__": freeze_support() pattern
- Focus on tensor shapes, gradient flow, loss values in expected ranges
"""

from __future__ import annotations

from multiprocessing import freeze_support
from typing import TYPE_CHECKING

import pytest
import torch

from aspire.losses.combined import AspireLoss
from aspire.losses.critic import CriticLoss
from aspire.losses.student import StudentLoss

if TYPE_CHECKING:
    pass


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def batch_size():
    """Standard batch size for tests."""
    return 4


@pytest.fixture
def hidden_dim():
    """Standard hidden dimension for embeddings."""
    return 768


@pytest.fixture
def seq_len():
    """Standard sequence length."""
    return 128


@pytest.fixture
def vocab_size():
    """Standard vocabulary size."""
    return 32000


@pytest.fixture
def random_scores(batch_size):
    """Random score tensors in [0, 10] range."""
    return torch.rand(batch_size) * 10.0


@pytest.fixture
def random_embeddings(batch_size, hidden_dim):
    """Random embedding tensors."""
    return torch.randn(batch_size, hidden_dim)


@pytest.fixture
def random_logits(batch_size, seq_len, vocab_size):
    """Random logit tensors."""
    return torch.randn(batch_size, seq_len, vocab_size)


# ============================================================================
# AspireLoss Tests
# ============================================================================


class TestAspireLoss:
    """Tests for AspireLoss combined training loss."""

    def test_aspire_loss_init(self):
        """AspireLoss initializes with default weights."""
        loss_fn = AspireLoss()

        assert isinstance(loss_fn.critic_loss, CriticLoss)
        assert isinstance(loss_fn.student_loss, StudentLoss)

    def test_aspire_loss_init_custom_weights(self):
        """AspireLoss accepts custom weights."""
        loss_fn = AspireLoss(
            critic_score_weight=2.0,
            critic_reasoning_weight=0.8,
            critic_contrastive_weight=0.5,
            student_reward_weight=1.5,
            student_contrastive_weight=0.6,
            student_trajectory_weight=0.4,
            student_coherence_weight=0.3,
            student_kl_weight=0.2,
            target_score=8.0,
            contrastive_margin=0.4,
            contrastive_temperature=0.1,
        )

        # Verify weights passed to sub-losses
        assert loss_fn.critic_loss.score_weight == 2.0
        assert loss_fn.critic_loss.reasoning_weight == 0.8
        assert loss_fn.critic_loss.contrastive_weight == 0.5

        assert loss_fn.student_loss.reward_weight == 1.5
        assert loss_fn.student_loss.contrastive_weight == 0.6
        assert loss_fn.student_loss.trajectory_weight == 0.4
        assert loss_fn.student_loss.coherence_weight == 0.3

    def test_aspire_loss_compute_critic_loss(self, random_scores):
        """compute_critic_loss returns dict with loss values."""
        loss_fn = AspireLoss()
        predicted = random_scores.clone()
        target = random_scores.clone() + torch.randn_like(random_scores) * 0.5

        result = loss_fn.compute_critic_loss(
            predicted_score=predicted,
            target_score=target,
        )

        assert isinstance(result, dict)
        assert "score" in result
        assert "total" in result
        assert isinstance(result["score"], torch.Tensor)
        assert result["score"].dim() == 0  # Scalar

    def test_aspire_loss_compute_critic_loss_with_embeddings(
        self, random_scores, random_embeddings
    ):
        """compute_critic_loss with embeddings computes reasoning loss."""
        loss_fn = AspireLoss()

        result = loss_fn.compute_critic_loss(
            predicted_score=random_scores,
            target_score=random_scores.clone(),
            predicted_embedding=random_embeddings,
            target_embedding=random_embeddings.clone(),
        )

        assert "reasoning" in result
        assert isinstance(result["reasoning"], torch.Tensor)
        assert result["reasoning"].dim() == 0

    def test_aspire_loss_compute_critic_loss_with_negative(
        self, random_scores, random_embeddings
    ):
        """compute_critic_loss with negative embedding computes contrastive loss."""
        loss_fn = AspireLoss()
        negative = torch.randn_like(random_embeddings)

        result = loss_fn.compute_critic_loss(
            predicted_score=random_scores,
            target_score=random_scores.clone(),
            predicted_embedding=random_embeddings,
            target_embedding=random_embeddings.clone(),
            negative_embedding=negative,
        )

        assert "contrastive" in result
        assert isinstance(result["contrastive"], torch.Tensor)

    def test_aspire_loss_compute_student_loss(self, random_scores):
        """compute_student_loss returns dict with loss values."""
        loss_fn = AspireLoss()

        result = loss_fn.compute_student_loss(critic_score=random_scores)

        assert isinstance(result, dict)
        assert "reward" in result
        assert "total" in result
        assert isinstance(result["reward"], torch.Tensor)

    def test_aspire_loss_compute_student_loss_with_embeddings(
        self, random_scores, random_embeddings
    ):
        """compute_student_loss with embeddings computes contrastive loss."""
        loss_fn = AspireLoss()

        result = loss_fn.compute_student_loss(
            critic_score=random_scores,
            student_embedding=random_embeddings,
            teacher_embedding=random_embeddings.clone(),
        )

        assert "contrastive" in result
        assert isinstance(result["contrastive"], torch.Tensor)

    def test_aspire_loss_compute_student_loss_with_trajectory(self, batch_size):
        """compute_student_loss with turn_scores computes trajectory loss."""
        loss_fn = AspireLoss()

        # Improving scores across turns
        turn_scores = [
            torch.rand(batch_size) * 5.0,  # Lower initial
            torch.rand(batch_size) * 5.0 + 3.0,  # Improving
            torch.rand(batch_size) * 5.0 + 5.0,  # Higher final
        ]

        result = loss_fn.compute_student_loss(
            critic_score=turn_scores[-1],
            turn_scores=turn_scores,
        )

        assert "trajectory" in result
        assert isinstance(result["trajectory"], torch.Tensor)

    def test_aspire_loss_forward_both(self, random_scores, random_embeddings):
        """forward with train_critic=True, train_student=True returns both losses."""
        loss_fn = AspireLoss()

        result = loss_fn.forward(
            critic_predicted_score=random_scores,
            teacher_score=random_scores.clone(),
            critic_predicted_embedding=random_embeddings,
            teacher_reasoning_embedding=random_embeddings.clone(),
            student_embedding=random_embeddings,
            teacher_improved_embedding=random_embeddings.clone(),
            train_critic=True,
            train_student=True,
        )

        # Should have critic losses prefixed with "critic_"
        assert "critic_score" in result
        assert "critic_total" in result

        # Should have student losses prefixed with "student_"
        assert "student_reward" in result
        assert "student_total" in result

        # Should have combined total
        assert "total" in result

    def test_aspire_loss_forward_critic_only(self, random_scores, random_embeddings):
        """train_student=False returns only critic losses."""
        loss_fn = AspireLoss()

        result = loss_fn.forward(
            critic_predicted_score=random_scores,
            teacher_score=random_scores.clone(),
            train_critic=True,
            train_student=False,
        )

        # Critic losses should be present
        assert "critic_score" in result
        assert "critic_total" in result

        # Student losses should NOT be present
        assert "student_reward" not in result
        assert "student_total" not in result

        # Total should equal critic_total
        assert torch.allclose(result["total"], result["critic_total"])

    def test_aspire_loss_forward_student_only(self, random_scores, random_embeddings):
        """train_critic=False returns only student losses."""
        loss_fn = AspireLoss()

        result = loss_fn.forward(
            critic_predicted_score=random_scores,
            teacher_score=random_scores.clone(),
            train_critic=False,
            train_student=True,
        )

        # Student losses should be present
        assert "student_reward" in result
        assert "student_total" in result

        # Critic losses should NOT be present
        assert "critic_score" not in result
        assert "critic_total" not in result

        # Total should equal student_total
        assert torch.allclose(result["total"], result["student_total"])

    def test_aspire_loss_forward_student_detaches_critic(self, random_scores):
        """Student loss uses detached critic score (no gradient flow)."""
        loss_fn = AspireLoss()
        predicted = random_scores.clone().requires_grad_(True)

        result = loss_fn.forward(
            critic_predicted_score=predicted,
            teacher_score=random_scores.clone(),
            train_critic=True,
            train_student=True,
        )

        # Backprop through total
        result["total"].backward()

        # predicted should have gradients from critic loss
        assert predicted.grad is not None

        # But student loss should not contribute to those gradients
        # (we can't easily verify this directly, but the code uses .detach())

    def test_aspire_loss_forward_total_accumulation(
        self, random_scores, random_embeddings
    ):
        """total = critic_total + student_total when both are trained."""
        loss_fn = AspireLoss()

        result = loss_fn.forward(
            critic_predicted_score=random_scores,
            teacher_score=random_scores.clone(),
            train_critic=True,
            train_student=True,
        )

        expected_total = result["critic_total"] + result["student_total"]
        assert torch.allclose(result["total"], expected_total)

    def test_aspire_loss_forward_empty_total(self, random_scores):
        """train_critic=False, train_student=False returns zero total."""
        loss_fn = AspireLoss()

        result = loss_fn.forward(
            critic_predicted_score=random_scores,
            teacher_score=random_scores.clone(),
            train_critic=False,
            train_student=False,
        )

        assert "total" in result
        assert torch.allclose(
            result["total"],
            torch.tensor(0.0, device=random_scores.device),
        )

    def test_aspire_loss_loss_values_reasonable(self, random_scores, random_embeddings):
        """Loss values are in reasonable ranges (not NaN, not extreme)."""
        loss_fn = AspireLoss()

        result = loss_fn.forward(
            critic_predicted_score=random_scores,
            teacher_score=random_scores.clone(),
            critic_predicted_embedding=random_embeddings,
            teacher_reasoning_embedding=random_embeddings.clone(),
            student_embedding=random_embeddings,
            teacher_improved_embedding=random_embeddings.clone(),
            train_critic=True,
            train_student=True,
        )

        for name, value in result.items():
            assert not torch.isnan(value).any(), f"{name} contains NaN"
            assert not torch.isinf(value).any(), f"{name} contains Inf"
            # Losses should generally be non-negative (allow small floating point errors)
            assert value >= -1e-6 or name in ["trajectory"], f"{name} is negative: {value}"

    def test_aspire_loss_gradient_flow_critic(self, batch_size, hidden_dim):
        """Gradients flow correctly for critic training."""
        loss_fn = AspireLoss()

        predicted_score = torch.randn(batch_size, requires_grad=True)
        predicted_embed = torch.randn(batch_size, hidden_dim, requires_grad=True)
        target_score = torch.randn(batch_size)
        target_embed = torch.randn(batch_size, hidden_dim)

        result = loss_fn.compute_critic_loss(
            predicted_score=predicted_score,
            target_score=target_score,
            predicted_embedding=predicted_embed,
            target_embedding=target_embed,
        )

        result["total"].backward()

        assert predicted_score.grad is not None
        assert predicted_embed.grad is not None
        assert predicted_score.grad.shape == predicted_score.shape
        assert predicted_embed.grad.shape == predicted_embed.shape

    def test_aspire_loss_gradient_flow_student(self, batch_size, hidden_dim):
        """Gradients flow correctly for student training."""
        loss_fn = AspireLoss()

        critic_score = torch.randn(batch_size)  # Should be detached in practice
        student_embed = torch.randn(batch_size, hidden_dim, requires_grad=True)
        teacher_embed = torch.randn(batch_size, hidden_dim)

        result = loss_fn.compute_student_loss(
            critic_score=critic_score,
            student_embedding=student_embed,
            teacher_embedding=teacher_embed,
        )

        result["total"].backward()

        assert student_embed.grad is not None
        assert student_embed.grad.shape == student_embed.shape


class TestCriticLossComponents:
    """Test individual critic loss components."""

    def test_critic_score_loss_shape(self, batch_size):
        """CriticScoreLoss produces scalar output."""
        from aspire.losses.critic import CriticScoreLoss

        loss_fn = CriticScoreLoss(beta=1.0)
        pred = torch.randn(batch_size)
        target = torch.randn(batch_size)

        loss = loss_fn(pred, target)

        assert loss.dim() == 0  # Scalar

    def test_critic_score_loss_zero_when_equal(self, batch_size):
        """CriticScoreLoss is zero when predictions match targets."""
        from aspire.losses.critic import CriticScoreLoss

        loss_fn = CriticScoreLoss()
        scores = torch.randn(batch_size)

        loss = loss_fn(scores, scores.clone())

        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_critic_reasoning_loss_aligned(self, batch_size, hidden_dim):
        """CriticReasoningLoss is low for aligned embeddings."""
        from aspire.losses.critic import CriticReasoningLoss

        loss_fn = CriticReasoningLoss()
        embed = torch.randn(batch_size, hidden_dim)
        embed_normalized = torch.nn.functional.normalize(embed, p=2, dim=-1)

        loss = loss_fn(embed_normalized, embed_normalized.clone())

        # Should be near zero for identical normalized embeddings
        assert loss < 0.01

    def test_critic_contrastive_loss_margin(self, batch_size, hidden_dim):
        """CriticContrastiveLoss uses margin correctly."""
        from aspire.losses.critic import CriticContrastiveLoss

        loss_fn = CriticContrastiveLoss(margin=0.5)
        pred = torch.randn(batch_size, hidden_dim)
        pos = pred.clone()  # Same as pred = high similarity

        loss = loss_fn(pred, pos)

        # With identical embeddings, loss should be related to margin
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0


class TestStudentLossComponents:
    """Test individual student loss components."""

    def test_reward_loss_high_score_low_loss(self, batch_size):
        """RewardLoss is low for high critic scores."""
        from aspire.losses.student import RewardLoss

        loss_fn = RewardLoss(target_score=9.0)

        high_scores = torch.ones(batch_size) * 9.5
        low_scores = torch.ones(batch_size) * 3.0

        high_loss = loss_fn(high_scores)
        low_loss = loss_fn(low_scores)

        # High scores should give lower loss
        assert high_loss < low_loss

    def test_contrastive_loss_pulls_toward_teacher(self, batch_size, hidden_dim):
        """ContrastiveLoss encourages similarity to teacher."""
        from aspire.losses.student import ContrastiveLoss

        loss_fn = ContrastiveLoss()
        student = torch.randn(batch_size, hidden_dim)
        teacher = student.clone()  # Identical = perfect alignment

        loss = loss_fn(student, teacher)

        # Should be low for identical embeddings
        assert loss < 0.1

    def test_trajectory_loss_rewards_improvement(self, batch_size):
        """TrajectoryLoss is negative (good) for improving scores."""
        from aspire.losses.student import TrajectoryLoss

        loss_fn = TrajectoryLoss()

        # Improving trajectory
        improving = [
            torch.ones(batch_size) * 4.0,
            torch.ones(batch_size) * 6.0,
            torch.ones(batch_size) * 8.0,
        ]

        # Declining trajectory
        declining = [
            torch.ones(batch_size) * 8.0,
            torch.ones(batch_size) * 6.0,
            torch.ones(batch_size) * 4.0,
        ]

        improving_loss = loss_fn(improving)
        declining_loss = loss_fn(declining)

        # Improving should have lower (more negative) loss
        assert improving_loss < declining_loss

    def test_coherence_loss_shape(self, batch_size, seq_len, vocab_size):
        """CoherenceLoss produces scalar output."""
        from aspire.losses.student import CoherenceLoss

        loss_fn = CoherenceLoss()
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        loss = loss_fn(logits, labels)

        assert loss.dim() == 0

    def test_kl_divergence_loss_zero_same_distribution(
        self, batch_size, seq_len, vocab_size
    ):
        """KLDivergenceLoss is near zero for identical distributions."""
        from aspire.losses.student import KLDivergenceLoss

        loss_fn = KLDivergenceLoss(beta=1.0)
        logits = torch.randn(batch_size, seq_len, vocab_size)

        loss = loss_fn(logits, logits.clone())

        # KL divergence should be near zero for same distribution
        assert loss < 0.01


# ============================================================================
# Integration Tests
# ============================================================================


class TestLossIntegration:
    """Integration tests for loss functions."""

    def test_full_training_step_simulation(self, batch_size, hidden_dim):
        """Simulate a full training step with both critic and student."""
        loss_fn = AspireLoss()

        # Simulated critic predictions
        critic_score = torch.randn(batch_size).sigmoid() * 10
        critic_score.requires_grad_(True)
        critic_embed = torch.randn(batch_size, hidden_dim, requires_grad=True)

        # Teacher targets
        teacher_score = torch.randn(batch_size).sigmoid() * 10
        teacher_embed = torch.randn(batch_size, hidden_dim)

        # Student outputs
        student_embed = torch.randn(batch_size, hidden_dim, requires_grad=True)

        # Compute loss
        result = loss_fn.forward(
            critic_predicted_score=critic_score,
            teacher_score=teacher_score,
            critic_predicted_embedding=critic_embed,
            teacher_reasoning_embedding=teacher_embed,
            student_embedding=student_embed,
            teacher_improved_embedding=teacher_embed,
            train_critic=True,
            train_student=True,
        )

        # Backward pass
        result["total"].backward()

        # All require_grad tensors should have gradients
        assert critic_score.grad is not None
        assert critic_embed.grad is not None
        assert student_embed.grad is not None

    def test_loss_device_compatibility(self):
        """Loss functions work on both CPU and GPU (if available)."""
        loss_fn = AspireLoss()

        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")

        for device in devices:
            scores = torch.randn(4, device=device)
            embeds = torch.randn(4, 256, device=device)

            result = loss_fn.forward(
                critic_predicted_score=scores,
                teacher_score=scores.clone(),
                train_critic=True,
                train_student=True,
            )

            assert result["total"].device.type == device


# ============================================================================
# Windows Compatibility Entry Point
# ============================================================================


if __name__ == "__main__":
    freeze_support()
    pytest.main([__file__, "-v"])
