"""
Comprehensive tests for ASPIRE loss functions.

Tests the mathematical correctness and edge cases of:
- CriticLoss (score, reasoning, contrastive)
- StudentLoss (reward, contrastive, trajectory, coherence, KL)
"""

import pytest
import torch

from aspire.losses.critic import (
    CriticContrastiveLoss,
    CriticLoss,
    CriticReasoningLoss,
    CriticScoreLoss,
)
from aspire.losses.student import (
    CoherenceLoss,
    ContrastiveLoss,
    KLDivergenceLoss,
    RewardLoss,
    StudentLoss,
    TrajectoryLoss,
)


class TestCriticScoreLoss:
    """Tests for CriticScoreLoss class."""

    @pytest.fixture
    def loss_fn(self):
        """Create loss function instance."""
        return CriticScoreLoss(beta=1.0)

    def test_perfect_prediction(self, loss_fn):
        """Loss should be zero for perfect predictions."""
        predicted = torch.tensor([5.0, 7.0, 3.0])
        target = torch.tensor([5.0, 7.0, 3.0])

        loss = loss_fn(predicted, target)

        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_known_values(self, loss_fn):
        """Test with known expected values."""
        predicted = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([2.0, 2.0, 5.0])

        loss = loss_fn(predicted, target)

        # Smooth L1 loss with beta=1.0
        # Differences: |1|, |0|, |2|
        # For |diff| < beta: 0.5 * diff^2 / beta
        # For |diff| >= beta: |diff| - 0.5 * beta
        # diff=1: 0.5 * 1 / 1 = 0.5
        # diff=0: 0.5 * 0 / 1 = 0
        # diff=2: 2 - 0.5 = 1.5
        # Mean: (0.5 + 0 + 1.5) / 3 = 0.667
        expected = (0.5 + 0.0 + 1.5) / 3.0
        assert loss.item() == pytest.approx(expected, rel=1e-4)

    def test_is_differentiable(self, loss_fn):
        """Loss should be differentiable for backprop."""
        predicted = torch.tensor([5.0, 3.0], requires_grad=True)
        target = torch.tensor([6.0, 4.0])

        loss = loss_fn(predicted, target)
        loss.backward()

        assert predicted.grad is not None
        assert not torch.isnan(predicted.grad).any()

    def test_handles_batch_size_one(self, loss_fn):
        """Loss should work with batch size of 1."""
        predicted = torch.tensor([5.0])
        target = torch.tensor([6.0])

        loss = loss_fn(predicted, target)

        # diff=1, smooth L1 = 0.5
        assert loss.item() == pytest.approx(0.5, rel=1e-5)

    def test_handles_large_values(self, loss_fn):
        """Loss should handle large score values."""
        predicted = torch.tensor([1000.0, 2000.0])
        target = torch.tensor([1000.0, 2000.0])

        loss = loss_fn(predicted, target)

        assert loss.item() == pytest.approx(0.0, abs=1e-4)
        assert not torch.isnan(loss)


class TestCriticReasoningLoss:
    """Tests for CriticReasoningLoss class."""

    @pytest.fixture
    def loss_fn(self):
        """Create loss function instance."""
        return CriticReasoningLoss(margin=0.0)

    def test_identical_embeddings(self, loss_fn):
        """Loss should be zero for identical embeddings."""
        embedding = torch.randn(4, 64)

        loss = loss_fn(embedding, embedding)

        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_opposite_embeddings(self, loss_fn):
        """Loss should be high for opposite embeddings."""
        embedding = torch.randn(4, 64)
        opposite = -embedding

        loss = loss_fn(embedding, opposite)

        # Loss = 1 - cosine_sim, cosine_sim = -1 for opposite
        # Loss = 1 - (-1) = 2
        assert loss.item() == pytest.approx(2.0, rel=1e-4)

    def test_orthogonal_embeddings(self, loss_fn):
        """Loss should be 1 for orthogonal embeddings."""
        # Create orthogonal vectors
        embedding1 = torch.zeros(4, 64)
        embedding1[:, 0] = 1.0
        embedding2 = torch.zeros(4, 64)
        embedding2[:, 1] = 1.0

        loss = loss_fn(embedding1, embedding2)

        # Cosine similarity = 0, loss = 1 - 0 = 1
        assert loss.item() == pytest.approx(1.0, rel=1e-4)


class TestCriticContrastiveLoss:
    """Tests for CriticContrastiveLoss class."""

    @pytest.fixture
    def loss_fn(self):
        """Create loss function instance."""
        return CriticContrastiveLoss(temperature=0.07, margin=0.5)

    def test_correct_ordering(self, loss_fn):
        """Loss should be low when positive is closer than negative."""
        predicted = torch.randn(4, 64)
        positive = predicted + 0.1 * torch.randn_like(predicted)  # Close
        negative = torch.randn_like(predicted)  # Random/far

        loss = loss_fn(predicted, positive, negative)

        # Should be relatively low
        assert loss.item() < 2.0

    def test_without_negative(self, loss_fn):
        """Should work with margin loss when no negative provided."""
        predicted = torch.randn(4, 64)
        positive = predicted + 0.1 * torch.randn_like(predicted)

        loss = loss_fn(predicted, positive, None)

        assert loss.item() >= 0
        assert not torch.isnan(loss)


class TestCriticLossCombined:
    """Tests for combined CriticLoss class."""

    @pytest.fixture
    def loss_fn(self):
        """Create loss function instance."""
        return CriticLoss(
            score_weight=1.0,
            reasoning_weight=0.5,
            contrastive_weight=0.3,
        )

    def test_returns_dict(self, loss_fn):
        """Should return dict with loss components."""
        predicted_score = torch.tensor([7.5, 3.2])
        target_score = torch.tensor([8.0, 3.0])

        losses = loss_fn(predicted_score, target_score)

        assert "score" in losses
        assert "total" in losses
        assert losses["total"].item() > 0

    def test_includes_reasoning_loss(self, loss_fn):
        """Should include reasoning loss when embeddings provided."""
        predicted_score = torch.tensor([7.5, 3.2])
        target_score = torch.tensor([8.0, 3.0])
        pred_embed = torch.randn(2, 64)
        target_embed = torch.randn(2, 64)

        losses = loss_fn(
            predicted_score, target_score,
            predicted_embedding=pred_embed,
            target_embedding=target_embed,
        )

        assert "reasoning" in losses

    def test_includes_contrastive_loss(self, loss_fn):
        """Should include contrastive loss when negative provided."""
        predicted_score = torch.tensor([7.5, 3.2])
        target_score = torch.tensor([8.0, 3.0])
        pred_embed = torch.randn(2, 64)
        target_embed = torch.randn(2, 64)
        neg_embed = torch.randn(2, 64)

        losses = loss_fn(
            predicted_score, target_score,
            predicted_embedding=pred_embed,
            target_embedding=target_embed,
            negative_embedding=neg_embed,
        )

        assert "contrastive" in losses


class TestRewardLoss:
    """Tests for RewardLoss class."""

    @pytest.fixture
    def loss_fn(self):
        """Create loss function instance."""
        return RewardLoss(target_score=9.0, margin=1.0)

    def test_high_score_low_loss(self, loss_fn):
        """High critic scores should result in lower loss."""
        high_score = torch.tensor([9.0])
        low_score = torch.tensor([3.0])

        high_loss = loss_fn(high_score)
        low_loss = loss_fn(low_score)

        assert high_loss.item() < low_loss.item()

    def test_above_target_zero_loss(self, loss_fn):
        """Scores at or above target should have zero loss."""
        perfect_score = torch.tensor([10.0])

        loss = loss_fn(perfect_score)

        assert loss.item() == pytest.approx(0.0, abs=1e-5)


class TestContrastiveLoss:
    """Tests for ContrastiveLoss class."""

    @pytest.fixture
    def loss_fn(self):
        """Create loss function instance."""
        return ContrastiveLoss(margin=0.5, temperature=0.07)

    def test_pulls_toward_teacher(self, loss_fn):
        """Should encourage student to be closer to teacher."""
        student = torch.randn(4, 64)
        teacher = student + 0.1 * torch.randn_like(student)

        loss = loss_fn(student, teacher)

        # Loss is 1 - cos_sim, should be small for similar embeddings
        assert loss.item() < 0.5


class TestTrajectoryLoss:
    """Tests for TrajectoryLoss class."""

    @pytest.fixture
    def loss_fn(self):
        """Create loss function instance."""
        return TrajectoryLoss(improvement_bonus=0.5)

    def test_rewards_improvement(self, loss_fn):
        """Should reward improving scores across turns."""
        improving = [
            torch.tensor([5.0]),
            torch.tensor([6.0]),
            torch.tensor([7.0]),
        ]
        declining = [
            torch.tensor([7.0]),
            torch.tensor([6.0]),
            torch.tensor([5.0]),
        ]

        improving_loss = loss_fn(improving)
        declining_loss = loss_fn(declining)

        assert improving_loss.item() < declining_loss.item()

    def test_single_turn_returns_zero(self, loss_fn):
        """Single turn should return zero loss."""
        single = [torch.tensor([5.0])]

        loss = loss_fn(single)

        assert loss.item() == pytest.approx(0.0, abs=1e-5)


class TestCoherenceLoss:
    """Tests for CoherenceLoss class."""

    @pytest.fixture
    def loss_fn(self):
        """Create loss function instance."""
        return CoherenceLoss(target_perplexity=10.0)

    def test_confident_predictions_lower_loss(self, loss_fn):
        """More confident predictions should have lower perplexity/loss."""
        batch_size, seq_len, vocab_size = 2, 10, 100

        # Create labels
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Confident logits: logits[t] should predict labels[t+1] due to shift
        # So logits[:, t, :] predicts labels[:, t+1]
        confident_logits = torch.zeros(batch_size, seq_len, vocab_size)
        for b in range(batch_size):
            for s in range(seq_len - 1):  # Last position doesn't predict anything
                confident_logits[b, s, labels[b, s + 1]] = 10.0

        # Uncertain: uniform distribution
        uncertain_logits = torch.zeros(batch_size, seq_len, vocab_size)

        confident_loss = loss_fn(confident_logits, labels)
        uncertain_loss = loss_fn(uncertain_logits, labels)

        # Confident predictions should have lower loss
        assert confident_loss.item() < uncertain_loss.item()


class TestKLDivergenceLoss:
    """Tests for KLDivergenceLoss class."""

    @pytest.fixture
    def loss_fn(self):
        """Create loss function instance."""
        return KLDivergenceLoss(beta=0.1)

    def test_same_distribution_zero_kl(self, loss_fn):
        """KL divergence should be zero for identical distributions."""
        logits = torch.randn(2, 10, 50)

        kl = loss_fn(logits, logits)

        assert kl.item() == pytest.approx(0.0, abs=1e-4)

    def test_different_distributions_positive_kl(self, loss_fn):
        """KL divergence should be positive for different distributions."""
        logits_p = torch.randn(2, 10, 50)
        logits_q = torch.randn(2, 10, 50)

        kl = loss_fn(logits_p, logits_q)

        assert kl.item() > 0


class TestStudentLossCombined:
    """Tests for combined StudentLoss class."""

    @pytest.fixture
    def loss_fn(self):
        """Create loss function instance."""
        return StudentLoss(
            reward_weight=1.0,
            contrastive_weight=0.5,
            trajectory_weight=0.3,
        )

    def test_returns_dict(self, loss_fn):
        """Should return dict with loss components."""
        critic_score = torch.tensor([7.5, 3.2])

        losses = loss_fn(critic_score)

        assert "reward" in losses
        assert "total" in losses

    def test_includes_contrastive(self, loss_fn):
        """Should include contrastive loss when embeddings provided."""
        critic_score = torch.tensor([7.5, 3.2])
        student_embed = torch.randn(2, 64)
        teacher_embed = torch.randn(2, 64)

        losses = loss_fn(
            critic_score,
            student_embedding=student_embed,
            teacher_embedding=teacher_embed,
        )

        assert "contrastive" in losses

    def test_includes_trajectory(self, loss_fn):
        """Should include trajectory loss when turn scores provided."""
        critic_score = torch.tensor([7.5, 3.2])
        turn_scores = [
            torch.tensor([5.0, 4.0]),
            torch.tensor([6.0, 5.0]),
            torch.tensor([7.0, 6.0]),
        ]

        losses = loss_fn(critic_score, turn_scores=turn_scores)

        assert "trajectory" in losses


class TestLossNumericalStability:
    """Test numerical stability of loss functions."""

    def test_critic_loss_with_zeros(self):
        """Critic loss should handle zero predictions."""
        loss_fn = CriticScoreLoss()
        predicted = torch.zeros(4)
        target = torch.zeros(4)

        loss = loss_fn(predicted, target)

        assert loss.item() == 0.0
        assert not torch.isnan(loss)

    def test_reward_loss_with_extreme_scores(self):
        """Reward loss should handle extreme score values."""
        loss_fn = RewardLoss()

        # Very high score
        high_score = torch.tensor([10.0])
        high_loss = loss_fn(high_score)
        assert not torch.isnan(high_loss)
        assert not torch.isinf(high_loss)

        # Very low score
        low_score = torch.tensor([0.0])
        low_loss = loss_fn(low_score)
        assert not torch.isnan(low_loss)
        assert not torch.isinf(low_loss)

    def test_empty_batch_handling(self):
        """Loss functions should handle empty tensors gracefully."""
        loss_fn = CriticScoreLoss()

        # Empty batch
        predicted = torch.tensor([])
        target = torch.tensor([])

        # Should either return 0 or raise a clear error
        try:
            loss = loss_fn(predicted, target)
            # nan is acceptable for empty batch
            assert loss.item() == 0.0 or torch.isnan(loss)
        except (ValueError, RuntimeError):
            pass  # Acceptable to raise error for empty batch

    def test_kl_divergence_with_extreme_logits(self):
        """KL divergence should handle extreme logit values."""
        loss_fn = KLDivergenceLoss(beta=0.1)

        # Very large logits
        large_logits = torch.ones(2, 5, 100) * 100

        kl = loss_fn(large_logits, large_logits)

        assert not torch.isnan(kl)
        # KL of same distribution should be ~0
        assert kl.item() == pytest.approx(0.0, abs=1e-3)
