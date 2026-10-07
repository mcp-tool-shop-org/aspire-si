"""Coverage-closing tests for aspire.losses.student."""

import pytest
import torch
import torch.nn.functional as F

from aspire.losses.student import (
    CoherenceLoss,
    ContrastiveLoss,
    KLDivergenceLoss,
    RewardLoss,
    StudentLoss,
    TrajectoryLoss,
)


class TestKLDivergenceMasking:
    def test_masked_kl_averages_only_over_valid_positions(self):
        torch.manual_seed(0)
        student = torch.randn(2, 3, 5)
        reference = torch.randn(2, 3, 5)
        mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
        loss = KLDivergenceLoss(beta=0.5)(student, reference, mask)

        per_pos = F.kl_div(
            F.log_softmax(student, -1), F.softmax(reference, -1), reduction="none"
        ).sum(-1)
        expected = 0.5 * (per_pos * mask).sum() / 3  # three valid positions
        torch.testing.assert_close(loss, expected)

    def test_masked_differs_from_unmasked_when_padding_diverges(self):
        student = torch.zeros(1, 2, 4)
        reference = torch.zeros(1, 2, 4)
        reference[0, 1] = torch.tensor([5.0, 0.0, 0.0, 0.0])  # only the padded position diverges
        kl = KLDivergenceLoss(beta=1.0)
        assert kl(student, reference).item() > 0
        assert kl(student, reference, torch.tensor([[1, 0]])).item() == pytest.approx(0.0, abs=1e-6)

    def test_all_masked_does_not_divide_by_zero(self):
        loss = KLDivergenceLoss()(torch.randn(1, 2, 3), torch.randn(1, 2, 3), torch.zeros(1, 2))
        assert torch.isfinite(loss) and loss.item() == 0.0

    def test_identical_distributions_give_zero(self):
        logits = torch.randn(2, 3, 4)
        assert KLDivergenceLoss()(logits, logits.clone()).item() == pytest.approx(0.0, abs=1e-6)


class TestStudentLossBranches:
    def _inputs(self):
        torch.manual_seed(1)
        return {
            "critic_score": torch.tensor([6.0, 8.0]),
            "student_embedding": torch.randn(2, 4),
            "teacher_embedding": torch.randn(2, 4),
            "turn_scores": [torch.tensor([5.0, 6.0]), torch.tensor([7.0, 8.0])],
            "student_logits": torch.randn(2, 4, 6),
            "labels": torch.randint(0, 6, (2, 4)),
            "reference_logits": torch.randn(2, 4, 6),
            "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]]),
        }

    def test_reward_only(self):
        sl = StudentLoss()
        out = sl(torch.tensor([3.0]))
        assert set(out) == {"reward", "total"}
        torch.testing.assert_close(out["total"], sl.reward_weight * out["reward"])

    def test_single_turn_score_skips_trajectory(self):
        out = StudentLoss()(torch.tensor([3.0]), turn_scores=[torch.tensor([3.0])])
        assert "trajectory" not in out

    def test_embeddings_need_both_sides_for_contrastive(self):
        sl = StudentLoss()
        assert "contrastive" not in sl(torch.tensor([3.0]), student_embedding=torch.randn(1, 4))
        assert "contrastive" not in sl(torch.tensor([3.0]), teacher_embedding=torch.randn(1, 4))

    def test_coherence_requires_logits_and_labels(self):
        sl = StudentLoss()
        out = sl(torch.tensor([3.0]), student_logits=torch.randn(1, 3, 5))
        assert "coherence" not in out and "kl" not in out
        out = sl(torch.tensor([3.0]), labels=torch.randint(0, 5, (1, 3)))
        assert "coherence" not in out

    def test_total_is_weighted_sum_of_all_components(self):
        sl = StudentLoss(
            reward_weight=2.0,
            contrastive_weight=0.25,
            trajectory_weight=0.5,
            coherence_weight=0.75,
            kl_weight=0.3,
        )
        out = sl(**self._inputs())
        assert set(out) == {"reward", "contrastive", "trajectory", "coherence", "kl", "total"}
        expected = (
            2.0 * out["reward"]
            + 0.25 * out["contrastive"]
            + 0.5 * out["trajectory"]
            + 0.75 * out["coherence"]
            + out["kl"]  # beta already folded into KL loss
        )
        torch.testing.assert_close(out["total"], expected)

    def test_kl_only_adds_when_reference_logits_given(self):
        inputs = self._inputs()
        inputs.pop("reference_logits")
        out = StudentLoss()(**inputs)
        assert "kl" not in out and "coherence" in out

    def test_kl_weight_scales_kl_term(self):
        inputs = self._inputs()
        small = StudentLoss(kl_weight=0.1)(**inputs)["kl"]
        large = StudentLoss(kl_weight=1.0)(**inputs)["kl"]
        torch.testing.assert_close(large, small * 10, rtol=1e-4, atol=1e-6)

    def test_total_is_differentiable_through_all_components(self):
        inputs = self._inputs()
        for key in ("critic_score", "student_embedding", "student_logits"):
            inputs[key] = inputs[key].clone().requires_grad_(True)
        out = StudentLoss()(**inputs)
        out["total"].backward()
        assert inputs["student_logits"].grad is not None
        assert inputs["student_embedding"].grad is not None
        assert inputs["critic_score"].grad is not None


class TestComponentEdges:
    def test_trajectory_with_single_score_is_zero_on_same_device(self):
        score = torch.tensor([5.0, 6.0])
        out = TrajectoryLoss()([score])
        assert out.item() == 0.0 and out.device == score.device

    def test_trajectory_rewards_improvement_and_penalizes_decline(self):
        up = TrajectoryLoss()([torch.tensor([1.0]), torch.tensor([3.0])])
        down = TrajectoryLoss()([torch.tensor([3.0]), torch.tensor([1.0])])
        assert up.item() == pytest.approx(-2.0)
        assert down.item() == pytest.approx(2.0 + 2.0)

    def test_reward_loss_policy_gradient_term(self):
        score = torch.tensor([8.0])
        logprobs = torch.tensor([[-1.0, -3.0]])
        base = RewardLoss()(score)
        with_pg = RewardLoss()(score, response_logprobs=logprobs)
        # reward = 0.8 - 0.5 = 0.3, mean logprob = -2 => loss - (0.3 * -2)
        assert with_pg.item() == pytest.approx(base.item() + 0.6)

    def test_contrastive_triplet_vs_pull(self):
        s = torch.tensor([[1.0, 0.0]])
        t = torch.tensor([[1.0, 0.0]])
        n = torch.tensor([[1.0, 0.0]])
        assert ContrastiveLoss()(s, t).item() == pytest.approx(0.0, abs=1e-6)
        # pos_sim = neg_sim = 1 -> relu(margin) = margin
        assert ContrastiveLoss(margin=0.5)(s, t, n).item() == pytest.approx(0.5)

    def test_coherence_masked_ignores_padded_tokens(self):
        logits = torch.zeros(1, 3, 4)
        labels = torch.tensor([[0, 1, 2]])
        logits[0, 1, 2] = 50.0  # position 1 predicts label[2]=2 confidently
        # mask drops the first predicted token (index 1 in the shifted view), keeps the second
        loss = CoherenceLoss(target_perplexity=1.0)(logits, labels, torch.tensor([[1, 0, 1]]))
        assert loss.item() == pytest.approx(0.0, abs=1e-3)
        unmasked = CoherenceLoss(target_perplexity=1.0)(logits, labels)
        assert unmasked.item() > loss.item()
