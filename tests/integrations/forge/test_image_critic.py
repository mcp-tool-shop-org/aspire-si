"""
Tests for the Forge image critic models (CLIP head, latent critic, loss).

All models are tiny and run on CPU; the CLIP encoder is a fake module that
exposes ``encode_image`` like an OpenCLIP model.
"""

import os

os.environ["XFORMERS_DISABLED"] = "1"

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from integrations.forge.image_critic import (
    CLIPImageCritic,
    ImageCriticLoss,
    ImageCriticOutput,
    LatentCritic,
)

CLIP_DIM = 16
HIDDEN = 8
DIM_NAMES = ["aesthetic", "composition", "color", "lighting", "style", "prompt_adherence"]


class FakeCLIP(nn.Module):
    """Minimal CLIP stand-in: flattens a [B, 3, 2, 2] image and projects it."""

    def __init__(self, with_config: bool = True):
        super().__init__()
        self.proj = nn.Linear(12, CLIP_DIM)
        if with_config:
            self.config = SimpleNamespace(hidden_size=CLIP_DIM)

    def encode_image(self, images):
        return self.proj(images.flatten(1))


def _make_critic(**kwargs):
    torch.manual_seed(0)
    return CLIPImageCritic(FakeCLIP(), hidden_dim=HIDDEN, **kwargs)


def _images(batch=3):
    torch.manual_seed(1)
    return torch.randn(batch, 3, 2, 2)


class TestImageCriticOutput:
    """Tests for the output dataclass."""

    def test_optional_fields_default_to_none(self):
        out = ImageCriticOutput(overall_score=torch.tensor([1.0]))

        assert out.aesthetic_score is None
        assert out.reasoning_embedding is None
        assert out.features is None


class TestCLIPImageCritic:
    """Tests for CLIPImageCritic."""

    def test_clip_dim_from_config(self):
        assert _make_critic().clip_dim == CLIP_DIM

    def test_clip_dim_defaults_to_768_without_config(self):
        critic = CLIPImageCritic(FakeCLIP(with_config=False), hidden_dim=HIDDEN)

        assert critic.clip_dim == 768
        assert critic.mlp[0].in_features == 768

    def test_clip_is_frozen(self):
        critic = _make_critic()

        assert all(not p.requires_grad for p in critic.clip_model.parameters())

    def test_mlp_layer_count_follows_num_layers(self):
        for n in (1, 2, 3):
            critic = _make_critic(num_layers=n)
            linears = [m for m in critic.mlp if isinstance(m, nn.Linear)]
            assert len(linears) == n
            assert linears[0].in_features == CLIP_DIM
            assert all(m.in_features == HIDDEN for m in linears[1:])

    def test_dimension_heads_cover_all_dimensions(self):
        assert list(_make_critic().dimension_heads.keys()) == DIM_NAMES

    def test_bias_free_linear_layers_are_handled(self):
        critic = _make_critic()
        critic.overall_head = nn.Linear(HIDDEN, 1, bias=False)

        critic._init_weights()

        assert critic.overall_head.bias is None
        assert critic(_images(2)).overall_score.shape == (2,)

    def test_head_biases_start_at_zero(self):
        critic = _make_critic()

        assert torch.all(critic.overall_head.bias == 0)
        assert all(torch.all(h.bias == 0) for h in critic.dimension_heads.values())

    def test_init_does_not_touch_frozen_clip_weights(self):
        clip = FakeCLIP()
        before = {k: v.clone() for k, v in clip.state_dict().items()}

        CLIPImageCritic(clip, hidden_dim=HIDDEN)

        for key, value in clip.state_dict().items():
            assert torch.equal(value, before[key]), key

    def test_forward_shapes_and_ranges(self):
        critic = _make_critic().eval()
        out = critic(_images(3))

        assert out.overall_score.shape == (3,)
        for name in DIM_NAMES:
            score = getattr(out, f"{name}_score")
            assert score.shape == (3,)
            assert torch.all(score > 0) and torch.all(score < 10)
        assert torch.all(out.overall_score > 0) and torch.all(out.overall_score < 10)
        assert out.reasoning_embedding.shape == (3, HIDDEN)

    def test_reasoning_embedding_is_unit_norm(self):
        out = _make_critic().eval()(_images(4))

        norms = out.reasoning_embedding.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(4), atol=1e-5)

    def test_features_only_when_requested(self):
        critic = _make_critic().eval()

        assert critic(_images(2)).features is None
        features = critic(_images(2), return_features=True).features
        assert features.shape == (2, HIDDEN)

    def test_forward_is_deterministic_in_eval(self):
        critic = _make_critic().eval()
        images = _images(2)

        assert torch.equal(critic(images).overall_score, critic(images).overall_score)

    def test_zero_head_gives_midpoint_score(self):
        critic = _make_critic().eval()
        nn.init.zeros_(critic.overall_head.weight)

        out = critic(_images(2))

        assert torch.allclose(out.overall_score, torch.full((2,), 5.0))

    def test_gradients_reach_heads_but_not_clip(self):
        critic = _make_critic()
        critic.train()
        out = critic(_images(3))
        out.overall_score.sum().backward()

        assert critic.overall_head.weight.grad is not None
        assert critic.mlp[0].weight.grad is not None
        assert all(p.grad is None for p in critic.clip_model.parameters())

    def test_predict_score_is_mean_float_and_switches_to_eval(self):
        critic = _make_critic()
        critic.train()
        images = _images(4)

        score = critic.predict_score(images)

        assert isinstance(score, float)
        assert critic.training is False
        expected = critic(images).overall_score.mean().item()
        assert score == pytest.approx(expected)

    def test_trainable_parameters_exclude_clip(self):
        critic = _make_critic()
        params = critic.get_trainable_parameters()
        clip_ids = {id(p) for p in critic.clip_model.parameters()}

        assert params and all(id(p) not in clip_ids for p in params)
        assert all(p.requires_grad for p in params)
        # every non-CLIP parameter is trainable and listed exactly once
        head_ids = {id(p) for n, p in critic.named_parameters() if not n.startswith("clip_model")}
        assert {id(p) for p in params} == head_ids
        assert len(params) == len(head_ids)


class TestLatentCritic:
    """Tests for LatentCritic."""

    @pytest.fixture
    def critic(self):
        torch.manual_seed(0)
        return LatentCritic(latent_channels=4, hidden_dim=16).eval()

    @pytest.fixture
    def latents(self):
        torch.manual_seed(2)
        return torch.randn(2, 4, 8, 8)

    def test_forward_shapes(self, critic, latents):
        out = critic(latents)

        assert out.overall_score.shape == (2,)
        assert torch.all(out.overall_score > 0) and torch.all(out.overall_score < 10)
        assert out.reasoning_embedding.shape == (2, 16)
        assert out.features.shape == (2, 16)

    def test_per_dimension_scores_not_produced(self, critic, latents):
        out = critic(latents)

        assert out.aesthetic_score is None
        assert out.prompt_adherence_score is None

    def test_reasoning_embedding_is_unit_norm(self, critic, latents):
        out = critic(latents)

        assert torch.allclose(out.reasoning_embedding.norm(dim=-1), torch.ones(2), atol=1e-5)

    def test_timestep_is_accepted_and_ignored(self, critic, latents):
        base = critic(latents).overall_score
        with_t = critic(latents, timestep=torch.tensor([10, 20])).overall_score

        assert torch.equal(base, with_t)

    def test_accepts_other_latent_sizes_and_batch_one(self, critic):
        out = critic(torch.randn(1, 4, 16, 16))

        assert out.overall_score.shape == (1,)

    def test_custom_latent_channels(self):
        critic = LatentCritic(latent_channels=3, hidden_dim=16).eval()

        assert critic(torch.randn(2, 3, 8, 8)).overall_score.shape == (2,)

    def test_guidance_scale_matches_formula(self, critic, latents):
        score = critic(latents).overall_score
        guidance = critic.get_guidance_scale(latents, target_score=10.0)

        expected = torch.clamp((10.0 - score) / 10.0, 0.0, 1.0)
        assert torch.allclose(guidance, expected)
        assert guidance.shape == (2,)

    def test_guidance_scale_is_zero_when_score_exceeds_target(self, critic, latents):
        guidance = critic.get_guidance_scale(latents, target_score=0.5)

        assert torch.all(guidance == 0.0)

    def test_guidance_scale_stays_in_unit_interval_for_huge_target(self, critic, latents):
        guidance = critic.get_guidance_scale(latents, target_score=1000.0)

        # scores are in (0, 10), so the guidance approaches but never exceeds 1
        assert torch.all(guidance > 0.99) and torch.all(guidance <= 1.0)

    def test_guidance_scale_switches_to_eval_and_disables_grad(self, latents):
        critic = LatentCritic(latent_channels=4, hidden_dim=16)
        critic.train()

        guidance = critic.get_guidance_scale(latents)

        assert critic.training is False
        assert guidance.requires_grad is False

    def test_lower_scoring_latents_get_more_guidance(self, critic):
        low = torch.zeros(1, 4, 8, 8)
        # force a known low and high score through the head bias
        with torch.no_grad():
            critic.overall_head.weight.zero_()
            critic.overall_head.bias.fill_(-5.0)
        g_low_quality = critic.get_guidance_scale(low)
        with torch.no_grad():
            critic.overall_head.bias.fill_(5.0)
        g_high_quality = critic.get_guidance_scale(low)

        assert g_low_quality.item() > g_high_quality.item()


def _output(overall, **kwargs):
    return ImageCriticOutput(overall_score=overall, **kwargs)


class TestImageCriticLoss:
    """Tests for ImageCriticLoss."""

    def test_overall_only(self):
        pred = _output(torch.tensor([2.0, 8.0]))
        target = torch.tensor([3.0, 6.0])

        losses = ImageCriticLoss()(pred, target)

        expected = F.smooth_l1_loss(pred.overall_score, target)
        assert set(losses) == {"overall", "total"}
        assert losses["overall"].item() == pytest.approx(expected.item())
        assert losses["total"].item() == pytest.approx(expected.item())
        # smooth L1 with diffs 1 and 2: (0.5*1 + (2 - 0.5)) / 2
        assert losses["overall"].item() == pytest.approx(1.0)

    def test_score_weight_scales_total(self):
        pred = _output(torch.tensor([2.0, 8.0]))
        target = torch.tensor([3.0, 6.0])

        losses = ImageCriticLoss(score_weight=2.5)(pred, target)

        assert losses["total"].item() == pytest.approx(2.5 * losses["overall"].item())

    def test_dimension_loss_averages_matching_dimensions(self):
        pred = _output(
            torch.tensor([5.0]),
            aesthetic_score=torch.tensor([4.0]),
            color_score=torch.tensor([9.0]),
        )
        targets = {"aesthetic": torch.tensor([5.0]), "color": torch.tensor([7.0])}

        losses = ImageCriticLoss(dimension_weight=0.5)(pred, torch.tensor([5.0]), targets)

        # aesthetic diff 1 -> 0.5; color diff 2 -> 1.5; mean 1.0
        assert losses["dimensions"].item() == pytest.approx(1.0)
        assert losses["total"].item() == pytest.approx(0.0 + 0.5 * 1.0)

    def test_unknown_dimension_names_are_skipped(self):
        pred = _output(torch.tensor([5.0]), aesthetic_score=torch.tensor([4.0]))
        targets = {"technical": torch.tensor([1.0]), "bogus": torch.tensor([1.0])}

        losses = ImageCriticLoss()(pred, torch.tensor([5.0]), targets)

        assert "dimensions" not in losses

    def test_unknown_names_do_not_pollute_known_ones(self):
        pred = _output(torch.tensor([5.0]), aesthetic_score=torch.tensor([4.0]))
        targets = {"bogus": torch.tensor([1.0]), "aesthetic": torch.tensor([5.0])}

        losses = ImageCriticLoss()(pred, torch.tensor([5.0]), targets)

        assert losses["dimensions"].item() == pytest.approx(0.5)

    def test_dimension_targets_ignored_when_prediction_has_no_dimensions(self):
        pred = _output(torch.tensor([5.0]))  # e.g. LatentCritic output
        targets = {"aesthetic": torch.tensor([5.0])}

        losses = ImageCriticLoss()(pred, torch.tensor([5.0]), targets)

        assert "dimensions" not in losses

    def test_reasoning_loss_is_one_minus_cosine(self):
        emb = F.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), dim=-1)
        target = F.normalize(torch.tensor([[1.0, 0.0], [1.0, 0.0]]), dim=-1)
        pred = _output(torch.tensor([5.0, 5.0]), reasoning_embedding=emb)

        losses = ImageCriticLoss(reasoning_weight=0.3)(pred, torch.tensor([5.0, 5.0]), None, target)

        # cosine similarities are 1 and 0 -> mean 0.5 -> loss 0.5
        assert losses["reasoning"].item() == pytest.approx(0.5)
        assert losses["total"].item() == pytest.approx(0.3 * 0.5)

    def test_identical_reasoning_gives_zero_loss(self):
        emb = F.normalize(torch.randn(3, 5), dim=-1)
        pred = _output(torch.tensor([1.0] * 3), reasoning_embedding=emb)

        losses = ImageCriticLoss()(pred, torch.tensor([1.0] * 3), None, emb.clone())

        assert losses["reasoning"].item() == pytest.approx(0.0, abs=1e-6)

    def test_reasoning_target_ignored_without_prediction_embedding(self):
        pred = _output(torch.tensor([1.0]))

        losses = ImageCriticLoss()(pred, torch.tensor([1.0]), None, torch.randn(1, 4))

        assert "reasoning" not in losses

    def test_total_combines_all_terms(self):
        emb = F.normalize(torch.tensor([[1.0, 0.0]]), dim=-1)
        target_emb = F.normalize(torch.tensor([[0.0, 1.0]]), dim=-1)
        pred = _output(
            torch.tensor([4.0]),
            aesthetic_score=torch.tensor([6.0]),
            reasoning_embedding=emb,
        )
        loss_fn = ImageCriticLoss(score_weight=2.0, dimension_weight=3.0, reasoning_weight=4.0)

        losses = loss_fn(pred, torch.tensor([5.0]), {"aesthetic": torch.tensor([5.0])}, target_emb)

        # overall diff 1 -> 0.5, dimension diff 1 -> 0.5, reasoning cos 0 -> 1.0
        assert losses["total"].item() == pytest.approx(2.0 * 0.5 + 3.0 * 0.5 + 4.0 * 1.0)
        assert set(losses) == {"overall", "dimensions", "reasoning", "total"}

    def test_end_to_end_gradient_to_critic_heads(self):
        torch.manual_seed(0)
        critic = _make_critic()
        critic.train()
        out = critic(_images(3))
        targets = {name: torch.full((3,), 6.0) for name in DIM_NAMES}
        target_emb = F.normalize(torch.randn(3, HIDDEN), dim=-1)

        losses = ImageCriticLoss()(out, torch.full((3,), 7.0), targets, target_emb)
        losses["total"].backward()

        assert torch.isfinite(losses["total"])
        assert critic.overall_head.weight.grad is not None
        assert critic.dimension_heads["color"].weight.grad is not None
        assert critic.reasoning_head.weight.grad is not None
