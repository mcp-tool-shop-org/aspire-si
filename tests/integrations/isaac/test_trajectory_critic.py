"""
Tests for integrations.isaac.trajectory_critic (encoder, heads, critic, loss).
"""

import math

import pytest
import torch

from integrations.isaac.config import CriticArchitecture, CriticConfig
from integrations.isaac.trajectory_critic import (
    CriticOutput,
    MotionCriticHead,
    PositionalEncoding,
    TrajectoryCritic,
    TrajectoryCriticLoss,
    TrajectoryEncoder,
)

STATE_DIM = 3
ACTION_DIM = 2
HIDDEN = 16
MAX_LEN = 8


def make_encoder(arch=CriticArchitecture.TRANSFORMER, **overrides):
    kwargs = dict(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=HIDDEN,
        num_layers=2,
        num_heads=2,
        max_seq_len=MAX_LEN,
        architecture=arch,
        dropout=0.0,
    )
    kwargs.update(overrides)
    return TrajectoryEncoder(**kwargs).eval()


def make_config(arch=CriticArchitecture.TRANSFORMER, **overrides):
    kwargs = dict(
        architecture=arch,
        hidden_dim=HIDDEN,
        num_layers=2,
        num_heads=2,
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        max_trajectory_len=MAX_LEN,
        dropout=0.0,
    )
    kwargs.update(overrides)
    return CriticConfig(**kwargs)


def batch(b=2, seq=5):
    torch.manual_seed(0)
    return torch.randn(b, seq, STATE_DIM), torch.randn(b, seq, ACTION_DIM)


class TestPositionalEncoding:
    def test_buffer_is_sinusoidal(self):
        pe = PositionalEncoding(d_model=8, max_len=10, dropout=0.0).pe
        assert pe.shape == (1, 10, 8)
        assert pe[0, 0, 0::2].tolist() == [0.0] * 4          # sin(0)
        assert pe[0, 0, 1::2].tolist() == [1.0] * 4          # cos(0)
        assert pe[0, 3, 0].item() == pytest.approx(math.sin(3.0))
        assert pe[0, 3, 1].item() == pytest.approx(math.cos(3.0))

    def test_adds_encoding_for_the_actual_sequence_length(self):
        module = PositionalEncoding(d_model=8, max_len=10, dropout=0.0)
        x = torch.zeros(2, 4, 8)
        out = module(x)
        assert out.shape == (2, 4, 8)
        assert torch.allclose(out[0], module.pe[0, :4])

    def test_dropout_only_active_in_training(self):
        module = PositionalEncoding(d_model=8, max_len=10, dropout=0.5).eval()
        x = torch.zeros(1, 4, 8)
        assert torch.allclose(module(x), module.pe[:, :4])

    def test_pe_is_a_buffer_not_a_parameter(self):
        module = PositionalEncoding(d_model=8, max_len=10)
        assert list(module.parameters()) == []
        assert "pe" in dict(module.named_buffers())


class TestTrajectoryEncoder:
    @pytest.mark.parametrize(
        "arch",
        [CriticArchitecture.TRANSFORMER, CriticArchitecture.LSTM, CriticArchitecture.TCN],
    )
    def test_sequence_architectures_preserve_sequence_length(self, arch):
        states, actions = batch(2, 5)
        out, attn = make_encoder(arch)(states, actions)
        assert out.shape == (2, 5, HIDDEN)
        assert attn is None
        assert torch.isfinite(out).all()

    def test_mlp_pools_to_single_step(self):
        states, actions = batch(2, 5)
        out, attn = make_encoder(CriticArchitecture.MLP)(states, actions)
        assert out.shape == (2, 1, HIDDEN)
        assert attn is None

    def test_mlp_pads_short_and_truncates_long_inputs(self):
        enc = make_encoder(CriticArchitecture.MLP)
        short = enc(*batch(2, 3))[0]
        exact = enc(*batch(2, MAX_LEN))[0]
        long = enc(*batch(2, MAX_LEN + 4))[0]
        assert short.shape == exact.shape == long.shape == (2, 1, HIDDEN)

    def test_mlp_truncation_ignores_steps_past_max_len(self):
        enc = make_encoder(CriticArchitecture.MLP)
        states, actions = batch(1, MAX_LEN + 3)
        longer = enc(states, actions)[0]
        clipped = enc(states[:, :MAX_LEN], actions[:, :MAX_LEN])[0]
        assert torch.allclose(longer, clipped, atol=1e-6)
        assert enc.max_seq_len == MAX_LEN

    def test_tcn_has_one_conv_per_layer_with_exponential_dilation(self):
        enc = make_encoder(CriticArchitecture.TCN, num_layers=3)
        convs = [m for m in enc.encoder if isinstance(m, torch.nn.Conv1d)]
        assert [c.dilation[0] for c in convs] == [1, 2, 4]

    def test_lstm_is_bidirectional_with_half_hidden_size(self):
        enc = make_encoder(CriticArchitecture.LSTM)
        assert enc.encoder.bidirectional
        assert enc.encoder.hidden_size == HIDDEN // 2

    def test_lstm_single_layer_skips_dropout(self):
        enc = make_encoder(CriticArchitecture.LSTM, num_layers=1, dropout=0.3)
        assert enc.encoder.dropout == 0

    def test_transformer_mask_hides_padded_steps(self):
        enc = make_encoder(CriticArchitecture.TRANSFORMER)
        states, actions = batch(1, 6)
        mask = torch.tensor([[True, True, True, False, False, False]])
        base = enc(states, actions, mask)[0]
        # changing padded timesteps must not change the valid positions
        states2, actions2 = states.clone(), actions.clone()
        states2[:, 3:] += 50.0
        actions2[:, 3:] -= 50.0
        perturbed = enc(states2, actions2, mask)[0]
        assert torch.allclose(base[:, :3], perturbed[:, :3], atol=1e-5)

    def test_transformer_without_mask_sees_every_step(self):
        enc = make_encoder(CriticArchitecture.TRANSFORMER)
        states, actions = batch(1, 6)
        base = enc(states, actions)[0]
        states2 = states.clone()
        states2[:, 5] += 50.0
        assert not torch.allclose(base[:, :3], enc(states2, actions)[0][:, :3], atol=1e-5)

    def test_lstm_mask_with_full_lengths_matches_unmasked(self):
        enc = make_encoder(CriticArchitecture.LSTM)
        states, actions = batch(2, 5)
        unmasked = enc(states, actions)[0]
        masked = enc(states, actions, torch.ones(2, 5, dtype=torch.bool))[0]
        assert masked.shape == unmasked.shape
        assert torch.allclose(masked, unmasked, atol=1e-5)

    def test_lstm_mask_with_ragged_lengths_runs(self):
        enc = make_encoder(CriticArchitecture.LSTM)
        states, actions = batch(2, 5)
        mask = torch.tensor([[True] * 3 + [False] * 2, [True] * 5])
        out = enc(states, actions, mask)[0]
        assert out.shape == (2, 5, HIDDEN)

    def test_mask_is_ignored_by_tcn(self):
        enc = make_encoder(CriticArchitecture.TCN)
        states, actions = batch(1, 5)
        mask = torch.tensor([[True, True, False, False, False]])
        assert torch.allclose(enc(states, actions, mask)[0], enc(states, actions)[0])

    def test_output_is_layer_normalised(self):
        out = make_encoder(CriticArchitecture.TCN)(*batch(2, 5))[0]
        assert out.mean(dim=-1).abs().max().item() < 1e-4


class TestMotionCriticHead:
    def head(self, **kwargs):
        kwargs.setdefault("hidden_dim", HIDDEN)
        kwargs.setdefault("action_dim", ACTION_DIM)
        kwargs.setdefault("reasoning_dim", 12)
        return MotionCriticHead(**kwargs).eval()

    def test_sequence_input_produces_every_output(self):
        out = self.head()(torch.randn(3, 5, HIDDEN))
        assert out.score.shape == (3,)
        assert set(out.dimension_scores) == {"safety", "efficiency", "smoothness", "goal_achievement"}
        assert all(v.shape == (3,) for v in out.dimension_scores.values())
        assert out.timestep_scores.shape == (3, 5)
        assert out.action_improvement.shape == (3, 5, ACTION_DIM)
        assert out.reasoning_embedding.shape == (3, 12)
        assert out.attention_weights is None

    def test_scores_are_bounded_zero_to_ten(self):
        out = self.head()(torch.randn(4, 5, HIDDEN) * 100)
        for t in (out.score, out.timestep_scores, *out.dimension_scores.values()):
            assert t.min() >= 0.0 and t.max() <= 10.0

    def test_improvement_is_bounded_by_tanh(self):
        out = self.head()(torch.randn(2, 4, HIDDEN) * 100)
        assert out.action_improvement.abs().max() <= 1.0

    def test_pooled_input_skips_sequence_heads(self):
        out = self.head()(torch.randn(3, HIDDEN))
        assert out.score.shape == (3,)
        assert out.timestep_scores is None
        assert out.action_improvement is None
        assert out.reasoning_embedding.shape == (3, 12)

    def test_pooling_modes_select_the_expected_positions(self):
        head = self.head(predict_dimensions=False, predict_timesteps=False,
                         predict_improvement=False, predict_reasoning=False)
        enc = torch.randn(2, 4, HIDDEN)
        mean = head(enc, pool="mean").score
        cls = head(enc, pool="cls").score
        last = head(enc, pool="last").score
        assert torch.allclose(cls, head(enc[:, 0]).score)
        assert torch.allclose(last, head(enc[:, -1]).score)
        assert torch.allclose(mean, head(enc.mean(dim=1)).score)
        assert not torch.allclose(cls, last)

    def test_unknown_pool_rejected(self):
        with pytest.raises(ValueError, match="Unknown pool: max"):
            self.head()(torch.randn(2, 4, HIDDEN), pool="max")

    def test_disabled_heads_return_none(self):
        head = self.head(predict_score=False, predict_dimensions=False, predict_timesteps=False,
                         predict_improvement=False, predict_reasoning=False)
        assert head.score_head is None and head.dimension_heads is None
        assert head.timestep_head is None and head.improvement_head is None
        assert head.reasoning_head is None
        out = head(torch.randn(2, 4, HIDDEN))
        assert out.score is None and out.dimension_scores is None
        assert out.timestep_scores is None and out.action_improvement is None
        assert out.reasoning_embedding is None

    def test_improvement_head_requires_positive_action_dim(self):
        head = self.head(action_dim=0, predict_improvement=True)
        assert head.improvement_head is None
        assert head(torch.randn(2, 4, HIDDEN)).action_improvement is None


class TestTrajectoryCritic:
    def test_default_config_is_used_when_none_given(self):
        critic = TrajectoryCritic()
        assert isinstance(critic.config, CriticConfig)
        assert critic.encoder.hidden_dim == 256

    @pytest.mark.parametrize("arch", list(CriticArchitecture))
    def test_forward_shapes_for_every_architecture(self, arch):
        critic = TrajectoryCritic(make_config(arch)).eval()
        states, actions = batch(3, 5)
        out = critic(states, actions)
        assert out.score.shape == (3,)
        assert 0.0 <= out.score.min() and out.score.max() <= 10.0
        assert out.reasoning_embedding.shape == (3, 256)
        assert out.attention_weights is None
        if arch is CriticArchitecture.MLP:
            assert out.timestep_scores.shape == (3, 1)
        else:
            assert out.timestep_scores.shape == (3, 5)
            assert out.action_improvement.shape == (3, 5, ACTION_DIM)

    def test_predict_flags_are_forwarded_to_heads(self):
        critic = TrajectoryCritic(make_config(predict_score=False, predict_reasoning=False,
                                              predict_improvement=False))
        assert critic.heads.score_head is None
        assert critic.heads.reasoning_head is None
        assert critic.heads.improvement_head is None
        assert critic.heads.dimension_heads is not None   # always on

    def test_score_trajectory_matches_forward_score(self):
        critic = TrajectoryCritic(make_config()).eval()
        states, actions = batch(2, 5)
        assert torch.allclose(critic.score_trajectory(states, actions), critic(states, actions).score)

    def test_get_action_improvement_returns_head_output(self):
        critic = TrajectoryCritic(make_config(CriticArchitecture.TCN)).eval()
        states, actions = batch(2, 5)
        delta = critic.get_action_improvement(states, actions)
        assert delta.shape == actions.shape
        assert delta.abs().max() <= 1.0
        assert torch.allclose(delta, critic(states, actions).action_improvement)

    def test_get_action_improvement_is_zero_when_head_disabled(self):
        critic = TrajectoryCritic(make_config(predict_improvement=False)).eval()
        states, actions = batch(2, 5)
        delta = critic.get_action_improvement(states, actions)
        assert delta.shape == actions.shape
        assert torch.count_nonzero(delta) == 0

    def test_masked_transformer_forward(self):
        critic = TrajectoryCritic(make_config()).eval()
        states, actions = batch(2, 5)
        mask = torch.tensor([[True] * 3 + [False] * 2, [True] * 5])
        out = critic(states, actions, mask)
        assert out.score.shape == (2,)
        assert torch.isfinite(out.score).all()

    def test_critic_is_trainable_end_to_end(self):
        critic = TrajectoryCritic(make_config(CriticArchitecture.TCN))
        states, actions = batch(4, 5)
        target = torch.tensor([2.0, 4.0, 6.0, 8.0])
        opt = torch.optim.Adam(critic.parameters(), lr=1e-2)
        loss_fn = TrajectoryCriticLoss()
        first = None
        for _ in range(15):
            opt.zero_grad()
            loss = loss_fn(critic(states, actions), target)["total"]
            first = first if first is not None else loss.item()
            loss.backward()
            opt.step()
        assert loss.item() < first

    def test_lstm_masked_improvement_matches_action_shape(self):
        critic = TrajectoryCritic(make_config(CriticArchitecture.LSTM)).eval()
        states, actions = batch(2, 6)
        mask = torch.tensor([[True] * 3 + [False] * 3, [True] * 4 + [False] * 2])
        delta = critic.get_action_improvement(states, actions, mask)
        assert delta.shape == actions.shape


class TestTrajectoryCriticLoss:
    def output(self, **kwargs):
        defaults = dict(score=torch.tensor([2.0, 4.0]))
        defaults.update(kwargs)
        return CriticOutput(**defaults)

    def test_score_loss_is_mse(self):
        losses = TrajectoryCriticLoss()(self.output(), torch.tensor([3.0, 4.0]))
        assert losses["score"].item() == pytest.approx(0.5)
        assert losses["total"].item() == pytest.approx(0.5)
        for key in ("dimensions", "reasoning", "timesteps"):
            assert losses[key].item() == 0.0

    def test_missing_critic_score_contributes_zero(self):
        losses = TrajectoryCriticLoss()(self.output(score=None), torch.tensor([3.0, 4.0]))
        assert losses["score"].item() == 0.0 and losses["total"].item() == 0.0

    def test_dimension_loss_averages_only_matching_dimensions(self):
        out = self.output(dimension_scores={
            "safety": torch.tensor([1.0, 1.0]),
            "efficiency": torch.tensor([3.0, 3.0]),
            "smoothness": torch.tensor([9.0, 9.0]),   # no teacher label -> ignored
        })
        teacher = {"safety": torch.tensor([2.0, 2.0]), "efficiency": torch.tensor([5.0, 5.0])}
        losses = TrajectoryCriticLoss()(out, torch.tensor([2.0, 4.0]), teacher_dimensions=teacher)
        assert losses["dimensions"].item() == pytest.approx((1.0 + 4.0) / 2)

    def test_dimension_loss_zero_when_no_names_overlap(self):
        out = self.output(dimension_scores={"safety": torch.tensor([1.0, 1.0])})
        losses = TrajectoryCriticLoss()(
            out, torch.tensor([2.0, 4.0]), teacher_dimensions={"other": torch.tensor([0.0, 0.0])}
        )
        assert losses["dimensions"].item() == 0.0

    def test_dimension_loss_zero_without_teacher_dimensions(self):
        out = self.output(dimension_scores={"safety": torch.tensor([1.0, 1.0])})
        assert TrajectoryCriticLoss()(out, torch.tensor([2.0, 4.0]))["dimensions"].item() == 0.0

    def test_reasoning_loss_is_one_minus_cosine(self):
        out = self.output(reasoning_embedding=torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
        teacher = torch.tensor([[1.0, 0.0], [1.0, 0.0]])   # cosine 1 and 0
        losses = TrajectoryCriticLoss()(out, torch.tensor([2.0, 4.0]),
                                        teacher_reasoning_embedding=teacher)
        assert losses["reasoning"].item() == pytest.approx(0.5)

    def test_reasoning_loss_zero_without_teacher_embedding(self):
        out = self.output(reasoning_embedding=torch.ones(2, 2))
        assert TrajectoryCriticLoss()(out, torch.tensor([2.0, 4.0]))["reasoning"].item() == 0.0

    def test_timestep_loss_is_mse(self):
        out = self.output(timestep_scores=torch.tensor([[1.0, 1.0], [1.0, 1.0]]))
        losses = TrajectoryCriticLoss()(out, torch.tensor([2.0, 4.0]),
                                        teacher_timestep_scores=torch.tensor([[2.0, 2.0], [3.0, 3.0]]))
        assert losses["timesteps"].item() == pytest.approx((1 + 1 + 4 + 4) / 4)

    def test_timestep_loss_zero_without_teacher_scores(self):
        out = self.output(timestep_scores=torch.ones(2, 2))
        assert TrajectoryCriticLoss()(out, torch.tensor([2.0, 4.0]))["timesteps"].item() == 0.0

    def test_total_applies_component_weights(self):
        loss_fn = TrajectoryCriticLoss(score_weight=2.0, dimension_weight=3.0,
                                       reasoning_weight=5.0, timestep_weight=7.0)
        out = self.output(
            dimension_scores={"safety": torch.tensor([1.0, 1.0])},
            reasoning_embedding=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            timestep_scores=torch.zeros(2, 2),
        )
        losses = loss_fn(
            out,
            torch.tensor([3.0, 4.0]),                              # mse 0.5
            teacher_dimensions={"safety": torch.tensor([2.0, 2.0])},   # mse 1.0
            teacher_reasoning_embedding=torch.tensor([[1.0, 0.0], [1.0, 0.0]]),   # 0.5
            teacher_timestep_scores=torch.ones(2, 2),              # 1.0
        )
        assert losses["total"].item() == pytest.approx(2 * 0.5 + 3 * 1.0 + 5 * 0.5 + 7 * 1.0)

    def test_perfect_prediction_has_zero_loss(self):
        out = self.output()
        assert TrajectoryCriticLoss()(out, torch.tensor([2.0, 4.0]))["total"].item() == 0.0
