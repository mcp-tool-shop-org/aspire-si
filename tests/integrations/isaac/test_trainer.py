"""
Tests for integrations.isaac.trainer (dataset, policy, and the ASPIRE Isaac training loop).
"""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from integrations.isaac.config import (
    CriticArchitecture,
    CriticConfig,
    IsaacAspireConfig,
    TrainingConfig,
)
from integrations.isaac.isaac_wrapper import (
    AspireIsaacEnv,
    DummyIsaacEnv,
    StateActionPair,
    Trajectory,
)
from integrations.isaac.motion_teacher import MotionCritique, MotionTeacher
from integrations.isaac.trainer import (
    AspireIsaacTrainer,
    SimplePolicy,
    TrainingMetrics,
    TrajectoryDataset,
)
from integrations.isaac.trajectory_critic import CriticOutput, TrajectoryCritic

STATE_DIM = 4
ACTION_DIM = 2
EPISODE_LEN = 12   # the trainer skips episodes of <= 10 steps


def make_config(tmp_path=None, **training):
    cfg = IsaacAspireConfig(
        critic=CriticConfig(
            architecture=CriticArchitecture.TCN,
            hidden_dim=16,
            num_layers=2,
            num_heads=2,
            max_trajectory_len=16,
            dropout=0.0,
        ),
        training=TrainingConfig(**{"num_envs": 2, "episodes_per_epoch": 2, "epochs": 2, **training}),
        device="cpu",
    )
    if tmp_path is not None:
        cfg.training.checkpoint_dir = str(tmp_path / "ckpt")
    return cfg


def make_env(num_envs=2, episode_length=EPISODE_LEN):
    return AspireIsaacEnv(DummyIsaacEnv(
        num_envs=num_envs, state_dim=STATE_DIM, action_dim=ACTION_DIM,
        episode_length=episode_length, device="cpu",
    ))


def make_trainer(tmp_path=None, **kwargs):
    training = kwargs.pop("training", {})
    config = kwargs.pop("config", None) or make_config(tmp_path, **training)
    return AspireIsaacTrainer(env=kwargs.pop("env", None) or make_env(), config=config, **kwargs)


def make_traj(n=12, reward=1.0, success=False):
    pairs = [
        StateActionPair(
            state=np.full(STATE_DIM, 0.1 * i, dtype=np.float32),
            action=np.full(ACTION_DIM, 0.05 * i, dtype=np.float32),
            timestamp=0.02 * i,
        )
        for i in range(n)
    ]
    return Trajectory(pairs=pairs, total_reward=reward, success=success)


def make_critique(score=5.0):
    return MotionCritique(overall_score=score, teacher_name="t")


class ScriptedEnv:
    """Single-env stand-in whose episodes have scripted lengths (then repeat the last)."""

    def __init__(self, lengths):
        self.lengths = list(lengths)
        self.num_envs = 1
        self.state_dim = STATE_DIM
        self.action_dim = ACTION_DIM
        self.episode = 0
        self.t = 0
        self.rewards = []

    def reset(self):
        self.episode, self.t = 0, 0
        return {"obs": torch.zeros(1, STATE_DIM)}

    def step(self, action):
        self.t += 1
        length = self.lengths[min(self.episode, len(self.lengths) - 1)]
        done = self.t >= length
        completed = []
        if done:
            traj = Trajectory(pairs=[
                StateActionPair(state=np.zeros(STATE_DIM), action=np.zeros(ACTION_DIM), timestamp=0.02 * i)
                for i in range(length)
            ], total_reward=float(self.episode + 1), success=bool(self.episode % 2))
            completed.append(traj)
            self.episode += 1
            self.t = 0
        obs = torch.zeros(1, STATE_DIM)
        return obs, torch.zeros(1), torch.tensor([done]), {}, completed


class StubCritic(nn.Module):
    """Critic that returns a fixed score; has a parameter so optimizers can be built."""

    def __init__(self, value):
        super().__init__()
        self.value = value
        self.w = nn.Parameter(torch.zeros(1))

    def forward(self, states, actions, mask=None):
        return CriticOutput(score=torch.full((states.shape[0],), float(self.value)))


def params_snapshot(module):
    return [p.detach().clone() for p in module.parameters()]


def params_changed(before, module):
    return any(not torch.equal(a, b) for a, b in zip(before, module.parameters()))


class TestTrajectoryDataset:
    def test_short_trajectory_is_zero_padded_with_mask(self):
        ds = TrajectoryDataset([make_traj(5)], [make_critique(7.5)], max_length=8)
        item = ds[0]
        assert item["states"].shape == (8, STATE_DIM) and item["actions"].shape == (8, ACTION_DIM)
        assert item["mask"].tolist() == [True] * 5 + [False] * 3
        assert torch.count_nonzero(item["states"][5:]) == 0
        assert torch.count_nonzero(item["actions"][5:]) == 0
        assert item["seq_len"].item() == 5 and item["seq_len"].dtype == torch.long
        assert item["score"].item() == pytest.approx(7.5) and item["score"].dtype == torch.float32

    def test_long_trajectory_is_truncated(self):
        traj = make_traj(10)
        ds = TrajectoryDataset([traj], [make_critique()], max_length=6)
        item = ds[0]
        assert item["states"].shape == (6, STATE_DIM)
        assert item["mask"].all()
        assert item["seq_len"].item() == 6
        assert item["states"][5, 0].item() == pytest.approx(0.5)   # step 5 kept, step 6+ dropped

    def test_exact_length_trajectory_is_unchanged(self):
        ds = TrajectoryDataset([make_traj(6)], [make_critique()], max_length=6)
        item = ds[0]
        assert item["mask"].all() and item["seq_len"].item() == 6

    def test_len_and_indexing_pair_trajectory_with_its_critique(self):
        ds = TrajectoryDataset([make_traj(5), make_traj(5)], [make_critique(1.0), make_critique(9.0)], 8)
        assert len(ds) == 2
        assert ds[1]["score"].item() == pytest.approx(9.0)


class TestSimplePolicy:
    def test_forward_shapes_and_initial_std(self):
        policy = SimplePolicy(STATE_DIM, ACTION_DIM, hidden_dim=8)
        mean, std = policy(torch.randn(5, STATE_DIM))
        assert mean.shape == (5, ACTION_DIM) and std.shape == (5, ACTION_DIM)
        assert torch.allclose(std, torch.ones(5, ACTION_DIM))   # log_std starts at zero

    def test_std_follows_log_std_parameter(self):
        policy = SimplePolicy(STATE_DIM, ACTION_DIM, hidden_dim=8)
        with torch.no_grad():
            policy.log_std.fill_(np.log(0.5))
        _, std = policy(torch.zeros(1, STATE_DIM))
        assert torch.allclose(std, torch.full((1, ACTION_DIM), 0.5))

    def test_num_layers_controls_backbone_depth(self):
        three = SimplePolicy(STATE_DIM, ACTION_DIM, hidden_dim=8, num_layers=3)
        four = SimplePolicy(STATE_DIM, ACTION_DIM, hidden_dim=8, num_layers=4)
        assert len(three.backbone) == 2 * 3    # (Linear, LayerNorm, GELU) per hidden block
        assert len(four.backbone) == 3 * 3

    def test_single_layer_policy_accepts_states(self):
        policy = SimplePolicy(STATE_DIM, ACTION_DIM, hidden_dim=8, num_layers=1)
        mean, _ = policy(torch.zeros(1, STATE_DIM))
        assert mean.shape == (1, ACTION_DIM)

    def test_deterministic_action_is_the_mean(self):
        policy = SimplePolicy(STATE_DIM, ACTION_DIM, hidden_dim=8)
        state = torch.randn(3, STATE_DIM)
        assert torch.equal(policy.get_action(state, deterministic=True), policy(state)[0])

    def test_sampled_action_is_tanh_bounded_and_stochastic(self):
        torch.manual_seed(1)
        policy = SimplePolicy(STATE_DIM, ACTION_DIM, hidden_dim=8)
        with torch.no_grad():
            policy.log_std.fill_(3.0)   # large noise to exercise the clamp
        state = torch.zeros(64, STATE_DIM)
        a1, a2 = policy.get_action(state), policy.get_action(state)
        assert a1.abs().max() <= 1.0
        assert not torch.equal(a1, a2)

    def test_log_prob_matches_diagonal_gaussian(self):
        policy = SimplePolicy(STATE_DIM, ACTION_DIM, hidden_dim=8)
        with torch.no_grad():
            policy.log_std.copy_(torch.tensor([0.0, 0.5]))
        state, action = torch.randn(4, STATE_DIM), torch.randn(4, ACTION_DIM)
        mean, std = policy(state)
        expected = torch.distributions.Normal(mean, std).log_prob(action).sum(dim=-1)
        got = policy.log_prob(state, action)
        assert got.shape == (4,)
        assert torch.allclose(got, expected, atol=1e-5)


class TestTrainerInit:
    def test_dimensions_flow_from_env_into_critic_config(self):
        trainer = make_trainer()
        assert trainer.config.critic.state_dim == STATE_DIM
        assert trainer.config.critic.action_dim == ACTION_DIM
        assert trainer.critic.encoder.state_dim == STATE_DIM
        assert trainer.policy.mean_head.out_features == ACTION_DIM

    def test_default_components_follow_config(self):
        trainer = make_trainer()
        assert isinstance(trainer.teacher, MotionTeacher)
        assert [t.name for t in trainer.teacher.teachers] == [
            "Safety Inspector", "Efficiency Expert", "Grace Coach",
        ]
        assert isinstance(trainer.critic, TrajectoryCritic)
        assert isinstance(trainer.policy, SimplePolicy)
        assert trainer.buffer.max_size == trainer.config.training.trajectory_batch_size * 100
        assert trainer.metrics_history == []
        assert trainer.device == "cpu"

    def test_optimizer_learning_rates_come_from_config(self):
        trainer = make_trainer()
        assert trainer.critic_optimizer.param_groups[0]["lr"] == pytest.approx(3e-4)
        assert trainer.policy_optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)

    def test_supplied_components_are_used(self):
        cfg = make_config()
        cfg.critic.state_dim, cfg.critic.action_dim = STATE_DIM, ACTION_DIM
        teacher = MotionTeacher(personas=["grace_coach"])
        critic = TrajectoryCritic(cfg.critic)
        policy = SimplePolicy(STATE_DIM, ACTION_DIM, hidden_dim=8)
        trainer = make_trainer(config=cfg, teacher=teacher, critic=critic, policy=policy)
        assert trainer.teacher is teacher
        assert trainer.critic is critic
        assert trainer.policy is policy

    def test_env_name_creates_environment_via_factory(self):
        fake_env = make_env()
        cfg = make_config()
        with patch("integrations.isaac.isaac_wrapper.create_isaac_env", return_value=fake_env) as factory:
            trainer = AspireIsaacTrainer("Some-Env-v0", config=cfg)
        factory.assert_called_once_with("Some-Env-v0", num_envs=2, device="cpu")
        assert trainer.env is fake_env

    def test_config_defaults_when_not_given(self):
        cfg = make_config()
        with patch("integrations.isaac.trainer.IsaacAspireConfig", return_value=cfg) as cls:
            trainer = AspireIsaacTrainer(make_env())
        cls.assert_called_once_with()
        assert trainer.config is cfg

    def test_wandb_initialised_when_enabled(self):
        fake_wandb = MagicMock()
        with patch.dict(sys.modules, {"wandb": fake_wandb}):
            trainer = make_trainer(training={"use_wandb": True, "wandb_project": "proj"})
        fake_wandb.init.assert_called_once_with(project="proj")
        assert trainer.use_wandb is True

    def test_missing_wandb_disables_logging(self, capsys):
        with patch.dict(sys.modules, {"wandb": None}):
            trainer = make_trainer(training={"use_wandb": True})
        assert trainer.use_wandb is False
        assert "wandb not installed, disabling" in capsys.readouterr().out


class TestCollectTrajectories:
    def test_collects_requested_episodes_with_critiques_and_buffers_them(self):
        trainer = make_trainer()
        trajs, critiques = trainer.collect_trajectories(num_episodes=2)
        assert len(trajs) == len(critiques) == 2
        assert all(len(t) == EPISODE_LEN for t in trajs)
        assert all(0.0 <= c.overall_score <= 10.0 for c in critiques)
        assert all(c.teacher_name == "Motion Teacher Committee" for c in critiques)
        assert len(trainer.buffer) == 2
        assert trainer.buffer.scores == [c.overall_score for c in critiques]

    def test_stops_mid_batch_once_enough_episodes_collected(self):
        trainer = make_trainer()
        trajs, _ = trainer.collect_trajectories(num_episodes=1)   # 2 envs finish together
        assert len(trajs) == 1

    def test_short_episodes_are_skipped(self):
        env = ScriptedEnv([5, 12])
        trainer = make_trainer(env=env)
        trajs, critiques = trainer.collect_trajectories(num_episodes=1)
        assert [len(t) for t in trajs] == [12]
        assert trajs[0].total_reward == 2.0   # the 5-step first episode (reward 1.0) was dropped
        assert len(critiques) == 1 and len(trainer.buffer) == 1

    def test_episode_of_exactly_ten_steps_is_skipped(self):
        env = ScriptedEnv([10, 11])
        trajs, _ = make_trainer(env=env).collect_trajectories(num_episodes=1)
        assert [len(t) for t in trajs] == [11]

    def test_tensor_observation_is_accepted(self):
        env = ScriptedEnv([12])
        env.reset = lambda: torch.zeros(1, STATE_DIM)
        trajs, _ = make_trainer(env=env).collect_trajectories(num_episodes=1)
        assert len(trajs) == 1

    def test_uses_policy_actions(self):
        trainer = make_trainer(env=ScriptedEnv([12]))
        spy = MagicMock(wraps=trainer.policy.get_action)
        trainer.policy.get_action = spy
        trainer.collect_trajectories(num_episodes=1)
        assert spy.call_count == 12
        assert spy.call_args.args[0].shape == (1, STATE_DIM)


class TestTrainCriticEpoch:
    def test_empty_input_is_a_noop(self):
        trainer = make_trainer()
        before = params_snapshot(trainer.critic)
        assert trainer.train_critic_epoch([], []) == 0.0
        assert not params_changed(before, trainer.critic)

    def test_returns_finite_loss_and_updates_critic(self):
        trainer = make_trainer()
        trajs = [make_traj(12), make_traj(8), make_traj(14)]
        critiques = [make_critique(2.0), make_critique(5.0), make_critique(9.0)]
        before = params_snapshot(trainer.critic)
        loss = trainer.train_critic_epoch(trajs, critiques)
        assert np.isfinite(loss) and loss > 0.0
        assert params_changed(before, trainer.critic)
        assert trainer.critic.training is True

    def test_batches_in_chunks_of_thirty_two(self):
        trainer = make_trainer()
        steps = []
        original_step = trainer.critic_optimizer.step
        trainer.critic_optimizer.step = lambda *a, **k: (steps.append(1), original_step(*a, **k))[1]
        trainer.train_critic_epoch([make_traj(12)] * 33, [make_critique(4.0)] * 33)
        assert len(steps) == 2   # 32 + 1

    def test_repeated_epochs_reduce_loss_on_fixed_data(self):
        torch.manual_seed(0)
        trainer = make_trainer(training={"critic_lr": 1e-2})
        trajs = [make_traj(12), make_traj(12)]
        critiques = [make_critique(2.0), make_critique(8.0)]
        losses = [trainer.train_critic_epoch(trajs, critiques) for _ in range(10)]
        assert losses[-1] < losses[0]


class TestTrainPolicyEpoch:
    def test_empty_input_is_a_noop(self):
        trainer = make_trainer()
        before = params_snapshot(trainer.policy)
        assert trainer.train_policy_epoch([], []) == 0.0
        assert not params_changed(before, trainer.policy)

    def test_trajectories_shorter_than_ten_are_skipped(self):
        trainer = make_trainer()
        before = params_snapshot(trainer.policy)
        loss = trainer.train_policy_epoch([make_traj(9)], [make_critique()])
        assert loss == 0.0
        assert not params_changed(before, trainer.policy)

    def test_updates_policy_and_sets_modes(self):
        trainer = make_trainer()
        before = params_snapshot(trainer.policy)
        loss = trainer.train_policy_epoch([make_traj(12), make_traj(10)], [make_critique()] * 2)
        assert np.isfinite(loss)
        assert params_changed(before, trainer.policy)
        assert trainer.policy.training is True
        assert trainer.critic.training is False

    def mean_log_prob(self, trainer, traj):
        states, actions, _ = traj.to_tensors(device="cpu")
        with torch.no_grad():
            return trainer.policy.log_prob(states, actions).mean().item()

    def test_high_critic_score_makes_taken_actions_more_likely(self):
        torch.manual_seed(0)
        trainer = make_trainer(training={"entropy_bonus": 0.0},
                               critic=StubCritic(10.0))
        traj = make_traj(12)
        before = self.mean_log_prob(trainer, traj)
        trainer.train_policy_epoch([traj], [make_critique()])
        assert self.mean_log_prob(trainer, traj) > before

    def test_low_critic_score_makes_taken_actions_less_likely(self):
        torch.manual_seed(0)
        trainer = make_trainer(training={"entropy_bonus": 0.0},
                               critic=StubCritic(0.0))
        traj = make_traj(12)
        before = self.mean_log_prob(trainer, traj)
        trainer.train_policy_epoch([traj], [make_critique()])
        assert self.mean_log_prob(trainer, traj) < before

    def test_neutral_critic_score_gives_zero_loss_without_entropy_bonus(self):
        trainer = make_trainer(training={"entropy_bonus": 0.0}, critic=StubCritic(5.0))
        loss = trainer.train_policy_epoch([make_traj(12)], [make_critique()])
        assert loss == pytest.approx(0.0, abs=1e-6)

    def test_entropy_bonus_lowers_the_loss(self):
        traj = make_traj(12)
        losses = {}
        for bonus in (0.0, 1.0):
            torch.manual_seed(0)
            trainer = make_trainer(training={"entropy_bonus": bonus}, critic=StubCritic(5.0))
            losses[bonus] = trainer.train_policy_epoch([traj], [make_critique()])
        # entropy of a unit-variance Gaussian is 0.5*log(2*pi*e) ~ 1.419
        assert losses[1.0] == pytest.approx(losses[0.0] - 0.5 * np.log(2 * np.pi * np.e), abs=1e-5)


class TestTrain:
    def test_runs_epochs_and_records_metrics(self, capsys):
        trainer = make_trainer()
        history = trainer.train(epochs=2)
        assert history is trainer.metrics_history
        assert [m.epoch for m in history] == [1, 2]
        for m in history:
            assert isinstance(m, TrainingMetrics)
            assert m.trajectories_collected == 2
            assert np.isfinite(m.critic_loss) and np.isfinite(m.policy_loss)
            assert 0.0 <= m.mean_trajectory_score <= 10.0
            assert 0.0 <= m.success_rate <= 1.0
            assert m.time_elapsed >= 0.0
        out = capsys.readouterr().out
        assert "Starting ASPIRE Isaac training for 2 epochs" in out
        assert "Parallel environments: 2" in out
        assert "Critic architecture: tcn" in out
        assert "Training complete!" in out

    def test_epochs_default_to_config(self):
        trainer = make_trainer(training={"epochs": 1})
        assert len(trainer.train()) == 1

    def test_callback_receives_metrics_every_epoch(self):
        seen = []
        make_trainer().train(epochs=2, callback=seen.append)
        assert [m.epoch for m in seen] == [1, 2]

    def test_update_frequencies_skip_off_epochs(self):
        trainer = make_trainer(training={"critic_update_frequency": 2, "policy_update_frequency": 2})
        history = trainer.train(epochs=2)
        assert history[0].critic_loss > 0.0 and history[0].policy_loss != 0.0
        assert history[1].critic_loss == 0.0 and history[1].policy_loss == 0.0

    def test_epoch_without_trajectories_is_skipped(self, capsys):
        trainer = make_trainer()
        trainer.collect_trajectories = lambda num_episodes: ([], [])
        history = trainer.train(epochs=2)
        assert history == []
        assert capsys.readouterr().out.count("No trajectories collected, skipping epoch") == 2

    def test_metrics_aggregate_trajectory_statistics(self):
        trainer = make_trainer()
        trajs = [make_traj(12, reward=2.0, success=True), make_traj(12, reward=4.0, success=False)]
        critiques = [make_critique(4.0), make_critique(8.0)]
        trainer.collect_trajectories = lambda num_episodes: (trajs, critiques)
        (m,) = trainer.train(epochs=1)
        assert m.mean_trajectory_score == pytest.approx(6.0)
        assert m.mean_episode_reward == pytest.approx(3.0)
        assert m.success_rate == pytest.approx(0.5)
        assert m.trajectories_collected == 2

    def test_checkpoint_written_on_save_frequency(self, tmp_path):
        trainer = make_trainer(tmp_path, training={"save_frequency": 2})
        trainer.train(epochs=3)
        files = sorted(p.name for p in (tmp_path / "ckpt").iterdir())
        assert files == ["checkpoint_epoch_2.pt"]

    def test_wandb_logging_per_epoch(self):
        fake_wandb = MagicMock()
        with patch.dict(sys.modules, {"wandb": fake_wandb}):
            trainer = make_trainer(training={"use_wandb": True})
            trainer.train(epochs=1)
        (call,) = fake_wandb.log.call_args_list
        payload = call.args[0]
        assert payload["epoch"] == 1
        assert set(payload) == {
            "epoch", "critic_loss", "policy_loss", "mean_score", "mean_reward", "success_rate",
        }

    def test_zero_epochs_runs_nothing(self):
        trainer = make_trainer(training={"epochs": 1})
        trainer.collect_trajectories = MagicMock(return_value=([], []))
        trainer.train(epochs=0)
        trainer.collect_trajectories.assert_not_called()


class TestCheckpointing:
    def test_save_checkpoint_contents(self, tmp_path):
        trainer = make_trainer(tmp_path)
        trainer.metrics_history.append(TrainingMetrics(1, 0.5, 0.25, 6.0, 1.0, 0.5, 2, 0.1))
        trainer.save_checkpoint(7)
        path = tmp_path / "ckpt" / "checkpoint_epoch_7.pt"
        assert path.exists()
        ckpt = torch.load(path, weights_only=True)  # plain values only: loading runs no code
        assert set(ckpt) == {
            "epoch", "critic_state_dict", "policy_state_dict", "critic_optimizer",
            "policy_optimizer", "config", "metrics_history",
        }
        assert ckpt["epoch"] == 7
        assert ckpt["metrics_history"][0]["critic_loss"] == 0.5
        assert ckpt["config"]["device"] == "cpu"
        assert set(ckpt["policy_state_dict"]) == set(trainer.policy.state_dict())

    def test_load_checkpoint_restores_weights_and_history(self, tmp_path):
        source = make_trainer(tmp_path)
        source.train_critic_epoch([make_traj(12)], [make_critique(3.0)])
        source.metrics_history.append(TrainingMetrics(1, 0.5, 0.25, 6.0, 1.0, 0.5, 2, 0.1))
        source.save_checkpoint(3)
        path = str(tmp_path / "ckpt" / "checkpoint_epoch_3.pt")

        target = make_trainer(tmp_path)
        target.load_checkpoint(path)  # the real weights_only=True load
        for a, b in zip(source.critic.parameters(), target.critic.parameters()):
            assert torch.equal(a, b)
        for a, b in zip(source.policy.parameters(), target.policy.parameters()):
            assert torch.equal(a, b)
        assert [m.epoch for m in target.metrics_history] == [1]

    def test_load_checkpoint_without_history_defaults_to_empty(self, tmp_path, capsys):
        trainer = make_trainer(tmp_path)
        ckpt = {
            "epoch": 4,
            "critic_state_dict": trainer.critic.state_dict(),
            "policy_state_dict": trainer.policy.state_dict(),
            "critic_optimizer": trainer.critic_optimizer.state_dict(),
            "policy_optimizer": trainer.policy_optimizer.state_dict(),
        }
        trainer.metrics_history.append(TrainingMetrics(1, 0.5, 0.25, 6.0, 1.0, 0.5, 2, 0.1))
        with patch("integrations.isaac.trainer.torch.load", return_value=ckpt) as load:
            trainer.load_checkpoint("somewhere.pt")
        assert load.call_args.kwargs["map_location"] == "cpu"
        assert trainer.metrics_history == []
        assert "Loaded checkpoint from somewhere.pt (epoch 4)" in capsys.readouterr().out

    def test_checkpoint_round_trips_through_save_and_load(self, tmp_path):
        trainer = make_trainer(tmp_path)
        trainer.save_checkpoint(1)
        make_trainer(tmp_path).load_checkpoint(str(tmp_path / "ckpt" / "checkpoint_epoch_1.pt"))


class TestEvaluate:
    def test_returns_summary_statistics(self):
        trainer = make_trainer()
        result = trainer.evaluate(num_episodes=2)
        assert set(result) == {"mean_reward", "std_reward", "success_rate", "mean_score", "std_score"}
        assert all(np.isfinite(v) for v in result.values())
        assert 0.0 <= result["success_rate"] <= 1.0
        assert 0.0 <= result["mean_score"] <= 10.0
        assert trainer.policy.training is False and trainer.critic.training is False

    def test_statistics_come_from_completed_episodes(self):
        env = ScriptedEnv([3])   # rewards 1, 2, 3, 4 ; success on odd episode indices
        trainer = make_trainer(env=env)
        trainer.teacher = MagicMock()
        trainer.teacher.critique.side_effect = [make_critique(s) for s in (2.0, 4.0, 6.0, 8.0)]
        result = trainer.evaluate(num_episodes=4)
        assert result["mean_reward"] == pytest.approx(2.5)
        assert result["std_reward"] == pytest.approx(np.std([1, 2, 3, 4]))
        assert result["success_rate"] == pytest.approx(0.5)
        assert result["mean_score"] == pytest.approx(5.0)
        assert result["std_score"] == pytest.approx(np.std([2, 4, 6, 8]))

    def test_deterministic_flag_is_forwarded_to_policy(self):
        trainer = make_trainer(env=ScriptedEnv([2]))
        trainer.teacher = MagicMock()
        trainer.teacher.critique.return_value = make_critique()
        spy = MagicMock(wraps=trainer.policy.get_action)
        trainer.policy.get_action = spy
        trainer.evaluate(num_episodes=1, deterministic=False)
        assert all(c.kwargs == {"deterministic": False} for c in spy.call_args_list)
        spy.reset_mock()
        trainer.evaluate(num_episodes=1)
        assert all(c.kwargs == {"deterministic": True} for c in spy.call_args_list)

    def test_critic_refinement_flag_does_not_change_actions(self):
        torch.manual_seed(0)
        trainer = make_trainer(env=ScriptedEnv([2]))
        trainer.teacher = MagicMock()
        trainer.teacher.critique.return_value = make_critique()
        with_refine = trainer.evaluate(num_episodes=1, use_critic_refinement=True)
        without = trainer.evaluate(num_episodes=1, use_critic_refinement=False)
        assert with_refine == without

    def test_tensor_observation_is_accepted(self):
        env = ScriptedEnv([2])
        env.reset = lambda: torch.zeros(1, STATE_DIM)
        trainer = make_trainer(env=env)
        trainer.teacher = MagicMock()
        trainer.teacher.critique.return_value = make_critique(6.0)
        assert trainer.evaluate(num_episodes=1)["mean_score"] == pytest.approx(6.0)

    def test_empty_trajectories_are_not_counted(self):
        class EmptyThenReal(ScriptedEnv):
            def step(self, action):
                obs, r, d, i, completed = super().step(action)
                if not hasattr(self, "_sent_empty"):
                    self._sent_empty = True
                    completed = [Trajectory()] + completed
                return obs, r, d, i, completed

        trainer = make_trainer(env=EmptyThenReal([2]))
        trainer.teacher = MagicMock()
        trainer.teacher.critique.return_value = make_critique()
        trainer.evaluate(num_episodes=1)
        assert trainer.teacher.critique.call_count == 1
