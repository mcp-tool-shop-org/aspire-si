"""
Tests for integrations.isaac.isaac_wrapper (trajectories, buffer, env wrapper, dummy env).
"""

import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from integrations.isaac.isaac_wrapper import (
    AspireIsaacEnv,
    DummyIsaacEnv,
    IsaacEnvProtocol,
    StateActionPair,
    Trajectory,
    TrajectoryBuffer,
    create_isaac_env,
)
from integrations.isaac.motion_teacher import TrajectoryData


def make_pair(t=0.0, state_dim=4, action_dim=2, **kwargs):
    return StateActionPair(
        state=np.full(state_dim, t, dtype=np.float32),
        action=np.full(action_dim, -t, dtype=np.float32),
        timestamp=t,
        **kwargs,
    )


def make_traj(n=5, **kwargs):
    return Trajectory(pairs=[make_pair(0.1 * i) for i in range(n)], **kwargs)


class TestStateActionPair:
    def test_optional_fields_default(self):
        pair = make_pair()
        assert pair.collision is False
        assert pair.contact_forces is None
        assert pair.energy == 0.0
        assert pair.goal_distance is None


class TestTrajectory:
    def test_len_counts_pairs(self):
        assert len(Trajectory()) == 0
        assert len(make_traj(7)) == 7

    def test_to_tensors_shapes_dtypes_and_values(self):
        states, actions, stamps = make_traj(5).to_tensors(device="cpu")
        assert states.shape == (5, 4) and actions.shape == (5, 2) and stamps.shape == (5,)
        assert states.dtype == actions.dtype == stamps.dtype == torch.float32
        assert stamps.tolist() == pytest.approx([0.0, 0.1, 0.2, 0.3, 0.4])
        assert actions[2, 0].item() == pytest.approx(-0.2)

    def test_to_teacher_format_minimal(self):
        goal, init = np.ones(4), np.zeros(4)
        traj = make_traj(5, task_description="reach", goal_state=goal, initial_state=init)
        data = traj.to_teacher_format()
        assert isinstance(data, TrajectoryData)
        assert data.states.shape == (5, 4)
        assert data.actions.shape == (5, 2)
        assert data.timestamps.shape == (5,)
        assert data.task_description == "reach"
        assert data.goal_state is goal and data.initial_state is init
        assert data.collisions.dtype == bool and not data.collisions.any()
        assert data.energy_usage.tolist() == [0.0] * 5
        assert data.goal_distances is None
        assert data.forces is None

    def test_to_teacher_format_carries_privileged_info(self):
        pairs = [
            make_pair(
                0.1 * i,
                collision=(i == 2),
                contact_forces=np.array([i, 0.0, 0.0]),
                energy=float(i),
                goal_distance=1.0 - 0.1 * i,
            )
            for i in range(4)
        ]
        data = Trajectory(pairs=pairs).to_teacher_format()
        assert data.collisions.tolist() == [False, False, True, False]
        assert data.forces.shape == (4, 3)
        assert data.forces[3, 0] == 3.0
        assert data.energy_usage.tolist() == [0.0, 1.0, 2.0, 3.0]
        assert data.goal_distances == pytest.approx([1.0, 0.9, 0.8, 0.7])
        assert data.length == 4


class TestTrajectoryBuffer:
    def test_empty_buffer(self):
        buf = TrajectoryBuffer()
        assert len(buf) == 0
        assert buf.sample(4) == []
        assert buf.get_statistics() == {}

    def test_add_records_trajectory_and_score(self):
        buf = TrajectoryBuffer()
        traj = make_traj()
        buf.add(traj, score=8.0)
        assert len(buf) == 1
        assert buf.trajectories[0] is traj
        assert buf.scores == [8.0]

    def test_missing_score_defaults_to_neutral_five_with_unit_priority(self):
        buf = TrajectoryBuffer()
        buf.add(make_traj())
        assert buf.scores == [5.0]
        assert buf.priorities.tolist() == [1.0]

    def test_priority_grows_with_distance_from_neutral(self):
        buf = TrajectoryBuffer(priority_alpha=0.5)
        buf.add(make_traj(), score=9.0)   # |9-5|+1 = 5
        buf.add(make_traj(), score=3.0)   # |3-5|+1 = 3
        assert buf.priorities[0] == pytest.approx(5 ** 0.5)
        assert buf.priorities[1] == pytest.approx(3 ** 0.5)

    def test_non_prioritized_buffer_tracks_no_priorities(self):
        buf = TrajectoryBuffer(prioritized=False)
        buf.add(make_traj(), score=9.0)
        assert buf.priorities.size == 0

    def test_zero_score_gets_priority_for_maximal_distance_from_neutral(self):
        buf = TrajectoryBuffer(priority_alpha=0.6)
        buf.add(make_traj(), score=0.0)
        assert buf.priorities[0] == pytest.approx(6.0 ** 0.6)

    def test_eviction_drops_oldest_and_keeps_arrays_aligned(self):
        buf = TrajectoryBuffer(max_size=3)
        trajs = [make_traj(i + 1) for i in range(5)]
        for i, t in enumerate(trajs):
            buf.add(t, score=float(i + 1))
        assert len(buf) == 3
        assert buf.trajectories == trajs[2:]
        assert buf.scores == [3.0, 4.0, 5.0]
        assert len(buf.priorities) == 3

    def test_eviction_without_priorities(self):
        buf = TrajectoryBuffer(max_size=2, prioritized=False)
        for i in range(4):
            buf.add(make_traj(i + 1), score=float(i))
        assert buf.scores == [2.0, 3.0]
        assert buf.priorities.size == 0

    def test_sample_clamps_batch_size_and_has_no_duplicates(self):
        buf = TrajectoryBuffer()
        for i in range(4):
            buf.add(make_traj(i + 1), score=float(i + 1))
        batch = buf.sample(100)
        assert len(batch) == 4
        assert len({id(t) for t in batch}) == 4

    def test_sample_uniform_when_not_prioritized(self):
        buf = TrajectoryBuffer(prioritized=False)
        for i in range(6):
            buf.add(make_traj(i + 1))
        batch = buf.sample(3)
        assert len(batch) == 3
        assert all(t in buf.trajectories for t in batch)

    def test_sample_prioritized_prefers_high_priority(self):
        np.random.seed(0)
        buf = TrajectoryBuffer(priority_alpha=1.0)
        hot = make_traj(3)
        buf.add(hot, score=0.1)           # priority ~5.9
        for _ in range(5):
            buf.add(make_traj(3), score=5.0)  # priority 1
        hits = sum(buf.sample(1)[0] is hot for _ in range(300))
        # expected share 5.9/10.9 ~ 54%, vs 17% if uniform
        assert hits > 100

    def test_sample_truncates_to_max_length(self):
        buf = TrajectoryBuffer(prioritized=False)
        buf.add(make_traj(10, task_description="x", success=True, total_reward=2.5))
        (t,) = buf.sample(1, max_length=4)
        assert len(t) == 4
        assert (t.task_description, t.success, t.total_reward) == ("x", True, 2.5)
        assert len(buf.trajectories[0]) == 10   # original untouched

    def test_truncate_returns_same_object_when_short_enough(self):
        buf = TrajectoryBuffer()
        traj = make_traj(3)
        assert buf._truncate(traj, 3) is traj
        assert buf._truncate(traj, 10) is traj

    def test_statistics(self):
        buf = TrajectoryBuffer()
        buf.add(make_traj(2), score=2.0)
        buf.add(make_traj(4), score=6.0)
        stats = buf.get_statistics()
        assert stats["size"] == 2
        assert stats["mean_score"] == pytest.approx(4.0)
        assert stats["std_score"] == pytest.approx(2.0)
        assert stats["min_score"] == 2.0 and stats["max_score"] == 6.0
        assert stats["mean_length"] == pytest.approx(3.0)

    def test_clear_resets_everything(self):
        buf = TrajectoryBuffer()
        buf.add(make_traj(), score=1.0)
        buf.clear()
        assert len(buf) == 0 and buf.scores == [] and buf.priorities.size == 0
        assert buf.get_statistics() == {}


class FakeEnv:
    """Scriptable env returning whatever obs/infos the test supplies."""

    def __init__(self, num_envs=2, state_dim=3, action_dim=2, obs_space=None, act_space=None):
        self.num_envs = num_envs
        self.observation_space = obs_space or types.SimpleNamespace(shape=(state_dim,))
        self.action_space = act_space or types.SimpleNamespace(shape=(action_dim,))
        self.reset_obs = {"obs": torch.zeros(num_envs, state_dim)}
        self.step_returns = None

    def reset(self):
        return self.reset_obs

    def step(self, actions):
        return self.step_returns


class TestAspireIsaacEnvInit:
    def test_dimensions_from_shape_attributes(self):
        env = AspireIsaacEnv(FakeEnv(num_envs=3, state_dim=7, action_dim=4))
        assert (env.state_dim, env.action_dim, env.num_envs) == (7, 4, 3)
        assert len(env.current_trajectories) == 3
        assert env.episode_start_times == [0.0] * 3
        assert env.step_count == 0

    def test_dict_observation_space_without_shape_attribute(self):
        space = {"obs": types.SimpleNamespace(shape=(9,))}
        env = AspireIsaacEnv(FakeEnv(obs_space=space))
        assert env.state_dim == 9

    def test_unknown_spaces_leave_dims_zero(self):
        env = AspireIsaacEnv(FakeEnv(obs_space=object(), act_space=object()))
        assert env.state_dim == 0 and env.action_dim == 0

    def test_protocol_is_runtime_checkable(self):
        assert isinstance(FakeEnv(), IsaacEnvProtocol)
        assert not isinstance(object(), IsaacEnvProtocol)


class TestAspireIsaacEnvReset:
    def test_reset_clears_in_progress_trajectories(self):
        env = AspireIsaacEnv(DummyIsaacEnv(num_envs=2, episode_length=50, device="cpu"))
        obs = env.reset()
        env.step(torch.zeros(2, 7))
        assert len(env.current_trajectories[0]) == 1
        env.reset()
        assert all(len(t) == 0 for t in env.current_trajectories)
        assert env.step_count == 0
        assert obs["obs"].shape == (2, 14)

    def test_reset_assigns_goal_per_env(self):
        inner = FakeEnv(num_envs=2, state_dim=3)
        inner.reset_obs = {
            "obs": torch.zeros(2, 3),
            "goal": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        }
        env = AspireIsaacEnv(inner)
        env.reset()
        assert env.current_trajectories[0].goal_state.tolist() == [1.0, 2.0]
        assert env.current_trajectories[1].goal_state.tolist() == [3.0, 4.0]

    def test_custom_goal_key(self):
        inner = FakeEnv(num_envs=1, state_dim=3)
        inner.reset_obs = {"obs": torch.zeros(1, 3), "target": torch.tensor([[5.0]])}
        env = AspireIsaacEnv(inner, goal_key="target")
        env.reset()
        assert env.current_trajectories[0].goal_state.tolist() == [5.0]

    def test_reset_infers_state_dim_from_dict_obs(self):
        inner = FakeEnv(num_envs=2, obs_space=object())
        inner.reset_obs = {"obs": torch.zeros(2, 6)}
        env = AspireIsaacEnv(inner)
        env.reset()
        assert env.state_dim == 6

    def test_reset_infers_state_dim_from_tensor_obs(self):
        inner = FakeEnv(num_envs=2, obs_space=object())
        inner.reset_obs = torch.zeros(2, 8)
        env = AspireIsaacEnv(inner)
        obs = env.reset()
        assert env.state_dim == 8
        assert obs is inner.reset_obs

    def test_reset_with_unrecognised_obs_keeps_state_dim_zero(self):
        inner = FakeEnv(num_envs=1, obs_space=object())
        inner.reset_obs = [1, 2, 3]
        env = AspireIsaacEnv(inner)
        env.reset()
        assert env.state_dim == 0


class TestAspireIsaacEnvStep:
    def run_step(self, inner, actions=None, **env_kwargs):
        env = AspireIsaacEnv(inner, **env_kwargs)
        env.reset()
        actions = actions if actions is not None else torch.ones(inner.num_envs, 2)
        return env, env.step(actions)

    def test_step_records_state_action_and_time(self):
        inner = FakeEnv(num_envs=2, state_dim=3)
        obs = {"obs": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])}
        inner.step_returns = (obs, torch.zeros(2), torch.tensor([False, False]), {})
        env, (out_obs, rewards, dones, infos, completed) = self.run_step(
            inner, actions=torch.tensor([[0.5, 0.5], [0.25, 0.75]])
        )
        assert completed == []
        assert out_obs is obs and infos == {}
        pair = env.current_trajectories[1].pairs[0]
        assert pair.state.tolist() == [4.0, 5.0, 6.0]
        assert pair.action.tolist() == [0.25, 0.75]
        assert pair.timestamp == pytest.approx(0.0)
        assert env.step_count == 1

    def test_timestamps_advance_at_fifty_hertz(self):
        inner = FakeEnv(num_envs=1, state_dim=3)
        inner.step_returns = ({"obs": torch.zeros(1, 3)}, torch.zeros(1), torch.tensor([False]), {})
        env, _ = self.run_step(inner, actions=torch.zeros(1, 2))
        env.step(torch.zeros(1, 2))
        env.step(torch.zeros(1, 2))
        stamps = [p.timestamp for p in env.current_trajectories[0].pairs]
        assert stamps == pytest.approx([0.0, 0.02, 0.04])

    def test_tensor_observation(self):
        inner = FakeEnv(num_envs=1, state_dim=3)
        inner.step_returns = (torch.tensor([[7.0, 8.0, 9.0]]), torch.zeros(1), torch.tensor([False]), {})
        env, _ = self.run_step(inner)
        assert env.current_trajectories[0].pairs[0].state.tolist() == [7.0, 8.0, 9.0]

    def test_unrecognised_observation_falls_back_to_zero_states(self):
        inner = FakeEnv(num_envs=2, state_dim=3)
        inner.step_returns = ("junk", torch.zeros(2), torch.tensor([False, False]), {})
        env, _ = self.run_step(inner)
        assert env.current_trajectories[0].pairs[0].state.tolist() == [0.0, 0.0, 0.0]

    def test_privileged_info_is_collected(self):
        inner = FakeEnv(num_envs=2, state_dim=3)
        infos = {
            "collision": torch.tensor([True, False]),
            "contact_forces": torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]),
            "energy": torch.tensor([4.0, 5.0]),
            "goal_distance": torch.tensor([0.5, 0.25]),
        }
        inner.step_returns = ({"obs": torch.zeros(2, 3)}, torch.zeros(2), torch.tensor([False, False]), infos)
        env, _ = self.run_step(inner)
        p0 = env.current_trajectories[0].pairs[0]
        p1 = env.current_trajectories[1].pairs[0]
        assert bool(p0.collision) is True and bool(p1.collision) is False
        assert p0.contact_forces.tolist() == [1.0, 2.0, 3.0]
        assert (p0.energy, p1.energy) == (4.0, 5.0)
        assert (p0.goal_distance, p1.goal_distance) == (0.5, 0.25)

    def test_privileged_info_ignored_when_disabled(self):
        inner = FakeEnv(num_envs=1, state_dim=3)
        infos = {"collision": torch.tensor([True]), "energy": torch.tensor([9.0])}
        inner.step_returns = ({"obs": torch.zeros(1, 3)}, torch.zeros(1), torch.tensor([False]), infos)
        env, _ = self.run_step(inner, collect_privileged=False)
        pair = env.current_trajectories[0].pairs[0]
        assert bool(pair.collision) is False
        assert pair.energy == 0.0
        assert pair.contact_forces is None and pair.goal_distance is None

    def test_completed_trajectory_is_returned_and_slot_restarted(self):
        inner = FakeEnv(num_envs=2, state_dim=3)
        obs = {"obs": torch.zeros(2, 3), "goal": torch.tensor([[1.0], [2.0]])}
        rewards = torch.tensor([3.5, -1.0])
        dones = torch.tensor([True, False])
        infos = {"success": torch.tensor([True, False])}
        inner.step_returns = (obs, rewards, dones, infos)
        env, (_, _, _, _, completed) = self.run_step(inner)
        assert len(completed) == 1
        done_traj = completed[0]
        assert len(done_traj) == 1
        assert done_traj.total_reward == pytest.approx(3.5)
        assert done_traj.success is True
        # finished slot gets a fresh trajectory with the new goal; the other keeps its pair
        assert len(env.current_trajectories[0]) == 0
        assert env.current_trajectories[0].goal_state.tolist() == [1.0]
        assert len(env.current_trajectories[1]) == 1
        assert env.episode_start_times[0] == pytest.approx(0.0)

    def test_completed_without_success_info_leaves_success_false(self):
        inner = FakeEnv(num_envs=1, state_dim=3)
        inner.step_returns = ({"obs": torch.zeros(1, 3)}, torch.tensor([2.0]), torch.tensor([True]), {})
        _, (_, _, _, _, completed) = self.run_step(inner)
        assert completed[0].success is False
        assert completed[0].total_reward == pytest.approx(2.0)

    def test_second_episode_timestamps_restart_from_episode_start(self):
        inner = FakeEnv(num_envs=1, state_dim=3)
        inner.step_returns = ({"obs": torch.zeros(1, 3)}, torch.zeros(1), torch.tensor([True]), {})
        env = AspireIsaacEnv(inner)
        env.reset()
        env.step(torch.zeros(1, 2))               # step 0 -> done at t=0.0
        inner.step_returns = ({"obs": torch.zeros(1, 3)}, torch.zeros(1), torch.tensor([False]), {})
        env.step(torch.zeros(1, 2))               # t=0.02, start=0.0
        env.step(torch.zeros(1, 2))               # t=0.04
        stamps = [p.timestamp for p in env.current_trajectories[0].pairs]
        assert stamps == pytest.approx([0.02, 0.04])

    def test_get_current_trajectories_returns_a_copy_of_the_list(self):
        env = AspireIsaacEnv(FakeEnv(num_envs=2))
        copy = env.get_current_trajectories()
        assert copy == env.current_trajectories and copy is not env.current_trajectories
        copy.pop()
        assert len(env.current_trajectories) == 2


class TestDummyIsaacEnv:
    def test_spaces_and_defaults(self):
        env = DummyIsaacEnv(device="cpu")
        assert env.num_envs == 4
        assert env.observation_space.shape == (14,)
        assert env.action_space.shape == (7,)

    def test_reset_shapes_and_step_counter(self):
        env = DummyIsaacEnv(num_envs=3, state_dim=6, action_dim=2, device="cpu")
        obs = env.reset()
        assert obs["obs"].shape == (3, 6)
        assert obs["goal"].shape == (3, 3)
        assert env._step == 0

    def test_step_dynamics_integrate_action_into_velocity_and_position(self):
        torch.manual_seed(0)
        env = DummyIsaacEnv(num_envs=2, state_dim=4, action_dim=2, device="cpu")
        env.reset()
        env._states = torch.zeros(2, 4)
        env._goals = torch.zeros(2, 2)
        obs, rewards, dones, infos = env.step(torch.ones(2, 2))
        vel = obs["obs"][:, 2:]
        pos = obs["obs"][:, :2]
        assert torch.allclose(vel, torch.full((2, 2), 0.1))
        assert torch.allclose(pos, torch.full((2, 2), 0.1 * 0.02))
        assert torch.allclose(rewards, -infos["goal_distance"])
        assert torch.allclose(infos["energy"], torch.tensor([2.0, 2.0]))
        assert infos["collision"].dtype == torch.bool
        assert infos["success"].tolist() == [True, True]   # distance ~0.003 < 0.1
        assert not dones.any()

    def test_episode_ends_after_episode_length_steps(self):
        env = DummyIsaacEnv(num_envs=2, state_dim=4, action_dim=2, episode_length=3, device="cpu")
        env.reset()
        flags = [env.step(torch.zeros(2, 2))[2].all().item() for _ in range(3)]
        assert flags == [False, False, True]

    def test_new_episode_after_done_is_not_immediately_done_again(self):
        env = DummyIsaacEnv(num_envs=2, state_dim=4, action_dim=2, episode_length=2, device="cpu")
        env.reset()
        env.step(torch.zeros(2, 2))
        env.step(torch.zeros(2, 2))            # done
        dones = env.step(torch.zeros(2, 2))[2]  # first step of the next episode
        assert not dones.any()

    def test_wrapper_collects_full_length_trajectories_from_dummy(self):
        env = AspireIsaacEnv(DummyIsaacEnv(num_envs=2, state_dim=4, action_dim=2,
                                           episode_length=5, device="cpu"))
        env.reset()
        completed = []
        for _ in range(5):
            completed = env.step(torch.zeros(2, 2))[4]
        assert len(completed) == 2
        assert all(len(t) == 5 for t in completed)
        assert completed[0].goal_state.shape == (2,)


def fake_module(name, **attrs):
    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)
    return mod


class TestCreateIsaacEnv:
    def isaac_lab_modules(self, env_obj):
        env_cls = MagicMock(return_value=env_obj)
        parse = MagicMock(return_value="CFG")
        mods = {
            "omni": fake_module("omni"),
            "omni.isaac": fake_module("omni.isaac"),
            "omni.isaac.lab": fake_module("omni.isaac.lab"),
            "omni.isaac.lab.envs": fake_module("omni.isaac.lab.envs", ManagerBasedRLEnv=env_cls),
            "omni.isaac.lab_tasks": fake_module("omni.isaac.lab_tasks"),
            "omni.isaac.lab_tasks.utils": fake_module("omni.isaac.lab_tasks.utils", parse_env_cfg=parse),
        }
        return mods, env_cls, parse

    def isaac_gym_modules(self, task_map):
        return {
            "isaacgym": fake_module("isaacgym", gymapi=object()),
            "isaacgym.gymapi": fake_module("isaacgym.gymapi"),
            "isaacgymenvs": fake_module("isaacgymenvs"),
            "isaacgymenvs.utils": fake_module("isaacgymenvs.utils"),
            "isaacgymenvs.utils.utils": fake_module("isaacgymenvs.utils.utils", set_seed=lambda *a: None),
            "isaacgymenvs.tasks": fake_module("isaacgymenvs.tasks", isaacgym_task_map=task_map),
        }

    def block_isaac(self, monkeypatch, *names):
        # A None entry in sys.modules makes `import x` raise ImportError.
        for name in names:
            monkeypatch.setitem(sys.modules, name, None)

    def test_isaac_lab_path(self, monkeypatch):
        inner = FakeEnv(num_envs=5, state_dim=3, action_dim=2)
        mods, env_cls, parse = self.isaac_lab_modules(inner)
        for k, v in mods.items():
            monkeypatch.setitem(sys.modules, k, v)
        wrapped = create_isaac_env("Cartpole-v0", num_envs=5, device="cuda")
        parse.assert_called_once_with("Cartpole-v0", num_envs=5, use_gpu=True)
        env_cls.assert_called_once_with(cfg="CFG")
        assert isinstance(wrapped, AspireIsaacEnv)
        assert wrapped.env is inner and wrapped.num_envs == 5

    def test_isaac_lab_cpu_disables_gpu_flag(self, monkeypatch):
        mods, _, parse = self.isaac_lab_modules(FakeEnv())
        for k, v in mods.items():
            monkeypatch.setitem(sys.modules, k, v)
        create_isaac_env("X-v0", num_envs=2, device="cpu")
        assert parse.call_args.kwargs["use_gpu"] is False

    def test_isaac_gym_fallback_strips_version_and_dashes(self, monkeypatch):
        self.block_isaac(monkeypatch, "omni.isaac.lab.envs", "omni.isaac.lab_tasks.utils")
        inner = FakeEnv(num_envs=2)
        task_cls = MagicMock(return_value=inner)
        for k, v in self.isaac_gym_modules({"FrankaCubeStack": task_cls}).items():
            monkeypatch.setitem(sys.modules, k, v)
        wrapped = create_isaac_env("Franka-Cube-Stack-v0", num_envs=2, device="cpu", headless=False)
        task_cls.assert_called_once_with(num_envs=2, headless=False, device="cpu")
        assert wrapped.env is inner

    def test_isaac_gym_unknown_task_raises_value_error(self, monkeypatch):
        self.block_isaac(monkeypatch, "omni.isaac.lab.envs", "omni.isaac.lab_tasks.utils")
        for k, v in self.isaac_gym_modules({}).items():
            monkeypatch.setitem(sys.modules, k, v)
        with pytest.raises(ValueError, match="Unknown Isaac Gym task: Nope"):
            create_isaac_env("Nope-v0")

    def test_neither_backend_installed_raises_import_error_with_hint(self, monkeypatch):
        self.block_isaac(
            monkeypatch,
            "omni.isaac.lab.envs", "omni.isaac.lab_tasks.utils",
            "isaacgym", "isaacgymenvs.utils.utils", "isaacgymenvs.tasks",
        )
        with pytest.raises(ImportError, match="Neither Isaac Lab nor Isaac Gym found"):
            create_isaac_env("FrankaCubeStack-v0")
