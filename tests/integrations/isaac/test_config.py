"""
Tests for integrations.isaac.config (configuration dataclasses and validation).
"""

import warnings
from unittest.mock import patch

import pytest
import torch

import integrations.isaac as isaac_pkg
from integrations.isaac.config import (
    CriticArchitecture,
    CriticConfig,
    IsaacAspireConfig,
    TaskType,
    TeacherConfig,
    TrainingConfig,
)


class TestEnums:
    def test_task_type_values_are_lowercase_strings(self):
        assert {t.value for t in TaskType} == {
            "reaching", "manipulation", "locomotion", "navigation", "assembly",
        }
        assert TaskType.REACHING == "reaching"

    def test_critic_architecture_values(self):
        assert [a.value for a in CriticArchitecture] == ["transformer", "lstm", "tcn", "mlp"]

    def test_critic_architecture_from_value(self):
        assert CriticArchitecture("tcn") is CriticArchitecture.TCN
        with pytest.raises(ValueError):
            CriticArchitecture("gru")


class TestTeacherConfig:
    def test_defaults(self):
        cfg = TeacherConfig()
        assert cfg.personas == ["safety_inspector", "efficiency_expert", "grace_coach"]
        assert cfg.strategy == "vote"
        assert cfg.use_physics_oracle is True
        assert cfg.use_vision_teacher is False
        assert cfg.safety_weight == 2.0
        assert cfg.efficiency_weight == 1.0
        assert cfg.smoothness_weight == 0.5
        assert cfg.goal_achievement_weight == 1.5

    def test_persona_list_is_not_shared_between_instances(self):
        a, b = TeacherConfig(), TeacherConfig()
        a.personas.append("physics_oracle")
        assert "physics_oracle" not in b.personas


class TestCriticConfig:
    def test_defaults(self):
        cfg = CriticConfig()
        assert cfg.architecture is CriticArchitecture.TRANSFORMER
        assert (cfg.hidden_dim, cfg.num_layers, cfg.num_heads) == (256, 4, 8)
        assert cfg.state_dim == 0 and cfg.action_dim == 0
        assert cfg.max_trajectory_len == 256
        assert cfg.predict_score and cfg.predict_reasoning and cfg.predict_improvement
        assert cfg.dropout == pytest.approx(0.1)


class TestTrainingConfig:
    def test_defaults(self):
        cfg = TrainingConfig()
        assert cfg.env_name == "FrankaCubeStack-v0"
        assert cfg.num_envs == 512
        assert cfg.epochs == 100
        assert cfg.episodes_per_epoch == 100
        assert cfg.critic_lr == pytest.approx(3e-4)
        assert cfg.policy_lr == pytest.approx(1e-4)
        assert cfg.checkpoint_dir == "checkpoints/isaac"
        assert cfg.use_wandb is False


class TestIsaacAspireConfig:
    def test_default_composition(self):
        cfg = IsaacAspireConfig()
        assert isinstance(cfg.teacher, TeacherConfig)
        assert isinstance(cfg.critic, CriticConfig)
        assert isinstance(cfg.training, TrainingConfig)
        assert cfg.device == ("cuda" if torch.cuda.is_available() else "cpu")
        assert cfg.seed == 42

    def test_the_default_device_follows_cuda(self):
        with patch("torch.cuda.is_available", return_value=False):
            assert IsaacAspireConfig().device == "cpu"
        with patch("torch.cuda.is_available", return_value=True):
            assert IsaacAspireConfig().device == "cuda"

    def test_default_config_emits_no_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            IsaacAspireConfig()

    def test_sub_configs_are_independent_per_instance(self):
        a, b = IsaacAspireConfig(), IsaacAspireConfig()
        a.critic.state_dim = 99
        assert b.critic.state_dim == 0

    @pytest.mark.parametrize("bad", [0, -3])
    def test_num_envs_below_one_rejected(self, bad):
        with pytest.raises(ValueError, match="num_envs must be >= 1"):
            IsaacAspireConfig(training=TrainingConfig(num_envs=bad))

    def test_num_envs_of_one_is_accepted(self):
        cfg = IsaacAspireConfig(training=TrainingConfig(num_envs=1))
        assert cfg.training.num_envs == 1

    def test_low_safety_weight_warns(self):
        with pytest.warns(UserWarning, match="Safety weight < 1.0"):
            IsaacAspireConfig(teacher=TeacherConfig(safety_weight=0.5))

    def test_safety_weight_of_exactly_one_does_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            IsaacAspireConfig(teacher=TeacherConfig(safety_weight=1.0))


class TestPackageExports:
    def test_all_names_resolve(self):
        for name in isaac_pkg.__all__:
            assert hasattr(isaac_pkg, name), name

    def test_exports_are_the_real_classes(self):
        from integrations.isaac.config import IsaacAspireConfig as Cfg
        from integrations.isaac.trainer import AspireIsaacTrainer

        assert isaac_pkg.IsaacAspireConfig is Cfg
        assert isaac_pkg.AspireIsaacTrainer is AspireIsaacTrainer
