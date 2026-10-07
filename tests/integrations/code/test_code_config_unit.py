"""Unit tests for integrations/code/config.py."""

from __future__ import annotations

import pytest

from integrations.code.config import (
    CodeAspireConfig,
    CodeDimension,
    CriticArchitecture,
    CriticConfig,
    Language,
    TeacherConfig,
    TrainingConfig,
)


class TestEnums:
    def test_language_values_are_lowercase_strings(self):
        assert Language.PYTHON == "python"
        assert Language("rust") is Language.RUST
        assert Language.UNKNOWN.value == "unknown"

    def test_language_rejects_unknown_value(self):
        with pytest.raises(ValueError):
            Language("cobol")

    def test_code_dimension_members(self):
        assert {d.value for d in CodeDimension} == {
            "correctness", "style", "security", "performance",
            "maintainability", "architecture", "documentation", "testing",
        }

    def test_critic_architecture_members(self):
        assert [a.value for a in CriticArchitecture] == ["transformer", "codebert", "graphnn", "hybrid"]


class TestDefaults:
    def test_teacher_defaults(self):
        cfg = TeacherConfig()
        assert cfg.personas == ["correctness_checker", "style_guide", "security_auditor"]
        assert cfg.strategy == "vote"
        assert cfg.languages == [Language.PYTHON, Language.JAVASCRIPT, Language.TYPESCRIPT]
        assert cfg.correctness_weight > cfg.security_weight > cfg.style_weight

    def test_default_factories_do_not_share_state(self):
        a, b = TeacherConfig(), TeacherConfig()
        a.personas.append("extra")
        assert "extra" not in b.personas

    def test_critic_defaults(self):
        cfg = CriticConfig()
        assert cfg.architecture == CriticArchitecture.TRANSFORMER
        assert (cfg.hidden_dim, cfg.num_layers, cfg.num_heads) == (512, 6, 8)
        assert cfg.pretrained_model == "microsoft/codebert-base"

    def test_training_defaults(self):
        cfg = TrainingConfig()
        assert cfg.batch_size == 8
        assert cfg.gradient_accumulation_steps == 4
        assert cfg.use_wandb is False
        assert cfg.checkpoint_dir == "checkpoints/code"

    def test_top_level_defaults(self):
        cfg = CodeAspireConfig()
        assert cfg.device == "cuda"
        assert cfg.seed == 42
        assert isinstance(cfg.teacher, TeacherConfig)
        assert isinstance(cfg.critic, CriticConfig)
        assert isinstance(cfg.training, TrainingConfig)


class TestValidation:
    @pytest.mark.parametrize("bad", [0, -1, -100])
    def test_batch_size_must_be_positive(self, bad):
        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            CodeAspireConfig(training=TrainingConfig(batch_size=bad))

    def test_batch_size_one_is_allowed(self):
        assert CodeAspireConfig(training=TrainingConfig(batch_size=1)).training.batch_size == 1

    def test_at_least_one_persona_required(self):
        with pytest.raises(ValueError, match="At least one teacher persona required"):
            CodeAspireConfig(teacher=TeacherConfig(personas=[]))

    def test_batch_size_checked_before_personas(self):
        with pytest.raises(ValueError, match="batch_size"):
            CodeAspireConfig(
                teacher=TeacherConfig(personas=[]),
                training=TrainingConfig(batch_size=0),
            )
