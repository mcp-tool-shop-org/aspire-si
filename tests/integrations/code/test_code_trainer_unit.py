"""
Unit tests for integrations/code/trainer.py (AspireCodeTrainer).

Everything runs on CPU with a tiny critic (hidden_dim=16), a character-level
fake tokenizer, and a tiny language model standing in for the student. No
HuggingFace download, no network, no GPU.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from integrations.code.code_critic import CodeCritic, CriticOutput
from integrations.code.code_teacher import CodeCritique, CodeSample, CodeTeacher
from integrations.code.config import CodeAspireConfig, Language, TrainingConfig
from integrations.code.data import CodeReviewPair, save_training_data
from integrations.code.trainer import AspireCodeTrainer, TrainingMetrics

from .conftest import FakeTokenizer

# ============================================================================
# Helpers
# ============================================================================

VOCAB = 100


class TinyLM(nn.Module):
    """Minimal causal-LM look-alike: returns an object with a ``loss`` attribute."""

    def __init__(self, vocab: int = VOCAB, dim: int = 8):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.out = nn.Linear(dim, vocab)
        self.vocab = vocab
        self.saved: list[str] = []

    def forward(self, input_ids, attention_mask=None, labels=None):
        logits = self.out(self.emb(input_ids))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.reshape(-1, self.vocab), labels.reshape(-1))
        return SimpleNamespace(loss=loss, logits=logits)

    def save_pretrained(self, path):
        self.saved.append(path)


class ConstLM(nn.Module):
    """Student whose language-modelling loss is exactly ``value`` (but still differentiable)."""

    def __init__(self, value: float = 2.0):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(1))
        self.value = value

    def forward(self, input_ids, attention_mask=None, labels=None):
        return SimpleNamespace(loss=(self.w * 0).sum() + self.value)

    def save_pretrained(self, path):
        pass


class ConstCritic(nn.Module):
    """Critic that always predicts ``value`` for every sample."""

    def __init__(self, value: float = 7.0):
        super().__init__()
        self.p = nn.Parameter(torch.zeros(1))
        self.value = value
        self.last_mode: bool | None = None

    def forward(self, input_ids, attention_mask=None):
        self.last_mode = self.training
        batch = input_ids.shape[0]
        return CriticOutput(score=torch.full((batch,), self.value) + self.p * 0)


def make_pairs(scores=(2.0, 4.0, 6.0, 8.0, 9.0)) -> list[CodeReviewPair]:
    pairs = []
    for i, score in enumerate(scores):
        critique = CodeCritique(
            overall_score=score,
            reasoning="r",
            weaknesses=[f"issue {i}"],
            teacher_name="T",
            language=Language.PYTHON,
        )
        pairs.append(
            CodeReviewPair(
                code=f"def f{i}(x):\n    return x + {i * 7}\n",
                language=Language.PYTHON,
                critique=critique,
            )
        )
    return pairs


@pytest.fixture
def config(tiny_critic_config, tmp_path):
    return CodeAspireConfig(
        critic=tiny_critic_config,
        training=TrainingConfig(
            student_model="stub/model",
            batch_size=2,
            gradient_accumulation_steps=2,
            epochs=2,
            max_length=16,
            critic_lr=1e-2,
            student_lr=1e-2,
            student_reward_weight=1.0,
            save_frequency=1,
            checkpoint_dir=str(tmp_path / "ckpt"),
        ),
        device="cpu",
    )


@pytest.fixture
def teacher():
    return CodeTeacher(personas=["architecture_reviewer"])


@pytest.fixture
def make_trainer(config, teacher, tiny_critic):
    def _make(student=None, **kw):
        kw.setdefault("teacher", teacher)
        kw.setdefault("critic", tiny_critic)
        # An explicit non-string, non-None student avoids touching HuggingFace.
        with patch.object(AspireCodeTrainer, "_load_student_model", return_value=(None, None)):
            return AspireCodeTrainer(config=config, student_model=student, **kw)

    return _make


@pytest.fixture
def trainer(make_trainer):
    """Critic-only trainer (no student)."""
    return make_trainer(student=None)


@pytest.fixture
def student_trainer(make_trainer, fake_tokenizer):
    t = make_trainer(student=TinyLM())
    t.tokenizer = fake_tokenizer
    return t


def snapshot(module: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def changed(before: dict[str, torch.Tensor], module: nn.Module) -> bool:
    return any(not torch.equal(before[k], v) for k, v in module.state_dict().items())


# ============================================================================
# TrainingMetrics
# ============================================================================


def test_training_metrics_dataclass_fields():
    m = TrainingMetrics(1, 0.5, 0.25, 6.0, 7.0, 1.0, 3.2)
    assert (m.epoch, m.critic_loss, m.student_loss) == (1, 0.5, 0.25)
    assert (m.mean_predicted_score, m.mean_actual_score, m.score_mae, m.time_elapsed) == (6.0, 7.0, 1.0, 3.2)


# ============================================================================
# __init__
# ============================================================================


class TestInit:
    def test_default_config_is_created_when_none(self, config, teacher, tiny_critic):
        with patch("integrations.code.trainer.CodeAspireConfig", return_value=config) as cfg_cls:
            t = AspireCodeTrainer(student_model=TinyLM(), teacher=teacher, critic=tiny_critic)
        cfg_cls.assert_called_once_with()
        assert t.config is config
        assert t.device == "cpu"

    def test_teacher_built_from_config_when_missing(self, config, tiny_critic):
        config.teacher.personas = ["style_guide", "security_auditor"]
        config.teacher.strategy = "rotate"
        config.teacher.use_llm_teacher = False
        config.teacher.llm_model = "some-model"
        with patch("integrations.code.trainer.CodeTeacher") as teacher_cls:
            t = AspireCodeTrainer(config=config, student_model=TinyLM(), critic=tiny_critic)
        teacher_cls.assert_called_once_with(
            personas=["style_guide", "security_auditor"],
            strategy="rotate",
            use_llm=False,
            llm_model="some-model",
        )
        assert t.teacher is teacher_cls.return_value

    def test_critic_built_from_critic_config_when_missing(self, config, teacher):
        t = AspireCodeTrainer(config=config, student_model=TinyLM(), teacher=teacher)
        assert isinstance(t.critic, CodeCritic)
        assert t.critic.config is config.critic
        assert t.critic.encoder.hidden_dim == 16

    def test_supplied_components_are_used_and_moved_to_device(self, config, teacher):
        critic, student = ConstCritic(), TinyLM()
        t = AspireCodeTrainer(config=config, student_model=student, teacher=teacher, critic=critic)
        assert t.teacher is teacher
        assert t.critic is critic
        assert t.student is student
        assert t.tokenizer is None  # must be supplied separately for module students
        assert next(t.student.parameters()).device.type == "cpu"

    def test_optimizers_use_configured_learning_rates(self, student_trainer):
        assert student_trainer.critic_optimizer.param_groups[0]["lr"] == 1e-2
        assert student_trainer.student_optimizer.param_groups[0]["lr"] == 1e-2
        n_student = sum(len(g["params"]) for g in student_trainer.student_optimizer.param_groups)
        assert n_student == len(list(student_trainer.student.parameters()))

    def test_no_student_means_no_student_optimizer(self, trainer):
        assert trainer.student is None
        assert trainer.student_optimizer is None
        assert trainer.metrics_history == []

    def test_string_student_triggers_loader_with_config_default(self, config, teacher, tiny_critic):
        with patch.object(
            AspireCodeTrainer, "_load_student_model", return_value=(TinyLM(), "tok")
        ) as loader:
            t = AspireCodeTrainer(config=config, teacher=teacher, critic=tiny_critic)
        loader.assert_called_once_with("stub/model")
        assert t.tokenizer == "tok"
        assert isinstance(t.student, TinyLM)

    def test_explicit_string_student_overrides_config(self, config, teacher, tiny_critic):
        with patch.object(
            AspireCodeTrainer, "_load_student_model", return_value=(TinyLM(), "tok")
        ) as loader:
            AspireCodeTrainer(config=config, student_model="other/model", teacher=teacher, critic=tiny_critic)
        loader.assert_called_once_with("other/model")

    def test_wandb_initialised_when_enabled(self, config, teacher, tiny_critic):
        config.training.use_wandb = True
        config.training.wandb_project = "proj-x"
        fake_wandb = MagicMock()
        with patch.dict(sys.modules, {"wandb": fake_wandb}):
            t = AspireCodeTrainer(config=config, student_model=TinyLM(), teacher=teacher, critic=tiny_critic)
        fake_wandb.init.assert_called_once_with(project="proj-x")
        assert t.use_wandb is True

    def test_wandb_disabled_when_not_installed(self, config, teacher, tiny_critic):
        config.training.use_wandb = True
        with patch.dict(sys.modules, {"wandb": None}):
            t = AspireCodeTrainer(config=config, student_model=TinyLM(), teacher=teacher, critic=tiny_critic)
        assert t.use_wandb is False

    def test_wandb_not_touched_when_disabled(self, student_trainer):
        assert student_trainer.use_wandb is False


# ============================================================================
# _load_student_model
# ============================================================================


class TestLoadStudentModel:
    @pytest.fixture
    def hf(self):
        tokenizer = MagicMock()
        tokenizer.pad_token = None
        tokenizer.eos_token = "<eos>"
        model = MagicMock(name="base_model")
        peft_model = MagicMock(name="peft_model")
        with patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer) as tok_fp, patch(
            "transformers.AutoModelForCausalLM.from_pretrained", return_value=model
        ) as model_fp, patch("peft.get_peft_model", return_value=peft_model) as get_peft:
            yield SimpleNamespace(
                tokenizer=tokenizer, model=model, peft_model=peft_model,
                tok_fp=tok_fp, model_fp=model_fp, get_peft=get_peft,
            )

    def test_loads_with_lora_and_sets_pad_token(self, trainer, hf):
        trainer.config.training.lora_r = 4
        trainer.config.training.lora_alpha = 9
        trainer.config.training.lora_dropout = 0.2
        model, tokenizer = trainer._load_student_model("org/m")

        hf.tok_fp.assert_called_once_with("org/m")
        assert hf.model_fp.call_args.args == ("org/m",)
        assert hf.model_fp.call_args.kwargs["torch_dtype"] == torch.float16
        assert hf.model_fp.call_args.kwargs["device_map"] is None  # cpu
        assert tokenizer.pad_token == "<eos>"
        peft_cfg = hf.get_peft.call_args.args[1]
        assert (peft_cfg.r, peft_cfg.lora_alpha, peft_cfg.lora_dropout) == (4, 9, 0.2)
        assert hf.get_peft.call_args.args[0] is hf.model
        hf.peft_model.print_trainable_parameters.assert_called_once()
        assert model is hf.peft_model

    def test_existing_pad_token_is_kept(self, trainer, hf):
        hf.tokenizer.pad_token = "<pad>"
        _, tokenizer = trainer._load_student_model("org/m")
        assert tokenizer.pad_token == "<pad>"

    def test_lora_can_be_disabled(self, trainer, hf):
        trainer.config.training.use_lora = False
        model, _ = trainer._load_student_model("org/m")
        hf.get_peft.assert_not_called()
        assert model is hf.model

    def test_cuda_uses_auto_device_map(self, trainer, hf):
        trainer.device = "cuda"
        trainer._load_student_model("org/m")
        assert hf.model_fp.call_args.kwargs["device_map"] == "auto"

    def test_missing_dependency_returns_none_pair(self, trainer, capsys):
        with patch.dict(sys.modules, {"peft": None}):
            assert trainer._load_student_model("org/m") == (None, None)
        out = capsys.readouterr().out
        assert "Could not load model" in out
        assert "pip install transformers peft" in out


# ============================================================================
# _get_trainable_params
# ============================================================================


class TestTrainableParams:
    def test_only_requires_grad_params_returned(self, student_trainer):
        student_trainer.student.emb.weight.requires_grad_(False)
        params = student_trainer._get_trainable_params()
        assert all(p.requires_grad for p in params)
        assert len(params) == len(list(student_trainer.student.parameters())) - 1

    def test_object_without_parameters_gives_empty_list(self, trainer):
        trainer.student = object()
        assert trainer._get_trainable_params() == []


# ============================================================================
# train_critic / _validate_critic
# ============================================================================


class TestTrainCritic:
    def test_returns_one_finite_loss_per_epoch_and_updates_weights(self, trainer, capsys):
        before = snapshot(trainer.critic)
        losses = trainer.train_critic(make_pairs(), epochs=3)
        assert len(losses) == 3
        assert all(isinstance(x, float) and np.isfinite(x) and x > 0 for x in losses)
        assert changed(before, trainer.critic)
        out = capsys.readouterr().out
        assert "Training critic for 3 epochs on 5 samples" in out
        assert "Epoch 3/3" in out

    def test_defaults_to_config_epochs(self, trainer):
        assert len(trainer.train_critic(make_pairs())) == trainer.config.training.epochs == 2

    def test_loss_decreases_when_overfitting_small_set(self, trainer):
        torch.manual_seed(0)
        pairs = make_pairs()
        trainer.config.training.batch_size = 5
        losses = trainer.train_critic(pairs, epochs=25)
        assert losses[-1] < losses[0] * 0.5

    def test_critic_left_in_train_mode_without_validation(self, trainer):
        trainer.train_critic(make_pairs(), epochs=1)
        assert trainer.critic.training is True

    def test_checkpoint_saved_every_epoch_with_expected_contents(self, trainer, config, capsys):
        from pathlib import Path

        trainer.train_critic(make_pairs(), epochs=2)
        ckpt = Path(config.training.checkpoint_dir)
        assert sorted(p.name for p in ckpt.iterdir()) == ["critic_epoch_1.pt", "critic_epoch_2.pt"]
        data = torch.load(ckpt / "critic_epoch_2.pt")
        assert data["epoch"] == 2
        assert set(data) == {"epoch", "model_state_dict", "optimizer_state_dict"}
        assert set(data["model_state_dict"]) == set(trainer.critic.state_dict())
        assert "Saved critic checkpoint" in capsys.readouterr().out

    def test_save_frequency_respected(self, trainer, config):
        from pathlib import Path

        config.training.save_frequency = 2
        trainer.train_critic(make_pairs(), epochs=3)
        names = sorted(p.name for p in Path(config.training.checkpoint_dir).iterdir())
        assert names == ["critic_epoch_2.pt"]

    def test_validation_runs_each_epoch_and_is_reported(self, trainer, capsys):
        val = make_pairs((3.0, 7.0))
        with patch.object(trainer, "_validate_critic", return_value=1.2345) as validate:
            trainer.train_critic(make_pairs(), val_pairs=val, epochs=2)
        assert validate.call_count == 2
        assert validate.call_args.args[0] is val
        assert "Validation Loss: 1.2345" in capsys.readouterr().out

    def test_validation_skipped_for_empty_list(self, trainer):
        with patch.object(trainer, "_validate_critic") as validate:
            trainer.train_critic(make_pairs(), val_pairs=[], epochs=1)
        validate.assert_not_called()

    def test_trainer_tokenizer_preferred_over_critic_tokenizer(self, trainer, fake_tokenizer):
        own = FakeTokenizer()
        trainer.tokenizer = own
        trainer.train_critic(make_pairs(), epochs=1)
        assert own.calls and all(c["padding"] == "max_length" and c["max_length"] == 16 for c in own.calls)
        assert fake_tokenizer.calls == []  # critic's tokenizer untouched

    def test_critic_tokenizer_used_when_trainer_has_none(self, trainer, fake_tokenizer):
        assert trainer.tokenizer is None
        trainer.train_critic(make_pairs(), epochs=1)
        assert len(fake_tokenizer.calls) == 5  # one per sample

    def test_gradient_clipping_applied(self, trainer):
        with patch("torch.nn.utils.clip_grad_norm_") as clip:
            trainer.train_critic(make_pairs(), epochs=1)
        assert clip.call_count == 3  # 5 samples / batch 2
        assert clip.call_args.args[1] == 1.0

    def test_validate_critic_matches_manual_mean_batch_mse(self, trainer, fake_tokenizer):
        pairs = make_pairs((1.0, 5.0, 9.0))
        trainer.critic.train()
        got = trainer._validate_critic(pairs, fake_tokenizer)

        assert trainer.critic.training is False  # switched to eval
        with torch.no_grad():
            preds = []
            for pair in pairs:
                tok = fake_tokenizer(pair.code, padding="max_length", truncation=True, max_length=16)
                preds.append(trainer.critic(tok["input_ids"], tok["attention_mask"]).score.item())
        actual = [p.critique.overall_score for p in pairs]
        batch1 = np.mean([(preds[i] - actual[i]) ** 2 for i in (0, 1)])
        batch2 = (preds[2] - actual[2]) ** 2
        assert got == pytest.approx(float((batch1 + batch2) / 2), rel=1e-4)

    def test_validate_critic_does_not_change_weights(self, trainer, fake_tokenizer):
        before = snapshot(trainer.critic)
        trainer._validate_critic(make_pairs(), fake_tokenizer)
        assert not changed(before, trainer.critic)

    def test_validate_critic_empty_returns_zero(self, trainer, fake_tokenizer):
        assert trainer._validate_critic([], fake_tokenizer) == 0.0


# ============================================================================
# train_student
# ============================================================================


class TestTrainStudent:
    def test_requires_student(self, trainer):
        with pytest.raises(ValueError, match="No student model loaded"):
            trainer.train_student(make_pairs())

    @pytest.mark.parametrize(
        "lm_loss,critic_score,weight,expected",
        [
            (2.0, 7.0, 1.0, 2.0 - (7.0 - 5.0) / 5.0),   # high critic score lowers the loss
            (2.0, 3.0, 1.0, 2.0 + (5.0 - 3.0) / 5.0),   # low critic score raises it
            (2.0, 10.0, 0.5, 2.0 - 0.5 * 1.0),
            (1.5, 5.0, 3.0, 1.5),                       # neutral score -> no shaping
            (2.0, 7.0, 0.0, 2.0),                       # zero weight disables reward shaping
        ],
    )
    def test_epoch_loss_combines_lm_loss_and_critic_reward(
        self, make_trainer, fake_tokenizer, lm_loss, critic_score, weight, expected
    ):
        t = make_trainer(student=ConstLM(lm_loss), critic=ConstCritic(critic_score))
        t.tokenizer = fake_tokenizer
        t.config.training.student_reward_weight = weight
        losses = t.train_student(make_pairs(), epochs=2)
        assert losses == pytest.approx([expected, expected], abs=1e-5)

    @pytest.mark.parametrize("accum,expected_steps", [(1, 3), (2, 1), (3, 1), (4, 0)])
    def test_optimizer_steps_follow_gradient_accumulation(
        self, make_trainer, fake_tokenizer, accum, expected_steps
    ):
        t = make_trainer(student=ConstLM(), critic=ConstCritic())
        t.tokenizer = fake_tokenizer
        t.config.training.gradient_accumulation_steps = accum
        with patch.object(t.student_optimizer, "step") as step:
            t.train_student(make_pairs(), epochs=1)  # 5 samples / batch 2 -> 3 batches
        assert step.call_count == expected_steps

    def test_student_weights_change_and_critic_is_frozen(self, student_trainer):
        student_before = snapshot(student_trainer.student)
        critic_before = snapshot(student_trainer.critic)
        student_trainer.config.training.gradient_accumulation_steps = 1
        losses = student_trainer.train_student(make_pairs(), epochs=2)
        assert len(losses) == 2 and all(np.isfinite(losses))
        assert changed(student_before, student_trainer.student)
        assert not changed(critic_before, student_trainer.critic)
        assert student_trainer.critic.training is False
        assert student_trainer.student.training is True
        assert all(p.grad is None for p in student_trainer.critic.parameters())

    def test_student_loss_decreases_on_repeated_data(self, student_trainer):
        torch.manual_seed(0)
        cfg = student_trainer.config.training
        cfg.gradient_accumulation_steps = 1
        cfg.batch_size = 5
        cfg.student_reward_weight = 0.0  # isolate the LM objective
        losses = student_trainer.train_student(make_pairs(), epochs=20)
        assert losses[-1] < losses[0]

    def test_student_mode_dataset_uses_trainer_tokenizer(self, student_trainer):
        student_trainer.train_student(make_pairs(), epochs=1)
        tok = student_trainer.tokenizer
        # student mode tokenises prompt and target per sample
        assert len(tok.calls) == 10
        assert all(c["max_length"] == 16 for c in tok.calls)

    def test_checkpoints_saved_for_student_and_tokenizer(self, student_trainer, config):
        student_trainer.train_student(make_pairs(), epochs=2)
        from pathlib import Path

        expected = [str(Path(config.training.checkpoint_dir) / f"student_epoch_{e}") for e in (1, 2)]
        assert student_trainer.student.saved == expected
        assert student_trainer.tokenizer.saved_to == expected

    def test_student_save_frequency_respected(self, student_trainer):
        student_trainer.config.training.save_frequency = 2
        student_trainer.train_student(make_pairs(), epochs=3)
        assert len(student_trainer.student.saved) == 1
        assert student_trainer.student.saved[0].endswith("student_epoch_2")

    def test_gradient_clip_uses_trainable_params(self, student_trainer):
        student_trainer.config.training.gradient_accumulation_steps = 1
        with patch("torch.nn.utils.clip_grad_norm_") as clip:
            student_trainer.train_student(make_pairs(), epochs=1)
        assert clip.call_count == 3
        params = clip.call_args.args[0]
        assert len(params) == len(list(student_trainer.student.parameters()))

    def test_student_training_falls_back_to_critic_tokenizer(self, make_trainer):
        t = make_trainer(student=TinyLM())  # tokenizer is None by design ("must be provided separately")
        assert t.tokenizer is None
        losses = t.train_student(make_pairs(), epochs=1)
        assert len(losses) == 1


# ============================================================================
# train (full pipeline)
# ============================================================================


class TestTrain:
    def test_critic_only_pipeline_with_student_missing(self, trainer, capsys):
        result = trainer.train(make_pairs(), critic_epochs=1, student_epochs=5)
        assert set(result) == {"critic_losses", "student_losses"}
        assert len(result["critic_losses"]) == 1
        assert result["student_losses"] == []
        out = capsys.readouterr().out
        assert "ASPIRE Code Training" in out
        assert "Training samples: 5" in out
        assert "Skipping student training (no student model)" in out
        assert "Training Complete!" in out

    def test_default_epochs_come_from_config(self, trainer):
        result = trainer.train(make_pairs())
        assert len(result["critic_losses"]) == 2

    def test_both_phases_run_with_independent_epoch_counts(self, student_trainer, capsys):
        result = student_trainer.train(make_pairs(), critic_epochs=1, student_epochs=3)
        assert len(result["critic_losses"]) == 1
        assert len(result["student_losses"]) == 3
        out = capsys.readouterr().out
        assert out.index("Phase 1: Training Critic") < out.index("Phase 2: Training Student")

    def test_paths_are_loaded_from_json(self, student_trainer, tmp_path):
        train_file = tmp_path / "train.json"
        val_file = tmp_path / "val.json"
        save_training_data(make_pairs(), str(train_file))
        save_training_data(make_pairs((5.0, 6.0)), str(val_file))
        with patch.object(student_trainer, "train_critic", wraps=student_trainer.train_critic) as tc:
            result = student_trainer.train(str(train_file), str(val_file), critic_epochs=1, student_epochs=1)
        train_arg, val_arg = tc.call_args.args
        assert [p.code for p in train_arg] == [p.code for p in make_pairs()]
        assert [p.critique.overall_score for p in val_arg] == [5.0, 6.0]
        assert len(result["student_losses"]) == 1

    def test_validation_data_forwarded(self, trainer):
        val = make_pairs((4.0,))
        with patch.object(trainer, "train_critic", return_value=[0.0]) as tc:
            trainer.train(make_pairs(), val)
        assert tc.call_args.args[1] is val


# ============================================================================
# checkpoints
# ============================================================================


class TestCheckpoints:
    def test_critic_checkpoint_round_trip(self, trainer, config):
        from pathlib import Path

        trainer._save_checkpoint(7, "critic")
        path = Path(config.training.checkpoint_dir) / "critic_epoch_7.pt"
        assert path.exists()
        saved = snapshot(trainer.critic)

        with torch.no_grad():
            for p in trainer.critic.parameters():
                p.add_(1.0)
        assert changed(saved, trainer.critic)

        trainer.load_checkpoint(critic_path=str(path))
        assert not changed(saved, trainer.critic)

    def test_checkpoint_dir_created_recursively(self, trainer, config):
        from pathlib import Path

        config.training.checkpoint_dir = str(Path(config.training.checkpoint_dir) / "a" / "b")
        trainer._save_checkpoint(1, "critic")
        assert (Path(config.training.checkpoint_dir) / "critic_epoch_1.pt").exists()

    def test_student_checkpoint_saves_model_and_tokenizer(self, student_trainer, config):
        student_trainer._save_checkpoint(3, "student")
        from pathlib import Path

        expected = str(Path(config.training.checkpoint_dir) / "student_epoch_3")
        assert student_trainer.student.saved == [expected]
        assert student_trainer.tokenizer.saved_to == [expected]

    def test_student_checkpoint_without_tokenizer(self, make_trainer):
        t = make_trainer(student=TinyLM())
        t._save_checkpoint(1, "student")
        assert len(t.student.saved) == 1

    def test_unknown_component_is_rejected_cleanly(self, trainer):
        try:
            trainer._save_checkpoint(1, "bogus")
        except UnboundLocalError:
            pytest.fail("UnboundLocalError leaked from _save_checkpoint")
        except ValueError:
            pass

    def test_load_checkpoint_with_no_paths_is_noop(self, trainer, capsys):
        saved = snapshot(trainer.critic)
        trainer.load_checkpoint()
        assert not changed(saved, trainer.critic)
        assert capsys.readouterr().out == ""

    def test_load_student_checkpoint_rewraps_base_model(self, student_trainer, capsys):
        base = object()
        student_trainer.student = SimpleNamespace(base_model=SimpleNamespace(model=base))
        sentinel = object()
        with patch("peft.PeftModel.from_pretrained", return_value=sentinel) as fp:
            student_trainer.load_checkpoint(student_path="some/dir")
        fp.assert_called_once_with(base, "some/dir")
        assert student_trainer.student is sentinel
        assert "Loaded student from some/dir" in capsys.readouterr().out

    def test_student_path_ignored_without_student(self, trainer):
        with patch("peft.PeftModel.from_pretrained") as fp:
            trainer.load_checkpoint(student_path="some/dir")
        fp.assert_not_called()


# ============================================================================
# evaluate / score_code / critique_code
# ============================================================================


class TestEvaluationAndScoring:
    def _predictions(self, trainer, pairs, tokenizer):
        preds = []
        trainer.critic.eval()
        for pair in pairs:
            tok = tokenizer(pair.code, padding="max_length", truncation=True, max_length=16)
            with torch.no_grad():
                preds.append(trainer.critic(tok["input_ids"], tok["attention_mask"]).score.item())
        return np.array(preds)

    def test_evaluate_metrics_match_manual_computation(self, trainer, fake_tokenizer):
        pairs = make_pairs()
        metrics = trainer.evaluate(pairs)

        preds = self._predictions(trainer, pairs, fake_tokenizer)
        actual = np.array([p.critique.overall_score for p in pairs])
        assert set(metrics) == {"mae", "mse", "rmse", "correlation", "mean_predicted", "mean_actual"}
        assert metrics["mae"] == pytest.approx(np.abs(preds - actual).mean())
        assert metrics["mse"] == pytest.approx(((preds - actual) ** 2).mean())
        assert metrics["rmse"] == pytest.approx(np.sqrt(metrics["mse"]))
        assert metrics["mean_actual"] == pytest.approx(actual.mean())
        assert metrics["mean_predicted"] == pytest.approx(preds.mean())
        assert metrics["correlation"] == pytest.approx(np.corrcoef(preds, actual)[0, 1], nan_ok=True)

    def test_evaluate_puts_critic_in_eval_mode_and_keeps_weights(self, trainer):
        trainer.critic.train()
        before = snapshot(trainer.critic)
        trainer.evaluate(make_pairs())
        assert trainer.critic.training is False
        assert not changed(before, trainer.critic)

    def test_evaluate_prefers_trainer_tokenizer(self, trainer, fake_tokenizer):
        own = FakeTokenizer()
        trainer.tokenizer = own
        trainer.evaluate(make_pairs())
        assert len(own.calls) == 5
        assert fake_tokenizer.calls == []

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_evaluate_perfect_critic_has_zero_error(self, trainer):
        trainer.critic = ConstCritic(6.0)
        pairs = make_pairs((6.0, 6.0, 6.0))
        trainer.tokenizer = FakeTokenizer()
        metrics = trainer.evaluate(pairs)
        assert metrics["mae"] == 0.0
        assert metrics["rmse"] == 0.0
        assert metrics["mean_predicted"] == 6.0
        assert np.isnan(metrics["correlation"])  # constant series have no defined correlation

    def test_score_code_returns_python_float_in_range(self, trainer):
        score = trainer.score_code("def f(): return 1")
        assert isinstance(score, float)
        assert 0.0 <= score <= 10.0

    def test_score_code_delegates_to_critic_with_device(self, trainer):
        with patch.object(trainer.critic, "score_code", return_value=torch.tensor([4.25])) as sc:
            assert trainer.score_code("x = 1") == pytest.approx(4.25)
        sc.assert_called_once_with("x = 1", device="cpu")

    def test_critique_code_overrides_teacher_score_with_critic_score(self, trainer):
        teacher_critique = CodeCritique(
            overall_score=9.9, reasoning="teacher says fine", teacher_name="T", language=Language.PYTHON
        )
        trainer.teacher = MagicMock()
        trainer.teacher.critique.return_value = teacher_critique
        code = "import os\n\ndef f():\n    return os.getcwd()\n"
        with patch.object(trainer, "score_code", return_value=3.5) as score:
            result = trainer.critique_code(code)

        score.assert_called_once_with(code)
        assert result.overall_score == 3.5
        assert result.reasoning == "teacher says fine"  # detailed feedback kept from the teacher
        sample = trainer.teacher.critique.call_args.args[0]
        assert isinstance(sample, CodeSample)
        assert sample.code == code
        assert sample.language == Language.PYTHON  # detected from content

    def test_critique_code_end_to_end_with_real_teacher(self, trainer):
        code = 'def add(a: int, b: int) -> int:\n    """Add."""\n    return a + b\n'
        result = trainer.critique_code(code)
        assert isinstance(result, CodeCritique)
        assert result.teacher_name == "Code Teacher Committee"
        assert "Architecture Reviewer" in result.reasoning
        assert result.overall_score == pytest.approx(trainer.score_code(code))
