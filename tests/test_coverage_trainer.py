"""Coverage-closing tests for aspire.trainer (geometry export, 8-bit optimizer, main)."""

import runpy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from aspire.config import AspireConfig


def _dialogue(evaluation):
    return SimpleNamespace(prompt="p", scored_response="r", final_evaluation=evaluation)


def _encoded(batch, length):
    """Stand in for tokenizing prompts and responses."""
    return patch(
        "aspire.trainer.encode_exchanges",
        return_value=(torch.zeros(batch, length, dtype=torch.long), torch.ones(batch, length, dtype=torch.long)),
    )


@pytest.fixture
def make_trainer(tmp_path):
    """Build an AspireTrainer with every heavy dependency mocked out."""
    patches = {
        name: patch(f"aspire.trainer.{name}")
        for name in (
            "AutoModelForCausalLM",
            "AutoTokenizer",
            "get_peft_model",
            "prepare_model_for_kbit_training",
            "CriticHead",
            "SeparateCritic",
            "SharedEncoderCritic",
            "get_teacher",
            "AspireLoss",
            "DialogueGenerator",
            "DialogueManager",
            "DialogueFormatter",
            "AdamW",
            "GeometryRecorder",
            "console",
        )
    }
    mocks = {name: p.start() for name, p in patches.items()}

    model = MagicMock()
    model.config.hidden_size = 8
    model.parameters.return_value = iter([torch.nn.Parameter(torch.zeros(2, 2))])
    mocks["AutoModelForCausalLM"].from_pretrained.return_value = model
    mocks["get_peft_model"].return_value = model
    mocks["prepare_model_for_kbit_training"].return_value = model

    tokenizer = MagicMock()
    tokenizer.pad_token = "<pad>"
    mocks["AutoTokenizer"].from_pretrained.return_value = tokenizer

    critic = MagicMock()
    critic.to.return_value = critic
    critic.get_trainable_parameters.return_value = [torch.nn.Parameter(torch.zeros(2, 2))]
    mocks["CriticHead"].return_value = critic

    mocks["model"] = model
    mocks["critic"] = critic
    mocks["tmp_path"] = tmp_path

    def build(**overrides):
        from aspire.trainer import AspireTrainer

        config = AspireConfig()
        config.device = "cpu"
        config.training.output_dir = tmp_path / "out"
        config.student.load_in_4bit = False
        config.student.load_in_8bit = False
        config.student.use_lora = False
        config.training.optimizer = "adamw"
        for key, value in overrides.items():
            if "__" in key:
                section, attr = key.split("__")
                setattr(getattr(config, section), attr, value)
            else:
                setattr(config, key, value)
        return AspireTrainer(config), config

    mocks["build"] = build
    yield mocks
    for p in patches.values():
        p.stop()


class TestGeometryWiring:
    def test_no_recorder_by_default(self, make_trainer):
        trainer, _ = make_trainer["build"]()
        assert trainer.geometry is None
        make_trainer["GeometryRecorder"].assert_not_called()

    def test_recorder_built_from_config(self, make_trainer):
        make_trainer["get_teacher"].return_value.name = "Prof Plum"
        trainer, config = make_trainer["build"](
            training__geometry_export=True,
            training__geometry_every=3,
            training__geometry_window=7,
            experiment_name="exp-1",
            seed=123,
        )
        make_trainer["GeometryRecorder"].assert_called_once_with(
            run_id="exp-1",
            condition="Prof Plum",
            seed=123,
            window=7,
            every=3,
            # three epochs (the default) replay the cached first-epoch scores
            scalar_source="replayed",
        )
        assert trainer.geometry is make_trainer["GeometryRecorder"].return_value

    def test_write_geometry_success_returns_path_string(self, make_trainer):
        trainer, config = make_trainer["build"](training__geometry_export=True, training__num_epochs=5)
        trainer.geometry.write.return_value = Path("somewhere") / "geometry.json"
        result = trainer._write_geometry(training_items=12)
        assert result == str(Path("somewhere") / "geometry.json")
        args, kwargs = trainer.geometry.write.call_args
        assert args[0] == Path(config.training.output_dir) / "geometry.json"
        assert kwargs == {"training_items": 12, "cycles": 5}

    def test_write_geometry_value_error_gives_none_and_reports(self, make_trainer):
        trainer, _ = make_trainer["build"](training__geometry_export=True)
        trainer.geometry.write.side_effect = ValueError("too few steps")
        assert trainer._write_geometry(3) is None
        printed = " ".join(str(c.args[0]) for c in make_trainer["console"].print.call_args_list)
        assert "No geometry export: too few steps" in printed

    def test_train_reports_geometry_export_in_metrics(self, make_trainer):
        trainer, _ = make_trainer["build"](training__geometry_export=True, training__num_epochs=2)
        trainer.geometry.write.return_value = Path("g.json")
        epoch_metrics = {"loss": 1.0, "critic_loss": 0.5, "student_loss": 0.25}
        with patch("aspire.trainer.get_scheduler"), patch.object(
            trainer, "_train_epoch", return_value=epoch_metrics
        ), patch.object(trainer, "_save_checkpoint") as save:
            metrics = trainer.train(["a", "b", "c", "d"])
        assert metrics["geometry_export"] == "g.json"
        assert metrics["train_loss"] == [1.0, 1.0]
        assert [c.args[0] for c in save.call_args_list] == [1, 2]
        _, kwargs = trainer.geometry.write.call_args
        assert kwargs == {"training_items": 4, "cycles": 2}

    def test_a_one_epoch_run_has_live_scalars(self, make_trainer):
        make_trainer["build"](training__geometry_export=True, training__num_epochs=1)
        assert make_trainer["GeometryRecorder"].call_args.kwargs["scalar_source"] == "live"

    def test_train_writes_the_export_after_every_epoch(self, make_trainer):
        """A run stopped early (a deadline, Ctrl+C) keeps the steps it recorded."""
        trainer, _ = make_trainer["build"](training__geometry_export=True, training__num_epochs=3)
        trainer.geometry.write.return_value = Path("g.json")
        epoch_metrics = {"loss": 1.0, "critic_loss": 0.5, "student_loss": 0.25}
        with patch("aspire.trainer.get_scheduler"), patch.object(
            trainer, "_train_epoch", return_value=epoch_metrics
        ), patch.object(trainer, "_save_checkpoint"):
            trainer.train(["a", "b"])
        cycles = [c.kwargs["cycles"] for c in trainer.geometry.write.call_args_list]
        assert cycles == [1, 2, 3]

    def test_an_epoch_export_too_short_to_write_is_quiet(self, make_trainer):
        trainer, _ = make_trainer["build"](training__geometry_export=True)
        trainer.geometry.write.side_effect = ValueError("too few steps")
        assert trainer._write_geometry(3, cycles=1, quiet=True) is None
        printed = " ".join(str(c.args[0]) for c in make_trainer["console"].print.call_args_list)
        assert "No geometry export" not in printed

    def test_train_geometry_failure_records_none(self, make_trainer):
        trainer, _ = make_trainer["build"](training__geometry_export=True, training__num_epochs=1)
        trainer.geometry.write.side_effect = ValueError("nope")
        epoch_metrics = {"loss": 1.0, "critic_loss": 0.5, "student_loss": 0.25}
        with patch("aspire.trainer.get_scheduler"), patch.object(
            trainer, "_train_epoch", return_value=epoch_metrics
        ), patch.object(trainer, "_save_checkpoint"):
            metrics = trainer.train(["a"])
        assert "geometry_export" in metrics and metrics["geometry_export"] is None

    def test_train_without_geometry_has_no_export_key(self, make_trainer):
        trainer, _ = make_trainer["build"](training__num_epochs=1)
        epoch_metrics = {"loss": 1.0, "critic_loss": 0.5, "student_loss": 0.25}
        with patch("aspire.trainer.get_scheduler"), patch.object(
            trainer, "_train_epoch", return_value=epoch_metrics
        ), patch.object(trainer, "_save_checkpoint"):
            metrics = trainer.train(["a"])
        assert "geometry_export" not in metrics

    def test_compute_batch_loss_records_geometry_with_teacher_evals(self, make_trainer):
        trainer, _ = make_trainer["build"](training__geometry_export=True)
        hidden = torch.randn(2, 3, 8)
        trainer.student_model = MagicMock(return_value=SimpleNamespace(hidden_states=[hidden * 0, hidden]))
        trainer.critic = MagicMock(
            return_value=SimpleNamespace(score=torch.tensor([1.0, 2.0]), reasoning_embedding=torch.zeros(2, 4))
        )
        trainer.loss_fn = MagicMock(return_value={"total": torch.tensor(1.0)})
        trainer.teacher = SimpleNamespace(name="Prof Plum")
        evals = [SimpleNamespace(overall_score=7.0), SimpleNamespace(overall_score=3.0)]
        dialogues = [
            SimpleNamespace(prompt=f"p{i}", scored_response=f"r{i}", final_evaluation=e)
            for i, e in enumerate(evals)
        ]
        mask = torch.ones(2, 3, dtype=torch.long)
        batch = {"input_ids": torch.zeros(2, 3, dtype=torch.long), "attention_mask": mask}

        with patch(
            "aspire.trainer.encode_exchanges", return_value=(torch.zeros(2, 3, dtype=torch.long), mask)
        ) as encode:
            trainer._compute_batch_loss(batch, dialogues)

        # the critic reads each prompt with the response the teacher scored, not the prompt alone
        assert encode.call_args.args[1:] == (["p0", "p1"], ["r0", "r1"])
        args, kwargs = trainer.geometry.record.call_args
        assert args[0] is hidden and args[1] is mask
        assert args[2] == evals
        assert kwargs == {"teacher_name": "Prof Plum"}
        # teacher scores are forwarded to the loss as float32
        loss_kwargs = trainer.loss_fn.call_args.kwargs
        assert loss_kwargs["teacher_score"].tolist() == [7.0, 3.0]
        assert loss_kwargs["teacher_score"].dtype == torch.float32

    def test_compute_batch_loss_teacher_without_name_falls_back(self, make_trainer):
        trainer, _ = make_trainer["build"](training__geometry_export=True)
        hidden = torch.randn(1, 2, 8)
        trainer.student_model = MagicMock(return_value=SimpleNamespace(hidden_states=[hidden]))
        trainer.critic = MagicMock(
            return_value=SimpleNamespace(score=torch.tensor([1.0]), reasoning_embedding=torch.zeros(1, 4))
        )
        trainer.loss_fn = MagicMock(return_value={"total": torch.tensor(1.0)})
        trainer.teacher = object()  # no .name attribute
        dialogues = [_dialogue(SimpleNamespace(overall_score=5.0))]
        batch = {
            "input_ids": torch.zeros(1, 2, dtype=torch.long),
            "attention_mask": torch.ones(1, 2, dtype=torch.long),
        }
        with _encoded(1, 2):
            trainer._compute_batch_loss(batch, dialogues)
        assert trainer.geometry.record.call_args.kwargs == {"teacher_name": "teacher"}

    def test_compute_batch_loss_without_geometry_does_not_record(self, make_trainer):
        trainer, _ = make_trainer["build"]()
        hidden = torch.randn(1, 2, 8)
        trainer.student_model = MagicMock(return_value=SimpleNamespace(hidden_states=[hidden]))
        trainer.critic = MagicMock(
            return_value=SimpleNamespace(score=torch.tensor([1.0]), reasoning_embedding=torch.zeros(1, 4))
        )
        trainer.loss_fn = MagicMock(return_value={"total": torch.tensor(1.0)})
        dialogues = [_dialogue(SimpleNamespace(overall_score=5.0))]
        batch = {
            "input_ids": torch.zeros(1, 2, dtype=torch.long),
            "attention_mask": torch.ones(1, 2, dtype=torch.long),
        }
        with _encoded(1, 2):
            out = trainer._compute_batch_loss(batch, dialogues)
        assert out["total"].item() == 1.0


class TestEightBitOptimizer:
    @pytest.mark.parametrize("name", ["adamw_8bit", "paged_adamw_8bit"])
    def test_uses_bitsandbytes_when_available(self, make_trainer, name):
        fake_bnb = MagicMock()
        with patch("aspire.trainer.bnb", fake_bnb):
            trainer, config = make_trainer["build"](
                training__optimizer=name, training__learning_rate=1e-3, training__weight_decay=0.2
            )
        _, kwargs = fake_bnb.optim.AdamW8bit.call_args
        assert kwargs == {"lr": 1e-3, "weight_decay": 0.2}
        assert trainer.student_optimizer is fake_bnb.optim.AdamW8bit.return_value

    @pytest.mark.parametrize("name", ["adamw_8bit", "paged_adamw_8bit"])
    def test_missing_bitsandbytes_raises_clear_import_error(self, make_trainer, name):
        with patch("aspire.trainer.bnb", None), pytest.raises(ImportError, match="bitsandbytes"):
            make_trainer["build"](training__optimizer=name)

    def test_adamw_does_not_touch_bitsandbytes(self, make_trainer):
        fake_bnb = MagicMock()
        with patch("aspire.trainer.bnb", fake_bnb):
            make_trainer["build"](training__optimizer="adamw")
        fake_bnb.optim.AdamW8bit.assert_not_called()


class TestUnknownOptimizer:
    def test_unknown_optimizer_name_fails_at_init(self, make_trainer):
        with pytest.raises((ValueError, KeyError)):
            make_trainer["build"](training__optimizer="sgd")


class TestGradientCheckpointing:
    def test_enabled_when_configured(self, make_trainer):
        make_trainer["build"](student__use_gradient_checkpointing=True)
        make_trainer["model"].gradient_checkpointing_enable.assert_called_once()

    def test_not_enabled_when_off(self, make_trainer):
        make_trainer["build"](student__use_gradient_checkpointing=False)
        make_trainer["model"].gradient_checkpointing_enable.assert_not_called()


class TestMain:
    def test_main_builds_trainer_and_trains_on_example_prompts(self):
        import aspire.trainer as trainer_module

        with patch.object(trainer_module, "freeze_support") as freeze, patch.object(
            trainer_module, "AspireConfig"
        ) as config_cls, patch.object(trainer_module, "AspireTrainer") as trainer_cls:
            trainer_module.main()
        freeze.assert_called_once_with()
        trainer_cls.assert_called_once_with(config_cls.return_value)
        (prompts,), _ = trainer_cls.return_value.train.call_args
        assert len(prompts) == 3 and all(isinstance(p, str) and p for p in prompts)

    @pytest.mark.filterwarnings("ignore:.*found in sys.modules.*:RuntimeWarning")
    def test_running_as_script_invokes_main(self):
        class Sentinel(Exception):
            pass

        # Re-executes the module as __main__; its main() builds AspireConfig() first.
        with patch("aspire.config.AspireConfig", side_effect=Sentinel), pytest.raises(Sentinel):
            runpy.run_module("aspire.trainer", run_name="__main__")
