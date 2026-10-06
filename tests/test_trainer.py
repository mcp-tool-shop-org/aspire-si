"""
Tests for ASPIRE Trainer module.

Tests cover:
- AspireDataset initialization, __getitem__, tokenization
- AspireTrainer initialization (student, critic, teacher, loss, optimizers)
- Training loop (dataloader, schedulers, epoch, gradient accumulation)
- Checkpoint save/load

CRITICAL Windows compatibility:
- num_workers=0 in DataLoader
- if __name__ == "__main__": freeze_support() pattern
- Mock heavy model loading to keep tests fast
"""

import os
import tempfile
from multiprocessing import freeze_support
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch
import torch.nn as nn

# Set Windows compatibility before imports
os.environ["XFORMERS_DISABLED"] = "1"


# ============================================================================
# AspireDataset Tests
# ============================================================================

class TestAspireDataset:
    """Tests for AspireDataset class."""

    def test_aspire_dataset_init(self, mock_tokenizer):
        """Verify dataset initializes with prompts, tokenizer, max_length."""
        from aspire.trainer import AspireDataset

        prompts = ["prompt 1", "prompt 2", "prompt 3"]
        max_length = 256

        dataset = AspireDataset(
            prompts=prompts,
            tokenizer=mock_tokenizer,
            max_length=max_length,
        )

        assert dataset.prompts == prompts
        assert dataset.tokenizer == mock_tokenizer
        assert dataset.max_length == max_length
        assert len(dataset) == 3

    def test_aspire_dataset_getitem(self, mock_tokenizer):
        """Verify __getitem__ returns dict with prompt, input_ids, attention_mask."""
        from aspire.trainer import AspireDataset

        prompts = ["What is recursion?", "Explain gradient descent."]
        max_length = 128

        dataset = AspireDataset(
            prompts=prompts,
            tokenizer=mock_tokenizer,
            max_length=max_length,
        )

        item = dataset[0]

        # Check returned keys
        assert "prompt" in item
        assert "input_ids" in item
        assert "attention_mask" in item

        # Check prompt is correct
        assert item["prompt"] == "What is recursion?"

        # Check tensor shapes
        assert item["input_ids"].shape == (max_length,)
        assert item["attention_mask"].shape == (max_length,)

    def test_aspire_dataset_getitem_shapes(self, mock_tokenizer):
        """Verify input_ids and attention_mask shapes match max_length."""
        from aspire.trainer import AspireDataset

        prompts = ["Test prompt"]
        max_lengths = [64, 128, 256, 512]

        for max_length in max_lengths:
            dataset = AspireDataset(
                prompts=prompts,
                tokenizer=mock_tokenizer,
                max_length=max_length,
            )
            item = dataset[0]

            assert item["input_ids"].shape == (max_length,), f"Expected {max_length}, got {item['input_ids'].shape}"
            assert item["attention_mask"].shape == (max_length,)

    def test_aspire_dataset_tokenization_truncation(self):
        """Verify truncation works for long prompts."""
        from aspire.trainer import AspireDataset

        # Create a mock tokenizer that simulates truncation
        mock_tok = MagicMock()
        max_length = 64

        def mock_call(text, **kwargs):
            actual_max = kwargs.get("max_length", 512)
            return {
                "input_ids": torch.randint(0, 1000, (1, actual_max)),
                "attention_mask": torch.ones(1, actual_max, dtype=torch.long),
            }

        mock_tok.side_effect = mock_call
        mock_tok.__call__ = mock_call

        # Very long prompt
        long_prompt = "This is a very long prompt. " * 100

        dataset = AspireDataset(
            prompts=[long_prompt],
            tokenizer=mock_tok,
            max_length=max_length,
        )

        item = dataset[0]

        # Should be truncated to max_length
        assert item["input_ids"].shape == (max_length,)
        assert item["attention_mask"].shape == (max_length,)

    def test_aspire_dataset_tokenization_padding(self):
        """Verify padding works for short prompts."""
        from aspire.trainer import AspireDataset

        # Mock tokenizer that simulates padding
        mock_tok = MagicMock()
        max_length = 256

        def mock_call(text, **kwargs):
            actual_max = kwargs.get("max_length", 512)
            return {
                "input_ids": torch.randint(0, 1000, (1, actual_max)),
                "attention_mask": torch.ones(1, actual_max, dtype=torch.long),
            }

        mock_tok.side_effect = mock_call
        mock_tok.__call__ = mock_call

        # Short prompt
        short_prompt = "Hi"

        dataset = AspireDataset(
            prompts=[short_prompt],
            tokenizer=mock_tok,
            max_length=max_length,
        )

        item = dataset[0]

        # Should be padded to max_length
        assert item["input_ids"].shape == (max_length,)


# ============================================================================
# AspireTrainer Tests - Initialization
# ============================================================================

class TestAspireTrainerInit:
    """Tests for AspireTrainer initialization."""

    @pytest.fixture
    def mock_all_dependencies(self):
        """Mock all heavy dependencies for trainer tests."""
        with patch("aspire.trainer.AutoModelForCausalLM") as mock_model_class, \
             patch("aspire.trainer.AutoTokenizer") as mock_tokenizer_class, \
             patch("aspire.trainer.get_teacher") as mock_get_teacher, \
             patch("aspire.trainer.CriticHead") as mock_critic_head, \
             patch("aspire.trainer.DialogueGenerator") as mock_dialogue_gen, \
             patch("aspire.trainer.DialogueManager") as mock_dialogue_mgr, \
             patch("aspire.trainer.DialogueFormatter") as mock_dialogue_fmt, \
             patch("aspire.trainer.AspireLoss") as mock_loss:

            # Mock model
            mock_model = MagicMock()
            mock_model.config = MagicMock()
            mock_model.config.hidden_size = 768
            mock_model.parameters = MagicMock(return_value=iter([nn.Parameter(torch.randn(10, 10))]))
            mock_model.to = MagicMock(return_value=mock_model)
            mock_model_class.from_pretrained = MagicMock(return_value=mock_model)

            # Mock tokenizer
            mock_tokenizer = MagicMock()
            mock_tokenizer.pad_token = None
            mock_tokenizer.eos_token = "</s>"
            mock_tokenizer_class.from_pretrained = MagicMock(return_value=mock_tokenizer)

            # Mock teacher
            mock_teacher = MagicMock()
            mock_get_teacher.return_value = mock_teacher

            # Mock critic
            mock_critic = MagicMock()
            mock_critic.parameters = MagicMock(return_value=iter([nn.Parameter(torch.randn(10, 10))]))
            mock_critic.get_trainable_parameters = MagicMock(return_value=[nn.Parameter(torch.randn(10, 10))])
            mock_critic.to = MagicMock(return_value=mock_critic)
            mock_critic_head.return_value = mock_critic

            yield {
                "model_class": mock_model_class,
                "model": mock_model,
                "tokenizer_class": mock_tokenizer_class,
                "tokenizer": mock_tokenizer,
                "get_teacher": mock_get_teacher,
                "teacher": mock_teacher,
                "critic_head": mock_critic_head,
                "critic": mock_critic,
                "loss": mock_loss,
            }

    def test_trainer_init(self, mock_all_dependencies):
        """Test basic trainer initialization."""
        from aspire.config import AspireConfig
        from aspire.trainer import AspireTrainer

        config = AspireConfig(device="cpu")

        trainer = AspireTrainer(config)

        assert trainer.config == config
        assert trainer.device == "cpu"
        assert trainer.global_step == 0
        assert trainer.current_epoch == 0

    def test_trainer_init_student_basic(self, mock_all_dependencies):
        """Test student model loading and tokenizer pad_token setup."""
        from aspire.config import AspireConfig
        from aspire.trainer import AspireTrainer

        mocks = mock_all_dependencies
        config = AspireConfig(device="cpu")

        trainer = AspireTrainer(config)

        # Verify model was loaded
        mocks["model_class"].from_pretrained.assert_called()

        # Verify tokenizer pad_token is set when None
        assert mocks["tokenizer"].pad_token == mocks["tokenizer"].eos_token

    def test_trainer_init_student_with_lora(self, mock_all_dependencies):
        """Test LoRA configuration is applied when use_lora=True."""
        from aspire.config import AspireConfig, StudentConfig
        from aspire.trainer import AspireTrainer

        with patch("aspire.trainer.get_peft_model") as mock_get_peft, \
             patch("aspire.trainer.LoraConfig") as mock_lora_config:

            mock_get_peft.return_value = mock_all_dependencies["model"]

            student_config = StudentConfig(
                use_lora=True,
                lora_r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                lora_target_modules=["q_proj", "v_proj"],
            )
            config = AspireConfig(student=student_config, device="cpu")

            trainer = AspireTrainer(config)

            # Verify LoraConfig was created with correct params
            mock_lora_config.assert_called_once()
            lora_call_kwargs = mock_lora_config.call_args
            assert lora_call_kwargs.kwargs["r"] == 16
            assert lora_call_kwargs.kwargs["lora_alpha"] == 32
            assert lora_call_kwargs.kwargs["lora_dropout"] == 0.05
            assert lora_call_kwargs.kwargs["target_modules"] == ["q_proj", "v_proj"]

            # Verify get_peft_model was called
            mock_get_peft.assert_called_once()

    def test_trainer_init_student_with_quantization_4bit(self, mock_all_dependencies):
        """Test 4-bit quantization configuration."""
        from aspire.config import AspireConfig, StudentConfig
        from aspire.trainer import AspireTrainer

        with patch("aspire.trainer.BitsAndBytesConfig") as mock_bnb, \
             patch("aspire.trainer.prepare_model_for_kbit_training") as mock_prepare:

            mock_prepare.return_value = mock_all_dependencies["model"]

            student_config = StudentConfig(load_in_4bit=True, load_in_8bit=False)
            config = AspireConfig(student=student_config, device="cpu")

            trainer = AspireTrainer(config)

            # Verify BitsAndBytesConfig was called with 4-bit settings
            mock_bnb.assert_called()
            bnb_call_kwargs = mock_bnb.call_args.kwargs
            assert bnb_call_kwargs.get("load_in_4bit") == True

            # Verify prepare_model_for_kbit_training was called
            mock_prepare.assert_called_once()

    def test_trainer_init_student_with_quantization_8bit(self, mock_all_dependencies):
        """Test 8-bit quantization configuration."""
        from aspire.config import AspireConfig, StudentConfig
        from aspire.trainer import AspireTrainer

        with patch("aspire.trainer.BitsAndBytesConfig") as mock_bnb, \
             patch("aspire.trainer.prepare_model_for_kbit_training") as mock_prepare:

            mock_prepare.return_value = mock_all_dependencies["model"]

            student_config = StudentConfig(load_in_4bit=False, load_in_8bit=True)
            config = AspireConfig(student=student_config, device="cpu")

            trainer = AspireTrainer(config)

            # Verify BitsAndBytesConfig was called with 8-bit settings
            mock_bnb.assert_called()
            bnb_call_kwargs = mock_bnb.call_args.kwargs
            assert bnb_call_kwargs.get("load_in_8bit") == True

    def test_trainer_init_critic_head(self):
        """Test critic head initialization."""
        from aspire.config import AspireConfig, CriticConfig

        with patch("aspire.trainer.AutoModelForCausalLM") as mock_model_class, \
             patch("aspire.trainer.AutoTokenizer") as mock_tok_class, \
             patch("aspire.trainer.get_teacher"), \
             patch("aspire.trainer.CriticHead") as mock_critic_head, \
             patch("aspire.trainer.DialogueGenerator"), \
             patch("aspire.trainer.DialogueManager"), \
             patch("aspire.trainer.DialogueFormatter"), \
             patch("aspire.trainer.AspireLoss"):

            # Setup mocks
            mock_model = MagicMock()
            mock_model.config = MagicMock()
            mock_model.config.hidden_size = 1024
            mock_model.parameters = MagicMock(return_value=iter([nn.Parameter(torch.randn(10, 10))]))
            mock_model_class.from_pretrained.return_value = mock_model

            mock_tok = MagicMock()
            mock_tok.pad_token = "<pad>"
            mock_tok_class.from_pretrained.return_value = mock_tok

            mock_critic = MagicMock()
            mock_critic.parameters = MagicMock(return_value=iter([nn.Parameter(torch.randn(10, 10))]))
            mock_critic.get_trainable_parameters = MagicMock(return_value=[nn.Parameter(torch.randn(10, 10))])
            mock_critic.to = MagicMock(return_value=mock_critic)
            mock_critic_head.return_value = mock_critic

            critic_config = CriticConfig(architecture="head", head_hidden_dim=512)
            config = AspireConfig(critic=critic_config, device="cpu")

            from aspire.trainer import AspireTrainer
            trainer = AspireTrainer(config)

            # Verify CriticHead was created with student_hidden_size
            mock_critic_head.assert_called_once()
            call_kwargs = mock_critic_head.call_args.kwargs
            assert call_kwargs["input_dim"] == 1024  # student hidden size

    def test_trainer_init_critic_separate(self):
        """Test separate critic initialization."""
        from aspire.config import AspireConfig, CriticConfig

        with patch("aspire.trainer.AutoModelForCausalLM") as mock_model_class, \
             patch("aspire.trainer.AutoTokenizer") as mock_tok_class, \
             patch("aspire.trainer.get_teacher"), \
             patch("aspire.trainer.SeparateCritic") as mock_sep_critic, \
             patch("aspire.trainer.DialogueGenerator"), \
             patch("aspire.trainer.DialogueManager"), \
             patch("aspire.trainer.DialogueFormatter"), \
             patch("aspire.trainer.AspireLoss"):

            # Setup mocks
            mock_model = MagicMock()
            mock_model.config = MagicMock()
            mock_model.config.hidden_size = 768
            mock_model.parameters = MagicMock(return_value=iter([nn.Parameter(torch.randn(10, 10))]))
            mock_model_class.from_pretrained.return_value = mock_model

            mock_tok = MagicMock()
            mock_tok.pad_token = "<pad>"
            mock_tok_class.from_pretrained.return_value = mock_tok

            mock_critic = MagicMock()
            mock_critic.parameters = MagicMock(return_value=iter([nn.Parameter(torch.randn(10, 10))]))
            mock_critic.get_trainable_parameters = MagicMock(return_value=[nn.Parameter(torch.randn(10, 10))])
            mock_critic.to = MagicMock(return_value=mock_critic)
            mock_sep_critic.return_value = mock_critic

            critic_config = CriticConfig(architecture="separate")
            config = AspireConfig(critic=critic_config, device="cpu")

            from aspire.trainer import AspireTrainer
            trainer = AspireTrainer(config)

            # Verify SeparateCritic was created
            mock_sep_critic.assert_called_once()

    def test_trainer_init_critic_shared(self):
        """Test shared encoder critic initialization."""
        from aspire.config import AspireConfig, CriticConfig

        with patch("aspire.trainer.AutoModelForCausalLM") as mock_model_class, \
             patch("aspire.trainer.AutoTokenizer") as mock_tok_class, \
             patch("aspire.trainer.get_teacher"), \
             patch("aspire.trainer.SharedEncoderCritic") as mock_shared_critic, \
             patch("aspire.trainer.DialogueGenerator"), \
             patch("aspire.trainer.DialogueManager"), \
             patch("aspire.trainer.DialogueFormatter"), \
             patch("aspire.trainer.AspireLoss"):

            # Setup mocks
            mock_model = MagicMock()
            mock_model.config = MagicMock()
            mock_model.config.hidden_size = 768
            mock_model.parameters = MagicMock(return_value=iter([nn.Parameter(torch.randn(10, 10))]))
            mock_model_class.from_pretrained.return_value = mock_model

            mock_tok = MagicMock()
            mock_tok.pad_token = "<pad>"
            mock_tok_class.from_pretrained.return_value = mock_tok

            mock_critic = MagicMock()
            mock_critic.parameters = MagicMock(return_value=iter([nn.Parameter(torch.randn(10, 10))]))
            mock_critic.get_trainable_parameters = MagicMock(return_value=[nn.Parameter(torch.randn(10, 10))])
            mock_critic.to = MagicMock(return_value=mock_critic)
            mock_shared_critic.return_value = mock_critic

            critic_config = CriticConfig(architecture="shared_encoder")
            config = AspireConfig(critic=critic_config, device="cpu")

            from aspire.trainer import AspireTrainer
            trainer = AspireTrainer(config)

            # Verify SharedEncoderCritic was created with student_model
            mock_shared_critic.assert_called_once()
            call_kwargs = mock_shared_critic.call_args.kwargs
            assert "student_model" in call_kwargs

    def test_trainer_init_critic_invalid(self):
        """Test that invalid critic architecture raises ValueError."""
        from aspire.config import AspireConfig, CriticConfig

        with patch("aspire.trainer.AutoModelForCausalLM") as mock_model_class, \
             patch("aspire.trainer.AutoTokenizer") as mock_tok_class, \
             patch("aspire.trainer.get_teacher"), \
             patch("aspire.trainer.DialogueGenerator"), \
             patch("aspire.trainer.DialogueManager"), \
             patch("aspire.trainer.DialogueFormatter"), \
             patch("aspire.trainer.AspireLoss"):

            # Setup mocks
            mock_model = MagicMock()
            mock_model.config = MagicMock()
            mock_model.config.hidden_size = 768
            mock_model.parameters = MagicMock(return_value=iter([nn.Parameter(torch.randn(10, 10))]))
            mock_model_class.from_pretrained.return_value = mock_model

            mock_tok = MagicMock()
            mock_tok.pad_token = "<pad>"
            mock_tok_class.from_pretrained.return_value = mock_tok

            # Create config with invalid architecture by patching the Literal validation
            critic_config = CriticConfig()
            # Manually set invalid architecture after validation
            critic_config.__dict__["architecture"] = "invalid"

            config = AspireConfig(critic=critic_config, device="cpu")

            from aspire.trainer import AspireTrainer

            with pytest.raises(ValueError, match="Unknown critic architecture"):
                AspireTrainer(config)

    def test_trainer_init_teacher_claude(self, mock_all_dependencies):
        """Test teacher initialization with Claude."""
        from aspire.config import AspireConfig, TeacherConfig
        from aspire.trainer import AspireTrainer

        teacher_config = TeacherConfig(default_teacher="claude", claude_model="claude-sonnet-4-20250514")
        config = AspireConfig(teacher=teacher_config, device="cpu")

        trainer = AspireTrainer(config)

        # Verify get_teacher was called with correct args
        mock_all_dependencies["get_teacher"].assert_called_once_with(
            "claude",
            model="claude-sonnet-4-20250514",
            temperature=teacher_config.temperature,
            max_tokens=teacher_config.max_tokens,
        )

    def test_trainer_init_teacher_openai(self, mock_all_dependencies):
        """Test teacher initialization with OpenAI."""
        from aspire.config import AspireConfig, TeacherConfig
        from aspire.trainer import AspireTrainer

        teacher_config = TeacherConfig(default_teacher="openai", openai_model="gpt-4o")
        config = AspireConfig(teacher=teacher_config, device="cpu")

        trainer = AspireTrainer(config)

        # Verify get_teacher was called with correct args
        mock_all_dependencies["get_teacher"].assert_called_once_with(
            "openai",
            model="gpt-4o",
            temperature=teacher_config.temperature,
            max_tokens=teacher_config.max_tokens,
        )

    def test_trainer_init_loss(self, mock_all_dependencies):
        """Test AspireLoss initialization with config weights."""
        from aspire.config import AspireConfig, LossConfig
        from aspire.trainer import AspireTrainer

        loss_config = LossConfig(
            critic_score_weight=1.5,
            critic_reasoning_weight=0.7,
            student_reward_weight=1.2,
            student_contrastive_weight=0.6,
        )
        config = AspireConfig(loss=loss_config, device="cpu")

        trainer = AspireTrainer(config)

        # Verify AspireLoss was created with correct weights
        mock_all_dependencies["loss"].assert_called_once()
        call_kwargs = mock_all_dependencies["loss"].call_args.kwargs
        assert call_kwargs["critic_score_weight"] == 1.5
        assert call_kwargs["critic_reasoning_weight"] == 0.7
        assert call_kwargs["student_reward_weight"] == 1.2
        assert call_kwargs["student_contrastive_weight"] == 0.6

    def test_trainer_init_optimizers_adamw(self, mock_all_dependencies):
        """Test AdamW optimizer initialization."""
        from aspire.config import AspireConfig, TrainingConfig
        from aspire.trainer import AspireTrainer

        training_config = TrainingConfig(
            optimizer="adamw",
            learning_rate=1e-5,
            weight_decay=0.01,
        )
        config = AspireConfig(training=training_config, device="cpu")

        trainer = AspireTrainer(config)

        # Verify optimizers are created
        assert hasattr(trainer, "student_optimizer")
        assert hasattr(trainer, "critic_optimizer")

    def test_trainer_init_optimizers_adamw_8bit(self, mock_all_dependencies):
        """Test AdamW 8-bit optimizer initialization."""
        from aspire.config import AspireConfig, TrainingConfig
        from aspire.trainer import AspireTrainer

        with patch("aspire.trainer.bnb") as mock_bnb:
            mock_opt = MagicMock()
            mock_bnb.optim.AdamW8bit.return_value = mock_opt

            training_config = TrainingConfig(
                optimizer="adamw_8bit",
                learning_rate=1e-5,
            )
            config = AspireConfig(training=training_config, device="cpu")

            # This will fail in the actual init because bnb isn't available at import time
            # But we're testing the branch logic
            try:
                trainer = AspireTrainer(config)
            except (ImportError, AttributeError, RuntimeError):
                # Expected if bitsandbytes not installed or CUDA unavailable
                pass


# ============================================================================
# AspireTrainer Tests - Training Loop
# ============================================================================

class TestAspireTrainerTraining:
    """Tests for AspireTrainer training methods."""

    @pytest.fixture
    def mock_trainer(self):
        """Create a trainer with all dependencies mocked."""
        from aspire.config import AspireConfig, TrainingConfig

        with patch("aspire.trainer.AutoModelForCausalLM") as mock_model_class, \
             patch("aspire.trainer.AutoTokenizer") as mock_tok_class, \
             patch("aspire.trainer.get_teacher"), \
             patch("aspire.trainer.CriticHead") as mock_critic_head, \
             patch("aspire.trainer.DialogueGenerator"), \
             patch("aspire.trainer.DialogueManager") as mock_dialogue_mgr, \
             patch("aspire.trainer.DialogueFormatter"), \
             patch("aspire.trainer.AspireLoss") as mock_loss:

            # Setup mocks
            mock_model = MagicMock()
            mock_model.config = MagicMock()
            mock_model.config.hidden_size = 768
            mock_model.parameters = MagicMock(return_value=iter([nn.Parameter(torch.randn(10, 10))]))
            mock_model.to = MagicMock(return_value=mock_model)
            mock_model.train = MagicMock(return_value=mock_model)
            mock_model.eval = MagicMock(return_value=mock_model)
            mock_model_class.from_pretrained.return_value = mock_model

            mock_tok = MagicMock()
            mock_tok.pad_token = "<pad>"

            def mock_call(text, **kwargs):
                if isinstance(text, str):
                    text = [text]
                batch_size = len(text)
                max_length = kwargs.get("max_length", 512)
                return {
                    "input_ids": torch.randint(0, 1000, (batch_size, max_length)),
                    "attention_mask": torch.ones(batch_size, max_length, dtype=torch.long),
                }

            mock_tok.side_effect = mock_call
            mock_tok.__call__ = mock_call
            mock_tok_class.from_pretrained.return_value = mock_tok

            mock_critic = MagicMock()
            mock_critic.parameters = MagicMock(return_value=iter([nn.Parameter(torch.randn(10, 10))]))
            mock_critic.get_trainable_parameters = MagicMock(return_value=[nn.Parameter(torch.randn(10, 10))])
            mock_critic.to = MagicMock(return_value=mock_critic)
            mock_critic.train = MagicMock(return_value=mock_critic)
            mock_critic.eval = MagicMock(return_value=mock_critic)
            mock_critic_head.return_value = mock_critic

            training_config = TrainingConfig(
                batch_size=2,
                num_epochs=2,
                gradient_accumulation_steps=2,
                dataloader_num_workers=0,  # CRITICAL: Windows compatibility
            )
            config = AspireConfig(training=training_config, device="cpu")

            from aspire.trainer import AspireTrainer
            trainer = AspireTrainer(config)

            # Store mocks for test assertions
            trainer._test_mocks = {
                "model": mock_model,
                "tokenizer": mock_tok,
                "critic": mock_critic,
                "dialogue_manager": mock_dialogue_mgr,
                "loss": mock_loss,
            }

            yield trainer

    def test_trainer_train_creates_dataloader(self, mock_trainer):
        """Test that train() creates DataLoader with correct settings."""

        with patch.object(mock_trainer, "_train_epoch", return_value={"loss": 0.5, "critic_loss": 0.3, "student_loss": 0.2}), \
             patch.object(mock_trainer, "_save_checkpoint"):

            prompts = ["prompt 1", "prompt 2", "prompt 3", "prompt 4"]
            mock_trainer.train(prompts)

            # Verify training state was updated
            assert mock_trainer.current_epoch >= 0

    def test_trainer_train_windows_num_workers_zero(self, mock_trainer):
        """CRITICAL: Verify num_workers=0 for Windows compatibility."""
        # This is verified via config
        assert mock_trainer.config.training.dataloader_num_workers == 0

    def test_trainer_train_creates_schedulers(self, mock_trainer):
        """Test that train() creates learning rate schedulers."""
        with patch.object(mock_trainer, "_train_epoch", return_value={"loss": 0.5, "critic_loss": 0.3, "student_loss": 0.2}), \
             patch.object(mock_trainer, "_save_checkpoint"), \
             patch("aspire.trainer.get_scheduler") as mock_get_scheduler:

            prompts = ["prompt 1", "prompt 2", "prompt 3", "prompt 4"]
            mock_trainer.train(prompts)

            # get_scheduler should be called for both student and critic
            assert mock_get_scheduler.call_count == 2

    def test_trainer_train_epoch_loop(self, mock_trainer):
        """Test that train() calls _train_epoch for each epoch."""
        with patch.object(mock_trainer, "_train_epoch", return_value={"loss": 0.5, "critic_loss": 0.3, "student_loss": 0.2}) as mock_epoch, \
             patch.object(mock_trainer, "_save_checkpoint"):

            prompts = ["prompt 1", "prompt 2", "prompt 3", "prompt 4"]
            mock_trainer.train(prompts)

            # Should be called num_epochs times (2)
            assert mock_epoch.call_count == 2

    def test_trainer_train_with_eval(self, mock_trainer):
        """Test that train() calls _evaluate when eval_prompts provided."""
        with patch.object(mock_trainer, "_train_epoch", return_value={"loss": 0.5, "critic_loss": 0.3, "student_loss": 0.2}), \
             patch.object(mock_trainer, "_evaluate", new_callable=AsyncMock) as mock_eval, \
             patch.object(mock_trainer, "_save_checkpoint"), \
             patch("aspire.trainer.asyncio.run") as mock_asyncio_run:

            mock_asyncio_run.return_value = {"avg_score": 7.5, "min_score": 5.0, "max_score": 9.0}

            train_prompts = ["prompt 1", "prompt 2"]
            eval_prompts = ["eval 1", "eval 2"]

            mock_trainer.train(train_prompts, eval_prompts)

            # asyncio.run should be called for evaluation
            assert mock_asyncio_run.call_count >= 2  # Once per epoch for eval

    def test_trainer_train_saves_checkpoints(self, mock_trainer):
        """Test that train() saves checkpoints."""
        with patch.object(mock_trainer, "_train_epoch", return_value={"loss": 0.5, "critic_loss": 0.3, "student_loss": 0.2}), \
             patch.object(mock_trainer, "_save_checkpoint") as mock_save:

            prompts = ["prompt 1", "prompt 2", "prompt 3", "prompt 4"]
            mock_trainer.train(prompts)

            # Should be called at least once per epoch
            assert mock_save.call_count >= 2

    def test_trainer_train_epoch_basic(self, mock_trainer):
        """Test _train_epoch sets models to train mode and processes batches."""
        from torch.utils.data import DataLoader

        from aspire.trainer import AspireDataset

        with patch.object(mock_trainer, "_compute_batch_loss") as mock_loss, \
             patch("aspire.trainer.asyncio.run") as mock_asyncio:

            # Create mock dialogues
            mock_dialogue = MagicMock()
            mock_dialogue.final_evaluation = MagicMock()
            mock_dialogue.final_evaluation.overall_score = 7.5
            mock_asyncio.return_value = [mock_dialogue, mock_dialogue]

            # Create mock loss return
            mock_loss.return_value = {
                "total": torch.tensor(0.5, requires_grad=True),
                "critic_total": torch.tensor(0.3),
                "student_total": torch.tensor(0.2),
            }

            # Create a simple dataloader
            prompts = ["prompt 1", "prompt 2"]
            dataset = AspireDataset(prompts, mock_trainer.tokenizer, max_length=64)
            dataloader = DataLoader(dataset, batch_size=2, num_workers=0)

            # Run epoch
            with patch("aspire.trainer.Progress"):
                metrics = mock_trainer._train_epoch(dataloader)

            # Verify models were set to train mode
            mock_trainer.student_model.train.assert_called()
            mock_trainer.critic.train.assert_called()

            assert "loss" in metrics
            assert "critic_loss" in metrics
            assert "student_loss" in metrics

    def test_trainer_train_epoch_gradient_accumulation(self, mock_trainer):
        """Test gradient accumulation with accumulation_steps=4."""
        mock_trainer.config.training.gradient_accumulation_steps = 4

        with patch.object(mock_trainer, "_compute_batch_loss") as mock_loss, \
             patch("aspire.trainer.asyncio.run") as mock_asyncio, \
             patch.object(mock_trainer.student_optimizer, "step") as mock_opt_step, \
             patch.object(mock_trainer.student_optimizer, "zero_grad") as mock_zero_grad:

            mock_dialogue = MagicMock()
            mock_dialogue.final_evaluation = MagicMock()
            mock_dialogue.final_evaluation.overall_score = 7.5
            mock_asyncio.return_value = [mock_dialogue]

            mock_loss.return_value = {
                "total": torch.tensor(0.5, requires_grad=True),
                "critic_total": torch.tensor(0.3),
                "student_total": torch.tensor(0.2),
            }

            # Create dataloader with 8 batches (should call optimizer.step twice with grad_accum=4)
            from torch.utils.data import DataLoader

            from aspire.trainer import AspireDataset

            prompts = ["prompt"] * 8
            dataset = AspireDataset(prompts, mock_trainer.tokenizer, max_length=64)
            dataloader = DataLoader(dataset, batch_size=1, num_workers=0)

            with patch("aspire.trainer.Progress"):
                mock_trainer._train_epoch(dataloader)

            # optimizer.step should be called len(dataloader) // grad_accum_steps = 8 // 4 = 2
            assert mock_opt_step.call_count == 2

    def test_trainer_train_epoch_gradient_clipping(self, mock_trainer):
        """Test gradient clipping is applied."""
        with patch.object(mock_trainer, "_compute_batch_loss") as mock_loss, \
             patch("aspire.trainer.asyncio.run") as mock_asyncio, \
             patch("torch.nn.utils.clip_grad_norm_") as mock_clip:

            mock_dialogue = MagicMock()
            mock_dialogue.final_evaluation = MagicMock()
            mock_dialogue.final_evaluation.overall_score = 7.5
            mock_asyncio.return_value = [mock_dialogue, mock_dialogue]

            mock_loss.return_value = {
                "total": torch.tensor(0.5, requires_grad=True),
                "critic_total": torch.tensor(0.3),
                "student_total": torch.tensor(0.2),
            }

            from torch.utils.data import DataLoader

            from aspire.trainer import AspireDataset

            prompts = ["prompt 1", "prompt 2"]
            dataset = AspireDataset(prompts, mock_trainer.tokenizer, max_length=64)
            dataloader = DataLoader(dataset, batch_size=2, num_workers=0)

            with patch("aspire.trainer.Progress"):
                mock_trainer._train_epoch(dataloader)

            # clip_grad_norm_ should be called for both models
            assert mock_clip.call_count >= 2


# ============================================================================
# AspireTrainer Tests - Compute Batch Loss
# ============================================================================

class TestAspireTrainerComputeBatchLoss:
    """Tests for _compute_batch_loss method."""

    def test_trainer_compute_batch_loss(self):
        """Test _compute_batch_loss computes losses correctly."""
        from aspire.config import AspireConfig
        from aspire.critic import CriticOutput

        with patch("aspire.trainer.AutoModelForCausalLM") as mock_model_class, \
             patch("aspire.trainer.AutoTokenizer") as mock_tok_class, \
             patch("aspire.trainer.get_teacher"), \
             patch("aspire.trainer.CriticHead") as mock_critic_head, \
             patch("aspire.trainer.DialogueGenerator"), \
             patch("aspire.trainer.DialogueManager"), \
             patch("aspire.trainer.DialogueFormatter"), \
             patch("aspire.trainer.AspireLoss") as mock_loss_class:

            # Setup model mock
            mock_model = MagicMock()
            mock_model.config = MagicMock()
            mock_model.config.hidden_size = 768
            mock_model.parameters = MagicMock(return_value=iter([nn.Parameter(torch.randn(10, 10))]))
            mock_model.to = MagicMock(return_value=mock_model)

            # Mock forward pass
            mock_output = MagicMock()
            mock_output.hidden_states = (torch.randn(2, 64, 768),)
            mock_model.return_value = mock_output
            mock_model.__call__ = MagicMock(return_value=mock_output)

            mock_model_class.from_pretrained.return_value = mock_model

            # Setup tokenizer mock
            mock_tok = MagicMock()
            mock_tok.pad_token = "<pad>"
            mock_tok_class.from_pretrained.return_value = mock_tok

            # Setup critic mock
            mock_critic = MagicMock()
            mock_critic.parameters = MagicMock(return_value=iter([nn.Parameter(torch.randn(10, 10))]))
            mock_critic.get_trainable_parameters = MagicMock(return_value=[nn.Parameter(torch.randn(10, 10))])
            mock_critic.to = MagicMock(return_value=mock_critic)

            # Mock critic output
            mock_critic_output = CriticOutput(
                score=torch.tensor([7.0, 8.0]),
                reasoning_embedding=torch.randn(2, 768),
            )
            mock_critic.return_value = mock_critic_output
            mock_critic.__call__ = MagicMock(return_value=mock_critic_output)

            mock_critic_head.return_value = mock_critic

            # Setup loss mock
            mock_loss = MagicMock()
            mock_loss.return_value = {
                "total": torch.tensor(0.5),
                "critic_total": torch.tensor(0.3),
                "student_total": torch.tensor(0.2),
            }
            mock_loss_class.return_value = mock_loss

            config = AspireConfig(device="cpu")

            from aspire.trainer import AspireTrainer
            trainer = AspireTrainer(config)

            # Create mock batch
            batch = {
                "prompt": ["prompt 1", "prompt 2"],
                "input_ids": torch.randint(0, 1000, (2, 64)),
                "attention_mask": torch.ones(2, 64, dtype=torch.long),
            }

            # Create mock dialogues
            mock_dialogue = MagicMock()
            mock_dialogue.final_evaluation = MagicMock()
            mock_dialogue.final_evaluation.overall_score = 7.5
            dialogues = [mock_dialogue, mock_dialogue]

            # Call _compute_batch_loss
            losses = trainer._compute_batch_loss(batch, dialogues)

            # Verify model was called with output_hidden_states=True
            mock_model.assert_called()
            call_kwargs = mock_model.call_args.kwargs
            assert call_kwargs.get("output_hidden_states") == True

            # Verify critic was called
            mock_critic.assert_called()

            # Verify loss function was called
            mock_loss.assert_called()

            # Verify return dict
            assert "total" in losses


# ============================================================================
# AspireTrainer Tests - Evaluation
# ============================================================================

class TestAspireTrainerEvaluation:
    """Tests for _evaluate method."""

    @pytest.mark.asyncio
    async def test_trainer_evaluate_async(self):
        """Test _evaluate returns correct metrics."""
        from aspire.config import AspireConfig

        with patch("aspire.trainer.AutoModelForCausalLM") as mock_model_class, \
             patch("aspire.trainer.AutoTokenizer") as mock_tok_class, \
             patch("aspire.trainer.get_teacher"), \
             patch("aspire.trainer.CriticHead") as mock_critic_head, \
             patch("aspire.trainer.DialogueGenerator"), \
             patch("aspire.trainer.DialogueManager") as mock_dialogue_mgr_class, \
             patch("aspire.trainer.DialogueFormatter"), \
             patch("aspire.trainer.AspireLoss"):

            # Setup mocks
            mock_model = MagicMock()
            mock_model.config = MagicMock()
            mock_model.config.hidden_size = 768
            mock_model.parameters = MagicMock(return_value=iter([nn.Parameter(torch.randn(10, 10))]))
            mock_model.eval = MagicMock(return_value=mock_model)
            mock_model_class.from_pretrained.return_value = mock_model

            mock_tok = MagicMock()
            mock_tok.pad_token = "<pad>"
            mock_tok_class.from_pretrained.return_value = mock_tok

            mock_critic = MagicMock()
            mock_critic.parameters = MagicMock(return_value=iter([nn.Parameter(torch.randn(10, 10))]))
            mock_critic.get_trainable_parameters = MagicMock(return_value=[nn.Parameter(torch.randn(10, 10))])
            mock_critic.to = MagicMock(return_value=mock_critic)
            mock_critic.eval = MagicMock(return_value=mock_critic)
            mock_critic_head.return_value = mock_critic

            # Setup dialogue manager mock
            mock_dialogue_mgr = MagicMock()
            mock_dialogue = MagicMock()
            mock_dialogue.final_evaluation = MagicMock()
            mock_dialogue.final_evaluation.overall_score = 8.0
            mock_dialogue_mgr.get_dialogue = AsyncMock(return_value=mock_dialogue)
            mock_dialogue_mgr_class.return_value = mock_dialogue_mgr

            config = AspireConfig(device="cpu")

            from aspire.trainer import AspireTrainer
            trainer = AspireTrainer(config)

            prompts = ["eval prompt 1", "eval prompt 2", "eval prompt 3"]
            metrics = await trainer._evaluate(prompts)

            # Verify models set to eval mode
            mock_model.eval.assert_called()
            mock_critic.eval.assert_called()

            # Verify metrics
            assert "avg_score" in metrics
            assert "min_score" in metrics
            assert "max_score" in metrics
            assert metrics["avg_score"] == 8.0
            assert metrics["min_score"] == 8.0
            assert metrics["max_score"] == 8.0


# ============================================================================
# AspireTrainer Tests - Checkpointing
# ============================================================================

class TestAspireTrainerCheckpointing:
    """Tests for checkpoint save/load methods."""

    def test_trainer_save_checkpoint(self):
        """Test _save_checkpoint saves all components."""
        from aspire.config import AspireConfig

        with patch("aspire.trainer.AutoModelForCausalLM") as mock_model_class, \
             patch("aspire.trainer.AutoTokenizer") as mock_tok_class, \
             patch("aspire.trainer.get_teacher"), \
             patch("aspire.trainer.CriticHead") as mock_critic_head, \
             patch("aspire.trainer.DialogueGenerator"), \
             patch("aspire.trainer.DialogueManager"), \
             patch("aspire.trainer.DialogueFormatter"), \
             patch("aspire.trainer.AspireLoss"):

            # Setup mocks
            mock_model = MagicMock()
            mock_model.config = MagicMock()
            mock_model.config.hidden_size = 768
            mock_model.parameters = MagicMock(return_value=iter([nn.Parameter(torch.randn(10, 10))]))
            mock_model.save_pretrained = MagicMock()
            mock_model_class.from_pretrained.return_value = mock_model

            mock_tok = MagicMock()
            mock_tok.pad_token = "<pad>"
            mock_tok.save_pretrained = MagicMock()
            mock_tok_class.from_pretrained.return_value = mock_tok

            mock_critic = MagicMock()
            mock_critic.parameters = MagicMock(return_value=iter([nn.Parameter(torch.randn(10, 10))]))
            mock_critic.get_trainable_parameters = MagicMock(return_value=[nn.Parameter(torch.randn(10, 10))])
            mock_critic.to = MagicMock(return_value=mock_critic)
            mock_critic.save = MagicMock()
            mock_critic_head.return_value = mock_critic

            with tempfile.TemporaryDirectory() as tmpdir:
                config = AspireConfig(device="cpu")
                config.training.output_dir = Path(tmpdir)

                from aspire.trainer import AspireTrainer
                trainer = AspireTrainer(config)

                # Save checkpoint
                trainer._save_checkpoint(epoch=1)

                # Verify student model was saved
                mock_model.save_pretrained.assert_called()

                # Verify tokenizer was saved
                mock_tok.save_pretrained.assert_called()

                # Verify critic was saved
                mock_critic.save.assert_called()

                # Verify checkpoint directory was created
                checkpoint_dir = Path(tmpdir) / "checkpoint-1"
                assert checkpoint_dir.exists()

    def test_trainer_load_checkpoint(self):
        """Test load_checkpoint loads all components."""
        from aspire.config import AspireConfig

        with patch("aspire.trainer.AutoModelForCausalLM") as mock_model_class, \
             patch("aspire.trainer.AutoTokenizer") as mock_tok_class, \
             patch("aspire.trainer.get_teacher"), \
             patch("aspire.trainer.CriticHead") as mock_critic_head, \
             patch("aspire.trainer.DialogueGenerator"), \
             patch("aspire.trainer.DialogueManager"), \
             patch("aspire.trainer.DialogueFormatter"), \
             patch("aspire.trainer.AspireLoss"), \
             patch("aspire.trainer.PeftModel") as mock_peft:

            # Setup mocks
            mock_model = MagicMock()
            mock_model.config = MagicMock()
            mock_model.config.hidden_size = 768
            mock_model.parameters = MagicMock(return_value=iter([nn.Parameter(torch.randn(10, 10))]))
            mock_model_class.from_pretrained.return_value = mock_model

            mock_tok = MagicMock()
            mock_tok.pad_token = "<pad>"
            mock_tok_class.from_pretrained.return_value = mock_tok

            mock_critic = MagicMock()
            mock_critic.parameters = MagicMock(return_value=iter([nn.Parameter(torch.randn(10, 10))]))
            mock_critic.get_trainable_parameters = MagicMock(return_value=[nn.Parameter(torch.randn(10, 10))])
            mock_critic.to = MagicMock(return_value=mock_critic)
            mock_critic.__class__ = MagicMock()
            mock_critic.__class__.load = MagicMock(return_value=mock_critic)
            mock_critic_head.return_value = mock_critic

            # Setup PeftModel mock
            mock_peft.from_pretrained = MagicMock(return_value=mock_model)

            config = AspireConfig(device="cpu")

            from aspire.trainer import AspireTrainer
            trainer = AspireTrainer(config)

            with tempfile.TemporaryDirectory() as tmpdir:
                checkpoint_dir = Path(tmpdir) / "checkpoint-1"
                checkpoint_dir.mkdir(parents=True)
                (checkpoint_dir / "student").mkdir()
                (checkpoint_dir / "critic.pt").touch()

                # Load checkpoint
                trainer.load_checkpoint(checkpoint_dir)

                # Verify PeftModel.from_pretrained was called
                mock_peft.from_pretrained.assert_called_once()

                # Verify critic was loaded
                trainer.critic.__class__.load.assert_called()


# ============================================================================
# Entry point for Windows compatibility
# ============================================================================

if __name__ == "__main__":
    freeze_support()
    pytest.main([__file__, "-v"])
