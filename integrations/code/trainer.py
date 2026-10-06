"""
ASPIRE Code Trainer - Training loop for code models with internalized judgment.

The training loop:
1. Collect code samples (from repos or datasets)
2. Have code teachers critique each sample
3. Train the critic to predict teacher judgments
4. Train the student to generate better code guided by critic

After training, the model self-refines code using the internalized critic.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .config import CodeAspireConfig
from .code_teacher import CodeTeacher, CodeSample, CodeCritique
from .code_critic import CodeCritic, CodeCriticLoss, CriticOutput
from .data import CodeReviewDataset, CodeReviewPair, load_training_data


@dataclass
class TrainingMetrics:
    """Metrics from a training epoch."""

    epoch: int
    critic_loss: float
    student_loss: float
    mean_predicted_score: float
    mean_actual_score: float
    score_mae: float  # Mean absolute error
    time_elapsed: float


class AspireCodeTrainer:
    """
    ASPIRE training for code models.

    Trains a code model to develop internalized judgment about code quality.
    """

    def __init__(
        self,
        config: CodeAspireConfig | None = None,
        student_model: nn.Module | str | None = None,
        teacher: CodeTeacher | None = None,
        critic: CodeCritic | None = None,
    ):
        if config is None:
            config = CodeAspireConfig()

        self.config = config
        self.device = config.device

        # Initialize teacher
        self.teacher = teacher or CodeTeacher(
            personas=config.teacher.personas,
            strategy=config.teacher.strategy,
            use_llm=config.teacher.use_llm_teacher,
            llm_model=config.teacher.llm_model,
        )

        # Initialize critic
        self.critic = critic or CodeCritic(config.critic)
        self.critic = self.critic.to(self.device)

        # Initialize student model
        if student_model is None:
            student_model = config.training.student_model

        if isinstance(student_model, str):
            self.student, self.tokenizer = self._load_student_model(student_model)
        else:
            self.student = student_model
            self.tokenizer = None  # Must be provided separately

        if self.student is not None:
            self.student = self.student.to(self.device)

        # Optimizers
        self.critic_optimizer = optim.AdamW(
            self.critic.parameters(),
            lr=config.training.critic_lr,
        )

        if self.student is not None:
            self.student_optimizer = optim.AdamW(
                self._get_trainable_params(),
                lr=config.training.student_lr,
            )
        else:
            self.student_optimizer = None

        # Loss functions
        self.critic_loss_fn = CodeCriticLoss()

        # Metrics
        self.metrics_history: list[TrainingMetrics] = []

        # Logging
        self.use_wandb = config.training.use_wandb
        if self.use_wandb:
            try:
                import wandb
                wandb.init(project=config.training.wandb_project)
            except ImportError:
                self.use_wandb = False

    def _load_student_model(self, model_name: str):
        """Load a HuggingFace model as student."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import get_peft_model, LoraConfig, TaskType

            tokenizer = AutoTokenizer.from_pretrained(model_name)

            # Ensure pad token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto" if self.device == "cuda" else None,
            )

            # Apply LoRA if configured
            if self.config.training.use_lora:
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    r=self.config.training.lora_r,
                    lora_alpha=self.config.training.lora_alpha,
                    lora_dropout=self.config.training.lora_dropout,
                )
                model = get_peft_model(model, peft_config)
                model.print_trainable_parameters()

            return model, tokenizer

        except ImportError as e:
            print(f"Could not load model: {e}")
            print("Install with: pip install transformers peft")
            return None, None

    def _get_trainable_params(self):
        """Get trainable parameters (LoRA params if using LoRA)."""
        if hasattr(self.student, "parameters"):
            return [p for p in self.student.parameters() if p.requires_grad]
        return []

    def train_critic(
        self,
        train_pairs: list[CodeReviewPair],
        val_pairs: list[CodeReviewPair] | None = None,
        epochs: int | None = None,
    ) -> list[float]:
        """
        Train the critic to predict teacher judgments.

        Args:
            train_pairs: Training code-critique pairs
            val_pairs: Optional validation pairs
            epochs: Number of epochs

        Returns:
            List of training losses per epoch
        """
        epochs = epochs or self.config.training.epochs

        # Create dataset
        if self.tokenizer is None:
            # Use critic's tokenizer
            tokenizer = self.critic.get_tokenizer()
        else:
            tokenizer = self.tokenizer

        train_dataset = CodeReviewDataset(
            train_pairs,
            tokenizer,
            max_length=self.config.training.max_length,
            mode="critic",
        )

        # Create dataloader (Windows-compatible: num_workers=0)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=0,
        )

        losses = []

        print(f"Training critic for {epochs} epochs on {len(train_pairs)} samples")

        for epoch in range(epochs):
            epoch_start = time.time()
            epoch_loss = 0.0
            num_batches = 0

            self.critic.train()

            for batch in train_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                scores = batch["score"].to(self.device)

                # Forward pass
                self.critic_optimizer.zero_grad()
                output = self.critic(input_ids, attention_mask)

                # Compute loss
                loss_dict = self.critic_loss_fn(output, scores)
                loss = loss_dict["total"]

                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                self.critic_optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / max(num_batches, 1)
            losses.append(avg_loss)

            elapsed = time.time() - epoch_start
            print(f"Epoch {epoch + 1}/{epochs}: Loss = {avg_loss:.4f}, Time = {elapsed:.1f}s")

            # Validation
            if val_pairs:
                val_loss = self._validate_critic(val_pairs, tokenizer)
                print(f"  Validation Loss: {val_loss:.4f}")

            # Save checkpoint
            if (epoch + 1) % self.config.training.save_frequency == 0:
                self._save_checkpoint(epoch + 1, "critic")

        return losses

    def _validate_critic(
        self,
        val_pairs: list[CodeReviewPair],
        tokenizer,
    ) -> float:
        """Validate critic on held-out data."""
        val_dataset = CodeReviewDataset(
            val_pairs,
            tokenizer,
            max_length=self.config.training.max_length,
            mode="critic",
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=0,
        )

        self.critic.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                scores = batch["score"].to(self.device)

                output = self.critic(input_ids, attention_mask)
                loss_dict = self.critic_loss_fn(output, scores)

                total_loss += loss_dict["total"].item()
                num_batches += 1

        return total_loss / max(num_batches, 1)

    def train_student(
        self,
        train_pairs: list[CodeReviewPair],
        epochs: int | None = None,
    ) -> list[float]:
        """
        Train the student model guided by the critic.

        Args:
            train_pairs: Training code-critique pairs
            epochs: Number of epochs

        Returns:
            List of training losses per epoch
        """
        if self.student is None:
            raise ValueError("No student model loaded")

        epochs = epochs or self.config.training.epochs

        train_dataset = CodeReviewDataset(
            train_pairs,
            self.tokenizer,
            max_length=self.config.training.max_length,
            mode="student",
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=0,
        )

        losses = []
        accumulation_steps = self.config.training.gradient_accumulation_steps

        print(f"Training student for {epochs} epochs")

        for epoch in range(epochs):
            epoch_start = time.time()
            epoch_loss = 0.0
            num_batches = 0

            self.student.train()
            self.critic.eval()

            for i, batch in enumerate(train_loader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Forward pass through student
                outputs = self.student(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

                # Language modeling loss
                lm_loss = outputs.loss

                # Get critic score for generated output
                with torch.no_grad():
                    critic_output = self.critic(input_ids, attention_mask)
                    critic_score = critic_output.score

                # Combine losses
                # Higher critic score = lower loss (reward shaping)
                reward_bonus = (critic_score.mean() - 5.0) / 5.0  # Normalize
                loss = lm_loss - self.config.training.student_reward_weight * reward_bonus

                # Gradient accumulation
                loss = loss / accumulation_steps
                loss.backward()

                if (i + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self._get_trainable_params(), 1.0)
                    self.student_optimizer.step()
                    self.student_optimizer.zero_grad()

                epoch_loss += loss.item() * accumulation_steps
                num_batches += 1

            avg_loss = epoch_loss / max(num_batches, 1)
            losses.append(avg_loss)

            elapsed = time.time() - epoch_start
            print(f"Epoch {epoch + 1}/{epochs}: Loss = {avg_loss:.4f}, Time = {elapsed:.1f}s")

            if (epoch + 1) % self.config.training.save_frequency == 0:
                self._save_checkpoint(epoch + 1, "student")

        return losses

    def train(
        self,
        train_data: list[CodeReviewPair] | str,
        val_data: list[CodeReviewPair] | str | None = None,
        critic_epochs: int | None = None,
        student_epochs: int | None = None,
    ):
        """
        Full training pipeline: critic then student.

        Args:
            train_data: Training pairs or path to JSON file
            val_data: Validation pairs or path
            critic_epochs: Epochs for critic training
            student_epochs: Epochs for student training
        """
        # Load data if paths provided
        if isinstance(train_data, str):
            train_data = load_training_data(train_data)
        if isinstance(val_data, str):
            val_data = load_training_data(val_data)

        print("=" * 60)
        print("ASPIRE Code Training")
        print("=" * 60)
        print(f"Training samples: {len(train_data)}")
        print(f"Teacher: {self.teacher}")
        print(f"Critic: {self.config.critic.architecture.value}")
        print()

        # Phase 1: Train critic
        print("Phase 1: Training Critic")
        print("-" * 40)
        critic_losses = self.train_critic(
            train_data,
            val_data,
            epochs=critic_epochs or self.config.training.epochs,
        )

        # Phase 2: Train student (if available)
        if self.student is not None:
            print("\nPhase 2: Training Student")
            print("-" * 40)
            student_losses = self.train_student(
                train_data,
                epochs=student_epochs or self.config.training.epochs,
            )
        else:
            print("\nSkipping student training (no student model)")
            student_losses = []

        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)

        return {
            "critic_losses": critic_losses,
            "student_losses": student_losses,
        }

    def _save_checkpoint(self, epoch: int, component: str):
        """Save model checkpoint."""
        checkpoint_dir = Path(self.config.training.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if component == "critic":
            path = checkpoint_dir / f"critic_epoch_{epoch}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.critic.state_dict(),
                    "optimizer_state_dict": self.critic_optimizer.state_dict(),
                },
                path,
            )
        elif component == "student" and self.student is not None:
            path = checkpoint_dir / f"student_epoch_{epoch}"
            self.student.save_pretrained(str(path))
            if self.tokenizer:
                self.tokenizer.save_pretrained(str(path))

        print(f"  Saved {component} checkpoint: {path}")

    def load_checkpoint(self, critic_path: str | None = None, student_path: str | None = None):
        """Load model checkpoints."""
        if critic_path:
            checkpoint = torch.load(critic_path, map_location=self.device)
            self.critic.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded critic from {critic_path}")

        if student_path and self.student is not None:
            from peft import PeftModel
            self.student = PeftModel.from_pretrained(
                self.student.base_model.model,
                student_path,
            )
            print(f"Loaded student from {student_path}")

    def evaluate(
        self,
        test_pairs: list[CodeReviewPair],
    ) -> dict[str, float]:
        """
        Evaluate the trained models.

        Returns:
            Dictionary of evaluation metrics
        """
        if self.tokenizer is None:
            tokenizer = self.critic.get_tokenizer()
        else:
            tokenizer = self.tokenizer

        self.critic.eval()

        all_predicted = []
        all_actual = []

        for pair in test_pairs:
            tokens = tokenizer(
                pair.code,
                padding="max_length",
                truncation=True,
                max_length=self.config.training.max_length,
                return_tensors="pt",
            )

            with torch.no_grad():
                input_ids = tokens["input_ids"].to(self.device)
                attention_mask = tokens["attention_mask"].to(self.device)

                output = self.critic(input_ids, attention_mask)
                predicted = output.score.item()

            all_predicted.append(predicted)
            all_actual.append(pair.critique.overall_score)

        # Compute metrics
        import numpy as np

        predicted = np.array(all_predicted)
        actual = np.array(all_actual)

        mae = np.mean(np.abs(predicted - actual))
        mse = np.mean((predicted - actual) ** 2)
        correlation = np.corrcoef(predicted, actual)[0, 1]

        return {
            "mae": mae,
            "mse": mse,
            "rmse": np.sqrt(mse),
            "correlation": correlation,
            "mean_predicted": np.mean(predicted),
            "mean_actual": np.mean(actual),
        }

    def score_code(self, code: str) -> float:
        """
        Score a piece of code using the trained critic.

        Args:
            code: Code to evaluate

        Returns:
            Quality score (0-10)
        """
        return self.critic.score_code(code, device=self.device).item()

    def critique_code(self, code: str) -> CodeCritique:
        """
        Get full critique of code using both critic and teacher.

        Uses critic for score, teacher for detailed feedback.
        """
        from .analysis import detect_language

        language = detect_language(code)
        sample = CodeSample(code=code, language=language)

        # Get teacher critique
        critique = self.teacher.critique(sample)

        # Override score with critic's prediction (after training)
        critic_score = self.score_code(code)
        critique.overall_score = critic_score

        return critique
