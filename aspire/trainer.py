"""
ASPIRE Trainer - the core training loop.

Orchestrates the training of student and critic models through
adversarial dialogue with teacher models.
"""

import asyncio
import os
from multiprocessing import freeze_support
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_scheduler,
)

try:  # 8-bit optimizers need bitsandbytes, which not every platform can load.
    import bitsandbytes as bnb
except ImportError:  # pragma: no cover - depends on the platform
    bnb = None

from aspire.config import AspireConfig, TeacherConfig
from aspire.critic import CriticHead, SeparateCritic, SharedEncoderCritic
from aspire.dialogue import DialogueFormatter, DialogueGenerator, DialogueManager
from aspire.errors import ConfigError
from aspire.geometry import GeometryRecorder
from aspire.judge import encode_exchanges
from aspire.losses import AspireLoss
from aspire.teachers import BaseTeacher, CompositeTeacher, get_teacher

# Windows compatibility
os.environ["XFORMERS_DISABLED"] = "1"

console = Console()

# Teachers that call the OpenAI API; every other built-in API teacher calls Claude.
OPENAI_TEACHERS = ("openai", "gpt4")


def _local_name(model: str) -> str:
    return f"local:{model.replace(chr(92), '/').rstrip('/').rsplit('/', 1)[-1]}"


def build_teacher(cfg: TeacherConfig, spec: str | None = None, device: str | None = None) -> BaseTeacher:
    """The teacher a config names.

    ``spec`` defaults to ``cfg.default_teacher``. It is a registered name, ``local`` (the model
    at ``cfg.local_model_path``), ``local:<model>``, or ``composite`` (``cfg.composite_members``,
    each of which is a spec of its own).
    """
    spec = (spec or cfg.default_teacher).strip()
    common = {"temperature": cfg.temperature, "max_tokens": cfg.max_tokens}

    if spec == "composite":
        if not cfg.composite_members:
            raise ConfigError(
                "The composite teacher has no members.",
                hint="Set teacher.composite_members, "
                'for example ["local:Qwen/Qwen2.5-3B-Instruct", "claude"].',
            )
        members = [build_teacher(cfg, member, device) for member in cfg.composite_members]
        names = [member.name for member in members]
        if len(set(names)) != len(names):
            raise ConfigError(
                f"Two composite members have the same name ({', '.join(names)}).",
                hint="Each member's scores are kept under its name, so the members must differ.",
            )
        return CompositeTeacher(teachers=members, strategy=cfg.composite_strategy, **common)

    if spec == "local" or spec.startswith("local:"):
        model = spec[len("local:") :].strip() if spec.startswith("local:") else cfg.local_model_path
        if not model:
            raise ConfigError(
                "The local teacher has no model.",
                hint="Pass --teacher-model, or set teacher.local_model_path (a Hugging Face name or a path).",
            )
        return get_teacher(
            "local",
            model_name_or_path=model,
            load_in_4bit=cfg.local_load_in_4bit,
            device=device,
            name=_local_name(model),
            **common,
        )

    model = cfg.openai_model if spec.lower() in OPENAI_TEACHERS else cfg.claude_model
    return get_teacher(spec, model=model, **common)


def load_peft_weights_into(model: PeftModel, adapter_dir: Path) -> None:
    """Load a saved LoRA adapter's weights into a model that already has the adapter."""
    from peft import load_peft_weights, set_peft_model_state_dict

    result = set_peft_model_state_dict(model, load_peft_weights(str(adapter_dir)))
    missing = [k for k in getattr(result, "missing_keys", []) or [] if "lora_" in k]
    if missing:
        raise ConfigError(
            f"{adapter_dir} does not match the student's LoRA adapter ({len(missing)} weights missing).",
            hint="Evaluate with the config the checkpoint was trained with (its config.yaml).",
        )


def describe_teacher(teacher: BaseTeacher) -> str:
    """A run's teacher in a few words, for the geometry export's condition."""
    if isinstance(teacher, CompositeTeacher):
        members = " + ".join(member.name for member in teacher.teachers)
        return f"composite ({teacher.strategy}): {members}"
    return str(getattr(teacher, "name", "teacher"))


class AspireDataset(Dataset):
    """Dataset for ASPIRE training."""

    def __init__(
        self,
        prompts: list[str],
        tokenizer: AutoTokenizer,
        max_length: int = 512,
    ):
        self.prompts = prompts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        prompt = self.prompts[idx]

        # Tokenize
        encoded = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "prompt": prompt,
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }


class AspireTrainer:
    """
    Main trainer for ASPIRE.

    Coordinates:
    1. Student model training
    2. Critic model training
    3. Dialogue generation with teachers
    4. Loss computation and backpropagation
    """

    def __init__(self, config: AspireConfig):
        self.config = config
        self.device = config.device

        # Set seed
        torch.manual_seed(config.seed)

        # Initialize components
        self._init_student()
        self._init_critic()
        self._init_teacher()
        self._init_loss()
        self._init_optimizers()

        # Dialogue components
        self.dialogue_generator = DialogueGenerator(
            student_model=self.student_model,
            student_tokenizer=self.tokenizer,
            teacher=self.teacher,
            max_turns=config.teacher.max_dialogue_turns,
            evaluate_each_turn=config.teacher.evaluate_each_turn,
            student_max_length=config.student.max_length,
            device=self.device,
        )
        self.dialogue_manager = DialogueManager(
            generator=self.dialogue_generator,
            cache_dir=config.training.output_dir / "dialogue_cache",
        )
        self.dialogue_formatter = DialogueFormatter(format_type="chat")

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.geometry: GeometryRecorder | None = None
        if config.training.geometry_export:
            self.geometry = GeometryRecorder(
                run_id=config.experiment_name,
                condition=describe_teacher(self.teacher),
                seed=config.seed,
                window=config.training.geometry_window,
                every=config.training.geometry_every,
                # Dialogues are cached, so epochs after the first replay the first epoch's scores.
                scalar_source="replayed" if config.training.num_epochs > 1 else "live",
            )

        console.print("[green]ASPIRE Trainer initialized[/green]")

    def _init_student(self) -> None:
        """Initialize student model with optional LoRA."""
        cfg = self.config.student

        console.print(f"Loading student model: {cfg.model_name_or_path}")

        # Quantization config
        quantization_config = None
        if cfg.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif cfg.load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        # Load model
        self.student_model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name_or_path,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Prepare for training
        if cfg.load_in_4bit or cfg.load_in_8bit:
            self.student_model = prepare_model_for_kbit_training(self.student_model)

        # Apply LoRA if configured
        if cfg.use_lora:
            lora_config = LoraConfig(
                r=cfg.lora_r,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                target_modules=cfg.lora_target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.student_model = get_peft_model(self.student_model, lora_config)
            console.print("[blue]LoRA applied to student model[/blue]")

        # Gradient checkpointing
        if cfg.use_gradient_checkpointing:
            self.student_model.gradient_checkpointing_enable()

        # Get hidden size for critic
        self.student_hidden_size = self.student_model.config.hidden_size

    def _init_critic(self) -> None:
        """Initialize critic model."""
        cfg = self.config.critic

        console.print(f"Initializing critic: {cfg.architecture}")

        if cfg.init_seed is not None:
            # A separate stream for the critic's weights. The critic is built on the CPU, so only
            # the CPU generator is reseeded (torch.manual_seed would also reseed the GPU's, which
            # the run's sampling draws from), and fork_rng puts it back afterwards.
            with torch.random.fork_rng(devices=[]):
                torch.default_generator.manual_seed(cfg.init_seed)
                self._build_critic(cfg)
        else:
            self._build_critic(cfg)
        self.critic = self.critic.to(self.device)

    def _build_critic(self, cfg: Any) -> None:
        if cfg.architecture == "head":
            self.critic = CriticHead(
                input_dim=self.student_hidden_size,
                hidden_dim=cfg.head_hidden_dim,
                num_layers=cfg.head_num_layers,
                reasoning_dim=cfg.reasoning_embedding_dim,
            )
        elif cfg.architecture == "separate":
            self.critic = SeparateCritic(
                model_name_or_path=cfg.separate_model_name,
                hidden_dim=cfg.head_hidden_dim,
                reasoning_dim=cfg.reasoning_embedding_dim,
                load_in_4bit=cfg.separate_load_in_4bit,
            )
        elif cfg.architecture == "shared_encoder":
            self.critic = SharedEncoderCritic(
                student_model=self.student_model,
                hidden_dim=cfg.head_hidden_dim,
                reasoning_dim=cfg.reasoning_embedding_dim,
            )
        else:
            raise ValueError(f"Unknown critic architecture: {cfg.architecture}")

    def _init_teacher(self) -> None:
        """Initialize teacher model(s)."""
        cfg = self.config.teacher

        console.print(f"Initializing teacher: {cfg.default_teacher}")

        self.teacher = build_teacher(cfg, device=self.device)

    def _init_loss(self) -> None:
        """Initialize loss functions."""
        cfg = self.config.loss

        self.loss_fn = AspireLoss(
            critic_score_weight=cfg.critic_score_weight,
            critic_reasoning_weight=cfg.critic_reasoning_weight,
            student_reward_weight=cfg.student_reward_weight,
            student_contrastive_weight=cfg.student_contrastive_weight,
            student_trajectory_weight=cfg.student_trajectory_weight,
            student_coherence_weight=cfg.student_coherence_weight,
            contrastive_margin=cfg.contrastive_margin,
            contrastive_temperature=cfg.contrastive_temperature,
        )

    def _init_optimizers(self) -> None:
        """Initialize optimizers and schedulers."""
        cfg = self.config.training

        # Student optimizer
        student_params = [p for p in self.student_model.parameters() if p.requires_grad]

        if cfg.optimizer == "adamw":
            self.student_optimizer = AdamW(
                student_params,
                lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
            )
        elif cfg.optimizer in ["adamw_8bit", "paged_adamw_8bit"]:
            if bnb is None:
                raise ImportError("The 8-bit optimizer needs bitsandbytes, which is not installed.")

            self.student_optimizer = bnb.optim.AdamW8bit(
                student_params,
                lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {cfg.optimizer!r}")

        # Critic optimizer
        critic_params = self.critic.get_trainable_parameters()
        self.critic_optimizer = AdamW(
            critic_params,
            lr=cfg.critic_learning_rate,
            weight_decay=cfg.weight_decay,
        )

    def train(
        self,
        train_prompts: list[str],
        eval_prompts: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Main training loop.

        Args:
            train_prompts: List of training prompts
            eval_prompts: Optional list of evaluation prompts

        Returns:
            Training metrics
        """
        cfg = self.config.training

        # Create dataset and dataloader
        train_dataset = AspireDataset(
            prompts=train_prompts,
            tokenizer=self.tokenizer,
            max_length=self.config.student.max_length,
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.dataloader_num_workers,  # 0 for Windows
        )

        # Calculate total steps
        num_update_steps = len(train_dataloader) // cfg.gradient_accumulation_steps
        total_steps = num_update_steps * cfg.num_epochs

        # Learning rate schedulers
        self.student_scheduler = get_scheduler(
            cfg.lr_scheduler,
            optimizer=self.student_optimizer,
            num_warmup_steps=int(total_steps * cfg.warmup_ratio),
            num_training_steps=total_steps,
        )

        self.critic_scheduler = get_scheduler(
            cfg.lr_scheduler,
            optimizer=self.critic_optimizer,
            num_warmup_steps=int(total_steps * cfg.warmup_ratio),
            num_training_steps=total_steps,
        )

        # Training loop
        console.print("\n[bold green]Starting ASPIRE training[/bold green]")
        console.print(f"  Epochs: {cfg.num_epochs}")
        console.print(f"  Train samples: {len(train_prompts)}")
        console.print(f"  Batch size: {cfg.batch_size}")
        console.print(f"  Total steps: {total_steps}")

        metrics: dict[str, Any] = {"train_loss": [], "critic_loss": [], "student_loss": []}

        for epoch in range(cfg.num_epochs):
            self.current_epoch = epoch
            epoch_metrics = self._train_epoch(train_dataloader)
            metrics["train_loss"].append(epoch_metrics["loss"])
            metrics["critic_loss"].append(epoch_metrics["critic_loss"])
            metrics["student_loss"].append(epoch_metrics["student_loss"])

            console.print(
                f"Epoch {epoch + 1}/{cfg.num_epochs} - "
                f"Loss: {epoch_metrics['loss']:.4f} - "
                f"Critic: {epoch_metrics['critic_loss']:.4f} - "
                f"Student: {epoch_metrics['student_loss']:.4f}"
            )

            # Evaluation
            if eval_prompts:
                eval_metrics = asyncio.run(self._evaluate(eval_prompts))
                console.print(f"  Eval score: {eval_metrics['avg_score']:.2f}")

            # Save checkpoint
            if (epoch + 1) % 1 == 0:  # Save every epoch
                self._save_checkpoint(epoch + 1)

            # Write the export so far after each epoch, so a run stopped before the end
            # (a deadline, Ctrl+C, a crash) keeps what it recorded. The last write is below.
            if self.geometry is not None and epoch + 1 < cfg.num_epochs:
                self._write_geometry(len(train_prompts), cycles=epoch + 1, quiet=True)

        if self.geometry is not None:
            metrics["geometry_export"] = self._write_geometry(len(train_prompts))

        return metrics

    def _write_geometry(
        self, training_items: int, cycles: int | None = None, quiet: bool = False
    ) -> str | None:
        """Write the ScalarScope export, or say why there is none.

        ``cycles`` is the number of epochs it covers (all of them by default); ``quiet`` skips
        the messages, for the write after each epoch.
        """
        assert self.geometry is not None
        try:
            path = self.geometry.write(
                Path(self.config.training.output_dir) / "geometry.json",
                training_items=training_items,
                cycles=self.config.training.num_epochs if cycles is None else cycles,
            )
        except ValueError as error:
            if not quiet:
                console.print(f"[yellow]No geometry export: {error}[/yellow]")
            return None
        if not quiet:
            console.print(f"  Geometry export: {path}")
        return str(path)

    def _train_epoch(self, dataloader: DataLoader) -> dict[str, float]:
        """Train for one epoch."""
        self.student_model.train()
        self.critic.train()

        total_loss = 0.0
        total_critic_loss = 0.0
        total_student_loss = 0.0
        num_batches = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        ) as progress:
            task = progress.add_task(f"Epoch {self.current_epoch + 1}", total=len(dataloader))

            for batch_idx, batch in enumerate(dataloader):
                # Generate dialogue for batch (async)
                prompts = batch["prompt"]
                dialogues = asyncio.run(
                    self.dialogue_manager.get_dialogues(prompts, max_concurrent=3)
                )

                # Compute losses
                losses = self._compute_batch_loss(batch, dialogues)

                # Backward pass
                loss = losses["total"]
                loss = loss / self.config.training.gradient_accumulation_steps
                loss.backward()

                # Gradient accumulation
                if (batch_idx + 1) % self.config.training.gradient_accumulation_steps == 0:
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(
                        self.student_model.parameters(),
                        self.config.training.max_grad_norm,
                    )
                    torch.nn.utils.clip_grad_norm_(
                        self.critic.parameters(),
                        self.config.training.max_grad_norm,
                    )

                    # Optimizer step
                    self.student_optimizer.step()
                    self.critic_optimizer.step()
                    self.student_scheduler.step()
                    self.critic_scheduler.step()

                    self.student_optimizer.zero_grad()
                    self.critic_optimizer.zero_grad()

                    self.global_step += 1

                # Track metrics
                total_loss += losses["total"].item()
                total_critic_loss += losses.get("critic_total", torch.tensor(0.0)).item()
                total_student_loss += losses.get("student_total", torch.tensor(0.0)).item()
                num_batches += 1

                progress.update(task, advance=1)

        return {
            "loss": total_loss / num_batches,
            "critic_loss": total_critic_loss / num_batches,
            "student_loss": total_student_loss / num_batches,
        }

    def _compute_batch_loss(
        self,
        batch: dict[str, torch.Tensor],
        dialogues: list,
    ) -> dict[str, torch.Tensor]:
        """Compute loss for a batch.

        The critic reads the student's hidden states over each prompt and the response the
        teacher scored, so that it learns to judge responses (not prompts).
        """
        input_ids, attention_mask = encode_exchanges(
            self.tokenizer,
            [d.prompt for d in dialogues],
            [d.scored_response for d in dialogues],
            max_length=self.config.student.max_length,
        )
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        # Get student outputs
        student_outputs = self.student_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # The critic is float32; a bf16 or quantized student hands it bf16 states.
        student_hidden = student_outputs.hidden_states[-1].float()

        if self.geometry is not None:
            self.geometry.record(
                student_hidden,
                attention_mask,
                [d.final_evaluation for d in dialogues],
                teacher_name=getattr(self.teacher, "name", "teacher"),
            )

        # Get critic predictions
        critic_output = self.critic(hidden_states=student_hidden, attention_mask=attention_mask)

        # Get teacher scores from dialogues
        teacher_scores = torch.tensor(
            [d.final_evaluation.overall_score for d in dialogues],
            device=self.device,
            dtype=torch.float32,
        )

        # Compute losses
        losses = self.loss_fn(
            critic_predicted_score=critic_output.score,
            teacher_score=teacher_scores,
            critic_predicted_embedding=critic_output.reasoning_embedding,
        )

        return losses

    async def _evaluate(self, prompts: list[str]) -> dict[str, float]:
        """Evaluate on a set of prompts."""
        self.student_model.eval()
        self.critic.eval()

        scores = []

        for prompt in prompts:
            dialogue = await self.dialogue_manager.get_dialogue(prompt)
            scores.append(dialogue.final_evaluation.overall_score)

        return {
            "avg_score": sum(scores) / len(scores),
            "min_score": min(scores),
            "max_score": max(scores),
        }

    def _save_checkpoint(self, epoch: int) -> None:
        """Save training checkpoint."""
        output_dir = self.config.training.output_dir / f"checkpoint-{epoch}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save student model
        self.student_model.save_pretrained(output_dir / "student")
        self.tokenizer.save_pretrained(output_dir / "student")

        # Save critic
        self.critic.save(str(output_dir / "critic.pt"))

        # Save config
        self.config.to_yaml(output_dir / "config.yaml")

        console.print(f"[green]Checkpoint saved to {output_dir}[/green]")

    def load_checkpoint(self, checkpoint_dir: Path) -> None:
        """Load from checkpoint."""
        student_dir = Path(checkpoint_dir) / "student"
        if isinstance(self.student_model, PeftModel):
            # The trainer already wrapped the student in a fresh LoRA adapter; wrapping it
            # again left the saved weights unloaded (they were looked up under a doubled
            # prefix), so evaluation ran on an untrained adapter. Load them into it.
            load_peft_weights_into(self.student_model, student_dir)
        else:
            self.student_model = PeftModel.from_pretrained(self.student_model, student_dir)

        # Load critic
        self.critic = self.critic.__class__.load(str(checkpoint_dir / "critic.pt"))
        self.critic = self.critic.to(self.device)

        console.print(f"[green]Loaded checkpoint from {checkpoint_dir}[/green]")


def main():
    """Main entry point."""
    # Windows multiprocessing fix
    freeze_support()

    # Example usage
    config = AspireConfig()
    trainer = AspireTrainer(config)

    # Example prompts (in practice, load from dataset)
    train_prompts = [
        "Explain the concept of recursion in programming.",
        "What are the trade-offs between SQL and NoSQL databases?",
        "How does gradient descent work in machine learning?",
    ]

    trainer.train(train_prompts)


if __name__ == "__main__":
    main()
