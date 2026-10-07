"""
Judging a response with a trained critic, without the teacher.

The critic scores the student's last hidden layer over a prompt and a response, laid out the way
the student saw them in training: its own chat format when its tokenizer has one, the prompt and
the response on separate paragraphs otherwise. ``critic_score`` scores one response,
``critic_scores`` a batch, and ``Judge.from_checkpoint`` loads what ``aspire train`` saved
(student, critic and config) so that ``judge.score(prompt, response)`` needs nothing else.

The score is the critic's prediction of the teacher's 0-10 score. It is only as good as the
training run behind it: a critic trained on a few dozen prompts has seen a few dozen scores.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from aspire.dialogue.generator import uses_chat_template
from aspire.errors import ConfigError


def format_exchange(tokenizer: Any, prompt: str, response: str) -> str:
    """A prompt and its response as one text, in the student's chat format if it has one."""
    if uses_chat_template(tokenizer):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}],
            tokenize=False,
        )
    return f"{prompt}\n\n{response}"


def encode_exchanges(
    tokenizer: Any,
    prompts: Sequence[str],
    responses: Sequence[str],
    max_length: int = 2048,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Token ids and attention mask for each prompt and response, padded to the longest."""
    if len(prompts) != len(responses):
        raise ValueError(f"{len(prompts)} prompts but {len(responses)} responses.")
    texts = [format_exchange(tokenizer, p, r) for p, r in zip(prompts, responses)]
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        # A chat template already starts with the model's special tokens.
        add_special_tokens=not uses_chat_template(tokenizer),
    )
    return encoded["input_ids"], encoded["attention_mask"]


def _device_of(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except (StopIteration, AttributeError, TypeError):
        return torch.device("cpu")


@torch.no_grad()
def critic_scores(
    student: Any,
    tokenizer: Any,
    critic: Any,
    prompts: Sequence[str],
    responses: Sequence[str],
    max_length: int = 2048,
) -> list[float]:
    """The critic's 0-10 score for each prompt and response."""
    input_ids, attention_mask = encode_exchanges(tokenizer, prompts, responses, max_length)
    device = _device_of(student)
    input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
    student.eval()
    critic.eval()
    outputs = student(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    hidden = outputs.hidden_states[-1]
    critic_device = _device_of(critic)
    output = critic(
        hidden_states=hidden.to(critic_device, torch.float32),
        attention_mask=attention_mask.to(critic_device),
    )
    return [float(score) for score in output.score.reshape(-1).tolist()]


def critic_score(
    student: Any,
    tokenizer: Any,
    critic: Any,
    prompt: str,
    response: str,
    max_length: int = 2048,
) -> float:
    """The critic's 0-10 score for one response to a prompt."""
    return critic_scores(student, tokenizer, critic, [prompt], [response], max_length)[0]


class Judge:
    """A trained student and critic, ready to score responses."""

    def __init__(self, student: Any, tokenizer: Any, critic: Any, max_length: int = 2048):
        self.student = student
        self.tokenizer = tokenizer
        self.critic = critic
        self.max_length = max_length

    def score(self, prompt: str, response: str) -> float:
        """The critic's 0-10 score for one response."""
        return critic_score(self.student, self.tokenizer, self.critic, prompt, response, self.max_length)

    def score_many(self, prompts: Sequence[str], responses: Sequence[str]) -> list[float]:
        """The critic's 0-10 score for each prompt and response."""
        return critic_scores(self.student, self.tokenizer, self.critic, prompts, responses, self.max_length)

    @classmethod
    def from_checkpoint(cls, checkpoint: str | Path, device: str | None = None) -> Judge:
        """Load a checkpoint directory written by ``aspire train`` (``outputs/checkpoint-N``)."""
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        from aspire.config import AspireConfig
        from aspire.critic import CriticHead

        checkpoint = Path(checkpoint)
        for needed in ("config.yaml", "critic.pt", "student"):
            if not (checkpoint / needed).exists():
                raise ConfigError(
                    f"{checkpoint} has no {needed}.",
                    hint="Pass a checkpoint directory written by `aspire train` "
                    "(for example outputs/checkpoint-1).",
                )
        cfg = AspireConfig.from_yaml(checkpoint / "config.yaml")
        if cfg.critic.architecture != "head":
            raise ConfigError(
                f"Judging supports the default critic architecture (head), not {cfg.critic.architecture!r}.",
                hint="Train with critic.architecture: head, "
                "or load the critic yourself and call critic_scores.",
            )
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        student_dir = checkpoint / "student"
        quantization_config = None
        if device != "cpu" and cfg.student.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif device != "cpu" and cfg.student.load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        adapter = (student_dir / "adapter_config.json").exists()
        base = cfg.student.model_name_or_path if adapter else str(student_dir)
        student = AutoModelForCausalLM.from_pretrained(
            base,
            quantization_config=quantization_config,
            device_map={"": device},
            torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
        )
        if adapter:
            from peft import PeftModel

            student = PeftModel.from_pretrained(student, str(student_dir))
        tokenizer = AutoTokenizer.from_pretrained(str(student_dir))
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        critic = CriticHead.load(str(checkpoint / "critic.pt")).to(device)
        return cls(student, tokenizer, critic, max_length=cfg.student.max_length)
