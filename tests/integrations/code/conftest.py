"""Shared fixtures for the integrations/code tests (CPU-only, no network)."""

from __future__ import annotations

import pytest
import torch

from integrations.code.code_critic import CodeCritic
from integrations.code.config import CriticArchitecture, CriticConfig


class FakeTokenizer:
    """Deterministic character-level tokenizer that mimics the HF call signature."""

    pad_token = None
    eos_token = "<eos>"

    def __init__(self, vocab: int = 100):
        self.vocab = vocab
        self.calls: list[dict] = []
        self.saved_to: list[str] = []

    def __call__(self, text, padding=False, truncation=False, max_length=None, return_tensors=None):
        texts = [text] if isinstance(text, str) else list(text)
        self.calls.append({"padding": padding, "max_length": max_length, "n": len(texts)})
        ids = []
        for t in texts:
            row = [(ord(c) % (self.vocab - 1)) + 1 for c in t]  # 1..vocab-1, 0 is padding
            if truncation and max_length is not None:
                row = row[:max_length]
            ids.append(row)
        if padding == "max_length":
            width = max_length
        elif padding:
            width = max(len(r) for r in ids)
        else:
            width = None
        masks = []
        if width is not None:
            for i, row in enumerate(ids):
                masks.append([1] * len(row) + [0] * (width - len(row)))
                ids[i] = row + [0] * (width - len(row))
        else:
            masks = [[1] * len(r) for r in ids]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }

    def save_pretrained(self, path):
        self.saved_to.append(str(path))

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


@pytest.fixture
def fake_tokenizer():
    return FakeTokenizer()


@pytest.fixture
def tiny_critic_config():
    return CriticConfig(
        architecture=CriticArchitecture.TRANSFORMER,
        hidden_dim=16,
        num_layers=1,
        num_heads=2,
        max_code_length=16,
        pretrained_model=None,
        dropout=0.0,
    )


@pytest.fixture
def tiny_critic(tiny_critic_config, fake_tokenizer):
    torch.manual_seed(0)
    critic = CodeCritic(tiny_critic_config)
    critic._tokenizer = fake_tokenizer  # avoids any HuggingFace download
    return critic
