"""critic.init_seed: the critic's starting weights get their own stream (2026-10-08 critic-init test).

The test separates the critic's initial weights from everything else the run's seed controls, so
changing one seed must not change the other's stream.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from aspire.config import AspireConfig, CriticConfig
from aspire.trainer import AspireTrainer


def build(critic: CriticConfig, run_seed: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """The critic's initial weights, and the next draw from the run's stream after building it."""
    torch.manual_seed(run_seed)
    torch.rand(3)  # the student is built first and draws from the run's stream
    trainer = AspireTrainer.__new__(AspireTrainer)
    trainer.config = SimpleNamespace(critic=critic)
    trainer.student_hidden_size = 32
    trainer.student_model = None
    trainer.device = "cpu"
    trainer._init_critic()
    weights = {k: v.detach().clone() for k, v in trainer.critic.state_dict().items()}
    return weights, torch.rand(5)


def head(init_seed: int | None) -> CriticConfig:
    return CriticConfig(
        architecture="head", head_hidden_dim=16, reasoning_embedding_dim=8, init_seed=init_seed
    )


def same(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


class TestCriticInitSeed:
    def test_default_is_none_and_validates(self):
        assert CriticConfig().init_seed is None
        assert AspireConfig(critic={"init_seed": 43}).critic.init_seed == 43

    def test_the_same_init_seed_gives_the_same_critic_whatever_the_run_seed(self):
        a, _ = build(head(42), run_seed=42)
        b, _ = build(head(42), run_seed=44)
        assert same(a, b)

    def test_a_different_init_seed_gives_a_different_critic(self):
        a, _ = build(head(42), run_seed=42)
        b, _ = build(head(43), run_seed=42)
        assert not same(a, b)

    def test_changing_the_init_seed_leaves_the_run_stream_untouched(self):
        _, after_42 = build(head(42), run_seed=42)
        _, after_43 = build(head(43), run_seed=42)
        torch.manual_seed(42)
        torch.rand(3)
        untouched = torch.rand(5)  # as if no critic had been built
        assert torch.equal(after_42, after_43) and torch.equal(after_42, untouched)

    def test_changing_the_run_seed_changes_the_run_stream(self):
        _, a = build(head(42), run_seed=42)
        _, b = build(head(42), run_seed=43)
        assert not torch.equal(a, b)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_the_gpu_stream_is_not_reseeded(self):
        torch.manual_seed(42)
        before = torch.cuda.get_rng_state()
        build(head(7), run_seed=42)
        torch.manual_seed(42)
        assert torch.equal(torch.cuda.get_rng_state(), before)

    def test_without_an_init_seed_the_critic_draws_from_the_run_stream_as_before(self):
        a, after_a = build(head(None), run_seed=42)
        b, after_b = build(head(None), run_seed=43)
        assert not same(a, b)
        torch.manual_seed(42)
        torch.rand(3)
        assert not torch.equal(after_a, torch.rand(5))  # the critic consumed part of the stream
