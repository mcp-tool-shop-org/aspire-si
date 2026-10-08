"""CPU checks that fast_heads (vendored from rnd's cuda-graphs experiment) reproduces critic_heads'
train_head and score_set. The CUDA-graph path needs a GPU and was checked by rnd's bench (1.1.4.1.5)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "sft-experiment"))

import fast_heads as fh  # noqa: E402
from critic_heads import HPARAMS, score_set, train_head  # noqa: E402


def fake_cache(n_pairs: int, dim: int, max_len: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    rng = random.Random(seed)
    feats, masks = [], []
    for _ in range(2 * n_pairs):
        t = rng.randint(min(3, max_len), max_len)
        feats.append(torch.randn(t, dim, generator=g).half())
        masks.append(torch.ones(t, dtype=torch.long))
    return feats, masks, [(2 * i, 2 * i + 1) for i in range(n_pairs)]


def test_batch_plan_matches_the_original_loop():
    index = [(2 * i, 2 * i + 1) for i in range(19)]
    flip = [i % 3 == 0 for i in range(19)]
    rng, pairs, expect = random.Random(7), list(range(19)), []
    for _ in range(2):
        rng.shuffle(pairs)
        for start in range(0, 19, 8):
            ks = []
            for i in pairs[start : start + 8]:
                s, f = index[i]
                ks += [f, s] if flip[i] else [s, f]
            expect.append(ks)
    assert fh.batch_plan(index, flip, 7, 2, 8) == expect


@pytest.mark.parametrize("role, pooling, max_len", [("auditor", "attention", 9), ("advocate", "mean", 1)])
def test_the_fast_eager_path_reproduces_train_head(role, pooling, max_len):
    feats, masks, index = fake_cache(13, 24, max_len)
    hp = {**HPARAMS, "hidden_dim": 16, "dropout": 0.0, "epochs": 2, "batch_pairs": 4}
    ref = train_head(role, pooling, 42, feats, masks, index, None, "cpu", hp)
    ref_strong, ref_flawed = score_set(ref, feats, masks, index, "cpu")
    x, m, lens = fh.stack_features(feats, masks, "cpu")
    head, losses = fh.train_head_fast(role, pooling, 42, x, m, lens, index, None, hp)
    strong, flawed = fh.score_set_fast(head, x, m, lens, index, chunk=5)
    assert len(losses) == 2 * 4
    assert strong == pytest.approx(ref_strong, abs=1e-4) and flawed == pytest.approx(ref_flawed, abs=1e-4)
    assert all(torch.allclose(p, q, atol=1e-5) for p, q in zip(ref.parameters(), head.parameters()))


def test_a_head_does_not_depend_on_heads_trained_before_it():
    feats, masks, index = fake_cache(13, 24, 9)
    hp = {**HPARAMS, "hidden_dim": 16, "dropout": 0.3, "epochs": 2, "batch_pairs": 4}
    x, m, lens = fh.stack_features(feats, masks, "cpu")
    torch.manual_seed(1234)
    alone, losses_alone = fh.train_head_fast("auditor", "attention", 43, x, m, lens, index, None, hp)
    torch.manual_seed(1234)
    fh.train_head_fast("auditor", "attention", 42, x, m, lens, index, None, hp)
    torch.rand(1000)
    after, losses_after = fh.train_head_fast("auditor", "attention", 43, x, m, lens, index, None, hp)
    assert losses_alone == losses_after
    assert all(torch.equal(p, q) for p, q in zip(alone.parameters(), after.parameters()))


def test_stack_pads_with_a_zero_mask_in_the_stored_dtype():
    feats, masks, _ = fake_cache(2, 4, 6)
    x, m, lens = fh.stack_features(feats, masks, "cpu")
    assert x.dtype == torch.float16
    for k, t in enumerate(lens):
        assert int(m[k].sum()) == t and torch.equal(x[k, :t], feats[k]) and float(x[k, t:].abs().sum()) == 0.0
