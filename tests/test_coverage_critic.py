"""Coverage-closing tests for aspire.critic (base, shared, separate)."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from aspire.critic.base import BaseCritic, CriticOutput
from aspire.critic.separate import SeparateCritic
from aspire.critic.shared import SharedEncoderCritic

# ---------------------------------------------------------------------------
# BaseCritic
# ---------------------------------------------------------------------------


class _StubCritic(BaseCritic):
    """Minimal concrete critic whose abstract bodies delegate to the base."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lin = nn.Linear(self.hidden_dim, 1)

    def forward(self, input_ids=None, attention_mask=None, hidden_states=None, **kwargs):
        # Exercise the abstract (no-op) base body too.
        assert super().forward(input_ids, attention_mask, hidden_states) is None
        return CriticOutput(score=self.lin(hidden_states).mean())

    def get_trainable_parameters(self):
        assert super().get_trainable_parameters() is None
        return list(self.parameters())


class TestBaseCriticAbstractBodies:
    def test_abstract_methods_are_noops_and_subclass_works(self):
        critic = _StubCritic(hidden_dim=4)
        hs = torch.ones(2, 3, 4)
        out = critic(hidden_states=hs)
        assert out.score.dim() == 0
        assert len(critic.get_trainable_parameters()) == 2  # weight + bias

    def test_cannot_instantiate_abstract_base(self):
        with pytest.raises(TypeError):
            BaseCritic()  # type: ignore[abstract]

    def test_predict_score_returns_float_and_sets_eval(self):
        critic = _StubCritic(hidden_dim=4)
        critic.train()
        score = critic.predict_score(hidden_states=torch.ones(1, 2, 4))
        assert isinstance(score, float)
        assert critic.training is False


# ---------------------------------------------------------------------------
# SharedEncoderCritic
# ---------------------------------------------------------------------------


class _TinyStudent(nn.Module):
    def __init__(self, hidden=8, vocab=20):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden)
        self.emb = nn.Embedding(vocab, hidden)

    def forward(self, input_ids=None, attention_mask=None, output_hidden_states=False):
        h = self.emb(input_ids)
        return SimpleNamespace(hidden_states=(h * 0.5, h))


class TestSharedTrainableParameters:
    def test_dimension_heads_and_adapter_are_included_but_student_is_not(self):
        student = _TinyStudent()
        critic = SharedEncoderCritic(student, hidden_dim=8, reasoning_dim=6, num_dimensions=2, use_adapters=True)
        ids = {id(p) for p in critic.get_trainable_parameters()}
        for head in critic.dimension_heads:
            assert {id(p) for p in head.parameters()} <= ids
        assert {id(p) for p in critic.adapter.parameters()} <= ids
        assert not ids & {id(p) for p in student.parameters()}

    def test_no_dimension_heads_by_default(self):
        critic = SharedEncoderCritic(_TinyStudent(), hidden_dim=8, reasoning_dim=6)
        assert not hasattr(critic, "dimension_heads")
        assert not hasattr(critic, "adapter")


class TestSharedGetAttentionWeights:
    def test_attention_weights_are_a_masked_distribution(self):
        critic = SharedEncoderCritic(_TinyStudent(), hidden_dim=8, reasoning_dim=6)
        ids = torch.tensor([[1, 2, 3, 4], [5, 6, 0, 0]])
        mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
        weights = critic.get_attention_weights(ids, mask)
        assert weights.shape == (2, 4)
        torch.testing.assert_close(weights.sum(dim=-1), torch.ones(2))
        # Padded positions receive exactly zero attention.
        assert torch.all(weights[1, 2:] == 0)
        assert critic.training is False

    def test_attention_weights_without_mask(self):
        critic = SharedEncoderCritic(_TinyStudent(), hidden_dim=8, reasoning_dim=6)
        weights = critic.get_attention_weights(torch.tensor([[1, 2, 3]]))
        torch.testing.assert_close(weights.sum(dim=-1), torch.ones(1))


# ---------------------------------------------------------------------------
# SeparateCritic
# ---------------------------------------------------------------------------


class _FakeEncoder(nn.Module):
    """Real small module standing in for a HF encoder."""

    def __init__(self, hidden=8, pooler=False):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden)
        self.emb = nn.Embedding(50, hidden)
        self.pooler = pooler
        self.device = torch.device("cpu")

    def forward(self, input_ids=None, attention_mask=None, output_hidden_states=False):
        last = self.emb(input_ids)
        return SimpleNamespace(
            last_hidden_state=last,
            pooler_output=last[:, 0] if self.pooler else None,
        )


class _FakeTokenizer:
    def __init__(self, pad_token=None):
        self.pad_token = pad_token
        self.eos_token = "<eos>"

    def __call__(self, text, **kwargs):
        ids = torch.arange(1, 5).repeat(len(text), 1)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


@pytest.fixture
def patched_hf():
    with patch("aspire.critic.separate.AutoModel") as model_cls, patch(
        "aspire.critic.separate.AutoTokenizer"
    ) as tok_cls, patch("aspire.critic.separate.BitsAndBytesConfig") as bnb_cls:
        encoder = _FakeEncoder()
        model_cls.from_pretrained.return_value = encoder
        tok_cls.from_pretrained.return_value = _FakeTokenizer()
        yield SimpleNamespace(model_cls=model_cls, tok_cls=tok_cls, bnb_cls=bnb_cls, encoder=encoder)


def _make(**kw):
    kw.setdefault("hidden_dim", 8)
    kw.setdefault("reasoning_dim", 4)
    kw.setdefault("device", "cpu")
    return SeparateCritic("x", **kw)


class TestSeparateCriticConstruction:
    def test_existing_pad_token_is_preserved(self, patched_hf):
        patched_hf.tok_cls.from_pretrained.return_value = _FakeTokenizer(pad_token="<pad>")
        assert _make().tokenizer.pad_token == "<pad>"

    def test_missing_pad_token_falls_back_to_eos(self, patched_hf):
        assert _make().tokenizer.pad_token == "<eos>"

    def test_4bit_quantization_config_and_device_map(self, patched_hf):
        _make(load_in_4bit=True)
        kwargs = patched_hf.bnb_cls.call_args.kwargs
        assert kwargs["load_in_4bit"] is True
        assert kwargs["bnb_4bit_quant_type"] == "nf4"
        assert kwargs["bnb_4bit_use_double_quant"] is True
        load_kwargs = patched_hf.model_cls.from_pretrained.call_args.kwargs
        assert load_kwargs["quantization_config"] is patched_hf.bnb_cls.return_value
        assert load_kwargs["device_map"] == "auto"

    def test_8bit_quantization_config(self, patched_hf):
        _make(load_in_8bit=True)
        patched_hf.bnb_cls.assert_called_once_with(load_in_8bit=True)
        assert "quantization_config" in patched_hf.model_cls.from_pretrained.call_args.kwargs

    def test_4bit_takes_precedence_over_8bit(self, patched_hf):
        _make(load_in_4bit=True, load_in_8bit=True)
        patched_hf.bnb_cls.assert_called_once()
        assert patched_hf.bnb_cls.call_args.kwargs["load_in_4bit"] is True

    def test_frozen_encoder_params_excluded_from_trainables(self, patched_hf):
        critic = _make(freeze_encoder=True)
        assert all(not p.requires_grad for p in critic.encoder.parameters())
        trainable = {id(p) for p in critic.get_trainable_parameters()}
        assert not trainable & {id(p) for p in critic.encoder.parameters()}


class TestSeparateCriticForward:
    def test_mean_pooling_without_attention_mask(self, patched_hf):
        critic = _make()
        critic.eval()
        ids = torch.tensor([[1, 2, 3, 4]])
        out = critic(input_ids=ids)
        pooled = critic.encoder(input_ids=ids).last_hidden_state.mean(dim=1)
        torch.testing.assert_close(out.hidden_states, critic.projection(pooled))

    def test_masked_mean_pooling_ignores_padding(self, patched_hf):
        critic = _make()
        critic.eval()
        ids = torch.tensor([[1, 2, 3, 4]])
        mask = torch.tensor([[1, 1, 0, 0]])
        out = critic(input_ids=ids, attention_mask=mask)
        last = critic.encoder(input_ids=ids).last_hidden_state
        torch.testing.assert_close(out.hidden_states, critic.projection(last[:, :2].mean(dim=1)))

    def test_pooler_output_preferred_when_present(self, patched_hf):
        critic = _make()
        critic.encoder.pooler = True
        critic.eval()
        ids = torch.tensor([[1, 2, 3, 4]])
        out = critic(input_ids=ids)
        expected = critic.projection(critic.encoder(input_ids=ids).pooler_output)
        torch.testing.assert_close(out.hidden_states, expected)

    def test_output_ranges_and_dimension_heads(self, patched_hf):
        critic = _make(num_dimensions=3)
        critic.eval()
        out = critic(text=["a", "b"])
        assert out.score.shape == (2,)
        assert torch.all((out.score >= 0) & (out.score <= 10))
        torch.testing.assert_close(out.reasoning_embedding.norm(dim=-1), torch.ones(2))
        assert set(out.dimension_scores) == {"dim_0", "dim_1", "dim_2"}
        assert all(v.shape == (2, 1) for v in out.dimension_scores.values())

    def test_trainable_parameters_include_dimension_heads_and_encoder(self, patched_hf):
        critic = _make(num_dimensions=2)
        ids = {id(p) for p in critic.get_trainable_parameters()}
        for head in critic.dimension_heads:
            assert {id(p) for p in head.parameters()} <= ids
        assert {id(p) for p in critic.encoder.parameters()} <= ids

    def test_encode_text_returns_features_in_eval_no_grad(self, patched_hf):
        critic = _make()
        critic.train()
        feats = critic.encode_text("hello")
        assert feats.shape == (1, 8)
        assert feats.requires_grad is False
        assert critic.training is False

    def test_encode_text_batch(self, patched_hf):
        assert _make().encode_text(["a", "b", "c"]).shape == (3, 8)

    def test_requires_input(self, patched_hf):
        with pytest.raises(ValueError, match="input_ids or text"):
            _make()(hidden_states=torch.zeros(1, 2, 8))
