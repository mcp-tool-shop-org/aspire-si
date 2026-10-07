"""
Unit tests for integrations/code/code_critic.py.

All models are tiny (hidden_dim=16, one layer) and run on CPU. HuggingFace
classes are mocked so nothing is downloaded.
"""

from __future__ import annotations

import logging
import math
import sys
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from integrations.code.code_critic import (
    CodeCritic,
    CodeCriticHead,
    CodeCriticLoss,
    CodeEncoder,
    CriticOutput,
    PositionalEncoding,
    create_critic_from_pretrained,
)
from integrations.code.config import CriticArchitecture, CriticConfig

DIMENSIONS = CodeCriticHead.DIMENSIONS


# ============================================================================
# PositionalEncoding
# ============================================================================


class TestPositionalEncoding:
    def test_buffer_shape_and_first_row(self):
        pe = PositionalEncoding(d_model=8, max_len=32, dropout=0.0)
        assert pe.pe.shape == (1, 32, 8)
        # position 0: sin(0)=0 on even dims, cos(0)=1 on odd dims
        assert torch.allclose(pe.pe[0, 0, 0::2], torch.zeros(4))
        assert torch.allclose(pe.pe[0, 0, 1::2], torch.ones(4))

    def test_known_value_at_position_one(self):
        pe = PositionalEncoding(d_model=4, max_len=4, dropout=0.0)
        assert pe.pe[0, 1, 0].item() == pytest.approx(math.sin(1.0), abs=1e-6)
        assert pe.pe[0, 1, 1].item() == pytest.approx(math.cos(1.0), abs=1e-6)

    def test_forward_adds_encoding_and_truncates_to_sequence(self):
        pe = PositionalEncoding(d_model=8, max_len=32, dropout=0.0).eval()
        x = torch.zeros(2, 5, 8)
        out = pe(x)
        assert out.shape == (2, 5, 8)
        assert torch.allclose(out[0], pe.pe[0, :5])
        assert torch.allclose(out[0], out[1])

    def test_sequence_longer_than_max_len_raises(self):
        pe = PositionalEncoding(d_model=8, max_len=4, dropout=0.0)
        with pytest.raises(RuntimeError):
            pe(torch.zeros(1, 6, 8))

    def test_dropout_active_in_train_mode_only(self):
        pe = PositionalEncoding(d_model=8, max_len=16, dropout=0.9)
        x = torch.ones(1, 16, 8)
        pe.eval()
        assert torch.allclose(pe(x), pe(x))
        pe.train()
        torch.manual_seed(0)
        assert (pe(x) == 0).any()


# ============================================================================
# CodeEncoder
# ============================================================================


class TestCodeEncoder:
    @pytest.fixture
    def encoder(self):
        torch.manual_seed(0)
        return CodeEncoder(
            vocab_size=50, hidden_dim=16, num_layers=1, num_heads=2, max_seq_len=16, dropout=0.0
        ).eval()

    def test_scratch_encoder_output_shape(self, encoder):
        ids = torch.randint(0, 50, (3, 7))
        encoding, attention = encoder(ids)
        assert encoding.shape == (3, 7, 16)
        assert attention is None

    def test_output_is_layer_normalised(self, encoder):
        encoding, _ = encoder(torch.randint(0, 50, (2, 6)))
        # LayerNorm with default affine init -> per-token mean ~0, var ~1
        assert torch.allclose(encoding.mean(-1), torch.zeros(2, 6), atol=1e-5)
        assert torch.allclose(encoding.var(-1, unbiased=False), torch.ones(2, 6), atol=1e-2)

    def test_padding_mask_hides_padded_token_ids(self, encoder):
        ids_a = torch.tensor([[5, 6, 7, 0, 0]])
        ids_b = torch.tensor([[5, 6, 7, 40, 41]])
        mask = torch.tensor([[1, 1, 1, 0, 0]])
        with torch.no_grad():
            enc_a, _ = encoder(ids_a, mask)
            enc_b, _ = encoder(ids_b, mask)
            enc_c, _ = encoder(ids_b)  # no mask: padding tokens leak into attention
        assert torch.allclose(enc_a[0, :3], enc_b[0, :3], atol=1e-5)
        assert not torch.allclose(enc_b[0, :3], enc_c[0, :3], atol=1e-5)

    def test_architecture_stored(self, encoder):
        assert encoder.architecture == CriticArchitecture.TRANSFORMER
        assert encoder.hidden_dim == 16
        assert not hasattr(encoder, "backbone")

    def test_codebert_without_pretrained_name_builds_scratch(self):
        enc = CodeEncoder(
            vocab_size=20, hidden_dim=8, num_layers=1, num_heads=2,
            architecture=CriticArchitecture.CODEBERT, pretrained_model=None,
        )
        assert hasattr(enc, "embedding")
        assert not hasattr(enc, "backbone")

    @patch("transformers.AutoTokenizer")
    @patch("transformers.AutoModel")
    def test_codebert_backbone_with_projection(self, mock_model_cls, mock_tok_cls):
        backbone = MagicMock()
        backbone.config.hidden_size = 24
        mock_model_cls.from_pretrained.return_value = backbone
        mock_tok_cls.from_pretrained.return_value = "tok"

        enc = CodeEncoder(
            hidden_dim=8,
            architecture=CriticArchitecture.CODEBERT,
            pretrained_model="some/model",
        )

        mock_model_cls.from_pretrained.assert_called_once_with("some/model")
        mock_tok_cls.from_pretrained.assert_called_once_with("some/model")
        assert enc.tokenizer == "tok"
        assert isinstance(enc.projection, nn.Linear)
        assert (enc.projection.in_features, enc.projection.out_features) == (24, 8)

    @patch("transformers.AutoTokenizer")
    @patch("transformers.AutoModel")
    def test_codebert_backbone_identity_projection_when_dims_match(self, mock_model_cls, mock_tok_cls):
        backbone = MagicMock()
        backbone.config.hidden_size = 8
        mock_model_cls.from_pretrained.return_value = backbone
        enc = CodeEncoder(
            hidden_dim=8, architecture=CriticArchitecture.CODEBERT, pretrained_model="m"
        )
        assert isinstance(enc.projection, nn.Identity)

    @patch("transformers.AutoTokenizer")
    @patch("transformers.AutoModel")
    def test_codebert_forward_uses_backbone_and_returns_last_attention(self, mock_model_cls, mock_tok_cls):
        hidden = torch.randn(2, 4, 8)
        attn_first, attn_last = torch.zeros(2, 2, 4, 4), torch.ones(2, 2, 4, 4)
        backbone = MagicMock()
        backbone.config.hidden_size = 8
        backbone.return_value = MagicMock(last_hidden_state=hidden, attentions=(attn_first, attn_last))
        mock_model_cls.from_pretrained.return_value = backbone

        enc = CodeEncoder(
            hidden_dim=8, architecture=CriticArchitecture.CODEBERT, pretrained_model="m"
        )
        ids = torch.zeros(2, 4, dtype=torch.long)
        mask = torch.ones(2, 4, dtype=torch.long)
        encoding, attention = enc(ids, mask)

        backbone.assert_called_once()
        kwargs = backbone.call_args.kwargs
        assert kwargs["output_attentions"] is True
        assert torch.equal(kwargs["attention_mask"], mask)
        assert encoding.shape == (2, 4, 8)
        assert attention is attn_last

    @patch("transformers.AutoTokenizer")
    @patch("transformers.AutoModel")
    def test_codebert_forward_without_attentions_returns_none(self, mock_model_cls, mock_tok_cls):
        backbone = MagicMock()
        backbone.config.hidden_size = 8
        backbone.return_value = MagicMock(last_hidden_state=torch.randn(1, 3, 8), attentions=None)
        mock_model_cls.from_pretrained.return_value = backbone
        enc = CodeEncoder(
            hidden_dim=8, architecture=CriticArchitecture.CODEBERT, pretrained_model="m"
        )
        _, attention = enc(torch.zeros(1, 3, dtype=torch.long))
        assert attention is None

    def test_codebert_import_error_message(self):
        # A None entry in sys.modules makes `from transformers import ...` raise ImportError
        with patch.dict(sys.modules, {"transformers": None}):
            with pytest.raises(ImportError, match="pip install transformers"):
                CodeEncoder(
                    hidden_dim=8,
                    architecture=CriticArchitecture.CODEBERT,
                    pretrained_model="m",
                )


# ============================================================================
# CodeCriticHead
# ============================================================================


class TestCodeCriticHead:
    @pytest.fixture
    def head(self):
        torch.manual_seed(0)
        return CodeCriticHead(hidden_dim=16, reasoning_dim=6, fix_dim=10).eval()

    def test_full_output_shapes(self, head):
        out = head(torch.randn(3, 5, 16))
        assert out.score.shape == (3,)
        assert set(out.dimension_scores) == set(DIMENSIONS)
        assert all(v.shape == (3,) for v in out.dimension_scores.values())
        assert out.token_scores.shape == (3, 5)
        assert out.reasoning_embedding.shape == (3, 6)
        assert out.fix_embedding.shape == (3, 10)
        assert out.attention_weights is None

    def test_scores_bounded_zero_to_ten(self, head):
        out = head(torch.randn(8, 4, 16) * 50)
        for t in (out.score, out.token_scores, *out.dimension_scores.values()):
            assert (t >= 0).all() and (t <= 10).all()

    @pytest.mark.parametrize("pool", ["mean", "cls", "max"])
    def test_pooling_modes_run_and_differ(self, head, pool):
        enc = torch.randn(2, 6, 16)
        out = head(enc, pool=pool)
        assert out.score.shape == (2,)

    def test_pooling_modes_produce_distinct_scores(self, head):
        torch.manual_seed(1)
        enc = torch.randn(2, 6, 16)
        scores = {p: head(enc, pool=p).score for p in ("mean", "cls", "max")}
        assert not torch.allclose(scores["mean"], scores["cls"])
        assert not torch.allclose(scores["mean"], scores["max"])

    def test_cls_pool_only_depends_on_first_token(self, head):
        enc = torch.randn(1, 6, 16)
        other = enc.clone()
        other[:, 1:] = torch.randn(1, 5, 16)
        assert torch.allclose(head(enc, pool="cls").score, head(other, pool="cls").score)

    def test_unknown_pool_raises(self, head):
        with pytest.raises(ValueError, match="Unknown pool: median"):
            head(torch.randn(1, 2, 16), pool="median")

    def test_all_heads_disabled(self):
        head = CodeCriticHead(
            hidden_dim=16,
            predict_score=False,
            predict_dimensions=False,
            predict_tokens=False,
            predict_reasoning=False,
            predict_fix=False,
        )
        assert head.score_head is None
        assert head.dimension_heads is None
        assert head.token_head is None
        assert head.reasoning_head is None
        assert head.fix_head is None
        out = head(torch.randn(2, 3, 16))
        assert out.score is None
        assert out.dimension_scores is None
        assert out.token_scores is None
        assert out.reasoning_embedding is None
        assert out.fix_embedding is None

    def test_gradients_reach_all_heads(self):
        head = CodeCriticHead(hidden_dim=16)
        out = head(torch.randn(2, 4, 16))
        total = (
            out.score.sum()
            + sum(v.sum() for v in out.dimension_scores.values())
            + out.token_scores.sum()
            + out.reasoning_embedding.sum()
            + out.fix_embedding.sum()
        )
        total.backward()
        no_grad = [n for n, p in head.named_parameters() if p.grad is None]
        assert no_grad == []


# ============================================================================
# CodeCritic
# ============================================================================


class TestCodeCritic:
    def test_default_config_is_used_when_none(self):
        # Build the default config's encoder lazily patched to keep this cheap
        with patch("integrations.code.code_critic.CodeEncoder") as enc_cls, patch(
            "integrations.code.code_critic.CodeCriticHead"
        ) as head_cls:
            critic = CodeCritic()
        assert isinstance(critic.config, CriticConfig)
        kwargs = enc_cls.call_args.kwargs
        assert kwargs["hidden_dim"] == 512
        assert kwargs["num_layers"] == 6
        assert kwargs["max_seq_len"] == 2048
        assert kwargs["architecture"] == CriticArchitecture.TRANSFORMER
        head_kwargs = head_cls.call_args.kwargs
        assert head_kwargs["predict_dimensions"] is True
        assert head_kwargs["predict_tokens"] is True

    def test_config_flags_propagate_to_heads(self, tiny_critic_config):
        tiny_critic_config.predict_score = False
        tiny_critic_config.predict_reasoning = False
        tiny_critic_config.predict_fix = False
        critic = CodeCritic(tiny_critic_config)
        assert critic.heads.score_head is None
        assert critic.heads.reasoning_head is None
        assert critic.heads.fix_head is None
        assert critic.heads.dimension_heads is not None

    def test_forward_shapes(self, tiny_critic):
        ids = torch.randint(1, 50, (3, 8))
        mask = torch.ones(3, 8, dtype=torch.long)
        out = tiny_critic(ids, mask)
        assert out.score.shape == (3,)
        assert out.token_scores.shape == (3, 8)
        assert out.reasoning_embedding.shape == (3, 256)
        assert out.fix_embedding.shape == (3, 512)
        assert out.attention_weights is None

    def test_forward_without_mask(self, tiny_critic):
        out = tiny_critic(torch.randint(1, 50, (2, 5)))
        assert out.score.shape == (2,)

    def test_forward_is_deterministic_in_eval(self, tiny_critic):
        tiny_critic.eval()
        ids = torch.randint(1, 50, (2, 5))
        with torch.no_grad():
            a = tiny_critic(ids).score
            b = tiny_critic(ids).score
        assert torch.equal(a, b)

    # ---- tokenizer ----

    def test_get_tokenizer_returns_cached_instance(self, tiny_critic, fake_tokenizer):
        assert tiny_critic.get_tokenizer() is fake_tokenizer

    def test_get_tokenizer_uses_encoder_tokenizer_when_present(self, tiny_critic_config):
        critic = CodeCritic(tiny_critic_config)
        critic.encoder.tokenizer = "encoder-tok"
        assert critic.get_tokenizer() == "encoder-tok"
        assert critic._tokenizer == "encoder-tok"

    def test_get_tokenizer_loads_from_pretrained_name(self, tiny_critic_config):
        tiny_critic_config.pretrained_model = "my/model"
        critic = CodeCritic(tiny_critic_config)
        with patch("transformers.AutoTokenizer.from_pretrained", return_value="loaded") as fp:
            assert critic.get_tokenizer() == "loaded"
            assert critic.get_tokenizer() == "loaded"  # second call is cached
        fp.assert_called_once_with("my/model")

    def test_get_tokenizer_defaults_to_codebert_name(self, tiny_critic_config):
        critic = CodeCritic(tiny_critic_config)
        with patch("transformers.AutoTokenizer.from_pretrained", return_value="cb") as fp:
            critic.get_tokenizer()
        fp.assert_called_once_with("microsoft/codebert-base")

    def test_get_tokenizer_failure_logs_warning_and_returns_none(self, tiny_critic_config, caplog):
        critic = CodeCritic(tiny_critic_config)
        with patch("transformers.AutoTokenizer.from_pretrained", side_effect=OSError("offline")):
            with caplog.at_level(logging.WARNING, logger="integrations.code.code_critic"):
                assert critic.get_tokenizer() is None
        assert "offline" in caplog.text
        assert "microsoft/codebert-base" in caplog.text

    def test_tokenize_wraps_single_string_and_pads(self, tiny_critic, fake_tokenizer):
        tokens = tiny_critic.tokenize("abc")
        assert tokens["input_ids"].shape == (1, 3)
        call = fake_tokenizer.calls[-1]
        assert call["padding"] is True
        assert call["max_length"] == 16  # config.max_code_length
        assert call["n"] == 1

    def test_tokenize_batch_and_explicit_max_length(self, tiny_critic, fake_tokenizer):
        tokens = tiny_critic.tokenize(["abcdef", "ab"], max_length=4)
        assert tokens["input_ids"].shape == (2, 4)
        assert tokens["attention_mask"].tolist() == [[1, 1, 1, 1], [1, 1, 0, 0]]

    def test_tokenize_without_tokenizer_raises(self, tiny_critic):
        tiny_critic._tokenizer = None
        with patch("transformers.AutoTokenizer.from_pretrained", side_effect=OSError("x")):
            with pytest.raises(ValueError, match="No tokenizer available"):
                tiny_critic.tokenize("code")

    # ---- scoring ----

    def test_score_code_single_string(self, tiny_critic):
        score = tiny_critic.score_code("def f(): pass", device="cpu")
        assert score.shape == (1,)
        assert 0.0 <= score.item() <= 10.0
        assert not tiny_critic.training  # score_code leaves the model in eval mode

    def test_score_code_batch_matches_forward(self, tiny_critic):
        codes = ["x = 1", "y = 2222"]
        scores = tiny_critic.score_code(codes, device="cpu")
        assert scores.shape == (2,)
        tokens = tiny_critic.tokenize(codes)
        with torch.no_grad():
            expected = tiny_critic(tokens["input_ids"], tokens["attention_mask"]).score
        assert torch.allclose(scores, expected)

    def test_score_code_returns_no_grad_tensor(self, tiny_critic):
        assert tiny_critic.score_code("x", device="cpu").requires_grad is False

    # ---- problem tokens ----

    def test_get_problem_tokens_threshold_extremes(self, tiny_critic):
        code = "abcdef"
        assert tiny_critic.get_problem_tokens(code, threshold=-1.0, device="cpu") == []
        everything = tiny_critic.get_problem_tokens(code, threshold=11.0, device="cpu")
        assert [(s, e) for s, e, _ in everything] == [(i, i + 1) for i in range(6)]
        assert all(0.0 <= sc <= 10.0 for _, _, sc in everything)

    def test_get_problem_tokens_filters_by_score(self, tiny_critic):
        code = "abcdefgh"
        everything = tiny_critic.get_problem_tokens(code, threshold=11.0, device="cpu")
        scores = sorted(sc for _, _, sc in everything)
        cut = (scores[3] + scores[4]) / 2
        flagged = tiny_critic.get_problem_tokens(code, threshold=cut, device="cpu")
        assert len(flagged) == 4
        assert all(sc < cut for _, _, sc in flagged)

    def test_get_problem_tokens_returns_empty_without_token_head(self, tiny_critic):
        tiny_critic.heads.token_head = None
        assert tiny_critic.get_problem_tokens("abc", threshold=11.0, device="cpu") == []


# ============================================================================
# CodeCriticLoss
# ============================================================================


class TestCodeCriticLoss:
    def _output(self, score, **kw):
        return CriticOutput(score=score, **kw)

    def test_score_only_is_mse(self):
        loss_fn = CodeCriticLoss()
        losses = loss_fn(self._output(torch.tensor([2.0, 4.0])), torch.tensor([3.0, 6.0]))
        assert losses["score"].item() == pytest.approx((1 + 4) / 2)
        assert losses["dimensions"].item() == 0.0
        assert losses["tokens"].item() == 0.0
        assert losses["reasoning"].item() == 0.0
        assert losses["total"].item() == pytest.approx(2.5)

    def test_missing_score_head_gives_zero_score_loss(self):
        loss_fn = CodeCriticLoss()
        losses = loss_fn(self._output(None), torch.tensor([3.0]))
        assert losses["score"].item() == 0.0
        assert losses["total"].item() == 0.0

    def test_dimension_loss_averages_only_shared_dimensions(self):
        pred = {"correctness": torch.tensor([1.0]), "style": torch.tensor([5.0]), "security": torch.tensor([0.0])}
        true = {"correctness": torch.tensor([3.0]), "style": torch.tensor([5.0])}
        losses = CodeCriticLoss()(
            self._output(torch.tensor([1.0]), dimension_scores=pred),
            torch.tensor([1.0]),
            teacher_dimensions=true,
        )
        # mean(MSE(correctness)=4, MSE(style)=0) = 2 ; security ignored
        assert losses["dimensions"].item() == pytest.approx(2.0)
        assert losses["total"].item() == pytest.approx(0.5 * 2.0)

    def test_dimension_loss_zero_when_no_overlap(self):
        losses = CodeCriticLoss()(
            self._output(torch.tensor([1.0]), dimension_scores={"style": torch.tensor([1.0])}),
            torch.tensor([1.0]),
            teacher_dimensions={"other": torch.tensor([9.0])},
        )
        assert losses["dimensions"].item() == 0.0

    def test_token_loss_aligns_to_shorter_length(self):
        pred = torch.tensor([[1.0, 2.0, 3.0, 9.0]])
        true = torch.tensor([[1.0, 4.0]])
        losses = CodeCriticLoss()(
            self._output(torch.tensor([0.0]), token_scores=pred),
            torch.tensor([0.0]),
            teacher_token_scores=true,
        )
        assert losses["tokens"].item() == pytest.approx((0 + 4) / 2)

    def test_reasoning_loss_is_one_minus_cosine(self):
        emb = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        target = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        losses = CodeCriticLoss()(
            self._output(torch.tensor([0.0, 0.0]), reasoning_embedding=emb),
            torch.tensor([0.0, 0.0]),
            teacher_reasoning_embedding=target,
        )
        # cosines are 1 and 0 -> mean(1 - cos) = 0.5
        assert losses["reasoning"].item() == pytest.approx(0.5)

    def test_total_uses_custom_weights(self):
        loss_fn = CodeCriticLoss(
            score_weight=2.0, dimension_weight=3.0, token_weight=5.0, reasoning_weight=7.0
        )
        out = self._output(
            torch.tensor([0.0]),
            dimension_scores={"style": torch.tensor([1.0])},
            token_scores=torch.tensor([[2.0]]),
            reasoning_embedding=torch.tensor([[0.0, 1.0]]),
        )
        losses = loss_fn(
            out,
            torch.tensor([1.0]),
            teacher_dimensions={"style": torch.tensor([3.0])},
            teacher_reasoning_embedding=torch.tensor([[1.0, 0.0]]),
            teacher_token_scores=torch.tensor([[5.0]]),
        )
        assert losses["score"].item() == pytest.approx(1.0)
        assert losses["dimensions"].item() == pytest.approx(4.0)
        assert losses["tokens"].item() == pytest.approx(9.0)
        assert losses["reasoning"].item() == pytest.approx(1.0)
        assert losses["total"].item() == pytest.approx(2 * 1 + 3 * 4 + 5 * 9 + 7 * 1)

    def test_total_is_differentiable_through_critic(self, tiny_critic):
        ids = torch.randint(1, 50, (2, 6))
        out = tiny_critic(ids)
        losses = CodeCriticLoss()(out, torch.tensor([5.0, 8.0]))
        losses["total"].backward()
        grads = [p.grad for p in tiny_critic.heads.score_head.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)

    def test_zero_loss_when_prediction_equals_target(self):
        score = torch.tensor([4.0, 7.0])
        losses = CodeCriticLoss()(self._output(score), score.clone())
        assert losses["total"].item() == 0.0


# ============================================================================
# create_critic_from_pretrained
# ============================================================================


class TestCreateCriticFromPretrained:
    @patch("transformers.AutoTokenizer")
    @patch("transformers.AutoModel")
    def test_builds_codebert_critic(self, mock_model_cls, mock_tok_cls):
        backbone = MagicMock()
        backbone.config.hidden_size = 32
        mock_model_cls.from_pretrained.return_value = backbone
        mock_tok_cls.from_pretrained.return_value = "tok"

        critic = create_critic_from_pretrained("org/code-model", hidden_dim=32, num_layers=1)

        assert critic.config.architecture == CriticArchitecture.CODEBERT
        assert critic.config.pretrained_model == "org/code-model"
        assert critic.config.hidden_dim == 32
        assert critic.encoder.backbone is backbone
        assert critic.get_tokenizer() == "tok"

    def test_default_model_name(self):
        with patch("integrations.code.code_critic.CodeCritic") as critic_cls:
            create_critic_from_pretrained()
        cfg = critic_cls.call_args.args[0]
        assert cfg.pretrained_model == "microsoft/codebert-base"
        assert cfg.architecture == CriticArchitecture.CODEBERT
