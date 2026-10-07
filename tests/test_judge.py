"""Tests for aspire.judge: scoring a response with a trained critic and no teacher."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from aspire.config import AspireConfig
from aspire.critic import CriticHead
from aspire.errors import ConfigError
from aspire.judge import Judge, critic_score, critic_scores, encode_exchanges, format_exchange


class _Tok:
    """A tokenizer that records how it was called."""

    def __init__(self, chat_template=None):
        self.chat_template = chat_template
        self.calls = []
        self.chat_calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.chat_calls.append((messages, kwargs))
        return f"<chat>{messages[0]['content']}|{messages[1]['content']}</chat>"

    def __call__(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        n = len(texts)
        return {
            "input_ids": torch.ones(n, 5, dtype=torch.long),
            "attention_mask": torch.ones(n, 5, dtype=torch.long),
        }


class _Student(nn.Module):
    """Returns a fixed-width hidden state in the dtype it is built with."""

    def __init__(self, dtype=torch.float32, hidden=8):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1, dtype=dtype))
        self.hidden = hidden
        self.dtype = dtype
        self.seen = {}

    def forward(self, input_ids, attention_mask, output_hidden_states=False):
        self.seen = {"output_hidden_states": output_hidden_states, "shape": tuple(input_ids.shape)}
        torch.manual_seed(0)
        h = torch.randn(input_ids.shape[0], input_ids.shape[1], self.hidden).to(self.dtype)
        return SimpleNamespace(hidden_states=(h * 0, h))


def _critic():
    torch.manual_seed(1)
    return CriticHead(input_dim=8, hidden_dim=16, reasoning_dim=12)


class TestFormatExchange:
    def test_plain_text_without_a_chat_template(self):
        assert format_exchange(_Tok(), "the prompt", "the reply") == "the prompt\n\nthe reply"

    def test_non_string_template_is_plain_text(self):
        assert format_exchange(_Tok(chat_template=3), "a", "b") == "a\n\nb"

    def test_chat_template(self):
        tok = _Tok(chat_template="{{ x }}")
        assert format_exchange(tok, "a", "b") == "<chat>a|b</chat>"
        messages, kwargs = tok.chat_calls[0]
        assert messages == [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
        assert kwargs == {"tokenize": False}


class TestEncodeExchanges:
    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="2 prompts but 1 responses"):
            encode_exchanges(_Tok(), ["a", "b"], ["x"])

    def test_plain_tokenizer_adds_special_tokens_and_pads(self):
        tok = _Tok()
        ids, mask = encode_exchanges(tok, ["a", "b"], ["x", "y"], max_length=64)
        texts, kwargs = tok.calls[0]
        assert texts == ["a\n\nx", "b\n\ny"]
        assert kwargs["padding"] is True
        assert kwargs["truncation"] is True
        assert kwargs["max_length"] == 64
        assert kwargs["return_tensors"] == "pt"
        assert kwargs["add_special_tokens"] is True
        assert ids.shape == mask.shape == (2, 5)

    def test_chat_template_skips_special_tokens(self):
        tok = _Tok(chat_template="{{ x }}")
        encode_exchanges(tok, ["a"], ["x"])
        assert tok.calls[0][1]["add_special_tokens"] is False
        assert tok.calls[0][0] == ["<chat>a|x</chat>"]


class TestCriticScores:
    def test_one_float_in_range_per_pair(self):
        scores = critic_scores(_Student(), _Tok(), _critic(), ["a", "b", "c"], ["x", "y", "z"])
        assert len(scores) == 3
        assert all(isinstance(s, float) and 0.0 <= s <= 10.0 for s in scores)

    def test_critic_score_is_one_float(self):
        student, critic = _Student(), _critic()
        score = critic_score(student, _Tok(), critic, "a", "x")
        assert isinstance(score, float) and 0.0 <= score <= 10.0
        assert score == pytest.approx(critic_scores(student, _Tok(), critic, ["a"], ["x"])[0])

    def test_asks_the_student_for_hidden_states_and_uses_eval_mode(self):
        student, critic = _Student(), _critic()
        student.train()
        critic.train()
        critic_scores(student, _Tok(), critic, ["a"], ["x"], max_length=32)
        assert student.seen["output_hidden_states"] is True
        assert not student.training and not critic.training

    def test_bfloat16_hidden_states_reach_the_critic_as_float32(self):
        student, critic = _Student(dtype=torch.bfloat16), _critic()
        seen = {}

        def hook(module, args, kwargs):
            seen["dtype"] = kwargs["hidden_states"].dtype
            seen["has_mask"] = kwargs.get("attention_mask") is not None

        critic.register_forward_pre_hook(hook, with_kwargs=True)
        scores = critic_scores(student, _Tok(), critic, ["a", "b"], ["x", "y"])
        assert seen == {"dtype": torch.float32, "has_mask": True}
        assert len(scores) == 2

    def test_a_student_without_parameters_runs_on_cpu(self):
        class Bare:
            def eval(self):
                pass

            def __call__(self, **kwargs):
                return SimpleNamespace(hidden_states=(torch.randn(1, 5, 8),))

        assert len(critic_scores(Bare(), _Tok(), _critic(), ["a"], ["x"])) == 1


class TestJudge:
    def test_score_delegates(self):
        judge = Judge("student", "tok", "critic", max_length=99)
        with patch("aspire.judge.critic_score", return_value=4.5) as one:
            assert judge.score("p", "r") == 4.5
        one.assert_called_once_with("student", "tok", "critic", "p", "r", 99)

    def test_score_many_delegates(self):
        judge = Judge("student", "tok", "critic")
        with patch("aspire.judge.critic_scores", return_value=[1.0, 2.0]) as many:
            assert judge.score_many(["p", "q"], ["r", "s"]) == [1.0, 2.0]
        many.assert_called_once_with("student", "tok", "critic", ["p", "q"], ["r", "s"], 2048)

    def test_end_to_end_with_fakes(self):
        judge = Judge(_Student(), _Tok(), _critic())
        assert 0.0 <= judge.score("p", "r") <= 10.0
        assert len(judge.score_many(["p", "q"], ["r", "s"])) == 2


class TestFromCheckpoint:
    def _checkpoint(self, tmp_path, *, config=True, critic=True, student=True, architecture="head", **student_cfg):
        ckpt = tmp_path / "checkpoint-1"
        ckpt.mkdir()
        if config:
            cfg = AspireConfig(device="cpu")
            cfg.critic.architecture = architecture
            for key, value in student_cfg.items():
                setattr(cfg.student, key, value)
            cfg.to_yaml(ckpt / "config.yaml")
        if critic:
            _critic().save(str(ckpt / "critic.pt"))
        if student:
            (ckpt / "student").mkdir()
        return ckpt

    @pytest.mark.parametrize("missing", ["config", "critic", "student"])
    def test_missing_pieces_are_named(self, tmp_path, missing):
        ckpt = self._checkpoint(tmp_path, **{missing: False})
        name = {"config": "config.yaml", "critic": "critic.pt", "student": "student"}[missing]
        with pytest.raises(ConfigError, match=name):
            Judge.from_checkpoint(ckpt)

    def test_non_head_critic_is_rejected(self, tmp_path):
        ckpt = self._checkpoint(tmp_path, architecture="separate")
        with pytest.raises(ConfigError, match="separate"):
            Judge.from_checkpoint(ckpt)

    def _load(self, ckpt, device="cpu", adapter=False):
        if adapter:
            (ckpt / "student" / "adapter_config.json").write_text("{}", encoding="utf-8")
        tokenizer = MagicMock()
        tokenizer.pad_token = None
        tokenizer.eos_token = "<eos>"
        with patch("transformers.AutoModelForCausalLM") as model_cls, patch(
            "transformers.AutoTokenizer"
        ) as tok_cls, patch("transformers.BitsAndBytesConfig") as bnb, patch("peft.PeftModel") as peft:
            tok_cls.from_pretrained.return_value = tokenizer
            judge = Judge.from_checkpoint(ckpt, device=device)
        return judge, tokenizer, model_cls, bnb, peft

    def test_loads_a_full_student_on_cpu(self, tmp_path):
        ckpt = self._checkpoint(tmp_path, max_length=777)
        judge, tokenizer, model_cls, bnb, peft = self._load(ckpt)
        args, kwargs = model_cls.from_pretrained.call_args
        assert args == (str(ckpt / "student"),)
        assert kwargs["quantization_config"] is None
        assert kwargs["torch_dtype"] == torch.float32
        assert kwargs["device_map"] == {"": "cpu"}
        peft.from_pretrained.assert_not_called()
        bnb.assert_not_called()
        assert judge.student is model_cls.from_pretrained.return_value
        assert tokenizer.pad_token == "<eos>"
        assert judge.max_length == 777
        assert isinstance(judge.critic, CriticHead)

    def test_adapter_checkpoint_loads_the_base_then_the_adapter(self, tmp_path):
        ckpt = self._checkpoint(tmp_path)
        judge, _, model_cls, _, peft = self._load(ckpt, adapter=True)
        assert model_cls.from_pretrained.call_args.args == (AspireConfig().student.model_name_or_path,)
        peft.from_pretrained.assert_called_once_with(model_cls.from_pretrained.return_value, str(ckpt / "student"))
        assert judge.student is peft.from_pretrained.return_value

    def test_4bit_on_gpu_builds_a_quantization_config(self, tmp_path):
        ckpt = self._checkpoint(tmp_path, load_in_4bit=True)
        with patch.object(CriticHead, "to", lambda self, device: self):
            _, _, model_cls, bnb, _ = self._load(ckpt, device="cuda")
        assert bnb.call_args.kwargs["load_in_4bit"] is True
        assert model_cls.from_pretrained.call_args.kwargs["quantization_config"] is bnb.return_value
        assert model_cls.from_pretrained.call_args.kwargs["torch_dtype"] == torch.bfloat16

    def test_8bit_on_gpu_builds_a_quantization_config(self, tmp_path):
        ckpt = self._checkpoint(tmp_path, load_in_4bit=False, load_in_8bit=True)
        with patch.object(CriticHead, "to", lambda self, device: self):
            _, _, _, bnb, _ = self._load(ckpt, device="cuda")
        bnb.assert_called_once_with(load_in_8bit=True)

    def test_device_defaults_to_this_machine(self, tmp_path):
        ckpt = self._checkpoint(tmp_path)
        tokenizer = MagicMock()
        tokenizer.pad_token = "<pad>"
        with patch("transformers.AutoModelForCausalLM") as model_cls, patch(
            "transformers.AutoTokenizer"
        ) as tok_cls, patch("transformers.BitsAndBytesConfig"), patch("torch.cuda.is_available", return_value=False):
            tok_cls.from_pretrained.return_value = tokenizer
            Judge.from_checkpoint(ckpt)
        assert model_cls.from_pretrained.call_args.kwargs["device_map"] == {"": "cpu"}
        assert tokenizer.pad_token == "<pad>"


# ---------------------------------------------------------------------------
# `aspire judge`
# ---------------------------------------------------------------------------

import json  # noqa: E402

from typer.testing import CliRunner  # noqa: E402

from aspire import cli  # noqa: E402

runner = CliRunner()


@pytest.fixture
def fake_judge():
    judge = MagicMock()
    judge.score_many.side_effect = lambda prompts, responses: [float(i) + 0.12345 for i, _ in enumerate(prompts)]
    with patch("aspire.judge.Judge.from_checkpoint", return_value=judge) as load:
        judge.load = load
        yield judge


def _pairs_file(tmp_path, data):
    path = tmp_path / "pairs.json"
    path.write_text(data if isinstance(data, str) else json.dumps(data), encoding="utf-8")
    return path


class TestJudgeCommand:
    def test_prompt_and_response(self, fake_judge, tmp_path):
        result = runner.invoke(cli.app, ["judge", str(tmp_path), "--prompt", "2+2?", "--response", "4"])
        assert result.exit_code == 0, result.output
        fake_judge.score_many.assert_called_once_with(["2+2?"], ["4"])
        fake_judge.load.assert_called_once_with(tmp_path, device=None)
        assert "0.12" in result.output and "2+2?" in result.output

    def test_device_is_passed_on(self, fake_judge, tmp_path):
        runner.invoke(cli.app, ["judge", str(tmp_path), "--prompt", "p", "--response", "r", "--device", "cpu"])
        assert fake_judge.load.call_args.kwargs["device"] == "cpu"

    def test_pairs_file_with_json_output(self, fake_judge, tmp_path):
        pairs = _pairs_file(tmp_path, [{"prompt": "a", "response": "x"}, {"prompt": "b", "response": "y"}])
        result = runner.invoke(cli.app, ["judge", str(tmp_path), "--pairs", str(pairs), "--json"])
        assert result.exit_code == 0, result.output
        fake_judge.score_many.assert_called_once_with(["a", "b"], ["x", "y"])
        assert json.loads(result.stdout) == [
            {"prompt": "a", "response": "x", "score": 0.1235},
            {"prompt": "b", "response": "y", "score": 1.1235},
        ]

    def test_long_responses_are_shortened_in_the_table(self, fake_judge, tmp_path):
        long = "word " * 40
        result = runner.invoke(cli.app, ["judge", str(tmp_path), "--prompt", "p", "--response", long])
        assert result.exit_code == 0, result.output
        assert "..." in result.output

    def test_nothing_to_score(self, fake_judge, tmp_path):
        result = runner.invoke(cli.app, ["judge", str(tmp_path)])
        assert result.exit_code == 1
        assert isinstance(result.exception, ConfigError) and "Nothing to score" in str(result.exception)
        fake_judge.load.assert_not_called()

    def test_prompt_without_response_is_nothing_to_score(self, fake_judge, tmp_path):
        result = runner.invoke(cli.app, ["judge", str(tmp_path), "--prompt", "p"])
        assert result.exit_code == 1
        assert "Nothing to score" in str(result.exception)

    def test_pairs_with_prompt_is_rejected(self, fake_judge, tmp_path):
        pairs = _pairs_file(tmp_path, [{"prompt": "a", "response": "x"}])
        result = runner.invoke(cli.app, ["judge", str(tmp_path), "--pairs", str(pairs), "--prompt", "p"])
        assert result.exit_code == 1
        assert isinstance(result.exception, ConfigError) and "not both" in str(result.exception)
        fake_judge.load.assert_not_called()

    def test_missing_pairs_file(self, fake_judge, tmp_path):
        result = runner.invoke(cli.app, ["judge", str(tmp_path), "--pairs", str(tmp_path / "nope.json")])
        assert result.exit_code == 1
        assert "does not exist" in str(result.exception)

    def test_invalid_json_pairs_file(self, fake_judge, tmp_path):
        result = runner.invoke(cli.app, ["judge", str(tmp_path), "--pairs", str(_pairs_file(tmp_path, "{oops"))])
        assert result.exit_code == 1
        assert "not valid JSON" in str(result.exception)

    @pytest.mark.parametrize(
        "data",
        [[], {"prompt": "a"}, [{"prompt": "a"}], [{"prompt": "a", "response": 3}], ["text"]],
    )
    def test_malformed_or_empty_pairs_file(self, fake_judge, tmp_path, data):
        result = runner.invoke(cli.app, ["judge", str(tmp_path), "--pairs", str(_pairs_file(tmp_path, data))])
        assert result.exit_code == 1
        assert "not a list of prompt and response pairs" in str(result.exception)
        fake_judge.load.assert_not_called()
