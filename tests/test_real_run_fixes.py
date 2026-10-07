"""Regression tests for the bugs fixed after the first real training run."""

import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch
from peft import PeftModel
from typer.testing import CliRunner

from aspire.cli import app, tolerate_unencodable_output
from aspire.config import AspireConfig, TeacherConfig
from aspire.critic.head import CriticHead, MultiHeadCriticHead
from aspire.dialogue.generator import DialogueGenerator, GeneratedDialogue
from aspire.dialogue.manager import DialogueManager
from aspire.errors import ConfigError
from aspire.teachers import CompositeTeacher
from aspire.teachers.base import (
    BaseTeacher,
    ChallengeType,
    DialogueHistory,
    DialogueTurn,
    DimensionScore,
    EvaluationDimension,
    TeacherChallenge,
    TeacherEvaluation,
)
from aspire.teachers.local import (
    MAX_INPUT_TOKENS,
    LocalTeacher,
    extract_json_object,
)
from aspire.teachers.registry import TeacherRegistry, get_teacher
from aspire.trainer import AspireTrainer, build_teacher, describe_teacher, load_peft_weights_into

runner = CliRunner()


class _Dummy(BaseTeacher):
    """A teacher that keeps the kwargs it was built with."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        super().__init__(**kwargs)

    async def challenge(self, prompt, student_response, dialogue_history=None, challenge_type=None):
        return None

    async def evaluate(self, prompt, student_response, dialogue_history=None, generate_improved=True):
        return None


# ---------------------------------------------------------------------------
# 1. teacher registry: ``name`` kwarg
# ---------------------------------------------------------------------------


class TestRegistryNameKwarg:
    def test_get_teacher_passes_name_to_constructor(self):
        with patch.dict(TeacherRegistry._teachers, {"dummy": _Dummy}):
            teacher = get_teacher("dummy", name="x", temperature=0.1)
        assert isinstance(teacher, _Dummy)
        assert teacher.name == "x"
        assert teacher.kwargs == {"name": "x", "temperature": 0.1}

    def test_registry_create_passes_name_to_constructor(self):
        with patch.dict(TeacherRegistry._teachers, {"dummy": _Dummy}):
            teacher = TeacherRegistry.create("dummy", name="y")
        assert teacher.name == "y"

    def test_unknown_teacher_still_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown teacher"):
            get_teacher("no-such-teacher", name="x")


# ---------------------------------------------------------------------------
# 2. tolerate_unencodable_output
# ---------------------------------------------------------------------------


class _Stream:
    def __init__(self, encoding, with_reconfigure=True):
        self.encoding = encoding
        if with_reconfigure:
            self.reconfigure = MagicMock()


class TestTolerateUnencodableOutput:
    @pytest.mark.parametrize("encoding", ["cp1252", "ascii", "cp437"])
    def test_non_utf_streams_are_reconfigured(self, monkeypatch, encoding):
        out, err = _Stream(encoding), _Stream(encoding)
        monkeypatch.setattr(sys, "stdout", out)
        monkeypatch.setattr(sys, "stderr", err)
        tolerate_unencodable_output()
        out.reconfigure.assert_called_once_with(errors="replace")
        err.reconfigure.assert_called_once_with(errors="replace")

    @pytest.mark.parametrize("encoding", ["utf-8", "UTF-8", "utf8", "utf-16", "utf-32"])
    def test_utf_streams_are_left_alone(self, monkeypatch, encoding):
        out, err = _Stream(encoding), _Stream(encoding)
        monkeypatch.setattr(sys, "stdout", out)
        monkeypatch.setattr(sys, "stderr", err)
        tolerate_unencodable_output()
        out.reconfigure.assert_not_called()
        err.reconfigure.assert_not_called()

    def test_stream_without_reconfigure_does_not_crash(self, monkeypatch):
        monkeypatch.setattr(sys, "stdout", _Stream("cp1252", with_reconfigure=False))
        monkeypatch.setattr(sys, "stderr", _Stream(None, with_reconfigure=False))
        tolerate_unencodable_output()

    def test_stream_without_encoding_is_reconfigured(self, monkeypatch):
        out = _Stream(None)
        monkeypatch.setattr(sys, "stdout", out)
        monkeypatch.setattr(sys, "stderr", _Stream("utf-8"))
        tolerate_unencodable_output()
        out.reconfigure.assert_called_once_with(errors="replace")

    def test_the_original_failure_and_its_fix(self, monkeypatch):
        raw = io.BytesIO()
        stream = io.TextIOWrapper(raw, encoding="cp1252", write_through=True)
        with pytest.raises(UnicodeEncodeError):
            stream.write("\u2834")
        monkeypatch.setattr(sys, "stdout", stream)
        monkeypatch.setattr(sys, "stderr", io.TextIOWrapper(io.BytesIO(), encoding="cp1252"))
        tolerate_unencodable_output()
        stream.write("\u2834")
        stream.flush()
        assert raw.getvalue().endswith(b"?")


# ---------------------------------------------------------------------------
# 3. LocalTeacher
# ---------------------------------------------------------------------------


class _Batch(dict):
    def to(self, device):
        return self


class _Tok:
    pad_token = "<pad>"
    eos_token = "<eos>"
    pad_token_id = 0

    def __init__(self, chat_template=None):
        self.chat_template = chat_template
        self.calls = []

    def __call__(self, text, **kwargs):
        self.calls.append((text, kwargs, self.truncation_side))
        return _Batch(input_ids=torch.ones(1, 4, dtype=torch.long))

    def decode(self, ids, skip_special_tokens=True):
        return "decoded"


@pytest.fixture
def make_local():
    def make(chat_template=None, context=None):
        with patch("aspire.teachers.local.AutoTokenizer") as tok_cls, patch(
            "aspire.teachers.local.AutoModelForCausalLM"
        ) as model_cls, patch("aspire.teachers.local.BitsAndBytesConfig"):
            tok = _Tok(chat_template)
            tok_cls.from_pretrained.return_value = tok
            model = MagicMock()
            model.generate.return_value = torch.arange(10).unsqueeze(0)
            model.config = SimpleNamespace(**({} if context is None else {"max_position_embeddings": context}))
            model_cls.from_pretrained.return_value = model
            teacher = LocalTeacher("some/model", device="cpu", name="the-local")
        return teacher, tok

    return make


class TestLocalInputLimit:
    def test_context_minus_new_tokens(self, make_local):
        teacher, _ = make_local(context=2048)
        assert teacher._input_limit(256) == 2048 - 256

    def test_capped_at_max_input_tokens(self, make_local):
        teacher, _ = make_local(context=131072)
        assert teacher._input_limit(256) == MAX_INPUT_TOKENS == 4096

    def test_minimum_is_64(self, make_local):
        teacher, _ = make_local(context=100)
        assert teacher._input_limit(768) == 64

    @pytest.mark.parametrize("context", [None, "long", 0, -5])
    def test_falls_back_without_an_int_context(self, make_local, context):
        teacher, _ = make_local()
        teacher.model.config = SimpleNamespace(max_position_embeddings=context)
        assert teacher._input_limit(256) == MAX_INPUT_TOKENS

    def test_falls_back_when_config_missing(self, make_local):
        teacher, _ = make_local()
        teacher.model = SimpleNamespace()
        assert teacher._input_limit(256) == MAX_INPUT_TOKENS


class TestLocalGenerate:
    def test_truncates_from_the_left_with_the_context_limit(self, make_local):
        teacher, tok = make_local(context=1000)
        assert teacher._generate("text", 200, 0.5) == "decoded"
        _, kwargs, side = tok.calls[0]
        assert side == "left"
        assert kwargs["max_length"] == 800
        assert kwargs["truncation"] is True

    def test_chat_template_disables_special_tokens(self, make_local):
        teacher, tok = make_local(chat_template="{{ messages }}")
        teacher._generate("text", 10, 0.5)
        assert tok.calls[0][1]["add_special_tokens"] is False

    def test_no_chat_template_keeps_special_tokens(self, make_local):
        teacher, tok = make_local(chat_template=None)
        teacher._generate("text", 10, 0.5)
        assert tok.calls[0][1]["add_special_tokens"] is True


class TestLocalParseEvaluation:
    BODY = {
        "overall_score": 7.5,
        "dimension_scores": [
            {"dimension": "clarity", "score": 12, "explanation": "clear"},
            {"dimension": "reasoning", "score": -3, "explanation": "weak"},
            {"dimension": "made_up", "score": 5, "explanation": "unknown"},
            {"dimension": "nuance", "score": "n/a", "explanation": "bad score"},
            "not-a-dict",
        ],
        "reasoning": "because",
        "strengths": ["a", 3],
        "weaknesses": ["b"],
        "improved_response": "better",
    }

    def _check(self, ev):
        assert ev.overall_score == 7.5
        assert ev.metadata == {"teacher": "the-local", "parse": "json"}
        scores = {d.dimension: d.score for d in ev.dimension_scores}
        assert scores == {EvaluationDimension.CLARITY: 10.0, EvaluationDimension.REASONING: 0.0}
        assert ev.reasoning == "because"
        assert ev.strengths == ["a"]
        assert ev.improved_response == "better"

    def test_bare_json(self, make_local):
        teacher, _ = make_local()
        self._check(teacher._parse_evaluation(json.dumps(self.BODY), True))

    def test_fenced_json(self, make_local):
        teacher, _ = make_local()
        reply = "Sure!\n```json\n" + json.dumps(self.BODY) + "\n```\nDone."
        self._check(teacher._parse_evaluation(reply, True))

    def test_improved_response_ignored_when_not_requested(self, make_local):
        teacher, _ = make_local()
        assert teacher._parse_evaluation(json.dumps(self.BODY), False).improved_response is None

    def test_free_text_score(self, make_local):
        teacher, _ = make_local()
        ev = teacher._parse_evaluation("Score: 7\nPretty good.", True)
        assert ev.overall_score == 7.0
        assert ev.metadata["parse"] == "text"
        assert ev.metadata["teacher"] == "the-local"

    def test_nothing_gives_the_default(self, make_local):
        teacher, _ = make_local()
        ev = teacher._parse_evaluation("no numbers here", True)
        assert ev.overall_score == 5.0
        assert ev.metadata["parse"] == "default"

    def test_json_without_overall_score_falls_back_to_text(self, make_local):
        teacher, _ = make_local()
        ev = teacher._parse_evaluation('{"reasoning": "x"} Score: 4', True)
        assert ev.overall_score == 4.0
        assert ev.metadata["parse"] == "text"

    @pytest.mark.parametrize("text", ["plain words", "{not json}", "[1, 2]", ""])
    def test_extract_json_object_returns_none(self, text):
        assert extract_json_object(text) is None

    def test_extract_json_object_finds_the_object(self):
        assert extract_json_object('prefix {"a": 1} suffix') == {"a": 1}


# ---------------------------------------------------------------------------
# 4. config round trip
# ---------------------------------------------------------------------------


class TestConfigYamlRoundTrip:
    def test_round_trip_with_paths(self, tmp_path):
        cfg = AspireConfig(device="cpu")
        cfg.training.output_dir = tmp_path / "outs"
        cfg.curriculum.data_dir = tmp_path / "data"
        cfg.teacher.default_teacher = "local"
        path = tmp_path / "cfg.yaml"
        cfg.to_yaml(path)
        assert "!!python" not in path.read_text(encoding="utf-8")
        loaded = AspireConfig.from_yaml(path)
        assert loaded.training.output_dir == tmp_path / "outs"
        assert loaded.curriculum.data_dir == tmp_path / "data"
        assert isinstance(loaded.training.output_dir, Path)
        assert loaded.teacher.default_teacher == "local"
        assert loaded.model_dump() == cfg.model_dump()

    @pytest.mark.parametrize("cuda", [True, False])
    def test_device_default_follows_cuda(self, cuda):
        with patch("torch.cuda.is_available", return_value=cuda):
            assert AspireConfig().device == ("cuda" if cuda else "cpu")

    def test_device_default_matches_this_machine(self):
        assert AspireConfig().device == ("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# 5. critic save / load
# ---------------------------------------------------------------------------


class TestCriticSaveLoad:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"pooling": "last", "num_layers": 3},
            {"pooling": "attention", "num_layers": 1, "num_dimensions": 3},
        ],
    )
    def test_critic_head_round_trip(self, tmp_path, kwargs):
        critic = CriticHead(input_dim=8, hidden_dim=16, reasoning_dim=12, **kwargs).eval()
        path = str(tmp_path / "critic.pt")
        critic.save(path)
        loaded = CriticHead.load(path).eval()
        assert loaded.init_config() == critic.init_config()
        hidden = torch.randn(2, 5, 8)
        mask = torch.ones(2, 5, dtype=torch.long)
        with torch.no_grad():
            expected = critic(hidden_states=hidden, attention_mask=mask)
            actual = loaded(hidden_states=hidden, attention_mask=mask)
        assert torch.allclose(expected.score, actual.score)
        assert torch.allclose(expected.reasoning_embedding, actual.reasoning_embedding)

    def test_multi_head_round_trip(self, tmp_path):
        critic = MultiHeadCriticHead(
            input_dim=8, hidden_dim=16, num_layers=1, reasoning_dim=12, num_heads=2
        ).eval()
        path = str(tmp_path / "multi.pt")
        critic.save(path)
        loaded = MultiHeadCriticHead.load(path).eval()
        assert isinstance(loaded, MultiHeadCriticHead)
        assert loaded.num_heads == 2
        hidden = torch.randn(2, 5, 8)
        with torch.no_grad():
            assert torch.allclose(critic(hidden_states=hidden).score, loaded(hidden_states=hidden).score)

    def test_load_kwargs_override_config(self, tmp_path):
        path = str(tmp_path / "critic.pt")
        CriticHead(input_dim=8, hidden_dim=16, reasoning_dim=12).save(path)
        assert CriticHead.load(path, dropout=0.5).dropout == 0.5


# ---------------------------------------------------------------------------
# 6. dialogue cache
# ---------------------------------------------------------------------------


def _evaluation(**overrides):
    base = dict(
        overall_score=8.0,
        dimension_scores=[
            DimensionScore(EvaluationDimension.CLARITY, 9.0, "clear"),
            DimensionScore(EvaluationDimension.NUANCE, 6.5, "ok"),
        ],
        reasoning="r",
        improved_response="imp",
        strengths=["s1"],
        weaknesses=["w1"],
        suggestions=["g1"],
        metadata={"teacher_scores": {"a": 7.0, "b": 9.0}},
    )
    base.update(overrides)
    return TeacherEvaluation(**base)


def _history(replies):
    history = DialogueHistory(prompt="p", initial_response="init")
    for i, reply in enumerate(replies, 1):
        history.add_turn(DialogueTurn(i, TeacherChallenge(ChallengeType.SOCRATIC, f"q{i}"), reply))
    return history


@pytest.fixture
def manager(tmp_path):
    generator = MagicMock()
    generator.teacher.name = "t"
    return DialogueManager(generator, cache_dir=tmp_path / "cache")


class TestDialogueCacheRoundTrip:
    def test_final_evaluation_survives_the_cache(self, manager):
        dialogue = GeneratedDialogue(
            prompt="p",
            initial_response="init",
            history=_history(["first", "second"]),
            final_evaluation=_evaluation(),
            turn_evaluations=[],
            metadata={"num_turns": 2},
            final_response="second",
        )
        manager._save_to_cache(dialogue)
        loaded = manager._load_from_cache("p")
        ev = loaded.final_evaluation
        assert [(d.dimension, d.score, d.explanation) for d in ev.dimension_scores] == [
            (EvaluationDimension.CLARITY, 9.0, "clear"),
            (EvaluationDimension.NUANCE, 6.5, "ok"),
        ]
        assert ev.strengths == ["s1"] and ev.weaknesses == ["w1"] and ev.suggestions == ["g1"]
        assert ev.metadata == {"teacher_scores": {"a": 7.0, "b": 9.0}}
        assert ev.improved_response == "imp"
        assert loaded.scored_response == "second"
        assert loaded.metadata == {"num_turns": 2}

    def test_scored_response_is_last_turn_when_not_set(self, manager):
        dialogue = GeneratedDialogue("p", "init", _history(["a", "b"]), _evaluation(), [], {})
        manager._save_to_cache(dialogue)
        assert manager._load_from_cache("p").scored_response == "b"

    def _write(self, manager, data):
        manager._get_cache_path("p").write_text(json.dumps(data), encoding="utf-8")

    def _old(self, **extra):
        data = {
            "prompt": "p",
            "initial_response": "init",
            "final_evaluation": _evaluation().to_dict(),
            "turns": [{"challenge": "q1", "response": "r1"}, {"challenge": "q2", "response": "r2"}],
        }
        data.update(extra)
        return data

    def test_old_format_falls_back_to_last_turn_response(self, manager):
        self._write(manager, self._old())
        assert manager._load_from_cache("p").scored_response == "r2"

    def test_old_format_without_turns_falls_back_to_initial_response(self, manager):
        self._write(manager, self._old(turns=[]))
        assert manager._load_from_cache("p").scored_response == "init"

    def test_corrupt_dimension_entries_are_skipped(self, manager):
        data = self._old()
        data["final_evaluation"]["dimension_scores"] = [
            {"dimension": "clarity", "score": 4, "explanation": "ok"},
            {"dimension": "bogus", "score": 4},
            {"dimension": "nuance", "score": "x"},
            {"score": 3},
            None,
        ]
        self._write(manager, data)
        ev = manager._load_from_cache("p").final_evaluation
        assert [(d.dimension, d.score) for d in ev.dimension_scores] == [(EvaluationDimension.CLARITY, 4.0)]

    def test_iterate_cached_reads_the_new_format(self, manager):
        manager._save_to_cache(GeneratedDialogue("p", "init", _history(["a"]), _evaluation(), [], {}, "a"))
        assert [d.scored_response for d in manager.iterate_cached()] == ["a"]


# ---------------------------------------------------------------------------
# 7. dialogue generator
# ---------------------------------------------------------------------------


class _ChatTok:
    pad_token = "<pad>"
    eos_token = "<eos>"
    pad_token_id = 0
    eos_token_id = 1

    def __init__(self, chat_template):
        self.chat_template = chat_template
        self.chat_calls = []
        self.calls = []

    def apply_chat_template(self, chat, **kwargs):
        self.chat_calls.append((chat, kwargs))
        return "CHAT"

    def __call__(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return _Batch(input_ids=torch.ones(1, 3, dtype=torch.long))

    def decode(self, ids, skip_special_tokens=True):
        return "  reply  "


def _generator(tokenizer, teacher=None, max_turns=2):
    model = MagicMock()
    model.generate.return_value = torch.arange(8).unsqueeze(0)
    return DialogueGenerator(
        student_model=model,
        student_tokenizer=tokenizer,
        teacher=teacher or MagicMock(),
        max_turns=max_turns,
        device="cpu",
    )


class TestScoredResponse:
    def _dialogue(self, **kwargs):
        return GeneratedDialogue("p", "init", kwargs.pop("history", _history([])), _evaluation(), [], {}, **kwargs)

    def test_explicit_final_response_wins(self):
        assert self._dialogue(history=_history(["a"]), final_response="x").scored_response == "x"

    def test_last_turn_reply(self):
        assert self._dialogue(history=_history(["a", "b"])).scored_response == "b"

    def test_initial_response_when_no_turns(self):
        assert self._dialogue().scored_response == "init"


class TestGenerateDialogueFinalResponse:
    async def test_final_response_is_the_last_reply(self):
        teacher = MagicMock()
        teacher.name = "t"
        teacher.challenge = AsyncMock(return_value=TeacherChallenge(ChallengeType.SOCRATIC, "why?"))
        teacher.evaluate = AsyncMock(return_value=_evaluation())
        generator = _generator(_ChatTok(None), teacher, max_turns=2)
        replies = iter(["initial", "reply one", "reply two"])
        with patch.object(generator, "_generate_student_response", side_effect=lambda *a, **k: next(replies)):
            dialogue = await generator.generate_dialogue("p")
        assert dialogue.final_response == "reply two"
        assert dialogue.scored_response == "reply two"
        assert dialogue.initial_response == "initial"
        # the final evaluation judged the last reply
        assert teacher.evaluate.await_args_list[-1].kwargs["student_response"] == "reply two"

    async def test_final_response_is_initial_without_turns(self):
        teacher = MagicMock()
        teacher.name = "t"
        teacher.evaluate = AsyncMock(return_value=_evaluation())
        generator = _generator(_ChatTok(None), teacher, max_turns=0)
        dialogue = await generator.generate_dialogue("p", initial_response="given")
        assert dialogue.final_response == "given"


class TestStudentInputFormat:
    def test_chat_template_layout(self):
        tok = _ChatTok("{{ x }}")
        generator = _generator(tok)
        history = _history(["r1", "r2"])
        assert generator._format_student_input("the prompt", "latest?", history) == "CHAT"
        chat, kwargs = tok.chat_calls[0]
        assert kwargs == {"tokenize": False, "add_generation_prompt": True}
        assert [m["role"] for m in chat] == [
            "system", "user", "assistant", "user", "assistant", "user", "assistant", "user",
        ]
        assert chat[1]["content"] == "the prompt"
        assert [m["content"] for m in chat[2:]] == ["init", "q1", "r1", "q2", "r2", "latest?"]

    def test_chat_template_first_prompt_has_no_challenge(self):
        tok = _ChatTok("{{ x }}")
        _generator(tok)._format_student_input("the prompt")
        chat, kwargs = tok.chat_calls[0]
        assert [m["role"] for m in chat] == ["system", "user"]
        assert kwargs["add_generation_prompt"] is True

    def test_chat_template_with_challenge_but_no_history(self):
        tok = _ChatTok("{{ x }}")
        _generator(tok)._format_student_input("the prompt", "ch", None)
        chat, _ = tok.chat_calls[0]
        assert [m["role"] for m in chat] == ["system", "user", "user"]

    @pytest.mark.parametrize("template", [None, 5])
    def test_plain_text_without_a_str_template(self, template):
        tok = _ChatTok(template)
        text = _generator(tok)._format_student_input("the prompt", "latest?", _history(["r1"]))
        assert tok.chat_calls == []
        assert "Task: the prompt" in text
        assert "Your initial response: init" in text
        assert "Challenge: q1" in text and "Your response: r1" in text
        assert text.endswith("Challenge: latest?\n\nYour response:")

    def test_plain_text_first_prompt(self):
        text = _generator(_ChatTok(None))._format_student_input("the prompt")
        assert text.endswith("Task: the prompt")

    @pytest.mark.parametrize("template,expected", [("{{ x }}", False), (None, True)])
    def test_generate_passes_add_special_tokens(self, template, expected):
        tok = _ChatTok(template)
        reply = _generator(tok)._generate_student_response("p")
        assert reply == "reply"
        assert tok.calls[0][1]["add_special_tokens"] is expected


# ---------------------------------------------------------------------------
# 8. trainer: build_teacher, describe_teacher, load_checkpoint
# ---------------------------------------------------------------------------


def _cfg(**kwargs):
    return TeacherConfig(claude_model="claude-x", openai_model="gpt-x", **kwargs)


def _named(name):
    return _Dummy(name=name)


class TestBuildTeacher:
    def test_local_uses_configured_model_and_names_it(self):
        cfg = _cfg(default_teacher="local", local_model_path="C:\\models\\Org\\Tiny", local_load_in_4bit=True)
        with patch("aspire.trainer.get_teacher") as get:
            build_teacher(cfg, device="cpu")
        args, kwargs = get.call_args
        assert args == ("local",)
        assert kwargs["model_name_or_path"] == "C:\\models\\Org\\Tiny"
        assert kwargs["name"] == "local:Tiny"
        assert kwargs["load_in_4bit"] is True
        assert kwargs["device"] == "cpu"
        assert kwargs["temperature"] == 0.7 and kwargs["max_tokens"] == 1024

    def test_local_without_model_is_a_config_error(self):
        with patch("aspire.trainer.get_teacher") as get, pytest.raises(ConfigError, match="no model"):
            build_teacher(_cfg(default_teacher="local"))
        get.assert_not_called()

    def test_local_colon_parses_the_model(self):
        with patch("aspire.trainer.get_teacher") as get:
            build_teacher(_cfg(), "local:Org/Model")
        assert get.call_args.kwargs["model_name_or_path"] == "Org/Model"
        assert get.call_args.kwargs["name"] == "local:Model"

    def test_local_colon_with_blank_model_is_a_config_error(self):
        with pytest.raises(ConfigError):
            build_teacher(_cfg(local_model_path="ignored"), "local:  ")

    def test_composite_builds_members_with_the_strategy(self):
        cfg = _cfg(
            default_teacher="composite",
            composite_members=["local:Org/A", "claude"],
            composite_strategy="rotate",
        )

        def fake(spec, **kwargs):
            return _named(kwargs.get("name", spec))

        with patch("aspire.trainer.get_teacher", side_effect=fake):
            teacher = build_teacher(cfg)
        assert isinstance(teacher, CompositeTeacher)
        assert teacher.strategy == "rotate"
        assert [t.name for t in teacher.teachers] == ["local:A", "claude"]

    def test_composite_without_members(self):
        with pytest.raises(ConfigError, match="no members"):
            build_teacher(_cfg(default_teacher="composite"))

    def test_composite_members_must_have_distinct_names(self):
        cfg = _cfg(default_teacher="composite", composite_members=["claude", "claude"])
        with patch("aspire.trainer.get_teacher", side_effect=lambda spec, **k: _named(spec)):
            with pytest.raises(ConfigError, match="same name"):
                build_teacher(cfg)

    @pytest.mark.parametrize("spec", ["openai", "gpt4", "OpenAI"])
    def test_openai_teachers_get_the_openai_model(self, spec):
        with patch("aspire.trainer.get_teacher") as get:
            build_teacher(_cfg(), spec)
        assert get.call_args.args == (spec,)
        assert get.call_args.kwargs["model"] == "gpt-x"

    @pytest.mark.parametrize("spec", ["claude", "socratic", "adversarial", "guide"])
    def test_claude_and_personas_get_the_claude_model(self, spec):
        with patch("aspire.trainer.get_teacher") as get:
            build_teacher(_cfg(), spec)
        assert get.call_args.kwargs["model"] == "claude-x"

    def test_spec_defaults_to_default_teacher(self):
        with patch("aspire.trainer.get_teacher") as get:
            build_teacher(_cfg(default_teacher="  gpt4 "))
        assert get.call_args.args == ("gpt4",)
        assert get.call_args.kwargs["model"] == "gpt-x"


class TestDescribeTeacher:
    def test_composite(self):
        composite = CompositeTeacher(teachers=[_named("a"), _named("b")], strategy="vote")
        assert describe_teacher(composite) == "composite (vote): a + b"

    def test_single(self):
        assert describe_teacher(_named("solo")) == "solo"

    def test_without_a_name(self):
        assert describe_teacher(object()) == "teacher"


class TestLoadCheckpointPeft:
    def _trainer(self, student):
        trainer = object.__new__(AspireTrainer)
        trainer.student_model = student
        trainer.device = "cpu"
        critic = MagicMock()
        critic.__class__ = MagicMock()
        critic.__class__.load = MagicMock(return_value=critic)
        critic.to.return_value = critic
        trainer.critic = critic
        return trainer

    def test_peft_student_is_loaded_into_not_wrapped_again(self, tmp_path):
        student = MagicMock(spec=PeftModel)
        trainer = self._trainer(student)
        with patch("aspire.trainer.load_peft_weights_into") as load_into, patch(
            "aspire.trainer.PeftModel.from_pretrained"
        ) as wrap, patch("aspire.trainer.console"):
            trainer.load_checkpoint(tmp_path)
        load_into.assert_called_once_with(student, tmp_path / "student")
        wrap.assert_not_called()
        assert trainer.student_model is student

    def test_plain_student_is_wrapped(self, tmp_path):
        trainer = self._trainer(MagicMock())
        with patch("aspire.trainer.load_peft_weights_into") as load_into, patch(
            "aspire.trainer.PeftModel.from_pretrained"
        ) as wrap, patch("aspire.trainer.console"):
            trainer.load_checkpoint(tmp_path)
        load_into.assert_not_called()
        wrap.assert_called_once()
        assert trainer.student_model is wrap.return_value


class TestLoadPeftWeightsInto:
    def test_matching_weights_load(self, tmp_path):
        result = SimpleNamespace(missing_keys=["base.weight"], unexpected_keys=[])
        with patch("peft.load_peft_weights", return_value={"w": 1}) as load, patch(
            "peft.set_peft_model_state_dict", return_value=result
        ) as setter:
            model = MagicMock()
            load_peft_weights_into(model, tmp_path)
        load.assert_called_once_with(str(tmp_path))
        setter.assert_called_once_with(model, {"w": 1})

    def test_missing_lora_keys_raise(self, tmp_path):
        result = SimpleNamespace(missing_keys=["layer.lora_A.default.weight", "layer.lora_B.default.weight"])
        with patch("peft.load_peft_weights", return_value={}), patch(
            "peft.set_peft_model_state_dict", return_value=result
        ):
            with pytest.raises(ConfigError, match="2 weights missing"):
                load_peft_weights_into(MagicMock(), tmp_path)

    def test_result_without_missing_keys_is_fine(self, tmp_path):
        with patch("peft.load_peft_weights", return_value={}), patch(
            "peft.set_peft_model_state_dict", return_value=None
        ):
            load_peft_weights_into(MagicMock(), tmp_path)


# ---------------------------------------------------------------------------
# 9. CLI train: config values vs flags
# ---------------------------------------------------------------------------


def _train(tmp_path, args, **config_edits):
    cfg = AspireConfig(device="cpu")
    cfg.teacher.default_teacher = "socratic"
    cfg.training.num_epochs = 7
    cfg.training.output_dir = tmp_path / "from-config"
    for key, value in config_edits.items():
        section, attr = key.split("__")
        setattr(getattr(cfg, section), attr, value)
    path = tmp_path / "cfg.yaml"
    cfg.to_yaml(path)
    prompts = tmp_path / "prompts.json"
    prompts.write_text(json.dumps(["one prompt"]), encoding="utf-8")
    with patch("aspire.trainer.AspireTrainer") as trainer_cls:
        result = runner.invoke(app, ["train", "--config", str(path), "--prompts", str(prompts), *args])
    assert result.exit_code == 0, result.output
    trainer_cls.return_value.train.assert_called_once_with(["one prompt"])
    return trainer_cls.call_args.args[0]


class TestTrainConfigPrecedence:
    def test_config_values_are_kept_without_flags(self, tmp_path):
        cfg = _train(tmp_path, [])
        assert cfg.teacher.default_teacher == "socratic"
        assert cfg.training.num_epochs == 7
        assert cfg.training.output_dir == tmp_path / "from-config"

    def test_flags_override_the_config(self, tmp_path):
        cfg = _train(tmp_path, ["--teacher", "openai", "--epochs", "2", "--output", str(tmp_path / "flag-out")])
        assert cfg.teacher.default_teacher == "openai"
        assert cfg.training.num_epochs == 2
        assert cfg.training.output_dir == tmp_path / "flag-out"

    def test_model_flags(self, tmp_path):
        cfg = _train(tmp_path, ["--teacher", "local", "--teacher-model", "Org/Teach", "--student-model", "Org/Stud"])
        assert cfg.teacher.default_teacher == "local"
        assert cfg.teacher.local_model_path == "Org/Teach"
        assert cfg.student.model_name_or_path == "Org/Stud"

    def test_model_flags_absent_keep_config(self, tmp_path):
        cfg = _train(tmp_path, [], teacher__local_model_path="Org/FromConfig")
        assert cfg.teacher.local_model_path == "Org/FromConfig"

    def test_without_config_the_defaults_apply(self):
        with patch("aspire.trainer.AspireTrainer") as trainer_cls:
            result = runner.invoke(app, ["train"])
        assert result.exit_code == 0, result.output
        cfg = trainer_cls.call_args.args[0]
        assert cfg.teacher.default_teacher == "claude"
        assert cfg.training.num_epochs == 3
        assert cfg.training.output_dir == Path("outputs")
