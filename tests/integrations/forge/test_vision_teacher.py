"""
Tests for the Forge vision-language teachers.

No network and no real PIL images are needed: the anthropic/openai clients are
replaced with AsyncMock-backed fakes and images are tiny stand-ins that
implement ``save``. If Pillow is not installed (it is an optional ``forge``
extra), a minimal ``PIL.Image`` stub is registered only while the module under
test is imported, so the real package is never shadowed.
"""

import base64
import importlib
import importlib.util
import json
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


def _load_vision_teacher():
    """Import vision_teacher, stubbing PIL for the import only if it is missing."""
    if importlib.util.find_spec("PIL") is not None:
        return importlib.import_module("integrations.forge.vision_teacher")

    pil = types.ModuleType("PIL")
    image_mod = types.ModuleType("PIL.Image")
    image_mod.Image = type("Image", (), {})
    pil.Image = image_mod
    saved = {name: sys.modules.get(name) for name in ("PIL", "PIL.Image")}
    sys.modules["PIL"] = pil
    sys.modules["PIL.Image"] = image_mod
    try:
        return importlib.import_module("integrations.forge.vision_teacher")
    finally:
        for name, mod in saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod


vt = _load_vision_teacher()

PNG_BYTES = b"\x89PNG-fake-bytes"


class FakeImage:
    """Stand-in for a PIL image; writes known bytes and records the format."""

    def __init__(self):
        self.saved_format = None

    def save(self, buffer, format=None):
        self.saved_format = format
        buffer.write(PNG_BYTES)


FULL_PAYLOAD = {
    "overall_score": 7.5,
    "aesthetic_score": 7.0,
    "composition_score": 8.0,
    "color_score": 6.5,
    "lighting_score": 5.5,
    "style_score": 6.0,
    "prompt_adherence_score": 9.0,
    "technical_score": 4.5,
    "reasoning": "Solid overall",
    "strengths": ["palette"],
    "weaknesses": ["noise"],
    "suggestions": ["denoise"],
    "improved_description": "A cleaner version",
}

FENCE = "`" * 3


def _claude_response(text):
    return SimpleNamespace(content=[SimpleNamespace(text=text)])


def _gpt_response(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


@pytest.fixture
def claude_client(monkeypatch):
    """Patch anthropic.AsyncAnthropic inside the module; return the fake client."""
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=_claude_response(json.dumps(FULL_PAYLOAD)))
    factory = MagicMock(return_value=client)
    monkeypatch.setattr(vt, "anthropic", SimpleNamespace(AsyncAnthropic=factory))
    client.factory = factory
    return client


@pytest.fixture
def gpt_client(monkeypatch):
    """Patch openai.AsyncOpenAI inside the module; return the fake client."""
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=_gpt_response(json.dumps(FULL_PAYLOAD)))
    factory = MagicMock(return_value=client)
    monkeypatch.setattr(vt, "openai", SimpleNamespace(AsyncOpenAI=factory))
    client.factory = factory
    return client


class TestImageCritique:
    """Tests for the ImageCritique dataclass."""

    def test_defaults(self):
        critique = vt.ImageCritique(overall_score=6.0)

        assert critique.overall_score == 6.0
        assert critique.aesthetic_score == 0.0
        assert critique.technical_score == 0.0
        assert critique.reasoning == ""
        assert critique.strengths == []
        assert critique.improved_description is None

    def test_list_defaults_are_not_shared(self):
        a = vt.ImageCritique(overall_score=1.0)
        b = vt.ImageCritique(overall_score=2.0)
        a.strengths.append("x")

        assert b.strengths == []

    def test_to_dict_structure(self):
        critique = vt.ImageCritique(
            overall_score=7.5,
            aesthetic_score=1.0,
            composition_score=2.0,
            color_score=3.0,
            lighting_score=4.0,
            style_score=5.0,
            prompt_adherence_score=6.0,
            technical_score=7.0,
            reasoning="why",
            strengths=["s"],
            weaknesses=["w"],
            suggestions=["g"],
            improved_description="better",
        )

        assert critique.to_dict() == {
            "overall_score": 7.5,
            "scores": {
                "aesthetic": 1.0,
                "composition": 2.0,
                "color": 3.0,
                "lighting": 4.0,
                "style": 5.0,
                "prompt_adherence": 6.0,
                "technical": 7.0,
            },
            "reasoning": "why",
            "strengths": ["s"],
            "weaknesses": ["w"],
            "suggestions": ["g"],
            "improved_description": "better",
        }


class _Concrete(vt.BaseVisionTeacher):
    async def critique(self, image, prompt, context=None):
        return vt.ImageCritique(overall_score=1.0)


class TestBaseVisionTeacher:
    """Tests for BaseVisionTeacher helpers."""

    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError, match="abstract"):
            vt.BaseVisionTeacher()

    def test_default_name_and_persona(self):
        teacher = _Concrete()

        assert teacher.name == "Vision Teacher"
        assert teacher.persona == "balanced"

    async def test_abstract_critique_body_returns_none(self):
        result = await vt.BaseVisionTeacher.critique(_Concrete(), FakeImage(), "p")

        assert result is None

    async def test_concrete_critique_runs(self):
        critique = await _Concrete().critique(FakeImage(), "p")

        assert critique.overall_score == 1.0

    @pytest.mark.parametrize(
        ("persona", "marker"),
        [
            ("balanced", "art critic"),
            ("technical", "technical image quality analyst"),
            ("artistic", "artistic visionary"),
            ("composition", "composition and design expert"),
            ("color", "color theory specialist"),
            ("harsh", "demanding critic"),
            ("encouraging", "supportive mentor"),
        ],
    )
    def test_system_prompt_per_persona(self, persona, marker):
        assert marker in _Concrete(persona=persona)._get_system_prompt()

    def test_system_prompts_are_distinct(self):
        personas = ["balanced", "technical", "artistic", "composition", "color", "harsh", "encouraging"]
        prompts = {_Concrete(persona=p)._get_system_prompt() for p in personas}

        assert len(prompts) == len(personas)

    def test_unknown_persona_falls_back_to_balanced(self):
        unknown = _Concrete(persona="nonexistent")._get_system_prompt()

        assert unknown == _Concrete(persona="balanced")._get_system_prompt()

    def test_encode_image_is_base64_png(self):
        image = FakeImage()
        encoded = _Concrete()._encode_image(image)

        assert image.saved_format == "PNG"
        assert base64.b64decode(encoded) == PNG_BYTES
        assert isinstance(encoded, str)


class TestClaudeVisionTeacher:
    """Tests for ClaudeVisionTeacher."""

    def test_init(self, claude_client):
        teacher = vt.ClaudeVisionTeacher(api_key="sk-test", persona="harsh")

        assert teacher.name == "Claude Vision"
        assert teacher.model == "claude-sonnet-4-20250514"
        assert teacher.persona == "harsh"
        assert teacher.client is claude_client
        claude_client.factory.assert_called_once_with(api_key="sk-test")

    def test_init_custom_model(self, claude_client):
        assert vt.ClaudeVisionTeacher(model="claude-x").model == "claude-x"

    async def test_critique_parses_full_payload(self, claude_client):
        teacher = vt.ClaudeVisionTeacher()
        critique = await teacher.critique(FakeImage(), "a red fox")

        assert critique.overall_score == 7.5
        assert critique.aesthetic_score == 7.0
        assert critique.composition_score == 8.0
        assert critique.color_score == 6.5
        assert critique.lighting_score == 5.5
        assert critique.style_score == 6.0
        assert critique.prompt_adherence_score == 9.0
        assert critique.technical_score == 4.5
        assert critique.reasoning == "Solid overall"
        assert critique.strengths == ["palette"]
        assert critique.weaknesses == ["noise"]
        assert critique.suggestions == ["denoise"]
        assert critique.improved_description == "A cleaner version"

    async def test_request_shape(self, claude_client):
        teacher = vt.ClaudeVisionTeacher(model="claude-x", persona="technical")
        await teacher.critique(FakeImage(), "a red fox", context="for a poster")

        kwargs = claude_client.messages.create.call_args.kwargs
        assert kwargs["model"] == "claude-x"
        assert kwargs["max_tokens"] == 2048
        assert "technical image quality analyst" in kwargs["system"]

        (message,) = kwargs["messages"]
        assert message["role"] == "user"
        image_block, text_block = message["content"]
        assert image_block["type"] == "image"
        assert image_block["source"]["type"] == "base64"
        assert image_block["source"]["media_type"] == "image/png"
        assert base64.b64decode(image_block["source"]["data"]) == PNG_BYTES
        assert text_block["type"] == "text"
        assert "**Original Prompt:** a red fox" in text_block["text"]
        assert "**Additional Context:** for a poster" in text_block["text"]

    async def test_request_without_context_omits_context_line(self, claude_client):
        await vt.ClaudeVisionTeacher().critique(FakeImage(), "a red fox")

        text = claude_client.messages.create.call_args.kwargs["messages"][0]["content"][1]["text"]
        assert "Additional Context" not in text

    async def test_parses_json_code_fence(self, claude_client):
        body = f"Here you go:\n{FENCE}json\n" + json.dumps({"overall_score": 3.0}) + f"\n{FENCE}\nDone."
        claude_client.messages.create.return_value = _claude_response(body)

        critique = await vt.ClaudeVisionTeacher().critique(FakeImage(), "p")

        assert critique.overall_score == 3.0

    async def test_parses_bare_code_fence(self, claude_client):
        body = f"{FENCE}\n" + json.dumps({"overall_score": 4.0, "reasoning": "r"}) + f"\n{FENCE}"
        claude_client.messages.create.return_value = _claude_response(body)

        critique = await vt.ClaudeVisionTeacher().critique(FakeImage(), "p")

        assert critique.overall_score == 4.0
        assert critique.reasoning == "r"

    async def test_missing_optional_fields_default(self, claude_client):
        claude_client.messages.create.return_value = _claude_response('{"overall_score": 8}')

        critique = await vt.ClaudeVisionTeacher().critique(FakeImage(), "p")

        assert critique.overall_score == 8
        assert critique.aesthetic_score == 0
        assert critique.reasoning == ""
        assert critique.strengths == []
        assert critique.improved_description is None

    async def test_invalid_json_falls_back_to_neutral(self, claude_client):
        claude_client.messages.create.return_value = _claude_response("not json at all")

        critique = await vt.ClaudeVisionTeacher().critique(FakeImage(), "p")

        assert critique.overall_score == 5.0
        assert critique.reasoning == "not json at all"
        assert critique.aesthetic_score == 0.0

    async def test_missing_overall_score_falls_back(self, claude_client):
        text = '{"aesthetic_score": 9}'
        claude_client.messages.create.return_value = _claude_response(text)

        critique = await vt.ClaudeVisionTeacher().critique(FakeImage(), "p")

        assert critique.overall_score == 5.0
        assert critique.reasoning == text
        assert critique.aesthetic_score == 0.0

    async def test_non_object_json_falls_back(self, claude_client):
        claude_client.messages.create.return_value = _claude_response("[1, 2, 3]")

        critique = await vt.ClaudeVisionTeacher().critique(FakeImage(), "p")

        assert critique.overall_score == 5.0

    async def test_api_error_propagates(self, claude_client):
        claude_client.messages.create.side_effect = RuntimeError("rate limited")

        with pytest.raises(RuntimeError, match="rate limited"):
            await vt.ClaudeVisionTeacher().critique(FakeImage(), "p")


class TestGPT4VisionTeacher:
    """Tests for GPT4VisionTeacher."""

    def test_init(self, gpt_client):
        teacher = vt.GPT4VisionTeacher(api_key="sk-oa", persona="encouraging")

        assert teacher.name == "GPT-4 Vision"
        assert teacher.model == "gpt-4o"
        assert teacher.persona == "encouraging"
        assert teacher.client is gpt_client
        gpt_client.factory.assert_called_once_with(api_key="sk-oa")

    async def test_critique_parses_full_payload(self, gpt_client):
        critique = await vt.GPT4VisionTeacher().critique(FakeImage(), "a red fox")

        assert critique.overall_score == 7.5
        assert critique.technical_score == 4.5
        assert critique.strengths == ["palette"]
        assert critique.improved_description == "A cleaner version"

    async def test_request_shape(self, gpt_client):
        teacher = vt.GPT4VisionTeacher(model="gpt-x", persona="color")
        await teacher.critique(FakeImage(), "a red fox", context="for a poster")

        kwargs = gpt_client.chat.completions.create.call_args.kwargs
        assert kwargs["model"] == "gpt-x"
        assert kwargs["max_tokens"] == 2048
        assert kwargs["response_format"] == {"type": "json_object"}

        system, user = kwargs["messages"]
        assert system["role"] == "system"
        assert "color theory specialist" in system["content"]
        image_part, text_part = user["content"]
        url = image_part["image_url"]["url"]
        assert url.startswith("data:image/png;base64,")
        assert base64.b64decode(url.split(",", 1)[1]) == PNG_BYTES
        assert "**Original Prompt:** a red fox" in text_part["text"]
        assert "**Additional Context:** for a poster" in text_part["text"]

    async def test_request_without_context(self, gpt_client):
        await vt.GPT4VisionTeacher().critique(FakeImage(), "a red fox")

        kwargs = gpt_client.chat.completions.create.call_args.kwargs
        assert "Additional Context" not in kwargs["messages"][1]["content"][1]["text"]

    async def test_invalid_json_falls_back(self, gpt_client):
        gpt_client.chat.completions.create.return_value = _gpt_response("oops")

        critique = await vt.GPT4VisionTeacher().critique(FakeImage(), "p")

        assert critique.overall_score == 5.0
        assert critique.reasoning == "oops"

    async def test_missing_overall_score_falls_back(self, gpt_client):
        text = '{"reasoning": "no score"}'
        gpt_client.chat.completions.create.return_value = _gpt_response(text)

        critique = await vt.GPT4VisionTeacher().critique(FakeImage(), "p")

        assert critique.overall_score == 5.0
        assert critique.reasoning == text

    async def test_missing_optional_fields_default(self, gpt_client):
        gpt_client.chat.completions.create.return_value = _gpt_response('{"overall_score": 2}')

        critique = await vt.GPT4VisionTeacher().critique(FakeImage(), "p")

        assert critique.overall_score == 2
        assert critique.suggestions == []
        assert critique.prompt_adherence_score == 0


class TestGetVisionTeacher:
    """Tests for the get_vision_teacher factory."""

    def test_claude(self, claude_client):
        teacher = vt.get_vision_teacher("claude")

        assert isinstance(teacher, vt.ClaudeVisionTeacher)
        assert teacher.persona == "balanced"

    def test_default_is_claude(self, claude_client):
        assert isinstance(vt.get_vision_teacher(), vt.ClaudeVisionTeacher)

    @pytest.mark.parametrize("name", ["gpt4v", "gpt4", "GPT4V", "Gpt4"])
    def test_gpt_aliases_case_insensitive(self, gpt_client, name):
        assert isinstance(vt.get_vision_teacher(name), vt.GPT4VisionTeacher)

    def test_persona_and_kwargs_forwarded(self, claude_client):
        teacher = vt.get_vision_teacher("claude", persona="harsh", model="claude-y", api_key="k")

        assert teacher.persona == "harsh"
        assert teacher.model == "claude-y"
        claude_client.factory.assert_called_once_with(api_key="k")

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown teacher type: llava"):
            vt.get_vision_teacher("llava")
