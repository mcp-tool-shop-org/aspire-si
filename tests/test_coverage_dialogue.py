"""Coverage-closing tests for aspire.dialogue (manager, formatter)."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from aspire.dialogue.formatter import DialogueFormatter
from aspire.dialogue.generator import GeneratedDialogue
from aspire.dialogue.manager import DialogueManager
from aspire.teachers.base import (
    ChallengeType,
    DialogueHistory,
    DialogueTurn,
    TeacherChallenge,
    TeacherEvaluation,
)

# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------


def make_dialogue(
    prompt="What is 2+2?",
    initial="four?",
    turns=(),
    improved="Four, because 2+2=4.",
    score=7.0,
    turn_eval_reasoning=None,
):
    history = DialogueHistory(prompt=prompt, initial_response=initial)
    for i, (challenge, reply) in enumerate(turns, start=1):
        ev = (
            TeacherEvaluation(overall_score=6.0, dimension_scores=[], reasoning=turn_eval_reasoning)
            if turn_eval_reasoning
            else None
        )
        history.add_turn(
            DialogueTurn(i, TeacherChallenge(ChallengeType.PROBE_REASONING, challenge), reply, evaluation=ev)
        )
    final = TeacherEvaluation(
        overall_score=score, dimension_scores=[], reasoning="final reasoning", improved_response=improved
    )
    return GeneratedDialogue(
        prompt=prompt,
        initial_response=initial,
        history=history,
        final_evaluation=final,
        turn_evaluations=[],
        metadata={"k": "v"},
    )


@pytest.fixture
def generator():
    gen = SimpleNamespace(teacher=SimpleNamespace(name="T1"))
    gen.generate_dialogue = AsyncMock(side_effect=lambda prompt: make_dialogue(prompt=prompt))
    gen.generate_batch = AsyncMock(
        side_effect=lambda prompts, max_concurrent=5: [make_dialogue(prompt=p) for p in prompts]
    )
    return gen


# ---------------------------------------------------------------------------
# DialogueManager
# ---------------------------------------------------------------------------


class TestManagerCacheDisabled:
    def test_cache_dir_not_created_when_disabled(self, generator, tmp_path):
        target = tmp_path / "nocache"
        DialogueManager(generator, cache_dir=target, use_cache=False)
        assert not target.exists()

    def test_load_returns_none_and_save_is_noop_when_disabled(self, generator, tmp_path):
        mgr = DialogueManager(generator, cache_dir=tmp_path / "c", use_cache=False)
        assert mgr._load_from_cache("p") is None
        mgr._save_to_cache(make_dialogue(prompt="p"))
        assert not (tmp_path / "c").exists()

    async def test_get_dialogue_always_generates_when_disabled(self, generator, tmp_path):
        mgr = DialogueManager(generator, cache_dir=tmp_path / "c", use_cache=False)
        await mgr.get_dialogue("p")
        await mgr.get_dialogue("p")
        assert generator.generate_dialogue.await_count == 2


class TestManagerGetDialogue:
    async def test_second_call_hits_cache(self, generator, tmp_path):
        mgr = DialogueManager(generator, cache_dir=tmp_path)
        first = await mgr.get_dialogue("p")
        second = await mgr.get_dialogue("p")
        assert generator.generate_dialogue.await_count == 1
        assert second.prompt == first.prompt == "p"
        assert second.final_evaluation.reasoning == "final reasoning"

    async def test_force_regenerate_bypasses_and_refreshes_cache(self, generator, tmp_path):
        mgr = DialogueManager(generator, cache_dir=tmp_path)
        await mgr.get_dialogue("p")
        await mgr.get_dialogue("p", force_regenerate=True)
        assert generator.generate_dialogue.await_count == 2
        assert mgr.cache_stats()["count"] == 1


class TestManagerBatch:
    async def test_batch_mixes_cached_and_generated_preserving_order(self, generator, tmp_path):
        mgr = DialogueManager(generator, cache_dir=tmp_path)
        mgr._save_to_cache(make_dialogue(prompt="b", improved="cached-improved"))
        results = await mgr.get_dialogues(["a", "b", "c"], max_concurrent=2)
        assert [r.prompt for r in results] == ["a", "b", "c"]
        # Only the uncached prompts were generated, with the concurrency forwarded.
        generator.generate_batch.assert_awaited_once()
        args, kwargs = generator.generate_batch.call_args
        assert args[0] == ["a", "c"] and kwargs["max_concurrent"] == 2
        assert results[1].final_evaluation.improved_response == "cached-improved"
        # Newly generated ones were cached.
        assert mgr.cache_stats()["count"] == 3

    async def test_batch_all_cached_skips_generation(self, generator, tmp_path):
        mgr = DialogueManager(generator, cache_dir=tmp_path)
        for p in ("a", "b"):
            mgr._save_to_cache(make_dialogue(prompt=p))
        results = await mgr.get_dialogues(["a", "b"])
        assert [r.prompt for r in results] == ["a", "b"]
        generator.generate_batch.assert_not_awaited()

    async def test_batch_force_regenerate_ignores_cache(self, generator, tmp_path):
        mgr = DialogueManager(generator, cache_dir=tmp_path)
        mgr._save_to_cache(make_dialogue(prompt="a"))
        await mgr.get_dialogues(["a"], force_regenerate=True)
        assert generator.generate_batch.call_args.args[0] == ["a"]

    async def test_empty_batch(self, generator, tmp_path):
        mgr = DialogueManager(generator, cache_dir=tmp_path)
        assert await mgr.get_dialogues([]) == []
        generator.generate_batch.assert_not_awaited()


class TestManagerCacheFiles:
    def test_cache_key_depends_on_teacher_name_and_prompt(self, generator, tmp_path):
        mgr = DialogueManager(generator, cache_dir=tmp_path)
        key = mgr._get_cache_key("p")
        assert key != mgr._get_cache_key("q")
        generator.teacher.name = "T2"
        assert key != mgr._get_cache_key("p")

    def test_corrupt_cache_file_is_a_miss(self, generator, tmp_path):
        mgr = DialogueManager(generator, cache_dir=tmp_path)
        mgr._get_cache_path("p").write_text("{not json")
        assert mgr._load_from_cache("p") is None

    def test_cache_file_missing_fields_is_a_miss(self, generator, tmp_path):
        mgr = DialogueManager(generator, cache_dir=tmp_path)
        mgr._get_cache_path("p").write_text(json.dumps({"prompt": "p"}))
        assert mgr._load_from_cache("p") is None

    def test_iterate_cached_empty_when_dir_missing(self, generator, tmp_path):
        mgr = DialogueManager(generator, cache_dir=tmp_path / "gone", use_cache=False)
        assert list(mgr.iterate_cached()) == []

    def test_iterate_cached_yields_valid_and_skips_bad_files(self, generator, tmp_path):
        mgr = DialogueManager(generator, cache_dir=tmp_path)
        mgr._save_to_cache(make_dialogue(prompt="good", improved="imp", score=8.0))
        (tmp_path / "broken.json").write_text("not json")
        (tmp_path / "partial.json").write_text(json.dumps({"prompt": "x"}))
        items = list(mgr.iterate_cached())
        assert len(items) == 1
        d = items[0]
        assert d.prompt == "good"
        assert d.final_evaluation.overall_score == 8.0
        assert d.final_evaluation.improved_response == "imp"
        assert d.metadata == {"k": "v"}

    def test_clear_cache_counts_and_removes_only_json(self, generator, tmp_path):
        mgr = DialogueManager(generator, cache_dir=tmp_path)
        mgr._save_to_cache(make_dialogue(prompt="a"))
        mgr._save_to_cache(make_dialogue(prompt="b"))
        (tmp_path / "keep.txt").write_text("x")
        assert mgr.clear_cache() == 2
        assert (tmp_path / "keep.txt").exists()
        assert mgr.cache_stats()["count"] == 0

    def test_clear_cache_when_directory_missing(self, generator, tmp_path):
        mgr = DialogueManager(generator, cache_dir=tmp_path / "gone", use_cache=False)
        assert mgr.clear_cache() == 0

    def test_cache_stats_when_directory_missing(self, generator, tmp_path):
        mgr = DialogueManager(generator, cache_dir=tmp_path / "gone", use_cache=False)
        assert mgr.cache_stats() == {"count": 0, "size_bytes": 0}

    def test_cache_stats_reports_sizes(self, generator, tmp_path):
        mgr = DialogueManager(generator, cache_dir=tmp_path)
        mgr._save_to_cache(make_dialogue(prompt="a", turns=[("q", "a")]))
        stats = mgr.cache_stats()
        on_disk = mgr._get_cache_path("a").stat().st_size
        assert stats["count"] == 1 and stats["size_bytes"] == on_disk
        assert stats["size_mb"] == pytest.approx(on_disk / (1024 * 1024))

    def test_saved_turns_are_serialized(self, generator, tmp_path):
        mgr = DialogueManager(generator, cache_dir=tmp_path)
        mgr._save_to_cache(make_dialogue(prompt="a", turns=[("q1", "a1")], turn_eval_reasoning="why"))
        data = json.loads(mgr._get_cache_path("a").read_text())
        assert data["turns"][0]["challenge"] == "q1"
        assert data["turns"][0]["response"] == "a1"
        assert data["turns"][0]["evaluation"]["reasoning"] == "why"


# ---------------------------------------------------------------------------
# DialogueFormatter
# ---------------------------------------------------------------------------


class TestFormatterCritic:
    def test_explicit_response_overrides_history(self):
        d = make_dialogue(turns=[("q1", "a1"), ("q2", "a2")])
        text = DialogueFormatter().format_for_critic(d, response_to_evaluate="custom")
        assert text.splitlines()[:2] == ["Task: What is 2+2?", "Response: custom"]
        # Context contains all turns but the last
        assert "Q: q1" in text and "A: a1" in text
        assert "q2" not in text and "a2" not in text

    def test_response_defaults_to_last_turn(self):
        d = make_dialogue(turns=[("q1", "a1"), ("q2", "a2")])
        text = DialogueFormatter().format_for_critic(d)
        assert "Response: a2" in text

    def test_response_defaults_to_initial_when_no_turns(self):
        text = DialogueFormatter().format_for_critic(make_dialogue(initial="init"))
        assert "Response: init" in text
        assert "Dialogue context" not in text

    def test_single_turn_has_empty_context(self):
        text = DialogueFormatter().format_for_critic(make_dialogue(turns=[("q1", "a1")]))
        assert "Dialogue context:" in text
        assert "Q: " not in text


class TestFormatterTargets:
    @pytest.mark.parametrize("fmt", ["standard", "chat", "instruction"])
    def test_target_is_improved_response_by_default(self, fmt):
        out = DialogueFormatter(format_type=fmt).format_dialogue(make_dialogue(turns=[("q", "a")]))
        assert out.target_text == "Four, because 2+2=4."
        assert out.score == 7.0 and out.num_turns == 1

    @pytest.mark.parametrize("fmt", ["standard", "chat", "instruction"])
    def test_target_falls_back_to_last_student_turn(self, fmt):
        d = make_dialogue(turns=[("q", "a1"), ("q2", "a2")], improved=None)
        out = DialogueFormatter(format_type=fmt).format_dialogue(d)
        assert out.target_text == "a2"

    @pytest.mark.parametrize("fmt", ["standard", "chat", "instruction"])
    def test_target_falls_back_to_initial_without_turns(self, fmt):
        out = DialogueFormatter(format_type=fmt).format_dialogue(make_dialogue(initial="init", improved=None))
        assert out.target_text == "init"

    @pytest.mark.parametrize("fmt", ["standard", "chat", "instruction"])
    def test_use_improved_false_ignores_improved(self, fmt):
        d = make_dialogue(turns=[("q", "a1")])
        out = DialogueFormatter(format_type=fmt).format_dialogue(d, use_improved_as_target=False)
        assert out.target_text == "a1"
        assert out.improved_response == "Four, because 2+2=4."

    def test_unknown_format_rejected(self):
        with pytest.raises(ValueError, match="Unknown format type: weird"):
            DialogueFormatter(format_type="weird").format_dialogue(make_dialogue())


class TestFormatterShapes:
    def test_standard_full_conversation_lists_everything(self):
        d = make_dialogue(turns=[("q1", "a1")])
        out = DialogueFormatter(format_type="standard").format_dialogue(d)
        assert out.input_text == "What is 2+2?"
        for piece in ("Prompt: What is 2+2?", "Initial: four?", "Challenge: q1", "Response: a1", "Improved: Four"):
            assert piece in out.full_conversation

    def test_standard_full_conversation_omits_improved_when_absent(self):
        out = DialogueFormatter(format_type="standard").format_dialogue(make_dialogue(improved=None))
        assert "Improved:" not in out.full_conversation

    def test_chat_with_turns_input_excludes_final_assistant_message(self):
        d = make_dialogue(turns=[("q1", "a1")])
        out = DialogueFormatter(format_type="chat", system_message="SYS").format_dialogue(d)
        assert out.input_text.startswith("<|system|>\nSYS\n<|end|>")
        assert "four?" in out.input_text and "q1" in out.input_text
        assert "a1" not in out.input_text
        assert out.full_conversation.endswith("<|assistant|>\na1\n<|end|>")

    def test_instruction_includes_feedback_only_when_enabled_and_available(self):
        d = make_dialogue(turns=[("q1", "a1")], turn_eval_reasoning="needs more rigor")
        with_reasoning = DialogueFormatter(format_type="instruction", include_reasoning=True)
        assert "needs more rigor" in with_reasoning.format_dialogue(d).input_text
        without = DialogueFormatter(format_type="instruction", include_reasoning=False)
        assert "needs more rigor" not in without.format_dialogue(d).input_text
        # include_reasoning but the last turn was never evaluated: header only
        d2 = make_dialogue(turns=[("q1", "a1")])
        text = with_reasoning.format_dialogue(d2).input_text
        assert "### Feedback from previous attempt:" in text
        assert text.endswith("### Response:")

    def test_instruction_full_conversation_is_input_plus_target(self):
        out = DialogueFormatter(format_type="instruction").format_dialogue(make_dialogue())
        assert out.full_conversation == out.input_text + "\n" + out.target_text

    def test_chat_without_turns_input_contains_the_prompt(self):
        out = DialogueFormatter(format_type="chat").format_dialogue(make_dialogue(improved=None))
        assert "What is 2+2?" in out.input_text
