from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from aspire.dialogue.generator import GeneratedDialogue
from aspire.dialogue.manager import DialogueManager
from aspire.teachers.base import DialogueHistory, TeacherEvaluation


def _fake_dialogue(prompt: str, teacher_name: str = "t") -> GeneratedDialogue:
    teacher_eval = TeacherEvaluation(overall_score=0.5, dimension_scores=[], reasoning="ok", improved_response=None)
    history = DialogueHistory(prompt=prompt, initial_response="hi")
    # History.turns is used in save; ensure it's present
    return GeneratedDialogue(
        prompt=prompt,
        initial_response="hi",
        history=history,
        final_evaluation=teacher_eval,
        turn_evaluations=[],
        metadata={"teacher": teacher_name},
    )


@pytest.mark.asyncio
async def test_get_dialogue_hits_cache_and_avoids_regeneration(tmp_path: Path):
    teacher = MagicMock()
    teacher.name = "teacherA"
    gen = MagicMock()
    gen.teacher = teacher
    gen.generate_dialogue = AsyncMock(side_effect=lambda p: _fake_dialogue(p, teacher.name))

    mgr = DialogueManager(generator=gen, cache_dir=tmp_path, use_cache=True)

    d1 = await mgr.get_dialogue("p1")
    d2 = await mgr.get_dialogue("p1")

    assert d1.prompt == "p1"
    assert d2.prompt == "p1"
    # Generator should be called only once thanks to cache
    assert gen.generate_dialogue.await_count == 1


@pytest.mark.asyncio
async def test_get_dialogues_uses_batch_only_for_misses(tmp_path: Path):
    teacher = MagicMock()
    teacher.name = "teacherA"
    gen = MagicMock()
    gen.teacher = teacher
    gen.generate_batch = AsyncMock(side_effect=lambda prompts, max_concurrent=5: [_fake_dialogue(p, teacher.name) for p in prompts])

    mgr = DialogueManager(generator=gen, cache_dir=tmp_path, use_cache=True)

    # Prime cache for p1
    gen_single = MagicMock()
    gen_single.teacher = teacher
    gen_single.generate_dialogue = AsyncMock(side_effect=lambda p: _fake_dialogue(p, teacher.name))
    mgr.generator = gen_single
    await mgr.get_dialogue("p1")
    assert gen_single.generate_dialogue.await_count == 1

    # Now batch: p1 should be cache hit, p2/p3 misses
    mgr.generator = gen
    out = await mgr.get_dialogues(["p1", "p2", "p3"], max_concurrent=7)

    assert [d.prompt for d in out] == ["p1", "p2", "p3"]
    gen.generate_batch.assert_awaited_once()
    args, kwargs = gen.generate_batch.await_args
    assert args[0] == ["p2", "p3"]
    assert kwargs["max_concurrent"] == 7


def test_cache_key_includes_teacher_name(tmp_path: Path):
    teacherA = MagicMock(); teacherA.name = "A"
    teacherB = MagicMock(); teacherB.name = "B"
    genA = MagicMock(); genA.teacher = teacherA
    genB = MagicMock(); genB.teacher = teacherB

    mgrA = DialogueManager(generator=genA, cache_dir=tmp_path, use_cache=True)
    mgrB = DialogueManager(generator=genB, cache_dir=tmp_path, use_cache=True)

    assert mgrA._get_cache_key("same") != mgrB._get_cache_key("same")
