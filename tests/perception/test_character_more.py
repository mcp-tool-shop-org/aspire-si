"""Additional behaviour tests for aspire.perception.character (coverage gaps)."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from aspire.perception.character import (
    AutobiographicalMemory,
    BehavioralTrait,
    CharacterCore,
    MemoryEntry,
    TraitDimension,
    ValueAnchor,
    ValueType,
    create_analytical_character,
    create_compassionate_character,
    create_socratic_character,
)

# ---------------------------------------------------------------------------
# ValueAnchor / BehavioralTrait
# ---------------------------------------------------------------------------


class TestValueAnchorMore:
    def test_context_match_is_case_insensitive_substring(self):
        anchor = ValueAnchor(ValueType.THOROUGHNESS, activation_contexts=["Debug"])
        assert anchor.applies_in_context("we are DEBUGGING a crash")
        assert not anchor.applies_in_context("casual chat")

    def test_get_resolution_known_and_unknown(self):
        anchor = ValueAnchor(
            ValueType.DIRECTNESS, conflict_resolutions={ValueType.WARMTH: "yield"}
        )
        assert anchor.get_resolution(ValueType.WARMTH) == "yield"
        assert anchor.get_resolution(ValueType.PATIENCE) is None


class TestBehavioralTraitMore:
    def _trait(self, position, **kw):
        kw.setdefault("stability", 1.0)  # no random jitter
        return BehavioralTrait(TraitDimension.CAUTIOUS_BOLD, position=position, **kw)

    def test_stable_trait_has_no_jitter(self):
        assert self._trait(0.42).get_effective_position() == 0.42

    def test_context_modifiers_stack_and_clamp(self):
        trait = self._trait(0.5, context_modifiers={"urgent": 0.3, "deadline": 0.4})
        assert trait.get_effective_position("urgent") == pytest.approx(0.8)
        assert trait.get_effective_position("URGENT deadline") == 1.0  # clamped at 1
        negative = self._trait(0.1, context_modifiers={"calm": -0.5})
        assert negative.get_effective_position("calm") == 0.0  # clamped at 0

    def test_unstable_trait_jitter_uses_gauss_scaled_by_instability(self):
        trait = BehavioralTrait(TraitDimension.CAUTIOUS_BOLD, position=0.5, stability=0.5)
        with patch("random.gauss", return_value=0.05) as gauss:
            result = trait.get_effective_position()
        gauss.assert_called_once()
        assert gauss.call_args.args[0] == 0
        assert gauss.call_args.args[1] == pytest.approx(0.1)  # (1-0.5)*0.2
        assert result == pytest.approx(0.55)

    def test_jitter_is_clamped_to_unit_interval(self):
        trait = BehavioralTrait(TraitDimension.CAUTIOUS_BOLD, position=0.95, stability=0.0)
        with patch("random.gauss", return_value=0.5):
            assert trait.get_effective_position() == 1.0
        with patch("random.gauss", return_value=-5.0):
            assert trait.get_effective_position() == 0.0

    @pytest.mark.parametrize(
        "position,expected",
        [
            (0.0, "strongly cautious"),
            (0.29, "strongly cautious"),
            (0.3, "somewhat cautious"),
            (0.44, "somewhat cautious"),
            (0.45, "balanced"),
            (0.55, "balanced"),
            (0.56, "somewhat bold"),
            (0.69, "somewhat bold"),
            (0.7, "strongly bold"),
            (1.0, "strongly bold"),
        ],
    )
    def test_tendency_bands(self, position, expected):
        assert self._trait(position).get_behavior_tendency() == expected

    def test_tendency_uses_first_and_last_pole_of_the_name(self):
        trait = BehavioralTrait(TraitDimension.ACCOMMODATING_PRINCIPLED, position=0.9, stability=1.0)
        assert trait.get_behavior_tendency() == "strongly principled"

    def test_tendency_for_single_word_dimension_uses_less_more(self):
        trait = BehavioralTrait(SimpleNamespace(value="warmth"), position=0.1, stability=1.0)
        assert trait.get_behavior_tendency() == "strongly less"
        trait.position = 0.9
        assert trait.get_behavior_tendency() == "strongly more"

    def test_tendency_respects_context_modifier(self):
        trait = self._trait(0.2, context_modifiers={"risky": 0.6})
        assert trait.get_behavior_tendency() == "strongly cautious"
        assert trait.get_behavior_tendency("risky") == "strongly bold"


# ---------------------------------------------------------------------------
# MemoryEntry / AutobiographicalMemory
# ---------------------------------------------------------------------------


class TestMemoryEntryMore:
    def test_dict_roundtrip_drops_embedding(self):
        entry = MemoryEntry(
            "sit",
            "act",
            "out",
            "lesson",
            timestamp=12.5,
            context_tags=["a", "b"],
            emotional_valence=-0.5,
            embedding=torch.ones(3),
        )
        data = entry.to_dict()
        assert "embedding" not in data
        json.dumps(data)  # serializable
        restored = MemoryEntry.from_dict(data)
        assert restored.situation == "sit" and restored.timestamp == 12.5
        assert restored.context_tags == ["a", "b"] and restored.emotional_valence == -0.5
        assert restored.embedding is None


def _mem(**kw):
    return AutobiographicalMemory(embedding_dim=4, **kw)


def _add(memory, situation, tags=None, embedding=None, lesson="l"):
    memory.add_memory(situation, "act", "out", lesson, context_tags=tags, embedding=embedding)


class TestAutobiographicalMemoryMore:
    def test_index_tracks_tags_to_positions(self):
        memory = _mem()
        _add(memory, "one", ["x", "y"])
        _add(memory, "two", ["x"])
        assert memory.memory_index == {"x": [0, 1], "y": [0]}
        assert [m.situation for m in memory.get_behavioral_precedents("x")] == ["one", "two"]
        assert memory.get_behavioral_precedents("missing") == []

    def test_pruning_keeps_newest_and_rebuilds_index(self):
        memory = _mem(max_memories=2)
        _add(memory, "oldest", ["t1"])
        _add(memory, "middle", ["t2"])
        memory.memories[0].timestamp = 1.0
        memory.memories[1].timestamp = 2.0
        _add(memory, "newest", ["t2", "t3"])  # exceeds the limit -> prune
        assert [m.situation for m in memory.memories] == ["newest", "middle"]
        assert "t1" not in memory.memory_index
        assert memory.memory_index == {"t2": [0, 1], "t3": [0]}
        # the index must point at the right entries after reordering
        assert [m.situation for m in memory.get_behavioral_precedents("t3")] == ["newest"]

    def test_recall_by_tags_ranks_by_overlap_and_honours_top_k(self):
        memory = _mem()
        _add(memory, "debugging a python crash", ["python"])
        _add(memory, "writing poetry", ["art"])
        _add(memory, "python crash triage", ["triage"])
        hits = memory.recall_by_situation("python crash debugging", top_k=5)
        assert [m.situation for m in hits] == ["debugging a python crash", "python crash triage"]
        assert len(memory.recall_by_situation("python crash debugging", top_k=1)) == 1
        assert memory.recall_by_situation("nothing relevant") == []

    def test_recall_by_embedding_orders_by_cosine_similarity(self):
        memory = _mem()
        _add(memory, "far", embedding=torch.tensor([0.0, 1.0, 0.0, 0.0]))
        _add(memory, "no-embedding")
        _add(memory, "near", embedding=torch.tensor([1.0, 0.1, 0.0, 0.0]))
        query = torch.tensor([1.0, 0.0, 0.0, 0.0])
        hits = memory.recall_by_situation("anything", top_k=5, embedding=query)
        assert [m.situation for m in hits] == ["near", "far"]  # skips the un-embedded memory
        assert [m.situation for m in memory.recall_by_situation("x", top_k=1, embedding=query)] == ["near"]

    def test_embedding_query_falls_back_to_tags_when_no_memory_has_embedding(self):
        memory = _mem()
        _add(memory, "tagged situation", ["alpha"])
        hits = memory.recall_by_situation("alpha", embedding=torch.ones(4))
        assert [m.situation for m in hits] == ["tagged situation"]

    def test_autosave_and_reload(self, tmp_path):
        path = tmp_path / "nested" / "mem.json"
        memory = _mem(storage_path=path)
        _add(memory, "persisted", ["p"], lesson="keep me")
        assert path.exists()
        reloaded = _mem(storage_path=path)
        assert [m.situation for m in reloaded.memories] == ["persisted"]
        assert reloaded.memories[0].lesson == "keep me"
        assert reloaded.memory_index == {"p": [0]}

    def test_save_and_load_are_noops_without_storage(self, tmp_path):
        memory = _mem()
        memory._save()
        memory._load()
        assert memory.memories == []
        gone = _mem(storage_path=tmp_path / "gone.json")
        gone._load()  # path does not exist
        assert gone.memories == []

    def test_load_tolerates_missing_keys(self, tmp_path):
        path = tmp_path / "m.json"
        path.write_text("{}")
        memory = _mem(storage_path=path)
        assert memory.memories == [] and memory.memory_index == {}


# ---------------------------------------------------------------------------
# CharacterCore
# ---------------------------------------------------------------------------


class TestCharacterCoreMore:
    def test_defaults(self):
        char = CharacterCore()
        assert set(char.values) == {
            ValueType.TRUTH_SEEKING,
            ValueType.HELPFULNESS,
            ValueType.INTELLECTUAL_HONESTY,
        }
        assert set(char.traits) == set(TraitDimension)
        assert all(t.position == 0.5 for t in char.traits.values())

    def test_identity_hash_stable_and_sensitive_to_changes(self):
        a, b = CharacterCore(name="A"), CharacterCore(name="A")
        assert a.get_identity_hash() == b.get_identity_hash()
        assert len(a.get_identity_hash()) == 16
        before = a.get_identity_hash()
        a.set_trait(TraitDimension.FORMAL_CASUAL, 0.9)
        assert a.get_identity_hash() != before
        mid = a.get_identity_hash()
        a.set_value(ValueType.WARMTH, priority=0.9)
        assert a.get_identity_hash() != mid
        assert CharacterCore(name="B").get_identity_hash() != before

    def test_identity_hash_is_cached_until_invalidated(self):
        char = CharacterCore()
        first = char.get_identity_hash()
        char.traits[TraitDimension.FORMAL_CASUAL].position = 0.99  # bypasses setters
        assert char.get_identity_hash() == first
        char._invalidate_hash()
        assert char.get_identity_hash() != first

    def test_active_values_sorted_and_filtered_by_context(self):
        char = CharacterCore()
        char.set_value(ValueType.THOROUGHNESS, priority=0.99, activation_contexts=["audit"])
        names = [v.value_type for v in char.get_active_values("casual chat")]
        assert ValueType.THOROUGHNESS not in names
        active = char.get_active_values("security audit")
        assert active[0].value_type == ValueType.THOROUGHNESS
        priorities = [v.priority for v in active]
        assert priorities == sorted(priorities, reverse=True)

    def test_trait_profile_covers_all_dimensions_and_applies_modifiers(self):
        char = CharacterCore()
        char.set_trait(
            TraitDimension.CAUTIOUS_BOLD, 0.5, stability=1.0, context_modifiers={"crisis": -0.4}
        )
        profile = char.get_trait_profile("crisis mode")
        assert set(profile) == set(TraitDimension)
        assert profile[TraitDimension.CAUTIOUS_BOLD] == pytest.approx(0.1)

    def test_resolve_conflict_missing_anchor_returns_the_other(self):
        char = CharacterCore()
        assert char.resolve_value_conflict(ValueType.WARMTH, ValueType.TRUTH_SEEKING) == ValueType.TRUTH_SEEKING
        assert char.resolve_value_conflict(ValueType.TRUTH_SEEKING, ValueType.WARMTH) == ValueType.TRUTH_SEEKING

    def test_resolve_conflict_explicit_rules_from_first_anchor(self):
        char = CharacterCore()
        char.set_value(
            ValueType.WARMTH, priority=0.1, conflict_resolutions={ValueType.DIRECTNESS: "override"}
        )
        char.set_value(ValueType.DIRECTNESS, priority=0.99)
        assert char.resolve_value_conflict(ValueType.WARMTH, ValueType.DIRECTNESS) == ValueType.WARMTH
        char.set_value(
            ValueType.WARMTH, priority=0.99, conflict_resolutions={ValueType.DIRECTNESS: "yield"}
        )
        assert char.resolve_value_conflict(ValueType.WARMTH, ValueType.DIRECTNESS) == ValueType.DIRECTNESS

    def test_resolve_conflict_explicit_rules_from_second_anchor(self):
        char = CharacterCore()
        char.set_value(ValueType.WARMTH, priority=0.99)
        char.set_value(
            ValueType.DIRECTNESS, priority=0.1, conflict_resolutions={ValueType.WARMTH: "yield"}
        )
        assert char.resolve_value_conflict(ValueType.WARMTH, ValueType.DIRECTNESS) == ValueType.WARMTH
        char.set_value(
            ValueType.DIRECTNESS, priority=0.1, conflict_resolutions={ValueType.WARMTH: "override"}
        )
        assert char.resolve_value_conflict(ValueType.WARMTH, ValueType.DIRECTNESS) == ValueType.DIRECTNESS

    def test_resolve_conflict_by_priority_with_context_boost(self):
        char = CharacterCore()
        char.set_value(ValueType.WARMTH, priority=0.5)
        char.set_value(ValueType.DIRECTNESS, priority=0.52, activation_contexts=["review"])
        # Outside DIRECTNESS's context only the always-applicable WARMTH is boosted: 0.55 > 0.52.
        assert char.resolve_value_conflict(ValueType.WARMTH, ValueType.DIRECTNESS, "chat") == ValueType.WARMTH
        # In context both are boosted (0.55 vs 0.572), so the higher raw priority wins.
        assert char.resolve_value_conflict(ValueType.WARMTH, ValueType.DIRECTNESS, "code review") == ValueType.DIRECTNESS
        # Ties go to the first argument.
        char.set_value(ValueType.DIRECTNESS, priority=0.5)
        assert char.resolve_value_conflict(ValueType.WARMTH, ValueType.DIRECTNESS) == ValueType.WARMTH
        assert char.resolve_value_conflict(ValueType.DIRECTNESS, ValueType.WARMTH) == ValueType.DIRECTNESS

    def test_context_boost_can_flip_the_outcome(self):
        char = CharacterCore()
        char.set_value(ValueType.WARMTH, priority=0.6, activation_contexts=["grief"])
        char.set_value(ValueType.DIRECTNESS, priority=0.62, activation_contexts=["review"])
        # Only WARMTH is boosted in a grief context: 0.66 > 0.62.
        assert char.resolve_value_conflict(ValueType.WARMTH, ValueType.DIRECTNESS, "grief") == ValueType.WARMTH
        assert char.resolve_value_conflict(ValueType.WARMTH, ValueType.DIRECTNESS, "") == ValueType.DIRECTNESS

    def test_character_prompt_sections(self):
        char = CharacterCore(name="Ada", description="A careful engineer.")
        prompt = char.generate_character_prompt()
        assert prompt.startswith("CHARACTER: Ada\n\nA careful engineer.")
        assert "CORE VALUES (in priority order):" in prompt
        assert prompt.index("truth_seeking") < prompt.index("helpfulness")
        assert "BEHAVIORAL TENDENCIES:" in prompt
        assert "RELEVANT PAST EXPERIENCE" not in prompt
        for dim in TraitDimension:
            assert dim.value in prompt

    def test_character_prompt_limits_values_to_top_five_and_omits_empty_description(self):
        char = CharacterCore(name="X")
        for vt in (ValueType.WARMTH, ValueType.PATIENCE, ValueType.CLARITY, ValueType.FAIRNESS):
            char.set_value(vt, priority=0.1, description=f"{vt.value} desc")
        prompt = char.generate_character_prompt()
        assert prompt.count("\n• ") - len(TraitDimension) == 5
        assert "\n\n\n" not in prompt

    def test_character_prompt_without_values(self):
        char = CharacterCore()
        char.values.clear()
        prompt = char.generate_character_prompt()
        assert "CORE VALUES" not in prompt and "BEHAVIORAL TENDENCIES" in prompt

    def test_character_prompt_includes_truncated_relevant_memories(self):
        char = CharacterCore()
        long_text = "refactor " + "x" * 200
        char.record_experience(long_text, "a" * 300, "ok", "measure first", context_tags=["refactor"])
        prompt = char.generate_character_prompt("refactor")
        assert "RELEVANT PAST EXPERIENCE:" in prompt
        assert f"• Situation: {long_text[:100]}..." in prompt
        assert long_text[:101] not in prompt
        assert f"  Action: {'a' * 100}..." in prompt
        assert "  Lesson: measure first" in prompt

    def test_record_experience_sets_valence(self):
        char = CharacterCore()
        char.record_experience("s1", "a", "o", "l", was_positive=True)
        char.record_experience("s2", "a", "o", "l", was_positive=False)
        assert [m.emotional_valence for m in char.memory.memories] == [0.5, -0.5]


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def test_save_without_any_path_raises(self):
        with pytest.raises(ValueError, match="No save path"):
            CharacterCore().save()

    def test_save_default_path_and_roundtrip(self, tmp_path):
        char = CharacterCore(name="Round", description="Trip", storage_dir=tmp_path)
        char.set_value(ValueType.WARMTH, priority=0.77, strength=0.33, description="warm")
        char.set_trait(
            TraitDimension.FORMAL_CASUAL, 0.9, stability=0.6, context_modifiers={"party": 0.05}
        )
        char.save()
        assert (tmp_path / "character.json").exists()
        loaded = CharacterCore.load(tmp_path / "character.json")
        assert loaded.name == "Round" and loaded.description == "Trip"
        warm = loaded.values[ValueType.WARMTH]
        assert (warm.priority, warm.strength, warm.description) == (0.77, 0.33, "warm")
        trait = loaded.traits[TraitDimension.FORMAL_CASUAL]
        assert (trait.position, trait.stability, trait.context_modifiers) == (0.9, 0.6, {"party": 0.05})
        assert loaded.get_identity_hash() == char.get_identity_hash()

    def test_save_rejects_path_outside_storage_dir(self, tmp_path):
        storage = tmp_path / "store"
        char = CharacterCore(storage_dir=storage)
        with pytest.raises(ValueError, match="within storage directory"):
            char.save(tmp_path / "elsewhere.json")
        with pytest.raises(ValueError, match="within storage directory"):
            char.save(storage / ".." / "escape.json")
        assert not (tmp_path / "escape.json").exists()

    def test_save_outside_allowed_when_flagged_or_inside(self, tmp_path):
        storage = tmp_path / "store"
        char = CharacterCore(storage_dir=storage)
        char.save(storage / "sub" / "ok.json")
        assert (storage / "sub" / "ok.json").exists()
        char.save(tmp_path / "out.json", allow_outside_storage=True)
        assert (tmp_path / "out.json").exists()

    def test_save_with_explicit_path_and_no_storage_dir(self, tmp_path):
        CharacterCore().save(tmp_path / "free.json")
        assert (tmp_path / "free.json").exists()

    def _write(self, tmp_path, data):
        path = tmp_path / "c.json"
        path.write_text(json.dumps(data))
        return path

    def test_load_rejects_non_object_documents(self, tmp_path):
        with pytest.raises(ValueError, match="JSON object"):
            CharacterCore.load(self._write(tmp_path, [1, 2]))

    def test_load_rejects_non_dict_values_or_traits(self, tmp_path):
        with pytest.raises(ValueError, match="'values' must be a dictionary"):
            CharacterCore.load(self._write(tmp_path, {"values": []}))
        with pytest.raises(ValueError, match="'traits' must be a dictionary"):
            CharacterCore.load(self._write(tmp_path, {"traits": "no"}))

    def test_load_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            CharacterCore.load(tmp_path / "nope.json")

    def test_load_truncates_name_and_description(self, tmp_path):
        path = self._write(tmp_path, {"name": "n" * 1000, "description": "d" * 10000})
        char = CharacterCore.load(path)
        assert len(char.name) == 256 and len(char.description) == 4096

    def test_load_missing_name_uses_default(self, tmp_path):
        assert CharacterCore.load(self._write(tmp_path, {})).name == "Agent"

    def test_load_skips_unknown_and_malformed_entries(self, tmp_path):
        before = CharacterCore().get_identity_hash()
        path = self._write(
            tmp_path,
            {
                "values": {"not_a_value": {"priority": 0.1}, "warmth": "oops", "patience": 5},
                "traits": {
                    "not_a_trait": {"position": 0.1},
                    "formal_casual": ["bad"],
                    "cautious_bold": {"stability": 0.2},  # no position: set_trait raises TypeError
                },
            },
        )
        char = CharacterCore.load(path)
        assert ValueType.WARMTH not in char.values and ValueType.PATIENCE not in char.values
        assert char.get_identity_hash() == before

    def test_load_ignores_unsafe_keys_and_types(self, tmp_path):
        path = self._write(
            tmp_path,
            {
                "values": {"warmth": {"priority": 0.3, "strength": 0.2, "evil": 1, "description": {"a": 1}}},
                "traits": {"formal_casual": {"position": 0.8, "evil": 1, "stability": "high"}},
            },
        )
        char = CharacterCore.load(path)
        warm = char.values[ValueType.WARMTH]
        assert (warm.priority, warm.strength) == (0.3, 0.2)
        assert warm.description == ""  # dict-valued description was filtered out
        trait = char.traits[TraitDimension.FORMAL_CASUAL]
        assert trait.position == 0.8 and trait.stability == 0.7  # str stability filtered -> default

    def test_load_caps_number_of_values_and_traits(self, tmp_path):
        values = {v.value: {"priority": 0.1} for v in ValueType}
        traits = {t.value: {"position": 0.9} for t in TraitDimension}
        path = self._write(tmp_path, {"values": values, "traits": traits})
        char = CharacterCore.load(path, max_values=2, max_traits=3)
        loaded_values = [v for v in char.values.values() if v.priority == 0.1]
        loaded_traits = [t for t in char.traits.values() if t.position == 0.9]
        assert len(loaded_values) == 2
        assert len(loaded_traits) == 3

    def test_load_uses_file_directory_as_storage_dir(self, tmp_path):
        path = self._write(tmp_path, {"name": "S"})
        assert CharacterCore.load(path).storage_dir == tmp_path

    # ---- former defects (regression tests) -----------------------------------------------

    def test_activation_contexts_survive_roundtrip(self, tmp_path):
        char = CharacterCore(storage_dir=tmp_path)
        char.set_value(ValueType.THOROUGHNESS, priority=0.9, activation_contexts=["audit"])
        char.save()
        loaded = CharacterCore.load(tmp_path / "character.json")
        assert loaded.values[ValueType.THOROUGHNESS].activation_contexts == ["audit"]

    def test_whitelisted_contexts_key_does_not_drop_the_value(self, tmp_path):
        path = tmp_path / "c.json"
        path.write_text(json.dumps({"values": {"warmth": {"priority": 0.3, "contexts": ["grief"]}}}))
        char = CharacterCore.load(path)
        assert char.values[ValueType.WARMTH].priority == 0.3

    def test_non_numeric_priority_does_not_poison_later_use(self, tmp_path):
        path = tmp_path / "c.json"
        path.write_text(json.dumps({"values": {"warmth": {"priority": "high"}}}))
        char = CharacterCore.load(path)
        char.get_active_values()  # must not raise


# ---------------------------------------------------------------------------
# templates
# ---------------------------------------------------------------------------


class TestTemplates:
    def test_socratic(self):
        char = create_socratic_character()
        assert char.name == "Socratic Guide"
        assert char.values[ValueType.TRUTH_SEEKING].priority == 0.95
        assert char.traits[TraitDimension.SUPPORTIVE_CHALLENGING].position == 0.7
        assert char.traits[TraitDimension.VERBOSE_CONCISE].position == 0.3
        assert char.traits[TraitDimension.ANALYTICAL_INTUITIVE].position == 0.6

    def test_compassionate(self):
        char = create_compassionate_character()
        assert char.name == "Compassionate Guide"
        assert {ValueType.WARMTH, ValueType.PATIENCE} <= set(char.values)
        assert char.traits[TraitDimension.EMPATHETIC_DETACHED].position == 0.8
        assert char.traits[TraitDimension.SUPPORTIVE_CHALLENGING].position == 0.3

    def test_analytical(self):
        char = create_analytical_character()
        assert char.name == "Analytical Partner"
        assert {ValueType.EVIDENCE_BASED, ValueType.THOROUGHNESS, ValueType.CLARITY} <= set(char.values)
        assert char.traits[TraitDimension.ANALYTICAL_INTUITIVE].position == 0.9
        assert char.traits[TraitDimension.CAUTIOUS_BOLD].position == 0.3

    def test_templates_have_distinct_identities(self):
        hashes = {
            create_socratic_character().get_identity_hash(),
            create_compassionate_character().get_identity_hash(),
            create_analytical_character().get_identity_hash(),
            CharacterCore().get_identity_hash(),
        }
        assert len(hashes) == 4
