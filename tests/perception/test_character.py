"""
Tests for Character Persistence module.

Tests value anchoring, behavioral traits, autobiographical memory,
and character consistency mechanisms.
"""

import os

os.environ["XFORMERS_DISABLED"] = "1"

import tempfile
from pathlib import Path

import pytest

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


class TestValueAnchor:
    """Tests for ValueAnchor class."""

    def test_basic_value_creation(self):
        """Test creating a basic value anchor."""
        value = ValueAnchor(
            value_type=ValueType.TRUTH_SEEKING,
            priority=0.9,
            description="Committed to truth",
            strength=0.8,
        )

        assert value.value_type == ValueType.TRUTH_SEEKING
        assert value.priority == 0.9
        assert value.strength == 0.8

    def test_applies_in_context_no_contexts(self):
        """Test value applies when no specific contexts are set."""
        value = ValueAnchor(value_type=ValueType.HELPFULNESS)

        assert value.applies_in_context("any context") is True
        assert value.applies_in_context("") is True

    def test_applies_in_context_with_contexts(self):
        """Test value applies only in specified contexts."""
        value = ValueAnchor(
            value_type=ValueType.THOROUGHNESS,
            activation_contexts=["technical", "debugging"],
        )

        assert value.applies_in_context("This is a technical question") is True
        assert value.applies_in_context("debugging the issue") is True
        assert value.applies_in_context("casual conversation") is False

    def test_conflict_resolution(self):
        """Test conflict resolution strategies."""
        value = ValueAnchor(
            value_type=ValueType.HELPFULNESS,
            conflict_resolutions={
                ValueType.HARMLESSNESS: "yield",  # Yield to safety
            },
        )

        assert value.get_resolution(ValueType.HARMLESSNESS) == "yield"
        assert value.get_resolution(ValueType.CLARITY) is None


class TestBehavioralTrait:
    """Tests for BehavioralTrait class."""

    def test_basic_trait_creation(self):
        """Test creating a basic behavioral trait."""
        trait = BehavioralTrait(
            dimension=TraitDimension.VERBOSE_CONCISE,
            position=0.3,  # Leaning concise
            stability=0.8,
        )

        assert trait.dimension == TraitDimension.VERBOSE_CONCISE
        assert trait.position == 0.3
        assert trait.stability == 0.8

    def test_get_effective_position_no_context(self):
        """Test getting position without context modifiers."""
        trait = BehavioralTrait(
            dimension=TraitDimension.FORMAL_CASUAL,
            position=0.5,
            stability=1.0,  # No variance
        )

        assert trait.get_effective_position() == 0.5

    def test_get_effective_position_with_context(self):
        """Test position adjusts with context."""
        trait = BehavioralTrait(
            dimension=TraitDimension.FORMAL_CASUAL,
            position=0.5,
            stability=1.0,
            context_modifiers={
                "professional": -0.2,  # More formal
                "casual chat": 0.2,  # More casual
            },
        )

        assert trait.get_effective_position("professional meeting") == 0.3
        assert trait.get_effective_position("casual chat with friend") == 0.7
        assert trait.get_effective_position("neutral context") == 0.5

    def test_position_clamping(self):
        """Test that position is clamped to 0-1."""
        trait = BehavioralTrait(
            dimension=TraitDimension.CAUTIOUS_BOLD,
            position=0.9,
            stability=1.0,
            context_modifiers={"risk": 0.5},
        )

        # Should be clamped to 1.0
        pos = trait.get_effective_position("high risk situation")
        assert pos == 1.0

    def test_get_behavior_tendency(self):
        """Test getting human-readable tendency."""
        trait = BehavioralTrait(
            dimension=TraitDimension.ANALYTICAL_INTUITIVE,
            position=0.1,
            stability=1.0,
        )

        tendency = trait.get_behavior_tendency()
        assert "strongly" in tendency and "analytical" in tendency

        trait.position = 0.9
        tendency = trait.get_behavior_tendency()
        assert "strongly" in tendency and "intuitive" in tendency


class TestAutobiographicalMemory:
    """Tests for AutobiographicalMemory class."""

    @pytest.fixture
    def memory(self):
        return AutobiographicalMemory(max_memories=10)

    def test_add_memory(self, memory):
        """Test adding a memory entry."""
        memory.add_memory(
            situation="User asked about recursion",
            action_taken="Explained with factorial example",
            outcome="User understood",
            lesson="Concrete examples help with abstract concepts",
            context_tags=["teaching", "programming"],
        )

        assert len(memory.memories) == 1
        assert "teaching" in memory.memory_index
        assert 0 in memory.memory_index["teaching"]

    def test_recall_by_tags(self, memory):
        """Test recalling memories by tag matching."""
        memory.add_memory(
            situation="Debugging Python error",
            action_taken="Checked stack trace",
            outcome="Found the bug",
            lesson="Always check stack trace first",
            context_tags=["debugging", "python"],
        )
        memory.add_memory(
            situation="Explaining JavaScript closures",
            action_taken="Used counter example",
            outcome="User understood",
            lesson="Counter example works for closures",
            context_tags=["teaching", "javascript"],
        )

        recalled = memory.recall_by_situation("debugging Python issue", top_k=5)

        # Should recall the debugging memory
        assert len(recalled) >= 1
        assert any("Python" in m.situation for m in recalled)

    def test_get_behavioral_precedents(self, memory):
        """Test getting all memories with a specific tag."""
        for i in range(3):
            memory.add_memory(
                situation=f"Teaching situation {i}",
                action_taken=f"Action {i}",
                outcome="Good",
                lesson=f"Lesson {i}",
                context_tags=["teaching"],
            )

        precedents = memory.get_behavioral_precedents("teaching")

        assert len(precedents) == 3

    def test_max_memories_pruning(self, memory):
        """Test that memories are pruned when limit is exceeded."""
        for i in range(15):
            memory.add_memory(
                situation=f"Situation {i}",
                action_taken=f"Action {i}",
                outcome="OK",
                lesson=f"Lesson {i}",
            )

        assert len(memory.memories) == 10  # Max limit

    def test_persistence(self):
        """Test saving and loading memories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "memories.json"

            # Create and populate memory
            memory = AutobiographicalMemory(storage_path=path)
            memory.add_memory(
                situation="Test situation",
                action_taken="Test action",
                outcome="Test outcome",
                lesson="Test lesson",
                context_tags=["test"],
            )

            # Load in new instance
            memory2 = AutobiographicalMemory(storage_path=path)

            assert len(memory2.memories) == 1
            assert memory2.memories[0].lesson == "Test lesson"


class TestCharacterCore:
    """Tests for CharacterCore class."""

    @pytest.fixture
    def character(self):
        return CharacterCore(name="TestAgent", description="A test agent")

    def test_initialization(self, character):
        """Test character initializes with defaults."""
        assert character.name == "TestAgent"
        assert len(character.values) > 0  # Default values
        assert len(character.traits) > 0  # Default traits

    def test_set_value(self, character):
        """Test setting a value."""
        character.set_value(
            ValueType.CREATIVITY,
            priority=0.8,
            strength=0.9,
            description="Loves creative solutions",
        )

        assert ValueType.CREATIVITY in character.values
        assert character.values[ValueType.CREATIVITY].priority == 0.8

    def test_set_trait(self, character):
        """Test setting a trait."""
        character.set_trait(
            TraitDimension.VERBOSE_CONCISE,
            position=0.2,  # Very concise
            stability=0.9,
        )

        assert character.traits[TraitDimension.VERBOSE_CONCISE].position == 0.2

    def test_get_active_values(self, character):
        """Test getting active values sorted by priority."""
        character.set_value(ValueType.TRUTH_SEEKING, priority=0.9)
        character.set_value(ValueType.HELPFULNESS, priority=0.7)
        character.set_value(ValueType.CLARITY, priority=0.8)

        active = character.get_active_values()

        # Should be sorted by priority (highest first)
        priorities = [v.priority for v in active]
        assert priorities == sorted(priorities, reverse=True)

    def test_get_trait_profile(self, character):
        """Test getting trait profile."""
        profile = character.get_trait_profile()

        assert TraitDimension.VERBOSE_CONCISE in profile
        assert all(0 <= v <= 1 for v in profile.values())

    def test_resolve_value_conflict(self, character):
        """Test resolving conflicts between values."""
        character.set_value(ValueType.HELPFULNESS, priority=0.8)
        character.set_value(ValueType.HARMLESSNESS, priority=0.95)

        winner = character.resolve_value_conflict(
            ValueType.HELPFULNESS,
            ValueType.HARMLESSNESS,
        )

        # Higher priority should win
        assert winner == ValueType.HARMLESSNESS

    def test_generate_character_prompt(self, character):
        """Test generating character prompt."""
        character.set_value(ValueType.TRUTH_SEEKING, priority=0.9,
                           description="Committed to truth")
        character.set_trait(TraitDimension.FORMAL_CASUAL, position=0.3)

        prompt = character.generate_character_prompt()

        assert "TestAgent" in prompt
        assert "truth" in prompt.lower()
        assert len(prompt) > 100

    def test_record_experience(self, character):
        """Test recording an experience."""
        character.record_experience(
            situation="User asked a tricky question",
            action_taken="Took time to think before answering",
            outcome="User appreciated the thoughtful response",
            lesson="Thoughtful responses build trust",
            context_tags=["complex_query"],
            was_positive=True,
        )

        assert len(character.memory.memories) > 0

    def test_identity_hash(self, character):
        """Test identity hash changes with character changes."""
        hash1 = character.get_identity_hash()

        character.set_value(ValueType.CREATIVITY, priority=0.99)

        hash2 = character.get_identity_hash()

        assert hash1 != hash2

    def test_save_and_load(self, character):
        """Test saving and loading character."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "character.json"

            character.set_value(ValueType.CREATIVITY, priority=0.85)
            character.set_trait(TraitDimension.CAUTIOUS_BOLD, position=0.7)
            character.save(path)

            loaded = CharacterCore.load(path)

            assert loaded.name == "TestAgent"
            assert ValueType.CREATIVITY in loaded.values
            assert loaded.values[ValueType.CREATIVITY].priority == 0.85
            assert loaded.traits[TraitDimension.CAUTIOUS_BOLD].position == 0.7


class TestCharacterTemplates:
    """Tests for pre-built character templates."""

    def test_socratic_character(self):
        """Test Socratic character template."""
        char = create_socratic_character()

        assert char.name == "Socratic Guide"
        assert ValueType.TRUTH_SEEKING in char.values
        assert char.values[ValueType.TRUTH_SEEKING].priority > 0.9

        # Should be more challenging than supportive
        support_challenge = char.traits[TraitDimension.SUPPORTIVE_CHALLENGING]
        assert support_challenge.position > 0.5

    def test_compassionate_character(self):
        """Test Compassionate character template."""
        char = create_compassionate_character()

        assert char.name == "Compassionate Guide"
        assert ValueType.WARMTH in char.values

        # Should be more empathetic
        empathy = char.traits[TraitDimension.EMPATHETIC_DETACHED]
        assert empathy.position > 0.7

    def test_analytical_character(self):
        """Test Analytical character template."""
        char = create_analytical_character()

        assert char.name == "Analytical Partner"
        assert ValueType.EVIDENCE_BASED in char.values

        # Should be highly analytical
        analytical = char.traits[TraitDimension.ANALYTICAL_INTUITIVE]
        assert analytical.position > 0.8


class TestMemoryEntry:
    """Tests for MemoryEntry dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        entry = MemoryEntry(
            situation="Test situation",
            action_taken="Test action",
            outcome="Test outcome",
            lesson="Test lesson",
            context_tags=["tag1", "tag2"],
            emotional_valence=0.5,
        )

        d = entry.to_dict()

        assert d["situation"] == "Test situation"
        assert d["context_tags"] == ["tag1", "tag2"]
        assert "timestamp" in d

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "situation": "Test",
            "action_taken": "Action",
            "outcome": "Outcome",
            "lesson": "Lesson",
            "timestamp": 123456.0,
            "context_tags": ["test"],
            "emotional_valence": -0.3,
        }

        entry = MemoryEntry.from_dict(data)

        assert entry.situation == "Test"
        assert entry.emotional_valence == -0.3


# Windows compatibility
if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    pytest.main([__file__, "-v"])
