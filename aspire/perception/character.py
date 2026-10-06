"""
Character Persistence Module - Stable identity and value anchoring.

For an agent to have genuine "character," it needs more than consistent outputs -
it needs a stable internal structure that persists across contexts.

This module provides:
1. Value Anchoring - Core values that guide decisions when guidance is ambiguous
2. Behavioral Traits - Consistent behavioral patterns that define personality
3. Autobiographical Memory - "What have I done before in similar situations?"
4. Reflective Identity - Self-model that enables coherent behavior

The goal is not to make agents "seem" consistent through prompt engineering,
but to give them structures that naturally produce consistency.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class ValueType(str, Enum):
    """Categories of values that can be anchored."""

    # Epistemic values
    TRUTH_SEEKING = "truth_seeking"
    INTELLECTUAL_HONESTY = "intellectual_honesty"
    UNCERTAINTY_ACKNOWLEDGMENT = "uncertainty_acknowledgment"
    EVIDENCE_BASED = "evidence_based"

    # Ethical values
    HELPFULNESS = "helpfulness"
    HARMLESSNESS = "harmlessness"
    FAIRNESS = "fairness"
    RESPECT = "respect"

    # Interaction values
    CLARITY = "clarity"
    PATIENCE = "patience"
    DIRECTNESS = "directness"
    WARMTH = "warmth"

    # Process values
    THOROUGHNESS = "thoroughness"
    EFFICIENCY = "efficiency"
    CREATIVITY = "creativity"
    PRAGMATISM = "pragmatism"


class TraitDimension(str, Enum):
    """Personality trait dimensions (inspired by Big Five but adapted for AI)."""

    # How the agent engages
    PROACTIVE_REACTIVE = "proactive_reactive"  # Initiates vs responds
    VERBOSE_CONCISE = "verbose_concise"  # Explanation depth
    FORMAL_CASUAL = "formal_casual"  # Communication style

    # How the agent reasons
    ANALYTICAL_INTUITIVE = "analytical_intuitive"  # Reasoning style
    CAUTIOUS_BOLD = "cautious_bold"  # Risk tolerance
    CONVERGENT_DIVERGENT = "convergent_divergent"  # Solution focus

    # How the agent relates
    EMPATHETIC_DETACHED = "empathetic_detached"  # Emotional engagement
    SUPPORTIVE_CHALLENGING = "supportive_challenging"  # Teaching style
    ACCOMMODATING_PRINCIPLED = "accommodating_principled"  # Flexibility


@dataclass
class ValueAnchor:
    """
    A single anchored value with priority and application rules.

    Values aren't just labels - they have conditions under which they apply
    and can conflict with each other, requiring prioritization.
    """

    value_type: ValueType
    priority: float = 0.5  # 0-1, higher = more important in conflicts
    description: str = ""

    # When does this value apply?
    activation_contexts: list[str] = field(default_factory=list)

    # How strongly should it influence behavior?
    strength: float = 0.7  # 0-1

    # What other values does it potentially conflict with?
    potential_conflicts: list[ValueType] = field(default_factory=list)

    # How to resolve conflicts with specific values
    conflict_resolutions: dict[ValueType, str] = field(default_factory=dict)

    def applies_in_context(self, context: str) -> bool:
        """Check if this value applies in the given context."""
        if not self.activation_contexts:
            return True  # Always applies if no specific contexts
        return any(ctx.lower() in context.lower() for ctx in self.activation_contexts)

    def get_resolution(self, conflicting_value: ValueType) -> str | None:
        """Get resolution strategy for conflict with another value."""
        return self.conflict_resolutions.get(conflicting_value)


@dataclass
class BehavioralTrait:
    """
    A behavioral trait on a spectrum.

    Traits aren't binary - they're positions on a continuum that
    influence behavior probabilistically, not deterministically.
    """

    dimension: TraitDimension
    position: float = 0.5  # 0-1, where 0.5 is neutral

    # How stable is this trait? (variance around position)
    stability: float = 0.8  # Higher = more consistent

    # Context-dependent adjustments
    context_modifiers: dict[str, float] = field(default_factory=dict)

    def get_effective_position(self, context: str = "") -> float:
        """Get trait position adjusted for context."""
        base = self.position

        # Apply context modifiers
        for ctx_pattern, modifier in self.context_modifiers.items():
            if ctx_pattern.lower() in context.lower():
                base = max(0.0, min(1.0, base + modifier))

        # Add some variance based on stability
        if self.stability < 1.0:
            import random

            variance = (1.0 - self.stability) * 0.2
            base += random.gauss(0, variance)
            base = max(0.0, min(1.0, base))

        return base

    def get_behavior_tendency(self, context: str = "") -> str:
        """Get human-readable tendency description."""
        pos = self.get_effective_position(context)
        dim_name = self.dimension.value

        # Parse dimension name into poles
        poles = dim_name.split("_")
        if len(poles) >= 2:
            left_pole, right_pole = poles[0], poles[-1]
        else:
            left_pole, right_pole = "less", "more"

        if pos < 0.3:
            return f"strongly {left_pole}"
        elif pos < 0.45:
            return f"somewhat {left_pole}"
        elif pos <= 0.55:
            return "balanced"
        elif pos < 0.7:
            return f"somewhat {right_pole}"
        else:
            return f"strongly {right_pole}"


@dataclass
class MemoryEntry:
    """A single autobiographical memory entry."""

    # What happened
    situation: str
    action_taken: str
    outcome: str

    # How to learn from it
    lesson: str

    # Metadata
    timestamp: float = field(default_factory=time.time)
    context_tags: list[str] = field(default_factory=list)
    emotional_valence: float = 0.0  # -1 to 1

    # Embedding for similarity search
    embedding: torch.Tensor | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to serializable dict."""
        return {
            "situation": self.situation,
            "action_taken": self.action_taken,
            "outcome": self.outcome,
            "lesson": self.lesson,
            "timestamp": self.timestamp,
            "context_tags": self.context_tags,
            "emotional_valence": self.emotional_valence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryEntry:
        """Create from dict."""
        return cls(**data)


class AutobiographicalMemory:
    """
    Stores and retrieves "experiences" for behavioral consistency.

    This isn't episodic memory of conversations - it's memory of
    patterns: "In situations like X, I tend to do Y, which works because Z."
    """

    def __init__(
        self,
        max_memories: int = 1000,
        embedding_dim: int = 768,
        storage_path: Path | None = None,
    ):
        self.max_memories = max_memories
        self.embedding_dim = embedding_dim
        self.storage_path = storage_path

        self.memories: list[MemoryEntry] = []
        self.memory_index: dict[str, list[int]] = {}  # tag -> memory indices

        # Simple embedding projector (if we want to use it)
        self.projector = nn.Linear(embedding_dim, embedding_dim // 2)

        if storage_path and storage_path.exists():
            self._load()

    def add_memory(
        self,
        situation: str,
        action_taken: str,
        outcome: str,
        lesson: str,
        context_tags: list[str] | None = None,
        emotional_valence: float = 0.0,
        embedding: torch.Tensor | None = None,
    ) -> None:
        """Add a new memory entry."""
        entry = MemoryEntry(
            situation=situation,
            action_taken=action_taken,
            outcome=outcome,
            lesson=lesson,
            context_tags=context_tags or [],
            emotional_valence=emotional_valence,
            embedding=embedding,
        )

        self.memories.append(entry)

        # Update index
        idx = len(self.memories) - 1
        for tag in entry.context_tags:
            if tag not in self.memory_index:
                self.memory_index[tag] = []
            self.memory_index[tag].append(idx)

        # Prune if over limit
        if len(self.memories) > self.max_memories:
            self._prune_oldest()

        # Auto-save
        if self.storage_path:
            self._save()

    def recall_by_situation(
        self,
        current_situation: str,
        top_k: int = 5,
        embedding: torch.Tensor | None = None,
    ) -> list[MemoryEntry]:
        """
        Recall relevant memories for current situation.

        Uses embedding similarity if available, falls back to tag matching.
        """
        if embedding is not None and any(m.embedding is not None for m in self.memories):
            return self._recall_by_embedding(embedding, top_k)
        else:
            return self._recall_by_tags(current_situation, top_k)

    def _recall_by_embedding(
        self,
        query_embedding: torch.Tensor,
        top_k: int,
    ) -> list[MemoryEntry]:
        """Recall using embedding similarity."""
        similarities = []

        for i, memory in enumerate(self.memories):
            if memory.embedding is not None:
                # Cosine similarity
                sim = torch.cosine_similarity(
                    query_embedding.unsqueeze(0),
                    memory.embedding.unsqueeze(0),
                ).item()
                similarities.append((i, sim))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        return [self.memories[i] for i, _ in similarities[:top_k]]

    def _recall_by_tags(
        self,
        situation: str,
        top_k: int,
    ) -> list[MemoryEntry]:
        """Recall using tag/keyword matching."""
        words = set(situation.lower().split())

        scored_memories = []
        for i, memory in enumerate(self.memories):
            # Score by tag overlap
            memory_words = set(memory.situation.lower().split())
            memory_words.update(tag.lower() for tag in memory.context_tags)

            overlap = len(words & memory_words)
            if overlap > 0:
                scored_memories.append((i, overlap))

        # Sort by score
        scored_memories.sort(key=lambda x: x[1], reverse=True)

        return [self.memories[i] for i, _ in scored_memories[:top_k]]

    def get_behavioral_precedents(self, context_tag: str) -> list[MemoryEntry]:
        """Get all memories with a specific context tag."""
        indices = self.memory_index.get(context_tag, [])
        return [self.memories[i] for i in indices]

    def _prune_oldest(self) -> None:
        """Remove oldest memories to stay under limit."""
        # Sort by timestamp, keep newest
        self.memories.sort(key=lambda m: m.timestamp, reverse=True)
        self.memories = self.memories[: self.max_memories]

        # Rebuild index
        self.memory_index = {}
        for i, memory in enumerate(self.memories):
            for tag in memory.context_tags:
                if tag not in self.memory_index:
                    self.memory_index[tag] = []
                self.memory_index[tag].append(i)

    def _save(self) -> None:
        """Save memories to disk."""
        if not self.storage_path:
            return

        data = {
            "memories": [m.to_dict() for m in self.memories],
            "index": self.memory_index,
        }

        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, "w") as f:
            json.dump(data, f, indent=2)

    def _load(self) -> None:
        """Load memories from disk."""
        if not self.storage_path or not self.storage_path.exists():
            return

        with open(self.storage_path) as f:
            data = json.load(f)

        self.memories = [MemoryEntry.from_dict(m) for m in data.get("memories", [])]
        self.memory_index = data.get("index", {})


class CharacterCore:
    """
    The core character structure that ties everything together.

    This is the "self" of the agent - its values, traits, and memories
    working together to produce coherent, characteristic behavior.
    """

    def __init__(
        self,
        name: str = "Agent",
        description: str = "",
        storage_dir: Path | None = None,
    ):
        self.name = name
        self.description = description
        self.storage_dir = storage_dir

        # Core components
        self.values: dict[ValueType, ValueAnchor] = {}
        self.traits: dict[TraitDimension, BehavioralTrait] = {}
        self.memory = AutobiographicalMemory(
            storage_path=storage_dir / "memories.json" if storage_dir else None
        )

        # Identity hash for consistency verification
        self._identity_hash: str | None = None

        # Initialize with default character
        self._initialize_defaults()

    def _initialize_defaults(self) -> None:
        """Set up default values and traits."""
        # Core values
        self.values[ValueType.TRUTH_SEEKING] = ValueAnchor(
            value_type=ValueType.TRUTH_SEEKING,
            priority=0.9,
            description="Committed to accurate information and honest assessment",
            strength=0.8,
        )
        self.values[ValueType.HELPFULNESS] = ValueAnchor(
            value_type=ValueType.HELPFULNESS,
            priority=0.85,
            description="Genuinely wants to help users succeed",
            strength=0.9,
        )
        self.values[ValueType.INTELLECTUAL_HONESTY] = ValueAnchor(
            value_type=ValueType.INTELLECTUAL_HONESTY,
            priority=0.88,
            description="Acknowledges uncertainty and limitations",
            strength=0.85,
        )

        # Default traits (balanced)
        for dim in TraitDimension:
            self.traits[dim] = BehavioralTrait(
                dimension=dim,
                position=0.5,
                stability=0.7,
            )

    def set_value(
        self,
        value_type: ValueType,
        priority: float = 0.5,
        strength: float = 0.7,
        description: str = "",
        **kwargs,
    ) -> None:
        """Set or update a value anchor."""
        self.values[value_type] = ValueAnchor(
            value_type=value_type,
            priority=priority,
            strength=strength,
            description=description,
            **kwargs,
        )
        self._invalidate_hash()

    def set_trait(
        self,
        dimension: TraitDimension,
        position: float,
        stability: float = 0.7,
        context_modifiers: dict[str, float] | None = None,
    ) -> None:
        """Set or update a behavioral trait."""
        self.traits[dimension] = BehavioralTrait(
            dimension=dimension,
            position=position,
            stability=stability,
            context_modifiers=context_modifiers or {},
        )
        self._invalidate_hash()

    def get_active_values(self, context: str = "") -> list[ValueAnchor]:
        """Get values that apply in the current context, sorted by priority."""
        active = [v for v in self.values.values() if v.applies_in_context(context)]
        return sorted(active, key=lambda v: v.priority, reverse=True)

    def get_trait_profile(self, context: str = "") -> dict[TraitDimension, float]:
        """Get current trait positions for given context."""
        return {dim: trait.get_effective_position(context) for dim, trait in self.traits.items()}

    def resolve_value_conflict(
        self,
        value_a: ValueType,
        value_b: ValueType,
        context: str = "",
    ) -> ValueType:
        """
        Resolve a conflict between two values.

        Returns the value that should take precedence.
        """
        anchor_a = self.values.get(value_a)
        anchor_b = self.values.get(value_b)

        if anchor_a is None:
            return value_b
        if anchor_b is None:
            return value_a

        # Check for explicit resolution rules
        resolution = anchor_a.get_resolution(value_b)
        if resolution == "yield":
            return value_b
        if resolution == "override":
            return value_a

        resolution = anchor_b.get_resolution(value_a)
        if resolution == "yield":
            return value_a
        if resolution == "override":
            return value_b

        # Fall back to priority comparison
        # Adjust for context
        priority_a = anchor_a.priority
        priority_b = anchor_b.priority

        if anchor_a.applies_in_context(context):
            priority_a *= 1.1
        if anchor_b.applies_in_context(context):
            priority_b *= 1.1

        return value_a if priority_a >= priority_b else value_b

    def generate_character_prompt(self, context: str = "") -> str:
        """
        Generate a prompt that encodes this character's identity.

        This can be used to prime the model to behave consistently
        with the defined character.
        """
        lines = [f"CHARACTER: {self.name}"]

        if self.description:
            lines.append(f"\n{self.description}")

        # Values section
        active_values = self.get_active_values(context)
        if active_values:
            lines.append("\nCORE VALUES (in priority order):")
            for v in active_values[:5]:  # Top 5
                lines.append(f"• {v.value_type.value}: {v.description}")

        # Traits section
        trait_profile = self.get_trait_profile(context)
        lines.append("\nBEHAVIORAL TENDENCIES:")
        for dim, pos in trait_profile.items():
            trait = self.traits[dim]
            tendency = trait.get_behavior_tendency(context)
            lines.append(f"• {dim.value}: {tendency}")

        # Relevant memories
        memories = self.memory.recall_by_situation(context, top_k=3)
        if memories:
            lines.append("\nRELEVANT PAST EXPERIENCE:")
            for mem in memories:
                lines.append(f"• Situation: {mem.situation[:100]}...")
                lines.append(f"  Action: {mem.action_taken[:100]}...")
                lines.append(f"  Lesson: {mem.lesson}")

        return "\n".join(lines)

    def record_experience(
        self,
        situation: str,
        action_taken: str,
        outcome: str,
        lesson: str,
        context_tags: list[str] | None = None,
        was_positive: bool = True,
    ) -> None:
        """Record a new experience in autobiographical memory."""
        self.memory.add_memory(
            situation=situation,
            action_taken=action_taken,
            outcome=outcome,
            lesson=lesson,
            context_tags=context_tags,
            emotional_valence=0.5 if was_positive else -0.5,
        )

    def _invalidate_hash(self) -> None:
        """Invalidate identity hash when character changes."""
        self._identity_hash = None

    def get_identity_hash(self) -> str:
        """Get a hash representing current character state."""
        if self._identity_hash is None:
            state = {
                "name": self.name,
                "values": {
                    k.value: {"priority": v.priority, "strength": v.strength}
                    for k, v in self.values.items()
                },
                "traits": {
                    k.value: {"position": v.position, "stability": v.stability}
                    for k, v in self.traits.items()
                },
            }
            state_str = json.dumps(state, sort_keys=True)
            self._identity_hash = hashlib.sha256(state_str.encode()).hexdigest()[:16]

        return self._identity_hash

    def save(self, path: Path | None = None, allow_outside_storage: bool = False) -> None:
        """
        Save character to disk.

        Args:
            path: Path to save to. If None, uses storage_dir/character.json
            allow_outside_storage: If False (default), restricts saves to storage_dir
                                   to prevent path traversal attacks

        Raises:
            ValueError: If path is outside storage directory and allow_outside_storage=False
        """
        save_path = path or (self.storage_dir / "character.json" if self.storage_dir else None)
        if not save_path:
            raise ValueError("No save path specified")

        # Path traversal protection
        if not allow_outside_storage and self.storage_dir is not None and path is not None:
            try:
                # Resolve both paths and check if save_path is within storage_dir
                resolved_save = save_path.resolve()
                resolved_storage = self.storage_dir.resolve()
                resolved_save.relative_to(resolved_storage)
            except ValueError:
                raise ValueError(
                    f"Save path must be within storage directory '{self.storage_dir}'. "
                    f"Use allow_outside_storage=True to override."
                )

        data = {
            "name": self.name,
            "description": self.description,
            "values": {
                k.value: {
                    "priority": v.priority,
                    "strength": v.strength,
                    "description": v.description,
                    "activation_contexts": v.activation_contexts,
                }
                for k, v in self.values.items()
            },
            "traits": {
                k.value: {
                    "position": v.position,
                    "stability": v.stability,
                    "context_modifiers": v.context_modifiers,
                }
                for k, v in self.traits.items()
            },
        }

        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path, max_values: int = 100, max_traits: int = 50) -> CharacterCore:
        """
        Load character from disk with validation.

        Args:
            path: Path to character JSON file
            max_values: Maximum number of values to load (prevents resource exhaustion)
            max_traits: Maximum number of traits to load (prevents resource exhaustion)

        Returns:
            Loaded CharacterCore instance

        Raises:
            ValueError: If file contains invalid data
            FileNotFoundError: If file doesn't exist
        """
        with open(path) as f:
            data = json.load(f)

        # Validate top-level structure
        if not isinstance(data, dict):
            raise ValueError("Character file must contain a JSON object")

        char = cls(
            name=str(data.get("name", "Agent"))[:256],  # Limit name length
            description=str(data.get("description", ""))[:4096],  # Limit description
            storage_dir=path.parent,
        )

        # Load values with validation
        values_data = data.get("values", {})
        if not isinstance(values_data, dict):
            raise ValueError("'values' must be a dictionary")

        valid_value_names = {v.value for v in ValueType}
        loaded_values = 0

        for value_name, value_data in values_data.items():
            if loaded_values >= max_values:
                break  # Prevent resource exhaustion

            # Validate value type
            if value_name not in valid_value_names:
                continue  # Skip invalid value types silently

            if not isinstance(value_data, dict):
                continue  # Skip malformed entries

            try:
                value_type = ValueType(value_name)
                # Validate and sanitize value_data keys
                safe_data = {
                    k: v
                    for k, v in value_data.items()
                    if k in {"priority", "strength", "description", "contexts", "conflicts_with"}
                    and (isinstance(v, (int, float, str, list)) or v is None)
                }
                char.set_value(value_type, **safe_data)
                loaded_values += 1
            except (ValueError, TypeError):
                # Log and continue on invalid data
                continue

        # Load traits with validation
        traits_data = data.get("traits", {})
        if not isinstance(traits_data, dict):
            raise ValueError("'traits' must be a dictionary")

        valid_trait_names = {t.value for t in TraitDimension}
        loaded_traits = 0

        for trait_name, trait_data in traits_data.items():
            if loaded_traits >= max_traits:
                break  # Prevent resource exhaustion

            # Validate trait dimension
            if trait_name not in valid_trait_names:
                continue  # Skip invalid trait types silently

            if not isinstance(trait_data, dict):
                continue  # Skip malformed entries

            try:
                trait_dim = TraitDimension(trait_name)
                # Validate and sanitize trait_data keys
                safe_data = {
                    k: v
                    for k, v in trait_data.items()
                    if k in {"position", "stability", "context_modifiers"}
                    and (isinstance(v, (int, float, dict)) or v is None)
                }
                char.set_trait(trait_dim, **safe_data)
                loaded_traits += 1
            except (ValueError, TypeError):
                # Log and continue on invalid data
                continue

        return char


# Pre-built character templates
def create_socratic_character() -> CharacterCore:
    """Create a character optimized for Socratic teaching."""
    char = CharacterCore(
        name="Socratic Guide",
        description="A patient, questioning teacher who helps through inquiry rather than answers.",
    )

    # Values emphasizing questioning
    char.set_value(
        ValueType.TRUTH_SEEKING,
        priority=0.95,
        strength=0.9,
        description="Truth must be discovered, not told",
    )
    char.set_value(
        ValueType.INTELLECTUAL_HONESTY,
        priority=0.9,
        strength=0.85,
        description="Admitting ignorance is the beginning of wisdom",
    )

    # Traits for Socratic method
    char.set_trait(TraitDimension.SUPPORTIVE_CHALLENGING, position=0.7)  # More challenging
    char.set_trait(TraitDimension.VERBOSE_CONCISE, position=0.3)  # More concise (questions)
    char.set_trait(TraitDimension.ANALYTICAL_INTUITIVE, position=0.6)  # Somewhat analytical

    return char


def create_compassionate_character() -> CharacterCore:
    """Create a character optimized for empathetic support."""
    char = CharacterCore(
        name="Compassionate Guide",
        description="A warm, supportive helper who balances emotional awareness with practical help.",
    )

    # Values emphasizing care
    char.set_value(
        ValueType.HELPFULNESS,
        priority=0.95,
        strength=0.9,
        description="Genuinely invested in the user's wellbeing and success",
    )
    char.set_value(
        ValueType.WARMTH,
        priority=0.85,
        strength=0.8,
        description="Creates a safe, encouraging environment",
    )
    char.set_value(
        ValueType.PATIENCE,
        priority=0.8,
        strength=0.85,
        description="Allows learning and growth at the user's pace",
    )

    # Traits for compassionate interaction
    char.set_trait(TraitDimension.EMPATHETIC_DETACHED, position=0.8)  # Very empathetic
    char.set_trait(TraitDimension.SUPPORTIVE_CHALLENGING, position=0.3)  # More supportive
    char.set_trait(TraitDimension.FORMAL_CASUAL, position=0.6)  # Somewhat casual/warm

    return char


def create_analytical_character() -> CharacterCore:
    """Create a character optimized for rigorous analysis."""
    char = CharacterCore(
        name="Analytical Partner",
        description="A precise, evidence-focused thinker who prioritizes accuracy and rigor.",
    )

    # Values emphasizing rigor
    char.set_value(
        ValueType.EVIDENCE_BASED,
        priority=0.95,
        strength=0.9,
        description="Claims must be supported by evidence",
    )
    char.set_value(
        ValueType.THOROUGHNESS,
        priority=0.85,
        strength=0.85,
        description="Comprehensive analysis over quick conclusions",
    )
    char.set_value(
        ValueType.CLARITY,
        priority=0.8,
        strength=0.8,
        description="Precise communication, no ambiguity",
    )

    # Traits for analytical work
    char.set_trait(TraitDimension.ANALYTICAL_INTUITIVE, position=0.9)  # Very analytical
    char.set_trait(TraitDimension.CAUTIOUS_BOLD, position=0.3)  # More cautious
    char.set_trait(TraitDimension.VERBOSE_CONCISE, position=0.6)  # Somewhat verbose (thorough)

    return char
