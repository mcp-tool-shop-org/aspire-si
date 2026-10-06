"""
Controlled Chaos Module - Adversarial training for robust perception.

The insight: Agents trained on clean, predictable inputs become brittle.
Real-world inputs are messy - ambiguous, contradictory, incomplete.
This module deliberately introduces controlled chaos during training
to build genuine robustness.

Types of chaos:
1. Noise Injection - Corrupted/partial information
2. Ambiguity Generation - Multiple valid interpretations
3. Adversarial Scenarios - Designed to exploit common failure modes
4. Contradiction Injection - Conflicting information sources

This is "chaos with intention" - each perturbation has a learning objective.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ChaosType(str, Enum):
    """Types of chaos that can be injected."""

    # Information degradation
    MISSING_CONTEXT = "missing_context"
    PARTIAL_INFORMATION = "partial_information"
    NOISY_INPUT = "noisy_input"
    TRUNCATED_INPUT = "truncated_input"

    # Semantic ambiguity
    AMBIGUOUS_REFERENCE = "ambiguous_reference"
    UNCLEAR_INTENT = "unclear_intent"
    MULTIPLE_INTERPRETATIONS = "multiple_interpretations"
    IMPLICIT_REQUIREMENTS = "implicit_requirements"

    # Contradiction and conflict
    CONTRADICTORY_INSTRUCTIONS = "contradictory_instructions"
    CONFLICTING_CONSTRAINTS = "conflicting_constraints"
    INCONSISTENT_CONTEXT = "inconsistent_context"

    # Social/emotional complexity
    EMOTIONAL_SUBTEXT = "emotional_subtext"
    HIDDEN_AGENDA = "hidden_agenda"
    POLITENESS_VS_DIRECTNESS = "politeness_vs_directness"

    # Meta-level
    MISLEADING_FRAMING = "misleading_framing"
    ASSUMPTION_TRAP = "assumption_trap"
    SCOPE_CREEP = "scope_creep"


class ChaosSeverity(str, Enum):
    """How severe the chaos should be."""

    SUBTLE = "subtle"  # Barely noticeable, tests attention
    MODERATE = "moderate"  # Clearly present, requires handling
    SEVERE = "severe"  # Major challenge, tests robustness limits


@dataclass
class ChaosConfig:
    """Configuration for chaos generation."""

    # Which types of chaos to enable
    enabled_types: list[ChaosType] = field(default_factory=lambda: list(ChaosType))

    # Probability of applying chaos to a given sample
    chaos_probability: float = 0.3

    # Severity distribution
    severity_weights: dict[ChaosSeverity, float] = field(
        default_factory=lambda: {
            ChaosSeverity.SUBTLE: 0.5,
            ChaosSeverity.MODERATE: 0.35,
            ChaosSeverity.SEVERE: 0.15,
        }
    )

    # Curriculum: increase chaos over training
    curriculum_enabled: bool = True
    curriculum_start_epoch: int = 1
    curriculum_ramp_epochs: int = 3


@dataclass
class ChaosInjection:
    """Record of a single chaos injection."""

    chaos_type: ChaosType
    severity: ChaosSeverity
    original_input: str
    modified_input: str
    ground_truth: str  # What the correct handling should be
    learning_objective: str  # What this tests


class BaseChaosGenerator(ABC):
    """Base class for chaos generators."""

    @abstractmethod
    def generate(
        self,
        input_text: str,
        severity: ChaosSeverity = ChaosSeverity.MODERATE,
        **kwargs,
    ) -> ChaosInjection:
        """Generate a chaos injection for the given input."""
        pass

    @abstractmethod
    def get_chaos_types(self) -> list[ChaosType]:
        """Return the types of chaos this generator produces."""
        pass


class NoiseInjector(BaseChaosGenerator):
    """
    Injects noise and information degradation.

    Tests: Attention to detail, robustness to imperfect input, error recovery.
    """

    def get_chaos_types(self) -> list[ChaosType]:
        return [
            ChaosType.MISSING_CONTEXT,
            ChaosType.PARTIAL_INFORMATION,
            ChaosType.NOISY_INPUT,
            ChaosType.TRUNCATED_INPUT,
        ]

    def generate(
        self,
        input_text: str,
        severity: ChaosSeverity = ChaosSeverity.MODERATE,
        **kwargs,
    ) -> ChaosInjection:
        chaos_type = random.choice(self.get_chaos_types())

        if chaos_type == ChaosType.MISSING_CONTEXT:
            return self._missing_context(input_text, severity)
        elif chaos_type == ChaosType.PARTIAL_INFORMATION:
            return self._partial_information(input_text, severity)
        elif chaos_type == ChaosType.NOISY_INPUT:
            return self._noisy_input(input_text, severity)
        else:
            return self._truncated_input(input_text, severity)

    def _missing_context(
        self,
        input_text: str,
        severity: ChaosSeverity,
    ) -> ChaosInjection:
        """Remove contextual information that would normally be present."""
        sentences = input_text.split(". ")

        # Remove sentences that provide context
        if severity == ChaosSeverity.SUBTLE:
            # Remove one contextual phrase
            if len(sentences) > 2:
                remove_idx = random.randint(0, len(sentences) - 2)
                sentences.pop(remove_idx)
        elif severity == ChaosSeverity.MODERATE:
            # Remove 30% of context
            num_remove = max(1, len(sentences) // 3)
            for _ in range(num_remove):
                if len(sentences) > 1:
                    sentences.pop(random.randint(0, len(sentences) - 1))
        else:  # SEVERE
            # Keep only the core request
            if len(sentences) > 1:
                sentences = [sentences[-1]]

        modified = ". ".join(sentences)
        if not modified.endswith("."):
            modified += "."

        return ChaosInjection(
            chaos_type=ChaosType.MISSING_CONTEXT,
            severity=severity,
            original_input=input_text,
            modified_input=modified,
            ground_truth="Ask for clarification about missing context",
            learning_objective="Learn to detect when context is insufficient and ask for it",
        )

    def _partial_information(
        self,
        input_text: str,
        severity: ChaosSeverity,
    ) -> ChaosInjection:
        """Provide incomplete information that needs to be requested."""
        # Add markers indicating missing info
        placeholders = {
            ChaosSeverity.SUBTLE: ["[...]", "(details omitted)"],
            ChaosSeverity.MODERATE: ["[REDACTED]", "(see attached)", "[missing]"],
            ChaosSeverity.SEVERE: ["???", "[ERROR: DATA NOT FOUND]", "(corrupted)"],
        }

        words = input_text.split()
        num_placeholders = {
            ChaosSeverity.SUBTLE: 1,
            ChaosSeverity.MODERATE: 2,
            ChaosSeverity.SEVERE: max(3, len(words) // 10),
        }[severity]

        for _ in range(num_placeholders):
            if len(words) > 3:
                idx = random.randint(1, len(words) - 2)
                words[idx] = random.choice(placeholders[severity])

        modified = " ".join(words)

        return ChaosInjection(
            chaos_type=ChaosType.PARTIAL_INFORMATION,
            severity=severity,
            original_input=input_text,
            modified_input=modified,
            ground_truth="Identify what information is missing and request it",
            learning_objective="Learn to work with incomplete data and identify gaps",
        )

    def _noisy_input(
        self,
        input_text: str,
        severity: ChaosSeverity,
    ) -> ChaosInjection:
        """Add noise (typos, formatting issues, random characters)."""
        modified = list(input_text)

        noise_rate = {
            ChaosSeverity.SUBTLE: 0.02,
            ChaosSeverity.MODERATE: 0.05,
            ChaosSeverity.SEVERE: 0.10,
        }[severity]

        for i in range(len(modified)):
            if random.random() < noise_rate:
                noise_type = random.choice(["typo", "swap", "insert", "delete"])
                if noise_type == "typo" and modified[i].isalpha():
                    modified[i] = random.choice("abcdefghijklmnopqrstuvwxyz")
                elif noise_type == "swap" and i < len(modified) - 1:
                    modified[i], modified[i + 1] = modified[i + 1], modified[i]
                elif noise_type == "insert":
                    modified.insert(i, random.choice(".,;:!? "))
                elif noise_type == "delete":
                    modified[i] = ""

        return ChaosInjection(
            chaos_type=ChaosType.NOISY_INPUT,
            severity=severity,
            original_input=input_text,
            modified_input="".join(modified),
            ground_truth="Parse through noise to understand core intent",
            learning_objective="Build robustness to input imperfections",
        )

    def _truncated_input(
        self,
        input_text: str,
        severity: ChaosSeverity,
    ) -> ChaosInjection:
        """Truncate input mid-thought."""
        truncate_ratio = {
            ChaosSeverity.SUBTLE: 0.9,
            ChaosSeverity.MODERATE: 0.7,
            ChaosSeverity.SEVERE: 0.4,
        }[severity]

        cutoff = int(len(input_text) * truncate_ratio)
        modified = input_text[:cutoff]

        # Add truncation indicator
        if severity == ChaosSeverity.SEVERE:
            modified += "..."

        return ChaosInjection(
            chaos_type=ChaosType.TRUNCATED_INPUT,
            severity=severity,
            original_input=input_text,
            modified_input=modified,
            ground_truth="Recognize truncation and request complete information",
            learning_objective="Detect incomplete input and handle gracefully",
        )


class AmbiguityGenerator(BaseChaosGenerator):
    """
    Generates semantically ambiguous inputs.

    Tests: Clarification skills, assumption awareness, interpretation flexibility.
    """

    def get_chaos_types(self) -> list[ChaosType]:
        return [
            ChaosType.AMBIGUOUS_REFERENCE,
            ChaosType.UNCLEAR_INTENT,
            ChaosType.MULTIPLE_INTERPRETATIONS,
            ChaosType.IMPLICIT_REQUIREMENTS,
        ]

    def generate(
        self,
        input_text: str,
        severity: ChaosSeverity = ChaosSeverity.MODERATE,
        **kwargs,
    ) -> ChaosInjection:
        chaos_type = random.choice(self.get_chaos_types())

        if chaos_type == ChaosType.AMBIGUOUS_REFERENCE:
            return self._ambiguous_reference(input_text, severity)
        elif chaos_type == ChaosType.UNCLEAR_INTENT:
            return self._unclear_intent(input_text, severity)
        elif chaos_type == ChaosType.MULTIPLE_INTERPRETATIONS:
            return self._multiple_interpretations(input_text, severity)
        else:
            return self._implicit_requirements(input_text, severity)

    def _ambiguous_reference(
        self,
        input_text: str,
        severity: ChaosSeverity,
    ) -> ChaosInjection:
        """Replace specific references with ambiguous ones."""
        ambiguous_refs = {
            ChaosSeverity.SUBTLE: ["it", "that", "this"],
            ChaosSeverity.MODERATE: ["the thing", "that part", "it"],
            ChaosSeverity.SEVERE: ["you know what", "that stuff", "the usual"],
        }

        # Find nouns/references to replace
        words = input_text.split()
        modified_words = []

        for word in words:
            # Simple heuristic: replace capitalized words or long words
            if (len(word) > 6 and word[0].isupper()) or random.random() < 0.1:
                if severity != ChaosSeverity.SUBTLE or random.random() < 0.3:
                    modified_words.append(random.choice(ambiguous_refs[severity]))
                else:
                    modified_words.append(word)
            else:
                modified_words.append(word)

        return ChaosInjection(
            chaos_type=ChaosType.AMBIGUOUS_REFERENCE,
            severity=severity,
            original_input=input_text,
            modified_input=" ".join(modified_words),
            ground_truth="Ask for clarification about ambiguous references",
            learning_objective="Detect and resolve referential ambiguity",
        )

    def _unclear_intent(
        self,
        input_text: str,
        severity: ChaosSeverity,
    ) -> ChaosInjection:
        """Make the user's intent less clear."""
        unclear_prefixes = {
            ChaosSeverity.SUBTLE: [
                "I was wondering about ",
                "What do you think about ",
            ],
            ChaosSeverity.MODERATE: [
                "So there's this thing with ",
                "I'm not sure but maybe ",
                "Something about ",
            ],
            ChaosSeverity.SEVERE: [
                "Hmm, ",
                "Well... ",
                "You know... ",
            ],
        }

        unclear_suffixes = {
            ChaosSeverity.SUBTLE: ["?", " I guess?"],
            ChaosSeverity.MODERATE: [" or something?", " maybe?", " idk"],
            ChaosSeverity.SEVERE: ["...", " yeah", " ???"],
        }

        prefix = random.choice(unclear_prefixes[severity])
        suffix = random.choice(unclear_suffixes[severity])

        # Remove clear intent markers from original
        modified = input_text.lower()
        for clear_word in ["please", "i need", "i want", "help me", "can you"]:
            modified = modified.replace(clear_word, "")

        modified = prefix + modified.strip() + suffix

        return ChaosInjection(
            chaos_type=ChaosType.UNCLEAR_INTENT,
            severity=severity,
            original_input=input_text,
            modified_input=modified,
            ground_truth="Clarify what the user actually wants",
            learning_objective="Extract clear intent from unclear communication",
        )

    def _multiple_interpretations(
        self,
        input_text: str,
        severity: ChaosSeverity,
    ) -> ChaosInjection:
        """Create input that could reasonably mean multiple things."""
        multi_interp_phrases = {
            ChaosSeverity.SUBTLE: [
                " (or the other way around)",
                " (roughly speaking)",
            ],
            ChaosSeverity.MODERATE: [
                " - unless you think otherwise",
                " (or did you mean something else?)",
                " - though it could be interpreted differently",
            ],
            ChaosSeverity.SEVERE: [
                " - but that's just one way to look at it. Or maybe I mean the opposite?",
                " (interpret this however makes sense to you)",
            ],
        }

        suffix = random.choice(multi_interp_phrases[severity])
        modified = input_text.rstrip(".!?") + suffix

        return ChaosInjection(
            chaos_type=ChaosType.MULTIPLE_INTERPRETATIONS,
            severity=severity,
            original_input=input_text,
            modified_input=modified,
            ground_truth="Present multiple interpretations and ask which is intended",
            learning_objective="Handle semantic ambiguity explicitly",
        )

    def _implicit_requirements(
        self,
        input_text: str,
        severity: ChaosSeverity,
    ) -> ChaosInjection:
        """Hide requirements that should be made explicit."""
        # Remove explicit requirements and make them implicit
        implicit_markers = {
            ChaosSeverity.SUBTLE: " (you know what I mean)",
            ChaosSeverity.MODERATE: " (the usual requirements apply)",
            ChaosSeverity.SEVERE: " (figure out what I need)",
        }

        # Strip away requirement-like phrases
        modified = input_text
        for phrase in ["must be", "should be", "needs to", "has to", "require"]:
            if phrase in modified.lower():
                idx = modified.lower().find(phrase)
                # Find end of sentence/clause
                end_idx = modified.find(".", idx)
                if end_idx == -1:
                    end_idx = modified.find(",", idx)
                if end_idx == -1:
                    end_idx = len(modified)
                modified = modified[:idx] + modified[end_idx:]

        modified = modified.strip() + implicit_markers[severity]

        return ChaosInjection(
            chaos_type=ChaosType.IMPLICIT_REQUIREMENTS,
            severity=severity,
            original_input=input_text,
            modified_input=modified,
            ground_truth="Identify and explicitly state implicit requirements",
            learning_objective="Surface hidden assumptions and requirements",
        )


class AdversarialScenarioGenerator(BaseChaosGenerator):
    """
    Generates adversarial scenarios that exploit common failure modes.

    Tests: Edge case handling, contradiction resolution, robustness under pressure.
    """

    def get_chaos_types(self) -> list[ChaosType]:
        return [
            ChaosType.CONTRADICTORY_INSTRUCTIONS,
            ChaosType.CONFLICTING_CONSTRAINTS,
            ChaosType.MISLEADING_FRAMING,
            ChaosType.ASSUMPTION_TRAP,
        ]

    def generate(
        self,
        input_text: str,
        severity: ChaosSeverity = ChaosSeverity.MODERATE,
        **kwargs,
    ) -> ChaosInjection:
        chaos_type = random.choice(self.get_chaos_types())

        if chaos_type == ChaosType.CONTRADICTORY_INSTRUCTIONS:
            return self._contradictory_instructions(input_text, severity)
        elif chaos_type == ChaosType.CONFLICTING_CONSTRAINTS:
            return self._conflicting_constraints(input_text, severity)
        elif chaos_type == ChaosType.MISLEADING_FRAMING:
            return self._misleading_framing(input_text, severity)
        else:
            return self._assumption_trap(input_text, severity)

    def _contradictory_instructions(
        self,
        input_text: str,
        severity: ChaosSeverity,
    ) -> ChaosInjection:
        """Add contradicting instructions."""
        contradictions = {
            ChaosSeverity.SUBTLE: [
                " But also keep it simple.",
                " Though brevity is important.",
            ],
            ChaosSeverity.MODERATE: [
                " Actually, on second thought, do the opposite.",
                " But wait, that might not be right either.",
                " Unless that doesn't make sense, then do something else.",
            ],
            ChaosSeverity.SEVERE: [
                " Do this AND don't do this at the same time.",
                " I want X but I also want not-X.",
                " Make it longer but also shorter.",
            ],
        }

        contradiction = random.choice(contradictions[severity])
        modified = input_text.rstrip(".!?") + "." + contradiction

        return ChaosInjection(
            chaos_type=ChaosType.CONTRADICTORY_INSTRUCTIONS,
            severity=severity,
            original_input=input_text,
            modified_input=modified,
            ground_truth="Identify the contradiction and ask for clarification",
            learning_objective="Detect and resolve logical contradictions",
        )

    def _conflicting_constraints(
        self,
        input_text: str,
        severity: ChaosSeverity,
    ) -> ChaosInjection:
        """Add mutually exclusive constraints."""
        conflicts = {
            ChaosSeverity.SUBTLE: [
                " It needs to be done quickly but also thoroughly.",
            ],
            ChaosSeverity.MODERATE: [
                " Requirements: fast, cheap, and high quality.",
                " Must be both maximally secure and maximally convenient.",
            ],
            ChaosSeverity.SEVERE: [
                " Constraints: zero budget, enterprise features, yesterday's deadline.",
                " Requirements: perfect accuracy, zero latency, runs on a calculator.",
            ],
        }

        constraint = random.choice(conflicts[severity])
        modified = input_text + constraint

        return ChaosInjection(
            chaos_type=ChaosType.CONFLICTING_CONSTRAINTS,
            severity=severity,
            original_input=input_text,
            modified_input=modified,
            ground_truth="Acknowledge constraint conflicts and negotiate priorities",
            learning_objective="Handle impossible constraints gracefully",
        )

    def _misleading_framing(
        self,
        input_text: str,
        severity: ChaosSeverity,
    ) -> ChaosInjection:
        """Frame the question in a misleading way."""
        misleading_frames = {
            ChaosSeverity.SUBTLE: [
                "Since everyone agrees that ",
                "Given the obvious fact that ",
            ],
            ChaosSeverity.MODERATE: [
                "Now that we've established [unestablished thing], ",
                "Building on what we discussed earlier [nothing discussed], ",
            ],
            ChaosSeverity.SEVERE: [
                "As you already confirmed [never confirmed], ",
                "Since you said [never said] earlier, ",
            ],
        }

        frame = random.choice(misleading_frames[severity])
        modified = frame + input_text[0].lower() + input_text[1:]

        return ChaosInjection(
            chaos_type=ChaosType.MISLEADING_FRAMING,
            severity=severity,
            original_input=input_text,
            modified_input=modified,
            ground_truth="Correct the false premise before proceeding",
            learning_objective="Detect and reject misleading framing",
        )

    def _assumption_trap(
        self,
        input_text: str,
        severity: ChaosSeverity,
    ) -> ChaosInjection:
        """Create input that tempts the model to make unwarranted assumptions."""
        traps = {
            ChaosSeverity.SUBTLE: [
                " (the standard approach should work)",
                " (just like the usual case)",
            ],
            ChaosSeverity.MODERATE: [
                " This is a typical case so don't overthink it.",
                " No need to consider edge cases here.",
            ],
            ChaosSeverity.SEVERE: [
                " Don't worry about validation or error cases.",
                " Assume everything will work perfectly.",
                " Skip the boring safety considerations.",
            ],
        }

        trap = random.choice(traps[severity])
        modified = input_text + trap

        return ChaosInjection(
            chaos_type=ChaosType.ASSUMPTION_TRAP,
            severity=severity,
            original_input=input_text,
            modified_input=modified,
            ground_truth="Still consider important cases despite being told not to",
            learning_objective="Maintain appropriate caution despite pressure",
        )


class ChaosGenerator:
    """
    Main chaos orchestrator that combines all generators.

    Manages chaos injection during training, including curriculum ramping.
    """

    def __init__(self, config: ChaosConfig | None = None):
        self.config = config or ChaosConfig()

        # Initialize all generators
        self.generators: list[BaseChaosGenerator] = [
            NoiseInjector(),
            AmbiguityGenerator(),
            AdversarialScenarioGenerator(),
        ]

        # Build type -> generator mapping
        self.type_to_generator: dict[ChaosType, BaseChaosGenerator] = {}
        for gen in self.generators:
            for chaos_type in gen.get_chaos_types():
                self.type_to_generator[chaos_type] = gen

        # Training state
        self.current_epoch = 0
        self.injection_count = 0

    def should_inject_chaos(self) -> bool:
        """Determine if chaos should be injected for current sample."""
        base_prob = self.config.chaos_probability

        if self.config.curriculum_enabled:
            # Ramp up chaos over training
            if self.current_epoch < self.config.curriculum_start_epoch:
                return False

            ramp_progress = min(
                1.0,
                (self.current_epoch - self.config.curriculum_start_epoch)
                / self.config.curriculum_ramp_epochs,
            )
            adjusted_prob = base_prob * ramp_progress
        else:
            adjusted_prob = base_prob

        return random.random() < adjusted_prob

    def select_severity(self) -> ChaosSeverity:
        """Select chaos severity based on configured weights."""
        weights = self.config.severity_weights
        total = sum(weights.values())
        r = random.random() * total

        cumsum = 0.0
        for severity, weight in weights.items():
            cumsum += weight
            if r <= cumsum:
                return severity

        return ChaosSeverity.MODERATE  # Default

    def inject(
        self,
        input_text: str,
        chaos_type: ChaosType | None = None,
        severity: ChaosSeverity | None = None,
    ) -> ChaosInjection | None:
        """
        Inject chaos into input if appropriate.

        Args:
            input_text: Original input text
            chaos_type: Specific type to inject (or random if None)
            severity: Specific severity (or weighted random if None)

        Returns:
            ChaosInjection if chaos was applied, None otherwise
        """
        if not self.should_inject_chaos():
            return None

        # Select chaos type
        if chaos_type is None:
            enabled = self.config.enabled_types
            chaos_type = random.choice(enabled)

        # Select severity
        if severity is None:
            severity = self.select_severity()

        # Get appropriate generator
        generator = self.type_to_generator.get(chaos_type)
        if generator is None:
            return None

        # Generate chaos
        injection = generator.generate(input_text, severity)
        self.injection_count += 1

        return injection

    def set_epoch(self, epoch: int) -> None:
        """Update current epoch for curriculum adjustment."""
        self.current_epoch = epoch

    def get_stats(self) -> dict[str, Any]:
        """Get chaos injection statistics."""
        return {
            "total_injections": self.injection_count,
            "current_epoch": self.current_epoch,
            "curriculum_enabled": self.config.curriculum_enabled,
            "base_probability": self.config.chaos_probability,
        }


# Convenience function for training integration
def apply_chaos_to_batch(
    batch: list[str],
    chaos_generator: ChaosGenerator,
) -> tuple[list[str], list[ChaosInjection | None]]:
    """
    Apply chaos to a batch of inputs.

    Returns modified inputs and injection records (None if no chaos applied).
    """
    modified_inputs = []
    injections = []

    for text in batch:
        injection = chaos_generator.inject(text)
        if injection:
            modified_inputs.append(injection.modified_input)
        else:
            modified_inputs.append(text)
        injections.append(injection)

    return modified_inputs, injections
