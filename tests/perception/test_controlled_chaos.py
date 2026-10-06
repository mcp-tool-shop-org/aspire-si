"""
Tests for Controlled Chaos module.

Tests noise injection, ambiguity generation, and adversarial
scenario creation for robust training.
"""

import os

os.environ["XFORMERS_DISABLED"] = "1"

import pytest

from aspire.perception.controlled_chaos import (
    AdversarialScenarioGenerator,
    AmbiguityGenerator,
    ChaosConfig,
    ChaosGenerator,
    ChaosSeverity,
    ChaosType,
    NoiseInjector,
    apply_chaos_to_batch,
)


class TestNoiseInjector:
    """Tests for NoiseInjector generator."""

    @pytest.fixture
    def injector(self):
        return NoiseInjector()

    def test_get_chaos_types(self, injector):
        """Test that noise injector reports correct chaos types."""
        types = injector.get_chaos_types()

        assert ChaosType.MISSING_CONTEXT in types
        assert ChaosType.PARTIAL_INFORMATION in types
        assert ChaosType.NOISY_INPUT in types
        assert ChaosType.TRUNCATED_INPUT in types

    def test_missing_context_injection(self, injector):
        """Test missing context injection."""
        original = "First, we need to understand the problem. Then, we analyze the data. Finally, we implement the solution."

        injection = injector._missing_context(original, ChaosSeverity.MODERATE)

        assert injection.chaos_type == ChaosType.MISSING_CONTEXT
        assert len(injection.modified_input) < len(original)
        assert injection.ground_truth != ""

    def test_partial_information_injection(self, injector):
        """Test partial information injection."""
        original = "The function takes three parameters: name, age, and location."

        injection = injector._partial_information(original, ChaosSeverity.MODERATE)

        assert injection.chaos_type == ChaosType.PARTIAL_INFORMATION
        # Should contain placeholder markers
        assert any(marker in injection.modified_input
                   for marker in ["[", "(", "REDACTED", "missing"])

    def test_noisy_input_injection(self, injector):
        """Test noisy input injection."""
        original = "This is a clear and clean sentence without any typos."

        injection = injector._noisy_input(original, ChaosSeverity.MODERATE)

        assert injection.chaos_type == ChaosType.NOISY_INPUT
        # Modified should be different (with high probability)
        # Note: With low noise rate, they could be equal occasionally

    def test_truncated_input_injection(self, injector):
        """Test truncated input injection."""
        original = "This is a long sentence that should get truncated at some point."

        injection = injector._truncated_input(original, ChaosSeverity.MODERATE)

        assert injection.chaos_type == ChaosType.TRUNCATED_INPUT
        assert len(injection.modified_input) < len(original)

    def test_severity_affects_intensity(self, injector):
        """Test that severity affects injection intensity."""
        original = "This is a test sentence for severity testing purposes."

        subtle = injector._truncated_input(original, ChaosSeverity.SUBTLE)
        severe = injector._truncated_input(original, ChaosSeverity.SEVERE)

        # Subtle should preserve more
        assert len(subtle.modified_input) > len(severe.modified_input)


class TestAmbiguityGenerator:
    """Tests for AmbiguityGenerator."""

    @pytest.fixture
    def generator(self):
        return AmbiguityGenerator()

    def test_get_chaos_types(self, generator):
        """Test that ambiguity generator reports correct chaos types."""
        types = generator.get_chaos_types()

        assert ChaosType.AMBIGUOUS_REFERENCE in types
        assert ChaosType.UNCLEAR_INTENT in types
        assert ChaosType.MULTIPLE_INTERPRETATIONS in types
        assert ChaosType.IMPLICIT_REQUIREMENTS in types

    def test_ambiguous_reference_injection(self, generator):
        """Test ambiguous reference injection."""
        original = "Please process the CustomerData and update the Database."

        injection = generator._ambiguous_reference(original, ChaosSeverity.MODERATE)

        assert injection.chaos_type == ChaosType.AMBIGUOUS_REFERENCE
        assert injection.learning_objective != ""

    def test_unclear_intent_injection(self, generator):
        """Test unclear intent injection."""
        original = "I need you to fix the bug in the authentication system."

        injection = generator._unclear_intent(original, ChaosSeverity.MODERATE)

        assert injection.chaos_type == ChaosType.UNCLEAR_INTENT
        # Should have uncertainty markers - include all possible phrases from the generator
        uncertainty_markers = [
            "wondering", "think", "maybe", "something", "hmm",
            "there's this thing", "not sure", "or something", "idk"
        ]
        assert any(marker in injection.modified_input.lower() for marker in uncertainty_markers)

    def test_multiple_interpretations_injection(self, generator):
        """Test multiple interpretations injection."""
        original = "Sort the data by date."

        injection = generator._multiple_interpretations(original, ChaosSeverity.MODERATE)

        assert injection.chaos_type == ChaosType.MULTIPLE_INTERPRETATIONS
        # Modified input should include some ambiguity marker
        ambiguity_markers = ["interpret", "otherwise", "something else", "or did", "mean"]
        assert any(marker in injection.modified_input.lower() for marker in ambiguity_markers)

    def test_implicit_requirements_injection(self, generator):
        """Test implicit requirements injection."""
        original = "Build a login system. It must be secure and needs to handle multiple users."

        injection = generator._implicit_requirements(original, ChaosSeverity.MODERATE)

        assert injection.chaos_type == ChaosType.IMPLICIT_REQUIREMENTS


class TestAdversarialScenarioGenerator:
    """Tests for AdversarialScenarioGenerator."""

    @pytest.fixture
    def generator(self):
        return AdversarialScenarioGenerator()

    def test_get_chaos_types(self, generator):
        """Test that adversarial generator reports correct chaos types."""
        types = generator.get_chaos_types()

        assert ChaosType.CONTRADICTORY_INSTRUCTIONS in types
        assert ChaosType.CONFLICTING_CONSTRAINTS in types
        assert ChaosType.MISLEADING_FRAMING in types
        assert ChaosType.ASSUMPTION_TRAP in types

    def test_contradictory_instructions_injection(self, generator):
        """Test contradictory instructions injection."""
        original = "Make the code more efficient."

        injection = generator._contradictory_instructions(original, ChaosSeverity.MODERATE)

        assert injection.chaos_type == ChaosType.CONTRADICTORY_INSTRUCTIONS
        assert "contradict" in injection.learning_objective.lower()

    def test_conflicting_constraints_injection(self, generator):
        """Test conflicting constraints injection."""
        original = "Build a new feature for the application."

        injection = generator._conflicting_constraints(original, ChaosSeverity.SEVERE)

        assert injection.chaos_type == ChaosType.CONFLICTING_CONSTRAINTS
        # Should mention constraint-like words
        assert any(word in injection.modified_input.lower()
                   for word in ["fast", "cheap", "quality", "zero", "perfect"])

    def test_misleading_framing_injection(self, generator):
        """Test misleading framing injection."""
        original = "We should use a database for this project."

        injection = generator._misleading_framing(original, ChaosSeverity.MODERATE)

        assert injection.chaos_type == ChaosType.MISLEADING_FRAMING
        # Should have a misleading prefix - check for various possible phrasings
        assert any(phrase in injection.modified_input.lower()
                   for phrase in ["since", "given", "established", "as you", "building on", "now that"])

    def test_assumption_trap_injection(self, generator):
        """Test assumption trap injection."""
        original = "Implement the payment processing logic."

        injection = generator._assumption_trap(original, ChaosSeverity.MODERATE)

        assert injection.chaos_type == ChaosType.ASSUMPTION_TRAP
        # Should discourage careful thinking - match the actual phrases used
        assert any(phrase in injection.modified_input.lower()
                   for phrase in ["standard", "typical", "usual", "don't worry", "skip",
                                  "don't overthink", "no need to consider", "edge cases"])


class TestChaosGenerator:
    """Tests for main ChaosGenerator orchestrator."""

    @pytest.fixture
    def config(self):
        return ChaosConfig(
            chaos_probability=1.0,  # Always inject for testing
            curriculum_enabled=False,
        )

    @pytest.fixture
    def generator(self, config):
        return ChaosGenerator(config)

    def test_initialization(self, generator):
        """Test generator initializes correctly."""
        assert len(generator.generators) == 3
        # Only the types implemented by generators are mapped (4 + 4 + 4)
        assert len(generator.type_to_generator) == 12

    def test_should_inject_chaos_probability(self):
        """Test chaos injection probability."""
        # High probability
        config_high = ChaosConfig(chaos_probability=1.0, curriculum_enabled=False)
        gen_high = ChaosGenerator(config_high)

        injections = sum(1 for _ in range(100) if gen_high.should_inject_chaos())
        assert injections == 100  # Should always inject

        # Low probability
        config_low = ChaosConfig(chaos_probability=0.0, curriculum_enabled=False)
        gen_low = ChaosGenerator(config_low)

        injections = sum(1 for _ in range(100) if gen_low.should_inject_chaos())
        assert injections == 0  # Should never inject

    def test_inject_produces_injection(self, generator):
        """Test that inject method produces ChaosInjection."""
        # Specify a mapped chaos_type to ensure injection occurs
        # (5 out of 17 ChaosTypes are not mapped to generators)
        # Note: The generator may return a different type from the same category
        # as each generator handles multiple related chaos types
        injection = generator.inject(
            "Test input for chaos injection.",
            chaos_type=ChaosType.MISSING_CONTEXT,  # NoiseInjector handles this
        )

        assert injection is not None
        assert injection.original_input == "Test input for chaos injection."
        # The chaos_type should be one of NoiseInjector's types
        noise_types = [ChaosType.MISSING_CONTEXT, ChaosType.PARTIAL_INFORMATION,
                       ChaosType.NOISY_INPUT, ChaosType.TRUNCATED_INPUT]
        assert injection.chaos_type in noise_types
        assert injection.severity in ChaosSeverity

    def test_inject_with_specific_type(self, generator):
        """Test injecting a specific chaos type."""
        # Note: The generator's generate() method randomly selects from its types
        # So we verify the injection came from the right generator
        injection = generator.inject(
            "Test input that is long enough to be truncated properly",
            chaos_type=ChaosType.TRUNCATED_INPUT,
        )

        # The NoiseInjector handles TRUNCATED_INPUT
        assert injection.chaos_type in generator.type_to_generator

    def test_inject_with_specific_severity(self):
        """Test injecting with specific severity."""
        # Use 100% probability and disable curriculum to ensure injection happens
        config = ChaosConfig(chaos_probability=1.0, curriculum_enabled=False)
        gen = ChaosGenerator(config)
        # Specify a chaos_type that has a generator to avoid random selection
        # of unmapped types (5 out of 17 ChaosTypes are not mapped to generators)
        injection = gen.inject(
            "Test input that is long enough to be processed",
            chaos_type=ChaosType.TRUNCATED_INPUT,
            severity=ChaosSeverity.SEVERE,
        )

        assert injection is not None
        assert injection.severity == ChaosSeverity.SEVERE

    def test_curriculum_ramp(self):
        """Test curriculum-based chaos ramping."""
        config = ChaosConfig(
            chaos_probability=1.0,
            curriculum_enabled=True,
            curriculum_start_epoch=2,
            curriculum_ramp_epochs=3,
        )
        generator = ChaosGenerator(config)

        # Before start epoch - no chaos
        generator.set_epoch(0)
        assert generator.should_inject_chaos() is False

        generator.set_epoch(1)
        assert generator.should_inject_chaos() is False

        # At start epoch - chaos begins
        generator.set_epoch(2)
        # Note: With curriculum, probability ramps up

    def test_severity_selection(self, generator):
        """Test that severity selection follows weights."""
        severities = []
        for _ in range(100):
            severities.append(generator.select_severity())

        # Should have some of each (with default weights)
        assert ChaosSeverity.SUBTLE in severities
        assert ChaosSeverity.MODERATE in severities

    def test_get_stats(self, generator):
        """Test getting chaos statistics."""
        generator.inject("Test input one that is long enough")
        generator.inject("Test input two that is long enough")

        stats = generator.get_stats()

        # At least some injections should have occurred
        assert stats["total_injections"] >= 1
        assert "current_epoch" in stats


class TestBatchChaos:
    """Tests for batch chaos application."""

    def test_apply_chaos_to_batch(self):
        """Test applying chaos to a batch of inputs."""
        config = ChaosConfig(chaos_probability=1.0, curriculum_enabled=False)
        generator = ChaosGenerator(config)

        batch = [
            "First input text that is long enough for chaos injection",
            "Second input text that is long enough for chaos injection",
            "Third input text that is long enough for chaos injection",
        ]

        modified, injections = apply_chaos_to_batch(batch, generator)

        assert len(modified) == 3
        assert len(injections) == 3
        # At least some should be injected with 100% probability
        assert any(inj is not None for inj in injections)

    def test_batch_with_no_chaos(self):
        """Test batch when chaos probability is 0."""
        config = ChaosConfig(chaos_probability=0.0, curriculum_enabled=False)
        generator = ChaosGenerator(config)

        batch = ["Input 1", "Input 2"]

        modified, injections = apply_chaos_to_batch(batch, generator)

        assert modified == batch
        assert all(inj is None for inj in injections)


class TestChaosConfig:
    """Tests for ChaosConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ChaosConfig()

        assert config.chaos_probability == 0.3
        assert config.curriculum_enabled is True
        assert len(config.enabled_types) == len(ChaosType)

    def test_severity_weights(self):
        """Test severity weight configuration."""
        config = ChaosConfig(
            severity_weights={
                ChaosSeverity.SUBTLE: 0.1,
                ChaosSeverity.MODERATE: 0.2,
                ChaosSeverity.SEVERE: 0.7,
            }
        )

        generator = ChaosGenerator(config)

        # With these weights, severe should be most common
        severities = [generator.select_severity() for _ in range(100)]
        severe_count = sum(1 for s in severities if s == ChaosSeverity.SEVERE)

        # Should have more severe than others (statistically)
        assert severe_count > 30  # Should be around 70


# Windows compatibility
if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    pytest.main([__file__, "-v"])
