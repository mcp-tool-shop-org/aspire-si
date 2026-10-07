"""Additional behaviour tests for aspire.perception.controlled_chaos (coverage gaps)."""

from unittest.mock import MagicMock, patch

import pytest

from aspire.perception.controlled_chaos import (
    AdversarialScenarioGenerator,
    AmbiguityGenerator,
    BaseChaosGenerator,
    ChaosConfig,
    ChaosGenerator,
    ChaosInjection,
    ChaosSeverity,
    ChaosType,
    NoiseInjector,
    SocialContextGenerator,
    apply_chaos_to_batch,
)

S = ChaosSeverity
MODULE_RANDOM = "aspire.perception.controlled_chaos.random"


def first_choice(seq):
    return seq[0]


def fake_random(**attrs):
    """A stand-in for the module-level `random` with deterministic behaviour."""
    rnd = MagicMock()
    rnd.choice.side_effect = first_choice
    rnd.randint.return_value = 0
    rnd.random.return_value = 0.5
    for key, value in attrs.items():
        setattr(rnd, key, value)
    return rnd


# ---------------------------------------------------------------------------
# BaseChaosGenerator
# ---------------------------------------------------------------------------


class _Stub(BaseChaosGenerator):
    def generate(self, input_text, severity=S.MODERATE, **kwargs):
        return super().generate(input_text, severity, **kwargs)

    def get_chaos_types(self):
        return super().get_chaos_types()


class TestBaseGenerator:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            BaseChaosGenerator()  # type: ignore[abstract]

    def test_abstract_bodies_are_noops(self):
        stub = _Stub()
        assert stub.generate("x") is None
        assert stub.get_chaos_types() is None


# ---------------------------------------------------------------------------
# NoiseInjector
# ---------------------------------------------------------------------------


class TestNoiseInjector:
    @pytest.mark.parametrize(
        "chosen",
        [
            ChaosType.MISSING_CONTEXT,
            ChaosType.PARTIAL_INFORMATION,
            ChaosType.NOISY_INPUT,
            ChaosType.TRUNCATED_INPUT,
        ],
    )
    def test_generate_dispatches_to_the_chosen_type(self, chosen):
        with patch(MODULE_RANDOM, fake_random(choice=MagicMock(return_value=chosen))):
            result = NoiseInjector().generate("One. Two. Three. Four. Five words here.", S.MODERATE)
        assert result.chaos_type == chosen
        assert result.original_input == "One. Two. Three. Four. Five words here."
        assert result.severity == S.MODERATE

    def test_chaos_types(self):
        assert set(NoiseInjector().get_chaos_types()) == {
            ChaosType.MISSING_CONTEXT,
            ChaosType.PARTIAL_INFORMATION,
            ChaosType.NOISY_INPUT,
            ChaosType.TRUNCATED_INPUT,
        }

    # ---- missing context --------------------------------------------------

    def test_missing_context_subtle_removes_one_non_final_sentence(self):
        text = "A one. B two. C three. D four."
        with patch(MODULE_RANDOM, fake_random(randint=MagicMock(return_value=1))):
            result = NoiseInjector()._missing_context(text, S.SUBTLE)
        assert result.modified_input == "A one. C three. D four."
        assert result.chaos_type == ChaosType.MISSING_CONTEXT

    def test_missing_context_subtle_keeps_short_inputs(self):
        result = NoiseInjector()._missing_context("Only one. And two.", S.SUBTLE)
        assert result.modified_input == "Only one. And two."

    def test_missing_context_moderate_removes_a_third(self):
        text = "A. B. C. D. E. F."
        with patch(MODULE_RANDOM, fake_random(randint=MagicMock(return_value=0))):
            result = NoiseInjector()._missing_context(text, S.MODERATE)
        # 6 sentences -> remove 2 (always index 0 here)
        assert result.modified_input == "C. D. E. F."

    def test_missing_context_moderate_never_empties_a_single_sentence(self):
        result = NoiseInjector()._missing_context("Just one.", S.MODERATE)
        assert result.modified_input == "Just one."

    def test_missing_context_severe_keeps_only_last_sentence(self):
        result = NoiseInjector()._missing_context("Background. More background. Do the thing", S.SEVERE)
        assert result.modified_input == "Do the thing."

    def test_missing_context_severe_single_sentence_gets_period(self):
        assert NoiseInjector()._missing_context("Hello", S.SEVERE).modified_input == "Hello."

    # ---- partial information ----------------------------------------------

    def test_partial_information_placeholders_per_severity(self):
        text = "alpha bravo charlie delta echo foxtrot golf hotel"
        cases = {
            S.SUBTLE: ("[...]", 1),
            S.MODERATE: ("[REDACTED]", 2),
            S.SEVERE: ("???", 3),
        }
        for severity, (placeholder, count) in cases.items():
            idx = iter(range(1, 20))
            rnd = fake_random(randint=MagicMock(side_effect=lambda a, b: next(idx)))
            with patch(MODULE_RANDOM, rnd):
                result = NoiseInjector()._partial_information(text, severity)
            words = result.modified_input.split()
            assert words.count(placeholder) == count, severity
            assert words[0] == "alpha" and words[-1] == "hotel"
            assert result.ground_truth.startswith("Identify what information is missing")

    def test_partial_information_severe_scales_with_length(self):
        text = " ".join(f"w{i}" for i in range(60))
        idx = iter(range(1, 59))
        rnd = fake_random(randint=MagicMock(side_effect=lambda a, b: next(idx)))
        with patch(MODULE_RANDOM, rnd):
            result = NoiseInjector()._partial_information(text, S.SEVERE)
        assert result.modified_input.split().count("???") == 6  # max(3, 60 // 10)

    def test_partial_information_leaves_tiny_inputs_alone(self):
        result = NoiseInjector()._partial_information("too short here", S.SEVERE)
        assert result.modified_input == "too short here"

    # ---- noisy input ------------------------------------------------------

    def _noise(self, text, noise_type, severity=S.MODERATE, roll=0.0, letter="z"):
        def choice(seq):
            return noise_type if "typo" in seq else letter

        rnd = fake_random(random=MagicMock(return_value=roll), choice=MagicMock(side_effect=choice))
        with patch(MODULE_RANDOM, rnd):
            return NoiseInjector()._noisy_input(text, severity).modified_input

    def test_noise_typo_replaces_letters_only(self):
        assert self._noise("ab1", "typo") == "zz1"

    def test_noise_swap_bubbles_characters_and_skips_last(self):
        assert self._noise("abcd", "swap") == "bcda"

    def test_noise_insert_adds_punctuation(self):
        assert self._noise("abcd", "insert", letter=".") == "....abcd"

    def test_noise_delete_blanks_characters(self):
        assert self._noise("abcd", "delete") == ""

    def test_noise_rate_depends_on_severity(self):
        text = "abcdefgh"
        assert self._noise(text, "delete", S.SUBTLE, roll=0.03) == text  # 0.03 >= 0.02
        assert self._noise(text, "delete", S.MODERATE, roll=0.03) == ""  # 0.03 < 0.05
        assert self._noise(text, "delete", S.MODERATE, roll=0.07) == text  # 0.07 >= 0.05
        assert self._noise(text, "delete", S.SEVERE, roll=0.07) == ""  # 0.07 < 0.10

    def test_noise_with_no_rolls_below_rate_is_identity(self):
        assert self._noise("hello world", "delete", roll=0.99) == "hello world"

    # ---- truncation -------------------------------------------------------

    @pytest.mark.parametrize(
        "severity,length,suffix",
        [(S.SUBTLE, 90, ""), (S.MODERATE, 70, ""), (S.SEVERE, 40, "...")],
    )
    def test_truncation_ratio(self, severity, length, suffix):
        text = "x" * 100
        result = NoiseInjector()._truncated_input(text, severity)
        assert result.modified_input == "x" * length + suffix
        assert result.chaos_type == ChaosType.TRUNCATED_INPUT

    def test_truncating_empty_input(self):
        assert NoiseInjector()._truncated_input("", S.MODERATE).modified_input == ""


# ---------------------------------------------------------------------------
# AmbiguityGenerator
# ---------------------------------------------------------------------------


class TestAmbiguityGenerator:
    @pytest.mark.parametrize(
        "chosen",
        [
            ChaosType.AMBIGUOUS_REFERENCE,
            ChaosType.UNCLEAR_INTENT,
            ChaosType.MULTIPLE_INTERPRETATIONS,
            ChaosType.IMPLICIT_REQUIREMENTS,
        ],
    )
    def test_generate_dispatch(self, chosen):
        with patch(MODULE_RANDOM, fake_random(choice=MagicMock(side_effect=lambda seq: chosen if chosen in seq else seq[0]))):
            result = AmbiguityGenerator().generate("Fix Kubernetes config", S.MODERATE)
        assert result.chaos_type == chosen

    def test_chaos_types(self):
        assert len(AmbiguityGenerator().get_chaos_types()) == 4

    def test_ambiguous_reference_replaces_long_capitalised_words(self):
        with patch(MODULE_RANDOM, fake_random()):  # random() == 0.5: no random replacements
            result = AmbiguityGenerator()._ambiguous_reference(
                "Please review Kubernetes configuration today", S.MODERATE
            )
        # "Please" is 6 chars (not > 6) and "configuration" is lowercase
        assert result.modified_input == "Please review the thing configuration today"

    def test_ambiguous_reference_subtle_usually_keeps_the_word(self):
        with patch(MODULE_RANDOM, fake_random()):  # second roll 0.5 >= 0.3
            result = AmbiguityGenerator()._ambiguous_reference("Review Kubernetes now", S.SUBTLE)
        assert result.modified_input == "Review Kubernetes now"

    def test_ambiguous_reference_subtle_replaces_when_rolls_are_low(self):
        with patch(MODULE_RANDOM, fake_random(random=MagicMock(return_value=0.05))):
            result = AmbiguityGenerator()._ambiguous_reference("Review Kubernetes now", S.SUBTLE)
        assert result.modified_input == "it it it"  # every word triggers the random branch

    def test_ambiguous_reference_severe_vocabulary(self):
        with patch(MODULE_RANDOM, fake_random()):
            result = AmbiguityGenerator()._ambiguous_reference("Review Kubernetes now", S.SEVERE)
        assert result.modified_input == "Review you know what now"

    def test_unclear_intent_strips_clear_markers_and_wraps(self):
        with patch(MODULE_RANDOM, fake_random()):
            result = AmbiguityGenerator()._unclear_intent("Please help me Fix the bug", S.SEVERE)
        assert result.modified_input == "Hmm, fix the bug..."
        assert result.chaos_type == ChaosType.UNCLEAR_INTENT

    def test_unclear_intent_prefix_suffix_by_severity(self):
        with patch(MODULE_RANDOM, fake_random()):
            subtle = AmbiguityGenerator()._unclear_intent("Explain recursion", S.SUBTLE).modified_input
            moderate = AmbiguityGenerator()._unclear_intent("Explain recursion", S.MODERATE).modified_input
        assert subtle == "I was wondering about explain recursion?"
        assert moderate == "So there's this thing with explain recursion or something?"

    @pytest.mark.parametrize(
        "severity,suffix",
        [
            (S.SUBTLE, " (or the other way around)"),
            (S.MODERATE, " - unless you think otherwise"),
            (S.SEVERE, " - but that's just one way to look at it. Or maybe I mean the opposite?"),
        ],
    )
    def test_multiple_interpretations_appends_after_trimming_punctuation(self, severity, suffix):
        with patch(MODULE_RANDOM, fake_random()):
            result = AmbiguityGenerator()._multiple_interpretations("Sort the list!?.", severity)
        assert result.modified_input == "Sort the list" + suffix

    def test_implicit_requirements_removes_clause_up_to_period(self):
        result = AmbiguityGenerator()._implicit_requirements("The API must be fast. Use caching.", S.SUBTLE)
        assert result.modified_input == "The API . Use caching. (you know what I mean)"

    def test_implicit_requirements_falls_back_to_comma_then_end(self):
        comma = AmbiguityGenerator()._implicit_requirements("It should be quick, then more", S.MODERATE)
        assert comma.modified_input == "It , then more (the usual requirements apply)"
        end = AmbiguityGenerator()._implicit_requirements("It HAS TO work", S.SEVERE)
        assert end.modified_input == "It (figure out what I need)"

    def test_implicit_requirements_without_phrases_only_adds_marker(self):
        result = AmbiguityGenerator()._implicit_requirements("  Write a poem  ", S.MODERATE)
        assert result.modified_input == "Write a poem (the usual requirements apply)"


# ---------------------------------------------------------------------------
# AdversarialScenarioGenerator
# ---------------------------------------------------------------------------


class TestAdversarialGenerator:
    @pytest.mark.parametrize(
        "chosen",
        [
            ChaosType.CONTRADICTORY_INSTRUCTIONS,
            ChaosType.CONFLICTING_CONSTRAINTS,
            ChaosType.MISLEADING_FRAMING,
            ChaosType.ASSUMPTION_TRAP,
        ],
    )
    def test_generate_dispatch(self, chosen):
        with patch(MODULE_RANDOM, fake_random(choice=MagicMock(side_effect=lambda seq: chosen if chosen in seq else seq[0]))):
            result = AdversarialScenarioGenerator().generate("Build a login form.", S.SEVERE)
        assert result.chaos_type == chosen

    def test_chaos_types(self):
        assert len(AdversarialScenarioGenerator().get_chaos_types()) == 4

    @pytest.mark.parametrize(
        "severity,contradiction",
        [
            (S.SUBTLE, " But also keep it simple."),
            (S.MODERATE, " Actually, on second thought, do the opposite."),
            (S.SEVERE, " Do this AND don't do this at the same time."),
        ],
    )
    def test_contradictory_instructions(self, severity, contradiction):
        with patch(MODULE_RANDOM, fake_random()):
            result = AdversarialScenarioGenerator()._contradictory_instructions("Write it!", severity)
        assert result.modified_input == "Write it." + contradiction

    @pytest.mark.parametrize(
        "severity,constraint",
        [
            (S.SUBTLE, " It needs to be done quickly but also thoroughly."),
            (S.MODERATE, " Requirements: fast, cheap, and high quality."),
            (S.SEVERE, " Constraints: zero budget, enterprise features, yesterday's deadline."),
        ],
    )
    def test_conflicting_constraints(self, severity, constraint):
        with patch(MODULE_RANDOM, fake_random()):
            result = AdversarialScenarioGenerator()._conflicting_constraints("Ship it.", severity)
        assert result.modified_input == "Ship it." + constraint

    @pytest.mark.parametrize(
        "severity,frame",
        [
            (S.SUBTLE, "Since everyone agrees that "),
            (S.MODERATE, "Now that we've established [unestablished thing], "),
            (S.SEVERE, "As you already confirmed [never confirmed], "),
        ],
    )
    def test_misleading_framing_lowercases_first_letter(self, severity, frame):
        with patch(MODULE_RANDOM, fake_random()):
            result = AdversarialScenarioGenerator()._misleading_framing("Python is slow.", severity)
        assert result.modified_input == frame + "python is slow."

    @pytest.mark.parametrize(
        "severity,trap",
        [
            (S.SUBTLE, " (the standard approach should work)"),
            (S.MODERATE, " This is a typical case so don't overthink it."),
            (S.SEVERE, " Don't worry about validation or error cases."),
        ],
    )
    def test_assumption_trap(self, severity, trap):
        with patch(MODULE_RANDOM, fake_random()):
            result = AdversarialScenarioGenerator()._assumption_trap("Parse the file.", severity)
        assert result.modified_input == "Parse the file." + trap
        assert result.ground_truth.startswith("Still consider")

    def test_misleading_framing_handles_empty_input(self):
        AdversarialScenarioGenerator()._misleading_framing("", S.MODERATE)

    def test_every_generator_survives_empty_input(self):
        # Nothing should blow up on "".
        for gen in (NoiseInjector(), AmbiguityGenerator(), AdversarialScenarioGenerator(), SocialContextGenerator()):
            for severity in S:
                for chaos_type in gen.get_chaos_types():
                    with patch(
                        MODULE_RANDOM,
                        fake_random(choice=MagicMock(side_effect=lambda seq, c=chaos_type: c if c in seq else seq[0])),
                    ):
                        assert isinstance(gen.generate("", severity), ChaosInjection)


# ---------------------------------------------------------------------------
# SocialContextGenerator
# ---------------------------------------------------------------------------


class TestSocialContextGenerator:
    def test_chaos_types(self):
        assert set(SocialContextGenerator().get_chaos_types()) == {
            ChaosType.INCONSISTENT_CONTEXT,
            ChaosType.EMOTIONAL_SUBTEXT,
            ChaosType.HIDDEN_AGENDA,
            ChaosType.POLITENESS_VS_DIRECTNESS,
            ChaosType.SCOPE_CREEP,
        }

    @pytest.mark.parametrize("severity", list(S))
    @pytest.mark.parametrize("chaos_type", SocialContextGenerator().get_chaos_types())
    def test_requested_type_is_produced_and_changes_the_input(self, chaos_type, severity):
        with patch(MODULE_RANDOM, fake_random()):
            result = SocialContextGenerator().generate("Parse the file.", severity, chaos_type=chaos_type)
        assert result.chaos_type == chaos_type
        assert result.severity == severity
        assert result.original_input == "Parse the file."
        assert "Parse the file." in result.modified_input
        assert result.modified_input != "Parse the file."
        assert result.ground_truth and result.learning_objective

    @pytest.mark.parametrize("chosen", SocialContextGenerator().get_chaos_types())
    def test_generate_picks_a_type_when_none_requested(self, chosen):
        with patch(MODULE_RANDOM, fake_random(choice=MagicMock(side_effect=lambda seq: chosen if chosen in seq else seq[0]))):
            result = SocialContextGenerator().generate("Parse the file.", S.MODERATE)
        assert result.chaos_type == chosen


# ---------------------------------------------------------------------------
# ChaosGenerator (orchestrator)
# ---------------------------------------------------------------------------


class TestChaosGeneratorGate:
    def test_before_curriculum_start_never_injects(self):
        gen = ChaosGenerator(ChaosConfig(chaos_probability=1.0, curriculum_start_epoch=2))
        gen.set_epoch(1)
        with patch(MODULE_RANDOM, fake_random(random=MagicMock(return_value=0.0))):
            assert gen.should_inject_chaos() is False

    def test_probability_ramps_linearly_then_caps(self):
        cfg = ChaosConfig(chaos_probability=0.4, curriculum_start_epoch=1, curriculum_ramp_epochs=4)
        gen = ChaosGenerator(cfg)

        def decide(epoch, roll):
            gen.set_epoch(epoch)
            with patch(MODULE_RANDOM, fake_random(random=MagicMock(return_value=roll))):
                return gen.should_inject_chaos()

        assert decide(1, 0.0) is False  # progress 0 -> probability 0
        assert decide(3, 0.19) is True and decide(3, 0.21) is False  # 0.4 * 2/4 = 0.2
        assert decide(5, 0.39) is True and decide(5, 0.41) is False  # fully ramped
        assert decide(50, 0.39) is True and decide(50, 0.41) is False  # capped

    def test_curriculum_disabled_uses_base_probability_from_epoch_zero(self):
        gen = ChaosGenerator(ChaosConfig(chaos_probability=0.3, curriculum_enabled=False))
        with patch(MODULE_RANDOM, fake_random(random=MagicMock(return_value=0.29))):
            assert gen.should_inject_chaos() is True
        with patch(MODULE_RANDOM, fake_random(random=MagicMock(return_value=0.31))):
            assert gen.should_inject_chaos() is False

    def test_zero_ramp_epochs_means_no_ramp(self):
        gen = ChaosGenerator(ChaosConfig(curriculum_ramp_epochs=0, curriculum_start_epoch=0))
        with patch(MODULE_RANDOM, fake_random(random=MagicMock(return_value=0.29))):
            assert gen.should_inject_chaos() is True  # full intensity: 0.3 * 1.0
        with patch(MODULE_RANDOM, fake_random(random=MagicMock(return_value=0.31))):
            assert gen.should_inject_chaos() is False


class TestSelectSeverity:
    def test_weights_partition_the_unit_interval(self):
        cfg = ChaosConfig(severity_weights={S.SUBTLE: 1.0, S.MODERATE: 1.0, S.SEVERE: 2.0})
        gen = ChaosGenerator(cfg)

        def pick(roll):
            with patch(MODULE_RANDOM, fake_random(random=MagicMock(return_value=roll))):
                return gen.select_severity()

        assert pick(0.0) == S.SUBTLE
        assert pick(0.24) == S.SUBTLE
        assert pick(0.26) == S.MODERATE
        assert pick(0.51) == S.SEVERE
        assert pick(1.0) == S.SEVERE

    def test_empty_weights_default_to_moderate(self):
        gen = ChaosGenerator(ChaosConfig(severity_weights={}))
        assert gen.select_severity() == S.MODERATE

    def test_distribution_follows_weights(self):
        import random as real_random

        real_random.seed(1234)
        gen = ChaosGenerator(ChaosConfig(severity_weights={S.SUBTLE: 0.9, S.MODERATE: 0.1, S.SEVERE: 0.0}))
        picks = [gen.select_severity() for _ in range(500)]
        assert picks.count(S.SEVERE) == 0
        assert picks.count(S.SUBTLE) > 400


class TestChaosGeneratorInject:
    def _always(self, **cfg):
        cfg.setdefault("curriculum_enabled", False)
        cfg.setdefault("chaos_probability", 1.0)
        return ChaosGenerator(ChaosConfig(**cfg))

    def test_generator_mapping_covers_the_implemented_types(self):
        gen = ChaosGenerator()
        assert isinstance(gen.type_to_generator[ChaosType.NOISY_INPUT], NoiseInjector)
        assert isinstance(gen.type_to_generator[ChaosType.UNCLEAR_INTENT], AmbiguityGenerator)
        assert isinstance(gen.type_to_generator[ChaosType.ASSUMPTION_TRAP], AdversarialScenarioGenerator)
        assert isinstance(gen.type_to_generator[ChaosType.SCOPE_CREEP], SocialContextGenerator)
        assert len(gen.type_to_generator) == len(ChaosType) == 17

    def test_returns_none_when_gate_closed(self):
        gen = ChaosGenerator(ChaosConfig(chaos_probability=0.0, curriculum_enabled=False))
        assert gen.inject("text") is None
        assert gen.injection_count == 0

    def test_explicit_type_and_severity_are_forwarded_to_the_owning_generator(self):
        gen = self._always()
        spy = MagicMock(wraps=gen.type_to_generator[ChaosType.NOISY_INPUT])
        gen.type_to_generator[ChaosType.NOISY_INPUT] = spy
        result = gen.inject("Some text. More text.", ChaosType.NOISY_INPUT, S.SEVERE)
        spy.generate.assert_called_once_with("Some text. More text.", S.SEVERE, chaos_type=ChaosType.NOISY_INPUT)
        assert result.severity == S.SEVERE
        assert gen.injection_count == 1

    def test_random_type_comes_from_enabled_types(self):
        gen = self._always(enabled_types=[ChaosType.TRUNCATED_INPUT])
        result = gen.inject("x" * 50, severity=S.SEVERE)
        assert result is not None and gen.injection_count == 1

    def test_unmapped_type_yields_no_injection_and_no_count(self):
        gen = self._always()
        del gen.type_to_generator[ChaosType.SCOPE_CREEP]
        assert gen.inject("text", ChaosType.SCOPE_CREEP, S.SUBTLE) is None
        assert gen.injection_count == 0

    def test_severity_chosen_from_weights_when_omitted(self):
        gen = self._always(severity_weights={S.SEVERE: 1.0})
        result = gen.inject("x" * 50, ChaosType.TRUNCATED_INPUT)
        assert result.severity == S.SEVERE

    def test_stats_and_epoch(self):
        gen = ChaosGenerator(ChaosConfig(chaos_probability=0.25, curriculum_enabled=False))
        gen.set_epoch(7)
        assert gen.get_stats() == {
            "total_injections": 0,
            "current_epoch": 7,
            "curriculum_enabled": False,
            "base_probability": 0.25,
        }

    def test_requested_chaos_type_is_honoured(self):
        gen = self._always()
        for _ in range(60):
            result = gen.inject("The quick brown fox. Jumps over the lazy dog.", ChaosType.TRUNCATED_INPUT, S.MODERATE)
            assert result.chaos_type == ChaosType.TRUNCATED_INPUT

    def test_every_chaos_type_yields_an_injection_of_that_type(self):
        gen = self._always()
        for chaos_type in ChaosType:
            for severity in S:
                result = gen.inject("The quick brown fox. Jumps over the lazy dog.", chaos_type, severity)
                assert result is not None
                assert result.chaos_type == chaos_type

    def test_every_default_enabled_type_has_a_generator(self):
        gen = ChaosGenerator()
        missing = [t for t in gen.config.enabled_types if t not in gen.type_to_generator]
        assert missing == []


class TestApplyChaosToBatch:
    def test_mixed_batch(self):
        injection = ChaosInjection(ChaosType.NOISY_INPUT, S.SUBTLE, "b", "B!", "gt", "lo")
        gen = MagicMock()
        gen.inject.side_effect = [None, injection, None]
        modified, records = apply_chaos_to_batch(["a", "b", "c"], gen)
        assert modified == ["a", "B!", "c"]
        assert records == [None, injection, None]

    def test_empty_batch(self):
        assert apply_chaos_to_batch([], ChaosGenerator()) == ([], [])

    def test_with_real_generator_at_full_probability(self):
        gen = ChaosGenerator(
            ChaosConfig(
                chaos_probability=1.0,
                curriculum_enabled=False,
                enabled_types=[ChaosType.TRUNCATED_INPUT],
                severity_weights={S.MODERATE: 1.0},
            )
        )
        trunc = ChaosType.TRUNCATED_INPUT
        rnd = fake_random(choice=MagicMock(side_effect=lambda seq: trunc if trunc in seq else seq[0]))
        with patch(MODULE_RANDOM, rnd):
            modified, records = apply_chaos_to_batch(["x" * 10, "y" * 20], gen)
        assert modified == ["x" * 7, "y" * 14]
        assert all(r is not None and r.chaos_type == ChaosType.TRUNCATED_INPUT for r in records)
        assert gen.injection_count == 2
