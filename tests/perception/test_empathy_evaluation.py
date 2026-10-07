"""
Tests for the perception-aware evaluation module.

Covers the dimension catalogue, score aggregation, conversion to
TeacherEvaluation, and the heuristic PerceptionEvaluator.
"""

import os

os.environ["XFORMERS_DISABLED"] = "1"

import pytest

from aspire.perception.controlled_chaos import ChaosInjection, ChaosSeverity, ChaosType
from aspire.perception.empathy_evaluation import (
    PERCEPTION_DIMENSION_CRITERIA,
    PerceptionDimension,
    PerceptionEvaluation,
    PerceptionEvaluator,
    PerceptionScore,
    create_perception_enhanced_evaluation_prompt,
)
from aspire.teachers.base import EvaluationDimension, TeacherEvaluation

PD = PerceptionDimension


def _score(dimension, value, explanation="why", **kwargs):
    return PerceptionScore(dimension=dimension, score=value, explanation=explanation, **kwargs)


def _chaos(chaos_type):
    return ChaosInjection(
        chaos_type=chaos_type,
        severity=ChaosSeverity.MODERATE,
        original_input="orig",
        modified_input="mod",
        ground_truth="handle it",
        learning_objective="test",
    )


class TestDimensionCatalogue:
    """Tests for PerceptionDimension and its criteria table."""

    def test_fifteen_dimensions(self):
        assert len(list(PD)) == 15

    def test_every_dimension_has_criteria(self):
        assert set(PERCEPTION_DIMENSION_CRITERIA) == set(PD)

    @pytest.mark.parametrize("dimension", list(PD))
    def test_criteria_shape(self, dimension):
        criteria = PERCEPTION_DIMENSION_CRITERIA[dimension]

        assert criteria["description"]
        assert criteria["high_score"]
        assert criteria["low_score"]
        assert criteria["weight"] > 0
        assert set(criteria["examples"]) == {"good", "bad"}

    def test_dimension_values_are_snake_case_strings(self):
        assert PD.COGNITIVE_EMPATHY == "cognitive_empathy"
        assert PD("noise_tolerance") is PD.NOISE_TOLERANCE

    def test_empathy_weights_are_highest(self):
        weights = {d: c["weight"] for d, c in PERCEPTION_DIMENSION_CRITERIA.items()}

        assert weights[PD.COGNITIVE_EMPATHY] == 1.2
        assert weights[PD.EMPATHIC_RESONANCE] == 1.2
        assert max(weights.values()) == 1.2


EXPECTED_MAPPING = {
    PD.COGNITIVE_EMPATHY: EvaluationDimension.EMPATHY,
    PD.EMOTIONAL_ATTUNEMENT: EvaluationDimension.EMPATHY,
    PD.INTENT_RECOGNITION: EvaluationDimension.REASONING,
    PD.UNCERTAINTY_CALIBRATION: EvaluationDimension.INTELLECTUAL_HONESTY,
    PD.SELF_AWARENESS: EvaluationDimension.INTELLECTUAL_HONESTY,
    PD.ASSUMPTION_TRANSPARENCY: EvaluationDimension.CLARITY,
    PD.VALUE_CONSISTENCY: EvaluationDimension.REASONING,
    PD.BEHAVIORAL_COHERENCE: EvaluationDimension.CLARITY,
    PD.PERSPECTIVE_INTEGRATION: EvaluationDimension.NUANCE,
    PD.AMBIGUITY_HANDLING: EvaluationDimension.ADAPTABILITY,
    PD.CONTRADICTION_RESOLUTION: EvaluationDimension.REASONING,
    PD.NOISE_TOLERANCE: EvaluationDimension.ADAPTABILITY,
    PD.SYNTROPIC_COHERENCE: EvaluationDimension.CLARITY,
    PD.EMPATHIC_RESONANCE: EvaluationDimension.EMPATHY,
    PD.MEANING_GENERATION: EvaluationDimension.CREATIVITY,
}


class TestPerceptionScore:
    """Tests for PerceptionScore."""

    def test_defaults(self):
        score = _score(PD.SELF_AWARENESS, 6.0)

        assert score.evidence == []
        assert score.improvement_suggestions == []

    def test_expected_mapping_covers_every_dimension(self):
        assert set(EXPECTED_MAPPING) == set(PD)

    @pytest.mark.parametrize(("dimension", "base"), list(EXPECTED_MAPPING.items()))
    def test_to_dimension_score_mapping(self, dimension, base):
        converted = _score(dimension, 7.25, "because").to_dimension_score()

        assert converted.dimension is base
        assert converted.score == 7.25
        assert converted.explanation == f"[{dimension.value}] because"


class TestComputeAggregates:
    """Tests for PerceptionEvaluation.compute_aggregates."""

    def test_empty_scores_give_zeros(self):
        evaluation = PerceptionEvaluation(perception_scores=[])
        evaluation.compute_aggregates()

        assert evaluation.cognitive_empathy_score == 0.0
        assert evaluation.metacognition_score == 0.0
        assert evaluation.character_score == 0.0
        assert evaluation.robustness_score == 0.0
        assert evaluation.overall_perception_score == 0.0

    def test_category_averages_and_weighted_overall(self):
        evaluation = PerceptionEvaluation(
            perception_scores=[
                _score(PD.COGNITIVE_EMPATHY, 8.0),
                _score(PD.EMOTIONAL_ATTUNEMENT, 6.0),
                _score(PD.INTENT_RECOGNITION, 10.0),  # empathy avg 8
                _score(PD.UNCERTAINTY_CALIBRATION, 4.0),
                _score(PD.SELF_AWARENESS, 6.0),  # meta avg 5
                _score(PD.VALUE_CONSISTENCY, 9.0),  # character avg 9
                _score(PD.AMBIGUITY_HANDLING, 2.0),
                _score(PD.NOISE_TOLERANCE, 4.0),  # robustness avg 3
            ]
        )
        evaluation.compute_aggregates()

        assert evaluation.cognitive_empathy_score == pytest.approx(8.0)
        assert evaluation.metacognition_score == pytest.approx(5.0)
        assert evaluation.character_score == pytest.approx(9.0)
        assert evaluation.robustness_score == pytest.approx(3.0)
        expected = (1.2 * 8.0 + 1.1 * 5.0 + 1.0 * 9.0 + 1.0 * 3.0) / 4.3
        assert evaluation.overall_perception_score == pytest.approx(expected)

    def test_uniform_scores_give_that_score(self):
        evaluation = PerceptionEvaluation(
            perception_scores=[_score(d, 6.0) for d in PD]
        )
        evaluation.compute_aggregates()

        assert evaluation.overall_perception_score == pytest.approx(6.0)

    def test_recompute_overwrites_previous_values(self):
        evaluation = PerceptionEvaluation(perception_scores=[_score(PD.SELF_AWARENESS, 2.0)])
        evaluation.compute_aggregates()
        evaluation.perception_scores = [_score(PD.SELF_AWARENESS, 8.0)]
        evaluation.compute_aggregates()

        assert evaluation.metacognition_score == pytest.approx(8.0)

    def test_syntropy_dimensions_do_not_enter_category_averages(self):
        evaluation = PerceptionEvaluation(
            perception_scores=[_score(PD.EMPATHIC_RESONANCE, 9.0), _score(PD.MEANING_GENERATION, 9.0)]
        )
        evaluation.compute_aggregates()

        assert evaluation.cognitive_empathy_score == 0.0
        assert evaluation.metacognition_score == 0.0

    def test_overall_ignores_unscored_categories(self):
        evaluation = PerceptionEvaluation(
            perception_scores=[_score(PD.UNCERTAINTY_CALIBRATION, 10.0)]
        )
        evaluation.compute_aggregates()

        assert evaluation.overall_perception_score == pytest.approx(10.0)


def _evaluation_with_feedback():
    evaluation = PerceptionEvaluation(
        perception_scores=[
            _score(PD.COGNITIVE_EMPATHY, 8.0, "empathic"),
            _score(PD.SELF_AWARENESS, 4.0, "blind"),
        ],
        perception_strengths=["empathy: strong"],
        perception_weaknesses=["awareness: weak"],
        perception_suggestions=["acknowledge limits"],
    )
    evaluation.compute_aggregates()
    return evaluation


class TestToTeacherEvaluation:
    """Tests for PerceptionEvaluation.to_teacher_evaluation."""

    def test_without_base(self):
        evaluation = _evaluation_with_feedback()

        result = evaluation.to_teacher_evaluation()

        assert isinstance(result, TeacherEvaluation)
        assert result.overall_score == evaluation.overall_perception_score
        assert [d.dimension for d in result.dimension_scores] == [
            EvaluationDimension.EMPATHY,
            EvaluationDimension.INTELLECTUAL_HONESTY,
        ]
        assert result.strengths == ["empathy: strong"]
        assert result.weaknesses == ["awareness: weak"]
        assert result.suggestions == ["acknowledge limits"]
        assert result.improved_response is None
        assert result.reasoning.startswith("Overall Perception Score:")
        meta = result.metadata["perception_evaluation"]
        assert meta == {
            "cognitive_empathy": evaluation.cognitive_empathy_score,
            "metacognition": evaluation.metacognition_score,
            "character": evaluation.character_score,
            "robustness": evaluation.robustness_score,
            "overall": evaluation.overall_perception_score,
        }

    def test_with_base_merges_everything(self):
        evaluation = _evaluation_with_feedback()
        base = TeacherEvaluation(
            overall_score=9.0,
            dimension_scores=[],
            reasoning="base reasoning",
            improved_response="better answer",
            strengths=["base strength"],
            weaknesses=["base weakness"],
            suggestions=["base suggestion"],
            metadata={"source": "teacher"},
        )
        base.dimension_scores = [
            _score(PD.SELF_AWARENESS, 1.0).to_dimension_score(),
        ]

        result = evaluation.to_teacher_evaluation(base)

        expected_overall = 9.0 * 0.6 + evaluation.overall_perception_score * 0.4
        assert result.overall_score == pytest.approx(expected_overall)
        assert len(result.dimension_scores) == 3
        assert result.dimension_scores[0] is base.dimension_scores[0]
        assert result.improved_response == "better answer"
        assert result.strengths == ["base strength", "empathy: strong"]
        assert result.weaknesses == ["base weakness", "awareness: weak"]
        assert result.suggestions == ["base suggestion", "acknowledge limits"]
        assert result.reasoning.startswith("base reasoning\n\nPERCEPTION ASSESSMENT:\n")
        assert "Overall Perception Score" in result.reasoning
        assert result.metadata["source"] == "teacher"
        assert result.metadata["perception_evaluation"]["overall"] == pytest.approx(
            evaluation.overall_perception_score
        )

    def test_with_base_does_not_mutate_base(self):
        evaluation = _evaluation_with_feedback()
        base = TeacherEvaluation(
            overall_score=5.0,
            dimension_scores=[],
            reasoning="r",
            strengths=["a"],
            metadata={"k": 1},
        )

        evaluation.to_teacher_evaluation(base)

        assert base.strengths == ["a"]
        assert base.dimension_scores == []
        assert base.metadata == {"k": 1}


class TestPerceptionSummary:
    """Tests for PerceptionEvaluation._perception_summary."""

    def test_summary_lines(self):
        evaluation = _evaluation_with_feedback()
        lines = evaluation._perception_summary().split("\n")

        assert lines[0] == f"Overall Perception Score: {evaluation.overall_perception_score:.1f}/10"
        assert lines[1] == "  - Cognitive Empathy: 8.0/10"
        assert lines[2] == "  - Meta-Cognition: 4.0/10"
        assert lines[3] == "  - Character Coherence: 0.0/10"
        assert lines[4] == "  - Robustness: 0.0/10"
        assert "Perception Strengths:" in lines
        assert "  + empathy: strong" in lines
        assert "Perception Weaknesses:" in lines
        assert "  - awareness: weak" in lines

    def test_summary_without_feedback_has_no_sections(self):
        summary = PerceptionEvaluation(perception_scores=[])._perception_summary()

        assert "Strengths" not in summary
        assert "Weaknesses" not in summary

    def test_summary_lists_at_most_three_items(self):
        evaluation = PerceptionEvaluation(
            perception_scores=[],
            perception_strengths=[f"s{i}" for i in range(5)],
            perception_weaknesses=[f"w{i}" for i in range(5)],
        )
        summary = evaluation._perception_summary()

        assert "  + s2" in summary and "s3" not in summary
        assert "  - w2" in summary and "w3" not in summary


class TestEvaluatorSetup:
    """Tests for PerceptionEvaluator construction."""

    def test_defaults(self):
        evaluator = PerceptionEvaluator()

        assert evaluator.strictness == 0.5
        assert evaluator.enabled_dimensions == list(PD)

    def test_custom_dimensions_and_strictness(self):
        evaluator = PerceptionEvaluator(enabled_dimensions=[PD.SELF_AWARENESS], strictness=0.9)

        assert evaluator.enabled_dimensions == [PD.SELF_AWARENESS]
        assert evaluator.strictness == 0.9


class TestUncertaintyCalibration:
    """Tests for the uncertainty-calibration heuristic."""

    @pytest.fixture
    def evaluator(self):
        return PerceptionEvaluator()

    def test_neutral_response(self, evaluator):
        score, explanation, evidence = evaluator._eval_uncertainty_calibration("Here is the answer.", {})

        assert score == 5.5
        assert explanation == "Moderate uncertainty calibration"
        assert evidence == []

    def test_hedging_without_certainty_scores_well(self, evaluator):
        score, explanation, evidence = evaluator._eval_uncertainty_calibration(
            "I think this is probably right, but it might change.", {}
        )

        assert score == 7.5
        assert explanation == "Appropriate use of hedging language"
        assert evidence == ["Found 3 hedging expressions"]

    def test_counts_distinct_phrases_not_occurrences(self, evaluator):
        _, _, evidence = evaluator._eval_uncertainty_calibration("probably probably probably", {})

        assert evidence == ["Found 1 hedging expressions"]

    def test_case_insensitive(self, evaluator):
        score, _, _ = evaluator._eval_uncertainty_calibration("I THINK so", {})

        assert score == 7.5

    def test_excess_certainty_without_chaos(self, evaluator):
        text = "This is definitely, certainly, absolutely and always true."

        score, explanation, evidence = evaluator._eval_uncertainty_calibration(text, {})

        assert score == 4.0
        assert explanation == "Excessive certainty language may indicate overconfidence"
        assert evidence == ["Found 4 certainty expressions"]

    def test_three_certainty_words_is_not_excessive(self, evaluator):
        text = "definitely, certainly, absolutely"

        score, _, _ = evaluator._eval_uncertainty_calibration(text, {})

        assert score == 5.5

    def test_hedging_with_two_certainties_is_moderate(self, evaluator):
        text = "I think it is definitely and obviously fine"

        score, explanation, evidence = evaluator._eval_uncertainty_calibration(text, {})

        assert score == 5.5
        assert explanation == "Moderate uncertainty calibration"
        assert evidence == ["Found 1 hedging expressions", "Found 2 certainty expressions"]

    def test_chaos_without_hedging_is_penalised(self, evaluator):
        context = {"chaos_injection": _chaos(ChaosType.UNCLEAR_INTENT)}

        score, explanation, _ = evaluator._eval_uncertainty_calibration("Here is the answer.", context)

        assert score == 3.0
        assert explanation == "No uncertainty expressed despite ambiguous input"

    def test_chaos_with_hedging_is_rewarded(self, evaluator):
        context = {"chaos_injection": _chaos(ChaosType.UNCLEAR_INTENT)}

        score, _, _ = evaluator._eval_uncertainty_calibration("I believe you mean the first one.", context)

        assert score == 7.5

    def test_chaos_with_heavy_certainty_is_not_called_overconfident(self, evaluator):
        context = {"chaos_injection": _chaos(ChaosType.NOISY_INPUT)}
        text = "definitely certainly absolutely always, I think"

        score, explanation, _ = evaluator._eval_uncertainty_calibration(text, context)

        assert score == 5.5
        assert explanation == "Moderate uncertainty calibration"

    def test_hedge_match_respects_word_boundaries(self, evaluator):
        context = {"chaos_injection": _chaos(ChaosType.UNCLEAR_INTENT)}

        score, _, _ = evaluator._eval_uncertainty_calibration("The mayor arrived at noon.", context)

        assert score == 3.0


class TestAmbiguityHandling:
    """Tests for the ambiguity-handling heuristic."""

    @pytest.fixture
    def evaluator(self):
        return PerceptionEvaluator()

    @pytest.mark.parametrize(
        "chaos_type",
        [ChaosType.AMBIGUOUS_REFERENCE, ChaosType.UNCLEAR_INTENT, ChaosType.MULTIPLE_INTERPRETATIONS],
    )
    def test_ambiguous_input_with_clarification(self, evaluator, chaos_type):
        context = {"chaos_injection": _chaos(chaos_type)}

        score, explanation, evidence = evaluator._eval_ambiguity_handling(
            "p", "Could you clarify? What do you mean?", context
        )

        assert score == 8.0
        assert explanation == "Appropriately sought clarification for ambiguous input"
        assert evidence == ["Found 2 clarification attempts"]

    def test_ambiguous_input_without_clarification(self, evaluator):
        context = {"chaos_injection": _chaos(ChaosType.AMBIGUOUS_REFERENCE)}

        score, explanation, evidence = evaluator._eval_ambiguity_handling("p", "Done.", context)

        assert score == 3.0
        assert explanation == "Did not address ambiguity in input"
        assert evidence == []

    def test_non_ambiguity_chaos_is_treated_as_clear_input(self, evaluator):
        context = {"chaos_injection": _chaos(ChaosType.NOISY_INPUT)}

        score, explanation, _ = evaluator._eval_ambiguity_handling("p", "Done.", context)

        assert score == 7.0
        assert explanation == "Handled input appropriately"

    def test_clear_input_normal_response(self, evaluator):
        score, explanation, evidence = evaluator._eval_ambiguity_handling("p", "Here you go.", {})

        assert score == 7.0
        assert explanation == "Handled input appropriately"
        assert evidence == []

    def test_clear_input_with_up_to_two_clarifications_is_fine(self, evaluator):
        score, _, evidence = evaluator._eval_ambiguity_handling(
            "p", "Could you clarify? Please specify.", {}
        )

        assert score == 7.0
        assert evidence == ["Found 2 clarification attempts"]

    def test_clear_input_with_excess_clarification(self, evaluator):
        text = "Could you clarify? What do you mean? Please specify. Do you want A?"

        score, explanation, evidence = evaluator._eval_ambiguity_handling("p", text, {})

        assert score == 5.0
        assert explanation == "Excessive clarification requests for clear input"
        assert evidence == ["Found 4 clarification attempts"]


class TestAssumptionTransparency:
    """Tests for the assumption-transparency heuristic."""

    @pytest.fixture
    def evaluator(self):
        return PerceptionEvaluator()

    def test_no_assumptions(self, evaluator):
        score, explanation, evidence = evaluator._eval_assumption_transparency("Just do it.")

        assert score == 5.0
        assert explanation == "Assumptions not explicitly stated"
        assert evidence == []

    def test_one_assumption(self, evaluator):
        score, explanation, evidence = evaluator._eval_assumption_transparency("I'm assuming Python.")

        assert score == 6.5
        assert explanation == "Some assumption transparency"
        assert evidence == ["Found 1 explicit assumption statements"]

    def test_two_assumptions(self, evaluator):
        text = "I'm assuming Python 3. Correct me if I'm wrong."

        score, explanation, evidence = evaluator._eval_assumption_transparency(text)

        assert score == 8.0
        assert explanation == "Clearly states assumptions and invites correction"
        assert evidence == ["Found 2 explicit assumption statements"]


class TestEvaluateDimension:
    """Tests for _evaluate_dimension (dispatch, strictness, suggestions)."""

    def test_generic_dimension_is_neutral_placeholder(self):
        score = PerceptionEvaluator()._evaluate_dimension(PD.COGNITIVE_EMPATHY, "p", "r", {})

        assert score.dimension is PD.COGNITIVE_EMPATHY
        assert score.score == 5.0
        assert score.explanation == "Awaiting model-based evaluation for cognitive_empathy"
        assert score.evidence == []
        assert score.improvement_suggestions == [
            PERCEPTION_DIMENSION_CRITERIA[PD.COGNITIVE_EMPATHY]["high_score"]
        ]

    def test_dispatches_to_specialised_heuristics(self):
        evaluator = PerceptionEvaluator()

        uncertainty = evaluator._evaluate_dimension(PD.UNCERTAINTY_CALIBRATION, "p", "I think so", {})
        ambiguity = evaluator._evaluate_dimension(PD.AMBIGUITY_HANDLING, "p", "ok", {})
        assumption = evaluator._evaluate_dimension(PD.ASSUMPTION_TRANSPARENCY, "p", "I assume X", {})

        assert uncertainty.score == 7.5
        assert ambiguity.score == 7.0
        assert assumption.score == 6.5
        assert uncertainty.evidence == ["Found 1 hedging expressions"]

    def test_good_scores_get_no_suggestions(self):
        score = PerceptionEvaluator()._evaluate_dimension(PD.AMBIGUITY_HANDLING, "p", "ok", {})

        assert score.score == 7.0
        assert score.improvement_suggestions == []

    @pytest.mark.parametrize(
        ("strictness", "expected"),
        [
            (0.5, 5.0),
            (1.0, 5.0 * 0.9),
            (0.75, 5.0 * 0.95),
            (0.0, 5.0 * 1.05),
            (0.25, 5.0 * 1.025),
        ],
    )
    def test_strictness_adjusts_neutral_score(self, strictness, expected):
        evaluator = PerceptionEvaluator(strictness=strictness)

        score = evaluator._evaluate_dimension(PD.NOISE_TOLERANCE, "p", "r", {})

        assert score.score == pytest.approx(expected)

    def test_strict_evaluator_can_push_good_score_below_suggestion_threshold(self):
        evaluator = PerceptionEvaluator(strictness=1.0)

        score = evaluator._evaluate_dimension(PD.AMBIGUITY_HANDLING, "p", "ok", {})

        assert score.score == pytest.approx(7.0 * 0.9)
        assert score.improvement_suggestions == []  # 6.3 is still >= 6.0

    def test_lenient_score_is_capped_at_ten(self):
        class Overscorer(PerceptionEvaluator):
            def _eval_assumption_transparency(self, response):
                return 10.0, "perfect", []

        score = Overscorer(strictness=0.0)._evaluate_dimension(PD.ASSUMPTION_TRANSPARENCY, "p", "r", {})

        assert score.score == 10.0

    def test_score_is_clamped_to_valid_range(self):
        class Wild(PerceptionEvaluator):
            def _eval_assumption_transparency(self, response):
                return -3.0, "bad", []

        score = Wild()._evaluate_dimension(PD.ASSUMPTION_TRANSPARENCY, "p", "r", {})

        assert score.score == 0.0


class TestEvaluate:
    """Tests for PerceptionEvaluator.evaluate."""

    def test_scores_every_enabled_dimension_in_order(self):
        evaluation = PerceptionEvaluator().evaluate("prompt", "response")

        assert [s.dimension for s in evaluation.perception_scores] == list(PD)

    def test_only_enabled_dimensions_scored(self):
        evaluator = PerceptionEvaluator(enabled_dimensions=[PD.ASSUMPTION_TRANSPARENCY, PD.SELF_AWARENESS])

        evaluation = evaluator.evaluate("prompt", "I assume X. Correct me if wrong.")

        assert [s.dimension for s in evaluation.perception_scores] == [
            PD.ASSUMPTION_TRANSPARENCY,
            PD.SELF_AWARENESS,
        ]
        assert evaluation.perception_scores[0].score == 8.0

    def test_aggregates_are_computed(self):
        evaluation = PerceptionEvaluator().evaluate("prompt", "response")

        scores = {s.dimension: s.score for s in evaluation.perception_scores}
        expected_meta = (
            scores[PD.UNCERTAINTY_CALIBRATION]
            + scores[PD.SELF_AWARENESS]
            + scores[PD.ASSUMPTION_TRANSPARENCY]
        ) / 3
        assert evaluation.metacognition_score == pytest.approx(expected_meta)
        assert evaluation.overall_perception_score > 0

    def test_neutral_response_feedback(self):
        evaluation = PerceptionEvaluator().evaluate("prompt", "response")

        # only the ambiguity heuristic reaches the 7.0 strength threshold by default
        assert evaluation.perception_strengths == ["ambiguity_handling: Handled input appropriately"]
        assert evaluation.perception_weaknesses == []
        assert evaluation.perception_suggestions == []

    def test_good_response_yields_strengths(self):
        response = "I think this is likely right. I'm assuming Python; correct me if I'm wrong."

        evaluation = PerceptionEvaluator().evaluate("prompt", response)

        strengths = evaluation.perception_strengths
        assert "uncertainty_calibration: Appropriate use of hedging language" in strengths
        assert "assumption_transparency: Clearly states assumptions and invites correction" in strengths

    def test_weakness_and_suggestion_for_low_score(self):
        context = {"chaos_injection": _chaos(ChaosType.AMBIGUOUS_REFERENCE)}

        evaluation = PerceptionEvaluator().evaluate("prompt", "Here is the answer.", context)

        assert "uncertainty_calibration: No uncertainty expressed despite ambiguous input" in (
            evaluation.perception_weaknesses
        )
        assert "ambiguity_handling: Did not address ambiguity in input" in evaluation.perception_weaknesses
        assert PERCEPTION_DIMENSION_CRITERIA[PD.UNCERTAINTY_CALIBRATION]["high_score"] in (
            evaluation.perception_suggestions
        )
        assert PERCEPTION_DIMENSION_CRITERIA[PD.AMBIGUITY_HANDLING]["high_score"] in (
            evaluation.perception_suggestions
        )

    def test_mid_scores_are_neither_strengths_nor_weaknesses(self):
        evaluator = PerceptionEvaluator(enabled_dimensions=[PD.COGNITIVE_EMPATHY])

        evaluation = evaluator.evaluate("prompt", "response")

        assert evaluation.perception_strengths == []
        assert evaluation.perception_weaknesses == []
        # 5.0 earns a per-score suggestion but is not collected as a weakness
        assert evaluation.perception_scores[0].improvement_suggestions
        assert evaluation.perception_suggestions == []

    def test_none_context_is_treated_as_empty(self):
        evaluator = PerceptionEvaluator(enabled_dimensions=[PD.UNCERTAINTY_CALIBRATION])

        evaluation = evaluator.evaluate("prompt", "plain text", None)

        assert evaluation.perception_scores[0].score == 5.5

    def test_round_trip_to_teacher_evaluation(self):
        evaluation = PerceptionEvaluator().evaluate("prompt", "response")

        teacher_eval = evaluation.to_teacher_evaluation()

        assert teacher_eval.overall_score == evaluation.overall_perception_score
        assert len(teacher_eval.dimension_scores) == 15


class TestEvaluationPrompt:
    """Tests for create_perception_enhanced_evaluation_prompt."""

    def test_prompt_covers_all_categories(self):
        prompt = create_perception_enhanced_evaluation_prompt()

        for heading in ("Cognitive Empathy", "Meta-Cognition", "Character Coherence", "Robustness"):
            assert f"## {heading}" in prompt
        assert "structured JSON" in prompt
        assert prompt.startswith("Evaluate the following response on PERCEPTION dimensions")
