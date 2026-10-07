"""Additional behaviour tests for aspire.perception.metacognition (coverage gaps)."""

import pytest
import torch

from aspire.perception.metacognition import (
    CalibrationRecord,
    ConfidenceCalibrator,
    ConfidenceLevel,
    MetaCognitionModule,
    ReflectiveInsight,
    ReflectiveLoop,
    UncertaintyEstimate,
    UncertaintyEstimator,
    UncertaintyType,
)

# ---------------------------------------------------------------------------
# UncertaintyEstimate
# ---------------------------------------------------------------------------


class TestUncertaintyEstimateText:
    @pytest.mark.parametrize(
        "magnitude,prefix",
        [
            (0.0, "I'm fairly confident about"),
            (0.19, "I'm fairly confident about"),
            (0.2, "I'm somewhat uncertain about"),
            (0.39, "I'm somewhat uncertain about"),
            (0.4, "I have moderate uncertainty about"),
            (0.59, "I have moderate uncertainty about"),
            (0.6, "I'm quite uncertain about"),
            (0.79, "I'm quite uncertain about"),
            (0.8, "I'm very uncertain about"),
            (1.0, "I'm very uncertain about"),
        ],
    )
    def test_prefix_bands(self, magnitude, prefix):
        est = UncertaintyEstimate("the date", UncertaintyType.FACTUAL, magnitude, "No source.")
        assert est.to_natural_language() == f"{prefix} the date. No source."

    def test_reducible_by_lists_at_most_three(self):
        est = UncertaintyEstimate(
            "x", UncertaintyType.PROCEDURAL, 0.5, "Unclear.", reducible_by=["a", "b", "c", "d"]
        )
        text = est.to_natural_language()
        assert text.endswith("This could be clarified by: a, b, c")
        assert "d" not in text.split("by:")[1]


# ---------------------------------------------------------------------------
# UncertaintyEstimator
# ---------------------------------------------------------------------------


@pytest.fixture
def estimator():
    torch.manual_seed(0)
    est = UncertaintyEstimator(hidden_dim=8)
    est.eval()
    return est


class TestUncertaintyEstimatorForward:
    def test_pooling_with_float_mask_ignores_padding(self, estimator):
        hs = torch.randn(2, 4, 8)
        mask = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
        out = estimator(hs, mask)
        torch.testing.assert_close(out["pooled"][0], hs[0, :2].mean(dim=0))
        torch.testing.assert_close(out["pooled"][1], hs[1].mean(dim=0))

    def test_unmasked_pooling_is_plain_mean(self, estimator):
        hs = torch.randn(3, 5, 8)
        out = estimator(hs)
        torch.testing.assert_close(out["pooled"], hs.mean(dim=1))
        assert out["overall_uncertainty"].shape == (3, 1)
        assert out["uncertainty_type_logits"].shape == (3, len(UncertaintyType))
        assert out["ood_score"].shape == (3, 1)
        assert torch.all((out["overall_uncertainty"] >= 0) & (out["overall_uncertainty"] <= 1))

    def test_mahalanobis_blend_only_after_enough_samples(self, estimator):
        hs = torch.randn(2, 3, 8)
        base = estimator(hs)["ood_score"]
        estimator.num_samples = torch.tensor(100.0)  # boundary: strictly greater is required
        torch.testing.assert_close(estimator(hs)["ood_score"], base)

        estimator.num_samples = torch.tensor(101.0)
        estimator.mean_hidden = torch.zeros(8)
        estimator.var_hidden = torch.full((8,), 2.0)
        blended = estimator(hs)["ood_score"]
        pooled = hs.mean(dim=1)
        mahal = ((pooled - 0.0) ** 2 / 2.0).mean(dim=-1, keepdim=True)
        expected = (base + torch.sigmoid(mahal - 1.0)) / 2
        torch.testing.assert_close(blended, expected)

    def test_far_away_input_scores_more_ood_than_typical_input(self, estimator):
        estimator.num_samples = torch.tensor(500.0)
        near = torch.zeros(1, 2, 8)
        far = torch.full((1, 2, 8), 10.0)
        # The learned head is shared, so any difference comes from the Mahalanobis term.
        near_ood = estimator(near)["ood_score"].item()
        far_ood = estimator(far)["ood_score"].item()
        assert far_ood > near_ood


class TestDistributionStats:
    def test_first_update_sets_mean_and_count(self, estimator):
        hs = torch.randn(4, 3, 8)
        estimator.update_distribution_stats(hs)
        torch.testing.assert_close(estimator.mean_hidden, hs.mean(dim=1).mean(dim=0))
        assert estimator.num_samples.item() == 4.0
        torch.testing.assert_close(estimator.var_hidden, hs.mean(dim=1).var(dim=0))

    def test_incremental_updates_track_the_overall_mean(self, estimator):
        a, b = torch.randn(3, 2, 8), torch.randn(5, 2, 8)
        estimator.update_distribution_stats(a)
        estimator.update_distribution_stats(b)
        all_pooled = torch.cat([a.mean(dim=1), b.mean(dim=1)])
        torch.testing.assert_close(estimator.mean_hidden, all_pooled.mean(dim=0), atol=1e-6, rtol=1e-5)
        assert estimator.num_samples.item() == 8.0
        assert torch.all(estimator.var_hidden > 0) and torch.isfinite(estimator.var_hidden).all()

    def test_masked_update_uses_masked_pooling(self, estimator):
        hs = torch.randn(2, 4, 8)
        mask = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
        estimator.update_distribution_stats(hs, mask)
        expected = torch.stack([hs[0, :2].mean(dim=0), hs[1, :1].mean(dim=0)]).mean(dim=0)
        torch.testing.assert_close(estimator.mean_hidden, expected)

    def test_update_does_not_build_graph(self, estimator):
        hs = torch.randn(2, 3, 8, requires_grad=True)
        estimator.update_distribution_stats(hs)
        assert not estimator.mean_hidden.requires_grad

    def test_single_sample_update_keeps_variance_finite(self, estimator):
        estimator.update_distribution_stats(torch.randn(1, 3, 8))
        assert torch.isfinite(estimator.var_hidden).all()


class TestLogitUncertainty:
    def test_entropy_extremes(self, estimator):
        uniform = torch.zeros(2, 10)
        peaked = torch.tensor([[50.0] + [0.0] * 9] * 2)
        assert estimator.estimate_from_logits(uniform, "entropy") == pytest.approx([1.0, 1.0], abs=1e-4)
        assert estimator.estimate_from_logits(peaked, "entropy").max().item() < 1e-3

    def test_margin_extremes(self, estimator):
        uniform = torch.zeros(1, 4)
        peaked = torch.tensor([[30.0, 0.0, 0.0, 0.0]])
        assert estimator.estimate_from_logits(uniform, "margin").item() == pytest.approx(1.0)
        assert estimator.estimate_from_logits(peaked, "margin").item() == pytest.approx(0.0, abs=1e-4)

    def test_variance_extremes(self, estimator):
        uniform = torch.zeros(1, 4)
        peaked = torch.tensor([[30.0, 0.0, 0.0, 0.0]])
        assert estimator.estimate_from_logits(uniform, "variance").item() == pytest.approx(1.0)
        # one-hot: sum(p^2) - 1/V = 1 - 0.25 -> 1 - 0.75
        assert estimator.estimate_from_logits(peaked, "variance").item() == pytest.approx(0.25, abs=1e-3)

    def test_sequence_shaped_logits_keep_leading_dims(self, estimator):
        logits = torch.randn(2, 3, 7)
        for method in ("entropy", "margin", "variance"):
            assert estimator.estimate_from_logits(logits, method).shape == (2, 3)

    def test_unknown_method_rejected(self, estimator):
        with pytest.raises(ValueError, match="Unknown method: bogus"):
            estimator.estimate_from_logits(torch.zeros(1, 3), "bogus")


# ---------------------------------------------------------------------------
# ConfidenceCalibrator
# ---------------------------------------------------------------------------


class TestCalibrator:
    def test_uncalibrated_domain_regresses_toward_half(self):
        cal = ConfidenceCalibrator()
        assert cal.calibrate(1.0) == pytest.approx(0.9)
        assert cal.calibrate(0.0) == pytest.approx(0.1)
        assert cal.calibrate(0.5) == 0.5

    def test_curve_is_built_every_hundred_records(self):
        cal = ConfidenceCalibrator()
        for _ in range(99):
            cal.record(0.9, 1.0, "d")
        assert "d" not in cal.calibration_curves
        cal.record(0.9, 1.0, "d")
        assert len(cal.calibration_curves["d"]) == 10
        assert isinstance(cal.calibration_records["d"][0], CalibrationRecord)

    def test_curve_values_smoothed_toward_expected_and_empty_bins_default(self):
        cal = ConfidenceCalibrator(num_bins=10, smoothing=0.1)
        for _ in range(100):
            cal.record(0.05, 1.0, "d")  # always right though only 5% confident
        curve = cal.calibration_curves["d"]
        assert curve[0] == pytest.approx(0.1 * 0.05 + 0.9 * 1.0)
        assert curve[1] == pytest.approx(0.15)  # empty bin -> expected value
        assert curve[9] == pytest.approx(0.95)

    def test_domains_are_independent(self):
        cal = ConfidenceCalibrator()
        for _ in range(100):
            cal.record(0.95, 0.0, "math")
        assert cal.calibrate(0.95, "math") != cal.calibrate(0.95, "other")
        assert cal.calibrate(0.95, "other") == pytest.approx(0.5 + 0.45 * 0.8)

    def test_overconfident_domain_is_pulled_down(self):
        cal = ConfidenceCalibrator()
        for _ in range(100):
            cal.record(0.85, 0.0, "d")  # claims 85%, is never right
        uncalibrated = ConfidenceCalibrator().calibrate(0.85)
        assert cal.calibrate(0.85, "d") < uncalibrated
        assert cal.calibrate(0.8, "d") == pytest.approx(0.1 * 0.85)  # bin start: smoothed accuracy 0

    def test_interpolation_between_bins_and_last_bin(self):
        cal = ConfidenceCalibrator(num_bins=4)
        cal.calibration_curves["d"] = [0.1, 0.3, 0.6, 0.9]
        assert cal.calibrate(0.125, "d") == pytest.approx(0.2)
        assert cal.calibrate(0.25, "d") == pytest.approx(0.3)
        assert cal.calibrate(0.9, "d") == pytest.approx(0.9)  # last bin has no neighbour
        assert cal.calibrate(1.0, "d") == pytest.approx(0.9)

    def test_calibrated_value_is_clamped(self):
        cal = ConfidenceCalibrator(num_bins=2)
        cal.calibration_curves["d"] = [-1.0, 3.0]
        assert cal.calibrate(0.1, "d") == 0.0  # interpolates to -0.2
        assert cal.calibrate(0.4, "d") == 1.0  # interpolates to 2.2
        assert cal.calibrate(1.0, "d") == 1.0  # last bin, 3.0

    def test_calibration_error_defaults_and_values(self):
        cal = ConfidenceCalibrator()
        assert cal.get_calibration_error("none") == 0.5
        cal.calibration_records["empty"] = []
        assert cal.get_calibration_error("empty") == 0.5
        # Perfect calibration: 80% confident and right 80% of the time -> ECE 0
        for i in range(10):
            cal.record(0.8, 1.0 if i < 8 else 0.0, "ok")
        assert cal.get_calibration_error("ok") == pytest.approx(0.0, abs=1e-9)
        # Always 90% confident, always wrong
        for _ in range(4):
            cal.record(0.9, 0.0, "bad")
        assert cal.get_calibration_error("bad") == pytest.approx(0.9)

    def test_calibration_error_is_weighted_across_bins(self):
        cal = ConfidenceCalibrator()
        cal.record(0.1, 0.0, "d")  # bin 1: |0.1 - 0| = 0.1, weight 1/4
        for _ in range(3):
            cal.record(0.9, 0.0, "d")  # bin 9: 0.9, weight 3/4
        assert cal.get_calibration_error("d") == pytest.approx(0.25 * 0.1 + 0.75 * 0.9)

    def test_confidence_level_thresholds(self):
        cal = ConfidenceCalibrator()
        cases = [
            (1.0, ConfidenceLevel.CERTAIN),
            (0.95, ConfidenceLevel.CERTAIN),
            (0.949, ConfidenceLevel.HIGH),
            (0.80, ConfidenceLevel.HIGH),
            (0.60, ConfidenceLevel.MODERATE),
            (0.40, ConfidenceLevel.LOW),
            (0.20, ConfidenceLevel.VERY_LOW),
            (0.199, ConfidenceLevel.UNCERTAIN),
            (0.0, ConfidenceLevel.UNCERTAIN),
        ]
        for value, level in cases:
            assert cal.get_confidence_level(value) is level, value

    def test_hedging_language_for_each_level(self):
        cal = ConfidenceCalibrator()
        assert cal.get_hedging_language(ConfidenceLevel.CERTAIN) == ""
        assert cal.get_hedging_language(ConfidenceLevel.HIGH) == "I believe "
        assert cal.get_hedging_language(ConfidenceLevel.MODERATE) == "I think "
        assert cal.get_hedging_language(ConfidenceLevel.LOW).startswith("I'm not certain")
        assert "best guess" in cal.get_hedging_language(ConfidenceLevel.VERY_LOW)
        assert cal.get_hedging_language(ConfidenceLevel.UNCERTAIN).startswith("I don't know")
        assert cal.get_hedging_language("not a level") == ""


# ---------------------------------------------------------------------------
# ReflectiveLoop
# ---------------------------------------------------------------------------


class TestReflectiveLoop:
    def test_default_prompts_and_custom_prompts(self):
        assert len(ReflectiveLoop().reflection_prompts) == 12
        assert ReflectiveLoop(reflection_prompts=["only one"]).reflection_prompts == ["only one"]

    def test_reflect_with_explicit_indices(self):
        loop = ReflectiveLoop()
        insights = loop.reflect("ctx", "resp", prompt_indices=[0, 4])
        assert [i.category for i in insights] == ["consistency", "perspective"]
        assert insights[0].observation == f"Reflection prompt: {loop.reflection_prompts[0]}"
        assert all(isinstance(i, ReflectiveInsight) for i in insights)
        assert loop.insights == insights
        assert loop.detected_patterns == {"consistency": 1, "perspective": 1}

    def test_reflect_auto_selects_when_indices_missing_or_empty(self):
        loop = ReflectiveLoop()
        auto = loop.reflect("short", "fine answer")
        assert [i.category for i in auto] == ["perspective", "completeness"]
        assert len(loop.reflect("short", "fine answer", prompt_indices=[])) == 2

    def test_reflect_skips_prompts_that_yield_no_insight(self):
        loop = ReflectiveLoop()
        results = iter([None, ReflectiveInsight("quality", "o", "i")])
        loop._generate_insight = lambda prompt, context, response: next(results)
        insights = loop.reflect("c", "r", prompt_indices=[0, 1])
        assert [i.category for i in insights] == ["quality"]
        assert loop.detected_patterns == {"quality": 1}

    def test_select_prompts_triggers(self):
        loop = ReflectiveLoop()
        p = loop.reflection_prompts
        assert loop._select_relevant_prompts("c", "plain") == [p[4], p[6]]
        assert loop._select_relevant_prompts("c" * 501, "plain") == [p[0], p[4], p[6]]
        assert loop._select_relevant_prompts("c" * 500, "plain")[0] == p[4]  # needs > 500
        assert p[2] in loop._select_relevant_prompts("c", "Obviously this works")
        assert p[9] in loop._select_relevant_prompts("c", "You must do it")
        assert p[10] in loop._select_relevant_prompts("c", "x" * 1001)
        assert p[10] not in loop._select_relevant_prompts("c", "x" * 1000)

    def test_select_prompts_never_exceeds_five(self):
        loop = ReflectiveLoop()
        selected = loop._select_relevant_prompts("c" * 600, "obviously always " + "x" * 1100)
        assert len(selected) == 5

    def test_perspective_and_completeness_always_selected(self):
        loop = ReflectiveLoop()
        selected = loop._select_relevant_prompts("c" * 600, "obviously always " + "x" * 1100)
        assert loop.reflection_prompts[4] in selected
        assert loop.reflection_prompts[6] in selected

    @pytest.mark.parametrize(
        "prompt,category",
        [
            ("Is this consistent?", "consistency"),
            ("Am I contradicting myself?", "consistency"),
            ("What assumptions am I making?", "assumptions"),
            ("What am I taking for granted?", "assumptions"),
            ("How might this appear?", "perspective"),
            ("From the user's perspective", "perspective"),
            ("What context might I be missing?", "completeness"),
            ("What haven't I addressed?", "completeness"),
            ("Is there bias here?", "bias"),
            ("Is this unfair?", "bias"),
            ("Is this helpful?", "quality"),
            ("Am I being clear?", "quality"),
            ("Something else entirely", "general"),
        ],
    )
    def test_categorize_prompt(self, prompt, category):
        assert ReflectiveLoop()._categorize_prompt(prompt) == category

    def test_default_prompts_categorize_sensibly(self):
        loop = ReflectiveLoop()
        categories = [loop._categorize_prompt(p) for p in loop.reflection_prompts]
        assert categories[0] == "consistency" and categories[2] == "assumptions"
        assert categories[4] == "perspective" and categories[6] == "completeness"
        assert categories[8] == "bias" and categories[10] == "quality"

    def test_insights_are_pruned_to_max(self):
        loop = ReflectiveLoop(max_insights=3)
        for i in range(5):
            loop._record_insight(ReflectiveInsight("c", f"obs{i}", "imp"))
        assert [i.observation for i in loop.insights] == ["obs2", "obs3", "obs4"]
        # pattern counts are cumulative, not pruned
        assert loop.detected_patterns == {"c": 5}

    def test_pattern_summary_empty_and_populated(self):
        loop = ReflectiveLoop()
        assert loop.get_pattern_summary() == {"patterns": {}, "total_reflections": 0}
        loop.detected_patterns = {"bias": 1, "quality": 3}
        summary = loop.get_pattern_summary()
        assert summary["total_reflections"] == 4
        assert summary["patterns"]["quality"] == {"count": 3, "percentage": 75.0}
        assert summary["most_common"] == "quality"

    def test_training_prompt_numbers_selected_prompts(self):
        loop = ReflectiveLoop()
        text = loop.get_reflection_prompt_for_training("c", "plain")
        assert text.startswith("Before finalizing this response")
        assert f"1. {loop.reflection_prompts[4]}\n" in text
        assert f"2. {loop.reflection_prompts[6]}\n" in text
        assert "3. " not in text
        assert "Is the confidence level appropriate?" in text


# ---------------------------------------------------------------------------
# MetaCognitionModule
# ---------------------------------------------------------------------------


@pytest.fixture
def meta():
    torch.manual_seed(0)
    module = MetaCognitionModule(hidden_dim=8)
    module.eval()
    return module


class TestMetaCognitionModule:
    def test_forward_without_logits(self, meta):
        out = meta(torch.randn(2, 3, 8))
        for key in ("should_hedge", "should_clarify", "should_ask_user", "meta_confidence"):
            assert out[key].shape == (2, 1)
            assert torch.all((out[key] >= 0) & (out[key] <= 1))
        assert {"overall_uncertainty", "ood_score", "pooled", "uncertainty_type_logits"} <= set(out)

    def test_forward_logits_feed_the_meta_head(self, meta):
        hs = torch.randn(2, 3, 8)
        flat = torch.zeros(2, 3, 11)
        peaked = torch.zeros(2, 3, 11)
        peaked[..., 0] = 40.0
        out_none = meta(hs)
        out_flat = meta(hs, output_logits=flat)
        out_peaked = meta(hs, output_logits=peaked)
        # Shapes are unchanged, but the entropy feature changes the predictions.
        assert out_flat["meta_confidence"].shape == (2, 1)
        assert not torch.allclose(out_flat["meta_confidence"], out_peaked["meta_confidence"])
        assert not torch.allclose(out_none["meta_confidence"], out_flat["meta_confidence"])

    def test_forward_with_attention_mask(self, meta):
        hs = torch.randn(2, 4, 8)
        mask = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
        out = meta(hs, attention_mask=mask)
        torch.testing.assert_close(out["pooled"][0], hs[0, :2].mean(dim=0))

    def test_forward_accepts_batch_by_vocab_logits(self, meta):
        meta(torch.randn(2, 3, 8), output_logits=torch.randn(2, 11))

    def test_gradients_flow_through_meta_head(self):
        module = MetaCognitionModule(hidden_dim=8)
        out = module(torch.randn(2, 3, 8), output_logits=torch.randn(2, 3, 5))
        out["meta_confidence"].sum().backward()
        assert module.meta_head[0].weight.grad is not None
        assert module.uncertainty_estimator.uncertainty_head[0].weight.grad is not None

    def _outputs(self, hedge=0.0, clarify=0.0, ask=0.0, conf=0.9, unc=0.1, ood=0.1):
        t = lambda v: torch.tensor([[v]])  # noqa: E731
        return {
            "should_hedge": t(hedge),
            "should_clarify": t(clarify),
            "should_ask_user": t(ask),
            "meta_confidence": t(conf),
            "overall_uncertainty": t(unc),
            "ood_score": t(ood),
        }

    def test_recommendation_adequate(self, meta):
        assert meta.get_action_recommendation(self._outputs()) == "Response confidence is adequate"

    def test_recommendation_below_thresholds_is_adequate(self, meta):
        outputs = self._outputs(hedge=0.5, clarify=0.5, ask=0.5, conf=0.5)
        assert meta.get_action_recommendation(outputs) == "Response confidence is adequate"

    def test_recommendation_combines_all_triggers_in_order(self, meta):
        rec = meta.get_action_recommendation(self._outputs(hedge=0.9, clarify=0.8, ask=0.7, conf=0.1))
        assert rec.split("; ") == [
            "Add hedging language to express uncertainty",
            "Clarify or expand on key points",
            "Ask the user for clarification",
            "Consider that this response may need revision",
        ]

    def test_recommendation_single_triggers(self, meta):
        assert "hedging" in meta.get_action_recommendation(self._outputs(hedge=0.7))
        assert "Clarify" in meta.get_action_recommendation(self._outputs(clarify=0.7))
        assert "Ask the user" in meta.get_action_recommendation(self._outputs(ask=0.7))
        assert "revision" in meta.get_action_recommendation(self._outputs(conf=0.3))

    def test_metacognitive_prompt_quiet_case(self, meta):
        prompt = meta.generate_metacognitive_prompt(self._outputs(unc=0.5, ood=0.5))
        assert prompt.startswith("METACOGNITIVE CHECK:")
        assert "Uncertainty level" not in prompt and "unusual" not in prompt
        assert "Before finalizing this response" in prompt

    def test_metacognitive_prompt_flags_uncertainty_and_ood(self, meta):
        prompt = meta.generate_metacognitive_prompt(self._outputs(unc=0.9, ood=0.9), context="ctx")
        # confidence = 1 - 0.9 = 0.1 -> UNCERTAIN
        assert f"Uncertainty level: {ConfidenceLevel.UNCERTAIN.value}." in prompt
        assert "This query may be unusual." in prompt

    def test_metacognitive_prompt_uncertainty_level_follows_value(self, meta):
        prompt = meta.generate_metacognitive_prompt(self._outputs(unc=0.55, ood=0.0))
        assert f"Uncertainty level: {ConfidenceLevel.LOW.value}." in prompt  # confidence 0.45
