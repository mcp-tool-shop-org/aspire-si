"""
Tests for Meta-Cognition module.

Tests uncertainty estimation, confidence calibration,
and reflective loop mechanisms.
"""

import os

os.environ["XFORMERS_DISABLED"] = "1"

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


class TestUncertaintyEstimate:
    """Tests for UncertaintyEstimate dataclass."""

    def test_basic_creation(self):
        """Test creating an uncertainty estimate."""
        estimate = UncertaintyEstimate(
            subject="The exact date of the event",
            uncertainty_type=UncertaintyType.FACTUAL,
            magnitude=0.7,
            source="No reliable source available",
            reducible_by=["checking primary sources", "asking the user"],
        )

        assert estimate.magnitude == 0.7
        assert estimate.is_reducible is True

    def test_to_natural_language(self):
        """Test conversion to natural language."""
        low_uncertainty = UncertaintyEstimate(
            subject="Python syntax",
            uncertainty_type=UncertaintyType.FACTUAL,
            magnitude=0.1,
            source="Well-documented",
        )

        high_uncertainty = UncertaintyEstimate(
            subject="user's specific needs",
            uncertainty_type=UncertaintyType.CONTEXTUAL,
            magnitude=0.9,
            source="Not enough context provided",
            reducible_by=["asking for clarification"],
        )

        low_text = low_uncertainty.to_natural_language()
        high_text = high_uncertainty.to_natural_language()

        assert "fairly confident" in low_text
        assert "very uncertain" in high_text
        assert "clarification" in high_text


class TestUncertaintyEstimator:
    """Tests for UncertaintyEstimator neural module."""

    @pytest.fixture
    def estimator(self):
        return UncertaintyEstimator(hidden_dim=64)

    def test_forward_pass(self, estimator):
        """Test forward pass produces expected outputs."""
        batch_size = 2
        seq_len = 10
        hidden_dim = 64

        hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
        attention_mask = torch.ones(batch_size, seq_len)

        outputs = estimator(hidden_states, attention_mask)

        assert "overall_uncertainty" in outputs
        assert "uncertainty_type_logits" in outputs
        assert "ood_score" in outputs

        assert outputs["overall_uncertainty"].shape == (batch_size, 1)
        assert outputs["ood_score"].shape == (batch_size, 1)

    def test_forward_without_mask(self, estimator):
        """Test forward pass works without attention mask."""
        hidden_states = torch.randn(2, 10, 64)

        outputs = estimator(hidden_states)

        assert outputs["overall_uncertainty"].shape[0] == 2

    def test_update_distribution_stats(self, estimator):
        """Test updating distribution statistics."""
        hidden_states = torch.randn(10, 10, 64)

        estimator.update_distribution_stats(hidden_states)

        assert estimator.num_samples.item() == 10

    def test_estimate_from_logits_entropy(self, estimator):
        """Test uncertainty estimation from logits using entropy."""
        # Confident distribution (low entropy)
        confident_logits = torch.zeros(2, 100)
        confident_logits[:, 0] = 10.0  # Strong peak

        # Uncertain distribution (high entropy)
        uncertain_logits = torch.zeros(2, 100)  # Uniform

        confident_uncertainty = estimator.estimate_from_logits(
            confident_logits, method="entropy"
        )
        uncertain_uncertainty = estimator.estimate_from_logits(
            uncertain_logits, method="entropy"
        )

        assert confident_uncertainty.mean() < uncertain_uncertainty.mean()

    def test_estimate_from_logits_margin(self, estimator):
        """Test uncertainty estimation using margin method."""
        logits = torch.randn(2, 100)

        uncertainty = estimator.estimate_from_logits(logits, method="margin")

        assert uncertainty.shape == (2,)
        assert (uncertainty >= 0).all() and (uncertainty <= 1).all()


class TestConfidenceCalibrator:
    """Tests for ConfidenceCalibrator."""

    @pytest.fixture
    def calibrator(self):
        return ConfidenceCalibrator(num_bins=10, smoothing=0.1)

    def test_record_calibration(self, calibrator):
        """Test recording calibration data."""
        calibrator.record(predicted=0.8, actual=1.0, domain="general")
        calibrator.record(predicted=0.8, actual=0.0, domain="general")

        assert len(calibrator.calibration_records["general"]) == 2

    def test_calibrate_no_data(self, calibrator):
        """Test calibration with no prior data regresses to mean."""
        raw = 0.9
        calibrated = calibrator.calibrate(raw, domain="unknown")

        # Should regress toward 0.5
        assert calibrated < raw

    def test_calibration_curve_update(self, calibrator):
        """Test that calibration curve is updated."""
        # Add many records to trigger curve update
        for i in range(100):
            # Simulate overconfident model
            calibrator.record(
                predicted=0.9,
                actual=1.0 if i % 2 == 0 else 0.0,  # 50% accuracy at 90% confidence
                domain="test",
            )

        # Curve should now exist
        assert "test" in calibrator.calibration_curves

    def test_get_confidence_level(self, calibrator):
        """Test converting confidence to discrete level."""
        assert calibrator.get_confidence_level(0.98) == ConfidenceLevel.CERTAIN
        assert calibrator.get_confidence_level(0.85) == ConfidenceLevel.HIGH
        assert calibrator.get_confidence_level(0.65) == ConfidenceLevel.MODERATE
        assert calibrator.get_confidence_level(0.45) == ConfidenceLevel.LOW
        assert calibrator.get_confidence_level(0.25) == ConfidenceLevel.VERY_LOW
        assert calibrator.get_confidence_level(0.10) == ConfidenceLevel.UNCERTAIN

    def test_get_hedging_language(self, calibrator):
        """Test getting hedging language for confidence levels."""
        assert calibrator.get_hedging_language(ConfidenceLevel.CERTAIN) == ""
        assert "believe" in calibrator.get_hedging_language(ConfidenceLevel.HIGH).lower()
        assert "think" in calibrator.get_hedging_language(ConfidenceLevel.MODERATE).lower()
        # LOW says "not certain" which contains "certain"
        assert "certain" in calibrator.get_hedging_language(ConfidenceLevel.LOW).lower()

    def test_get_calibration_error(self, calibrator):
        """Test ECE calculation."""
        # Perfect calibration: predicted confidence matches accuracy
        for i in range(100):
            conf = (i % 10) / 10 + 0.05  # 0.05, 0.15, ..., 0.95
            actual = 1.0 if torch.rand(1).item() < conf else 0.0
            calibrator.record(predicted=conf, actual=actual, domain="perfect")

        ece = calibrator.get_calibration_error("perfect")

        # Should be relatively low (not perfect due to randomness)
        assert ece < 0.5


class TestReflectiveLoop:
    """Tests for ReflectiveLoop."""

    @pytest.fixture
    def reflector(self):
        return ReflectiveLoop()

    def test_default_prompts(self, reflector):
        """Test that default prompts are loaded."""
        assert len(reflector.reflection_prompts) > 0
        assert any("consistent" in p.lower() for p in reflector.reflection_prompts)

    def test_reflect_produces_insights(self, reflector):
        """Test that reflect produces insights."""
        insights = reflector.reflect(
            context="User asking about Python programming",
            current_response="Python is a great language for beginners.",
            prompt_indices=[0, 1],  # Use first two prompts
        )

        assert len(insights) == 2
        assert all(isinstance(i, ReflectiveInsight) for i in insights)

    def test_categorize_prompt(self, reflector):
        """Test prompt categorization."""
        assert reflector._categorize_prompt("Is this consistent?") == "consistency"
        assert reflector._categorize_prompt("What assumption am I making?") == "assumptions"
        assert reflector._categorize_prompt("How would this appear?") == "perspective"
        assert reflector._categorize_prompt("What's missing?") == "completeness"

    def test_select_relevant_prompts(self, reflector):
        """Test automatic prompt selection based on content."""
        # Response with strong claims
        response = "Obviously, this is definitely the only correct approach."

        prompts = reflector._select_relevant_prompts("Some context", response)

        # Should include some prompts
        assert len(prompts) > 0
        assert len(prompts) <= 5  # Max limit

    def test_get_reflection_prompt_for_training(self, reflector):
        """Test generating full reflection prompt."""
        prompt = reflector.get_reflection_prompt_for_training(
            context="Technical question",
            response="Here's the answer.",
        )

        assert "reflect" in prompt.lower()
        assert "?" in prompt  # Should have questions

    def test_pattern_tracking(self, reflector):
        """Test that patterns are tracked."""
        for _ in range(5):
            reflector.reflect("context", "response", [0])  # Consistency checks

        summary = reflector.get_pattern_summary()

        assert summary["total_reflections"] == 5
        assert "consistency" in summary["patterns"]


class TestMetaCognitionModule:
    """Tests for combined MetaCognitionModule."""

    @pytest.fixture
    def module(self):
        return MetaCognitionModule(hidden_dim=64)

    def test_forward_pass(self, module):
        """Test forward pass produces expected outputs."""
        hidden_states = torch.randn(2, 10, 64)
        attention_mask = torch.ones(2, 10)

        outputs = module(hidden_states, attention_mask)

        assert "should_hedge" in outputs
        assert "should_clarify" in outputs
        assert "should_ask_user" in outputs
        assert "meta_confidence" in outputs

        assert outputs["should_hedge"].shape == (2, 1)

    def test_forward_with_logits(self, module):
        """Test forward pass with output logits."""
        hidden_states = torch.randn(2, 10, 64)
        output_logits = torch.randn(2, 10, 1000)

        outputs = module(hidden_states, output_logits=output_logits)

        assert "overall_uncertainty" in outputs

    def test_get_action_recommendation(self, module):
        """Test getting action recommendation."""
        hidden_states = torch.randn(1, 10, 64)
        outputs = module(hidden_states)

        # Manually set high values to test recommendation generation
        outputs["should_hedge"] = torch.tensor([[0.8]])
        outputs["should_clarify"] = torch.tensor([[0.7]])
        outputs["should_ask_user"] = torch.tensor([[0.3]])
        outputs["meta_confidence"] = torch.tensor([[0.3]])

        recommendation = module.get_action_recommendation(outputs)

        assert "hedging" in recommendation.lower()
        assert "clarify" in recommendation.lower()

    def test_generate_metacognitive_prompt(self, module):
        """Test generating metacognitive prompt."""
        hidden_states = torch.randn(1, 10, 64)
        outputs = module(hidden_states)

        prompt = module.generate_metacognitive_prompt(outputs, context="Test context")

        assert "METACOGNITIVE" in prompt
        assert len(prompt) > 50


class TestCalibrationRecord:
    """Tests for CalibrationRecord dataclass."""

    def test_creation(self):
        """Test creating a calibration record."""
        record = CalibrationRecord(
            predicted_confidence=0.85,
            actual_correctness=1.0,
            domain="programming",
        )

        assert record.predicted_confidence == 0.85
        assert record.actual_correctness == 1.0
        assert record.timestamp > 0


class TestReflectiveInsight:
    """Tests for ReflectiveInsight dataclass."""

    def test_creation(self):
        """Test creating a reflective insight."""
        insight = ReflectiveInsight(
            category="consistency",
            observation="Previous statement contradicted",
            implication="May confuse user",
            action="Clarify the contradiction",
            confidence=0.7,
        )

        assert insight.category == "consistency"
        assert insight.action is not None


class TestIntegration:
    """Integration tests for metacognition components."""

    def test_full_metacognition_pipeline(self):
        """Test full pipeline from hidden states to recommendations."""
        module = MetaCognitionModule(hidden_dim=64)

        # Simulate a forward pass
        hidden_states = torch.randn(1, 10, 64)
        output_logits = torch.randn(1, 10, 100)

        outputs = module(hidden_states, output_logits=output_logits)

        # Get recommendation
        recommendation = module.get_action_recommendation(outputs)
        assert isinstance(recommendation, str)

        # Generate prompt
        prompt = module.generate_metacognitive_prompt(outputs)
        assert isinstance(prompt, str)

    def test_calibrator_with_estimator(self):
        """Test using calibrator with estimator outputs."""
        estimator = UncertaintyEstimator(hidden_dim=64)
        calibrator = ConfidenceCalibrator()

        # Get uncertainty estimate
        hidden_states = torch.randn(1, 10, 64)
        outputs = estimator(hidden_states)

        raw_confidence = 1 - outputs["overall_uncertainty"].mean().item()

        # Calibrate
        calibrated = calibrator.calibrate(raw_confidence)

        assert 0 <= calibrated <= 1


# Windows compatibility
if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    pytest.main([__file__, "-v"])
