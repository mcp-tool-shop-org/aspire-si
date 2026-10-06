"""
Tests for Theory of Mind module.

Tests mental state tracking, belief/intent/emotion modeling,
and perspective-taking capabilities.
"""

import os

os.environ["XFORMERS_DISABLED"] = "1"

import pytest
import torch

from aspire.perception.theory_of_mind import (
    BeliefState,
    EmotionalState,
    EmotionalValence,
    EmotionType,
    IntentCategory,
    IntentState,
    KnowledgeState,
    MentalStateTracker,
)


class TestBeliefState:
    """Tests for BeliefState tracking."""

    def test_add_explicit_belief(self):
        """Test adding an explicit belief."""
        state = BeliefState()
        state.add_belief("Python is a good language", confidence=0.8, explicit=True)

        assert "Python is a good language" in state.explicit_beliefs
        assert state.explicit_beliefs["Python is a good language"] == 0.8

    def test_add_inferred_belief(self):
        """Test adding an inferred belief."""
        state = BeliefState()
        state.add_belief("User knows programming", confidence=0.6, explicit=False)

        assert "User knows programming" in state.inferred_beliefs
        assert state.inferred_beliefs["User knows programming"] == 0.6

    def test_confidence_clamping(self):
        """Test that confidence is clamped to 0-1 range."""
        state = BeliefState()
        state.add_belief("Overconfident", confidence=1.5)
        state.add_belief("Underconfident", confidence=-0.5)

        assert state.explicit_beliefs["Overconfident"] == 1.0
        assert state.explicit_beliefs["Underconfident"] == 0.0

    def test_flag_misconception(self):
        """Test flagging a belief as a misconception."""
        state = BeliefState()
        state.add_belief("Python is compiled", confidence=0.7)
        state.flag_misconception("Python is compiled", "Python is interpreted")

        assert "Python is compiled" in state.misconceptions
        assert "Python is compiled" not in state.explicit_beliefs
        assert state.misconceptions["Python is compiled"] == "Python is interpreted"

    def test_get_belief_confidence(self):
        """Test getting confidence for beliefs."""
        state = BeliefState()
        state.add_belief("Explicit belief", confidence=0.9, explicit=True)
        state.add_belief("Inferred belief", confidence=0.8, explicit=False)

        # Explicit beliefs return full confidence
        assert state.get_belief_confidence("Explicit belief") == 0.9
        # Inferred beliefs are discounted
        assert state.get_belief_confidence("Inferred belief") == pytest.approx(0.64)
        # Unknown beliefs return 0
        assert state.get_belief_confidence("Unknown") == 0.0


class TestIntentState:
    """Tests for IntentState tracking."""

    def test_update_intent(self):
        """Test updating intent state."""
        state = IntentState()
        state.update_intent(
            surface=IntentCategory.DEBUGGING,
            surface_conf=0.85,
            underlying=IntentCategory.LEARNING,
            underlying_conf=0.7,
        )

        assert state.surface_intent == IntentCategory.DEBUGGING
        assert state.surface_confidence == 0.85
        assert state.underlying_intent == IntentCategory.LEARNING
        assert state.underlying_confidence == 0.7

    def test_intent_trajectory_tracking(self):
        """Test that intent trajectory is tracked."""
        state = IntentState()

        intents = [
            IntentCategory.SEEKING_INFORMATION,
            IntentCategory.CLARIFYING,
            IntentCategory.DEBUGGING,
        ]

        for intent in intents:
            state.update_intent(intent, 0.8)

        assert len(state.intent_trajectory) == 3
        assert state.intent_trajectory[0][0] == IntentCategory.SEEKING_INFORMATION
        assert state.intent_trajectory[2][0] == IntentCategory.DEBUGGING

    def test_intent_stability(self):
        """Test that intent stability decreases with rapid changes."""
        state = IntentState()
        initial_stability = state.intent_stability

        # Rapidly change intents
        for intent in [IntentCategory.LEARNING, IntentCategory.DEBUGGING,
                       IntentCategory.VENTING, IntentCategory.CLARIFYING]:
            state.update_intent(intent, 0.8)

        assert state.intent_stability < initial_stability

    def test_detect_frustration_signals(self):
        """Test frustration detection from intent patterns."""
        state = IntentState()

        # Need at least 3 trajectory entries
        state.update_intent(IntentCategory.SEEKING_INFORMATION, 0.7)
        state.update_intent(IntentCategory.CLARIFYING, 0.7)
        state.update_intent(IntentCategory.CLARIFYING, 0.7)

        # Manually set frustration indicators
        state.intent_stability = 0.3  # Low stability
        state.urgency = 0.8  # High urgency

        frustration = state.detect_frustration_signals()
        # Should detect: low stability (0.3) + clarifying count >= 2 (0.3) + high urgency (0.2) = 0.8
        assert frustration >= 0.5  # Should detect frustration


class TestEmotionalState:
    """Tests for EmotionalState tracking."""

    def test_update_emotion(self):
        """Test updating emotional state."""
        state = EmotionalState()
        state.update_emotion(EmotionType.CURIOSITY, intensity=0.8)

        assert state.primary_emotion == EmotionType.CURIOSITY
        assert state.primary_intensity == 0.8
        assert state.valence == EmotionalValence.POSITIVE

    def test_valence_detection(self):
        """Test that valence is correctly detected."""
        state = EmotionalState()

        # Test positive emotion
        state.update_emotion(EmotionType.ENTHUSIASM, 0.7)
        assert state.valence == EmotionalValence.POSITIVE

        # Test negative emotion
        state.update_emotion(EmotionType.FRUSTRATION, 0.7)
        assert state.valence == EmotionalValence.NEGATIVE

        # Test neutral emotion
        state.update_emotion(EmotionType.ANALYTICAL, 0.7)
        assert state.valence == EmotionalValence.NEUTRAL

    def test_emotional_trend(self):
        """Test emotional trend detection."""
        state = EmotionalState()

        # Simulate improving trend - need positive > negative + 1
        # So we need 4+ positive and 1- negative in last 5
        for emotion in [EmotionType.FRUSTRATION,  # 1 negative
                        EmotionType.CURIOSITY,     # 1 positive
                        EmotionType.SATISFACTION,  # 2 positive
                        EmotionType.ENTHUSIASM,    # 3 positive
                        EmotionType.RELIEF]:       # 4 positive
            state.update_emotion(emotion, 0.7)

        trend = state.get_emotional_trend()
        # 4 positive > 1 negative + 1, so should be "improving"
        assert trend == "improving"

    def test_rapport_adjustment(self):
        """Test rapport level adjustment."""
        state = EmotionalState()
        initial_rapport = state.rapport_level

        state.adjust_rapport(0.2)
        assert state.rapport_level == initial_rapport + 0.2

        # Test clamping
        state.adjust_rapport(10.0)
        assert state.rapport_level == 1.0


class TestKnowledgeState:
    """Tests for KnowledgeState tracking."""

    def test_record_topic(self):
        """Test recording discussed topics."""
        state = KnowledgeState()
        state.record_topic("machine learning")
        state.record_topic("machine learning")
        state.record_topic("deep learning")

        assert state.topics_covered["machine learning"] == 2
        assert state.topics_covered["deep learning"] == 1

    def test_has_been_explained(self):
        """Test checking if topic has been explained."""
        state = KnowledgeState()
        state.record_topic("topic A")
        state.record_topic("topic A")
        state.record_topic("topic B")

        assert state.has_been_explained("topic A", threshold=2) is True
        assert state.has_been_explained("topic B", threshold=2) is False
        assert state.has_been_explained("topic C", threshold=1) is False

    def test_understanding_tracking(self):
        """Test tracking demonstrated understanding and confusion."""
        state = KnowledgeState()

        state.record_confusion("recursion")
        assert "recursion" in state.confusion_points

        state.record_understanding("recursion")
        assert "recursion" in state.demonstrated_understanding
        assert "recursion" not in state.confusion_points


class TestMentalStateTracker:
    """Tests for the neural MentalStateTracker module."""

    @pytest.fixture
    def tracker(self):
        """Create a tracker instance."""
        return MentalStateTracker(hidden_dim=64)

    def test_forward_pass(self, tracker):
        """Test forward pass produces expected outputs."""
        batch_size = 2
        seq_len = 10
        hidden_dim = 64

        hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
        attention_mask = torch.ones(batch_size, seq_len)

        outputs = tracker(hidden_states, attention_mask)

        assert "emotion_logits" in outputs
        assert "intent_logits" in outputs
        assert "belief_embedding" in outputs
        assert "frustration" in outputs

        assert outputs["emotion_logits"].shape == (batch_size, len(EmotionType))
        assert outputs["intent_logits"].shape == (batch_size, len(IntentCategory))
        assert outputs["frustration"].shape == (batch_size, 2)

    def test_forward_without_mask(self, tracker):
        """Test forward pass works without attention mask."""
        hidden_states = torch.randn(2, 10, 64)

        outputs = tracker(hidden_states)

        assert outputs["emotion_logits"].shape[0] == 2

    def test_update_from_predictions(self, tracker):
        """Test updating internal state from predictions."""
        hidden_states = torch.randn(1, 10, 64)
        predictions = tracker(hidden_states)

        tracker.update_from_predictions(predictions)

        # State should be updated
        assert tracker.emotional_state.primary_emotion is not None

    def test_perspective_prompt_generation(self, tracker):
        """Test generation of perspective-taking prompt."""
        # Set up some state
        tracker.emotional_state.update_emotion(EmotionType.FRUSTRATION, 0.8)
        tracker.intent_state.update_intent(IntentCategory.DEBUGGING, 0.9)

        prompt = tracker.get_perspective_prompt()

        assert "frustration" in prompt.lower()
        assert len(prompt) > 0

    def test_reset(self, tracker):
        """Test reset clears all state."""
        tracker.emotional_state.update_emotion(EmotionType.CURIOSITY, 0.7)
        tracker.reset()

        assert tracker.emotional_state.primary_emotion is None
        assert tracker.intent_state.surface_intent is None

    def test_trainable_parameters(self, tracker):
        """Test that parameters are trainable."""
        params = tracker.get_trainable_parameters()

        assert len(params) > 0
        assert all(p.requires_grad for p in params)


class TestIntegration:
    """Integration tests for Theory of Mind components."""

    def test_full_mental_state_update_cycle(self):
        """Test a full cycle of mental state tracking."""
        tracker = MentalStateTracker(hidden_dim=64)

        # Simulate a conversation with changing states
        for _ in range(5):
            hidden_states = torch.randn(1, 10, 64)
            predictions = tracker(hidden_states)
            tracker.update_from_predictions(predictions)

        # Should have history
        assert len(tracker.emotional_state.emotion_history) > 0
        assert len(tracker.intent_state.intent_trajectory) > 0

    def test_perspective_prompt_with_full_state(self):
        """Test perspective prompt with multiple state components."""
        tracker = MentalStateTracker(hidden_dim=64)

        # Set up rich state
        tracker.emotional_state.update_emotion(EmotionType.CONFUSION, 0.7)
        tracker.intent_state.update_intent(IntentCategory.LEARNING, 0.8)
        tracker.intent_state.urgency = 0.8
        tracker.knowledge_state.domain_expertise = 0.2
        tracker.belief_state.flag_misconception(
            "Python is compiled",
            "Python is interpreted"
        )

        prompt = tracker.get_perspective_prompt()

        # Should mention multiple aspects
        assert "confusion" in prompt.lower() or "learning" in prompt.lower()
        assert len(prompt) > 100  # Should be substantial


# Windows compatibility
if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    pytest.main([__file__, "-v"])
