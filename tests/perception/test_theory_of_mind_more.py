"""Additional behaviour tests for aspire.perception.theory_of_mind (coverage gaps)."""

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

# ---------------------------------------------------------------------------
# BeliefState
# ---------------------------------------------------------------------------


class TestBeliefStateMore:
    def test_confidence_is_clamped_and_routed_by_explicitness(self):
        beliefs = BeliefState()
        beliefs.add_belief("a", 1.7)
        beliefs.add_belief("b", -0.4, explicit=False)
        beliefs.add_belief("c")
        assert beliefs.explicit_beliefs == {"a": 1.0, "c": 0.7}
        assert beliefs.inferred_beliefs == {"b": 0.0}

    def test_inferred_beliefs_are_discounted(self):
        beliefs = BeliefState()
        beliefs.add_belief("inferred", 0.5, explicit=False)
        assert beliefs.get_belief_confidence("inferred") == pytest.approx(0.4)
        assert beliefs.get_belief_confidence("unknown") == 0.0

    def test_explicit_wins_over_inferred(self):
        beliefs = BeliefState()
        beliefs.add_belief("x", 0.9, explicit=False)
        beliefs.add_belief("x", 0.6, explicit=True)
        assert beliefs.get_belief_confidence("x") == 0.6

    def test_flag_misconception_removes_from_both_belief_stores(self):
        beliefs = BeliefState()
        beliefs.add_belief("earth is flat", 0.8)
        beliefs.add_belief("earth is flat", 0.5, explicit=False)
        beliefs.flag_misconception("earth is flat", "It is an oblate spheroid")
        assert beliefs.misconceptions == {"earth is flat": "It is an oblate spheroid"}
        assert beliefs.get_belief_confidence("earth is flat") == 0.0

    def test_flag_unknown_belief_is_safe(self):
        beliefs = BeliefState()
        beliefs.flag_misconception("never held", "fix")
        assert "never held" in beliefs.misconceptions

    def test_last_updated_advances(self):
        beliefs = BeliefState()
        beliefs.last_updated = 0.0
        beliefs.add_belief("a")
        assert beliefs.last_updated > 0
        beliefs.last_updated = 0.0
        beliefs.flag_misconception("a", "b")
        assert beliefs.last_updated > 0


# ---------------------------------------------------------------------------
# IntentState
# ---------------------------------------------------------------------------

I = IntentCategory  # noqa: E741


class TestIntentStateMore:
    def test_stability_drops_on_shift_and_recovers_on_repeat(self):
        state = IntentState()
        state.update_intent(I.LEARNING, 0.9)
        assert state.intent_stability == 1.0  # first update cannot be a shift
        state.update_intent(I.DEBUGGING, 0.8)
        assert state.intent_stability == pytest.approx(0.8)
        state.update_intent(I.DEBUGGING, 0.8)
        assert state.intent_stability == pytest.approx(0.9)
        state.update_intent(I.DEBUGGING, 0.8)
        state.update_intent(I.DEBUGGING, 0.8)
        assert state.intent_stability == 1.0  # capped

    def test_stability_floor_is_zero(self):
        state = IntentState()
        for i in range(12):
            state.update_intent(I.LEARNING if i % 2 else I.DEBUGGING, 0.5)
        assert state.intent_stability == 0.0

    def test_underlying_intent_only_set_when_given(self):
        state = IntentState()
        state.update_intent(I.LEARNING, 0.7, I.EXPLORING, 0.4)
        assert (state.underlying_intent, state.underlying_confidence) == (I.EXPLORING, 0.4)
        state.update_intent(I.LEARNING, 0.9)  # no underlying given: keep previous
        assert (state.underlying_intent, state.underlying_confidence) == (I.EXPLORING, 0.4)
        assert (state.surface_intent, state.surface_confidence) == (I.LEARNING, 0.9)

    def test_trajectory_is_bounded_to_twenty_latest(self):
        state = IntentState()
        for i in range(25):
            state.update_intent(I.LEARNING, i / 100)
        assert len(state.intent_trajectory) == 20
        assert state.intent_trajectory[0][1] == pytest.approx(0.05)
        assert state.intent_trajectory[-1][1] == pytest.approx(0.24)

    def test_frustration_needs_three_data_points(self):
        state = IntentState(urgency=1.0)
        state.update_intent(I.CLARIFYING, 0.5)
        state.update_intent(I.CLARIFYING, 0.5)
        assert state.detect_frustration_signals() == 0.0

    def test_frustration_signals_accumulate(self):
        state = IntentState()
        for intent in (I.LEARNING, I.CLARIFYING, I.DEBUGGING, I.CLARIFYING):
            state.update_intent(intent, 0.5)
        # unstable (0.3) + repeated clarification (0.3)
        assert state.intent_stability < 0.5
        assert state.detect_frustration_signals() == pytest.approx(0.6)
        state.urgency = 0.9
        assert state.detect_frustration_signals() == pytest.approx(0.8)

    def test_frustration_urgency_alone(self):
        state = IntentState(urgency=0.9)
        for _ in range(3):
            state.update_intent(I.BUILDING, 0.5)
        assert state.detect_frustration_signals() == pytest.approx(0.2)

    def test_frustration_is_capped_at_one(self):
        state = IntentState(urgency=1.0)
        for intent in (I.LEARNING, I.CLARIFYING, I.DEBUGGING, I.CLARIFYING, I.EXPLORING, I.CLARIFYING):
            state.update_intent(intent, 0.5)
        assert state.detect_frustration_signals() <= 1.0

    def test_calm_conversation_has_no_frustration(self):
        state = IntentState()
        for _ in range(5):
            state.update_intent(I.LEARNING, 0.9)
        assert state.detect_frustration_signals() == 0.0


# ---------------------------------------------------------------------------
# EmotionalState
# ---------------------------------------------------------------------------

E = EmotionType

POSITIVE = [E.CURIOSITY, E.SATISFACTION, E.ENTHUSIASM, E.RELIEF, E.GRATITUDE]
NEGATIVE = [E.FRUSTRATION, E.CONFUSION, E.IMPATIENCE, E.DISAPPOINTMENT, E.ANXIETY]
NEUTRAL = [E.FOCUSED, E.ANALYTICAL, E.EXPLORATORY]


class TestEmotionalStateMore:
    @pytest.mark.parametrize("emotion", POSITIVE)
    def test_positive_valence(self, emotion):
        state = EmotionalState()
        state.update_emotion(emotion, 0.5)
        assert state.valence is EmotionalValence.POSITIVE

    @pytest.mark.parametrize("emotion", NEGATIVE)
    def test_negative_valence(self, emotion):
        state = EmotionalState()
        state.update_emotion(emotion, 0.5)
        assert state.valence is EmotionalValence.NEGATIVE

    @pytest.mark.parametrize("emotion", NEUTRAL)
    def test_neutral_valence(self, emotion):
        state = EmotionalState(valence=EmotionalValence.POSITIVE)
        state.update_emotion(emotion, 0.5)
        assert state.valence is EmotionalValence.NEUTRAL

    def test_every_emotion_type_is_classified(self):
        assert set(POSITIVE) | set(NEGATIVE) | set(NEUTRAL) == set(EmotionType)

    def test_intensity_clamped_and_secondary_recorded(self):
        state = EmotionalState()
        state.update_emotion(E.FRUSTRATION, 3.0, secondary=E.ANXIETY, secondary_intensity=0.4)
        assert state.primary_intensity == 1.0
        assert (state.secondary_emotion, state.secondary_intensity) == (E.ANXIETY, 0.4)
        state.update_emotion(E.RELIEF, -2.0)
        assert state.primary_intensity == 0.0
        # no new secondary given: the previous one is retained
        assert state.secondary_emotion is E.ANXIETY

    def test_history_is_bounded_to_fifty(self):
        state = EmotionalState()
        for i in range(60):
            state.update_emotion(E.FOCUSED, i / 100)
        assert len(state.emotion_history) == 50
        assert state.emotion_history[0][1] == pytest.approx(0.10)

    def test_trend_needs_three_points(self):
        state = EmotionalState()
        state.update_emotion(E.CURIOSITY, 0.5)
        state.update_emotion(E.CURIOSITY, 0.5)
        assert state.get_emotional_trend() == "insufficient_data"

    def test_trend_improving_declining_stable(self):
        def trend(emotions):
            state = EmotionalState()
            for e in emotions:
                state.update_emotion(e, 0.5)
            return state.get_emotional_trend()

        assert trend([E.CURIOSITY, E.SATISFACTION, E.RELIEF]) == "improving"
        assert trend([E.FRUSTRATION, E.CONFUSION, E.IMPATIENCE]) == "declining"
        assert trend([E.CURIOSITY, E.CONFUSION, E.FOCUSED]) == "stable"
        # a margin of exactly one is not enough to call a trend
        assert trend([E.CURIOSITY, E.FOCUSED, E.FOCUSED]) == "stable"
        assert trend([E.CONFUSION, E.FOCUSED, E.FOCUSED]) == "stable"
        assert trend([E.CURIOSITY, E.CURIOSITY, E.FOCUSED]) == "improving"  # margin of two

    def test_trend_uses_only_the_last_five(self):
        state = EmotionalState()
        for e in [E.FRUSTRATION] * 10 + [E.CURIOSITY] * 4 + [E.FOCUSED]:
            state.update_emotion(e, 0.5)
        assert state.get_emotional_trend() == "improving"

    @pytest.mark.parametrize("emotion,expected", [(E.ANXIETY, "declining"), (E.GRATITUDE, "improving")])
    def test_trend_counts_every_classified_emotion(self, emotion, expected):
        state = EmotionalState()
        for _ in range(5):
            state.update_emotion(emotion, 0.8)
        assert state.get_emotional_trend() == expected

    def test_adjust_rapport_is_clamped(self):
        state = EmotionalState()
        state.adjust_rapport(0.8)
        assert state.rapport_level == 1.0
        state.adjust_rapport(-2.0)
        assert state.rapport_level == 0.0
        state.adjust_rapport(0.25)
        assert state.rapport_level == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# KnowledgeState
# ---------------------------------------------------------------------------


class TestKnowledgeStateMore:
    def test_topic_counts_and_explained_threshold(self):
        state = KnowledgeState()
        assert not state.has_been_explained("loops")
        state.record_topic("loops")
        assert state.has_been_explained("loops")
        assert not state.has_been_explained("loops", threshold=2)
        state.record_topic("loops")
        assert state.topics_covered == {"loops": 2}
        assert state.has_been_explained("loops", threshold=2)

    def test_understanding_clears_confusion(self):
        state = KnowledgeState()
        state.record_confusion("recursion")
        assert "recursion" in state.confusion_points
        state.record_understanding("recursion")
        assert "recursion" in state.demonstrated_understanding
        assert "recursion" not in state.confusion_points

    def test_understanding_of_unconfused_concept_is_safe(self):
        state = KnowledgeState()
        state.record_understanding("variables")
        assert state.demonstrated_understanding == {"variables"}

    def test_confusion_does_not_revoke_understanding(self):
        state = KnowledgeState()
        state.record_understanding("x")
        state.record_confusion("x")
        assert "x" in state.demonstrated_understanding and "x" in state.confusion_points


# ---------------------------------------------------------------------------
# MentalStateTracker
# ---------------------------------------------------------------------------


@pytest.fixture
def tracker():
    torch.manual_seed(0)
    return MentalStateTracker(hidden_dim=16).eval()


def _one_hot(index, size, scale=20.0):
    logits = torch.zeros(1, size)
    logits[0, index] = scale
    return logits


def _predictions(emotion=E.FOCUSED, intent=I.LEARNING, frustration=(0.1, 0.1)):
    return {
        "emotion_logits": _one_hot(list(EmotionType).index(emotion), len(EmotionType)),
        "intent_logits": _one_hot(list(IntentCategory).index(intent), len(IntentCategory)),
        "frustration": torch.tensor([list(frustration)]),
    }


class TestTrackerForward:
    def test_output_shapes_and_ranges(self, tracker):
        out = tracker(torch.randn(3, 5, 16))
        assert out["emotion_logits"].shape == (3, len(EmotionType))
        assert out["intent_logits"].shape == (3, len(IntentCategory))
        assert out["belief_embedding"].shape == (3, 8)
        assert out["frustration"].shape == (3, 2)
        assert torch.all((out["frustration"] > 0) & (out["frustration"] < 1))
        assert out["pooled_state"].shape == (3, 16)

    def test_unmasked_pooling_is_the_mean(self, tracker):
        hs = torch.randn(2, 5, 16)
        torch.testing.assert_close(tracker(hs)["pooled_state"], hs.mean(dim=1))

    def test_masked_pooling_ignores_padding(self, tracker):
        hs = torch.randn(2, 4, 16)
        mask = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
        pooled = tracker(hs, mask)["pooled_state"]
        torch.testing.assert_close(pooled[0], hs[0, :2].mean(dim=0))
        torch.testing.assert_close(pooled[1], hs[1].mean(dim=0))

    def test_gradients_reach_every_head(self):
        module = MentalStateTracker(hidden_dim=16)
        out = module(torch.randn(2, 4, 16))
        loss = (
            out["emotion_logits"].sum()
            + out["intent_logits"].sum()
            + out["belief_embedding"].sum()
            + out["frustration"].sum()
        )
        loss.backward()
        for head in (module.emotion_encoder, module.intent_encoder, module.belief_encoder, module.frustration_detector):
            assert head[0].weight.grad is not None

    def test_trainable_parameters_are_all_parameters(self, tracker):
        assert len(tracker.get_trainable_parameters()) == len(list(tracker.parameters()))
        assert all(p.requires_grad for p in tracker.get_trainable_parameters())


class TestTrackerStateUpdates:
    def test_update_sets_emotion_and_intent_from_argmax(self, tracker):
        tracker.update_from_predictions(_predictions(E.CURIOSITY, I.DEBUGGING))
        assert tracker.emotional_state.primary_emotion is E.CURIOSITY
        assert tracker.emotional_state.valence is EmotionalValence.POSITIVE
        assert tracker.emotional_state.primary_intensity == pytest.approx(1.0, abs=1e-3)
        assert tracker.intent_state.surface_intent is I.DEBUGGING
        assert tracker.intent_state.surface_confidence == pytest.approx(1.0, abs=1e-3)

    def test_intensity_reflects_softmax_confidence(self, tracker):
        preds = _predictions()
        preds["emotion_logits"] = torch.zeros(1, len(EmotionType))  # uniform
        tracker.update_from_predictions(preds)
        assert tracker.emotional_state.primary_intensity == pytest.approx(1 / len(EmotionType))

    def test_frustrated_prediction_lowers_rapport_in_proportion_to_intensity(self, tracker):
        tracker.update_from_predictions(_predictions(frustration=(0.9, 0.5)))
        assert tracker.emotional_state.rapport_level == pytest.approx(0.5 - 0.1 * 0.5)

    def test_unfrustrated_prediction_leaves_rapport_alone(self, tracker):
        tracker.update_from_predictions(_predictions(frustration=(0.5, 1.0)))  # 0.5 is not > 0.5
        assert tracker.emotional_state.rapport_level == 0.5

    def test_updates_accumulate_history(self, tracker):
        for _ in range(3):
            tracker.update_from_predictions(_predictions())
        assert len(tracker.emotional_state.emotion_history) == 3
        assert len(tracker.intent_state.intent_trajectory) == 3

    def test_reset_replaces_all_states(self, tracker):
        old = (tracker.belief_state, tracker.intent_state, tracker.emotional_state, tracker.knowledge_state)
        tracker.update_from_predictions(_predictions())
        tracker.knowledge_state.record_topic("x")
        tracker.reset()
        new = (tracker.belief_state, tracker.intent_state, tracker.emotional_state, tracker.knowledge_state)
        assert all(a is not b for a, b in zip(old, new))
        assert tracker.emotional_state.primary_emotion is None
        assert tracker.knowledge_state.topics_covered == {}


class TestPerspectivePrompt:
    def test_empty_when_nothing_is_known(self, tracker):
        assert tracker.get_perspective_prompt() == ""

    def test_strong_emotion_wording(self, tracker):
        tracker.emotional_state.update_emotion(E.FRUSTRATION, 0.9)
        prompt = tracker.get_perspective_prompt()
        assert prompt.startswith("PERSPECTIVE TAKING:\n")
        assert "strongly frustration" in prompt
        assert prompt.endswith("\n\n")

    def test_moderate_emotion_wording_and_weak_emotion_omitted(self, tracker):
        tracker.emotional_state.update_emotion(E.CONFUSION, 0.5)
        assert "The user seems confusion." in tracker.get_perspective_prompt()
        tracker.emotional_state.update_emotion(E.CONFUSION, 0.4)  # boundary: not > 0.4
        assert tracker.get_perspective_prompt() == ""

    def test_intent_line_humanises_the_category(self, tracker):
        tracker.intent_state.update_intent(I.SEEKING_INFORMATION, 0.8)
        prompt = tracker.get_perspective_prompt()
        assert "The user is seeking information." in prompt

    def test_frustration_warning_threshold(self, tracker):
        for intent in (I.LEARNING, I.CLARIFYING, I.DEBUGGING, I.CLARIFYING):
            tracker.intent_state.update_intent(intent, 0.5)
        assert "IMPORTANT: The user may be frustrated." in tracker.get_perspective_prompt()

    def test_expertise_guidance(self, tracker):
        tracker.knowledge_state.domain_expertise = 0.1
        assert "novice" in tracker.get_perspective_prompt()
        tracker.knowledge_state.domain_expertise = 0.9
        assert "experienced" in tracker.get_perspective_prompt()
        tracker.knowledge_state.domain_expertise = 0.5
        assert tracker.get_perspective_prompt() == ""

    def test_misconceptions_are_capped_at_two(self, tracker):
        for i in range(3):
            tracker.belief_state.flag_misconception(f"belief{i}", f"fix{i}")
        prompt = tracker.get_perspective_prompt()
        assert "belief0" in prompt and "fix0" in prompt
        assert "belief1" in prompt
        assert "belief2" not in prompt

    def test_all_sections_appear_as_bullets_in_order(self, tracker):
        tracker.emotional_state.update_emotion(E.ANXIETY, 0.8)
        tracker.intent_state.update_intent(I.VENTING, 0.9)
        tracker.knowledge_state.domain_expertise = 0.1
        tracker.belief_state.flag_misconception("b", "c")
        lines = [ln for ln in tracker.get_perspective_prompt().splitlines() if ln.startswith("• ")]
        assert len(lines) == 4
        assert "anxiety" in lines[0] and "venting" in lines[1]
        assert "novice" in lines[2] and "believes 'b'" in lines[3]
