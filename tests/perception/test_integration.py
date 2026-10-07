"""
Tests for the Perception Integration module.

Covers PerceptionConfig/PerceptionState, the unified PerceptionModule,
PerceptionLoss, and the trainer-integration wrapper. Everything runs on CPU
with tiny tensors (hidden_dim=32).
"""

import os

os.environ["XFORMERS_DISABLED"] = "1"

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch
import torch.nn.functional as F

from aspire.perception.controlled_chaos import ChaosInjection, ChaosSeverity, ChaosType
from aspire.perception.empathy_evaluation import PerceptionEvaluation
from aspire.perception.integration import (
    PerceptionConfig,
    PerceptionLoss,
    PerceptionModule,
    PerceptionState,
    integrate_perception_with_trainer,
)
from aspire.perception.theory_of_mind import EmotionType, IntentCategory

H = 32


def _config(**overrides):
    base = {"tom_hidden_dim": H, "chaos_probability": 1.0, "chaos_curriculum": False}
    base.update(overrides)
    return PerceptionConfig(**base)


def _module(**overrides):
    torch.manual_seed(0)
    return PerceptionModule(_config(**overrides), hidden_dim=H).eval()


def _disabled(**overrides):
    flags = {
        "tom_enabled": False,
        "chaos_enabled": False,
        "character_enabled": False,
        "metacognition_enabled": False,
        "perception_eval_enabled": False,
    }
    flags.update(overrides)
    return _module(**flags)


def _hidden(batch=2, seq=5):
    torch.manual_seed(1)
    return torch.randn(batch, seq, H)


def _injection(chaos_type=ChaosType.AMBIGUOUS_REFERENCE):
    return ChaosInjection(
        chaos_type=chaos_type,
        severity=ChaosSeverity.MODERATE,
        original_input="fix it",
        modified_input="fix it, you know, the thing",
        ground_truth="ask which thing",
        learning_objective="ambiguity",
    )


class TestPerceptionConfigAndState:
    """Tests for the plain dataclasses."""

    def test_config_defaults(self):
        config = PerceptionConfig()

        assert config.tom_enabled and config.chaos_enabled and config.character_enabled
        assert config.metacognition_enabled and config.perception_eval_enabled
        assert config.tom_hidden_dim == 768
        assert config.chaos_probability == 0.3
        assert config.chaos_curriculum is True
        assert config.character_storage_dir is None
        assert config.perception_eval_strictness == 0.5
        assert config.perception_loss_weight == 0.3
        assert config.update_perception_every_n_steps == 1

    def test_state_defaults(self):
        state = PerceptionState()

        assert state.user_frustration == 0.0
        assert state.user_expertise == 0.5
        assert state.user_intent == "" and state.user_emotion == ""
        assert state.should_hedge is False and state.should_clarify is False
        assert state.confidence_level == 0.5
        assert state.character_prompt == ""
        assert state.active_values == []
        assert state.chaos_applied is False
        assert state.chaos_type is None
        assert state.expected_handling == ""

    def test_state_lists_are_independent(self):
        a, b = PerceptionState(), PerceptionState()
        a.active_values.append("x")

        assert b.active_values == []


class TestConstruction:
    """Tests for PerceptionModule.__init__."""

    def test_all_components_enabled(self):
        module = _module()

        assert module.mental_state_tracker is not None
        assert module.chaos_generator is not None
        assert module.character is not None
        assert module.metacognition is not None
        assert module.evaluator is not None
        assert module.hidden_dim == H

    def test_head_input_dim_depends_on_enabled_components(self):
        assert _module().perception_head[0].in_features == H + H // 2 + 4
        assert _disabled().perception_head[0].in_features == H
        assert _disabled(tom_enabled=True).perception_head[0].in_features == H + H // 2
        assert _disabled(metacognition_enabled=True).perception_head[0].in_features == H + 4

    def test_head_outputs_six_scores(self):
        head = _module().perception_head

        assert head[-1].out_features == 6
        assert head[0].out_features == H // 2

    def test_disabled_components_are_none(self):
        module = _disabled()

        assert module.mental_state_tracker is None
        assert module.chaos_generator is None
        assert module.character is None
        assert module.metacognition is None
        assert module.evaluator is None

    def test_config_values_are_forwarded(self):
        module = _module(
            chaos_probability=0.7,
            chaos_curriculum=True,
            perception_eval_strictness=0.9,
        )

        assert module.chaos_generator.config.chaos_probability == 0.7
        assert module.chaos_generator.config.curriculum_enabled is True
        assert module.evaluator.strictness == 0.9

    def test_default_config_is_created_when_none(self):
        module = PerceptionModule()

        assert module.config == PerceptionConfig()
        assert module.hidden_dim == 768
        assert module.perception_head[0].in_features == 768 + 384 + 4

    def test_character_storage_dir_is_forwarded(self, tmp_path):
        module = _module(character_storage_dir=tmp_path)

        assert module.character.storage_dir == tmp_path

    def test_initial_state(self):
        module = _module()

        assert module._current_state == PerceptionState()
        assert module._chaos_injection is None


class TestForward:
    """Tests for PerceptionModule.forward."""

    def test_output_keys_and_shapes(self):
        module = _module()
        out = module(_hidden(2, 5))

        assert out["perception_scores"].shape == (2, 6)
        assert out["pooled_features"].shape == (2, H)
        assert out["tom_emotion_logits"].shape == (2, len(EmotionType))
        assert out["tom_intent_logits"].shape == (2, len(IntentCategory))
        assert out["tom_belief_embedding"].shape == (2, H // 2)
        for key in ("should_hedge", "should_clarify", "should_ask_user", "meta_confidence"):
            assert out[f"meta_{key}"].shape == (2, 1)
        assert out["meta_overall_uncertainty"].shape == (2, 1)

    def test_scores_are_probabilities(self):
        scores = _module()(_hidden(3, 4))["perception_scores"]

        assert torch.all(scores > 0) and torch.all(scores < 1)

    def test_minimal_module_has_no_component_outputs(self):
        out = _disabled()(_hidden())

        assert set(out) == {"perception_scores", "pooled_features"}

    def test_pooling_without_mask_is_plain_mean(self):
        hidden = _hidden(2, 5)

        pooled = _disabled()(hidden)["pooled_features"]

        assert torch.allclose(pooled, hidden.mean(dim=1))

    def test_masked_pooling_ignores_padding(self):
        hidden = _hidden(2, 5)
        mask = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 1, 1]])

        pooled = _disabled()(hidden, attention_mask=mask)["pooled_features"]

        assert torch.allclose(pooled[0], hidden[0, :2].mean(dim=0), atol=1e-6)
        assert torch.allclose(pooled[1], hidden[1].mean(dim=0), atol=1e-6)

    def test_mask_changes_scores(self):
        module = _disabled()
        hidden = _hidden(1, 6)
        full = module(hidden)["perception_scores"]
        masked = module(hidden, attention_mask=torch.tensor([[1, 0, 0, 0, 0, 0]]))["perception_scores"]

        assert not torch.allclose(full, masked)

    def test_full_mask_matches_no_mask(self):
        module = _module()
        hidden = _hidden(2, 4)

        a = module(hidden)["perception_scores"]
        b = module(hidden, attention_mask=torch.ones(2, 4))["perception_scores"]

        assert torch.allclose(a, b, atol=1e-6)

    def test_output_logits_path(self):
        module = _module()
        logits = torch.randn(2, 5, 11)

        out = module(_hidden(2, 5), output_logits=logits)
        without = module(_hidden(2, 5))

        assert out["perception_scores"].shape == (2, 6)
        assert not torch.allclose(out["perception_scores"], without["perception_scores"])

    def test_output_logits_two_dimensional(self):
        module = _module()

        out = module(_hidden(2, 5), output_logits=torch.randn(2, 11))

        assert out["perception_scores"].shape == (2, 6)

    def test_gradients_reach_perception_head(self):
        module = _module()
        module.train()
        out = module(_hidden(2, 5))
        out["perception_scores"].sum().backward()

        assert module.perception_head[0].weight.grad is not None
        assert module.mental_state_tracker.belief_encoder[0].weight.grad is not None


class TestPrepareInput:
    """Tests for PerceptionModule.prepare_input."""

    def test_without_chaos_returns_prompt_unchanged(self):
        module = _module()

        prompt, state = module.prepare_input("explain recursion", apply_chaos=False)

        assert prompt == "explain recursion"
        assert state.chaos_applied is False
        assert state.chaos_type is None
        assert module._current_state is state

    def test_character_context_is_populated(self):
        module = _module()

        _, state = module.prepare_input("explain recursion", apply_chaos=False)

        assert state.character_prompt.startswith("CHARACTER: Agent")
        assert "CORE VALUES" in state.character_prompt
        # default character has three context-free values, sorted by priority
        assert state.active_values == ["truth_seeking", "intellectual_honesty", "helpfulness"]

    def test_chaos_injection_applied(self, monkeypatch):
        module = _module()
        injection = _injection()
        monkeypatch.setattr(module.chaos_generator, "inject", lambda text: injection)

        prompt, state = module.prepare_input("fix it")

        assert prompt == "fix it, you know, the thing"
        assert state.chaos_applied is True
        assert state.chaos_type == "ambiguous_reference"
        assert state.expected_handling == "ask which thing"
        assert module._chaos_injection is injection

    def test_chaos_skipped_when_generator_declines(self, monkeypatch):
        module = _module()
        monkeypatch.setattr(module.chaos_generator, "inject", lambda text: None)

        prompt, state = module.prepare_input("fix it")

        assert prompt == "fix it"
        assert state.chaos_applied is False

    def test_real_generator_with_probability_one(self):
        module = _module()
        # some ChaosTypes have no generator and silently yield no injection; pin one that does
        module.chaos_generator.config.enabled_types = [ChaosType.MISSING_CONTEXT]

        prompt, state = module.prepare_input("Please summarise the report for the team.")

        assert state.chaos_applied is True
        noise_types = {"missing_context", "partial_information", "noisy_input", "truncated_input"}
        assert state.chaos_type in noise_types
        assert module._chaos_injection.modified_input == prompt
        assert module._chaos_injection.original_input == "Please summarise the report for the team."

    def test_disabled_chaos_never_applies(self):
        module = _module(chaos_enabled=False)

        prompt, state = module.prepare_input("fix it")

        assert prompt == "fix it"
        assert state.chaos_applied is False

    def test_disabled_character_leaves_context_empty(self):
        module = _module(character_enabled=False)

        _, state = module.prepare_input("hi", apply_chaos=False)

        assert state.character_prompt == ""
        assert state.active_values == []

    def test_curriculum_blocks_chaos_until_ramped(self):
        module = _module(chaos_curriculum=True)
        module.chaos_generator.config.enabled_types = [ChaosType.MISSING_CONTEXT]

        _, early = module.prepare_input("hello there")
        module.set_training_epoch(4)  # past start epoch + ramp -> full probability
        _, late = module.prepare_input("hello there")

        assert early.chaos_applied is False
        assert late.chaos_applied is True
        assert module.chaos_generator.current_epoch == 4

    def test_set_training_epoch_without_generator_is_noop(self):
        module = _module(chaos_enabled=False)

        module.set_training_epoch(3)  # must not raise

    def test_chaos_injection_does_not_leak_into_next_prompt(self, monkeypatch):
        module = _module()
        monkeypatch.setattr(module.chaos_generator, "inject", lambda text: _injection())
        module.prepare_input("fix it")

        module.prepare_input("a clear request", apply_chaos=False)

        assert module._chaos_injection is None


class TestUpdateMentalState:
    """Tests for PerceptionModule.update_mental_state."""

    def test_state_mirrors_tracker(self):
        module = _module()
        module.prepare_input("hi", apply_chaos=False)

        module.update_mental_state(_hidden(1, 5))

        tracker = module.mental_state_tracker
        state = module._current_state
        assert tracker.intent_state.surface_intent is not None
        assert state.user_intent == tracker.intent_state.surface_intent.value
        assert state.user_emotion == tracker.emotional_state.primary_emotion.value
        assert state.user_intent in {i.value for i in IntentCategory}
        assert state.user_emotion in {e.value for e in EmotionType}
        assert state.user_expertise == tracker.knowledge_state.domain_expertise == 0.5
        assert state.user_frustration == 0.0  # fewer than three turns observed

    def test_frustration_signal_is_copied_after_enough_turns(self):
        module = _module()
        for _ in range(3):
            module.update_mental_state(_hidden(1, 5))
        tracker = module.mental_state_tracker
        tracker.intent_state.intent_stability = 0.2  # rapid intent shifts
        tracker.intent_state.intent_trajectory = [(IntentCategory.CLARIFYING, 0.9)] * 3

        module.update_mental_state(_hidden(1, 5))

        assert module._current_state.user_frustration == pytest.approx(
            tracker.intent_state.detect_frustration_signals()
        )

    def test_attention_mask_is_accepted(self):
        module = _module()

        module.update_mental_state(_hidden(1, 4), attention_mask=torch.ones(1, 4))

        assert len(module.mental_state_tracker.intent_state.intent_trajectory) == 1

    def test_no_tracker_is_noop(self):
        module = _module(tom_enabled=False)

        module.update_mental_state(_hidden(1, 4))

        assert module._current_state == PerceptionState()

    def test_state_untouched_when_tracker_reports_nothing(self, monkeypatch):
        module = _module()
        monkeypatch.setattr(module.mental_state_tracker, "update_from_predictions", lambda p: None)

        module.update_mental_state(_hidden(1, 4))

        assert module._current_state.user_intent == ""
        assert module._current_state.user_emotion == ""

    def test_runs_without_building_graph(self):
        module = _module()
        module.train()
        calls = []
        original = module.mental_state_tracker.update_from_predictions

        def spy(predictions):
            calls.append(predictions)
            return original(predictions)

        module.mental_state_tracker.update_from_predictions = spy
        module.update_mental_state(_hidden(1, 4))

        assert all(not v.requires_grad for v in calls[0].values())


class TestPerceptionPrompt:
    """Tests for PerceptionModule.get_perception_prompt."""

    def test_empty_when_nothing_enabled(self):
        assert _disabled().get_perception_prompt() == ""

    def test_empty_for_fresh_tracker_without_character_context(self):
        module = _module(character_enabled=False)

        assert module.get_perception_prompt() == ""

    def test_includes_character_prompt(self):
        module = _module()
        module.prepare_input("hello", apply_chaos=False)

        prompt = module.get_perception_prompt()

        assert prompt == module._current_state.character_prompt

    def test_includes_perspective_prompt_after_update(self):
        module = _module(character_enabled=False)
        module.update_mental_state(_hidden(1, 4))

        prompt = module.get_perception_prompt()

        assert prompt.startswith("PERSPECTIVE TAKING:")
        assert module.mental_state_tracker.get_perspective_prompt().strip() in prompt

    def test_includes_chaos_hint(self, monkeypatch):
        module = _module(character_enabled=False)
        monkeypatch.setattr(module.chaos_generator, "inject", lambda text: _injection())
        module.prepare_input("fix it")

        prompt = module.get_perception_prompt()

        assert prompt == "NOTE: Input may contain ambiguous_reference. Expected handling: ask which thing"

    def test_parts_are_newline_joined_in_order(self, monkeypatch):
        module = _module()
        monkeypatch.setattr(module.chaos_generator, "inject", lambda text: _injection())
        module.prepare_input("fix it")

        prompt = module.get_perception_prompt()

        character, note = module._current_state.character_prompt, prompt.split("\n")[-1]
        assert prompt == f"{character}\n{note}"
        assert note.startswith("NOTE: Input may contain")


class TestEvaluateResponse:
    """Tests for PerceptionModule.evaluate_response."""

    def test_none_when_evaluator_disabled(self):
        assert _module(perception_eval_enabled=False).evaluate_response("p", "r") is None

    def test_returns_evaluation(self):
        module = _module()

        evaluation = module.evaluate_response("prompt", "I think this works.")

        assert isinstance(evaluation, PerceptionEvaluation)
        assert len(evaluation.perception_scores) == 15

    def test_context_is_assembled_from_module_state(self, monkeypatch):
        module = _module()
        injection = _injection()
        monkeypatch.setattr(module.chaos_generator, "inject", lambda text: injection)
        module.prepare_input("fix it")
        spy = MagicMock(return_value="sentinel")
        monkeypatch.setattr(module.evaluator, "evaluate", spy)

        result = module.evaluate_response("prompt", "response")

        assert result == "sentinel"
        prompt, response, context = spy.call_args.args
        assert (prompt, response) == ("prompt", "response")
        assert context["mental_state"] is module.mental_state_tracker
        assert context["chaos_injection"] is injection
        assert context["character"] is module.character
        assert context["perception_state"] is module._current_state

    def test_chaos_context_drives_heuristics(self, monkeypatch):
        module = _module()
        monkeypatch.setattr(module.chaos_generator, "inject", lambda text: _injection())
        module.prepare_input("fix it")

        evaluation = module.evaluate_response("fix it", "Done, here is the fix.")

        scores = {s.dimension.value: s.score for s in evaluation.perception_scores}
        assert scores["ambiguity_handling"] == 3.0
        assert scores["uncertainty_calibration"] == 3.0


class TestRecordExperience:
    """Tests for PerceptionModule.record_experience."""

    def test_successful_experience(self):
        module = _module()
        module._current_state.user_intent = "debugging"

        module.record_experience("the prompt", "the response", "it worked", was_successful=True)

        (memory,) = module.character.memory.memories
        assert memory.situation == "the prompt"
        assert memory.action_taken == "the response"
        assert memory.outcome == "it worked"
        assert memory.lesson == "This approach worked well for debugging"
        assert memory.emotional_valence == 0.5
        assert memory.context_tags == ["intent_debugging"]

    def test_failed_experience(self):
        module = _module()
        module._current_state.user_intent = "learning"

        module.record_experience("p", "r", "it failed", was_successful=False)

        (memory,) = module.character.memory.memories
        assert memory.lesson == "Consider alternative approach for learning"
        assert memory.emotional_valence == -0.5

    def test_chaos_tag_added(self):
        module = _module()
        module._current_state.chaos_applied = True
        module._current_state.chaos_type = "noisy_input"

        module.record_experience("p", "r", "o", True)

        (memory,) = module.character.memory.memories
        assert memory.context_tags == ["chaos_noisy_input"]
        assert module.character.memory.memory_index == {"chaos_noisy_input": [0]}

    def test_both_tags_in_order(self):
        module = _module()
        module._current_state.chaos_applied = True
        module._current_state.chaos_type = "noisy_input"
        module._current_state.user_intent = "building"

        module.record_experience("p", "r", "o", True)

        assert module.character.memory.memories[0].context_tags == ["chaos_noisy_input", "intent_building"]

    def test_long_text_is_truncated_to_200_chars(self):
        module = _module()

        module.record_experience("p" * 500, "r" * 500, "o", True)

        memory = module.character.memory.memories[0]
        assert memory.situation == "p" * 200
        assert memory.action_taken == "r" * 200

    def test_no_character_is_noop(self):
        module = _module(character_enabled=False)

        module.record_experience("p", "r", "o", True)  # must not raise

        assert module.character is None

    def test_persisted_when_storage_dir_configured(self, tmp_path):
        module = _module(character_storage_dir=tmp_path)

        module.record_experience("the prompt", "the response", "ok", True)

        data = json.loads((tmp_path / "memories.json").read_text())
        assert data["memories"][0]["situation"] == "the prompt"


class TestTrainableParametersAndState:
    """Tests for parameter collection, (de)serialisation and reset."""

    def test_trainable_parameters_cover_all_parameters_once(self):
        module = _module()

        params = module.get_trainable_parameters()

        assert {id(p) for p in params} == {id(p) for p in module.parameters()}
        assert len(params) == len({id(p) for p in params})

    def test_trainable_parameters_minimal_module_is_head_only(self):
        module = _disabled()

        params = module.get_trainable_parameters()

        assert [id(p) for p in params] == [id(p) for p in module.perception_head.parameters()]

    def test_state_dict_perception_contents(self):
        module = _module()

        state = module.state_dict_perception()

        assert set(state) == {
            "perception_head",
            "config",
            "mental_state_tracker",
            "metacognition",
            "character_identity",
        }
        assert state["config"] is module.config
        assert state["character_identity"] == module.character.get_identity_hash()
        assert set(state["perception_head"]) == set(module.perception_head.state_dict())

    def test_state_dict_perception_minimal(self):
        state = _disabled().state_dict_perception()

        assert set(state) == {"perception_head", "config"}

    def test_load_roundtrip_restores_weights(self):
        source = _module()
        torch.manual_seed(99)
        target = PerceptionModule(_config(), hidden_dim=H).eval()
        hidden = _hidden(2, 4)
        assert not torch.allclose(source(hidden)["perception_scores"], target(hidden)["perception_scores"])

        target.load_state_dict_perception(source.state_dict_perception())

        assert torch.allclose(source(hidden)["perception_scores"], target(hidden)["perception_scores"])

    def test_load_head_only_state_leaves_other_components(self):
        source = _module()
        torch.manual_seed(7)
        target = PerceptionModule(_config(), hidden_dim=H).eval()
        tracker_before = target.mental_state_tracker.emotion_encoder[0].weight.clone()
        state = source.state_dict_perception()
        del state["mental_state_tracker"], state["metacognition"]

        target.load_state_dict_perception(state)

        assert torch.equal(target.perception_head[0].weight, source.perception_head[0].weight)
        assert torch.equal(target.mental_state_tracker.emotion_encoder[0].weight, tracker_before)
        assert not torch.equal(tracker_before, source.mental_state_tracker.emotion_encoder[0].weight)

    def test_load_ignores_component_state_when_component_disabled(self):
        full = _module()
        minimal = _disabled()
        state = minimal.state_dict_perception()
        state["mental_state_tracker"] = full.mental_state_tracker.state_dict()
        state["metacognition"] = full.metacognition.state_dict()

        minimal.load_state_dict_perception(state)  # must not raise or build components

        assert minimal.mental_state_tracker is None

    def test_load_missing_head_raises_keyerror(self):
        with pytest.raises(KeyError):
            _module().load_state_dict_perception({})

    def test_reset_conversation_state(self, monkeypatch):
        module = _module()
        monkeypatch.setattr(module.chaos_generator, "inject", lambda text: _injection())
        module.prepare_input("fix it")
        module.update_mental_state(_hidden(1, 4))
        assert module.mental_state_tracker.intent_state.surface_intent is not None

        module.reset_conversation_state()

        assert module._current_state == PerceptionState()
        assert module._chaos_injection is None
        assert module.mental_state_tracker.intent_state.surface_intent is None
        assert module.mental_state_tracker.emotional_state.primary_emotion is None

    def test_reset_without_tracker(self):
        module = _module(tom_enabled=False)
        module._current_state.user_intent = "x"

        module.reset_conversation_state()

        assert module._current_state.user_intent == ""


class TestPerceptionLoss:
    """Tests for PerceptionLoss."""

    @pytest.fixture
    def outputs(self):
        torch.manual_seed(3)
        return {
            "tom_emotion_logits": torch.randn(4, len(EmotionType), requires_grad=True),
            "tom_intent_logits": torch.randn(4, len(IntentCategory), requires_grad=True),
            "meta_overall_uncertainty": torch.rand(4, 1, requires_grad=True),
            "perception_scores": torch.rand(4, 6, requires_grad=True),
        }

    @pytest.fixture
    def targets(self):
        torch.manual_seed(4)
        return {
            "emotion_target": torch.tensor([0, 3, 5, 7]),
            "intent_target": torch.tensor([1, 2, 3, 4]),
            "uncertainty_target": torch.rand(4, 1),
            "chaos_handling_target": torch.rand(4, 6),
        }

    def test_default_weights(self):
        loss = PerceptionLoss()

        assert (loss.tom_weight, loss.metacognition_weight) == (0.3, 0.3)
        assert (loss.character_weight, loss.chaos_weight) == (0.2, 0.2)

    def test_total_matches_manual_computation(self, outputs, targets):
        losses = PerceptionLoss(tom_weight=0.5, metacognition_weight=0.25, chaos_weight=2.0)(
            outputs, targets
        )

        emotion = F.cross_entropy(outputs["tom_emotion_logits"], targets["emotion_target"])
        intent = F.cross_entropy(outputs["tom_intent_logits"], targets["intent_target"])
        uncertainty = F.mse_loss(outputs["meta_overall_uncertainty"], targets["uncertainty_target"])
        chaos = F.mse_loss(outputs["perception_scores"], targets["chaos_handling_target"])
        assert set(losses) == {"tom_emotion", "tom_intent", "uncertainty", "chaos_handling", "total"}
        assert losses["tom_emotion"].item() == pytest.approx(emotion.item())
        assert losses["tom_intent"].item() == pytest.approx(intent.item())
        assert losses["uncertainty"].item() == pytest.approx(uncertainty.item())
        assert losses["chaos_handling"].item() == pytest.approx(chaos.item())
        expected = 0.5 * emotion + 0.5 * intent + 0.25 * uncertainty + 2.0 * chaos
        assert losses["total"].item() == pytest.approx(expected.item())

    def test_only_present_targets_contribute(self, outputs, targets):
        losses = PerceptionLoss()(outputs, {"emotion_target": targets["emotion_target"]})

        assert set(losses) == {"tom_emotion", "total"}
        assert losses["total"].item() == pytest.approx(0.3 * losses["tom_emotion"].item())

    def test_target_without_matching_output_is_ignored(self, targets):
        outputs = {"perception_scores": torch.rand(4, 6)}

        losses = PerceptionLoss()(outputs, targets)

        assert set(losses) == {"chaos_handling", "total"}

    def test_no_matching_pairs_gives_zero_total(self):
        losses = PerceptionLoss()({"perception_scores": torch.rand(2, 6)}, {})

        assert set(losses) == {"total"}
        assert losses["total"].item() == 0.0

    def test_perfect_prediction_has_zero_regression_losses(self):
        scores = torch.rand(3, 6)
        losses = PerceptionLoss()(
            {"perception_scores": scores, "meta_overall_uncertainty": torch.rand(3, 1)},
            {"chaos_handling_target": scores.clone()},
        )

        assert losses["chaos_handling"].item() == pytest.approx(0.0)
        assert losses["total"].item() == pytest.approx(0.0)

    def test_total_is_differentiable(self, outputs, targets):
        losses = PerceptionLoss()(outputs, targets)
        losses["total"].backward()

        for tensor in outputs.values():
            assert tensor.grad is not None and torch.isfinite(tensor.grad).all()

    def test_end_to_end_with_module_outputs(self):
        module = _module()
        module.train()
        out = module(_hidden(2, 5))
        targets = {
            "emotion_target": torch.tensor([1, 2]),
            "intent_target": torch.tensor([0, 4]),
            "uncertainty_target": torch.tensor([[0.2], [0.8]]),
            "chaos_handling_target": torch.full((2, 6), 0.5),
        }

        losses = PerceptionLoss()(out, targets)
        losses["total"].backward()

        assert losses["total"].item() > 0
        assert module.perception_head[0].weight.grad is not None
        assert module.metacognition.meta_head[0].weight.grad is not None


class _FakeTrainer:
    """Minimal AspireTrainer stand-in for the integration wrapper."""

    def __init__(self, result=None):
        self.param = torch.nn.Parameter(torch.zeros(1))
        self.optimizer = torch.optim.SGD([self.param], lr=0.1)
        self.result = result if result is not None else {}
        self._train_step = AsyncMock(return_value=self.result)
        self.original_step = self._train_step


class TestIntegratePerceptionWithTrainer:
    """Tests for integrate_perception_with_trainer."""

    def test_attaches_module_and_extends_optimizer(self, capsys):
        module = _module()
        trainer = _FakeTrainer()

        integrate_perception_with_trainer(trainer, module)

        assert trainer.perception_module is module
        group_params = trainer.optimizer.param_groups[0]["params"]
        perception_params = module.get_trainable_parameters()
        assert len(group_params) == 1 + len(perception_params)
        assert group_params[0] is trainer.param
        assert all(any(p is g for g in group_params) for p in perception_params)

    def test_prints_summary(self, capsys):
        module = _module(chaos_enabled=False)

        integrate_perception_with_trainer(_FakeTrainer(), module)

        out = capsys.readouterr().out.splitlines()
        assert out == [
            "Perception module integrated with trainer",
            "  - ToM enabled: True",
            "  - Chaos enabled: False",
            "  - Character enabled: True",
            "  - Metacognition enabled: True",
        ]

    def test_optimizer_can_update_perception_parameters(self):
        module = _module()
        module.train()
        trainer = _FakeTrainer()
        integrate_perception_with_trainer(trainer, module)
        head_weight = module.perception_head[0].weight
        before = head_weight.detach().clone()

        module(_hidden(2, 5))["perception_scores"].sum().backward()
        trainer.optimizer.step()

        assert not torch.equal(head_weight.detach(), before)

    def test_no_params_skips_optimizer_extension(self, monkeypatch):
        module = _module()
        monkeypatch.setattr(module, "get_trainable_parameters", lambda: [])
        trainer = _FakeTrainer()

        integrate_perception_with_trainer(trainer, module)

        assert len(trainer.optimizer.param_groups[0]["params"]) == 1

    async def test_wrapped_step_prepares_prompt_and_passes_state(self, monkeypatch):
        module = _module()
        monkeypatch.setattr(module.chaos_generator, "inject", lambda text: _injection())
        trainer = _FakeTrainer(result={"loss": 1.0})
        integrate_perception_with_trainer(trainer, module)

        result = await trainer._train_step(prompt="fix it", step=3)

        assert result == {"loss": 1.0}
        trainer.original_step.assert_awaited_once()
        kwargs = trainer.original_step.call_args.kwargs
        assert kwargs["prompt"] == "fix it, you know, the thing"
        assert kwargs["step"] == 3
        assert isinstance(kwargs["perception_state"], PerceptionState)
        assert kwargs["perception_state"].chaos_type == "ambiguous_reference"

    async def test_wrapped_step_without_prompt_kwarg_passes_through(self):
        module = _module()
        trainer = _FakeTrainer(result={"loss": 2.0})
        integrate_perception_with_trainer(trainer, module)

        result = await trainer._train_step("positional", flag=True)

        assert result == {"loss": 2.0}
        trainer.original_step.assert_awaited_once_with("positional", flag=True)
        assert module._current_state == PerceptionState()

    async def test_wrapped_step_updates_mental_state_from_hidden_states(self):
        module = _module()
        hidden = _hidden(1, 4)
        mask = torch.ones(1, 4)
        trainer = _FakeTrainer(result={"hidden_states": hidden, "attention_mask": mask})
        integrate_perception_with_trainer(trainer, module)
        module.update_mental_state = MagicMock(wraps=module.update_mental_state)
        module.forward = MagicMock(wraps=module.forward)

        result = await trainer._train_step(prompt="hello")

        assert result["hidden_states"] is hidden
        module.update_mental_state.assert_called_once_with(hidden, mask)
        module.forward.assert_called_once_with(hidden, mask)
        assert module._current_state.user_intent != ""

    async def test_wrapped_step_without_hidden_states_skips_perception(self):
        module = _module()
        module.update_mental_state = MagicMock()
        trainer = _FakeTrainer(result={"loss": 0.5})
        integrate_perception_with_trainer(trainer, module)

        await trainer._train_step(prompt="hello")

        module.update_mental_state.assert_not_called()

    async def test_hidden_states_without_mask(self):
        module = _module()
        trainer = _FakeTrainer(result={"hidden_states": _hidden(1, 3)})
        integrate_perception_with_trainer(trainer, module)
        module.update_mental_state = MagicMock()

        await trainer._train_step(prompt="hello")

        args = module.update_mental_state.call_args.args
        assert args[1] is None
