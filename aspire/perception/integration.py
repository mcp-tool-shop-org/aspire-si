"""
Perception Integration Module - Connecting perception to ASPIRE training.

This module provides the glue between the perception components and
ASPIRE's training infrastructure. It:

1. Wraps perception modules for easy integration with AspireTrainer
2. Provides perception-aware loss functions
3. Enables curriculum-based perception training
4. Tracks perception metrics for monitoring

Usage:
    from aspire.perception import PerceptionModule
    from aspire import AspireTrainer, AspireConfig

    perception = PerceptionModule(config)
    trainer = AspireTrainer(config, perception_module=perception)

    # Perception is now integrated into training
    results = await trainer.train(teacher, prompts)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from aspire.perception.character import CharacterCore
from aspire.perception.controlled_chaos import ChaosConfig, ChaosGenerator, ChaosInjection
from aspire.perception.empathy_evaluation import PerceptionEvaluation, PerceptionEvaluator
from aspire.perception.metacognition import MetaCognitionModule
from aspire.perception.theory_of_mind import MentalStateTracker


@dataclass
class PerceptionConfig:
    """Configuration for the perception module."""

    # Theory of Mind
    tom_enabled: bool = True
    tom_hidden_dim: int = 768

    # Controlled Chaos
    chaos_enabled: bool = True
    chaos_probability: float = 0.3
    chaos_curriculum: bool = True

    # Character
    character_enabled: bool = True
    character_storage_dir: Path | None = None

    # Meta-Cognition
    metacognition_enabled: bool = True

    # Perception Evaluation
    perception_eval_enabled: bool = True
    perception_eval_strictness: float = 0.5

    # Training integration
    perception_loss_weight: float = 0.3
    update_perception_every_n_steps: int = 1


@dataclass
class PerceptionState:
    """Current state of perception modules for a conversation."""

    # Mental model of user
    user_frustration: float = 0.0
    user_expertise: float = 0.5
    user_intent: str = ""
    user_emotion: str = ""

    # Response generation context
    should_hedge: bool = False
    should_clarify: bool = False
    confidence_level: float = 0.5

    # Character state
    character_prompt: str = ""
    active_values: list[str] = field(default_factory=list)

    # Chaos state (if applied)
    chaos_applied: bool = False
    chaos_type: str | None = None
    expected_handling: str = ""


class PerceptionModule(nn.Module):
    """
    Unified perception module for ASPIRE integration.

    Combines all perception components into a single trainable module
    that can be integrated with the AspireTrainer.
    """

    def __init__(
        self,
        config: PerceptionConfig | None = None,
        hidden_dim: int = 768,
    ):
        super().__init__()

        self.config = config or PerceptionConfig()
        self.hidden_dim = hidden_dim

        # Initialize components based on config
        if self.config.tom_enabled:
            self.mental_state_tracker = MentalStateTracker(hidden_dim=self.config.tom_hidden_dim)
        else:
            self.mental_state_tracker = None

        if self.config.chaos_enabled:
            chaos_config = ChaosConfig(
                chaos_probability=self.config.chaos_probability,
                curriculum_enabled=self.config.chaos_curriculum,
            )
            self.chaos_generator = ChaosGenerator(chaos_config)
        else:
            self.chaos_generator = None

        if self.config.character_enabled:
            self.character = CharacterCore(storage_dir=self.config.character_storage_dir)
        else:
            self.character = None

        if self.config.metacognition_enabled:
            self.metacognition = MetaCognitionModule(hidden_dim=hidden_dim)
        else:
            self.metacognition = None

        if self.config.perception_eval_enabled:
            self.evaluator = PerceptionEvaluator(strictness=self.config.perception_eval_strictness)
        else:
            self.evaluator = None

        # Combined perception head for end-to-end training
        input_dim = hidden_dim
        if self.config.tom_enabled:
            input_dim += hidden_dim // 2  # Mental state features
        if self.config.metacognition_enabled:
            input_dim += 4  # Meta-cognition outputs

        self.perception_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 6),  # Perception scores
        )

        # Current perception state
        self._current_state = PerceptionState()
        self._chaos_injection: ChaosInjection | None = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_logits: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass combining all perception modules.

        Args:
            hidden_states: [batch, seq_len, hidden_dim] from language model
            attention_mask: Optional attention mask
            output_logits: Optional model output logits

        Returns:
            Dictionary with all perception predictions
        """
        outputs = {}
        features = []

        # Pool hidden states for base features
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).expand(hidden_states.size())
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = hidden_states.mean(dim=1)

        features.append(pooled)

        # Theory of Mind
        if self.mental_state_tracker is not None:
            tom_out = self.mental_state_tracker(hidden_states, attention_mask)
            outputs.update({f"tom_{k}": v for k, v in tom_out.items()})
            features.append(tom_out["belief_embedding"])

        # Meta-Cognition
        if self.metacognition is not None:
            meta_out = self.metacognition(hidden_states, attention_mask, output_logits)
            outputs.update({f"meta_{k}": v for k, v in meta_out.items()})
            meta_features = torch.cat(
                [
                    meta_out["should_hedge"],
                    meta_out["should_clarify"],
                    meta_out["should_ask_user"],
                    meta_out["meta_confidence"],
                ],
                dim=-1,
            )
            features.append(meta_features)

        # Combined perception
        combined = torch.cat(features, dim=-1)
        perception_scores = self.perception_head(combined)
        perception_scores = torch.sigmoid(perception_scores)

        outputs["perception_scores"] = perception_scores
        outputs["pooled_features"] = pooled

        return outputs

    def prepare_input(
        self,
        prompt: str,
        apply_chaos: bool = True,
    ) -> tuple[str, PerceptionState]:
        """
        Prepare input for training/inference with perception context.

        Args:
            prompt: Original user prompt
            apply_chaos: Whether to potentially apply chaos injection

        Returns:
            Tuple of (processed_prompt, perception_state)
        """
        state = PerceptionState()

        # Apply chaos if enabled and dice roll succeeds
        processed_prompt = prompt
        if apply_chaos and self.chaos_generator is not None:
            injection = self.chaos_generator.inject(prompt)
            if injection:
                processed_prompt = injection.modified_input
                state.chaos_applied = True
                state.chaos_type = injection.chaos_type.value
                state.expected_handling = injection.ground_truth
                self._chaos_injection = injection

        # Generate character context
        if self.character is not None:
            state.character_prompt = self.character.generate_character_prompt(prompt)
            state.active_values = [
                v.value_type.value for v in self.character.get_active_values(prompt)
            ]

        self._current_state = state
        return processed_prompt, state

    def update_mental_state(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> None:
        """Update mental state tracking from model outputs."""
        if self.mental_state_tracker is not None:
            with torch.no_grad():
                predictions = self.mental_state_tracker(hidden_states, attention_mask)
                self.mental_state_tracker.update_from_predictions(predictions)

            # Update current state
            self._current_state.user_frustration = (
                self.mental_state_tracker.intent_state.detect_frustration_signals()
            )
            self._current_state.user_expertise = (
                self.mental_state_tracker.knowledge_state.domain_expertise
            )

            if self.mental_state_tracker.intent_state.surface_intent:
                self._current_state.user_intent = (
                    self.mental_state_tracker.intent_state.surface_intent.value
                )

            if self.mental_state_tracker.emotional_state.primary_emotion:
                self._current_state.user_emotion = (
                    self.mental_state_tracker.emotional_state.primary_emotion.value
                )

    def get_perception_prompt(self) -> str:
        """
        Generate perception-aware prompt augmentation.

        This prompt can be prepended to encourage perception-aware responses.
        """
        parts = []

        # Theory of Mind prompt
        if self.mental_state_tracker is not None:
            tom_prompt = self.mental_state_tracker.get_perspective_prompt()
            if tom_prompt:
                parts.append(tom_prompt)

        # Character prompt
        if self._current_state.character_prompt:
            parts.append(self._current_state.character_prompt)

        # Chaos handling hint (for training - tells model what's expected)
        if self._current_state.chaos_applied:
            parts.append(
                f"NOTE: Input may contain {self._current_state.chaos_type}. "
                f"Expected handling: {self._current_state.expected_handling}"
            )

        return "\n".join(parts)

    def evaluate_response(
        self,
        prompt: str,
        response: str,
    ) -> PerceptionEvaluation | None:
        """
        Evaluate a response on perception dimensions.

        Args:
            prompt: Original prompt
            response: Agent's response

        Returns:
            PerceptionEvaluation if evaluator is enabled
        """
        if self.evaluator is None:
            return None

        context = {
            "mental_state": self.mental_state_tracker,
            "chaos_injection": self._chaos_injection,
            "character": self.character,
            "perception_state": self._current_state,
        }

        return self.evaluator.evaluate(prompt, response, context)

    def record_experience(
        self,
        prompt: str,
        response: str,
        outcome: str,
        was_successful: bool,
    ) -> None:
        """
        Record an experience in autobiographical memory.

        This helps build consistent behavior over time.
        """
        if self.character is not None:
            # Determine relevant context tags
            tags = []
            if self._current_state.chaos_applied:
                tags.append(f"chaos_{self._current_state.chaos_type}")
            if self._current_state.user_intent:
                tags.append(f"intent_{self._current_state.user_intent}")

            # Derive lesson from outcome
            if was_successful:
                lesson = f"This approach worked well for {self._current_state.user_intent}"
            else:
                lesson = f"Consider alternative approach for {self._current_state.user_intent}"

            self.character.record_experience(
                situation=prompt[:200],
                action_taken=response[:200],
                outcome=outcome,
                lesson=lesson,
                context_tags=tags,
                was_positive=was_successful,
            )

    def set_training_epoch(self, epoch: int) -> None:
        """Update epoch for curriculum adjustments."""
        if self.chaos_generator is not None:
            self.chaos_generator.set_epoch(epoch)

    def get_trainable_parameters(self) -> list[nn.Parameter]:
        """Get all trainable parameters from perception modules."""
        params = list(self.perception_head.parameters())

        if self.mental_state_tracker is not None:
            params.extend(self.mental_state_tracker.parameters())

        if self.metacognition is not None:
            params.extend(self.metacognition.parameters())

        return params

    def state_dict_perception(self) -> dict[str, Any]:
        """Get state dict for perception modules only."""
        state = {
            "perception_head": self.perception_head.state_dict(),
            "config": self.config,
        }

        if self.mental_state_tracker is not None:
            state["mental_state_tracker"] = self.mental_state_tracker.state_dict()

        if self.metacognition is not None:
            state["metacognition"] = self.metacognition.state_dict()

        if self.character is not None:
            state["character_identity"] = self.character.get_identity_hash()

        return state

    def load_state_dict_perception(self, state: dict[str, Any]) -> None:
        """Load state dict for perception modules."""
        self.perception_head.load_state_dict(state["perception_head"])

        if "mental_state_tracker" in state and self.mental_state_tracker is not None:
            self.mental_state_tracker.load_state_dict(state["mental_state_tracker"])

        if "metacognition" in state and self.metacognition is not None:
            self.metacognition.load_state_dict(state["metacognition"])

    def reset_conversation_state(self) -> None:
        """Reset state for new conversation."""
        self._current_state = PerceptionState()
        self._chaos_injection = None

        if self.mental_state_tracker is not None:
            self.mental_state_tracker.reset()


class PerceptionLoss(nn.Module):
    """
    Loss function for training perception modules.

    Combines multiple objectives:
    1. Mental state prediction accuracy
    2. Uncertainty calibration
    3. Character consistency
    4. Chaos handling
    """

    def __init__(
        self,
        tom_weight: float = 0.3,
        metacognition_weight: float = 0.3,
        character_weight: float = 0.2,
        chaos_weight: float = 0.2,
    ):
        super().__init__()

        self.tom_weight = tom_weight
        self.metacognition_weight = metacognition_weight
        self.character_weight = character_weight
        self.chaos_weight = chaos_weight

    def forward(
        self,
        perception_outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Compute perception loss.

        Args:
            perception_outputs: Output from PerceptionModule.forward()
            targets: Target values for each component

        Returns:
            Dictionary with total loss and component losses
        """
        losses = {}
        total_loss = torch.tensor(0.0, device=next(iter(perception_outputs.values())).device)

        # Theory of Mind loss
        if "tom_emotion_logits" in perception_outputs and "emotion_target" in targets:
            emotion_loss = F.cross_entropy(
                perception_outputs["tom_emotion_logits"],
                targets["emotion_target"],
            )
            losses["tom_emotion"] = emotion_loss
            total_loss = total_loss + self.tom_weight * emotion_loss

        if "tom_intent_logits" in perception_outputs and "intent_target" in targets:
            intent_loss = F.cross_entropy(
                perception_outputs["tom_intent_logits"],
                targets["intent_target"],
            )
            losses["tom_intent"] = intent_loss
            total_loss = total_loss + self.tom_weight * intent_loss

        # Meta-cognition loss (calibration)
        if "meta_overall_uncertainty" in perception_outputs and "uncertainty_target" in targets:
            uncertainty_loss = F.mse_loss(
                perception_outputs["meta_overall_uncertainty"],
                targets["uncertainty_target"],
            )
            losses["uncertainty"] = uncertainty_loss
            total_loss = total_loss + self.metacognition_weight * uncertainty_loss

        # Chaos handling loss
        if "perception_scores" in perception_outputs and "chaos_handling_target" in targets:
            chaos_loss = F.mse_loss(
                perception_outputs["perception_scores"],
                targets["chaos_handling_target"],
            )
            losses["chaos_handling"] = chaos_loss
            total_loss = total_loss + self.chaos_weight * chaos_loss

        losses["total"] = total_loss
        return losses


def integrate_perception_with_trainer(
    trainer,  # AspireTrainer instance
    perception_module: PerceptionModule,
) -> None:
    """
    Integrate perception module with an existing AspireTrainer.

    This modifies the trainer's training loop to include perception.

    Args:
        trainer: AspireTrainer instance
        perception_module: PerceptionModule instance
    """
    # Store reference
    trainer.perception_module = perception_module

    # Add perception parameters to optimizer
    perception_params = perception_module.get_trainable_parameters()
    if perception_params:
        for param_group in trainer.optimizer.param_groups:
            param_group["params"].extend(perception_params)

    # Wrap the original training step
    original_train_step = trainer._train_step

    async def perception_train_step(*args, **kwargs):
        # Prepare input with perception
        if "prompt" in kwargs:
            processed_prompt, state = perception_module.prepare_input(kwargs["prompt"])
            kwargs["prompt"] = processed_prompt
            kwargs["perception_state"] = state

        # Run original training step
        result = await original_train_step(*args, **kwargs)

        # Update perception state from model outputs
        if "hidden_states" in result:
            perception_module.update_mental_state(
                result["hidden_states"],
                result.get("attention_mask"),
            )

        # Compute perception loss
        if "hidden_states" in result:
            perception_module(
                result["hidden_states"],
                result.get("attention_mask"),
            )
            # Note: Would need targets to compute loss - this is structural

        return result

    trainer._train_step = perception_train_step

    print("Perception module integrated with trainer")
    print(f"  - ToM enabled: {perception_module.config.tom_enabled}")
    print(f"  - Chaos enabled: {perception_module.config.chaos_enabled}")
    print(f"  - Character enabled: {perception_module.config.character_enabled}")
    print(f"  - Metacognition enabled: {perception_module.config.metacognition_enabled}")
