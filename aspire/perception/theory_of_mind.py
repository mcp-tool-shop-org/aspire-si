"""
Theory of Mind (ToM) Module - Mental state tracking and perspective-taking.

This module enables agents to maintain explicit models of what other agents/users
likely know, believe, intend, and feel. It's the foundation of cognitive empathy.

The key insight is that empathy isn't just sentiment detection - it's the ability
to model another mind's trajectory through a conversation, anticipating what they
might be thinking based on what information they have access to.

Architecture:
┌─────────────────────────────────────────────────────────────┐
│                  MentalStateTracker                         │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Belief    │  │   Intent    │  │  Emotional  │         │
│  │   State     │  │   State     │  │   State     │         │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘         │
│         │                │                │                 │
│         └────────────────┼────────────────┘                 │
│                          │                                  │
│                  ┌───────▼───────┐                         │
│                  │  Knowledge    │                         │
│                  │    State      │                         │
│                  └───────────────┘                         │
└─────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

import torch
import torch.nn as nn


class EmotionalValence(str, Enum):
    """Basic emotional valence categories."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


class EmotionType(str, Enum):
    """Specific emotion types for more nuanced tracking."""

    # Positive emotions
    CURIOSITY = "curiosity"
    SATISFACTION = "satisfaction"
    ENTHUSIASM = "enthusiasm"
    RELIEF = "relief"
    GRATITUDE = "gratitude"

    # Negative emotions
    FRUSTRATION = "frustration"
    CONFUSION = "confusion"
    IMPATIENCE = "impatience"
    DISAPPOINTMENT = "disappointment"
    ANXIETY = "anxiety"

    # Neutral/Task-focused
    FOCUSED = "focused"
    ANALYTICAL = "analytical"
    EXPLORATORY = "exploratory"


class IntentCategory(str, Enum):
    """Categories of user intent."""

    SEEKING_INFORMATION = "seeking_information"
    SEEKING_VALIDATION = "seeking_validation"
    DEBUGGING = "debugging"
    LEARNING = "learning"
    EXPLORING = "exploring"
    VENTING = "venting"
    TESTING_LIMITS = "testing_limits"
    BUILDING = "building"
    CLARIFYING = "clarifying"


@dataclass
class BeliefState:
    """
    Tracks what the user likely believes to be true.

    This isn't about what IS true, but what the user's mental model contains.
    The agent needs to know what assumptions the user is operating under.
    """

    # Explicit beliefs extracted from conversation
    explicit_beliefs: dict[str, float] = field(default_factory=dict)  # belief -> confidence

    # Inferred beliefs (not stated but implied)
    inferred_beliefs: dict[str, float] = field(default_factory=dict)

    # Misconceptions detected (beliefs that conflict with ground truth)
    misconceptions: dict[str, str] = field(default_factory=dict)  # belief -> correction

    # Knowledge gaps (things the user doesn't know they don't know)
    knowledge_gaps: list[str] = field(default_factory=list)

    # Timestamp for staleness tracking
    last_updated: float = field(default_factory=time.time)

    def add_belief(
        self,
        belief: str,
        confidence: float = 0.7,
        explicit: bool = True,
    ) -> None:
        """Add a belief with confidence score."""
        target = self.explicit_beliefs if explicit else self.inferred_beliefs
        target[belief] = min(1.0, max(0.0, confidence))
        self.last_updated = time.time()

    def flag_misconception(self, belief: str, correction: str) -> None:
        """Flag a belief as incorrect and provide correction."""
        self.misconceptions[belief] = correction
        # Remove from beliefs
        self.explicit_beliefs.pop(belief, None)
        self.inferred_beliefs.pop(belief, None)
        self.last_updated = time.time()

    def get_belief_confidence(self, belief: str) -> float:
        """Get confidence for a specific belief."""
        if belief in self.explicit_beliefs:
            return self.explicit_beliefs[belief]
        if belief in self.inferred_beliefs:
            return self.inferred_beliefs[belief] * 0.8  # Discount inferred
        return 0.0


@dataclass
class IntentState:
    """
    Tracks the user's goals and intentions.

    Intent is layered:
    - Surface intent: What they're explicitly asking for
    - Underlying intent: Why they're asking (often implicit)
    - Meta-intent: Their broader goal in this session
    """

    # Current turn's explicit intent
    surface_intent: IntentCategory | None = None
    surface_confidence: float = 0.0

    # Underlying motivation
    underlying_intent: IntentCategory | None = None
    underlying_confidence: float = 0.0

    # Session-level goal
    meta_intent: str | None = None

    # Intent history for trajectory tracking
    intent_trajectory: list[tuple[IntentCategory, float]] = field(default_factory=list)

    # Urgency level (affects response style)
    urgency: float = 0.5  # 0 = relaxed exploration, 1 = urgent need

    # Whether intent has shifted recently (might indicate frustration)
    intent_stability: float = 1.0  # 0 = rapidly shifting, 1 = stable

    last_updated: float = field(default_factory=time.time)

    def update_intent(
        self,
        surface: IntentCategory,
        surface_conf: float,
        underlying: IntentCategory | None = None,
        underlying_conf: float = 0.0,
    ) -> None:
        """Update intent state and track trajectory."""
        # Track stability
        if self.surface_intent and self.surface_intent != surface:
            self.intent_stability = max(0.0, self.intent_stability - 0.2)
        else:
            self.intent_stability = min(1.0, self.intent_stability + 0.1)

        self.surface_intent = surface
        self.surface_confidence = surface_conf

        if underlying:
            self.underlying_intent = underlying
            self.underlying_confidence = underlying_conf

        self.intent_trajectory.append((surface, surface_conf))
        # Keep trajectory bounded
        if len(self.intent_trajectory) > 20:
            self.intent_trajectory = self.intent_trajectory[-20:]

        self.last_updated = time.time()

    def detect_frustration_signals(self) -> float:
        """Detect signals of user frustration from intent trajectory."""
        if len(self.intent_trajectory) < 3:
            return 0.0

        signals = 0.0

        # Rapid intent shifts
        if self.intent_stability < 0.5:
            signals += 0.3

        # Repeated clarification requests
        recent = self.intent_trajectory[-5:]
        clarifying_count = sum(1 for i, _ in recent if i == IntentCategory.CLARIFYING)
        if clarifying_count >= 2:
            signals += 0.3

        # High urgency
        if self.urgency > 0.7:
            signals += 0.2

        return min(1.0, signals)


@dataclass
class EmotionalState:
    """
    Tracks the user's emotional state and trajectory.

    Emotional tracking is crucial for appropriate response calibration.
    A frustrated user needs different handling than an enthusiastic one.
    """

    # Current emotional state
    primary_emotion: EmotionType | None = None
    primary_intensity: float = 0.5  # 0-1 scale

    # Secondary emotion (emotions are often complex)
    secondary_emotion: EmotionType | None = None
    secondary_intensity: float = 0.0

    # Overall valence
    valence: EmotionalValence = EmotionalValence.NEUTRAL

    # Emotional trajectory (for trend detection)
    emotion_history: list[tuple[EmotionType, float, float]] = field(
        default_factory=list
    )  # (emotion, intensity, timestamp)

    # Rapport indicators
    rapport_level: float = 0.5  # 0 = poor, 1 = excellent
    trust_level: float = 0.5

    last_updated: float = field(default_factory=time.time)

    def update_emotion(
        self,
        emotion: EmotionType,
        intensity: float,
        secondary: EmotionType | None = None,
        secondary_intensity: float = 0.0,
    ) -> None:
        """Update emotional state."""
        self.primary_emotion = emotion
        self.primary_intensity = min(1.0, max(0.0, intensity))

        if secondary:
            self.secondary_emotion = secondary
            self.secondary_intensity = secondary_intensity

        # Determine valence
        positive_emotions = {
            EmotionType.CURIOSITY,
            EmotionType.SATISFACTION,
            EmotionType.ENTHUSIASM,
            EmotionType.RELIEF,
            EmotionType.GRATITUDE,
        }
        negative_emotions = {
            EmotionType.FRUSTRATION,
            EmotionType.CONFUSION,
            EmotionType.IMPATIENCE,
            EmotionType.DISAPPOINTMENT,
            EmotionType.ANXIETY,
        }

        if emotion in positive_emotions:
            self.valence = EmotionalValence.POSITIVE
        elif emotion in negative_emotions:
            self.valence = EmotionalValence.NEGATIVE
        else:
            self.valence = EmotionalValence.NEUTRAL

        # Track history
        now = time.time()
        self.emotion_history.append((emotion, intensity, now))
        if len(self.emotion_history) > 50:
            self.emotion_history = self.emotion_history[-50:]

        self.last_updated = now

    def get_emotional_trend(self) -> str:
        """Detect emotional trajectory: improving, declining, or stable."""
        if len(self.emotion_history) < 3:
            return "insufficient_data"

        recent = self.emotion_history[-5:]
        positive_count = sum(
            1
            for e, _, _ in recent
            if e
            in {
                EmotionType.CURIOSITY,
                EmotionType.SATISFACTION,
                EmotionType.ENTHUSIASM,
                EmotionType.RELIEF,
            }
        )
        negative_count = sum(
            1
            for e, _, _ in recent
            if e
            in {
                EmotionType.FRUSTRATION,
                EmotionType.CONFUSION,
                EmotionType.IMPATIENCE,
                EmotionType.DISAPPOINTMENT,
            }
        )

        if positive_count > negative_count + 1:
            return "improving"
        elif negative_count > positive_count + 1:
            return "declining"
        return "stable"

    def adjust_rapport(self, delta: float) -> None:
        """Adjust rapport level based on interaction quality."""
        self.rapport_level = min(1.0, max(0.0, self.rapport_level + delta))
        self.last_updated = time.time()


@dataclass
class KnowledgeState:
    """
    Tracks what information the user has been exposed to.

    This is crucial for avoiding repetition and building on shared context.
    """

    # Topics discussed
    topics_covered: dict[str, int] = field(default_factory=dict)  # topic -> mention count

    # Information provided by agent
    information_shared: list[str] = field(default_factory=list)

    # Questions asked by user
    questions_asked: list[str] = field(default_factory=list)

    # Things the user has demonstrated understanding of
    demonstrated_understanding: set[str] = field(default_factory=set)

    # Things the user seems confused about
    confusion_points: set[str] = field(default_factory=set)

    # Estimated expertise level in current domain
    domain_expertise: float = 0.5  # 0 = novice, 1 = expert

    last_updated: float = field(default_factory=time.time)

    def record_topic(self, topic: str) -> None:
        """Record that a topic was discussed."""
        self.topics_covered[topic] = self.topics_covered.get(topic, 0) + 1
        self.last_updated = time.time()

    def record_understanding(self, concept: str) -> None:
        """Record that user demonstrated understanding of a concept."""
        self.demonstrated_understanding.add(concept)
        self.confusion_points.discard(concept)
        self.last_updated = time.time()

    def record_confusion(self, concept: str) -> None:
        """Record that user seems confused about a concept."""
        self.confusion_points.add(concept)
        self.last_updated = time.time()

    def has_been_explained(self, topic: str, threshold: int = 1) -> bool:
        """Check if topic has been adequately covered."""
        return self.topics_covered.get(topic, 0) >= threshold


class MentalStateTracker(nn.Module):
    """
    Neural module for tracking mental states from conversation.

    This integrates with the ASPIRE training loop to learn to predict
    mental states that lead to better teaching outcomes.
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_emotions: int = len(EmotionType),
        num_intents: int = len(IntentCategory),
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Emotion prediction head
        self.emotion_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_emotions),
        )

        # Intent prediction head
        self.intent_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_intents),
        )

        # Belief extraction (produces embedding that can be decoded)
        self.belief_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )

        # Frustration detector (binary + intensity)
        self.frustration_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 2),  # [is_frustrated, intensity]
        )

        # State aggregator (combines all signals)
        self.state_aggregator = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            batch_first=True,
        )

        # Current tracked states (not learnable, just storage)
        self._belief_state = BeliefState()
        self._intent_state = IntentState()
        self._emotional_state = EmotionalState()
        self._knowledge_state = KnowledgeState()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass to predict mental states from hidden representations.

        Args:
            hidden_states: [batch, seq_len, hidden_dim] from language model
            attention_mask: Optional attention mask

        Returns:
            Dictionary of predictions:
            - emotion_logits: [batch, num_emotions]
            - intent_logits: [batch, num_intents]
            - belief_embedding: [batch, hidden_dim // 2]
            - frustration: [batch, 2] (is_frustrated, intensity)
        """
        # Pool to get sequence representation
        if attention_mask is not None:
            # Masked mean pooling
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
            sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled = sum_hidden / sum_mask
        else:
            # Simple mean pooling
            pooled = hidden_states.mean(dim=1)

        # Predict each component
        emotion_logits = self.emotion_encoder(pooled)
        intent_logits = self.intent_encoder(pooled)
        belief_embedding = self.belief_encoder(pooled)
        frustration = self.frustration_detector(pooled)
        frustration = torch.sigmoid(frustration)  # Bound to 0-1

        return {
            "emotion_logits": emotion_logits,
            "intent_logits": intent_logits,
            "belief_embedding": belief_embedding,
            "frustration": frustration,
            "pooled_state": pooled,
        }

    def update_from_predictions(
        self,
        predictions: dict[str, torch.Tensor],
    ) -> None:
        """Update internal state tracking from model predictions."""
        with torch.no_grad():
            # Update emotion
            emotion_probs = torch.softmax(predictions["emotion_logits"], dim=-1)
            emotion_idx = emotion_probs.argmax(dim=-1).item()
            emotion = list(EmotionType)[emotion_idx]
            intensity = emotion_probs.max().item()
            self._emotional_state.update_emotion(emotion, intensity)

            # Update intent
            intent_probs = torch.softmax(predictions["intent_logits"], dim=-1)
            intent_idx = intent_probs.argmax(dim=-1).item()
            intent = list(IntentCategory)[intent_idx]
            confidence = intent_probs.max().item()
            self._intent_state.update_intent(intent, confidence)

            # Update frustration-related states
            frustration = predictions["frustration"]
            if frustration[0, 0].item() > 0.5:  # is_frustrated
                intensity = frustration[0, 1].item()
                self._emotional_state.adjust_rapport(-0.1 * intensity)

    def get_perspective_prompt(self) -> str:
        """
        Generate a perspective-taking prompt based on current mental state.

        This can be prepended to the agent's reasoning to encourage
        perspective-taking before responding.
        """
        prompts = []

        # Emotional awareness
        if self._emotional_state.primary_emotion:
            emotion = self._emotional_state.primary_emotion.value
            intensity = self._emotional_state.primary_intensity
            if intensity > 0.7:
                prompts.append(
                    f"The user appears strongly {emotion}. Consider how this affects "
                    f"what kind of response would be most helpful."
                )
            elif intensity > 0.4:
                prompts.append(f"The user seems {emotion}. Keep this in mind when responding.")

        # Intent awareness
        if self._intent_state.surface_intent:
            intent = self._intent_state.surface_intent.value.replace("_", " ")
            prompts.append(f"The user is {intent}. Make sure the response addresses this need.")

        # Frustration check
        frustration = self._intent_state.detect_frustration_signals()
        if frustration > 0.5:
            prompts.append(
                "IMPORTANT: The user may be frustrated. Consider acknowledging "
                "difficulty and being extra clear and helpful."
            )

        # Knowledge level
        expertise = self._knowledge_state.domain_expertise
        if expertise < 0.3:
            prompts.append("The user appears to be a novice. Avoid jargon and explain concepts.")
        elif expertise > 0.7:
            prompts.append("The user appears experienced. You can use technical language freely.")

        # Misconceptions
        if self._belief_state.misconceptions:
            misconceptions = list(self._belief_state.misconceptions.items())[:2]
            for belief, correction in misconceptions:
                prompts.append(
                    f"Note: User believes '{belief}' but this is incorrect. "
                    f"Consider gently addressing: {correction}"
                )

        if prompts:
            return "PERSPECTIVE TAKING:\n" + "\n".join(f"• {p}" for p in prompts) + "\n\n"
        return ""

    @property
    def belief_state(self) -> BeliefState:
        return self._belief_state

    @property
    def intent_state(self) -> IntentState:
        return self._intent_state

    @property
    def emotional_state(self) -> EmotionalState:
        return self._emotional_state

    @property
    def knowledge_state(self) -> KnowledgeState:
        return self._knowledge_state

    def reset(self) -> None:
        """Reset all tracked states."""
        self._belief_state = BeliefState()
        self._intent_state = IntentState()
        self._emotional_state = EmotionalState()
        self._knowledge_state = KnowledgeState()

    def get_trainable_parameters(self) -> list[nn.Parameter]:
        """Get all trainable parameters."""
        return list(self.parameters())
