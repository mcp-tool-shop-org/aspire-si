"""
Syntropy Module - The ordering force opposing entropy in AI cognition.

Syntropy (negentropy) represents the measure of order, organization, and coherence
within a system. While entropy quantifies disorder and randomness, syntropy reflects
the presence of structure, meaning, and intentional organization.

Key Insight from Szent-Györgyi (1974): Machines wear out with use, but living
organisms improve through activity. Syntropy captures this life-affirming,
order-creating force that distinguishes genuine understanding from mere pattern matching.

In the context of AI empathy and cognition, syntropy manifests as:
1. Coherence - Responses that hang together meaningfully
2. Resonance - Alignment between agent and user mental states
3. Integration - Unifying disparate information into understanding
4. Emergence - Creating new meaning greater than sum of parts
5. Attunement - Converging toward shared understanding (syntropic attractor)

Mathematical Foundation:
- Negentropy: J(x) = H(x_gaussian) - H(x)
- Measures distance from maximum entropy (Gaussian) distribution
- Higher negentropy = more structured/organized signal

References:
- Schrödinger, E. (1944). What is Life? - introduced "negative entropy"
- Szent-Györgyi, A. (1974). Proposed "syntropy" as life's ordering principle
- Fantappiè, L. (1941). Discovered syntropic (converging) wave solutions
- Hyvärinen & Oja (2000). ICA using negentropy maximization
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class SyntropicDimension(str, Enum):
    """Dimensions of syntropic organization in cognition."""

    # Structural coherence
    SEMANTIC_COHERENCE = "semantic_coherence"  # Ideas connect meaningfully
    LOGICAL_CONSISTENCY = "logical_consistency"  # No contradictions
    NARRATIVE_FLOW = "narrative_flow"  # Natural progression of thought

    # Relational attunement
    EMPATHIC_RESONANCE = "empathic_resonance"  # Emotional alignment
    INTENTIONAL_ALIGNMENT = "intentional_alignment"  # Goal convergence
    KNOWLEDGE_BRIDGING = "knowledge_bridging"  # Meeting at understanding level

    # Emergent organization
    MEANING_GENERATION = "meaning_generation"  # Creating understanding
    INSIGHT_CRYSTALLIZATION = "insight_crystallization"  # Clarity emergence
    INTEGRATION_DEPTH = "integration_depth"  # Unifying diverse elements

    # Temporal coherence
    CONTEXTUAL_CONTINUITY = "contextual_continuity"  # Maintains thread
    PROGRESSIVE_REFINEMENT = "progressive_refinement"  # Improves over interaction
    ANTICIPATORY_COHERENCE = "anticipatory_coherence"  # Prepares for future


class SyntropicState(str, Enum):
    """States of syntropic flow in interaction."""

    ENTROPIC = "entropic"  # Dissolving into disorder
    NEUTRAL = "neutral"  # Neither gaining nor losing order
    SYNTROPIC = "syntropic"  # Actively creating order
    RESONANT = "resonant"  # High coherence, mutual understanding
    CRYSTALLIZED = "crystallized"  # Insight has formed


@dataclass
class SyntropicMeasurement:
    """A measurement of syntropy at a point in time."""

    # Overall syntropy score (-1 to 1, negative = entropic)
    syntropy_score: float

    # Dimensional breakdown
    dimension_scores: dict[SyntropicDimension, float] = field(default_factory=dict)

    # Derived state
    state: SyntropicState = SyntropicState.NEUTRAL

    # Trajectory (is syntropy increasing or decreasing?)
    trajectory: float = 0.0  # -1 to 1

    # Negentropy measure (always >= 0)
    negentropy: float = 0.0

    # Timestamp
    timestamp: float = field(default_factory=time.time)

    def compute_state(self) -> SyntropicState:
        """Compute syntropic state from score."""
        if self.syntropy_score < -0.3:
            return SyntropicState.ENTROPIC
        elif self.syntropy_score < 0.2:
            return SyntropicState.NEUTRAL
        elif self.syntropy_score < 0.5:
            return SyntropicState.SYNTROPIC
        elif self.syntropy_score < 0.8:
            return SyntropicState.RESONANT
        else:
            return SyntropicState.CRYSTALLIZED


@dataclass
class CoherenceField:
    """
    Represents the coherence field between agent and user.

    Inspired by Fantappiè's discovery of converging wave solutions,
    this models the attractor basin toward shared understanding.
    """

    # Field strength (0 = no coherence, 1 = perfect resonance)
    strength: float = 0.0

    # Field stability (resistance to disruption)
    stability: float = 0.5

    # Attractor position (where understanding is converging)
    attractor_embedding: torch.Tensor | None = None

    # Divergence measure (how far current state is from attractor)
    divergence: float = 1.0

    # History of field evolution (bounded deque prevents resource exhaustion)
    strength_history: deque[float] = field(default_factory=lambda: deque(maxlen=100))

    def update(self, new_strength: float, new_embedding: torch.Tensor | None = None) -> None:
        """Update coherence field with new measurement."""
        # Exponential moving average for stability
        alpha = 0.3
        self.strength = alpha * new_strength + (1 - alpha) * self.strength

        # Track history (deque automatically maintains bounded size)
        self.strength_history.append(self.strength)

        # Update attractor if provided
        if new_embedding is not None:
            if self.attractor_embedding is None:
                self.attractor_embedding = new_embedding.clone()
            else:
                # Gradual attractor shift
                self.attractor_embedding = 0.9 * self.attractor_embedding + 0.1 * new_embedding

        # Compute stability from variance using numerically stable method
        if len(self.strength_history) >= 5:
            recent = list(self.strength_history)[-10:]
            n = len(recent)
            mean = sum(recent) / n
            # Two-pass variance for numerical stability
            variance = sum((x - mean) ** 2 for x in recent) / n
            self.stability = max(0.0, min(1.0, 1.0 - variance * 4))

    def get_convergence_rate(self) -> float:
        """Compute how quickly field is converging."""
        if len(self.strength_history) < 3:
            return 0.0

        # Convert deque to list for slicing
        history_list = list(self.strength_history)
        recent = history_list[-5:]
        if len(recent) < 2:
            return 0.0

        # Simple linear trend
        n = len(recent)
        x_mean = (n - 1) / 2
        y_mean = sum(recent) / n

        numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(recent))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return 0.0

        slope = numerator / denominator
        return max(-1.0, min(1.0, slope * 5))  # Scale to -1 to 1


def compute_negentropy_approximation(
    x: torch.Tensor,
    method: str = "exp",
) -> torch.Tensor:
    """
    Compute negentropy approximation for a tensor.

    Negentropy J(y) ≈ [E{G(y)} - E{G(ν)}]²

    where G is a non-quadratic function and ν is standard Gaussian.

    Higher negentropy indicates more structured (less Gaussian) distribution,
    which we interpret as more organized/meaningful representation.

    Args:
        x: Input tensor (assumed zero mean, unit variance after normalization)
        method: Approximation method - "exp", "logcosh", or "kurtosis"

    Returns:
        Negentropy estimate (always non-negative)
    """
    # Normalize to zero mean, unit variance
    x_normalized = (x - x.mean()) / (x.std() + 1e-8)

    if method == "exp":
        # G(u) = -exp(-u²/2)
        # E[G(ν)] for standard Gaussian = -1/sqrt(2)
        g_x = -torch.exp(-(x_normalized**2) / 2)
        g_gaussian = -1.0 / math.sqrt(2)
        negentropy = (g_x.mean() - g_gaussian) ** 2

    elif method == "logcosh":
        # G(u) = log(cosh(u))
        # E[G(ν)] for standard Gaussian ≈ 0.3746
        g_x = torch.log(torch.cosh(x_normalized.clamp(-10, 10)))
        g_gaussian = 0.3746
        negentropy = (g_x.mean() - g_gaussian) ** 2

    elif method == "kurtosis":
        # Kurtosis-based: J ≈ (1/12) * E[x³]² + (1/48) * (E[x⁴] - 3)²
        # Gaussian has kurtosis = 3, skewness = 0
        skewness = (x_normalized**3).mean()
        kurtosis = (x_normalized**4).mean()
        negentropy = (1 / 12) * skewness**2 + (1 / 48) * (kurtosis - 3) ** 2

    else:
        raise ValueError(f"Unknown method: {method}")

    return negentropy


def compute_semantic_coherence(
    embeddings: torch.Tensor,
    window_size: int = 5,
) -> torch.Tensor:
    """
    Compute semantic coherence from a sequence of embeddings.

    Coherence is measured as the consistency of semantic relationships
    across the sequence - high coherence means ideas flow naturally.

    Args:
        embeddings: [seq_len, hidden_dim] or [batch, seq_len, hidden_dim]
        window_size: Size of sliding window for local coherence

    Returns:
        Coherence score [0, 1]
    """
    if embeddings.dim() == 2:
        embeddings = embeddings.unsqueeze(0)

    batch_size, seq_len, hidden_dim = embeddings.shape

    if seq_len < 2:
        return torch.ones(batch_size)

    # Normalize embeddings
    embeddings_norm = F.normalize(embeddings, p=2, dim=-1)

    # Compute adjacent similarities
    adjacent_sim = torch.sum(
        embeddings_norm[:, :-1] * embeddings_norm[:, 1:], dim=-1
    )  # [batch, seq_len-1]

    # Local coherence: mean similarity in sliding windows
    if seq_len >= window_size:
        local_coherences = []
        for i in range(seq_len - window_size + 1):
            window = embeddings_norm[:, i : i + window_size]
            # Compute mean pairwise similarity in window
            sim_matrix = torch.bmm(window, window.transpose(1, 2))
            # Exclude diagonal by creating mask and zeroing diagonal
            mask = ~torch.eye(window_size, dtype=torch.bool, device=sim_matrix.device)
            mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
            # Get sum of off-diagonal elements and divide by count
            num_off_diag = window_size * (window_size - 1)
            local_sim = (sim_matrix * mask.float()).sum(dim=(1, 2)) / num_off_diag
            local_coherences.append(local_sim)

        local_coherence = torch.stack(local_coherences, dim=1).mean(dim=1)
    else:
        local_coherence = adjacent_sim.mean(dim=-1)

    # Global coherence: similarity of first and last with overall mean
    mean_embedding = embeddings_norm.mean(dim=1)
    first_to_mean = torch.sum(embeddings_norm[:, 0] * mean_embedding, dim=-1)
    last_to_mean = torch.sum(embeddings_norm[:, -1] * mean_embedding, dim=-1)
    global_coherence = (first_to_mean + last_to_mean) / 2

    # Combined coherence
    coherence = 0.6 * local_coherence + 0.4 * global_coherence

    return coherence.clamp(0, 1)


class SyntropicResonanceDetector(nn.Module):
    """
    Detects resonance between agent and user representations.

    Resonance occurs when representations converge toward shared understanding,
    creating a syntropic attractor basin.
    """

    def __init__(self, hidden_dim: int = 768):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Project both representations to resonance space
        self.agent_projector = nn.Linear(hidden_dim, hidden_dim // 2)
        self.user_projector = nn.Linear(hidden_dim, hidden_dim // 2)

        # Resonance detector
        self.resonance_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )

        # Attractor field predictor
        self.attractor_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
        )

    def forward(
        self,
        agent_state: torch.Tensor,
        user_state: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Detect resonance between agent and user states.

        Args:
            agent_state: [batch, hidden_dim] agent's representation
            user_state: [batch, hidden_dim] modeled user state

        Returns:
            Dictionary with resonance measurements
        """
        # Project to resonance space
        agent_proj = self.agent_projector(agent_state)
        user_proj = self.user_projector(user_state)

        # Cosine similarity in projected space
        similarity = F.cosine_similarity(agent_proj, user_proj, dim=-1)

        # Concatenate for resonance detection
        combined = torch.cat([agent_proj, user_proj], dim=-1)

        # Resonance score
        resonance = self.resonance_head(combined).squeeze(-1)

        # Predict attractor (midpoint that representations converge toward)
        attractor = self.attractor_predictor(combined)

        # Divergence from attractor
        agent_divergence = 1 - F.cosine_similarity(agent_proj, attractor, dim=-1)
        user_divergence = 1 - F.cosine_similarity(user_proj, attractor, dim=-1)
        total_divergence = (agent_divergence + user_divergence) / 2

        return {
            "resonance": resonance,
            "similarity": similarity,
            "attractor": attractor,
            "divergence": total_divergence,
            "agent_proj": agent_proj,
            "user_proj": user_proj,
        }


class SyntropicIntegrator(nn.Module):
    """
    Measures and promotes syntropic integration of information.

    Integration is the syntropic process of unifying disparate information
    into coherent understanding - creating order from complexity.
    """

    def __init__(self, hidden_dim: int = 768, num_heads: int = 8):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Multi-head attention for integration
        self.integration_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True,
        )

        # Integration quality assessor
        self.integration_scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Emergent meaning detector
        self.emergence_detector = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        representations: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Assess syntropic integration of representations.

        Args:
            representations: [batch, seq_len, hidden_dim]
            attention_mask: Optional mask

        Returns:
            Integration measurements
        """
        batch_size, seq_len, _ = representations.shape

        # Self-attention for integration
        if attention_mask is not None:
            # Convert to attention mask format (True = masked)
            key_padding_mask = ~attention_mask.bool()
        else:
            key_padding_mask = None

        integrated, attention_weights = self.integration_attention(
            representations,
            representations,
            representations,
            key_padding_mask=key_padding_mask,
        )

        # Pool integrated representation
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).expand(integrated.size())
            pooled = (integrated * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = integrated.mean(dim=1)

        # Integration quality score
        integration_score = self.integration_scorer(pooled).squeeze(-1)

        # Detect emergent meaning (is the whole > sum of parts?)
        # Compare pooled representation to mean of individual representations
        simple_mean = representations.mean(dim=1)
        emergence_input = torch.cat([pooled, simple_mean], dim=-1)
        emergence_score = self.emergence_detector(emergence_input).squeeze(-1)

        # Compute attention entropy (lower entropy = more focused integration)
        attention_entropy = (
            -(attention_weights * torch.log(attention_weights + 1e-10)).sum(dim=-1).mean(dim=-1)
        )
        max_entropy = math.log(seq_len)
        normalized_entropy = attention_entropy / max_entropy

        # Integration coherence (inverse of entropy)
        coherence = 1 - normalized_entropy

        return {
            "integration_score": integration_score,
            "emergence_score": emergence_score,
            "coherence": coherence,
            "integrated": integrated,
            "pooled": pooled,
            "attention_weights": attention_weights,
        }


class SyntropicFlowTracker:
    """
    Tracks the flow of syntropy across an interaction.

    Like tracking the flow of a river, this monitors how order/meaning
    accumulates or dissipates across conversation turns.
    """

    def __init__(self, window_size: int = 10, max_history: int = 100):
        self.window_size = window_size
        self.max_history = max_history

        # History of measurements (bounded deque prevents resource exhaustion)
        self.measurements: deque[SyntropicMeasurement] = deque(maxlen=max_history)

        # Coherence field
        self.coherence_field = CoherenceField()

        # Cumulative syntropy (like accumulated negative entropy)
        self.cumulative_syntropy: float = 0.0

        # Peak syntropy achieved
        self.peak_syntropy: float = 0.0

    def record(self, measurement: SyntropicMeasurement) -> None:
        """Record a new syntropic measurement."""
        # deque automatically maintains bounded size
        self.measurements.append(measurement)

        # Update coherence field
        self.coherence_field.update(measurement.syntropy_score)

        # Update cumulative (only add positive contributions)
        if measurement.syntropy_score > 0:
            self.cumulative_syntropy += measurement.syntropy_score * 0.1

        # Track peak
        self.peak_syntropy = max(self.peak_syntropy, measurement.syntropy_score)

    def get_trajectory(self) -> float:
        """Compute syntropic trajectory (is order increasing or decreasing?)."""
        if len(self.measurements) < 3:
            return 0.0

        # Convert deque to list for slicing (deque doesn't support negative slicing)
        measurements_list = list(self.measurements)
        recent = [m.syntropy_score for m in measurements_list[-self.window_size :]]
        n = len(recent)

        # Linear regression slope
        x_mean = (n - 1) / 2
        y_mean = sum(recent) / n

        numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(recent))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return 0.0

        return max(-1.0, min(1.0, numerator / denominator * 3))

    def get_state(self) -> SyntropicState:
        """Get current syntropic state."""
        if not self.measurements:
            return SyntropicState.NEUTRAL

        # Convert deque to list for slicing
        measurements_list = list(self.measurements)
        recent_measurements = measurements_list[-5:]
        recent_avg = sum(m.syntropy_score for m in recent_measurements) / len(recent_measurements)
        trajectory = self.get_trajectory()

        if recent_avg < -0.2 or trajectory < -0.3:
            return SyntropicState.ENTROPIC
        elif recent_avg < 0.2:
            return SyntropicState.NEUTRAL
        elif recent_avg < 0.5:
            return SyntropicState.SYNTROPIC
        elif recent_avg < 0.8:
            return SyntropicState.RESONANT
        else:
            return SyntropicState.CRYSTALLIZED

    def get_summary(self) -> dict[str, Any]:
        """Get summary of syntropic flow."""
        # Get last measurement (deque supports indexing but not slicing)
        current_syntropy = self.measurements[-1].syntropy_score if self.measurements else 0.0
        return {
            "current_syntropy": current_syntropy,
            "trajectory": self.get_trajectory(),
            "state": self.get_state().value,
            "cumulative": self.cumulative_syntropy,
            "peak": self.peak_syntropy,
            "coherence_field_strength": self.coherence_field.strength,
            "coherence_field_stability": self.coherence_field.stability,
            "convergence_rate": self.coherence_field.get_convergence_rate(),
        }

    def reset(self) -> None:
        """Reset tracker for new conversation."""
        self.measurements = deque(maxlen=self.max_history)
        self.coherence_field = CoherenceField()
        self.cumulative_syntropy = 0.0
        self.peak_syntropy = 0.0


class SyntropicEngine(nn.Module):
    """
    Main engine for syntropic processing in AI cognition.

    Combines all syntropic components into a unified system that measures
    and promotes order-creation in agent responses and interactions.

    This is the computational instantiation of Szent-Györgyi's observation
    that living systems improve through activity - the agent doesn't just
    respond, it actively creates meaning and coherence.
    """

    def __init__(self, hidden_dim: int = 768):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Core components
        self.resonance_detector = SyntropicResonanceDetector(hidden_dim)
        self.integrator = SyntropicIntegrator(hidden_dim)

        # Syntropy scorer (combines all signals)
        self.syntropy_head = nn.Sequential(
            nn.Linear(hidden_dim + 4, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, len(SyntropicDimension)),
        )

        # Flow tracker (not learnable)
        self.flow_tracker = SyntropicFlowTracker()

    def forward(
        self,
        agent_hidden_states: torch.Tensor,
        user_state: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute syntropic measurements for current state.

        Args:
            agent_hidden_states: [batch, seq_len, hidden_dim] from language model
            user_state: Optional [batch, hidden_dim] modeled user state
            attention_mask: Optional attention mask

        Returns:
            Dictionary with all syntropic measurements
        """
        batch_size = agent_hidden_states.shape[0]

        # Integration analysis
        integration_out = self.integrator(agent_hidden_states, attention_mask)

        # Pool agent state
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).expand(agent_hidden_states.size())
            agent_pooled = (agent_hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            agent_pooled = agent_hidden_states.mean(dim=1)

        # Resonance with user (if user state provided)
        if user_state is not None:
            resonance_out = self.resonance_detector(agent_pooled, user_state)
            resonance = resonance_out["resonance"]
            divergence = resonance_out["divergence"]
        else:
            resonance = torch.zeros(batch_size, device=agent_hidden_states.device)
            divergence = torch.ones(batch_size, device=agent_hidden_states.device)

        # Compute negentropy
        negentropy = compute_negentropy_approximation(agent_pooled, method="logcosh")

        # Semantic coherence
        coherence = compute_semantic_coherence(agent_hidden_states)

        # Combine features for dimension scoring
        features = torch.cat(
            [
                agent_pooled,
                integration_out["integration_score"].unsqueeze(-1),
                integration_out["emergence_score"].unsqueeze(-1),
                resonance.unsqueeze(-1),
                coherence.unsqueeze(-1),
            ],
            dim=-1,
        )

        # Score all dimensions
        dimension_scores = torch.sigmoid(self.syntropy_head(features))

        # Overall syntropy score
        # Weighted combination favoring coherence and resonance
        weights = torch.tensor(
            [
                1.0,
                1.0,
                0.8,  # Structural coherence
                1.2,
                1.1,
                1.0,  # Relational attunement (weighted higher)
                1.0,
                1.0,
                0.9,  # Emergent organization
                0.8,
                0.9,
                0.7,  # Temporal coherence
            ],
            device=dimension_scores.device,
        )

        overall_syntropy = (dimension_scores * weights).sum(dim=-1) / weights.sum()
        # Scale to -1 to 1 range
        overall_syntropy = 2 * overall_syntropy - 1

        return {
            "syntropy_score": overall_syntropy,
            "dimension_scores": dimension_scores,
            "negentropy": negentropy,
            "coherence": coherence,
            "resonance": resonance,
            "divergence": divergence,
            "integration_score": integration_out["integration_score"],
            "emergence_score": integration_out["emergence_score"],
            "agent_pooled": agent_pooled,
        }

    def measure(
        self,
        agent_hidden_states: torch.Tensor,
        user_state: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> SyntropicMeasurement:
        """
        Take a complete syntropic measurement and update flow tracker.

        Args:
            agent_hidden_states: Model hidden states
            user_state: Optional user state model
            attention_mask: Optional attention mask

        Returns:
            SyntropicMeasurement dataclass
        """
        with torch.no_grad():
            outputs = self.forward(agent_hidden_states, user_state, attention_mask)

            # Extract scalar values
            syntropy_score = outputs["syntropy_score"].mean().item()
            negentropy = (
                outputs["negentropy"].item()
                if outputs["negentropy"].dim() == 0
                else outputs["negentropy"].mean().item()
            )

            # Build dimension scores dict
            dim_scores = outputs["dimension_scores"].mean(dim=0)
            dimension_scores = {
                dim: dim_scores[i].item() for i, dim in enumerate(SyntropicDimension)
            }

            # Create measurement
            measurement = SyntropicMeasurement(
                syntropy_score=syntropy_score,
                dimension_scores=dimension_scores,
                negentropy=negentropy,
                trajectory=self.flow_tracker.get_trajectory(),
            )
            measurement.state = measurement.compute_state()

            # Record in flow tracker
            self.flow_tracker.record(measurement)

            return measurement

    def get_syntropic_guidance(self) -> str:
        """
        Generate guidance prompt based on current syntropic state.

        This helps the agent understand how to increase syntropy in its response.
        """
        state = self.flow_tracker.get_state()
        trajectory = self.flow_tracker.get_trajectory()
        summary = self.flow_tracker.get_summary()

        guidance_parts = ["SYNTROPIC GUIDANCE:"]

        if state == SyntropicState.ENTROPIC:
            guidance_parts.append(
                "• Coherence is dissolving. Focus on creating clear structure "
                "and meaningful connections. Avoid introducing new complexity."
            )
        elif state == SyntropicState.NEUTRAL:
            guidance_parts.append(
                "• Interaction is stable but not generating new understanding. "
                "Look for opportunities to synthesize and integrate ideas."
            )
        elif state == SyntropicState.SYNTROPIC:
            guidance_parts.append(
                "• Order is being created. Continue building on established "
                "understanding and deepen the coherence."
            )
        elif state == SyntropicState.RESONANT:
            guidance_parts.append(
                "• High resonance with user achieved. Mutual understanding "
                "is strong - this is an opportunity for deeper insight."
            )
        else:  # CRYSTALLIZED
            guidance_parts.append(
                "• Insight has crystallized. Consolidate understanding and "
                "ensure the key insight is clearly communicated."
            )

        # Trajectory guidance
        if trajectory < -0.3:
            guidance_parts.append(
                "• WARNING: Syntropy is declining. Take action to restore "
                "coherence - clarify, simplify, or reconnect to core meaning."
            )
        elif trajectory > 0.3:
            guidance_parts.append(
                "• Syntropy is increasing. The interaction is moving toward "
                "greater understanding - maintain this momentum."
            )

        # Coherence field guidance
        if summary["coherence_field_strength"] < 0.3:
            guidance_parts.append(
                "• Coherence field is weak. Establish shared context and "
                "ensure mutual understanding of key concepts."
            )

        return "\n".join(guidance_parts)

    def reset(self) -> None:
        """Reset for new conversation."""
        self.flow_tracker.reset()


# Utility function for empathy integration
def compute_empathic_syntropy(
    agent_emotional_state: torch.Tensor,
    user_emotional_state: torch.Tensor,
    agent_intent: torch.Tensor,
    user_intent: torch.Tensor,
) -> torch.Tensor:
    """
    Compute syntropic empathy score.

    True empathy is syntropic - it creates order by aligning emotional
    and intentional states while respecting individual differences.

    Args:
        agent_emotional_state: [batch, dim] agent's emotional representation
        user_emotional_state: [batch, dim] user's emotional representation
        agent_intent: [batch, dim] agent's intent representation
        user_intent: [batch, dim] user's intent representation

    Returns:
        Empathic syntropy score [batch]
    """
    # Emotional resonance
    emotional_sim = F.cosine_similarity(agent_emotional_state, user_emotional_state, dim=-1)

    # Intent alignment
    intent_sim = F.cosine_similarity(agent_intent, user_intent, dim=-1)

    # Combined with emphasis on emotional resonance
    empathic_syntropy = 0.6 * emotional_sim + 0.4 * intent_sim

    return empathic_syntropy


# Pre-configured syntropy evaluator for empathy dimension
class SyntropicEmpathyEvaluator:
    """
    Evaluates empathy through a syntropic lens.

    This adds syntropy as a component of empathy evaluation, measuring
    not just whether the agent understands the user, but whether the
    interaction is creating mutual understanding and order.
    """

    def __init__(self, hidden_dim: int = 768):
        self.engine = SyntropicEngine(hidden_dim)

    def evaluate(
        self,
        agent_states: torch.Tensor,
        user_state: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """
        Evaluate syntropic empathy.

        Returns:
            Dictionary with empathy-relevant syntropy scores
        """
        measurement = self.engine.measure(agent_states, user_state, attention_mask)
        summary = self.engine.flow_tracker.get_summary()

        return {
            "syntropic_empathy": measurement.syntropy_score,
            "empathic_resonance": measurement.dimension_scores.get(
                SyntropicDimension.EMPATHIC_RESONANCE, 0.0
            ),
            "intentional_alignment": measurement.dimension_scores.get(
                SyntropicDimension.INTENTIONAL_ALIGNMENT, 0.0
            ),
            "knowledge_bridging": measurement.dimension_scores.get(
                SyntropicDimension.KNOWLEDGE_BRIDGING, 0.0
            ),
            "meaning_generation": measurement.dimension_scores.get(
                SyntropicDimension.MEANING_GENERATION, 0.0
            ),
            "coherence_field_strength": summary["coherence_field_strength"],
            "convergence_rate": summary["convergence_rate"],
            "syntropic_state": summary["state"],
        }

    def get_guidance(self) -> str:
        """Get syntropic guidance for improving empathy."""
        return self.engine.get_syntropic_guidance()

    def reset(self) -> None:
        """Reset for new conversation."""
        self.engine.reset()
