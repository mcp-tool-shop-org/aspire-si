"""
Meta-Cognition Module - Uncertainty tracking and self-reflection.

Meta-cognition is "thinking about thinking" - the ability to:
1. Monitor one's own cognitive processes
2. Estimate confidence and uncertainty accurately
3. Recognize the limits of one's knowledge
4. Reflect on and improve one's reasoning

This is crucial for agents because overconfident agents are dangerous,
while appropriately uncertain agents know when to ask for help or hedge.

Architecture:
┌─────────────────────────────────────────────────────────────┐
│                    Meta-Cognition Stack                      │
├─────────────────────────────────────────────────────────────┤
│  Layer 3: Reflective Loop                                   │
│  "Am I being consistent? What could I be missing?"          │
├─────────────────────────────────────────────────────────────┤
│  Layer 2: Confidence Calibrator                             │
│  "How confident should I actually be?"                      │
├─────────────────────────────────────────────────────────────┤
│  Layer 1: Uncertainty Estimator                             │
│  "What am I uncertain about and why?"                       │
└─────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class UncertaintyType(str, Enum):
    """Types of uncertainty the agent can experience."""

    # Epistemic uncertainty (knowledge gaps)
    FACTUAL = "factual"  # Don't know the facts
    DEFINITIONAL = "definitional"  # Unclear what terms mean
    PROCEDURAL = "procedural"  # Don't know how to do something

    # Aleatoric uncertainty (inherent randomness)
    STOCHASTIC = "stochastic"  # Outcome is genuinely random
    CONTEXTUAL = "contextual"  # Depends on unknown context

    # Model uncertainty
    OUT_OF_DISTRIBUTION = "out_of_distribution"  # Input unlike training data
    CONFLICTING_EVIDENCE = "conflicting_evidence"  # Multiple valid interpretations

    # Meta-uncertainty
    UNKNOWN_UNKNOWNS = "unknown_unknowns"  # Don't know what I don't know


class ConfidenceLevel(str, Enum):
    """Calibrated confidence levels with associated behaviors."""

    CERTAIN = "certain"  # >95% - State directly
    HIGH = "high"  # 80-95% - State with minor hedging
    MODERATE = "moderate"  # 60-80% - Acknowledge uncertainty
    LOW = "low"  # 40-60% - Strong hedging, offer alternatives
    VERY_LOW = "very_low"  # 20-40% - Express significant doubt
    UNCERTAIN = "uncertain"  # <20% - Decline or ask for help


@dataclass
class UncertaintyEstimate:
    """A structured uncertainty estimate for a claim or response."""

    # What we're uncertain about
    subject: str

    # Type and magnitude
    uncertainty_type: UncertaintyType
    magnitude: float  # 0-1, where 1 is maximum uncertainty

    # Source of uncertainty
    source: str  # Why are we uncertain?

    # What would reduce uncertainty?
    reducible_by: list[str] = field(default_factory=list)

    # Is this uncertainty reducible at all?
    is_reducible: bool = True

    timestamp: float = field(default_factory=time.time)

    def to_natural_language(self) -> str:
        """Convert to natural language expression."""
        if self.magnitude < 0.2:
            prefix = "I'm fairly confident about"
        elif self.magnitude < 0.4:
            prefix = "I'm somewhat uncertain about"
        elif self.magnitude < 0.6:
            prefix = "I have moderate uncertainty about"
        elif self.magnitude < 0.8:
            prefix = "I'm quite uncertain about"
        else:
            prefix = "I'm very uncertain about"

        result = f"{prefix} {self.subject}. {self.source}"

        if self.reducible_by:
            result += f" This could be clarified by: {', '.join(self.reducible_by[:3])}"

        return result


@dataclass
class ReflectiveInsight:
    """An insight from reflective analysis."""

    category: str  # What type of insight
    observation: str  # What was observed
    implication: str  # What it means
    action: str | None = None  # What to do about it

    confidence: float = 0.5
    timestamp: float = field(default_factory=time.time)


@dataclass
class CalibrationRecord:
    """Record for tracking calibration over time."""

    predicted_confidence: float
    actual_correctness: float  # 1 if correct, 0 if wrong
    domain: str
    timestamp: float = field(default_factory=time.time)


class UncertaintyEstimator(nn.Module):
    """
    Neural module for estimating uncertainty from hidden states.

    This learns to predict when the model is likely to be wrong,
    enabling better-calibrated confidence estimates.
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_uncertainty_types: int = len(UncertaintyType),
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Overall uncertainty head
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Uncertainty type classifier
        self.type_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_uncertainty_types),
        )

        # OOD detector (out-of-distribution)
        self.ood_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )

        # Track seen distributions for OOD detection
        self.register_buffer(
            "mean_hidden",
            torch.zeros(hidden_dim),
        )
        self.register_buffer(
            "var_hidden",
            torch.ones(hidden_dim),
        )
        self.register_buffer(
            "num_samples",
            torch.tensor(0.0),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Estimate uncertainty from hidden states.

        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            attention_mask: Optional mask

        Returns:
            Dictionary with:
            - overall_uncertainty: [batch, 1]
            - uncertainty_type_logits: [batch, num_types]
            - ood_score: [batch, 1]
        """
        # Pool hidden states
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).expand(hidden_states.size())
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = hidden_states.mean(dim=1)

        # Estimate uncertainty
        overall = self.uncertainty_head(pooled)
        type_logits = self.type_classifier(pooled)
        ood = self.ood_detector(pooled)

        # Also compute Mahalanobis-style OOD score
        if self.num_samples > 100:
            diff = pooled - self.mean_hidden.unsqueeze(0)
            mahal = (diff**2 / self.var_hidden.unsqueeze(0).clamp(min=1e-6)).mean(
                dim=-1, keepdim=True
            )
            # Normalize to 0-1 range
            mahal_ood = torch.sigmoid(mahal - 1.0)  # Centered at distance 1
            # Combine with learned OOD
            ood = (ood + mahal_ood) / 2

        return {
            "overall_uncertainty": overall,
            "uncertainty_type_logits": type_logits,
            "ood_score": ood,
            "pooled": pooled,
        }

    def update_distribution_stats(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> None:
        """Update running statistics for OOD detection."""
        with torch.no_grad():
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).expand(hidden_states.size())
                pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            else:
                pooled = hidden_states.mean(dim=1)

            # Running mean and variance update
            batch_mean = pooled.mean(dim=0)
            batch_var = pooled.var(dim=0)
            batch_size = pooled.size(0)

            n = self.num_samples
            new_n = n + batch_size

            # Welford's online algorithm
            delta = batch_mean - self.mean_hidden
            self.mean_hidden = self.mean_hidden + delta * batch_size / new_n

            # Update variance
            self.var_hidden = (
                n * self.var_hidden + batch_size * batch_var + n * batch_size * delta**2 / new_n
            ) / new_n

            self.num_samples = new_n

    def estimate_from_logits(
        self,
        logits: torch.Tensor,
        method: str = "entropy",
    ) -> torch.Tensor:
        """
        Estimate uncertainty directly from output logits.

        Args:
            logits: Model output logits [batch, vocab_size] or [batch, seq, vocab]
            method: "entropy", "margin", or "variance"

        Returns:
            Uncertainty score [batch] or [batch, seq]
        """
        probs = F.softmax(logits, dim=-1)

        if method == "entropy":
            # Shannon entropy
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
            # Normalize by log(vocab_size)
            max_entropy = torch.log(torch.tensor(logits.size(-1), dtype=torch.float))
            return entropy / max_entropy

        elif method == "margin":
            # Difference between top two probabilities
            sorted_probs = probs.sort(dim=-1, descending=True).values
            margin = sorted_probs[..., 0] - sorted_probs[..., 1]
            return 1.0 - margin  # Lower margin = higher uncertainty

        elif method == "variance":
            # Variance of probability distribution
            mean_prob = probs.mean(dim=-1, keepdim=True)
            variance = ((probs - mean_prob) ** 2).mean(dim=-1)
            return 1.0 - variance * logits.size(-1)  # Normalize

        else:
            raise ValueError(f"Unknown method: {method}")


class ConfidenceCalibrator:
    """
    Calibrates raw confidence scores to be more accurate.

    Uses historical performance to adjust confidence predictions,
    addressing the common problem of over/under-confidence.
    """

    def __init__(
        self,
        num_bins: int = 10,
        smoothing: float = 0.1,
    ):
        self.num_bins = num_bins
        self.smoothing = smoothing

        # Calibration data per domain
        self.calibration_records: dict[str, list[CalibrationRecord]] = {}

        # Learned calibration curves
        self.calibration_curves: dict[str, list[float]] = {}

    def record(
        self,
        predicted: float,
        actual: float,
        domain: str = "general",
    ) -> None:
        """Record a prediction for calibration learning."""
        record = CalibrationRecord(
            predicted_confidence=predicted,
            actual_correctness=actual,
            domain=domain,
        )

        if domain not in self.calibration_records:
            self.calibration_records[domain] = []

        self.calibration_records[domain].append(record)

        # Update calibration curve periodically
        if len(self.calibration_records[domain]) % 100 == 0:
            self._update_calibration_curve(domain)

    def calibrate(
        self,
        raw_confidence: float,
        domain: str = "general",
    ) -> float:
        """
        Apply calibration to a raw confidence score.

        Returns adjusted confidence that better reflects true probability.
        """
        if domain not in self.calibration_curves:
            # No calibration data - return with slight regression to mean
            return 0.5 + (raw_confidence - 0.5) * 0.8

        curve = self.calibration_curves[domain]

        # Find which bin this falls into
        bin_idx = min(int(raw_confidence * self.num_bins), self.num_bins - 1)

        # Linear interpolation between bin boundaries
        if bin_idx < self.num_bins - 1:
            bin_start = bin_idx / self.num_bins
            bin_end = (bin_idx + 1) / self.num_bins
            t = (raw_confidence - bin_start) / (bin_end - bin_start)

            calibrated = curve[bin_idx] * (1 - t) + curve[bin_idx + 1] * t
        else:
            calibrated = curve[bin_idx]

        return max(0.0, min(1.0, calibrated))

    def _update_calibration_curve(self, domain: str) -> None:
        """Recompute calibration curve from records."""
        records = self.calibration_records[domain]

        # Bin the predictions
        bins = [[] for _ in range(self.num_bins)]
        for record in records:
            bin_idx = min(
                int(record.predicted_confidence * self.num_bins),
                self.num_bins - 1,
            )
            bins[bin_idx].append(record.actual_correctness)

        # Compute actual accuracy per bin
        curve = []
        for i, bin_records in enumerate(bins):
            if bin_records:
                actual = sum(bin_records) / len(bin_records)
            else:
                # No data - use expected value with smoothing
                expected = (i + 0.5) / self.num_bins
                actual = expected

            # Apply smoothing toward expected
            expected = (i + 0.5) / self.num_bins
            smoothed = self.smoothing * expected + (1 - self.smoothing) * actual
            curve.append(smoothed)

        self.calibration_curves[domain] = curve

    def get_calibration_error(self, domain: str = "general") -> float:
        """Compute Expected Calibration Error (ECE) for a domain."""
        if domain not in self.calibration_records:
            return 0.5  # Maximum uncertainty

        records = self.calibration_records[domain]
        if not records:
            return 0.5

        # Bin and compute ECE
        bins = [[] for _ in range(self.num_bins)]
        for record in records:
            bin_idx = min(
                int(record.predicted_confidence * self.num_bins),
                self.num_bins - 1,
            )
            bins[bin_idx].append((record.predicted_confidence, record.actual_correctness))

        ece = 0.0
        total = len(records)

        for bin_records in bins:
            if bin_records:
                avg_conf = sum(p for p, _ in bin_records) / len(bin_records)
                avg_acc = sum(a for _, a in bin_records) / len(bin_records)
                ece += len(bin_records) / total * abs(avg_conf - avg_acc)

        return ece

    def get_confidence_level(self, calibrated_confidence: float) -> ConfidenceLevel:
        """Convert calibrated confidence to a discrete level."""
        if calibrated_confidence >= 0.95:
            return ConfidenceLevel.CERTAIN
        elif calibrated_confidence >= 0.80:
            return ConfidenceLevel.HIGH
        elif calibrated_confidence >= 0.60:
            return ConfidenceLevel.MODERATE
        elif calibrated_confidence >= 0.40:
            return ConfidenceLevel.LOW
        elif calibrated_confidence >= 0.20:
            return ConfidenceLevel.VERY_LOW
        else:
            return ConfidenceLevel.UNCERTAIN

    def get_hedging_language(self, level: ConfidenceLevel) -> str:
        """Get appropriate hedging language for confidence level."""
        hedges = {
            ConfidenceLevel.CERTAIN: "",
            ConfidenceLevel.HIGH: "I believe ",
            ConfidenceLevel.MODERATE: "I think ",
            ConfidenceLevel.LOW: "I'm not certain, but ",
            ConfidenceLevel.VERY_LOW: "I'm quite uncertain, but my best guess is ",
            ConfidenceLevel.UNCERTAIN: "I don't know for sure. If I had to guess, ",
        }
        return hedges.get(level, "")


class ReflectiveLoop:
    """
    Implements periodic self-reflection for improved reasoning.

    This module enables the agent to step back and analyze its own
    reasoning process, catching potential errors and biases.
    """

    def __init__(
        self,
        reflection_prompts: list[str] | None = None,
        max_insights: int = 100,
    ):
        self.reflection_prompts = reflection_prompts or self._default_prompts()
        self.max_insights = max_insights

        # History of insights
        self.insights: list[ReflectiveInsight] = []

        # Patterns detected
        self.detected_patterns: dict[str, int] = {}

    def _default_prompts(self) -> list[str]:
        """Default reflection prompts."""
        return [
            # Consistency checks
            "Is this response consistent with what I said earlier?",
            "Am I contradicting any of my stated values or principles?",
            # Assumption awareness
            "What assumptions am I making that might not be valid?",
            "What am I taking for granted that I should question?",
            # Perspective taking
            "How might this appear from the user's perspective?",
            "What context might I be missing?",
            # Completeness
            "What important aspects haven't I addressed?",
            "What follow-up questions might this raise?",
            # Bias detection
            "Am I favoring a particular viewpoint unfairly?",
            "Am I being appropriately uncertain vs overconfident?",
            # Quality check
            "Is this actually helpful, or just verbose?",
            "Am I being clear and precise?",
        ]

    def reflect(
        self,
        context: str,
        current_response: str,
        prompt_indices: list[int] | None = None,
    ) -> list[ReflectiveInsight]:
        """
        Perform reflection on current response.

        This generates reflection prompts that can be used to
        critique and potentially improve the response.

        Args:
            context: The conversation context
            current_response: The response being reflected upon
            prompt_indices: Specific prompts to use (or None for auto-select)

        Returns:
            List of reflective insights
        """
        insights = []

        # Select prompts
        if prompt_indices:
            prompts = [self.reflection_prompts[i] for i in prompt_indices]
        else:
            # Auto-select based on context
            prompts = self._select_relevant_prompts(context, current_response)

        # Generate reflection for each prompt
        for prompt in prompts:
            insight = self._generate_insight(prompt, context, current_response)
            if insight:
                insights.append(insight)
                self._record_insight(insight)

        return insights

    def _select_relevant_prompts(
        self,
        context: str,
        response: str,
    ) -> list[str]:
        """Select relevant reflection prompts based on content."""
        selected = []

        # Always include consistency check if context is substantial
        if len(context) > 500:
            selected.append(self.reflection_prompts[0])  # Consistency

        # Check for assumption-heavy language
        assumption_markers = ["obviously", "clearly", "of course", "everyone knows"]
        if any(marker in response.lower() for marker in assumption_markers):
            selected.append(self.reflection_prompts[2])  # Assumptions

        # Check for strong claims
        strong_claim_markers = ["always", "never", "definitely", "certainly", "must"]
        if any(marker in response.lower() for marker in strong_claim_markers):
            selected.append(self.reflection_prompts[9])  # Overconfidence check

        # Check response length
        if len(response) > 1000:
            selected.append(self.reflection_prompts[10])  # Verbosity check

        # Always include perspective taking and completeness
        selected.append(self.reflection_prompts[4])  # Perspective
        selected.append(self.reflection_prompts[6])  # Completeness

        return selected[:5]  # Max 5 prompts

    def _generate_insight(
        self,
        prompt: str,
        context: str,
        response: str,
    ) -> ReflectiveInsight | None:
        """
        Generate an insight from a reflection prompt.

        In a full implementation, this would call a model to reflect.
        Here we provide the structure for integration.
        """
        # This is a template - actual reflection requires model inference
        # The structure enables integration with the training loop

        category = self._categorize_prompt(prompt)

        return ReflectiveInsight(
            category=category,
            observation=f"Reflection prompt: {prompt}",
            implication="Requires analysis",
            action="Review response with this lens",
            confidence=0.5,
        )

    def _categorize_prompt(self, prompt: str) -> str:
        """Categorize a reflection prompt."""
        prompt_lower = prompt.lower()

        if "consistent" in prompt_lower or "contradict" in prompt_lower:
            return "consistency"
        elif "assumption" in prompt_lower or "taking for granted" in prompt_lower:
            return "assumptions"
        elif "perspective" in prompt_lower or "appear" in prompt_lower:
            return "perspective"
        elif "missing" in prompt_lower or "addressed" in prompt_lower:
            return "completeness"
        elif "bias" in prompt_lower or "unfair" in prompt_lower:
            return "bias"
        elif "helpful" in prompt_lower or "clear" in prompt_lower:
            return "quality"
        else:
            return "general"

    def _record_insight(self, insight: ReflectiveInsight) -> None:
        """Record insight for pattern detection."""
        self.insights.append(insight)

        # Track patterns
        self.detected_patterns[insight.category] = (
            self.detected_patterns.get(insight.category, 0) + 1
        )

        # Prune if over limit
        if len(self.insights) > self.max_insights:
            self.insights = self.insights[-self.max_insights :]

    def get_reflection_prompt_for_training(
        self,
        context: str,
        response: str,
    ) -> str:
        """
        Generate a full reflection prompt for model training.

        This can be used to train the model to self-reflect.
        """
        prompts = self._select_relevant_prompts(context, response)

        reflection_text = """Before finalizing this response, reflect on the following:

"""
        for i, prompt in enumerate(prompts, 1):
            reflection_text += f"{i}. {prompt}\n"

        reflection_text += """
Based on this reflection:
- What should be revised or clarified?
- What additional context or caveats are needed?
- Is the confidence level appropriate?
"""
        return reflection_text

    def get_pattern_summary(self) -> dict[str, Any]:
        """Get summary of detected reflection patterns."""
        total = sum(self.detected_patterns.values())
        if total == 0:
            return {"patterns": {}, "total_reflections": 0}

        return {
            "patterns": {
                k: {"count": v, "percentage": v / total * 100}
                for k, v in self.detected_patterns.items()
            },
            "total_reflections": total,
            "most_common": max(self.detected_patterns, key=self.detected_patterns.get)
            if self.detected_patterns
            else None,
        }


class MetaCognitionModule(nn.Module):
    """
    Combined meta-cognition module for training integration.

    Brings together uncertainty estimation, calibration, and reflection
    into a single trainable module.
    """

    def __init__(
        self,
        hidden_dim: int = 768,
    ):
        super().__init__()

        self.uncertainty_estimator = UncertaintyEstimator(hidden_dim)
        self.calibrator = ConfidenceCalibrator()
        self.reflective_loop = ReflectiveLoop()

        # Combined meta-cognition head
        self.meta_head = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim // 2),  # +3 for uncertainty features
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 4),  # [should_hedge, should_clarify, should_ask, confidence]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_logits: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass for meta-cognition.

        Args:
            hidden_states: [batch, seq, hidden_dim]
            attention_mask: Optional attention mask
            output_logits: Optional model output logits for entropy-based uncertainty

        Returns:
            Dictionary with meta-cognitive predictions
        """
        # Get uncertainty estimates
        uncertainty_out = self.uncertainty_estimator(hidden_states, attention_mask)

        # Combine features
        features = [uncertainty_out["pooled"]]

        # Add uncertainty features
        features.append(uncertainty_out["overall_uncertainty"])
        features.append(uncertainty_out["ood_score"])

        # Add logit-based uncertainty if available
        if output_logits is not None:
            entropy_uncertainty = self.uncertainty_estimator.estimate_from_logits(
                output_logits, method="entropy"
            )
            if entropy_uncertainty.dim() == 2:
                entropy_uncertainty = entropy_uncertainty.mean(dim=1, keepdim=True)
            features.append(entropy_uncertainty)
        else:
            features.append(torch.zeros_like(uncertainty_out["overall_uncertainty"]))

        # Concatenate
        combined = torch.cat(features, dim=-1)

        # Meta-cognition predictions
        meta_out = self.meta_head(combined)
        meta_out = torch.sigmoid(meta_out)

        return {
            **uncertainty_out,
            "should_hedge": meta_out[:, 0:1],
            "should_clarify": meta_out[:, 1:2],
            "should_ask_user": meta_out[:, 2:3],
            "meta_confidence": meta_out[:, 3:4],
        }

    def get_action_recommendation(
        self,
        outputs: dict[str, torch.Tensor],
    ) -> str:
        """Get natural language action recommendation from outputs."""
        with torch.no_grad():
            hedge = outputs["should_hedge"].mean().item()
            clarify = outputs["should_clarify"].mean().item()
            ask = outputs["should_ask_user"].mean().item()
            confidence = outputs["meta_confidence"].mean().item()

            recommendations = []

            if hedge > 0.6:
                recommendations.append("Add hedging language to express uncertainty")
            if clarify > 0.6:
                recommendations.append("Clarify or expand on key points")
            if ask > 0.6:
                recommendations.append("Ask the user for clarification")

            if confidence < 0.4:
                recommendations.append("Consider that this response may need revision")

            if not recommendations:
                recommendations.append("Response confidence is adequate")

            return "; ".join(recommendations)

    def generate_metacognitive_prompt(
        self,
        outputs: dict[str, torch.Tensor],
        context: str = "",
    ) -> str:
        """Generate a prompt to encourage metacognitive behavior."""
        with torch.no_grad():
            uncertainty = outputs["overall_uncertainty"].mean().item()
            ood = outputs["ood_score"].mean().item()

            prompt_parts = ["METACOGNITIVE CHECK:"]

            if uncertainty > 0.5:
                level = self.calibrator.get_confidence_level(1 - uncertainty)
                prompt_parts.append(
                    f"• Uncertainty level: {level.value}. "
                    f"Consider expressing appropriate confidence."
                )

            if ood > 0.5:
                prompt_parts.append(
                    "• This query may be unusual. "
                    "Double-check assumptions and consider asking for clarification."
                )

            # Add reflection prompts
            reflection = self.reflective_loop.get_reflection_prompt_for_training(
                context,
                "",  # Empty response since we're pre-generating
            )
            prompt_parts.append(reflection)

            return "\n".join(prompt_parts)
