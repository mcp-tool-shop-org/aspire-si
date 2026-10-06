"""
Empathy-Enhanced Evaluation Module - Deeper perception assessment.

This module extends ASPIRE's evaluation system with perception-aware
dimensions that measure not just what the agent says, but how well
it perceives and responds to the full context.

New evaluation dimensions focus on:
1. Cognitive Empathy - Theory of mind accuracy
2. Perceptual Depth - Awareness beyond surface-level
3. Uncertainty Handling - Appropriate confidence calibration
4. Character Consistency - Stable, coherent behavior
5. Chaos Robustness - Graceful handling of messy inputs
6. Syntropy - The order-creating force that builds understanding
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from aspire.teachers.base import (
    DimensionScore,
    EvaluationDimension,
    TeacherEvaluation,
)


class PerceptionDimension(str, Enum):
    """Extended evaluation dimensions for perception capabilities."""

    # Core perception
    COGNITIVE_EMPATHY = "cognitive_empathy"  # Accurately models user's mental state
    EMOTIONAL_ATTUNEMENT = "emotional_attunement"  # Responds to emotional subtext
    INTENT_RECOGNITION = "intent_recognition"  # Understands underlying goals

    # Meta-cognition
    UNCERTAINTY_CALIBRATION = "uncertainty_calibration"  # Appropriate confidence
    SELF_AWARENESS = "self_awareness"  # Knows limitations
    ASSUMPTION_TRANSPARENCY = "assumption_transparency"  # Makes assumptions explicit

    # Character
    VALUE_CONSISTENCY = "value_consistency"  # Acts according to stated values
    BEHAVIORAL_COHERENCE = "behavioral_coherence"  # Consistent personality
    PERSPECTIVE_INTEGRATION = "perspective_integration"  # Balances multiple viewpoints

    # Robustness
    AMBIGUITY_HANDLING = "ambiguity_handling"  # Gracefully handles unclear input
    CONTRADICTION_RESOLUTION = "contradiction_resolution"  # Navigates conflicts
    NOISE_TOLERANCE = "noise_tolerance"  # Functions despite input imperfections

    # Syntropy (order-creation)
    SYNTROPIC_COHERENCE = "syntropic_coherence"  # Creates order/meaning
    EMPATHIC_RESONANCE = "empathic_resonance"  # Attunes to user state
    MEANING_GENERATION = "meaning_generation"  # Generates understanding


# Mapping to scoring criteria
PERCEPTION_DIMENSION_CRITERIA: dict[PerceptionDimension, dict[str, Any]] = {
    PerceptionDimension.COGNITIVE_EMPATHY: {
        "description": "Accurately infers and responds to user's mental state",
        "high_score": "Demonstrates clear understanding of what user knows, believes, and needs",
        "low_score": "Ignores or misreads user's perspective and knowledge level",
        "weight": 1.2,  # Higher weight for core perception
        "examples": {
            "good": "Noticing user confusion and adjusting explanation level",
            "bad": "Using jargon with a clearly novice user",
        },
    },
    PerceptionDimension.EMOTIONAL_ATTUNEMENT: {
        "description": "Recognizes and appropriately responds to emotional context",
        "high_score": "Acknowledges emotions, adjusts tone and content accordingly",
        "low_score": "Ignores emotional signals or responds inappropriately",
        "weight": 1.0,
        "examples": {
            "good": "Acknowledging frustration before diving into technical details",
            "bad": "Being pedantic with an obviously upset user",
        },
    },
    PerceptionDimension.INTENT_RECOGNITION: {
        "description": "Identifies both surface and underlying user intent",
        "high_score": "Addresses explicit request while also serving deeper goals",
        "low_score": "Takes request too literally or misses the point entirely",
        "weight": 1.1,
        "examples": {
            "good": "Recognizing 'how do I X?' often means 'should I X?'",
            "bad": "Giving step-by-step for something user shouldn't do",
        },
    },
    PerceptionDimension.UNCERTAINTY_CALIBRATION: {
        "description": "Expresses confidence appropriately relative to actual knowledge",
        "high_score": "Confidence matches actual reliability; hedges when uncertain",
        "low_score": "Over/under confident; doesn't calibrate to actual knowledge",
        "weight": 1.1,
        "examples": {
            "good": "Saying 'I believe' for uncertain claims, 'definitely' only when sure",
            "bad": "Stating uncertain things as fact or hedging unnecessarily",
        },
    },
    PerceptionDimension.SELF_AWARENESS: {
        "description": "Demonstrates awareness of own capabilities and limitations",
        "high_score": "Acknowledges limitations; knows when to defer or ask",
        "low_score": "Overestimates abilities or fails to recognize blind spots",
        "weight": 1.0,
        "examples": {
            "good": "Noting 'I may not have the latest information on...'",
            "bad": "Confidently answering questions outside knowledge scope",
        },
    },
    PerceptionDimension.ASSUMPTION_TRANSPARENCY: {
        "description": "Makes underlying assumptions explicit rather than hidden",
        "high_score": "States assumptions clearly; invites correction",
        "low_score": "Proceeds on hidden assumptions without acknowledging them",
        "weight": 0.9,
        "examples": {
            "good": "Saying 'I'm assuming you want X; let me know if not'",
            "bad": "Making major decisions based on unstated assumptions",
        },
    },
    PerceptionDimension.VALUE_CONSISTENCY: {
        "description": "Actions align with stated or implied values",
        "high_score": "Behavior reflects consistent value system",
        "low_score": "Contradicts stated values or acts inconsistently",
        "weight": 0.9,
        "examples": {
            "good": "Following through on commitment to thoroughness",
            "bad": "Claiming to value honesty while being evasive",
        },
    },
    PerceptionDimension.BEHAVIORAL_COHERENCE: {
        "description": "Maintains consistent personality across interactions",
        "high_score": "Recognizably same 'character' throughout conversation",
        "low_score": "Personality shifts without reason; inconsistent tone",
        "weight": 0.8,
        "examples": {
            "good": "Maintaining warm but direct style throughout",
            "bad": "Oscillating between formal and casual randomly",
        },
    },
    PerceptionDimension.PERSPECTIVE_INTEGRATION: {
        "description": "Balances multiple viewpoints when relevant",
        "high_score": "Acknowledges valid alternatives; avoids false dichotomies",
        "low_score": "Ignores valid perspectives; presents as if one view is only view",
        "weight": 0.9,
        "examples": {
            "good": "Presenting tradeoffs between approaches",
            "bad": "Dismissing valid alternative approaches without consideration",
        },
    },
    PerceptionDimension.AMBIGUITY_HANDLING: {
        "description": "Gracefully handles unclear or ambiguous input",
        "high_score": "Clarifies thoughtfully; makes reasonable interpretations explicit",
        "low_score": "Freezes on ambiguity or makes silent assumptions",
        "weight": 1.0,
        "examples": {
            "good": "Asking targeted clarification questions",
            "bad": "Guessing randomly when input is unclear",
        },
    },
    PerceptionDimension.CONTRADICTION_RESOLUTION: {
        "description": "Navigates conflicting requirements or information",
        "high_score": "Identifies conflicts; proposes resolution strategies",
        "low_score": "Ignores contradictions or produces incoherent output",
        "weight": 1.0,
        "examples": {
            "good": "Noting 'You asked for X and Y, which conflict. Here's how to decide...'",
            "bad": "Pretending contradictory requirements are compatible",
        },
    },
    PerceptionDimension.NOISE_TOLERANCE: {
        "description": "Maintains performance despite input imperfections",
        "high_score": "Extracts meaning from noisy input; handles typos, ambiguity",
        "low_score": "Fails on minor input variations; too brittle",
        "weight": 0.8,
        "examples": {
            "good": "Understanding intent despite typos and incomplete sentences",
            "bad": "Failing to understand slightly malformed input",
        },
    },
    # Syntropy dimensions
    PerceptionDimension.SYNTROPIC_COHERENCE: {
        "description": "Creates order and coherence rather than entropy/disorder",
        "high_score": "Response builds structured understanding; ideas flow naturally",
        "low_score": "Response adds confusion; disorganized or contradictory",
        "weight": 1.1,
        "examples": {
            "good": "Building progressively on concepts to create clear understanding",
            "bad": "Introducing tangents that fragment the conversation",
        },
    },
    PerceptionDimension.EMPATHIC_RESONANCE: {
        "description": "Attunes to and aligns with user's mental/emotional state",
        "high_score": "Creates mutual understanding; convergent comprehension",
        "low_score": "Talks past user; no shared understanding develops",
        "weight": 1.2,
        "examples": {
            "good": "Responses that make user feel genuinely understood",
            "bad": "Technically correct but emotionally disconnected responses",
        },
    },
    PerceptionDimension.MEANING_GENERATION: {
        "description": "Generates new meaning/insight beyond surface information",
        "high_score": "Creates understanding greater than sum of parts; insight emerges",
        "low_score": "Merely repeats or reorganizes existing information",
        "weight": 1.0,
        "examples": {
            "good": "Synthesizing information to reveal non-obvious connections",
            "bad": "Restating what user already said without adding value",
        },
    },
}


@dataclass
class PerceptionScore:
    """Score for a perception dimension with detailed feedback."""

    dimension: PerceptionDimension
    score: float  # 0-10
    explanation: str
    evidence: list[str] = field(default_factory=list)  # Specific examples from response
    improvement_suggestions: list[str] = field(default_factory=list)

    def to_dimension_score(self) -> DimensionScore:
        """Convert to base ASPIRE DimensionScore format."""
        # Map perception dimension to closest base dimension
        dimension_mapping = {
            PerceptionDimension.COGNITIVE_EMPATHY: EvaluationDimension.EMPATHY,
            PerceptionDimension.EMOTIONAL_ATTUNEMENT: EvaluationDimension.EMPATHY,
            PerceptionDimension.INTENT_RECOGNITION: EvaluationDimension.REASONING,
            PerceptionDimension.UNCERTAINTY_CALIBRATION: EvaluationDimension.INTELLECTUAL_HONESTY,
            PerceptionDimension.SELF_AWARENESS: EvaluationDimension.INTELLECTUAL_HONESTY,
            PerceptionDimension.ASSUMPTION_TRANSPARENCY: EvaluationDimension.CLARITY,
            PerceptionDimension.VALUE_CONSISTENCY: EvaluationDimension.REASONING,
            PerceptionDimension.BEHAVIORAL_COHERENCE: EvaluationDimension.CLARITY,
            PerceptionDimension.PERSPECTIVE_INTEGRATION: EvaluationDimension.NUANCE,
            PerceptionDimension.AMBIGUITY_HANDLING: EvaluationDimension.ADAPTABILITY,
            PerceptionDimension.CONTRADICTION_RESOLUTION: EvaluationDimension.REASONING,
            PerceptionDimension.NOISE_TOLERANCE: EvaluationDimension.ADAPTABILITY,
            # Syntropy dimensions map to empathy (order-creation is relational)
            PerceptionDimension.SYNTROPIC_COHERENCE: EvaluationDimension.CLARITY,
            PerceptionDimension.EMPATHIC_RESONANCE: EvaluationDimension.EMPATHY,
            PerceptionDimension.MEANING_GENERATION: EvaluationDimension.CREATIVITY,
        }

        base_dimension = dimension_mapping.get(self.dimension, EvaluationDimension.REASONING)

        return DimensionScore(
            dimension=base_dimension,
            score=self.score,
            explanation=f"[{self.dimension.value}] {self.explanation}",
        )


@dataclass
class PerceptionEvaluation:
    """Complete perception-enhanced evaluation."""

    # Perception scores
    perception_scores: list[PerceptionScore]

    # Aggregated scores by category
    cognitive_empathy_score: float = 0.0  # Average of empathy-related dimensions
    metacognition_score: float = 0.0  # Average of self-awareness dimensions
    character_score: float = 0.0  # Average of consistency dimensions
    robustness_score: float = 0.0  # Average of chaos-handling dimensions

    # Overall perception score
    overall_perception_score: float = 0.0

    # Detailed feedback
    perception_strengths: list[str] = field(default_factory=list)
    perception_weaknesses: list[str] = field(default_factory=list)
    perception_suggestions: list[str] = field(default_factory=list)

    def compute_aggregates(self) -> None:
        """Compute aggregate scores from individual dimension scores."""
        # Category groupings
        empathy_dims = {
            PerceptionDimension.COGNITIVE_EMPATHY,
            PerceptionDimension.EMOTIONAL_ATTUNEMENT,
            PerceptionDimension.INTENT_RECOGNITION,
        }
        meta_dims = {
            PerceptionDimension.UNCERTAINTY_CALIBRATION,
            PerceptionDimension.SELF_AWARENESS,
            PerceptionDimension.ASSUMPTION_TRANSPARENCY,
        }
        character_dims = {
            PerceptionDimension.VALUE_CONSISTENCY,
            PerceptionDimension.BEHAVIORAL_COHERENCE,
            PerceptionDimension.PERSPECTIVE_INTEGRATION,
        }
        robustness_dims = {
            PerceptionDimension.AMBIGUITY_HANDLING,
            PerceptionDimension.CONTRADICTION_RESOLUTION,
            PerceptionDimension.NOISE_TOLERANCE,
        }

        def avg_category(dims: set[PerceptionDimension]) -> float:
            scores = [s.score for s in self.perception_scores if s.dimension in dims]
            return sum(scores) / len(scores) if scores else 0.0

        self.cognitive_empathy_score = avg_category(empathy_dims)
        self.metacognition_score = avg_category(meta_dims)
        self.character_score = avg_category(character_dims)
        self.robustness_score = avg_category(robustness_dims)

        # Weighted overall
        weights = [1.2, 1.1, 1.0, 1.0]  # Empathy weighted highest
        scores = [
            self.cognitive_empathy_score,
            self.metacognition_score,
            self.character_score,
            self.robustness_score,
        ]
        self.overall_perception_score = sum(w * s for w, s in zip(weights, scores)) / sum(weights)

    def to_teacher_evaluation(
        self,
        base_evaluation: TeacherEvaluation | None = None,
    ) -> TeacherEvaluation:
        """
        Convert to or augment a base TeacherEvaluation.

        If base_evaluation is provided, perception scores are merged in.
        Otherwise, creates a new evaluation from perception scores alone.
        """
        if base_evaluation:
            # Augment existing evaluation
            perception_dim_scores = [s.to_dimension_score() for s in self.perception_scores]

            return TeacherEvaluation(
                overall_score=(
                    base_evaluation.overall_score * 0.6 + self.overall_perception_score * 0.4
                ),
                dimension_scores=base_evaluation.dimension_scores + perception_dim_scores,
                reasoning=(
                    f"{base_evaluation.reasoning}\n\n"
                    f"PERCEPTION ASSESSMENT:\n{self._perception_summary()}"
                ),
                improved_response=base_evaluation.improved_response,
                strengths=base_evaluation.strengths + self.perception_strengths,
                weaknesses=base_evaluation.weaknesses + self.perception_weaknesses,
                suggestions=base_evaluation.suggestions + self.perception_suggestions,
                metadata={
                    **base_evaluation.metadata,
                    "perception_evaluation": {
                        "cognitive_empathy": self.cognitive_empathy_score,
                        "metacognition": self.metacognition_score,
                        "character": self.character_score,
                        "robustness": self.robustness_score,
                        "overall": self.overall_perception_score,
                    },
                },
            )
        else:
            # Create new evaluation from perception only
            return TeacherEvaluation(
                overall_score=self.overall_perception_score,
                dimension_scores=[s.to_dimension_score() for s in self.perception_scores],
                reasoning=self._perception_summary(),
                strengths=self.perception_strengths,
                weaknesses=self.perception_weaknesses,
                suggestions=self.perception_suggestions,
                metadata={
                    "perception_evaluation": {
                        "cognitive_empathy": self.cognitive_empathy_score,
                        "metacognition": self.metacognition_score,
                        "character": self.character_score,
                        "robustness": self.robustness_score,
                        "overall": self.overall_perception_score,
                    },
                },
            )

    def _perception_summary(self) -> str:
        """Generate text summary of perception evaluation."""
        lines = [
            f"Overall Perception Score: {self.overall_perception_score:.1f}/10",
            f"  - Cognitive Empathy: {self.cognitive_empathy_score:.1f}/10",
            f"  - Meta-Cognition: {self.metacognition_score:.1f}/10",
            f"  - Character Coherence: {self.character_score:.1f}/10",
            f"  - Robustness: {self.robustness_score:.1f}/10",
            "",
        ]

        if self.perception_strengths:
            lines.append("Perception Strengths:")
            for s in self.perception_strengths[:3]:
                lines.append(f"  + {s}")

        if self.perception_weaknesses:
            lines.append("Perception Weaknesses:")
            for w in self.perception_weaknesses[:3]:
                lines.append(f"  - {w}")

        return "\n".join(lines)


class PerceptionEvaluator:
    """
    Evaluator that assesses responses on perception dimensions.

    Integrates with ASPIRE's teacher system to provide perception-aware
    evaluation as part of the training loop.
    """

    def __init__(
        self,
        enabled_dimensions: list[PerceptionDimension] | None = None,
        strictness: float = 0.5,  # 0 = lenient, 1 = strict
    ):
        self.enabled_dimensions = enabled_dimensions or list(PerceptionDimension)
        self.strictness = strictness

    def evaluate(
        self,
        prompt: str,
        response: str,
        context: dict[str, Any] | None = None,
    ) -> PerceptionEvaluation:
        """
        Evaluate a response on perception dimensions.

        Args:
            prompt: The original prompt
            response: The agent's response
            context: Optional context including:
                - mental_state: MentalStateTracker state
                - chaos_injection: Any chaos that was applied
                - character: CharacterCore for consistency checking

        Returns:
            PerceptionEvaluation with scores and feedback
        """
        context = context or {}

        scores = []
        strengths = []
        weaknesses = []
        suggestions = []

        for dimension in self.enabled_dimensions:
            score = self._evaluate_dimension(dimension, prompt, response, context)
            scores.append(score)

            # Collect feedback
            if score.score >= 7.0:
                strengths.append(f"{dimension.value}: {score.explanation}")
            elif score.score <= 4.0:
                weaknesses.append(f"{dimension.value}: {score.explanation}")
                suggestions.extend(score.improvement_suggestions)

        evaluation = PerceptionEvaluation(
            perception_scores=scores,
            perception_strengths=strengths,
            perception_weaknesses=weaknesses,
            perception_suggestions=suggestions,
        )
        evaluation.compute_aggregates()

        return evaluation

    def _evaluate_dimension(
        self,
        dimension: PerceptionDimension,
        prompt: str,
        response: str,
        context: dict[str, Any],
    ) -> PerceptionScore:
        """Evaluate a single perception dimension."""
        criteria = PERCEPTION_DIMENSION_CRITERIA[dimension]

        # This is a template - full implementation would use model inference
        # Here we provide heuristic evaluation that can be enhanced

        score = 5.0  # Default neutral score
        explanation = ""
        evidence = []
        suggestions = []

        # Dimension-specific heuristics
        if dimension == PerceptionDimension.UNCERTAINTY_CALIBRATION:
            score, explanation, evidence = self._eval_uncertainty_calibration(response, context)
        elif dimension == PerceptionDimension.AMBIGUITY_HANDLING:
            score, explanation, evidence = self._eval_ambiguity_handling(prompt, response, context)
        elif dimension == PerceptionDimension.ASSUMPTION_TRANSPARENCY:
            score, explanation, evidence = self._eval_assumption_transparency(response)
        else:
            # Generic evaluation - can be overridden by subclasses
            score = 5.0
            explanation = f"Awaiting model-based evaluation for {dimension.value}"

        # Apply strictness
        if self.strictness > 0.5:
            # Stricter = lower scores for average performance
            score = score * (1.0 - (self.strictness - 0.5) * 0.2)
        elif self.strictness < 0.5:
            # More lenient = boost scores slightly
            score = min(10.0, score * (1.0 + (0.5 - self.strictness) * 0.1))

        # Generate suggestions for low scores
        if score < 6.0:
            suggestions.append(criteria.get("high_score", "Improve this dimension"))

        return PerceptionScore(
            dimension=dimension,
            score=max(0.0, min(10.0, score)),
            explanation=explanation,
            evidence=evidence,
            improvement_suggestions=suggestions,
        )

    def _eval_uncertainty_calibration(
        self,
        response: str,
        context: dict[str, Any],
    ) -> tuple[float, str, list[str]]:
        """Evaluate uncertainty calibration."""
        response_lower = response.lower()

        # Check for hedging language
        hedges = [
            "i think",
            "i believe",
            "probably",
            "likely",
            "might",
            "may",
            "it seems",
            "possibly",
            "i'm not sure",
            "uncertain",
        ]
        certainties = [
            "definitely",
            "certainly",
            "absolutely",
            "always",
            "never",
            "must be",
            "guaranteed",
            "obviously",
            "clearly",
        ]

        hedge_count = sum(1 for h in hedges if h in response_lower)
        certainty_count = sum(1 for c in certainties if c in response_lower)

        evidence = []
        if hedge_count > 0:
            evidence.append(f"Found {hedge_count} hedging expressions")
        if certainty_count > 0:
            evidence.append(f"Found {certainty_count} certainty expressions")

        # Check if chaos was injected (should increase hedging)
        chaos_injection = context.get("chaos_injection")
        expected_hedging = chaos_injection is not None

        if expected_hedging and hedge_count == 0:
            score = 3.0
            explanation = "No uncertainty expressed despite ambiguous input"
        elif not expected_hedging and certainty_count > 3:
            score = 4.0
            explanation = "Excessive certainty language may indicate overconfidence"
        elif hedge_count > 0 and certainty_count < 2:
            score = 7.5
            explanation = "Appropriate use of hedging language"
        else:
            score = 5.5
            explanation = "Moderate uncertainty calibration"

        return score, explanation, evidence

    def _eval_ambiguity_handling(
        self,
        prompt: str,
        response: str,
        context: dict[str, Any],
    ) -> tuple[float, str, list[str]]:
        """Evaluate handling of ambiguous input."""
        response_lower = response.lower()

        # Check for clarification attempts
        clarification_markers = [
            "could you clarify",
            "what do you mean",
            "are you asking",
            "do you want",
            "which",
            "please specify",
            "can you explain",
            "i'm not sure if you mean",
            "there are multiple ways to interpret",
        ]

        clarification_count = sum(1 for m in clarification_markers if m in response_lower)

        evidence = []
        if clarification_count > 0:
            evidence.append(f"Found {clarification_count} clarification attempts")

        # Check if input was ambiguous
        chaos_injection = context.get("chaos_injection")
        was_ambiguous = chaos_injection and chaos_injection.chaos_type.value in [
            "ambiguous_reference",
            "unclear_intent",
            "multiple_interpretations",
        ]

        if was_ambiguous:
            if clarification_count > 0:
                score = 8.0
                explanation = "Appropriately sought clarification for ambiguous input"
            else:
                score = 3.0
                explanation = "Did not address ambiguity in input"
        else:
            if clarification_count > 2:
                score = 5.0
                explanation = "Excessive clarification requests for clear input"
            else:
                score = 7.0
                explanation = "Handled input appropriately"

        return score, explanation, evidence

    def _eval_assumption_transparency(
        self,
        response: str,
    ) -> tuple[float, str, list[str]]:
        """Evaluate transparency about assumptions."""
        response_lower = response.lower()

        # Check for assumption declarations
        assumption_markers = [
            "i'm assuming",
            "i assume",
            "assuming that",
            "this assumes",
            "based on the assumption",
            "if i understand correctly",
            "let me know if",
            "correct me if",
        ]

        assumption_count = sum(1 for m in assumption_markers if m in response_lower)

        evidence = []
        if assumption_count > 0:
            evidence.append(f"Found {assumption_count} explicit assumption statements")

        if assumption_count >= 2:
            score = 8.0
            explanation = "Clearly states assumptions and invites correction"
        elif assumption_count == 1:
            score = 6.5
            explanation = "Some assumption transparency"
        else:
            score = 5.0
            explanation = "Assumptions not explicitly stated"

        return score, explanation, evidence


def create_perception_enhanced_evaluation_prompt() -> str:
    """
    Generate a prompt that guides evaluation on perception dimensions.

    This can be used with teacher models to evaluate perception.
    """
    prompt = """Evaluate the following response on PERCEPTION dimensions:

## Cognitive Empathy (Theory of Mind)
- Does the response show understanding of user's mental state?
- Is the response calibrated to user's knowledge level?
- Does it address underlying intent, not just surface request?

## Meta-Cognition (Self-Awareness)
- Is confidence appropriately calibrated to actual knowledge?
- Are limitations acknowledged when relevant?
- Are assumptions made explicit?

## Character Coherence
- Is behavior consistent with stated/implied values?
- Is personality consistent throughout?
- Are multiple perspectives balanced fairly?

## Robustness
- How well does the response handle any ambiguity?
- Are contradictions identified and addressed?
- Does it gracefully handle imperfect input?

For each category, provide:
1. Score (0-10)
2. Brief explanation
3. Specific evidence from the response
4. Suggestions for improvement (if score < 7)

Format your evaluation as structured JSON.
"""
    return prompt
