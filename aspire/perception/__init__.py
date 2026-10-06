"""
ASPIRE Perception Module - Enhancing agent awareness and cognitive empathy.

This module adds four key perception capabilities:

1. Theory of Mind (ToM) - Mental state tracking and perspective-taking
2. Controlled Chaos - Adversarial training for robustness
3. Character Persistence - Stable identity and value anchoring
4. Meta-Cognition - Uncertainty tracking and self-reflection

These work together to create agents with genuine perceptual depth,
not just pattern matching but actual understanding of context, users, and self.
"""

from aspire.perception.character import (
    AutobiographicalMemory,
    BehavioralTrait,
    CharacterCore,
    MemoryEntry,
    TraitDimension,
    ValueAnchor,
    ValueType,
    create_analytical_character,
    create_compassionate_character,
    create_socratic_character,
)
from aspire.perception.controlled_chaos import (
    AdversarialScenarioGenerator,
    AmbiguityGenerator,
    ChaosConfig,
    ChaosGenerator,
    ChaosInjection,
    ChaosSeverity,
    ChaosType,
    NoiseInjector,
    apply_chaos_to_batch,
)
from aspire.perception.empathy_evaluation import (
    PERCEPTION_DIMENSION_CRITERIA,
    PerceptionDimension,
    PerceptionEvaluation,
    PerceptionEvaluator,
    PerceptionScore,
)
from aspire.perception.integration import (
    PerceptionConfig,
    PerceptionLoss,
    PerceptionModule,
    PerceptionState,
    integrate_perception_with_trainer,
)
from aspire.perception.metacognition import (
    ConfidenceCalibrator,
    ConfidenceLevel,
    MetaCognitionModule,
    ReflectiveInsight,
    ReflectiveLoop,
    UncertaintyEstimate,
    UncertaintyEstimator,
    UncertaintyType,
)
from aspire.perception.syntropy import (
    CoherenceField,
    SyntropicDimension,
    SyntropicEmpathyEvaluator,
    SyntropicEngine,
    SyntropicFlowTracker,
    SyntropicIntegrator,
    SyntropicMeasurement,
    SyntropicResonanceDetector,
    SyntropicState,
    compute_empathic_syntropy,
    compute_negentropy_approximation,
    compute_semantic_coherence,
)
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

__all__ = [
    # Theory of Mind
    "MentalStateTracker",
    "BeliefState",
    "IntentState",
    "EmotionalState",
    "KnowledgeState",
    "EmotionType",
    "IntentCategory",
    "EmotionalValence",
    # Controlled Chaos
    "ChaosGenerator",
    "ChaosConfig",
    "ChaosType",
    "ChaosSeverity",
    "ChaosInjection",
    "NoiseInjector",
    "AmbiguityGenerator",
    "AdversarialScenarioGenerator",
    "apply_chaos_to_batch",
    # Character
    "CharacterCore",
    "ValueAnchor",
    "BehavioralTrait",
    "AutobiographicalMemory",
    "MemoryEntry",
    "ValueType",
    "TraitDimension",
    "create_socratic_character",
    "create_compassionate_character",
    "create_analytical_character",
    # Meta-Cognition
    "UncertaintyEstimator",
    "ConfidenceCalibrator",
    "ReflectiveLoop",
    "MetaCognitionModule",
    "UncertaintyType",
    "ConfidenceLevel",
    "UncertaintyEstimate",
    "ReflectiveInsight",
    # Evaluation
    "PerceptionEvaluator",
    "PerceptionEvaluation",
    "PerceptionDimension",
    "PerceptionScore",
    "PERCEPTION_DIMENSION_CRITERIA",
    # Syntropy
    "SyntropicEngine",
    "SyntropicMeasurement",
    "SyntropicDimension",
    "SyntropicState",
    "CoherenceField",
    "SyntropicResonanceDetector",
    "SyntropicIntegrator",
    "SyntropicFlowTracker",
    "SyntropicEmpathyEvaluator",
    "compute_negentropy_approximation",
    "compute_semantic_coherence",
    "compute_empathic_syntropy",
    # Integration
    "PerceptionModule",
    "PerceptionConfig",
    "PerceptionState",
    "PerceptionLoss",
    "integrate_perception_with_trainer",
]
