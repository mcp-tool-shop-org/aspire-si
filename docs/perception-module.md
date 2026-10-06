# ASPIRE Perception Module

## Overview

The Perception Module enhances AI agents with deeper cognitive capabilities beyond simple pattern matching. It implements five interconnected systems that work together to create agents with genuine perceptual depth.

## Architecture

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│                              PERCEPTION MODULE                                     │
├───────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │  Theory of Mind │  │   Controlled    │  │    Character    │  │  Syntropy   │ │
│  │     (ToM)       │  │     Chaos       │  │   Persistence   │  │ (Negentropy)│ │
│  │                 │  │                 │  │                 │  │             │ │
│  │ • Belief State  │  │ • Noise Inject  │  │ • Value Anchors │  │ • Coherence │ │
│  │ • Intent State  │  │ • Ambiguity Gen │  │ • Traits        │  │ • Resonance │ │
│  │ • Emotion State │  │ • Adversarial   │  │ • Memory        │  │ • Flow      │ │
│  │ • Knowledge     │  │   Scenarios     │  │                 │  │ • Emergence │ │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘  └──────┬──────┘ │
│           │                    │                    │                   │        │
│           └────────────────────┼────────────────────┼───────────────────┘        │
│                                │                    │                            │
│                     ┌──────────▼────────────────────▼─────────┐                  │
│                     │            Meta-Cognition               │                  │
│                     │                                         │                  │
│                     │  • Uncertainty Estimation               │                  │
│                     │  • Confidence Calibration               │                  │
│                     │  • Reflective Loop                      │                  │
│                     └─────────────────────────────────────────┘                  │
│                                                                                   │
└───────────────────────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Theory of Mind (ToM)

**Purpose**: Enable agents to model what other agents/users likely know, believe, intend, and feel.

**Key Classes**:
- `MentalStateTracker` - Neural module for tracking mental states
- `BeliefState` - Tracks explicit/inferred beliefs and misconceptions
- `IntentState` - Models surface, underlying, and meta-intent
- `EmotionalState` - Tracks emotional valence and trajectory
- `KnowledgeState` - Tracks what information the user has been exposed to

**Usage**:
```python
from aspire.perception import MentalStateTracker

tracker = MentalStateTracker(hidden_dim=768)

# Forward pass with model hidden states
outputs = tracker(hidden_states, attention_mask)

# Get perspective-taking prompt
prompt = tracker.get_perspective_prompt()
# Returns: "PERSPECTIVE TAKING:\n• The user appears strongly frustrated..."
```

### 2. Controlled Chaos

**Purpose**: Train robust agents by deliberately injecting controlled perturbations during training.

**Chaos Types**:
- **Noise Injection**: Missing context, partial information, typos, truncation
- **Ambiguity Generation**: Unclear references, multiple interpretations, implicit requirements
- **Adversarial Scenarios**: Contradictions, conflicting constraints, misleading framing

**Usage**:
```python
from aspire.perception import ChaosGenerator, ChaosConfig

config = ChaosConfig(
    chaos_probability=0.3,
    curriculum_enabled=True,
    curriculum_start_epoch=2,
)
generator = ChaosGenerator(config)

# Inject chaos into input
injection = generator.inject("Original user prompt")
if injection:
    modified_input = injection.modified_input
    expected_handling = injection.ground_truth
```

### 3. Character Persistence

**Purpose**: Give agents stable identity through values, traits, and autobiographical memory.

**Key Classes**:
- `CharacterCore` - The unified character structure
- `ValueAnchor` - Core values with priorities and conflict resolution
- `BehavioralTrait` - Personality traits on spectrums
- `AutobiographicalMemory` - "What have I done before in similar situations?"

**Pre-built Characters**:
```python
from aspire.perception import (
    create_socratic_character,
    create_compassionate_character,
    create_analytical_character,
)

# Create a Socratic teacher character
character = create_socratic_character()

# Generate character prompt for model priming
prompt = character.generate_character_prompt(context="debugging session")
```

### 4. Meta-Cognition

**Purpose**: Enable agents to think about their own thinking - uncertainty, confidence, and self-reflection.

**Key Classes**:
- `UncertaintyEstimator` - Neural module for uncertainty prediction
- `ConfidenceCalibrator` - Calibrates raw confidence to actual accuracy
- `ReflectiveLoop` - Periodic self-reflection prompts

**Usage**:
```python
from aspire.perception import MetaCognitionModule, ConfidenceCalibrator

metacog = MetaCognitionModule(hidden_dim=768)
outputs = metacog(hidden_states)

# Get action recommendation
recommendation = metacog.get_action_recommendation(outputs)
# Returns: "Add hedging language; Consider that this response may need revision"

# Calibrate confidence
calibrator = ConfidenceCalibrator()
calibrator.record(predicted=0.9, actual=1.0, domain="programming")
calibrated = calibrator.calibrate(raw_confidence=0.85, domain="programming")
```

### 5. Syntropy

**Purpose**: Measure and promote order-creation (negentropy) in agent cognition. While entropy quantifies disorder, syntropy reflects structure, meaning, and coherence.

**Mathematical Foundation**:
- Negentropy: J(x) = H(x_gaussian) - H(x)
- Based on Schrödinger (1944), Szent-Györgyi (1974), and ICA literature

**Key Classes**:
- `SyntropicEngine` - Main engine for syntropic processing
- `SyntropicResonanceDetector` - Detects alignment between agent and user states
- `SyntropicIntegrator` - Measures information integration quality
- `SyntropicFlowTracker` - Tracks syntropy across interaction
- `SyntropicEmpathyEvaluator` - Evaluates empathy through syntropic lens
- `CoherenceField` - Models the attractor basin toward shared understanding

**Usage**:
```python
from aspire.perception import SyntropicEngine, SyntropicEmpathyEvaluator

# Create engine
engine = SyntropicEngine(hidden_dim=768)

# Measure syntropy from hidden states
measurement = engine.measure(hidden_states, user_state)
print(f"Syntropy: {measurement.syntropy_score}")  # -1 to 1
print(f"State: {measurement.state}")  # ENTROPIC, NEUTRAL, SYNTROPIC, RESONANT, CRYSTALLIZED

# Get guidance for improving syntropy
guidance = engine.get_syntropic_guidance()
# Returns: "SYNTROPIC GUIDANCE:\n• High resonance with user achieved..."

# Use empathy evaluator
evaluator = SyntropicEmpathyEvaluator(hidden_dim=768)
result = evaluator.evaluate(agent_states, user_state)
print(f"Empathic resonance: {result['empathic_resonance']}")
```

**Syntropic Dimensions** (12 total):
- Structural: semantic_coherence, logical_consistency, narrative_flow
- Relational: empathic_resonance, intentional_alignment, knowledge_bridging
- Emergent: meaning_generation, insight_crystallization, integration_depth
- Temporal: contextual_continuity, progressive_refinement, anticipatory_coherence

## Integration with ASPIRE Training

### Basic Integration

```python
from aspire import AspireTrainer, AspireConfig
from aspire.perception import PerceptionModule, PerceptionConfig

# Configure perception
perception_config = PerceptionConfig(
    tom_enabled=True,
    chaos_enabled=True,
    chaos_probability=0.3,
    character_enabled=True,
    metacognition_enabled=True,
)

# Create perception module
perception = PerceptionModule(perception_config)

# Integrate with trainer
from aspire.perception.integration import integrate_perception_with_trainer

config = AspireConfig()
trainer = AspireTrainer(config)
integrate_perception_with_trainer(trainer, perception)

# Training now includes perception
results = await trainer.train(teacher, prompts)
```

### Perception-Enhanced Evaluation

```python
from aspire.perception import PerceptionEvaluator

evaluator = PerceptionEvaluator(strictness=0.5)

evaluation = evaluator.evaluate(
    prompt="How do I fix this bug?",
    response="...",
    context={
        "chaos_injection": injection,
        "character": character,
    }
)

print(f"Cognitive Empathy: {evaluation.cognitive_empathy_score}")
print(f"Metacognition: {evaluation.metacognition_score}")
print(f"Character: {evaluation.character_score}")
print(f"Robustness: {evaluation.robustness_score}")
```

## Evaluation Dimensions

The perception module adds 15 evaluation dimensions:

| Dimension | Category | Description |
|-----------|----------|-------------|
| `cognitive_empathy` | Empathy | Accurately models user's mental state |
| `emotional_attunement` | Empathy | Responds to emotional subtext |
| `intent_recognition` | Empathy | Understands underlying goals |
| `uncertainty_calibration` | Meta-cognition | Appropriate confidence |
| `self_awareness` | Meta-cognition | Knows limitations |
| `assumption_transparency` | Meta-cognition | Makes assumptions explicit |
| `value_consistency` | Character | Acts according to values |
| `behavioral_coherence` | Character | Consistent personality |
| `perspective_integration` | Character | Balances viewpoints |
| `ambiguity_handling` | Robustness | Handles unclear input |
| `contradiction_resolution` | Robustness | Navigates conflicts |
| `noise_tolerance` | Robustness | Functions despite imperfections |
| `syntropic_coherence` | Syntropy | Creates order/meaning rather than disorder |
| `empathic_resonance` | Syntropy | Attunes to and aligns with user state |
| `meaning_generation` | Syntropy | Generates understanding beyond surface info |

## Curriculum Learning

The perception module supports curriculum-based training:

1. **Foundation** (Epochs 0-1): No chaos, basic mental state tracking
2. **Robustness** (Epochs 2-4): Gradually increasing chaos injection
3. **Character** (Epochs 5+): Full character consistency enforcement

```python
config = ChaosConfig(
    curriculum_enabled=True,
    curriculum_start_epoch=2,
    curriculum_ramp_epochs=3,  # Ramp from 0 to full chaos over 3 epochs
)
```

## Windows Compatibility

All perception modules are fully Windows-compatible:

- No `xformers` dependency (set `XFORMERS_DISABLED=1`)
- DataLoader `num_workers=0` (Windows multiprocessing constraint)
- Compatible with RTX 5080 (SM 12.0 Blackwell architecture)

## Testing

Run perception tests:
```bash
pytest tests/perception/ -v
```

Current coverage: 163 tests, all passing.
