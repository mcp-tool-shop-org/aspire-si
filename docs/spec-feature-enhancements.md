# ASPIRE Feature Enhancements Specification

> **Status**: Draft
> **Author**: Claude (AI Development Partner)
> **Date**: 2026-01-23
> **Target**: ASPIRE v2.0
> **Research**: Incorporates 2026 state-of-the-art best practices

---

## Executive Summary

This specification outlines four major enhancements to transform ASPIRE from a training framework into a complete AI reasoning ecosystem:

1. **Curriculum Engine** - Dynamic difficulty progression with spaced repetition
2. **Teacher Debate Mode** - Multi-perspective adversarial reasoning
3. **Critic-Only Inference API** - Runtime reasoning enhancement
4. **Trajectory Visualizer** - Interpretable dialogue analysis

---

## 0. 2026 Best Practices Summary

This section synthesizes cutting-edge research and industry practices from 2025-2026 that inform each component.

### 0.1 Curriculum Learning (2026 State-of-the-Art)

**Key Research:**
- [Adaptive Difficulty Curriculum Learning (ADCL)](https://arxiv.org/abs/2505.08364) - Addresses "Difficulty Shift" phenomenon where model's perception of difficulty changes during training
- [CAMPUS Framework](https://arxiv.org/html/2509.13790) - Competence-Aware Multi-Perspective Curriculum with softmax sub-curricula selection
- [E2H Reasoner](https://arxiv.org/abs/2506.06632) - Easy-to-Hard curriculum with scheduled "fading" of easy tasks to prevent overfitting

**Best Practices:**
1. **Dynamic difficulty re-estimation** - Periodically re-assess difficulty within batches as model capabilities evolve
2. **Competence-aware scheduling** - Use negative perplexity or learned reward as on-the-fly competence measure
3. **Easy task fading** - Initially emphasize easy tasks, then fade them out to prevent overfitting
4. **Multi-perspective sub-curricula** - Maintain multiple difficulty-ordered queues and soft-select based on competence
5. **Monitor for misalignment** - Watch for negative transfer, curriculum rigidity, and catastrophic forgetting

**Caveat:** [Recent research](https://openreview.net/forum?id=sHn5rq6L0O) shows CL benefits vary by task - random sampling can be competitive for some mathematical reasoning tasks.

### 0.2 Spaced Repetition (FSRS Algorithm)

**Key Research:**
- [FSRS (Free Spaced Repetition Scheduler)](https://github.com/open-spaced-repetition/fsrs4anki) - Modern ML-based algorithm achieving 20-30% fewer reviews than SM-2
- [DRL-SRS](https://www.mdpi.com/2076-3417/14/13/5591) - Deep Reinforcement Learning for spaced repetition as Partially Observable MDP
- [Memory Dynamics Research](https://www.researchgate.net/publication/369045947) - Stability + Retrievability dual-variable memory model

**Best Practices (FSRS over SM-2):**
1. **Use FSRS algorithm** - [20-30% more efficient](https://domenic.me/fsrs/) than traditional SM-2
2. **Model both stability and retrievability** - Two-variable memory representation captures forgetting dynamics
3. **Personalized parameters** - Run optimizer over user's own review history for personalized scheduling
4. **Target specific retention** - Allow configurable retention targets (70-97% reasonable range)
5. **Flexible timing** - Support early reviews and delayed reviews with automatic adaptation
6. **Local-first** - Run entirely on device for privacy and offline capability

**FSRS Formula:**
```
R(t) = (1 + t/(9*S))^(-1)  # Retrievability decay
S' = S * e^(w * (R - 0.9))  # Stability update after review
```

### 0.3 Multi-Agent Debate (2026 State-of-the-Art)

**Key Research:**
- [Multi-Agent Debate for Factuality](https://composable-models.github.io/llm_debate/) - Significantly reduces hallucinations through debate
- [A-HMAD (Adaptive Heterogeneous MAD)](https://link.springer.com/article/10.1007/s44443-025-00353-3) - 4-6% accuracy improvement over original MAD
- [Controlled Study of MAD](https://arxiv.org/abs/2511.07784) - Identifies intrinsic reasoning strength and group diversity as key success factors
- [DMAD (Diverse MAD)](https://openreview.net/forum?id=t6QHYUOQL7) - Breaks "fixed mental set" by encouraging distinct reasoning approaches

**Best Practices:**
1. **Heterogeneous agents** - Use diverse personas/reasoning styles, not just different prompts
2. **Trained consensus optimizer** - Learn to aggregate debate transcripts (first learning-based aggregation in MAD)
3. **Adaptive stability detection** - Mathematically formalize debate convergence
4. **Minority override capability** - Train aggregator to recognize when minority agent is correct
5. **Rational behavior enforcement** - Agents following high-quality arguments achieve >90% correction rate
6. **Watch for "strong model paradox"** - Weaker agents may struggle to leverage sophisticated arguments from stronger models

**Key Insight:** Majority pressure can suppress independent correction - design debate protocols that allow minority positions to be heard.

### 0.4 Critic-Guided Self-Refinement (2026 Inference Scaling)

**Key Research:**
- [State of LLMs 2025](https://magazine.sebastianraschka.com/p/state-of-llms-2025) - Predicts 2026 progress will come more from inference-time scaling than training
- [Critic-CoT Framework](https://www.emergentmind.com/topics/critic-cot-framework) - Couples chain-of-thought with explicit critic interventions
- [ThinkPRM](https://arxiv.org/abs/2504.16828) - Process Reward Models that generate verification chain-of-thought
- [Two-Player Critic-Actor](https://arxiv.org/html/2411.16579v1) - Scalable oversight with critic providing step-level feedback
- [SCRPO](https://arxiv.org/html/2512.05387v2) - Self Critique and Refinement-based Preference Optimization

**Best Practices:**
1. **Process Reward Models (PRMs)** - Score intermediate steps, not just final outcomes
2. **ThinkPRM approach** - Train verifiers to generate verification CoT (1% of labels needed vs discriminative PRMs)
3. **Compute-optimal allocation** - Use question difficulty to predict efficacy of test-time compute
4. **Critic at test-time AND training** - Distill inference-time refinement into training (SCRPO)
5. **Tool-interactive critiquing** - Allow critic to use external tools for validation
6. **Beware overthinking** - [Longer reasoning isn't always better](https://medium.com/the-synaptic-stack/test-time-scaling-are-longer-reasoning-chains-always-better-de0844a110ff); use Shortest Majority Vote when appropriate

**2026 Prediction:** "A lot of LLM benchmark progress will come from improved tooling and inference-time scaling rather than from training or the core model itself."

### 0.5 Knowledge Graphs for Concept Scaffolding

**Key Research:**
- [LLM-empowered KG Construction Survey](https://arxiv.org/abs/2510.20345) - Comprehensive review of LLM+KG integration
- [Neo4j LLM Knowledge Graph Builder](https://neo4j.com/blog/developer/llm-knowledge-graph-builder-release/) - Entity extraction and relationship mapping
- [GraphRAG and LightRAG](https://medium.com/@claudiubranzan/from-llms-to-knowledge-graphs-building-production-ready-graph-systems-in-2025-2b4aff1ec99a) - 90% hallucination reduction with sub-50ms latency

**Best Practices:**
1. **LLM-assisted ontology construction** - GPT-4/Claude achieve quality comparable to novice human modelers
2. **Few-shot concept extraction** - No need for thousands of labeled examples
3. **Dual-level retrieval (LightRAG)** - 10x token reduction vs full GraphRAG
4. **Entity graph across documents** - Relate concepts spread across multiple sources
5. **Schema teaching** - Define your domain schema, let LLM extract accordingly

### 0.6 LLM Observability and Visualization

**Key Research:**
- [LLM Observability Tools Comparison 2026](https://lakefs.io/blog/llm-observability-tools/) - Comprehensive tooling review
- [LIDA](https://microsoft.github.io/lida/) - Grammar-agnostic visualization generation
- [LLM Visualization Techniques](https://ai2.work/technology/ai-tech-llm-visualization-2025/) - Strategic insights for 2025-2026

**Best Practices:**
1. **Attention attribution** - Visualize attention patterns for interpretability
2. **Embedding visualizations** - Track representation drift during training
3. **Saliency methods** - Understand which inputs drive outputs
4. **Natural language to visualization** - Use LLMs to generate visualizations from prompts
5. **Real-time data integration** - Dashboard should support live training metrics
6. **Multimodal support** - Visualize both text and hidden state dynamics

---

## 1. Curriculum Engine

### 1.1 Problem Statement

Currently, ASPIRE uses static prompts with fixed difficulty. The existing `CurriculumConfig` only adjusts teacher weights per stage but doesn't:
- Track per-concept mastery
- Implement spaced repetition for retention
- Generate adversarial examples targeting weaknesses
- Scaffold complex concepts from simpler ones

### 1.2 Proposed Architecture

```
aspire/curriculum/
├── __init__.py
├── engine.py              # Main CurriculumEngine orchestrator
├── difficulty.py          # DifficultyEstimator + progression
├── scaffolding.py         # ConceptGraph + prerequisites
├── spaced_repetition.py   # SM-2 algorithm implementation
├── adversarial.py         # Weakness-targeted prompt generation
└── analytics.py           # Mastery tracking + visualization data
```

### 1.3 Core Components

#### 1.3.1 CurriculumEngine

```python
@dataclass
class CurriculumConfig:
    """Extended curriculum configuration."""
    # Existing fields
    stages: list[str] = field(default_factory=lambda: [
        "foundation", "reasoning", "nuance", "adversarial", "transfer"
    ])
    stage_epochs: dict[str, int] = field(default_factory=dict)
    advancement_threshold: float = 0.7

    # NEW: Difficulty progression
    initial_difficulty: float = 0.3
    difficulty_increment: float = 0.1
    max_difficulty: float = 1.0
    difficulty_window: int = 10  # Recent prompts to consider

    # NEW: Spaced repetition
    enable_spaced_repetition: bool = True
    retention_target: float = 0.9
    min_interval_hours: float = 1.0
    max_interval_days: float = 30.0

    # NEW: Concept scaffolding
    concept_graph_path: str | None = None
    prerequisite_threshold: float = 0.6

    # NEW: Adversarial curriculum
    weakness_sampling_ratio: float = 0.3  # 30% prompts target weaknesses
    adversarial_temperature: float = 1.2


class CurriculumEngine:
    """
    Orchestrates adaptive curriculum for ASPIRE training.

    Responsibilities:
    - Track mastery per concept/skill
    - Schedule prompts based on spaced repetition
    - Generate adversarial prompts targeting weaknesses
    - Ensure prerequisite concepts are mastered first
    """

    def __init__(
        self,
        config: CurriculumConfig,
        concept_graph: ConceptGraph | None = None,
        teacher: BaseTeacher | None = None,  # For adversarial generation
    ):
        self.config = config
        self.difficulty_estimator = DifficultyEstimator()
        self.spaced_repetition = SpacedRepetitionScheduler(config)
        self.concept_graph = concept_graph or ConceptGraph()
        self.adversarial_generator = AdversarialCurriculumGenerator(teacher)
        self.mastery_tracker = MasteryTracker()

    def get_next_batch(
        self,
        batch_size: int,
        available_prompts: list[Prompt],
    ) -> list[ScheduledPrompt]:
        """
        Select next batch of prompts based on curriculum state.

        Selection strategy:
        1. Due for review (spaced repetition) - priority
        2. New concepts with satisfied prerequisites
        3. Adversarial prompts targeting weaknesses
        4. Difficulty-appropriate new prompts
        """
        batch = []

        # 1. Spaced repetition reviews (highest priority)
        due_reviews = self.spaced_repetition.get_due_items(
            limit=int(batch_size * 0.4)
        )
        batch.extend(due_reviews)

        # 2. Weakness-targeted adversarial prompts
        if random.random() < self.config.weakness_sampling_ratio:
            weaknesses = self.mastery_tracker.get_weakest_concepts(k=3)
            adversarial = self.adversarial_generator.generate(
                weaknesses=weaknesses,
                difficulty=self._current_difficulty(),
                count=int(batch_size * 0.3),
            )
            batch.extend(adversarial)

        # 3. New prompts with satisfied prerequisites
        remaining = batch_size - len(batch)
        eligible = self._filter_by_prerequisites(available_prompts)
        difficulty_matched = self._filter_by_difficulty(eligible)
        batch.extend(difficulty_matched[:remaining])

        return batch

    def update_mastery(
        self,
        prompt: Prompt,
        evaluation: TeacherEvaluation,
        dialogue: GeneratedDialogue,
    ):
        """Update mastery tracking after a dialogue."""
        # Extract concepts from prompt
        concepts = self.concept_graph.extract_concepts(prompt.text)

        # Update mastery for each concept
        for concept in concepts:
            self.mastery_tracker.update(
                concept=concept,
                score=evaluation.overall_score,
                dimensions=evaluation.dimension_scores,
            )

        # Update spaced repetition schedule
        self.spaced_repetition.record_review(
            item_id=prompt.id,
            quality=self._score_to_quality(evaluation.overall_score),
        )

        # Adjust difficulty if needed
        self._maybe_adjust_difficulty(evaluation.overall_score)
```

#### 1.3.2 Spaced Repetition (FSRS Algorithm - 2026 Best Practice)

> **2026 Update:** We use FSRS (Free Spaced Repetition Scheduler) instead of SM-2.
> FSRS achieves [20-30% fewer reviews](https://domenic.me/fsrs/) for the same retention.
> See [Section 0.2](#02-spaced-repetition-fsrs-algorithm) for research background.

```python
@dataclass
class FSRSParameters:
    """
    FSRS model parameters (21 total).

    These can be personalized by running the optimizer
    on the model's own review history.
    """
    # Initial stability values by rating (Again, Hard, Good, Easy)
    w0: float = 0.4    # Again
    w1: float = 0.6    # Hard
    w2: float = 2.4    # Good
    w3: float = 5.8    # Easy

    # Difficulty parameters
    w4: float = 4.93   # Initial difficulty weight
    w5: float = 0.94   # Difficulty decay
    w6: float = 0.86   # Difficulty floor
    w7: float = 0.01   # Difficulty delta

    # Stability parameters
    w8: float = 1.49   # Stability growth base
    w9: float = 0.14   # Grade modifier
    w10: float = 0.94  # Retrievability modifier
    w11: float = 2.18  # Stability ceiling
    w12: float = 0.05  # Stability floor
    w13: float = 0.34  # Hard penalty
    w14: float = 1.26  # Easy bonus

    # Forgetting curve parameters
    w15: float = 0.29  # Forgetting curve modifier
    w16: float = 2.61  # Relearning stability base


@dataclass
class FSRSReviewItem:
    """Item in FSRS spaced repetition system."""
    item_id: str
    prompt: str
    concepts: list[str]

    # FSRS state (Stability-Retrievability model)
    difficulty: float = 0.3      # D: 0-1, how hard this item is
    stability: float = 1.0       # S: days until R drops to 90%
    retrievability: float = 1.0  # R: current probability of recall

    # Scheduling
    last_review: datetime | None = None
    next_review: datetime | None = None

    # History
    review_history: list[FSRSReviewRecord] = field(default_factory=list)
    reps: int = 0
    lapses: int = 0  # Number of times forgotten


@dataclass
class FSRSReviewRecord:
    """Record of a single review."""
    timestamp: datetime
    rating: int  # 1=Again, 2=Hard, 3=Good, 4=Easy
    stability_before: float
    stability_after: float
    retrievability: float
    elapsed_days: float


class FSRSScheduler:
    """
    FSRS-based spaced repetition scheduler.

    FSRS (Free Spaced Repetition Scheduler) is a modern algorithm that:
    - Models memory with Stability (S) and Retrievability (R)
    - Uses ML-optimized parameters instead of arbitrary formulas
    - Achieves 20-30% fewer reviews than SM-2 for same retention

    References:
    - https://github.com/open-spaced-repetition/fsrs4anki
    - https://pypi.org/project/fsrs/
    """

    def __init__(
        self,
        config: CurriculumConfig,
        params: FSRSParameters | None = None,
    ):
        self.config = config
        self.params = params or FSRSParameters()
        self.items: dict[str, FSRSReviewItem] = {}
        self.desired_retention: float = config.retention_target  # e.g., 0.9

    def calculate_retrievability(
        self,
        stability: float,
        elapsed_days: float,
    ) -> float:
        """
        Calculate retrievability using FSRS forgetting curve.

        R(t) = (1 + t/(9*S))^(-1)

        Where:
        - t = elapsed time in days
        - S = stability (days until R = 0.9)
        """
        if elapsed_days <= 0:
            return 1.0
        return (1 + elapsed_days / (9 * stability)) ** -1

    def calculate_interval(
        self,
        stability: float,
        desired_retention: float | None = None,
    ) -> float:
        """
        Calculate optimal interval for desired retention.

        Derived from: R = (1 + t/(9*S))^(-1)
        Solving for t: t = 9*S * (R^(-1) - 1)
        """
        r = desired_retention or self.desired_retention
        interval = 9 * stability * (r ** -1 - 1)

        # Clamp interval
        return max(
            self.config.min_interval_hours / 24,
            min(interval, self.config.max_interval_days)
        )

    def update_stability(
        self,
        item: FSRSReviewItem,
        rating: int,  # 1=Again, 2=Hard, 3=Good, 4=Easy
        retrievability: float,
    ) -> float:
        """
        Update stability after review using FSRS formula.

        For successful recall (rating >= 2):
        S' = S * (1 + e^(w8) * (11-D) * S^(-w9) * (e^(w10*(1-R)) - 1) *
             (if Hard: w13, if Easy: w14, else 1))

        For forgotten (rating == 1):
        S' = w15 * D^(-w16) * ((S+1)^w17 - 1) * e^(w18*(1-R))
        """
        w = self.params
        S = item.stability
        D = item.difficulty
        R = retrievability

        if rating == 1:  # Again - forgotten
            # Relearning stability formula
            new_stability = w.w15 * (D ** -0.5) * ((S + 1) ** 0.2 - 1)
            item.lapses += 1
        else:
            # Successful recall - stability grows
            hard_penalty = w.w13 if rating == 2 else 1.0
            easy_bonus = w.w14 if rating == 4 else 1.0

            stability_growth = (
                math.exp(w.w8) *
                (11 - D) *
                (S ** -w.w9) *
                (math.exp(w.w10 * (1 - R)) - 1) *
                hard_penalty *
                easy_bonus
            )

            new_stability = S * (1 + stability_growth)

        # Clamp stability to reasonable bounds
        return max(0.1, min(new_stability, 36500))  # Max ~100 years

    def update_difficulty(
        self,
        item: FSRSReviewItem,
        rating: int,
    ) -> float:
        """
        Update difficulty based on rating.

        D' = D - w6 * (rating - 3)
        Clamped to [0.1, 1.0]
        """
        w = self.params
        delta = w.w7 * (rating - 3)  # Easier if rating > 3, harder if < 3
        new_difficulty = item.difficulty - delta
        return max(0.1, min(1.0, new_difficulty))

    def record_review(
        self,
        item_id: str,
        rating: int,  # 1=Again, 2=Hard, 3=Good, 4=Easy
    ):
        """
        Record a review and update scheduling.

        Rating mapping from critic score (0-10):
        - 0-3: Again (1) - complete failure
        - 3-5: Hard (2) - struggled significantly
        - 5-7: Good (3) - correct with effort
        - 7-10: Easy (4) - smooth recall
        """
        item = self.items[item_id]
        now = datetime.now()

        # Calculate elapsed time
        if item.last_review:
            elapsed_days = (now - item.last_review).total_seconds() / 86400
        else:
            elapsed_days = 0

        # Calculate current retrievability
        retrievability = self.calculate_retrievability(item.stability, elapsed_days)

        # Store old stability for record
        old_stability = item.stability

        # Update stability and difficulty
        item.stability = self.update_stability(item, rating, retrievability)
        item.difficulty = self.update_difficulty(item, rating)
        item.retrievability = 1.0  # Reset after review
        item.reps += 1

        # Calculate next interval
        interval_days = self.calculate_interval(item.stability)

        # Schedule next review
        item.last_review = now
        item.next_review = now + timedelta(days=interval_days)

        # Record history
        item.review_history.append(FSRSReviewRecord(
            timestamp=now,
            rating=rating,
            stability_before=old_stability,
            stability_after=item.stability,
            retrievability=retrievability,
            elapsed_days=elapsed_days,
        ))

    def score_to_rating(self, score: float) -> int:
        """Convert critic score (0-10) to FSRS rating (1-4)."""
        if score < 3:
            return 1  # Again
        elif score < 5:
            return 2  # Hard
        elif score < 7:
            return 3  # Good
        else:
            return 4  # Easy

    def get_due_items(self, limit: int = 10) -> list[FSRSReviewItem]:
        """Get items due for review, sorted by urgency."""
        now = datetime.now()
        due = []

        for item in self.items.values():
            if item.next_review and item.next_review <= now:
                # Calculate how overdue
                overdue_days = (now - item.next_review).total_seconds() / 86400
                # More overdue = higher priority
                urgency = overdue_days / max(item.stability, 0.1)
                due.append((urgency, item))

        # Sort by urgency (most urgent first)
        due.sort(key=lambda x: -x[0])

        return [item for _, item in due[:limit]]

    def optimize_parameters(
        self,
        review_history: list[tuple[FSRSReviewItem, list[FSRSReviewRecord]]],
    ) -> FSRSParameters:
        """
        Optimize FSRS parameters using review history.

        Uses maximum likelihood estimation to find parameters
        that best predict actual review outcomes.

        This is what makes FSRS personalized - it learns YOUR
        forgetting patterns from YOUR review history.
        """
        # Implementation would use scipy.optimize or torch
        # to minimize prediction error on historical reviews
        raise NotImplementedError(
            "Parameter optimization requires scipy/torch. "
            "See https://github.com/open-spaced-repetition/fsrs-optimizer"
        )
```

#### 1.3.3 Concept Scaffolding

```python
class ConceptGraph:
    """
    Directed acyclic graph of concepts and prerequisites.

    Example:
        recursion -> [loops, functions, call_stack]
        async_await -> [callbacks, promises, event_loop]
    """

    def __init__(self, graph_path: str | None = None):
        self.graph: nx.DiGraph = nx.DiGraph()
        if graph_path:
            self.load(graph_path)

    def add_concept(
        self,
        concept: str,
        prerequisites: list[str] | None = None,
        difficulty: float = 0.5,
        metadata: dict | None = None,
    ):
        """Add concept with prerequisites."""
        self.graph.add_node(concept, difficulty=difficulty, **(metadata or {}))

        if prerequisites:
            for prereq in prerequisites:
                self.graph.add_edge(prereq, concept)

    def get_eligible_concepts(
        self,
        mastery: dict[str, float],
        threshold: float = 0.6,
    ) -> list[str]:
        """
        Get concepts whose prerequisites are mastered.

        A concept is eligible if:
        1. All prerequisites have mastery >= threshold
        2. The concept itself has mastery < 1.0 (not fully mastered)
        """
        eligible = []

        for concept in self.graph.nodes:
            # Check prerequisites
            prereqs = list(self.graph.predecessors(concept))
            prereqs_mastered = all(
                mastery.get(p, 0.0) >= threshold for p in prereqs
            )

            # Check not already mastered
            concept_mastery = mastery.get(concept, 0.0)

            if prereqs_mastered and concept_mastery < 1.0:
                eligible.append(concept)

        return eligible

    def get_learning_path(
        self,
        target_concept: str,
        mastery: dict[str, float],
    ) -> list[str]:
        """Get ordered list of concepts to learn to reach target."""
        # Topological sort of prerequisites
        ancestors = nx.ancestors(self.graph, target_concept)
        subgraph = self.graph.subgraph(ancestors | {target_concept})

        # Filter to unmastered
        path = [
            c for c in nx.topological_sort(subgraph)
            if mastery.get(c, 0.0) < 0.9
        ]

        return path
```

#### 1.3.4 Adversarial Curriculum Generator

```python
class AdversarialCurriculumGenerator:
    """
    Generates prompts that target identified weaknesses.

    Uses teacher model to create challenging variations
    that stress-test specific concepts or skills.
    """

    def __init__(self, teacher: BaseTeacher | None = None):
        self.teacher = teacher

    async def generate(
        self,
        weaknesses: list[WeaknessProfile],
        difficulty: float,
        count: int,
    ) -> list[AdversarialPrompt]:
        """
        Generate adversarial prompts targeting weaknesses.

        Weakness types:
        - concept: Struggles with specific topic
        - dimension: Low scores on reasoning/clarity/etc.
        - edge_case: Fails on boundary conditions
        - transfer: Can't apply knowledge to new domains
        """
        prompts = []

        for weakness in weaknesses:
            if weakness.type == "concept":
                prompt = await self._generate_concept_challenge(
                    concept=weakness.concept,
                    failure_patterns=weakness.failure_patterns,
                    difficulty=difficulty,
                )
            elif weakness.type == "dimension":
                prompt = await self._generate_dimension_challenge(
                    dimension=weakness.dimension,
                    difficulty=difficulty,
                )
            elif weakness.type == "edge_case":
                prompt = await self._generate_edge_case(
                    concept=weakness.concept,
                    known_failures=weakness.failure_examples,
                )

            prompts.append(prompt)

        return prompts[:count]

    async def _generate_concept_challenge(
        self,
        concept: str,
        failure_patterns: list[str],
        difficulty: float,
    ) -> AdversarialPrompt:
        """Generate a prompt that challenges a weak concept."""

        system_prompt = f"""Generate a challenging question about {concept}.

The student has shown weakness in these areas:
{chr(10).join(f'- {p}' for p in failure_patterns)}

Create a question at difficulty {difficulty:.1f}/1.0 that:
1. Directly tests the weak areas
2. Requires deep understanding, not memorization
3. Has a clear correct answer for evaluation

Output format:
QUESTION: [the question]
KEY_CONCEPTS: [comma-separated concepts tested]
EXPECTED_REASONING: [brief outline of correct approach]
"""

        response = await self.teacher.generate(system_prompt)
        return self._parse_adversarial_response(response, concept)
```

### 1.4 Integration with AspireTrainer

```python
class AspireTrainer:
    def __init__(self, config: AspireConfig):
        # ... existing init ...

        # NEW: Initialize curriculum engine
        if config.curriculum.enable_curriculum:
            self.curriculum_engine = CurriculumEngine(
                config=config.curriculum,
                teacher=self.teacher,
            )
        else:
            self.curriculum_engine = None

    def train(self, train_prompts: list[str], eval_prompts: list[str]):
        # Convert to Prompt objects with IDs
        prompts = [Prompt(id=str(i), text=p) for i, p in enumerate(train_prompts)]

        for epoch in range(self.config.training.num_epochs):
            # NEW: Use curriculum engine for batch selection
            if self.curriculum_engine:
                batch_prompts = self.curriculum_engine.get_next_batch(
                    batch_size=self.config.training.batch_size,
                    available_prompts=prompts,
                )
            else:
                batch_prompts = random.sample(prompts, self.config.training.batch_size)

            # ... training loop ...

            # NEW: Update curriculum after each dialogue
            for prompt, dialogue in zip(batch_prompts, dialogues):
                if self.curriculum_engine:
                    self.curriculum_engine.update_mastery(
                        prompt=prompt,
                        evaluation=dialogue.final_evaluation,
                        dialogue=dialogue,
                    )
```

---

## 2. Teacher Debate Mode

> **2026 Best Practices Applied:**
> - [A-HMAD](https://link.springer.com/article/10.1007/s44443-025-00353-3): Adaptive heterogeneous agents for 4-6% accuracy gains
> - [DMAD](https://openreview.net/forum?id=t6QHYUOQL7): Diverse reasoning approaches to break "fixed mental set"
> - [MAD LLM Judges](https://openreview.net/forum?id=Vusd1Hw2D9): Trained consensus optimizer for minority override
> - See [Section 0.3](#03-multi-agent-debate-2026-state-of-the-art) for full research background.

### 2.1 Problem Statement

The current `CompositeTeacher` with `strategy="debate"` just falls back to voting. A true debate system would have teachers:
- Argue with each other about evaluation
- Defend their positions with evidence
- Update judgments based on peer arguments
- Reach consensus or document disagreements

### 2.2 Proposed Architecture

```
aspire/teachers/
├── debate/
│   ├── __init__.py
│   ├── engine.py          # DebateEngine orchestrator
│   ├── argument.py        # Argument/Rebuttal data structures
│   ├── moderator.py       # Debate flow control
│   ├── consensus.py       # Consensus detection + resolution
│   └── transcript.py      # Debate history + analysis
```

### 2.3 Core Components

#### 2.3.1 Data Structures

```python
@dataclass
class DebateArgument:
    """A single argument in a teacher debate."""
    teacher_id: str
    teacher_name: str
    position: str  # "support", "oppose", "neutral"
    claim: str
    evidence: list[str]
    cited_dimensions: list[EvaluationDimension]
    confidence: float  # 0-1
    references_arguments: list[str]  # IDs of arguments this responds to
    timestamp: datetime = field(default_factory=datetime.now)

    def to_prompt_context(self) -> str:
        """Format for inclusion in debate prompt."""
        return f"""
{self.teacher_name} ({self.position}, confidence: {self.confidence:.0%}):
"{self.claim}"

Evidence:
{chr(10).join(f'  - {e}' for e in self.evidence)}
"""


@dataclass
class DebateRound:
    """One round of debate (all teachers speak once)."""
    round_number: int
    arguments: list[DebateArgument]
    consensus_check: ConsensusResult | None = None


@dataclass
class DebateTranscript:
    """Complete debate record."""
    prompt: str
    student_response: str
    dialogue_history: DialogueHistory | None
    rounds: list[DebateRound]
    final_consensus: ConsensusResult
    total_duration_seconds: float

    # Analytics
    position_changes: list[PositionChange]
    key_disagreements: list[Disagreement]
    strongest_arguments: list[DebateArgument]


@dataclass
class ConsensusResult:
    """Result of consensus check."""
    reached: bool
    final_score: float | None
    final_evaluation: TeacherEvaluation | None
    agreement_level: float  # 0-1, how much teachers agree
    dissenting_opinions: list[DebateArgument]
    resolution_method: str  # "unanimous", "majority", "moderator", "timeout"
```

#### 2.3.2 Debate Engine

```python
class DebateEngine:
    """
    Orchestrates multi-teacher debates about student responses.

    Debate flow:
    1. Opening statements - each teacher gives initial evaluation
    2. Rebuttal rounds - teachers respond to each other
    3. Consensus check - attempt to reach agreement
    4. Resolution - final evaluation synthesis

    2026 Best Practices Implemented:
    - Heterogeneous agents with diverse reasoning styles (A-HMAD)
    - Trained consensus optimizer for minority override capability
    - Adaptive stability detection for convergence
    - Rational behavior scoring to weight arguments

    References:
    - https://arxiv.org/abs/2511.07784 (Controlled MAD Study)
    - https://link.springer.com/article/10.1007/s44443-025-00353-3 (A-HMAD)
    """

    def __init__(
        self,
        teachers: list[BaseTeacher],
        moderator: DebateModerator | None = None,
        max_rounds: int = 3,
        consensus_threshold: float = 0.8,
        min_confidence_to_argue: float = 0.6,
        # 2026 additions
        use_trained_consensus: bool = True,
        enable_minority_override: bool = True,
        rationality_weighting: bool = True,
    ):
        self.teachers = teachers
        self.moderator = moderator or DefaultModerator()
        self.max_rounds = max_rounds
        self.consensus_threshold = consensus_threshold
        self.min_confidence = min_confidence_to_argue

    async def debate(
        self,
        prompt: str,
        student_response: str,
        dialogue_history: DialogueHistory | None = None,
    ) -> DebateTranscript:
        """
        Run a full debate and return transcript.
        """
        transcript = DebateTranscript(
            prompt=prompt,
            student_response=student_response,
            dialogue_history=dialogue_history,
            rounds=[],
            position_changes=[],
            key_disagreements=[],
            strongest_arguments=[],
        )

        start_time = time.time()

        # Round 0: Opening statements (parallel)
        opening_round = await self._opening_round(
            prompt, student_response, dialogue_history
        )
        transcript.rounds.append(opening_round)

        # Check for early consensus
        consensus = self._check_consensus(opening_round.arguments)
        if consensus.reached:
            transcript.final_consensus = consensus
            transcript.total_duration_seconds = time.time() - start_time
            return transcript

        # Rebuttal rounds
        for round_num in range(1, self.max_rounds + 1):
            rebuttal_round = await self._rebuttal_round(
                round_num=round_num,
                previous_arguments=self._get_all_arguments(transcript),
                prompt=prompt,
                student_response=student_response,
            )
            transcript.rounds.append(rebuttal_round)

            # Track position changes
            changes = self._detect_position_changes(transcript)
            transcript.position_changes.extend(changes)

            # Check consensus
            consensus = self._check_consensus(rebuttal_round.arguments)
            if consensus.reached:
                break

        # Final resolution
        transcript.final_consensus = self._resolve_debate(transcript)
        transcript.total_duration_seconds = time.time() - start_time

        # Analytics
        transcript.key_disagreements = self._identify_disagreements(transcript)
        transcript.strongest_arguments = self._rank_arguments(transcript)

        return transcript

    async def _opening_round(
        self,
        prompt: str,
        student_response: str,
        dialogue_history: DialogueHistory | None,
    ) -> DebateRound:
        """Get initial positions from all teachers (parallel)."""

        async def get_opening(teacher: BaseTeacher) -> DebateArgument:
            evaluation = await teacher.evaluate(
                prompt=prompt,
                student_response=student_response,
                dialogue_history=dialogue_history,
                generate_improved=False,
            )

            return DebateArgument(
                teacher_id=teacher.name,
                teacher_name=teacher.name,
                position=self._score_to_position(evaluation.overall_score),
                claim=evaluation.reasoning,
                evidence=evaluation.strengths + evaluation.weaknesses,
                cited_dimensions=[ds.dimension for ds in evaluation.dimension_scores],
                confidence=self._calculate_confidence(evaluation),
                references_arguments=[],
            )

        arguments = await asyncio.gather(*[
            get_opening(t) for t in self.teachers
        ])

        return DebateRound(round_number=0, arguments=list(arguments))

    async def _rebuttal_round(
        self,
        round_num: int,
        previous_arguments: list[DebateArgument],
        prompt: str,
        student_response: str,
    ) -> DebateRound:
        """Teachers respond to each other's arguments."""

        async def get_rebuttal(teacher: BaseTeacher) -> DebateArgument:
            # Get arguments from other teachers
            other_arguments = [
                a for a in previous_arguments
                if a.teacher_id != teacher.name
            ]

            rebuttal_prompt = self._format_rebuttal_prompt(
                teacher=teacher,
                prompt=prompt,
                student_response=student_response,
                other_arguments=other_arguments,
            )

            response = await teacher.generate(rebuttal_prompt)
            return self._parse_rebuttal_response(response, teacher)

        arguments = await asyncio.gather(*[
            get_rebuttal(t) for t in self.teachers
        ])

        return DebateRound(round_number=round_num, arguments=list(arguments))

    def _format_rebuttal_prompt(
        self,
        teacher: BaseTeacher,
        prompt: str,
        student_response: str,
        other_arguments: list[DebateArgument],
    ) -> str:
        """Format prompt for rebuttal generation."""

        return f"""You are {teacher.name} in a debate about evaluating a student response.

ORIGINAL PROMPT: {prompt}

STUDENT RESPONSE: {student_response}

OTHER TEACHERS' ARGUMENTS:
{chr(10).join(a.to_prompt_context() for a in other_arguments)}

Your task:
1. Consider the other teachers' points
2. Defend your position OR update it if convinced
3. Identify specific points of agreement/disagreement
4. Cite specific evidence from the student response

Respond in this format:
POSITION: [support/oppose/neutral]
CONFIDENCE: [0.0-1.0]
MAIN_CLAIM: [your central argument]
EVIDENCE:
- [evidence point 1]
- [evidence point 2]
RESPONDING_TO: [which arguments you're addressing]
CONCESSIONS: [any points you now agree with]
"""

    def _check_consensus(self, arguments: list[DebateArgument]) -> ConsensusResult:
        """Check if teachers have reached consensus."""

        scores = [self._position_to_score(a.position) for a in arguments]
        confidences = [a.confidence for a in arguments]

        # Weighted agreement based on confidence
        weighted_scores = [s * c for s, c in zip(scores, confidences)]
        avg_score = sum(weighted_scores) / sum(confidences)

        # Calculate agreement level (inverse of variance)
        variance = sum((s - avg_score) ** 2 for s in scores) / len(scores)
        agreement = 1.0 / (1.0 + variance * 10)  # Scale to 0-1

        if agreement >= self.consensus_threshold:
            return ConsensusResult(
                reached=True,
                final_score=avg_score * 10,  # Scale to 0-10
                agreement_level=agreement,
                dissenting_opinions=[],
                resolution_method="consensus",
            )

        return ConsensusResult(
            reached=False,
            agreement_level=agreement,
            dissenting_opinions=[a for a in arguments if abs(self._position_to_score(a.position) - avg_score) > 0.3],
        )

    def _resolve_debate(self, transcript: DebateTranscript) -> ConsensusResult:
        """Final resolution when consensus not reached."""

        all_arguments = self._get_all_arguments(transcript)

        # Method 1: Weighted average by confidence and argument strength
        final_score = self._weighted_resolution(all_arguments)

        # Synthesize final evaluation
        final_eval = TeacherEvaluation(
            overall_score=final_score,
            reasoning=self._synthesize_reasoning(transcript),
            dimension_scores=self._aggregate_dimensions(all_arguments),
            strengths=self._collect_unique(all_arguments, "strengths"),
            weaknesses=self._collect_unique(all_arguments, "weaknesses"),
            suggestions=self._collect_unique(all_arguments, "suggestions"),
        )

        # Identify remaining disagreements
        dissenting = [
            a for a in all_arguments
            if abs(self._position_to_score(a.position) * 10 - final_score) > 2.0
        ]

        return ConsensusResult(
            reached=True,  # Forced resolution
            final_score=final_score,
            final_evaluation=final_eval,
            agreement_level=transcript.rounds[-1].consensus_check.agreement_level
                if transcript.rounds[-1].consensus_check else 0.5,
            dissenting_opinions=dissenting,
            resolution_method="weighted_synthesis",
        )
```

#### 2.3.3 Integration with CompositeTeacher

```python
class CompositeTeacher(BaseTeacher):
    """Updated to include true debate strategy."""

    async def _debate_evaluation(
        self,
        prompt: str,
        student_response: str,
        dialogue_history: DialogueHistory | None,
        generate_improved: bool,
    ) -> TeacherEvaluation:
        """
        True debate-based evaluation.

        Teachers argue about the evaluation, potentially changing
        their positions based on peer arguments.
        """
        # Create debate engine with current teachers
        debate_engine = DebateEngine(
            teachers=self.teachers,
            max_rounds=self.debate_config.max_rounds,
            consensus_threshold=self.debate_config.consensus_threshold,
        )

        # Run debate
        transcript = await debate_engine.debate(
            prompt=prompt,
            student_response=student_response,
            dialogue_history=dialogue_history,
        )

        # Store transcript for analysis
        self._last_debate_transcript = transcript

        # Generate improved response if requested
        improved = None
        if generate_improved:
            improved = await self._generate_improved_from_debate(
                prompt, student_response, transcript
            )

        eval = transcript.final_consensus.final_evaluation
        eval.improved_response = improved

        return eval
```

### 2.4 Trained Consensus Optimizer (2026 Best Practice)

> **Key Innovation:** Train a consensus optimizer on debate transcripts with known
> ground-truth answers. This is the first learning-based aggregation method in
> MAD frameworks, enabling minority override when a single agent is correct.

```python
class TrainedConsensusOptimizer(nn.Module):
    """
    Learned aggregation for multi-agent debate.

    Instead of simple voting or averaging, this model learns to:
    - Weight arguments by quality, not just quantity
    - Recognize when minority positions are correct
    - Account for agent reasoning patterns

    Training:
    - Dataset: Debate transcripts with ground-truth answers
    - Loss: Reward choosing correct answer, penalize incorrect
    - Features: Argument text, confidence, citations, position changes

    Reference: https://openreview.net/forum?id=Vusd1Hw2D9
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_attention_heads: int = 4,
        encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        super().__init__()

        # Encode arguments
        self.argument_encoder = SentenceTransformer(encoder_model)
        self.embed_dim = self.argument_encoder.get_sentence_embedding_dimension()

        # Cross-argument attention (who responds to whom)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=num_attention_heads,
            batch_first=True,
        )

        # Argument quality scorer
        self.quality_head = nn.Sequential(
            nn.Linear(self.embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Final aggregation
        self.aggregation_head = nn.Sequential(
            nn.Linear(self.embed_dim + 3, hidden_dim),  # +3 for metadata
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        arguments: list[DebateArgument],
        metadata: dict,
    ) -> ConsensusResult:
        """
        Aggregate debate arguments into final decision.

        Args:
            arguments: All arguments from debate
            metadata: {round_num, position_changes, confidence_shifts}

        Returns:
            ConsensusResult with learned aggregation
        """
        # Encode all arguments
        texts = [a.claim for a in arguments]
        embeddings = self.argument_encoder.encode(
            texts, convert_to_tensor=True
        )  # [N, embed_dim]

        # Self-attention across arguments
        attended, attn_weights = self.cross_attention(
            embeddings.unsqueeze(0),
            embeddings.unsqueeze(0),
            embeddings.unsqueeze(0),
        )
        attended = attended.squeeze(0)  # [N, embed_dim]

        # Score argument quality
        quality_scores = self.quality_head(attended).squeeze(-1)  # [N]
        quality_weights = F.softmax(quality_scores, dim=0)

        # Aggregate with quality weighting
        weighted_embedding = (attended * quality_weights.unsqueeze(-1)).sum(dim=0)

        # Add metadata features
        meta_features = torch.tensor([
            metadata.get("num_position_changes", 0) / 10,
            metadata.get("avg_confidence", 0.5),
            metadata.get("rounds", 1) / 5,
        ])
        combined = torch.cat([weighted_embedding, meta_features])

        # Final score prediction
        final_score = torch.sigmoid(self.aggregation_head(combined)) * 10

        # Identify if minority should override
        minority_override = self._check_minority_override(
            arguments, quality_scores, attn_weights
        )

        return ConsensusResult(
            reached=True,
            final_score=final_score.item(),
            agreement_level=1.0 - quality_scores.std().item(),
            resolution_method="trained_consensus",
            minority_override_applied=minority_override,
        )

    def _check_minority_override(
        self,
        arguments: list[DebateArgument],
        quality_scores: torch.Tensor,
        attn_weights: torch.Tensor,
    ) -> bool:
        """
        Check if a minority position should override majority.

        Conditions for override:
        1. Minority argument has significantly higher quality score
        2. Minority argument is well-attended by other arguments
        3. Other arguments reference/concede to minority

        This enables catching cases where one agent is correct
        even when outnumbered.
        """
        # Group by position
        positions = [a.position for a in arguments]
        position_counts = Counter(positions)

        if len(position_counts) < 2:
            return False

        minority_pos = min(position_counts.keys(), key=lambda p: position_counts[p])
        minority_indices = [i for i, a in enumerate(arguments) if a.position == minority_pos]

        # Check if minority has highest quality
        minority_quality = quality_scores[minority_indices].mean()
        majority_quality = quality_scores[[i for i in range(len(arguments)) if i not in minority_indices]].mean()

        return minority_quality > majority_quality * 1.3  # 30% higher triggers override


class RationalityScorer:
    """
    Score arguments for rational behavior.

    Research shows agents following high-quality arguments achieve >90%
    correction rate, while irrational behaviors result in <55% success.

    Rational behaviors to reward:
    - Following high-quality peer arguments
    - Processing peer input effectively
    - Providing evidence-based reasoning
    - Acknowledging valid counterpoints

    Irrational behaviors to penalize:
    - Ignoring peer arguments
    - Repeating same points without addressing rebuttals
    - Confidence without evidence
    - Ad hominem or appeal to authority
    """

    def score_argument(
        self,
        argument: DebateArgument,
        previous_arguments: list[DebateArgument],
    ) -> float:
        """
        Score argument rationality (0-1).

        Higher scores indicate more rational behavior.
        """
        score = 0.5  # Baseline

        # Reward: References previous arguments
        if argument.references_arguments:
            score += 0.1

        # Reward: Provides evidence
        if len(argument.evidence) >= 2:
            score += 0.15

        # Reward: Acknowledges counterpoints
        if any("concede" in e.lower() or "agree" in e.lower() for e in argument.evidence):
            score += 0.1

        # Reward: Cites specific dimensions
        if argument.cited_dimensions:
            score += 0.05 * len(argument.cited_dimensions)

        # Penalize: High confidence without evidence
        if argument.confidence > 0.8 and len(argument.evidence) < 2:
            score -= 0.2

        # Penalize: No engagement with previous arguments
        if previous_arguments and not argument.references_arguments:
            score -= 0.15

        return max(0.0, min(1.0, score))
```

### 2.5 Student Observation Mode

```python
class DebateObserver:
    """
    Allows student to observe teacher debates.

    The student watches teachers argue, learning:
    - How to evaluate responses
    - Multiple perspectives on quality
    - How to handle disagreement
    - Evidence-based reasoning
    """

    def format_debate_for_student(
        self,
        transcript: DebateTranscript,
        include_resolution: bool = True,
    ) -> str:
        """
        Format debate transcript as training data for student.

        This teaches the student to:
        1. See multiple evaluation perspectives
        2. Understand how arguments are constructed
        3. Learn when to update beliefs
        """
        formatted = f"""[TEACHER DEBATE OBSERVATION]

Topic: Evaluating response to "{transcript.prompt[:100]}..."

"""
        for round in transcript.rounds:
            formatted += f"\n=== Round {round.round_number} ===\n"
            for arg in round.arguments:
                formatted += f"\n{arg.teacher_name} ({arg.position}):\n"
                formatted += f'"{arg.claim}"\n'
                if arg.references_arguments:
                    formatted += f"(Responding to previous arguments)\n"

        if include_resolution:
            formatted += f"\n=== RESOLUTION ===\n"
            formatted += f"Final score: {transcript.final_consensus.final_score:.1f}/10\n"
            formatted += f"Agreement level: {transcript.final_consensus.agreement_level:.0%}\n"

            if transcript.final_consensus.dissenting_opinions:
                formatted += "\nDissenting views:\n"
                for dissent in transcript.final_consensus.dissenting_opinions:
                    formatted += f"- {dissent.teacher_name}: {dissent.claim[:100]}...\n"

        return formatted
```

---

## 3. Critic-Only Inference API

> **2026 Best Practices Applied:**
> - [Test-Time Scaling](https://testtimescaling.github.io/): Inference-time compute will drive 2026 progress
> - [ThinkPRM](https://arxiv.org/abs/2504.16828): Process Reward Models with verification chain-of-thought
> - [Critic-CoT](https://www.emergentmind.com/topics/critic-cot-framework): Coupled reasoning with explicit critic interventions
> - [SCRPO](https://arxiv.org/html/2512.05387v2): Distill inference-time refinement into training
> - See [Section 0.4](#04-critic-guided-self-refinement-2026-inference-scaling) for full research background.

### 3.1 Problem Statement

Currently, using ASPIRE's trained critic requires:
1. Loading the full trainer
2. Having the student model loaded
3. Complex setup for inference

Users want a simple API:
```python
response = generate_with_judgment("Explain entropy.")
```

### 3.2 Proposed Architecture

```
aspire/
├── inference/
│   ├── __init__.py
│   ├── api.py              # High-level API (generate_with_judgment)
│   ├── critic_runtime.py   # Standalone critic loading
│   ├── refinement.py       # Self-refinement loop
│   └── serving.py          # FastAPI server (optional)
```

### 3.3 Core Components

#### 3.3.1 High-Level API

```python
# aspire/inference/api.py

from typing import Callable, Iterator
from dataclasses import dataclass

@dataclass
class JudgedResponse:
    """Response with critic judgment."""
    text: str
    score: float  # 0-10
    reasoning_embedding: torch.Tensor | None
    dimension_scores: dict[str, float] | None

    # Refinement metadata
    num_refinements: int
    refinement_history: list[RefinementStep] | None

    # Confidence
    confidence: float  # Based on critic certainty


@dataclass
class RefinementStep:
    """One step in self-refinement."""
    attempt: int
    text: str
    score: float
    feedback: str | None


class AspireInference:
    """
    High-level inference API for ASPIRE-trained models.

    Provides:
    - Simple generation with critic scoring
    - Self-refinement loops
    - Streaming with live scores
    - Batch processing
    """

    def __init__(
        self,
        model_path: str,
        critic_path: str | None = None,
        device: str = "cuda",
        load_in_4bit: bool = True,
    ):
        self.model, self.tokenizer = self._load_model(model_path, load_in_4bit)
        self.critic = self._load_critic(critic_path or f"{model_path}/critic.pt")
        self.device = device

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        """Simple generation without judgment."""
        # Standard generation
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=max_tokens, temperature=temperature)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def judge(
        self,
        prompt: str,
        response: str,
    ) -> CriticOutput:
        """Score a response using the critic."""
        # Encode prompt + response
        text = f"{prompt}\n\n{response}"
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        # Get hidden states
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]

        # Critic scoring
        critic_output = self.critic(hidden_states=hidden_states)

        return critic_output

    def generate_with_judgment(
        self,
        prompt: str,
        min_score: float = 7.0,
        max_refinements: int = 3,
        temperature: float = 0.7,
        refinement_temperature: float = 0.9,  # Higher for diversity
        return_history: bool = False,
    ) -> JudgedResponse:
        """
        Generate response and refine until critic is satisfied.

        The self-refinement loop:
        1. Generate initial response
        2. Score with critic
        3. If score < min_score, generate refinement
        4. Repeat until satisfied or max_refinements
        """
        history = []

        # Initial generation
        response = self.generate(prompt, temperature=temperature)
        critic_output = self.judge(prompt, response)

        history.append(RefinementStep(
            attempt=0,
            text=response,
            score=critic_output.score.item(),
            feedback=None,
        ))

        # Refinement loop
        for i in range(max_refinements):
            if critic_output.score.item() >= min_score:
                break

            # Generate feedback prompt
            feedback = self._generate_feedback(prompt, response, critic_output)

            # Refine
            refinement_prompt = f"""{prompt}

Previous attempt: {response}

Feedback: {feedback}

Improved response:"""

            response = self.generate(
                refinement_prompt,
                temperature=refinement_temperature
            )
            critic_output = self.judge(prompt, response)

            history.append(RefinementStep(
                attempt=i + 1,
                text=response,
                score=critic_output.score.item(),
                feedback=feedback,
            ))

        return JudgedResponse(
            text=response,
            score=critic_output.score.item(),
            reasoning_embedding=critic_output.reasoning_embedding,
            dimension_scores=self._extract_dimension_scores(critic_output),
            num_refinements=len(history) - 1,
            refinement_history=history if return_history else None,
            confidence=self._calculate_confidence(critic_output),
        )

    def stream_with_judgment(
        self,
        prompt: str,
        min_score: float = 7.0,
    ) -> Iterator[tuple[str, float | None]]:
        """
        Stream generation with periodic scoring.

        Yields (token, score) pairs where score is updated
        periodically as generation progresses.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        generated_ids = inputs["input_ids"].clone()
        current_score = None
        tokens_since_score = 0
        score_interval = 20  # Score every N tokens

        for _ in range(512):  # Max tokens
            with torch.no_grad():
                outputs = self.model(
                    input_ids=generated_ids,
                    output_hidden_states=True,
                )

                # Sample next token
                next_token_logits = outputs.logits[:, -1, :]
                next_token = torch.multinomial(
                    torch.softmax(next_token_logits / 0.7, dim=-1),
                    num_samples=1
                )

                generated_ids = torch.cat([generated_ids, next_token], dim=-1)

                # Decode token
                token_text = self.tokenizer.decode(next_token[0])

                # Periodic scoring
                tokens_since_score += 1
                if tokens_since_score >= score_interval:
                    hidden_states = outputs.hidden_states[-1]
                    critic_output = self.critic(hidden_states=hidden_states)
                    current_score = critic_output.score.item()
                    tokens_since_score = 0

                yield token_text, current_score

                # Stop on EOS
                if next_token[0].item() == self.tokenizer.eos_token_id:
                    break

        # Final score
        with torch.no_grad():
            outputs = self.model(input_ids=generated_ids, output_hidden_states=True)
            critic_output = self.critic(hidden_states=outputs.hidden_states[-1])
            yield "", critic_output.score.item()


# Convenience function
_default_inference: AspireInference | None = None

def generate_with_judgment(
    prompt: str,
    model_path: str | None = None,
    min_score: float = 7.0,
    max_refinements: int = 3,
    **kwargs,
) -> JudgedResponse:
    """
    One-liner for ASPIRE inference.

    Usage:
        from aspire import generate_with_judgment

        response = generate_with_judgment("Explain entropy.")
        print(f"Score: {response.score}/10")
        print(response.text)
    """
    global _default_inference

    if _default_inference is None:
        if model_path is None:
            raise ValueError("Must provide model_path on first call")
        _default_inference = AspireInference(model_path)

    return _default_inference.generate_with_judgment(
        prompt=prompt,
        min_score=min_score,
        max_refinements=max_refinements,
        **kwargs,
    )
```

#### 3.3.2 FastAPI Server

```python
# aspire/inference/serving.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="ASPIRE Inference API")

class GenerateRequest(BaseModel):
    prompt: str
    min_score: float = 7.0
    max_refinements: int = 3
    temperature: float = 0.7
    return_history: bool = False

class JudgeRequest(BaseModel):
    prompt: str
    response: str

class GenerateResponse(BaseModel):
    text: str
    score: float
    num_refinements: int
    confidence: float
    dimension_scores: Optional[dict[str, float]] = None
    refinement_history: Optional[list[dict]] = None

class JudgeResponse(BaseModel):
    score: float
    dimension_scores: Optional[dict[str, float]] = None


inference: AspireInference | None = None

@app.on_event("startup")
async def startup():
    global inference
    import os
    model_path = os.environ.get("ASPIRE_MODEL_PATH", "./aspire-model")
    inference = AspireInference(model_path)

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate response with self-refinement."""
    result = inference.generate_with_judgment(
        prompt=request.prompt,
        min_score=request.min_score,
        max_refinements=request.max_refinements,
        temperature=request.temperature,
        return_history=request.return_history,
    )

    return GenerateResponse(
        text=result.text,
        score=result.score,
        num_refinements=result.num_refinements,
        confidence=result.confidence,
        dimension_scores=result.dimension_scores,
        refinement_history=[
            {"attempt": r.attempt, "text": r.text, "score": r.score}
            for r in (result.refinement_history or [])
        ],
    )

@app.post("/judge", response_model=JudgeResponse)
async def judge(request: JudgeRequest):
    """Score an existing response."""
    result = inference.judge(request.prompt, request.response)

    return JudgeResponse(
        score=result.score.item(),
        dimension_scores=inference._extract_dimension_scores(result),
    )

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": inference is not None}


# CLI entry point
def serve(host: str = "0.0.0.0", port: int = 8000):
    """Run the inference server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)
```

### 3.4 Process Reward Model Integration (2026 Best Practice)

> **ThinkPRM approach**: Instead of discriminative scoring, generate verification
> chain-of-thought. This requires only 1% of the process labels while achieving
> superior results on ProcessBench, MATH-500, and AIME benchmarks.

```python
class ThinkPRMCritic:
    """
    Process Reward Model that generates verification chain-of-thought.

    Instead of just outputting a score, ThinkPRM verbalizes its step-by-step
    verification of each reasoning step. This approach:
    - Requires 1% of the training labels vs discriminative PRMs
    - Provides interpretable verification reasoning
    - Outperforms LLM-as-Judge on challenging benchmarks

    Reference: https://arxiv.org/abs/2504.16828
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
    ):
        self.model, self.tokenizer = self._load_model(model_path)
        self.device = device

    def verify_step(
        self,
        problem: str,
        solution_step: str,
        previous_steps: list[str],
    ) -> StepVerification:
        """
        Verify a single reasoning step.

        Returns verification with:
        - is_correct: bool
        - verification_cot: str (chain of thought explaining verification)
        - error_type: str | None (if incorrect)
        - suggested_fix: str | None
        """
        prompt = self._format_verification_prompt(
            problem, solution_step, previous_steps
        )

        # Generate verification chain-of-thought
        verification_cot = self.model.generate(prompt, max_tokens=512)

        # Parse verification result from CoT
        return self._parse_verification(verification_cot)

    def verify_solution(
        self,
        problem: str,
        solution: str,
    ) -> SolutionVerification:
        """
        Verify entire solution step-by-step.

        Returns:
        - overall_correct: bool
        - step_verifications: list of per-step results
        - first_error_step: int | None
        - verification_summary: str
        """
        # Split solution into steps
        steps = self._extract_steps(solution)

        verifications = []
        for i, step in enumerate(steps):
            verification = self.verify_step(
                problem=problem,
                solution_step=step,
                previous_steps=steps[:i],
            )
            verifications.append(verification)

            # Early termination on error (optional)
            if not verification.is_correct:
                break

        return SolutionVerification(
            overall_correct=all(v.is_correct for v in verifications),
            step_verifications=verifications,
            first_error_step=next(
                (i for i, v in enumerate(verifications) if not v.is_correct),
                None
            ),
            verification_summary=self._summarize_verifications(verifications),
        )


class ComputeOptimalInference:
    """
    Compute-optimal test-time scaling.

    Uses question difficulty to predict how much test-time compute
    will be beneficial, avoiding wasted computation on easy questions
    and under-computing on hard ones.

    Reference: https://arxiv.org/abs/2408.03314
    """

    def __init__(
        self,
        model: AspireInference,
        difficulty_estimator: DifficultyEstimator | None = None,
    ):
        self.model = model
        self.difficulty_estimator = difficulty_estimator or PerplexityDifficultyEstimator()

    def generate_optimal(
        self,
        prompt: str,
        compute_budget: float = 1.0,  # Relative compute multiplier
    ) -> JudgedResponse:
        """
        Generate with compute-optimal allocation.

        Strategy based on 2026 research:
        - Easy questions: Minimal refinement (1-2 attempts)
        - Medium questions: Standard refinement (3-5 attempts)
        - Hard questions: Extended refinement + parallel sampling

        This achieves equivalent quality to best-of-N with 4x less compute.
        """
        # Estimate difficulty
        difficulty = self.difficulty_estimator.estimate(prompt, self.model)

        # Allocate compute based on difficulty
        if difficulty < 0.3:  # Easy
            max_refinements = 1
            parallel_samples = 1
            use_prm = False
        elif difficulty < 0.7:  # Medium
            max_refinements = int(3 * compute_budget)
            parallel_samples = 1
            use_prm = True
        else:  # Hard
            max_refinements = int(5 * compute_budget)
            parallel_samples = min(4, int(2 * compute_budget))
            use_prm = True

        if parallel_samples > 1:
            # Parallel sampling with PRM-guided selection
            responses = self._parallel_generate(
                prompt,
                n=parallel_samples,
                max_refinements=max_refinements,
            )
            # Select best using PRM or majority vote
            return self._select_best(responses, use_prm=use_prm)
        else:
            return self.model.generate_with_judgment(
                prompt=prompt,
                max_refinements=max_refinements,
            )

    def _parallel_generate(
        self,
        prompt: str,
        n: int,
        max_refinements: int,
    ) -> list[JudgedResponse]:
        """Generate n responses in parallel with diverse sampling."""
        # Use different temperatures for diversity
        temperatures = [0.5 + 0.2 * i for i in range(n)]

        responses = []
        for temp in temperatures:
            response = self.model.generate_with_judgment(
                prompt=prompt,
                max_refinements=max_refinements,
                temperature=temp,
            )
            responses.append(response)

        return responses

    def _select_best(
        self,
        responses: list[JudgedResponse],
        use_prm: bool = True,
    ) -> JudgedResponse:
        """
        Select best response using PRM or Shortest Majority Vote.

        2026 insight: Longer reasoning isn't always better.
        Shortest Majority Vote often outperforms just picking the longest.
        """
        if use_prm and self.prm is not None:
            # PRM-guided selection
            scored = [(self.prm.score(r.text), r) for r in responses]
            return max(scored, key=lambda x: x[0])[1]
        else:
            # Shortest Majority Vote
            # Group by answer, pick shortest from majority group
            answer_groups = defaultdict(list)
            for r in responses:
                answer = self._extract_answer(r.text)
                answer_groups[answer].append(r)

            # Find majority answer
            majority_answer = max(answer_groups.keys(), key=lambda a: len(answer_groups[a]))

            # Return shortest response with majority answer
            majority_responses = answer_groups[majority_answer]
            return min(majority_responses, key=lambda r: len(r.text))
```

### 3.5 CLI Integration

```python
# Add to aspire/cli.py

@app.command()
def infer(
    prompt: str = typer.Argument(..., help="Prompt to generate from"),
    model_path: str = typer.Option("./output", help="Path to trained model"),
    min_score: float = typer.Option(7.0, help="Minimum acceptable score"),
    max_refinements: int = typer.Option(3, help="Max refinement attempts"),
    show_history: bool = typer.Option(False, help="Show refinement history"),
):
    """Generate with critic-guided self-refinement."""
    from aspire.inference import AspireInference

    console.print(f"[blue]Loading model from {model_path}...[/blue]")
    inference = AspireInference(model_path)

    console.print(f"[blue]Generating response...[/blue]")
    result = inference.generate_with_judgment(
        prompt=prompt,
        min_score=min_score,
        max_refinements=max_refinements,
        return_history=show_history,
    )

    console.print(Panel(result.text, title="Response"))
    console.print(f"Score: [green]{result.score:.1f}/10[/green]")
    console.print(f"Refinements: {result.num_refinements}")
    console.print(f"Confidence: {result.confidence:.0%}")

    if show_history and result.refinement_history:
        console.print("\n[yellow]Refinement History:[/yellow]")
        for step in result.refinement_history:
            console.print(f"  Attempt {step.attempt}: {step.score:.1f}/10")


@app.command()
def serve(
    model_path: str = typer.Option("./output", help="Path to trained model"),
    host: str = typer.Option("0.0.0.0", help="Host to bind to"),
    port: int = typer.Option(8000, help="Port to listen on"),
):
    """Start the inference API server."""
    import os
    os.environ["ASPIRE_MODEL_PATH"] = model_path

    from aspire.inference.serving import serve as run_server
    console.print(f"[green]Starting server at http://{host}:{port}[/green]")
    run_server(host=host, port=port)
```

---

## 4. Trajectory Visualizer

> **2026 Best Practices Applied:**
> - [LIDA](https://microsoft.github.io/lida/): Grammar-agnostic visualization generation from natural language
> - [LLM Observability 2026](https://lakefs.io/blog/llm-observability-tools/): Interpretability through attention attribution
> - [TransformerLens](https://github.com/JShollaj/awesome-llm-interpretability): Mechanistic interpretability patterns
> - See [Section 0.6](#06-llm-observability-and-visualization) for full research background.

### 4.1 Problem Statement

ASPIRE generates rich data during training:
- Dialogue trees with challenges/responses
- Critic scores over time
- Reasoning evolution across turns
- Refinement steps

This data is currently logged but not visualized, making it hard to:
- Understand model learning dynamics
- Debug training issues
- Present results to stakeholders
- Analyze specific dialogues

### 4.2 Proposed Architecture

```
aspire/visualization/
├── __init__.py
├── dashboard.py           # Main Gradio/Streamlit dashboard
├── dialogue_tree.py       # Interactive dialogue visualization
├── score_curves.py        # Critic score over training
├── reasoning_deltas.py    # How reasoning changes
├── export.py              # Export to various formats
└── components/
    ├── tree_renderer.py   # D3.js tree visualization
    ├── chart_builder.py   # Plotly charts
    └── diff_viewer.py     # Response diff highlighting
```

### 4.3 Core Components

#### 4.3.1 Data Structures for Visualization

```python
@dataclass
class TrajectoryData:
    """Complete trajectory for visualization."""
    dialogue_id: str
    prompt: str

    # Tree structure
    nodes: list[TrajectoryNode]
    edges: list[TrajectoryEdge]

    # Score evolution
    score_curve: list[ScorePoint]
    dimension_curves: dict[str, list[ScorePoint]]

    # Reasoning analysis
    reasoning_deltas: list[ReasoningDelta]

    # Metadata
    teacher_name: str
    num_turns: int
    total_tokens: int
    duration_seconds: float


@dataclass
class TrajectoryNode:
    """Node in dialogue tree."""
    id: str
    type: str  # "prompt", "challenge", "response", "evaluation"
    content: str
    score: float | None

    # Position in tree
    depth: int
    parent_id: str | None

    # Styling
    color: str | None = None
    size: float = 1.0


@dataclass
class ScorePoint:
    """Point on score curve."""
    turn: int
    score: float
    confidence: float | None = None
    event: str | None = None  # "challenge", "refinement", etc.


@dataclass
class ReasoningDelta:
    """Change in reasoning between turns."""
    from_turn: int
    to_turn: int

    # Text changes
    added_concepts: list[str]
    removed_concepts: list[str]
    modified_claims: list[tuple[str, str]]  # (before, after)

    # Semantic shift
    embedding_distance: float
    semantic_drift_direction: str  # "more_specific", "more_general", etc.
```

#### 4.3.2 Trajectory Extractor

```python
class TrajectoryExtractor:
    """Extracts visualization data from dialogues."""

    def extract(self, dialogue: GeneratedDialogue) -> TrajectoryData:
        """Convert dialogue to visualization format."""
        nodes = []
        edges = []
        score_curve = []

        # Root: prompt
        prompt_node = TrajectoryNode(
            id="prompt",
            type="prompt",
            content=dialogue.prompt,
            score=None,
            depth=0,
            parent_id=None,
        )
        nodes.append(prompt_node)

        # Initial response
        initial_node = TrajectoryNode(
            id="initial",
            type="response",
            content=dialogue.initial_response,
            score=dialogue.turn_evaluations[0].overall_score if dialogue.turn_evaluations else None,
            depth=1,
            parent_id="prompt",
        )
        nodes.append(initial_node)
        edges.append(TrajectoryEdge(source="prompt", target="initial"))

        # Dialogue turns
        prev_id = "initial"
        for i, turn in enumerate(dialogue.history.turns):
            # Challenge node
            challenge_id = f"challenge_{i}"
            challenge_node = TrajectoryNode(
                id=challenge_id,
                type="challenge",
                content=turn.challenge.content,
                score=None,
                depth=2 + i * 2,
                parent_id=prev_id,
                color=self._challenge_type_color(turn.challenge.challenge_type),
            )
            nodes.append(challenge_node)
            edges.append(TrajectoryEdge(source=prev_id, target=challenge_id))

            # Response node
            response_id = f"response_{i}"
            response_node = TrajectoryNode(
                id=response_id,
                type="response",
                content=turn.student_response,
                score=turn.evaluation.overall_score if turn.evaluation else None,
                depth=3 + i * 2,
                parent_id=challenge_id,
            )
            nodes.append(response_node)
            edges.append(TrajectoryEdge(source=challenge_id, target=response_id))

            # Score curve
            if turn.evaluation:
                score_curve.append(ScorePoint(
                    turn=i + 1,
                    score=turn.evaluation.overall_score,
                    event="challenge",
                ))

            prev_id = response_id

        # Final evaluation
        if dialogue.final_evaluation:
            eval_node = TrajectoryNode(
                id="final_eval",
                type="evaluation",
                content=dialogue.final_evaluation.reasoning,
                score=dialogue.final_evaluation.overall_score,
                depth=len(nodes),
                parent_id=prev_id,
                color="green" if dialogue.final_evaluation.overall_score >= 7 else "orange",
            )
            nodes.append(eval_node)

            score_curve.append(ScorePoint(
                turn=len(dialogue.history.turns) + 1,
                score=dialogue.final_evaluation.overall_score,
                event="final",
            ))

        # Extract reasoning deltas
        reasoning_deltas = self._extract_reasoning_deltas(dialogue)

        return TrajectoryData(
            dialogue_id=dialogue.metadata.get("id", "unknown"),
            prompt=dialogue.prompt,
            nodes=nodes,
            edges=edges,
            score_curve=score_curve,
            dimension_curves=self._extract_dimension_curves(dialogue),
            reasoning_deltas=reasoning_deltas,
            teacher_name=dialogue.metadata.get("teacher", "unknown"),
            num_turns=len(dialogue.history.turns),
            total_tokens=self._count_tokens(dialogue),
            duration_seconds=dialogue.metadata.get("duration", 0),
        )

    def _extract_reasoning_deltas(
        self,
        dialogue: GeneratedDialogue
    ) -> list[ReasoningDelta]:
        """Analyze how reasoning changes across turns."""
        deltas = []

        responses = [dialogue.initial_response]
        responses.extend([t.student_response for t in dialogue.history.turns])

        for i in range(len(responses) - 1):
            delta = self._compute_delta(responses[i], responses[i + 1])
            delta.from_turn = i
            delta.to_turn = i + 1
            deltas.append(delta)

        return deltas

    def _compute_delta(self, before: str, after: str) -> ReasoningDelta:
        """Compute semantic delta between responses."""
        # Extract concepts (simple keyword extraction)
        before_concepts = set(self._extract_concepts(before))
        after_concepts = set(self._extract_concepts(after))

        return ReasoningDelta(
            from_turn=0,
            to_turn=0,
            added_concepts=list(after_concepts - before_concepts),
            removed_concepts=list(before_concepts - after_concepts),
            modified_claims=self._find_modified_claims(before, after),
            embedding_distance=self._compute_embedding_distance(before, after),
            semantic_drift_direction=self._classify_drift(before, after),
        )
```

#### 4.3.3 Gradio Dashboard

```python
# aspire/visualization/dashboard.py

import gradio as gr
import plotly.graph_objects as go
import plotly.express as px
from typing import Optional

class AspireDashboard:
    """
    Interactive visualization dashboard for ASPIRE training.

    Features:
    - Dialogue tree explorer
    - Score curves over training
    - Reasoning delta analysis
    - Training metrics overview
    """

    def __init__(
        self,
        checkpoint_dir: str,
        dialogue_cache_dir: str,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.dialogue_cache_dir = dialogue_cache_dir
        self.extractor = TrajectoryExtractor()

        # Load data
        self.dialogues = self._load_dialogues()
        self.training_metrics = self._load_metrics()

    def create_app(self) -> gr.Blocks:
        """Create the Gradio application."""

        with gr.Blocks(title="ASPIRE Trajectory Visualizer") as app:
            gr.Markdown("# ASPIRE Training Visualizer")

            with gr.Tabs():
                # Tab 1: Dialogue Explorer
                with gr.Tab("Dialogue Explorer"):
                    self._create_dialogue_tab()

                # Tab 2: Score Curves
                with gr.Tab("Score Analysis"):
                    self._create_score_tab()

                # Tab 3: Reasoning Deltas
                with gr.Tab("Reasoning Evolution"):
                    self._create_reasoning_tab()

                # Tab 4: Training Overview
                with gr.Tab("Training Metrics"):
                    self._create_metrics_tab()

        return app

    def _create_dialogue_tab(self):
        """Dialogue tree visualization."""

        with gr.Row():
            with gr.Column(scale=1):
                dialogue_selector = gr.Dropdown(
                    choices=[d.prompt[:50] + "..." for d in self.dialogues],
                    label="Select Dialogue",
                )

                gr.Markdown("### Dialogue Info")
                info_text = gr.Markdown()

            with gr.Column(scale=3):
                tree_plot = gr.Plot(label="Dialogue Tree")

                with gr.Accordion("Turn Details", open=False):
                    turn_content = gr.Markdown()

        def update_tree(selection_idx):
            if selection_idx is None:
                return None, "", ""

            dialogue = self.dialogues[selection_idx]
            trajectory = self.extractor.extract(dialogue)

            # Create tree visualization
            fig = self._create_tree_figure(trajectory)

            # Info text
            info = f"""
**Teacher:** {trajectory.teacher_name}
**Turns:** {trajectory.num_turns}
**Final Score:** {trajectory.score_curve[-1].score:.1f}/10
**Tokens:** {trajectory.total_tokens}
"""

            return fig, info, ""

        dialogue_selector.change(
            update_tree,
            inputs=[dialogue_selector],
            outputs=[tree_plot, info_text, turn_content],
        )

    def _create_tree_figure(self, trajectory: TrajectoryData) -> go.Figure:
        """Create interactive tree visualization."""

        # Calculate positions
        positions = self._calculate_tree_positions(trajectory.nodes)

        # Create edges
        edge_x, edge_y = [], []
        for edge in trajectory.edges:
            x0, y0 = positions[edge.source]
            x1, y1 = positions[edge.target]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='#888'),
            hoverinfo='none',
            mode='lines',
        )

        # Create nodes
        node_x = [positions[n.id][0] for n in trajectory.nodes]
        node_y = [positions[n.id][1] for n in trajectory.nodes]
        node_colors = [self._get_node_color(n) for n in trajectory.nodes]
        node_sizes = [20 + (n.score or 5) * 3 for n in trajectory.nodes]
        node_text = [self._get_node_hover(n) for n in trajectory.nodes]

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=[n.type[0].upper() for n in trajectory.nodes],
            textposition='middle center',
            hovertext=node_text,
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=2, color='white'),
            ),
        )

        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                showlegend=False,
                hovermode='closest',
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='white',
            )
        )

        return fig

    def _create_score_tab(self):
        """Score curve visualization."""

        with gr.Row():
            epoch_slider = gr.Slider(
                minimum=1, maximum=10, value=1, step=1,
                label="Epoch",
            )
            dimension_selector = gr.CheckboxGroup(
                choices=["Overall", "Correctness", "Reasoning", "Clarity", "Nuance"],
                value=["Overall"],
                label="Dimensions",
            )

        score_plot = gr.Plot(label="Score Evolution")

        with gr.Row():
            histogram_plot = gr.Plot(label="Score Distribution")
            improvement_plot = gr.Plot(label="Improvement per Turn")

        def update_score_plots(epoch, dimensions):
            # Filter dialogues by epoch
            epoch_dialogues = [d for d in self.dialogues if d.metadata.get("epoch") == epoch]

            # Score evolution
            fig1 = self._create_score_curve(epoch_dialogues, dimensions)

            # Histogram
            scores = [d.final_evaluation.overall_score for d in epoch_dialogues if d.final_evaluation]
            fig2 = px.histogram(x=scores, nbins=20, title="Score Distribution")

            # Improvement
            improvements = []
            for d in epoch_dialogues:
                if len(d.turn_evaluations) >= 2:
                    first = d.turn_evaluations[0].overall_score if d.turn_evaluations[0] else 5
                    last = d.final_evaluation.overall_score if d.final_evaluation else 5
                    improvements.append(last - first)

            fig3 = px.histogram(x=improvements, nbins=20, title="Score Improvement")

            return fig1, fig2, fig3

        epoch_slider.change(
            update_score_plots,
            inputs=[epoch_slider, dimension_selector],
            outputs=[score_plot, histogram_plot, improvement_plot],
        )

    def _create_reasoning_tab(self):
        """Reasoning delta visualization."""

        dialogue_selector = gr.Dropdown(
            choices=[d.prompt[:50] + "..." for d in self.dialogues],
            label="Select Dialogue",
        )

        with gr.Row():
            delta_plot = gr.Plot(label="Reasoning Delta")
            concept_flow = gr.Plot(label="Concept Flow")

        with gr.Accordion("Detailed Changes"):
            diff_view = gr.HTML()

        def update_reasoning(selection_idx):
            if selection_idx is None:
                return None, None, ""

            dialogue = self.dialogues[selection_idx]
            trajectory = self.extractor.extract(dialogue)

            # Delta visualization
            fig1 = self._create_delta_chart(trajectory.reasoning_deltas)

            # Concept flow (Sankey-like)
            fig2 = self._create_concept_flow(trajectory)

            # Diff view
            html = self._create_diff_html(dialogue)

            return fig1, fig2, html

        dialogue_selector.change(
            update_reasoning,
            inputs=[dialogue_selector],
            outputs=[delta_plot, concept_flow, diff_view],
        )

    def _create_metrics_tab(self):
        """Training metrics overview."""

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Training Progress")
                loss_plot = gr.Plot()

            with gr.Column():
                gr.Markdown("### Critic Accuracy")
                critic_plot = gr.Plot()

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Score by Teacher")
                teacher_plot = gr.Plot()

            with gr.Column():
                gr.Markdown("### Curriculum Progress")
                curriculum_plot = gr.Plot()

        # Auto-populate on load
        def load_metrics():
            metrics = self.training_metrics

            fig1 = px.line(
                x=metrics["steps"],
                y=metrics["loss"],
                title="Training Loss",
            )

            fig2 = px.line(
                x=metrics["steps"],
                y=metrics["critic_accuracy"],
                title="Critic Accuracy",
            )

            fig3 = px.box(
                x=metrics["teacher_names"],
                y=metrics["teacher_scores"],
                title="Score by Teacher",
            )

            fig4 = self._create_curriculum_progress()

            return fig1, fig2, fig3, fig4

        app.load(
            load_metrics,
            outputs=[loss_plot, critic_plot, teacher_plot, curriculum_plot],
        )

    def launch(self, share: bool = False, port: int = 7860):
        """Launch the dashboard."""
        app = self.create_app()
        app.launch(share=share, server_port=port)


# CLI entry point
def visualize(
    checkpoint_dir: str = "./output",
    cache_dir: str = "./dialogue_cache",
    port: int = 7860,
    share: bool = False,
):
    """Launch the trajectory visualizer."""
    dashboard = AspireDashboard(checkpoint_dir, cache_dir)
    dashboard.launch(share=share, port=port)
```

#### 4.3.4 Export Functionality

```python
# aspire/visualization/export.py

class TrajectoryExporter:
    """Export trajectories to various formats."""

    def to_html(
        self,
        trajectory: TrajectoryData,
        output_path: str,
        include_charts: bool = True,
    ):
        """Export as standalone HTML report."""
        # ... generate HTML with embedded charts

    def to_json(
        self,
        trajectory: TrajectoryData,
        output_path: str,
    ):
        """Export as JSON for external tools."""
        # ... serialize to JSON

    def to_mermaid(
        self,
        trajectory: TrajectoryData,
    ) -> str:
        """Export dialogue tree as Mermaid diagram."""
        lines = ["graph TD"]

        for node in trajectory.nodes:
            label = node.content[:30].replace('"', "'")
            lines.append(f'    {node.id}["{node.type}: {label}..."]')

        for edge in trajectory.edges:
            lines.append(f"    {edge.source} --> {edge.target}")

        return "\n".join(lines)

    def to_wandb(
        self,
        trajectory: TrajectoryData,
        run: "wandb.Run",
    ):
        """Log trajectory to Weights & Biases."""
        import wandb

        # Log tree as HTML
        run.log({
            "dialogue_tree": wandb.Html(self.to_html(trajectory, None)),
            "score_curve": wandb.plot.line_series(
                xs=[p.turn for p in trajectory.score_curve],
                ys=[[p.score for p in trajectory.score_curve]],
                keys=["Score"],
                title="Score Evolution",
            ),
        })
```

### 4.4 CLI Integration

```python
# Add to aspire/cli.py

@app.command()
def visualize(
    checkpoint_dir: str = typer.Option("./output", help="Checkpoint directory"),
    cache_dir: str = typer.Option("./dialogue_cache", help="Dialogue cache"),
    port: int = typer.Option(7860, help="Port for dashboard"),
    share: bool = typer.Option(False, help="Create public link"),
):
    """Launch the trajectory visualizer dashboard."""
    from aspire.visualization.dashboard import AspireDashboard

    console.print(f"[green]Launching visualizer at http://localhost:{port}[/green]")
    dashboard = AspireDashboard(checkpoint_dir, cache_dir)
    dashboard.launch(share=share, port=port)


@app.command()
def export_dialogue(
    dialogue_id: str = typer.Argument(..., help="Dialogue ID to export"),
    format: str = typer.Option("html", help="Export format (html, json, mermaid)"),
    output: str = typer.Option(None, help="Output path"),
):
    """Export a dialogue trajectory."""
    from aspire.visualization.export import TrajectoryExporter

    # Load dialogue
    dialogue = load_dialogue(dialogue_id)
    trajectory = TrajectoryExtractor().extract(dialogue)

    exporter = TrajectoryExporter()

    if format == "html":
        exporter.to_html(trajectory, output or f"{dialogue_id}.html")
    elif format == "json":
        exporter.to_json(trajectory, output or f"{dialogue_id}.json")
    elif format == "mermaid":
        mermaid = exporter.to_mermaid(trajectory)
        if output:
            Path(output).write_text(mermaid)
        else:
            console.print(mermaid)
```

---

## 5. Implementation Roadmap

### Phase 1: Foundation (Week 1-2)
- [ ] Curriculum Engine core (`engine.py`, `difficulty.py`)
- [ ] Spaced repetition scheduler
- [ ] Critic-Only Inference API (`api.py`)

### Phase 2: Debate System (Week 3-4)
- [ ] Debate data structures
- [ ] DebateEngine implementation
- [ ] Integration with CompositeTeacher
- [ ] Student observation mode

### Phase 3: Visualization (Week 5-6)
- [ ] Trajectory extraction
- [ ] Gradio dashboard
- [ ] Score curve visualization
- [ ] Reasoning delta analysis

### Phase 4: Advanced Features (Week 7-8)
- [ ] Adversarial curriculum generator
- [ ] Concept scaffolding graph
- [ ] FastAPI inference server
- [ ] Export functionality
- [ ] W&B integration

### Phase 5: Testing & Documentation (Week 9-10)
- [ ] Unit tests for all components
- [ ] Integration tests
- [ ] Documentation
- [ ] Example notebooks

---

## 6. Dependencies

### New Dependencies
```toml
[project.optional-dependencies]
visualization = [
    "gradio>=4.0",
    "plotly>=5.0",
    "networkx>=3.0",
    "transformerlens>=1.0",  # For mechanistic interpretability
]
serving = [
    "fastapi>=0.100",
    "uvicorn>=0.23",
]
curriculum = [
    "fsrs>=4.0",  # FSRS spaced repetition (2026 best practice)
    "scipy>=1.10",  # For FSRS parameter optimization
]
debate = [
    "sentence-transformers>=2.2",  # For consensus optimizer
]
```

### 2026 Algorithm Dependencies

| Feature | Package | Purpose |
|---------|---------|---------|
| Spaced Repetition | `fsrs>=4.0` | FSRS algorithm (20-30% more efficient than SM-2) |
| Consensus Optimizer | `sentence-transformers` | Argument embedding for learned aggregation |
| PRM Verification | (included) | ThinkPRM for step-wise verification |
| Difficulty Estimation | (included) | Perplexity-based compute allocation |

### Windows Compatibility Notes
- All visualization uses Gradio (no multiprocessing issues)
- FastAPI server works correctly on Windows
- Curriculum engine uses `dataloader_num_workers=0`
- FSRS runs entirely locally (no cloud dependency)

---

## 7. Open Questions

1. **Debate transcript storage**: Should we store full transcripts or just summaries?
2. **Curriculum persistence**: How to handle curriculum state across training restarts?
3. **Visualization performance**: How to handle large numbers of dialogues efficiently?
4. **Inference caching**: Should we cache critic scores for repeated prompts?

---

## 8. Success Metrics

### Core Metrics

| Feature | Metric | Target | Research Basis |
|---------|--------|--------|----------------|
| Curriculum Engine | Score improvement per epoch | +0.5 points | [ADCL](https://arxiv.org/abs/2505.08364) reports accelerated convergence |
| FSRS Scheduler | Review efficiency vs SM-2 | 20-30% fewer reviews | [FSRS benchmarks](https://domenic.me/fsrs/) |
| Teacher Debate | Evaluation accuracy improvement | 4-6% | [A-HMAD results](https://link.springer.com/article/10.1007/s44443-025-00353-3) |
| Debate Consensus | Minority override success rate | >90% when applicable | [MAD study](https://arxiv.org/abs/2511.07784) |
| Inference API | Compute efficiency vs best-of-N | 4x reduction | [Test-time scaling](https://arxiv.org/abs/2408.03314) |
| PRM Verification | Label efficiency vs discriminative | 99% reduction | [ThinkPRM](https://arxiv.org/abs/2504.16828) |
| Visualizer | Dashboard load time | <2s | Standard UX target |

### 2026-Specific Benchmarks

| Capability | Benchmark | Target |
|------------|-----------|--------|
| Process verification | ProcessBench | Match ThinkPRM baseline |
| Mathematical reasoning | MATH-500 | +5% with best-of-N selection |
| Competition math | AIME '24 | Improvement with reward-guided search |
| Hallucination reduction | Custom factuality test | Target 90% (GraphRAG baseline) |

### Qualitative Success Indicators

1. **Curriculum Engine**
   - Model shows consistent progression through difficulty levels
   - No catastrophic forgetting when advancing stages
   - Weakness-targeted prompts show higher improvement rates

2. **Teacher Debate**
   - Debate transcripts show genuine position changes based on evidence
   - Minority override triggers appropriately on contrived test cases
   - Rationality scores correlate with argument quality

3. **Inference API**
   - Users report "it just works" experience
   - Streaming scores update smoothly
   - Self-refinement produces visibly better outputs

4. **Trajectory Visualizer**
   - Researchers can identify training issues from visualizations
   - Dialogue trees clearly show reasoning evolution
   - Export formats integrate with existing workflows

---

## 9. References

### Research Papers (2025-2026)

1. **Curriculum Learning**
   - [Adaptive Difficulty Curriculum Learning](https://arxiv.org/abs/2505.08364) - EMNLP 2025
   - [CAMPUS Framework](https://arxiv.org/html/2509.13790) - Competence-aware scheduling
   - [E2H Reasoner](https://arxiv.org/abs/2506.06632) - Easy-to-hard curriculum for RL

2. **Spaced Repetition**
   - [FSRS Algorithm](https://github.com/open-spaced-repetition/fsrs4anki) - Open source
   - [DRL-SRS](https://www.mdpi.com/2076-3417/14/13/5591) - Deep RL for scheduling
   - [Memory Dynamics](https://www.researchgate.net/publication/369045947) - Stability/retrievability model

3. **Multi-Agent Debate**
   - [Controlled MAD Study](https://arxiv.org/abs/2511.07784) - Identifies success factors
   - [A-HMAD](https://link.springer.com/article/10.1007/s44443-025-00353-3) - Heterogeneous agents
   - [DMAD](https://openreview.net/forum?id=t6QHYUOQL7) - Diverse reasoning approaches

4. **Test-Time Scaling**
   - [Scaling Test-Time Compute](https://arxiv.org/abs/2408.03314) - Compute-optimal allocation
   - [ThinkPRM](https://arxiv.org/abs/2504.16828) - Verbalized process verification
   - [State of LLMs 2025](https://magazine.sebastianraschka.com/p/state-of-llms-2025) - 2026 predictions

5. **Knowledge Graphs**
   - [LLM-KG Survey](https://arxiv.org/abs/2510.20345) - Comprehensive review
   - [GraphRAG/LightRAG](https://medium.com/@claudiubranzan/from-llms-to-knowledge-graphs-building-production-ready-graph-systems-in-2025-2b4aff1ec99a) - Production systems

6. **Visualization & Observability**
   - [LIDA](https://microsoft.github.io/lida/) - Microsoft's visualization tool
   - [LLM Observability 2026](https://lakefs.io/blog/llm-observability-tools/) - Tool comparison

---

*End of specification*
