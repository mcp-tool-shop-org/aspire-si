---
title: Teachers
description: Teacher personas in ASPIRE — Socratic, Scientific, Creative, Adversarial, Compassionate, and composite strategies.
sidebar:
  order: 3
---

Teachers are the heart of ASPIRE. Each teacher persona embodies a different philosophy of mentorship, and the persona you choose shapes the kind of thinking your student develops.

## The five personas

### Socratic

*"What assumption are you making?"*

The Socratic teacher never gives answers. It asks questions that expose gaps in reasoning, force the student to examine its own assumptions, and guide it toward deeper understanding through dialogue. Produces students with strong independent reasoning and intellectual humility.

### Scientific

*"What's your evidence?"*

The Scientific teacher demands precision, evidence, and rigor. Every claim needs support. Every argument needs data. Sloppy reasoning gets challenged immediately. Produces students with technical precision and a habit of grounding claims in evidence.

### Creative

*"What if we tried the opposite?"*

The Creative teacher pushes against conventional thinking. It asks the student to consider alternative perspectives, invert assumptions, and explore unexpected connections. Produces students with lateral thinking ability and a willingness to innovate.

### Adversarial

*"I disagree. Defend your position."*

The Adversarial teacher takes the opposing view on everything. It forces the student to build robust arguments, anticipate counterpoints, and defend positions under pressure. Produces students with conviction, strong argumentation, and resilience to challenges.

### Compassionate

*"How might someone feel about this?"*

The Compassionate teacher brings ethical reasoning and human impact into every discussion. It asks the student to consider perspectives beyond pure logic, weigh consequences for people, and develop wisdom alongside intelligence. Produces students with ethical awareness and balanced judgment.

## Composite teachers

Real mentorship rarely comes from a single voice. ASPIRE lets you combine multiple teachers into a composite that applies different perspectives to the same problem.

```python
from aspire.teachers import CompositeTeacher
from aspire.teachers import SocraticTeacher
from aspire.teachers import ScientificTeacher
from aspire.teachers import AdversarialTeacher

teacher = CompositeTeacher(
    teachers=[
        SocraticTeacher(),
        ScientificTeacher(),
        AdversarialTeacher(),
    ],
    strategy="vote",
)
```

### Composition strategies

**vote** — All teachers evaluate the student's response independently. Scores are combined as a weighted average. Good for balanced, well-rounded feedback.

**rotate** — Teachers take turns across dialogue rounds. Round 1 might be Socratic, round 2 Scientific, round 3 Adversarial. Good for exposing the student to varied challenge styles within a single dialogue.

**specialize** — Each teacher handles the challenge types it prefers. When a Socratic-style question is needed, the Socratic teacher handles it; when an edge case is needed, the Scientific teacher steps in. Good for leveraging each teacher's strengths.

**random** — A teacher is selected at random for each turn, weighted by the configured weights. A lightweight alternative to rotate with more variety.

**debate** — Teachers evaluate independently and their results are synthesized into a unified critique. Currently implemented as weighted voting with rich feedback combination; future versions will add multi-turn teacher-to-teacher discussion.

### Choosing a strategy

| Strategy | API cost | Feedback richness | Best for |
|----------|----------|-------------------|----------|
| vote     | N calls per evaluation (one per teacher) | High — multiple perspectives | General-purpose training |
| rotate   | 1 call per turn | Moderate — varied over time | Cost-conscious training |
| specialize | 1 call per turn | Moderate — targeted expertise | Domain-specific training |
| random   | 1 call per turn | Moderate — weighted variety | Lightweight variety |
| debate   | N calls per evaluation | High — synthesized perspectives | High-quality, small-batch training |

### Curriculum-aware composition

The `CurriculumCompositeTeacher` adjusts teacher weights based on the current training stage. Early stages might emphasize the Compassionate teacher, while later stages shift weight toward the Adversarial teacher:

```python
from aspire.teachers import CompassionateTeacher, SocraticTeacher, AdversarialTeacher
from aspire.teachers.composite import CurriculumCompositeTeacher

teacher = CurriculumCompositeTeacher(
    teachers=[CompassionateTeacher(), SocraticTeacher(), AdversarialTeacher()],
    stage_weights={
        "foundation": [0.6, 0.3, 0.1],
        "reasoning":  [0.2, 0.5, 0.3],
        "adversarial": [0.1, 0.2, 0.7],
    },
    current_stage="foundation",
)

# Later, when the student improves:
teacher.set_stage("adversarial")
```

## Custom teachers

You can create your own teacher by subclassing `BaseTeacher`. A teacher must implement two async methods: `challenge` (pose a question or objection) and `evaluate` (score the response with reasoning).

```python
from aspire.teachers.base import (
    BaseTeacher,
    ChallengeType,
    DialogueHistory,
    TeacherChallenge,
    TeacherEvaluation,
)

class MyTeacher(BaseTeacher):
    async def challenge(
        self,
        prompt: str,
        student_response: str,
        dialogue_history: DialogueHistory | None = None,
        challenge_type: ChallengeType | None = None,
    ) -> TeacherChallenge:
        # Return a challenge that pushes the student
        ...

    async def evaluate(
        self,
        prompt: str,
        student_response: str,
        dialogue_history: DialogueHistory | None = None,
        generate_improved: bool = True,
    ) -> TeacherEvaluation:
        # Return scores, reasoning, strengths, weaknesses, suggestions
        ...
```

Custom teachers can wrap any backend: a local model, an API, a rule-based system, or even a human-in-the-loop. Register them with the teacher registry so the CLI can find them:

```python
from aspire.teachers.registry import TeacherRegistry

TeacherRegistry.register("my_teacher", MyTeacher)
```

You can also use the `register_teacher` decorator for a more concise registration:

```python
from aspire.teachers.registry import register_teacher

@register_teacher("my_teacher")
class MyTeacher(BaseTeacher):
    ...
```

After registration, `aspire dialogue --teacher my_teacher` and `aspire train --teacher my_teacher` work automatically.
