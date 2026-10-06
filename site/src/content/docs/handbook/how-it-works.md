---
title: How It Works
description: The four-stage ASPIRE pipeline — adversarial dialogue, critic training, student training, and inference-time self-refinement.
sidebar:
  order: 2
---

ASPIRE trains AI judgment through a four-stage pipeline. Each stage builds on the previous one, culminating in a student model that can self-refine at inference time without any teacher API calls.

## Stage 1: Adversarial Dialogue

The student generates a response. The teacher challenges it. They go back and forth, probing weaknesses, demanding clarity, pushing deeper.

```
Student: "Recursion works by calling itself."

Teacher (Socratic): "But what prevents infinite regress?
                     What's the mechanism that grounds the recursion?"

Student: "The base case stops it when..."

Teacher: "You say 'stops it' — but how does the computer know
          to check the base case before recursing?"
```

This adversarial exchange produces rich training data: not just right answers, but the reasoning process that leads to right answers. The teacher's challenges expose gaps in the student's understanding that flat supervision would never surface.

## Stage 2: Critic Training

The critic learns to predict the teacher's judgment. Not just the score, but the reasoning behind it.

```python
critic_loss = predict_teacher_judgment(
    score=True,      # "This deserves a 7/10"
    reasoning=True,  # "Because the explanation lacks depth on X"
)
```

The critic is a separate model (or a lightweight head on the student's encoder) that internalizes the teacher's evaluation criteria. After training, the critic can assess student outputs the way the teacher would, without calling the teacher at all.

This is the key insight: the critic becomes an internalized mentor. It learns not just what the teacher would score, but why.

## Stage 3: Student Training

The student trains against four signals simultaneously:

```python
student_loss = (
    reward_from_critic +      # Higher score = better
    contrastive_to_teacher +  # Pull toward teacher's improved version
    trajectory_improvement +  # Get better across dialogue turns
    coherence_regularization  # Maintain consistent reasoning
)
```

**Reward from critic** — The critic scores the student's output. Higher scores mean the response aligns with what the teacher would approve.

**Contrastive to teacher** — The student's representation is pulled toward the teacher's improved version of the response, learning the direction of improvement.

**Trajectory improvement** — The student should get better across dialogue turns, not just produce one good response. This encourages genuine reasoning development.

**Coherence regularization** — Keeps the student's reasoning internally consistent across turns, preventing contradictions or drift during refinement.

## Stage 4: Inference Magic

After training, the student self-refines using the internalized critic. No teacher API calls needed.

```python
def generate_with_judgment(prompt):
    response = student.generate(prompt)

    while critic.score(response) < threshold:
        response = student.refine(response, critic.feedback)

    return response  # Self-improved through internalized judgment
```

The student generates a draft, the critic evaluates it, and the student refines based on the critic's feedback. This loop runs entirely locally. The teacher's wisdom has been distilled into the critic, and the student has learned to use that internalized judgment to improve its own output.

## Why this matters

Standard fine-tuning teaches models to match outputs. ASPIRE teaches models to develop judgment. The difference shows up at inference time: a fine-tuned model produces its best guess in one shot, while an ASPIRE-trained model iteratively refines its output using internalized quality criteria.

The student doesn't just predict what the teacher would say. It understands what the teacher understands. The map becomes the territory.
