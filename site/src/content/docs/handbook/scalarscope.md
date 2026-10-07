---
title: Watching Runs in ScalarScope
description: Export a training run's dynamics from aspire-si and compare runs in ScalarScope.
sidebar:
  order: 6
---

A loss curve says how much a run learned. It does not say how: whether the student settled
or kept wandering, whether the teachers' judgements pulled the same way, or when a run
regressed. aspire-si can record that as a training-dynamics export, and
[ScalarScope](https://github.com/mcp-tool-shop-org/scalarscope) draws it and compares two runs.

## Turn it on

```bash
aspire train --prompts data/prompts.json --geometry
```

or, in the config file:

```yaml
training:
  geometry_export: true
  geometry_every: 1    # batches averaged into one export step
  geometry_window: 4   # steps either side in the windowed measures
```

The trainer records every batch and writes `geometry.json` to the output directory when training
ends. A run shorter than two recorded steps writes nothing and says so.

## No model? Try the demo

```bash
python examples/geometry_demo.py outputs/geometry-demo
```

This simulates two runs, one steady and one that regresses partway through, and writes both
exports. It needs no model, GPU or API key. Open the two files on either side of ScalarScope's
Compare page.

## What is in the export

| Field | What it holds |
|-------|---------------|
| Trajectory | The student's last hidden layer, pooled over the unmasked tokens and the batch, projected onto its first two principal components and scaled so the farthest point is at distance 1. |
| Velocity | The displacement per step. |
| Curvature | The signed turning angle between steps, over pi. A turn counts in full only when the steps around it are as long as the run's average step, so jitter after convergence does not read as instability. |
| Effective dimension | The participation ratio of the hidden states in a window around the step: roughly how many directions the student is moving in. |
| Scalars | Every evaluation dimension the teachers scored, from 0–10 scaled to 0–1. |
| Eigenvalues | Per step, the spectrum of how the dimension scores vary together in a window, as fractions that sum to 1. A large first value means one direction explains the teachers' judgements. |
| Professors | One arrow per teacher: the direction in the 2-D state space along which that teacher's score rises. A composite teacher gives one arrow per member. |
| Failures | Steps where a dimension drops at least 0.1 below the median of the five steps before it. A dip lasting several steps is one failure. |

The file is ScalarScope's geometry format (schema 1.0). ScalarScope 3 reads every dimension;
ScalarScope 2.0 reads only its five named dimensions.

## From Python

```python
from aspire.geometry import GeometryRecorder

recorder = GeometryRecorder(run_id="baseline", condition="socratic teacher", seed=42)

for batch in batches:
    hidden = model(**batch, output_hidden_states=True).hidden_states[-1]
    recorder.record(hidden, batch["attention_mask"], evaluations, teacher_name="socratic")

recorder.write("outputs/geometry.json", training_items=len(prompts), cycles=epochs)
```

`record_step(state, scores, teacher_scores)` takes a state vector and 0–10 scores directly, when
your loop is not ASPIRE's trainer.

## Memory

The recorder keeps one pooled vector of the model's hidden size per step. For long runs set
`geometry_every` so several batches average into one step.
