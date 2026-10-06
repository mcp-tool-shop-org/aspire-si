"""
Training-dynamics export for ScalarScope.

A ``GeometryRecorder`` watches a training run one step at a time: the student's pooled hidden
state, the teachers' per-dimension scores, and each teacher's overall score. ``build()`` turns
that history into the geometry run ScalarScope reads (schema 1.0, the format ScalarScope 2.0
introduced), so a run can be opened, scrubbed, and compared with another run there.

What each field holds:

- ``trajectory``: the pooled hidden states projected onto their first two principal
  components and scaled so the farthest point sits at distance 1. ``t`` runs from 0 to 1.
  ``velocity`` is the displacement per step (a central difference), ``curvature`` the signed
  turning angle between successive displacements divided by pi (so it lies in [-1, 1]),
  scaled down where the steps around the turn are shorter than the run's average step so that
  jitter at a standstill does not read as instability, and ``effective_dim`` the
  participation ratio of the hidden states in a window around the step.
- ``scalars``: every evaluation dimension the teachers scored, scaled from 0-10 to 0-1.
- ``geometry.eigenvalues``: for each step, the spectrum of the covariance of the dimension
  scores in a window around it, as fractions of the total (so they sum to 1, or are all 0
  when the scores do not move). The first value large against the rest means one direction
  explains how the teachers' judgements vary together. ``anisotropy`` is the first over
  the second.
- ``evaluators.professors``: one per teacher, the unit direction in the 2-D state space along
  which that teacher's overall score rises (a least-squares fit of the score on the state).
- ``failures``: dips, where a dimension falls at least ``dip_threshold`` below the median of
  the steps before it.

The recorder holds the pooled states in memory, one vector of the model's hidden size per step;
record every few batches (``every``) for long runs.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "1.0"
# Teacher scores are 0-10; the export holds 0-1 as ScalarScope's samples do.
SCORE_SCALE = 10.0
# A ratio with a zero second eigenvalue is written as this instead of infinity.
MAX_ANISOTROPY = 1000.0


@dataclass
class _Step:
    state: np.ndarray
    scores: dict[str, float]
    teacher_scores: dict[str, float]


@dataclass
class GeometryRecorder:
    """Collects one step of training at a time and writes ScalarScope's geometry export."""

    run_id: str
    condition: str = ""
    seed: int = 0
    holdout_professor: str | None = None
    conscience_tier: str = "UNKNOWN"
    # Steps either side of a step in the effective-dimension and eigenvalue windows.
    window: int = 4
    # Record one export step for this many calls to `record`, averaging them.
    every: int = 1
    # A dimension falling this far (0-1 scale) below the median before it is a failure.
    dip_threshold: float = 0.1
    # Steps before a step that the dip median is taken over.
    dip_window: int = 5
    _steps: list[_Step] = field(default_factory=list, init=False, repr=False)
    _pending: list[_Step] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.run_id.strip() and not self.condition.strip():
            raise ValueError("A geometry export needs a run_id or a condition.")
        if self.every < 1:
            raise ValueError("every must be 1 or more.")
        if self.window < 1:
            raise ValueError("window must be 1 or more.")

    def __len__(self) -> int:
        return len(self._steps)

    def record(
        self,
        hidden_states: Any,
        attention_mask: Any | None = None,
        evaluations: Iterable[Any] = (),
        teacher_name: str = "teacher",
    ) -> None:
        """Record a batch: the last hidden layer, its mask, and the batch's teacher evaluations.

        ``hidden_states`` is (batch, sequence, hidden) and may be a torch tensor; it is pooled
        over the unmasked tokens and then over the batch. Each evaluation is a
        ``TeacherEvaluation``; its dimension scores are averaged over the batch, and a
        composite teacher's per-teacher scores (``metadata["teacher_scores"]``) become one
        professor each. A single teacher is one professor named ``metadata["teacher"]``, or
        ``teacher_name``.
        """
        state = pool_hidden_states(hidden_states, attention_mask)
        dimension_sums: dict[str, list[float]] = {}
        teacher_sums: dict[str, list[float]] = {}
        for evaluation in evaluations:
            for dimension_score in getattr(evaluation, "dimension_scores", []) or []:
                name = getattr(dimension_score.dimension, "value", dimension_score.dimension)
                dimension_sums.setdefault(str(name), []).append(float(dimension_score.score))
            metadata = getattr(evaluation, "metadata", None) or {}
            per_teacher = metadata.get("teacher_scores")
            if isinstance(per_teacher, Mapping) and per_teacher:
                for name, score in per_teacher.items():
                    teacher_sums.setdefault(str(name), []).append(float(score))
            else:
                name = str(metadata.get("teacher", teacher_name))
                teacher_sums.setdefault(name, []).append(float(evaluation.overall_score))
        self.record_step(
            state,
            {name: float(np.mean(values)) for name, values in dimension_sums.items()},
            {name: float(np.mean(values)) for name, values in teacher_sums.items()},
        )

    def record_step(
        self,
        state: Sequence[float] | np.ndarray,
        scores: Mapping[str, float],
        teacher_scores: Mapping[str, float] | None = None,
    ) -> None:
        """Record one step directly: a state vector and 0-10 scores by dimension and teacher."""
        vector = np.asarray(state, dtype=np.float64).reshape(-1)
        if self._steps and vector.shape != self._steps[0].state.shape:
            raise ValueError(
                f"Every state must have the same size; the first had {self._steps[0].state.size}, "
                f"this one has {vector.size}."
            )
        self._pending.append(_Step(vector, dict(scores), dict(teacher_scores or {})))
        if len(self._pending) >= self.every:
            self._flush()

    def _flush(self) -> None:
        if not self._pending:
            return
        pending, self._pending = self._pending, []
        self._steps.append(
            _Step(
                np.mean([step.state for step in pending], axis=0),
                _mean_maps(step.scores for step in pending),
                _mean_maps(step.teacher_scores for step in pending),
            )
        )

    def build(self, training_items: int = 0, cycles: int = 0) -> dict[str, Any]:
        """The export as a JSON-ready dict. Needs at least two recorded steps."""
        self._flush()
        if len(self._steps) < 2:
            raise ValueError("A geometry export needs at least two recorded steps.")
        states = np.stack([step.state for step in self._steps])
        count, input_dim = states.shape
        times = [i / (count - 1) for i in range(count)]

        points, components, explained = _project(states)
        velocity = _velocity(points)
        curvature = _curvature(points)
        effective = [_participation_ratio(states[_window(i, count, self.window)]) for i in range(count)]

        dimensions = _ordered_names(step.scores for step in self._steps)
        score_rows = _forward_fill([step.scores for step in self._steps], dimensions)
        score_matrix = np.array([[row[name] / SCORE_SCALE for name in dimensions] for row in score_rows])

        eigenvalues = []
        anisotropy = []
        for i in range(count):
            values = _score_spectrum(score_matrix[_window(i, count, self.window)]) if dimensions else []
            eigenvalues.append({"t": times[i], "values": values})
            anisotropy.append({"t": times[i], "ratio": _ratio(values)})

        teachers = _ordered_names(step.teacher_scores for step in self._steps)
        teacher_rows = _forward_fill([step.teacher_scores for step in self._steps], teachers)
        professors = [
            {
                "name": name,
                "vector": _professor_direction(points, np.array([row[name] for row in teacher_rows])),
                "holdout": name == self.holdout_professor,
            }
            for name in teachers
        ]

        return {
            "schema_version": SCHEMA_VERSION,
            "run_metadata": {
                "run_id": self.run_id,
                "condition": self.condition,
                "seed": int(self.seed),
                "training_items": int(training_items),
                "cycles": int(cycles),
                "holdout_professor": self.holdout_professor,
                "conscience_tier": self.conscience_tier,
            },
            "reduction": {
                "method": "PCA",
                "input_dim": int(input_dim),
                "output_dim": 2,
                "explained_variance": explained,
                "components": components,
            },
            "trajectory": {
                "timesteps": [
                    {
                        "t": times[i],
                        "state_2d": [float(points[i, 0]), float(points[i, 1])],
                        "velocity": [float(velocity[i, 0]), float(velocity[i, 1])],
                        "curvature": float(curvature[i]),
                        "effective_dim": float(effective[i]),
                    }
                    for i in range(count)
                ]
            },
            "scalars": {
                "dimensions": dimensions,
                "values": [
                    {"t": times[i], **{name: float(score_matrix[i, j]) for j, name in enumerate(dimensions)}}
                    for i in range(count)
                ],
            },
            "geometry": {"eigenvalues": eigenvalues, "anisotropy": anisotropy},
            "evaluators": {"latent_dim": len(dimensions), "professors": professors},
            "failures": _dips(times, score_matrix, dimensions, self.dip_threshold, self.dip_window),
        }

    def write(self, path: str | Path, training_items: int = 0, cycles: int = 0) -> Path:
        """Write the export as JSON and return its path."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        document = self.build(training_items=training_items, cycles=cycles)
        target.write_text(json.dumps(document, indent=1, allow_nan=False), encoding="utf-8")
        return target


def pool_hidden_states(hidden_states: Any, attention_mask: Any | None = None) -> np.ndarray:
    """Mean over the unmasked tokens of each sequence, then over the batch: one vector."""
    hidden = _to_numpy(hidden_states)
    if hidden.ndim == 1:
        return hidden
    if hidden.ndim == 2:
        hidden = hidden[None, ...]
    if hidden.ndim != 3:
        raise ValueError(f"Hidden states must be (batch, sequence, hidden); got shape {hidden.shape}.")
    if attention_mask is None:
        mask = np.ones(hidden.shape[:2])
    else:
        mask = _to_numpy(attention_mask).reshape(hidden.shape[:2])
    weights = mask[..., None]
    totals = weights.sum(axis=1)
    pooled = (hidden * weights).sum(axis=1) / np.maximum(totals, 1.0)
    return pooled.mean(axis=0)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
        if hasattr(value, "float"):
            value = value.float()
        value = value.cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _mean_maps(maps: Iterable[Mapping[str, float]]) -> dict[str, float]:
    collected: dict[str, list[float]] = {}
    for mapping in maps:
        for name, value in mapping.items():
            collected.setdefault(name, []).append(value)
    return {name: float(np.mean(values)) for name, values in collected.items()}


def _ordered_names(maps: Iterable[Mapping[str, float]]) -> list[str]:
    names: list[str] = []
    for mapping in maps:
        for name in mapping:
            if name not in names:
                names.append(name)
    return names


def _forward_fill(rows: list[dict[str, float]], names: list[str]) -> list[dict[str, float]]:
    """A step missing a name takes the last value before it, or the first value after it."""
    filled: list[dict[str, float]] = []
    first = {name: next((row[name] for row in rows if name in row), 0.0) for name in names}
    last = dict(first)
    for row in rows:
        for name in names:
            if name in row:
                last[name] = row[name]
        filled.append({name: last[name] for name in names})
    return filled


def _project(states: np.ndarray) -> tuple[np.ndarray, list[list[float]], list[float]]:
    """PCA to two dimensions, scaled so the farthest point is at distance 1.

    Each component's sign is fixed so its largest-magnitude entry is positive, which makes the
    picture the same from run to run of the same data.
    """
    centered = states - states.mean(axis=0)
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    variance = singular**2
    total = float(variance.sum())
    components = np.zeros((2, states.shape[1]))
    explained = [0.0, 0.0]
    for k in range(min(2, vt.shape[0])):
        component = vt[k]
        if component[np.argmax(np.abs(component))] < 0:
            component = -component
        components[k] = component
        explained[k] = float(variance[k] / total) if total > 0 else 0.0
    points = centered @ components.T
    radius = float(np.max(np.linalg.norm(points, axis=1)))
    if radius > 0:
        points = points / radius
    return points, components.tolist(), explained


def _velocity(points: np.ndarray) -> np.ndarray:
    return np.gradient(points, axis=0)


def _curvature(points: np.ndarray) -> np.ndarray:
    """Signed turning angle between the displacement into and out of each step, over pi.

    A turn counts in full only when both steps around it are at least the run's average step
    (its path length over its step count); a shorter step scales it down in proportion. A run
    that has converged still jitters, and the direction of a tiny jitter is noise, not a change
    of course. The average, not the median, is the yardstick, because a run that sits converged
    for most of its length has a median step that is itself jitter.
    """
    steps = np.diff(points, axis=0)
    lengths = np.linalg.norm(steps, axis=1)
    typical = float(np.mean(lengths)) if len(lengths) else 0.0
    turning = np.zeros(len(points))
    for i in range(1, len(points) - 1):
        before, after = steps[i - 1], steps[i]
        shorter = min(lengths[i - 1], lengths[i])
        if shorter < 1e-12 or typical <= 0:
            continue
        cross = before[0] * after[1] - before[1] * after[0]
        dot = float(before @ after)
        turning[i] = math.atan2(cross, dot) / math.pi * min(1.0, shorter / typical)
    return turning


def _window(i: int, count: int, half: int) -> slice:
    """Steps i-half..i+half, kept inside the run and at least two long."""
    start, stop = max(0, i - half), min(count, i + half + 1)
    if stop - start < 2:
        start, stop = max(0, stop - 2), min(count, start + 2)
    return slice(start, stop)


def _participation_ratio(states: np.ndarray) -> float:
    """(sum of eigenvalues)^2 / sum of squared eigenvalues of the states' covariance.

    Uses the small Gram matrix of the window, which has the same non-zero eigenvalues.
    """
    centered = states - states.mean(axis=0)
    gram = centered @ centered.T
    values = np.clip(np.linalg.eigvalsh(gram), 0.0, None)
    squares = float((values**2).sum())
    return float(values.sum() ** 2 / squares) if squares > 0 else 0.0


def _score_spectrum(scores: np.ndarray) -> list[float]:
    """Eigenvalues of the dimension scores' covariance, descending, as fractions of the total."""
    if scores.shape[1] == 1:
        variance = float(scores.var())
        return [1.0 if variance > 0 else 0.0]
    covariance = np.atleast_2d(np.cov(scores, rowvar=False, bias=True))
    values = np.sort(np.clip(np.linalg.eigvalsh(covariance), 0.0, None))[::-1]
    total = float(values.sum())
    if total <= 1e-15:
        return [0.0] * len(values)
    return [float(value / total) for value in values]


def _ratio(values: list[float]) -> float:
    if len(values) < 2 or values[0] <= 0:
        return 0.0
    if values[1] <= values[0] / MAX_ANISOTROPY:
        return MAX_ANISOTROPY
    return float(values[0] / values[1])


def _professor_direction(points: np.ndarray, scores: np.ndarray) -> list[float]:
    """The unit direction in the 2-D state space along which the score rises; zero if flat."""
    centered_points = points - points.mean(axis=0)
    centered_scores = scores - scores.mean()
    if float(np.abs(centered_scores).max(initial=0.0)) < 1e-12:
        return [0.0, 0.0]
    beta, *_ = np.linalg.lstsq(centered_points, centered_scores, rcond=None)
    norm = float(np.linalg.norm(beta))
    if norm < 1e-12:
        return [0.0, 0.0]
    return [float(beta[0] / norm), float(beta[1] / norm)]


def _dips(
    times: list[float],
    scores: np.ndarray,
    dimensions: list[str],
    threshold: float,
    window: int,
) -> list[dict[str, Any]]:
    """A failure where a dimension drops `threshold` or more below the median of the steps
    before it; a dip lasting several steps is one failure, at its first step."""
    failures = []
    for j, name in enumerate(dimensions):
        in_dip = False
        for i in range(1, len(times)):
            baseline = float(np.median(scores[max(0, i - window) : i, j]))
            drop = baseline - float(scores[i, j])
            if drop >= threshold:
                if not in_dip:
                    severity = (
                        "HIGH" if drop >= 3 * threshold else "MEDIUM" if drop >= 2 * threshold else "LOW"
                    )
                    failures.append(
                        {
                            "t": times[i],
                            "category": f"{name}_dip",
                            "severity": severity,
                            "description": f"{name} fell from {baseline:.2f} to {scores[i, j]:.2f}",
                        }
                    )
                in_dip = True
            else:
                in_dip = False
    failures.sort(key=lambda failure: failure["t"])
    return failures
