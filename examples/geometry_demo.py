"""
Write two simulated training-dynamics exports to open in ScalarScope.

No model, GPU, or teacher API is used: a student's hidden state is simulated as a point
drifting toward a target, and the teachers' scores rise as it gets closer. One run learns
steadily; the other is noisier and regresses partway through. Open both files in ScalarScope
(Compare, one on each side) to see the trajectories, the professor arrows, the eigenvalue
spectrum, the failures, and the five geometry deltas between them.

    python examples/geometry_demo.py [output directory]
"""

import sys
from pathlib import Path

import numpy as np

from aspire.geometry import GeometryRecorder
from aspire.teachers.base import EvaluationDimension

HIDDEN = 64
STEPS = 60
DIMENSIONS = [dimension.value for dimension in EvaluationDimension]


def simulate(
    run_id: str, condition: str, seed: int, noise: float, regression_at: int | None
) -> GeometryRecorder:
    rng = np.random.default_rng(seed)
    target = rng.normal(size=HIDDEN)
    # Each dimension cares about the target in its own way; two teachers weigh different halves.
    weights = rng.uniform(0.5, 1.5, size=len(DIMENSIONS))
    halves = (np.r_[np.ones(HIDDEN // 2), np.zeros(HIDDEN - HIDDEN // 2)],
              np.r_[np.zeros(HIDDEN // 2), np.ones(HIDDEN - HIDDEN // 2)])

    recorder = GeometryRecorder(run_id=run_id, condition=condition, seed=seed, holdout_professor="skeptic")
    state = np.zeros(HIDDEN)
    for step in range(STEPS):
        pull = 0.08 if regression_at is None or not (regression_at <= step < regression_at + 8) else -0.15
        state = state + pull * (target - state) + rng.normal(scale=noise, size=HIDDEN)
        closeness = 1.0 / (1.0 + np.linalg.norm(target - state) / np.sqrt(HIDDEN))
        scores = {
            name: float(np.clip(10 * closeness ** (1 / weight) + rng.normal(scale=0.3), 0, 10))
            for name, weight in zip(DIMENSIONS, weights)
        }
        teachers = {
            name: float(10 / (1 + np.linalg.norm(half * (target - state)) / np.sqrt(HIDDEN / 2)))
            for name, half in zip(("socratic", "skeptic"), halves)
        }
        recorder.record_step(state, scores, teachers)
    return recorder


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs/geometry-demo")
    runs = [
        simulate("steady", "Steady curriculum", seed=7, noise=0.02, regression_at=None),
        simulate("regressing", "Noisy curriculum with a regression", seed=11, noise=0.06, regression_at=30),
    ]
    for recorder in runs:
        path = recorder.write(out / f"{recorder.run_id}.json", training_items=STEPS * 4, cycles=1)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
