"""
Tests for the ScalarScope training-dynamics export (aspire.geometry).
"""

import json
import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from aspire.geometry import (
    MAX_ANISOTROPY,
    GeometryRecorder,
    pool_hidden_states,
)
from aspire.teachers.base import DimensionScore, EvaluationDimension, TeacherEvaluation


def evaluation(overall, scores, metadata=None):
    return TeacherEvaluation(
        overall_score=overall,
        dimension_scores=[
            DimensionScore(dimension=EvaluationDimension(name), score=value, explanation="")
            for name, value in scores.items()
        ],
        reasoning="",
        metadata=metadata or {},
    )


def recorder(**kwargs):
    kwargs.setdefault("run_id", "run")
    return GeometryRecorder(**kwargs)


def line_run(count=12, dims=6):
    """States along one direction, scores rising with it."""
    rec = recorder()
    direction = np.arange(1, dims + 1, dtype=float)
    for i in range(count):
        rec.record_step(direction * i, {"correctness": 3 + 0.5 * i, "clarity": 4 + 0.25 * i}, {"P1": 2 + 0.5 * i})
    return rec


class TestShape:
    def test_the_export_has_scalarscopes_fields_and_reads_back(self, tmp_path):
        path = line_run().write(tmp_path / "out" / "geometry.json", training_items=40, cycles=3)
        document = json.loads(path.read_text(encoding="utf-8"))
        assert document["schema_version"] == "1.0"
        assert set(document) == {
            "schema_version", "run_metadata", "reduction", "trajectory",
            "scalars", "geometry", "evaluators", "failures",
        }
        meta = document["run_metadata"]
        assert (meta["run_id"], meta["training_items"], meta["cycles"]) == ("run", 40, 3)
        steps = document["trajectory"]["timesteps"]
        assert len(steps) == 12
        assert steps[0]["t"] == 0.0 and steps[-1]["t"] == 1.0
        assert all(len(step["state_2d"]) == 2 and len(step["velocity"]) == 2 for step in steps)
        assert document["scalars"]["dimensions"] == ["correctness", "clarity"]
        assert document["evaluators"]["latent_dim"] == 2
        assert document["reduction"]["input_dim"] == 6
        assert len(document["reduction"]["components"]) == 2
        assert len(document["reduction"]["components"][0]) == 6

    def test_scores_are_scaled_to_zero_one(self):
        values = line_run().build()["scalars"]["values"]
        assert values[0]["correctness"] == pytest.approx(0.3)
        assert values[0]["clarity"] == pytest.approx(0.4)
        assert values[0]["t"] == 0.0

    def test_two_steps_are_needed(self):
        rec = recorder()
        rec.record_step([0.0, 1.0], {"correctness": 5})
        with pytest.raises(ValueError, match="at least two"):
            rec.build()

    def test_states_must_keep_their_size(self):
        rec = recorder()
        rec.record_step([0.0, 1.0], {})
        with pytest.raises(ValueError, match="same size"):
            rec.record_step([0.0, 1.0, 2.0], {})

    @pytest.mark.parametrize("kwargs", [{"run_id": " ", "condition": ""}, {"every": 0}, {"window": 0}])
    def test_bad_settings_say_why(self, kwargs):
        with pytest.raises(ValueError):
            GeometryRecorder(**{"run_id": "r", **kwargs})


class TestTrajectory:
    def test_a_straight_run_is_one_component_scaled_to_unit_radius(self):
        document = line_run().build()
        assert document["reduction"]["explained_variance"][0] == pytest.approx(1.0)
        points = np.array([step["state_2d"] for step in document["trajectory"]["timesteps"]])
        assert np.max(np.linalg.norm(points, axis=1)) == pytest.approx(1.0)
        assert np.allclose(points[:, 1], 0.0, atol=1e-9)
        assert all(step["curvature"] == pytest.approx(0.0) for step in document["trajectory"]["timesteps"])

    def test_the_picture_is_the_same_each_time(self):
        assert line_run().build() == line_run().build()

    def test_a_left_turn_is_plus_half_and_a_right_turn_minus_half(self):
        rec = recorder()
        for point in [(0, 0), (1, 0), (1, 1), (2, 1), (2, 0)]:
            rec.record_step(point, {})
        curvature = [step["curvature"] for step in rec.build()["trajectory"]["timesteps"]]
        assert curvature[0] == 0.0 and curvature[-1] == 0.0
        # PCA may mirror the picture; the turns keep their alternation either way.
        assert abs(curvature[1]) == pytest.approx(0.5)
        assert curvature[2] == pytest.approx(-curvature[1])

    def test_jitter_at_a_standstill_is_not_a_sharp_turn(self):
        rec = recorder()
        points = [(float(i), 0.0) for i in range(8)]
        # Converged: tiny back-and-forth steps a hundredth of the approach steps.
        points += [(7.0 + 0.01 * (i % 2), 0.01 * ((i // 2) % 2)) for i in range(1, 9)]
        for point in points:
            rec.record_step(point, {})
        curvature = [step["curvature"] for step in rec.build()["trajectory"]["timesteps"]]
        assert max(abs(value) for value in curvature) < 0.05

    def test_velocity_is_displacement_per_step(self):
        steps = line_run(count=5).build()["trajectory"]["timesteps"]
        speeds = [math.hypot(*step["velocity"]) for step in steps]
        # Five points centred and scaled to radius 1 run from -1 to 1 in steps of 0.5.
        assert speeds == pytest.approx([0.5] * 5)

    def test_effective_dimension_counts_the_directions_used(self):
        flat = line_run().build()["trajectory"]["timesteps"]
        assert all(step["effective_dim"] == pytest.approx(1.0) for step in flat)
        rng = np.random.default_rng(3)
        rec = recorder(window=10)
        for _ in range(30):
            rec.record_step(rng.normal(size=8), {})
        spread = [step["effective_dim"] for step in rec.build()["trajectory"]["timesteps"]]
        assert min(spread) > 3.0


class TestSpectrum:
    def test_eigenvalues_are_fractions_that_sum_to_one(self):
        rng = np.random.default_rng(1)
        rec = recorder()
        for i in range(10):
            rec.record_step([i, 0.0], {"correctness": rng.uniform(3, 9), "clarity": rng.uniform(3, 9),
                                       "nuance": rng.uniform(3, 9)})
        for step in rec.build()["geometry"]["eigenvalues"]:
            assert len(step["values"]) == 3
            assert sum(step["values"]) == pytest.approx(1.0)
            assert step["values"] == sorted(step["values"], reverse=True)

    def test_scores_moving_together_give_one_dominant_direction(self):
        geometry = line_run().build()["geometry"]
        assert geometry["eigenvalues"][5]["values"][0] == pytest.approx(1.0)
        assert geometry["anisotropy"][5]["ratio"] == MAX_ANISOTROPY

    def test_scores_that_never_move_have_no_spectrum(self):
        rec = recorder()
        for i in range(4):
            rec.record_step([i, 0.0], {"correctness": 5, "clarity": 5})
        geometry = rec.build()["geometry"]
        assert all(step["values"] == [0.0, 0.0] for step in geometry["eigenvalues"])
        assert all(step["ratio"] == 0.0 for step in geometry["anisotropy"])

    def test_one_dimension_is_one_eigenvalue(self):
        rec = recorder()
        for i in range(4):
            rec.record_step([i, 0.0], {"correctness": i})
        assert rec.build()["geometry"]["eigenvalues"][1]["values"] == [1.0]


class TestProfessors:
    def test_a_teacher_points_where_its_score_rises(self):
        rec = recorder(holdout_professor="P2")
        for x, y in [(0, 0), (1, 0.2), (2, -0.1), (3, 0.3), (4, 0.0), (5, -0.2)]:
            rec.record_step([x, y], {}, {"P1": 2.0 + x, "P2": 5.0 + 10 * y})
        document = rec.build()
        points = np.array([step["state_2d"] for step in document["trajectory"]["timesteps"]])
        professors = {professor["name"]: professor for professor in document["evaluators"]["professors"]}
        assert professors["P2"]["holdout"] and not professors["P1"]["holdout"]
        p1 = np.array(professors["P1"]["vector"])
        assert np.linalg.norm(p1) == pytest.approx(1.0)
        # Moving along P1's arrow raises P1's score.
        projected = points @ p1
        assert np.corrcoef(projected, [2.0 + x for x in range(6)])[0, 1] > 0.99

    def test_a_teacher_that_never_changes_has_no_direction(self):
        rec = recorder()
        for i in range(3):
            rec.record_step([i, 0.0], {}, {"P1": 7.0})
        assert rec.build()["evaluators"]["professors"][0]["vector"] == [0.0, 0.0]


class TestFailures:
    def test_a_dip_is_one_failure_at_its_first_step(self):
        rec = recorder()
        scores = [6, 6, 6, 6, 2, 2, 6, 6]
        for i, score in enumerate(scores):
            rec.record_step([i, 0.0], {"correctness": score})
        failures = rec.build()["failures"]
        assert len(failures) == 1
        failure = failures[0]
        assert failure["category"] == "correctness_dip"
        assert failure["t"] == pytest.approx(4 / 7)
        assert failure["severity"] == "HIGH"
        assert "0.60" in failure["description"] and "0.20" in failure["description"]

    def test_small_wobbles_are_not_failures(self):
        rec = recorder()
        for i, score in enumerate([6, 5.5, 6, 5.6, 6]):
            rec.record_step([i, 0.0], {"correctness": score})
        assert rec.build()["failures"] == []

    def test_severity_follows_the_size_of_the_drop(self):
        rec = recorder()
        for i, score in enumerate([8, 8, 6.8, 8, 8, 5.9]):
            rec.record_step([i, 0.0], {"correctness": score})
        assert [failure["severity"] for failure in rec.build()["failures"]] == ["LOW", "MEDIUM"]


class TestRecording:
    def test_padding_is_left_out_of_the_pooled_state(self):
        hidden = torch.zeros(1, 3, 2)
        hidden[0, 0] = torch.tensor([1.0, 2.0])
        hidden[0, 1] = torch.tensor([3.0, 4.0])
        hidden[0, 2] = torch.tensor([100.0, 100.0])
        mask = torch.tensor([[1, 1, 0]])
        assert pool_hidden_states(hidden, mask).tolist() == [2.0, 3.0]
        assert pool_hidden_states(hidden.numpy()).tolist() == pytest.approx([104 / 3, 106 / 3])

    def test_bad_hidden_shapes_say_why(self):
        with pytest.raises(ValueError, match="batch, sequence, hidden"):
            pool_hidden_states(np.zeros((1, 2, 3, 4)))

    def test_one_dimensional_and_two_dimensional_states_are_accepted(self):
        assert pool_hidden_states([1.0, 2.0]).tolist() == [1.0, 2.0]
        assert pool_hidden_states(np.array([[1.0, 2.0], [3.0, 4.0]])).tolist() == [2.0, 3.0]

    def test_a_batch_records_its_mean_scores_and_teacher(self):
        rec = recorder()
        for step in range(2):
            rec.record(
                torch.full((2, 4, 3), float(step)),
                torch.ones(2, 4),
                [
                    evaluation(6.0, {"correctness": 6.0, "clarity": 8.0}),
                    evaluation(8.0, {"correctness": 8.0, "clarity": 6.0}),
                ],
                teacher_name="socratic",
            )
        document = rec.build()
        assert document["scalars"]["values"][0]["correctness"] == pytest.approx(0.7)
        assert [professor["name"] for professor in document["evaluators"]["professors"]] == ["socratic"]

    def test_a_composite_teacher_gives_one_professor_per_teacher(self):
        rec = recorder()
        for step in range(3):
            rec.record(
                torch.full((1, 2, 2), float(step)),
                None,
                [evaluation(7.0, {"correctness": 7.0}, {"teacher_scores": {"socratic": 6.0 + step, "critic": 8.0}})],
            )
        names = [professor["name"] for professor in rec.build()["evaluators"]["professors"]]
        assert names == ["socratic", "critic"]

    def test_every_averages_batches_into_one_step(self):
        rec = recorder(every=2)
        for i in range(5):
            rec.record_step([float(i), 0.0], {"correctness": float(i)})
        assert len(rec) == 2
        document = rec.build()  # the odd batch out is flushed as a last step
        assert len(document["trajectory"]["timesteps"]) == 3
        assert [row["correctness"] for row in document["scalars"]["values"]] == pytest.approx([0.05, 0.25, 0.4])

    def test_a_missing_dimension_carries_its_last_value(self):
        rec = recorder()
        rec.record_step([0.0, 0.0], {"clarity": 5.0})
        rec.record_step([1.0, 0.0], {"clarity": 6.0, "nuance": 4.0})
        rec.record_step([2.0, 0.0], {"nuance": 7.0})
        values = rec.build()["scalars"]["values"]
        assert [row["clarity"] for row in values] == pytest.approx([0.5, 0.6, 0.6])
        assert [row["nuance"] for row in values] == pytest.approx([0.4, 0.4, 0.7])


class TestTrainer:
    def test_a_training_batch_is_recorded_and_written(self, tmp_path):
        from aspire.config import AspireConfig
        from aspire.trainer import AspireTrainer

        config = AspireConfig(device="cpu")
        config.training.output_dir = tmp_path
        config.training.num_epochs = 2
        trainer = AspireTrainer.__new__(AspireTrainer)
        trainer.config = config
        trainer.device = "cpu"
        trainer.teacher = SimpleNamespace(name="socratic")
        trainer.geometry = GeometryRecorder(run_id="unit")
        trainer.critic = MagicMock(return_value=SimpleNamespace(score=torch.tensor([7.0]),
                                                               reasoning_embedding=torch.zeros(1, 4)))
        trainer.loss_fn = MagicMock(return_value={"total": torch.tensor(0.0)})

        for step in range(3):
            trainer.student_model = MagicMock(
                return_value=SimpleNamespace(hidden_states=(torch.full((1, 3, 4), float(step)),))
            )
            dialogue = SimpleNamespace(final_evaluation=evaluation(6.0 + step, {"correctness": 6.0 + step}))
            batch = {"input_ids": torch.zeros(1, 3, dtype=torch.long), "attention_mask": torch.ones(1, 3)}
            trainer._compute_batch_loss(batch, [dialogue])

        written = trainer._write_geometry(training_items=3)
        document = json.loads((tmp_path / "geometry.json").read_text(encoding="utf-8"))
        assert written == str(tmp_path / "geometry.json")
        assert len(document["trajectory"]["timesteps"]) == 3
        assert document["run_metadata"]["cycles"] == 2
        assert document["evaluators"]["professors"][0]["name"] == "socratic"

    def test_too_short_a_run_writes_nothing(self, tmp_path):
        from aspire.config import AspireConfig
        from aspire.trainer import AspireTrainer

        trainer = AspireTrainer.__new__(AspireTrainer)
        trainer.config = AspireConfig(device="cpu")
        trainer.config.training.output_dir = tmp_path
        trainer.geometry = GeometryRecorder(run_id="unit")
        assert trainer._write_geometry(training_items=0) is None
        assert not (tmp_path / "geometry.json").exists()
