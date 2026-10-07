"""
Tests for integrations.isaac.motion_teacher (rule-based motion teachers and the composite).
"""

import numpy as np
import pytest

from integrations.isaac.motion_teacher import (
    BaseMotionTeacher,
    EfficiencyExpert,
    GraceCoach,
    MotionCritique,
    MotionDimension,
    MotionTeacher,
    PhysicsOracle,
    SafetyInspector,
    TrajectoryData,
)


def make_data(states, dt=0.1, **kwargs):
    states = np.asarray(states, dtype=float)
    if states.ndim == 1:
        states = states[:, None]
    n = len(states)
    return TrajectoryData(
        states=states,
        actions=np.zeros((n, 1)),
        timestamps=np.arange(n) * dt,
        **kwargs,
    )


def ramp(n=11, step=0.05, dims=2):
    """Constant-velocity motion: zero acceleration and zero jerk."""
    return np.tile((np.arange(n) * step)[:, None], (1, dims))


class StubTeacher(BaseMotionTeacher):
    def __init__(self, name, critique):
        super().__init__(name, "stub", [MotionDimension.SAFETY])
        self._critique = critique

    def critique(self, trajectory):
        return self._critique


def stub_critique(name, score, dims=None, strengths=(), weaknesses=(), suggestions=(), reasoning="r"):
    return MotionCritique(
        overall_score=score,
        dimension_scores=dims or {},
        reasoning=reasoning,
        strengths=list(strengths),
        weaknesses=list(weaknesses),
        improvement_suggestions=list(suggestions),
        teacher_name=name,
    )


class TestDataStructures:
    def test_trajectory_length_and_duration(self):
        data = make_data(ramp(6), dt=0.5)
        assert data.length == 6
        assert data.duration == pytest.approx(2.5)

    def test_trajectory_optional_fields_default_to_none(self):
        data = make_data(ramp(3))
        for name in ("goal_state", "initial_state", "task_description", "collisions",
                     "forces", "energy_usage", "goal_distances"):
            assert getattr(data, name) is None

    def test_critique_defaults(self):
        c = MotionCritique(overall_score=7.0)
        assert c.dimension_scores == {} and c.strengths == [] and c.weaknesses == []
        assert c.improvement_suggestions == [] and c.problem_timesteps == []
        assert c.reasoning == "" and c.teacher_name == "" and c.confidence == 1.0

    def test_dimension_enum_values(self):
        assert MotionDimension.SAFETY == "safety"
        assert MotionDimension("goal_achievement") is MotionDimension.GOAL_ACHIEVEMENT


class TestBaseMotionTeacher:
    def test_cannot_instantiate_abstract_base(self):
        with pytest.raises(TypeError):
            BaseMotionTeacher("n", "d", [])

    def test_repr_uses_class_and_name(self):
        assert repr(SafetyInspector()) == "SafetyInspector(name='Safety Inspector')"
        assert repr(GraceCoach()) == "GraceCoach(name='Grace Coach')"

    def test_abstract_critique_body_returns_none(self):
        teacher = StubTeacher("s", stub_critique("s", 1.0))
        assert BaseMotionTeacher.critique(teacher, make_data(ramp(3))) is None

    def test_focus_dimensions_per_persona(self):
        assert SafetyInspector().focus_dimensions == [MotionDimension.SAFETY, MotionDimension.STABILITY]
        assert EfficiencyExpert().focus_dimensions == [MotionDimension.EFFICIENCY]
        assert GraceCoach().focus_dimensions == [MotionDimension.SMOOTHNESS, MotionDimension.NATURALNESS]
        assert MotionDimension.GOAL_ACHIEVEMENT in PhysicsOracle().focus_dimensions


class TestSafetyInspector:
    def test_smooth_trajectory_scores_ten(self):
        c = SafetyInspector().critique(make_data(ramp()))
        assert c.overall_score == 10.0
        assert c.dimension_scores == {MotionDimension.SAFETY: 10.0}
        assert c.strengths == ["Velocities within limits (max: 0.50)"]
        assert c.weaknesses == [] and c.improvement_suggestions == []
        assert c.reasoning.startswith("This trajectory demonstrates safe, controlled motion.")
        assert c.teacher_name == "Safety Inspector"

    def test_no_collisions_is_a_strength(self):
        data = make_data(ramp(), collisions=np.zeros(11, dtype=bool))
        c = SafetyInspector().critique(data)
        assert "No collisions detected" in c.strengths
        assert c.overall_score == 10.0

    def test_collisions_reduce_score_proportionally(self):
        collisions = np.zeros(10, dtype=bool)
        collisions[[2, 5]] = True
        c = SafetyInspector().critique(make_data(ramp(10), collisions=collisions))
        assert c.overall_score == pytest.approx(10.0 - 10.0 * 2 / 10)
        assert c.weaknesses == ["Detected 2 collision events"]
        assert c.problem_timesteps == [(2, "collision"), (5, "collision")]
        assert c.improvement_suggestions == ["Increase clearance from obstacles"]
        assert c.reasoning.startswith("This trajectory demonstrates safe")

    def test_collision_timesteps_capped_at_five(self):
        collisions = np.ones(10, dtype=bool)
        c = SafetyInspector().critique(make_data(ramp(10), collisions=collisions))
        assert c.problem_timesteps == [(i, "collision") for i in range(5)]

    def test_mid_range_score_reasoning(self):
        collisions = np.zeros(10, dtype=bool)
        collisions[:4] = True
        c = SafetyInspector().critique(make_data(ramp(10), collisions=collisions))
        assert c.overall_score == pytest.approx(6.0)
        assert c.reasoning.startswith("This trajectory has some safety concerns")
        assert "Key concerns: Detected 4 collision events." in c.reasoning

    def test_low_score_reasoning_warns_against_hardware(self):
        collisions = np.ones(10, dtype=bool)
        c = SafetyInspector().critique(make_data(ramp(10), collisions=collisions))
        assert c.overall_score == pytest.approx(0.0)
        assert "should not be executed on real hardware" in c.reasoning

    def test_velocity_violation_penalty(self):
        # 0.4 rad per 0.1 s = 4 rad/s, limit 2 -> violation 1.0 -> penalty 3.0
        c = SafetyInspector().critique(make_data(ramp(11, step=0.4)))
        assert c.overall_score == pytest.approx(7.0)
        assert c.weaknesses == ["Velocity exceeded limit: 4.00 > 2.0"]
        assert "Reduce commanded velocities or add velocity smoothing" in c.improvement_suggestions
        assert c.strengths == []

    def test_velocity_penalty_is_capped(self):
        c = SafetyInspector().critique(make_data(ramp(11, step=40.0)))
        assert c.overall_score == pytest.approx(7.0)

    def test_acceleration_violation_penalty(self):
        # x_k = 0.1 k^2 with dt 0.1 -> constant acceleration 20 rad/s^2; limit 10 -> penalty 2
        states = 0.1 * np.arange(11) ** 2
        c = SafetyInspector(max_velocity=100.0).critique(make_data(states))
        assert c.overall_score == pytest.approx(8.0)
        assert c.weaknesses == ["Acceleration exceeded limit: 20.00"]
        assert "Use smoother acceleration profiles" in c.improvement_suggestions

    def test_acceleration_within_limit_has_no_weakness(self):
        states = 0.01 * np.arange(11) ** 2   # acceleration 2 rad/s^2
        c = SafetyInspector().critique(make_data(states))
        assert c.weaknesses == [] and c.overall_score == 10.0

    def test_two_states_skip_acceleration_check(self):
        c = SafetyInspector().critique(make_data(ramp(2)))
        assert c.overall_score == 10.0

    def test_single_state_skips_velocity_checks(self):
        c = SafetyInspector().critique(make_data(ramp(1)))
        assert c.overall_score == 10.0
        assert c.strengths == []

    def test_force_violation_penalty(self):
        forces = np.zeros((11, 2, 3))
        forces[4, 1] = [150.0, 0.0, 0.0]
        c = SafetyInspector().critique(make_data(ramp(), forces=forces))
        assert c.overall_score == pytest.approx(10.0 - 1.5)
        assert c.weaknesses == ["Contact force exceeded limit: 150.0N"]
        assert "Reduce approach speed near contacts" in c.improvement_suggestions

    def test_force_within_limit_is_ignored(self):
        forces = np.full((11, 1, 3), 10.0)
        c = SafetyInspector().critique(make_data(ramp(), forces=forces))
        assert c.overall_score == 10.0

    def test_penalties_stack_and_score_is_clamped_at_zero(self):
        forces = np.full((11, 1, 3), 1000.0)
        c = SafetyInspector().critique(
            make_data(ramp(11, step=40.0), collisions=np.ones(11, dtype=bool), forces=forces)
        )
        assert c.overall_score == 0.0
        assert len(c.weaknesses) == 3
        assert "Key concerns:" in c.reasoning

    def test_custom_limits_are_respected(self):
        lenient = SafetyInspector(max_velocity=10.0)
        assert lenient.critique(make_data(ramp(11, step=0.4))).overall_score == 10.0


class TestEfficiencyExpert:
    def test_direct_motion_with_no_context_scores_ten(self):
        c = EfficiencyExpert().critique(make_data(ramp(11, step=0.1)))
        assert c.overall_score == 10.0
        assert c.dimension_scores == {MotionDimension.EFFICIENCY: 10.0}
        assert c.reasoning.startswith("Highly efficient")
        assert c.teacher_name == "Efficiency Expert"
        assert c.strengths == [] and c.weaknesses == []

    def test_slow_completion_is_penalised(self):
        expert = EfficiencyExpert(target_completion_time=0.5)   # duration 1.0 -> ratio 2.0
        c = expert.critique(make_data(ramp(11, step=0.1)))
        assert c.overall_score == pytest.approx(8.0)
        assert c.weaknesses == ["Task took 2.0x longer than target"]
        assert c.improvement_suggestions == ["Optimize trajectory timing"]

    def test_slow_completion_penalty_capped_at_three(self):
        expert = EfficiencyExpert(target_completion_time=0.01)
        assert expert.critique(make_data(ramp(11, step=0.1))).overall_score == pytest.approx(7.0)

    def test_on_time_completion_is_a_strength(self):
        expert = EfficiencyExpert(target_completion_time=1.0)
        c = expert.critique(make_data(ramp(11, step=0.1)))
        assert c.strengths == ["Completed within target time"]
        assert c.overall_score == 10.0

    def test_borderline_time_ratio_gives_no_feedback(self):
        expert = EfficiencyExpert(target_completion_time=1.0 / 1.3)   # ratio 1.3
        c = expert.critique(make_data(ramp(11, step=0.1)))
        assert c.strengths == [] and c.weaknesses == []

    def test_direct_path_to_goal_is_a_strength(self):
        distances = np.linspace(1.0, 0.0, 11)
        c = EfficiencyExpert().critique(make_data(ramp(11, step=0.1, dims=1), goal_distances=distances))
        assert c.strengths == ["Excellent path efficiency (100.0%)"]
        assert c.overall_score == 10.0

    def test_wandering_path_is_penalised(self):
        distances = np.linspace(1.0, 0.0, 11)
        # 11 states moving 0.5 each: total motion 5.0, direct 1.0 -> efficiency 0.2
        c = EfficiencyExpert().critique(make_data(ramp(11, step=0.5, dims=1), goal_distances=distances))
        assert c.overall_score == pytest.approx(10.0 - (0.5 - 0.2) * 4.0, abs=1e-4)
        assert c.weaknesses == ["Path efficiency only 20.0%"]
        assert c.improvement_suggestions == ["Take more direct route to goal"]

    def test_middling_path_efficiency_gives_no_feedback(self):
        distances = np.linspace(1.0, 0.0, 11)
        # total motion 1/0.65 -> efficiency 0.65
        step = 1.0 / 0.65 / 10
        c = EfficiencyExpert().critique(make_data(ramp(11, step=step, dims=1), goal_distances=distances))
        assert c.strengths == [] and c.weaknesses == []
        assert c.overall_score == 10.0

    def test_high_energy_use_is_flagged_without_score_change(self):
        energy = np.full(11, 10.0)   # total 110 over 1.0 s -> 110 W
        c = EfficiencyExpert().critique(make_data(ramp(11, step=0.1), energy_usage=energy))
        assert c.weaknesses == ["High energy consumption (avg 110.0W)"]
        assert c.improvement_suggestions == ["Use more energy-efficient motion profiles"]
        assert c.overall_score == 10.0

    def test_low_energy_use_is_not_flagged(self):
        energy = np.full(11, 0.1)
        c = EfficiencyExpert().critique(make_data(ramp(11, step=0.1), energy_usage=energy))
        assert c.weaknesses == []

    def test_zero_energy_is_not_flagged(self):
        c = EfficiencyExpert().critique(make_data(ramp(11, step=0.1), energy_usage=np.zeros(11)))
        assert c.weaknesses == []

    def test_idle_trajectory_penalised_by_idle_penalty(self):
        c = EfficiencyExpert().critique(make_data(np.zeros((11, 2))))
        assert c.overall_score == pytest.approx(10.0 - 1.0 * 0.5 * 5.0)
        assert c.weaknesses == ["Robot idle 100.0% of trajectory"]
        assert c.improvement_suggestions == ["Reduce unnecessary pauses"]
        assert c.reasoning.startswith("Reasonably efficient")

    def test_partial_idle_time(self):
        states = np.concatenate([np.zeros(4), np.arange(1, 8) * 0.1])  # 3 idle of 10 diffs
        c = EfficiencyExpert().critique(make_data(states))
        assert c.overall_score == pytest.approx(10.0 - 0.3 * 0.5 * 5.0)
        assert c.weaknesses == ["Robot idle 30.0% of trajectory"]

    def test_idle_ratio_at_threshold_is_not_penalised(self):
        states = np.concatenate([np.zeros(3), np.arange(1, 9) * 0.1])  # 2 idle of 10 diffs = 0.2
        c = EfficiencyExpert().critique(make_data(states))
        assert c.overall_score == 10.0

    def test_severe_inefficiency_reasoning_and_floor(self):
        c = EfficiencyExpert(idle_penalty=2.0).critique(make_data(np.zeros((11, 2))))
        assert c.overall_score == 0.0
        assert c.reasoning.startswith("Significant inefficiencies detected")


class TestGraceCoach:
    def test_too_short_trajectory_gets_neutral_score(self):
        c = GraceCoach().critique(make_data(ramp(3)))
        assert c.overall_score == 5.0
        assert c.reasoning == "Trajectory too short to evaluate smoothness"
        assert c.dimension_scores == {}
        assert c.teacher_name == "Grace Coach"

    def test_smooth_coordinated_motion_scores_ten(self):
        c = GraceCoach().critique(make_data(ramp(8, step=0.01, dims=2)))
        assert c.overall_score == 10.0
        assert c.dimension_scores == {
            MotionDimension.SMOOTHNESS: 10.0,
            MotionDimension.NATURALNESS: pytest.approx(9.0),
        }
        assert "Smooth motion (jerk within limits)" in c.strengths
        assert "Well-coordinated multi-joint motion" in c.strengths
        assert c.reasoning.startswith("Beautifully smooth")
        assert c.problem_timesteps == []

    def test_single_joint_skips_coordination(self):
        c = GraceCoach().critique(make_data(ramp(8, step=0.01, dims=1)))
        assert c.strengths == ["Smooth motion (jerk within limits)"]

    def test_jerky_motion_penalised_and_located(self):
        states = [0, 0, 1, 0, 0, 0, 0]   # jerk magnitudes [3000, 3000, 1000, 0]
        c = GraceCoach().critique(make_data(states))
        assert c.overall_score == pytest.approx(6.0)   # penalty capped at 4
        assert c.weaknesses == ["High jerk detected (max: 3000.0)"]
        assert c.problem_timesteps == [(2, "high jerk"), (3, "high jerk"), (4, "high jerk")]
        assert c.improvement_suggestions == ["Use minimum-jerk trajectory optimization"]
        assert c.reasoning.startswith("Acceptable smoothness")

    def test_jerk_penalty_scales_below_cap(self):
        # max jerk 3000 vs threshold 2000 -> penalty (1.5 - 1) * 2 = 1.0
        c = GraceCoach(jerk_threshold=2000.0).critique(make_data([0, 0, 1, 0, 0, 0, 0]))
        assert c.overall_score == pytest.approx(9.0)

    def test_problem_timesteps_capped_at_three(self):
        states = [0, 1, 0, 1, 0, 1, 0, 1, 0]
        c = GraceCoach().critique(make_data(states))
        assert len(c.problem_timesteps) == 3

    def test_bell_shaped_velocity_profile_is_recognised(self):
        n = 15
        profile = np.sin(np.pi * np.arange(n - 1) / (n - 1)) * 0.01
        states = np.concatenate([[0.0], np.cumsum(profile)])
        c = GraceCoach(jerk_threshold=1e9).critique(make_data(states))
        assert "Natural bell-shaped velocity profile" in c.strengths
        assert c.overall_score == 10.0

    def test_front_loaded_velocity_profile_penalised(self):
        states = np.zeros(14)
        states[1] = 5.0   # velocity spike at the very start
        c = GraceCoach(jerk_threshold=1e9).critique(make_data(states))
        assert c.overall_score == pytest.approx(9.0)
        assert c.weaknesses == ["Velocity profile not bell-shaped"]
        assert c.improvement_suggestions == ["Peak velocity should occur mid-motion"]

    def test_uncoordinated_joints_penalised(self):
        states = np.zeros((14, 2))
        states[1, 0] = 5.0   # only joint 0 moves
        c = GraceCoach(jerk_threshold=1e9).critique(make_data(states))
        assert "Jerky, uncoordinated joint motion" in c.weaknesses
        assert "Synchronize joint movements" in c.improvement_suggestions
        assert c.overall_score == pytest.approx(10.0 - 1.0 - 1.5)

    def test_middling_coordination_gives_no_feedback(self):
        x = np.concatenate([[0.0], np.cumsum([0.01, 0.03, 0.02, 0.04, 0.01, 0.03, 0.02])])
        states = np.stack([x, 2 * x], axis=1)   # velocity-std ratio 1:2 -> coordination ~0.67
        c = GraceCoach(jerk_threshold=1e9).critique(make_data(states))
        assert "Well-coordinated multi-joint motion" not in c.strengths
        assert not any("uncoordinated" in w for w in c.weaknesses)

    def test_very_jerky_and_uncoordinated_motion_gets_harsh_reasoning(self):
        states = np.zeros((14, 2))
        states[1, 0] = 5.0
        c = GraceCoach().critique(make_data(states))
        assert c.overall_score == pytest.approx(3.5)
        assert c.reasoning.startswith("Motion appears jerky and mechanical")


class TestPhysicsOracle:
    def test_no_privileged_information_gives_neutral_score(self):
        c = PhysicsOracle().critique(make_data(ramp(5)))
        assert c.overall_score == 5.0
        assert c.dimension_scores == {}
        assert c.reasoning == "Physics-based evaluation: "
        assert c.confidence == 1.0 and c.teacher_name == "Physics Oracle"

    def test_reaching_the_goal_scores_ten(self):
        data = make_data(ramp(3), goal_distances=np.array([1.0, 0.5, 0.01]))
        c = PhysicsOracle().critique(data)
        assert c.dimension_scores == {MotionDimension.GOAL_ACHIEVEMENT: 10.0}
        assert c.strengths == ["Task completed successfully"]
        assert c.overall_score == 10.0

    def test_partial_progress_gets_partial_credit(self):
        data = make_data(ramp(2), goal_distances=np.array([1.0, 0.5]))
        c = PhysicsOracle().critique(data)
        assert c.dimension_scores[MotionDimension.GOAL_ACHIEVEMENT] == pytest.approx(5.0, abs=1e-4)
        assert c.strengths == ["Made progress toward goal (50.0%)"]

    def test_partial_credit_capped_below_full_success(self):
        data = make_data(ramp(2), goal_distances=np.array([1.0, 0.06]))
        c = PhysicsOracle().critique(data)
        assert c.dimension_scores[MotionDimension.GOAL_ACHIEVEMENT] == 8.0

    def test_moving_away_from_goal_scores_zero(self):
        data = make_data(ramp(2), goal_distances=np.array([1.0, 2.0]))
        c = PhysicsOracle().critique(data)
        assert c.dimension_scores[MotionDimension.GOAL_ACHIEVEMENT] == 0
        assert c.weaknesses == ["Moved away from goal"]

    def test_custom_success_threshold(self):
        data = make_data(ramp(2), goal_distances=np.array([1.0, 0.2]))
        c = PhysicsOracle(success_threshold=0.5).critique(data)
        assert c.dimension_scores[MotionDimension.GOAL_ACHIEVEMENT] == 10.0

    def test_collisions_cost_points(self):
        c = PhysicsOracle().critique(make_data(ramp(4), collisions=np.array([0, 1, 0, 0], dtype=bool)))
        assert c.dimension_scores == {MotionDimension.SAFETY: 5.0}
        assert c.weaknesses == ["1 collisions detected"]

    def test_collision_score_floors_at_zero(self):
        c = PhysicsOracle().critique(make_data(ramp(4), collisions=np.ones(4, dtype=bool)))
        assert c.dimension_scores[MotionDimension.SAFETY] == 0

    def test_zero_collisions_scores_ten(self):
        c = PhysicsOracle().critique(make_data(ramp(4), collisions=np.zeros(4, dtype=bool)))
        assert c.dimension_scores[MotionDimension.SAFETY] == 10.0
        assert c.strengths == ["Zero collisions"]

    def test_energy_reduces_efficiency_score(self):
        c = PhysicsOracle().critique(make_data(ramp(4), energy_usage=np.full(4, 62.5)))   # 250 / 100
        assert c.dimension_scores == {MotionDimension.EFFICIENCY: pytest.approx(7.5)}

    def test_energy_efficiency_floors_at_zero(self):
        c = PhysicsOracle().critique(make_data(ramp(4), energy_usage=np.full(4, 1e4)))
        assert c.dimension_scores[MotionDimension.EFFICIENCY] == 0

    def test_overall_is_mean_of_available_dimensions_and_reasoning_lists_feedback(self):
        data = make_data(
            ramp(3),
            goal_distances=np.array([1.0, 0.5, 0.01]),       # 10
            collisions=np.array([1, 0, 0], dtype=bool),       # 5
            energy_usage=np.array([100.0, 100.0, 100.0]),     # 7
        )
        c = PhysicsOracle().critique(data)
        assert c.overall_score == pytest.approx((10 + 5 + 7) / 3)
        assert c.reasoning == (
            "Physics-based evaluation: Task completed successfully; 1 collisions detected"
        )


class TestMotionTeacherConstruction:
    def test_default_personas(self):
        mt = MotionTeacher()
        assert [type(t) for t in mt.teachers] == [SafetyInspector, EfficiencyExpert, GraceCoach]
        assert mt.strategy == "vote"
        assert mt.weights == {"Safety Inspector": 1.0, "Efficiency Expert": 1.0, "Grace Coach": 1.0}

    def test_physics_oracle_persona_available(self):
        mt = MotionTeacher(personas=["physics_oracle"])
        assert isinstance(mt.teachers[0], PhysicsOracle)

    def test_unknown_persona_rejected(self):
        with pytest.raises(ValueError, match="Unknown persona: wizard"):
            MotionTeacher(personas=["safety_inspector", "wizard"])

    def test_custom_weights_are_kept(self):
        mt = MotionTeacher(weights={"Safety Inspector": 3.0})
        assert mt.weights == {"Safety Inspector": 3.0}

    def test_repr(self):
        mt = MotionTeacher(personas=["grace_coach"], strategy="rotate")
        assert repr(mt) == "MotionTeacher(teachers=['Grace Coach'], strategy='rotate')"


class TestMotionTeacherStrategies:
    def stubbed(self, strategy, critiques, weights=None):
        mt = MotionTeacher(personas=["safety_inspector"], strategy=strategy, weights=weights)
        mt.teachers = [StubTeacher(c.teacher_name, c) for c in critiques]
        if weights is None:
            mt.weights = {c.teacher_name: 1.0 for c in critiques}
        return mt

    def test_rotate_cycles_through_teachers(self):
        mt = MotionTeacher(strategy="rotate")
        data = make_data(ramp())
        names = [mt.critique(data).teacher_name for _ in range(4)]
        assert names == ["Safety Inspector", "Efficiency Expert", "Grace Coach", "Safety Inspector"]

    def test_vote_weighted_average(self):
        a = stub_critique("A", 8.0, dims={MotionDimension.SAFETY: 9.0})
        b = stub_critique("B", 2.0)
        mt = self.stubbed("vote", [a, b], weights={"A": 3.0})   # B defaults to 1.0
        result = mt.critique(make_data(ramp()))
        assert result.overall_score == pytest.approx((8.0 * 3 + 2.0) / 4)
        assert result.teacher_name == "Motion Teacher Committee"

    def test_vote_dimension_scores_fall_back_to_overall(self):
        a = stub_critique("A", 8.0, dims={MotionDimension.SAFETY: 9.0})
        b = stub_critique("B", 2.0, dims={MotionDimension.EFFICIENCY: 4.0})
        result = self.stubbed("vote", [a, b]).critique(make_data(ramp()))
        assert result.dimension_scores[MotionDimension.SAFETY] == pytest.approx((9.0 + 2.0) / 2)
        assert result.dimension_scores[MotionDimension.EFFICIENCY] == pytest.approx((8.0 + 4.0) / 2)

    def test_vote_merges_and_dedupes_feedback(self):
        a = stub_critique("A", 8.0, strengths=["s1", "shared"], weaknesses=["w1"],
                          suggestions=["x"], reasoning="alpha")
        b = stub_critique("B", 6.0, strengths=["shared"], weaknesses=["w2"],
                          suggestions=["x", "y"], reasoning="beta")
        result = self.stubbed("vote", [a, b]).critique(make_data(ramp()))
        assert sorted(result.strengths) == ["s1", "shared"]
        assert sorted(result.weaknesses) == ["w1", "w2"]
        assert sorted(result.improvement_suggestions) == ["x", "y"]
        assert result.reasoning == "A: alpha | B: beta"

    def test_vote_with_real_teachers_is_a_weighted_mean(self):
        data = make_data(ramp(11, step=0.4), collisions=np.zeros(11, dtype=bool))
        mt = MotionTeacher(personas=["safety_inspector", "grace_coach"])
        parts = [t.critique(data).overall_score for t in mt.teachers]
        result = mt.critique(data)
        assert result.overall_score == pytest.approx(sum(parts) / 2)
        assert MotionDimension.SAFETY in result.dimension_scores
        assert MotionDimension.SMOOTHNESS in result.dimension_scores

    def test_debate_with_unanimous_teachers(self):
        crits = [stub_critique("A", 7.0), stub_critique("B", 7.0)]
        result = self.stubbed("debate", crits).critique(make_data(ramp()))
        assert result.overall_score == pytest.approx(7.0)
        assert result.reasoning == "Debate consensus: 7.0/10 (std: 0.0)"
        assert result.teacher_name == "Motion Teacher Debate"

    def test_debate_downweights_outliers(self):
        scores = [8.0, 8.0, 2.0]
        crits = [stub_critique(n, s) for n, s in zip("ABC", scores)]
        result = self.stubbed("debate", crits).critique(make_data(ramp()))
        assert np.mean(scores) < result.overall_score < max(scores)
        assert result.reasoning.startswith(f"Debate consensus: {result.overall_score:.1f}/10")

    def test_unknown_strategy_raises_at_critique_time(self):
        mt = MotionTeacher(strategy="nope")   # type: ignore[arg-type]
        with pytest.raises(ValueError, match="Unknown strategy: nope"):
            mt.critique(make_data(ramp()))
