"""Additional behaviour tests for aspire.perception.syntropy (coverage gaps)."""

import math

import pytest
import torch
import torch.nn.functional as F

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

# ---------------------------------------------------------------------------
# SyntropicMeasurement / CoherenceField
# ---------------------------------------------------------------------------


class TestMeasurementState:
    @pytest.mark.parametrize(
        "score,state",
        [
            (-1.0, SyntropicState.ENTROPIC),
            (-0.31, SyntropicState.ENTROPIC),
            (-0.3, SyntropicState.NEUTRAL),
            (0.19, SyntropicState.NEUTRAL),
            (0.2, SyntropicState.SYNTROPIC),
            (0.49, SyntropicState.SYNTROPIC),
            (0.5, SyntropicState.RESONANT),
            (0.79, SyntropicState.RESONANT),
            (0.8, SyntropicState.CRYSTALLIZED),
            (1.0, SyntropicState.CRYSTALLIZED),
        ],
    )
    def test_compute_state_bands(self, score, state):
        assert SyntropicMeasurement(syntropy_score=score).compute_state() is state


class TestCoherenceField:
    def test_strength_is_an_exponential_moving_average(self):
        field = CoherenceField()
        field.update(1.0)
        assert field.strength == pytest.approx(0.3)
        field.update(1.0)
        assert field.strength == pytest.approx(0.3 * 1.0 + 0.7 * 0.3)
        assert list(field.strength_history) == pytest.approx([0.3, 0.51])

    def test_history_is_bounded(self):
        field = CoherenceField()
        for _ in range(150):
            field.update(0.5)
        assert len(field.strength_history) == 100

    def test_attractor_is_cloned_then_drifts_slowly(self):
        field = CoherenceField()
        first = torch.tensor([1.0, 0.0])
        field.update(0.5, first)
        assert torch.equal(field.attractor_embedding, first)
        first[0] = 99.0  # mutating the caller's tensor must not affect the field
        assert field.attractor_embedding[0] == 1.0
        field.update(0.5, torch.tensor([0.0, 1.0]))
        torch.testing.assert_close(field.attractor_embedding, torch.tensor([0.9, 0.1]))

    def test_attractor_untouched_without_embedding(self):
        field = CoherenceField()
        field.update(0.5)
        assert field.attractor_embedding is None

    def test_stability_untouched_until_five_samples(self):
        field = CoherenceField()
        for _ in range(4):
            field.update(1.0)
        assert field.stability == 0.5  # default

    def test_stability_is_high_for_steady_and_low_for_oscillating_signal(self):
        steady, noisy = CoherenceField(), CoherenceField()
        for _ in range(40):
            steady.update(0.5)
        for i in range(40):
            noisy.update(1.0 if i % 2 == 0 else 0.0)
        assert steady.stability > 0.99
        assert noisy.stability < steady.stability
        assert 0.0 <= noisy.stability <= 1.0

    def test_stability_formula(self):
        field = CoherenceField()
        for v in (0.1, 0.9, 0.2, 0.8, 0.3, 0.7):
            field.update(v)
        recent = list(field.strength_history)[-10:]
        mean = sum(recent) / len(recent)
        var = sum((x - mean) ** 2 for x in recent) / len(recent)
        assert field.stability == pytest.approx(max(0.0, min(1.0, 1 - 4 * var)))

    def test_stability_clamped_at_zero(self):
        field = CoherenceField()
        field.strength_history.extend([0.0, 10.0, 0.0, 10.0])
        field.update(0.0)
        assert field.stability == 0.0

    def test_convergence_rate_needs_three_points(self):
        field = CoherenceField()
        field.strength_history.extend([0.1, 0.2])
        assert field.get_convergence_rate() == 0.0

    def test_convergence_rate_sign_and_clamp(self):
        rising, falling, flat, steep = (CoherenceField() for _ in range(4))
        rising.strength_history.extend([0.0, 0.05, 0.10, 0.15, 0.20])
        falling.strength_history.extend([0.20, 0.15, 0.10, 0.05, 0.0])
        flat.strength_history.extend([0.4] * 5)
        steep.strength_history.extend([0.0, 1.0, 2.0, 3.0, 4.0])
        assert rising.get_convergence_rate() == pytest.approx(0.25)  # slope 0.05 * 5
        assert falling.get_convergence_rate() == pytest.approx(-0.25)
        assert flat.get_convergence_rate() == 0.0
        assert steep.get_convergence_rate() == 1.0
        steep.strength_history.clear()
        steep.strength_history.extend([4.0, 3.0, 2.0, 1.0, 0.0])
        assert steep.get_convergence_rate() == -1.0

    def test_convergence_rate_only_looks_at_last_five(self):
        field = CoherenceField()
        field.strength_history.extend([5.0, 4.0, 3.0] + [0.1] * 5)
        assert field.get_convergence_rate() == 0.0

    def test_convergence_rate_with_degenerate_history_object(self):
        class Odd:
            """Reports 3 entries but yields only one (defensive branch)."""

            def __len__(self):
                return 3

            def __iter__(self):
                return iter([0.5])

        field = CoherenceField()
        field.strength_history = Odd()
        assert field.get_convergence_rate() == 0.0


# ---------------------------------------------------------------------------
# negentropy / semantic coherence
# ---------------------------------------------------------------------------


class TestNegentropy:
    @pytest.mark.parametrize("method", ["exp", "logcosh", "kurtosis"])
    def test_gaussian_scores_near_zero_and_structured_signal_scores_higher(self, method):
        torch.manual_seed(0)
        gaussian = torch.randn(20000)
        bimodal = torch.tensor([1.0, -1.0]).repeat(10000)
        g = compute_negentropy_approximation(gaussian, method).item()
        b = compute_negentropy_approximation(bimodal, method).item()
        assert g >= 0 and b >= 0
        assert g < 1e-3
        assert b > 10 * g

    @pytest.mark.parametrize("method", ["exp", "logcosh", "kurtosis"])
    def test_invariant_to_scale_and_shift(self, method):
        torch.manual_seed(1)
        x = torch.rand(500)
        a = compute_negentropy_approximation(x, method)
        b = compute_negentropy_approximation(3.0 * x + 5.0, method)
        torch.testing.assert_close(a, b, rtol=1e-3, atol=1e-6)

    def test_kurtosis_of_two_point_distribution(self):
        bimodal = torch.tensor([1.0, -1.0]).repeat(5000)
        value = compute_negentropy_approximation(bimodal, "kurtosis").item()
        assert value == pytest.approx((1 / 48) * (1 - 3) ** 2, rel=0.01)

    def test_unknown_method_rejected(self):
        with pytest.raises(ValueError, match="Unknown method: nope"):
            compute_negentropy_approximation(torch.randn(10), "nope")

    def test_constant_input_is_finite(self):
        for method in ("exp", "logcosh", "kurtosis"):
            assert torch.isfinite(compute_negentropy_approximation(torch.ones(8), method))


class TestSemanticCoherence:
    def test_single_token_sequences_are_perfectly_coherent(self):
        out = compute_semantic_coherence(torch.randn(3, 1, 8))
        assert out.shape == (3,) and torch.all(out == 1.0)

    def test_2d_input_is_treated_as_one_batch_element(self):
        out = compute_semantic_coherence(torch.randn(6, 8))
        assert out.shape == (1,)

    def test_identical_embeddings_are_fully_coherent(self):
        v = torch.randn(1, 1, 8).expand(2, 7, 8)
        torch.testing.assert_close(compute_semantic_coherence(v), torch.ones(2))

    def test_short_sequences_use_adjacent_similarity(self):
        # seq_len (3) < window_size (5): local coherence = mean adjacent similarity
        v = torch.randn(1, 1, 8).expand(1, 3, 8)
        torch.testing.assert_close(compute_semantic_coherence(v), torch.ones(1))

    def test_alternating_directions_clamp_to_zero(self):
        v = torch.tensor([1.0, 0.0, 0.0, 0.0])
        seq = torch.stack([v, -v, v, -v, v, -v]).unsqueeze(0)
        assert compute_semantic_coherence(seq, window_size=2).item() == 0.0

    def test_coherent_beats_random_and_output_is_bounded(self):
        torch.manual_seed(3)
        base = torch.randn(1, 1, 16)
        coherent = base + 0.05 * torch.randn(4, 12, 16)
        scrambled = torch.randn(4, 12, 16)
        c = compute_semantic_coherence(coherent)
        s = compute_semantic_coherence(scrambled)
        assert torch.all(c > s)
        assert torch.all((c >= 0) & (c <= 1)) and torch.all((s >= 0) & (s <= 1))

    def test_windowed_path_matches_manual_computation(self):
        torch.manual_seed(4)
        emb = torch.randn(1, 6, 8)
        norm = F.normalize(emb, dim=-1)
        window = 3
        locals_ = []
        for i in range(6 - window + 1):
            w = norm[0, i : i + window]
            sim = w @ w.T
            locals_.append((sim.sum() - sim.diagonal().sum()) / (window * (window - 1)))
        local = torch.stack(locals_).mean()
        mean = norm.mean(dim=1)[0]
        glob = ((norm[0, 0] @ mean) + (norm[0, -1] @ mean)) / 2
        expected = (0.6 * local + 0.4 * glob).clamp(0, 1)
        torch.testing.assert_close(compute_semantic_coherence(emb, window_size=window)[0], expected)


# ---------------------------------------------------------------------------
# Resonance detector / integrator
# ---------------------------------------------------------------------------


class TestResonanceDetector:
    def test_outputs_and_definitions(self):
        torch.manual_seed(0)
        det = SyntropicResonanceDetector(hidden_dim=16).eval()
        agent, user = torch.randn(3, 16), torch.randn(3, 16)
        out = det(agent, user)
        assert out["resonance"].shape == (3,) and torch.all((out["resonance"] > 0) & (out["resonance"] < 1))
        assert out["agent_proj"].shape == (3, 8) and out["attractor"].shape == (3, 8)
        torch.testing.assert_close(out["similarity"], F.cosine_similarity(out["agent_proj"], out["user_proj"], dim=-1))
        expected = (
            (1 - F.cosine_similarity(out["agent_proj"], out["attractor"], dim=-1))
            + (1 - F.cosine_similarity(out["user_proj"], out["attractor"], dim=-1))
        ) / 2
        torch.testing.assert_close(out["divergence"], expected)
        assert torch.all((out["divergence"] >= 0) & (out["divergence"] <= 2))

    def test_gradients_reach_both_projectors(self):
        det = SyntropicResonanceDetector(hidden_dim=16)
        out = det(torch.randn(2, 16), torch.randn(2, 16))
        (out["resonance"].sum() + out["divergence"].sum()).backward()
        assert det.agent_projector.weight.grad is not None
        assert det.user_projector.weight.grad is not None


class TestIntegrator:
    def test_outputs_without_mask(self):
        torch.manual_seed(0)
        integ = SyntropicIntegrator(hidden_dim=16, num_heads=4).eval()
        out = integ(torch.randn(2, 5, 16))
        assert out["integration_score"].shape == (2,)
        assert out["emergence_score"].shape == (2,)
        assert out["integrated"].shape == (2, 5, 16) and out["pooled"].shape == (2, 16)
        assert out["attention_weights"].shape == (2, 5, 5)
        # attention entropy is bounded by log(seq_len), so coherence lies in [0, 1]
        assert torch.all((out["coherence"] >= -1e-5) & (out["coherence"] <= 1 + 1e-5))

    def test_mask_zeroes_attention_to_padding_and_ignores_it_when_pooling(self):
        torch.manual_seed(0)
        integ = SyntropicIntegrator(hidden_dim=16, num_heads=4).eval()
        hs = torch.randn(1, 6, 16)
        mask = torch.tensor([[1, 1, 1, 1, 0, 0]])
        out = integ(hs, mask)
        assert torch.all(out["attention_weights"][..., 4:] == 0)
        mask_f = mask.unsqueeze(-1).float()
        expected = (out["integrated"] * mask_f).sum(dim=1) / mask_f.sum(dim=1)
        torch.testing.assert_close(out["pooled"], expected)

    def test_padding_content_does_not_change_the_pooled_result(self):
        torch.manual_seed(0)
        integ = SyntropicIntegrator(hidden_dim=16, num_heads=4).eval()
        hs = torch.randn(1, 6, 16)
        mask = torch.tensor([[1, 1, 1, 1, 0, 0]])
        changed = hs.clone()
        changed[:, 4:] = torch.randn(1, 2, 16) * 50
        torch.testing.assert_close(integ(hs, mask)["pooled"], integ(changed, mask)["pooled"], atol=1e-5, rtol=1e-4)

    def test_single_token_sequence_has_finite_coherence(self):
        integ = SyntropicIntegrator(hidden_dim=16, num_heads=4).eval()
        out = integ(torch.randn(2, 1, 16))
        assert torch.isfinite(out["coherence"]).all()


# ---------------------------------------------------------------------------
# SyntropicFlowTracker
# ---------------------------------------------------------------------------


def _record_all(tracker, scores):
    for s in scores:
        tracker.record(SyntropicMeasurement(syntropy_score=s))
    return tracker


class TestFlowTracker:
    def test_record_tracks_cumulative_peak_and_field(self):
        tracker = SyntropicFlowTracker()
        _record_all(tracker, [0.5, -0.4, 0.2])
        assert tracker.cumulative_syntropy == pytest.approx(0.5 * 0.1 + 0.2 * 0.1)  # negatives ignored
        assert tracker.peak_syntropy == 0.5
        assert len(tracker.coherence_field.strength_history) == 3

    def test_peak_never_goes_negative(self):
        tracker = _record_all(SyntropicFlowTracker(), [-0.9, -0.5])
        assert tracker.peak_syntropy == 0.0

    def test_history_is_bounded(self):
        tracker = _record_all(SyntropicFlowTracker(max_history=4), [0.1 * i for i in range(10)])
        assert len(tracker.measurements) == 4
        assert tracker.measurements[-1].syntropy_score == pytest.approx(0.9)

    def test_trajectory_needs_three_measurements(self):
        assert _record_all(SyntropicFlowTracker(), [0.1, 0.9]).get_trajectory() == 0.0

    def test_trajectory_sign_scale_and_clamp(self):
        up = _record_all(SyntropicFlowTracker(), [0.0, 0.1, 0.2, 0.3])
        down = _record_all(SyntropicFlowTracker(), [0.3, 0.2, 0.1, 0.0])
        flat = _record_all(SyntropicFlowTracker(), [0.4, 0.4, 0.4])
        steep = _record_all(SyntropicFlowTracker(), [0.0, 5.0, 10.0])
        assert up.get_trajectory() == pytest.approx(0.1 * 3)
        assert down.get_trajectory() == pytest.approx(-0.1 * 3)
        assert flat.get_trajectory() == 0.0
        assert steep.get_trajectory() == 1.0

    def test_trajectory_uses_only_the_window(self):
        tracker = _record_all(SyntropicFlowTracker(window_size=3), [5.0, 4.0, 3.0, 0.1, 0.1, 0.1])
        assert tracker.get_trajectory() == 0.0

    def test_window_of_one_has_no_trend(self):
        tracker = _record_all(SyntropicFlowTracker(window_size=1), [0.1, 0.5, 0.9])
        assert tracker.get_trajectory() == 0.0

    def test_empty_state_is_neutral(self):
        assert SyntropicFlowTracker().get_state() is SyntropicState.NEUTRAL

    @pytest.mark.parametrize(
        "score,state",
        [
            (-0.5, SyntropicState.ENTROPIC),
            (-0.19, SyntropicState.NEUTRAL),
            (0.0, SyntropicState.NEUTRAL),
            (0.25, SyntropicState.SYNTROPIC),
            (0.5, SyntropicState.RESONANT),
            (0.85, SyntropicState.CRYSTALLIZED),
        ],
    )
    def test_state_bands_from_recent_average(self, score, state):
        assert _record_all(SyntropicFlowTracker(), [score] * 5).get_state() is state

    def test_state_uses_only_the_last_five(self):
        tracker = _record_all(SyntropicFlowTracker(), [-0.9] * 10 + [0.6] * 5)
        assert tracker.get_state() is SyntropicState.RESONANT

    def test_steep_decline_is_entropic_even_with_a_high_average(self):
        tracker = _record_all(SyntropicFlowTracker(), [0.9, 0.75, 0.6, 0.45, 0.3])
        assert tracker.get_trajectory() < -0.3
        assert sum(m.syntropy_score for m in tracker.measurements) / 5 > 0.2
        assert tracker.get_state() is SyntropicState.ENTROPIC

    def test_summary(self):
        empty = SyntropicFlowTracker().get_summary()
        assert empty["current_syntropy"] == 0.0 and empty["state"] == "neutral"
        tracker = _record_all(SyntropicFlowTracker(), [0.1, 0.2, 0.3])
        s = tracker.get_summary()
        assert s["current_syntropy"] == 0.3 and s["peak"] == 0.3
        assert s["cumulative"] == pytest.approx(0.06)
        assert s["trajectory"] == tracker.get_trajectory()
        assert s["coherence_field_strength"] == tracker.coherence_field.strength
        assert set(s) == {
            "current_syntropy", "trajectory", "state", "cumulative", "peak",
            "coherence_field_strength", "coherence_field_stability", "convergence_rate",
        }  # fmt: skip

    def test_reset(self):
        tracker = _record_all(SyntropicFlowTracker(max_history=7), [0.5, 0.6, 0.7])
        tracker.reset()
        assert len(tracker.measurements) == 0 and tracker.measurements.maxlen == 7
        assert tracker.cumulative_syntropy == 0.0 and tracker.peak_syntropy == 0.0
        assert tracker.coherence_field.strength == 0.0


# ---------------------------------------------------------------------------
# SyntropicEngine
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    torch.manual_seed(0)
    return SyntropicEngine(hidden_dim=16).eval()


class TestEngine:
    def test_forward_without_user_state(self, engine):
        out = engine(torch.randn(2, 5, 16))
        assert out["syntropy_score"].shape == (2,)
        assert torch.all((out["syntropy_score"] >= -1) & (out["syntropy_score"] <= 1))
        assert out["dimension_scores"].shape == (2, len(SyntropicDimension))
        assert torch.all(out["resonance"] == 0) and torch.all(out["divergence"] == 1)
        assert out["negentropy"].dim() == 0

    def test_forward_with_user_state_uses_resonance_detector(self, engine):
        hs, user = torch.randn(2, 5, 16), torch.randn(2, 16)
        out = engine(hs, user_state=user)
        expected = engine.resonance_detector(hs.mean(dim=1), user)
        torch.testing.assert_close(out["resonance"], expected["resonance"])
        torch.testing.assert_close(out["divergence"], expected["divergence"])

    def test_forward_with_attention_mask_pools_valid_tokens(self, engine):
        hs = torch.randn(1, 6, 16)
        mask = torch.tensor([[1, 1, 1, 0, 0, 0]])
        out = engine(hs, attention_mask=mask)
        torch.testing.assert_close(out["agent_pooled"], hs[:, :3].mean(dim=1))

    def test_syntropy_score_is_the_weighted_mean_rescaled(self, engine):
        out = engine(torch.randn(2, 5, 16))
        dims = out["dimension_scores"]
        weights = torch.tensor([1.0, 1.0, 0.8, 1.2, 1.1, 1.0, 1.0, 1.0, 0.9, 0.8, 0.9, 0.7])
        expected = 2 * (dims * weights).sum(dim=-1) / weights.sum() - 1
        torch.testing.assert_close(out["syntropy_score"], expected)

    def test_measure_builds_measurement_and_updates_tracker(self, engine):
        m1 = engine.measure(torch.randn(2, 5, 16))
        assert isinstance(m1, SyntropicMeasurement)
        assert set(m1.dimension_scores) == set(SyntropicDimension)
        assert m1.state is m1.compute_state()
        assert m1.negentropy >= 0
        assert m1.trajectory == 0.0
        assert len(engine.flow_tracker.measurements) == 1
        engine.measure(torch.randn(2, 5, 16))
        engine.measure(torch.randn(2, 5, 16))
        before = engine.flow_tracker.get_trajectory()  # three measurements recorded so far
        m4 = engine.measure(torch.randn(2, 5, 16))
        assert len(engine.flow_tracker.measurements) == 4
        # the measurement carries the trajectory as it stood *before* it was recorded
        assert m4.trajectory == pytest.approx(before)

    def test_measure_with_user_state_and_mask_does_not_track_grads(self, engine):
        m = engine.measure(torch.randn(2, 4, 16), torch.randn(2, 16), torch.ones(2, 4))
        assert isinstance(m.syntropy_score, float)

    def test_measure_handles_non_scalar_negentropy(self, engine, monkeypatch):
        import aspire.perception.syntropy as mod

        monkeypatch.setattr(mod, "compute_negentropy_approximation", lambda x, method="exp": torch.tensor([0.2, 0.4]))
        assert engine.measure(torch.randn(2, 4, 16)).negentropy == pytest.approx(0.3)

    def test_reset_clears_tracker(self, engine):
        engine.measure(torch.randn(1, 4, 16))
        engine.reset()
        assert len(engine.flow_tracker.measurements) == 0

    def test_gradients_flow_through_forward(self):
        eng = SyntropicEngine(hidden_dim=16)
        out = eng(torch.randn(2, 4, 16), torch.randn(2, 16))
        out["syntropy_score"].sum().backward()
        assert eng.syntropy_head[0].weight.grad is not None


class TestGuidance:
    def _engine_with(self, scores):
        eng = SyntropicEngine(hidden_dim=16)
        _record_all(eng.flow_tracker, scores)
        return eng

    def test_each_state_has_its_own_guidance(self):
        texts = {
            "entropic": ("Coherence is dissolving", [-0.5] * 5),
            "neutral": ("stable but not generating", [0.0] * 5),
            "syntropic": ("Order is being created", [0.3] * 5),
            "resonant": ("High resonance with user", [0.6] * 5),
            "crystallized": ("Insight has crystallized", [0.9] * 5),
        }
        for _, (fragment, scores) in texts.items():
            guidance = self._engine_with(scores).get_syntropic_guidance()
            assert guidance.startswith("SYNTROPIC GUIDANCE:")
            assert fragment in guidance
        # exactly one state paragraph each
        assert "Order is being created" not in self._engine_with([0.9] * 5).get_syntropic_guidance()

    def test_declining_trajectory_warning(self):
        guidance = self._engine_with([0.9, 0.75, 0.6, 0.45, 0.3]).get_syntropic_guidance()
        assert "WARNING: Syntropy is declining" in guidance
        assert "Syntropy is increasing" not in guidance

    def test_increasing_trajectory_message(self):
        guidance = self._engine_with([0.0, 0.15, 0.3, 0.45, 0.6]).get_syntropic_guidance()
        assert "Syntropy is increasing" in guidance
        assert "WARNING" not in guidance

    def test_weak_coherence_field_message(self):
        assert "Coherence field is weak" in self._engine_with([]).get_syntropic_guidance()
        assert "Coherence field is weak" not in self._engine_with([0.9] * 8).get_syntropic_guidance()


# ---------------------------------------------------------------------------
# empathy helpers
# ---------------------------------------------------------------------------


class TestEmpathy:
    def test_empathic_syntropy_weights_emotion_over_intent(self):
        a = torch.tensor([[1.0, 0.0]])
        same, opposite = torch.tensor([[1.0, 0.0]]), torch.tensor([[-1.0, 0.0]])
        assert compute_empathic_syntropy(a, same, a, opposite).item() == pytest.approx(0.6 - 0.4)
        assert compute_empathic_syntropy(a, opposite, a, same).item() == pytest.approx(-0.6 + 0.4)
        assert compute_empathic_syntropy(a, same, a, same).item() == pytest.approx(1.0)

    def test_empathic_syntropy_is_per_batch_element(self):
        e1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        e2 = torch.tensor([[1.0, 0.0], [0.0, -1.0]])
        out = compute_empathic_syntropy(e1, e2, e1, e1)
        assert out.shape == (2,)
        assert out[0].item() == pytest.approx(1.0)
        assert out[1].item() == pytest.approx(-0.6 + 0.4)

    def test_evaluator_reports_all_keys_and_guides(self):
        torch.manual_seed(0)
        ev = SyntropicEmpathyEvaluator(hidden_dim=16)
        result = ev.evaluate(torch.randn(2, 5, 16), torch.randn(2, 16))
        assert set(result) == {
            "syntropic_empathy", "empathic_resonance", "intentional_alignment", "knowledge_bridging",
            "meaning_generation", "coherence_field_strength", "convergence_rate", "syntropic_state",
        }  # fmt: skip
        assert result["syntropic_state"] in {s.value for s in SyntropicState}
        assert ev.get_guidance().startswith("SYNTROPIC GUIDANCE:")
        assert len(ev.engine.flow_tracker.measurements) == 1
        ev.reset()
        assert len(ev.engine.flow_tracker.measurements) == 0

    def test_evaluator_accepts_mask_and_no_user_state(self):
        ev = SyntropicEmpathyEvaluator(hidden_dim=16)
        result = ev.evaluate(torch.randn(1, 4, 16), attention_mask=torch.ones(1, 4))
        assert math.isfinite(result["syntropic_empathy"])
