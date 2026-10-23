"""Smoke test S1's pure parts: the pairing rule, the readout and pass line, and the device guard."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "examples" / "verifier-distill"))

import s1_nli_critic as s1  # noqa: E402

CLAIMS = {"a": "Claim A.", "b": "Claim B.", "c": "Claim C.", "d": "Claim D."}


def _line(cid, gold, verdict="supported", quote="q", found=True, ct="grounded"):
    return {
        "claim_id": cid,
        "gold_label": gold,
        "model_verdict": verdict,
        "quote_found": found,
        "evidence_quote": quote,
        "check_type": ct,
    }


class TestPairing:
    def test_rule_keeps_supported_found_quotes_and_dedupes_on_normalised_quote(self):
        lines = [
            _line("a", "supported", quote="x  =\n 1"),
            _line("a", "supported", quote="x = 1"),  # the same quote once whitespace is normalised
            _line("a", "supported", verdict="unsupported", quote="y"),  # cited to reject: never a pair
            _line("b", "unsupported", quote="z", found=False),  # not found: never a pair
            _line("b", "unsupported", quote="w"),
            _line("c", "cannot_tell", quote="v", ct="reasoning"),
            _line("d", "supported", quote=None),
        ]
        pairs = s1.build_pairs(lines, CLAIMS)
        assert [(p["claim_id"], p["quote"], p["role"]) for p in pairs] == [
            ("a", "x = 1", "positive"),
            ("b", "w", "fooled:unsupported"),
            ("c", "v", "fooled:cannot_tell"),
        ]
        assert pairs[0]["claim"] == "Claim A."

    def test_dedupe_happens_after_the_verdict_filter(self):
        # The bug a first count had: a rejection seen first must not swallow a later acceptance.
        lines = [
            _line("a", "supported", verdict="unsupported", quote="q"),
            _line("a", "supported", quote="q"),
        ]
        assert len(s1.build_pairs(lines, CLAIMS)) == 1

    def test_a_claim_missing_from_gold_stops_the_build(self):
        with pytest.raises(ValueError, match="not in the gold"):
            s1.build_pairs([_line("zz", "supported")], CLAIMS)


def _pairs(n_pos, n_fool, ct="grounded"):
    pos = [{"claim_id": f"p{i}", "quote": "q", "check_type": ct, "role": "positive"} for i in range(n_pos)]
    fool = [
        {"claim_id": f"f{i}", "quote": "q", "check_type": ct, "role": "fooled:unsupported"}
        for i in range(n_fool)
    ]
    return pos + fool


class TestReadout:
    def test_passes_only_when_both_lines_hold(self):
        pairs = _pairs(100, 100)
        labels = {(p["claim_id"], "q"): "entailment" for p in pairs[:90]}  # 90/100 positives entail
        labels |= {(p["claim_id"], "q"): "neutral" for p in pairs[90:]}
        labels[("f0", "q")] = "entailment"  # 1/100 fooled entail: Wilson upper ~0.054
        out = s1.readout(pairs, labels)["grounded"]
        assert out["positive"]["rate"] == 0.9 and out["fooled:unsupported"]["entailed"] == 1
        assert out["pass"] is True

    def test_fails_on_a_wide_fooled_interval_even_at_zero_entailed(self):
        pairs = _pairs(20, 10)  # 0/10 fooled entail: Wilson upper ~0.28 > 0.15
        labels = {
            (p["claim_id"], "q"): ("entailment" if p["role"] == "positive" else "neutral") for p in pairs
        }
        out = s1.readout(pairs, labels)["grounded"]
        assert out["fooled:unsupported"]["wilson95"][0] == 0.0
        assert out["pass"] is False

    def test_an_incomplete_run_never_passes_and_cannot_tell_is_not_gated(self):
        pairs = _pairs(100, 100) + [
            {"claim_id": "c0", "quote": "q", "check_type": "grounded", "role": "fooled:cannot_tell"}
        ]
        labels = {
            (p["claim_id"], "q"): ("entailment" if p["role"] == "positive" else "neutral") for p in pairs[:-1]
        }
        labels[("c0", "q")] = "entailment"  # reported, never gated
        assert s1.readout(pairs, labels)["grounded"]["pass"] is True
        del labels[("p0", "q")]
        out = s1.readout(pairs, labels)["grounded"]
        assert out["complete"] is False and out["pass"] is False

    def test_wilson_edges_are_exact(self):
        assert s1.wilson(0, 10)[0] == 0.0 and s1.wilson(10, 10)[1] == 1.0 and s1.wilson(0, 0) == (0.0, 1.0)


class _Core:
    def __init__(self, devices):
        self.available_devices = list(devices)
        self.names = devices

    def get_property(self, device, _):
        return self.names[device]


class TestDeviceGuard:
    def test_igpu_is_found_by_its_intel_name_and_the_5090_is_never_picked(self):
        core = _Core(
            {
                "CPU": "Intel CPU",
                "GPU.0": "NVIDIA GeForce RTX 5090 (dGPU)",
                "GPU.1": "Intel(R) Graphics (iGPU)",
                "NPU": "Intel(R) AI Boost",
            }
        )
        assert s1.pick_device(core, "iGPU") == "GPU.1"
        assert s1.pick_device(core, "NPU") == "NPU"
        with pytest.raises(SystemExit):
            s1.pick_device(_Core({"GPU.0": "NVIDIA GeForce RTX 5090"}), "iGPU")
        with pytest.raises(SystemExit):
            s1.pick_device(core, "GPU.0")

    def test_a_hung_call_times_out(self):
        import time

        value, err = s1.call_with_timeout(lambda: time.sleep(2) or 1, 0.05)
        assert (value, err) == (None, "timeout")
        assert s1.call_with_timeout(lambda: 7, 1.0) == (7, None)
