"""Smoke test S1's pure parts: the pairing rule, the readout and pass line, and the device guard."""

import hashlib
import json
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


# -- the fixes from review ji: a run that was cut short, stopped or left the device unhealthy never reads as a pass --


def _whole_run():
    """A run that clears the pass line on every check: 100 positives (90 entail), 100 fooled (1 entails), 1 cannot_tell."""
    pairs = _pairs(100, 100) + [
        {"claim_id": "c0", "quote": "q", "check_type": "grounded", "role": "fooled:cannot_tell"}
    ]
    labels = {
        (p["claim_id"], "q"): ("entailment" if p["role"] == "positive" else "neutral") for p in pairs
    }
    return pairs, labels


class TestWholeRun:
    def test_the_baseline_whole_run_passes(self):
        pairs, labels = _whole_run()
        out = s1.readout(pairs, labels, stopped=None, limit=None, health_after="OK")["grounded"]
        assert out["pass"] is True and out["complete"] is True

    def test_a_limit_run_reads_not_passed_and_not_complete_even_when_every_scored_pair_entails(self):
        pairs, labels = _whole_run()
        out = s1.readout(pairs, labels, limit=300)["grounded"]  # every pair of the cut set was scored
        assert out["pass"] is False and out["complete"] is False

    def test_a_stopped_run_reads_not_passed_and_not_complete(self):
        pairs, labels = _whole_run()
        stopped = {"pair": ["c0", "q"], "error": "timeout"}  # the stop came on the last pair; all earlier ones scored
        out = s1.readout(pairs, labels, stopped=stopped)["grounded"]
        assert out["pass"] is False and out["complete"] is False

    def test_an_unscored_cannot_tell_pair_makes_the_check_type_incomplete(self):
        pairs, labels = _whole_run()
        del labels[("c0", "q")]  # the trailing cannot_tell pair was never scored
        out = s1.readout(pairs, labels)["grounded"]
        assert out["complete"] is False and out["pass"] is False

    def test_an_unhealthy_device_afterwards_reads_not_passed(self):
        pairs, labels = _whole_run()
        out = s1.readout(pairs, labels, health_after="Error")["grounded"]
        assert out["pass"] is False and out["complete"] is False

    def test_no_limit_of_zero_or_none_cuts_the_run(self):
        pairs, labels = _whole_run()
        assert s1.readout(pairs, labels, limit=0)["grounded"]["pass"] is True
        assert s1.readout(pairs, labels, limit=None)["grounded"]["pass"] is True


class TestExitAndVerdict:
    def test_a_stop_exits_non_zero_and_a_whole_healthy_run_exits_zero(self):
        stopped = {"pair": ["a", "q"], "error": "timeout"}
        assert s1.exit_status(stopped, "OK") == 1
        assert s1.exit_status(None, "Error") == 1
        assert s1.exit_status(None, "") == 1
        assert s1.exit_status(None, "OK") == 0

    def test_s1_ok_is_printed_only_for_a_whole_healthy_run(self):
        stopped = {"pair": ["a", "q"], "error": "timeout"}
        assert s1.verdict_line(None, "OK", None) == "S1-OK"
        assert s1.verdict_line(None, "OK", 20) == "S1-LIMITED"
        assert s1.verdict_line(stopped, "OK", None) == "S1-STOPPED"
        assert s1.verdict_line(stopped, "OK", 20) == "S1-STOPPED"
        assert s1.verdict_line(None, "Error", None) == "S1-UNHEALTHY"

    def test_the_run_refuses_to_start_unless_health_reads_ok_whichever_device(self):
        s1.require_healthy_start("OK")
        for health in ("Error", "Unknown", "Degraded", ""):
            with pytest.raises(SystemExit, match="not starting"):
                s1.require_healthy_start(health)


def _write_gold(root):
    names = {"grounded.jsonl": None, "prs/grounded-prs.jsonl": True, "diffs/reasoning-diffs.jsonl": False}
    for i, (name, flag) in enumerate(names.items()):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        row = {"id": f"g{i}", "claim": f"Claim {i}."}
        if flag is not None:
            row["has_doc_comment"] = flag
        path.write_text(json.dumps(row) + "\n", encoding="utf-8", newline="\n")
    return names


class TestGoldProvenance:
    def test_the_receipt_carries_the_sha256_of_all_three_gold_files(self, tmp_path):
        names = _write_gold(tmp_path)
        _claims, _flags, gold_hashes = s1.read_gold(tmp_path)
        assert sorted(gold_hashes) == sorted(names) == sorted(s1.GOLD_FILES)
        for name in names:
            assert gold_hashes[name] == hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
        record = s1.provenance({"x/verdicts.jsonl": "ab"}, gold_hashes, "e48cc90c", 40)
        assert record["gold_files_sha256"] == gold_hashes and record["verdict_files_sha256"] == {"x/verdicts.jsonl": "ab"}
        assert record["rnd_commit"] == "e48cc90c" and record["limit"] == 40  # the limit is on the receipt

    def test_a_changed_gold_file_changes_its_hash(self, tmp_path):
        _write_gold(tmp_path)
        before = s1.read_gold(tmp_path)[2]
        (tmp_path / "grounded.jsonl").write_text('{"id": "g0", "claim": "Edited."}\n', encoding="utf-8", newline="\n")
        after = s1.read_gold(tmp_path)[2]
        assert before["grounded.jsonl"] != after["grounded.jsonl"]
        assert before["prs/grounded-prs.jsonl"] == after["prs/grounded-prs.jsonl"]

    def test_load_claims_still_returns_the_claim_text(self, tmp_path):
        _write_gold(tmp_path)
        assert s1.load_claims(tmp_path) == {"g0": "Claim 0.", "g1": "Claim 1.", "g2": "Claim 2."}


class TestStrata:
    def test_pairs_carry_the_gold_items_doc_comment_flag(self, tmp_path):
        _write_gold(tmp_path)
        claims, flags, _ = s1.read_gold(tmp_path)
        assert flags == {"g0": None, "g1": True, "g2": False}
        lines = [_line("g0", "supported", quote="a"), _line("g1", "supported", quote="b"), _line("g2", "supported", quote="c")]
        pairs = s1.build_pairs(lines, claims, flags)
        assert [p["has_doc_comment"] for p in pairs] == [None, True, False]
        assert s1.build_pairs([_line("a", "supported")], CLAIMS)[0]["has_doc_comment"] is None

    def test_the_readout_reports_doc_comment_and_code_only_pairs_as_separate_strata_and_never_gates_on_them(self):
        pairs = []
        for i in range(100):  # 100 positives with a doc comment that entail, 100 code-only that do not
            pairs.append({"claim_id": f"d{i}", "quote": "q", "check_type": "grounded", "role": "positive",
                          "has_doc_comment": True})
            pairs.append({"claim_id": f"k{i}", "quote": "q", "check_type": "grounded", "role": "positive",
                          "has_doc_comment": False})
        for i in range(100):
            pairs.append({"claim_id": f"u{i}", "quote": "q", "check_type": "grounded", "role": "fooled:unsupported",
                          "has_doc_comment": i % 2 == 0})
        labels = {}
        for p in pairs:
            entails = p["role"] == "positive" and p["has_doc_comment"]
            labels[(p["claim_id"], "q")] = "entailment" if entails else "neutral"
        out = s1.readout(pairs, labels)["grounded"]
        assert out["positive"]["rate"] == 0.5 and out["pass"] is False  # pooled, the prose comments mask the code lines
        strata = out["strata"]
        assert set(strata) == {"doc_comment", "code_only"}  # no pair is unflagged
        assert strata["doc_comment"]["positive"]["rate"] == 1.0 and strata["code_only"]["positive"]["rate"] == 0.0
        assert strata["doc_comment"]["fooled:unsupported"]["n"] == 50 == strata["code_only"]["fooled:unsupported"]["n"]
        # the same pooled result with the strata removed is the same decision: the strata are reported, not gated
        stripped = [{k: v for k, v in p.items() if k != "has_doc_comment"} for p in pairs]
        assert s1.readout(stripped, labels)["grounded"]["pass"] == out["pass"]

    def test_an_unflagged_pair_gets_its_own_stratum(self):
        pairs = [{"claim_id": "a", "quote": "q", "check_type": "grounded", "role": "positive", "has_doc_comment": None}]
        out = s1.readout(pairs, {("a", "q"): "entailment"})["grounded"]
        assert set(out["strata"]) == {"unflagged"} and out["strata"]["unflagged"]["positive"]["entailed"] == 1


class TestQuoteNormalisation:
    def test_a_whitespace_only_quote_is_no_quote(self):
        lines = [_line("a", "supported", quote="   \n\t "), _line("b", "unsupported", quote="w")]
        assert [p["claim_id"] for p in s1.build_pairs(lines, CLAIMS)] == ["b"]
