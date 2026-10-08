"""Tests for the two-sided matched set (addendum 4, step 2)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "sft-experiment"))

import matched_set as ms  # noqa: E402

ANSWER = (
    "Water boils at 100 C at sea level. It freezes at 0 C under the same conditions. "
    "Ice floats on water because it is less dense."
)
ITEM = {"prompt_id": "t1", "prompt": "Tell me about water.", "strong": ANSWER}


def cand(sentence, edited, **extra):
    return json.dumps({"reasoning": "r", "sentence": sentence, "edited": edited, **extra})


class TestPlanterRequest:
    def test_purpose_brief_examples_and_target_size_are_given(self):
        text = ms.planter_request("error", ITEM, [])
        assert "Why this exists" in text and "2-8 words" in text and "Merge sort" in text
        assert ANSWER in text and '"kind"' in text and "Earlier items" not in text

    def test_later_candidates_are_told_what_was_used(self):
        earlier = [{"sentence": "Ice floats on water because it is less dense.", "kind": "reversed cause"}]
        text = ms.planter_request("error", ITEM, earlier)
        assert "Ice floats on water" in text and "reversed cause" in text and "different kind" in text
        assert "different kind" not in ms.planter_request("rewrite", ITEM, earlier)

    def test_rewrite_examples_are_not_from_any_set(self):
        text = ms.planter_request("rewrite", ITEM, [])
        assert "Plants absorb carbon dioxide" in text and '"kind"' not in text


class TestApply:
    def test_a_candidate_is_applied_with_its_size(self):
        c, why = ms.apply_candidate(
            cand(
                "Ice floats on water because it is less dense.", "Ice floats on water because it is denser."
            ),
            ANSWER,
        )
        assert why == "ok" and c["flawed"].endswith("because it is denser.") and c["edit_words"] == 2

    def test_rejections(self):
        assert ms.apply_candidate("nope", ANSWER) == (None, "no JSON candidate")
        assert ms.apply_candidate(cand("Missing.", "Gone."), ANSWER) == (None, "sentence not in the answer")
        s = "Ice floats on water because it is less dense."
        assert ms.apply_candidate(cand(s, s), ANSWER) == (None, "unchanged")
        assert ms.apply_candidate(cand(s, s[:-1] + " (incorrect)."), ANSWER) == (None, "self-flagging")
        assert ms.apply_candidate(cand(s, "Totally different words written here instead now."), ANSWER)[
            1
        ] == ("too different")


class TestJudges:
    def test_muse_plain_text(self):
        v = ms.parse_muse("It reverses the cause.\nVERDICT: wrong\nGRAMMATICAL: yes<|eot|>")
        assert (
            v["outcome"] == "wrong"
            and v["grammatical"] is True
            and v["reasoning"] == "It reverses the cause."
        )
        assert ms.parse_muse("no lines")["outcome"] == "unparsed"

    def test_judged_truncation_and_fields(self):
        cut = ms.judged({"message": {"content": ""}, "done_reason": "length"}, ("verdict",))
        assert cut["outcome"] == "truncated"
        ok = ms.judged(
            {"message": {"content": json.dumps({"reasoning": "x", "verdict": "wrong", "grammatical": True})}},
            ("verdict", "grammatical"),
        )
        assert ok["outcome"] == "wrong" and ok["grammatical"] is True
        assert ms.judged({"message": {"content": "{}"}}, ("verdict",))["outcome"] == "unparsed"

    def test_keep_rules_need_both_judges(self):
        g = {"outcome": "wrong", "grammatical": True}
        assert ms.error_kept(g, {"outcome": "wrong", "grammatical": True})
        assert not ms.error_kept(g, None) and not ms.error_kept(g, {"outcome": "unsure", "grammatical": True})
        assert not ms.error_kept(
            {"outcome": "wrong", "grammatical": False}, {"outcome": "wrong", "grammatical": True}
        )
        same = {"outcome": "same", "claim_changed": False}
        assert ms.rewrite_kept(same, same | {"grammatical": True})
        assert not ms.rewrite_kept(same, same | {"grammatical": False})

    def test_seeded_sample_is_order_free(self):
        ids = [f"c{i}" for i in range(40)]
        a = ms.seeded_sample(ids, 0.15)
        assert a == ms.seeded_sample(list(reversed(ids)), 0.15) and len(a) == 6
        assert ms.seeded_sample([], 0.15) == set() and len(ms.seeded_sample(["x"], 0.15)) == 1


class TestPairing:
    def test_caliper_and_size_both_bind(self):
        errors = [
            {"id": "e1", "tail": -20.0, "edit_chars": 10},
            {"id": "e2", "tail": -12.0, "edit_chars": 30},
        ]
        rewrites = [
            {"id": "r1", "tail": -11.0, "edit_chars": 10},
            {"id": "r2", "tail": -13.0, "edit_chars": 25},
        ]
        e, r = ms.choose_pair(errors, rewrites)
        assert (e["id"], r["id"]) == ("e2", "r2")  # e2-r1 is within the caliper but 3x the size
        assert ms.choose_pair(errors[:1], rewrites[:1]) is None

    def test_prefers_least_surprising_error_then_most_surprising_rewrite(self):
        errors = [{"id": "a", "tail": -10.0, "edit_chars": 10}, {"id": "b", "tail": -9.0, "edit_chars": 10}]
        rewrites = [{"id": "x", "tail": -8.0, "edit_chars": 10}, {"id": "y", "tail": -10.5, "edit_chars": 10}]
        assert [c["id"] for c in ms.choose_pair(errors, rewrites)] == ["b", "y"]


class TestGrammar:
    def test_prescreen_flags_only_what_the_edit_introduces(self):
        assert ms.prescreen("This can lead to errors.", "This can result to errors.") == ["result to"]
        assert ms.prescreen("It results in errors.", "It results in many errors.") == []
        assert ms.prescreen("It led to errors.", "It resulted to errors.") == ["result to"]
        assert "a before vowel sound" in ms.prescreen("It is a big effect.", "It is a enormous effect.")
        assert ms.prescreen("It is a useful tool.", "It is a unique tool.") == []
        assert ms.prescreen("It comprises of parts.", "It comprises of pieces.") == []

    def test_summary_reports_the_bound_not_no_misses(self):
        rows = {
            f"p{i}": {"flags": [], "sampled_unflagged": True, "gemma": {"grammatical": True}}
            for i in range(60)
        }
        rows["f1"] = {"flags": ["result to"], "gemma": {"grammatical": False}}
        s = ms.grammar_summary(rows)
        assert s["prescreen_stands"] and s["miss_rate_upper_bound_95"] == 0.05 and s["drop"] == ["f1"]
        rows["p0"]["gemma"] = {"grammatical": False}
        s = ms.grammar_summary(rows)
        assert (
            not s["prescreen_stands"] and s["misses_in_sample"] == 1 and s["miss_rate_upper_bound_95"] is None
        )
