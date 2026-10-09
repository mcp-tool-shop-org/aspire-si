"""Tests for the context-rich meaning re-check (addendum 4)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "sft-experiment"))

import recheck as rc  # noqa: E402

PAIR = {
    "pair_id": "t1-w1",
    "prompt": "Why does ice float?",
    "strong": "Ice is less dense than water. So it floats.",
    "flawed": "Ice is less dense than water. Hence it floats.",
}


def reply(content, done="stop", thinking=""):
    return {"message": {"content": content, "thinking": thinking}, "done_reason": done}


class TestRequest:
    def test_both_full_answers_the_sentences_and_the_schema_are_sent(self):
        body = rc.request_body("mistral-small:24b", PAIR)
        text = body["messages"][1]["content"]
        assert PAIR["strong"] in text and PAIR["flawed"] in text and "Hence it floats." in text
        assert (
            body["format"]["required"][0] == "reasoning" and body["format"]["additionalProperties"] is False
        )
        assert "think" not in body and body["options"]["num_predict"] == 2048

    def test_gemma_thinks_with_room(self):
        body = rc.request_body("gemma4:31b", PAIR)
        assert body["think"] is True and body["options"]["num_predict"] == 16000
        assert body["options"]["num_ctx"] == 24576

    def test_the_worked_examples_are_not_from_any_set(self):
        assert "Plants use sunlight" in rc.REQUEST and "{strong}" in rc.REQUEST


class TestParse:
    def test_verdicts_truncation_and_unparsed(self):
        ok = rc.parse(reply(json.dumps({"reasoning": "r", "verdict": "same", "claim_changed": False})))
        assert ok["outcome"] == "same" and rc.keeps(ok) and ok["reasoning"] == "r"
        unsure = rc.parse(reply(json.dumps({"reasoning": "r", "verdict": "unsure", "claim_changed": False})))
        assert unsure["outcome"] == "unsure" and not rc.keeps(unsure)
        assert rc.parse(reply("", done="length", thinking="x" * 50))["outcome"] == "truncated"
        assert rc.parse(reply("{}"))["outcome"] == "unparsed"
        same_but_changed = rc.parse(
            reply(json.dumps({"reasoning": "", "verdict": "same", "claim_changed": True}))
        )
        assert not rc.keeps(same_but_changed)


class TestCombine:
    def test_a_pair_the_first_judge_dropped_need_not_be_asked_again(self):
        same = {"outcome": "same", "claim_changed": False}
        changed = {"outcome": "changed", "claim_changed": True}
        verdicts = {"mistral": {"a": same, "b": changed}, "gemma": {"a": same}}
        r = rc.combine(verdicts, ["a", "b"], old_dropped=set())
        assert (
            r["kept_ids"] == ["a"] and r["outcomes"]["gemma"]["not asked (the other judge dropped it)"] == 1
        )

    def test_both_judges_must_keep_and_the_first_check_is_compared(self):
        same = {"outcome": "same", "claim_changed": False}
        changed = {"outcome": "changed", "claim_changed": True}
        verdicts = {
            "mistral": {"a": same, "b": same, "c": changed, "d": {"outcome": "truncated"}},
            "gemma": {"a": same, "b": changed, "c": changed, "d": same},
        }
        r = rc.combine(verdicts, ["a", "b", "c", "d"], old_dropped={"a", "c"})
        assert r["kept_ids"] == ["a"] and r["judges_disagree"] == 2
        assert [(d["pair_id"], d["kept_by"]) for d in r["disagreements"]] == [
            ("b", "mistral"),
            ("d", "gemma"),
        ]
        assert r["disagreements"][0]["gemma"]["outcome"] == "changed"
        assert r["agreement"] == {"both_keep": 1, "only_mistral": 1, "only_gemma": 1, "neither": 1}
        assert r["outcomes"]["mistral"] == {"same": 2, "changed": 1, "truncated": 1}
        assert r["against_first_check"] == {
            "kept_by_both_checks": 0,
            "kept_now_dropped_before": 1,
            "dropped_now_kept_before": 2,
            "dropped_by_both": 1,
        }
