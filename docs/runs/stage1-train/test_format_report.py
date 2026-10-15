"""Format failures and truncations are classified apart from correctness; the report is descriptive, and it's
never written for the sealed set."""

import hashlib
import json

import pytest

import eval_adapter
import format_report

OK = {"verdict": "supported", "verdict_raw": "supported", "deciding": "x", "escalate": "no"}


def rec(tid, final=OK, truncated=False, closed=True, right=True, category="c1", tier=3):
    return {"task_id": tid, "category": category, "tier": tier, "final": dict(final),
            "turns": [{"truncated": truncated, "thinking_closed": closed}], "grade": {"verdict_correct": right}}


@pytest.mark.parametrize("r, cls", [
    (rec("a", truncated=True), "truncated"),
    (rec("b", closed=False), "thinking_unclosed"),
    (rec("c", final={**OK, "verdict": None, "verdict_raw": None}), "verdict_missing"),
    (rec("d", final={**OK, "verdict": None, "verdict_raw": "probably"}), "verdict_unrecognised"),
    (rec("e", final={**OK, "deciding": None}), "deciding_missing"),
    (rec("f", final={**OK, "escalate": None}), "escalate_missing"),
    (rec("g"), "ok"),
])
def test_each_task_gets_the_first_class_that_holds(r, cls):
    assert format_report.classify(r) == cls


def test_a_truncated_answer_counts_as_truncated_not_as_a_missing_line():
    r = rec("t", final={**OK, "verdict": None, "verdict_raw": None}, truncated=True)
    assert format_report.classify(r) == "truncated"


def test_the_summary_counts_classes_apart_from_correctness(tmp_path):
    recs = [rec("a", truncated=True), rec("b", right=False), rec("c"), rec("d", final={**OK, "escalate": None},
                                                                        tier=5)]
    raw = tmp_path / "raw.jsonl"
    raw.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    rep = format_report.write(raw)
    assert rep["counts"]["ok"] == 2 and rep["counts"]["truncated"] == 1 and rep["counts"]["escalate_missing"] == 1
    assert rep["ok_tasks"] == 2 and rep["ok_verdict_correct"] == 1
    assert rep["ids"]["truncated"] == ["a"] and rep["by"]["tier"]["5"]["escalate_missing"] == 1
    assert json.loads((tmp_path / "format-report.json").read_text(encoding="utf-8")) == rep


def test_the_sealed_set_is_recognised_so_no_report_is_written_for_it(tmp_path, monkeypatch):
    sealed = tmp_path / "sealed.jsonl"
    sealed.write_text('{"id": "sx-1"}\r\n', encoding="utf-8", newline="")
    monkeypatch.setattr(eval_adapter, "SEALED_SHA256", hashlib.sha256(b'{"id": "sx-1"}\n').hexdigest())
    dev = tmp_path / "dev.jsonl"
    dev.write_text('{"id": "pl-1"}\n', encoding="utf-8")
    assert eval_adapter.is_sealed(sealed) and not eval_adapter.is_sealed(dev)
