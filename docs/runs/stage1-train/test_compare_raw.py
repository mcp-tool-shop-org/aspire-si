"""G0's drift check reads M1 and M2 against A_0 under rules (a)-(c), with (c) meaning stop, not re-baseline."""

import json

import pytest

import compare_raw


def raw(tmp_path, name, tasks):
    """tasks: {task_id: [(text, new_tokens), ...]}"""
    p = tmp_path / f"{name}.jsonl"
    rows = [{"task_id": tid, "turns": [{"text": t, "prompt_tokens": 100, "new_tokens": n, "truncated": False,
                                        "seconds": 12.3 + k} for k, (t, n) in enumerate(turns)]}
            for tid, turns in tasks.items()]
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return str(p)


BASE = {"pl-01": [("<think>a</think>VERDICT: supported", 40)],
        "pl-02": [("<think>b</think>VERDICT: unsupported", 50), ("<think>c</think>VERDICT: unsupported", 30)]}


def run(tmp_path, a0, m1, m2):
    return compare_raw.main(["--a0", raw(tmp_path, "a0", a0), "--m1", raw(tmp_path, "m1", m1),
                             "--m2", raw(tmp_path, "m2", m2)])


def test_a_all_equal_proceeds_and_timing_is_ignored(tmp_path, capsys):
    assert run(tmp_path, BASE, BASE, BASE) == 0
    assert capsys.readouterr().out.strip().endswith("proceed on the existing A_0")


def test_b_runs_agree_but_differ_from_a0_means_re_baseline(tmp_path, capsys):
    moved = {**BASE, "pl-02": [("<think>b</think>VERDICT: unsupported", 50),
                               ("<think>d</think>VERDICT: cannot_tell", 31)]}
    assert run(tmp_path, BASE, moved, moved) == 1
    out = capsys.readouterr().out
    assert "m1_vs_a0: pl-02 turn 2: text differs" in out and "re-baseline A_0" in out


def test_c_runs_disagree_means_stop_not_re_baseline(tmp_path, capsys):
    other = {**BASE, "pl-01": [("<think>a</think>VERDICT: supported", 41)]}  # only new_tokens differs
    assert run(tmp_path, BASE, BASE, other) == 2  # M1 equals A_0, but M2 doesn't equal M1: still (c)
    out = capsys.readouterr().out
    assert "m1_vs_m2: pl-01 turn 1: new_tokens differs" in out
    assert "stop and fix the harness's determinism" in out and "re-baseline (G0b)" not in out


def test_a_missing_task_or_turn_counts_as_a_difference(tmp_path):
    fewer = {"pl-01": BASE["pl-01"]}
    assert run(tmp_path, BASE, fewer, fewer) == 1
    one_turn = {**BASE, "pl-02": BASE["pl-02"][:1]}
    assert run(tmp_path, BASE, one_turn, one_turn) == 1


TWICE = '{"task_id": "pl-01", "turns": []}\n' * 2  # one task id twice


@pytest.mark.parametrize("content", ["not json\n", TWICE])
def test_a_malformed_file_gives_no_verdict(tmp_path, capsys, content):
    bad = tmp_path / "bad.jsonl"
    bad.write_text(content, encoding="utf-8")
    good = raw(tmp_path, "good", BASE)
    assert compare_raw.main(["--a0", good, "--m1", str(bad), "--m2", good]) == 3
    assert capsys.readouterr().out.startswith("no verdict")
