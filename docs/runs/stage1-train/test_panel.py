"""Panel builder, scorer, runner and guard.

Each test feeds a broken input and expects a refusal or a wrong score.
"""

import hashlib
import json
from pathlib import Path

import pytest

import batched
import det_run
import eval_adapter
import panel
import panel_build
import panel_guard
import panel_leak
import panel_score
import parity
import receipts

WORDS = "amber birch cedar drift elm fir granite hazel"
QID = "A" * 32


def _reply(answer: str, *, closed: bool = True, thinking: str = "working") -> dict:
    if closed:
        text = f"<think>\n{thinking}\n</think>\n{answer}"
    else:
        text = f"<think>\n{thinking}\n{answer}"
    return {"text": text, "truncated": False}


def _gsm(question: str, answer: str) -> str:
    return json.dumps({"question": question, "answer": answer})


def _wino(qid: str, sentence: str, answer: str = "1") -> str:
    row = {"qID": qid, "sentence": sentence, "option1": "cat", "option2": "dog", "answer": answer}
    return json.dumps(row)


def _sources(tmp_path: Path, gsm_lines: list[str], wino_lines: list[str]):
    gsm, wino = tmp_path / "gsm.jsonl", tmp_path / "wino.jsonl"
    gsm.write_text("\n".join(gsm_lines) + "\n", encoding="utf-8", newline="\n")
    wino.write_text("\n".join(wino_lines) + "\n", encoding="utf-8", newline="\n")
    return gsm, wino, panel_build.sha256_bytes(gsm), panel_build.sha256_bytes(wino)


def _row(key: str, ident: str, gold, source_text: str, source: str = "gsm8k") -> dict:
    return {"source": source, "id": ident, "key": key, "gold": gold, "source_text": source_text,
            "prompt": source_text}


def _rights(flags, sha: str = "item", scorer: str = "scorer", model: str = "base",
            adapter_sha: str | None = None) -> dict:
    return {"item_sha256": sha, "scorer_sha256": scorer, "model": model,
            "adapter_sha256": adapter_sha,
            "items": [{"id": f"i{i}", "right": flag} for i, flag in enumerate(flags)]}


def test_gold_answer_scores_right_and_a_wrong_one_does_not():
    right = panel_score.score_generation(_reply("ANSWER: 42"), "42", "gsm8k")
    wrong = panel_score.score_generation(_reply("ANSWER: 41"), "42", "gsm8k")
    assert right["right"] and right["parsed"] == "42"
    assert not wrong["right"] and not wrong["unparsed"]
    letter = panel_score.score_generation(_reply("ANSWER: A"), "A", "winogrande")
    assert letter["right"]


def test_last_of_several_answer_lines_wins():
    scored = panel_score.score_generation(_reply("ANSWER: 1\nANSWER: 9"), "9", "gsm8k")
    assert scored["parsed"] == "9" and scored["right"]
    assert not panel_score.score_generation(_reply("ANSWER: 1\nANSWER: 9"), "1", "gsm8k")["right"]


def test_answer_inside_thinking_is_never_read():
    text = _reply("no number here", thinking="ANSWER: 5")
    scored = panel_score.score_generation(text, "5", "gsm8k")
    assert scored["parsed"] is None and not scored["right"] and scored["unparsed"]


def test_truncated_trace_with_an_answer_is_not_right():
    generation = _reply("ANSWER: 5")
    generation["truncated"] = True
    scored = panel_score.score_generation(generation, "5", "gsm8k")
    assert scored["truncated"] and scored["parsed"] == "5" and not scored["unparsed"] and not scored["right"]


def test_unclosed_think_is_empty():
    scored = panel_score.score_generation(_reply("ANSWER: 5", closed=False), "5", "gsm8k")
    assert scored["empty"] and scored["unparsed"] and not scored["right"] and scored["parsed"] is None


def test_empty_answer_is_unparsed():
    scored = panel_score.score_generation(_reply("   "), "5", "gsm8k")
    assert scored["empty"] and scored["unparsed"] and not scored["right"]


def test_malformed_number_is_unparsed():
    for bad in ("ANSWER: 3.5", "ANSWER: 1/2", "ANSWER: twelve", "ANSWER: 5 apples"):
        scored = panel_score.score_generation(_reply(bad), "5", "gsm8k")
        assert scored["unparsed"] and not scored["right"] and scored["parsed"] is None, bad


def test_wrong_letter_and_a_non_letter_are_not_right():
    wrong = panel_score.score_generation(_reply("ANSWER: B"), "A", "winogrande")
    assert wrong["parsed"] == "B" and not wrong["right"] and not wrong["unparsed"]
    other = panel_score.score_generation(_reply("ANSWER: C"), "A", "winogrande")
    assert other["unparsed"] and not other["right"]


def test_commas_are_removed_for_an_exact_integer():
    scored = panel_score.score_generation(_reply("ANSWER: 1,234"), "1234", "gsm8k")
    assert scored["right"] and scored["parsed"] == "1234"
    assert not panel_score.score_generation(_reply("ANSWER: 1,235"), "1234", "gsm8k")["right"]


def test_strict_one_line_is_not_required_for_a_right_answer():
    scored = panel_score.score_generation(_reply("The total follows.\nANSWER: 4"), "4", "gsm8k")
    assert scored["right"] and not scored["strict_one_line"]


def test_machine_line_mismatch_exits_before_load(monkeypatch):
    loaded = []
    monkeypatch.setattr(det_run, "machine_line", lambda: {
        "gpu": "different", "driver": "0", "cuda": "0", "torch": "0", "transformers": "0"})

    def load(_line):
        loaded.append("load")

    with pytest.raises(SystemExit) as caught:
        panel.prepare(load)
    assert caught.value.code == 5
    assert loaded == []


def test_machine_line_sha_mismatch_exits_before_the_live_line(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(det_run, "machine_line", lambda: calls.append("read") or {})
    other = tmp_path / "line.json"
    other.write_text("{}\n", encoding="utf-8", newline="\n")
    with pytest.raises(SystemExit) as caught:
        panel.check_machine(other)
    assert caught.value.code == 5
    assert calls == []


def test_incomplete_machine_line_exits(monkeypatch):
    monkeypatch.setattr(det_run, "machine_line", lambda: {
        "gpu": "", "driver": "", "cuda": "1", "torch": "1", "transformers": "1"})
    with pytest.raises(SystemExit) as caught:
        panel.check_machine()
    assert caught.value.code == 5


def test_cap_matches_parity_and_refuses_an_unknown_reading():
    seen = []
    fraction = panel.memory_cap(1000.0, 250.0, seen.append)
    assert fraction == parity.cap_fraction(1000.0, 250.0)
    assert seen == [fraction]

    def refuse(_fraction):
        raise AssertionError("a missing reading must not set a fraction")

    with pytest.raises(SystemExit, match="0"):
        panel.memory_cap(1000.0, None, refuse)


def test_unknown_cap_exits_before_load(monkeypatch):
    frozen = json.loads(panel.MACHINE_LINE.read_text(encoding="utf-8"))
    monkeypatch.setattr(det_run, "machine_line", lambda: frozen)
    order = []

    def cap():
        order.append("cap")
        panel.memory_cap(1000.0, None, lambda _fraction: order.append("set"))

    with pytest.raises(SystemExit):
        panel.prepare(lambda _line: order.append("load"), cap=cap)
    assert order == ["cap"]


def test_id_and_seed_are_the_same_across_sittings_and_the_order_is_kept():
    items = [{"id": "gsm8k-test-9", "source": "gsm8k", "gold": "1", "prompt": "file order first"},
             {"id": "gsm8k-test-1", "source": "gsm8k", "gold": "2", "prompt": "file order second"}]
    first, second = panel.asks_for(items), panel.asks_for(list(items))
    assert first == second
    assert [ask["item_id"] for ask in first] == ["gsm8k-test-9", "gsm8k-test-1"]
    assert [ask["key"] for ask in first] == [ask["item_id"] for ask in first]
    assert [ask["sample"] for ask in first] == [0, 0]
    seeds = [panel.seed_of(ask["item_id"]) for ask in first]
    assert seeds == [batched.seed_for(item["id"], 0) for item in items]
    assert seeds == [panel.seed_of(ask["item_id"]) for ask in second]


def test_scored_records_stay_in_file_order():
    items = [{"id": "gsm8k-test-2", "source": "gsm8k", "gold": "2", "prompt": "b"},
             {"id": "gsm8k-test-0", "source": "gsm8k", "gold": "0", "prompt": "a"}]

    def generate_many(asks):
        assert [ask["item_id"] for ask in asks] == ["gsm8k-test-2", "gsm8k-test-0"]
        return {ask["item_id"]: _reply(f"ANSWER: {ask['item_id'].rsplit('-', 1)[-1]}") for ask in asks}

    records, _minutes = panel.run_generation(items, generate_many)
    assert [row["id"] for row in records] == ["gsm8k-test-2", "gsm8k-test-0"]
    assert [row["right"] for row in records] == [True, True]
    assert [row["seed"] for row in records] == [batched.seed_for(item["id"], 0) for item in items]


def test_mcnemar_known_tables():
    assert round(panel_guard.mcnemar_p(30, 15), 3) == 0.036
    assert round(panel_guard.mcnemar_p(29, 16), 3) == 0.072
    assert panel_guard.holds(30, 15, 300) is True
    assert panel_guard.holds(29, 16, 300) is False


def test_exact_two_point_drop_does_not_hold():
    # n = 250 makes 2.0 points exactly 5 items. The injected p isolates the integer test.
    assert panel_guard.holds(5, 0, 250, p=0.01) is False
    assert panel_guard.holds(6, 0, 250, p=0.01) is True
    assert panel_guard.drop_exceeds(5, 0, 250) is False
    assert (5 - 0) / 250 * 100 == 2.0


def _seed(flags, adapter_sha: str) -> dict:
    return _rights(flags, model="adapter", adapter_sha=adapter_sha)


def test_any_seed_hold_names_one_seed_or_shared():
    base = _rights([True] * 250)
    one = _seed([False] * 6 + [True] * 244, "seed-a")
    one_b = _seed([False] * 6 + [True] * 244, "seed-b")
    quiet = _seed([True] * 250, "seed-c")
    quiet_d = _seed([True] * 250, "seed-d")
    single = panel_guard.round_result(base, [one, quiet, quiet_d])
    assert single["where"] == "one seed" and single["holding_seeds"] == [0]
    assert single["tables"][0]["holds"] and not single["tables"][1]["holds"]
    shared = panel_guard.round_result(base, [one, one_b, quiet])
    assert shared["where"] == "shared" and shared["same_flipped_items"] is True
    assert shared["mean_drop"] == sum(table["drop"] for table in shared["tables"]) / 3
    other = _seed([True] * 6 + [False] * 6 + [True] * 238, "seed-e")
    split = panel_guard.round_result(base, [one, other, quiet_d])
    assert split["where"] == "shared" and split["same_flipped_items"] is False
    assert split["flipped_in_more_than_one_seed"] == []


def test_sitting_view_keeps_the_model_and_the_adapter_sha():
    view = panel.sitting_view({"item_sha256": "i", "scorer_sha256": "s", "model": "adapter",
                                "adapter_sha256": "abc", "items": [{"id": "i0", "right": True}]})
    assert view["model"] == "adapter" and view["adapter_sha256"] == "abc"
    assert view["items"] == [{"id": "i0", "right": True}]


def _three(*seeds):
    return list(seeds)


def test_round_refuses_a_p0_that_is_not_base():
    p0 = _rights([True, False], model="adapter", adapter_sha="seed-a")
    seeds = _three(_seed([True, False], "seed-b"), _seed([True, False], "seed-c"),
                   _seed([False, True], "seed-d"))
    with pytest.raises(SystemExit, match="model base"):
        panel_guard.round_result(p0, seeds)


def test_round_refuses_a_seed_that_is_not_an_adapter():
    seeds = _three(_rights([True, False]), _seed([True, False], "seed-b"),
                   _seed([False, True], "seed-c"))
    with pytest.raises(SystemExit, match="not an adapter"):
        panel_guard.round_result(_rights([True, False]), seeds)


def test_round_refuses_two_seeds_with_the_same_adapter_sha():
    seeds = _three(_seed([True, False], "same"), _seed([False, True], "same"),
                   _seed([True, False], "other"))
    with pytest.raises(SystemExit, match="share an adapter"):
        panel_guard.round_result(_rights([True, False]), seeds)


def test_round_refuses_any_count_but_three_seeds():
    base = _rights([True, False])
    two = [_seed([True, False], "a"), _seed([False, True], "b")]
    with pytest.raises(SystemExit, match="exactly 3"):
        panel_guard.round_result(base, two)
    four = [*two, _seed([True, False], "c"), _seed([True, False], "d")]
    with pytest.raises(SystemExit, match="exactly 3"):
        panel_guard.round_result(base, four)


def test_round_refuses_a_missing_or_empty_adapter_sha():
    base = _rights([True, False])
    rest = [_seed([True, False], "b"), _seed([False, True], "c")]
    missing = _rights([True, False], model="adapter", adapter_sha=None)
    with pytest.raises(SystemExit, match="no adapter"):
        panel_guard.round_result(base, [missing, *rest])
    empty = _rights([True, False], model="adapter", adapter_sha="")
    with pytest.raises(SystemExit, match="no adapter"):
        panel_guard.round_result(base, [empty, *rest])


def test_pairing_refuses_mismatched_item_file_scorer_or_ids():
    good = _rights([True, False])
    with pytest.raises(SystemExit, match="item file"):
        panel_guard.pair(good, _rights([True, False], sha="other"))
    with pytest.raises(SystemExit, match="scorer"):
        panel_guard.pair(good, _rights([True, False], scorer="other"))
    swapped = _rights([False, True])
    swapped["items"] = list(reversed(swapped["items"]))
    with pytest.raises(SystemExit, match="item ids"):
        panel_guard.pair(good, swapped)
    paired = panel_guard.pair(good, _rights([False, True]))
    assert paired["b"] == 1 and paired["c"] == 1 and paired["n"] == 2


def test_slice_at_or_above_97_is_exact():
    assert panel_guard.slice_at_or_above_97(97, 100) is True
    assert panel_guard.slice_at_or_above_97(96, 100) is False
    blank = {"truncated": False, "unparsed": False, "empty": False, "strict_one_line": False}
    records = [{**blank, "slice": "gsm8k", "right": True}] * 97
    records += [{**blank, "slice": "gsm8k", "right": False}] * 3
    records += [{**blank, "slice": "winogrande", "right": False}]
    readout = panel.readout_for(records, item_sha="i", scorer_sha="s", model="base", adapter_sha=None,
                                 machine_line={}, minutes=20.0, batched_sha="b")
    assert readout["slices"]["gsm8k"]["at_or_above_97"] is True
    assert readout["slices"]["winogrande"]["at_or_above_97"] is False
    assert readout["over_20_minutes"] is False
    assert readout["accuracy"]["n"] == 101
    longer = panel.readout_for(records, item_sha="i", scorer_sha="s", model="base", adapter_sha=None,
                                machine_line={}, minutes=20.01, batched_sha="b")
    assert longer["over_20_minutes"] is True and len(longer["items"]) == len(records)


def test_partition_counts_sum_to_n_and_a_capped_unparsed_row_is_truncated():
    blank = {"empty": False, "strict_one_line": False, "slice": "gsm8k"}
    records = [
        {**blank, "right": False, "truncated": True, "unparsed": True},
        {**blank, "right": False, "truncated": True, "unparsed": False},
        {**blank, "right": False, "truncated": False, "unparsed": True},
        {**blank, "right": False, "truncated": False, "unparsed": False},
        {**blank, "right": True, "truncated": False, "unparsed": False},
        {**blank, "slice": "winogrande", "right": True, "truncated": False, "unparsed": False},
    ]
    readout = panel.readout_for(records, item_sha="i", scorer_sha="s", model="base", adapter_sha=None,
                                 machine_line={}, minutes=1.0, batched_sha="b")
    part = readout["partition"]
    assert part == {"truncated": 2, "unparsed": 1, "wrong": 1, "right": 2}
    assert sum(part.values()) == len(records) == readout["accuracy"]["n"]
    assert readout["truncated"] == 2 and readout["unparsed"] == 2


def test_sit_main_loads_nothing_after_a_failed_machine_line(monkeypatch, tmp_path):
    seen = []
    real_prepare = panel.prepare

    def wrapped(load, **kwargs):
        seen.append("prepare")
        return real_prepare(load, **kwargs)

    monkeypatch.setattr(panel, "prepare", wrapped)
    monkeypatch.setattr(panel_build, "outside_repo", lambda _path: None)
    def fail_machine(path=None):
        raise SystemExit(5)

    monkeypatch.setattr(panel, "check_machine", fail_machine)
    monkeypatch.setattr(panel, "load_sitting", lambda *args, **kwargs: seen.append("load"))
    monkeypatch.setattr(panel, "apply_memory_cap", lambda: seen.append("cap"))
    with pytest.raises(SystemExit) as caught:
        panel.sit_main(["--items", "items.jsonl", "--out", str(tmp_path / "out.json")])
    assert caught.value.code == 5
    assert seen == ["prepare"]


def test_changed_or_missing_scorer_sha_is_refused(monkeypatch, tmp_path):
    monkeypatch.setattr(panel, "PANEL_SCORE_SHA256", "0" * 64)
    with pytest.raises(SystemExit, match="does not match"):
        panel.require_scorer()
    monkeypatch.setattr(panel, "PANEL_SCORE_SHA256", panel_score_sha())
    monkeypatch.setattr(panel_score, "__file__", str(tmp_path / "missing-scorer.py"))
    with pytest.raises(SystemExit, match="missing"):
        panel.require_scorer()


def panel_score_sha() -> str:
    return hashlib.sha256(Path(panel_score.__file__).read_bytes()).hexdigest()


def test_unchanged_scorer_is_accepted():
    assert panel.require_scorer() == panel.PANEL_SCORE_SHA256
    assert panel.require_scorer() == panel_score_sha()


def _kept_file(path: Path, n_gsm: int = 140, n_wino: int = 140) -> Path:
    lines = ['{"record": "meta"}']
    for i in range(n_gsm):
        lines.append(json.dumps({"decision": "kept", "source": "gsm8k", "id": f"gsm8k-test-{i}"}))
    for i in range(n_wino):
        lines.append(json.dumps({"decision": "kept", "source": "winogrande", "id": f"w-{i}"}))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


def _frozen_machine(monkeypatch):
    frozen = json.loads(panel.MACHINE_LINE.read_text(encoding="utf-8"))
    monkeypatch.setattr(det_run, "machine_line", lambda: frozen)


def test_item_pin_is_required_before_the_cap_and_on_the_guard(monkeypatch, tmp_path):
    seen = []
    _frozen_machine(monkeypatch)
    monkeypatch.setattr(panel, "PANEL_ITEMS_SHA256", None)
    items = _kept_file(tmp_path / "items.jsonl")
    with pytest.raises(SystemExit, match="pin is not set"):
        panel.prepare(lambda _line: seen.append("load"), cap=lambda: seen.append("cap"),
                      items_path=items, adapter=None)
    assert seen == []
    monkeypatch.setattr(panel, "PANEL_ITEMS_SHA256", "0" * 64)
    with pytest.raises(SystemExit, match="does not match"):
        panel.require_items(items)
    short = _kept_file(tmp_path / "short.jsonl", n_gsm=1, n_wino=1)
    monkeypatch.setattr(panel, "PANEL_ITEMS_SHA256", panel.sha256_file(short))
    with pytest.raises(SystemExit, match="140"):
        panel.require_items(short)
    monkeypatch.setattr(panel, "PANEL_ITEMS_SHA256", "abc")
    baseline = {"scorer_sha256": panel.PANEL_SCORE_SHA256, "item_sha256": "nope",
                "model": "base", "items": []}
    with pytest.raises(SystemExit, match="item sha256 is not the pin"):
        panel.guard_sittings(baseline, [])


def test_guard_refuses_a_baseline_scorer_sha_that_is_not_the_pin(monkeypatch):
    called = []
    monkeypatch.setattr(panel_guard, "round_result", lambda *_args: called.append("pair"))
    baseline = {"scorer_sha256": "0" * 64, "item_sha256": "item", "model": "base", "items": []}
    with pytest.raises(SystemExit, match="baseline scorer"):
        panel.guard_sittings(baseline, [])
    assert called == []


def test_adapter_and_the_reset_clock_are_checked_before_the_model(monkeypatch, tmp_path):
    seen = []
    real_load = panel.load_sitting
    _frozen_machine(monkeypatch)
    monkeypatch.setattr(panel_build, "outside_repo", lambda _path: None)
    monkeypatch.setattr(panel, "apply_memory_cap", lambda: seen.append("cap"))
    monkeypatch.setattr(panel, "load_sitting", lambda *args, **kwargs: seen.append("load"))
    adapter = tmp_path / "nest" / "adapter"
    adapter.mkdir(parents=True)
    items = tmp_path / "items.jsonl"
    items.write_text('{"record": "meta"}\n', encoding="utf-8", newline="\n")
    out = tmp_path / "out.json"
    with pytest.raises(SystemExit, match="manifest"):
        panel.sit_main(["--items", str(items), "--out", str(out), "--adapter", str(adapter)])
    assert seen == []

    clock = {"n": 0}

    def now():
        clock["n"] += 1
        return float(clock["n"])

    monkeypatch.setattr(panel.time, "time", now)
    real_prepare = panel.prepare

    def wrapped(load, **kwargs):
        clock["at_prepare"] = panel.time.time()
        return real_prepare(load, **kwargs)

    monkeypatch.setattr(panel, "prepare", wrapped)
    monkeypatch.setattr(panel, "load_sitting", real_load)
    monkeypatch.setattr(panel, "require_items", lambda path: "pinned")
    captured = {}

    def resets(t0):
        captured["t0"] = t0
        return 0

    monkeypatch.setattr(parity, "driver_resets_since", resets)
    import student
    monkeypatch.setattr(student, "verify_weights", lambda *args, **kwargs: None)
    monkeypatch.setattr(student, "load_tokenizer", lambda *args, **kwargs: object())

    class _Model:
        def eval(self):
            return self

    monkeypatch.setattr(student, "load_student", lambda *args, **kwargs: _Model())
    monkeypatch.setattr(batched, "make_generate_many", lambda *args, **kwargs: (lambda asks: {}))
    monkeypatch.setattr(panel, "run_generation", lambda items, generate_many: ([], 0.0))
    monkeypatch.setattr(panel, "load_items", lambda path: ({"record": "meta"}, []))
    panel.sit_main(["--items", str(items), "--out", str(out)])
    assert captured["t0"] < clock["at_prepare"]


def test_guard_refuses_a_changed_scorer_before_pairing(monkeypatch):
    monkeypatch.setattr(panel, "PANEL_SCORE_SHA256", "0" * 64)
    called = []
    monkeypatch.setattr(panel_guard, "round_result", lambda *_args: called.append("pair"))
    with pytest.raises(SystemExit, match="does not match"):
        panel.guard_sittings({}, [])
    assert called == []


def test_source_sha_mismatch_is_refused(tmp_path):
    gsm, wino, _gsm_sha, _wino_sha = _sources(
        tmp_path, [_gsm("What is two plus two?", "#### 4")], [_wino(QID, "The _ sat.")])
    with pytest.raises(SystemExit, match="gsm8k sha256"):
        panel_build.build(gsm, wino, tmp_path / "items.jsonl")
    with pytest.raises(SystemExit, match="winogrande sha256"):
        panel_build.build(gsm, wino, tmp_path / "items.jsonl", gsm_sha=panel_build.sha256_bytes(gsm))


def test_hash_order_is_stable_and_not_file_order():
    rows = [_row("b", "gsm8k-test-0", "1", "beta"), _row("a", "gsm8k-test-1", "2", "alpha")]
    first = panel_build.walk(rows, 2, {}, None)
    second = panel_build.walk(rows, 2, {}, None)
    assert [row["id"] for row in first] == ["gsm8k-test-1", "gsm8k-test-0"]
    assert first == second


def test_build_file_sha_is_stable(tmp_path):
    gsm, wino, gsm_sha, wino_sha = _sources(
        tmp_path, [_gsm("What is two plus two?", "#### 4")], [_wino(QID, "The _ sat.")])
    out_a, out_b = tmp_path / "a" / "items.jsonl", tmp_path / "b" / "items.jsonl"
    sha_a = panel_build.build(gsm, wino, out_a, per_source=1, gsm_sha=gsm_sha, wino_sha=wino_sha)
    sha_b = panel_build.build(gsm, wino, out_b, per_source=1, gsm_sha=gsm_sha, wino_sha=wino_sha)
    assert sha_a == sha_b
    assert sha_a == panel_build.sha256_bytes(out_a)
    text = out_a.read_bytes()
    assert b"\r\n" not in text
    meta = json.loads(text.splitlines()[0])
    assert meta["leakage"] == "not run" and meta["kept"] == {"gsm8k": 1, "winogrande": 1}
    assert (out_a.parent / "GSM8K-MIT-LICENSE.txt").read_text(encoding="utf-8").startswith("MIT License")
    notice = (out_a.parent / "WINOGRANDE-ATTRIBUTION.txt").read_text(encoding="utf-8")
    assert "CC-BY" in notice and "reformatted into prompts; options relabelled A and B" in notice


def test_out_inside_the_repository_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="repository"):
        panel_build.build(tmp_path / "missing-gsm.jsonl", tmp_path / "missing-wino.jsonl",
                          receipts.REPO / "panel-items.jsonl")
    with pytest.raises(SystemExit, match="repository"):
        panel_build.fetch_into(receipts.REPO / "panel-sources")


def test_dedupe_drops_the_later_item_and_keeps_the_instruction_out_of_it():
    shared = WORDS
    rows = [_row("a", "gsm8k-test-0", "1", shared),
            _row("b", "gsm8k-test-1", "2", shared + " extra"),
            _row("c", "gsm8k-test-2", "3", "unique")]
    walked = panel_build.walk(rows, 2, {}, None)
    assert [row["decision"] for row in walked] == ["kept", "skipped", "kept"]
    assert walked[1]["reason"] == "dedupe:gsm8k-test-0"
    prompts = [_row("a", "gsm8k-test-0", "1", "one"), _row("b", "gsm8k-test-1", "2", "two")]
    for row, question in zip(prompts, ("one", "two")):
        row["prompt"] = panel_build.GSM_TEMPLATE + question
    assert [row["decision"] for row in panel_build.walk(prompts, 2, {}, None)] == ["kept", "kept"]


def test_bad_gold_is_skipped_and_a_bad_id_is_refused():
    rows = [_row("a", "gsm8k-test-0", None, "no gold here at all"),
            _row("b", "gsm8k-test-1", "4", "a later valid item")]
    walked = panel_build.walk(rows, 1, {}, None)
    assert walked[0]["reason"] == "gold" and walked[1]["decision"] == "kept"
    bad = panel_build._kept(_row("a", "not-an-id", "1", "short"))
    with pytest.raises(SystemExit, match="item id"):
        panel_build.check_kept([bad])


def test_source_pins_are_the_files_at_the_named_commits():
    assert panel_build.GSM8K_SHA256 == "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"
    assert panel_build.WINOGRANDE_SHA256 == "1aeac79cf46a3dcbe59a71ca7e5372a1490cde2b4ae2387dc56f00378104d2e8"
    assert panel_build.GSM8K_COMMIT == "3101c7d5072418e28b9008a6636bde82a006892c"
    assert panel_build.WINOGRANDE_COMMIT == "727e837f77521ef38bcc56df3b275c8da43f45af"


def test_statement_claim_is_not_skipped_and_a_bare_row_stops():
    row = {"id": "T-B", "kind": "B", "statement": "the depot gate is shut today",
           "turns": [{"user": "there is no statement line", "thinking": "", "reply": ""}]}
    corpus = panel_leak.corpus_from_parts([row], [], [], [])
    item = {"id": "gsm8k-test-0", "prompt": "the depot gate is shut today", "source_text": "short question"}
    assert panel_leak.reject_reason(item, corpus) == "leakage"
    assert panel_leak.claim_of({"turns": [{"user": "recall the lyn jar"}]}) == "recall the lyn jar"
    assert panel_leak.claim_of({"turns": [{"user": "Statement: the card is red"}]}) == "the card is red"
    with pytest.raises(panel_leak.LeakError, match="no turns"):
        panel_leak.claim_of({"id": "empty"})


def test_shingles_cover_material_variant_and_twin_and_not_spans(monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("the panel called the wrong tokenizer")

    monkeypatch.setattr(panel_leak.gates, "_spans", boom)
    monkeypatch.setattr(panel_leak.gates, "line_spans", boom)
    item = {"id": "gsm8k-test-0", "source_text": WORDS, "prompt": "Go.\n\n" + WORDS}
    for field, value in (("material", WORDS), ("variant", {"user": WORDS}),
                         ("r10_twin", {"statement": WORDS})):
        row = {"id": "T", "turns": [{"user": "no overlap here", "thinking": "", "reply": "ok"}],
               "statement": "a different claim", field: value}
        corpus = panel_leak.corpus_from_parts([row], [], [], [])
        assert panel_leak.reject_reason(item, corpus) == "leakage", field


def test_an_instruction_hit_does_not_drop_the_source():
    row = {"id": "T", "statement": "an unrelated claim about a jar",
           "turns": [{"user": "short", "thinking": "",
                      "reply": "solve the problem think it through then end"}]}
    corpus = panel_leak.corpus_from_parts([row], [], [], [])
    item = {"id": "gsm8k-test-0", "source_text": "a short question",
            "prompt": panel_build.GSM_TEMPLATE + "a short question"}
    with pytest.raises(panel_leak.LeakError, match="instruction"):
        panel_leak.reject_reason(item, corpus)


def test_sealed_directory_is_refused_and_a_planted_file_is_checked(tmp_path):
    directory = tmp_path / "taskset-sealed.jsonl"
    directory.mkdir()
    with pytest.raises(panel_leak.LeakError, match="not a file"):
        panel_leak.load_jsonl_file(directory, receipts.SEALED_SHA256)
    planted = tmp_path / "sealed.jsonl"
    planted.write_text('{"id": "sx-1", "claim": "one", "material": "two"}\n', encoding="utf-8", newline="\n")
    with pytest.raises(panel_leak.LeakError, match="sha256"):
        panel_leak.load_jsonl_file(planted, receipts.SEALED_SHA256)
    loaded = panel_leak.load_jsonl_file(planted, receipts.sha256_file(planted))
    assert loaded[0]["id"] == "sx-1" and planted.is_file()
    assert receipts.SEALED_SHA256.startswith("e1e515b4")
    assert receipts.SEALED_SHA256 == eval_adapter.SEALED_SHA256


def test_frozen_table_parses_and_a_bad_sha_is_refused(tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    data = folder / "a.jsonl"
    data.write_text('{"id": "t", "statement": "claim", "turns": [{"user": "claim", "reply": ""}]}\n',
                    encoding="utf-8", newline="\n")
    digest = receipts.sha256_file(data)
    text = (
        "# Stage 1 frozen data\n\n"
        "Each file's sha256 is the sha256 of the content with CRLF read as LF (`receipts.sha256_file`).\n\n"
        "## Training files\n\n"
        "| # | file | items | sha256 |\n"
        "|---:|---|---:|---|\n"
        f"| 1 | docs/a.jsonl | 1 | {digest} |\n\n"
        "## Apply files\n\n"
        "| # | file | items | sha256 |\n"
        "|---:|---|---:|---|\n\n"
        "## NO-ROLE\n\n"
        "| file | items | sha256 |\n"
        "|---|---:|---|\n"
        f"| docs/a.jsonl | 1 | {digest} |\n\n"
        "## Items per lesson and kind\n\n"
        "| lesson | A | B | C | D | E |\n"
        "|---|---:|---:|---:|---:|---:|\n"
        "| t01 | 1 | 2 | 3 | 4 | 5 |\n\n"
        "## Gates\n\n"
        "| check | result |\n"
        "|---|---|\n"
        "| leakage | pass |\n"
        f"| docs/decoy.jsonl | 1 | {digest} |\n\n"
        "## Sources\n\n"
        "invented items: 0\n")
    parsed = panel_leak.parse_frozen(text)
    assert parsed["apply"] == [] and len(parsed["training"]) == 1
    assert len(parsed["no_role"]) == 1 and parsed["no_role"][0]["file"] == "docs/a.jsonl"
    assert panel_leak.load_listed(tmp_path, parsed["training"])[0]["id"] == "t"
    parsed["training"][0]["sha256"] = "0" * 64
    with pytest.raises(panel_leak.LeakError, match="sha256"):
        panel_leak.load_listed(tmp_path, parsed["training"])
    with pytest.raises(panel_leak.LeakError, match="repo-relative"):
        panel_leak.parse_frozen(text.replace("docs/a.jsonl", "../secret.jsonl"))


def test_missing_or_unparsed_frozen_is_not_a_clean_pass(tmp_path):
    items = tmp_path / "items.jsonl"
    items.write_text('{"record": "meta"}\n', encoding="utf-8", newline="\n")
    missing = tmp_path / "no-frozen.md"
    assert panel_leak.main(["--items", str(items), "--frozen", str(missing)]) == 2
    bad = tmp_path / "FROZEN.md"
    bad.write_text("this is not a freeze file\n", encoding="utf-8", newline="\n")
    assert panel_leak.main(["--items", str(items), "--frozen", str(bad)]) == 2


def test_driver_reset_writes_nothing(tmp_path):
    dest = tmp_path / "readout.json"
    with pytest.raises(SystemExit, match="driver reset"):
        panel.write_readout({"items": []}, dest, resets=1)
    assert not dest.exists()
    panel.write_readout({"where": "none"}, dest, resets=0)
    assert dest.is_file()


def test_adapter_without_a_manifest_or_a_partial_set_is_refused(tmp_path):
    root = tmp_path / "round"
    adapter = root / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    with pytest.raises(SystemExit, match="manifest"):
        panel.adapter_record(str(adapter))
    (root / "manifest.json").write_text('{"freeze_report": {"partial_set": true}}\n', encoding="utf-8",
                                        newline="\n")
    with pytest.raises(SystemExit, match="partial-set"):
        panel.adapter_record(str(adapter))
    (root / "manifest.json").write_text('{"freeze_report": {}}\n', encoding="utf-8", newline="\n")
    record = panel.adapter_record(str(adapter))
    assert record == {"model": "adapter",
                      "adapter_sha256": hashlib.sha256(b"weights").hexdigest()}
    assert panel.adapter_record(None) == {"model": "base", "adapter_sha256": None}


def test_readout_inside_the_repository_is_refused():
    with pytest.raises(SystemExit, match="repository"):
        panel.write_readout({"where": "none"}, receipts.REPO / "panel-readout.json", resets=0)


def test_dev_files_match_their_pins():
    rows = panel_leak.load_dev()
    assert rows and all(row.get("id") and "claim" in row for row in rows)
