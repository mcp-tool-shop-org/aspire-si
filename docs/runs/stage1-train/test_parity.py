"""B4's parity rules, on synthetic records: the set is fixed by rule, (i) holds Argon's paired tolerances, and
(ii) holds #118's relative rules against the batch-1 noise."""

import json
from pathlib import Path

import data
import parity

DATA = Path(__file__).resolve().parent.parent / "stage1-training-data"


def rec(right=True, stuck=False, tt=300, tier=4, lp=-0.5, truncated=False, clause=None):
    r = {"right": right, "stuck": stuck, "thinking_tokens": tt, "truncated": truncated, "tier": tier,
         "first_logprob": lp, **{c: False for c in parity.CLAUSES}}
    if clause:
        r[clause] = True
    return r


def draw(n_items=20, stuck_items=(), tt=300, wrong_items=(), lp=-0.5):
    return {(f"i{i}", s): rec(right=i not in wrong_items, stuck=i in stuck_items, tt=tt + i, lp=lp,
                              clause="long_not_progressing" if i in stuck_items else None)
            for i in range(n_items) for s in range(3)}


def test_the_parity_set_is_two_unspent_verdict_variants_per_lesson():
    items = [r for p in sorted(DATA.glob("*.jsonl")) if not p.name.endswith(".models.jsonl") for r in data.load(p)]
    chosen = parity.select(items)
    lessons = {r["lesson"] for r in chosen}
    assert len(chosen) == 2 * len(lessons) and all(r["kind"] == "E" and r["quiz"]["type"] == "verdict" for r in chosen)
    assert parity.select(list(reversed(items))) == chosen  # fixed by rule, not by file order


def test_i_passes_when_batching_changes_nothing_and_fails_on_a_stuck_shift():
    a = draw(stuck_items=(1, 2))
    assert parity.compare_i(a, draw(stuck_items=(1, 2)))["pass"]
    shifted = parity.compare_i(a, draw(stuck_items=(1, 2, 3, 4, 5)))
    assert not shifted["checks"]["stuck_rate"] and not shifted["pass"]


def test_i_holds_the_thinking_length_tolerances_and_logs_the_logprob_deviation():
    out = parity.compare_i(draw(tt=300), draw(tt=330, lp=-0.6))
    assert not out["checks"]["median_thinking"]  # +10% is over the 5% median tolerance
    assert abs(out["figures"]["first_logprob_dev_max"] - 0.1) < 1e-9 and "first_logprob" not in out["checks"]


def test_ii_judges_the_batched_loop_against_the_batch_1_noise():
    p1, p1p = draw(wrong_items=(0, 1)), draw(wrong_items=(0, 2))  # the two harness draws disagree on 2 of 20
    assert parity.compare_ii(draw(wrong_items=(0, 3)), p1, p1p)["pass"]  # the same disagreement: within the noise
    far = parity.compare_ii(draw(wrong_items=(0, 3, 4, 5, 6, 7)), p1, p1p)  # 6 of 20 differ from P1
    assert not far["checks"]["majority_agreement"]


def test_ii_holds_the_clause_and_truncation_rules():
    p1, p1p = draw(), draw()
    many_stuck = draw(stuck_items=tuple(range(5)))
    out = parity.compare_ii(many_stuck, p1, p1p)
    assert not out["checks"]["clauses"] and not out["checks"]["stuck_rate"]
    trunc = {k: {**r, "truncated": True} if k[0] in ("i0", "i1") else r for k, r in draw().items()}
    assert not parity.compare_ii(trunc, p1, p1p)["checks"]["truncations"]


def test_kappa_is_one_for_identical_and_low_for_unrelated_labels():
    assert parity.kappa([True, False, True], [True, False, True]) == 1.0
    assert parity.kappa([True, False] * 10, [False, True] * 10) < 0


def test_warned_ops_are_the_deterministic_warnings_only():
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.warn("index_copy_cuda does not have a deterministic implementation, but you set warn_only")
        warnings.warn("index_copy_cuda does not have a deterministic implementation, but you set warn_only")
        warnings.warn("an unrelated deprecation")
    assert parity.warned_ops(caught) == [
        "index_copy_cuda does not have a deterministic implementation, but you set warn_only"]


def test_a_finished_draw_is_saved_and_kept_and_an_incomplete_one_is_redone(tmp_path):
    rows = {("T05-E-01-v0", 0): rec(), ("T05-E-01-v0", 1): rec(right=False)}
    path = tmp_path / "draw-LB8.jsonl"
    parity.save_draw(path, rows)
    assert parity.load_draw(path, set(rows)) == rows
    assert parity.load_draw(path, set(rows) | {("T05-E-02-v0", 0)}) is None  # incomplete: run it again
    assert parity.load_draw(tmp_path / "draw-LB1.jsonl", set(rows)) is None


def test_progress_prints_a_line_per_generation_with_an_eta():
    lines = []
    p = parity.Progress("LB1", 4, out=lambda s, flush=True: lines.append(s))
    p.step("T05-E-01-v0|0", 300, 15.0)
    p.step("T05-E-01-v0|1", 16384, 800.0)
    assert len(lines) == 2 and lines[1].startswith("LB1 2/4 T05-E-01-v0|1 tokens=16384") and "eta=" in lines[1]
    assert p.tokens == 16684


def test_sizing_measures_the_cap_share_and_projects_batch_1_minutes():
    lb8 = {(f"i{i}", s): {**rec(), "new_tokens": 16384 if i == 0 else 300} for i in range(4) for s in range(3)}
    size = parity.sizing(lb8, 16384, tok_s=20.0, per_draw=12)
    assert size["at_cap"] == 3 and size["at_cap_share"] == 0.25
    assert size["minutes_per_batch1_draw"] == round((3 * 16384 + 9 * 300) / 20 / 60, 1)


def test_a_draw_that_cannot_finish_by_the_deadline_is_not_started():
    import time
    assert parity.fits(10, None) and parity.fits(10, time.time() + 3600)
    assert not parity.fits(120, time.time() + 3600)


def test_a_rerun_with_lb8_and_lb1_kept_still_sizes_and_refuses_a_draw_past_the_deadline(tmp_path):
    """R&D on #130: a kept draw comes back with its meta, so sizing exists on a rerun and --deadline can refuse."""
    import time
    keys = {(f"i{i}", s) for i in range(4) for s in range(3)}
    lb8 = {k: {**rec(), "new_tokens": 16384 if k[0] == "i0" else 300} for k in keys}
    lb1 = {k: rec() for k in keys}
    for name, rows, meta in (("LB8", lb8, {"seconds": 60.0, "tokens": 51852, "warned_ops": None}),
                             ("LB1", lb1, {"seconds": 2400.0, "tokens": 48000, "warned_ops": None})):
        (tmp_path / f"draw-{name}.meta.json").write_text(json.dumps(meta), encoding="utf-8")
        parity.save_draw(tmp_path / f"draw-{name}.jsonl", rows)
    ran, said = [], []

    def draw_fn(name):
        ran.append(name)
        return {k: rec() for k in keys}, {"seconds": 1.0, "tokens": 10, "warned_ops": []}
    draws, meta, size = parity.run_draws(tmp_path, keys, 12, 16384, draw_fn, deadline=time.time() + 60,
                                         say=lambda s, flush=True: said.append(s))
    assert ran == [] and set(draws) == {"LB8", "LB1"}  # both kept; P1 refused before it started
    assert size is not None and size["tok_s_batch1"] == 20.0 and size["at_cap"] == 3  # LB1's measured 48000/2400
    assert any(s.startswith("P1: not started") for s in said)
    assert meta["LB1"]["seconds"] == 2400.0
    draws, _, _ = parity.run_draws(tmp_path, keys, 12, 16384, draw_fn, deadline=None,
                                   say=lambda s, flush=True: None)
    assert ran == ["P1", "P1p"] and set(draws) == set(parity.DRAW_ORDER)  # no deadline: the rest run, LB8/LB1 kept


def test_a_draw_file_without_its_meta_is_run_again(tmp_path):
    keys = {("i0", 0)}
    parity.save_draw(tmp_path / "draw-LB8.jsonl", {("i0", 0): {**rec(), "new_tokens": 10}})
    ran = []

    def draw_fn(name):
        ran.append(name)
        return {("i0", 0): {**rec(), "new_tokens": 10}}, {"seconds": 1.0, "tokens": 10, "warned_ops": None}
    parity.run_draws(tmp_path, keys, 1, 16384, draw_fn, say=lambda s, flush=True: None)
    assert ran[0] == "LB8" and (tmp_path / "draw-LB8.meta.json").exists()


def test_writes_are_atomic_and_leave_no_temporary_file(tmp_path):
    path = tmp_path / "draw-LB8.jsonl"
    parity.save_draw(path, {("i0", 0): rec()})
    parity.write_atomic(tmp_path / "draw-LB8.meta.json", json.dumps({"seconds": 1.0}))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["draw-LB8.jsonl", "draw-LB8.meta.json"]
    assert all(line.endswith("}") for line in path.read_text(encoding="utf-8").splitlines())


def test_a_resumed_run_equals_an_uninterrupted_one(tmp_path):
    """Each row's sample depends only on (item, sample), never on draw order (Argon): a run stopped after LB8 and
    resumed gives the same draws as one that ran straight through."""
    import batched
    keys = {(f"i{i}", s) for i in range(3) for s in range(3)}

    def draw_fn(name):  # deterministic per key, as the per-item generators and the harness's re-seeding are
        return ({k: {**rec(), "new_tokens": batched.seed_for(f"{name}:{k[0]}", k[1]) % 500} for k in keys},
                {"seconds": 1.0, "tokens": 10, "warned_ops": None})
    quiet = {"say": lambda s, flush=True: None}
    (tmp_path / "whole").mkdir()
    (tmp_path / "resumed").mkdir()
    whole, _, _ = parity.run_draws(tmp_path / "whole", keys, 9, 16384, draw_fn, **quiet)
    stop_after_lb8 = iter([None, "a stop"])
    parity.run_draws(tmp_path / "resumed", keys, 9, 16384, draw_fn, check_fn=lambda: next(stop_after_lb8), **quiet)
    assert sorted(p.name for p in (tmp_path / "resumed").glob("draw-*.jsonl")) == ["draw-LB8.jsonl"]
    resumed, _, _ = parity.run_draws(tmp_path / "resumed", keys, 9, 16384, draw_fn, **quiet)
    assert resumed == whole


def test_a_driver_reset_or_a_changed_machine_line_stops_the_run_before_the_next_draw():
    line = {"driver": "617.42"}
    assert parity.make_check(line, 0.0, lambda: dict(line), resets_since=lambda t: 0)() is None
    assert "driver reset" in parity.make_check(line, 0.0, lambda: dict(line), resets_since=lambda t: 1)()
    assert "machine line changed" in parity.make_check(line, 0.0, lambda: {"driver": "618.0"},
                                                       resets_since=lambda t: 0)()


def test_the_worst_case_batch_fills_the_budget_exactly():
    import batched
    n = parity.worst_case_lengths(batched.B_TOK, batched.BATCH, batched.CHUNK_MAX)
    assert batched.BATCH * (n + batched.CHUNK_MAX) == batched.B_TOK - batched.B_TOK % batched.BATCH
    plan = batched.plan([{"key": f"k{i}", "ids": [0] * n, "new": []} for i in range(batched.BATCH)])
    assert plan == [([f"k{i}" for i in range(batched.BATCH)], batched.CHUNK_MAX, "first")]  # one full batch


def test_the_cap_counts_the_context_and_the_baseline_in_the_plans_26_gb():
    """R&D on #130: the plan caps memory.used (torch, the CUDA context and the idle baseline) at 26 GB, so the
    allocator gets 26e9 minus what was in use once CUDA started."""
    total = 32607 * 2**20  # the 5090's memory, in bytes
    used = 2.6e9  # the context plus the idle baseline at start
    f = parity.cap_fraction(total, used)
    assert abs(f * total + used - 26e9) < 1  # torch's share plus the start figure is the plan, exactly
    assert parity.within_plan(23.4e9, used) and not parity.within_plan(23.5e9, used)
    assert parity.cap_fraction(total, 30e9) == 0.0 and parity.cap_fraction(10e9, 0.0) == 1.0  # clamped
