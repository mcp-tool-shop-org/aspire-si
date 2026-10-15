"""check_against (the generator's one-at-a-time check) and the batch gate refuse the same puzzles (R&D, 2026-10-15):
over the real data, fed one row at a time in file order as a generator would, check_against's PUZZLE_SHAPE_REUSED
and PUZZLE_SHAPE_BATCH refusals are exactly the ones the batch gate's signatures imply, with and without the
sidecar models."""

from pathlib import Path

import pytest

from aspire_si_gates import check_against, gates, load_models
from aspire_si_gates.constants import PRE_AMENDMENT_LESSONS

import data

DATA = Path(__file__).resolve().parent.parent / "stage1-training-data"
FILES = sorted(p for p in DATA.glob("*.jsonl") if not p.name.endswith(".models.jsonl"))
SHAPE = {"PUZZLE_SHAPE_REUSED", "PUZZLE_SHAPE_BATCH"}


def batches() -> dict:
    """{batch prefix: (rows in file order, {id: model})}."""
    out: dict = {}
    for p in FILES:
        rows, models = out.setdefault(p.name.split("-", 1)[0], ([], {}))
        rows += data.load(p)
        models.update(load_models(p))
    return out


def gate_refusals(rows, models) -> dict:
    """What the batch gate's signatures imply, row by row: a puzzle whose signature an earlier puzzle of its lesson
    already has is REUSED; otherwise one with 2 earlier puzzles of that signature in the batch is BATCH."""
    seen: dict = {}
    out = {}
    for r in rows:
        if r.get("lesson") in PRE_AMENDMENT_LESSONS or not gates._is_puzzle(r):
            continue
        sig = gates.puzzle_sig(r, models)
        before = seen.setdefault(sig, [])
        if any(lesson == r["lesson"] for lesson, _ in before):
            out[r["id"]] = ("PUZZLE_SHAPE_REUSED", gates.describe_signature(sig))
        elif len(before) >= 2:
            out[r["id"]] = ("PUZZLE_SHAPE_BATCH", gates.describe_signature(sig))
        before.append((r["lesson"], r["id"]))
    return out


def one_at_a_time(rows, models) -> dict:
    out = {}
    for n, r in enumerate(rows):
        for code, msg in check_against(r, rows[:n], models=models):
            if code in SHAPE:
                out[r["id"]] = (code, msg)
    return out


@pytest.mark.parametrize("mode", ["sidecar models", "regex only", "one shared model"])
def test_check_against_refuses_exactly_what_the_batch_gate_implies(mode):
    """The real data passes both, so a third mode gives every puzzle of a batch one shared model: real rows, now
    refused (REUSED inside a lesson, BATCH across lessons), and the two must still refuse the same ones."""
    found, codes = 0, set()
    for name, (rows, models) in batches().items():
        if mode == "sidecar models":
            m = models
        elif mode == "regex only":
            m = None
        else:
            one = next(iter(models.values()), None)
            m = {r["id"]: one for r in rows if gates._is_puzzle(r)} if one else None
        expected = gate_refusals(rows, m)
        got = one_at_a_time(rows, m)
        assert set(got) == set(expected), name
        for i, (code, msg) in got.items():
            assert code == expected[i][0] and expected[i][1] in msg, (name, i)
        if got:
            with pytest.raises(data.DataError, match="share the structure"):
                gates.check_puzzles(rows, models=m)
        else:
            gates.check_puzzles(rows, models=m)
        found += len(got)
        codes |= {c for c, _ in got.values()}
    if mode == "one shared model":
        assert found > 10 and codes == SHAPE  # refusals of both codes on real rows
    else:
        assert found == 0  # the merged data passes the batch gate either way, so the generator's check passes it
