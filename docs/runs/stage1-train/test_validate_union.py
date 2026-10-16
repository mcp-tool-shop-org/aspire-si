"""The trainer's validation of a set made of several batches (the Dataset Specialist's ruling, 2026-10-11): per
item and per lesson over the union, the batch checks per batch file, freeze_report over the union."""

from pathlib import Path

import pytest

import data

DATA = Path(__file__).resolve().parent.parent / "stage1-training-data"
FILES = {str(DATA / f): data.load(DATA / f) for f in ("batch3-t05.jsonl", "batch4-t02.jsonl", "batch4-t06.jsonl")}
SIG = (5, ("negation",))


@pytest.fixture
def three_lessons_share_a_signature(monkeypatch):
    """One puzzle signature shared by a B item in each of three lessons (each lesson once). The data itself
    changes between fix rounds, so the sharing is planted through puzzle_repeats rather than found in today's
    items."""
    from aspire_si_gates import gates
    picks = [(r["lesson"], r["id"]) for rs in FILES.values() for r in rs[:1]]
    monkeypatch.setattr(gates, "puzzle_repeats", lambda rows, skip=(), models=None: {SIG: [x for x in picks
                                                                              if x[1] in {r["id"] for r in rows}]})
    return picks


def test_batches_that_each_pass_are_accepted_together_when_only_a_batch_cap_would_fail(three_lessons_share_a_signature):
    rows = [r for rs in FILES.values() for r in rs]
    with pytest.raises(data.DataError, match="at most 2 per batch"):
        data.validate(rows)  # the old whole-set validation applied the batch cap across batches
    report = data.validate_training_set(FILES)  # these three lessons also meet the freeze floors on their own
    assert report["shortfalls"] == [] and "partial_set" not in report


def test_the_batch_cap_still_holds_inside_one_batch(three_lessons_share_a_signature):
    as_one_batch = {f"batch9-{Path(p).name.split('-', 1)[1]}": rs for p, rs in FILES.items()}
    with pytest.raises(data.DataError, match="batch batch9: .*at most 2 per batch"):
        data.validate_training_set(as_one_batch, require_freeze=False)


def test_files_group_into_batches_by_their_prefix():
    assert data.batch_of("x/batch3-t05.jsonl") == "batch3" and data.batch_of("pilot-t04.jsonl") == "pilot"


def test_the_whole_current_dataset_validates_for_training():
    files = {str(p): data.load(p) for p in sorted(DATA.glob("*.jsonl")) if not p.name.endswith(".models.jsonl")}
    assert len(files) >= 10
    assert data.validate_training_set(files)["shortfalls"] == []


def test_part_of_the_data_is_refused_unless_marked_as_a_partial_set():
    """The freeze floors hold for the whole dataset; a smoke or a test trains on a few files with --partial-set."""
    part = {str(DATA / f): data.load(DATA / f) for f in ("pilot-t04.jsonl", "batch2-t01.jsonl")}
    with pytest.raises(data.DataError, match="freeze"):
        data.validate_training_set(part)
    report = data.validate_training_set(part, require_freeze=False)
    assert report["partial_set"] is True and "freeze" in report["freeze_refused"]


STAR_MAT = "The depot opens at 08:00 on weekdays.\nIt is closed on Sundays."
STAR_RIGHT = "VERDICT: unsupported\nDECIDING: It is closed on Sundays.\nESCALATE: no"


def star_row(rid="T05-D-ST-1-01", reply=STAR_RIGHT, **kw):
    return {"id": rid, "split": "star", "star_of": "T05-D-AP-01", "lesson": "T05", "trait": "t", "kind": "D",
            "tier": 2, "material": STAR_MAT, "verdict": "unsupported", "deciding": ["It is closed on Sundays."],
            "escalate": "no", "turns": [{"user": f"Material:\n{STAR_MAT}\nStatement: the depot opens on Sunday.",
                                         "thinking": "My own check: the Sunday line rules it out.",
                                         "reply": reply}], **kw}


def test_star_rows_skip_the_curriculum_gates_and_are_checked_by_the_trainers_scorer():
    """Argon's finding 7.2: a STaR row is the model's own answer, so the exact-reply, span, frame and freeze gates
    don't apply; its shape and its three lines (train.star_three_lines_ok, the pinned scorer) do."""
    import train
    files = {**FILES, "star-s1.jsonl": [star_row()]}
    with pytest.raises(data.DataError, match="star_check"):
        data.validate_training_set(files)  # no scorer given: refused, never waved through
    report = data.validate_training_set(files, star_check=train.star_three_lines_ok)
    assert report["star_rows"] == 1 and report["shortfalls"] == []
    wrong = {**FILES, "star-s1.jsonl": [star_row(reply=STAR_RIGHT.replace("unsupported", "supported"))]}
    with pytest.raises(data.DataError, match="don't pass the pinned scorer"):
        data.validate_training_set(wrong, star_check=train.star_three_lines_ok)
    for bad, msg in ((star_row(rid="T05-D-AP-01"), "ST-"), (star_row(star_of="T05-D-01"), "star_of"),
                     (star_row(rid="T06-D-ST-1-01"), "matching its lesson")):
        with pytest.raises(data.DataError, match=msg):
            data.validate_training_set({**FILES, "star-s1.jsonl": [bad]}, star_check=lambda r: True)


def test_an_apply_row_in_the_training_data_is_refused():
    """Argon's finding 7.2: split apply, or an -AP- id, never reaches training."""
    rs = next(iter(FILES.values()))
    for change in ({"split": "apply"}, {"id": rs[0]["id"].split("-")[0] + "-D-AP-01"}):
        files = {**FILES, "x.jsonl": [{**rs[0], **change}]}
        with pytest.raises(data.DataError, match="apply item"):
            data.validate_training_set(files, require_freeze=False)
    with pytest.raises(data.DataError, match="star check"):
        data.check_item(star_row())
