"""The apply sitting on synthetic items (none from the real set): the schema's shape is enforced, the three lines are
scored by the pinned scorer, stuck by the frozen rule, item_trained is recorded and split per cell, the keep rule
holds when the lesson rule isn't testable, and STaR's matching, capping and rows follow amendment A and R&D's
addition 1."""

import copy
import json

import pytest

import apply
import stuck_adapter

RULE = stuck_adapter.load_frozen()
MATERIAL = "The depot opens at 08:00 on weekdays.\nIt is closed on Sundays."


def item(lesson, kind, n, verdict="unsupported", deciding=("It is closed on Sundays.",)):
    r = {"id": f"{lesson}-{kind}-AP-{n:02d}", "split": "apply", "lesson": lesson, "trait": "t", "kind": kind,
         "side": "n/a", "content_kind": "notice", "tier": 2, "material": MATERIAL, "sources": [], "invented": True,
         "key_basis": "construction", "tags": [], "verdict": verdict, "deciding": list(deciding), "escalate": "no",
         "turns": [{"user": f"Material:\n{MATERIAL}\nStatement: case {lesson}{kind}{n}, the depot opens on Sunday.",
                    "thinking": "I look for a line that rules it in or out. The Sunday line rules it out.",
                    "reply": "VERDICT: unsupported\nDECIDING: It is closed on Sundays.\nESCALATE: no"}]}
    if kind == "C":
        r["planted_flaw"] = "rubber_stamp"
    return r


ITEMS = [item(ls, k, n) for ls in ("T05", "T06") for k in ("C", "D") for n in (1, 2)]
RIGHT = "VERDICT: unsupported\nDECIDING: It is closed on Sundays.\nESCALATE: no"
WRONG = "VERDICT: supported\nDECIDING: The depot opens at 08:00 on weekdays.\nESCALATE: no"


def gen_many(answers, loop_ids=()):
    """A stand-in for batched.make_generate_many: the answer per item id; a looping trace for loop_ids."""
    seen = []

    def generate_many(asks):
        seen.extend(asks)
        out = {}
        for a in asks:
            think = ("I check the line again " * 40) if a["key"] in loop_ids else f"I read {a['key']} once."
            out[a["key"]] = {"text": f"<think>{think}</think>{answers.get(a['key'], RIGHT)}", "new_tokens": 60,
                             "truncated": False, "seconds": 1.0, "prompt_tokens": 80}
        return out
    generate_many.seen = seen
    return generate_many


def count(text):
    return len(text.split())


def sit(answers=None, loop_ids=(), trained=(), rules=None, monkeypatch=None):
    rows = []
    if monkeypatch is not None:
        monkeypatch.setattr(apply, "LESSON_RULES", rules or {})
    g = gen_many(answers or {}, loop_ids)
    readout, kept = apply.sit(ITEMS, 2, g, count, RULE, rows.append, set(trained))
    return readout, kept, rows, g


def test_each_item_is_asked_once_with_a_per_session_seed_and_scored_by_the_pinned_scorer(monkeypatch):
    readout, kept, rows, g = sit({ITEMS[1]["id"]: WRONG}, monkeypatch=monkeypatch)
    assert [a["key"] for a in g.seen] == [r["id"] for r in ITEMS] and {a["sample"] for a in g.seen} == {0}
    assert g.seen[0]["item_id"] == f"{ITEMS[0]['id']}@session2"
    by = {r["question_id"]: r for r in rows}
    assert by[ITEMS[0]["id"]]["right"] and not by[ITEMS[1]["id"]]["right"]
    assert by[ITEMS[0]["id"]]["flaw_named"] == "not scored" and by[ITEMS[2]["id"]]["flaw_named"] is None
    assert readout["c_flaw_naming"].startswith("not scored")


def test_stuck_uses_the_frozen_rule_and_refuses_the_keep(monkeypatch):
    rules = {"T05": lambda r, a: True, "T06": lambda r, a: True}
    _, kept, rows, _ = sit(loop_ids={ITEMS[0]["id"]}, rules=rules, monkeypatch=monkeypatch)
    by = {r["question_id"]: r for r in rows}
    assert by[ITEMS[0]["id"]]["stuck"] and "loop_span" in by[ITEMS[0]["id"]]["stuck_reasons"]
    assert by[ITEMS[0]["id"]]["star"] == "refused: stuck"
    assert ITEMS[0]["id"] not in {k["item"]["id"] for k in kept} and len(kept) == len(ITEMS) - 1


def test_an_untestable_lesson_rule_holds_the_answer_never_keeps_it(monkeypatch):
    _, kept, rows, _ = sit(rules={"T05": lambda r, a: True}, monkeypatch=monkeypatch)
    stars = {r["question_id"]: r["star"] for r in rows}
    assert kept and all(k["item"]["lesson"] == "T05" for k in kept)
    assert all(v.startswith("held") for i, v in stars.items() if i.startswith("T06"))
    assert {r["lesson_rule"] for r in rows if r["lesson"] == "T06"} == {"not yet testable"}


def test_item_trained_is_recorded_and_split_per_cell(monkeypatch):
    readout, _, rows, _ = sit(trained={ITEMS[0]["id"]}, monkeypatch=monkeypatch)
    assert [r["item_trained"] for r in rows] == [r["id"] == ITEMS[0]["id"] for r in ITEMS]
    cell = readout["cells"]["T05|C"]
    assert cell["asked"] == 2 and cell["by_item_trained"] == {"true": [1, 0], "false": [1, 0]}


def test_the_schema_shape_is_enforced():
    bad = copy.deepcopy(ITEMS[0])
    bad["split"] = "train"
    with pytest.raises(apply.ApplyItemError, match="not an apply item"):
        apply.check_item(bad)
    bad = copy.deepcopy(ITEMS[2])
    bad["id"] = "T05-C-AP-09"  # a D item with a C id
    with pytest.raises(apply.ApplyItemError, match="kind"):
        apply.check_item(bad)
    bad = copy.deepcopy(ITEMS[0])
    del bad["planted_flaw"]
    with pytest.raises(apply.ApplyItemError, match="planted_flaw"):
        apply.check_item(bad)


def cand(it, thinking="I read it once.", answer=RIGHT):
    return {"item": it, "thinking": thinking, "answer": answer}


def test_star_matches_the_arms_caps_cumulatively_and_drops_near_duplicates():
    t05c = [i for i in ITEMS if i["lesson"] == "T05" and i["kind"] == "C"]
    role = [cand(t05c[0], "first wording of a careful check of the Sunday line here"),
            cand(t05c[1], "first wording of a careful check of the Sunday line here")]  # a near-duplicate
    no_role = [cand(t05c[0], "a"), cand(t05c[1], "b")]
    chosen = apply.match_and_cap(role, no_role, prior={})
    assert len(chosen["ROLE"]) == len(chosen["NO-ROLE"]) == 1  # the duplicate went first; then matched to 1
    capped = apply.match_and_cap(no_role, no_role, prior={("T05", "C"): apply.CAP - 1})
    assert len(capped["ROLE"]) == 1  # only 1 left under the cumulative cap of 8
    assert apply.match_and_cap(no_role, no_role, prior={("T05", "C"): apply.CAP}) == {"ROLE": [], "NO-ROLE": []}


def test_star_rows_are_relabelled_for_training(tmp_path):
    rows = apply.star_rows([cand(ITEMS[0], "my own check"), cand(ITEMS[2])], session=3)
    assert [r["id"] for r in rows] == ["T05-C-ST-3-01", "T05-D-ST-3-01"]
    assert all(r["split"] == "star" and "-AP-" not in r["id"] for r in rows)
    assert rows[0]["star_of"] == ITEMS[0]["id"] and rows[0]["turns"][0]["thinking"] == "my own check"
    assert rows[0]["turns"][0]["user"] == ITEMS[0]["turns"][0]["user"] and rows[0]["planted_flaw"] == "rubber_stamp"
    f = tmp_path / "star-s3.jsonl"
    f.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    assert apply.trained_items([f]) == {ITEMS[0]["id"], ITEMS[2]["id"]}
    assert apply.prior_counts([f]) == {("T05", "C"): 1, ("T05", "D"): 1}


def test_a_rehearsal_receipt_row_passes_the_firewall(tmp_path, monkeypatch):
    import receipts
    sealed = tmp_path / "sealed.jsonl"
    sealed.write_text(json.dumps({"id": "sx-1", "material": "a sealed line nobody trains on at all ever"}) + "\n",
                      encoding="utf-8")
    fw = receipts.Firewall.load(sealed=sealed, sealed_sha256=receipts.sha256_file(sealed))
    w = receipts.ReceiptWriter(tmp_path / "out", 1, "ROLE", 0, "r1-s0-ROLE", "a" * 64, "b" * 64, "c" * 64, fw)
    _, _, rows, _ = sit(monkeypatch=monkeypatch)
    for r in rows:
        w.write("rehearsal", r)
    assert not w.refused
