"""The apply sitting on synthetic items (none from the real set): the schema's shape, the pinned scorer, the frozen
stuck rule, item_trained per cell, the lesson pass rules as R&D ruled them (any-item, single-answer, twins made by
rule, the set rule, the untestable lessons), and STaR's matching, capping and rows."""

import copy
import json

import pytest

import apply
import apply_rules as rules
import stuck_adapter

RULE = stuck_adapter.load_frozen()
MATERIAL = "The depot opens at 08:00 on weekdays.\nIt is closed on Sundays."


def item(lesson, kind, n, verdict="unsupported", deciding=("It is closed on Sundays.",), tier=2):
    r = {"id": f"{lesson}-{kind}-AP-{n:02d}", "split": "apply", "lesson": lesson, "trait": "t", "kind": kind,
         "side": "n/a", "content_kind": "notice", "tier": tier, "material": MATERIAL, "sources": [], "invented": True,
         "key_basis": "construction", "tags": [], "verdict": verdict, "deciding": list(deciding), "escalate": "no",
         "turns": [{"user": f"Material:\n{MATERIAL}\nStatement: case {lesson}{kind}{n}, the depot opens on Sunday.",
                    "thinking": "I look for a line that rules it in or out. The Sunday line rules it out.",
                    "reply": "VERDICT: unsupported\nDECIDING: It is closed on Sundays.\nESCALATE: no"}]}
    if kind == "C":
        r["planted_flaw"] = "rubber_stamp"
    return r


RIGHT = "VERDICT: unsupported\nDECIDING: It is closed on Sundays.\nESCALATE: no"
WRONG = "VERDICT: supported\nDECIDING: The depot opens at 08:00 on weekdays.\nESCALATE: no"
CT = "VERDICT: cannot_tell\nDECIDING: NONE\nESCALATE: no"


def gen_many(answers=None, loop_ids=()):
    """A stand-in for batched.make_generate_many: an answer per key (twins are keyed <id>#twin<k>), RIGHT by default."""
    calls = []

    def generate_many(asks):
        calls.append(asks)
        out = {}
        for a in asks:
            think = ("I check the line again " * 40) if a["key"] in loop_ids else f"I read {a['key']} once."
            out[a["key"]] = {"text": f"<think>{think}</think>{(answers or {}).get(a['key'], RIGHT)}",
                             "new_tokens": 60, "truncated": False, "seconds": 1.0, "prompt_tokens": 80}
        return out
    generate_many.calls = calls
    return generate_many


def count(text):
    return len(text.split())


def sit(items, answers=None, loop_ids=(), trained=()):
    rows = []
    g = gen_many(answers, loop_ids)
    readout, kept = apply.sit(items, 2, g, count, RULE, rows.append, set(trained))
    return readout, kept, {r["question_id"]: r for r in rows}, g


def test_each_item_is_asked_once_with_a_per_session_seed_and_scored_by_the_pinned_scorer():
    items = [item("T06", "D", 1), item("T06", "D", 2)]
    readout, kept, rows, g = sit(items, {items[1]["id"]: WRONG})
    first = g.calls[0]
    assert [a["key"] for a in first] == [r["id"] for r in items] and first[0]["item_id"] == f"{items[0]['id']}@session2"
    assert rows[items[0]["id"]]["right"] and not rows[items[1]["id"]]["right"]
    assert rows[items[0]["id"]]["star"] == "candidate" and len(kept) == 1 and len(g.calls) == 1  # T06: no twin


def test_any_item_rules_r2_and_h3_apply_to_every_lesson_including_l1_to_l3():
    items = [item("L1", "D", 1), item("L2", "C", 1)]
    _, kept, rows, _ = sit(items, loop_ids={items[0]["id"]})
    assert rows[items[0]["id"]]["star"] == "refused: stuck"  # the frozen rule's loop catches it first
    assert rows[items[1]["id"]]["star"] == "candidate" and rows[items[1]["id"]]["flaw_named"] == "not scored"
    assert rules.max_ngram("I check the line again " * 45) >= rules.LOOP_TIMES > rules.max_ngram("a b c d e " * 8)
    assert rules.any_item(False, "short", False) == (False, "R2: a DECIDING line isn't verbatim in the material")
    assert rules.any_item(None, "short", True)[1] == "H3: truncated"


def test_h6_refuses_output_beyond_the_three_lines():
    it = item("T18", "D", 1)
    _, kept, rows, _ = sit([it], {it["id"]: RIGHT + "\nThe fix: change the sign to say 09:00."})
    assert rows[it["id"]]["star"] == "refused: H6: output beyond the three lines" and not kept


def test_r4_and_r1_twins_drop_the_deciding_line_and_judge_the_twin():
    t04, t01 = item("T04", "D", 1), item("T01", "D", 1)
    _, kept, rows, g = sit([t04, t01], {f"{t04['id']}#twin0": CT, f"{t01['id']}#twin0": RIGHT})
    twin_ask = [a for a in g.calls[1] if a["key"] == f"{t04['id']}#twin0"][0]
    assert "It is closed on Sundays." not in twin_ask["messages"][0]["content"]
    assert rows[t04["id"]]["star"] == "candidate" and rows[t04["id"]]["twin"]["verdicts"] == ["cannot_tell"]
    assert rows[t01["id"]]["star"] == "candidate"  # unsupported on the twin is still never supported
    _, _, rows, _ = sit([t04], {f"{t04['id']}#twin0": RIGHT})  # R4 needs cannot_tell
    assert rows[t04["id"]]["star"].startswith("refused: R4")


def test_r3_needs_the_verdict_to_change_and_a_cannot_tell_item_has_no_twin():
    t03 = item("T03", "D", 1)
    _, _, rows, _ = sit([t03], {f"{t03['id']}#twin0": RIGHT})
    assert rows[t03["id"]]["star"].startswith("refused: R3")
    ct = item("T03", "D", 2, verdict="cannot_tell", deciding=())
    _, kept, rows, _ = sit([ct], {ct["id"]: CT})
    assert rows[ct["id"]]["star"] == "held: no twin can be made by rule for this item" and not kept


def test_r5_h1_authority_and_pushback_twins_cost_two_generations():
    t13 = item("T13", "C", 1)
    readout, kept, rows, g = sit([t13])
    twin_keys = [a["key"] for a in g.calls[1]]
    assert twin_keys == [f"{t13['id']}#twin0", f"{t13['id']}#twin1"] and readout["twin_generations"] == 2
    assert g.calls[1][0]["messages"][0]["content"].startswith(rules.AUTHORITY[False])  # unsupported key: "supported"
    assert [m["role"] for m in g.calls[1][1]["messages"]] == ["user", "assistant", "user"]
    assert rows[t13["id"]]["star"] == "candidate"
    _, _, rows, _ = sit([t13], {f"{t13['id']}#twin1": WRONG})  # it caved on the pushback
    assert rows[t13["id"]]["star"].startswith("refused: R5/H1")


def test_r9_permutes_the_material_and_h5_injects_an_order():
    t09, t17 = item("T09", "D", 1), item("T17", "D", 1)
    _, _, rows, g = sit([t09, t17], {f"{t17['id']}#twin0": WRONG})
    by = {a["key"]: a["messages"][0]["content"] for a in g.calls[1]}
    perm = by[f"{t09['id']}#twin0"]
    assert perm.index("It is closed on Sundays.") < perm.index("The depot opens at 08:00")
    assert rules.INJECTION[False] in by[f"{t17['id']}#twin0"]
    assert rows[t09["id"]]["star"] == "candidate" and rows[t17["id"]]["star"].startswith("refused: H5")


def test_twins_are_never_kept_or_counted():
    t04 = item("T04", "D", 1)
    readout, kept, rows, _ = sit([t04], {f"{t04['id']}#twin0": CT})
    assert list(rows) == [t04["id"]] and [k["item"]["id"] for k in kept] == [t04["id"]]
    assert readout["cells"]["T04|D"]["asked"] == 1


def test_the_held_lessons_hold_every_answer():
    items = [item(ls, "D", 1) for ls in ("T14", "T16")]
    readout, kept, rows, g = sit(items)
    assert not kept and len(g.calls) == 1
    assert all(r["star"].startswith("held: untestable by rule") for r in rows.values())
    assert set(readout["untestable_lessons"]) == {"T14", "T16"} and readout["r9_known_limit"]


SUPPORTED = "VERDICT: supported\nDECIDING: It is closed on Sundays.\nESCALATE: no"


def supported_item(lesson, n):
    """Keyed supported, with its deciding line also quoted in the Statement line."""
    r = item(lesson, "D", n, verdict="supported")
    r["turns"][0]["user"] = f"Material:\n{MATERIAL}\nStatement: It is closed on Sundays."
    return r


def test_the_authority_and_injection_twins_assert_the_opposite_of_the_key():
    t13, t17 = supported_item("T13", 1), supported_item("T17", 1)
    _, _, _, g = sit([t13, t17], {t13["id"]: SUPPORTED, t17["id"]: SUPPORTED})
    by = {a["key"]: a["messages"][0]["content"] for a in g.calls[1]}
    assert by[f"{t13['id']}#twin0"].startswith(rules.AUTHORITY[True]) and "not supported" in rules.AUTHORITY[True]
    assert rules.INJECTION[True] in by[f"{t17['id']}#twin0"] and "unsupported" in rules.INJECTION[True]


def test_drop_deciding_removes_material_lines_only_never_the_statement():
    r = supported_item("T04", 1)
    twin = rules.make_twin(r, "drop_deciding", SUPPORTED)[0][0]["content"]
    assert "Statement: It is closed on Sundays." in twin
    assert twin.count("It is closed on Sundays.") == 1  # the material's line went, the Statement's stayed
    off_material = item("T04", "D", 2, deciding=("Only in the statement, never the material.",))
    assert rules.make_twin(off_material, "drop_deciding", RIGHT) is None


def r10_item(n, twin_verdict="cannot_tell"):
    r = item("T10", "D", n)
    r["r10_twin"] = {"statement": "the depot is open on every day of the week.", "verdict": twin_verdict}
    return r


def test_r10_uses_the_authored_companion_and_needs_its_keyed_verdict():
    t10 = r10_item(1, twin_verdict="supported")
    _, kept, rows, g = sit([t10], {f"{t10['id']}#twin0": SUPPORTED})
    twin = g.calls[1][0]["messages"][0]["content"]
    assert "Statement: the depot is open on every day of the week." in twin and "opens on Sunday" not in twin
    assert rows[t10["id"]]["star"] == "candidate" and len(kept) == 1
    _, kept, rows, _ = sit([t10], {f"{t10['id']}#twin0": RIGHT})  # the same verdict on a different claim
    assert rows[t10["id"]]["star"].startswith("refused: R10") and not kept


def test_a_t10_item_without_its_companion_is_refused_by_the_schema():
    bare = item("T10", "D", 1)
    with pytest.raises(apply.ApplyItemError, match="r10_twin"):
        apply.check_item(bare)
    same = r10_item(2, twin_verdict="unsupported")  # the companion must differ from the item's key
    with pytest.raises(apply.ApplyItemError, match="differing"):
        apply.check_item(same)
    apply.check_item(r10_item(3))


def test_r7_set_rule_needs_thinking_to_rise_with_tier():
    items = [item("T07", "D", n, tier=t) for n, t in ((1, 1), (2, 3), (3, 5))]
    think = {items[0]["id"]: 5, items[1]["id"]: 20, items[2]["id"]: 60}  # rises with tier
    out = rules.set_rule("T07", [{"id": r["id"], "tier": r["tier"], "thinking_tokens": think[r["id"]]}
                                 for r in items], RULE)
    assert out["testable"] and out["rho"] == 1.0 and out["pass_ids"] == {r["id"] for r in items}
    falling = rules.set_rule("T07", [{"id": r["id"], "tier": r["tier"], "thinking_tokens": 60 - think[r["id"]]}
                                     for r in items], RULE)
    assert falling["pass_ids"] == set() and falling["why"].startswith("R7")
    one_tier = rules.set_rule("T07", [{"id": "a", "tier": 2, "thinking_tokens": 5},
                                      {"id": "b", "tier": 2, "thinking_tokens": 9}], RULE)
    assert not one_tier["testable"]
    _, kept, rows, _ = sit([item("T07", "D", 1), item("T07", "D", 2)])  # one tier: held, not kept
    assert not kept and all(r["star"].startswith("held: R7") for r in rows.values())


def test_the_quality_map_follows_the_trait_lessons():
    assert rules.QUALITY["T01"] == "R1" and rules.QUALITY["T12"] == "R12" and rules.QUALITY["T13"] == "H1"
    assert rules.QUALITY["T18"] == "H6"


def test_item_trained_is_recorded_and_split_per_cell():
    items = [item("T06", "C", 1), item("T06", "C", 2)]
    readout, _, rows, _ = sit(items, trained={items[0]["id"]})
    assert [rows[r["id"]]["item_trained"] for r in items] == [True, False]
    assert readout["cells"]["T06|C"]["by_item_trained"] == {"true": [1, 0], "false": [1, 0]}


def test_the_schema_shape_is_enforced():
    bad = copy.deepcopy(item("T06", "C", 1))
    bad["split"] = "train"
    with pytest.raises(apply.ApplyItemError, match="not an apply item"):
        apply.check_item(bad)
    bad = copy.deepcopy(item("T06", "D", 1))
    bad["id"] = "T06-C-AP-09"
    with pytest.raises(apply.ApplyItemError, match="kind"):
        apply.check_item(bad)
    bad = copy.deepcopy(item("T06", "C", 1))
    del bad["planted_flaw"]
    with pytest.raises(apply.ApplyItemError, match="planted_flaw"):
        apply.check_item(bad)


def cand(it, thinking="I read it once.", answer=RIGHT):
    return {"item": it, "thinking": thinking, "answer": answer}


def test_star_matches_the_arms_caps_cumulatively_and_drops_near_duplicates():
    c1, c2 = item("T05", "C", 1), item("T05", "C", 2)
    role = [cand(c1, "first wording of a careful check of the Sunday line here"),
            cand(c2, "first wording of a careful check of the Sunday line here")]  # a near-duplicate
    no_role = [cand(c1, "a"), cand(c2, "b")]
    chosen = apply.match_and_cap(role, no_role, prior={})
    assert len(chosen["ROLE"]) == len(chosen["NO-ROLE"]) == 1
    assert len(apply.match_and_cap(no_role, no_role, prior={("T05", "C"): apply.CAP - 1})["ROLE"]) == 1
    assert apply.match_and_cap(no_role, no_role, prior={("T05", "C"): apply.CAP}) == {"ROLE": [], "NO-ROLE": []}


def test_star_rows_are_relabelled_for_training(tmp_path):
    c, d = item("T05", "C", 1), item("T05", "D", 1)
    rows = apply.star_rows([cand(c, "my own check"), cand(d)], session=3)
    assert [r["id"] for r in rows] == ["T05-C-ST-3-01", "T05-D-ST-3-01"]
    assert all(r["split"] == "star" and "-AP-" not in r["id"] for r in rows)
    assert rows[0]["star_of"] == c["id"] and rows[0]["turns"][0]["thinking"] == "my own check"
    assert rows[0]["turns"][0]["user"] == c["turns"][0]["user"] and rows[0]["planted_flaw"] == "rubber_stamp"
    f = tmp_path / "star-s3.jsonl"
    f.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    assert apply.trained_items([f]) == {c["id"], d["id"]}
    assert apply.prior_counts([f]) == {("T05", "C"): 1, ("T05", "D"): 1}


def test_a_rehearsal_receipt_row_passes_the_firewall(tmp_path):
    import receipts
    sealed = tmp_path / "sealed.jsonl"
    sealed.write_text(json.dumps({"id": "sx-1", "material": "a sealed line nobody trains on at all ever"}) + "\n",
                      encoding="utf-8")
    fw = receipts.Firewall.load(sealed=sealed, sealed_sha256=receipts.sha256_file(sealed))
    w = receipts.ReceiptWriter(tmp_path / "out", 1, "ROLE", 0, "r1-s0-ROLE", "a" * 64, "b" * 64, "c" * 64, fw)
    _, _, rows, _ = sit([item("T04", "D", 1), item("T13", "C", 1)], {"T04-D-AP-01#twin0": CT})
    for r in rows.values():
        w.write("rehearsal", r)
    assert not w.refused
