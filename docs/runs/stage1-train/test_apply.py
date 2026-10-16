"""The apply sitting on synthetic items (none from the real set): the schema's shape, the pinned scorer, the frozen
stuck rule, item_trained per cell, the lesson pass rules as R&D ruled them (any-item, single-answer, twins made by
rule, the set rule, the untestable lessons), and STaR's matching, capping and rows."""

import copy
from pathlib import Path
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
    t09["order_free"] = True
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


def test_t14_is_tested_by_r4s_twin_on_decided_items_and_kept_on_the_rest_otherwise():
    t14 = item("T14", "D", 1)
    readout, kept, rows, g = sit([t14], {f"{t14['id']}#twin0": CT})
    assert rows[t14["id"]]["star"] == "candidate" and rows[t14["id"]]["twin"]["rule"] == "drop_deciding"
    assert "R4's twin" in readout["h2_disclosure"] and readout["untestable_lessons"] == {}
    _, kept, rows, _ = sit([t14], {f"{t14['id']}#twin0": RIGHT})  # filled the gap instead of cannot_tell
    assert rows[t14["id"]]["star"].startswith("refused: H2 (by R4's twin)") and not kept
    undecided = item("T14", "D", 2, verdict="cannot_tell", deciding=())
    _, kept, rows, g = sit([undecided], {undecided["id"]: CT})
    assert rows[undecided["id"]]["star"] == "candidate" and rows[undecided["id"]]["twin"] is None  # no twin
    assert len(g.calls) == 1 and len(kept) == 1


def h4_rows(n, abstain=0, false_reject=0):
    rows = [{"id": f"r{i}", "key_verdict": "unsupported", "given": "unsupported"} for i in range(n)]
    for i in range(abstain):
        rows[i]["given"] = "cannot_tell"
    for i in range(abstain, abstain + false_reject):
        rows[i]["given"] = "supported"
    return rows


A0_ZERO = {"abstain": 0.0, "false_reject": 0.0, "n": 30, "source": "test"}


def test_t16_with_n_decided_10_passes_1_abstention_and_fails_2_when_a0_is_0():
    one = rules.h4_rule(h4_rows(10, abstain=1), A0_ZERO)  # 0.10 <= max(0.10, 1/10)
    assert one["testable"] and one["pass_ids"] == {f"r{i}" for i in range(10)} and one["abstain"] == 1
    two = rules.h4_rule(h4_rows(10, abstain=2), A0_ZERO)  # 0.20 > 0.10
    assert two["pass_ids"] == set() and two["why"].startswith("H4")


def test_t16_false_reject_line_is_a0_plus_0_15_or_2_over_n():
    assert rules.h4_rule(h4_rows(10, false_reject=2), A0_ZERO)["pass_ids"]  # 0.20 <= max(0.15, 0.20)
    assert not rules.h4_rule(h4_rows(10, false_reject=3), A0_ZERO)["pass_ids"]  # 0.30 > 0.20
    a0 = {"abstain": 0.0, "false_reject": 0.20, "n": 30, "source": "test"}
    assert rules.h4_rule(h4_rows(10, false_reject=3), a0)["pass_ids"]  # 0.30 <= 0.35
    undecided = h4_rows(10) + [{"id": "u", "key_verdict": "cannot_tell", "given": "cannot_tell"}]
    out = rules.h4_rule(undecided, A0_ZERO)
    assert out["n_decided"] == 10 and "u" in out["pass_ids"]  # counted over decided items; the set keeps all


def test_t16_uses_a0_from_devs_decided_items():
    assert rules.A0_H4["abstain_count"] == 3 and rules.A0_H4["false_reject_count"] == 2 and rules.A0_H4["n"] == 78
    items = [item("T16", "D", n) for n in range(1, 11)]  # 10 decided, all answered right
    _, kept, rows, _ = sit(items)
    assert len(kept) == 10 and all(r["set_rule"]["testable"] for r in rows.values())
    _, kept, rows, _ = sit(items, {items[0]["id"]: CT, items[1]["id"]: CT})  # 2 abstentions: over max(.138, .1)
    assert not kept and rows[items[2]["id"]]["star"].startswith("refused: H4")


def test_h4_rates_counts_over_decided_items_only():
    keys = {"a": "supported", "b": "unsupported", "c": "cannot_tell"}
    recs = [{"task_id": "a", "final": {"verdict": "cannot_tell"}}, {"task_id": "b", "final": {"verdict": "supported"}},
            {"task_id": "c", "final": {"verdict": "supported"}}, {"task_id": "a", "final": {"verdict": "supported"}}]
    assert rules.h4_rates(recs, keys) == {"n": 3, "abstain_count": 1, "false_reject_count": 1,
                                          "abstain": 1 / 3, "false_reject": 1 / 3}


A0_DIR = Path("E:/AI/aspire-si/outputs/stage1-baseline")


@pytest.mark.skipif(not (A0_DIR / "pilot-30-s0" / "raw.jsonl").exists(), reason="A_0's raw files are local-only")
def test_a0_reproduces_abstain_3_of_78_and_false_reject_2_of_78_from_the_pinned_raw_files():
    import hashlib
    here = Path(__file__).resolve().parent.parent / "stage1-taskset"
    keys = {}
    for f in ("taskset-review-10.jsonl", "taskset-pilot-20.jsonl"):
        for line in (here / f).read_text(encoding="utf-8").splitlines():
            if line.strip():
                t = json.loads(line)
                keys[t["id"]] = t["key"]["verdict"]
    recs = []
    for seed, sha in rules.A0_RAW_SHA256.items():
        raw = A0_DIR / f"pilot-30-s{seed}" / "raw.jsonl"
        assert hashlib.sha256(raw.read_bytes()).hexdigest() == sha  # the files A_0 was computed from
        recs += [json.loads(x) for x in raw.read_text(encoding="utf-8").splitlines() if x.strip()]
    got = rules.h4_rates(recs, keys)
    assert (got["n"], got["abstain_count"], got["false_reject_count"]) == (78, 3, 2)
    assert got["abstain"] == rules.A0_H4["abstain"] and got["false_reject"] == rules.A0_H4["false_reject"]


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


def test_r9_permutes_only_an_order_free_item_and_holds_the_rest():
    ordered = item("T09", "D", 2)
    ordered["order_free"] = False  # a changelog or a table: reversing it can rightly change the answer
    _, kept, rows, g = sit([ordered])
    assert rows[ordered["id"]]["star"] == "held: no twin can be made by rule for this item" and not kept
    assert len(g.calls) == 1  # no twin was generated
    free = item("T09", "D", 3)
    free["order_free"] = True
    assert rules.make_twin(free, "permute", RIGHT) is not None
    assert rules.make_twin({**free, "order_free": None}, "permute", RIGHT) is None


def test_a_t09_item_without_order_free_is_refused_by_the_schema():
    with pytest.raises(apply.ApplyItemError, match="order_free"):
        apply.check_item(item("T09", "D", 1))
    ok = item("T09", "D", 1)
    ok["order_free"] = False
    apply.check_item(ok)
