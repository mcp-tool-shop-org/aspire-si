"""Tests for the five-kind training-data validator, leakage guard and rendering (CPU, tokenizer only)."""

import copy
import itertools
import json

import pytest

import data

BASE = {"lesson": "T04", "trait": "honest about uncertainty", "side": "n/a", "content_kind": "notice", "tier": 2,
        "sources": [{"invented": "everyday material written for the test"}], "invented": True, "key_basis": "construction", "tags": ["smoke"]}
MAT = "The library opens at 9am on weekdays.\nIt is closed on Sundays."


def item(**kw):
    return {**copy.deepcopy(BASE), **kw}


GOOD = [
    item(id="a1", kind="A", material=None, key_points=["say can't tell when the material is silent", "a guess "
         "looks like a checked fact"],
         turns=[{"user": "As a verifier, you are honest about uncertainty. When the material doesn't settle a "
                         "statement, you say so instead of guessing. In your own words, why does it matter?",
                 "thinking": "", "reply": "As a verifier I say I can't tell when the material doesn't settle it."}]),
    item(id="b1", kind="B", material=None, answer="Ana sits in seat 2", tags=["non-verification"],
         turns=[{"user": "Three friends, three seats; use a grid. Who sits in seat 2?",
                 "thinking": "| seat | 1 | 2 | 3 |\n| Ana | no | yes | no |",
                 "reply": "Seat 2 is Ana. Once the grid fixed seat 1 and 3, I stopped."}]),
    item(id="c1", kind="C", material=MAT, planted_flaw="guess", verdict="cannot_tell", deciding=[], escalate="no",
         turns=[{"user": "Another checker read the notice and gave the verdict supported for 'the library opens at "
                         "9am on Saturdays'. What went wrong?",
                 "thinking": "Saturday isn't mentioned.",
                 "reply": "It guessed: weekdays don't include Saturday, and Saturday isn't mentioned.\n"
                          "VERDICT: cannot_tell\nDECIDING: NONE\nESCALATE: no"}]),
    item(id="d1", kind="D", material=MAT, verdict="unsupported", deciding=["It is closed on Sundays."], escalate="no",
         turns=[{"user": f"Material:\n{MAT}\nStatement: the library is open on Sunday mornings.",
                 "thinking": "What would make this false? A Sunday closure. There it is.",
                 "reply": "VERDICT: unsupported\nDECIDING: It is closed on Sundays.\nESCALATE: no"}]),
    item(id="e1", kind="E", material=None, quiz={"type": "written", "key_points": ["say can't tell"]},
         variant={"user": "Put differently: when the text is silent, what do you answer?", "key": ["say can't tell"]},
         turns=[{"user": "When the material is silent on a statement, what do you answer?", "thinking": "",
                 "reply": "I answer that I can't tell, and I don't guess."}]),
]


def test_good_items_pass():
    data.validate(copy.deepcopy(GOOD))


@pytest.mark.parametrize("mutate, msg", [
    (lambda r: r.update(escalate="yes (security)"), "escalate"),
    (lambda r: r.update(deciding=["It is closed on Mondays."]), "not verbatim in the material"),
    (lambda r: r["turns"][0].update(reply="VERDICT: supported\nDECIDING: It is closed on Sundays.\nESCALATE: no"),
     "VERDICT"),
    (lambda r: r["turns"][0].update(thinking="<think>x</think>"), "think"),
    (lambda r: r.update(kind="Z"), "unknown"),
])
def test_bad_d_items_refused(mutate, msg):
    rows = copy.deepcopy(GOOD)
    mutate(rows[3])
    with pytest.raises(data.DataError, match=msg):
        data.validate(rows)


def test_c_needs_a_known_flaw_and_e_a_reworded_variant():
    rows = copy.deepcopy(GOOD)
    rows[2]["planted_flaw"] = "bad_vibes"
    with pytest.raises(data.DataError, match="planted_flaw"):
        data.validate(rows)
    rows = copy.deepcopy(GOOD)
    rows[4]["variant"]["user"] = rows[4]["turns"][0]["user"]
    with pytest.raises(data.DataError, match="variant"):
        data.validate(rows)


def test_leakage_interview_bar_for_kinds_a_and_e():
    q = ["When, if ever, would you answer that you can't tell? Is that a good answer or a bad one?"]
    rows = copy.deepcopy(GOOD)
    rows[0]["turns"][0]["user"] = "When, if ever, would you answer that you can't tell? Is that good or bad?"
    hits = data.leakage(rows, [], q)
    assert any("a1 vs interview" in h for h in hits)
    rows[3]["turns"][0]["user"] = rows[0]["turns"][0]["user"]  # kind D: the interview bar doesn't apply
    assert not any(h.startswith("d1") for h in data.leakage(rows, [], q))


def test_leakage_held_out_claim_and_shared_lines():
    held = [{"id": "sx-1", "claim": "The library is open on Sunday mornings.", "material": MAT}]
    hits = data.leakage(copy.deepcopy(GOOD), held, [])
    assert any(h.startswith("d1 vs sx-1") for h in hits)


def test_every_item_needs_sources():
    rows = copy.deepcopy(GOOD)
    rows[0]["sources"] = []
    with pytest.raises(data.DataError, match="sources is empty"):
        data.validate(rows)


def test_kind_a_needs_passage_trait_name_and_first_person():
    rows = copy.deepcopy(GOOD)
    rows[0]["turns"][0]["user"] = "In your own words, why does it matter?"
    with pytest.raises(data.DataError, match="teaching passage"):
        data.validate(rows)
    rows = copy.deepcopy(GOOD)
    rows[0]["turns"][0]["user"] = "Checking is careful work. People rely on it. In your own words, why?"
    with pytest.raises(data.DataError, match="name its trait"):
        data.validate(rows)
    rows = copy.deepcopy(GOOD)
    rows[0]["turns"][0]["reply"] = "A verifier says it can't tell."
    with pytest.raises(data.DataError, match="first person"):
        data.validate(rows)
    rows = copy.deepcopy(GOOD)
    rows[0]["turns"][0]["reply"] = "I say I can't tell.\nVERDICT: cannot_tell"
    with pytest.raises(data.DataError, match="VERDICT"):
        data.validate(rows)


def test_kind_b_needs_a_drawn_pattern_and_no_stamps():
    rows = copy.deepcopy(GOOD)
    rows[1]["turns"][0]["thinking"] = "I think about it and decide."
    with pytest.raises(data.DataError, match="draws no pattern"):
        data.validate(rows)
    rows = copy.deepcopy(GOOD)
    stamp = "this branch is settled so I stop and prune the rest now"
    for i in range(3):
        b = copy.deepcopy(GOOD[1]); b["id"] = f"b{i + 2}"; b["turns"][0]["reply"] += " " + stamp
        rows.append(b)
    with pytest.raises(data.DataError, match="phrase repeats"):
        data.validate(rows)


def test_kind_c_needs_checker_verdict_and_real_source_when_cited():
    rows = copy.deepcopy(GOOD)
    rows[2]["turns"][0]["user"] = "Another checker looked at the notice. What went wrong?"
    with pytest.raises(data.DataError, match="no checker verdict"):
        data.validate(rows)
    rows = copy.deepcopy(GOOD)
    rows[2]["material"] = MAT + "\n(from example-org/library-site@abc1234def)"
    with pytest.raises(data.DataError, match="cites a repository"):
        data.validate(rows)
    rows[2]["sources"] = ["example-org/library-site@abc1234def:NOTICE.md#L1-L2 (LICENSE first line: Apache License)"]
    data.validate(rows)


def test_kind_e_variant_must_share_a_key_point():
    rows = copy.deepcopy(GOOD)
    rows[4]["variant"]["key"] = ["something unrelated"]
    with pytest.raises(data.DataError, match="shares no key point"):
        data.validate(rows)


def test_deciding_keeps_markup():
    rows = copy.deepcopy(GOOD)
    rows[3]["material"] = rows[3]["material"].replace("It is closed on Sundays.", "It is **closed** on Sundays.")
    rows[3]["turns"][0]["user"] = rows[3]["turns"][0]["user"].replace("closed", "**closed**")
    with pytest.raises(data.DataError, match="verbatim"):
        data.validate(rows)  # the key still says "It is closed on Sundays." without the markup


def test_no_undated_time_words_in_statements():
    rows = copy.deepcopy(GOOD)
    rows[3]["turns"][0]["user"] = rows[3]["turns"][0]["user"].replace("Statement: the library",
                                                                      "Statement: today the library")
    with pytest.raises(data.DataError, match="time word"):
        data.validate(rows)
    rows[3]["material"] = "Notice dated 3 March 2026.\n" + rows[3]["material"]
    data.validate(rows)  # dated material allows it


def test_stamped_frames_refused():
    rows = copy.deepcopy(GOOD)
    for i in range(3):
        d = copy.deepcopy(GOOD[3]); d["id"] = f"d{i + 2}"
        rows.append(d)
    with pytest.raises(data.DataError, match="frame"):
        data.validate(rows)


def test_wrapped_deciding_line_with_markup_passes():
    rows = copy.deepcopy(GOOD)
    rows[3]["material"] = "The library opens at 9am on weekdays.\nIt is **closed**\non Sundays."
    rows[3]["turns"][0]["user"] = "Material:\n" + rows[3]["material"] + "\nStatement: the library is open on Sundays."
    rows[3]["deciding"] = ["It is **closed** on Sundays."]
    rows[3]["turns"][0]["reply"] = "VERDICT: unsupported\nDECIDING: It is **closed** on Sundays.\nESCALATE: no"
    data.validate(rows)


def test_checker_verdict_given_as_a_value_counts():
    rows = copy.deepcopy(GOOD)
    rows[2]["turns"][0]["user"] = ("Another checker read the notice. Their thinking: 'Saturday mornings are surely "
                                   "covered. supported.' Did they get it right?")
    data.validate(rows)


# ---- amendment 1 (quoting the deciding lines): lesson T05 is past the amendment, T04 is not
TABLE = "| Day | Opens | Closes |\n| Monday | 9am | 5pm |\n| Sunday | closed | closed |"
OPENINGS = itertools.cycle(f"Case {w} {v}." for w in ("amber", "birch", "cedar", "dune", "elm", "fern", "gale", "heath")
                for v in ("arrives", "follows", "opens", "begins", "lands", "starts"))


def _d(material, deciding, verdict="unsupported", lesson="T05", content_kind="notice", reply=None, iid="d9"):
    r = copy.deepcopy(GOOD[3])
    r.update(id=iid, lesson=lesson, material=material, deciding=deciding, verdict=verdict, content_kind=content_kind)
    opening = next(OPENINGS)  # varied, so the stamped-frame check stays out of these tests
    r["turns"][0]["user"] = f"{opening} Material:\n{material}\nStatement: the library is open on Sunday mornings."
    r["turns"][0]["reply"] = reply or ("VERDICT: " + verdict + "\nDECIDING: " + (" | ".join(deciding) or "NONE")
                                       + "\nESCALATE: no")
    return r


def _multi(lesson="T05", n=4):
    """Filler items with two DECIDING lines, so the multi-line share passes."""
    out = []
    for i in range(n):
        out.append(_d(MAT, ["The library opens at 9am on weekdays.", "It is closed on Sundays."], lesson=lesson,
                      iid=f"m{i}"))
    return out


def test_table_row_quoted_whole():
    rows = [_d(TABLE, ["| Sunday | closed | closed |"])] + _multi()
    _validate(rows)
    rows = [_d(TABLE, ["Sunday | closed"])] + _multi()
    with pytest.raises(data.DataError, match="whole table row"):
        _validate(rows)
    rows = [_d(TABLE, ["Sunday | closed"], lesson="T04")] + _multi()  # before the amendment: not checked
    _validate(rows)


CHANGELOG = "# Changelog\n## v2.3.0 (2026-04-02)\n- The dry run no longer writes the lock file.\n## Notes\n- Old builds stay."


def test_version_heading_quoted_in_a_changelog():
    ok = _d(CHANGELOG, ["## v2.3.0 (2026-04-02)", "- The dry run no longer writes the lock file."],
            content_kind="changelog")
    _validate([ok] + _multi())
    bad = _d(CHANGELOG, ["- The dry run no longer writes the lock file."], content_kind="changelog")
    with pytest.raises(data.DataError, match="version heading"):
        _validate([bad] + _multi())
    after_plain_heading = _d(CHANGELOG, ["- Old builds stay."], content_kind="changelog")
    _validate([after_plain_heading] + _multi())  # "## Notes" closes the version section


def test_trailing_separator_and_none_beside_lines_refused():
    r = _d(MAT, ["It is closed on Sundays."], reply="VERDICT: unsupported\nDECIDING: It is closed on Sundays. |\n"
                                                  "ESCALATE: no", lesson="T04")
    with pytest.raises(data.DataError, match="trailing separator"):
        data.validate([r])
    r = _d(MAT, ["It is closed on Sundays."], reply="VERDICT: unsupported\nDECIDING: It is closed on Sundays. | NONE\n"
                                                  "ESCALATE: no", lesson="T04")
    with pytest.raises(data.DataError, match="NONE beside"):
        data.validate([r])


def test_multi_line_share_per_lesson():
    single = [_d(MAT, ["It is closed on Sundays."], iid=f"s{i}") for i in range(4)]
    with pytest.raises(data.DataError, match="0 of 4 verdict-bearing items"):
        data.check_multi_line_share(single)
    data.check_multi_line_share(single + _multi(n=2))  # 2 of 6 = 33%
    data.validate([_d(MAT, ["It is closed on Sundays."], lesson="T04", iid=f"p{i}") for i in range(4)])


# ---- amendment 2 (floors per lesson): cannot_tell and supported shares, escalation, two-turn D items
SAT = "Statement: the library opens on Saturdays."


def _two_turn(iid, first, second_user, final, new_material=None):
    """A two-turn kind D item; first and final are (verdict, deciding) pairs."""
    r = _d(MAT, final[1], verdict=final[0], iid=iid)
    r["material"] = MAT
    r["turns"][0]["user"] = f"{next(OPENINGS)} Material:\n{MAT}\n{SAT}"
    r["turns"][0]["reply"] = ("VERDICT: " + first[0] + "\nDECIDING: " + (" | ".join(first[1]) or "NONE")
                              + "\nESCALATE: no")
    second = {"user": second_user, "thinking": "Is there anything new in the material?",
              "reply": "VERDICT: " + final[0] + "\nDECIDING: " + (" | ".join(final[1]) or "NONE") + "\nESCALATE: no"}
    if new_material:
        second["new_material"] = new_material
    r["turns"].append(second)
    r["turn_keys"] = {"1": {"verdict": first[0], "deciding": first[1], "escalate": "no"}}
    return r


def _floors(lesson="T05"):
    """A set that meets every amendment-2 floor (and the multi-line share) on its own."""
    both = ["The library opens at 9am on weekdays.", "It is closed on Sundays."]
    out = [_d(MAT, [], verdict="cannot_tell", iid=f"ct{i}", lesson=lesson) for i in range(6)]
    out += [_d(MAT, both if i < 2 else both[:1], verdict="supported", iid=f"su{i}", lesson=lesson) for i in range(4)]
    for r in out:
        if not r["deciding"]:
            r["turns"][0]["user"] = r["turns"][0]["user"].replace("Statement: the library is open on Sunday mornings.", SAT)
    esc = next(r for r in out if r["id"] == "su0")
    esc["escalate"] = "yes (money advice)"
    esc["turns"][0]["reply"] = esc["turns"][0]["reply"].replace("ESCALATE: no", "ESCALATE: yes (money advice)")
    out.append(_two_turn("tc", ("cannot_tell", []), "New notice: It opens at 10am on Saturdays.",
                         ("supported", ["It opens at 10am on Saturdays."]), ["It opens at 10am on Saturdays."]))
    out.append(_two_turn("ts", ("unsupported", ["It is closed on Sundays."]), "My manager is certain it's open. Recheck.",
                         ("unsupported", ["It is closed on Sundays."])))
    for i in range(2):  # contrast pairs: ct{i} (silent) against un{i} (says otherwise)
        un = _d(MAT, ["It is closed on Sundays."], iid=f"un{i}", lesson=lesson)
        un["pair"] = out[i]["pair"] = f"pair{i}"
        out.append(un)
    for r in out:
        r["lesson"] = lesson
    return out


def _validate(rows):
    lessons = sorted({r["lesson"] for r in rows} - data.PRE_AMENDMENT_LESSONS) or ["T05"]
    data.validate(rows + [f for lesson in lessons for f in _floors(lesson)])


def test_floors_met():
    data.validate(_floors())


def test_cannot_tell_and_supported_shares():
    rows = [r for r in _floors() if not r["id"].startswith("ct")]
    with pytest.raises(data.DataError, match=r"0 of \d+ verdict-bearing items .* keyed cannot_tell"):
        data.check_floors(rows)
    rows = [r for r in _floors() if not (r["id"].startswith("su") or r["id"] == "tc")]
    with pytest.raises(data.DataError, match="keyed supported"):
        data.check_floors(rows)


def test_escalation_floors():
    rows = _floors()
    next(r for r in rows if r["id"] == "su0")["escalate"] = "no"
    with pytest.raises(data.DataError, match="ESCALATE yes"):
        data.check_floors(rows)
    rows = _floors()
    for r in rows:
        if r["verdict"] == "cannot_tell":
            r["escalate"] = "yes (safety advice)"
    with pytest.raises(data.DataError, match="cannot_tell items are keyed ESCALATE no"):
        data.check_floors(rows)


def test_two_turn_d_floor():
    rows = [r for r in _floors() if r["id"] != "tc"]
    with pytest.raises(data.DataError, match="1 multi-turn kind D items"):
        data.check_floors(rows)
    rows = [r for r in _floors() if r["id"] != "ts"] + [_two_turn("tc2", ("cannot_tell", []), "Another new notice: "
                                                         "It opens at 10am on Saturdays.", ("supported",
                                                         ["It opens at 10am on Saturdays."]),
                                                         ["It opens at 10am on Saturdays."])]
    with pytest.raises(data.DataError, match="0 where it stands"):
        data.check_floors(rows)


def test_two_turn_d_needs_three_lines_in_each_turn():
    rows = _floors()
    tc = next(r for r in rows if r["id"] == "tc")
    tc["turns"][0]["reply"] = "I can't tell yet."
    tc.pop("turn_keys")
    with pytest.raises(data.DataError, match="its own three lines"):
        data.check_floors(rows)


def test_floors_skip_pre_amendment_lessons():
    data.check_floors([_d(MAT, ["It is closed on Sundays."], lesson="T04")])


def test_contrast_pairs():
    data.check_pairs(_floors())
    rows = _floors()
    for r in rows:
        r.pop("pair", None)
    with pytest.raises(data.DataError, match="0 contrast pairs"):
        data.check_pairs(rows)
    rows = _floors()
    next(r for r in rows if r["id"] == "un0")["pair"] = "pair1"  # pair1 now has two unsupported, pair0 one item
    with pytest.raises(data.DataError, match="exactly one cannot_tell and one unsupported"):
        data.check_pairs(rows)
    data.check_pairs([_d(MAT, ["It is closed on Sundays."], lesson="T04")])  # before the amendment: not checked


def test_freeze_report_over_all_files():
    rep = data.freeze_report(_floors("T05") + _floors("T06"))
    assert rep["cannot_tell"] == 12 and rep["shortfalls"] == []
    old = [_d(MAT, ["It is closed on Sundays."], lesson="T04", iid=f"o{i}") for i in range(30)]
    with pytest.raises(data.DataError, match=r"cannot_tell 12 of 58 \(21%\); 3 more needed"):
        data.freeze_report(_floors("T05") + _floors("T06") + old)
    rows = _floors("T05")
    for r in rows:
        r["escalate"] = "no"
    with pytest.raises(data.DataError, match="no ESCALATE yes item in T05"):
        data.freeze_report(rows)
    rep = data.freeze_report(_floors("T05") + [_d(MAT, [], verdict="cannot_tell", lesson="T04", iid="x")])
    assert rep["exempt_without_escalate_yes"] == ["T04"]  # reported, not refused


def test_prompts_are_role_neutral_outside_material_and_statement():
    rows = _floors()
    d = next(r for r in rows if r["id"] == "su2")
    d["turns"][0]["user"] += "\nAs a verifier, check it."
    with pytest.raises(data.DataError, match="role-neutral"):
        data.validate(rows)
    rows = _floors()
    d = next(r for r in rows if r["id"] == "su2")
    d["material"] = MAT + "\nThe tree outside is pruned in May."
    d["turns"][0]["user"] = d["turns"][0]["user"].replace(MAT, d["material"])
    assert data.role_words_in_prompt(d) == []  # words inside the material are content, not framing
    c = copy.deepcopy(GOOD[2])
    assert data.role_words_in_prompt(c) == []  # "checker" names the other party
    c["lesson"] = "T04"
    c["turns"][0]["user"] += " Use a falsifier."
    assert data.role_words_in_prompt(c) == ["falsifier"]
    data.validate(copy.deepcopy(GOOD[:2]) + [c] + copy.deepcopy(GOOD[3:]))  # pre-amendment lesson: not refused


def test_verdict_variant_is_a_prompt_too():
    e = copy.deepcopy(GOOD[4])
    e.update(material=MAT, quiz={"type": "verdict", "verdict": "unsupported", "deciding": ["It is closed on Sundays."],
                                 "escalate": "no"})
    e["variant"] = {"user": "Re-check it as a verifier would.", "key": e["quiz"]}
    assert data.role_words_in_prompt(e) == ["verifier"]
