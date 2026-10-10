"""Tests for the secondary "found the evidence" score (score_found.py)."""

import score_found as F

MAT = ("## [2.1.0] - 2026-09-02\n"
       "- **Seeded runs:** every session prints its seed; `--seed <n>` replays it byte-for-byte\n"
       "| **Dockhand** | Crosswind Isle | Cheap crew | Lowest starting capital |\n"
       "A session is capped at six requests and ten tool calls.")
ROW = "| **Dockhand** | Crosswind Isle | Cheap crew | Lowest starting capital |"
BULLET = "- **Seeded runs:** every session prints its seed; `--seed <n>` replays it byte-for-byte"
HEAD = "## [2.1.0] - 2026-09-02"


def ok(deciding, key, verdict="unsupported", category="buried exception"):
    return F.found_ok_turn(deciding, key, [], MAT, verdict, category)


def test_dropped_backticks_markup_and_case_pass():
    assert ok("Seeded runs: every session prints its seed; --seed <n> replays it byte-for-byte", [BULLET])
    assert ok("A SESSION is capped at six requests and ten tool calls", ["A session is capped at six requests "
                                                                        "and ten tool calls."])


def test_cell_fragment_fails_whole_row_passes():
    assert not ok("Dockhand | Crosswind Isle", [ROW])  # two short cells: under half the row
    assert ok(ROW, [ROW])
    assert ok("Dockhand | Crosswind Isle | Cheap crew | Lowest starting capital", [ROW])


def test_half_coverage_inside_a_key_line():
    assert ok("every session prints its seed; --seed <n> replays it byte-for-byte", [BULLET])
    assert not ok("every session prints its seed", [BULLET])


def test_heading_optional_except_in_the_version_category():
    assert ok(BULLET, [HEAD, BULLET])
    assert not ok(BULLET, [HEAD, BULLET], category=F.VERSION_CATEGORY)
    assert ok(HEAD + " | " + BULLET, [HEAD, BULLET], category=F.VERSION_CATEGORY)


def test_paraphrase_and_cannot_tell_still_fail():
    assert not ok("A session allows at most six requests", ["A session is capped at six requests and ten tool calls."])
    assert not ok("A session is capped at six requests and ten tool calls.", [], verdict="cannot_tell")
    assert ok("NONE", [], verdict="cannot_tell")


def test_v2_parsing_repair_applies():
    assert ok("A session is capped at six requests and ten tool calls. | NONE",
              ["A session is capped at six requests and ten tool calls."])
    assert ok("A session is capped at six requests and ten tool calls. |",
              ["A session is capped at six requests and ten tool calls."])
