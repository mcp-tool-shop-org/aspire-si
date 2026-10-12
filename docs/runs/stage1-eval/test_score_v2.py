"""Tests for score v2 (score_v2.py): the parsing repair and nothing more."""

import score
import score_v2

MAT = ("- **Your RunPod API key**, read from the `RUNPOD_API_KEY` environment variable. offrig\n"
       "never writes it to disk.\n"
       "| `PORT` | `3456` | Server port |\n"
       "A session is capped at six requests and ten tool calls.")
KEY = ["A session is capped at six requests and ten tool calls."]


def both(deciding, key=KEY, verdict="unsupported"):
    pinned = score.deciding_ok(score.lines_of(deciding), key, [], MAT, verdict, score.norm)
    v2 = score_v2.deciding_ok_v2(deciding, key, [], MAT, verdict, score.norm)
    return pinned, v2


def test_lone_edge_separator_repaired():
    assert both("A session is capped at six requests and ten tool calls. |") == (False, True)
    assert both("| offrig never writes it to disk.", key=["offrig never writes it to disk."]) == (False, True)


def test_none_element_dropped_beside_real_lines():
    assert both("A session is capped at six requests and ten tool calls. | NONE") == (False, True)
    assert both("NONE", verdict="cannot_tell") == (True, True)  # a lone NONE is still NONE


def test_dropped_backticks_still_fail():
    key = ["Your RunPod API key, read from the `RUNPOD_API_KEY` environment variable."]
    assert both("Your RunPod API key, read from the RUNPOD_API_KEY environment variable. |", key=key) == (False, False)


def test_quoted_table_row_unchanged():
    key = ["| `PORT` | `3456` | Server port |"]
    assert both("| `PORT` | `3456` | Server port |", key=key) == (True, True)


def test_v2_never_fails_what_pinned_passes():
    for d in ["A session is capped at six requests and ten tool calls.", "| `PORT` | `3456` | Server port |"]:
        for key in (KEY, ["| `PORT` | `3456` | Server port |"]):
            p, v = both(d, key=key)
            assert v or not p
