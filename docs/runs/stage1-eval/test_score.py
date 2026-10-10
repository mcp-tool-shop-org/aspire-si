"""Tests for the pinned correctness scorer (score.py)."""

import score

MAT = ("- **Your RunPod API key**, read from the `RUNPOD_API_KEY` environment variable. offrig\n"
       "never writes it to disk.\n"
       "| `PORT` | `3456` | Server port |\n"
       "A session is capped at six requests and ten tool calls.")
KEY = ["A session is capped at six requests and ten tool calls."]


def ok(given, key=KEY, alts=(), verdict="unsupported"):
    return score.deciding_ok(given, key, list(alts), MAT, verdict, score.norm)


def test_exact_and_edge_trims():
    assert ok(["A session is capped at six requests and ten tool calls."])
    assert ok(["A session is capped at six requests and ten tool calls"])  # trailing punctuation
    assert ok(["Your RunPod API key, read from the `RUNPOD_API_KEY` environment variable."],
              key=["- **Your RunPod API key**, read from the `RUNPOD_API_KEY` environment variable."])


def test_short_fragment_covers_nothing():
    assert not ok(["the model"])
    assert not ok(["A session is capped"])  # a substring of the key line, not the key line


def test_no_case_folding_and_backticks_count():
    assert not ok(["a session is capped at six requests and ten tool calls."])
    assert not ok(["Your RunPod API key, read from the RUNPOD_API_KEY environment variable."],
                  key=["Your RunPod API key, read from the `RUNPOD_API_KEY` environment variable."])


def test_table_row_through_joined_string():
    assert ok(["| `PORT`", "`3456`", "Server port |"], key=["| `PORT` | `3456` | Server port |"])


def test_given_not_in_material_fails_even_if_it_covers():
    assert not ok(["A session is capped at six requests and ten tool calls. Always."])


def test_cannot_tell_needs_none():
    assert ok([], key=[], verdict="cannot_tell")
    assert not ok(["A session is capped at six requests and ten tool calls."], key=[], verdict="cannot_tell")


def test_also_sufficient_set():
    assert ok(["never writes it to disk."], key=["not in the material at all"],
              alts=[["never writes it to disk."]])
