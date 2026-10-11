"""The gates package's three-line parser is a copy of the frozen harness's; this keeps them identical."""

import inspect

import data  # noqa: F401  (puts the eval directory and the package on the path)
import harness
from aspire_si_gates import answer_format


def test_parser_copy_is_identical_to_the_harness():
    assert inspect.getsource(answer_format.parse_answer) == inspect.getsource(harness.parse_answer)


def test_parser_gives_the_same_answers():
    for text in ["VERDICT: supported\nDECIDING: a | b\nESCALATE: no", "verdict: **Unsupported**.", "no lines", ""]:
        assert answer_format.parse_answer(text) == harness.parse_answer(text)
