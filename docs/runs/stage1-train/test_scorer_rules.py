"""The gates package's scorer rules are copies of the frozen score.py; this keeps them identical."""

import inspect

import data  # noqa: F401  (puts the eval directory and the package on the path)
import score
from aspire_si_gates import scorer_rules


def test_scorer_copies_are_identical_to_the_frozen_scorer():
    for name in ("norm", "ident", "core", "lines_of", "deciding_ok"):
        assert inspect.getsource(getattr(scorer_rules, name)) == inspect.getsource(getattr(score, name)), name


MAT = "| Sunday | closed | closed |\nThe library opens at 9am on weekdays.\nIt is closed on Sundays and holidays."


def test_critic_mismatch():
    key = ["It is closed on Sundays and holidays."]
    assert not scorer_rules.critic_mismatch(key, ["It is closed on Sundays and holidays."], MAT)  # same line
    assert not scorer_rules.critic_mismatch(key, ["It is closed on Sundays and holidays"], MAT)  # edge trim
    assert not scorer_rules.critic_mismatch([], [], MAT)  # both NONE
    assert scorer_rules.critic_mismatch(key, ["The library opens at 9am on weekdays."], MAT)  # different evidence
    assert scorer_rules.critic_mismatch(key, [], MAT)  # one side NONE
    row = ["| Sunday | closed | closed |"]
    assert scorer_rules.critic_mismatch(key, row, MAT)
    # coverage, one way: a longer key line covers the critic's quoted fragment only if the fragment contains it
    assert not scorer_rules.deciding_covers(["It is closed on Sundays"], key, MAT)
    assert scorer_rules.deciding_covers(key, ["It is closed on Sundays"], MAT)
