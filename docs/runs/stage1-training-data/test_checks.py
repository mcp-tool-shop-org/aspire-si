"""checks.py's quote-balance gate: curly double quotes pair up; the curly apostrophe is never counted.

Run from this directory: python -m pytest test_checks.py
"""
import checks


def test_a_curly_quoted_phrase_is_balanced():
    assert checks.quote_balance_errors("The note says “x” and nothing else.") == []


def test_a_curly_apostrophe_is_not_a_quote():
    assert checks.quote_balance_errors("it’s open, and “it’s shut” is the claim") == []


def test_an_opening_curly_quote_without_its_close_is_unbalanced():
    errs = checks.quote_balance_errors("The note says “x and nothing else.")
    assert errs == ["unbalanced curly double quotes (1 “ against 0 ”)"]
    assert checks.quote_balance_errors("“a” “b”") == []


def test_the_other_marks_are_unchanged():
    assert checks.quote_balance_errors('say "x') == ["unbalanced straight double quote"]
    assert checks.quote_balance_errors("run `npm test") == ["unbalanced backtick"]
    assert checks.quote_balance_errors("a **bold") == ["unbalanced **"]
