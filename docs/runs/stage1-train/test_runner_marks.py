"""Every runner test that starts the tiny trainer is marked needs_tokenizer, so it skips, never fails, where the
student's tokenizer isn't reachable. One test lost its mark in a merge-conflict resolution (2026-10-15). This file
isn't Windows-only, so CI checks the marks too."""

import ast
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Tests that call run_runner with the tiny trainer but stop before it starts, so they need no tokenizer.
EXEMPT = {"test_refuses_to_start_without_a_live_watchdog_or_after_the_grant": "the runner refuses before train.py "
                                                                              "starts"}


def starts_the_tiny_trainer(f: ast.FunctionDef) -> bool:
    names = {n.id for n in ast.walk(f) if isinstance(n, ast.Name)}
    calls = [n for n in ast.walk(f) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == "run_runner"]
    default_trainer = any(not any(isinstance(a, ast.Constant) and a.value == "--trainer" for a in c.args)
                          for c in calls)
    return "TINY" in names or "plain_run" in names or default_trainer


def test_every_test_that_starts_the_tiny_trainer_is_marked_needs_tokenizer():
    tree = ast.parse((HERE / "test_runner.py").read_text(encoding="utf-8"))
    unmarked = [f.name for f in tree.body if isinstance(f, ast.FunctionDef) and f.name.startswith("test_")
                and starts_the_tiny_trainer(f) and f.name not in EXEMPT
                and not any(isinstance(d, ast.Name) and d.id == "needs_tokenizer" for d in f.decorator_list)]
    assert unmarked == []


def test_an_empty_cache_reads_as_no_tokenizer(tmp_path, monkeypatch):
    import test_runner
    monkeypatch.setenv("HF_HOME", str(tmp_path / "empty-hf"))
    assert test_runner.tokenizer_available() is False
    monkeypatch.delenv("HF_HOME")
    assert test_runner.tokenizer_available() is False  # no HF_HOME at all: skip, never fail
