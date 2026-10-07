"""
Additional unit tests for integrations/code/analysis.py.

External tools (ruff / mypy / bandit) are mocked at the subprocess boundary so
the tests are deterministic and need no installed linters. One test exercises
the real ruff binary and skips when it is not on PATH.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from integrations.code.analysis import (
    CodeAnalyzer,
    CodeFeatures,
    CodeIssue,
    StaticAnalysisResult,
    detect_language,
    extract_code_features,
    parse_code,
    quick_analyze,
)
from integrations.code.config import CodeDimension, Language

ALL_TOOLS = {"ruff": True, "mypy": True, "bandit": True, "semgrep": False}


def make_analyzer(available=None, **flags) -> CodeAnalyzer:
    with patch.object(CodeAnalyzer, "_check_tools", return_value=dict(available or ALL_TOOLS)):
        return CodeAnalyzer(**flags)


def completed(stdout: str = "", returncode: int = 0):
    return SimpleNamespace(stdout=stdout, stderr="", returncode=returncode)


def issue(dimension, severity="warning", line=1):
    return CodeIssue(line=line, column=0, message="m", severity=severity, dimension=dimension)


# ============================================================================
# detect_language
# ============================================================================


class TestDetectLanguage:
    @pytest.mark.parametrize(
        "filename,expected",
        [
            ("a.py", Language.PYTHON),
            ("a.js", Language.JAVASCRIPT),
            ("a.jsx", Language.JAVASCRIPT),
            ("a.ts", Language.TYPESCRIPT),
            ("a.tsx", Language.TYPESCRIPT),
            ("a.rs", Language.RUST),
            ("a.go", Language.GO),
            ("a.java", Language.JAVA),
            ("a.cpp", Language.CPP),
            ("a.cc", Language.CPP),
            ("a.c", Language.C),
            ("a.h", Language.C),
            ("a.cs", Language.CSHARP),
            ("a.rb", Language.RUBY),
            ("a.php", Language.PHP),
            ("a.swift", Language.SWIFT),
            ("a.kt", Language.KOTLIN),
            ("a.scala", Language.SCALA),
            ("DIR/Main.PY", Language.PYTHON),  # case-insensitive, path-aware
        ],
    )
    def test_extension_mapping(self, filename, expected):
        assert detect_language("", filename) == expected

    def test_extension_beats_content(self):
        assert detect_language("def f():\n    import os\n", "x.rs") == Language.RUST

    def test_unknown_extension_falls_back_to_content(self):
        assert detect_language("import os\ndef f(): pass\n", "notes.txt") == Language.PYTHON

    def test_filename_without_extension_falls_back_to_content(self):
        assert detect_language("import os\ndef f(): pass\n", "Makefile") == Language.PYTHON

    def test_typescript_by_interface_keyword(self):
        code = "interface User { name: string }\nconst u: User = { name: 'x' };\nfunction f() {}\n"
        assert detect_language(code) == Language.TYPESCRIPT

    def test_javascript_when_no_interface(self):
        assert detect_language("const x = 1;\nfunction f() { return x; }\n") == Language.JAVASCRIPT

    def test_typescript_requires_colon_space_too(self):
        # "interface " alone (without ": ") is not enough
        assert detect_language("function f() {}\ninterface{}") == Language.JAVASCRIPT

    def test_go_by_content(self):
        assert detect_language("package main\n\nfunc main() {}\n") == Language.GO

    def test_java_by_content(self):
        assert detect_language("public class Main { }") == Language.JAVA
        assert detect_language("class A { private void run() {} }") == Language.JAVA

    def test_def_without_import_is_unknown(self):
        assert detect_language("def f(): pass") == Language.UNKNOWN

    def test_empty_is_unknown(self):
        assert detect_language("") == Language.UNKNOWN

    def test_rust_detected_by_content(self):
        code = "fn main() {\n    let mut x = 5;\n    x += 1;\n}\n"
        assert detect_language(code) == Language.RUST


# ============================================================================
# parse_code
# ============================================================================


class TestParseCode:
    def test_python_structure_extraction(self):
        code = (
            "import os, sys\n"
            "from collections import OrderedDict\n"
            "from . import sibling\n"
            "from .pkg import thing\n"
            "class A:\n    def m(self): pass\n"
            "def f(): pass\n"
        )
        result = parse_code(code, Language.PYTHON)
        assert result["success"] is True
        assert result["language"] == "python"
        assert result["error"] is None
        assert sorted(result["functions"]) == ["f", "m"]
        assert result["classes"] == ["A"]
        # `from . import x` has no module and is skipped; relative `.pkg` keeps its bare module name
        assert sorted(result["imports"]) == ["collections", "os", "pkg", "sys"]

    def test_python_syntax_error_reports_message(self):
        result = parse_code("def f(:\n", Language.PYTHON)
        assert result["success"] is False
        assert result["ast"] is None
        assert result["error"]
        assert "functions" not in result

    def test_other_languages_succeed_with_note(self):
        result = parse_code("fn main() {}", Language.RUST)
        assert result["success"] is True
        assert result["ast"] is None
        assert result["language"] == "rust"
        assert "not implemented for rust" in result["note"]


# ============================================================================
# extract_code_features
# ============================================================================


class TestExtractCodeFeatures:
    def test_empty_code(self):
        f = extract_code_features("", Language.PYTHON)
        assert f.num_lines == 1  # "".split("\n") == [""]
        assert f.num_functions == 0 and f.cyclomatic_complexity == 1
        assert f.has_error_handling is False

    def test_non_python_only_counts_lines(self):
        f = extract_code_features("fn a() {}\nfn b() {}\n", Language.RUST)
        assert f == CodeFeatures(num_lines=3)

    def test_syntax_error_keeps_defaults_but_counts_lines(self):
        f = extract_code_features("def broken(:\n    pass\n", Language.PYTHON)
        assert f.num_lines == 3
        assert f.num_functions == 0
        assert f.ast_node_types == {}

    def test_structure_counts_and_node_types(self):
        code = "import a, b\nfrom c import d\nclass K:\n    def m(self): pass\ndef g(): pass\n"
        f = extract_code_features(code, Language.PYTHON)
        assert f.num_imports == 2  # one Import node + one ImportFrom node
        assert f.num_classes == 1
        assert f.num_functions == 2
        assert f.ast_node_types["FunctionDef"] == 2
        assert f.ast_node_types["ClassDef"] == 1
        assert f.ast_node_types["Module"] == 1

    def test_boolean_operators_add_complexity(self):
        # If(+1) + BoolOp(or)(+1) + nested BoolOp(and)(+1) + base 1
        f = extract_code_features("if a and b or c:\n    pass\n", Language.PYTHON)
        assert f.cyclomatic_complexity == 4

    def test_three_operand_boolop_adds_two(self):
        f = extract_code_features("x = a and b and c\n", Language.PYTHON)
        assert f.cyclomatic_complexity == 3

    def test_loops_and_except_handlers_add_complexity(self):
        code = (
            "for i in x:\n    pass\n"
            "while y:\n    pass\n"
            "try:\n    pass\nexcept A:\n    pass\nexcept B:\n    pass\n"
        )
        f = extract_code_features(code, Language.PYTHON)
        assert f.cyclomatic_complexity == 1 + 1 + 1 + 2
        assert f.has_error_handling is True

    def test_nesting_depth_counts_compound_statements(self):
        code = "while a:\n    for b in c:\n        if d:\n            pass\n"
        assert extract_code_features(code, Language.PYTHON).max_nesting_depth == 3

    def test_nesting_depth_counts_with_and_try(self):
        code = "with a:\n    try:\n        pass\n    finally:\n        pass\n"
        assert extract_code_features(code, Language.PYTHON).max_nesting_depth == 2

    def test_sibling_blocks_do_not_accumulate_depth(self):
        code = "if a:\n    pass\nif b:\n    pass\nif c:\n    pass\n"
        assert extract_code_features(code, Language.PYTHON).max_nesting_depth == 1

    def test_main_guard_both_quote_styles(self):
        assert extract_code_features('if __name__ == "__main__":\n    pass\n', Language.PYTHON).has_main_guard
        assert extract_code_features("if __name__ == '__main__':\n    pass\n", Language.PYTHON).has_main_guard
        assert not extract_code_features("x = 1\n", Language.PYTHON).has_main_guard

    def test_type_hints_need_return_annotation(self):
        assert extract_code_features("def f() -> int:\n    return 1\n", Language.PYTHON).has_type_hints
        assert not extract_code_features("def f():\n    return 1\n", Language.PYTHON).has_type_hints

    def test_async_functions_are_counted(self):
        code = "async def fetch():\n    return 1\n\ndef sync():\n    return 2\n"
        assert extract_code_features(code, Language.PYTHON).num_functions == 2


# ============================================================================
# CodeAnalyzer._check_tools
# ============================================================================


class TestCheckTools:
    def test_all_tools_available(self):
        with patch("integrations.code.analysis.subprocess.run", return_value=completed(returncode=0)) as run:
            analyzer = CodeAnalyzer()
        assert analyzer._available_tools == {"ruff": True, "mypy": True, "bandit": True, "semgrep": True}
        assert [c.args[0] for c in run.call_args_list] == [
            ["ruff", "--version"], ["mypy", "--version"], ["bandit", "--version"], ["semgrep", "--version"],
        ]
        assert all(c.kwargs["timeout"] == 5 for c in run.call_args_list)

    def test_nonzero_exit_means_unavailable(self):
        with patch("integrations.code.analysis.subprocess.run", return_value=completed(returncode=2)):
            analyzer = CodeAnalyzer()
        assert not any(analyzer._available_tools.values())

    def test_missing_binary_and_timeout_mean_unavailable(self):
        def fake_run(cmd, **kwargs):
            if cmd[0] == "ruff":
                raise FileNotFoundError(cmd[0])
            if cmd[0] == "mypy":
                raise subprocess.TimeoutExpired(cmd, 5)
            return completed(returncode=0)

        with patch("integrations.code.analysis.subprocess.run", side_effect=fake_run):
            analyzer = CodeAnalyzer()
        assert analyzer._available_tools == {"ruff": False, "mypy": False, "bandit": True, "semgrep": True}

    def test_flags_are_stored(self):
        analyzer = make_analyzer(use_ruff=False, use_mypy=False, use_bandit=True, use_semgrep=True)
        assert (analyzer.use_ruff, analyzer.use_mypy, analyzer.use_bandit, analyzer.use_semgrep) == (
            False, False, True, True,
        )


# ============================================================================
# CodeAnalyzer.analyze
# ============================================================================


class TestAnalyze:
    def test_non_python_short_circuits(self):
        analyzer = make_analyzer()
        with patch.object(analyzer, "_run_ruff") as ruff:
            result = analyzer.analyze("fn main() {}", Language.RUST)
        ruff.assert_not_called()
        assert result.issues == []
        assert result.tool_outputs == {"note": "Limited analysis for rust"}
        assert result.metrics == {}

    def test_score_arithmetic_per_tool(self):
        analyzer = make_analyzer()
        ruff = [issue(CodeDimension.STYLE)] * 3
        mypy = [issue(CodeDimension.CORRECTNESS, "error")] * 4
        bandit = [
            issue(CodeDimension.SECURITY, "error"),
            issue(CodeDimension.SECURITY, "warning"),
            issue(CodeDimension.SECURITY, "info"),
        ]
        with patch.object(analyzer, "_run_ruff", return_value=ruff), patch.object(
            analyzer, "_run_mypy", return_value=mypy
        ), patch.object(analyzer, "_run_bandit", return_value=bandit):
            result = analyzer.analyze("x = 1\n")

        assert result.style_score == pytest.approx(10 - 3 * 0.5)
        assert result.correctness_score == pytest.approx(10 - 4 * 0.5)
        assert result.security_score == pytest.approx(10 - 1 * 2.0 - 3 * 0.5)
        assert result.complexity_score == 10.0
        assert len(result.issues) == 10
        assert result.metrics == {
            "lines": 2, "functions": 0, "classes": 0, "complexity": 1, "max_depth": 0,
        }

    def test_scores_are_floored_at_zero(self):
        analyzer = make_analyzer()
        with patch.object(analyzer, "_run_ruff", return_value=[issue(CodeDimension.STYLE)] * 50), patch.object(
            analyzer, "_run_mypy", return_value=[issue(CodeDimension.CORRECTNESS)] * 50
        ), patch.object(analyzer, "_run_bandit", return_value=[issue(CodeDimension.SECURITY, "error")] * 50):
            result = analyzer.analyze("x = 1\n")
        assert result.style_score == 0
        assert result.correctness_score == 0
        assert result.security_score == 0

    def test_only_style_issues_reduce_style_score(self):
        analyzer = make_analyzer()
        mixed = [issue(CodeDimension.STYLE), issue(CodeDimension.SECURITY)]
        with patch.object(analyzer, "_run_ruff", return_value=mixed), patch.object(
            analyzer, "_run_mypy", return_value=[]
        ), patch.object(analyzer, "_run_bandit", return_value=[]):
            result = analyzer.analyze("x = 1\n")
        assert result.style_score == pytest.approx(9.5)

    def test_disabled_or_unavailable_tools_are_not_run(self):
        analyzer = make_analyzer(
            available={"ruff": True, "mypy": False, "bandit": True}, use_ruff=False, use_mypy=True, use_bandit=True
        )
        with patch.object(analyzer, "_run_ruff") as ruff, patch.object(analyzer, "_run_mypy") as mypy, patch.object(
            analyzer, "_run_bandit", return_value=[]
        ) as bandit:
            analyzer.analyze("x = 1\n")
        ruff.assert_not_called()  # flag off even though installed
        mypy.assert_not_called()  # flag on but not installed
        bandit.assert_called_once()

    def test_complexity_and_nesting_penalties(self):
        analyzer = make_analyzer(use_ruff=False, use_mypy=False, use_bandit=False)
        flat = "".join(f"if x{i}:\n    pass\n" for i in range(24))  # complexity 25
        assert analyzer.analyze(flat).complexity_score == pytest.approx(10 - (25 - 20) * 0.2)

        deep = "if a:\n if b:\n  if c:\n   if d:\n    if e:\n     if f:\n      pass\n"  # depth 6
        assert analyzer.analyze(deep).complexity_score == pytest.approx(10 - (6 - 4) * 1.0)

    def test_complexity_score_floor(self):
        analyzer = make_analyzer(use_ruff=False, use_mypy=False, use_bandit=False)
        code = "".join(" " * i + "if a:\n" for i in range(15)) + " " * 15 + "pass\n"
        assert analyzer.analyze(code).complexity_score == 0

    def test_temp_file_written_with_code_and_removed(self):
        analyzer = make_analyzer(use_mypy=False, use_bandit=False)
        seen = {}

        def fake_ruff(path):
            seen["path"] = path
            seen["text"] = Path(path).read_text()
            return []

        with patch.object(analyzer, "_run_ruff", side_effect=fake_ruff):
            analyzer.analyze("answer = 42\n")
        assert seen["path"].endswith(".py")
        assert seen["text"] == "answer = 42\n"
        assert not Path(seen["path"]).exists()

    def test_temp_file_removed_even_when_tool_raises(self):
        analyzer = make_analyzer(use_mypy=False, use_bandit=False)
        seen = {}

        def boom(path):
            seen["path"] = path
            raise RuntimeError("tool crashed")

        with patch.object(analyzer, "_run_ruff", side_effect=boom):
            with pytest.raises(RuntimeError, match="tool crashed"):
                analyzer.analyze("x = 1\n")
        assert not Path(seen["path"]).exists()

    @pytest.mark.skipif(shutil.which("ruff") is None, reason="ruff not on PATH")
    def test_real_ruff_flags_unused_import(self):
        analyzer = CodeAnalyzer(use_ruff=True, use_mypy=False, use_bandit=False)
        if not analyzer._available_tools["ruff"]:
            pytest.skip("ruff not runnable")
        result = analyzer.analyze("import os\n")
        assert any(i.rule_id == "F401" and i.dimension == CodeDimension.STYLE for i in result.issues)
        assert result.style_score < 10


# ============================================================================
# _run_ruff / _run_mypy / _run_bandit
# ============================================================================


class TestRunRuff:
    def test_parses_json_output(self):
        payload = [
            {"code": "F401", "message": "unused import", "location": {"row": 3, "column": 8}},
            {"message": "no location"},
        ]
        analyzer = make_analyzer()
        with patch("integrations.code.analysis.subprocess.run", return_value=completed(json.dumps(payload))) as run:
            issues = analyzer._run_ruff("f.py")

        cmd = run.call_args.args[0]
        assert cmd == ["ruff", "check", "--output-format=json", "f.py"]
        assert run.call_args.kwargs["timeout"] == 30
        assert len(issues) == 2
        first = issues[0]
        assert (first.line, first.column, first.message, first.rule_id) == (3, 8, "unused import", "F401")
        assert first.severity == "warning" and first.dimension == CodeDimension.STYLE
        assert (issues[1].line, issues[1].column, issues[1].rule_id) == (0, 0, "")

    def test_empty_stdout_means_no_issues(self):
        with patch("integrations.code.analysis.subprocess.run", return_value=completed("")):
            assert make_analyzer()._run_ruff("f.py") == []

    def test_invalid_json_is_swallowed(self):
        with patch("integrations.code.analysis.subprocess.run", return_value=completed("not json")):
            assert make_analyzer()._run_ruff("f.py") == []

    @pytest.mark.parametrize("exc", [subprocess.TimeoutExpired("ruff", 30), FileNotFoundError("ruff")])
    def test_tool_failures_are_swallowed(self, exc):
        with patch("integrations.code.analysis.subprocess.run", side_effect=exc):
            assert make_analyzer()._run_ruff("f.py") == []


class TestRunMypy:
    def test_parses_error_and_note_lines(self):
        out = (
            "f.py:3:5: error: Incompatible types in assignment\n"
            "f.py:9:1: note: See docs\n"
            "this line does not match\n"
            "\n"
        )
        with patch("integrations.code.analysis.subprocess.run", return_value=completed(out)) as run:
            issues = make_analyzer()._run_mypy("f.py")
        assert run.call_args.args[0] == ["mypy", "--no-error-summary", "--show-column-numbers", "f.py"]
        assert run.call_args.kwargs["timeout"] == 60
        assert [(i.line, i.column, i.severity, i.message) for i in issues] == [
            (3, 5, "error", "Incompatible types in assignment"),
            (9, 1, "warning", "See docs"),
        ]
        assert all(i.dimension == CodeDimension.CORRECTNESS and i.rule_id == "mypy" for i in issues)

    def test_windows_drive_letter_paths_parse(self):
        out = "C:\\tmp\\f.py:2:4: error: bad\n"
        with patch("integrations.code.analysis.subprocess.run", return_value=completed(out)):
            issues = make_analyzer()._run_mypy("f.py")
        assert [(i.line, i.column) for i in issues] == [(2, 4)]

    @pytest.mark.parametrize("exc", [subprocess.TimeoutExpired("mypy", 60), FileNotFoundError("mypy")])
    def test_tool_failures_are_swallowed(self, exc):
        with patch("integrations.code.analysis.subprocess.run", side_effect=exc):
            assert make_analyzer()._run_mypy("f.py") == []


class TestRunBandit:
    def test_parses_results_and_maps_severity(self):
        payload = {
            "results": [
                {"line_number": 4, "issue_text": "use of eval", "issue_severity": "HIGH", "test_id": "B307"},
                {"line_number": 5, "issue_text": "md5", "issue_severity": "MEDIUM", "test_id": "B303"},
                {"line_number": 6, "issue_text": "assert", "issue_severity": "LOW", "test_id": "B101"},
                {"issue_text": "weird", "issue_severity": "???"},
            ]
        }
        with patch("integrations.code.analysis.subprocess.run", return_value=completed(json.dumps(payload))) as run:
            issues = make_analyzer()._run_bandit("f.py")
        assert run.call_args.args[0] == ["bandit", "-f", "json", "-q", "f.py"]
        assert [(i.line, i.severity, i.rule_id) for i in issues] == [
            (4, "error", "B307"), (5, "warning", "B303"), (6, "info", "B101"), (0, "warning", ""),
        ]
        assert issues[0].message == "use of eval"
        assert all(i.dimension == CodeDimension.SECURITY and i.column == 0 for i in issues)

    def test_missing_results_key(self):
        with patch("integrations.code.analysis.subprocess.run", return_value=completed("{}")):
            assert make_analyzer()._run_bandit("f.py") == []

    def test_empty_and_invalid_output(self):
        analyzer = make_analyzer()
        with patch("integrations.code.analysis.subprocess.run", return_value=completed("")):
            assert analyzer._run_bandit("f.py") == []
        with patch("integrations.code.analysis.subprocess.run", return_value=completed("<html>")):
            assert analyzer._run_bandit("f.py") == []

    @pytest.mark.parametrize("exc", [subprocess.TimeoutExpired("bandit", 30), FileNotFoundError("bandit")])
    def test_tool_failures_are_swallowed(self, exc):
        with patch("integrations.code.analysis.subprocess.run", side_effect=exc):
            assert make_analyzer()._run_bandit("f.py") == []


# ============================================================================
# quick_analyze
# ============================================================================


class TestQuickAnalyze:
    def test_baseline_scores(self):
        r = quick_analyze("x = 1\n", Language.PYTHON)
        assert r["scores"] == {"correctness": 10.0, "style": 8.0, "complexity": 10.0}
        assert r["overall"] == pytest.approx((10 + 8 + 10) / 3)
        assert r["parsed"] is True
        assert r["language"] == "python"
        assert isinstance(r["features"], CodeFeatures)

    def test_complexity_penalty_above_ten(self):
        code = "".join(f"if x{i}:\n    pass\n" for i in range(14))  # complexity 15
        r = quick_analyze(code, Language.PYTHON)
        assert r["scores"]["complexity"] == pytest.approx(10 - (15 - 10) * 0.3)

    def test_nesting_penalty_above_three(self):
        code = "if a:\n if b:\n  if c:\n   if d:\n    if e:\n     pass\n"  # depth 5
        r = quick_analyze(code, Language.PYTHON)
        assert r["scores"]["complexity"] == pytest.approx(10 - (5 - 3) * 0.5)

    def test_penalties_combine_and_floor_at_zero(self):
        code = "".join(" " * i + "if a:\n" for i in range(30)) + " " * 30 + "pass\n"
        r = quick_analyze(code, Language.PYTHON)
        assert r["scores"]["complexity"] == 0

    def test_good_practice_bonuses(self):
        code = (
            "def f(x: int) -> int:\n"
            '    """Doc."""\n'
            "    try:\n        return x\n    except ValueError:\n        return 0\n"
        )
        r = quick_analyze(code, Language.PYTHON)
        assert r["scores"]["style"] == pytest.approx(9.0)  # 8 + docstring + type hints
        assert r["scores"]["correctness"] == 10.0  # 10 + 0.5 capped to 10

    def test_syntax_error_lowers_correctness(self):
        r = quick_analyze("def f(:\n", Language.PYTHON)
        assert r["parsed"] is False
        assert r["scores"]["correctness"] == 3.0

    def test_language_autodetected_when_omitted(self):
        r = quick_analyze("import os\n\ndef f():\n    return os.name\n")
        assert r["language"] == "python"

    def test_non_python_language_counts_as_parsed(self):
        r = quick_analyze("fn main() {}", Language.RUST)
        assert r["parsed"] is True
        assert r["scores"]["correctness"] == 10.0


# ============================================================================
# StaticAnalysisResult
# ============================================================================


class TestStaticAnalysisResultDefaults:
    def test_defaults_are_independent_and_clean(self):
        a, b = StaticAnalysisResult(), StaticAnalysisResult()
        a.issues.append(issue(CodeDimension.STYLE, "error"))
        assert b.issues == []
        assert (b.correctness_score, b.style_score, b.security_score, b.complexity_score) == (10.0,) * 4
        assert a.has_errors and a.error_count == 1 and a.warning_count == 0

    def test_mixed_severities_counted(self):
        r = StaticAnalysisResult(
            issues=[
                issue(CodeDimension.STYLE, "warning"),
                issue(CodeDimension.STYLE, "info"),
                issue(CodeDimension.STYLE, "hint"),
                issue(CodeDimension.STYLE, "error"),
            ]
        )
        assert (r.error_count, r.warning_count) == (1, 1)

