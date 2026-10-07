"""
Additional unit tests for integrations/code/code_teacher.py.

Static-analysis tools are never launched: ``CodeAnalyzer._check_tools`` is
patched for the whole module and individual tests inject canned
``StaticAnalysisResult`` objects. LLM clients are replaced by fakes.
"""

from __future__ import annotations

import math
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from integrations.code.analysis import CodeAnalyzer, CodeFeatures, CodeIssue, StaticAnalysisResult
from integrations.code.code_teacher import (
    ArchitectureReviewer,
    BaseCodeTeacher,
    CodeCritique,
    CodeSample,
    CodeTeacher,
    CorrectnessChecker,
    DocumentationCritic,
    LLMCodeTeacher,
    PerformanceAnalyst,
    SecurityAuditor,
    StyleGuide,
)
from integrations.code.config import CodeDimension, Language


@pytest.fixture(autouse=True)
def no_external_tools():
    with patch.object(CodeAnalyzer, "_check_tools", return_value={}):
        yield


def py(code: str) -> CodeSample:
    return CodeSample(code=code, language=Language.PYTHON)


def analysis_with(*issues: CodeIssue) -> StaticAnalysisResult:
    return StaticAnalysisResult(issues=list(issues))


def issue(dimension, severity="warning", line=1, message="msg") -> CodeIssue:
    return CodeIssue(line=line, column=0, message=message, severity=severity, dimension=dimension)


def with_analysis(teacher, *issues: CodeIssue):
    teacher.analyzer = MagicMock()
    teacher.analyzer.analyze.return_value = analysis_with(*issues)
    return teacher


def function_of_lines(name: str, body_lines: int) -> str:
    return f"def {name}():\n" + "    x = 1\n" * body_lines


# ============================================================================
# BaseCodeTeacher
# ============================================================================


class TestBaseCodeTeacher:
    def test_cannot_instantiate_abstract_base(self):
        with pytest.raises(TypeError):
            BaseCodeTeacher("n", "d", [CodeDimension.STYLE])

    def test_repr_uses_class_and_name(self):
        assert repr(ArchitectureReviewer()) == "ArchitectureReviewer(name='Architecture Reviewer')"

    def test_metadata(self):
        t = SecurityAuditor(use_static_analysis=False)
        assert (t.name, t.focus_dimensions) == ("Security Auditor", [CodeDimension.SECURITY])
        assert PerformanceAnalyst().focus_dimensions == [CodeDimension.PERFORMANCE]
        assert DocumentationCritic().focus_dimensions == [CodeDimension.DOCUMENTATION]
        assert StyleGuide(False).focus_dimensions == [CodeDimension.STYLE, CodeDimension.MAINTAINABILITY]


# ============================================================================
# CorrectnessChecker
# ============================================================================


class TestCorrectnessChecker:
    @pytest.fixture
    def checker(self):
        return CorrectnessChecker(use_static_analysis=False)

    def test_static_analysis_flag_controls_analyzer_creation(self):
        assert not hasattr(CorrectnessChecker(use_static_analysis=False), "analyzer")
        analyzer = CorrectnessChecker(use_static_analysis=True).analyzer
        assert (analyzer.use_ruff, analyzer.use_mypy, analyzer.use_bandit) == (True, True, False)

    def test_syntax_error_short_circuits(self, checker):
        c = checker.critique(py("def broken(:\n"))
        assert c.overall_score == 2.0
        assert c.dimension_scores == {CodeDimension.CORRECTNESS: 2.0}
        assert c.reasoning == "Code has syntax errors and cannot be parsed."
        assert c.weaknesses[0].startswith("Syntax error: ")
        assert c.suggestions == ["Fix syntax errors before other improvements"]
        assert c.teacher_name == "Correctness Checker"
        assert c.language == Language.PYTHON

    def test_clean_code_with_error_handling_scores_ten(self, checker):
        code = "try:\n    y = int('3')\nexcept ValueError:\n    y = 0\nprint(y)\n"
        c = checker.critique(py(code))
        assert c.overall_score == 10.0
        assert c.reasoning == "Code appears correct with no significant issues. "
        assert c.weaknesses == [] and c.suggestions == []

    def test_missing_error_handling_costs_half_point(self, checker):
        c = checker.critique(py("print(1)\n"))
        assert c.overall_score == 9.5
        assert c.suggestions == ["Consider adding error handling for robustness"]

    def test_while_true_without_break(self, checker):
        c = checker.critique(py("while True:\n    pass\n"))
        assert c.overall_score == pytest.approx(10 - 0.5 - 1.0)
        assert "Potential infinite loop: while True without break" in c.weaknesses

    def test_while_true_with_break_is_fine(self, checker):
        c = checker.critique(py("while True:\n    break\n"))
        assert c.overall_score == 9.5
        assert c.weaknesses == []

    def test_division_by_zero_literal(self, checker):
        c = checker.critique(py("def f(a):\n    return a / 0\n"))
        assert c.overall_score == pytest.approx(10 - 0.5 - 2.0)
        assert c.weaknesses == ["Possible division by zero"]
        assert c.reasoning == "Code has some correctness concerns that should be addressed. Main issues: Possible division by zero."

    def test_unused_variables_penalised_underscore_ignored(self, checker):
        code = "def f():\n    a = 1\n    b = 2\n    _ = 3\n    return 4\n"
        c = checker.critique(py(code))
        assert c.overall_score == pytest.approx(10 - 0.5 - 0.6)
        (suggestion,) = [s for s in c.suggestions if s.startswith("Unused variables")]
        assert set(suggestion.split(": ")[1].split(", ")) == {"a", "b"}

    def test_used_variables_are_not_flagged(self, checker):
        c = checker.critique(py("def f():\n    a = 1\n    return a\n"))
        assert not any("Unused" in s for s in c.suggestions)

    def test_unused_variable_list_truncated_to_three(self, checker):
        code = "def f():\n    a = 1\n    b = 2\n    c = 3\n    d = 4\n    return 0\n"
        c = checker.critique(py(code))
        (suggestion,) = [s for s in c.suggestions if s.startswith("Unused variables")]
        assert len(suggestion.split(": ")[1].split(", ")) == 3
        assert c.overall_score == pytest.approx(10 - 0.5 - 1.2)

    def test_non_python_skips_python_only_checks(self, checker):
        sample = CodeSample(code="let a = 1;\nlet b = 2;\n", language=Language.JAVASCRIPT)
        c = checker.critique(sample)
        # No AST available: no unused-variable check; no static analysis; error handling feature is False
        assert c.overall_score == 9.5
        assert c.language == Language.JAVASCRIPT

    def test_static_analysis_errors_and_warnings(self):
        checker = with_analysis(
            CorrectnessChecker(),
            issue(CodeDimension.CORRECTNESS, "error", line=3, message="bad type"),
            issue(CodeDimension.CORRECTNESS, "warning", line=5, message="meh"),
            issue(CodeDimension.STYLE, "error", line=7, message="ignored: wrong dimension"),
        )
        code = "try:\n    print(1)\nexcept Exception:\n    pass\n"
        c = checker.critique(py(code))
        assert c.overall_score == pytest.approx(10 - 1.5 - 0.5)
        assert c.weaknesses == ["Line 3: bad type"]
        assert c.line_comments == {3: "bad type", 5: "meh"}
        assert "No static analysis errors detected" not in c.strengths

    def test_static_analysis_clean_adds_strength(self):
        checker = with_analysis(CorrectnessChecker())
        c = checker.critique(py("try:\n    print(1)\nexcept Exception:\n    pass\n"))
        assert "No static analysis errors detected" in c.strengths
        checker.analyzer.analyze.assert_called_once()

    def test_static_analysis_not_used_for_other_languages(self):
        checker = with_analysis(CorrectnessChecker())
        checker.critique(CodeSample(code="let a = 1;", language=Language.JAVASCRIPT))
        checker.analyzer.analyze.assert_not_called()

    def test_score_clamped_at_zero_with_critical_reasoning(self):
        errors = [issue(CodeDimension.CORRECTNESS, "error", line=i, message=f"e{i}") for i in range(1, 11)]
        c = with_analysis(CorrectnessChecker(), *errors).critique(py("print(1)\n"))
        assert c.overall_score == 0.0
        assert c.reasoning.startswith("Code has significant correctness issues. Main issues: Line 1: e1; Line 2: e2.")

    @pytest.mark.parametrize(
        "n_errors,prefix",
        [(0, "Code appears correct"), (2, "Code has some correctness concerns"), (6, "Code has significant")],
    )
    def test_reasoning_bands(self, n_errors, prefix):
        errors = [issue(CodeDimension.CORRECTNESS, "error", line=i + 1) for i in range(n_errors)]
        code = "try:\n    print(1)\nexcept Exception:\n    pass\n"
        c = with_analysis(CorrectnessChecker(), *errors).critique(py(code))
        assert c.reasoning.startswith(prefix)

    @pytest.mark.parametrize("divisor", ["0", "0.0", "0.00"])
    def test_zero_divisor_spellings_are_flagged(self, checker, divisor):
        c = checker.critique(py(f"def f(a):\n    return a / {divisor}\n"))
        assert "Possible division by zero" in c.weaknesses

    def test_division_by_fraction_is_not_division_by_zero(self, checker):
        c = checker.critique(py("def f(a):\n    return a / 0.5\n"))
        assert "Possible division by zero" not in c.weaknesses


# ============================================================================
# StyleGuide
# ============================================================================

GOOD_STYLE = 'def add(a: int, b: int) -> int:\n    """Add."""\n    return a + b\n'


class TestStyleGuide:
    @pytest.fixture
    def guide(self):
        return StyleGuide(use_static_analysis=False)

    def test_static_analysis_flag(self):
        assert not hasattr(StyleGuide(False), "analyzer")
        a = StyleGuide(True).analyzer
        assert (a.use_ruff, a.use_mypy, a.use_bandit) == (True, False, False)

    def test_clean_code_scores_ten(self, guide):
        c = guide.critique(py(GOOD_STYLE))
        assert c.overall_score == 10.0
        assert c.strengths == ["Has docstrings", "Uses type hints"]
        assert c.dimension_scores == {CodeDimension.STYLE: 10.0, CodeDimension.MAINTAINABILITY: 9.0}
        assert c.reasoning == "Code follows good style conventions and is readable. "

    def test_missing_docstrings_and_hints(self, guide):
        c = guide.critique(py("def add(a, b):\n    return a + b\n"))
        assert c.overall_score == pytest.approx(10 - 1.0 - 0.5)
        assert "Add docstrings to functions and classes" in c.suggestions
        assert "Consider adding type hints for clarity" in c.suggestions

    def test_camel_case_flagged_with_names(self, guide):
        c = guide.critique(py(GOOD_STYLE + "\nmyValue = 1\notherName = 2\n"))
        (w,) = [w for w in c.weaknesses if "snake_case" in w]
        assert "myValue" in w and "otherName" in w
        assert c.overall_score == pytest.approx(10 - 0.5)

    def test_all_caps_names_other_than_whitelist_penalised(self, guide):
        base = guide.critique(py(GOOD_STYLE + "\nURL = 'x'\n")).overall_score
        assert base == 10.0  # whitelisted acronym
        flagged = guide.critique(py(GOOD_STYLE + "\nFOO = 'x'\n")).overall_score
        assert flagged == pytest.approx(10 - 0.2)

    def test_long_lines_penalty_capped_at_three_lines(self, guide):
        long_line = "# " + "x" * 120 + "\n"
        c = guide.critique(py(GOOD_STYLE + long_line * 5))
        assert c.overall_score == pytest.approx(10 - 0.9)
        (s,) = [s for s in c.suggestions if s.startswith("Lines too long")]
        assert s == "Lines too long (>100 chars): [4, 5, 6]"

    def test_single_long_line_penalty(self, guide):
        c = guide.critique(py(GOOD_STYLE + "# " + "x" * 120 + "\n"))
        assert c.overall_score == pytest.approx(10 - 0.3)

    def test_magic_numbers(self, guide):
        c = guide.critique(py(GOOD_STYLE + "\nlimit = 42\n"))
        assert c.overall_score == pytest.approx(10 - 0.3)
        assert "Consider using named constants instead of magic numbers" in c.suggestions

    @pytest.mark.parametrize("n", ["10", "100", "1000"])
    def test_common_round_numbers_are_not_magic(self, guide, n):
        c = guide.critique(py(GOOD_STYLE + f"\nlimit = {n}\n"))
        assert c.overall_score == 10.0

    def test_static_analysis_style_issues_deducted_with_cap(self):
        few = with_analysis(StyleGuide(), *[issue(CodeDimension.STYLE)] * 2).critique(py(GOOD_STYLE))
        assert few.overall_score == pytest.approx(10 - 0.6)
        assert few.weaknesses == ["2 style issues detected"]
        many = with_analysis(StyleGuide(), *[issue(CodeDimension.STYLE)] * 30).critique(py(GOOD_STYLE))
        assert many.overall_score == pytest.approx(10 - 3.0)

    def test_static_analysis_clean_adds_strength_and_ignores_other_dimensions(self):
        c = with_analysis(StyleGuide(), issue(CodeDimension.SECURITY)).critique(py(GOOD_STYLE))
        assert "Passes style checks" in c.strengths
        assert c.overall_score == 10.0

    def test_non_python_skips_naming_and_linter(self):
        guide = with_analysis(StyleGuide())
        c = guide.critique(CodeSample(code="const myValue = 1;\n", language=Language.JAVASCRIPT))
        guide.analyzer.analyze.assert_not_called()
        assert not any("snake_case" in w for w in c.weaknesses)

    def test_reasoning_bands(self, guide):
        mid = guide.critique(py("x = 1\n"))  # -1.0 docstrings, -0.5 hints => 8.5
        assert mid.reasoning.startswith("Code follows good style")
        worse = StyleGuide()
        with_analysis(worse, *[issue(CodeDimension.STYLE)] * 10)
        code = "def calcTotal(a):\n    return a  # " + "y" * 120 + "\n"
        c = worse.critique(py(code))
        # -3.0 style, -1.0 docs, -0.5 hints, -0.5 camel, -0.3 long line = 4.7
        assert c.overall_score == pytest.approx(4.7)
        assert c.reasoning == "Code has significant style issues affecting maintainability. "
        between = guide.critique(py("def calcTotal(a):\n    return a  # " + "y" * 120 + "\nz = 42\n"))
        assert between.overall_score == pytest.approx(10 - 1.0 - 0.5 - 0.5 - 0.3 - 0.3)
        assert between.reasoning == "Code style could be improved for better readability. "


# ============================================================================
# SecurityAuditor
# ============================================================================


class TestSecurityAuditor:
    @pytest.fixture
    def auditor(self):
        return SecurityAuditor(use_static_analysis=False)

    def test_static_analysis_flag(self):
        a = SecurityAuditor(True).analyzer
        assert (a.use_ruff, a.use_mypy, a.use_bandit) == (False, False, True)

    def test_clean_code(self, auditor):
        c = auditor.critique(py("def add(a, b):\n    return a + b\n"))
        assert c.overall_score == 10.0
        assert c.reasoning == "Code follows security best practices. "
        assert c.dimension_scores == {CodeDimension.SECURITY: 10.0}

    def test_dangerous_pattern_reports_first_matching_line(self, auditor):
        code = "x = 1\nresult = eval(user)\nother = eval(user2)\n"
        c = auditor.critique(py(code))
        assert c.weaknesses == ["Line 2: Code injection risk"]
        assert c.line_comments == {2: "SECURITY: Code injection risk"}
        assert c.suggestions == ["Never use eval() with untrusted input"]
        assert c.overall_score == pytest.approx(7.0)
        assert c.reasoning == "Code has minor security considerations to address. "

    @pytest.mark.parametrize(
        "snippet,risk",
        [
            ("os.system(cmd)", "Command injection risk"),
            ("subprocess.run(cmd, shell=True)", "Command injection risk"),
            ("pickle.loads(blob)", "Deserialization attack risk"),
            ("yaml.load(text)", "YAML deserialization risk"),
            ("h = md5(data)", "Weak cryptography"),
            ("h = sha1(data)", "Weak cryptography"),
            ("x = random.random()", "Weak randomness"),
            ("session.verify = False", "SSL verification disabled"),
            ("CORS(app)", "Potential CORS misconfiguration"),
            ("token = api_key", "Sensitive data exposure"),
        ],
    )
    def test_each_pattern_family_is_detected(self, auditor, snippet, risk):
        c = auditor.critique(py(snippet + "\n"))
        assert any(risk in w for w in c.weaknesses), c.weaknesses

    def test_each_matching_pattern_deducts_once(self, auditor):
        c = auditor.critique(py("eval(a)\neval(b)\nexec(c)\n"))
        assert c.overall_score == pytest.approx(10 - 2 * 3.0)

    @pytest.mark.parametrize(
        "snippet,penalty",
        [
            ("eval(a)", 3.0),
            ("exec(a)", 3.0),
            ("os.system(cmd)", 3.0),
            ("subprocess.run(cmd, shell=True)", 3.0),
            ("pickle.loads(blob)", 3.0),
            ("yaml.load(text)", 3.0),
            ('q = "SELECT * FROM t WHERE id = %s" % i', 3.0),
            ('q = f"SELECT * FROM t WHERE id = {i}"', 3.0),
            ("password = 1", 1.5),
            ("client_secret = 1", 1.5),
            ("token = api_key", 1.5),
            ("session.verify = False", 1.5),
            ("h = md5(data)", 1.0),
            ("h = sha1(data)", 1.0),
            ("x = random.random()", 1.0),
            ("CORS(app)", 1.0),
        ],
    )
    def test_deduction_depends_on_severity(self, auditor, snippet, penalty):
        c = auditor.critique(py(snippet + "\n"))
        assert len(c.weaknesses) == 1
        assert c.overall_score == pytest.approx(10 - penalty)

    @pytest.mark.parametrize("call", ["eval", "exec"])
    def test_eval_or_exec_of_input_is_called_out(self, auditor, call):
        c = auditor.critique(py(f"{call}(input())\n"))
        assert c.weaknesses == [f"Line 1: Code injection risk ({call} of user input)"]
        assert c.line_comments == {1: f"SECURITY: Code injection risk ({call} of user input)"}
        assert c.overall_score == pytest.approx(10 - 3.0 - 4.0 - 0.5)

    def test_eval_of_input_must_be_on_the_same_line(self, auditor):
        c = auditor.critique(py("name = input()\neval(name)\n"))
        assert c.weaknesses == ["Line 2: Code injection risk"]
        assert c.overall_score == pytest.approx(10 - 3.0 - 0.5)

    def test_eval_of_input_with_spacing_is_called_out(self, auditor):
        c = auditor.critique(py("eval( input() )\n"))
        assert c.weaknesses == ["Line 1: Code injection risk (eval of user input)"]

    def test_eval_of_input_scores_four_or_less(self):
        c = SecurityAuditor().critique(CodeSample(code="def f(): eval(input())", language=Language.PYTHON))
        assert c.overall_score <= 4.0
        assert "Line 1: Code injection risk (eval of user input)" in c.weaknesses

    def test_patterns_are_case_insensitive(self, auditor):
        c = auditor.critique(py("PASSWORD = 1\n"))
        assert c.weaknesses == ["Line 1: Sensitive data exposure"]

    def test_secret_word_variants_are_flagged(self, auditor):
        for line in ("SECRET_KEY = 1", "my_secret = 1", "secret = 1"):
            c = auditor.critique(py(line + "\n"))
            assert c.weaknesses == ["Line 1: Sensitive data exposure"], line

    def test_percent_formatted_sql_is_flagged(self, auditor):
        c = auditor.critique(py('query = "SELECT * FROM users WHERE id = %s" % user_id\n'))
        assert any("SQL injection" in w for w in c.weaknesses)

    def test_fstring_sql_is_flagged(self, auditor):
        c = auditor.critique(py('q = f"SELECT * FROM t WHERE id = {i}"\n'))
        assert any("SQL injection risk" in w for w in c.weaknesses)

    def test_unvalidated_input(self, auditor):
        c = auditor.critique(py("name = input('x')\nprint(name)\n"))
        assert "Validate user input before processing" in c.suggestions
        assert c.overall_score == 9.5

    def test_validated_input_not_flagged(self, auditor):
        c = auditor.critique(py("name = input('x')\nvalidate(name)\n"))
        assert "Validate user input before processing" not in c.suggestions

    def test_sensitive_operation_without_error_handling(self, auditor):
        c = auditor.critique(py("f = open('a.txt')\n"))
        assert "Add error handling for sensitive operations" in c.suggestions
        assert c.overall_score == 9.5

    def test_sensitive_operation_with_error_handling(self, auditor):
        code = "try:\n    f = open('a.txt')\nexcept OSError:\n    f = None\n"
        c = auditor.critique(py(code))
        assert c.suggestions == [] and c.overall_score == 10.0

    def test_positive_signals(self, auditor):
        code = (
            "import secrets, hashlib\n"
            "try:\n"
            "    t = secrets.token_hex(8)\n"
            "    h = hashlib.sha256(b'x')\n"
            "    cur.execute('SELECT 1 WHERE a = ?', (1,))\n"
            "except Exception:\n    pass\n"
        )
        c = auditor.critique(py(code))
        assert c.strengths == [
            "Uses secrets module for secure randomness",
            "Uses strong cryptographic hashing",
            "Appears to use parameterized queries",
        ]

    def test_secrets_module_is_not_a_hardcoded_secret(self, auditor):
        code = "import secrets\ntry:\n    t = secrets.token_hex(8)\nexcept Exception:\n    t = ''\n"
        c = auditor.critique(py(code))
        assert "Uses secrets module for secure randomness" in c.strengths
        assert c.overall_score == 10.0

    def test_sha512_counts_as_strong_hash(self, auditor):
        c = auditor.critique(py("h = hashlib.sha512(b'x')\n"))
        assert "Uses strong cryptographic hashing" in c.strengths

    def test_static_analysis_severity_mapping(self):
        auditor = with_analysis(
            SecurityAuditor(),
            issue(CodeDimension.SECURITY, "error", line=1, message="high one"),
            issue(CodeDimension.SECURITY, "warning", line=2, message="medium one"),
            issue(CodeDimension.SECURITY, "info", line=3, message="low one"),
            issue(CodeDimension.STYLE, "error", line=4, message="not security"),
        )
        c = auditor.critique(py("x = 1\n"))
        assert c.overall_score == pytest.approx(10 - 2.0 - 1.0 - 0.3)
        assert c.weaknesses == ["HIGH: high one", "MEDIUM: medium one"]
        assert c.line_comments == {1: "SECURITY: high one", 2: "SECURITY: medium one", 3: "SECURITY: low one"}

    def test_static_analysis_skipped_for_non_python(self):
        auditor = with_analysis(SecurityAuditor())
        auditor.critique(CodeSample(code="eval(x)", language=Language.JAVASCRIPT))
        auditor.analyzer.analyze.assert_not_called()

    @pytest.mark.parametrize(
        "score_code,prefix",
        [
            ("x = 1\n", "Code follows security best practices"),
            ("eval(a)\n", "Code has minor security considerations"),
            ("eval(a)\npassword = 1\n", "Code has security issues that should be fixed"),
            ("eval(a)\nexec(b)\nos.system(c)\nyaml.load(d)\n", "CRITICAL: Code has serious security vulnerabilities."),
        ],
    )
    def test_reasoning_bands(self, auditor, score_code, prefix):
        assert auditor.critique(py(score_code)).reasoning.startswith(prefix)

    def test_score_floor(self, auditor):
        code = "\n".join(
            ["eval(a)", "exec(b)", "os.system(c)", "yaml.load(d)", "md5(e)", "sha1(f)", "password = 1", "secret = 2"]
        )
        assert auditor.critique(py(code)).overall_score == 0.0


# ============================================================================
# ArchitectureReviewer
# ============================================================================


class TestArchitectureReviewer:
    @pytest.fixture
    def reviewer(self):
        return ArchitectureReviewer()

    def test_simple_code_strengths(self, reviewer):
        c = reviewer.critique(py("def f():\n    return 1\n"))
        assert c.overall_score == 10.0
        assert c.strengths == ["Good complexity level", "Flat, readable structure"]
        assert c.dimension_scores == {CodeDimension.ARCHITECTURE: 10.0, CodeDimension.MAINTAINABILITY: 9.0}
        assert c.reasoning == "Code has good architecture with clear structure. "

    def test_function_over_fifty_lines(self, reviewer):
        c = reviewer.critique(py(function_of_lines("big", 55)))
        assert c.overall_score == pytest.approx(9.5)
        assert c.weaknesses == ["Function 'big' is too long (55 lines)"]
        assert c.suggestions == ["Consider breaking 'big' into smaller functions"]

    def test_function_between_thirty_and_fifty_lines(self, reviewer):
        c = reviewer.critique(py(function_of_lines("mid", 40)))
        assert c.overall_score == pytest.approx(9.8)
        assert c.weaknesses == []

    def test_short_function_not_penalised(self, reviewer):
        assert reviewer.critique(py(function_of_lines("small", 29))).overall_score == 10.0

    def test_high_cyclomatic_complexity(self, reviewer):
        code = "".join(f"if x{i}:\n    pass\n" for i in range(16))  # complexity 17
        c = reviewer.critique(py(code))
        assert c.overall_score == pytest.approx(8.5)
        assert c.weaknesses == ["High cyclomatic complexity: 17"]
        assert "Reduce complexity by extracting methods" in c.suggestions
        assert "Good complexity level" not in c.strengths

    def test_moderate_cyclomatic_complexity(self, reviewer):
        code = "".join(f"if x{i}:\n    pass\n" for i in range(11))  # complexity 12
        c = reviewer.critique(py(code))
        assert c.overall_score == pytest.approx(9.5)
        assert c.suggestions == ["Consider simplifying complex logic"]

    def test_deep_nesting(self, reviewer):
        code = "if a:\n if b:\n  if c:\n   if d:\n    if e:\n     pass\n"
        c = reviewer.critique(py(code))
        assert c.overall_score == pytest.approx(9.0)
        assert c.weaknesses == ["Deep nesting: 5 levels"]
        assert "Use early returns or extract nested logic" in c.suggestions
        assert "Flat, readable structure" not in c.strengths

    def test_medium_nesting_has_no_comment(self, reviewer):
        code = "if a:\n if b:\n  if c:\n   pass\n"
        c = reviewer.critique(py(code))
        assert c.overall_score == 10.0
        assert "Flat, readable structure" not in c.strengths

    def test_god_class(self, reviewer):
        methods = "".join(f"    def m{i}(self):\n        return {i}\n" for i in range(16))
        c = reviewer.critique(py("class Big:\n" + methods))
        assert c.overall_score == pytest.approx(9.0)
        assert c.weaknesses == ["Class 'Big' has too many methods (16)"]
        assert "Single Responsibility" in c.suggestions[0]

    def test_large_but_not_god_class(self, reviewer):
        methods = "".join(f"    def m{i}(self):\n        return {i}\n" for i in range(12))
        assert reviewer.critique(py("class Mid:\n" + methods)).overall_score == pytest.approx(9.7)

    def test_mixed_concerns_in_single_unit(self, reviewer):
        code = "print('x')\nurl = 'http://x'\ncur.execute('q')\n"
        c = reviewer.critique(py(code))
        assert c.overall_score == pytest.approx(9.5)
        assert c.suggestions == [
            "Multiple concerns in one place: io, network, database. Consider separating."
        ]

    def test_mixed_concerns_tolerated_with_multiple_classes(self, reviewer):
        code = "class A: pass\nclass B: pass\nprint('x')\nurl = 'http://x'\ncur.execute('q')\n"
        assert reviewer.critique(py(code)).overall_score == 10.0

    def test_two_concerns_are_fine(self, reviewer):
        assert reviewer.critique(py("print('x')\nurl = 'http://x'\n")).overall_score == 10.0

    def test_non_python_uses_default_features(self, reviewer):
        c = reviewer.critique(CodeSample(code="function f() { return 1; }", language=Language.JAVASCRIPT))
        assert c.overall_score == 10.0
        assert c.strengths == ["Good complexity level", "Flat, readable structure"]

    def test_python_syntax_error_does_not_crash(self, reviewer):
        c = reviewer.critique(py("def f(:\n"))
        assert c.overall_score == 10.0

    def test_reasoning_bands(self, reviewer):
        code = "".join(function_of_lines(f"f{i}", 55) for i in range(5))
        good = reviewer.critique(py(function_of_lines("one", 55)))
        assert good.reasoning == "Code has good architecture with clear structure. "
        c = reviewer.critique(py(code))
        assert c.overall_score == pytest.approx(7.5)
        assert c.reasoning == "Architecture could be improved for better maintainability. "

        features = CodeFeatures(cyclomatic_complexity=20, max_nesting_depth=6)
        with patch("integrations.code.code_teacher.extract_code_features", return_value=features):
            bad = reviewer.critique(py(code))
        # 5 * 0.5 long functions + 1.5 complexity + 1.0 nesting
        assert bad.overall_score == pytest.approx(10 - 2.5 - 1.5 - 1.0)
        assert bad.reasoning == "Architecture could be improved for better maintainability. "

        methods = "".join(f"    def m{i}(self):\n        return {i}\n" for i in range(16))
        worst_code = code + "class Big:\n" + methods
        with patch("integrations.code.code_teacher.extract_code_features", return_value=features):
            worst = reviewer.critique(py(worst_code + "print('x')\nu='http://x'\ncur.execute('q')\n"))
        assert worst.overall_score == pytest.approx(10 - 2.5 - 1.5 - 1.0 - 1.0 - 0.5)
        assert worst.overall_score == pytest.approx(3.5)
        assert worst.reasoning == "Significant architectural issues affecting code quality. "

        many = "".join(function_of_lines(f"g{i}", 55) for i in range(12))
        low = reviewer.critique(py(many))
        assert low.overall_score == pytest.approx(4.0)
        assert low.reasoning == "Significant architectural issues affecting code quality. "


# ============================================================================
# PerformanceAnalyst
# ============================================================================


class TestPerformanceAnalyst:
    @pytest.fixture
    def analyst(self):
        return PerformanceAnalyst()

    def test_clean_code(self, analyst):
        c = analyst.critique(py("def f(x):\n    return x + 1\n"))
        assert c.overall_score == 10.0
        assert c.weaknesses == [] and c.suggestions == [] and c.strengths == []
        assert c.reasoning == "Code appears performant with no obvious issues. "
        assert c.dimension_scores == {CodeDimension.PERFORMANCE: 10.0}

    @pytest.mark.parametrize(
        "code,label",
        [
            ("for a in b: for c in d: pass\n", "Nested loops"),
            ("total += str(i)\n", "String concatenation in loop"),
            ("[out.append(i) for i in items]\n", "Append in loop"),
            ("for x in list(gen):\n    pass\n", "Converting to list unnecessarily"),
            ("for i in range(len(items)):\n    pass\n", "range(len()) pattern"),
            ("names = sorted(d.keys())\n", "Iterating over .keys()"),
            ("global counter\n", "Global variables"),
            ("from os import *\n", "Wildcard import"),
        ],
    )
    def test_each_antipattern_costs_half_a_point(self, analyst, code, label):
        c = analyst.critique(py(code))
        assert label in c.weaknesses
        assert c.overall_score <= 9.5

    def test_antipattern_suggestion_pairs(self, analyst):
        c = analyst.critique(py("for i in range(len(items)):\n    pass\n"))
        assert c.weaknesses == ["range(len()) pattern"]
        assert c.suggestions == ["Use enumerate() instead"]
        assert c.overall_score == 9.5

    def test_antipattern_matching_is_case_insensitive(self, analyst):
        assert "Global variables" in analyst.critique(py("GLOBAL x\n")).weaknesses

    def test_repeated_calls_suggest_caching(self, analyst):
        code = "a = expensive(1)\nb = expensive(2)\nc = expensive(3)\nd = expensive(4)\n"
        c = analyst.critique(py(code))
        assert "Consider caching repeated calls: expensive" in c.suggestions
        assert c.overall_score == 10.0

    def test_repeated_builtin_calls_ignored(self, analyst):
        code = "print(1)\nprint(2)\nprint(3)\nprint(4)\n"
        assert analyst.critique(py(code)).suggestions == []

    def test_three_repeats_is_not_enough(self, analyst):
        assert analyst.critique(py("f(1)\nf(2)\nf(3)\n")).suggestions == []

    def test_membership_hint_when_appending_and_testing_in(self, analyst):
        c = analyst.critique(py("if x in items:\n    items.append(x)\n"))
        assert "If checking membership, consider using a set instead of list" in c.suggestions

    def test_loop_plus_append_suggests_comprehension(self, analyst):
        code = "result = []\nfor x in data:\n    result.append(x)\n"
        c = analyst.critique(py(code))
        assert c.overall_score == pytest.approx(9.7)
        assert "Consider using list comprehension instead of loop + append" in c.suggestions

    def test_positive_patterns(self, analyst):
        code = (
            "import numpy as np\n"
            "from functools import lru_cache\n"
            "@lru_cache\n"
            "def gen():\n    yield 1\n"
            "seen = set()\n"
        )
        c = analyst.critique(py(code))
        assert c.strengths == [
            "Uses generators for memory efficiency",
            "Uses caching for expensive operations",
            "Uses NumPy for efficient array operations",
            "Uses sets for efficient membership testing",
        ]
        assert c.overall_score == 10.0

    def test_functools_cache_decorator_recognised(self, analyst):
        assert "Uses caching for expensive operations" in analyst.critique(py("@cache\ndef f(): pass\n")).strengths

    def test_two_separate_nested_loop_pairs_are_penalised(self, analyst):
        code = (
            "for a in x:\n    for b in y:\n        pass\n"
            "for c in x:\n    for d in y:\n        pass\n"
        )
        c = analyst.critique(py(code))
        assert "Multiple nested loops detected (2)" in c.weaknesses
        assert c.overall_score == pytest.approx(9.0)

    def test_single_nested_loop_pair_not_flagged_by_counter(self, analyst):
        code = "for a in x:\n    for b in y:\n        pass\n"
        c = analyst.critique(py(code))
        assert not any("Multiple nested loops" in w for w in c.weaknesses)

    def test_reasoning_bands_and_floor(self, analyst):
        worst = (
            "global g\nfrom os import *\nfor a in list(x): pass\ns += str(1)\n"
            "for i in range(len(x)): pass\nz = sorted(d.keys())\nout = [o.append(i) for i in x]\n"
            "for p in q: for r in t: pass\n"
            "for a in x:\n    for b in y:\n        pass\n"
            "for c in x:\n    for d in y:\n        pass\n"
            "res = []\nfor x in y:\n    res.append(x)\n"
        )
        c = analyst.critique(py(worst))
        assert c.overall_score == pytest.approx(10 - 8 * 0.5 - 1.0 - 0.3)
        assert c.reasoning == "Significant performance concerns detected. "
        mid = analyst.critique(py("global a\nfrom os import *\ntotal += str(1)\nz = sorted(d.keys())\n"
                                  "for i in range(len(x)): pass\n"))
        assert mid.overall_score == pytest.approx(7.5)
        assert mid.reasoning == "Some performance optimizations possible. "
        assert analyst.critique(py("global a\n")).reasoning == "Code appears performant with no obvious issues. "


# ============================================================================
# DocumentationCritic
# ============================================================================


class TestDocumentationCritic:
    @pytest.fixture
    def critic(self):
        return DocumentationCritic()

    def test_well_documented_module(self, critic):
        code = '"""Module doc."""\n\ndef add(a: int, b: int) -> int:\n    """Add."""\n    return a + b\n'
        c = critic.critique(py(code))
        assert c.overall_score == 10.0
        assert c.strengths == [
            "Has module docstring",
            "All public functions have docstrings",
            "Uses type hints for self-documentation",
        ]
        assert c.reasoning == "Code is well-documented and self-explanatory. "
        assert c.dimension_scores == {CodeDimension.DOCUMENTATION: 10.0}

    @pytest.mark.parametrize(
        "code",
        [
            "x = 1\n",              # first statement is not an expression
            "1 + 1\n",              # first statement is an expression but not a string
            "",                     # empty module
        ],
    )
    def test_missing_module_docstring(self, critic, code):
        c = critic.critique(py(code))
        assert "Add a module-level docstring" in c.suggestions
        assert c.overall_score == pytest.approx(9.5)

    def test_undocumented_public_functions_penalised_private_ignored(self, critic):
        code = '"""M."""\ndef a() -> int:\n    return 1\ndef b() -> int:\n    return 2\ndef _c() -> int:\n    return 3\n'
        c = critic.critique(py(code))
        assert c.weaknesses == ["Functions without docstrings: a, b"]
        assert c.overall_score == pytest.approx(10 - 0.6)
        assert "All public functions have docstrings" not in c.strengths

    def test_function_docstring_penalty_capped_and_listed_to_five(self, critic):
        funcs = "".join(f"def f{i}() -> int:\n    return {i}\n" for i in range(8))
        c = critic.critique(py('"""M."""\n' + funcs))
        assert c.weaknesses == ["Functions without docstrings: f0, f1, f2, f3, f4"]
        assert c.overall_score == pytest.approx(10 - 2.0)

    def test_only_private_undocumented_functions_counts_as_documented(self, critic):
        code = '"""M."""\ndef _hidden() -> int:\n    return 1\n'
        c = critic.critique(py(code))
        assert "All public functions have docstrings" in c.strengths

    def test_type_hints_missing_penalised_only_when_functions_exist(self, critic):
        with_funcs = critic.critique(py('"""M."""\ndef a():\n    """D."""\n'))
        assert "Add type hints to function signatures" in with_funcs.suggestions
        assert with_funcs.overall_score == pytest.approx(9.5)
        no_funcs = critic.critique(py('"""M."""\nx = 1\n'))
        assert "Add type hints to function signatures" not in no_funcs.suggestions
        assert no_funcs.overall_score == 10.0

    def test_sparse_comments_in_long_code(self, critic):
        body = "".join(f"x{i} = {i}\n" for i in range(25))
        c = critic.critique(py('"""M."""\n' + body))
        assert "Add more inline comments for complex logic" in c.suggestions
        assert c.overall_score == pytest.approx(9.5)

    def test_short_code_not_asked_for_comments(self, critic):
        c = critic.critique(py('"""M."""\nx = 1\n'))
        assert "Add more inline comments for complex logic" not in c.suggestions

    def test_balanced_comments_not_penalised(self, critic):
        body = "".join(f"x{i} = {i}\n" for i in range(24)) + "# note\n" * 2
        c = critic.critique(py('"""M."""\n' + body))
        assert c.overall_score == 10.0

    def test_over_commented_code(self, critic):
        code = '"""M."""\n# a\n# b\n# c\nx = 1\n'
        c = critic.critique(py(code))
        assert "Possibly over-commented - code should be self-documenting" in c.weaknesses
        assert c.overall_score == pytest.approx(9.7)

    def test_unclear_todos(self, critic):
        code = '"""M."""\n# TODO\n# FIXME: x\n# TODO: refactor the parser to stream tokens\nx = 1\n'
        c = critic.critique(py(code))
        assert "Add explanations to TODO/FIXME comments" in c.suggestions
        # two unclear markers (-0.4) and an over-commented module (-0.3)
        assert c.overall_score == pytest.approx(10 - 0.4 - 0.3)

    def test_non_python_skips_ast_checks(self, critic):
        c = critic.critique(CodeSample(code="function f() {}\n", language=Language.JAVASCRIPT))
        assert c.overall_score == 10.0
        assert "Add a module-level docstring" not in c.suggestions

    def test_python_syntax_error_skips_ast_checks(self, critic):
        c = critic.critique(py("def f(:\n"))
        assert c.overall_score == 10.0

    def test_reasoning_bands(self, critic):
        funcs = "".join(f"def f{i}():\n    return {i}\n" for i in range(8))
        mid = critic.critique(py(funcs))  # -0.5 module, -2.0 funcs, -0.5 hints = 7.0
        assert mid.overall_score == pytest.approx(7.0)
        assert mid.reasoning == "Documentation could be improved. "
        todos = "# TODO\n" * 8
        low = critic.critique(py(funcs + todos))
        assert low.overall_score == pytest.approx(7.0 - 1.6)  # 8 comments / 16 code lines: not over-commented
        assert low.reasoning == "Documentation could be improved. "
        many_todos = critic.critique(py(funcs + "# TODO\n" * 20))
        assert many_todos.overall_score == pytest.approx(7.0 - 4.0 - 0.3)
        assert many_todos.reasoning == "Code lacks sufficient documentation. "


# ============================================================================
# LLMCodeTeacher
# ============================================================================


class TestLLMCodeTeacher:
    def test_defaults(self):
        t = LLMCodeTeacher()
        assert t.model == "claude-sonnet-4-20250514"
        assert t.focus_dimensions == list(CodeDimension)
        assert t.name == "LLM Code Reviewer"
        assert t._client is None

    def test_custom_focus_and_model(self):
        t = LLMCodeTeacher(model="gpt-x", focus_dimensions=[CodeDimension.SECURITY])
        assert (t.model, t.focus_dimensions) == ("gpt-x", [CodeDimension.SECURITY])

    def test_claude_models_use_anthropic_client_cached(self):
        fake = MagicMock()
        with patch.dict(sys.modules, {"anthropic": fake}):
            t = LLMCodeTeacher(model="Claude-Opus")  # case-insensitive match
            assert t._get_client() is fake.Anthropic.return_value
            assert t._get_client() is fake.Anthropic.return_value
        fake.Anthropic.assert_called_once_with()

    def test_other_models_use_openai_client(self):
        fake = MagicMock()
        with patch.dict(sys.modules, {"openai": fake}):
            t = LLMCodeTeacher(model="gpt-4o")
            assert t._get_client() is fake.OpenAI.return_value
        fake.OpenAI.assert_called_once_with()

    @pytest.mark.parametrize(
        "model,module,hint", [("claude-x", "anthropic", "pip install anthropic"), ("gpt-x", "openai", "pip install openai")]
    )
    def test_missing_sdk_gives_install_hint(self, model, module, hint):
        with patch.dict(sys.modules, {module: None}):
            with pytest.raises(ImportError, match=hint):
                LLMCodeTeacher(model=model)._get_client()

    def test_critique_propagates_missing_sdk(self):
        with patch.dict(sys.modules, {"anthropic": None}):
            with pytest.raises(ImportError):
                LLMCodeTeacher().critique(py("x = 1"))

    @staticmethod
    def claude_teacher(text):
        t = LLMCodeTeacher(model="claude-test")
        t._client = MagicMock()
        t._client.messages.create.return_value = SimpleNamespace(content=[SimpleNamespace(text=text)])
        return t

    @staticmethod
    def openai_teacher(text):
        t = LLMCodeTeacher(model="gpt-test")
        t._client = MagicMock()
        t._client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
        )
        return t

    def test_claude_request_and_score_parsing(self):
        t = self.claude_teacher("Overall: 8.5/10. Solid code.")
        sample = CodeSample(code="x = 1", language=Language.PYTHON, task="add numbers", context="inside a CLI")
        c = t.critique(sample)

        kwargs = t._client.messages.create.call_args.kwargs
        assert kwargs["model"] == "claude-test"
        assert kwargs["max_tokens"] == 1000
        (message,) = kwargs["messages"]
        assert message["role"] == "user"
        prompt = message["content"]
        assert "Review this python code" in prompt
        assert "```python\nx = 1\n```" in prompt
        assert "Task: add numbers" in prompt
        assert "Context: inside a CLI" in prompt

        assert c.overall_score == 8.5
        assert c.reasoning == "Overall: 8.5/10. Solid code."
        assert c.teacher_name == "LLM Code Reviewer"
        assert c.language == Language.PYTHON
        assert c.confidence == 1.0

    def test_prompt_omits_task_and_context_when_absent(self):
        t = self.claude_teacher("7/10")
        t.critique(py("x = 1"))
        prompt = t._client.messages.create.call_args.kwargs["messages"][0]["content"]
        assert "Task:" not in prompt and "Context:" not in prompt

    def test_openai_request_and_response_shape(self):
        t = self.openai_teacher("I'd say 6 / 10")
        c = t.critique(py("x = 1"))
        kwargs = t._client.chat.completions.create.call_args.kwargs
        assert kwargs["model"] == "gpt-test"
        assert kwargs["max_tokens"] == 1000
        assert kwargs["messages"][0]["role"] == "user"
        assert c.overall_score == 6.0

    def test_default_score_when_response_has_no_rating(self):
        assert self.claude_teacher("Looks fine.").critique(py("x")).overall_score == 7.0

    def test_scores_are_clamped_to_range(self):
        assert self.claude_teacher("15/10").critique(py("x")).overall_score == 10
        assert self.claude_teacher("0/10").critique(py("x")).overall_score == 0

    def test_reasoning_truncated_to_500_chars(self):
        c = self.claude_teacher("9/10 " + "z" * 900).critique(py("x"))
        assert len(c.reasoning) == 500

    def test_api_failure_returns_low_confidence_neutral_critique(self):
        t = LLMCodeTeacher(model="claude-test")
        t._client = MagicMock()
        t._client.messages.create.side_effect = RuntimeError("boom")
        c = t.critique(CodeSample(code="x", language=Language.GO))
        assert c.overall_score == 5.0
        assert c.confidence == 0.0
        assert c.reasoning == "LLM review failed: boom"
        assert c.language == Language.GO

    def test_hundred_point_scale_is_not_read_as_ten_point(self):
        c = self.claude_teacher("I rate this 45/100").critique(py("x"))
        assert c.overall_score < 10

    def test_hundred_point_scale_is_scaled_to_ten(self):
        assert self.claude_teacher("I rate this 45/100").critique(py("x")).overall_score == pytest.approx(4.5)

    def test_zero_denominator_keeps_default_score(self):
        assert self.claude_teacher("5/0").critique(py("x")).overall_score == 7.0


# ============================================================================
# CodeTeacher (composite)
# ============================================================================


class StubTeacher(BaseCodeTeacher):
    def __init__(self, name, critique):
        super().__init__(name, "stub", [CodeDimension.STYLE])
        self._critique = critique

    def critique(self, sample):
        return self._critique


def stub_critique(name, score, **kw):
    return CodeCritique(overall_score=score, teacher_name=name, language=Language.PYTHON, **kw)


class TestCodeTeacherComposite:
    def test_default_personas(self):
        t = CodeTeacher()
        assert [x.name for x in t.teachers] == ["Correctness Checker", "Style Guide", "Security Auditor"]
        assert t.strategy == "vote"
        assert t.weights == {"Correctness Checker": 1.0, "Style Guide": 1.0, "Security Auditor": 1.0}

    def test_all_personas_registered(self):
        assert set(CodeTeacher.PERSONA_MAP) == {
            "correctness_checker", "style_guide", "security_auditor",
            "architecture_reviewer", "performance_analyst", "documentation_critic",
        }
        t = CodeTeacher(personas=list(CodeTeacher.PERSONA_MAP))
        assert len(t.teachers) == 6

    def test_unknown_persona_rejected(self):
        with pytest.raises(ValueError, match="Unknown persona: wizard"):
            CodeTeacher(personas=["style_guide", "wizard"])

    def test_llm_teacher_appended_when_requested(self):
        t = CodeTeacher(personas=["style_guide"], use_llm=True, llm_model="gpt-x")
        assert isinstance(t.teachers[-1], LLMCodeTeacher)
        assert t.teachers[-1].model == "gpt-x"
        assert "LLM Code Reviewer" in t.weights

    def test_eval_of_user_input_pulls_committee_below_seven(self):
        t = CodeTeacher(personas=["correctness_checker", "style_guide", "security_auditor"], strategy="vote")
        c = t.critique(CodeSample(code="def f(): eval(input())", language=Language.PYTHON))
        assert c.overall_score < 7.0
        assert "Line 1: Code injection risk (eval of user input)" in c.weaknesses

    def test_custom_weights_kept(self):
        t = CodeTeacher(personas=["style_guide"], weights={"Style Guide": 5.0})
        assert t.weights == {"Style Guide": 5.0}

    def test_repr(self):
        assert repr(CodeTeacher(personas=["architecture_reviewer"], strategy="rotate")) == (
            "CodeTeacher(teachers=['Architecture Reviewer'], strategy='rotate')"
        )

    def test_language_autodetected_and_written_back(self):
        t = CodeTeacher(personas=["architecture_reviewer"])
        sample = CodeSample(code="x", filename="app.rs")
        t.critique(sample)
        assert sample.language == Language.RUST

    def test_explicit_language_not_overridden(self):
        t = CodeTeacher(personas=["architecture_reviewer"])
        sample = CodeSample(code="import os\ndef f(): pass", language=Language.GO)
        t.critique(sample)
        assert sample.language == Language.GO

    def test_rotate_cycles_through_teachers(self):
        t = CodeTeacher(personas=["architecture_reviewer", "performance_analyst"], strategy="rotate")
        names = [t.critique(py("x = 1\n")).teacher_name for _ in range(5)]
        assert names == [
            "Architecture Reviewer", "Performance Analyst", "Architecture Reviewer",
            "Performance Analyst", "Architecture Reviewer",
        ]
        assert t._turn_idx == 5

    def test_vote_weighted_average_and_merging(self):
        t = CodeTeacher(personas=["architecture_reviewer"], weights={"A": 3.0, "B": 1.0})
        a = stub_critique(
            "A", 10.0,
            dimension_scores={CodeDimension.STYLE: 10.0},
            strengths=["s1", "shared"], weaknesses=["w1"], suggestions=["g1"],
            line_comments={1: "from A", 2: "only A"}, reasoning="fine",
        )
        b = stub_critique(
            "B", 4.0,
            dimension_scores={CodeDimension.SECURITY: 2.0},
            strengths=["shared"], weaknesses=["w2"], suggestions=["g1", "g2"],
            line_comments={1: "from B"}, reasoning="bad",
        )
        t.teachers = [StubTeacher("A", a), StubTeacher("B", b)]
        out = t.critique(py("x"))

        assert out.overall_score == pytest.approx((10 * 3 + 4 * 1) / 4)
        # a dimension missing from a critique falls back to that critique's overall score
        assert out.dimension_scores[CodeDimension.STYLE] == pytest.approx((10.0 + 4.0) / 2)
        assert out.dimension_scores[CodeDimension.SECURITY] == pytest.approx((10.0 + 2.0) / 2)
        assert sorted(out.strengths) == ["s1", "shared"]
        assert sorted(out.weaknesses) == ["w1", "w2"]
        assert sorted(out.suggestions) == ["g1", "g2"]
        assert out.line_comments == {1: "from B", 2: "only A"}
        assert out.reasoning == "A: fine | B: bad"
        assert out.teacher_name == "Code Teacher Committee"
        assert out.language == Language.PYTHON

    def test_vote_unknown_teacher_name_defaults_to_weight_one(self):
        t = CodeTeacher(personas=["architecture_reviewer"], weights={"A": 3.0})
        t.teachers = [StubTeacher("A", stub_critique("A", 10.0)), StubTeacher("B", stub_critique("B", 2.0))]
        assert t.critique(py("x")).overall_score == pytest.approx((30 + 2) / 4)

    def test_vote_reasoning_truncated(self):
        t = CodeTeacher(personas=["architecture_reviewer"])
        t.teachers = [StubTeacher("A", stub_critique("A", 5.0, reasoning="r" * 2000))]
        assert len(t.critique(py("x")).reasoning) == 1000

    def test_debate_equal_scores(self):
        t = CodeTeacher(personas=["architecture_reviewer"], strategy="debate")
        t.teachers = [StubTeacher(n, stub_critique(n, 8.0)) for n in "ABC"]
        out = t.critique(py("x"))
        assert out.overall_score == pytest.approx(8.0)
        assert out.reasoning == "Debate consensus: 8.0/10 (std: 0.0)"
        assert out.teacher_name == "Code Teacher Debate"

    def test_debate_downweights_outliers(self):
        t = CodeTeacher(personas=["architecture_reviewer"], strategy="debate")
        scores = [2.0, 8.0, 8.0]
        t.teachers = [StubTeacher(str(i), stub_critique(str(i), s)) for i, s in enumerate(scores)]
        out = t.critique(py("x"))

        mean = sum(scores) / 3
        std = math.sqrt(sum((s - mean) ** 2 for s in scores) / 3)
        weights = [1 / (1 + abs(s - mean) / std) for s in scores]
        expected = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
        assert out.overall_score == pytest.approx(expected)
        assert expected > mean  # outlier pulled toward the majority
        assert f"std: {std:.1f}" in out.reasoning

    def test_unknown_strategy_raises_at_critique_time(self):
        t = CodeTeacher(personas=["architecture_reviewer"], strategy="bogus")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="Unknown strategy: bogus"):
            t.critique(py("x = 1"))

    def test_vote_end_to_end_with_real_teachers(self):
        t = CodeTeacher(personas=["correctness_checker", "documentation_critic", "architecture_reviewer"])
        code = '"""M."""\n\ndef add(a: int, b: int) -> int:\n    """Add."""\n    return a + b\n'
        out = t.critique(py(code))
        assert 0.0 <= out.overall_score <= 10.0
        assert CodeDimension.CORRECTNESS in out.dimension_scores
        assert CodeDimension.DOCUMENTATION in out.dimension_scores
        assert out.reasoning.count(" | ") == 2
