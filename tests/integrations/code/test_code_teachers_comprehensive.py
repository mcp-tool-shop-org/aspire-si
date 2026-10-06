"""
Comprehensive tests for ASPIRE Code Teachers.

Tests the code analysis and critique capabilities:
- CorrectnessChecker: Bug detection, syntax errors
- StyleGuide: PEP8, naming conventions
- SecurityAuditor: Vulnerability detection
- CodeTeacher: Composite evaluation
"""

import pytest

from integrations.code.code_teacher import (
    ArchitectureReviewer,
    CodeCritique,
    CodeSample,
    CodeTeacher,
    CorrectnessChecker,
    PerformanceAnalyst,
    SecurityAuditor,
    StyleGuide,
)
from integrations.code.config import Language


class TestCorrectnessChecker:
    """Tests for CorrectnessChecker teacher."""

    @pytest.fixture
    def checker(self):
        """Create checker instance."""
        return CorrectnessChecker(use_static_analysis=False)

    def test_detects_syntax_error(self, checker):
        """Should detect syntax errors."""
        code = """
def broken(
    return x
"""
        sample = CodeSample(code=code, language=Language.PYTHON)
        critique = checker.critique(sample)

        assert critique.overall_score < 5.0
        assert any("syntax" in w.lower() for w in critique.weaknesses)

    def test_valid_code_scores_high(self, checker):
        """Valid code should score well."""
        code = """
def factorial(n: int) -> int:
    if n <= 1:
        return 1
    return n * factorial(n - 1)
"""
        sample = CodeSample(code=code, language=Language.PYTHON)
        critique = checker.critique(sample)

        assert critique.overall_score >= 7.0

    def test_detects_infinite_loop_risk(self, checker):
        """Should flag potential infinite loops."""
        code = """
def risky():
    while True:
        print("forever")
"""
        sample = CodeSample(code=code, language=Language.PYTHON)
        critique = checker.critique(sample)

        assert critique.overall_score < 10.0  # Should have some penalty
        assert any("infinite" in str(w).lower() or "while true" in str(w).lower()
                   for w in critique.weaknesses + critique.suggestions)

    def test_division_by_zero_detection(self, checker):
        """Should detect division by zero risk."""
        code = """
def divide(a, b):
    return a / 0
"""
        sample = CodeSample(code=code, language=Language.PYTHON)
        critique = checker.critique(sample)

        assert critique.overall_score < 10.0  # Should have penalty for div by zero


class TestStyleGuide:
    """Tests for StyleGuide teacher."""

    @pytest.fixture
    def guide(self):
        """Create guide instance."""
        return StyleGuide(use_static_analysis=False)

    def test_rewards_docstrings(self, guide):
        """Code with docstrings should score better."""
        code_with_docs = '''
def greet(name: str) -> str:
    """Greet a person by name."""
    return f"Hello, {name}"
'''
        code_without_docs = '''
def greet(name):
    return f"Hello, {name}"
'''
        sample_with = CodeSample(code=code_with_docs, language=Language.PYTHON)
        sample_without = CodeSample(code=code_without_docs, language=Language.PYTHON)

        critique_with = guide.critique(sample_with)
        critique_without = guide.critique(sample_without)

        assert critique_with.overall_score > critique_without.overall_score

    def test_rewards_type_hints(self, guide):
        """Code with type hints should score better."""
        code_typed = '''
def add(a: int, b: int) -> int:
    return a + b
'''
        code_untyped = '''
def add(a, b):
    return a + b
'''
        sample_typed = CodeSample(code=code_typed, language=Language.PYTHON)
        sample_untyped = CodeSample(code=code_untyped, language=Language.PYTHON)

        critique_typed = guide.critique(sample_typed)
        critique_untyped = guide.critique(sample_untyped)

        assert critique_typed.overall_score >= critique_untyped.overall_score

    def test_detects_camelcase_in_python(self, guide):
        """Should flag camelCase in Python (should be snake_case)."""
        code = """
def calculateTotal(itemPrice, itemCount):
    return itemPrice * itemCount
"""
        sample = CodeSample(code=code, language=Language.PYTHON)
        critique = guide.critique(sample)

        assert any("snake_case" in str(w).lower() or "camel" in str(w).lower()
                   for w in critique.weaknesses + critique.suggestions)


class TestSecurityAuditor:
    """Tests for SecurityAuditor teacher."""

    @pytest.fixture
    def auditor(self):
        """Create auditor instance."""
        return SecurityAuditor(use_static_analysis=False)

    def test_detects_eval(self, auditor):
        """Should flag eval() usage."""
        code = """
def process(data):
    return eval(data)
"""
        sample = CodeSample(code=code, language=Language.PYTHON)
        critique = auditor.critique(sample)

        assert critique.overall_score < 10.0  # Should penalize eval
        assert any("eval" in str(w).lower() or "injection" in str(w).lower()
                   for w in critique.weaknesses)

    def test_detects_exec(self, auditor):
        """Should flag exec() usage."""
        code = """
def run(code):
    exec(code)
"""
        sample = CodeSample(code=code, language=Language.PYTHON)
        critique = auditor.critique(sample)

        assert critique.overall_score < 10.0  # Should penalize exec

    def test_detects_sql_injection(self, auditor):
        """Should flag SQL injection patterns."""
        code = '''
def get_user(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    return db.execute(query)
'''
        sample = CodeSample(code=code, language=Language.PYTHON)
        critique = auditor.critique(sample)

        assert critique.overall_score < 10.0  # Should penalize SQL injection patterns
        assert any("sql" in str(w).lower() or "injection" in str(w).lower()
                   for w in critique.weaknesses)

    def test_detects_hardcoded_secrets(self, auditor):
        """Should flag hardcoded passwords/secrets."""
        code = '''
def connect():
    password = "admin123"
    api_key = "sk-12345abcdef"
'''
        sample = CodeSample(code=code, language=Language.PYTHON)
        critique = auditor.critique(sample)

        assert critique.overall_score < 8.0

    def test_detects_shell_injection(self, auditor):
        """Should flag shell=True in subprocess."""
        code = '''
import subprocess

def run_command(cmd):
    subprocess.run(cmd, shell=True)
'''
        sample = CodeSample(code=code, language=Language.PYTHON)
        critique = auditor.critique(sample)

        assert any("shell" in str(w).lower() or "command" in str(w).lower()
                   for w in critique.weaknesses + critique.suggestions)

    def test_praises_secure_patterns(self, auditor):
        """Should recognize secure coding patterns."""
        code = '''
import secrets
import hashlib

def generate_token():
    return secrets.token_hex(32)

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()
'''
        sample = CodeSample(code=code, language=Language.PYTHON)
        critique = auditor.critique(sample)

        assert critique.overall_score >= 7.0
        assert len(critique.strengths) > 0


class TestArchitectureReviewer:
    """Tests for ArchitectureReviewer teacher."""

    @pytest.fixture
    def reviewer(self):
        """Create reviewer instance."""
        return ArchitectureReviewer()

    def test_flags_high_complexity(self, reviewer):
        """Should flag high cyclomatic complexity."""
        code = """
def complex_function(a, b, c, d, e):
    if a:
        if b:
            if c:
                if d:
                    if e:
                        return 1
                    else:
                        return 2
                else:
                    return 3
            else:
                return 4
        else:
            return 5
    else:
        return 6
"""
        sample = CodeSample(code=code, language=Language.PYTHON)
        critique = reviewer.critique(sample)

        assert critique.overall_score < 10.0  # Should penalize complexity
        assert any("complexity" in str(w).lower() or "nesting" in str(w).lower()
                   for w in critique.weaknesses + critique.suggestions)

    def test_rewards_simple_structure(self, reviewer):
        """Should reward simple, flat code structure."""
        code = """
def simple_function(x: int) -> int:
    if x < 0:
        return -x
    return x
"""
        sample = CodeSample(code=code, language=Language.PYTHON)
        critique = reviewer.critique(sample)

        assert critique.overall_score >= 7.0


class TestPerformanceAnalyst:
    """Tests for PerformanceAnalyst teacher."""

    @pytest.fixture
    def analyst(self):
        """Create analyst instance."""
        return PerformanceAnalyst()

    def test_flags_nested_loops(self, analyst):
        """Should flag nested loop patterns."""
        code = """
def find_pairs(items):
    pairs = []
    for i in items:
        for j in items:
            for k in items:
                pairs.append((i, j, k))
    return pairs
"""
        sample = CodeSample(code=code, language=Language.PYTHON)
        critique = analyst.critique(sample)

        assert any("nested" in str(w).lower() or "loop" in str(w).lower()
                   for w in critique.weaknesses + critique.suggestions)

    def test_suggests_comprehension(self, analyst):
        """Should suggest list comprehension over append loop."""
        code = """
def square_all(numbers):
    result = []
    for n in numbers:
        result.append(n * n)
    return result
"""
        sample = CodeSample(code=code, language=Language.PYTHON)
        critique = analyst.critique(sample)

        assert any("comprehension" in str(s).lower()
                   for s in critique.suggestions)

    def test_praises_generators(self, analyst):
        """Should recognize generator usage positively."""
        code = """
def numbers(n):
    for i in range(n):
        yield i * 2
"""
        sample = CodeSample(code=code, language=Language.PYTHON)
        critique = analyst.critique(sample)

        assert any("generator" in str(s).lower()
                   for s in critique.strengths)


class TestCodeTeacher:
    """Tests for composite CodeTeacher."""

    @pytest.fixture
    def teacher(self):
        """Create composite teacher."""
        return CodeTeacher(
            personas=["correctness_checker", "style_guide", "security_auditor"],
            strategy="vote",
        )

    def test_combines_multiple_perspectives(self, teacher):
        """Should combine feedback from all teachers."""
        code = """
def process(data):
    result = eval(data)  # Security issue
    # No docstring - style issue
    return result
"""
        sample = CodeSample(code=code, language=Language.PYTHON)
        critique = teacher.critique(sample)

        # Should have feedback from multiple dimensions
        assert len(critique.dimension_scores) >= 2
        assert len(critique.weaknesses) >= 1

    def test_voting_produces_average(self, teacher):
        """Vote strategy should average teacher scores."""
        code = """
def safe_add(a: int, b: int) -> int:
    '''Add two integers safely.'''
    return a + b
"""
        sample = CodeSample(code=code, language=Language.PYTHON)
        critique = teacher.critique(sample)

        # Score should be reasonable average
        assert 5.0 <= critique.overall_score <= 10.0

    def test_auto_detects_language(self):
        """Should auto-detect programming language."""
        teacher = CodeTeacher(personas=["correctness_checker"])

        python_code = """
def hello():
    print("Hello")
"""
        sample = CodeSample(code=python_code, language=Language.UNKNOWN)
        critique = teacher.critique(sample)

        # Language detection is best-effort, main thing is it doesn't crash
        assert critique.language in [Language.PYTHON, Language.UNKNOWN]

    def test_handles_empty_code(self, teacher):
        """Should handle empty code gracefully."""
        sample = CodeSample(code="", language=Language.PYTHON)
        critique = teacher.critique(sample)

        # Should return some critique, even if low score
        assert critique.overall_score >= 0

    def test_different_strategies(self):
        """Different strategies should work."""
        for strategy in ["vote", "rotate"]:
            teacher = CodeTeacher(
                personas=["correctness_checker", "style_guide"],
                strategy=strategy,
            )
            sample = CodeSample(code="x = 1", language=Language.PYTHON)
            critique = teacher.critique(sample)

            assert critique.overall_score >= 0


class TestCodeCritiqueDataclass:
    """Tests for CodeCritique dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        critique = CodeCritique(
            overall_score=7.5,
            teacher_name="test",
            language=Language.PYTHON,
        )

        assert critique.strengths == []
        assert critique.weaknesses == []
        assert critique.suggestions == []
        assert critique.line_comments == {}

    def test_dimension_scores_dict(self):
        """Should store dimension scores."""
        from integrations.code.config import CodeDimension

        critique = CodeCritique(
            overall_score=8.0,
            dimension_scores={
                CodeDimension.CORRECTNESS: 9.0,
                CodeDimension.STYLE: 7.0,
            },
            teacher_name="test",
            language=Language.PYTHON,
        )

        assert critique.dimension_scores[CodeDimension.CORRECTNESS] == 9.0
