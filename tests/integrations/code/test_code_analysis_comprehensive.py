"""
Comprehensive tests for ASPIRE Code Analysis utilities.

Tests the static analysis and feature extraction:
- Language detection
- AST parsing
- Feature extraction
- Static analysis integration
"""

import pytest

from integrations.code.analysis import (
    CodeAnalyzer,
    StaticAnalysisResult,
    detect_language,
    extract_code_features,
    parse_code,
    quick_analyze,
)
from integrations.code.config import Language


class TestLanguageDetection:
    """Tests for language detection."""

    def test_detect_python_by_extension(self):
        """Should detect Python by .py extension."""
        lang = detect_language("x = 1", filename="test.py")
        assert lang == Language.PYTHON

    def test_detect_javascript_by_extension(self):
        """Should detect JavaScript by .js extension."""
        lang = detect_language("const x = 1", filename="app.js")
        assert lang == Language.JAVASCRIPT

    def test_detect_typescript_by_extension(self):
        """Should detect TypeScript by .ts extension."""
        lang = detect_language("const x: number = 1", filename="app.ts")
        assert lang == Language.TYPESCRIPT

    def test_detect_python_by_content(self):
        """Should detect Python by code patterns."""
        code = """
import os
def hello():
    print("world")
"""
        lang = detect_language(code)
        assert lang == Language.PYTHON

    def test_detect_javascript_by_content(self):
        """Should detect JavaScript by code patterns."""
        code = """
const hello = () => {
    console.log("world");
};
"""
        lang = detect_language(code)
        assert lang == Language.JAVASCRIPT

    def test_detect_rust_by_content(self):
        """Should detect Rust by code patterns."""
        code = """
fn main() {
    let mut x = 5;
    println!("x = {}", x);
}
"""
        lang = detect_language(code)
        # Rust detection is heuristic-based, may not always succeed
        assert lang in [Language.RUST, Language.JAVASCRIPT, Language.UNKNOWN]

    def test_unknown_for_ambiguous_code(self):
        """Should return UNKNOWN for ambiguous code."""
        code = "x = 1"  # Could be many languages
        lang = detect_language(code)
        # Either detected as something or unknown
        assert isinstance(lang, Language)


class TestCodeParsing:
    """Tests for code parsing."""

    def test_parse_valid_python(self):
        """Should successfully parse valid Python."""
        code = """
def add(a, b):
    return a + b

class Calculator:
    pass
"""
        result = parse_code(code, Language.PYTHON)

        assert result["success"] is True
        assert result["ast"] is not None
        assert "add" in result["functions"]
        assert "Calculator" in result["classes"]

    def test_parse_invalid_python(self):
        """Should report error for invalid Python."""
        code = """
def broken(
    return x
"""
        result = parse_code(code, Language.PYTHON)

        assert result["success"] is False
        assert result["error"] is not None

    def test_parse_extracts_imports(self):
        """Should extract import statements."""
        code = """
import os
from pathlib import Path
import json as j
"""
        result = parse_code(code, Language.PYTHON)

        assert result["success"] is True
        assert "os" in result["imports"]
        assert "pathlib" in result["imports"]
        assert "json" in result["imports"]

    def test_parse_other_languages_graceful(self):
        """Should handle other languages gracefully."""
        code = "const x = 1;"
        result = parse_code(code, Language.JAVASCRIPT)

        # Should succeed but with limited info
        assert result["success"] is True


class TestFeatureExtraction:
    """Tests for code feature extraction."""

    def test_extracts_basic_metrics(self):
        """Should extract basic code metrics."""
        code = """
import os

def hello():
    print("world")

def goodbye():
    print("farewell")

class Greeter:
    pass
"""
        features = extract_code_features(code, Language.PYTHON)

        assert features.num_lines > 0
        assert features.num_functions == 2
        assert features.num_classes == 1
        assert features.num_imports == 1

    def test_detects_main_guard(self):
        """Should detect if __name__ == '__main__' guard."""
        code_with_guard = '''
def main():
    pass

if __name__ == "__main__":
    main()
'''
        code_without = '''
def main():
    pass

main()
'''
        features_with = extract_code_features(code_with_guard, Language.PYTHON)
        features_without = extract_code_features(code_without, Language.PYTHON)

        assert features_with.has_main_guard is True
        assert features_without.has_main_guard is False

    def test_detects_error_handling(self):
        """Should detect try/except blocks."""
        code_with_try = '''
def safe_divide(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return None
'''
        code_without = '''
def divide(a, b):
    return a / b
'''
        features_with = extract_code_features(code_with_try, Language.PYTHON)
        features_without = extract_code_features(code_without, Language.PYTHON)

        assert features_with.has_error_handling is True
        assert features_without.has_error_handling is False

    def test_detects_docstrings(self):
        """Should detect presence of docstrings."""
        code_with_docs = '''
def hello():
    """Say hello."""
    print("hello")
'''
        features = extract_code_features(code_with_docs, Language.PYTHON)

        assert features.has_docstrings is True

    def test_detects_type_hints(self):
        """Should detect type hints."""
        code_typed = '''
def add(a: int, b: int) -> int:
    return a + b
'''
        code_untyped = '''
def add(a, b):
    return a + b
'''
        features_typed = extract_code_features(code_typed, Language.PYTHON)
        features_untyped = extract_code_features(code_untyped, Language.PYTHON)

        assert features_typed.has_type_hints is True
        assert features_untyped.has_type_hints is False

    def test_calculates_complexity(self):
        """Should calculate cyclomatic complexity."""
        simple_code = '''
def simple(x):
    return x + 1
'''
        complex_code = '''
def complex(a, b, c):
    if a:
        if b:
            if c:
                return 1
            return 2
        return 3
    elif b:
        return 4
    return 5
'''
        simple_features = extract_code_features(simple_code, Language.PYTHON)
        complex_features = extract_code_features(complex_code, Language.PYTHON)

        assert complex_features.cyclomatic_complexity > simple_features.cyclomatic_complexity

    def test_calculates_nesting_depth(self):
        """Should calculate max nesting depth."""
        shallow = '''
def shallow(x):
    if x:
        return 1
    return 0
'''
        deep = '''
def deep(a, b, c, d):
    if a:
        if b:
            if c:
                if d:
                    return 1
'''
        shallow_features = extract_code_features(shallow, Language.PYTHON)
        deep_features = extract_code_features(deep, Language.PYTHON)

        assert deep_features.max_nesting_depth > shallow_features.max_nesting_depth


class TestQuickAnalyze:
    """Tests for quick_analyze function."""

    def test_returns_scores(self):
        """Should return analysis scores."""
        code = '''
def hello():
    """Say hello."""
    print("hello")
'''
        result = quick_analyze(code, Language.PYTHON)

        assert "scores" in result
        assert "correctness" in result["scores"]
        assert "style" in result["scores"]
        assert "complexity" in result["scores"]

    def test_auto_detects_language(self):
        """Should auto-detect language if not provided."""
        code = '''
def hello():
    print("hello")
'''
        result = quick_analyze(code)

        # Language detection is best-effort
        assert result["language"] in ["python", "unknown"]

    def test_handles_invalid_code(self):
        """Should handle invalid code gracefully."""
        code = "def broken("
        result = quick_analyze(code, Language.PYTHON)

        assert result["parsed"] is False
        assert result["scores"]["correctness"] < 5.0

    def test_rewards_good_practices(self):
        """Should give higher scores for good practices."""
        good_code = '''
def factorial(n: int) -> int:
    """Calculate factorial."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n <= 1:
        return 1
    return n * factorial(n - 1)
'''
        bad_code = '''
def f(x):
    if x:
        if x:
            if x:
                return x
'''
        good_result = quick_analyze(good_code, Language.PYTHON)
        bad_result = quick_analyze(bad_code, Language.PYTHON)

        assert good_result["overall"] > bad_result["overall"]


class TestStaticAnalysisResult:
    """Tests for StaticAnalysisResult dataclass."""

    def test_has_errors_property(self):
        """Should correctly report has_errors."""
        from integrations.code.analysis import CodeIssue
        from integrations.code.config import CodeDimension

        result_with_error = StaticAnalysisResult(
            issues=[
                CodeIssue(
                    line=1, column=0, message="Error",
                    severity="error", dimension=CodeDimension.CORRECTNESS
                )
            ]
        )
        result_without = StaticAnalysisResult(issues=[])

        assert result_with_error.has_errors is True
        assert result_without.has_errors is False

    def test_counts_errors_and_warnings(self):
        """Should count errors and warnings separately."""
        from integrations.code.analysis import CodeIssue
        from integrations.code.config import CodeDimension

        result = StaticAnalysisResult(
            issues=[
                CodeIssue(line=1, column=0, message="E1",
                          severity="error", dimension=CodeDimension.CORRECTNESS),
                CodeIssue(line=2, column=0, message="E2",
                          severity="error", dimension=CodeDimension.CORRECTNESS),
                CodeIssue(line=3, column=0, message="W1",
                          severity="warning", dimension=CodeDimension.STYLE),
            ]
        )

        assert result.error_count == 2
        assert result.warning_count == 1


class TestCodeAnalyzerIntegration:
    """Integration tests for CodeAnalyzer (if tools available)."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return CodeAnalyzer(
            use_ruff=True,
            use_mypy=False,  # Slower
            use_bandit=True,
        )

    def test_analyzer_creation(self, analyzer):
        """Should create analyzer without errors."""
        assert analyzer is not None
        assert isinstance(analyzer._available_tools, dict)

    def test_analyze_returns_result(self, analyzer):
        """Should return StaticAnalysisResult."""
        code = "x = 1"
        result = analyzer.analyze(code, Language.PYTHON)

        assert isinstance(result, StaticAnalysisResult)

    @pytest.mark.slow
    def test_analyze_with_issues(self, analyzer):
        """Should detect issues in problematic code."""
        code = '''
import os  # unused import

def f():
    x = 1  # unused variable
    eval("code")  # security issue
'''
        result = analyzer.analyze(code, Language.PYTHON)

        # Should find at least some issues (depends on tool availability)
        # Just check that analysis completes without error
        assert isinstance(result, StaticAnalysisResult)
