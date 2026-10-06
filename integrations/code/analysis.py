"""
Code Analysis Utilities - Static analysis and feature extraction for code.

Provides tools for understanding code structure, detecting issues,
and extracting features for the code critic.
"""

from __future__ import annotations

import ast
import json
import logging
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import Language, CodeDimension

logger = logging.getLogger(__name__)


@dataclass
class CodeIssue:
    """A detected issue in code."""

    line: int
    column: int
    message: str
    severity: str  # "error", "warning", "info", "hint"
    dimension: CodeDimension
    rule_id: str = ""
    suggestion: str = ""


@dataclass
class StaticAnalysisResult:
    """Results from static analysis tools."""

    issues: list[CodeIssue] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    # Summary scores (0-10)
    correctness_score: float = 10.0
    style_score: float = 10.0
    security_score: float = 10.0
    complexity_score: float = 10.0

    # Tool outputs
    tool_outputs: dict[str, str] = field(default_factory=dict)

    @property
    def has_errors(self) -> bool:
        return any(i.severity == "error" for i in self.issues)

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warning")


@dataclass
class CodeFeatures:
    """Extracted features from code for critic input."""

    # Basic metrics
    num_lines: int = 0
    num_functions: int = 0
    num_classes: int = 0
    num_imports: int = 0

    # Complexity
    cyclomatic_complexity: float = 0.0
    max_nesting_depth: int = 0
    avg_function_length: float = 0.0

    # Style
    has_docstrings: bool = False
    has_type_hints: bool = False
    follows_naming_conventions: bool = True

    # Structure
    has_main_guard: bool = False
    has_error_handling: bool = False
    import_organization: str = "unknown"  # "good", "mixed", "poor"

    # AST summary (for neural processing)
    ast_node_types: dict[str, int] = field(default_factory=dict)


def detect_language(code: str, filename: str | None = None) -> Language:
    """Detect the programming language of code."""

    if filename:
        ext = Path(filename).suffix.lower()
        ext_map = {
            ".py": Language.PYTHON,
            ".js": Language.JAVASCRIPT,
            ".ts": Language.TYPESCRIPT,
            ".tsx": Language.TYPESCRIPT,
            ".jsx": Language.JAVASCRIPT,
            ".rs": Language.RUST,
            ".go": Language.GO,
            ".java": Language.JAVA,
            ".cpp": Language.CPP,
            ".cc": Language.CPP,
            ".c": Language.C,
            ".h": Language.C,
            ".cs": Language.CSHARP,
            ".rb": Language.RUBY,
            ".php": Language.PHP,
            ".swift": Language.SWIFT,
            ".kt": Language.KOTLIN,
            ".scala": Language.SCALA,
        }
        if ext in ext_map:
            return ext_map[ext]

    # Heuristic detection
    if "def " in code and "import " in code:
        return Language.PYTHON
    if "function " in code or "const " in code or "let " in code:
        if ": " in code and "interface " in code:
            return Language.TYPESCRIPT
        return Language.JAVASCRIPT
    if "fn " in code and "let mut" in code:
        return Language.RUST
    if "func " in code and "package " in code:
        return Language.GO
    if "public class " in code or "private void " in code:
        return Language.JAVA

    return Language.UNKNOWN


def parse_code(code: str, language: Language) -> dict[str, Any]:
    """
    Parse code into an AST or structural representation.

    Returns a dictionary with parsed information.
    """
    result = {
        "language": language.value,
        "success": False,
        "ast": None,
        "error": None,
    }

    if language == Language.PYTHON:
        try:
            tree = ast.parse(code)
            result["success"] = True
            result["ast"] = tree

            # Extract structure
            result["functions"] = [
                node.name for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef)
            ]
            result["classes"] = [
                node.name for node in ast.walk(tree)
                if isinstance(node, ast.ClassDef)
            ]
            result["imports"] = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    result["imports"].extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        result["imports"].append(node.module)

        except SyntaxError as e:
            result["error"] = str(e)

    # For other languages, we'd use tree-sitter or language-specific parsers
    # For now, return basic info
    else:
        result["success"] = True
        result["note"] = f"Full parsing not implemented for {language.value}"

    return result


def extract_code_features(code: str, language: Language) -> CodeFeatures:
    """Extract features from code for critic input."""

    features = CodeFeatures()

    lines = code.split("\n")
    features.num_lines = len(lines)

    if language == Language.PYTHON:
        try:
            tree = ast.parse(code)

            # Count structures
            for node in ast.walk(tree):
                node_type = type(node).__name__
                features.ast_node_types[node_type] = (
                    features.ast_node_types.get(node_type, 0) + 1
                )

                if isinstance(node, ast.FunctionDef):
                    features.num_functions += 1
                elif isinstance(node, ast.ClassDef):
                    features.num_classes += 1
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    features.num_imports += 1
                elif isinstance(node, ast.Try):
                    features.has_error_handling = True

            # Check for docstrings
            features.has_docstrings = any(
                isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
                for node in ast.walk(tree)
            )

            # Check for type hints
            features.has_type_hints = any(
                isinstance(node, ast.FunctionDef) and node.returns is not None
                for node in ast.walk(tree)
            )

            # Check for main guard
            features.has_main_guard = 'if __name__ == "__main__"' in code or "if __name__ == '__main__'" in code

            # Calculate complexity (simplified)
            complexity = 1
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.For, ast.While, ast.ExceptHandler)):
                    complexity += 1
                elif isinstance(node, ast.BoolOp):
                    complexity += len(node.values) - 1
            features.cyclomatic_complexity = complexity

            # Calculate nesting depth
            def get_depth(node, current=0):
                max_depth = current
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
                        max_depth = max(max_depth, get_depth(child, current + 1))
                    else:
                        max_depth = max(max_depth, get_depth(child, current))
                return max_depth

            features.max_nesting_depth = get_depth(tree)

        except SyntaxError:
            pass  # Return default features if parsing fails

    return features


class CodeAnalyzer:
    """
    Analyzes code using multiple static analysis tools.

    Combines results from linters, type checkers, and security scanners.
    """

    def __init__(
        self,
        use_ruff: bool = True,
        use_mypy: bool = True,
        use_bandit: bool = True,
        use_semgrep: bool = False,
    ):
        self.use_ruff = use_ruff
        self.use_mypy = use_mypy
        self.use_bandit = use_bandit
        self.use_semgrep = use_semgrep

        # Check tool availability
        self._available_tools = self._check_tools()

    def _check_tools(self) -> dict[str, bool]:
        """Check which tools are available."""
        tools = {}

        for tool in ["ruff", "mypy", "bandit", "semgrep"]:
            try:
                result = subprocess.run(
                    [tool, "--version"],
                    capture_output=True,
                    timeout=5,
                )
                tools[tool] = result.returncode == 0
            except (subprocess.TimeoutExpired, FileNotFoundError):
                tools[tool] = False

        return tools

    def analyze(
        self,
        code: str,
        language: Language = Language.PYTHON,
        filename: str | None = None,
    ) -> StaticAnalysisResult:
        """
        Run static analysis on code.

        Args:
            code: Source code to analyze
            language: Programming language
            filename: Optional filename for context

        Returns:
            StaticAnalysisResult with issues and metrics
        """
        result = StaticAnalysisResult()

        if language != Language.PYTHON:
            # For now, only Python has full analysis support
            result.tool_outputs["note"] = f"Limited analysis for {language.value}"
            return result

        # Write code to temp file for tools that need it
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
        ) as f:
            f.write(code)
            temp_path = f.name

        try:
            # Ruff (fast linter)
            if self.use_ruff and self._available_tools.get("ruff"):
                ruff_issues = self._run_ruff(temp_path)
                result.issues.extend(ruff_issues)
                result.style_score -= len([i for i in ruff_issues if i.dimension == CodeDimension.STYLE]) * 0.5
                result.style_score = max(0, result.style_score)

            # Mypy (type checker)
            if self.use_mypy and self._available_tools.get("mypy"):
                mypy_issues = self._run_mypy(temp_path)
                result.issues.extend(mypy_issues)
                result.correctness_score -= len(mypy_issues) * 0.5
                result.correctness_score = max(0, result.correctness_score)

            # Bandit (security)
            if self.use_bandit and self._available_tools.get("bandit"):
                bandit_issues = self._run_bandit(temp_path)
                result.issues.extend(bandit_issues)
                # Security issues are weighted more heavily
                high_severity = len([i for i in bandit_issues if i.severity == "error"])
                result.security_score -= high_severity * 2.0
                result.security_score -= len(bandit_issues) * 0.5
                result.security_score = max(0, result.security_score)

            # Extract features for complexity score
            features = extract_code_features(code, language)
            if features.cyclomatic_complexity > 20:
                result.complexity_score -= (features.cyclomatic_complexity - 20) * 0.2
            if features.max_nesting_depth > 4:
                result.complexity_score -= (features.max_nesting_depth - 4) * 1.0
            result.complexity_score = max(0, result.complexity_score)

            result.metrics = {
                "lines": features.num_lines,
                "functions": features.num_functions,
                "classes": features.num_classes,
                "complexity": features.cyclomatic_complexity,
                "max_depth": features.max_nesting_depth,
            }

        finally:
            # Clean up temp file
            Path(temp_path).unlink(missing_ok=True)

        return result

    def _run_ruff(self, filepath: str) -> list[CodeIssue]:
        """Run ruff linter."""
        issues = []

        try:
            result = subprocess.run(
                ["ruff", "check", "--output-format=json", filepath],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.stdout:
                try:
                    data = json.loads(result.stdout)
                    for item in data:
                        issues.append(CodeIssue(
                            line=item.get("location", {}).get("row", 0),
                            column=item.get("location", {}).get("column", 0),
                            message=item.get("message", ""),
                            severity="warning",
                            dimension=CodeDimension.STYLE,
                            rule_id=item.get("code", ""),
                        ))
                except json.JSONDecodeError as e:
                    logger.debug(f"Failed to parse ruff output: {e}")

        except subprocess.TimeoutExpired:
            logger.debug(f"Ruff timed out analyzing {filepath}")
        except FileNotFoundError:
            logger.debug("Ruff not available - skipping linting")

        return issues

    def _run_mypy(self, filepath: str) -> list[CodeIssue]:
        """Run mypy type checker."""
        issues = []

        try:
            result = subprocess.run(
                ["mypy", "--no-error-summary", "--show-column-numbers", filepath],
                capture_output=True,
                text=True,
                timeout=60,
            )

            for line in result.stdout.split("\n"):
                # Parse mypy output: file:line:col: severity: message
                match = re.match(r".+:(\d+):(\d+): (\w+): (.+)", line)
                if match:
                    line_num, col, severity, message = match.groups()
                    issues.append(CodeIssue(
                        line=int(line_num),
                        column=int(col),
                        message=message,
                        severity="error" if severity == "error" else "warning",
                        dimension=CodeDimension.CORRECTNESS,
                        rule_id="mypy",
                    ))

        except subprocess.TimeoutExpired:
            logger.debug(f"Mypy timed out analyzing {filepath}")
        except FileNotFoundError:
            logger.debug("Mypy not available - skipping type checking")

        return issues

    def _run_bandit(self, filepath: str) -> list[CodeIssue]:
        """Run bandit security scanner."""
        issues = []

        try:
            result = subprocess.run(
                ["bandit", "-f", "json", "-q", filepath],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.stdout:
                try:
                    data = json.loads(result.stdout)
                    for item in data.get("results", []):
                        severity_map = {
                            "HIGH": "error",
                            "MEDIUM": "warning",
                            "LOW": "info",
                        }
                        issues.append(CodeIssue(
                            line=item.get("line_number", 0),
                            column=0,
                            message=item.get("issue_text", ""),
                            severity=severity_map.get(item.get("issue_severity", ""), "warning"),
                            dimension=CodeDimension.SECURITY,
                            rule_id=item.get("test_id", ""),
                        ))
                except json.JSONDecodeError as e:
                    logger.debug(f"Failed to parse bandit output: {e}")

        except subprocess.TimeoutExpired:
            logger.debug(f"Bandit timed out analyzing {filepath}")
        except FileNotFoundError:
            logger.debug("Bandit not available - skipping security scanning")

        return issues


def quick_analyze(code: str, language: Language | None = None) -> dict[str, Any]:
    """
    Quick analysis without external tools.

    Useful when static analysis tools aren't available.
    """
    if language is None:
        language = detect_language(code)

    features = extract_code_features(code, language)
    parsed = parse_code(code, language)

    # Simple heuristic scoring
    scores = {
        "correctness": 10.0 if parsed["success"] else 3.0,
        "style": 8.0,
        "complexity": 10.0,
    }

    # Deduct for complexity
    if features.cyclomatic_complexity > 10:
        scores["complexity"] -= (features.cyclomatic_complexity - 10) * 0.3

    if features.max_nesting_depth > 3:
        scores["complexity"] -= (features.max_nesting_depth - 3) * 0.5

    # Bonus for good practices
    if features.has_docstrings:
        scores["style"] += 0.5
    if features.has_type_hints:
        scores["style"] += 0.5
    if features.has_error_handling:
        scores["correctness"] += 0.5

    # Cap scores
    scores = {k: max(0, min(10, v)) for k, v in scores.items()}

    return {
        "language": language.value,
        "features": features,
        "scores": scores,
        "parsed": parsed["success"],
        "overall": sum(scores.values()) / len(scores),
    }
