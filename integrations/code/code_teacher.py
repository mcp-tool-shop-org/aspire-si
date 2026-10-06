"""
Code Teachers - Experts that critique generated code.

Each teacher persona evaluates code from a different perspective:
- CorrectnessChecker: Does the code work? Any bugs?
- StyleGuide: Is it idiomatic? Readable?
- SecurityAuditor: Any vulnerabilities? Safe inputs?
- ArchitectureReviewer: Good structure? SOLID principles?
- PerformanceAnalyst: Efficient? Scalable?
- DocumentationCritic: Well explained? Good docstrings?
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

from .config import Language, CodeDimension
from .analysis import (
    CodeAnalyzer,
    StaticAnalysisResult,
    extract_code_features,
    parse_code,
    quick_analyze,
)


@dataclass
class CodeCritique:
    """A teacher's evaluation of code."""

    # Overall assessment
    overall_score: float  # 0-10

    # Per-dimension scores
    dimension_scores: dict[CodeDimension, float] = field(default_factory=dict)

    # Detailed reasoning
    reasoning: str = ""

    # Specific feedback
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)

    # Suggested improvements
    suggestions: list[str] = field(default_factory=list)

    # Line-specific comments
    line_comments: dict[int, str] = field(default_factory=dict)

    # Suggested fix (if applicable)
    suggested_fix: str | None = None

    # Metadata
    teacher_name: str = ""
    language: Language = Language.UNKNOWN
    confidence: float = 1.0


@dataclass
class CodeSample:
    """A code sample to be evaluated."""

    code: str
    language: Language = Language.UNKNOWN
    filename: str | None = None
    context: str | None = None  # Surrounding code or description
    task: str | None = None     # What the code is supposed to do


class BaseCodeTeacher(ABC):
    """Base class for code teachers."""

    def __init__(
        self,
        name: str,
        description: str,
        focus_dimensions: list[CodeDimension],
    ):
        self.name = name
        self.description = description
        self.focus_dimensions = focus_dimensions

    @abstractmethod
    def critique(self, sample: CodeSample) -> CodeCritique:
        """Evaluate code and provide detailed feedback."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


class CorrectnessChecker(BaseCodeTeacher):
    """
    Evaluates code for correctness.

    Focuses on:
    - Syntax errors
    - Type errors
    - Logic bugs
    - Edge cases
    - Error handling
    """

    def __init__(self, use_static_analysis: bool = True):
        super().__init__(
            name="Correctness Checker",
            description="Ensures code is functionally correct",
            focus_dimensions=[CodeDimension.CORRECTNESS],
        )
        self.use_static_analysis = use_static_analysis
        if use_static_analysis:
            self.analyzer = CodeAnalyzer(use_ruff=True, use_mypy=True, use_bandit=False)

    def critique(self, sample: CodeSample) -> CodeCritique:
        """Evaluate code correctness."""
        strengths = []
        weaknesses = []
        suggestions = []
        line_comments = {}

        score = 10.0

        # Parse the code
        parsed = parse_code(sample.code, sample.language)

        if not parsed["success"]:
            score = 2.0
            weaknesses.append(f"Syntax error: {parsed.get('error', 'Unknown')}")
            suggestions.append("Fix syntax errors before other improvements")

            return CodeCritique(
                overall_score=score,
                dimension_scores={CodeDimension.CORRECTNESS: score},
                reasoning="Code has syntax errors and cannot be parsed.",
                weaknesses=weaknesses,
                suggestions=suggestions,
                teacher_name=self.name,
                language=sample.language,
            )

        # Run static analysis if available
        if self.use_static_analysis and sample.language == Language.PYTHON:
            analysis = self.analyzer.analyze(sample.code, sample.language)

            # Process issues
            for issue in analysis.issues:
                if issue.dimension == CodeDimension.CORRECTNESS:
                    if issue.severity == "error":
                        score -= 1.5
                        weaknesses.append(f"Line {issue.line}: {issue.message}")
                    else:
                        score -= 0.5

                    line_comments[issue.line] = issue.message

            if not analysis.issues:
                strengths.append("No static analysis errors detected")

        # Check for common issues
        features = extract_code_features(sample.code, sample.language)

        if not features.has_error_handling:
            score -= 0.5
            suggestions.append("Consider adding error handling for robustness")

        # Check for obvious issues
        code_lower = sample.code.lower()

        # Potential infinite loops
        if "while true" in code_lower and "break" not in code_lower:
            score -= 1.0
            weaknesses.append("Potential infinite loop: while True without break")

        # Division by zero risk
        if "/ 0" in sample.code or "/0" in sample.code:
            score -= 2.0
            weaknesses.append("Possible division by zero")

        # Unused variables (simple check)
        if sample.language == Language.PYTHON and parsed.get("ast"):
            import ast
            tree = parsed["ast"]

            assigned = set()
            used = set()

            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            assigned.add(target.id)
                elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    used.add(node.id)

            unused = assigned - used - {"_"}
            if unused:
                score -= 0.3 * len(unused)
                suggestions.append(f"Unused variables: {', '.join(list(unused)[:3])}")

        score = max(0.0, min(10.0, score))

        # Generate reasoning
        if score >= 8.0:
            reasoning = "Code appears correct with no significant issues. "
        elif score >= 5.0:
            reasoning = "Code has some correctness concerns that should be addressed. "
        else:
            reasoning = "Code has significant correctness issues. "

        if weaknesses:
            reasoning += f"Main issues: {'; '.join(weaknesses[:2])}."

        return CodeCritique(
            overall_score=score,
            dimension_scores={CodeDimension.CORRECTNESS: score},
            reasoning=reasoning,
            strengths=strengths,
            weaknesses=weaknesses,
            suggestions=suggestions,
            line_comments=line_comments,
            teacher_name=self.name,
            language=sample.language,
        )


class StyleGuide(BaseCodeTeacher):
    """
    Evaluates code for style and readability.

    Focuses on:
    - PEP8 / language conventions
    - Naming conventions
    - Code organization
    - Readability
    - Idiomatic patterns
    """

    def __init__(self, use_static_analysis: bool = True):
        super().__init__(
            name="Style Guide",
            description="Ensures code follows best practices and is readable",
            focus_dimensions=[
                CodeDimension.STYLE,
                CodeDimension.MAINTAINABILITY,
            ],
        )
        self.use_static_analysis = use_static_analysis
        if use_static_analysis:
            self.analyzer = CodeAnalyzer(use_ruff=True, use_mypy=False, use_bandit=False)

    def critique(self, sample: CodeSample) -> CodeCritique:
        """Evaluate code style."""
        strengths = []
        weaknesses = []
        suggestions = []

        score = 10.0

        # Run linter
        if self.use_static_analysis and sample.language == Language.PYTHON:
            analysis = self.analyzer.analyze(sample.code, sample.language)

            style_issues = [i for i in analysis.issues if i.dimension == CodeDimension.STYLE]
            if style_issues:
                score -= min(3.0, len(style_issues) * 0.3)
                weaknesses.append(f"{len(style_issues)} style issues detected")
            else:
                strengths.append("Passes style checks")

        features = extract_code_features(sample.code, sample.language)

        # Check for good practices
        if features.has_docstrings:
            strengths.append("Has docstrings")
        else:
            score -= 1.0
            suggestions.append("Add docstrings to functions and classes")

        if features.has_type_hints:
            strengths.append("Uses type hints")
        else:
            score -= 0.5
            suggestions.append("Consider adding type hints for clarity")

        # Check naming conventions (Python)
        if sample.language == Language.PYTHON:
            import re

            # Check for camelCase (should be snake_case in Python)
            camel_case = re.findall(r'\b[a-z]+[A-Z][a-z]+\b', sample.code)
            if camel_case:
                score -= 0.5
                weaknesses.append(f"Use snake_case instead of camelCase: {', '.join(camel_case[:3])}")

            # Check for ALLCAPS that aren't constants
            lines = sample.code.split('\n')
            for i, line in enumerate(lines, 1):
                if '=' in line and not line.strip().startswith('#'):
                    # Simple check for non-constant ALLCAPS
                    match = re.search(r'\b([A-Z]{2,})\s*=\s*[^A-Z]', line)
                    if match and match.group(1) not in ['ID', 'URL', 'API', 'SQL', 'HTML', 'JSON', 'XML']:
                        score -= 0.2

        # Check line length
        long_lines = [i for i, line in enumerate(sample.code.split('\n'), 1) if len(line) > 100]
        if long_lines:
            score -= 0.3 * min(len(long_lines), 3)
            suggestions.append(f"Lines too long (>100 chars): {long_lines[:3]}")

        # Check for magic numbers
        import re
        magic_numbers = re.findall(r'[^0-9.](\d{2,})[^0-9.]', sample.code)
        magic_numbers = [n for n in magic_numbers if n not in ['10', '100', '1000']]
        if magic_numbers:
            score -= 0.3
            suggestions.append("Consider using named constants instead of magic numbers")

        score = max(0.0, min(10.0, score))

        # Generate reasoning
        if score >= 8.0:
            reasoning = "Code follows good style conventions and is readable. "
        elif score >= 5.0:
            reasoning = "Code style could be improved for better readability. "
        else:
            reasoning = "Code has significant style issues affecting maintainability. "

        return CodeCritique(
            overall_score=score,
            dimension_scores={
                CodeDimension.STYLE: score,
                CodeDimension.MAINTAINABILITY: score * 0.9,
            },
            reasoning=reasoning,
            strengths=strengths,
            weaknesses=weaknesses,
            suggestions=suggestions,
            teacher_name=self.name,
            language=sample.language,
        )


class SecurityAuditor(BaseCodeTeacher):
    """
    Evaluates code for security vulnerabilities.

    Focuses on:
    - Injection attacks (SQL, command, etc.)
    - Input validation
    - Sensitive data exposure
    - Authentication/authorization issues
    - Cryptographic weaknesses
    """

    def __init__(self, use_static_analysis: bool = True):
        super().__init__(
            name="Security Auditor",
            description="Identifies security vulnerabilities and risks",
            focus_dimensions=[CodeDimension.SECURITY],
        )
        self.use_static_analysis = use_static_analysis
        if use_static_analysis:
            self.analyzer = CodeAnalyzer(use_ruff=False, use_mypy=False, use_bandit=True)

    # Common dangerous patterns
    DANGEROUS_PATTERNS = {
        "eval(": ("Code injection risk", "Never use eval() with untrusted input"),
        "exec(": ("Code injection risk", "Avoid exec() - use safer alternatives"),
        "os.system(": ("Command injection risk", "Use subprocess with shell=False"),
        "shell=True": ("Command injection risk", "Avoid shell=True in subprocess"),
        "pickle.load": ("Deserialization attack risk", "Don't unpickle untrusted data"),
        "yaml.load(": ("YAML deserialization risk", "Use yaml.safe_load() instead"),
        "SELECT.*%s": ("SQL injection risk", "Use parameterized queries"),
        "f\"SELECT": ("SQL injection risk", "Never use f-strings for SQL"),
        "password": ("Sensitive data exposure", "Don't hardcode passwords"),
        "secret": ("Sensitive data exposure", "Use environment variables for secrets"),
        "api_key": ("Sensitive data exposure", "Don't commit API keys"),
        "md5(": ("Weak cryptography", "Use SHA-256 or better"),
        "sha1(": ("Weak cryptography", "Use SHA-256 or better"),
        "random.random": ("Weak randomness", "Use secrets module for security"),
        ".verify = False": ("SSL verification disabled", "Never disable SSL verification"),
        "CORS(app)": ("Potential CORS misconfiguration", "Configure CORS restrictively"),
    }

    def critique(self, sample: CodeSample) -> CodeCritique:
        """Evaluate code security."""
        strengths = []
        weaknesses = []
        suggestions = []
        line_comments = {}

        score = 10.0

        # Run bandit if available
        if self.use_static_analysis and sample.language == Language.PYTHON:
            analysis = self.analyzer.analyze(sample.code, sample.language)

            for issue in analysis.issues:
                if issue.dimension == CodeDimension.SECURITY:
                    if issue.severity == "error":
                        score -= 2.0
                        weaknesses.append(f"HIGH: {issue.message}")
                    elif issue.severity == "warning":
                        score -= 1.0
                        weaknesses.append(f"MEDIUM: {issue.message}")
                    else:
                        score -= 0.3

                    line_comments[issue.line] = f"SECURITY: {issue.message}"

        # Check for dangerous patterns
        code_lower = sample.code.lower()
        lines = sample.code.split('\n')

        for pattern, (risk, suggestion) in self.DANGEROUS_PATTERNS.items():
            if pattern.lower() in code_lower:
                # Find the line number
                for i, line in enumerate(lines, 1):
                    if pattern.lower() in line.lower():
                        score -= 1.5
                        weaknesses.append(f"Line {i}: {risk}")
                        suggestions.append(suggestion)
                        line_comments[i] = f"SECURITY: {risk}"
                        break

        # Check for input validation
        if "input(" in sample.code and "validate" not in code_lower and "check" not in code_lower:
            score -= 0.5
            suggestions.append("Validate user input before processing")

        # Check for proper error handling of sensitive operations
        sensitive_ops = ["open(", "connect(", "execute(", "request"]
        has_sensitive = any(op in sample.code for op in sensitive_ops)
        features = extract_code_features(sample.code, sample.language)

        if has_sensitive and not features.has_error_handling:
            score -= 0.5
            suggestions.append("Add error handling for sensitive operations")

        # Positive checks
        if "secrets." in sample.code:
            strengths.append("Uses secrets module for secure randomness")
        if "hashlib.sha256" in sample.code or "hashlib.sha512" in sample.code:
            strengths.append("Uses strong cryptographic hashing")
        if "paramstyle" in code_lower or "?" in sample.code and "execute" in sample.code:
            strengths.append("Appears to use parameterized queries")

        score = max(0.0, min(10.0, score))

        # Generate reasoning
        if score >= 9.0:
            reasoning = "Code follows security best practices. "
        elif score >= 7.0:
            reasoning = "Code has minor security considerations to address. "
        elif score >= 5.0:
            reasoning = "Code has security issues that should be fixed before deployment. "
        else:
            reasoning = "CRITICAL: Code has serious security vulnerabilities. "

        return CodeCritique(
            overall_score=score,
            dimension_scores={CodeDimension.SECURITY: score},
            reasoning=reasoning,
            strengths=strengths,
            weaknesses=weaknesses,
            suggestions=suggestions,
            line_comments=line_comments,
            teacher_name=self.name,
            language=sample.language,
        )


class ArchitectureReviewer(BaseCodeTeacher):
    """
    Evaluates code architecture and design.

    Focuses on:
    - SOLID principles
    - Separation of concerns
    - Code organization
    - Coupling and cohesion
    - Design patterns
    """

    def __init__(self):
        super().__init__(
            name="Architecture Reviewer",
            description="Evaluates code structure and design principles",
            focus_dimensions=[
                CodeDimension.ARCHITECTURE,
                CodeDimension.MAINTAINABILITY,
            ],
        )

    def critique(self, sample: CodeSample) -> CodeCritique:
        """Evaluate code architecture."""
        strengths = []
        weaknesses = []
        suggestions = []

        score = 10.0
        features = extract_code_features(sample.code, sample.language)

        # Check function length
        if sample.language == Language.PYTHON:
            parsed = parse_code(sample.code, sample.language)
            if parsed["success"] and parsed.get("ast"):
                import ast
                tree = parsed["ast"]

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        func_lines = node.end_lineno - node.lineno if hasattr(node, 'end_lineno') else 0
                        if func_lines > 50:
                            score -= 0.5
                            weaknesses.append(f"Function '{node.name}' is too long ({func_lines} lines)")
                            suggestions.append(f"Consider breaking '{node.name}' into smaller functions")
                        elif func_lines > 30:
                            score -= 0.2

        # Check complexity
        if features.cyclomatic_complexity > 15:
            score -= 1.5
            weaknesses.append(f"High cyclomatic complexity: {features.cyclomatic_complexity}")
            suggestions.append("Reduce complexity by extracting methods")
        elif features.cyclomatic_complexity > 10:
            score -= 0.5
            suggestions.append("Consider simplifying complex logic")
        else:
            strengths.append("Good complexity level")

        # Check nesting
        if features.max_nesting_depth > 4:
            score -= 1.0
            weaknesses.append(f"Deep nesting: {features.max_nesting_depth} levels")
            suggestions.append("Use early returns or extract nested logic")
        elif features.max_nesting_depth <= 2:
            strengths.append("Flat, readable structure")

        # Check for god objects (classes doing too much)
        if sample.language == Language.PYTHON:
            parsed = parse_code(sample.code, sample.language)
            if parsed["success"] and parsed.get("ast"):
                import ast
                tree = parsed["ast"]

                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
                        if len(methods) > 15:
                            score -= 1.0
                            weaknesses.append(f"Class '{node.name}' has too many methods ({len(methods)})")
                            suggestions.append("Consider splitting into smaller classes (Single Responsibility)")
                        elif len(methods) > 10:
                            score -= 0.3

        # Check for separation of concerns
        concerns = {
            "io": ["print(", "input(", "open(", "read(", "write("],
            "network": ["request", "http", "socket", "urllib"],
            "database": ["execute", "cursor", "commit", "SELECT", "INSERT"],
            "ui": ["tkinter", "pygame", "flask", "django"],
        }

        detected_concerns = []
        for concern, patterns in concerns.items():
            if any(p in sample.code for p in patterns):
                detected_concerns.append(concern)

        if len(detected_concerns) > 2 and features.num_classes <= 1:
            score -= 0.5
            suggestions.append(f"Multiple concerns in one place: {', '.join(detected_concerns)}. Consider separating.")

        score = max(0.0, min(10.0, score))

        # Generate reasoning
        if score >= 8.0:
            reasoning = "Code has good architecture with clear structure. "
        elif score >= 5.0:
            reasoning = "Architecture could be improved for better maintainability. "
        else:
            reasoning = "Significant architectural issues affecting code quality. "

        return CodeCritique(
            overall_score=score,
            dimension_scores={
                CodeDimension.ARCHITECTURE: score,
                CodeDimension.MAINTAINABILITY: score * 0.9,
            },
            reasoning=reasoning,
            strengths=strengths,
            weaknesses=weaknesses,
            suggestions=suggestions,
            teacher_name=self.name,
            language=sample.language,
        )


class PerformanceAnalyst(BaseCodeTeacher):
    """
    Evaluates code for performance.

    Focuses on:
    - Time complexity
    - Space complexity
    - Common performance pitfalls
    - Optimization opportunities
    """

    def __init__(self):
        super().__init__(
            name="Performance Analyst",
            description="Identifies performance issues and optimization opportunities",
            focus_dimensions=[CodeDimension.PERFORMANCE],
        )

    # Common performance anti-patterns
    ANTI_PATTERNS = {
        r"for.*in.*for.*in": ("Nested loops", "Consider if O(n^2) is necessary"),
        r"\+\s*=.*str": ("String concatenation in loop", "Use list and join() instead"),
        r"append.*for": ("Append in loop", "Consider list comprehension"),
        r"in\s+list\(": ("Converting to list unnecessarily", "Iterate directly over generator"),
        r"range\(len\(": ("range(len()) pattern", "Use enumerate() instead"),
        r"\.keys\(\)\s*\)": ("Iterating over .keys()", "Iterate over dict directly"),
        r"global ": ("Global variables", "Consider passing as parameters"),
        r"import.*\*": ("Wildcard import", "Import only what's needed"),
    }

    def critique(self, sample: CodeSample) -> CodeCritique:
        """Evaluate code performance."""
        import re

        strengths = []
        weaknesses = []
        suggestions = []

        score = 10.0

        # Check for anti-patterns
        for pattern, (issue, suggestion) in self.ANTI_PATTERNS.items():
            if re.search(pattern, sample.code, re.IGNORECASE):
                score -= 0.5
                weaknesses.append(issue)
                suggestions.append(suggestion)

        # Check for common inefficiencies
        lines = sample.code.split('\n')

        # Repeated function calls that could be cached
        import re
        func_calls = re.findall(r'\b(\w+)\([^)]*\)', sample.code)
        call_counts = {}
        for call in func_calls:
            call_counts[call] = call_counts.get(call, 0) + 1

        repeated = [f for f, c in call_counts.items() if c > 3 and f not in ['print', 'len', 'str', 'int', 'range']]
        if repeated:
            suggestions.append(f"Consider caching repeated calls: {', '.join(repeated[:3])}")

        # Check for list operations that could be sets
        if '.append(' in sample.code and ' in ' in sample.code:
            suggestions.append("If checking membership, consider using a set instead of list")

        # Check for comprehension opportunities
        if 'for ' in sample.code and '.append(' in sample.code:
            # Simple pattern: result = []; for x in y: result.append(...)
            if re.search(r'\[\s*\]\s*\n.*for.*\.append', sample.code, re.DOTALL):
                score -= 0.3
                suggestions.append("Consider using list comprehension instead of loop + append")

        # Positive patterns
        if 'yield ' in sample.code:
            strengths.append("Uses generators for memory efficiency")
        if '@lru_cache' in sample.code or '@cache' in sample.code:
            strengths.append("Uses caching for expensive operations")
        if 'numpy' in sample.code or 'np.' in sample.code:
            strengths.append("Uses NumPy for efficient array operations")
        if ' set(' in sample.code or '= set()' in sample.code:
            strengths.append("Uses sets for efficient membership testing")

        # Check for obvious O(n^2) patterns
        nested_loops = len(re.findall(r'for\s+\w+\s+in.*:\s*\n\s*for\s+\w+\s+in', sample.code))
        if nested_loops > 1:
            score -= 1.0
            weaknesses.append(f"Multiple nested loops detected ({nested_loops})")
            suggestions.append("Review nested loops for potential O(n^2) issues")

        score = max(0.0, min(10.0, score))

        # Generate reasoning
        if score >= 8.0:
            reasoning = "Code appears performant with no obvious issues. "
        elif score >= 5.0:
            reasoning = "Some performance optimizations possible. "
        else:
            reasoning = "Significant performance concerns detected. "

        return CodeCritique(
            overall_score=score,
            dimension_scores={CodeDimension.PERFORMANCE: score},
            reasoning=reasoning,
            strengths=strengths,
            weaknesses=weaknesses,
            suggestions=suggestions,
            teacher_name=self.name,
            language=sample.language,
        )


class DocumentationCritic(BaseCodeTeacher):
    """
    Evaluates code documentation.

    Focuses on:
    - Docstrings
    - Comments
    - README/usage examples
    - Type hints as documentation
    """

    def __init__(self):
        super().__init__(
            name="Documentation Critic",
            description="Evaluates code documentation and explanations",
            focus_dimensions=[CodeDimension.DOCUMENTATION],
        )

    def critique(self, sample: CodeSample) -> CodeCritique:
        """Evaluate code documentation."""
        strengths = []
        weaknesses = []
        suggestions = []

        score = 10.0
        features = extract_code_features(sample.code, sample.language)

        if sample.language == Language.PYTHON:
            parsed = parse_code(sample.code, sample.language)

            if parsed["success"] and parsed.get("ast"):
                import ast
                tree = parsed["ast"]

                # Check module docstring
                if tree.body and isinstance(tree.body[0], ast.Expr):
                    if isinstance(tree.body[0].value, ast.Constant) and isinstance(tree.body[0].value.value, str):
                        strengths.append("Has module docstring")
                    else:
                        score -= 0.5
                        suggestions.append("Add a module-level docstring")
                else:
                    score -= 0.5
                    suggestions.append("Add a module-level docstring")

                # Check function docstrings
                functions_without_docs = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        has_doc = (
                            node.body and
                            isinstance(node.body[0], ast.Expr) and
                            isinstance(node.body[0].value, ast.Constant) and
                            isinstance(node.body[0].value.value, str)
                        )
                        if not has_doc and not node.name.startswith('_'):
                            functions_without_docs.append(node.name)

                if functions_without_docs:
                    score -= min(2.0, len(functions_without_docs) * 0.3)
                    weaknesses.append(f"Functions without docstrings: {', '.join(functions_without_docs[:5])}")
                elif features.num_functions > 0:
                    strengths.append("All public functions have docstrings")

        # Check for type hints
        if features.has_type_hints:
            strengths.append("Uses type hints for self-documentation")
        elif features.num_functions > 0:
            score -= 0.5
            suggestions.append("Add type hints to function signatures")

        # Check comment quality
        lines = sample.code.split('\n')
        comment_lines = [l for l in lines if l.strip().startswith('#')]
        code_lines = [l for l in lines if l.strip() and not l.strip().startswith('#')]

        if code_lines:
            comment_ratio = len(comment_lines) / len(code_lines)

            if comment_ratio < 0.05 and len(code_lines) > 20:
                score -= 0.5
                suggestions.append("Add more inline comments for complex logic")
            elif comment_ratio > 0.5:
                score -= 0.3
                weaknesses.append("Possibly over-commented - code should be self-documenting")

        # Check for TODO/FIXME without explanation
        todos = [l for l in lines if 'TODO' in l or 'FIXME' in l]
        unclear_todos = [l for l in todos if len(l.split('TODO')[-1].strip()) < 10 or len(l.split('FIXME')[-1].strip()) < 10]
        if unclear_todos:
            score -= 0.2 * len(unclear_todos)
            suggestions.append("Add explanations to TODO/FIXME comments")

        score = max(0.0, min(10.0, score))

        # Generate reasoning
        if score >= 8.0:
            reasoning = "Code is well-documented and self-explanatory. "
        elif score >= 5.0:
            reasoning = "Documentation could be improved. "
        else:
            reasoning = "Code lacks sufficient documentation. "

        return CodeCritique(
            overall_score=score,
            dimension_scores={CodeDimension.DOCUMENTATION: score},
            reasoning=reasoning,
            strengths=strengths,
            weaknesses=weaknesses,
            suggestions=suggestions,
            teacher_name=self.name,
            language=sample.language,
        )


class LLMCodeTeacher(BaseCodeTeacher):
    """
    Teacher that uses an LLM (Claude/GPT-4) for deep code review.

    Provides nuanced feedback that static analysis can't catch.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        focus_dimensions: list[CodeDimension] | None = None,
    ):
        super().__init__(
            name="LLM Code Reviewer",
            description="Deep code review using large language models",
            focus_dimensions=focus_dimensions or list(CodeDimension),
        )
        self.model = model
        self._client = None

    def _get_client(self):
        """Lazy initialization of API client."""
        if self._client is None:
            if "claude" in self.model.lower():
                try:
                    import anthropic
                    self._client = anthropic.Anthropic()
                except ImportError:
                    raise ImportError("pip install anthropic")
            else:
                try:
                    import openai
                    self._client = openai.OpenAI()
                except ImportError:
                    raise ImportError("pip install openai")
        return self._client

    def critique(self, sample: CodeSample) -> CodeCritique:
        """Get LLM critique of code."""
        client = self._get_client()

        prompt = f"""Review this {sample.language.value} code and provide detailed feedback.

Code:
```{sample.language.value}
{sample.code}
```

{f"Task: {sample.task}" if sample.task else ""}
{f"Context: {sample.context}" if sample.context else ""}

Evaluate on:
1. Correctness - Does it work? Any bugs?
2. Style - Is it idiomatic and readable?
3. Security - Any vulnerabilities?
4. Performance - Any inefficiencies?
5. Architecture - Good structure?

Provide:
- Overall score (0-10)
- Specific strengths
- Specific weaknesses
- Actionable suggestions

Be concise but thorough."""

        try:
            if "claude" in self.model.lower():
                response = client.messages.create(
                    model=self.model,
                    max_tokens=1000,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = response.content[0].text
            else:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1000,
                )
                text = response.choices[0].message.content

            # Parse the response (simplified)
            score = 7.0  # Default
            import re
            score_match = re.search(r'(\d+(?:\.\d+)?)\s*/\s*10', text)
            if score_match:
                score = float(score_match.group(1))

            return CodeCritique(
                overall_score=min(10, max(0, score)),
                reasoning=text[:500],  # Truncate for storage
                teacher_name=self.name,
                language=sample.language,
            )

        except Exception as e:
            return CodeCritique(
                overall_score=5.0,
                reasoning=f"LLM review failed: {str(e)}",
                teacher_name=self.name,
                language=sample.language,
                confidence=0.0,
            )


class CodeTeacher:
    """
    Composite code teacher that combines multiple perspectives.

    Like ASPIRE's CompositeTeacher, but specialized for code.
    """

    PERSONA_MAP = {
        "correctness_checker": CorrectnessChecker,
        "style_guide": StyleGuide,
        "security_auditor": SecurityAuditor,
        "architecture_reviewer": ArchitectureReviewer,
        "performance_analyst": PerformanceAnalyst,
        "documentation_critic": DocumentationCritic,
    }

    def __init__(
        self,
        personas: list[str] | None = None,
        strategy: Literal["vote", "rotate", "debate"] = "vote",
        weights: dict[str, float] | None = None,
        use_llm: bool = False,
        llm_model: str = "claude-sonnet-4-20250514",
    ):
        if personas is None:
            personas = ["correctness_checker", "style_guide", "security_auditor"]

        self.teachers = []
        for persona in personas:
            if persona in self.PERSONA_MAP:
                self.teachers.append(self.PERSONA_MAP[persona]())
            else:
                raise ValueError(f"Unknown persona: {persona}")

        if use_llm:
            self.teachers.append(LLMCodeTeacher(model=llm_model))

        self.strategy = strategy
        self.weights = weights or {t.name: 1.0 for t in self.teachers}
        self._turn_idx = 0

    def critique(self, sample: CodeSample) -> CodeCritique:
        """Get combined critique from all teachers."""

        # Detect language if not specified
        if sample.language == Language.UNKNOWN:
            from .analysis import detect_language
            sample.language = detect_language(sample.code, sample.filename)

        if self.strategy == "rotate":
            teacher = self.teachers[self._turn_idx % len(self.teachers)]
            self._turn_idx += 1
            return teacher.critique(sample)

        elif self.strategy == "vote":
            critiques = [t.critique(sample) for t in self.teachers]

            total_weight = sum(self.weights.get(c.teacher_name, 1.0) for c in critiques)

            weighted_score = sum(
                c.overall_score * self.weights.get(c.teacher_name, 1.0)
                for c in critiques
            ) / total_weight

            # Combine dimension scores
            all_dimensions = set()
            for c in critiques:
                all_dimensions.update(c.dimension_scores.keys())

            combined_dimensions = {}
            for dim in all_dimensions:
                dim_scores = [c.dimension_scores.get(dim, c.overall_score) for c in critiques]
                combined_dimensions[dim] = sum(dim_scores) / len(dim_scores)

            # Combine feedback
            all_strengths = []
            all_weaknesses = []
            all_suggestions = []
            all_line_comments = {}

            for c in critiques:
                all_strengths.extend(c.strengths)
                all_weaknesses.extend(c.weaknesses)
                all_suggestions.extend(c.suggestions)
                all_line_comments.update(c.line_comments)

            # Generate combined reasoning
            reasoning_parts = [f"{c.teacher_name}: {c.reasoning}" for c in critiques]
            combined_reasoning = " | ".join(reasoning_parts)

            return CodeCritique(
                overall_score=weighted_score,
                dimension_scores=combined_dimensions,
                reasoning=combined_reasoning[:1000],  # Truncate
                strengths=list(set(all_strengths)),
                weaknesses=list(set(all_weaknesses)),
                suggestions=list(set(all_suggestions)),
                line_comments=all_line_comments,
                teacher_name="Code Teacher Committee",
                language=sample.language,
            )

        elif self.strategy == "debate":
            # Simplified: weight toward consensus
            critiques = [t.critique(sample) for t in self.teachers]
            scores = [c.overall_score for c in critiques]

            import numpy as np
            mean_score = np.mean(scores)
            std_score = np.std(scores)

            if std_score > 0:
                consensus_weights = [
                    1.0 / (1.0 + abs(s - mean_score) / std_score)
                    for s in scores
                ]
            else:
                consensus_weights = [1.0] * len(scores)

            final_score = np.average(scores, weights=consensus_weights)

            return CodeCritique(
                overall_score=final_score,
                reasoning=f"Debate consensus: {final_score:.1f}/10 (std: {std_score:.1f})",
                teacher_name="Code Teacher Debate",
                language=sample.language,
            )

        raise ValueError(f"Unknown strategy: {self.strategy}")

    def __repr__(self) -> str:
        teacher_names = [t.name for t in self.teachers]
        return f"CodeTeacher(teachers={teacher_names}, strategy='{self.strategy}')"
