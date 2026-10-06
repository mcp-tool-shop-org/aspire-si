"""
ASPIRE for Code - Teaching code models to develop programming judgment.

This integration brings ASPIRE's adversarial learning to code generation:
- Code teachers critique generated code for correctness, style, security
- Code critics internalize programming wisdom
- Models self-refine before outputting code

Example:
    from aspire.integrations.code import (
        CodeTeacher,
        CodeCritic,
        AspireCodeTrainer,
    )

    # Create teacher that evaluates code quality
    teacher = CodeTeacher(
        personas=["correctness_checker", "style_guide", "security_auditor"],
        strategy="vote",
    )

    # Train model to internalize good coding judgment
    trainer = AspireCodeTrainer(
        student_model="codellama/CodeLlama-7b-hf",
        teacher=teacher,
        critic=CodeCritic(),
    )
    trainer.train(dataset="code_review_data")
"""

from .code_teacher import (
    CodeTeacher,
    CorrectnessChecker,
    StyleGuide,
    SecurityAuditor,
    ArchitectureReviewer,
    PerformanceAnalyst,
    DocumentationCritic,
)
from .code_critic import (
    CodeCritic,
    CodeEncoder,
    CodeCriticHead,
)
from .analysis import (
    CodeAnalyzer,
    StaticAnalysisResult,
    extract_code_features,
    parse_code,
)
from .trainer import AspireCodeTrainer
from .config import CodeAspireConfig
from .data import (
    CodeReviewDataset,
    GitHubRepoCollector,
    generate_training_pairs,
)

__all__ = [
    # Teachers
    "CodeTeacher",
    "CorrectnessChecker",
    "StyleGuide",
    "SecurityAuditor",
    "ArchitectureReviewer",
    "PerformanceAnalyst",
    "DocumentationCritic",
    # Critics
    "CodeCritic",
    "CodeEncoder",
    "CodeCriticHead",
    # Analysis
    "CodeAnalyzer",
    "StaticAnalysisResult",
    "extract_code_features",
    "parse_code",
    # Training
    "AspireCodeTrainer",
    "CodeAspireConfig",
    # Data
    "CodeReviewDataset",
    "GitHubRepoCollector",
    "generate_training_pairs",
]
