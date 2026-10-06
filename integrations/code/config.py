"""
Configuration for ASPIRE Code Assistant integration.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


class Language(str, Enum):
    """Supported programming languages."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    RUST = "rust"
    GO = "go"
    JAVA = "java"
    CPP = "cpp"
    C = "c"
    CSHARP = "csharp"
    RUBY = "ruby"
    PHP = "php"
    SWIFT = "swift"
    KOTLIN = "kotlin"
    SCALA = "scala"
    UNKNOWN = "unknown"


class CodeDimension(str, Enum):
    """Dimensions of code quality."""
    CORRECTNESS = "correctness"         # Does it work?
    STYLE = "style"                     # Is it idiomatic?
    SECURITY = "security"               # Is it safe?
    PERFORMANCE = "performance"         # Is it efficient?
    MAINTAINABILITY = "maintainability" # Is it readable/changeable?
    ARCHITECTURE = "architecture"       # Is it well-structured?
    DOCUMENTATION = "documentation"     # Is it explained?
    TESTING = "testing"                 # Is it testable/tested?


class CriticArchitecture(str, Enum):
    """Code critic architectures."""
    TRANSFORMER = "transformer"     # Full attention over code
    CODEBERT = "codebert"          # Pre-trained code understanding
    GRAPHNN = "graphnn"            # AST-based graph neural network
    HYBRID = "hybrid"              # Combines sequence and structure


@dataclass
class TeacherConfig:
    """Configuration for code teachers."""

    # Which personas to use
    personas: list[str] = field(default_factory=lambda: [
        "correctness_checker",
        "style_guide",
        "security_auditor",
    ])

    # How to combine multiple teachers
    strategy: Literal["vote", "rotate", "debate"] = "vote"

    # Use LLM teacher (Claude/GPT-4)?
    use_llm_teacher: bool = True
    llm_model: str = "claude-sonnet-4-20250514"

    # Use static analysis tools?
    use_static_analysis: bool = True

    # Languages to support
    languages: list[Language] = field(default_factory=lambda: [
        Language.PYTHON,
        Language.JAVASCRIPT,
        Language.TYPESCRIPT,
    ])

    # Weights for different evaluation dimensions
    correctness_weight: float = 2.0    # Correctness is paramount
    security_weight: float = 1.5       # Security is critical
    style_weight: float = 1.0
    performance_weight: float = 0.8
    maintainability_weight: float = 1.0


@dataclass
class CriticConfig:
    """Configuration for code critic."""

    architecture: CriticArchitecture = CriticArchitecture.TRANSFORMER

    # Model dimensions
    hidden_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8

    # Input processing
    max_code_length: int = 2048  # tokens
    use_ast_features: bool = True
    use_docstring_features: bool = True

    # Output heads
    predict_score: bool = True
    predict_reasoning: bool = True
    predict_fix: bool = True  # Suggest code fix

    # Pre-trained backbone (optional)
    pretrained_model: str | None = "microsoft/codebert-base"

    # Training
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    """Training configuration for code ASPIRE."""

    # Student model
    student_model: str = "codellama/CodeLlama-7b-hf"

    # Dataset
    dataset_path: str = ""
    languages: list[str] = field(default_factory=lambda: ["python"])

    # Training parameters
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    epochs: int = 3
    max_length: int = 2048

    # Learning rates
    critic_lr: float = 1e-4
    student_lr: float = 5e-5

    # LoRA configuration (for efficient fine-tuning)
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # Loss weights
    critic_score_weight: float = 1.0
    critic_reasoning_weight: float = 0.5
    student_reward_weight: float = 1.0
    student_contrastive_weight: float = 0.3

    # Checkpointing
    save_frequency: int = 500
    checkpoint_dir: str = "checkpoints/code"

    # Logging
    use_wandb: bool = False
    wandb_project: str = "aspire-code"


@dataclass
class CodeAspireConfig:
    """Complete configuration for ASPIRE Code training."""

    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    critic: CriticConfig = field(default_factory=CriticConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Hardware
    device: str = "cuda"
    seed: int = 42

    def __post_init__(self):
        """Validate configuration."""
        if self.training.batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        if not self.teacher.personas:
            raise ValueError("At least one teacher persona required")
