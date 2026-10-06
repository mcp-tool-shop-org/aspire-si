"""
Code Critic - Learns to predict teacher judgments of code quality.

The critic observes code and predicts:
1. Quality score (what would the teacher rate this?)
2. Reasoning (why would the teacher give this score?)
3. Fix suggestion (how would the teacher improve it?)

After training, the critic enables self-refinement without teacher API calls.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CriticArchitecture, CriticConfig, CodeDimension

logger = logging.getLogger(__name__)


@dataclass
class CriticOutput:
    """Output from the code critic."""

    # Predicted teacher score (0-10)
    score: torch.Tensor  # (batch,)

    # Predicted dimension scores
    dimension_scores: dict[str, torch.Tensor] | None = None

    # Predicted reasoning embedding
    reasoning_embedding: torch.Tensor | None = None  # (batch, hidden_dim)

    # Suggested fix embedding (can be decoded to code)
    fix_embedding: torch.Tensor | None = None  # (batch, hidden_dim)

    # Token-level scores (which parts are problematic?)
    token_scores: torch.Tensor | None = None  # (batch, seq)

    # Attention weights (for interpretability)
    attention_weights: torch.Tensor | None = None


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class CodeEncoder(nn.Module):
    """
    Encodes code into a representation for the critic.

    Supports multiple architectures:
    - Transformer: Full attention over code tokens
    - CodeBERT: Pre-trained code understanding
    - GraphNN: AST-based structural encoding
    - Hybrid: Combines sequence and structure
    """

    def __init__(
        self,
        vocab_size: int = 50000,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        max_seq_len: int = 2048,
        architecture: CriticArchitecture = CriticArchitecture.TRANSFORMER,
        pretrained_model: str | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.architecture = architecture
        self.pretrained_model = pretrained_model

        if architecture == CriticArchitecture.CODEBERT and pretrained_model:
            # Use pre-trained CodeBERT
            try:
                from transformers import AutoModel, AutoTokenizer

                self.backbone = AutoModel.from_pretrained(pretrained_model)
                self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

                # Project to our hidden dim if different
                backbone_dim = self.backbone.config.hidden_size
                if backbone_dim != hidden_dim:
                    self.projection = nn.Linear(backbone_dim, hidden_dim)
                else:
                    self.projection = nn.Identity()

            except ImportError:
                raise ImportError("pip install transformers")

        else:
            # Build from scratch
            self.embedding = nn.Embedding(vocab_size, hidden_dim)
            self.pos_encoding = PositionalEncoding(hidden_dim, max_seq_len, dropout)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
            self.projection = nn.Identity()

        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Encode code tokens.

        Args:
            input_ids: (batch, seq) token IDs
            attention_mask: (batch, seq) mask for padding

        Returns:
            encoding: (batch, seq, hidden_dim)
            attention_weights: Optional attention for interpretability
        """
        attention_weights = None

        if self.architecture == CriticArchitecture.CODEBERT and hasattr(self, "backbone"):
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
            )
            encoding = outputs.last_hidden_state
            attention_weights = outputs.attentions[-1] if outputs.attentions else None

        else:
            # Custom transformer
            x = self.embedding(input_ids)
            x = self.pos_encoding(x)

            if attention_mask is not None:
                # Convert to transformer format (True = masked)
                src_key_padding_mask = ~attention_mask.bool()
            else:
                src_key_padding_mask = None

            encoding = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        encoding = self.projection(encoding)
        encoding = self.output_norm(encoding)

        return encoding, attention_weights


class CodeCriticHead(nn.Module):
    """
    Prediction heads for the code critic.

    Predicts:
    - Overall quality score
    - Per-dimension scores (correctness, style, security, etc.)
    - Per-token scores (which tokens are problematic?)
    - Reasoning embedding (for distillation)
    - Fix embedding (for code suggestion)
    """

    DIMENSIONS = [
        "correctness",
        "style",
        "security",
        "performance",
        "maintainability",
        "architecture",
        "documentation",
    ]

    def __init__(
        self,
        hidden_dim: int = 512,
        predict_score: bool = True,
        predict_dimensions: bool = True,
        predict_tokens: bool = True,
        predict_reasoning: bool = True,
        predict_fix: bool = True,
        reasoning_dim: int = 256,
        fix_dim: int = 512,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Score prediction (0-10)
        if predict_score:
            self.score_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid(),
            )
        else:
            self.score_head = None

        # Per-dimension scores
        if predict_dimensions:
            self.dimension_heads = nn.ModuleDict({
                dim: nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 4),
                    nn.GELU(),
                    nn.Linear(hidden_dim // 4, 1),
                    nn.Sigmoid(),
                )
                for dim in self.DIMENSIONS
            })
        else:
            self.dimension_heads = None

        # Per-token scores
        if predict_tokens:
            self.token_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid(),
            )
        else:
            self.token_head = None

        # Reasoning embedding
        if predict_reasoning:
            self.reasoning_head = nn.Sequential(
                nn.Linear(hidden_dim, reasoning_dim),
                nn.LayerNorm(reasoning_dim),
            )
        else:
            self.reasoning_head = None

        # Fix embedding
        if predict_fix:
            self.fix_head = nn.Sequential(
                nn.Linear(hidden_dim, fix_dim),
                nn.LayerNorm(fix_dim),
            )
        else:
            self.fix_head = None

    def forward(
        self,
        encoding: torch.Tensor,
        pool: Literal["mean", "cls", "max"] = "mean",
    ) -> CriticOutput:
        """
        Predict from code encoding.

        Args:
            encoding: (batch, seq, hidden) code representation
            pool: How to pool sequence for global predictions

        Returns:
            CriticOutput with all predictions
        """
        batch, seq, hidden = encoding.shape

        # Pool for global predictions
        if pool == "mean":
            pooled = encoding.mean(dim=1)
        elif pool == "cls":
            pooled = encoding[:, 0]
        elif pool == "max":
            pooled = encoding.max(dim=1).values
        else:
            raise ValueError(f"Unknown pool: {pool}")

        # Score prediction
        score = None
        if self.score_head is not None:
            score = self.score_head(pooled).squeeze(-1) * 10.0

        # Dimension scores
        dimension_scores = None
        if self.dimension_heads is not None:
            dimension_scores = {
                dim: head(pooled).squeeze(-1) * 10.0
                for dim, head in self.dimension_heads.items()
            }

        # Token scores
        token_scores = None
        if self.token_head is not None:
            token_scores = self.token_head(encoding).squeeze(-1) * 10.0

        # Reasoning embedding
        reasoning_embedding = None
        if self.reasoning_head is not None:
            reasoning_embedding = self.reasoning_head(pooled)

        # Fix embedding
        fix_embedding = None
        if self.fix_head is not None:
            fix_embedding = self.fix_head(pooled)

        return CriticOutput(
            score=score,
            dimension_scores=dimension_scores,
            token_scores=token_scores,
            reasoning_embedding=reasoning_embedding,
            fix_embedding=fix_embedding,
        )


class CodeCritic(nn.Module):
    """
    Complete code critic model.

    Combines encoder and prediction heads to evaluate code
    and predict what the teacher would think.
    """

    def __init__(self, config: CriticConfig | None = None):
        super().__init__()

        if config is None:
            config = CriticConfig()

        self.config = config

        # Build encoder
        self.encoder = CodeEncoder(
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            max_seq_len=config.max_code_length,
            architecture=config.architecture,
            pretrained_model=config.pretrained_model,
            dropout=config.dropout,
        )

        # Build prediction heads
        self.heads = CodeCriticHead(
            hidden_dim=config.hidden_dim,
            predict_score=config.predict_score,
            predict_dimensions=True,
            predict_tokens=True,
            predict_reasoning=config.predict_reasoning,
            predict_fix=config.predict_fix,
        )

        # Tokenizer (use CodeBERT's or build simple one)
        self._tokenizer = None

    def get_tokenizer(self):
        """Get or create tokenizer."""
        if self._tokenizer is None:
            if hasattr(self.encoder, "tokenizer"):
                self._tokenizer = self.encoder.tokenizer
            else:
                # Use a simple tokenizer
                try:
                    from transformers import AutoTokenizer
                    self._tokenizer = AutoTokenizer.from_pretrained(
                        self.config.pretrained_model or "microsoft/codebert-base"
                    )
                except (ImportError, OSError) as e:
                    # Fallback to basic tokenizer
                    logger.warning(
                        f"Failed to load tokenizer for "
                        f"{self.config.pretrained_model or 'microsoft/codebert-base'}: {e}. "
                        "Tokenizer will be None - use tokenize() with caution."
                    )
                    self._tokenizer = None
        return self._tokenizer

    def tokenize(
        self,
        code: str | list[str],
        max_length: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Tokenize code for input."""
        tokenizer = self.get_tokenizer()

        if tokenizer is None:
            raise ValueError("No tokenizer available")

        max_length = max_length or self.config.max_code_length

        if isinstance(code, str):
            code = [code]

        return tokenizer(
            code,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> CriticOutput:
        """
        Evaluate code.

        Args:
            input_ids: (batch, seq) token IDs
            attention_mask: (batch, seq) attention mask

        Returns:
            CriticOutput with score, reasoning, and suggestions
        """
        # Encode
        encoding, attention = self.encoder(input_ids, attention_mask)

        # Predict
        output = self.heads(encoding)
        output.attention_weights = attention

        return output

    def score_code(
        self,
        code: str | list[str],
        device: str = "cuda",
    ) -> torch.Tensor:
        """
        Get quality scores for code.

        Args:
            code: Code string(s) to evaluate
            device: Device to run on

        Returns:
            Scores tensor (batch,) in range [0, 10]
        """
        tokens = self.tokenize(code)
        tokens = {k: v.to(device) for k, v in tokens.items()}

        self.eval()
        with torch.no_grad():
            output = self.forward(
                tokens["input_ids"],
                tokens.get("attention_mask"),
            )

        return output.score

    def get_problem_tokens(
        self,
        code: str,
        threshold: float = 5.0,
        device: str = "cuda",
    ) -> list[tuple[int, int, float]]:
        """
        Find problematic tokens in code.

        Returns list of (start_char, end_char, score) for problem areas.
        """
        tokens = self.tokenize(code)
        tokens = {k: v.to(device) for k, v in tokens.items()}

        self.eval()
        with torch.no_grad():
            output = self.forward(
                tokens["input_ids"],
                tokens.get("attention_mask"),
            )

        if output.token_scores is None:
            return []

        # Find low-scoring tokens
        scores = output.token_scores[0].cpu().numpy()
        tokenizer = self.get_tokenizer()

        problems = []
        for i, score in enumerate(scores):
            if score < threshold:
                # Convert token position to character position
                # This is simplified - real implementation would use offset mapping
                token = tokenizer.decode([tokens["input_ids"][0, i].item()])
                problems.append((i, i + 1, float(score)))

        return problems


class CodeCriticLoss(nn.Module):
    """
    Loss function for training the code critic.

    Trains the critic to predict teacher judgments:
    - Score prediction loss (MSE)
    - Dimension score losses
    - Token-level losses
    - Reasoning distillation
    """

    def __init__(
        self,
        score_weight: float = 1.0,
        dimension_weight: float = 0.5,
        token_weight: float = 0.3,
        reasoning_weight: float = 0.3,
    ):
        super().__init__()

        self.score_weight = score_weight
        self.dimension_weight = dimension_weight
        self.token_weight = token_weight
        self.reasoning_weight = reasoning_weight

        self.mse = nn.MSELoss()
        self.cosine = nn.CosineSimilarity(dim=-1)

    def forward(
        self,
        critic_output: CriticOutput,
        teacher_score: torch.Tensor,
        teacher_dimensions: dict[str, torch.Tensor] | None = None,
        teacher_reasoning_embedding: torch.Tensor | None = None,
        teacher_token_scores: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute critic training loss.

        Args:
            critic_output: Predictions from critic
            teacher_score: Ground truth overall score (batch,)
            teacher_dimensions: Ground truth dimension scores
            teacher_reasoning_embedding: Teacher's reasoning embedding
            teacher_token_scores: Per-token ground truth

        Returns:
            Dictionary with individual losses and total
        """
        losses = {}
        device = teacher_score.device

        # Score prediction loss
        if critic_output.score is not None:
            losses["score"] = self.mse(critic_output.score, teacher_score)
        else:
            losses["score"] = torch.tensor(0.0, device=device)

        # Dimension score losses
        if (
            critic_output.dimension_scores is not None
            and teacher_dimensions is not None
        ):
            dim_losses = []
            for dim, pred in critic_output.dimension_scores.items():
                if dim in teacher_dimensions:
                    dim_losses.append(self.mse(pred, teacher_dimensions[dim]))

            if dim_losses:
                losses["dimensions"] = torch.stack(dim_losses).mean()
            else:
                losses["dimensions"] = torch.tensor(0.0, device=device)
        else:
            losses["dimensions"] = torch.tensor(0.0, device=device)

        # Token-level loss
        if (
            critic_output.token_scores is not None
            and teacher_token_scores is not None
        ):
            # Align lengths
            min_len = min(
                critic_output.token_scores.shape[1],
                teacher_token_scores.shape[1],
            )
            pred_tokens = critic_output.token_scores[:, :min_len]
            true_tokens = teacher_token_scores[:, :min_len]
            losses["tokens"] = self.mse(pred_tokens, true_tokens)
        else:
            losses["tokens"] = torch.tensor(0.0, device=device)

        # Reasoning distillation
        if (
            critic_output.reasoning_embedding is not None
            and teacher_reasoning_embedding is not None
        ):
            similarity = self.cosine(
                critic_output.reasoning_embedding,
                teacher_reasoning_embedding,
            )
            losses["reasoning"] = (1.0 - similarity).mean()
        else:
            losses["reasoning"] = torch.tensor(0.0, device=device)

        # Total weighted loss
        losses["total"] = (
            self.score_weight * losses["score"]
            + self.dimension_weight * losses["dimensions"]
            + self.token_weight * losses["tokens"]
            + self.reasoning_weight * losses["reasoning"]
        )

        return losses


def create_critic_from_pretrained(
    model_name: str = "microsoft/codebert-base",
    **kwargs,
) -> CodeCritic:
    """
    Create a code critic initialized from a pre-trained model.

    Args:
        model_name: HuggingFace model name
        **kwargs: Additional config overrides

    Returns:
        Initialized CodeCritic
    """
    config = CriticConfig(
        architecture=CriticArchitecture.CODEBERT,
        pretrained_model=model_name,
        **kwargs,
    )
    return CodeCritic(config)
