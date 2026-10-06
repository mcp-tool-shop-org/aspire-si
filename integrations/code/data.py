"""
Data Collection and Processing for ASPIRE Code Training.

Provides utilities for:
- Collecting code from GitHub repositories
- Generating training pairs (code, critique)
- Creating datasets for critic and student training
"""

from __future__ import annotations

import json
import logging
import os
import random
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import Dataset, IterableDataset

from .config import Language
from .code_teacher import CodeSample, CodeCritique, CodeTeacher
from .analysis import detect_language

logger = logging.getLogger(__name__)


@dataclass
class CodeReviewPair:
    """A code sample paired with its critique."""

    code: str
    language: Language
    critique: CodeCritique

    # Optional metadata
    filename: str | None = None
    repo: str | None = None
    commit: str | None = None

    # For contrastive learning
    improved_code: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "code": self.code,
            "language": self.language.value,
            "score": self.critique.overall_score,
            "reasoning": self.critique.reasoning,
            "strengths": self.critique.strengths,
            "weaknesses": self.critique.weaknesses,
            "suggestions": self.critique.suggestions,
            "filename": self.filename,
            "repo": self.repo,
            "improved_code": self.improved_code,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CodeReviewPair":
        """Create from dictionary."""
        critique = CodeCritique(
            overall_score=data["score"],
            reasoning=data.get("reasoning", ""),
            strengths=data.get("strengths", []),
            weaknesses=data.get("weaknesses", []),
            suggestions=data.get("suggestions", []),
            teacher_name="loaded",
            language=Language(data["language"]),
        )
        return cls(
            code=data["code"],
            language=Language(data["language"]),
            critique=critique,
            filename=data.get("filename"),
            repo=data.get("repo"),
            improved_code=data.get("improved_code"),
        )


class GitHubRepoCollector:
    """
    Collects code from GitHub repositories.

    Can clone repos or use GitHub API to fetch specific files.
    """

    # Popular repos known for good code quality
    QUALITY_REPOS = {
        Language.PYTHON: [
            "psf/requests",
            "pallets/flask",
            "django/django",
            "pytorch/pytorch",
            "huggingface/transformers",
            "python/cpython",
            "tiangolo/fastapi",
        ],
        Language.JAVASCRIPT: [
            "facebook/react",
            "vuejs/vue",
            "nodejs/node",
            "expressjs/express",
            "lodash/lodash",
        ],
        Language.TYPESCRIPT: [
            "microsoft/TypeScript",
            "microsoft/vscode",
            "angular/angular",
        ],
        Language.RUST: [
            "rust-lang/rust",
            "tokio-rs/tokio",
            "serde-rs/serde",
        ],
        Language.GO: [
            "golang/go",
            "kubernetes/kubernetes",
            "docker/docker-ce",
        ],
    }

    def __init__(
        self,
        cache_dir: str = "~/.cache/aspire/repos",
        github_token: str | None = None,
    ):
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.github_token = github_token or os.environ.get("GITHUB_TOKEN")

    def clone_repo(self, repo: str, shallow: bool = True) -> Path:
        """
        Clone a GitHub repository.

        Args:
            repo: Repository in format "owner/name"
            shallow: Whether to do a shallow clone

        Returns:
            Path to cloned repository
        """
        repo_path = self.cache_dir / repo.replace("/", "_")

        if repo_path.exists():
            return repo_path

        url = f"https://github.com/{repo}.git"
        cmd = ["git", "clone"]

        if shallow:
            cmd.extend(["--depth", "1"])

        cmd.extend([url, str(repo_path)])

        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to clone {repo}: {e.stderr.decode()}")

        return repo_path

    def collect_files(
        self,
        repo: str,
        language: Language,
        max_files: int = 100,
        min_lines: int = 10,
        max_lines: int = 500,
    ) -> Iterator[tuple[str, str]]:
        """
        Collect code files from a repository.

        Args:
            repo: Repository in format "owner/name"
            language: Language to collect
            max_files: Maximum number of files
            min_lines: Minimum lines per file
            max_lines: Maximum lines per file

        Yields:
            (filename, code) tuples
        """
        repo_path = self.clone_repo(repo)

        # Map language to extensions
        ext_map = {
            Language.PYTHON: [".py"],
            Language.JAVASCRIPT: [".js", ".mjs"],
            Language.TYPESCRIPT: [".ts", ".tsx"],
            Language.RUST: [".rs"],
            Language.GO: [".go"],
            Language.JAVA: [".java"],
            Language.CPP: [".cpp", ".cc", ".hpp", ".h"],
            Language.C: [".c", ".h"],
        }

        extensions = ext_map.get(language, [])
        if not extensions:
            return

        files_yielded = 0

        for ext in extensions:
            for path in repo_path.rglob(f"*{ext}"):
                if files_yielded >= max_files:
                    return

                # Skip tests, examples, vendored code
                path_str = str(path).lower()
                skip_patterns = ["test", "example", "vendor", "node_modules", "__pycache__"]
                if any(p in path_str for p in skip_patterns):
                    continue

                try:
                    code = path.read_text(encoding="utf-8", errors="ignore")
                except (OSError, UnicodeDecodeError) as e:
                    logger.debug(f"Skipping file {path}: {e}")
                    continue

                lines = code.split("\n")
                if not (min_lines <= len(lines) <= max_lines):
                    continue

                yield str(path.relative_to(repo_path)), code
                files_yielded += 1

    def collect_from_quality_repos(
        self,
        language: Language,
        files_per_repo: int = 50,
    ) -> Iterator[tuple[str, str, str]]:
        """
        Collect files from curated high-quality repos.

        Yields:
            (repo, filename, code) tuples
        """
        repos = self.QUALITY_REPOS.get(language, [])

        for repo in repos:
            try:
                for filename, code in self.collect_files(
                    repo, language, max_files=files_per_repo
                ):
                    yield repo, filename, code
            except (OSError, subprocess.CalledProcessError) as e:
                logger.warning(f"Failed to collect from {repo}: {e}")
                continue


def generate_training_pairs(
    teacher: CodeTeacher,
    code_samples: list[tuple[str, str, str]],  # (repo, filename, code)
    language: Language,
    include_improvements: bool = True,
) -> list[CodeReviewPair]:
    """
    Generate training pairs by having the teacher critique code.

    Args:
        teacher: CodeTeacher to generate critiques
        code_samples: List of (repo, filename, code) tuples
        language: Programming language
        include_improvements: Whether to include improved versions

    Returns:
        List of CodeReviewPair objects
    """
    pairs = []

    for repo, filename, code in code_samples:
        sample = CodeSample(
            code=code,
            language=language,
            filename=filename,
        )

        try:
            critique = teacher.critique(sample)
        except Exception as e:
            logger.warning(f"Failed to critique {filename}: {e}")
            continue

        pair = CodeReviewPair(
            code=code,
            language=language,
            critique=critique,
            filename=filename,
            repo=repo,
        )

        pairs.append(pair)

    return pairs


def save_training_data(
    pairs: list[CodeReviewPair],
    output_path: str,
):
    """Save training pairs to JSON."""
    data = [p.to_dict() for p in pairs]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Saved {len(pairs)} training pairs to {output_path}")


def load_training_data(input_path: str) -> list[CodeReviewPair]:
    """Load training pairs from JSON."""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return [CodeReviewPair.from_dict(d) for d in data]


class CodeReviewDataset(Dataset):
    """
    PyTorch Dataset for code review training.

    Supports both critic training (predict scores) and
    student training (generate better code).
    """

    def __init__(
        self,
        pairs: list[CodeReviewPair],
        tokenizer,
        max_length: int = 2048,
        mode: str = "critic",  # "critic" or "student"
    ):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        pair = self.pairs[idx]

        if self.mode == "critic":
            # Tokenize code
            tokens = self.tokenizer(
                pair.code,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

            return {
                "input_ids": tokens["input_ids"].squeeze(0),
                "attention_mask": tokens["attention_mask"].squeeze(0),
                "score": torch.tensor(pair.critique.overall_score, dtype=torch.float32),
                "language": pair.language.value,
            }

        elif self.mode == "student":
            # For student training, include the critique as input
            prompt = f"Review this code and improve it:\n\n```{pair.language.value}\n{pair.code}\n```\n\nIssues: {'; '.join(pair.critique.weaknesses[:3])}\n\nImproved code:"

            tokens = self.tokenizer(
                prompt,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

            # Target is improved code if available, else original
            target = pair.improved_code or pair.code

            target_tokens = self.tokenizer(
                target,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

            return {
                "input_ids": tokens["input_ids"].squeeze(0),
                "attention_mask": tokens["attention_mask"].squeeze(0),
                "labels": target_tokens["input_ids"].squeeze(0),
                "score": torch.tensor(pair.critique.overall_score, dtype=torch.float32),
            }

        raise ValueError(f"Unknown mode: {self.mode}")


class StreamingCodeDataset(IterableDataset):
    """
    Streaming dataset that generates training pairs on-the-fly.

    Useful for large-scale training without storing all pairs.
    """

    def __init__(
        self,
        collector: GitHubRepoCollector,
        teacher: CodeTeacher,
        tokenizer,
        language: Language,
        max_length: int = 2048,
        repos: list[str] | None = None,
    ):
        self.collector = collector
        self.teacher = teacher
        self.tokenizer = tokenizer
        self.language = language
        self.max_length = max_length
        self.repos = repos or collector.QUALITY_REPOS.get(language, [])

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        """Stream training examples."""
        for repo in self.repos:
            try:
                for filename, code in self.collector.collect_files(
                    repo, self.language
                ):
                    sample = CodeSample(
                        code=code,
                        language=self.language,
                        filename=filename,
                    )

                    try:
                        critique = self.teacher.critique(sample)
                    except Exception as e:
                        logger.debug(f"Failed to critique {filename}: {e}")
                        continue

                    # Tokenize
                    tokens = self.tokenizer(
                        code,
                        padding="max_length",
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors="pt",
                    )

                    yield {
                        "input_ids": tokens["input_ids"].squeeze(0),
                        "attention_mask": tokens["attention_mask"].squeeze(0),
                        "score": torch.tensor(critique.overall_score, dtype=torch.float32),
                        "repo": repo,
                        "filename": filename,
                    }

            except Exception as e:
                logger.warning(f"Error processing repo {repo}: {e}")
                continue


def create_balanced_dataset(
    pairs: list[CodeReviewPair],
    score_bins: int = 5,
    samples_per_bin: int | None = None,
) -> list[CodeReviewPair]:
    """
    Create a balanced dataset across score ranges.

    Helps prevent the critic from just predicting the mean score.
    """
    # Bin pairs by score
    bins = {i: [] for i in range(score_bins)}
    bin_size = 10.0 / score_bins

    for pair in pairs:
        bin_idx = min(int(pair.critique.overall_score / bin_size), score_bins - 1)
        bins[bin_idx].append(pair)

    # Sample from each bin
    if samples_per_bin is None:
        samples_per_bin = min(len(b) for b in bins.values() if b)

    balanced = []
    for bin_pairs in bins.values():
        if bin_pairs:
            sampled = random.sample(bin_pairs, min(len(bin_pairs), samples_per_bin))
            balanced.extend(sampled)

    random.shuffle(balanced)
    return balanced
