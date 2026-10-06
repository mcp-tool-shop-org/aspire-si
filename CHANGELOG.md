# Changelog

All notable changes to ASPIRE will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Renamed to **aspire-si**: the repository is `mcp-tool-shop-org/aspire-si` and the PyPI
  distribution is `aspire-si`. The import package is still `aspire`.
- Restored as its own repository with a fresh history; the versions below were released from
  the earlier `aspire-ai` repository, whose tags are not carried over.
- CI fails when a test fails (the test step used to be allowed to fail), installs the CPU build
  of torch, runs the geometry demo, and checks the Atlas map.
- Dependabot runs monthly, three pull requests at most, with grouped updates.

### Added

- Training-dynamics export for ScalarScope: `aspire.geometry.GeometryRecorder`,
  `aspire train --geometry`, and the `training.geometry_export`, `geometry_every` and
  `geometry_window` settings. The trainer records each batch and writes `geometry.json` in the
  output directory.
- A composite teacher's vote keeps each member's score in `metadata["teacher_scores"]`.
- `examples/geometry_demo.py` writes two simulated exports without a model or API key.
- Atlas map (`atlas/`).

### Fixed

- The trainer tests ran PEFT's real LoRA wrapper on mock models, which never returned; 26 of them
  hung until the timeout. They now pass the model through, and the epoch tests set up the
  schedulers that `train()` would create.
- `PeftModel` and `bitsandbytes` are imported where the trainer module loads, and a missing
  `bitsandbytes` says so when the 8-bit optimizer is chosen.
- The long-path test skips, instead of failing, on Windows machines without long paths enabled.

## [1.0.0] - 2026-02-27

### Added

- SECURITY.md with vulnerability reporting and data scope
- SHIP_GATE.md and SCORECARD.md for product standards
- Security & Data Scope section in README
- Scorecard in README
- Makefile with verify target (lint + test + build)
- Coverage reporting in CI with Codecov upload

### Changed

- Bumped version from 0.2.0 to 1.0.0

---

## [0.2.0] - 2026-02-23

### Fixed
- **Ruff lint**: resolved all lint errors in `aspire/` (unused imports, f-string syntax, import sorting)
- **Python 3.10 compatibility**: replaced backslash escapes in f-strings with single-quoted strings in teacher modules
- **Line length**: bumped to 110 and reformatted codebase

### Changed
- CI workflow: added `paths:` filters, `concurrency` group, `workflow_dispatch`, bumped `setup-python` to v6
- CI test matrix trimmed to Python 3.11 + 3.12 (dropped 3.10 — still supported at install time)
- Ruff config: added `N812` ignore for PyTorch `functional as F` convention

---

## [0.1.3] - 2026-02-18

### Fixed
- License metadata added to pyproject.toml
- Repository URLs updated to `mcp-tool-shop-org`
- Brand logo added to package

### Changed
- Moved repository to `mcp-tool-shop-org/aspire-ai`
- Added publish workflow for PyPI trusted publishing

---

## [0.1.2] - 2026-01-28

### Added
- Perception Module with cognitive empathy, syntropy, and security hardening
- Theory of Mind tracker, metacognition module, controlled chaos generator
- Character system with persistent personality and value alignment
- Empathy evaluation with perception-aware scoring
- Performance tests and issues punch list
- Press release for Perception Module launch
- CODE_OF_CONDUCT.md and authors metadata

### Changed
- Test coverage expanded to 659 tests

---

## [0.1.1] - 2026-01-22

### Added
- `aspire doctor` command for environment diagnostics
- `--version` / `-V` flag to CLI
- Helpful error messages for missing API keys
- SECURITY.md for vulnerability reporting
- Dependabot configuration for automated dependency updates
- Comprehensive CONTRIBUTING.md guide
- Input validation for all teacher implementations
- Expanded test suite with 7 new test files
- CI and PyPI badges to README

### Changed
- CLI now shows help when run without arguments
- Improved error handling across the codebase

### Security
- Added ClaudeTeacherError and OpenAITeacherError for better error handling
- API keys now validated before use with actionable error messages

## [0.1.0] - 2026-01-22

### Added
- Initial release of ASPIRE
- Core training loop with student-critic-teacher architecture
- Teacher implementations:
  - ClaudeTeacher (Anthropic API)
  - OpenAITeacher (OpenAI API)
  - LocalTeacher (local models via transformers)
- Teacher personas:
  - Socratic - teaches through questions
  - Scientific - demands evidence and rigor
  - Creative - encourages novel thinking
  - Adversarial - stress-tests reasoning
  - Compassionate - balances challenge with encouragement
- CompositeTeacher for multi-teacher ensembles
- Critic architectures:
  - CriticHead - lightweight MLP on student hidden states
  - SeparateCritic - independent encoder model
  - SharedEncoderCritic - shared encoder with student
- Loss functions:
  - Critic score prediction loss
  - Critic reasoning alignment loss
  - Student reward loss
  - Contrastive loss (student vs teacher improved)
  - Trajectory improvement loss
- Dialogue generation system with caching
- CLI commands: `train`, `evaluate`, `dialogue`, `teachers`, `init`
- Pydantic-based configuration with YAML support
- Integration modules:
  - Stable Diffusion WebUI Forge (image generation)
  - Isaac Gym/Lab (robotics)
  - Code assistants (code review)
- Full Windows compatibility (RTX 5080/Blackwell support)
- Comprehensive test suite

### Technical Details
- Python 3.10+ required
- PyTorch 2.0+ with CUDA support
- 4-bit and 8-bit quantization support via bitsandbytes
- LoRA fine-tuning via PEFT
- Async teacher API calls
