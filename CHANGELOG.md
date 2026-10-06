# Changelog

All notable changes to ASPIRE will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[1.0.0]: https://github.com/mcp-tool-shop-org/aspire-ai/compare/v0.2.0...v1.0.0
[0.2.0]: https://github.com/mcp-tool-shop-org/aspire-ai/compare/v0.1.3...v0.2.0
[0.1.3]: https://github.com/mcp-tool-shop-org/aspire-ai/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/mcp-tool-shop-org/aspire-ai/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/mcp-tool-shop-org/aspire-ai/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/mcp-tool-shop-org/aspire-ai/releases/tag/v0.1.0
