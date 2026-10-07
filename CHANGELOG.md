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
- Coverage is held at 90%: CI runs `pytest --cov-fail-under=90` over `aspire/` and
  `integrations/`, and `codecov.yml` sets the same target for the project and for each pull
  request. Codecov uploads over OIDC, without a stored token. Coverage is now 99%, up from 62%.
- `use_wandb` defaults to off, as it already did in the integrations: logging to Weights &
  Biases sends run data to a third party.
- The Isaac integration's default device is `cuda` only when torch can use one, and `cpu`
  otherwise, so its examples run on a machine without a GPU.
- The code integration's security auditor deducts by severity (high 3, medium 1.5, low 1) instead
  of a flat 1.5 per pattern. `eval` or `exec` applied straight to `input()` costs 4 more and is
  named as such, so `def f(): eval(input())` now scores 2.5 instead of 8.
- The handbook and landing page run on Astro 7 and Starlight 0.42, with the sidebar in the shape
  Starlight 0.39 and later require.
- The README, landing page and handbook no longer say the student refines its own output at
  inference time. ASPIRE ships the trained critic, which judges a response without the teacher;
  a refine loop around it is the user's to write, and the docs now show one.

### Added

- Training-dynamics export for ScalarScope: `aspire.geometry.GeometryRecorder`,
  `aspire train --geometry`, and the `training.geometry_export`, `geometry_every` and
  `geometry_window` settings. The trainer records each batch and writes `geometry.json` in the
  output directory.
- A composite teacher's vote keeps each member's score in `metadata["teacher_scores"]`.
- `examples/geometry_demo.py` writes two simulated exports without a model or API key.
- Atlas map (`atlas/`).
- `aspire.errors.AspireError`: every error ASPIRE raises on purpose has a `code`, `message`,
  `hint`, optional `cause` and `retryable`. Teacher, config and input errors subclass it.
- The CLI reports errors as code, message and hint without a traceback, and exits 1 for
  something the user can fix, 2 for a failure while running, 130 when interrupted. New global
  options `--quiet`, `--verbose` (prints the resolved settings) and `--debug` (shows tracebacks).
  The `aspire` entry point is now `aspire.cli:run`.
- `aspire train` and `aspire evaluate` say so when a config, prompts or checkpoint file is
  missing or malformed, instead of failing with a traceback or quietly using the defaults.
- `integrations.code` exports `CodeSample`, `CodeCritique` and `Language`, which its README and
  examples already imported from it.
- Handbook page "Watching Runs in ScalarScope".
- About 1,550 new tests, covering the Isaac, Forge and code integrations and the perception
  modules, which were mostly untested.

### Fixed

- The trainer tests ran PEFT's real LoRA wrapper on mock models, which never returned; 26 of them
  hung until the timeout. They now pass the model through, and the epoch tests set up the
  schedulers that `train()` would create.
- `PeftModel` and `bitsandbytes` are imported where the trainer module loads, and a missing
  `bitsandbytes` says so when the 8-bit optimizer is chosen.
- The long-path test skips, instead of failing, on Windows machines without long paths enabled.
- The integration READMEs, examples and quick starts imported `aspire.integrations`, which does
  not exist; they import `integrations` now.

Bugs the new tests found, each now covered by a test that failed before the fix:

- **Isaac:** a trajectory scored 0.0 got the lowest replay priority instead of the highest;
  `DummyIsaacEnv` never started a new episode, so collection could loop forever; `train(epochs=0)`
  ran the full schedule; checkpoints could not be loaded back (they now hold plain values and
  load with `weights_only=True`); a one-layer policy failed on any state size; the LSTM critic's
  improvement shape shrank under a padding mask.
- **Code:** Rust was detected as JavaScript; `async def` functions were not counted; `a / 0.5`
  was flagged as division by zero; the SQL-injection regex never matched; the `secrets` module was
  flagged as a leaked secret; an LLM score of `45/100` read as 10/10; a repository under a folder
  named like a skip word yielded no files; one failed clone stopped every remaining one; an empty
  balanced dataset raised; a sample's `commit` was lost on a round trip; a module student had no
  tokenizer; checkpointing an unknown component raised `UnboundLocalError`.
- **Forge:** the image critic re-initialised the frozen CLIP encoder's weights; a vision teacher
  reply that was JSON but not an object raised instead of falling back.
- **Perception:** a chaos injection carried over to later chaos-free prompts; an evaluation that
  scored some categories was diluted by the unscored ones (a perfect partial score read about
  2.6/10); "may" was found in "mayor" as a hedge; a character's saved activation contexts and
  values were lost on load, and a string priority crashed sorting; the uncertainty variance was
  NaN for one sample; the completeness reflection prompt could be dropped; `[batch, vocab]`
  logits failed; five chaos types had no generator and a requested type could come back as
  another; empty text and a zero-epoch ramp raised; syntropy coherence was NaN for one token;
  gratitude and anxiety were left out of the emotional trend.
- **Core:** a chat-formatted dialogue with no turns dropped the prompt; an unknown optimizer name
  left the trainer without one (it now raises).

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
