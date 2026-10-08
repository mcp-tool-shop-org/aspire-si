# Changelog

All notable changes to ASPIRE will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `examples/sft-experiment/`: the fine-tune-then-ASPIRE experiment (#11) and its
  [run report](docs/runs/2026-10-07-sft-then-aspire.md). It holds:
  - a dataset builder that keeps evaluation prompts out by word overlap and embeddings;
  - a judge set of one-sentence planted errors;
  - a cleaning step that records every drop;
  - QLoRA fine-tuning;
  - probes with schema 1.1 drift exports;
  - the held-out and judge measures, with intervals that resample prompts;
  - pod scripts that check the rented host's CUDA driver, GPU memory and download speed before
    installing anything.

  Result, over three training seeds ([run 1 report](docs/runs/2026-10-07-run-1-seeds.md)):
  fine-tuning the student before ASPIRE makes no reliable difference to how well the critic spots
  planted errors, with either teacher. A first single-seed result said it hurt the composite
  critic (0.65 against 0.87); two more seeds did not bear that out. The critic's accuracy varies
  much more between training runs than between conditions (the composite control ranged from
  0.43 to 0.87). The hidden-state trajectory replicates: the fine-tune moves the student about
  30 times as far as ASPIRE does, always in the same direction, and ASPIRE's own drift opposes
  that direction in every seed, with or without the fine-tune.

- Judges read order-averaged ([report](docs/runs/2026-10-08-kev-confirmation.md)): `fresh_pairs.py`
  plants errors on new prompts, `judge_kev.order_averaged` averages a judge's probabilities over
  both answer orders, and `judge_logprob.py` reads the teacher's A/B log-probabilities. By single
  choices every judge tried is position-biased and below 0.75. Order-averaged, on 149 fresh pairs,
  Kev-4B reaches 0.973 (CI 0.946 to 0.993), Kev-9B 0.859 and the 32B teacher 0.886. Kev-4B,
  read this way, is the reference judge for later comparisons.

- Step 2 ([report](docs/runs/2026-10-08-step-2.md)): critics trained on 128 prompts instead of 32
  score higher on planted errors (mean 0.745 against 0.635, every seed up by 0.10 to 0.13), but
  differ between seeds exactly as much (a range of 0.181 at both). The fine-tune comparison waits
  for a critic setup that varies less.

- `critic.init_seed` (config, default unset): seeds the critic's initial weights apart from the
  run's seed, leaving the run's own random stream (student adapter, data order, sampling) as if no
  critic had been built. Unset, nothing changes. Used by the critic-init test
  ([plan](docs/runs/2026-10-08-critic-init-plan.md)).

### Changed

- `aspire.teachers.local`: the scoring request and its parser are module functions
  (`evaluation_request`, `parse_evaluation`, `extract_score`, `extract_improved`), so other code
  scores exactly as the local teacher does. The request text and the parsed scores are unchanged.

## [1.2.0] - 2026-10-07

The first end-to-end run on real models: a local teacher, a composite of two, a checkpoint that
loads back, and a critic you can score responses with. Running it found the bugs below; each
is fixed and covered by a test that failed before the fix.

### Security

- Loading a checkpoint can no longer run code from it. The critic, the code trainer and the
  isaac trainer load `.pt` files with `torch.load(weights_only=True)`. aspire-si allows torch
  2.0, and before torch 2.6 the default unpickled any object, so a crafted checkpoint handed to
  `aspire judge` or `load_checkpoint` could execute code. A test fails if any `torch.load` call
  leaves the flag out.

### Added

- `aspire.judge`: `Judge.from_checkpoint(path)` loads what `aspire train` saved (student, its
  LoRA adapter, critic and config) and scores responses with `score(prompt, response)` and
  `score_many(...)`; `critic_score(student, tokenizer, critic, prompt, response)` does the same
  for models you loaded yourself. This is the scoring function the docs used to leave to you.
- `aspire judge CHECKPOINT --prompt ... --response ...` (or `--pairs file.json`, `--json`):
  the critic's 0-10 score without calling the teacher.
- Local and composite teachers from the config and the CLI: `--teacher local --teacher-model
  <model>`, `teacher.local_model_path`, `teacher.local_load_in_4bit`, and
  `default_teacher: composite` with `teacher.composite_members` (each a registered name or
  `local:<model>`) and `teacher.composite_strategy`. `--student-model` sets the student.
- `examples/local-run/`: a run that costs nothing (1B student, 3B local teacher, 24 GB GPU), its
  composite variant, and 32 prompts. `examples/pod-run/`: the configs of the runs behind
  ScalarScope's real fixtures (1.5B student, 32B teachers in 4-bit).
- The local teacher asks for JSON with a score per evaluation dimension, reads it (fenced or
  bare), and falls back to a score in the text. `metadata["parse"]` says which happened.
- An instruct model's own chat template is used for the student's turns, the local teacher's
  requests and the text the critic reads.
- `teacher.evaluate_each_turn` (default on): the teacher scores every turn as well as the last.
  Training reads only the final evaluation, so turning it off halves the teacher's work without
  changing what the student and critic train on.
- The geometry export is written after every epoch as well as at the end, so a run stopped
  early (a deadline, Ctrl+C, a crash) keeps the epochs it finished.
- `examples/pod-run/`: the configs and scripts of the first real-model runs, including
  `probe.py` and `drift.py`. `probe.py` replays a fixed set of exchanges through the base student
  and every epoch checkpoint; `drift.py` writes a geometry export of what training changed, with
  prompt identity removed. Run report: `docs/runs/2026-10-06-pod-run.md`.
- Geometry exports are schema 1.1. `run_metadata.step_axis` (`training_step` or
  `checkpoint_by_item`, with `checkpoints`) and `run_metadata.scalar_source` (`live`, `replayed`
  or `fixed_per_item`) say how to read the steps, so ScalarScope can withhold time-based
  comparisons on probe and drift exports and failure dips on replayed scores. The trainer marks a
  run of more than one epoch as `replayed`. `examples/pod-run/upgrade_export.py` adds the two
  facts to a 1.0 export written by the trainer, without touching its data.

### Changed

- The critic now reads the student's hidden states over the **prompt and the response the
  teacher scored**. It used to read the prompt alone, so it learned to predict a score from the
  question and could not judge a response.
- The README, handbook and landing page say what training does today. The critic learns the
  teacher's score (its reasoning head is not trained). The student's LoRA adapter learns
  representations that help the critic predict that score; it is not yet trained toward better
  answers, because the trainer does not feed the reward, contrastive or trajectory losses. That
  is planned for 1.3.0 ([#11](https://github.com/mcp-tool-shop-org/aspire-si/issues/11)). The
  translated READMEs carry the corrected sections in English until they are regenerated.
- The Claude teacher defaults to `claude-sonnet-5-5` (was `claude-sonnet-4-20250514`). It sends
  a `temperature` only to models that accept one (Sonnet 5 and later, Opus 4.7 and later and
  Fable reject a non-default value), reads the answer from text blocks so a thinking block
  before it is skipped, and raises `ASPIRE_CLAUDE_TEACHER` when Claude declines a request.
- The default `device` is `cuda` when torch can use it and `cpu` otherwise (was always `cuda`).
- `aspire train` options left off the command line keep the config file's values. `--teacher`,
  `--epochs` and `--output` used to overwrite them with `claude`, 3 and `outputs` every time.
- A geometry export's condition names the teacher (`local:Qwen2.5-32B-Instruct`, or
  `composite (vote): ...`) instead of the setting's value.

### Fixed

- Checkpoints could not be loaded: `config.yaml` was written with Python path tags that the
  YAML reader refuses. `aspire evaluate` failed on every checkpoint.
- `aspire evaluate` wrapped the student in a second LoRA adapter, so the trained weights were
  never loaded and evaluation ran on an untrained adapter.
- The default critic (`CriticHead`) could not be loaded from `critic.pt`: its saved settings
  lacked the input size. `MultiHeadCriticHead` round-trips too.
- A bf16 or 4-bit student handed bf16 hidden states to the float32 critic.
- The local teacher could not be built (the trainer passed it an API model name), and a
  composite teacher could not be built at all. `get_teacher(name, name=...)` failed because the
  registry's own parameter was called `name`; it is positional-only now.
- The local teacher always took 2048 input tokens plus up to 768 new ones, past the context of
  smaller models, and cut long requests from the end, losing the instructions. It now fits the
  request to the model's context and drops the beginning.
- Dialogues reloaded from the cache (every epoch after the first) lost their dimension scores,
  each composite member's score and the scored response. The geometry export of a run longer
  than one epoch had no scalars and a single professor after epoch 1.
- Persona teachers (`socratic`, `scientific`, ...) are Claude teachers but were given the OpenAI
  model name.
- On a Windows console or redirected output that uses a code page such as cp1252, the progress
  bar stopped training with a `UnicodeEncodeError`. Characters the console cannot show print as
  `?` now.

Found by the real-model run on a rented GPU (a 1.5B student, 32B teachers), each with a test:

- The student's turns were cut at 512 tokens from the end: the trainer never passed
  `student.max_length` to the dialogue generator, and truncation dropped the end of the input.
  By the third turn the cue to answer was gone, the student continued the dialogue
  mid-sentence, and that reply was what the teacher scored. The generator now gets the
  student's `max_length` and, past it, drops the beginning of the dialogue instead.
  (`TestPodRunFixes::test_student_input_loses_its_beginning_not_the_cue_to_answer`,
  `test_trainer_passes_the_students_max_length_and_turn_evaluation`)
- Local teachers wrote LaTeX (`\(`) and raw newlines inside JSON strings, so complete
  evaluations were rejected and fell back to the default 5.0. Invalid escapes are repaired and
  raw newlines accepted. (`test_teacher_json_with_latex_escapes_and_raw_newlines_parses`)
- An evaluation could take only 768 tokens, which cut verbose teachers off mid-JSON (a score
  and an explanation for each of nine dimensions, plus an improved response). It may take
  1536. (`test_evaluations_have_room_for_every_dimension`)

## [1.1.0] - 2026-10-06

The first release as aspire-si, and the first on PyPI.

### Changed

- Renamed to **aspire-si**: the repository is `mcp-tool-shop-org/aspire-si` and the PyPI
  distribution is `aspire-si`. The import package is still `aspire`.
- Restored as its own repository with a fresh history; the versions below were released from
  the earlier `aspire-ai` repository, whose tags are not carried over.
- CI fails when a test fails (the test step used to be allowed to fail), installs the CPU build
  of torch, runs the geometry demo, and checks the Atlas map.
- Dependabot runs monthly, three pull requests at most, with grouped updates.
- New logo: the rocket A reading ASPIRE-SI (brand slug `aspire-si`), in the README and the
  handbook header.
- The PyPI workflow is `release.yml`, publishing with Trusted Publishing through the `pypi`
  environment. It refuses a release whose tag does not match the version in `pyproject.toml`,
  and pins the publish action to a commit.
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
