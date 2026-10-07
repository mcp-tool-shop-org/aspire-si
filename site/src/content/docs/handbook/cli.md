---
title: CLI Reference
description: Complete command-line reference for aspire-si — every command, flag, and option.
sidebar:
  order: 5
---

ASPIRE provides a command-line interface for generating dialogues, training models, and evaluating checkpoints. Every command supports `--help` for inline documentation.

## Global options

| Flag | Description |
|------|-------------|
| `--version`, `-V` | Print the installed ASPIRE version and exit. |
| `--quiet`, `-q` | Print errors only. |
| `--verbose`, `-v` | Also print the resolved settings before training or evaluating. |
| `--debug` | Verbose, and show the full traceback when something fails. |
| `--help` | Show help for any command. |

Global options go before the command: `aspire --verbose train ...`.

## Errors and exit codes

When something goes wrong, ASPIRE prints a code, a message, and what to do, without a
traceback:

```
ASPIRE_CONFIG  The prompts file data/prompts.json is not valid JSON.
It should be a JSON list of prompt strings.
```

| Exit code | Meaning |
|-----------|---------|
| `0` | Success. |
| `1` | Something you can fix: a missing API key (`ASPIRE_MISSING_API_KEY`), a missing or unreadable config, prompts or checkpoint file (`ASPIRE_CONFIG`), invalid input (`ASPIRE_INVALID_INPUT`). |
| `2` | A failure while running. An error ASPIRE did not anticipate prints `ASPIRE_UNEXPECTED`; run again with `--debug` to see the traceback. |
| `130` | Interrupted with Ctrl+C. |

## aspire doctor

Check your environment for ASPIRE compatibility. Reports Python version, PyTorch / CUDA availability, API key presence, HuggingFace cache size, and free disk space.

```bash
aspire doctor
```

Exits with code 0 when all checks pass, 1 when any hard requirement is missing. No flags required.

## aspire diagnose

Structured environment diagnostics. Checks the same items as `doctor` plus individual dependency versions (torch, transformers, datasets, accelerate, peft, pydantic, typer, rich) and the installed ASPIRE version.

```bash
aspire diagnose
aspire diagnose --json
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--json` | `false` | Output results as a JSON object instead of a Rich table. |

Exits with code 0 when all checks pass, 1 otherwise.

## aspire teachers

List all available teacher personas.

```bash
aspire teachers
```

Prints every registered teacher name with a short description. The built-in names are: `claude`, `openai` (alias `gpt4`), `local`, `socratic` (alias `socrates`), `scientific` (alias `scientist`), `creative` (alias `innovator`), `adversarial` (alias `challenger`), `compassionate` (alias `guide`). No flags required.

## aspire dialogue

Generate an adversarial dialogue between a student and a teacher.

```bash
aspire dialogue "Your prompt here" \
    --teacher socratic \
    --turns 3 \
    --model microsoft/Phi-3-mini-4k-instruct
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--teacher`, `-t` | `socratic` | Teacher persona to use. Any registered teacher name (see `aspire teachers`). |
| `--turns`, `-n` | `3` | Number of dialogue turns (one teacher challenge + one student response = one turn). |
| `--model`, `-m` | `microsoft/Phi-3-mini-4k-instruct` | Student model. Any HuggingFace model identifier. |

Output goes to stdout. The dialogue command is useful for exploring how different teachers challenge the same prompt, generating training data, and debugging teacher behavior.

## aspire init

Initialize a training configuration file with sensible defaults.

```bash
aspire init --output my-config.yaml
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--output`, `-o` | `aspire-config.yaml` | Path for the generated configuration file. |

The generated YAML file contains all configurable parameters. Edit it to set your student model, teacher, dataset path, and training hyperparameters before running `aspire train`.

## aspire train

Train a student model using the ASPIRE pipeline.

```bash
aspire train \
    --config config.yaml \
    --prompts data/prompts.json \
    --teacher adversarial \
    --epochs 3
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--config`, `-c` | none | Path to the training configuration file (from `aspire init`). If omitted, uses built-in defaults. A path that does not exist is an error. |
| `--prompts`, `-p` | none | Path to a JSON list of prompt strings. If omitted, uses three built-in demo prompts. A missing or malformed file is an error. |
| `--teacher`, `-t` | config, else `claude` | Teacher: a name from `aspire teachers`, `local` (a model on this machine, see `--teacher-model`), or `composite` (the config's `teacher.composite_members`). |
| `--teacher-model` | config | The model for `--teacher local`: a Hugging Face name or a path. |
| `--student-model`, `-m` | config | The student model: a Hugging Face name or a path. |
| `--epochs`, `-e` | config, else `3` | Number of training epochs. |
| `--output`, `-o` | config, else `outputs` | Output directory for checkpoints, dialogue cache, and logs. |
| `--geometry` | off | Also write `geometry.json`, the run's training dynamics for ScalarScope. See [Watching Runs in ScalarScope](/aspire-si/handbook/scalarscope/). |

A flag given on the command line overrides the config file; a flag left out keeps the config's value. Training generates adversarial dialogues, trains the critic (and, through it, the student's LoRA adapter), and saves a checkpoint after each epoch. If no prompts file is provided, three demo prompts are used so you can verify the pipeline runs end to end.

A run that costs nothing, with a local teacher (a 1B student and a 3B teacher; fits a 24 GB GPU):

```bash
aspire train --config examples/local-run/local-teacher.yaml     --prompts examples/local-run/prompts.json --geometry
```

`examples/local-run/composite-teacher.yaml` adds a second local teacher; the two vote. `examples/pod-run/` holds the configs of the runs behind ScalarScope's real fixtures: a 1.5B student with 32B teachers in 4-bit, made on one rented 96 GB GPU.

A composite teacher in a config file names its members, each a registered name or `local:<model>`:

```yaml
teacher:
  default_teacher: composite
  composite_members:
    - local:Qwen/Qwen2.5-3B-Instruct
    - local:microsoft/Phi-4-mini-instruct
  composite_strategy: vote   # or rotate, specialize, random, debate
  local_load_in_4bit: false  # true loads local teachers in 4-bit
```

## aspire evaluate

Evaluate a trained checkpoint against a set of prompts.

```bash
aspire evaluate outputs/checkpoint-3 \
    --prompts data/eval.json
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--prompts`, `-p` | (required) | Path to evaluation prompts (JSON array of strings). |
| `--output`, `-o` | none | File path for evaluation results (JSON). If omitted, results print to stdout. |

The evaluate command loads the config, the student's trained LoRA adapter and the critic from the checkpoint directory, runs the student on each prompt, scores the results using the configured teacher, and prints a table of metrics (average, min, and max scores).

## aspire judge

Score responses with a checkpoint's trained critic, without calling the teacher.

```bash
aspire judge outputs/checkpoint-2 --prompt "What is 2+2?" --response "4"
aspire judge outputs/checkpoint-2 --pairs pairs.json --json
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--prompt` | none | The prompt the response answers. Use with `--response`. |
| `--response` | none | The response to score. |
| `--pairs` | none | A JSON list of `{"prompt": ..., "response": ...}` objects to score in one batch. |
| `--json` | `false` | Print the scores as JSON instead of a table. |
| `--device` | `cuda` if available | Where to load the student and critic. |

Each score is the critic's prediction of the teacher's 0-10 score. In Python, `aspire.judge.Judge.from_checkpoint(path)` gives the same scores through `score` and `score_many`.

## Project structure

```
aspire/
├── teachers/          # Pluggable teacher personas
│   ├── base.py        # BaseTeacher ABC + data structures
│   ├── claude.py      # Claude API teacher
│   ├── openai.py      # GPT-4 teacher
│   ├── local.py       # Local model teacher
│   ├── personas.py    # Socratic, Scientific, Creative, etc.
│   ├── composite.py   # Multi-teacher combinations
│   └── registry.py    # Dynamic teacher discovery and registration
│
├── critic/            # Internalized judgment models
│   ├── base.py        # BaseCritic ABC + CriticOutput
│   ├── head.py        # Lightweight MLP on student hidden states
│   ├── separate.py    # Independent encoder
│   └── shared.py      # Shared encoder with student
│
├── losses/            # Training objectives
│   ├── critic.py      # Score + reasoning alignment
│   ├── student.py     # Reward, contrastive, trajectory, coherence
│   └── combined.py    # Unified AspireLoss orchestrator
│
├── dialogue/          # Adversarial conversation engine
│   ├── generator.py   # Student-teacher dialogue generation
│   ├── manager.py     # Caching, batching, and retrieval
│   └── formatter.py   # Format dialogues for training (chat, standard, instruction)
│
├── perception/        # Experimental perception modules
│   ├── theory_of_mind.py    # Mental state tracking
│   ├── metacognition.py     # Uncertainty and self-reflection
│   ├── character.py         # Stable identity and value anchoring
│   ├── controlled_chaos.py  # Adversarial robustness training
│   ├── empathy_evaluation.py # Perception evaluation
│   ├── syntropy.py          # Coherence and resonance detection
│   └── integration.py       # Trainer integration hooks
│
├── judge.py           # Score responses with a trained critic (Judge, critic_score)
├── trainer.py         # Core training loop
├── config.py          # Pydantic configuration
└── cli.py             # Command-line interface (Typer + Rich)
```
