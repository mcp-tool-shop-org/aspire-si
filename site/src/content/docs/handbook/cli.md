---
title: CLI Reference
description: Complete command-line reference for aspire-ai — every command, flag, and option.
sidebar:
  order: 5
---

ASPIRE provides a command-line interface for generating dialogues, training models, and evaluating checkpoints. Every command supports `--help` for inline documentation.

## Global options

| Flag | Description |
|------|-------------|
| `--version`, `-V` | Print the installed ASPIRE version and exit. |
| `--help` | Show help for any command. |

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
| `--config`, `-c` | none | Path to the training configuration file (from `aspire init`). If omitted, uses built-in defaults. |
| `--prompts`, `-p` | none | Path to a JSON file containing training prompts. If omitted, uses three built-in demo prompts. |
| `--teacher`, `-t` | `claude` | Teacher model to use. Overrides the value in the config file. |
| `--epochs`, `-e` | `3` | Number of training epochs. Overrides the config. |
| `--output`, `-o` | `outputs` | Output directory for checkpoints, dialogue cache, and logs. |

Training generates adversarial dialogues, trains the critic and student, and saves a checkpoint after each epoch. If no prompts file is provided, three demo prompts are used so you can verify the pipeline runs end to end.

## aspire evaluate

Evaluate a trained checkpoint against a set of prompts.

```bash
aspire evaluate checkpoints/epoch-3 \
    --prompts data/eval.json
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--prompts`, `-p` | (required) | Path to evaluation prompts (JSON array of strings). |
| `--output`, `-o` | none | File path for evaluation results (JSON). If omitted, results print to stdout. |

The evaluate command loads the config and model from the checkpoint directory, runs the student on each prompt, scores the results using the configured teacher, and prints a table of metrics (average, min, and max scores).

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
├── trainer.py         # Core training loop
├── config.py          # Pydantic configuration
└── cli.py             # Command-line interface (Typer + Rich)
```
