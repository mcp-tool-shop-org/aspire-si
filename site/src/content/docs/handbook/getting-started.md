---
title: Getting Started
description: Install aspire-ai, set your API key, and run your first adversarial dialogue.
sidebar:
  order: 1
---

This guide walks you through installation, configuration, and your first interaction with ASPIRE.

## Requirements

- **Python 3.10+**
- **PyTorch 2.0+**
- **CUDA GPU** with 16GB+ VRAM recommended (training is GPU-intensive)
- **Anthropic API key** for the Claude teacher, or an OpenAI API key for GPT-4 teachers

Windows is fully supported, including RTX 5080 / Blackwell GPUs.

## Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/mcp-tool-shop-org/aspire-ai.git
cd aspire-ai
pip install -e .
```

## Set your API key

ASPIRE reads API keys from environment variables. Set the one that matches the teacher you plan to use.

```bash
# Windows
set ANTHROPIC_API_KEY=your-key-here

# Linux / macOS
export ANTHROPIC_API_KEY=your-key-here
```

You can also use `OPENAI_API_KEY` if you prefer GPT-4 as your teacher model.

## Verify your environment

The `doctor` command checks Python version, CUDA availability, API key presence, and dependency health:

```bash
aspire doctor
```

If anything is missing the output will tell you exactly what to fix.

## Try it out

### List available teachers

```bash
aspire teachers
```

This prints every built-in teacher persona with a short description of its philosophy and what kind of thinking it produces.

### Generate an adversarial dialogue

```bash
aspire dialogue "Explain why recursion works" --teacher socratic --turns 3
```

The student generates a response. The Socratic teacher challenges it. They go back and forth for three turns, with each round pushing the student toward deeper, clearer reasoning.

### Initialize a training config

```bash
aspire init --output my-config.yaml
```

This creates a configuration file with sensible defaults. Edit it to set your model, teacher, dataset, and training hyperparameters before running `aspire train`.

## Next steps

- Read [How It Works](/aspire-ai/handbook/how-it-works/) to understand the four-stage pipeline.
- Explore [Teachers](/aspire-ai/handbook/teachers/) to learn about each persona and composite strategies.
- See the full [CLI Reference](/aspire-ai/handbook/cli/) for every command and flag.
