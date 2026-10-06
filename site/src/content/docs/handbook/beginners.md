---
title: Beginners Guide
description: New to ASPIRE? A plain-language walkthrough of what it does, why it matters, and how to get your first result.
sidebar:
  order: 99
---

This page is for people who are new to ASPIRE and want to understand what it does before diving into the technical details.

## What ASPIRE is

ASPIRE stands for **Adversarial Student-Professor Internalized Reasoning Engine**. It is a Python framework for fine-tuning AI language models so they develop judgment, not just knowledge.

Standard fine-tuning gives a model a set of correct answers and says "match these." ASPIRE gives a model a wise teacher and says "learn to think like this teacher does." The difference matters at inference time: a standard model produces its best guess in one shot, while an ASPIRE-trained model refines its own output using an internalized sense of quality.

The framework provides a CLI, a Python API, and integrations for image generation (Stable Diffusion Forge), robotics (Isaac Gym), and code review.

## Who it is for

ASPIRE is designed for:

- **ML researchers** exploring alternatives to RLHF and standard fine-tuning who want adversarial dialogue as a training signal.
- **AI engineers** who want their models to self-critique at inference time without calling an external API.
- **Hobbyists with a GPU** (16GB+ VRAM recommended) who want to experiment with training small language models using teacher feedback from Claude or GPT-4.

You need familiarity with Python and a basic understanding of what language model fine-tuning is. You do not need to be an expert in reinforcement learning or loss functions to use the CLI.

## Key concepts

**Student model** -- The model being trained. This is any HuggingFace causal language model (the default is Phi-3-mini). It starts with generic capabilities and develops judgment through ASPIRE training.

**Teacher model** -- The source of wisdom. Teachers challenge the student's responses through adversarial dialogue. ASPIRE ships with five teacher personas (Socratic, Scientific, Creative, Adversarial, Compassionate) backed by Claude or GPT-4 APIs.

**Critic model** -- The internalized judge. The critic learns to predict what the teacher would think of a response. After training, the student uses the critic to self-refine without any teacher API calls. Three critic architectures are available: a lightweight MLP head, a separate encoder, or a shared encoder with the student.

**Adversarial dialogue** -- The core training signal. The student generates a response, the teacher challenges it, and they go back and forth. This produces richer training data than flat question-answer pairs because it exposes gaps in reasoning that simple supervision would miss.

**Perception module** -- An experimental extension that adds theory of mind, metacognition, character persistence, and controlled chaos capabilities. This is for advanced use cases where you want agents with deeper situational awareness.

## Prerequisites

Before installing ASPIRE, make sure you have:

1. **Python 3.10 or newer** installed and available on your PATH.
2. **PyTorch 2.0 or newer** with CUDA support if you have a GPU. CPU-only works but is slow for training.
3. **A GPU with 16GB+ VRAM** is recommended for training. The default configuration uses 4-bit quantization (QLoRA) to fit within 16GB. Inference and dialogue generation work on smaller GPUs.
4. **An API key** for at least one teacher backend: set `ANTHROPIC_API_KEY` for Claude teachers or `OPENAI_API_KEY` for GPT-4 teachers.

Windows, Linux, and macOS are all supported. Windows users with RTX 5080 / Blackwell GPUs are explicitly supported with automatic compatibility settings (`dataloader_num_workers=0`, `XFORMERS_DISABLED=1`).

## First steps

Install ASPIRE from source:

```bash
git clone https://github.com/mcp-tool-shop-org/aspire-ai.git
cd aspire-ai
pip install -e .
```

Set your API key:

```bash
# Windows
set ANTHROPIC_API_KEY=your-key-here

# Linux / macOS
export ANTHROPIC_API_KEY=your-key-here
```

Verify your environment:

```bash
aspire doctor
```

This checks Python, PyTorch, CUDA, API keys, and disk space. Fix anything it reports as missing before continuing.

Run your first adversarial dialogue:

```bash
aspire dialogue "Explain why recursion works" --teacher socratic --turns 3
```

This loads a student model, creates a Socratic teacher, and generates three rounds of challenge-and-response dialogue. The output shows each challenge, the student's response, and a final evaluation score with reasoning.

## Common terms

| Term | Meaning |
|------|---------|
| LoRA | Low-Rank Adaptation. A parameter-efficient fine-tuning method that trains small adapter matrices instead of the full model. ASPIRE uses LoRA by default. |
| QLoRA | Quantized LoRA. Combines 4-bit quantization with LoRA to fit training into less VRAM. |
| Critic head | A small MLP network attached to the student model's hidden states that predicts teacher scores. The lightest critic architecture. |
| Composite teacher | Multiple teacher personas combined into a committee using strategies like vote, rotate, specialize, random, or debate. |
| Curriculum | A staged training progression (foundation, reasoning, nuance, adversarial, transfer) where teacher weights and challenge difficulty change over time. |
| Dialogue turn | One cycle of teacher challenge followed by student response. |
| Checkpoint | A saved snapshot of the student model, critic, and config at a point during training. Saved after each epoch by default. |
| Challenge type | The kind of question or objection the teacher poses. Built-in types include probe_reasoning, edge_case, devils_advocate, socratic, clarification, extension, contradiction, and steelman. |
| Evaluation dimension | An axis the teacher scores responses on. Dimensions include correctness, reasoning, nuance, adaptability, clarity, intellectual_honesty, creativity, empathy, and practicality. |

## Where to go next

- [Getting Started](/aspire-ai/handbook/getting-started/) walks through installation, configuration, and your first interaction in more detail.
- [How It Works](/aspire-ai/handbook/how-it-works/) explains the four-stage pipeline from adversarial dialogue to inference-time self-refinement.
- [Teachers](/aspire-ai/handbook/teachers/) covers each persona, composite strategies, curriculum-aware composition, and how to create custom teachers.
- [Integrations](/aspire-ai/handbook/integrations/) describes the Stable Diffusion Forge, Isaac Gym, and code assistant extensions.
- [CLI Reference](/aspire-ai/handbook/cli/) documents every command, flag, and option.

If something is not working, run `aspire doctor` for a human-readable environment check. For machine-readable output (useful for bug reports), run `aspire diagnose --json` which returns a JSON object with the status of every dependency, your GPU, and the installed ASPIRE version.
