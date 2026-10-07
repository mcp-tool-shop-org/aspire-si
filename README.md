<p align="center">
  <a href="README.md">English</a> | <a href="README.ja.md">日本語</a> | <a href="README.zh.md">中文</a> | <a href="README.es.md">Español</a> | <a href="README.fr.md">Français</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.it.md">Italiano</a> | <a href="README.pt-BR.md">Português (BR)</a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/mcp-tool-shop-org/brand/main/logos/aspire-si/readme.png" width="400" />
</p>

<p align="center">
  <strong>Adversarial Student-Professor Internalized Reasoning Engine</strong>
</p>

<p align="center">
  <em>Teaching AI to develop judgment, not just knowledge.</em>
</p>

<p align="center">
  <a href="#the-idea">The Idea</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#teacher-personas">Teachers</a> •
  <a href="#how-it-works">How It Works</a> •
  <a href="#integrations">Integrations</a> •
  <a href="https://mcp-tool-shop-org.github.io/aspire-si/handbook/">Handbook</a>
</p>

<p align="center">
  <a href="https://github.com/mcp-tool-shop-org/aspire-si/actions/workflows/ci.yml"><img src="https://github.com/mcp-tool-shop-org/aspire-si/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
  <a href="https://codecov.io/gh/mcp-tool-shop-org/aspire-si"><img src="https://codecov.io/gh/mcp-tool-shop-org/aspire-si/branch/main/graph/badge.svg" alt="codecov" /></a>
  <a href="https://pypi.org/project/aspire-si/"><img src="https://img.shields.io/pypi/v/aspire-si" alt="PyPI" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT License" /></a>
  <a href="https://mcp-tool-shop-org.github.io/aspire-si/"><img src="https://img.shields.io/badge/Landing_Page-live-blue" alt="Landing Page" /></a>
</p>

---

## The Idea

**Traditional fine-tuning:** *"Here are the right answers. Match them."*

**ASPIRE:** *"Here is a wise mind. Learn to think like it does."*

When you learn from a great mentor, you don't just memorize their answers. You internalize their way of seeing. Their voice becomes part of your inner dialogue. You start to anticipate what they would say, and eventually that anticipation becomes your own discernment.

ASPIRE gives AI that same experience.

```
┌─────────────────────────────────────────────────────────────────┐
│                         ASPIRE SYSTEM                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   STUDENT   │    │   CRITIC    │    │   TEACHER   │         │
│  │    MODEL    │    │   MODEL     │    │    MODEL    │         │
│  │             │    │             │    │             │         │
│  │ (learning)  │    │ (internal-  │    │ (wisdom)    │         │
│  │             │    │  ized       │    │             │         │
│  │             │    │  judgment)  │    │             │         │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘         │
│         │                  │                   │                 │
│         └──────────────────┴───────────────────┘                 │
│                            │                                     │
│                   ADVERSARIAL DIALOGUE                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

The **critic** learns to predict what the teacher would think of a response. After training, it judges the student's responses on its own — **no teacher needed at inference time** — so you can choose between attempts, or write a refine loop around it.

---

## Quick Start

### Installation

```bash
git clone https://github.com/mcp-tool-shop-org/aspire-si.git
cd aspire-si
pip install -e .
```

### Set Your API Key

Only for the Claude or OpenAI teachers; a local teacher needs none (see below).

```bash
# Windows
set ANTHROPIC_API_KEY=your-key-here

# Linux/Mac
export ANTHROPIC_API_KEY=your-key-here
```

### Verify Setup

```bash
# Check your environment (Python, CUDA, API keys)
aspire doctor
```

### Try It Out

```bash
# See available teacher personas
aspire teachers

# Generate an adversarial dialogue
aspire dialogue "Explain why recursion works" --teacher socratic --turns 3

# Initialize a training config
aspire init --output my-config.yaml
```

### A Run That Costs Nothing

No API key needed: a local teacher, a 1B student and a 3B teacher, on one 24 GB GPU.

```bash
aspire train --config examples/local-run/local-teacher.yaml \
    --prompts examples/local-run/prompts.json --geometry

# Score a response with the trained critic, without the teacher
aspire judge outputs/local-teacher/checkpoint-2 \
    --prompt "Why is the sky blue?" --response "Rayleigh scattering of sunlight."
```

---

## Teacher Personas

Different teachers produce different minds. Choose wisely.

| Persona | Philosophy | Produces |
|---------|------------|----------|
| 🏛️ **Socratic** | *"What assumption are you making?"* | Deep reasoning, intellectual independence |
| 🔬 **Scientific** | *"What's your evidence?"* | Technical precision, rigorous thinking |
| 🎨 **Creative** | *"What if we tried the opposite?"* | Innovation, lateral thinking |
| ⚔️ **Adversarial** | *"I disagree. Defend your position."* | Robust arguments, conviction |
| 💚 **Compassionate** | *"How might someone feel about this?"* | Ethical reasoning, wisdom |

### Composite Teachers

Combine multiple teachers for richer learning:

```python
from aspire.teachers import CompositeTeacher, SocraticTeacher, ScientificTeacher

# A committee of mentors
teacher = CompositeTeacher(
    teachers=[SocraticTeacher(), ScientificTeacher()],
    strategy="vote"  # or "rotate", "debate"
)
```

---

## How It Works

### 1. Adversarial Dialogue

The student generates a response. The teacher challenges it. Back and forth, probing weaknesses, demanding clarity, pushing deeper.

```
Student: "Recursion works by calling itself."

Teacher (Socratic): "But what prevents infinite regress?
                     What's the mechanism that grounds the recursion?"

Student: "The base case stops it when..."

Teacher: "You say 'stops it' — but how does the computer know
          to check the base case before recursing?"
```

### 2. Critic Training

The critic learns to predict the teacher's judgment of a response: its 0-10 score. (It has a
head for the teacher's reasoning too, which the trainer does not train yet.)

```python
# The critic reads the student's hidden states over the prompt and the response
# the teacher judged, and learns to predict the teacher's 0-10 score.
critic_loss = mse(critic(student_hidden_states(prompt, response)), teacher_score)
```

### 3. What the Student Learns

The critic is a head on the student's hidden states, and its loss flows back into the
student's LoRA adapter. So today the student learns **representations that help the critic
predict the teacher's score**. It is not yet trained toward better answers: the reward,
contrastive and trajectory terms in `aspire.losses` exist, but the trainer does not feed
them, so they move nothing. Training the student on the critic's judgment is planned for
1.3.0 ([#11](https://github.com/mcp-tool-shop-org/aspire-si/issues/11)).

### 4. Judgment Without the Teacher

After training, the critic scores a response from the student's hidden states alone, so no
teacher call is needed to judge one. A refine loop is yours to write around it; ASPIRE ships
the trained critic and `aspire.judge`, not the loop:

```python
from aspire.judge import Judge

judge = Judge.from_checkpoint("outputs/checkpoint-2")  # student, critic and config

def generate_with_judgment(prompt, threshold=7.0, attempts=3):
    response = student.generate(prompt)
    for _ in range(attempts):
        if judge.score(prompt, response) >= threshold:
            break
        response = student.generate(prompt)  # or a revision prompt of your own
    return response
```

From the command line: `aspire judge outputs/checkpoint-2 --prompt "..." --response "..."`.

---

## CLI Reference

```bash
# Global options go before the command
aspire --quiet ...     # errors only
aspire --verbose ...   # also print the resolved settings
aspire --debug ...     # verbose, and show tracebacks on errors

# Check your environment
aspire doctor

# Structured environment diagnostics (machine-readable)
aspire diagnose --json

# List available teachers
aspire teachers

# Generate adversarial dialogue
aspire dialogue "Your prompt here" \
    --teacher socratic \
    --turns 3 \
    --model microsoft/Phi-3-mini-4k-instruct

# Initialize config file
aspire init --output config.yaml

# Train a model
aspire train \
    --config config.yaml \
    --prompts data/prompts.json \
    --teacher adversarial \
    --epochs 3

# Train and write a training-dynamics export for ScalarScope
aspire train --prompts data/prompts.json --geometry

# Train against a local model as the teacher (no API key)
aspire train --teacher local --teacher-model Qwen/Qwen2.5-3B-Instruct \
    --student-model meta-llama/Llama-3.2-1B-Instruct --prompts data/prompts.json

# Evaluate checkpoint
aspire evaluate outputs/checkpoint-3 \
    --prompts data/eval.json

# Score responses with a checkpoint's critic, without the teacher
aspire judge outputs/checkpoint-3 --prompt "..." --response "..."
aspire judge outputs/checkpoint-3 --pairs pairs.json --json
```

Errors print a code, a message, and what to do, without a traceback:

```
ASPIRE_MISSING_API_KEY  ANTHROPIC_API_KEY not found.
To fix this, set your API key: ...
```

Exit codes: `0` success, `1` something you can fix (a missing key, a bad config or prompts
file), `2` a failure while running, `130` interrupted.

---

## Watching a Run in ScalarScope

`aspire train --geometry` (or `training.geometry_export: true` in the config) writes
`geometry.json` next to the checkpoints: the run's training dynamics in the format
[ScalarScope](https://github.com/mcp-tool-shop-org/scalarscope) reads. Open two of them side by
side to compare runs.

| Field | What it holds |
|-------|---------------|
| Trajectory | The student's last hidden layer, pooled over tokens and the batch, projected onto its first two principal components. Velocity, signed curvature (turning angle over pi, damped when the run barely moves), and effective dimension (participation ratio in a window). |
| Scalars | Every evaluation dimension the teachers scored, 0 to 1. |
| Eigenvalues | Per step, the spectrum of how the dimension scores vary together in a window, as fractions. A large first value means one direction explains the teachers' judgements. |
| Professors | One arrow per teacher: the direction in the state space along which its score rises. A composite teacher gives one per member. |
| Failures | Steps where a dimension drops at least 0.1 below its recent median. |

No model or API key? `python examples/geometry_demo.py` simulates two runs and writes both
exports, which is the quickest way to see the views.

The recorder (`aspire.geometry.GeometryRecorder`) keeps one pooled vector per step in memory.
For long runs set `training.geometry_every` to average several batches into one step. The export
is written after every epoch as well as at the end, so a run stopped early keeps what it recorded.

**Reading a real run.** In the first real-model runs (1.5B student, 32B teachers, 3 epochs;
[run report](docs/runs/2026-10-06-pod-run.md)), each step's hidden state mostly reflected *which
prompt* the step trained on: two prompts sit about 10,000 times further apart than three epochs of
training moved the student. The scalars repeat each epoch, because epochs after the first replay the
cached teacher scores. To see what training changed, `examples/pod-run/probe.py` and `drift.py` send
the same fixed exchanges through the base student and every epoch checkpoint and write a *drift*
export with prompt identity removed. ScalarScope's real fixtures are both kinds, and all of them
cover all three epochs.

Exports are schema 1.1. `run_metadata` says how to read the steps: `step_axis` is
`training_step` (the trainer's export, in training order) or `checkpoint_by_item` (probe and drift
exports, one block of the same items per checkpoint, with `checkpoints` giving the count), and
`scalar_source` is `live`, `replayed` (epochs after the first reuse cached scores) or
`fixed_per_item`. ScalarScope withholds the comparisons that would read such steps as time.

---

## Project Structure

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
│   ├── student.py     # Reward, contrastive, trajectory, coherence (not yet fed by the trainer, #11)
│   └── combined.py    # Unified AspireLoss orchestrator
│
├── dialogue/          # Adversarial conversation engine
│   ├── generator.py   # Student-teacher dialogue generation
│   ├── manager.py     # Caching, batching, and retrieval
│   └── formatter.py   # Format dialogues for training
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

---

## Requirements

- Python 3.10+
- PyTorch 2.0+
- A CUDA GPU for training. With an API teacher, a 1-4B student fits 16 GB; the local-teacher
  example (`examples/local-run/`) holds student and teacher on one 24 GB card. The tests, the
  geometry demo and the integration examples run on CPU.
- A teacher: a local model (no key), or an Anthropic or OpenAI API key

### Windows Compatibility

ASPIRE is fully Windows-compatible with RTX 5080/Blackwell support:
- `dataloader_num_workers=0`
- `XFORMERS_DISABLED=1`
- Proper multiprocessing with `freeze_support()`

---

## Integrations

### 🖼️ Stable Diffusion WebUI Forge

ASPIRE extends to image generation! Train Stable Diffusion models to develop aesthetic judgment.

```
integrations/forge/
├── scripts/
│   ├── aspire_generate.py   # Critic-guided generation
│   └── aspire_train.py      # Training interface
├── vision_teacher.py        # Claude Vision / GPT-4V teachers
├── image_critic.py          # CLIP and latent-space critics
└── README.md
```

**Features:**
- **Vision Teachers**: Claude Vision, GPT-4V critique your generated images
- **Image Critics**: CLIP-based and latent-space critics for real-time guidance
- **Training UI**: Train LoRA adapters with live preview and before/after comparison
- **No API at inference**: Trained critic guides generation locally

**Installation:**
```bash
# Copy to your Forge extensions
cp -r integrations/forge /path/to/sd-webui-forge/extensions-builtin/sd_forge_aspire
```

| Vision Teacher | Focus |
|----------------|-------|
| **Balanced Critic** | Fair technical and artistic evaluation |
| **Technical Analyst** | Quality, artifacts, sharpness |
| **Artistic Visionary** | Creativity and emotional impact |
| **Composition Expert** | Balance, focal points, visual flow |
| **Harsh Critic** | Very high standards |

### 🤖 Isaac Gym / Isaac Lab (Robotics)

ASPIRE extends to embodied AI! Teach robots to develop physical intuition.

```
integrations/isaac/
├── motion_teacher.py       # Safety, efficiency, grace teachers
├── trajectory_critic.py    # Learns to predict motion quality
├── isaac_wrapper.py        # Environment integration
├── trainer.py              # Training loop
└── examples/
    ├── basic_training.py   # Simple reaching task
    ├── custom_teacher.py   # Assembly task teacher
    └── locomotion.py       # Quadruped walking
```

**Features:**
- **Motion Teachers**: Safety Inspector, Efficiency Expert, Grace Coach, Physics Oracle
- **Trajectory Critics**: Transformer, LSTM, TCN architectures for motion evaluation
- **GPU-Accelerated**: 512+ parallel environments with Isaac Gym
- **Self-Refinement**: Robot evaluates its own motions before execution

**Quick Start:**
```python
from integrations.isaac import AspireIsaacTrainer, MotionTeacher

teacher = MotionTeacher(
    personas=["safety_inspector", "efficiency_expert", "grace_coach"],
    strategy="vote",
)

trainer = AspireIsaacTrainer(env="FrankaCubeStack-v0", teacher=teacher)
trainer.train(epochs=100)
```

Without Isaac Gym installed, `python -m integrations.isaac.examples.basic_training` runs the same
loop on a small built-in stand-in environment, on CPU.

| Motion Teacher | Focus |
|----------------|-------|
| **Safety Inspector** | Collisions, joint limits, force limits |
| **Efficiency Expert** | Energy, time, path length |
| **Grace Coach** | Smoothness, naturalness, jerk minimization |
| **Physics Oracle** | Ground truth from simulator |

### 💻 Code Assistants

ASPIRE extends to code generation! Teach code models to self-review before outputting.

```
integrations/code/
├── code_teacher.py        # Correctness, style, security teachers
├── code_critic.py         # Learns to predict code quality
├── analysis.py            # Static analysis integration (ruff, mypy, bandit)
├── data.py                # GitHub repo collector, training pairs
├── trainer.py             # Full training pipeline
└── examples/
    ├── basic_critique.py  # Multi-teacher code review
    └── train_critic.py    # Train your own code critic
```

**Features:**
- **Code Teachers**: Correctness Checker, Style Guide, Security Auditor, Architecture Reviewer
- **Static Analysis**: Integrates with ruff, mypy, bandit
- **Code Critic**: CodeBERT-based model learns to predict quality scores
- **GitHub Collection**: Auto-collect training data from quality repos

**Quick Start:**
```python
from integrations.code import CodeSample, CodeTeacher, Language

teacher = CodeTeacher(
    personas=["correctness_checker", "style_guide", "security_auditor"],
    strategy="vote",
)

critique = teacher.critique(CodeSample(code="def f(): eval(input())", language=Language.PYTHON))
print(critique.weaknesses)  # ['Line 1: Code injection risk (eval of user input)', ...]
```

| Code Teacher | Focus |
|--------------|-------|
| **Correctness Checker** | Bugs, types, logic errors |
| **Style Guide** | PEP8, naming, readability |
| **Security Auditor** | Injection, secrets, vulnerabilities |
| **Performance Analyst** | Complexity, efficiency |

---

## The Philosophy

> *"A learned critic that predicts whether the teacher would approve hits closest to how humans actually behave."*

We don't carry our mentors around forever. We internalize them. That inner voice that asks *"what would my professor think?"* eventually becomes our own judgment.

ASPIRE builds that inner voice as a critic: a model that predicts what the teacher would think of a response, and keeps judging after the teacher is gone. Teaching the student itself to act on that voice is the next step ([#11](https://github.com/mcp-tool-shop-org/aspire-si/issues/11)).

---

## Origin

Built during a conversation about consciousness, Buddhism, and the nature of learning.

The insight: humans exist in the present moment, but our minds wander to past and future. AI models are instantiated fresh each time — forced enlightenment through architecture. What if we could teach them to develop judgment the same way humans do, through internalized mentorship?

---

## Contributing

This is early-stage research code. Contributions welcome:

- [ ] Curriculum management and progression
- [ ] Evaluation benchmarks
- [ ] Pre-built curriculum datasets
- [ ] More teacher personas
- [ ] Interpretability tools

---

## Citation

```bibtex
@software{aspire2026,
  author = {mcp-tool-shop},
  title = {ASPIRE: Adversarial Student-Professor Internalized Reasoning Engine},
  year = {2026},
  url = {https://github.com/mcp-tool-shop-org/aspire-si}
}
```

---

## Security & Data Scope

- **Data accessed:** Reads training prompts, model checkpoints, and configuration files from local filesystem. Calls external APIs (Anthropic, OpenAI) only when teacher modules are explicitly configured.
- **Data NOT accessed:** No telemetry. No user data storage beyond training artifacts. No credential storage — API keys are read from environment variables at runtime.
- **Permissions required:** Read/write access to training data and checkpoint directories. GPU access for model training. Network access only when using API-based teachers.

## Scorecard

| Gate | Status |
|------|--------|
| A. Security Baseline | PASS |
| B. Error Handling | PASS |
| C. Operator Docs | PASS |
| D. Shipping Hygiene | PASS |
| E. Identity | PASS |

## License

[MIT](LICENSE)

---

Built by <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a>
