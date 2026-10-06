<p align="center">
  <a href="README.ja.md">日本語</a> | <a href="README.zh.md">中文</a> | <a href="README.es.md">Español</a> | <a href="README.fr.md">Français</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.it.md">Italiano</a> | <a href="README.pt-BR.md">Português (BR)</a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/mcp-tool-shop-org/brand/main/logos/aspire-ai/readme.png" width="400" />
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
  <a href="#documentation">Docs</a>
</p>

<p align="center">
  <a href="https://github.com/mcp-tool-shop-org/aspire-ai/actions/workflows/ci.yml"><img src="https://github.com/mcp-tool-shop-org/aspire-ai/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
  <a href="https://codecov.io/gh/mcp-tool-shop-org/aspire-ai"><img src="https://codecov.io/gh/mcp-tool-shop-org/aspire-ai/branch/main/graph/badge.svg" alt="codecov" /></a>
  <a href="https://pypi.org/project/aspire-ai/"><img src="https://img.shields.io/pypi/v/aspire-ai" alt="PyPI" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT License" /></a>
  <a href="https://mcp-tool-shop-org.github.io/aspire-ai/"><img src="https://img.shields.io/badge/Landing_Page-live-blue" alt="Landing Page" /></a>
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

The **critic** learns to predict what the teacher would think. After training, the student uses this internalized critic to self-refine — **no teacher needed at inference time**.

---

## Quick Start

### Installation

```bash
git clone https://github.com/mcp-tool-shop-org/aspire-ai.git
cd aspire-ai
pip install -e .
```

### Set Your API Key

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

The critic learns to predict the teacher's judgment — not just the score, but the *reasoning*.

```python
critic_loss = predict_teacher_judgment(
    score=True,      # "This deserves a 7/10"
    reasoning=True,  # "Because the explanation lacks depth on X"
)
```

### 3. Student Training

The student learns from the critic's internalized judgment, pulling toward what the teacher would approve.

```python
student_loss = (
    reward_from_critic +      # Higher score = better
    contrastive_to_teacher +  # Pull toward teacher's improved version
    trajectory_improvement    # Get better across dialogue turns
)
```

### 4. Inference Magic

After training, the student self-refines using the internalized critic. **No teacher API calls needed.**

```python
def generate_with_judgment(prompt):
    response = student.generate(prompt)

    while critic.score(response) < threshold:
        response = student.refine(response, critic.feedback)

    return response  # Self-improved through internalized judgment
```

---

## CLI Reference

```bash
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

# Evaluate checkpoint
aspire evaluate checkpoints/epoch-3 \
    --prompts data/eval.json
```

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
│   ├── student.py     # Reward, contrastive, trajectory, coherence
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
├── trainer.py         # Core training loop
├── config.py          # Pydantic configuration
└── cli.py             # Command-line interface (Typer + Rich)
```

---

## Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA GPU (16GB+ VRAM recommended)
- Anthropic API key (for Claude teacher) or OpenAI API key

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
from aspire.integrations.isaac import AspireIsaacTrainer, MotionTeacher

teacher = MotionTeacher(
    personas=["safety_inspector", "efficiency_expert", "grace_coach"],
    strategy="vote",
)

trainer = AspireIsaacTrainer(env="FrankaCubeStack-v0", teacher=teacher)
trainer.train(epochs=100)
```

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
from aspire.integrations.code import CodeTeacher, CodeSample

teacher = CodeTeacher(
    personas=["correctness_checker", "style_guide", "security_auditor"],
    strategy="vote",
)

critique = teacher.critique(CodeSample(code="def f(): eval(input())", language="python"))
print(f"Score: {critique.overall_score}/10")  # Low score - security issue!
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

The student doesn't just predict what the teacher would say — it *understands* what the teacher understands. The map becomes the territory. The internalized critic becomes genuine discernment.

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
  url = {https://github.com/mcp-tool-shop-org/aspire-ai}
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
