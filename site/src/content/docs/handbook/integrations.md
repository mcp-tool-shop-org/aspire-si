---
title: Integrations
description: ASPIRE integrations with Stable Diffusion Forge, Isaac Gym / Isaac Lab, and code assistants.
sidebar:
  order: 4
---

ASPIRE extends beyond text. The same adversarial student-teacher-critic pattern applies to images, robotics, and code. Each integration adapts the core pipeline to a different domain.

## Stable Diffusion Forge

ASPIRE integrates with Stable Diffusion WebUI Forge to bring adversarial training to image generation. Vision teachers critique generated images, CLIP-based critics learn to predict those critiques, and LoRA adapters are trained with real-time guidance.

### Vision teachers

| Teacher | Focus |
|---------|-------|
| Balanced Critic | Fair technical and artistic evaluation |
| Technical Analyst | Quality, artifacts, sharpness |
| Artistic Visionary | Creativity and emotional impact |
| Composition Expert | Balance, focal points, visual flow |
| Harsh Critic | Very high standards, pushes quality ceiling |

Vision teachers use Claude Vision or GPT-4V to evaluate generated images. They produce structured critiques with scores and reasoning, just like text teachers.

### Image critics

Instead of text-based critics, the Forge integration uses CLIP-based and latent-space critics. These learn to predict vision teacher scores from the image embedding directly, enabling real-time guidance during generation without API calls.

### Training workflow

1. Generate images with Stable Diffusion.
2. Vision teachers critique each image (score + reasoning).
3. An image critic trains on these critiques.
4. Train a LoRA adapter using the critic's feedback.
5. At inference time, the critic guides generation locally.

### Installation

```bash
# Copy the integration into your Forge extensions
cp -r integrations/forge /path/to/sd-webui-forge/extensions-builtin/sd_forge_aspire
```

The training UI provides live preview with before/after comparison so you can watch the model improve in real time.

## Isaac Gym / Isaac Lab

ASPIRE integrates with NVIDIA Isaac Gym and Isaac Lab to bring adversarial training to robotics. Motion teachers evaluate robot trajectories, trajectory critics learn to predict those evaluations, and robot policies train with internalized physical intuition.

### Motion teachers

| Teacher | Focus |
|---------|-------|
| Safety Inspector | Collisions, joint limits, force limits |
| Efficiency Expert | Energy consumption, time, path length |
| Grace Coach | Smoothness, naturalness, jerk minimization |
| Physics Oracle | Ground truth from simulator physics |

Motion teachers evaluate full trajectories, not individual frames. They consider the quality of motion over time, catching problems that frame-by-frame analysis would miss.

### GPU-accelerated training

Isaac Gym runs 512+ parallel environments on a single GPU. ASPIRE leverages this for massive throughput: each environment runs a different scenario, and the critic trains on all of them simultaneously.

```python
from aspire.integrations.isaac import AspireIsaacTrainer, MotionTeacher

teacher = MotionTeacher(
    personas=["safety_inspector", "efficiency_expert", "grace_coach"],
    strategy="vote",
)

trainer = AspireIsaacTrainer(
    env="FrankaCubeStack-v0",
    teacher=teacher,
)
trainer.train(epochs=100)
```

### Self-refinement for robots

After training, the robot evaluates its own planned motions before execution. The trajectory critic scores a candidate trajectory, and if the score is below threshold the policy generates an alternative. This happens entirely on-device with no API calls.

## Code Assistants

ASPIRE integrates with code generation workflows to teach models to self-review before outputting code. Code teachers evaluate correctness, style, and security; code critics internalize those evaluations; and trained models catch their own mistakes.

### Code teachers

| Teacher | Focus |
|---------|-------|
| Correctness Checker | Bugs, type errors, logic errors |
| Style Guide | PEP8 compliance, naming conventions, readability |
| Security Auditor | Injection vulnerabilities, secrets exposure, unsafe patterns |
| Performance Analyst | Algorithmic complexity, efficiency, resource usage |

### Static analysis integration

Code teachers integrate with established static analysis tools for ground-truth signals:

- **ruff** — fast Python linting and formatting
- **mypy** — static type checking
- **bandit** — security vulnerability scanning

The static analysis results become part of the teacher's evaluation, combining tool-based precision with LLM-based reasoning about code quality.

### Training code critics

```python
from aspire.integrations.code import CodeTeacher, CodeSample

teacher = CodeTeacher(
    personas=["correctness_checker", "style_guide", "security_auditor"],
    strategy="vote",
)

critique = teacher.critique(
    CodeSample(code="def f(): eval(input())", language="python")
)
print(f"Score: {critique.overall_score}/10")  # Low — security issue
```

The code critic learns to predict these multi-teacher evaluations from the code embedding alone. After training, it can flag issues in generated code without calling any teacher API.

### GitHub data collection

The integration includes a data collector that gathers training pairs from quality GitHub repositories. It extracts code samples, runs them through the teacher pipeline, and produces labeled training data for critic training.
