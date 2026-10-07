<p align="center">
  <a href="README.ja.md">日本語</a> | <a href="README.md">English</a> | <a href="README.es.md">Español</a> | <a href="README.fr.md">Français</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.it.md">Italiano</a> | <a href="README.pt-BR.md">Português (BR)</a>
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

## 想法

**传统的微调：**“这里是正确的答案。请匹配它们。”

**ASPIRE：**“这里是一位睿智的导师。学习像他一样思考。”

当你向一位优秀的导师学习时，你不仅仅是记住他们的答案。你还会内化他们看待事物的方式。他们的声音会成为你内在对话的一部分。你开始预想他们会说什么，最终这种预想会变成你自己的判断力。

ASPIRE 赋予 AI 相同的体验。

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

**评论者** 学习预测导师对一个回答的看法。训练后，它会自行判断学生的回答——**在推理时不需要导师**——因此你可以选择不同的尝试，或者围绕它编写一个优化循环。

---

## 快速入门

### 安装

```bash
git clone https://github.com/mcp-tool-shop-org/aspire-si.git
cd aspire-si
pip install -e .
```

### 设置你的 API 密钥

仅适用于 Claude 或 OpenAI 导师；本地导师不需要（见下文）。

```bash
# Windows
set ANTHROPIC_API_KEY=your-key-here

# Linux/Mac
export ANTHROPIC_API_KEY=your-key-here
```

### 验证设置

```bash
# Check your environment (Python, CUDA, API keys)
aspire doctor
```

### 试用

```bash
# See available teacher personas
aspire teachers

# Generate an adversarial dialogue
aspire dialogue "Explain why recursion works" --teacher socratic --turns 3

# Initialize a training config
aspire init --output my-config.yaml
```

### 一次无需费用的运行

不需要 API 密钥：一个本地导师、一个 1B 的学生和一个 3B 的导师，在一块 24GB 的 GPU 上。

```bash
aspire train --config examples/local-run/local-teacher.yaml \
    --prompts examples/local-run/prompts.json --geometry

# Score a response with the trained critic, without the teacher
aspire judge outputs/local-teacher/checkpoint-2 \
    --prompt "Why is the sky blue?" --response "Rayleigh scattering of sunlight."
```

---

## 导师角色

不同的导师会产生不同的思维模式。明智地选择。

| 角色 | 哲学 | 产生 |
|---------|------------|----------|
| 🏛️ **苏格拉底式** | “你做了什么假设？” | 深入的推理，独立的思考 |
| 🔬 **科学式** | “你的证据是什么？” | 技术上的精确，严谨的思维 |
| 🎨 **创造式** | “如果我们尝试相反的方法会怎么样？” | 创新，横向思维 |
| ⚔️ **对抗式** | “我不同意。捍卫你的观点。” | 有力的论证，坚定的信念 |
| 💚 **富有同情心** | “有人可能会对这件事有什么感觉？” | 伦理推理，智慧 |

### 复合导师

组合多个导师以获得更丰富的学习体验：

```python
from aspire.teachers import CompositeTeacher, SocraticTeacher, ScientificTeacher

# A committee of mentors
teacher = CompositeTeacher(
    teachers=[SocraticTeacher(), ScientificTeacher()],
    strategy="vote"  # or "rotate", "debate"
)
```

---

## 工作原理

### 1. 对抗式对话

学生生成一个回答。导师挑战它。来回进行，探究弱点，要求清晰，深入挖掘。

```
Student: "Recursion works by calling itself."

Teacher (Socratic): "But what prevents infinite regress?
                     What's the mechanism that grounds the recursion?"

Student: "The base case stops it when..."

Teacher: "You say 'stops it' — but how does the computer know
          to check the base case before recursing?"
```

### 2. 评论者训练

评论者学习预测导师对一个回答的判断：它的 0-10 分。 （它还有一个关于导师推理的模块，但训练器尚未训练它。）

```python
# The critic reads the student's hidden states over the prompt and the response
# the teacher judged, and learns to predict the teacher's 0-10 score.
critic_loss = mse(critic(student_hidden_states(prompt, response)), teacher_score)
```

### 3. 学生学到的内容

评论者是一个位于学生隐藏状态上的模块，它的损失会反馈到学生的 LoRA 适配器中。因此，今天学生学习的是**有助于评论者预测导师评分的表示**。它尚未接受训练以获得更好的答案：在 `aspire.losses` 中存在奖励、对比和轨迹项，但训练器不会输入这些项，因此它们不会产生任何影响。计划在 1.3.0 版本中对学生进行评论者判断的训练（[#11](https://github.com/mcp-tool-shop-org/aspire-si/issues/11)）。

### 4. 无需导师的判断

训练后，评论者仅根据学生的隐藏状态对一个回答进行评分，因此不需要导师来判断。你可以围绕它编写一个优化循环；ASPIRE 提供了训练好的评论者和 `aspire.judge`，而不是循环：

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

从命令行：`aspire judge outputs/checkpoint-2 --prompt "..." --response "..."`。

---

## CLI 参考

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

错误会打印一个代码、一条消息以及如何处理，而不会显示回溯：

```
ASPIRE_MISSING_API_KEY  ANTHROPIC_API_KEY not found.
To fix this, set your API key: ...
```

退出代码：`0` 成功，`1` 你可以修复的问题（缺少密钥、错误的配置或提示文件），`2` 运行过程中发生的故障，`130` 中断。

---

## 在 ScalarScope 中观察运行

`aspire train --geometry`（或配置中的 `training.geometry_export: true`）会写入
`geometry.json` 位于检查点旁边：运行的训练动态，格式为
[ScalarScope](https://github.com/mcp-tool-shop-org/scalarscope) 可以读取。并排打开两个，以比较运行。

| 字段 | 它包含的内容 |
|-------|---------------|
| 轨迹 | 学生的最后一个隐藏层，在令牌和批次上进行池化，并投影到它的前两个主成分上。速度、有符号曲率（以 pi 为单位的转角，当运行几乎没有移动时会衰减）和有效维度（在窗口中的参与率）。 |
| 标量 | 每个评估维度，导师给出的分数，范围为 0 到 1。 |
| 特征值 | 每一步，维度分数在窗口中变化的程度，以分数形式表示。较大的第一个值意味着一个方向可以解释导师的判断。 |
| 教授 | 每个导师一个箭头：在状态空间中，其分数上升的方向。复合导师会为每个成员提供一个。 |
| 失败 | 在某个维度至少下降 0.1 以下其最近中值的步骤。 |

没有模型或 API 密钥？`python examples/geometry_demo.py` 模拟两次运行并写入两个
导出，这是查看视图的最快方法。

记录器（`aspire.geometry.GeometryRecorder`）在内存中保留每个步骤的一个池化向量。
对于长时间运行，请设置 `training.geometry_every` 以将多个批次平均到一个步骤。导出
在每个 epoch 结束时以及在最后也会写入，因此提前停止的运行会保留它记录的内容。

**分析实际运行结果。** 在第一次实际模型运行中（1.5B的学生模型，32B的教师模型，3个epochs；[运行报告](docs/runs/2026-10-06-pod-run.md)），每个步骤的隐藏状态主要反映了该步骤训练的*提示*是什么：两个提示之间的距离大约是三个epochs训练所移动的学生模型的10,000倍。标量值在每个epoch中重复出现，因为第一个epoch之后的epochs会重现缓存的教师评分。为了查看训练如何改变，`examples/pod-run/probe.py`和`drift.py`将相同的固定对话输入到基础学生模型和每个epoch的检查点中，并导出一个*漂移*文件，其中移除了提示标识。ScalarScope的实际测试用例包括这两种类型，并且它们都涵盖了所有三个epoch。

导出的数据采用架构 1.1。`run_metadata` 说明了如何读取步骤：`step_axis` 是
`training_step`（训练器的导出数据，按训练顺序排列）或 `checkpoint_by_item`（探针和漂移导出数据，每个检查点包含相同项目的一个块，`checkpoints` 表示数量），并且
`scalar_source` 是 `live`、`replayed`（首次重复使用缓存分数后的训练周期数）或
`fixed_per_item`。ScalarScope 会阻止读取此类步骤时将其视为时间进行比较。

---

## 项目结构

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

## 要求

- Python 3.10+
- PyTorch 2.0+
- 用于训练的CUDA GPU。使用API教师时，1-4B的学生模型可以在16GB的GPU上运行；本地教师示例（`examples/local-run/`）将学生模型和教师模型都放在一个24GB的GPU上。测试、几何演示和集成示例在CPU上运行。
- 一个教师：一个本地模型（无密钥），或者一个Anthropic或OpenAI的API密钥。

### Windows兼容性

ASPIRE完全兼容Windows，并支持RTX 5080/Blackwell：
- `dataloader_num_workers=0`
- `XFORMERS_DISABLED=1`
- 使用`freeze_support()`进行适当的多进程处理。

---

## 集成

### 🖼️ Stable Diffusion WebUI Forge

ASPIRE扩展到图像生成！训练Stable Diffusion模型以培养审美判断力。

```
integrations/forge/
├── scripts/
│   ├── aspire_generate.py   # Critic-guided generation
│   └── aspire_train.py      # Training interface
├── vision_teacher.py        # Claude Vision / GPT-4V teachers
├── image_critic.py          # CLIP and latent-space critics
└── README.md
```

**功能：**
- **视觉教师：** Claude Vision、GPT-4V对生成的图像进行评价。
- **图像评论者：** 基于CLIP和潜在空间的评论者，用于实时指导。
- **训练UI：** 训练LoRA适配器，并进行实时预览和前后对比。
- **推理时无需API：** 训练后的评论者可以本地指导生成。

**安装：**
```bash
# Copy to your Forge extensions
cp -r integrations/forge /path/to/sd-webui-forge/extensions-builtin/sd_forge_aspire
```

| 视觉教师 | 关注点 |
|----------------|-------|
| **Balanced Critic** | 公正的技术和艺术评价 |
| **Technical Analyst** | 质量、瑕疵、清晰度 |
| **Artistic Visionary** | 创造力和情感影响 |
| **Composition Expert** | 平衡、焦点、视觉流程 |
| **Harsh Critic** | 非常高的标准 |

### 🤖 Isaac Gym / Isaac Lab（机器人）

ASPIRE扩展到具身AI！教机器人培养物理直觉。

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

**功能：**
- **运动教师：** 安全检查员、效率专家、优雅教练、物理预言家。
- **轨迹评论者：** Transformer、LSTM、TCN架构，用于运动评估。
- **GPU加速：** 使用Isaac Gym，支持512+个并行环境。
- **自我完善：** 机器人在执行之前评估自己的运动。

**快速入门：**
```python
from integrations.isaac import AspireIsaacTrainer, MotionTeacher

teacher = MotionTeacher(
    personas=["safety_inspector", "efficiency_expert", "grace_coach"],
    strategy="vote",
)

trainer = AspireIsaacTrainer(env="FrankaCubeStack-v0", teacher=teacher)
trainer.train(epochs=100)
```

如果没有安装Isaac Gym，`python -m integrations.isaac.examples.basic_training`将在一个小型内置的模拟环境中，在CPU上运行相同的循环。

| 运动教师 | 关注点 |
|----------------|-------|
| **Safety Inspector** | 碰撞、关节限制、力限制 |
| **Efficiency Expert** | 能量、时间、路径长度 |
| **Grace Coach** | 平滑度、自然性、最小化抖动 |
| **Physics Oracle** | 来自模拟器的真实数据 |

### 💻 代码助手

ASPIRE扩展到代码生成！教代码模型在输出之前进行自我审查。

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

**功能：**
- **代码教师：** 正确性检查器、风格指南、安全审计员、架构审查员。
- **静态分析：** 集成ruff、mypy、bandit。
- **代码评论者：** 基于CodeBERT的模型，学习预测质量得分。
- **GitHub集合：** 自动从高质量的代码仓库中收集训练数据。

**快速入门：**
```python
from integrations.code import CodeSample, CodeTeacher, Language

teacher = CodeTeacher(
    personas=["correctness_checker", "style_guide", "security_auditor"],
    strategy="vote",
)

critique = teacher.critique(CodeSample(code="def f(): eval(input())", language=Language.PYTHON))
print(critique.weaknesses)  # ['Line 1: Code injection risk (eval of user input)', ...]
```

| 代码教师 | 关注点 |
|--------------|-------|
| **Correctness Checker** | 错误、类型、逻辑错误 |
| **Style Guide** | PEP8、命名、可读性 |
| **Security Auditor** | 注入、密钥、漏洞 |
| **Performance Analyst** | 复杂性、效率 |

---

## 哲学

> “一个能够预测教师是否会批准的、经过学习的评论者，最接近于人类的实际行为。”

我们不会永远带着导师。我们会将他们内化。那个内在的声音，它会问“我的教授会怎么想？”，最终会变成我们自己的判断。

ASPIRE将这种内在的声音构建为一个评论者：一个模型，它可以预测教师对某个响应的看法，并在教师离开后继续进行判断。教学生模型如何运用这种声音是下一步（[#11](https://github.com/mcp-tool-shop-org/aspire-si/issues/11)）。

---

## 起源

在关于意识、佛教和学习本质的对话中构建。

核心思想：人类存在于当下，但我们的思想会游离于过去和未来。AI模型每次都会被重新实例化——通过架构实现强制性的顿悟。如果我们能够像人类一样，通过内化的指导来教它们培养判断力，会怎么样呢？

---

## 贡献

这是一个早期阶段的研究代码。欢迎贡献：

- [ ] 课程管理和进度
- [ ] 评估基准
- [ ] 预构建的课程数据集
- [ ] 更多的教师角色
- [ ] 可解释性工具

---

## 引用

```bibtex
@software{aspire2026,
  author = {mcp-tool-shop},
  title = {ASPIRE: Adversarial Student-Professor Internalized Reasoning Engine},
  year = {2026},
  url = {https://github.com/mcp-tool-shop-org/aspire-si}
}
```

---

## 安全性和数据范围

- **访问的数据：** 从本地文件系统读取训练提示、模型检查点和配置文件。仅当明确配置了教师模块时，才会调用外部API（Anthropic、OpenAI）。
- **不访问的数据：** 没有遥测数据。没有用户数据存储，除了训练工件。没有凭据存储——API密钥在运行时从环境变量中读取。
- **所需的权限：** 对训练数据和检查点目录的读/写访问权限。用于模型训练的GPU访问权限。仅在使用基于API的教师时才需要网络访问权限。

## 评估

| 关卡 | 状态 |
|------|--------|
| A. 安全基线 | 通过 |
| B. 错误处理 | 通过 |
| C. 操作手册 | 通过 |
| D. 运输卫生 | 通过 |
| E. 身份验证 | 通过 |

## 许可证

[MIT](LICENSE)

---

由 <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a> 制作
