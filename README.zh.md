<p align="center">
  <a href="README.ja.md">日本語</a> | <a href="README.md">English</a> | <a href="README.es.md">Español</a> | <a href="README.fr.md">Français</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.it.md">Italiano</a> | <a href="README.pt-BR.md">Português (BR)</a>
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
  <a href="https://github.com/mcp-tool-shop-org/aspire-si/actions/workflows/ci.yml"><img src="https://github.com/mcp-tool-shop-org/aspire-si/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
  <a href="https://codecov.io/gh/mcp-tool-shop-org/aspire-si"><img src="https://codecov.io/gh/mcp-tool-shop-org/aspire-si/branch/main/graph/badge.svg" alt="codecov" /></a>
  <a href="https://pypi.org/project/aspire-si/"><img src="https://img.shields.io/pypi/v/aspire-si" alt="PyPI" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT License" /></a>
  <a href="https://mcp-tool-shop-org.github.io/aspire-si/"><img src="https://img.shields.io/badge/Landing_Page-live-blue" alt="Landing Page" /></a>
</p>

---

## The Idea

**传统的微调：** *"这里是正确的答案。请进行匹配。"*

**ASPIRE：** *"这里是一位睿智的思想。学习像它一样思考。"*

当你从一位伟大的导师那里学习时，你不仅仅是记住他们的答案。你是在内化他们的思考方式。他们的声音会成为你内心对话的一部分。你开始预料他们会说什么，最终，这种预料会转化为你自己的判断力。

ASPIRE 为 AI 提供了同样的体验。

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

**“批评家”** 学习预测导师的思考方式。经过训练后，学生利用这个内化的“批评家”进行自我完善——**推理阶段不需要导师**。

---

## 快速入门

### 安装

```bash
git clone https://github.com/mcp-tool-shop-org/aspire-si.git
cd aspire-si
pip install -e .
```

### 设置您的 API 密钥

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

---

## 导师角色

不同的导师会塑造不同的思维方式。请谨慎选择。

| 角色 | 理念 | 特点 |
|---------|------------|----------|
| 🏛️ **苏格拉底式** | *"你基于什么假设？"* | 深刻的推理，独立的思考 |
| 🔬 **科学式** | *"你的证据是什么？"* | 技术精确，严谨的思考 |
| 🎨 **创造式** | *"如果我们尝试相反的方法会怎么样？"* | 创新，发散性思维 |
| ⚔️ **对抗式** | *"我不同意。请为你的观点辩护。"* | 有力的论点，坚定 |
| 💚 **富有同情心** | *"这件事可能会让别人感到如何？"* | 伦理推理，智慧 |

### 组合导师

结合多个导师，以获得更丰富的学习体验：

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

### 1. 对话式学习

学生生成一个回复。导师对其进行挑战。来回进行，探究弱点，要求清晰，深入思考。

```
Student: "Recursion works by calling itself."

Teacher (Socratic): "But what prevents infinite regress?
                     What's the mechanism that grounds the recursion?"

Student: "The base case stops it when..."

Teacher: "You say 'stops it' — but how does the computer know
          to check the base case before recursing?"
```

### 2. 批评家训练

“批评家” 学习预测导师的判断——不仅是分数，而是*推理过程*。

```python
critic_loss = predict_teacher_judgment(
    score=True,      # "This deserves a 7/10"
    reasoning=True,  # "Because the explanation lacks depth on X"
)
```

### 3. 学生训练

学生从“批评家”的内化判断中学习，朝着导师会认可的方向发展。

```python
student_loss = (
    reward_from_critic +      # Higher score = better
    contrastive_to_teacher +  # Pull toward teacher's improved version
    trajectory_improvement    # Get better across dialogue turns
)
```

### 4. 推理魔法

经过训练后，学生使用内化的“批评家”进行自我完善。**推理阶段不需要调用导师的 API。**

```python
def generate_with_judgment(prompt):
    response = student.generate(prompt)

    while critic.score(response) < threshold:
        response = student.refine(response, critic.feedback)

    return response  # Self-improved through internalized judgment
```

---

## 命令行参考

```bash
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

## 项目结构

```
aspire/
├── teachers/          # Pluggable teacher personas
│   ├── claude.py      # Claude API teacher
│   ├── openai.py      # GPT-4 teacher
│   ├── local.py       # Local model teacher
│   ├── personas.py    # Socratic, Scientific, Creative, etc.
│   └── composite.py   # Multi-teacher combinations
│
├── critic/            # Internalized judgment models
│   ├── head.py        # Lightweight MLP on student hidden states
│   ├── separate.py    # Independent encoder
│   └── shared.py      # Shared encoder with student
│
├── losses/            # Training objectives
│   ├── critic.py      # Score + reasoning alignment
│   └── student.py     # Reward, contrastive, trajectory
│
├── dialogue/          # Adversarial conversation engine
│   ├── generator.py   # Student-teacher dialogue
│   └── manager.py     # Caching and batching
│
├── trainer.py         # Core training loop
├── config.py          # Pydantic configuration
└── cli.py             # Command-line interface
```

---

## 依赖

- Python 3.10+
- PyTorch 2.0+
- CUDA GPU (建议 16GB+ 的显存)
- Anthropic API 密钥（用于 Claude 导师）或 OpenAI API 密钥

### Windows 兼容性

ASPIRE 完美兼容 Windows，并支持 RTX 5080/Blackwell：
- `dataloader_num_workers=0`
- `XFORMERS_DISABLED=1`
- 使用 `freeze_support()` 实现正确的多进程

---

## 集成

### 🖼️ Stable Diffusion WebUI Forge

ASPIRE 扩展到图像生成！训练 Stable Diffusion 模型，培养审美判断力。

```
integrations/forge/
├── scripts/
│   ├── aspire_generate.py   # Critic-guided generation
│   └── aspire_train.py      # Training interface
├── vision_teacher.py        # Claude Vision / GPT-4V teachers
├── image_critic.py          # CLIP and latent-space critics
└── README.md
```

**特点：**
- **视觉导师：** Claude Vision、GPT-4V 评估您生成的图像
- **图像批评家：** 基于 CLIP 和潜在空间的批评家，提供实时指导
- **训练 UI：** 训练 LoRA 适配器，具有实时预览和前后比较功能
- **推理阶段无需 API：** 训练好的批评家在本地指导生成

**安装：**
```bash
# Copy to your Forge extensions
cp -r integrations/forge /path/to/sd-webui-forge/extensions-builtin/sd_forge_aspire
```

| 视觉导师 | 关注点 |
|----------------|-------|
| **Balanced Critic** | 公平的技术和艺术评估 |
| **Technical Analyst** | 质量、瑕疵、清晰度 |
| **Artistic Visionary** | 创造力和情感冲击 |
| **Composition Expert** | 平衡、焦点、视觉流程 |
| **Harsh Critic** | 极高的标准。 |

### 🤖 Isaac Gym / Isaac Lab (机器人学)

ASPIRE项目扩展到具身人工智能领域！旨在教会机器人发展身体直觉。

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

**功能特点：**
- **运动指导模块：** 包括安全检查员、效率专家、姿态教练和物理专家。
- **运动轨迹评估模块：** 采用Transformer、LSTM和TCN等架构进行运动评估。
- **GPU加速：** 配合Isaac Gym，支持512个以上的并行环境。
- **自我优化：** 机器人会在执行动作前，先评估自身的运动。

**快速入门：**
```python
from aspire.integrations.isaac import AspireIsaacTrainer, MotionTeacher

teacher = MotionTeacher(
    personas=["safety_inspector", "efficiency_expert", "grace_coach"],
    strategy="vote",
)

trainer = AspireIsaacTrainer(env="FrankaCubeStack-v0", teacher=teacher)
trainer.train(epochs=100)
```

| 运动教练。 | 关注点 |
|----------------|-------|
| **Safety Inspector** | 碰撞、关节限制、力限制。 |
| **Efficiency Expert** | 能量、时间、路径长度。 |
| **Grace Coach** | 流畅性、自然性、减少抖动。 |
| **Physics Oracle** | 模拟器提供的真实数据。 |

### 💻 代码辅助工具

ASPIRE现在也扩展到代码生成领域！我们正在训练代码模型，使其在输出代码之前能够进行自我审查。

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
- **代码助手：** 包含代码正确性检查器、代码风格指南、安全审计工具、架构审查工具。
- **静态分析：** 集成了 ruff、mypy、bandit 等工具。
- **代码质量评估：** 基于 CodeBERT 的模型，用于预测代码质量得分。
- **GitHub 数据收集：** 自动从高质量代码仓库中收集训练数据。

**快速入门：**
```python
from aspire.integrations.code import CodeTeacher, CodeSample

teacher = CodeTeacher(
    personas=["correctness_checker", "style_guide", "security_auditor"],
    strategy="vote",
)

critique = teacher.critique(CodeSample(code="def f(): eval(input())", language="python"))
print(f"Score: {critique.overall_score}/10")  # Low score - security issue!
```

| 代码教师 | 关注点 |
|--------------|-------|
| **Correctness Checker** | 错误、类型错误、逻辑错误。 |
| **Style Guide** | PEP8规范、命名、可读性。 |
| **Security Auditor** | 注入攻击、秘密、漏洞。 |
| **Performance Analyst** | 复杂性，效率。 |

---

## 哲学理念

“一位博学的评论者，他预测教师是否会认可某种观点，这更能反映人类的真实行为方式。”

我们不会永远依赖导师的指导。我们会将他们的教诲内化于心。那个总在问“我的教授会怎么想？”的内在声音，最终会转化为我们自己的判断。

学生不仅仅是预测老师会说什么，而是*理解*了老师所理解的内容。地图成为了现实，内化的批判意识转化为真正的洞察力。

---

## 起源

这部作品是在一次关于意识、佛教以及学习本质的对话中诞生的。

洞察：人类存在于当下，但我们的思想常常流连于过去和未来。人工智能模型每次运行都是全新的——通过架构来实现“顿悟”。如果我们可以像培养人类一样，通过内化的指导来教它们发展判断力，会怎么样呢？

---

## 贡献

这部分代码处于研究的早期阶段。欢迎贡献。

- [ ] 课程管理与进度跟踪
- [ ] 评估标准
- [ ] 预设的课程数据集
- [ ] 更多教师角色模型
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

## 安全与数据范围

- **访问的数据：** 从本地文件系统中读取训练提示、模型检查点和配置文件。只有在明确配置了“教师”模块时，才会调用外部API（Anthropic、OpenAI）。
- **未访问的数据：** 不收集任何遥测数据。除了训练产生的中间文件外，不存储任何用户数据。不存储任何凭证——API密钥在运行时从环境变量中读取。
- **所需权限：** 访问训练数据和检查点目录的读/写权限。进行模型训练需要访问GPU。只有在使用基于API的“教师”模块时才需要网络访问。

## 计分卡

| 门。 | 状态。 |
|------|--------|
| A. 安全基线。 | 通过。 |
| B. 错误处理。 | 通过。 |
| C. 操作手册。 | 通过。 |
| D. 航运卫生。 | 通过。 |
| E. 身份认同。 | 通过。 |

## 许可

[麻省理工学院] (LICENSE)

---

由 <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a> 制作。
