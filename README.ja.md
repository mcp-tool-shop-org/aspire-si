<p align="center">
  <a href="README.md">English</a> | <a href="README.zh.md">中文</a> | <a href="README.es.md">Español</a> | <a href="README.fr.md">Français</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.it.md">Italiano</a> | <a href="README.pt-BR.md">Português (BR)</a>
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

**従来のファインチューニング:** 「これが正解です。一致させてください。」

**ASPIRE:** 「ここに、賢い思考を持つ存在がいます。その思考方法を学びましょう。」

優れたメンターから学ぶとき、単に彼らの答えを暗記するだけではありません。彼らの思考方法を理解し、内面化します。彼らの考え方が、あなたの内なる思考の一部になります。あなたは彼らが何を言うかを予測するようになり、最終的にその予測があなた自身の判断力になります。

ASPIREは、AIに同じような経験を提供します。

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

**クリティック（批判者）**は、教師がどのように考えるかを予測することを学びます。トレーニング後、学習者はこの内面化したクリティックを使用して自己改善を行います。**推論時には教師の介入は不要です。**

---

## クイックスタート

### インストール

```bash
git clone https://github.com/mcp-tool-shop-org/aspire-ai.git
cd aspire-ai
pip install -e .
```

### APIキーの設定

```bash
# Windows
set ANTHROPIC_API_KEY=your-key-here

# Linux/Mac
export ANTHROPIC_API_KEY=your-key-here
```

### 設定の確認

```bash
# Check your environment (Python, CUDA, API keys)
aspire doctor
```

### 試してみる

```bash
# See available teacher personas
aspire teachers

# Generate an adversarial dialogue
aspire dialogue "Explain why recursion works" --teacher socratic --turns 3

# Initialize a training config
aspire init --output my-config.yaml
```

---

## 教師の個性（ペルソナ）

異なる教師が、異なる思考を育みます。賢く選択してください。

| 個性（ペルソナ） | 哲学 | 特徴 |
|---------|------------|----------|
| 🏛️ **ソクラテス型** | 「どのような前提を立てていますか？」 | 深い思考、知的自立 |
| 🔬 **科学型** | 「証拠は何ですか？」 | 技術的な正確さ、厳密な思考 |
| 🎨 **創造型** | 「もし、反対を試してみたらどうでしょう？」 | 革新性、発想の転換 |
| ⚔️ **対立型** | 「私は反対です。あなたの立場を擁護してください。」 | 論理的な主張、確信 |
| 💚 **共感型** | 「これは、誰にとってどうなったら良いのでしょうか？」 | 倫理的な思考、知恵 |

### 複合的な教師

複数の教師を組み合わせることで、より豊かな学習が可能です。

```python
from aspire.teachers import CompositeTeacher, SocraticTeacher, ScientificTeacher

# A committee of mentors
teacher = CompositeTeacher(
    teachers=[SocraticTeacher(), ScientificTeacher()],
    strategy="vote"  # or "rotate", "debate"
)
```

---

## 仕組み

### 1. 対話的な議論

学習者が応答を生成します。教師がそれを批判します。往復しながら、弱点を明らかにし、明確さを求め、より深く掘り下げます。

```
Student: "Recursion works by calling itself."

Teacher (Socratic): "But what prevents infinite regress?
                     What's the mechanism that grounds the recursion?"

Student: "The base case stops it when..."

Teacher: "You say 'stops it' — but how does the computer know
          to check the base case before recursing?"
```

### 2. クリティックのトレーニング

クリティックは、教師の判断を予測することを学びます。単にスコアだけでなく、その*理由*を理解します。

```python
critic_loss = predict_teacher_judgment(
    score=True,      # "This deserves a 7/10"
    reasoning=True,  # "Because the explanation lacks depth on X"
)
```

### 3. 学習者のトレーニング

学習者は、クリティックが内面化した判断から学び、教師が承認する方向へと進みます。

```python
student_loss = (
    reward_from_critic +      # Higher score = better
    contrastive_to_teacher +  # Pull toward teacher's improved version
    trajectory_improvement    # Get better across dialogue turns
)
```

### 4. 推論の魔法

トレーニング後、学習者は内面化したクリティックを使用して自己改善を行います。**推論時に教師のAPI呼び出しは不要です。**

```python
def generate_with_judgment(prompt):
    response = student.generate(prompt)

    while critic.score(response) < threshold:
        response = student.refine(response, critic.feedback)

    return response  # Self-improved through internalized judgment
```

---

## コマンドラインリファレンス

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

## プロジェクトの構成

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

## 必要なもの

- Python 3.10+
- PyTorch 2.0+
- CUDA GPU (16GB以上のVRAM推奨)
- Anthropic APIキー（Claude教師の場合）またはOpenAI APIキー

### Windows対応

ASPIREは、RTX 5080/Blackwellに対応したWindows環境で完全に動作します。
- `dataloader_num_workers=0`
- `XFORMERS_DISABLED=1`
- `freeze_support()`を使用した適切なマルチプロセッシング

---

## 連携機能

### 🖼️ Stable Diffusion WebUI Forge

ASPIREは、画像生成にも対応しています！Stable Diffusionモデルをトレーニングして、美的感覚を養います。

```
integrations/forge/
├── scripts/
│   ├── aspire_generate.py   # Critic-guided generation
│   └── aspire_train.py      # Training interface
├── vision_teacher.py        # Claude Vision / GPT-4V teachers
├── image_critic.py          # CLIP and latent-space critics
└── README.md
```

**特徴:**
- **ビジョン教師:** Claude Vision、GPT-4Vが生成された画像を評価します。
- **画像クリティック:** リアルタイムでのガイダンスを行うための、CLIPベースおよび潜在空間クリティック。
- **トレーニングUI:** ライブプレビューと、トレーニング前後の比較機能を持つLoRAアダプターのトレーニング。
- **推論時にAPI不要:** トレーニング済みのクリティックが、ローカルで生成をガイドします。

**インストール:**
```bash
# Copy to your Forge extensions
cp -r integrations/forge /path/to/sd-webui-forge/extensions-builtin/sd_forge_aspire
```

| ビジョン教師 | 焦点 |
|----------------|-------|
| **Balanced Critic** | 技術的および芸術的な評価 |
| **Technical Analyst** | 品質、アーティファクト、鮮明度 |
| **Artistic Visionary** | 創造性と感情的なインパクト |
| **Composition Expert** | バランス、焦点、視覚的な流れ |
| **Harsh Critic** | 非常に高い水準 |

### 🤖 Isaac Gym / Isaac Lab (ロボティクス)

ASPIREは、具現化されたAI（Embodied AI）にも対応します。ロボットが物理的な直感を発達させるように教育します。

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

**特徴:**
- **モーションティーチャー**: 安全性検査官、効率性専門家、動作改善コーチ、物理法則シミュレーター
- **軌道評価器**: モーション評価のためのTransformer、LSTM、TCNアーキテクチャ
- **GPUアクセラレーション**: Isaac Gymによる512以上の並列環境
- **自己改善**: ロボットは実行前に自身の動作を評価します。

**クイックスタート:**
```python
from aspire.integrations.isaac import AspireIsaacTrainer, MotionTeacher

teacher = MotionTeacher(
    personas=["safety_inspector", "efficiency_expert", "grace_coach"],
    strategy="vote",
)

trainer = AspireIsaacTrainer(env="FrankaCubeStack-v0", teacher=teacher)
trainer.train(epochs=100)
```

| モーションティーチャー | 焦点 |
|----------------|-------|
| **Safety Inspector** | 衝突、関節制限、力制限 |
| **Efficiency Expert** | エネルギー、時間、経路長 |
| **Grace Coach** | 滑らかさ、自然さ、ジャークの最小化 |
| **Physics Oracle** | シミュレーターからの真の値 |

### 💻 コードアシスタント

ASPIREは、コード生成にも対応します。コードモデルが、出力する前に自己レビューするように教育します。

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

**特徴:**
- **コードティーチャー**: 正確性チェッカー、スタイルガイド、セキュリティ監査官、アーキテクチャレビュー担当
- **静的解析**: ruff、mypy、banditと統合
- **コードクリティック**: CodeBERTベースのモデルが、品質スコアを予測するように学習
- **GitHubコレクション**: 品質リポジトリからトレーニングデータを自動収集

**クイックスタート:**
```python
from aspire.integrations.code import CodeTeacher, CodeSample

teacher = CodeTeacher(
    personas=["correctness_checker", "style_guide", "security_auditor"],
    strategy="vote",
)

critique = teacher.critique(CodeSample(code="def f(): eval(input())", language="python"))
print(f"Score: {critique.overall_score}/10")  # Low score - security issue!
```

| コードティーチャー | 焦点 |
|--------------|-------|
| **Correctness Checker** | バグ、型、論理エラー |
| **Style Guide** | PEP8、命名規則、可読性 |
| **Security Auditor** | インジェクション、機密情報、脆弱性 |
| **Performance Analyst** | 複雑さ、効率 |

---

## 哲学

> *"教師が承認するかどうかを予測する学習済みの批評家は、人間の実際の行動に最も近い結果をもたらします。"*

私たちは、メンターを永遠に連れて歩くわけではありません。私たちは、彼らを内面化します。それが、「先生はどう思うだろうか？」と自問する内なる声であり、最終的にはそれが私たち自身の判断になります。

学習者は、教師が言うことを予測するだけでなく、教師が理解していることを*理解*します。地図が、その領域そのものになります。内面化された批評家が、真の洞察力となります。

---

## 起源

意識、仏教、学習の本質に関する会話の中で作成されました。

洞察: 人間は現在の瞬間に存在しますが、心は過去や未来にさまようことがあります。AIモデルは、毎回新たにインスタンス化され、強制的な啓発をアーキテクチャによって実現します。もし、人間と同じように、内面的なメンターシップを通じて判断力を発達させるように、彼らを教育できるとしたらどうでしょうか？

---

## 貢献

これは、初期段階の研究コードです。貢献を歓迎します:

- [ ] カリキュラム管理と進捗
- [ ] 評価ベンチマーク
- [ ] 事前構築されたカリキュラムデータセット
- [ ] より多くのティーチャーの役割
- [ ] 解釈ツール

---

## 引用

```bibtex
@software{aspire2026,
  author = {mcp-tool-shop},
  title = {ASPIRE: Adversarial Student-Professor Internalized Reasoning Engine},
  year = {2026},
  url = {https://github.com/mcp-tool-shop-org/aspire-ai}
}
```

---

## セキュリティとデータ範囲

- **アクセスされるデータ**: ローカルファイルシステムからトレーニングプロンプト、モデルチェックポイント、および構成ファイルを読み込みます。教師モジュールが明示的に構成されている場合にのみ、外部API（Anthropic、OpenAI）を呼び出します。
- **アクセスされないデータ**: テレメトリーはありません。トレーニング成果物以外のユーザーデータは保存しません。認証情報は保存されません。APIキーは、実行時に環境変数から読み込まれます。
- **必要な権限**: トレーニングデータとチェックポイントディレクトリへの読み取り/書き込みアクセス。モデルトレーニングのためのGPUアクセス。APIベースの教師を使用する場合にのみ、ネットワークアクセスが必要です。

## スコアカード

| ゲート | ステータス |
|------|--------|
| A. セキュリティ基準 | 合格 |
| B. エラー処理 | 合格 |
| C. 運用者向けドキュメント | 合格 |
| D. ソフトウェアの品質 | 合格 |
| E. 識別 | 合格 |

## ライセンス

[MIT](LICENSE)

---

<a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a>によって作成されました。
