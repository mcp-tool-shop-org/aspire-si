<p align="center">
  <a href="README.md">English</a> | <a href="README.zh.md">中文</a> | <a href="README.es.md">Español</a> | <a href="README.fr.md">Français</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.it.md">Italiano</a> | <a href="README.pt-BR.md">Português (BR)</a>
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

## アイデア

**従来のファインチューニング:** 「これが正しい答えです。これに一致させてください。」

**ASPIRE:** 「ここに賢い思考を持つ存在がいます。その思考方法を学んでください。」

優れた指導者から学ぶとき、あなたは単に彼らの答えを暗記するだけではありません。あなたは彼らのものの見方を内面化します。彼らの声は、あなたの内なる対話の一部になります。あなたは彼らが何を言うかを予測し始め、最終的にはその予測があなた自身の洞察力になります。

ASPIREは、AIに同じ経験を与えます。

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

**批評家**は、教師が特定の応答をどのように評価するかを予測することを学びます。トレーニング後、教師なしで学生の応答を独自に評価します。したがって、試行錯誤の中から選択したり、それを改善するためのループを作成したりできます。

---

## クイックスタート

### インストール

```bash
git clone https://github.com/mcp-tool-shop-org/aspire-si.git
cd aspire-si
pip install -e .
```

### APIキーを設定する

ClaudeまたはOpenAIの教師にのみ適用されます。ローカルの教師には必要ありません（以下を参照）。

```bash
# Windows
set ANTHROPIC_API_KEY=your-key-here

# Linux/Mac
export ANTHROPIC_API_KEY=your-key-here
```

### セットアップの確認

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

### 費用のかからない実行

APIキーは不要です。ローカルの教師、10億パラメータの学生、30億パラメータの教師を、1つの24GB GPUで実行します。

```bash
aspire train --config examples/local-run/local-teacher.yaml \
    --prompts examples/local-run/prompts.json --geometry

# Score a response with the trained critic, without the teacher
aspire judge outputs/local-teacher/checkpoint-2 \
    --prompt "Why is the sky blue?" --response "Rayleigh scattering of sunlight."
```

---

## 教師の個性

異なる教師は、異なる思考を生み出します。賢く選択してください。

| 個性 | 哲学 | 生成するもの |
|---------|------------|----------|
| 🏛️ **ソクラテス的** | 「どのような前提に基づいていますか？」 | 深い推論、知的な独立性 |
| 🔬 **科学的** | 「どのような証拠がありますか？」 | 技術的な正確さ、厳密な思考 |
| 🎨 **創造的** | 「もし逆のことを試したらどうなるでしょうか？」 | 革新、横方向の思考 |
| ⚔️ **敵対的** | 「私は同意しません。あなたの立場を擁護してください。」 | 堅牢な議論、確信 |
| 💚 **思いやりのある** | 「誰かがこれについてどのように感じるでしょうか？」 | 倫理的な推論、知恵 |

### 複合教師

より豊かな学習のために、複数の教師を組み合わせます。

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

### 1. 敵対的な対話

学生は応答を生成します。教師はそれに異議を唱えます。往復しながら、弱点を突き止め、明確さを求め、より深く掘り下げます。

```
Student: "Recursion works by calling itself."

Teacher (Socratic): "But what prevents infinite regress?
                     What's the mechanism that grounds the recursion?"

Student: "The base case stops it when..."

Teacher: "You say 'stops it' — but how does the computer know
          to check the base case before recursing?"
```

### 2. 批評家のトレーニング

批評家は、教師が応答をどのように評価するか（0〜10のスコア）を予測することを学びます。（また、教師の推論に関するヘッドも持っていますが、トレーナーはまだそれをトレーニングしていません。）

```python
# The critic reads the student's hidden states over the prompt and the response
# the teacher judged, and learns to predict the teacher's 0-10 score.
critic_loss = mse(critic(student_hidden_states(prompt, response)), teacher_score)
```

### 3. 学生が学ぶこと

批評家は、学生の隠れ状態にあるヘッドであり、その損失は学生のLoRAアダプターにフィードバックされます。したがって、今日、学生は**批評家が教師のスコアを予測するのに役立つ表現**を学習します。まだより良い答えを出すようにトレーニングされていません。報酬、コントラスト、および軌跡項は`aspire.losses`に存在しますが、トレーナーはそれらをフィードしていないため、何も変化しません。批評家の判断に基づいて学生をトレーニングすることは、1.3.0で計画されています（[#11](https://github.com/mcp-tool-shop-org/aspire-si/issues/11)）。

### 4. 教師なしの判断

トレーニング後、批評家は学生の隠れ状態のみから応答を評価するため、1つの応答を評価するために教師を呼び出す必要はありません。それを改善するためのループは自分で作成できます。ASPIREは、トレーニングされた批評家と`aspire.judge`を、ループではなく提供します。

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

コマンドラインから：`aspire judge outputs/checkpoint-2 --prompt "..." --response "..."`。

---

## CLIリファレンス

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

エラーが発生した場合、トレースバックなしで、コード、メッセージ、および対処方法が表示されます。

```
ASPIRE_MISSING_API_KEY  ANTHROPIC_API_KEY not found.
To fix this, set your API key: ...
```

終了コード：`0`（成功）、`1`（修正可能な問題（APIキーの欠落、無効な構成またはプロンプトファイル））、`2`（実行中のエラー）、`130`（中断）。

---

## ScalarScopeで実行を監視する

`aspire train --geometry`（または構成の`training.geometry_export: true`）は、チェックポイントの横に`geometry.json`を書き込みます。これは、ScalarScope（[ScalarScope](https://github.com/mcp-tool-shop-org/scalarscope)）が読み取る形式で、実行のトレーニングダイナミクスを示します。2つを並べて開いて、実行を比較します。

| フィールド | 内容 |
|-------|---------------|
| 軌跡 | 学生の最後の隠れ層で、トークンとバッチにわたってプールされ、最初の2つの主成分に投影されます。速度、符号付き曲率（πで減衰された回転角、実行がほとんど動かない場合）、および有効次元（ウィンドウ内の参加比）。 |
| スカラー | 教師が評価したすべての評価次元（0〜1）。 |
| 固有値 | 各ステップで、次元スコアがウィンドウ内でどのように変化するかを示すスペクトル（分数として）。大きな最初の値は、1つの方向が教師の判断を説明することを意味します。 |
| 教授 | 教師ごとに1つの矢印：状態空間内の、そのスコアが上昇する方向。複合教師は、メンバーごとに1つを提供します。 |
| 失敗 | 次元が最近の平均値から少なくとも0.1低下したステップ。 |

モデルまたはAPIキーがない場合、`python examples/geometry_demo.py`は2つの実行をシミュレートし、両方のエクスポートを書き込みます。これは、ビューをすばやく確認できる最も簡単な方法です。

レコーダー（`aspire.geometry.GeometryRecorder`）は、各ステップで1つのプールされたベクトルをメモリに保持します。長い実行の場合は、`training.geometry_every`を設定して、いくつかのバッチを1つのステップに平均化します。エクスポートは、各エポックの終了時と、実行の最後に書き込まれるため、途中で停止した実行でも、記録されたものを保持します。

**実際の実行結果の確認。** 最初の実際のモデル実行（1.5Bの生徒、32Bの教師、3エポック；[実行レポート](docs/runs/2026-10-06-pod-run.md)）では、各ステップの隠れ状態は、主にそのステップが学習した*プロンプト*を反映していました。2つのプロンプトは、3エポックの学習によって生徒が変化するよりも、約10,000倍も離れています。スカラー値は各エポックで繰り返されます。これは、最初のエポック以降のエポックでは、キャッシュされた教師のスコアが再利用されるためです。学習によって何が変化したかを確認するには、`examples/pod-run/probe.py`と`drift.py`を使用して、同じ固定のプロンプトをベースの生徒モデルと各エポックのチェックポイントに送信し、プロンプトの識別子を削除した*ドリフト*エクスポートを作成します。ScalarScopeの実際のフィクスチャは、これら2つの種類のどちらかであり、すべて3つのエポックを網羅しています。

エクスポートはスキーマ1.1です。`run_metadata`には、手順の読み込み方法が記載されています。`step_axis`は、`training_step`（トレーナーによるエクスポートで、トレーニングの順に並んでいます）または`checkpoint_by_item`（プローブとドリフトのエクスポートで、各チェックポイントごとに同じ項目のブロックが1つずつ含まれ、`checkpoints`にはその数が示されます）です。そして、`scalar_source`は`live`、`replayed`（最初の再利用以降のキャッシュされたスコアの後のエポック数）または`fixed_per_item`です。ScalarScopeは、これらの手順を時間として読み込む比較を抑制します。

---

## プロジェクト構造

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

## 要件

- Python 3.10以上
- PyTorch 2.0以上
- トレーニング用のCUDA GPU。API教師を使用する場合、1〜4Bの生徒モデルは16GBに収まります。ローカル教師の例（`examples/local-run/`）では、生徒と教師を1つの24GBのカードに格納します。テスト、ジオメトリデモ、および統合例はCPUで実行されます。
- 教師：ローカルモデル（キーなし）、またはAnthropicまたはOpenAIのAPIキー

### Windowsとの互換性

ASPIREは、RTX 5080/Blackwellのサポートにより、完全にWindowsに対応しています。
- `dataloader_num_workers=0`
- `XFORMERS_DISABLED=1`
- `freeze_support()`を使用した適切なマルチプロセッシング

---

## 統合

### 🖼️ Stable Diffusion WebUI Forge

ASPIREは、画像生成にも対応します！Stable Diffusionモデルをトレーニングして、美的判断力を高めます。

```
integrations/forge/
├── scripts/
│   ├── aspire_generate.py   # Critic-guided generation
│   └── aspire_train.py      # Training interface
├── vision_teacher.py        # Claude Vision / GPT-4V teachers
├── image_critic.py          # CLIP and latent-space critics
└── README.md
```

**機能：**
- **ビジョン教師：** Claude Vision、GPT-4Vが生成された画像を評価します。
- **画像クリティック：** リアルタイムのガイダンスのための、CLIPベースおよび潜在空間のクリティック。
- **トレーニングUI：** ライブプレビューと、変更前/後の比較を使用して、LoRAアダプターをトレーニングします。
- **推論時のAPI不要：** トレーニングされたクリティックが、ローカルで生成をガイドします。

**インストール：**
```bash
# Copy to your Forge extensions
cp -r integrations/forge /path/to/sd-webui-forge/extensions-builtin/sd_forge_aspire
```

| ビジョン教師 | 焦点 |
|----------------|-------|
| **Balanced Critic** | 公正な技術的および芸術的評価 |
| **Technical Analyst** | 品質、アーティファクト、鮮明度 |
| **Artistic Visionary** | 創造性と感情的なインパクト |
| **Composition Expert** | バランス、焦点、視覚的な流れ |
| **Harsh Critic** | 非常に高い基準 |

### 🤖 Isaac Gym / Isaac Lab（ロボット工学）

ASPIREは、具現化されたAIにも対応します！ロボットに物理的な直感を開発させます。

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

**機能：**
- **モーション教師：** 安全検査官、効率専門家、優雅さコーチ、物理オラクル
- **軌道クリティック：** モーション評価のためのTransformer、LSTM、TCNアーキテクチャ
- **GPUアクセラレーション：** Isaac Gymを使用した512以上の並列環境
- **自己改善：** ロボットは実行前に、自身のモーションを評価します。

**クイックスタート：**
```python
from integrations.isaac import AspireIsaacTrainer, MotionTeacher

teacher = MotionTeacher(
    personas=["safety_inspector", "efficiency_expert", "grace_coach"],
    strategy="vote",
)

trainer = AspireIsaacTrainer(env="FrankaCubeStack-v0", teacher=teacher)
trainer.train(epochs=100)
```

Isaac Gymがインストールされていない場合、`python -m integrations.isaac.examples.basic_training`は、CPU上で、小さな組み込みの代替環境で同じループを実行します。

| モーション教師 | 焦点 |
|----------------|-------|
| **Safety Inspector** | 衝突、関節の制限、力の制限 |
| **Efficiency Expert** | エネルギー、時間、パスの長さ |
| **Grace Coach** | 滑らかさ、自然さ、ジャークの最小化 |
| **Physics Oracle** | シミュレーターからの真の値 |

### 💻 コードアシスタント

ASPIREは、コード生成にも対応します！コードモデルに、出力前に自己レビューさせます。

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

**機能：**
- **コード教師：** 正確性チェッカー、スタイルガイド、セキュリティ監査者、アーキテクチャレビュー担当者
- **静的解析：** ruff、mypy、banditと統合
- **コードクリティック：** CodeBERTベースのモデルが、品質スコアを予測するように学習します。
- **GitHubコレクション：** 品質の高いリポジトリから、トレーニングデータを自動的に収集します。

**クイックスタート：**
```python
from integrations.code import CodeSample, CodeTeacher, Language

teacher = CodeTeacher(
    personas=["correctness_checker", "style_guide", "security_auditor"],
    strategy="vote",
)

critique = teacher.critique(CodeSample(code="def f(): eval(input())", language=Language.PYTHON))
print(critique.weaknesses)  # ['Line 1: Code injection risk (eval of user input)', ...]
```

| コード教師 | 焦点 |
|--------------|-------|
| **Correctness Checker** | バグ、型、論理エラー |
| **Style Guide** | PEP8、命名、可読性 |
| **Security Auditor** | インジェクション、機密情報、脆弱性 |
| **Performance Analyst** | 複雑さ、効率 |

---

## 哲学

> 「教師が承認するかどうかを予測する学習済みのクリティックは、人間の実際の行動に最も近い。」

私たちは、メンターを永遠にそばに置いておくわけではありません。私たちは、彼らを内面化します。「先生ならどう思うだろうか？」と自問する内なる声は、最終的には私たち自身の判断になります。

ASPIREは、その内なる声をクリティックとして構築します。それは、教師が応答についてどう思うかを予測するモデルであり、教師がいなくなっても判断し続けます。生徒自身に、その声に基づいて行動させるように教えることが、次のステップです（[#11](https://github.com/mcp-tool-shop-org/aspire-si/issues/11)）。

---

## 起源

意識、仏教、学習の本質についての会話中に構築されました。

洞察：人間は現在に存在しますが、私たちの心は過去と未来をさまよいます。AIモデルは、毎回新たにインスタンス化されます。アーキテクチャを通じて強制的な悟り。もし、人間が内面化された指導を通じて行うように、AIに判断力を開発させることができたらどうでしょうか？

---

## 貢献

これは初期段階の研究コードです。貢献を歓迎します。

- [ ] カリキュラムの管理と進捗
- [ ] 評価ベンチマーク
- [ ] 事前に構築されたカリキュラムデータセット
- [ ] より多くの教師のペルソナ
- [ ] 解釈可能性ツール

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

## セキュリティとデータ範囲

- **アクセスされるデータ：** ローカルファイルシステムから、トレーニングプロンプト、モデルチェックポイント、および構成ファイルを読み取ります。教師モジュールが明示的に構成されている場合にのみ、外部API（Anthropic、OpenAI）を呼び出します。
- **アクセスされないデータ：** テレメトリーはありません。トレーニング成果物以外のユーザーデータの保存はありません。認証情報は保存されません。APIキーは、実行時に環境変数から読み取られます。
- **必要な権限：** トレーニングデータとチェックポイントディレクトリへの読み取り/書き込みアクセス。モデルトレーニングのためのGPUアクセス。APIベースの教師を使用する場合のみ、ネットワークアクセスが必要です。

## スコアカード

| ゲート | 状態 |
|------|--------|
| A. セキュリティベースライン | PASS |
| B. エラー処理 | PASS |
| C. オペレーター向けドキュメント | PASS |
| D. 配送時の衛生管理 | PASS |
| E. 識別情報 | PASS |

## ライセンス

[MIT](LICENSE)

---

MCP Tool Shop (<a href="https://mcp-tool-shop.github.io/">https://mcp-tool-shop.github.io/</a>) によって作成
