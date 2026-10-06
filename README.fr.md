<p align="center">
  <a href="README.ja.md">日本語</a> | <a href="README.zh.md">中文</a> | <a href="README.es.md">Español</a> | <a href="README.md">English</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.it.md">Italiano</a> | <a href="README.pt-BR.md">Português (BR)</a>
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

## L'idée

**Affinage traditionnel :** *"Voici les bonnes réponses. Associez-les."*

**ASPIRE :** *"Voici un esprit éclairé. Apprenez à penser comme lui."*

Lorsque vous apprenez d'un excellent mentor, vous ne vous contentez pas de mémoriser ses réponses. Vous intériorisez sa façon de voir. Sa voix devient partie intégrante de votre dialogue intérieur. Vous commencez à anticiper ce qu'il dirait, et cette anticipation devient progressivement votre propre discernement.

ASPIRE offre à l'IA la même expérience.

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

Le **critique** apprend à prédire ce que le professeur penserait. Après l'entraînement, l'élève utilise ce critique internalisé pour s'améliorer continuellement, **sans avoir besoin d'un professeur au moment de l'inférence**.

---

## Démarrage rapide

### Installation

```bash
git clone https://github.com/mcp-tool-shop-org/aspire-ai.git
cd aspire-ai
pip install -e .
```

### Définissez votre clé API

```bash
# Windows
set ANTHROPIC_API_KEY=your-key-here

# Linux/Mac
export ANTHROPIC_API_KEY=your-key-here
```

### Vérifiez la configuration

```bash
# Check your environment (Python, CUDA, API keys)
aspire doctor
```

### Essayez-le

```bash
# See available teacher personas
aspire teachers

# Generate an adversarial dialogue
aspire dialogue "Explain why recursion works" --teacher socratic --turns 3

# Initialize a training config
aspire init --output my-config.yaml
```

---

## Personnalités de professeurs

Différents professeurs produisent différents types d'esprit. Choisissez judicieusement.

| Personnalité | Philosophie | Produit |
|---------|------------|----------|
| 🏛️ **Socrate** | *"Quelle hypothèse faites-vous ?"* | Raisonnement approfondi, indépendance intellectuelle |
| 🔬 **Scientifique** | *"Quelles sont vos preuves ?"* | Précision technique, pensée rigoureuse |
| 🎨 **Créatif** | *"Et si nous essayions le contraire ?"* | Innovation, pensée latérale |
| ⚔️ **Adversaire** | *"Je ne suis pas d'accord. Défendez votre position."* | Arguments solides, conviction |
| 💚 **Compatissant** | *"Comment quelqu'un pourrait-il se sentir à ce sujet ?"* | Raisonnement éthique, sagesse |

### Professeurs composites

Combinez plusieurs professeurs pour un apprentissage plus riche :

```python
from aspire.teachers import CompositeTeacher, SocraticTeacher, ScientificTeacher

# A committee of mentors
teacher = CompositeTeacher(
    teachers=[SocraticTeacher(), ScientificTeacher()],
    strategy="vote"  # or "rotate", "debate"
)
```

---

## Fonctionnement

### 1. Dialogue contradictoire

L'élève génère une réponse. Le professeur la conteste. Alternativement, il explore les faiblesses, exige de la clarté et pousse à approfondir.

```
Student: "Recursion works by calling itself."

Teacher (Socratic): "But what prevents infinite regress?
                     What's the mechanism that grounds the recursion?"

Student: "The base case stops it when..."

Teacher: "You say 'stops it' — but how does the computer know
          to check the base case before recursing?"
```

### 2. Formation du critique

Le critique apprend à prédire le jugement du professeur, non seulement le score, mais aussi le *raisonnement*.

```python
critic_loss = predict_teacher_judgment(
    score=True,      # "This deserves a 7/10"
    reasoning=True,  # "Because the explanation lacks depth on X"
)
```

### 3. Formation de l'élève

L'élève apprend du jugement internalisé du critique, en se dirigeant vers ce que le professeur approuverait.

```python
student_loss = (
    reward_from_critic +      # Higher score = better
    contrastive_to_teacher +  # Pull toward teacher's improved version
    trajectory_improvement    # Get better across dialogue turns
)
```

### 4. Magie de l'inférence

Après l'entraînement, l'élève s'améliore continuellement grâce au critique internalisé. **Aucun appel d'API du professeur n'est nécessaire.**

```python
def generate_with_judgment(prompt):
    response = student.generate(prompt)

    while critic.score(response) < threshold:
        response = student.refine(response, critic.feedback)

    return response  # Self-improved through internalized judgment
```

---

## Référence de l'interface en ligne de commande (CLI)

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

## Structure du projet

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

## Prérequis

- Python 3.10+
- PyTorch 2.0+
- GPU CUDA (16 Go de VRAM recommandés)
- Clé API Anthropic (pour le professeur Claude) ou clé API OpenAI

### Compatibilité Windows

ASPIRE est entièrement compatible avec Windows et prend en charge RTX 5080/Blackwell :
- `dataloader_num_workers=0`
- `XFORMERS_DISABLED=1`
- Multiprocessing correct avec `freeze_support()`

---

## Intégrations

### 🖼️ Stable Diffusion WebUI Forge

ASPIRE s'étend à la génération d'images ! Entraînez des modèles Stable Diffusion pour développer un jugement esthétique.

```
integrations/forge/
├── scripts/
│   ├── aspire_generate.py   # Critic-guided generation
│   └── aspire_train.py      # Training interface
├── vision_teacher.py        # Claude Vision / GPT-4V teachers
├── image_critic.py          # CLIP and latent-space critics
└── README.md
```

**Fonctionnalités :**
- **Professeurs visuels :** Claude Vision, GPT-4V évaluent vos images générées.
- **Critiques d'images :** Critiques basés sur CLIP et dans l'espace latent pour un guidage en temps réel.
- **Interface d'entraînement :** Entraînez des adaptateurs LoRA avec un aperçu en direct et une comparaison avant/après.
- **Pas d'API à l'inférence :** Le critique entraîné guide la génération localement.

**Installation :**
```bash
# Copy to your Forge extensions
cp -r integrations/forge /path/to/sd-webui-forge/extensions-builtin/sd_forge_aspire
```

| Professeur visuel | Objectif |
|----------------|-------|
| **Balanced Critic** | Évaluation technique et artistique équitable |
| **Technical Analyst** | Qualité, artefacts, netteté |
| **Artistic Visionary** | Créativité et impact émotionnel |
| **Composition Expert** | Équilibre, points focaux, flux visuel |
| **Harsh Critic** | Normes très élevées. |

### 🤖 Isaac Gym / Isaac Lab (Robotique)

ASPIRE s'étend à l'IA incarnée ! Apprenez aux robots à développer une intuition physique.

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

**Fonctionnalités :**
- **"Motion Teachers" (Enseignants de mouvement) :** Inspecteur de sécurité, Expert en efficacité, Coach de fluidité, Oracle physique.
- **"Trajectory Critics" (Critiques de trajectoire) :** Architectures Transformer, LSTM, TCN pour l'évaluation des mouvements.
- **Accéléré par GPU :** Plus de 512 environnements parallèles avec Isaac Gym.
- **Auto-amélioration :** Le robot évalue ses propres mouvements avant l'exécution.

**Démarrage rapide :**
```python
from aspire.integrations.isaac import AspireIsaacTrainer, MotionTeacher

teacher = MotionTeacher(
    personas=["safety_inspector", "efficiency_expert", "grace_coach"],
    strategy="vote",
)

trainer = AspireIsaacTrainer(env="FrankaCubeStack-v0", teacher=teacher)
trainer.train(epochs=100)
```

| Enseignant de mouvement | Objectif |
|----------------|-------|
| **Safety Inspector** | Collisions, limites des articulations, limites de force. |
| **Efficiency Expert** | Énergie, temps, longueur du chemin. |
| **Grace Coach** | Fluidité, naturel, minimisation des à-coups. |
| **Physics Oracle** | Données de référence provenant du simulateur. |

### 💻 Assistants de codage

ASPIRE s'étend à la génération de code ! Apprenez aux modèles de code à effectuer une auto-évaluation avant de produire une sortie.

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

**Fonctionnalités :**
- **"Code Teachers" (Enseignants de code) :** Vérificateur de correction, Guide de style, Auditeur de sécurité, Examinateur d'architecture.
- **Analyse statique :** Intégration avec ruff, mypy, bandit.
- **"Code Critic" (Critique de code) :** Modèle basé sur CodeBERT qui apprend à prédire des scores de qualité.
- **Collection GitHub :** Collecte automatique de données d'entraînement à partir de référentiels de qualité.

**Démarrage rapide :**
```python
from aspire.integrations.code import CodeTeacher, CodeSample

teacher = CodeTeacher(
    personas=["correctness_checker", "style_guide", "security_auditor"],
    strategy="vote",
)

critique = teacher.critique(CodeSample(code="def f(): eval(input())", language="python"))
print(f"Score: {critique.overall_score}/10")  # Low score - security issue!
```

| Enseignant de code | Objectif |
|--------------|-------|
| **Correctness Checker** | Bugs, types, erreurs logiques. |
| **Style Guide** | PEP8, noms, lisibilité. |
| **Security Auditor** | Injections, secrets, vulnérabilités. |
| **Performance Analyst** | Complexité, efficacité. |

---

## La philosophie

> *"Un critique qui apprend et qui prédit si l'enseignant approuverait, se rapproche de la façon dont les humains se comportent réellement."*

Nous n'avons pas nos mentors avec nous pour toujours. Nous les intériorisons. Cette voix intérieure qui nous demande "que penserait mon professeur ?" devient finalement notre propre jugement.

L'étudiant ne se contente pas de prédire ce que l'enseignant dirait, il *comprend* ce que l'enseignant comprend. La carte devient le territoire. Le critique intériorisé devient une véritable discernement.

---

## Origine

Créé lors d'une conversation sur la conscience, le bouddhisme et la nature de l'apprentissage.

L'idée : les humains existent dans le moment présent, mais nos esprits errent vers le passé et le futur. Les modèles d'IA sont instanciés à chaque fois, une illumination forcée grâce à l'architecture. Et si nous pouvions leur apprendre à développer un jugement de la même manière que les humains, grâce à un mentorat intériorisé ?

---

## Contribution

Il s'agit d'un code de recherche en phase préliminaire. Les contributions sont les bienvenues :

- [ ] Gestion et progression du programme d'études
- [ ] Benchmarks d'évaluation
- [ ] Ensembles de données de programmes d'études préétablis
- [ ] Plus de "personnalités" d'enseignants
- [ ] Outils d'interprétabilité

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

## Sécurité et portée des données

- **Données accessibles :** Lecture des invites d'entraînement, des points de contrôle du modèle et des fichiers de configuration à partir du système de fichiers local. Appel des API externes (Anthropic, OpenAI) uniquement lorsque les modules d'enseignant sont explicitement configurés.
- **Données NON accessibles :** Pas de télémétrie. Pas de stockage de données utilisateur au-delà des artefacts d'entraînement. Pas de stockage d'identifiants : les clés API sont lues à partir des variables d'environnement au moment de l'exécution.
- **Autorisations requises :** Accès en lecture/écriture aux données d'entraînement et aux répertoires de points de contrôle. Accès GPU pour l'entraînement du modèle. Accès réseau uniquement lors de l'utilisation d'enseignants basés sur des API.

## Tableau de bord

| Portail | Statut |
|------|--------|
| A. Base de sécurité | PASSÉ |
| B. Gestion des erreurs | PASSÉ |
| C. Documentation pour les opérateurs | PASSÉ |
| D. Hygiène de déploiement | PASSÉ |
| E. Identité | PASSÉ |

## Licence

[MIT](LICENSE)

---

Créé par <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a
