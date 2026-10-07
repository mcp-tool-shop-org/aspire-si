<p align="center">
  <a href="README.ja.md">日本語</a> | <a href="README.zh.md">中文</a> | <a href="README.es.md">Español</a> | <a href="README.md">English</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.it.md">Italiano</a> | <a href="README.pt-BR.md">Português (BR)</a>
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

## L'idée

**Affinement traditionnel :** *"Voici les bonnes réponses. Associez-les."*

**ASPIRE :** *"Voici un esprit sage. Apprenez à penser comme lui."*

Lorsque vous apprenez auprès d'un excellent mentor, vous ne vous contentez pas de mémoriser ses réponses. Vous intériorisez sa façon de voir les choses. Sa voix devient une partie de votre dialogue intérieur. Vous commencez à anticiper ce qu'il dirait, et finalement, cette anticipation devient votre propre discernement.

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

Le **critique** apprend à prédire ce que l'enseignant penserait. Après l'entraînement, l'élève utilise ce critique intériorisé pour s'améliorer lui-même — **aucun enseignant n'est nécessaire au moment de l'inférence**.

---

## Démarrage rapide

### Installation

```bash
git clone https://github.com/mcp-tool-shop-org/aspire-si.git
cd aspire-si
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

## Personnalités d'enseignants

Différents enseignants produisent différents esprits. Choisissez judicieusement.

| Personnalité | Philosophie | Produit |
|---------|------------|----------|
| 🏛️ **Socratique** | *"Quelle hypothèse faites-vous ?"* | Raisonnement approfondi, indépendance intellectuelle |
| 🔬 **Scientifique** | *"Quelles sont vos preuves ?"* | Précision technique, pensée rigoureuse |
| 🎨 **Créatif** | *"Et si nous essayions l'inverse ?"* | Innovation, pensée latérale |
| ⚔️ **Adversarial** | *"Je ne suis pas d'accord. Défendez votre position."* | Arguments solides, conviction |
| 💚 **Compatissant** | *"Comment quelqu'un pourrait-il se sentir à ce sujet ?"* | Raisonnement éthique, sagesse |

### Enseignants composites

Combinez plusieurs enseignants pour un apprentissage plus riche :

```python
from aspire.teachers import CompositeTeacher, SocraticTeacher, ScientificTeacher

# A committee of mentors
teacher = CompositeTeacher(
    teachers=[SocraticTeacher(), ScientificTeacher()],
    strategy="vote"  # or "rotate", "debate"
)
```

---

## Comment ça marche

### 1. Dialogue adversarial

L'élève génère une réponse. L'enseignant la remet en question. De va-et-vient, sondant les faiblesses, exigeant de la clarté, allant plus loin.

```
Student: "Recursion works by calling itself."

Teacher (Socratic): "But what prevents infinite regress?
                     What's the mechanism that grounds the recursion?"

Student: "The base case stops it when..."

Teacher: "You say 'stops it' — but how does the computer know
          to check the base case before recursing?"
```

### 2. Entraînement du critique

The critic learns to predict the teacher's judgment of a response: its 0-10 score. (It has a
head for the teacher's reasoning too, which the trainer does not train yet.)

```python
# The critic reads the student's hidden states over the prompt and the response
# the teacher judged, and learns to predict the teacher's 0-10 score.
critic_loss = mse(critic(student_hidden_states(prompt, response)), teacher_score)
```

### 3. Entraînement de l'élève

The critic is a head on the student's hidden states, and its loss flows back into the
student's LoRA adapter. So today the student learns **representations that help the critic
predict the teacher's score**. It is not yet trained toward better answers: the reward,
contrastive and trajectory terms in `aspire.losses` exist, but the trainer does not feed
them, so they move nothing. Training the student on the critic's judgment is planned for
1.3.0 ([#11](https://github.com/mcp-tool-shop-org/aspire-si/issues/11)).

### 4. Jugement sans l'enseignant

Après l'entraînement, le critique évalue une réponse de l'élève à partir de ses états cachés, de sorte qu'aucun appel à l'API de l'enseignant n'est nécessaire pour la juger. Une boucle d'amélioration vous appartient pour l'écrire autour de cela ; ASPIRE fournit le critique entraîné, pas la boucle :

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

## Référence CLI

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

# Evaluate checkpoint
aspire evaluate outputs/checkpoint-3 \
    --prompts data/eval.json
```

Les erreurs affichent un code, un message et ce qu'il faut faire, sans trace de pile :

```
ASPIRE_MISSING_API_KEY  ANTHROPIC_API_KEY not found.
To fix this, set your API key: ...
```

Codes de sortie : `0` succès, `1` quelque chose que vous pouvez corriger (une clé manquante, une mauvaise configuration ou un fichier de prompts), `2` une erreur lors de l'exécution, `130` interruption.

---

## Surveillance d'une exécution dans ScalarScope

`aspire train --geometry` (ou `training.geometry_export: true` dans la configuration) écrit `geometry.json` à côté des points de contrôle : la dynamique d'entraînement de l'exécution dans le format que [ScalarScope](https://github.com/mcp-tool-shop-org/scalarscope) lit. Ouvrez-en deux côte à côte pour comparer les exécutions.

| Champ | Ce qu'il contient |
|-------|---------------|
| Trajectoire | La dernière couche cachée de l'élève, mise en pool sur les tokens et le lot, projetée sur ses deux premiers composants principaux. Vitesse, courbure signée (angle de rotation sur pi, amortie lorsque l'exécution bouge à peine) et dimension effective (taux de participation dans une fenêtre). |
| Scalaires | Chaque dimension d'évaluation sur laquelle les enseignants ont noté, de 0 à 1. |
| Valeurs propres | À chaque étape, le spectre de la façon dont les scores de dimension varient ensemble dans une fenêtre, sous forme de fractions. Une grande première valeur signifie qu'une direction explique les jugements des enseignants. |
| Professeurs | Une flèche par enseignant : la direction dans l'espace d'état le long de laquelle son score augmente. Un enseignant composite donne une flèche par membre. |
| Échecs | Étapes où une dimension diminue d'au moins 0,1 par rapport à sa médiane récente. |

Pas de modèle ou de clé API ? `python examples/geometry_demo.py` simule deux exécutions et écrit les deux fichiers d'exportation, ce qui est le moyen le plus rapide de voir les visualisations.

L'enregistreur (`aspire.geometry.GeometryRecorder`) conserve un vecteur mis en pool par étape en mémoire. Pour les longues exécutions, définissez `training.geometry_every` pour moyenner plusieurs lots en une seule étape.

---

## Structure du projet

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

## Exigences

- Python 3.10+
- PyTorch 2.0+
- Une GPU CUDA pour l'entraînement (16 Go de VRAM recommandés). Les tests, la démo de géométrie et les exemples d'intégration fonctionnent sur CPU.
- Clé API Anthropic (pour l'enseignant Claude) ou clé API OpenAI

### Compatibilité Windows

ASPIRE est entièrement compatible avec Windows avec le support RTX 5080/Blackwell :
- `dataloader_num_workers=0`
- `XFORMERS_DISABLED=1`
- Multiprocessing approprié avec `freeze_support()`

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
- **Enseignants de vision :** Claude Vision, GPT-4V critiquent vos images générées
- **Critiques d'images :** critiques basées sur CLIP et dans l'espace latent pour un guidage en temps réel
- **Interface d'entraînement :** entraînez des adaptateurs LoRA avec aperçu en direct et comparaison avant/après
- **Pas d'API à l'inférence :** le critique entraîné guide la génération localement

**Installation :**
```bash
# Copy to your Forge extensions
cp -r integrations/forge /path/to/sd-webui-forge/extensions-builtin/sd_forge_aspire
```

| Enseignant de vision | Focus |
|----------------|-------|
| **Balanced Critic** | Évaluation technique et artistique équitable |
| **Technical Analyst** | Qualité, artefacts, netteté |
| **Artistic Visionary** | Créativité et impact émotionnel |
| **Composition Expert** | Équilibre, points focaux, flux visuel |
| **Harsh Critic** | Très hautes exigences |

### 🤖 Isaac Gym / Isaac Lab (Robotique)

ASPIRE s’étend à l’IA incarnée ! Apprenez aux robots à développer une intuition physique.

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

**Fonctionnalités :**
- **Tuteurs de mouvement :** Inspecteur de sécurité, expert en efficacité, entraîneur en fluidité, oracle de la physique
- **Critiques de trajectoire :** Architectures Transformer, LSTM, TCN pour l’évaluation du mouvement
- **Accélération GPU :** 512 + environnements parallèles avec Isaac Gym
- **Auto-amélioration :** Le robot évalue ses propres mouvements avant l’exécution

**Démarrage rapide :**
```python
from integrations.isaac import AspireIsaacTrainer, MotionTeacher

teacher = MotionTeacher(
    personas=["safety_inspector", "efficiency_expert", "grace_coach"],
    strategy="vote",
)

trainer = AspireIsaacTrainer(env="FrankaCubeStack-v0", teacher=teacher)
trainer.train(epochs=100)
```

Sans Isaac Gym installé, `python -m integrations.isaac.examples.basic_training` exécute la même boucle sur un petit environnement de substitution intégré, sur CPU.

| Tuteur de mouvement | Focus |
|----------------|-------|
| **Safety Inspector** | Collisions, limites d’articulation, limites de force |
| **Efficiency Expert** | Énergie, temps, longueur du trajet |
| **Grace Coach** | Fluidité, naturel, minimisation des à-coups |
| **Physics Oracle** | Vérité terrain provenant du simulateur |

### 💻 Assistants de codage

ASPIRE s’étend à la génération de code ! Apprenez aux modèles de code à s’auto-évaluer avant de produire des résultats.

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

**Fonctionnalités :**
- **Tuteurs de code :** Vérificateur de correction, guide de style, auditeur de sécurité, évaluateur d’architecture
- **Analyse statique :** S’intègre à ruff, mypy, bandit
- **Critique de code :** Le modèle basé sur CodeBERT apprend à prédire les scores de qualité
- **Collection GitHub :** Collecte automatiquement les données d’entraînement à partir de référentiels de qualité

**Démarrage rapide :**
```python
from integrations.code import CodeSample, CodeTeacher, Language

teacher = CodeTeacher(
    personas=["correctness_checker", "style_guide", "security_auditor"],
    strategy="vote",
)

critique = teacher.critique(CodeSample(code="def f(): eval(input())", language=Language.PYTHON))
print(critique.weaknesses)  # ['Line 1: Code injection risk (eval of user input)', ...]
```

| Tuteur de code | Focus |
|--------------|-------|
| **Correctness Checker** | Bogues, types, erreurs logiques |
| **Style Guide** | PEP8, nommage, lisibilité |
| **Security Auditor** | Injection, secrets, vulnérabilités |
| **Performance Analyst** | Complexité, efficacité |

---

## La philosophie

> *"Un critique averti qui prédit si le tuteur approuverait se rapproche le plus de la façon dont les humains se comportent réellement."*

Nous n’emmenons pas nos mentors avec nous pour toujours. Nous les intégrons. Cette voix intérieure qui demande : « que penserait mon professeur ? » devient finalement notre propre jugement.

L’élève ne se contente pas de prédire ce que le tuteur dirait ; il *comprend* ce que le tuteur comprend. La carte devient le territoire. Le critique internalisé devient un véritable discernement.

---

## Origine

Créé lors d’une conversation sur la conscience, le bouddhisme et la nature de l’apprentissage.

L’idée : les humains existent dans le moment présent, mais leur esprit vagabonde dans le passé et le futur. Les modèles d’IA sont instanciés à chaque fois — une sorte d’illumination forcée grâce à l’architecture. Et si nous pouvions leur apprendre à développer un jugement de la même manière que les humains, grâce à un mentorat internalisé ?

---

## Contribution

Il s’agit d’un code de recherche en phase initiale. Les contributions sont les bienvenues :

- [ ] Gestion et progression du programme
- [ ] Évaluations comparatives
- [ ] Ensembles de données de programme préconstitués
- [ ] Plus de personnalités de tuteur
- [ ] Outils d’interprétabilité

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

## Sécurité et portée des données

- **Données consultées :** Lit les invites d’entraînement, les points de contrôle du modèle et les fichiers de configuration à partir du système de fichiers local. Appelle les API externes (Anthropic, OpenAI) uniquement lorsque les modules de tuteur sont explicitement configurés.
- **Données NON consultées :** Pas de télémétrie. Pas de stockage des données utilisateur au-delà des artefacts d’entraînement. Pas de stockage des informations d’identification : les clés d’API sont lues à partir des variables d’environnement au moment de l’exécution.
- **Autorisations requises :** Accès en lecture/écriture aux données d’entraînement et aux répertoires des points de contrôle. Accès GPU pour l’entraînement du modèle. Accès réseau uniquement lors de l’utilisation de tuteurs basés sur l’API.

## Tableau de bord

| Porte | État |
|------|--------|
| A. Base de référence de sécurité | PASSÉ |
| B. Gestion des erreurs | PASSÉ |
| C. Documentation pour les opérateurs | PASSÉ |
| D. Hygiène de déploiement | PASSÉ |
| E. Identité | PASSÉ |

## Licence

[MIT](LICENSE)

---

Créé par <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a>
