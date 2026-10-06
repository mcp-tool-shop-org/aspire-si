<p align="center">
  <a href="README.ja.md">日本語</a> | <a href="README.zh.md">中文</a> | <a href="README.es.md">Español</a> | <a href="README.fr.md">Français</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.md">English</a> | <a href="README.pt-BR.md">Português (BR)</a>
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

## L'idea

**Ottimizzazione tradizionale:** *"Ecco le risposte corrette. Abbinale."*

**ASPIRE:** *"Ecco una mente saggia. Impara a pensare come lei."*

Quando si impara da un grande mentore, non ci si limita a memorizzare le sue risposte. Si interiorizza il suo modo di vedere. La sua voce diventa parte del proprio dialogo interiore. Si inizia ad anticipare ciò che direbbe, e alla fine quell'anticipazione diventa la propria capacità di giudizio.

ASPIRE offre all'intelligenza artificiale la stessa esperienza.

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

Il **critico** impara a prevedere ciò che l'insegnante penserebbe. Dopo l'addestramento, lo studente utilizza questo critico interiorizzato per auto-perfezionarsi, **senza la necessità di un insegnante durante l'inferenza**.

---

## Guida rapida

### Installazione

```bash
git clone https://github.com/mcp-tool-shop-org/aspire-ai.git
cd aspire-ai
pip install -e .
```

### Imposta la tua chiave API

```bash
# Windows
set ANTHROPIC_API_KEY=your-key-here

# Linux/Mac
export ANTHROPIC_API_KEY=your-key-here
```

### Verifica la configurazione

```bash
# Check your environment (Python, CUDA, API keys)
aspire doctor
```

### Prova

```bash
# See available teacher personas
aspire teachers

# Generate an adversarial dialogue
aspire dialogue "Explain why recursion works" --teacher socratic --turns 3

# Initialize a training config
aspire init --output my-config.yaml
```

---

## Profili di insegnante

Insegnanti diversi producono menti diverse. Scegli con saggezza.

| Profilo | Filosofia | Risultati |
|---------|------------|----------|
| 🏛️ **Socrate** | *"Quale assunzione stai facendo?"* | Ragionamento profondo, indipendenza intellettuale |
| 🔬 **Scientifico** | *"Quali sono le tue prove?"* | Precisione tecnica, pensiero rigoroso |
| 🎨 **Creativo** | *"E se provassimo il contrario?"* | Innovazione, pensiero laterale |
| ⚔️ **Avversario** | *"Non sono d'accordo. Difendi la tua posizione."* | Argomentazioni solide, convinzione |
| 💚 **Compassionevole** | *"Come potrebbe sentirsi qualcuno riguardo a questo?"* | Ragionamento etico, saggezza |

### Insegnanti compositi

Combina più insegnanti per un apprendimento più ricco:

```python
from aspire.teachers import CompositeTeacher, SocraticTeacher, ScientificTeacher

# A committee of mentors
teacher = CompositeTeacher(
    teachers=[SocraticTeacher(), ScientificTeacher()],
    strategy="vote"  # or "rotate", "debate"
)
```

---

## Come funziona

### 1. Dialogo avversariale

Lo studente genera una risposta. L'insegnante la mette in discussione. Un botta e risposta, che esplora le debolezze, richiede chiarezza e approfondisce l'analisi.

```
Student: "Recursion works by calling itself."

Teacher (Socratic): "But what prevents infinite regress?
                     What's the mechanism that grounds the recursion?"

Student: "The base case stops it when..."

Teacher: "You say 'stops it' — but how does the computer know
          to check the base case before recursing?"
```

### 2. Addestramento del critico

Il critico impara a prevedere il giudizio dell'insegnante, non solo il punteggio, ma anche la *ragione*.

```python
critic_loss = predict_teacher_judgment(
    score=True,      # "This deserves a 7/10"
    reasoning=True,  # "Because the explanation lacks depth on X"
)
```

### 3. Addestramento dello studente

Lo studente impara dal giudizio interiorizzato del critico, orientandosi verso ciò che l'insegnante approverebbe.

```python
student_loss = (
    reward_from_critic +      # Higher score = better
    contrastive_to_teacher +  # Pull toward teacher's improved version
    trajectory_improvement    # Get better across dialogue turns
)
```

### 4. Magia dell'inferenza

Dopo l'addestramento, lo studente si auto-perfeziona utilizzando il critico interiorizzato. **Non sono necessarie chiamate API dell'insegnante.**

```python
def generate_with_judgment(prompt):
    response = student.generate(prompt)

    while critic.score(response) < threshold:
        response = student.refine(response, critic.feedback)

    return response  # Self-improved through internalized judgment
```

---

## Riferimento CLI

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

## Struttura del progetto

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

## Requisiti

- Python 3.10+
- PyTorch 2.0+
- GPU CUDA (si consiglia 16GB+ di VRAM)
- Chiave API Anthropic (per l'insegnante Claude) o chiave API OpenAI

### Compatibilità con Windows

ASPIRE è completamente compatibile con Windows e supporta RTX 5080/Blackwell:
- `dataloader_num_workers=0`
- `XFORMERS_DISABLED=1`
- Multiprocessing corretto con `freeze_support()`

---

## Integrazioni

### 🖼️ Stable Diffusion WebUI Forge

ASPIRE si estende alla generazione di immagini! Addestra modelli Stable Diffusion per sviluppare il senso estetico.

```
integrations/forge/
├── scripts/
│   ├── aspire_generate.py   # Critic-guided generation
│   └── aspire_train.py      # Training interface
├── vision_teacher.py        # Claude Vision / GPT-4V teachers
├── image_critic.py          # CLIP and latent-space critics
└── README.md
```

**Funzionalità:**
- **Insegnanti visivi**: Claude Vision, GPT-4V criticano le immagini generate
- **Critici di immagini**: Critici basati su CLIP e nello spazio latente per una guida in tempo reale
- **Interfaccia di addestramento**: Addestra gli adattatori LoRA con anteprima in diretta e confronto prima/dopo
- **Nessuna API durante l'inferenza**: Il critico addestrato guida la generazione localmente

**Installazione:**
```bash
# Copy to your Forge extensions
cp -r integrations/forge /path/to/sd-webui-forge/extensions-builtin/sd_forge_aspire
```

| Insegnante visivo | Focus |
|----------------|-------|
| **Balanced Critic** | Valutazione tecnica e artistica equa |
| **Technical Analyst** | Qualità, artefatti, nitidezza |
| **Artistic Visionary** | Creatività e impatto emotivo |
| **Composition Expert** | Equilibrio, punti focali, flusso visivo |
| **Harsh Critic** | Standard molto elevati |

### 🤖 Isaac Gym / Isaac Lab (Robotica)

ASPIRE si estende all'intelligenza artificiale incarnata! Insegna ai robot a sviluppare un'intuizione fisica.

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

**Funzionalità:**
- **Motion Teachers (Insegnanti di movimento):** Safety Inspector (Ispettore di sicurezza), Efficiency Expert (Esperto di efficienza), Grace Coach (Allenatore di eleganza), Physics Oracle (Oracolo della fisica)
- **Trajectory Critics (Critici di traiettoria):** Architetture Transformer, LSTM, TCN per la valutazione del movimento
- **Accelerazione GPU:** 512+ ambienti paralleli con Isaac Gym
- **Auto-perfezionamento:** Il robot valuta i propri movimenti prima dell'esecuzione

**Guida rapida:**
```python
from aspire.integrations.isaac import AspireIsaacTrainer, MotionTeacher

teacher = MotionTeacher(
    personas=["safety_inspector", "efficiency_expert", "grace_coach"],
    strategy="vote",
)

trainer = AspireIsaacTrainer(env="FrankaCubeStack-v0", teacher=teacher)
trainer.train(epochs=100)
```

| Motion Teacher (Insegnante di movimento) | Focus |
|----------------|-------|
| **Safety Inspector** | Collisioni, limiti delle articolazioni, limiti di forza |
| **Efficiency Expert** | Energia, tempo, lunghezza del percorso |
| **Grace Coach** | Fluidità, naturalezza, minimizzazione delle accelerazioni |
| **Physics Oracle** | Dati di riferimento dal simulatore |

### 💻 Code Assistants (Assistenti di programmazione)

ASPIRE si estende alla generazione di codice! Insegna ai modelli di codice a effettuare un'auto-revisione prima di produrre l'output.

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

**Funzionalità:**
- **Code Teachers (Insegnanti di programmazione):** Correctness Checker (Verificatore di correttezza), Style Guide (Guida di stile), Security Auditor (Revisore di sicurezza), Architecture Reviewer (Esaminatore dell'architettura)
- **Analisi statica:** Si integra con ruff, mypy, bandit
- **Code Critic (Critico di codice):** Modello basato su CodeBERT che impara a prevedere punteggi di qualità
- **GitHub Collection (Raccolta da GitHub):** Raccoglie automaticamente dati di addestramento da repository di alta qualità

**Guida rapida:**
```python
from aspire.integrations.code import CodeTeacher, CodeSample

teacher = CodeTeacher(
    personas=["correctness_checker", "style_guide", "security_auditor"],
    strategy="vote",
)

critique = teacher.critique(CodeSample(code="def f(): eval(input())", language="python"))
print(f"Score: {critique.overall_score}/10")  # Low score - security issue!
```

| Code Teacher (Insegnante di programmazione) | Focus |
|--------------|-------|
| **Correctness Checker** | Bug, tipi, errori logici |
| **Style Guide** | PEP8, nomenclatura, leggibilità |
| **Security Auditor** | Injection (iniezione), segreti, vulnerabilità |
| **Performance Analyst** | Complessità, efficienza |

---

## La filosofia

> *"Un critico addestrato che prevede se l'insegnante approverebbe, e questo si avvicina al modo in cui gli esseri umani si comportano effettivamente."*

Non portiamo con noi i nostri mentori per sempre. Li interiorizziamo. Quella voce interiore che chiede "cosa penserebbe il mio professore?" alla fine diventa il nostro stesso giudizio.

Lo studente non si limita a prevedere ciò che l'insegnante direbbe, ma *comprende* ciò che l'insegnante comprende. La mappa diventa il territorio. Il critico interiorizzato diventa una vera e propria capacità di discernimento.

---

## Origine

Sviluppato durante una conversazione sulla coscienza, il buddismo e la natura dell'apprendimento.

L'intuizione: gli esseri umani esistono nel momento presente, ma le nostre menti vagano nel passato e nel futuro. I modelli di intelligenza artificiale vengono istanziati ogni volta, una sorta di illuminazione forzata attraverso l'architettura. E se potessimo insegnare loro a sviluppare il giudizio nello stesso modo in cui fanno gli esseri umani, attraverso una mentorship interiorizzata?

---

## Contributi

Questo è codice di ricerca in fase iniziale. I contributi sono benvenuti:

- [ ] Gestione e progressione del curriculum
- [ ] Benchmark di valutazione
- [ ] Set di dati di curriculum predefiniti
- [ ] Altre personalità di insegnante
- [ ] Strumenti di interpretabilità

---

## Citazione

```bibtex
@software{aspire2026,
  author = {mcp-tool-shop},
  title = {ASPIRE: Adversarial Student-Professor Internalized Reasoning Engine},
  year = {2026},
  url = {https://github.com/mcp-tool-shop-org/aspire-ai}
}
```

---

## Sicurezza e ambito dei dati

- **Dati accessibili:** Legge i prompt di addestramento, i checkpoint del modello e i file di configurazione dal file system locale. Chiama API esterne (Anthropic, OpenAI) solo quando i moduli dell'insegnante sono configurati esplicitamente.
- **Dati NON accessibili:** Nessuna telemetria. Nessun archivio di dati utente al di là degli artefatti di addestramento. Nessun archivio di credenziali: le chiavi API vengono lette dalle variabili d'ambiente durante l'esecuzione.
- **Autorizzazioni richieste:** Accesso in lettura/scrittura ai dati di addestramento e alle directory dei checkpoint. Accesso alla GPU per l'addestramento del modello. Accesso alla rete solo quando si utilizzano insegnanti basati su API.

## Scorecard (Scheda di valutazione)

| Gate (Porta di controllo) | Status (Stato) |
|------|--------|
| A. Security Baseline (Base di sicurezza) | PASS (SUPERATO) |
| B. Error Handling (Gestione degli errori) | PASS (SUPERATO) |
| C. Operator Docs (Documentazione per gli operatori) | PASS (SUPERATO) |
| D. Shipping Hygiene (Igiene del rilascio) | PASS (SUPERATO) |
| E. Identity (Identità) | PASS (SUPERATO) |

## License (Licenza)

[MIT](LICENSE)

---

Creato da <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a
