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

## L'idea

**Fine-tuning tradizionale:** *"Ecco le risposte corrette. Abbinale."*

**ASPIRE:** *"Ecco una mente saggia. Impara a pensare come fa lei."*

Quando si impara da un grande mentore, non ci si limita a memorizzare le sue risposte. Si interiorizza il suo modo di vedere. La sua voce diventa parte del proprio dialogo interiore. Si inizia ad anticipare ciò che direbbe, e alla fine questa anticipazione diventa la propria capacità di discernimento.

ASPIRE offre all'IA la stessa esperienza.

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

Il **critico** impara a prevedere cosa penserebbe l'insegnante. Dopo l'addestramento, lo studente utilizza questo critico interiorizzato per auto-migliorarsi: **non è necessario alcun insegnante durante la fase di inferenza**.

---

## Guida rapida

### Installazione

```bash
git clone https://github.com/mcp-tool-shop-org/aspire-si.git
cd aspire-si
pip install -e .
```

### Imposta la tua chiave API

```bash
# Windows
set ANTHROPIC_API_KEY=your-key-here

# Linux/Mac
export ANTHROPIC_API_KEY=your-key-here
```

### Verifica l'installazione

```bash
# Check your environment (Python, CUDA, API keys)
aspire doctor
```

### Provalo

```bash
# See available teacher personas
aspire teachers

# Generate an adversarial dialogue
aspire dialogue "Explain why recursion works" --teacher socratic --turns 3

# Initialize a training config
aspire init --output my-config.yaml
```

---

## Profili di insegnanti

Insegnanti diversi producono menti diverse. Scegli con saggezza.

| Profilo | Filosofia | Produce |
|---------|------------|----------|
| 🏛️ **Socrate** | *"Quale presupposto stai facendo?"* | Ragionamento approfondito, indipendenza intellettuale |
| 🔬 **Scientifico** | *"Quali sono le tue prove?"* | Precisione tecnica, pensiero rigoroso |
| 🎨 **Creativo** | *"E se provassimo il contrario?"* | Innovazione, pensiero laterale |
| ⚔️ **Antagonista** | *"Sono in disaccordo. Difendi la tua posizione."* | Argomentazioni solide, convinzione |
| 💚 **Compassionevole** | *"Come potrebbe sentirsi qualcuno al riguardo?"* | Ragionamento etico, saggezza |

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

### 1. Dialogo antagonista

Lo studente genera una risposta. L'insegnante la mette in discussione. Un avanti e indietro, alla ricerca di debolezze, richiedendo chiarezza, spingendo più a fondo.

```
Student: "Recursion works by calling itself."

Teacher (Socratic): "But what prevents infinite regress?
                     What's the mechanism that grounds the recursion?"

Student: "The base case stops it when..."

Teacher: "You say 'stops it' — but how does the computer know
          to check the base case before recursing?"
```

### 2. Addestramento del critico

Il critico impara a prevedere il giudizio dell'insegnante, non solo il punteggio, ma anche il *ragionamento*.

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

### 4. Giudizio senza l'insegnante

Dopo l'addestramento, il critico valuta una risposta dello studente basandosi esclusivamente sui suoi stati nascosti, quindi non è necessaria alcuna chiamata API all'insegnante per giudicarla. Un ciclo di perfezionamento è a tua disposizione per essere implementato; ASPIRE fornisce il critico addestrato, non il ciclo:

```python
def generate_with_judgment(prompt, threshold=7.0, attempts=3):
    response = student.generate(prompt)
    for _ in range(attempts):
        if critic_score(student, critic, response) >= threshold:  # your wrapper around the critic
            break
        response = student.generate(prompt)  # or a revision prompt of your own
    return response
```

---

## Riferimento CLI

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

Gli errori stampano un codice, un messaggio e cosa fare, senza un traceback:

```
ASPIRE_MISSING_API_KEY  ANTHROPIC_API_KEY not found.
To fix this, set your API key: ...
```

Codici di uscita: `0` successo, `1` qualcosa che puoi correggere (una chiave mancante, una configurazione o un file di prompt errati), `2` un errore durante l'esecuzione, `130` interrotto.

---

## Monitoraggio di un'esecuzione in ScalarScope

`aspire train --geometry` (o `training.geometry_export: true` nella configurazione) scrive `geometry.json` accanto ai checkpoint: la dinamica dell'addestramento dell'esecuzione nel formato che [ScalarScope](https://github.com/mcp-tool-shop-org/scalarscope) legge. Apri due di essi uno accanto all'altro per confrontare le esecuzioni.

| Campo | Cosa contiene |
|-------|---------------|
| Traiettoria | L'ultimo livello nascosto dello studente, raggruppato sui token e sul batch, proiettato sulle sue prime due componenti principali. Velocità, curvatura con segno (angolo di svolta rispetto a pi, smorzata quando l'esecuzione si muove a malapena) e dimensione effettiva (rapporto di partecipazione in una finestra). |
| Scalari | Ogni dimensione di valutazione per cui gli insegnanti hanno dato un punteggio, da 0 a 1. |
| Autovalori | Per ogni passo, lo spettro di come le dimensioni dei punteggi variano insieme in una finestra, come frazioni. Un valore iniziale elevato significa che una direzione spiega i giudizi degli insegnanti. |
| Professori | Una freccia per ogni insegnante: la direzione nello spazio degli stati lungo la quale il suo punteggio aumenta. Un insegnante composito fornisce una freccia per ogni membro. |
| Fallimenti | Passi in cui una dimensione scende di almeno 0,1 al di sotto della sua mediana recente. |

Nessun modello o chiave API? `python examples/geometry_demo.py` simula due esecuzioni e scrive entrambi i file di esportazione, che è il modo più rapido per vedere le visualizzazioni.

Il registratore (`aspire.geometry.GeometryRecorder`) mantiene in memoria un vettore raggruppato per ogni passo. Per le esecuzioni lunghe, imposta `training.geometry_every` per calcolare la media di più batch in un unico passo.

---

## Struttura del progetto

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

## Requisiti

- Python 3.10+
- PyTorch 2.0+
- Una GPU CUDA per l'addestramento (si consiglia 16 GB+ di VRAM). I test, la demo della geometria e gli esempi di integrazione vengono eseguiti sulla CPU.
- Chiave API Anthropic (per l'insegnante Claude) o chiave API OpenAI

### Compatibilità con Windows

ASPIRE è completamente compatibile con Windows con supporto RTX 5080/Blackwell:
- `dataloader_num_workers=0`
- `XFORMERS_DISABLED=1`
- Multiprocessing corretto con `freeze_support()`

---

## Integrazioni

### 🖼️ Stable Diffusion WebUI Forge

ASPIRE si estende alla generazione di immagini! Addestra i modelli Stable Diffusion per sviluppare un giudizio estetico.

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
- **Insegnanti di visione:** Claude Vision, GPT-4V criticano le immagini generate
- **Critici di immagini:** Critici basati su CLIP e sullo spazio latente per una guida in tempo reale
- **Interfaccia utente di addestramento:** Addestra gli adattatori LoRA con anteprima in diretta e confronto prima/dopo
- **Nessuna API durante l'inferenza:** Il critico addestrato guida la generazione localmente

**Installazione:**
```bash
# Copy to your Forge extensions
cp -r integrations/forge /path/to/sd-webui-forge/extensions-builtin/sd_forge_aspire
```

| Insegnante di visione | Focus |
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
- **Insegnanti del movimento:** Ispettore di sicurezza, Esperto di efficienza, Allenatore di fluidità, Oracolo della fisica
- **Valutatori di traiettoria:** Architetture Transformer, LSTM, TCN per la valutazione del movimento
- **Accelerazione GPU:** 512+ ambienti paralleli con Isaac Gym
- **Auto-miglioramento:** Il robot valuta i propri movimenti prima dell'esecuzione

**Avvio rapido:**
```python
from integrations.isaac import AspireIsaacTrainer, MotionTeacher

teacher = MotionTeacher(
    personas=["safety_inspector", "efficiency_expert", "grace_coach"],
    strategy="vote",
)

trainer = AspireIsaacTrainer(env="FrankaCubeStack-v0", teacher=teacher)
trainer.train(epochs=100)
```

Senza Isaac Gym installato, `python -m integrations.isaac.examples.basic_training` esegue lo stesso ciclo su un piccolo ambiente di test integrato, sulla CPU.

| Insegnante del movimento | Focus |
|----------------|-------|
| **Safety Inspector** | Collisioni, limiti delle giunture, limiti di forza |
| **Efficiency Expert** | Energia, tempo, lunghezza del percorso |
| **Grace Coach** | Fluidità, naturalezza, minimizzazione degli scatti |
| **Physics Oracle** | Dati di riferimento dal simulatore |

### 💻 Assistenti per il codice

ASPIRE si estende alla generazione di codice! Insegna ai modelli di codice a effettuare un'auto-revisione prima di fornire l'output.

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
- **Insegnanti del codice:** Controllore di correttezza, Guida di stile, Revisore della sicurezza, Valutatore dell'architettura
- **Analisi statica:** Si integra con ruff, mypy, bandit
- **Valutatore del codice:** Il modello basato su CodeBERT impara a prevedere i punteggi di qualità
- **Raccolta GitHub:** Raccoglie automaticamente i dati di addestramento da repository di alta qualità

**Avvio rapido:**
```python
from integrations.code import CodeSample, CodeTeacher, Language

teacher = CodeTeacher(
    personas=["correctness_checker", "style_guide", "security_auditor"],
    strategy="vote",
)

critique = teacher.critique(CodeSample(code="def f(): eval(input())", language=Language.PYTHON))
print(critique.weaknesses)  # ['Line 1: Code injection risk (eval of user input)', ...]
```

| Insegnante del codice | Focus |
|--------------|-------|
| **Correctness Checker** | Bug, tipi, errori logici |
| **Style Guide** | PEP8, denominazione, leggibilità |
| **Security Auditor** | Iniezione, segreti, vulnerabilità |
| **Performance Analyst** | Complessità, efficienza |

---

## La filosofia

> *"Un critico esperto che prevede se l'insegnante approverebbe, si avvicina al modo in cui si comportano effettivamente gli esseri umani."*

Non portiamo con noi i nostri mentori per sempre. Li interiorizziamo. Quella voce interiore che chiede *"cosa penserebbe il mio professore?"* alla fine diventa il nostro giudizio.

Lo studente non si limita a prevedere cosa direbbe l'insegnante, ma *comprende* ciò che l'insegnante comprende. La mappa diventa il territorio. Il critico interiorizzato diventa un vero discernimento.

---

## Origine

Creato durante una conversazione sulla coscienza, il buddismo e la natura dell'apprendimento.

L'intuizione: gli esseri umani esistono nel momento presente, ma le nostre menti vagano nel passato e nel futuro. I modelli di intelligenza artificiale vengono istanziati ogni volta, con un'illuminazione forzata attraverso l'architettura. E se potessimo insegnare loro a sviluppare il giudizio nello stesso modo in cui lo fanno gli esseri umani, attraverso un tutoraggio interiorizzato?

---

## Contributi

Questo è un codice di ricerca in fase iniziale. I contributi sono benvenuti:

- [ ] Gestione e progressione del curriculum
- [ ] Benchmark di valutazione
- [ ] Dataset di curriculum predefiniti
- [ ] Più personalità di insegnanti
- [ ] Strumenti di interpretabilità

---

## Citazione

```bibtex
@software{aspire2026,
  author = {mcp-tool-shop},
  title = {ASPIRE: Adversarial Student-Professor Internalized Reasoning Engine},
  year = {2026},
  url = {https://github.com/mcp-tool-shop-org/aspire-si}
}
```

---

## Sicurezza e ambito dei dati

- **Dati a cui si accede:** Legge i prompt di addestramento, i checkpoint del modello e i file di configurazione dal file system locale. Chiama API esterne (Anthropic, OpenAI) solo quando i moduli dell'insegnante sono configurati esplicitamente.
- **Dati a cui NON si accede:** Nessun telemetria. Nessun archivio di dati utente oltre agli artefatti di addestramento. Nessun archivio di credenziali: le chiavi API vengono lette dalle variabili d'ambiente in fase di esecuzione.
- **Autorizzazioni richieste:** Accesso in lettura/scrittura ai dati di addestramento e alle directory dei checkpoint. Accesso alla GPU per l'addestramento del modello. Accesso alla rete solo quando si utilizzano insegnanti basati su API.

## Scheda di valutazione

| Gateway | Stato |
|------|--------|
| A. Baseline di sicurezza | PASS |
| B. Gestione degli errori | PASS |
| C. Documentazione per l'operatore | PASS |
| D. Igiene di rilascio | PASS |
| E. Identità | PASS |

## Licenza

[MIT](LICENSE)

---

Creato da <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a>
