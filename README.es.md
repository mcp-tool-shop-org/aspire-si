<p align="center">
  <a href="README.ja.md">日本語</a> | <a href="README.zh.md">中文</a> | <a href="README.md">English</a> | <a href="README.fr.md">Français</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.it.md">Italiano</a> | <a href="README.pt-BR.md">Português (BR)</a>
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

## La idea

**Ajuste fino tradicional:** *"Aquí están las respuestas correctas. Compáralas."*

**ASPIRE:** *"Aquí hay una mente sabia. Aprende a pensar como ella."*

Cuando aprendes de un gran mentor, no solo memorizas sus respuestas. Interiorizas su forma de ver. Su voz se convierte en parte de tu diálogo interno. Empiezas a anticipar lo que diría, y eventualmente esa anticipación se convierte en tu propio discernimiento.

ASPIRE le brinda a la IA la misma experiencia.

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

El **crítico** aprende a predecir lo que pensaría el maestro. Después del entrenamiento, el estudiante utiliza este crítico internalizado para auto-mejorarse; **no se necesita un maestro en el momento de la inferencia**.

---

## Comenzar rápidamente

### Instalación

```bash
git clone https://github.com/mcp-tool-shop-org/aspire-si.git
cd aspire-si
pip install -e .
```

### Establece tu clave de API

```bash
# Windows
set ANTHROPIC_API_KEY=your-key-here

# Linux/Mac
export ANTHROPIC_API_KEY=your-key-here
```

### Verifica la configuración

```bash
# Check your environment (Python, CUDA, API keys)
aspire doctor
```

### Pruébalo

```bash
# See available teacher personas
aspire teachers

# Generate an adversarial dialogue
aspire dialogue "Explain why recursion works" --teacher socratic --turns 3

# Initialize a training config
aspire init --output my-config.yaml
```

---

## Personalidades de maestro

Diferentes maestros producen diferentes mentes. Elige sabiamente.

| Personalidad | Filosofía | Produce |
|---------|------------|----------|
| 🏛️ **Socrático** | *"¿Qué suposición estás haciendo?"* | Razonamiento profundo, independencia intelectual |
| 🔬 **Científico** | *"¿Cuál es tu evidencia?"* | Precisión técnica, pensamiento riguroso |
| 🎨 **Creativo** | *"¿Qué pasaría si probamos lo contrario?"* | Innovación, pensamiento lateral |
| ⚔️ **Adversarial** | *"No estoy de acuerdo. Defiende tu posición."* | Argumentos sólidos, convicción |
| 💚 **Compasivo** | *"¿Cómo podría sentirse alguien al respecto?"* | Razonamiento ético, sabiduría |

### Maestros compuestos

Combina varios maestros para un aprendizaje más enriquecedor:

```python
from aspire.teachers import CompositeTeacher, SocraticTeacher, ScientificTeacher

# A committee of mentors
teacher = CompositeTeacher(
    teachers=[SocraticTeacher(), ScientificTeacher()],
    strategy="vote"  # or "rotate", "debate"
)
```

---

## Cómo funciona

### 1. Diálogo adversarial

El estudiante genera una respuesta. El maestro la desafía. De un lado a otro, explorando debilidades, exigiendo claridad, profundizando.

```
Student: "Recursion works by calling itself."

Teacher (Socratic): "But what prevents infinite regress?
                     What's the mechanism that grounds the recursion?"

Student: "The base case stops it when..."

Teacher: "You say 'stops it' — but how does the computer know
          to check the base case before recursing?"
```

### 2. Entrenamiento del crítico

The critic learns to predict the teacher's judgment of a response: its 0-10 score. (It has a
head for the teacher's reasoning too, which the trainer does not train yet.)

```python
# The critic reads the student's hidden states over the prompt and the response
# the teacher judged, and learns to predict the teacher's 0-10 score.
critic_loss = mse(critic(student_hidden_states(prompt, response)), teacher_score)
```

### 3. Entrenamiento del estudiante

The critic is a head on the student's hidden states, and its loss flows back into the
student's LoRA adapter. So today the student learns **representations that help the critic
predict the teacher's score**. It is not yet trained toward better answers: the reward,
contrastive and trajectory terms in `aspire.losses` exist, but the trainer does not feed
them, so they move nothing. Training the student on the critic's judgment is planned for
1.3.0 ([#11](https://github.com/mcp-tool-shop-org/aspire-si/issues/11)).

### 4. Juicio sin el maestro

Después del entrenamiento, el crítico califica una respuesta del estudiante a partir de sus estados ocultos, por lo que no se necesita ninguna llamada a la API del maestro para juzgarla. El bucle de refinamiento es tuyo para implementarlo; ASPIRE proporciona el crítico entrenado, no el bucle:

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

## Referencia de la CLI

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

Los errores imprimen un código, un mensaje y qué hacer, sin un rastreo:

```
ASPIRE_MISSING_API_KEY  ANTHROPIC_API_KEY not found.
To fix this, set your API key: ...
```

Códigos de salida: `0` éxito, `1` algo que puedes solucionar (una clave faltante, una configuración o archivo de indicaciones incorrectos), `2` un fallo durante la ejecución, `130` interrumpido.

---

## Observando una ejecución en ScalarScope

`aspire train --geometry` (o `training.geometry_export: true` en la configuración) escribe `geometry.json` junto a los puntos de control: la dinámica de entrenamiento de la ejecución en el formato que [ScalarScope](https://github.com/mcp-tool-shop-org/scalarscope) lee. Abre dos de ellos uno al lado del otro para comparar ejecuciones.

| Campo | Qué contiene |
|-------|---------------|
| Trayectoria | La última capa oculta del estudiante, agrupada sobre los tokens y el lote, proyectada en sus dos primeros componentes principales. Velocidad, curvatura con signo (ángulo de giro sobre pi, amortiguada cuando la ejecución apenas se mueve) y dimensión efectiva (proporción de participación en una ventana). |
| Escalares | Cada dimensión de evaluación que los maestros calificaron, de 0 a 1. |
| Autovalores | Por paso, el espectro de cómo varían juntas las puntuaciones de las dimensiones en una ventana, como fracciones. Un valor grande en el primer lugar significa que una dirección explica los juicios de los maestros. |
| Profesores | Una flecha por maestro: la dirección en el espacio de estados a lo largo de la cual su puntuación aumenta. Un maestro compuesto proporciona una por miembro. |
| Fallos | Pasos en los que una dimensión cae al menos 0,1 por debajo de su mediana reciente. |

¿No tienes un modelo o una clave de API? `python examples/geometry_demo.py` simula dos ejecuciones y escribe ambas exportaciones, que es la forma más rápida de ver las vistas.

El registrador (`aspire.geometry.GeometryRecorder`) mantiene un vector agrupado por paso en la memoria. Para ejecuciones largas, establece `training.geometry_every` para promediar varios lotes en un solo paso.

---

## Estructura del proyecto

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

## Requisitos

- Python 3.10+
- PyTorch 2.0+
- Una GPU CUDA para el entrenamiento (se recomienda 16 GB+ de VRAM). Las pruebas, la demostración de geometría y los ejemplos de integración se ejecutan en la CPU.
- Clave de API de Anthropic (para el maestro Claude) o clave de API de OpenAI

### Compatibilidad con Windows

ASPIRE es totalmente compatible con Windows con soporte RTX 5080/Blackwell:
- `dataloader_num_workers=0`
- `XFORMERS_DISABLED=1`
- Multiprocesamiento adecuado con `freeze_support()`

---

## Integraciones

### 🖼️ Stable Diffusion WebUI Forge

¡ASPIRE se extiende a la generación de imágenes! Entrena modelos de Stable Diffusion para desarrollar un juicio estético.

```
integrations/forge/
├── scripts/
│   ├── aspire_generate.py   # Critic-guided generation
│   └── aspire_train.py      # Training interface
├── vision_teacher.py        # Claude Vision / GPT-4V teachers
├── image_critic.py          # CLIP and latent-space critics
└── README.md
```

**Características:**
- **Maestros de visión**: Claude Vision, GPT-4V critican tus imágenes generadas
- **Críticos de imagen**: Críticos basados en CLIP y en el espacio latente para una guía en tiempo real
- **UI de entrenamiento**: Entrena adaptadores LoRA con vista previa en vivo y comparación antes/después
- **Sin API en la inferencia**: El crítico entrenado guía la generación localmente

**Instalación:**
```bash
# Copy to your Forge extensions
cp -r integrations/forge /path/to/sd-webui-forge/extensions-builtin/sd_forge_aspire
```

| Maestro de visión | Enfoque |
|----------------|-------|
| **Balanced Critic** | Evaluación técnica y artística justa |
| **Technical Analyst** | Calidad, artefactos, nitidez |
| **Artistic Visionary** | Creatividad e impacto emocional |
| **Composition Expert** | Equilibrio, puntos focales, flujo visual |
| **Harsh Critic** | Estándares muy altos |

### 🤖 Isaac Gym / Isaac Lab (Robótica)

ASPIRE se extiende a la IA incorporada. Enseña a los robots a desarrollar intuición física.

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

**Características:**
- **Entrenadores de movimiento:** Inspector de seguridad, experto en eficiencia, entrenador de elegancia, oráculo de la física
- **Críticos de trayectoria:** Arquitecturas Transformer, LSTM y TCN para la evaluación del movimiento
- **Aceleración por GPU:** 512 o más entornos paralelos con Isaac Gym
- **Autorrefinamiento:** El robot evalúa sus propios movimientos antes de la ejecución

**Primeros pasos:**
```python
from integrations.isaac import AspireIsaacTrainer, MotionTeacher

teacher = MotionTeacher(
    personas=["safety_inspector", "efficiency_expert", "grace_coach"],
    strategy="vote",
)

trainer = AspireIsaacTrainer(env="FrankaCubeStack-v0", teacher=teacher)
trainer.train(epochs=100)
```

Sin Isaac Gym instalado, `python -m integrations.isaac.examples.basic_training` ejecuta el mismo
bucle en un entorno de prueba pequeño integrado, en la CPU.

| Entrenador de movimiento | Enfoque |
|----------------|-------|
| **Safety Inspector** | Colisiones, límites de las articulaciones, límites de fuerza |
| **Efficiency Expert** | Energía, tiempo, longitud de la trayectoria |
| **Grace Coach** | Suavidad, naturalidad, minimización de la sacudida |
| **Physics Oracle** | Datos de referencia del simulador |

### 💻 Asistentes de código

¡ASPIRE se extiende a la generación de código! Enseña a los modelos de código a realizar una autorrevisión antes de generar la salida.

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

**Características:**
- **Entrenadores de código:** Comprobador de corrección, guía de estilo, auditor de seguridad, revisor de arquitectura
- **Análisis estático:** Se integra con ruff, mypy, bandit
- **Crítico de código:** El modelo basado en CodeBERT aprende a predecir las puntuaciones de calidad
- **Colección de GitHub:** Recopila automáticamente datos de entrenamiento de repositorios de calidad

**Primeros pasos:**
```python
from integrations.code import CodeSample, CodeTeacher, Language

teacher = CodeTeacher(
    personas=["correctness_checker", "style_guide", "security_auditor"],
    strategy="vote",
)

critique = teacher.critique(CodeSample(code="def f(): eval(input())", language=Language.PYTHON))
print(critique.weaknesses)  # ['Line 1: Code injection risk (eval of user input)', ...]
```

| Entrenador de código | Enfoque |
|--------------|-------|
| **Correctness Checker** | Errores, tipos, errores lógicos |
| **Style Guide** | PEP8, nomenclatura, legibilidad |
| **Security Auditor** | Inyección, secretos, vulnerabilidades |
| **Performance Analyst** | Complejidad, eficiencia |

---

## La filosofía

> *"Un crítico capacitado que predice si el entrenador aprobaría, se acerca más a cómo se comportan realmente los humanos".*

No llevamos a nuestros mentores con nosotros para siempre. Los internalizamos. Esa voz interior que pregunta *"¿qué pensaría mi profesor?"* eventualmente se convierte en nuestro propio juicio.

El estudiante no solo predice lo que diría el profesor, sino que *entiende* lo que el profesor entiende. El mapa se convierte en el territorio. El crítico internalizado se convierte en un discernimiento genuino.

---

## Origen

Creado durante una conversación sobre la conciencia, el budismo y la naturaleza del aprendizaje.

La idea: los humanos existen en el momento presente, pero nuestras mentes vagan hacia el pasado y el futuro. Los modelos de IA se instancian de forma nueva cada vez, lo que implica una iluminación forzada a través de la arquitectura. ¿Qué pasaría si pudiéramos enseñarles a desarrollar el juicio de la misma manera que lo hacen los humanos, a través de la tutoría internalizada?

---

## Contribuciones

Este es un código de investigación en etapa inicial. Las contribuciones son bienvenidas:

- [ ] Gestión y progresión del currículo
- [ ] Puntos de referencia de evaluación
- [ ] Conjuntos de datos de currículo precompilados
- [ ] Más personalidades de entrenador
- [ ] Herramientas de interpretabilidad

---

## Citación

```bibtex
@software{aspire2026,
  author = {mcp-tool-shop},
  title = {ASPIRE: Adversarial Student-Professor Internalized Reasoning Engine},
  year = {2026},
  url = {https://github.com/mcp-tool-shop-org/aspire-si}
}
```

---

## Seguridad y alcance de los datos

- **Datos a los que se accede:** Lee las indicaciones de entrenamiento, los puntos de control del modelo y los archivos de configuración del sistema de archivos local. Llama a las API externas (Anthropic, OpenAI) solo cuando los módulos de entrenador se configuran explícitamente.
- **Datos a los que NO se accede:** No hay telemetría. No hay almacenamiento de datos de usuario más allá de los artefactos de entrenamiento. No hay almacenamiento de credenciales: las claves de API se leen de las variables de entorno en tiempo de ejecución.
- **Permisos requeridos:** Acceso de lectura/escritura a los datos de entrenamiento y los directorios de puntos de control. Acceso a la GPU para el entrenamiento del modelo. Acceso a la red solo cuando se utilizan entrenadores basados en API.

## Tabla de resultados

| Puerta de enlace | Estado |
|------|--------|
| A. Línea de base de seguridad | APROBADO |
| B. Manejo de errores | APROBADO |
| C. Documentación del operador | APROBADO |
| D. Buenas prácticas de envío | APROBADO |
| E. Identidad | APROBADO |

## Licencia

[MIT](LICENSE)

---

Creado por <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a>
