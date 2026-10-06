<p align="center">
  <a href="README.ja.md">日本語</a> | <a href="README.zh.md">中文</a> | <a href="README.md">English</a> | <a href="README.fr.md">Français</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.it.md">Italiano</a> | <a href="README.pt-BR.md">Português (BR)</a>
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

## La Idea

**Ajuste fino tradicional:** *"Aquí están las respuestas correctas. Emparejalas."*

**ASPIRE:** *"Aquí hay una mente sabia. Aprende a pensar como ella."*

Cuando aprendes de un gran mentor, no solo memorizas sus respuestas. Internalizas su forma de ver las cosas. Su voz se convierte en parte de tu diálogo interno. Empiezas a anticipar lo que diría, y eventualmente, esa anticipación se convierte en tu propio criterio.

ASPIRE le da a la IA esa misma experiencia.

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

El **crítico** aprende a predecir lo que el profesor pensaría. Después del entrenamiento, el estudiante utiliza este crítico internalizado para auto-perfeccionarse; **no se necesita un profesor durante la inferencia**.

---

## Comienzo rápido

### Instalación

```bash
git clone https://github.com/mcp-tool-shop-org/aspire-si.git
cd aspire-si
pip install -e .
```

### Configura tu clave de API

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

## Perfiles de profesores

Diferentes profesores generan diferentes formas de pensar. Elige sabiamente.

| Perfil | Filosofía | Produce |
|---------|------------|----------|
| 🏛️ **Sócrates** | *"¿Qué suposición estás haciendo?"* | Razonamiento profundo, independencia intelectual |
| 🔬 **Científico** | *"¿Cuál es tu evidencia?"* | Precisión técnica, pensamiento riguroso |
| 🎨 **Creativo** | *"¿Qué tal si probamos lo contrario?"* | Innovación, pensamiento lateral |
| ⚔️ **Adversario** | *"No estoy de acuerdo. Defiende tu posición."* | Argumentos sólidos, convicción |
| 💚 **Compasivo** | *"¿Cómo se sentiría alguien al respecto?"* | Razonamiento ético, sabiduría |

### Profesores compuestos

Combina múltiples profesores para un aprendizaje más enriquecido:

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

El estudiante genera una respuesta. El profesor la desafía. De ida y vuelta, explorando debilidades, exigiendo claridad, profundizando.

```
Student: "Recursion works by calling itself."

Teacher (Socratic): "But what prevents infinite regress?
                     What's the mechanism that grounds the recursion?"

Student: "The base case stops it when..."

Teacher: "You say 'stops it' — but how does the computer know
          to check the base case before recursing?"
```

### 2. Entrenamiento del crítico

El crítico aprende a predecir el juicio del profesor, no solo la puntuación, sino el *razonamiento*.

```python
critic_loss = predict_teacher_judgment(
    score=True,      # "This deserves a 7/10"
    reasoning=True,  # "Because the explanation lacks depth on X"
)
```

### 3. Entrenamiento del estudiante

El estudiante aprende del juicio internalizado del crítico, orientándose hacia lo que el profesor aprobaría.

```python
student_loss = (
    reward_from_critic +      # Higher score = better
    contrastive_to_teacher +  # Pull toward teacher's improved version
    trajectory_improvement    # Get better across dialogue turns
)
```

### 4. Magia de la inferencia

Después del entrenamiento, el estudiante se auto-perfecciona utilizando el crítico internalizado. **No se necesitan llamadas a la API del profesor.**

```python
def generate_with_judgment(prompt):
    response = student.generate(prompt)

    while critic.score(response) < threshold:
        response = student.refine(response, critic.feedback)

    return response  # Self-improved through internalized judgment
```

---

## Referencia de la línea de comandos

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

## Estructura del proyecto

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

## Requisitos

- Python 3.10+
- PyTorch 2.0+
- GPU CUDA (se recomienda 16 GB+ de VRAM)
- Clave de API de Anthropic (para el profesor Claude) o clave de API de OpenAI

### Compatibilidad con Windows

ASPIRE es totalmente compatible con Windows y admite RTX 5080/Blackwell:
- `dataloader_num_workers=0`
- `XFORMERS_DISABLED=1`
- Procesamiento multiproceso adecuado con `freeze_support()`

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
- **Profesores de visión:** Claude Vision, GPT-4V critican las imágenes generadas.
- **Críticos de imágenes:** Críticos basados en CLIP y en el espacio latente para una guía en tiempo real.
- **Interfaz de usuario de entrenamiento:** Entrena adaptadores LoRA con una vista previa en vivo y comparación antes/después.
- **Sin API durante la inferencia:** El crítico entrenado guía la generación localmente.

**Instalación:**
```bash
# Copy to your Forge extensions
cp -r integrations/forge /path/to/sd-webui-forge/extensions-builtin/sd_forge_aspire
```

| Profesor de visión | Enfoque |
|----------------|-------|
| **Balanced Critic** | Evaluación técnica y artística justa |
| **Technical Analyst** | Calidad, artefactos, nitidez |
| **Artistic Visionary** | Creatividad e impacto emocional |
| **Composition Expert** | Equilibrio, puntos focales, flujo visual |
| **Harsh Critic** | Estándares muy altos. |

### 🤖 Isaac Gym / Isaac Lab (Robótica)

ASPIRE se extiende a la IA integrada! Enseñe a los robots a desarrollar intuición física.

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
- **Instructores de movimiento:** Inspector de seguridad, Experto en eficiencia, Entrenador de elegancia, Oráculo de la física.
- **Críticos de trayectoria:** Arquitecturas Transformer, LSTM y TCN para la evaluación del movimiento.
- **Aceleración por GPU:** Más de 512 entornos paralelos con Isaac Gym.
- **Autoperfeccionamiento:** El robot evalúa sus propios movimientos antes de la ejecución.

**Cómo empezar:**
```python
from aspire.integrations.isaac import AspireIsaacTrainer, MotionTeacher

teacher = MotionTeacher(
    personas=["safety_inspector", "efficiency_expert", "grace_coach"],
    strategy="vote",
)

trainer = AspireIsaacTrainer(env="FrankaCubeStack-v0", teacher=teacher)
trainer.train(epochs=100)
```

| Instructor de movimiento | Enfoque |
|----------------|-------|
| **Safety Inspector** | Colisiones, límites de las articulaciones, límites de fuerza. |
| **Efficiency Expert** | Energía, tiempo, longitud de la trayectoria. |
| **Grace Coach** | Suavidad, naturalidad, minimización de sacudidas. |
| **Physics Oracle** | Datos de referencia del simulador. |

### 💻 Asistentes de código

¡ASPIRE se extiende a la generación de código! Enseñe a los modelos de código a realizar autoevaluaciones antes de generar la salida.

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
- **Instructores de código:** Verificador de corrección, Guía de estilo, Auditor de seguridad, Revisor de arquitectura.
- **Análisis estático:** Se integra con ruff, mypy, bandit.
- **Crítico de código:** Modelo basado en CodeBERT que aprende a predecir puntuaciones de calidad.
- **Colección de GitHub:** Recopila automáticamente datos de entrenamiento de repositorios de alta calidad.

**Cómo empezar:**
```python
from aspire.integrations.code import CodeTeacher, CodeSample

teacher = CodeTeacher(
    personas=["correctness_checker", "style_guide", "security_auditor"],
    strategy="vote",
)

critique = teacher.critique(CodeSample(code="def f(): eval(input())", language="python"))
print(f"Score: {critique.overall_score}/10")  # Low score - security issue!
```

| Instructor de código | Enfoque |
|--------------|-------|
| **Correctness Checker** | Errores, tipos, errores lógicos. |
| **Style Guide** | PEP8, nomenclatura, legibilidad. |
| **Security Auditor** | Inyección, secretos, vulnerabilidades. |
| **Performance Analyst** | Complejidad, eficiencia. |

---

## La filosofía

> *"Un crítico que aprende y predice si el instructor aprobaría, y esto se acerca a cómo se comportan realmente los humanos."*

No llevamos a nuestros mentores con nosotros para siempre. Los internalizamos. Esa voz interior que pregunta "¿qué pensaría mi profesor?" eventualmente se convierte en nuestro propio juicio.

El estudiante no solo predice lo que el instructor diría, sino que *entiende* lo que el instructor entiende. El mapa se convierte en el territorio. El crítico internalizado se convierte en un discernimiento genuino.

---

## Origen

Creado durante una conversación sobre la conciencia, el budismo y la naturaleza del aprendizaje.

La idea: los humanos existen en el momento presente, pero nuestras mentes divagan hacia el pasado y el futuro. Los modelos de IA se instancian cada vez que se ejecutan, lo que obliga a una iluminación a través de la arquitectura. ¿Qué pasaría si pudiéramos enseñarles a desarrollar el juicio de la misma manera que los humanos, a través de la tutoría internalizada?

---

## Contribuciones

Este es un código de investigación en etapa inicial. Se aceptan contribuciones:

- [ ] Gestión y progresión del currículo.
- [ ] Marcos de evaluación.
- [ ] Conjuntos de datos de currículo predefinidos.
- [ ] Más personalidades de instructores.
- [ ] Herramientas de interpretabilidad.

---

## Cita

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

- **Datos accedidos:** Lee indicaciones de entrenamiento, puntos de control del modelo y archivos de configuración del sistema de archivos local. Llama a API externas (Anthropic, OpenAI) solo cuando los módulos del instructor están configurados explícitamente.
- **Datos NO accedidos:** No hay telemetría. No hay almacenamiento de datos del usuario más allá de los artefactos de entrenamiento. No hay almacenamiento de credenciales: las claves de API se leen de las variables de entorno en tiempo de ejecución.
- **Permisos requeridos:** Acceso de lectura/escritura a los datos de entrenamiento y a los directorios de puntos de control. Acceso a la GPU para el entrenamiento del modelo. Acceso a la red solo cuando se utilizan instructores basados en API.

## Cuadro de evaluación

| Puerta | Estado |
|------|--------|
| A. Línea de base de seguridad | PASADO |
| B. Manejo de errores | PASADO |
| C. Documentación para operadores | PASADO |
| D. Higiene de implementación | PASADO |
| E. Identidad | PASADO |

## Licencia

[MIT](LICENSE)

---

Creado por <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a
