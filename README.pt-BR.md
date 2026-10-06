<p align="center">
  <a href="README.ja.md">日本語</a> | <a href="README.zh.md">中文</a> | <a href="README.es.md">Español</a> | <a href="README.fr.md">Français</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.it.md">Italiano</a> | <a href="README.md">English</a>
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

## A Ideia

**Ajuste fino tradicional:** *"Aqui estão as respostas corretas. Combine-as."*

**ASPIRE:** *"Aqui está uma mente sábia. Aprenda a pensar como ela."*

Quando você aprende com um grande mentor, não apenas memoriza suas respostas. Você internaliza a maneira como ele pensa. Sua voz se torna parte do seu diálogo interno. Você começa a antecipar o que ele diria, e eventualmente essa antecipação se torna seu próprio discernimento.

ASPIRE oferece a mesma experiência para a IA.

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

O **crítico** aprende a prever o que o professor pensaria. Após o treinamento, o aluno usa esse crítico internalizado para se aprimorar continuamente — **sem a necessidade de um professor durante a inferência**.

---

## Início Rápido

### Instalação

```bash
git clone https://github.com/mcp-tool-shop-org/aspire-si.git
cd aspire-si
pip install -e .
```

### Configure sua Chave de API

```bash
# Windows
set ANTHROPIC_API_KEY=your-key-here

# Linux/Mac
export ANTHROPIC_API_KEY=your-key-here
```

### Verifique a Configuração

```bash
# Check your environment (Python, CUDA, API keys)
aspire doctor
```

### Experimente

```bash
# See available teacher personas
aspire teachers

# Generate an adversarial dialogue
aspire dialogue "Explain why recursion works" --teacher socratic --turns 3

# Initialize a training config
aspire init --output my-config.yaml
```

---

## Personagens de Professores

Professores diferentes produzem mentes diferentes. Escolha com sabedoria.

| Personagem | Filosofia | Produz |
|---------|------------|----------|
| 🏛️ **Sócrático** | *"Qual é a sua premissa?"* | Raciocínio profundo, independência intelectual |
| 🔬 **Científico** | *"Qual é a sua evidência?"* | Precisão técnica, pensamento rigoroso |
| 🎨 **Criativo** | *"E se tentássemos o oposto?"* | Inovação, pensamento lateral |
| ⚔️ **Adversarial** | *"Eu discordo. Defenda sua posição."* | Argumentos sólidos, convicção |
| 💚 **Compassivo** | *"Como alguém se sentiria em relação a isso?"* | Raciocínio ético, sabedoria |

### Professores Combinados

Combine vários professores para um aprendizado mais rico:

```python
from aspire.teachers import CompositeTeacher, SocraticTeacher, ScientificTeacher

# A committee of mentors
teacher = CompositeTeacher(
    teachers=[SocraticTeacher(), ScientificTeacher()],
    strategy="vote"  # or "rotate", "debate"
)
```

---

## Como Funciona

### 1. Diálogo Adversarial

O aluno gera uma resposta. O professor a desafia. Em um ciclo, explorando fraquezas, exigindo clareza, aprofundando o conhecimento.

```
Student: "Recursion works by calling itself."

Teacher (Socratic): "But what prevents infinite regress?
                     What's the mechanism that grounds the recursion?"

Student: "The base case stops it when..."

Teacher: "You say 'stops it' — but how does the computer know
          to check the base case before recursing?"
```

### 2. Treinamento do Crítico

O crítico aprende a prever o julgamento do professor — não apenas a pontuação, mas o *raciocínio*.

```python
critic_loss = predict_teacher_judgment(
    score=True,      # "This deserves a 7/10"
    reasoning=True,  # "Because the explanation lacks depth on X"
)
```

### 3. Treinamento do Aluno

O aluno aprende com o julgamento internalizado do crítico, buscando o que o professor aprovaria.

```python
student_loss = (
    reward_from_critic +      # Higher score = better
    contrastive_to_teacher +  # Pull toward teacher's improved version
    trajectory_improvement    # Get better across dialogue turns
)
```

### 4. Magia da Inferência

Após o treinamento, o aluno se aprimora continuamente usando o crítico internalizado. **Não são necessárias chamadas de API do professor.**

```python
def generate_with_judgment(prompt):
    response = student.generate(prompt)

    while critic.score(response) < threshold:
        response = student.refine(response, critic.feedback)

    return response  # Self-improved through internalized judgment
```

---

## Referência da Linha de Comando (CLI)

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

## Estrutura do Projeto

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
- GPU CUDA (16GB+ de VRAM recomendado)
- Chave de API da Anthropic (para o professor Claude) ou chave de API da OpenAI

### Compatibilidade com Windows

ASPIRE é totalmente compatível com Windows e suporta RTX 5080/Blackwell:
- `dataloader_num_workers=0`
- `XFORMERS_DISABLED=1`
- Processamento paralelo adequado com `freeze_support()`

---

## Integrações

### 🖼️ Stable Diffusion WebUI Forge

ASPIRE se estende à geração de imagens! Treine modelos Stable Diffusion para desenvolver senso estético.

```
integrations/forge/
├── scripts/
│   ├── aspire_generate.py   # Critic-guided generation
│   └── aspire_train.py      # Training interface
├── vision_teacher.py        # Claude Vision / GPT-4V teachers
├── image_critic.py          # CLIP and latent-space critics
└── README.md
```

**Recursos:**
- **Professores Visuais**: Claude Vision, GPT-4V criticam suas imagens geradas
- **Críticos de Imagem**: Críticos baseados em CLIP e em espaço latente para orientação em tempo real
- **Interface de Treinamento**: Treine adaptadores LoRA com visualização em tempo real e comparação antes/depois
- **Sem API durante a inferência**: O crítico treinado guia a geração localmente

**Instalação:**
```bash
# Copy to your Forge extensions
cp -r integrations/forge /path/to/sd-webui-forge/extensions-builtin/sd_forge_aspire
```

| Professor Visual | Foco |
|----------------|-------|
| **Balanced Critic** | Avaliação técnica e artística justa |
| **Technical Analyst** | Qualidade, artefatos, nitidez |
| **Artistic Visionary** | Criatividade e impacto emocional |
| **Composition Expert** | Equilíbrio, pontos focais, fluxo visual |
| **Harsh Critic** | Padrões muito elevados. |

### 🤖 Isaac Gym / Isaac Lab (Robótica)

O ASPIRE se estende à IA incorporada! Ensine robôs a desenvolver intuição física.

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
- **Professores de Movimento:** Inspetor de Segurança, Especialista em Eficiência, Treinador de Elegância, Oráculo da Física.
- **Críticos de Trajetória:** Arquiteturas Transformer, LSTM, TCN para avaliação de movimentos.
- **Acelerado por GPU:** Mais de 512 ambientes paralelos com o Isaac Gym.
- **Autoaperfeiçoamento:** O robô avalia seus próprios movimentos antes da execução.

**Como Começar:**
```python
from aspire.integrations.isaac import AspireIsaacTrainer, MotionTeacher

teacher = MotionTeacher(
    personas=["safety_inspector", "efficiency_expert", "grace_coach"],
    strategy="vote",
)

trainer = AspireIsaacTrainer(env="FrankaCubeStack-v0", teacher=teacher)
trainer.train(epochs=100)
```

| Professor de Movimento | Foco |
|----------------|-------|
| **Safety Inspector** | Colisões, limites das juntas, limites de força. |
| **Efficiency Expert** | Energia, tempo, comprimento do caminho. |
| **Grace Coach** | Suavidade, naturalidade, minimização de solavancos. |
| **Physics Oracle** | Dados de referência do simulador. |

### 💻 Assistentes de Código

O ASPIRE se estende à geração de código! Ensine modelos de código a fazer autoavaliação antes de gerar a saída.

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
- **Professores de Código:** Verificador de Correção, Guia de Estilo, Auditor de Segurança, Avaliador de Arquitetura.
- **Análise Estática:** Integra-se com ruff, mypy, bandit.
- **Crítico de Código:** Modelo baseado em CodeBERT que aprende a prever pontuações de qualidade.
- **Coleção do GitHub:** Coleta automaticamente dados de treinamento de repositórios de qualidade.

**Como Começar:**
```python
from aspire.integrations.code import CodeTeacher, CodeSample

teacher = CodeTeacher(
    personas=["correctness_checker", "style_guide", "security_auditor"],
    strategy="vote",
)

critique = teacher.critique(CodeSample(code="def f(): eval(input())", language="python"))
print(f"Score: {critique.overall_score}/10")  # Low score - security issue!
```

| Professor de Código | Foco |
|--------------|-------|
| **Correctness Checker** | Bugs, tipos, erros de lógica. |
| **Style Guide** | PEP8, nomenclatura, legibilidade. |
| **Security Auditor** | Injeção, segredos, vulnerabilidades. |
| **Performance Analyst** | Complexidade, eficiência. |

---

## A Filosofia

> *"Um crítico treinado que prevê se o professor aprovaria está mais próximo de como os humanos realmente se comportam."*

Nós não carregamos nossos mentores conosco para sempre. Nós os internalizamos. Aquela voz interior que pergunta *"o que meu professor pensaria?"* eventualmente se torna nosso próprio julgamento.

O aluno não apenas prevê o que o professor diria — ele *entende* o que o professor entende. O mapa se torna o território. O crítico internalizado se torna discernimento genuíno.

---

## Origem

Criado durante uma conversa sobre consciência, budismo e a natureza da aprendizagem.

A ideia: os humanos existem no momento presente, mas nossas mentes vagueiam para o passado e o futuro. Os modelos de IA são instanciados a cada vez — iluminação forçada através da arquitetura. E se pudéssemos ensiná-los a desenvolver o julgamento da mesma forma que os humanos, através da mentoria internalizada?

---

## Contribuições

Este é um código de pesquisa em estágio inicial. Contribuições são bem-vindas:

- [ ] Gerenciamento e progressão do currículo.
- [ ] Métricas de avaliação.
- [ ] Conjuntos de dados de currículo pré-construídos.
- [ ] Mais personas de professores.
- [ ] Ferramentas de interpretabilidade.

---

## Citação

```bibtex
@software{aspire2026,
  author = {mcp-tool-shop},
  title = {ASPIRE: Adversarial Student-Professor Internalized Reasoning Engine},
  year = {2026},
  url = {https://github.com/mcp-tool-shop-org/aspire-si}
}
```

---

## Segurança e Escopo de Dados

- **Dados acessados:** Lê prompts de treinamento, pontos de verificação do modelo e arquivos de configuração do sistema de arquivos local. Chama APIs externas (Anthropic, OpenAI) apenas quando os módulos do professor são configurados explicitamente.
- **Dados NÃO acessados:** Sem telemetria. Sem armazenamento de dados do usuário além dos artefatos de treinamento. Sem armazenamento de credenciais — as chaves da API são lidas de variáveis de ambiente em tempo de execução.
- **Permissões necessárias:** Acesso de leitura/gravação aos diretórios de dados de treinamento e de pontos de verificação. Acesso à GPU para treinamento do modelo. Acesso à rede apenas ao usar professores baseados em API.

## Tabela de Avaliação

| Porta de Entrada | Status |
|------|--------|
| A. Baseline de Segurança | APROVADO |
| B. Tratamento de Erros | APROVADO |
| C. Documentação para Operadores | APROVADO |
| D. Higiene de Implantação | APROVADO |
| E. Identidade | APROVADO |

## Licença

[MIT](LICENSE)

---

Criado por <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a
