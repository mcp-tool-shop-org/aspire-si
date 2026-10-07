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

## A Ideia

**Ajuste fino tradicional:** *"Aqui estão as respostas corretas. Combine-as."*

**ASPIRE:** *"Aqui está uma mente sábia. Aprenda a pensar como ela."*

Quando você aprende com um grande mentor, você não apenas memoriza suas respostas. Você internaliza sua forma de ver as coisas. A voz dele se torna parte do seu diálogo interno. Você começa a antecipar o que ele diria, e, eventualmente, essa antecipação se torna seu próprio discernimento.

ASPIRE oferece à IA a mesma experiência.

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

O **crítico** aprende a prever o que o professor pensaria sobre uma resposta. Após o treinamento, ele avalia as respostas do aluno por conta própria — **não é necessário o professor durante a inferência** —, para que você possa escolher entre as tentativas ou escrever um loop de refinamento em torno delas.

---

## Primeiros Passos

### Instalação

```bash
git clone https://github.com/mcp-tool-shop-org/aspire-si.git
cd aspire-si
pip install -e .
```

### Defina sua chave de API

Apenas para os professores Claude ou OpenAI; um professor local não precisa (veja abaixo).

```bash
# Windows
set ANTHROPIC_API_KEY=your-key-here

# Linux/Mac
export ANTHROPIC_API_KEY=your-key-here
```

### Verifique a configuração

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

### Uma execução que não custa nada

Nenhuma chave de API é necessária: um professor local, um aluno de 1B e um professor de 3B, em uma única GPU de 24 GB.

```bash
aspire train --config examples/local-run/local-teacher.yaml \
    --prompts examples/local-run/prompts.json --geometry

# Score a response with the trained critic, without the teacher
aspire judge outputs/local-teacher/checkpoint-2 \
    --prompt "Why is the sky blue?" --response "Rayleigh scattering of sunlight."
```

---

## Personalidades de Professor

Professores diferentes produzem mentes diferentes. Escolha com sabedoria.

| Personalidade | Filosofia | Produz |
|---------|------------|----------|
| 🏛️ **Socrático** | *"Que suposição você está fazendo?"* | Raciocínio profundo, independência intelectual |
| 🔬 **Científico** | *"Quais são suas evidências?"* | Precisão técnica, pensamento rigoroso |
| 🎨 **Criativo** | *"E se tentássemos o oposto?"* | Inovação, pensamento lateral |
| ⚔️ **Adversarial** | *"Eu discordo. Defenda sua posição."* | Argumentos sólidos, convicção |
| 💚 **Compassivo** | *"Como alguém se sentiria em relação a isso?"* | Raciocínio ético, sabedoria |

### Professores Compostos

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

## Como funciona

### 1. Diálogo Adversarial

O aluno gera uma resposta. O professor a desafia. De um lado para o outro, investigando fraquezas, exigindo clareza, aprofundando.

```
Student: "Recursion works by calling itself."

Teacher (Socratic): "But what prevents infinite regress?
                     What's the mechanism that grounds the recursion?"

Student: "The base case stops it when..."

Teacher: "You say 'stops it' — but how does the computer know
          to check the base case before recursing?"
```

### 2. Treinamento do Crítico

O crítico aprende a prever o julgamento do professor sobre uma resposta: sua pontuação de 0 a 10. (Ele também tem uma compreensão do raciocínio do professor, que o treinador ainda não treina.)

```python
# The critic reads the student's hidden states over the prompt and the response
# the teacher judged, and learns to predict the teacher's 0-10 score.
critic_loss = mse(critic(student_hidden_states(prompt, response)), teacher_score)
```

### 3. O que o Aluno Aprende

O crítico é uma camada nos estados ocultos do aluno, e sua perda é transmitida de volta para o adaptador LoRA do aluno. Portanto, hoje, o aluno aprende **representações que ajudam o crítico a prever a pontuação do professor**. Ele ainda não está sendo treinado para obter melhores respostas: os termos de recompensa, contraste e trajetória em `aspire.losses` existem, mas o treinador não os alimenta, então eles não alteram nada. O treinamento do aluno com base no julgamento do crítico está planejado para a versão 1.3.0 ([#11](https://github.com/mcp-tool-shop-org/aspire-si/issues/11)).

### 4. Julgamento Sem o Professor

Após o treinamento, o crítico avalia uma resposta dos estados ocultos do aluno, portanto, nenhuma chamada ao professor é necessária para avaliá-la. Um loop de refinamento fica a seu critério para ser escrito; o ASPIRE fornece o crítico treinado e `aspire.judge`, não o loop:

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

Na linha de comando: `aspire judge outputs/checkpoint-2 --prompt "..." --response "..."`.

---

## Referência da CLI

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

Os erros imprimem um código, uma mensagem e o que fazer, sem um rastreamento:

```
ASPIRE_MISSING_API_KEY  ANTHROPIC_API_KEY not found.
To fix this, set your API key: ...
```

Códigos de saída: `0` sucesso, `1` algo que você pode corrigir (uma chave ausente, uma configuração ou arquivo de prompts incorretos), `2` uma falha durante a execução, `130` interrompido.

---

## Observando uma execução no ScalarScope

`aspire train --geometry` (ou `training.geometry_export: true` na configuração) grava `geometry.json` ao lado dos pontos de verificação: a dinâmica de treinamento da execução no formato que o [ScalarScope](https://github.com/mcp-tool-shop-org/scalarscope) lê. Abra dois deles lado a lado para comparar as execuções.

| Campo | O que ele contém |
|-------|---------------|
| Trajetória | A última camada oculta do aluno, agrupada em tokens e no lote, projetada em seus dois primeiros componentes principais. Velocidade, curvatura assinada (ângulo de giro em relação a pi, amortecida quando a execução mal se move) e dimensão efetiva (razão de participação em uma janela). |
| Escalares | Toda dimensão de avaliação que os professores pontuaram, de 0 a 1. |
| Autovalores | Por etapa, o espectro de como as pontuações de dimensão variam juntas em uma janela, como frações. Um grande primeiro valor significa que uma direção explica os julgamentos dos professores. |
| Professores | Uma seta por professor: a direção no espaço de estados ao longo da qual sua pontuação aumenta. Um professor composto fornece uma por membro. |
| Falhas | Etapas em que uma dimensão cai pelo menos 0,1 abaixo de sua mediana recente. |

Sem modelo ou chave de API? `python examples/geometry_demo.py` simula duas execuções e grava ambas as exportações, que é a maneira mais rápida de ver as visualizações.

O gravador (`aspire.geometry.GeometryRecorder`) mantém um vetor agrupado por etapa na memória. Para execuções longas, defina `training.geometry_every` para calcular a média de vários lotes em uma etapa. A exportação é gravada após cada época, bem como no final, para que uma execução interrompida prematuramente mantenha o que foi registrado.

**Analisando uma execução real.** Nas primeiras execuções com o modelo real (1,5 bilhão de parâmetros para o aluno, 32 bilhões para os professores, 3 épocas; [relatório da execução](docs/runs/2026-10-06-pod-run.md)), o estado oculto de cada etapa refletia principalmente *qual prompt* a etapa usou para o treinamento: dois prompts estão cerca de 10.000 vezes mais distantes um do outro do que as três épocas de treinamento que afetaram o aluno. Os valores escalares se repetem em cada época, porque as épocas após a primeira repetem as pontuações do professor armazenadas em cache. Para ver o que o treinamento mudou, `examples/pod-run/probe.py` e `drift.py` enviam as mesmas sequências fixas através do modelo base e de cada ponto de verificação de época e criam uma exportação de *desvio* com a identificação do prompt removida. Os exemplos reais do ScalarScope são de ambos os tipos, e todos abrangem as três épocas.

As exportações utilizam o esquema 1.1. `run_metadata` indica como ler os passos: `step_axis` é `training_step` (a exportação do modelo, na ordem de treino) ou `checkpoint_by_item` (exportações de sondagem e ajuste, um bloco dos mesmos itens por ponto de verificação, com `checkpoints` a indicar a contagem) e `scalar_source` é `live`, `replayed` (épocas após a primeira reutilização das pontuações em cache) ou `fixed_per_item`. O ScalarScope omite as comparações que interpretariam esses passos como tempo.

---

## Estrutura do Projeto

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

## Requisitos

- Python 3.10+
- PyTorch 2.0+
- Uma GPU CUDA para treinamento. Com um professor baseado em API, um aluno de 1 a 4 bilhões de parâmetros se encaixa em 16 GB; o exemplo do professor local (`examples/local-run/`) mantém o aluno e o professor em uma única placa de 24 GB. Os testes, a demonstração de geometria e os exemplos de integração são executados na CPU.
- Um professor: um modelo local (sem chave) ou uma chave de API da Anthropic ou OpenAI

### Compatibilidade com Windows

O ASPIRE é totalmente compatível com Windows, com suporte para RTX 5080/Blackwell:
- `dataloader_num_workers=0`
- `XFORMERS_DISABLED=1`
- Processamento paralelo adequado com `freeze_support()`

---

## Integrações

### 🖼️ Stable Diffusion WebUI Forge

O ASPIRE se estende à geração de imagens! Treine modelos Stable Diffusion para desenvolver o senso estético.

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
- **Professores de Visão:** Claude Vision, GPT-4V avaliam suas imagens geradas
- **Críticos de Imagem:** Críticos baseados em CLIP e no espaço latente para orientação em tempo real
- **Interface de Treinamento:** Treine adaptadores LoRA com visualização ao vivo e comparação antes/depois
- **Sem API na inferência:** O crítico treinado orienta a geração localmente

**Instalação:**
```bash
# Copy to your Forge extensions
cp -r integrations/forge /path/to/sd-webui-forge/extensions-builtin/sd_forge_aspire
```

| Professor de Visão | Foco |
|----------------|-------|
| **Balanced Critic** | Avaliação técnica e artística justa |
| **Technical Analyst** | Qualidade, artefatos, nitidez |
| **Artistic Visionary** | Criatividade e impacto emocional |
| **Composition Expert** | Equilíbrio, pontos focais, fluxo visual |
| **Harsh Critic** | Padrões muito altos |

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

**Recursos:**
- **Professores de Movimento:** Safety Inspector, Efficiency Expert, Grace Coach, Physics Oracle
- **Críticos de Trajetória:** Arquiteturas Transformer, LSTM, TCN para avaliação de movimento
- **Acelerado por GPU:** 512+ ambientes paralelos com Isaac Gym
- **Autoaperfeiçoamento:** O robô avalia seus próprios movimentos antes da execução

**Início Rápido:**
```python
from integrations.isaac import AspireIsaacTrainer, MotionTeacher

teacher = MotionTeacher(
    personas=["safety_inspector", "efficiency_expert", "grace_coach"],
    strategy="vote",
)

trainer = AspireIsaacTrainer(env="FrankaCubeStack-v0", teacher=teacher)
trainer.train(epochs=100)
```

Sem o Isaac Gym instalado, `python -m integrations.isaac.examples.basic_training` executa o mesmo loop em um pequeno ambiente de simulação integrado, na CPU.

| Professor de Movimento | Foco |
|----------------|-------|
| **Safety Inspector** | Colisões, limites de articulação, limites de força |
| **Efficiency Expert** | Energia, tempo, comprimento do caminho |
| **Grace Coach** | Suavidade, naturalidade, minimização de solavancos |
| **Physics Oracle** | Dados reais do simulador |

### 💻 Assistentes de Código

O ASPIRE se estende à geração de código! Ensine modelos de código a se autoavaliar antes de gerar a saída.

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

**Recursos:**
- **Professores de Código:** Correctness Checker, Style Guide, Security Auditor, Architecture Reviewer
- **Análise Estática:** Integra-se com ruff, mypy, bandit
- **Crítico de Código:** Modelo baseado em CodeBERT aprende a prever pontuações de qualidade
- **Coleção do GitHub:** Coleta automaticamente dados de treinamento de repositórios de qualidade

**Início Rápido:**
```python
from integrations.code import CodeSample, CodeTeacher, Language

teacher = CodeTeacher(
    personas=["correctness_checker", "style_guide", "security_auditor"],
    strategy="vote",
)

critique = teacher.critique(CodeSample(code="def f(): eval(input())", language=Language.PYTHON))
print(critique.weaknesses)  # ['Line 1: Code injection risk (eval of user input)', ...]
```

| Professor de Código | Foco |
|--------------|-------|
| **Correctness Checker** | Bugs, tipos, erros de lógica |
| **Style Guide** | PEP8, nomenclatura, legibilidade |
| **Security Auditor** | Injeção, segredos, vulnerabilidades |
| **Performance Analyst** | Complexidade, eficiência |

---

## A Filosofia

> *"Um crítico treinado que prevê se o professor aprovaria se aproxima mais de como os humanos realmente se comportam."*

Não carregamos nossos mentores para sempre. Nós os internalizamos. Essa voz interior que pergunta *"o que meu professor pensaria?"* eventualmente se torna nosso próprio julgamento.

O ASPIRE constrói essa voz interior como um crítico: um modelo que prevê o que o professor pensaria sobre uma resposta e continua a julgar depois que o professor se foi. Ensinar o próprio aluno a agir com base nessa voz é o próximo passo ([#11](https://github.com/mcp-tool-shop-org/aspire-si/issues/11)).

---

## Origem

Criado durante uma conversa sobre consciência, budismo e a natureza da aprendizagem.

A percepção: os humanos existem no momento presente, mas nossas mentes vagam para o passado e o futuro. Os modelos de IA são instanciados do zero a cada vez — iluminação forçada através da arquitetura. E se pudéssemos ensiná-los a desenvolver o julgamento da mesma forma que os humanos, através da mentoria internalizada?

---

## Contribuições

Este é um código de pesquisa em estágio inicial. As contribuições são bem-vindas:

- [ ] Gerenciamento e progressão do currículo
- [ ] Bancos de dados de avaliação
- [ ] Conjuntos de dados de currículo pré-construídos
- [ ] Mais personas de professores
- [ ] Ferramentas de interpretabilidade

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

- **Dados acessados:** Lê prompts de treinamento, pontos de verificação do modelo e arquivos de configuração do sistema de arquivos local. Chama APIs externas (Anthropic, OpenAI) apenas quando os módulos de professor são explicitamente configurados.
- **Dados NÃO acessados:** Sem telemetria. Sem armazenamento de dados do usuário além dos artefatos de treinamento. Sem armazenamento de credenciais — as chaves de API são lidas das variáveis de ambiente em tempo de execução.
- **Permissões necessárias:** Acesso de leitura/gravação aos diretórios de dados de treinamento e pontos de verificação. Acesso à GPU para treinamento do modelo. Acesso à rede apenas quando usar professores baseados em API.

## Placar

| Portão | Status |
|------|--------|
| A. Linha de Base de Segurança | APROVADO |
| B. Tratamento de Erros | APROVADO |
| C. Documentação para Operadores | APROVADO |
| D. Boas Práticas de Envio | APROVADO |
| E. Identidade | APROVADO |

## Licença

[MIT](LICENSE)

---

Desenvolvido por <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a>
