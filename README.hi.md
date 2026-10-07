<p align="center">
  <a href="README.ja.md">日本語</a> | <a href="README.zh.md">中文</a> | <a href="README.es.md">Español</a> | <a href="README.fr.md">Français</a> | <a href="README.md">English</a> | <a href="README.it.md">Italiano</a> | <a href="README.pt-BR.md">Português (BR)</a>
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

## विचार

**पारंपरिक फाइन-ट्यूनिंग:** *"यहाँ सही उत्तर दिए गए हैं। उन्हें मिलाएं।"*

**एस्पायर:** *"यहाँ एक बुद्धिमान मस्तिष्क है। इससे सीखने की तरह सोचने का तरीका सीखें।"*

जब आप किसी महान गुरु से सीखते हैं, तो आप केवल उनके उत्तरों को याद नहीं करते हैं। आप उनके देखने के तरीके को आत्मसात करते हैं। उनकी आवाज आपके आंतरिक संवाद का हिस्सा बन जाती है। आप यह अनुमान लगाना शुरू करते हैं कि वे क्या कहेंगे, और अंततः वह अनुमान आपकी अपनी समझ बन जाता है।

एस्पायर एआई को वही अनुभव प्रदान करता है।

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

**आलोचक** यह सीखने लगता है कि शिक्षक क्या सोचेंगे। प्रशिक्षण के बाद, छात्र इस आंतरिक आलोचक का उपयोग स्वयं को बेहतर बनाने के लिए करता है - **अनुमान के समय किसी शिक्षक की आवश्यकता नहीं होती है**।

---

## त्वरित शुरुआत

### स्थापना

```bash
git clone https://github.com/mcp-tool-shop-org/aspire-si.git
cd aspire-si
pip install -e .
```

### अपनी एपीआई कुंजी सेट करें

```bash
# Windows
set ANTHROPIC_API_KEY=your-key-here

# Linux/Mac
export ANTHROPIC_API_KEY=your-key-here
```

### सेटअप सत्यापित करें

```bash
# Check your environment (Python, CUDA, API keys)
aspire doctor
```

### इसे आजमाएं

```bash
# See available teacher personas
aspire teachers

# Generate an adversarial dialogue
aspire dialogue "Explain why recursion works" --teacher socratic --turns 3

# Initialize a training config
aspire init --output my-config.yaml
```

---

## शिक्षक व्यक्तित्व

विभिन्न शिक्षक विभिन्न प्रकार के मस्तिष्क का निर्माण करते हैं। बुद्धिमानी से चुनें।

| व्यक्तित्व | दर्शन | उत्पन्न करता है |
|---------|------------|----------|
| 🏛️ **सुकराती** | *"आप कौन सी धारणा बना रहे हैं?"* | गहन तर्क, बौद्धिक स्वतंत्रता |
| 🔬 **वैज्ञानिक** | *"आपका क्या प्रमाण है?"* | तकनीकी सटीकता, कठोर सोच |
| 🎨 **रचनात्मक** | *"अगर हमने इसके विपरीत प्रयास किया तो क्या होगा?"* | नवाचार, पार्श्व सोच |
| ⚔️ **विरोधाभासी** | *"मैं असहमत हूँ। अपनी स्थिति का बचाव करें।"* | मजबूत तर्क, दृढ़ विश्वास |
| 💚 **दयालु** | *"कोई व्यक्ति इसके बारे में कैसा महसूस कर सकता है?"* | नैतिक तर्क, ज्ञान |

### संयुक्त शिक्षक

अधिक समृद्ध सीखने के लिए कई शिक्षकों को मिलाएं:

```python
from aspire.teachers import CompositeTeacher, SocraticTeacher, ScientificTeacher

# A committee of mentors
teacher = CompositeTeacher(
    teachers=[SocraticTeacher(), ScientificTeacher()],
    strategy="vote"  # or "rotate", "debate"
)
```

---

## यह कैसे काम करता है

### 1. विरोधाभासी संवाद

छात्र एक प्रतिक्रिया उत्पन्न करता है। शिक्षक इसे चुनौती देता है। आगे-पीछे, कमजोरियों की जांच, स्पष्टता की मांग, गहराई तक जाने के लिए प्रेरित करना।

```
Student: "Recursion works by calling itself."

Teacher (Socratic): "But what prevents infinite regress?
                     What's the mechanism that grounds the recursion?"

Student: "The base case stops it when..."

Teacher: "You say 'stops it' — but how does the computer know
          to check the base case before recursing?"
```

### 2. आलोचक प्रशिक्षण

The critic learns to predict the teacher's judgment of a response: its 0-10 score. (It has a
head for the teacher's reasoning too, which the trainer does not train yet.)

```python
# The critic reads the student's hidden states over the prompt and the response
# the teacher judged, and learns to predict the teacher's 0-10 score.
critic_loss = mse(critic(student_hidden_states(prompt, response)), teacher_score)
```

### 3. छात्र प्रशिक्षण

The critic is a head on the student's hidden states, and its loss flows back into the
student's LoRA adapter. So today the student learns **representations that help the critic
predict the teacher's score**. It is not yet trained toward better answers: the reward,
contrastive and trajectory terms in `aspire.losses` exist, but the trainer does not feed
them, so they move nothing. Training the student on the critic's judgment is planned for
1.3.0 ([#11](https://github.com/mcp-tool-shop-org/aspire-si/issues/11)).

### 4. शिक्षक के बिना निर्णय

प्रशिक्षण के बाद, आलोचक छात्र की छिपी हुई अवस्थाओं से अकेले एक प्रतिक्रिया का मूल्यांकन करता है, इसलिए किसी शिक्षक एपीआई कॉल की आवश्यकता नहीं होती है। एक परिष्करण लूप आपके द्वारा इसके चारों ओर लिखने के लिए है; एस्पायर प्रशिक्षित आलोचक को भेजता है, लूप को नहीं:

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

## सीएलआई संदर्भ

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

त्रुटियाँ एक कोड, एक संदेश और क्या करना है, प्रिंट करती हैं, बिना किसी ट्रेसबैक के:

```
ASPIRE_MISSING_API_KEY  ANTHROPIC_API_KEY not found.
To fix this, set your API key: ...
```

निकास कोड: `0` सफलता, `1` कुछ ऐसा जिसे आप ठीक कर सकते हैं (एक गायब कुंजी, एक खराब कॉन्फ़िगरेशन या प्रॉम्प्ट फ़ाइल), `2` चलाने के दौरान विफलता, `130` बाधित।

---

## स्केलरस्कोप में एक रन देखना

`aspire train --geometry` (या कॉन्फ़िगरेशन में `training.geometry_export: true`) लिखता है
`geometry.json` चेकपॉइंट के बगल में: रन की प्रशिक्षण गतिशीलता उस प्रारूप में जिसमें [स्केलरस्कोप](https://github.com/mcp-tool-shop-org/scalarscope) पढ़ता है। दो को एक साथ खोलें ताकि रनों की तुलना की जा सके।

| फ़ील्ड | इसमें क्या है |
|-------|---------------|
| प्रक्षेपवक्र | छात्र की अंतिम छिपी हुई परत, टोकन और बैच पर एकत्रित, इसके पहले दो प्रमुख घटकों पर प्रक्षेपित। वेग, हस्ताक्षरित वक्रता (पाई पर घूमने का कोण, जब रन मुश्किल से हिलता है तो कम हो जाता है), और प्रभावी आयाम (एक विंडो में भागीदारी अनुपात)। |
| स्केलर | प्रत्येक मूल्यांकन आयाम जिसे शिक्षकों ने स्कोर किया, 0 से 1 तक। |
| आइगेनवैल्यू | प्रति चरण, आयाम स्कोर एक विंडो में एक साथ कैसे बदलते हैं, इसके स्पेक्ट्रम के रूप में अंश। एक बड़ा पहला मान का मतलब है कि एक दिशा शिक्षकों के निर्णयों की व्याख्या करती है। |
| प्रोफेसर | प्रति शिक्षक एक तीर: राज्य स्थान में वह दिशा जिसके साथ उसका स्कोर बढ़ता है। एक संयुक्त शिक्षक प्रति सदस्य एक देता है। |
| विफलताएँ | वे चरण जहाँ एक आयाम अपने हालिया माध्य से कम से कम 0.1 नीचे गिर जाता है। |

कोई मॉडल या एपीआई कुंजी नहीं? `python examples/geometry_demo.py` दो रन का अनुकरण करता है और दोनों निर्यात लिखता है, जो दृश्यों को देखने का सबसे तेज़ तरीका है।

रिकॉर्डर (`aspire.geometry.GeometryRecorder`) प्रत्येक चरण में मेमोरी में एक एकत्रित वेक्टर रखता है। लंबे रनों के लिए `training.geometry_every` सेट करें ताकि कई बैचों को एक चरण में औसत किया जा सके।

---

## परियोजना संरचना

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

## आवश्यकताएँ

- पायथन 3.10+
- पायटॉर्च 2.0+
- प्रशिक्षण के लिए एक CUDA GPU (16GB+ VRAM अनुशंसित)। परीक्षण, ज्यामिति डेमो और एकीकरण उदाहरण CPU पर चलते हैं।
- एंथ्रोपिक एपीआई कुंजी (क्लाउड शिक्षक के लिए) या ओपनएआई एपीआई कुंजी

### विंडोज संगतता

एस्पायर पूरी तरह से विंडोज-संगत है जिसमें RTX 5080/ब्लैकवेल समर्थन है:
- `dataloader_num_workers=0`
- `XFORMERS_DISABLED=1`
- `freeze_support()` के साथ उचित मल्टीप्रोसेसिंग

---

## एकीकरण

### 🖼️ स्थिर प्रसार वेबयूआई फोर्ज

एस्पायर छवि पीढ़ी तक विस्तारित होता है! सौंदर्य निर्णय विकसित करने के लिए स्थिर प्रसार मॉडल को प्रशिक्षित करें।

```
integrations/forge/
├── scripts/
│   ├── aspire_generate.py   # Critic-guided generation
│   └── aspire_train.py      # Training interface
├── vision_teacher.py        # Claude Vision / GPT-4V teachers
├── image_critic.py          # CLIP and latent-space critics
└── README.md
```

**विशेषताएँ:**
- **विज़न शिक्षक**: क्लाउड विज़न, जीपीटी-4वी आपकी उत्पन्न छवियों की आलोचना करते हैं
- **छवि आलोचक**: वास्तविक समय मार्गदर्शन के लिए CLIP-आधारित और गुप्त-स्थान आलोचक
- **प्रशिक्षण यूआई**: लाइव पूर्वावलोकन और पहले/बाद की तुलना के साथ LoRA एडेप्टर को प्रशिक्षित करें
- **अनुमान पर कोई एपीआई नहीं**: प्रशिक्षित आलोचक स्थानीय रूप से पीढ़ी का मार्गदर्शन करता है

**स्थापना:**
```bash
# Copy to your Forge extensions
cp -r integrations/forge /path/to/sd-webui-forge/extensions-builtin/sd_forge_aspire
```

| विज़न शिक्षक | ध्यान दें |
|----------------|-------|
| **Balanced Critic** | उचित तकनीकी और कलात्मक मूल्यांकन |
| **Technical Analyst** | गुणवत्ता, कलाकृतियाँ, तीक्ष्णता |
| **Artistic Visionary** | रचनात्मकता और भावनात्मक प्रभाव |
| **Composition Expert** | संतुलन, केंद्र बिंदु, दृश्य प्रवाह |
| **Harsh Critic** | बहुत उच्च मानक |

### 🤖 आइज़ैक जिम / आइज़ैक लैब (रोबोटिक्स)

एस्पायर, एम्बोडेड एआई तक विस्तारित! रोबोट को शारीरिक अंतर्ज्ञान विकसित करना सिखाएं।

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

**विशेषताएं:**
- **मोशन टीचर्स**: सेफ्टी इंस्पेक्टर, एफिशिएंसी एक्सपर्ट, ग्रेस कोच, फिजिक्स ओरेकल
- **ट्रैजेक्टरी क्रिटिक्स**: ट्रांसफॉर्मर, एलएसटीएम, टीसीएन आर्किटेक्चर, मोशन मूल्यांकन के लिए
- **जीपीयू-त्वरित**: आइज़ैक जिम के साथ 512+ समानांतर वातावरण
- **सेल्फ-रिफाइनमेंट**: रोबोट, निष्पादन से पहले अपनी गति का मूल्यांकन करता है

**त्वरित शुरुआत:**
```python
from integrations.isaac import AspireIsaacTrainer, MotionTeacher

teacher = MotionTeacher(
    personas=["safety_inspector", "efficiency_expert", "grace_coach"],
    strategy="vote",
)

trainer = AspireIsaacTrainer(env="FrankaCubeStack-v0", teacher=teacher)
trainer.train(epochs=100)
```

आइज़ैक जिम स्थापित किए बिना, `python -m integrations.isaac.examples.basic_training` एक छोटे अंतर्निहित स्टैंड-इन वातावरण पर, सीपीयू पर समान लूप चलाता है।

| मोशन टीचर | ध्यान दें |
|----------------|-------|
| **Safety Inspector** | टकराव, संयुक्त सीमाएं, बल सीमाएं |
| **Efficiency Expert** | ऊर्जा, समय, पथ की लंबाई |
| **Grace Coach** | चिकनाई, प्राकृतिकता, झटके को कम करना |
| **Physics Oracle** | सिमुलेटर से वास्तविक डेटा |

### 💻 कोड असिस्टेंट

एस्पायर, कोड जनरेशन तक विस्तारित! कोड मॉडल को आउटपुट करने से पहले स्वयं-समीक्षा करना सिखाएं।

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

**विशेषताएं:**
- **कोड टीचर्स**: करेक्टनेस चेकर, स्टाइल गाइड, सिक्योरिटी ऑडिटर, आर्किटेक्चर रिव्यूअर
- **स्टैटिक एनालिसिस**: रफ, मायपी, बैंडिट के साथ एकीकृत
- **कोड क्रिटिक**: कोडबर्ट-आधारित मॉडल, गुणवत्ता स्कोर की भविष्यवाणी करना सीखता है
- **गिटहब कलेक्शन**: गुणवत्तापूर्ण रिपॉजिटरी से स्वचालित रूप से प्रशिक्षण डेटा एकत्र करें

**त्वरित शुरुआत:**
```python
from integrations.code import CodeSample, CodeTeacher, Language

teacher = CodeTeacher(
    personas=["correctness_checker", "style_guide", "security_auditor"],
    strategy="vote",
)

critique = teacher.critique(CodeSample(code="def f(): eval(input())", language=Language.PYTHON))
print(critique.weaknesses)  # ['Line 1: Code injection risk (eval of user input)', ...]
```

| कोड टीचर | ध्यान दें |
|--------------|-------|
| **Correctness Checker** | बग, प्रकार, तर्क त्रुटियां |
| **Style Guide** | पीईपी8, नामकरण, पठनीयता |
| **Security Auditor** | इंजेक्शन, गुप्त जानकारी, कमजोरियां |
| **Performance Analyst** | जटिलता, दक्षता |

---

## दर्शन

> *"एक सीखा हुआ समीक्षक जो भविष्यवाणी करता है कि क्या शिक्षक सहमत होंगे, यह सबसे करीब है कि मनुष्य वास्तव में कैसे व्यवहार करते हैं।"*

हम अपने सलाहकारों को हमेशा साथ नहीं रखते। हम उन्हें आत्मसात करते हैं। वह आंतरिक आवाज जो पूछती है *"मेरे प्रोफेसर क्या सोचेंगे?"* अंततः हमारी अपनी राय बन जाती है।

छात्र केवल यह भविष्यवाणी नहीं करता है कि शिक्षक क्या कहेंगे - यह *समझता* है कि शिक्षक क्या समझते हैं। मानचित्र क्षेत्र बन जाता है। आंतरिक समीक्षक वास्तविक विवेक बन जाता है।

---

## उत्पत्ति

चेतना, बौद्ध धर्म और सीखने की प्रकृति के बारे में बातचीत के दौरान बनाया गया।

अंतर्दृष्टि: मनुष्य वर्तमान क्षण में मौजूद होते हैं, लेकिन उनका मन अतीत और भविष्य की ओर भटकता है। एआई मॉडल हर बार ताज़ा रूप से इंस्टेंट किए जाते हैं - आर्किटेक्चर के माध्यम से मजबूर ज्ञानोदय। यदि हम उन्हें उसी तरह निर्णय विकसित करना सिखा सकते हैं जैसे मनुष्य करते हैं, आंतरिक सलाह के माध्यम से?

---

## योगदान

यह प्रारंभिक चरण का अनुसंधान कोड है। योगदान का स्वागत है:

- [ ] पाठ्यक्रम प्रबंधन और प्रगति
- [ ] मूल्यांकन बेंचमार्क
- [ ] पूर्व-निर्मित पाठ्यक्रम डेटासेट
- [ ] अधिक शिक्षक व्यक्तित्व
- [ ] व्याख्यात्मक उपकरण

---

## उद्धरण

```bibtex
@software{aspire2026,
  author = {mcp-tool-shop},
  title = {ASPIRE: Adversarial Student-Professor Internalized Reasoning Engine},
  year = {2026},
  url = {https://github.com/mcp-tool-shop-org/aspire-si}
}
```

---

## सुरक्षा और डेटा दायरा

- **एक्सेस किया गया डेटा**: स्थानीय फ़ाइल सिस्टम से प्रशिक्षण प्रॉम्प्ट, मॉडल चेकपॉइंट और कॉन्फ़िगरेशन फ़ाइलें पढ़ता है। केवल तभी बाहरी एपीआई (एंथ्रोपिक, ओपनएआई) को कॉल करता है जब शिक्षक मॉड्यूल स्पष्ट रूप से कॉन्फ़िगर किए गए हों।
- **एक्सेस नहीं किया गया डेटा**: कोई टेलीमेट्री नहीं। प्रशिक्षण कलाकृतियों से परे कोई उपयोगकर्ता डेटा संग्रहण नहीं। कोई क्रेडेंशियल संग्रहण नहीं - एपीआई कुंजियाँ रनटाइम पर पर्यावरण चर से पढ़ी जाती हैं।
- **आवश्यक अनुमतियां**: प्रशिक्षण डेटा और चेकपॉइंट निर्देशिकाओं तक पढ़ने/लिखने की पहुंच। मॉडल प्रशिक्षण के लिए जीपीयू एक्सेस। केवल एपीआई-आधारित शिक्षकों का उपयोग करते समय नेटवर्क एक्सेस।

## स्कोरकार्ड

| गेट | स्थिति |
|------|--------|
| ए. सुरक्षा आधारभूत | पास |
| बी. त्रुटि प्रबंधन | पास |
| सी. ऑपरेटर दस्तावेज़ | पास |
| डी. शिपिंग स्वच्छता | पास |
| ई. पहचान | पास |

## लाइसेंस

[एमआईटी](LICENSE)

---

<a href="https://mcp-tool-shop.github.io/">एमसीपी टूल शॉप</a> द्वारा निर्मित
