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

## The Idea

**पारंपरिक फाइन-ट्यूनिंग:** *"यहाँ सही उत्तर दिए गए हैं। उन्हें मिलाएं।"*

**एस्पिर (ASPIRE):** *"यहाँ एक बुद्धिमान मन है। इस तरह सोचने का तरीका सीखें।"*

जब आप किसी महान गुरु से सीखते हैं, तो आप केवल उनके उत्तरों को याद नहीं करते। आप उनकी देखने की शैली को आत्मसात करते हैं। उनकी आवाज आपकी आंतरिक बातचीत का हिस्सा बन जाती है। आप यह अनुमान लगाना शुरू कर देते हैं कि वे क्या कहेंगे, और अंततः वह अनुमान आपकी अपनी समझ बन जाता है।

एस्पिर (ASPIRE) एआई को भी वही अनुभव प्रदान करता है।

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

**आलोचक (क्रिटिक)** सीखता है कि शिक्षक क्या सोचेंगे। प्रशिक्षण के बाद, छात्र इस आंतरिक आलोचक का उपयोग करके स्वयं को बेहतर बनाता है - **अनुमान लगाने के समय किसी शिक्षक की आवश्यकता नहीं होती है।**

---

## शुरुआत कैसे करें

### स्थापना

```bash
git clone https://github.com/mcp-tool-shop-org/aspire-si.git
cd aspire-si
pip install -e .
```

### अपना एपीआई कुंजी सेट करें

```bash
# Windows
set ANTHROPIC_API_KEY=your-key-here

# Linux/Mac
export ANTHROPIC_API_KEY=your-key-here
```

### स्थापना की पुष्टि करें

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

## शिक्षक व्यक्तित्व (टीचर पर्सोना)

विभिन्न शिक्षक अलग-अलग तरह के विचार पैदा करते हैं। बुद्धिमानी से चुनें।

| व्यक्तित्व | दर्शन | परिणाम |
|---------|------------|----------|
| 🏛️ **सुकराती (Socratic)** | *"आप कौन सी धारणा बना रहे हैं?"* | गहरी तर्कशक्ति, बौद्धिक स्वतंत्रता |
| 🔬 **वैज्ञानिक (Scientific)** | *"आपका प्रमाण क्या है?"* | तकनीकी सटीकता, कठोर सोच |
| 🎨 **रचनात्मक (Creative)** | *"अगर हम विपरीत करने की कोशिश करें तो क्या होगा?"* | नवाचार, रचनात्मक सोच |
| ⚔️ **विरोधाभासी (Adversarial)** | *"मैं असहमत हूं। अपने दृष्टिकोण का बचाव करें।"* | मजबूत तर्क, दृढ़ विश्वास |
| 💚 **सहानुभूतिपूर्ण (Compassionate)** | *"इस बारे में किसी को कैसा लग सकता है?"* | नैतिक तर्क, ज्ञान |

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

छात्र एक प्रतिक्रिया उत्पन्न करता है। शिक्षक उसे चुनौती देता है। आगे-पीछे, कमजोरियों की जांच, स्पष्टता की मांग, गहराई में जाने का प्रयास।

```
Student: "Recursion works by calling itself."

Teacher (Socratic): "But what prevents infinite regress?
                     What's the mechanism that grounds the recursion?"

Student: "The base case stops it when..."

Teacher: "You say 'stops it' — but how does the computer know
          to check the base case before recursing?"
```

### 2. आलोचक प्रशिक्षण

आलोचक सीखता है कि शिक्षक का निर्णय क्या होगा - न केवल स्कोर, बल्कि *तर्क*।

```python
critic_loss = predict_teacher_judgment(
    score=True,      # "This deserves a 7/10"
    reasoning=True,  # "Because the explanation lacks depth on X"
)
```

### 3. छात्र प्रशिक्षण

छात्र आलोचक के आंतरिक निर्णय से सीखता है, और उस दिशा में आगे बढ़ता है जिसे शिक्षक स्वीकार करेंगे।

```python
student_loss = (
    reward_from_critic +      # Higher score = better
    contrastive_to_teacher +  # Pull toward teacher's improved version
    trajectory_improvement    # Get better across dialogue turns
)
```

### 4. जादू जैसा अनुमान

प्रशिक्षण के बाद, छात्र आंतरिक आलोचक का उपयोग करके स्वयं को बेहतर बनाता है। **अनुमान लगाने के समय किसी शिक्षक एपीआई कॉल की आवश्यकता नहीं होती है।**

```python
def generate_with_judgment(prompt):
    response = student.generate(prompt)

    while critic.score(response) < threshold:
        response = student.refine(response, critic.feedback)

    return response  # Self-improved through internalized judgment
```

---

## सीएलआई संदर्भ

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

## परियोजना संरचना

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

## आवश्यकताएं

- Python 3.10+
- PyTorch 2.0+
- CUDA GPU (16GB+ VRAM अनुशंसित)
- एंथ्रोपिक एपीआई कुंजी (क्लाउड शिक्षक के लिए) या ओपनएआई एपीआई कुंजी

### विंडोज संगतता

एस्पिर (ASPIRE) पूरी तरह से विंडोज-संगत है और RTX 5080/ब्लैकवेल का समर्थन करता है:
- `dataloader_num_workers=0`
- `XFORMERS_DISABLED=1`
- `freeze_support()` के साथ उचित मल्टीप्रोसेसिंग

---

## एकीकरण

### 🖼️ स्टेबल डिफ्यूजन वेबयूआई फोर्ज

एस्पिर (ASPIRE) छवि निर्माण तक भी विस्तारित होता है! सौंदर्य संबंधी निर्णय विकसित करने के लिए स्टेबल डिफ्यूजन मॉडल को प्रशिक्षित करें।

```
integrations/forge/
├── scripts/
│   ├── aspire_generate.py   # Critic-guided generation
│   └── aspire_train.py      # Training interface
├── vision_teacher.py        # Claude Vision / GPT-4V teachers
├── image_critic.py          # CLIP and latent-space critics
└── README.md
```

**विशेषताएं:**
- **विज़न शिक्षक (Vision Teachers)**: क्लाउड विज़न, जीपीटी-4वी आपकी उत्पन्न छवियों की आलोचना करते हैं।
- **इमेज क्रिटिक्स (Image Critics)**: वास्तविक समय में मार्गदर्शन के लिए सीएलआईपी-आधारित और लेटेंट-स्पेस क्रिटिक्स।
- **प्रशिक्षण यूआई (Training UI)**: लाइव पूर्वावलोकन और पहले/बाद की तुलना के साथ लोरा एडेप्टर को प्रशिक्षित करें।
- **अनुमान पर एपीआई नहीं (No API at inference)**: प्रशिक्षित आलोचक स्थानीय रूप से पीढ़ी का मार्गदर्शन करता है।

**स्थापना:**
```bash
# Copy to your Forge extensions
cp -r integrations/forge /path/to/sd-webui-forge/extensions-builtin/sd_forge_aspire
```

| विज़न शिक्षक | ध्यान केंद्रित करें |
|----------------|-------|
| **Balanced Critic** | उचित तकनीकी और कलात्मक मूल्यांकन |
| **Technical Analyst** | गुणवत्ता, कलाकृतियाँ, स्पष्टता |
| **Artistic Visionary** | रचनात्मकता और भावनात्मक प्रभाव |
| **Composition Expert** | संतुलन, फोकस बिंदु, दृश्य प्रवाह |
| **Harsh Critic** | बहुत उच्च मानक। |

### 🤖 आइज़ैक जिम / आइज़ैक लैब (रोबोटिक्स)

एएसपायर, एम्बोडीड एआई तक विस्तारित है! रोबोटों को भौतिक समझ विकसित करना सिखाएं।

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
- **मोशन टीचर्स:** सेफ्टी इंस्पेक्टर, एफिशिएंसी एक्सपर्ट, ग्रेस कोच, फिजिक्स ओरेकल
- **ट्रैजेक्टरी क्रिटिक्स:** मोशन मूल्यांकन के लिए ट्रांसफॉर्मर, एलएसटीएम, टीसीएन आर्किटेक्चर
- **जीपीयू-एक्सीलरेटेड:** आइज़ैक जिम के साथ 512+ समानांतर वातावरण
- **सेल्फ-रिफाइनमेंट:** रोबोट निष्पादन से पहले अपनी गति का मूल्यांकन करता है।

**शुरुआत कैसे करें:**
```python
from aspire.integrations.isaac import AspireIsaacTrainer, MotionTeacher

teacher = MotionTeacher(
    personas=["safety_inspector", "efficiency_expert", "grace_coach"],
    strategy="vote",
)

trainer = AspireIsaacTrainer(env="FrankaCubeStack-v0", teacher=teacher)
trainer.train(epochs=100)
```

| मोशन टीचर | ध्यान केंद्रित करें |
|----------------|-------|
| **Safety Inspector** | टकराव, जोड़ सीमाएं, बल सीमाएं |
| **Efficiency Expert** | ऊर्जा, समय, पथ की लंबाई |
| **Grace Coach** | सुगमता, स्वाभाविकता, जर्क न्यूनीकरण |
| **Physics Oracle** | सिमुलेटर से वास्तविक डेटा |

### 💻 कोड असिस्टेंट

एएसपायर, कोड जेनरेशन तक विस्तारित है! कोड मॉडल को आउटपुट करने से पहले स्वयं की समीक्षा करने के लिए सिखाएं।

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
- **कोड टीचर्स:** करेक्टनेस चेकर, स्टाइल गाइड, सिक्योरिटी ऑडिटर, आर्किटेक्चर रिव्यूवर
- **स्टैटिक एनालिसिस:** रफ, माईपी, बैंडिट के साथ एकीकृत
- **कोड क्रिटिक:** कोडबर्ट-आधारित मॉडल, जो गुणवत्ता स्कोर की भविष्यवाणी करना सीखता है।
- **गिटहब कलेक्शन:** गुणवत्ता वाले रिपॉजिटरी से स्वचालित रूप से प्रशिक्षण डेटा एकत्र करें।

**शुरुआत कैसे करें:**
```python
from aspire.integrations.code import CodeTeacher, CodeSample

teacher = CodeTeacher(
    personas=["correctness_checker", "style_guide", "security_auditor"],
    strategy="vote",
)

critique = teacher.critique(CodeSample(code="def f(): eval(input())", language="python"))
print(f"Score: {critique.overall_score}/10")  # Low score - security issue!
```

| कोड टीचर | ध्यान केंद्रित करें |
|--------------|-------|
| **Correctness Checker** | बग, प्रकार, लॉजिक त्रुटियां |
| **Style Guide** | पीईपी8, नामकरण, पठनीयता |
| **Security Auditor** | इंजेक्शन, सीक्रेट, कमजोरियां |
| **Performance Analyst** | जटिलता, दक्षता |

---

## दर्शन

> *"एक सीखा हुआ आलोचक जो यह भविष्यवाणी करता है कि क्या शिक्षक अनुमोदन करेगा, यह इस बात के सबसे करीब है कि मनुष्य वास्तव में कैसे व्यवहार करते हैं।"*

हम अपने गुरुओं को हमेशा अपने साथ नहीं रखते। हम उन्हें आत्मसात कर लेते हैं। वह आंतरिक आवाज जो पूछता है *"मेरे प्रोफेसर क्या सोचेंगे?"*, अंततः हमारी अपनी समझ बन जाती है।

छात्र केवल यह नहीं भविष्यवाणी करता कि शिक्षक क्या कहेंगे - बल्कि वह *समझता* है कि शिक्षक क्या समझता है। मानचित्र क्षेत्र बन जाता है। आत्मसात किया गया आलोचक वास्तविक समझ बन जाता है।

---

## उत्पत्ति

यह बातचीत चेतना, बौद्ध धर्म और सीखने की प्रकृति के बारे में हुई थी।

अंतर्दृष्टि: मनुष्य वर्तमान क्षण में मौजूद होते हैं, लेकिन हमारे दिमाग अतीत और भविष्य में भटकते हैं। एआई मॉडल हर बार नए बनाए जाते हैं - आर्किटेक्चर के माध्यम से मजबूर ज्ञानोदय। क्या होगा यदि हम उन्हें मनुष्यों की तरह, आंतरिक मार्गदर्शन के माध्यम से निर्णय विकसित करना सिखा सकें?

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

- **पहुंच किए गए डेटा:** स्थानीय फ़ाइल सिस्टम से प्रशिक्षण संकेत, मॉडल चेकपॉइंट और कॉन्फ़िगरेशन फ़ाइलें पढ़ता है। केवल तभी बाहरी एपीआई (एंथ्रोपिक, ओपनएआई) को कॉल करता है जब शिक्षक मॉड्यूल स्पष्ट रूप से कॉन्फ़िगर किए गए हों।
- **डेटा जो एक्सेस नहीं किया जाता:** कोई टेलीमेट्री नहीं। प्रशिक्षण कलाकृतियों से परे कोई उपयोगकर्ता डेटा संग्रहण नहीं। कोई क्रेडेंशियल संग्रहण नहीं - एपीआई कुंजियाँ रनटाइम पर पर्यावरण चर से पढ़ी जाती हैं।
- **आवश्यक अनुमतियाँ:** प्रशिक्षण डेटा और चेकपॉइंट निर्देशिकाओं तक पढ़ने/लिखने की पहुंच। मॉडल प्रशिक्षण के लिए जीपीयू एक्सेस। केवल एपीआई-आधारित शिक्षकों का उपयोग करते समय नेटवर्क एक्सेस।

## स्कोरकार्ड

| गेट | स्थिति |
|------|--------|
| ए. सुरक्षा बेसलाइन | पास |
| बी. त्रुटि हैंडलिंग | पास |
| सी. ऑपरेटर डॉक्स | पास |
| डी. शिपिंग स्वच्छता | पास |
| ई. पहचान | पास |

## लाइसेंस

[एमआईटी](LICENSE)

---

द्वारा निर्मित <a href="https://mcp-tool-shop.github.io/">एमसीपी टूल शॉप</a>
