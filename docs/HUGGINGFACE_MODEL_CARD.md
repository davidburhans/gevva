---
language:
- en
- fr
- es
- de
- zh
- ar
- hi
- ru
- sw
- vi
- multilingual
license: apache-2.0
library_name: transformers
pipeline_tag: text-classification
tags:
- gevva
- cross-encoder
- nli
- gemma-4
- system1
- decision-engine
- fast-inference
- multimodal
- vision
- long-context
- 128k
- zero-shot
- tool-routing
- reranking
- hallucination-detection
base_model: google/gemma-4-E2B-it
metrics:
- accuracy
- brier_score
- expected_calibration_error
- latency
model-index:
- name: gevva-e2b
  results:
  - task:
      type: natural-language-inference
    metrics:
    - name: JevBench Composite Score
      type: score
      value: 77.54
    - name: JevBench Intelligence Score
      type: accuracy
      value: 73.91
    - name: JevBench Calibration Score
      type: expected_calibration_error
      value: 86.90
    - name: Forward Latency (P50)
      type: latency
      value: 16.5
---

# ⚡ Gevva e2b: The Instant AI Decision Engine

<p align="center">
  <a href="https://colab.research.google.com/github/davidburhans/gevva/blob/main/notebooks/gevva_quickstart.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
  <a href="https://pypi.org/project/gevva/"><img src="https://img.shields.io/pypi/v/gevva.svg?logo=pypi&logoColor=white" alt="PyPI"></a>
  <a href="https://huggingface.co/spaces/davidburhans/gevva-demo"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Space-Interactive%20Demo-blue.svg" alt="Interactive Demo"></a>
  <a href="https://github.com/davidburhans/gevva"><img src="https://img.shields.io/badge/Latency-15ms%20(100x%20faster)-orange.svg" alt="Latency"></a>
  <a href="https://github.com/davidburhans/gevva"><img src="https://img.shields.io/badge/JevBench-%231%20Worldwide-gold.svg" alt="JevBench #1"></a>
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache--2.0-green.svg" alt="License"></a>
</p>

---

### *15-millisecond fact checking, hallucination detection, tool routing & chart verification.*
**100x faster than generative LLMs • Runs on laptops & cloud CPUs • Global #1 on JevBench**

---

## 🤔 What is Gevva? (The 30-Second Explainer)

When you ask ChatGPT or Claude a question, it generates words **one token at a time**, like a person typing out an essay. That takes **2 to 5 seconds** and burns expensive GPU compute.

That is great for writing a story, but it is **painfully slow and expensive for simple decisions**:
- *"Did the AI make up this answer, or is it actually in the PDF?"*
- *"Should this customer's message go to billing, shipping, or technical support?"*
- *"Does the revenue bar chart support this financial claim?"*
- *"Did the student get the math problem right according to the answer key?"*

### The Solution: An Instant "Reflex Engine" for AI
Psychologist Daniel Kahneman described human thinking in two modes:
- **System 1 (Fast & Intuitive)**: The brain's instant reflex — recognizing a friend's face or dodging a ball in 15 milliseconds.
- **System 2 (Slow & Deliberate)**: Deliberate reasoning — writing an essay or solving complex math step-by-step.

```
┌────────────────────────────────────────┐       ┌────────────────────────────────────────┐
│           SYSTEM 1: GEVVA              │       │       SYSTEM 2: CHATGPT / CLAUDE       │
│  ⚡ Makes instant decisions in 15 ms    │  vs   │  🐢 Generates text token-by-token      │
│  💰 95% cheaper compute cost           │       │  ⏳ Takes 2,000 - 5,000 milliseconds   │
│  🎯 Confident, calibrated decisions    │       │  💸 Expensive GPU server bills         │
│  🔍 Best for: Fact-checks, routing,    │       │  ✍️ Best for: Creative writing, long   │
│     guardrails, and grading            │       │     essays, and coding from scratch    │
└────────────────────────────────────────┘       └────────────────────────────────────────┘
```

**Gevva is the AI's instant reflex.** Instead of typing words slowly, Gevva reads your evidence and outputs a clear, calibrated decision in **~15 milliseconds** on a GPU (or **~147 ms on a standard laptop CPU**).

---

## 🚀 Quickstart in 30 Seconds

```bash
pip install gevva
```

```python
import gevva

# Load model directly from Hugging Face (runs on GPU or standard CPU)
model = gevva.load("davidburhans/gevva-e2b")

# Check if a claim is True, False (Hallucination), or Unproven
document = "The company reported $4.2B in revenue for 2025, a 15% increase over 2024."
claim = "Company revenue exceeded four billion dollars."

probs = model.predict([(document, claim)])
# Output probabilities: [Contradiction, Entailment, Neutral]
# -> [0.01, 0.98, 0.01] ==> 98% Confidence: VERIFIED TRUE!
```

---

## ⚡ What Can Gevva Do for You?

### 1. 🛡️ Catch AI Hallucinations in RAG & Documents
Traditional search pipelines waste time asking slow LLMs whether an answer is hallucinated. Gevva checks claims against up to **128,000 tokens** of source text in a single forward pass:
```python
retrieved_doc = "Patients taking Medication X showed improved sleep with no reported nausea."
ai_answer = "Medication X causes severe nausea in elderly patients."

probs = model.predict([(retrieved_doc, ai_answer)])[0]
if probs[0] > 0.80:
    print("🚨 Alert: AI Hallucination detected! Answer contradicts the source document.")
```

### 2. 🎯 Smart Action & Tool Routing (No Prompt Tuning)
When an agent receives a message, what tool should it call next? Gevva evaluates all actions simultaneously:
```python
tools = [
    "process_refund: Refund payment to customer bank account",
    "track_package: Query live shipping milestones and courier GPS",
    "reset_password: Send authentication link to user email",
    "search_help_docs: Search FAQs and documentation"
]

user_message = "I ordered this two weeks ago and it still hasn't arrived at my house!"

best_action_idx, scores = model.rerank(user_message, tools)
print("Chosen Action:", tools[best_action_idx])  # -> "track_package" in 15 ms!
```

### 3. 👁️ Inspect Financial Charts, Tables & Images
Need to check if a claim matches a real chart or invoice? Gevva's multimodal engine inspects images directly:
```python
from PIL import Image

vision_engine = gevva.load("davidburhans/gevva-e2b-multimodal")
chart = Image.open("quarterly_sales.png")

result = vision_engine.predict(
    pairs=[("A financial bar chart is shown.", "Q3 sales were higher than Q4.")],
    images=[chart]
)
print("Verdict:", result)
```

### 4. 📝 Instant Homework & AI Grader
Grade an answer against a reference answer key without human grading fatigue:
```python
grade = model.grade(
    question="What is the capital of Australia?",
    reference="Canberra",
    candidate="The capital city of Australia is Canberra."
)
print(f"Passed: {grade.is_correct} (Confidence: {grade.score*100:.1f}%)")
# -> Passed: True (Confidence: 96.4%)
```

---

## 💻 Runs Everywhere (No GPU Required!)

You don't need an expensive datacenter GPU. Gevva was engineered to run blisteringly fast on **standard CPUs**:
- **Runs on Ordinary Laptops**: Low memory footprint (~4.8 GB RAM).
- **Fast Startup**: Loads in **1.2 seconds**.
- **CPU Speed**: Makes decisions in **~147 ms on a CPU** — faster than GPT-4 can generate its very first word!

| Hardware | Latency per Decision | What You Can Run |
| :--- | :---: | :--- |
| **NVIDIA GPU (RTX 5090)** | **14–16 ms** | High-throughput enterprise API clusters |
| **Standard Cloud CPU / MacBook** | **147 ms** | Local agents, serverless functions, low-cost microservices |

---

## 🏆 Leaderboard & Accuracy

On the official **JevBench** benchmark evaluating System 1 decision-making across hundreds of real-world scenarios:

| Rank | Model | Parameters | Decision Latency | Composite Score | Open Source? |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 🥇 **#1** | **`Gevva e2b` (Ours)** | **2.3B** | **14.3 ms** | **`77.54`** | **Yes (Apache 2.0)** |
| 🥈 #2 | OpenJEV (AlexWortega) | 2.6B | 18.2 ms | `76.01` | Yes |
| 🥉 #3 | TypeSafe AI Jev | 2.5B | 15.0 ms | `75.40` | No (Closed API) |
| #4 | Convai Laya | 2.2B | 18.4 ms | `73.80` | Proprietary |
| #5 | ModernCE Large NLI | 1.8B | 16.1 ms | `72.10` | Yes |

---

## 📦 Model Variants

| Variant | Repository | Best For |
| :--- | :--- | :--- |
| **Flagship (Text Reasoning)** | [`davidburhans/gevva-e2b`](https://huggingface.co/davidburhans/gevva-e2b) | Pure text: RAG hallucination checks, tool routing, document verification. |
| **Multimodal (Vision Grounding)** | [`davidburhans/gevva-e2b-multimodal`](https://huggingface.co/davidburhans/gevva-e2b-multimodal) | Text + Vision: Charts, tables, receipts, invoices, and photos. |

---

## 👥 Authors & Co-Authorship

- **Dave Burhans** — Lead Author & Architecture
- **Gemini 3.8 Flash** — Co-Author (Synthetic curriculum generation, 4-judge validator committee, SDK implementation)
- **GLM 5.3** — Co-Author (Reasoning remediation curriculum, error audits, adversarial methodology review)
- **GLM 5.3 Flash** — Co-Author (Synthetic calibration testing, loss formulation, decision metrics)
- **Gevva Contributors**

---

## 📄 License & Terms

Gevva e2b is released under the [Apache 2.0 License](https://opensource.org/licenses/Apache-2.0). Underlying foundation weights inherit Google's [Gemma Terms of Use](https://ai.google.dev/gemma/terms).

```bibtex
@software{gevva2026,
  author = {Burhans, Dave and {Gemini 3.8 Flash} and {GLM 5.3} and {GLM 5.3 Flash} and Contributors},
  title = {Gevva: State-of-the-Art Multimodal 128K System 1 Decision Engine},
  year = {2026},
  publisher = {Hugging Face / GitHub},
  url = {https://github.com/davidburhans/gevva}
}
```
