<div align="center">

```
 ██████╗ ███████╗██╗   ██╗██╗   ██╗ █████╗ 
██╔════╝ ██╔════╝██║   ██║██║   ██║██╔══██╗
██║  ███╗█████╗  ██║   ██║██║   ██║███████║
██║   ██║██╔══╝  ╚██╗ ██╔╝╚██╗ ██╔╝██╔══██║
╚██████╔╝███████╗ ╚████╔╝  ╚████╔╝ ██║  ██║
 ╚═════╝ ╚══════╝  ╚═══╝    ╚═══╝  ╚═╝  ╚═╝
```

# Gevva: The Instant AI Decision Engine

### *15-millisecond fact checking, hallucination detection, tool routing & chart verification.*
**100x faster than ChatGPT • Runs on standard laptops & CPUs • High Accuracy on JevBench Public**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/davidburhans/gevva/blob/main/notebooks/gevva_quickstart.ipynb)
[![PyPI version](https://img.shields.io/pypi/v/gevva.svg?style=for-the-badge&logo=pypi&logoColor=white)](https://pypi.org/project/gevva/)
[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Live%20Demo-Hugging%20Face%20Space-blue.svg?style=for-the-badge)](https://huggingface.co/spaces/davidburhans/gevva-demo)
[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Models-Hugging%20Face%20Hub-yellow.svg?style=for-the-badge)](https://huggingface.co/davidburhans/gevva-e2b)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg?style=for-the-badge)](LICENSE)

[Try the Live Web Demo](https://huggingface.co/spaces/davidburhans/gevva-demo) • [Run in Colab (Free GPU)](https://colab.research.google.com/github/davidburhans/gevva/blob/main/notebooks/gevva_quickstart.ipynb) • [Quickstart](#-quickstart-in-30-seconds) • [Superpowers](#-what-can-gevva-do-for-you) • [No GPU Needed (CPU)](#-runs-everywhere-no-gpu-required) • [Leaderboard](#-leaderboard--accuracy) • [Fine-Tuning](#-train-on-your-own-data-in-1-line)

---

</div>

## 🤔 What is Gevva? (The 30-Second Explainer)

When you ask ChatGPT or Claude a question, it generates words **one token at a time**, like a person typing out an essay. That takes **2 to 5 seconds** and burns expensive GPU compute.

That is great for writing a story, but it is **painfully slow and expensive for simple decisions**:
- *"Did the AI make up this answer, or is it actually in the PDF?"*
- *"Should this customer's message go to billing, shipping, or technical support?"*
- *"Does the revenue bar chart support this financial claim?"*
- *"Did the student get the math problem right according to the answer key?"*

### The Solution: An Instant "Reflex Engine" for AI
Psychologist Daniel Kahneman famously described human thinking in two modes:
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

### 1. Install via pip
```bash
pip install gevva
```

### 2. Test a fact in 3 lines of Python
```python
from gevva import load

# Load the model (runs on GPU or standard CPU automatically)
engine = load("davidburhans/gevva-e2b")

# Check if a claim is True, False (Hallucination), or Unproven
document = "The company reported $4.2B in revenue for 2025, a 15% increase over 2024."
claim = "Company revenue exceeded four billion dollars."

probs = engine.predict([(document, claim)])
# Output probabilities: [Contradiction, Entailment, Neutral]
# -> [0.01, 0.98, 0.01] ==> 98% Confidence: VERIFIED TRUE!
```

Want to test it without writing code? Open the **[Live Interactive Demo](https://huggingface.co/spaces/davidburhans/gevva-demo)** in your browser!

---

## ⚡ What Can Gevva Do for You?

### 1. 🛡️ Catch AI Hallucinations in RAG & Documents
When building Retrieval-Augmented Generation (RAG) search engines, LLMs frequently invent fake facts that aren't in your documents. 

Use Gevva as an instant safety filter:
```python
retrieved_document = "Patients taking Medication X showed improved sleep duration with no reported nausea."
ai_answer = "Medication X causes severe nausea in elderly patients."

# Gevva checks if the answer matches the source in 15 ms:
probs = engine.predict([(retrieved_document, ai_answer)])[0]
if probs[0] > 0.80:
    print("🚨 Alert: AI Hallucination detected! Answer contradicts the source document.")
```

### 2. 🎯 Smart Action & Tool Routing (No Prompt Tuning)
When an agent receives a message, what tool should it call next? Instead of writing a long prompt, Gevva evaluates all available actions simultaneously:
```python
available_tools = [
    "process_refund: Refund payment to customer bank account",
    "track_package: Query live shipping milestones and courier GPS",
    "reset_password: Send authentication link to user email",
    "search_help_docs: Search FAQs and documentation"
]

user_message = "I ordered this two weeks ago and it still hasn't arrived at my house!"

best_action_idx, scores = engine.rerank(user_message, available_tools)
print("Chosen Action:", available_tools[best_action_idx])
# -> "track_package" (92% match) in 15 ms!
```

### 3. 👁️ Inspect Financial Charts, Tables & Images
Need to check if a claim matches a real chart or invoice? Gevva's multimodal engine inspects images directly:
```python
from PIL import Image

vision_engine = load("davidburhans/gevva-e2b-multimodal")
chart = Image.open("quarterly_sales.png")

# Check visual evidence against a factual statement:
result = vision_engine.predict(
    pairs=[("A financial bar chart is shown.", "Q3 sales were higher than Q4.")],
    images=[chart]
)
print("Verdict:", result)
```

### 4. 📝 Instant Homework & AI Grader
Grade an answer against a reference answer key without human grading fatigue:
```python
grade = engine.grade(
    question="What is the capital of Australia?",
    reference="Canberra",
    candidate="The capital city of Australia is Canberra."
)
print(f"Passed: {grade.is_correct} (Confidence: {grade.score*100:.1f}%)")
# -> Passed: True (Confidence: 96.4%)
```

---

## 💻 Runs Everywhere (No GPU Required!)

You don't need a multi-thousand-dollar GPU server. Gevva was engineered to run blisteringly fast on **standard CPUs**:

- **Runs on Ordinary Laptops**: Comfortable memory footprint (~4.8 GB RAM).
- **Fast Startup**: Loads in **1.2 seconds**.
- **CPU Speed**: Makes decisions in **~147 ms on a CPU** — faster than GPT-4 can generate its very first word!

| Hardware | Latency per Decision | What You Can Run |
| :--- | :---: | :--- |
| **NVIDIA GPU (RTX 5090)** | **14–16 ms** | High-throughput enterprise API clusters |
| **Standard Cloud CPU / MacBook** | **147 ms** | Local agents, serverless functions, low-cost microservices |

---

## 📊 JevBench Public Benchmark Results

Evaluated against the open **JevBench Public Dataset** (231 evaluation tasks across 12 challenge families):

| Model | Parameters | Decision Latency | Public Composite Score | Hard Accuracy | Open Weights? |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`Gevva e2b`** | **2.3B** | **14.3 ms** | **`77.54`** | 47.8% | **Yes (Apache 2.0)** |
| **`Gevva e4b`** | **4.5B** | **17.8 ms** | **`77.28`** | **55.0%** | **Yes (Apache 2.0)** |
| OpenJEV (AlexWortega) | 2.6B | 18.2 ms | `76.01` | ~45% | Yes |
| TypeSafe AI Jev | 2.5B | 15.0 ms | `75.40` | ~48% | No (Closed API) |
| Convai Laya | 2.2B | 18.4 ms | `73.80` | ~42% | Proprietary |

> **Note on Benchmarking**: These results are measured locally against the 231-item open public split of JevBench. We do not claim official standing on the full private benchmark suite until independent verification is completed by the benchmark maintainers.

*For deep academic benchmarking protocols, McNemar statistical tests, and error audits, see [EVALUATION_PROTOCOL.md](docs/EVALUATION_PROTOCOL.md).*

---

## 📦 Which Model Should I Choose?

| Model | Hugging Face ID | Best For |
| :--- | :--- | :--- |
| **Gevva e2b** | [`davidburhans/gevva-e2b`](https://huggingface.co/davidburhans/gevva-e2b) | Fast pure text: RAG hallucination checks, customer support routing, document verification. |
| **Gevva e2b Multimodal** | [`davidburhans/gevva-e2b-multimodal`](https://huggingface.co/davidburhans/gevva-e2b-multimodal) | Text + Vision: Verifying financial charts, tables, receipts, invoices, and photos. |
| **Gevva e4b** | [`davidburhans/gevva-e4b`](https://huggingface.co/davidburhans/gevva-e4b) | Deep reasoning: Complex policies, multi-hop evidence, high-stakes verification. |

---

## 🛠️ Train on Your Own Data in 1 Line

Want Gevva to learn your proprietary company tools, custom policies, or domain-specific language? 

Fine-tune with **zero boilerplate**:
```bash
# Automatically detects your column names and trains on your data:
gevva finetune --data my_company_data.jsonl --out-dir ./my_custom_model
```

Accepts `.jsonl`, `.csv`, `.tsv`, or `.parquet`. Gevva automatically handles column detection, label normalization, and train/validation splits. See the [Custom Fine-Tuning Guide](docs/CUSTOM_FINETUNING_GUIDE.md) for step-by-step recipes.

---

## 👥 Authors & Co-Authorship

- **Dave Burhans** — Lead Author & Architecture
- **Gemini 3.8 Flash** — Co-Author (Synthetic curriculum generation, 4-judge validator committee, SDK implementation)
- **GLM 5.3** — Co-Author (Reasoning remediation curriculum, error audits, adversarial methodology review)
- **GLM 5.3 Flash** — Co-Author (Synthetic calibration testing, loss formulation, decision metrics)
- **Gevva Contributors**

---

## 📄 License & Citation

Gevva is licensed under the open-source [Apache 2.0 License](LICENSE). Base model weights inherit the [Google Gemma Terms of Use](https://ai.google.dev/gemma/terms).

```bibtex
@software{gevva2026,
  author = {Burhans, Dave and {Gemini 3.8 Flash} and {GLM 5.3} and {GLM 5.3 Flash} and Contributors},
  title = {Gevva: State-of-the-Art Multimodal 128K System 1 Decision Engine},
  year = {2026},
  publisher = {Hugging Face / GitHub},
  url = {https://github.com/davidburhans/gevva}
}
```

<div align="center">
  <sub>Built with ❤️ by the Gevva Community. Fast decisions for smart agents.</sub>
</div>
