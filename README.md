<div align="center">

```
 ██████╗ ███████╗██╗   ██╗██╗   ██╗ █████╗ 
██╔════╝ ██╔════╝██║   ██║██║   ██║██╔══██╗
██║  ███╗█████╗  ██║   ██║██║   ██║███████║
██║   ██║██╔══╝  ╚██╗ ██╔╝╚██╗ ██╔╝██╔══██║
╚██████╔╝███████╗ ╚████╔╝  ╚████╔╝ ██║  ██║
 ╚═════╝ ╚══════╝  ╚═══╝    ╚═══╝  ╚═╝  ╚═╝
```

# Gevva: Multimodal System 1 Decision Engine & NLI Cross-Encoder

### *Fast fact-checking, hallucination detection, tool routing & document verification based on Google Gemma 4.*
**Single-pass sequence classification • Open weights (Apache 2.0) • Evaluated on ANLI, MNLI, HANS & JevBench Public**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/davidburhans/gevva/blob/main/notebooks/gevva_quickstart.ipynb)
[![PyPI version](https://img.shields.io/pypi/v/gevva.svg?style=for-the-badge&logo=pypi&logoColor=white)](https://pypi.org/project/gevva/)
[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Live%20Demo-Hugging%20Face%20Space-blue.svg?style=for-the-badge)](https://huggingface.co/spaces/davidburhans/gevva-demo)
[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Models-Hugging%20Face%20Hub-yellow.svg?style=for-the-badge)](https://huggingface.co/davidburhans/gevva-e2b)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg?style=for-the-badge)](LICENSE)

[Live Web Demo](https://huggingface.co/spaces/davidburhans/gevva-demo) • [Quickstart](#-quickstart-in-30-seconds) • [Superpowers](#-what-can-gevva-do-for-you) • [Hardware & Latency Profile](#-hardware-latency--memory-profile) • [Benchmarks & Empirical Accuracy](#-benchmarks--empirical-accuracy) • [Known Limitations](#-known-limitations--failure-modes) • [Fine-Tuning](#-train-on-your-own-data-in-1-line)

---

</div>

## 🤔 What is Gevva?

When you ask an autoregressive LLM (like ChatGPT or Claude) to verify a fact, it generates text **token-by-token**, typically taking **1 to 3 seconds** for an answer.

For discrete decision tasks, autoregressive generation is unnecessarily slow and expensive:
- *"Does this retrieved context support the generated claim, or is it a hallucination?"*
- *"Which of these 4 tools should be called for this user request?"*
- *"Did the student's answer match the reference solution?"*

### The System 1 Approach
Psychologist Daniel Kahneman described human thought in two modes: **System 1** (fast, automatic reflex) and **System 2** (slow, deliberate reasoning). 

**Gevva** is designed as a **System 1 decision engine**: a sequence classification cross-encoder built on Google's lightweight Gemma 4 foundation. Rather than generating text, it evaluates premise-hypothesis pairs in a single forward pass and outputs calibrated 3-class probability distributions:

$$\text{Class} \in \{\text{Contradiction (0)}, \text{Entailment (1)}, \text{Neutral (2)}\}$$

---

## 🚀 Quickstart in 30 Seconds

### 1. Install via pip
```bash
pip install gevva
```

### 2. Verify evidence in Python
```python
from gevva import load

# Load model (automatically uses CUDA, MPS, or CPU)
engine = load("davidburhans/gevva-e2b")

# Check if evidence entails or contradicts a statement
premise = "The Eiffel Tower is a wrought-iron lattice tower located on the Champ de Mars in Paris, France."
claim = "The Eiffel Tower is located in Paris."

probs = engine.predict([(premise, claim)])[0]
print(f"Entailment: {probs[1]*100:.1f}%, Contradiction: {probs[0]*100:.1f}%, Neutral: {probs[2]*100:.1f}%")
# -> Entailment: 96.1%, Contradiction: 0.9%, Neutral: 2.9%
```

---

## ⚡ What Can Gevva Do for You?

### 1. 🛡️ Catch AI Hallucinations in RAG Pipelines
In Retrieval-Augmented Generation (RAG), LLMs frequently introduce unsupported or contradictory claims. Use Gevva as a fast guardrail:

```python
retrieved_document = "Patients taking Medication X showed improved sleep duration with no reported nausea."
ai_answer = "Medication X causes severe nausea in elderly patients."

probs = engine.predict([(retrieved_document, ai_answer)])[0]
# Returns [Contradiction: 75.2%, Entailment: 3.3%, Neutral: 21.5%]
if probs[0] > 0.70:
    print("🚨 Alert: AI Hallucination detected! Claim contradicts source document.")
```

### 2. 🎯 Tool & Intent Routing
Evaluate multiple candidate tools simultaneously without prompt-tuning:

```python
available_tools = [
    "process_refund: Refund payment to customer bank account",
    "track_package: Query live shipping milestones and courier GPS",
    "reset_password: Send authentication link to user email",
    "search_help_docs: Search FAQs and documentation"
]

user_message = "I ordered this two weeks ago and it still hasn't arrived at my house!"

best_idx, scores = engine.rerank(user_message, available_tools)
print("Chosen Action:", available_tools[best_idx])
# -> "track_package: Query live shipping milestones and courier GPS"
```

### 3. 📝 Instant Answer & Rubric Grading
Grade student or agent responses against reference keys:

```python
grade = engine.grade(
    question="What is the capital of Australia?",
    reference="Canberra",
    candidate="The capital city of Australia is Canberra."
)
print(f"Passed: {grade.is_correct} (Confidence: {grade.score*100:.1f}%)")
# -> Passed: True (Confidence: 81.8%)
```

---

## 💻 Hardware, Latency & Memory Profile

Gevva models are built on Google's multimodal `gemma-4-E2B-it` and `gemma-4-E4B-it` checkpoints. Here are real measured performance numbers across hardware:

### Model Size & Download Footprint
- **Gevva e2b**: **5.10 Billion total parameters** (10.2 GB download in bfloat16).
  *(Includes 2.3B active text backbone + SigLIP vision encoder + 262K vocabulary embedding table).*
- **Gevva e4b**: **7.94 Billion total parameters** (15.88 GB download in bfloat16).
  *(Includes 4.5B active text backbone + SigLIP vision encoder + 262K vocabulary embedding table).*

### Measured Latency & Memory

| Input Type & Size | Hardware | Latency | Measured Memory | Notes |
| :--- | :--- | :---: | :---: | :--- |
| **Short sentence pair (<128 tokens)** | NVIDIA GeForce RTX 5090 | **14–19 ms** | 9.56 GB allocated VRAM (e2b) / 15.88 GB (e4b) | High-throughput GPU serving |
| **Short sentence pair (<128 tokens)** | Apple M4 Pro GPU (MPS) | **~74 ms** | ~4.6 GB RAM | Local developer machines (e2b default) |
| **Short sentence pair (<128 tokens)** | Apple M4 Pro CPU (FP32) | **~205 ms** | ~11.9 GB RAM | Full-precision CPU execution (e2b) |
| **Short sentence pair (<128 tokens)** | Apple M4 Pro CPU (BF16 default) | **~720 ms** | ~4.6 GB RAM | CPU emulation of bfloat16 (e2b) |
| **RAG Document Context (~900 tokens)** | Apple M4 Pro GPU (MPS) | **~0.84 s** | — | Realistic RAG passage check |
| **RAG Document Context (~900 tokens)** | Apple M4 Pro CPU | **~20 s** | — | Heavy for CPU; GPU recommended for long docs |

> **Context Length Note**: Gemma 4 natively supports up to 128K context via Rotary Position Embeddings (RoPE). However, the fine-tuning curriculum for Gevva was trained on sequences up to **2,048 tokens**. For optimal accuracy and latency in RAG pipelines, chunking retrieved evidence to ~1,000–2,000 tokens is strongly recommended.

---

## 📊 Benchmarks & Empirical Accuracy

### 1. Independent Evaluation on Unseen Public Test Sets
Evaluated on 250 items per test set against a standard fact-checking baseline (`nli-deberta-v3-base`, ~184M parameters):

| Test Set | Gevva e2b (5.1B) | DeBERTa-v3-base (184M) | Key Observations |
| :--- | :---: | :---: | :--- |
| **ANLI Round 1** (Adversarial NLI) | **60.0%** | 43.0% | Stronger on tricky multi-hop negations |
| **ANLI Round 2** (Adversarial NLI) | **47.0%** | 38.0% | Handles hard semantic shifts better |
| **ANLI Round 3** (Adversarial NLI) | **50.0%** | 38.0% | Outperforms baseline on complex claims |
| **MNLI Mismatched** (Standard fact-checking) | 88.4% | **89.6%** | Comparable on standard in-domain text |
| **HANS** (Word-overlap shortcut test) | **83.0%** | 80.0% | Overall accuracy |
| ↳ *HANS non-entailed cases only* | **68.0%** | 58.0% | Substantially more resistant to lexical traps |
| **RAG Hallucination Checks (Handwritten)** | **17/20 (85%)** | 14/20 (70%) | Strong general fact verification |
| **Tool / Action Routing (Handwritten)** | **15/18 (83%)** | — | Missed 3 login/auth routing edge cases |
| **Answer Grading (Handwritten)** | **14/14 (100%)** | — | Evaluates correctness reliably |

### 2. JevBench Public Dataset (231 Items)
Evaluated locally against the frozen open public split of JevBench (`jevbench/datasets/public` across 18 task families):

- **Gevva e2b**: **71.43% overall accuracy** (165/231), **47.75% on the Hard tier** (53/111).
  - JevBench Composite Score: **77.54** (at $T=1.6$).
- **Gevva e4b**: **76.62% overall accuracy** (177/231), **54.95% on the Hard tier** (61/111).
  - JevBench Composite Score: **77.28** (at $T=1.6$).

> **Transparency Note on JevBench**:
> - These scores were measured locally on the 231-item open public split and have not yet been evaluated by third-party maintainers on the private benchmark suite.
> - The JevBench composite score weights Intelligence (25%), Calibration (25%), Speed (25%), and a hard-coded Cost tariff (25%).
> - A temperature of $T=1.6$ was fitted specifically to minimize ECE on the 111 JevBench Hard tier items. However, on standard NLI test sets, $T=1.6$ flattens probabilities (raising ECE from 0.056 to 0.163). Therefore, **Gevva ships with standard $T=1.0$ default inference**.

---

## ⚠️ Known Limitations & Failure Modes

Every decision engine has blind spots. We encourage testing on your specific distribution:

1. **Numerical & Arithmetic Claims**:
   Gevva can be unreliable with subtle arithmetic discrepancies. In our audit, the model accepted *"revenue exceeded five billion"* when the document stated *"$4.2B"* (83% confidence). For strict financial and numeric validation, pair Gevva with programmatic regex or numeric validators.
2. **Authentication & Password Routing**:
   In tool-routing evaluations, requests involving account security, passwords, or authentication links were occasionally misrouted to general documentation rather than password-reset actions.
3. **CPU Latency on Long Documents**:
   While short queries evaluate in ~200 ms on CPU, full 900+ token documents take ~20 seconds on CPU. GPU acceleration (CUDA or Apple MPS) is strongly recommended for production RAG pipelines.
4. **Confidence Thresholding**:
   Because real-world calibration varies by domain, do not rely on a fixed `0.80` cutoff. Evaluate your specific positive/negative tradeoff curves to choose operational thresholds.

---

## 📦 Model Selection Guide

| Model | Hugging Face ID | Download Size | Best For |
| :--- | :--- | :---: | :--- |
| **Gevva e2b** | [`davidburhans/gevva-e2b`](https://huggingface.co/davidburhans/gevva-e2b) | ~10.2 GB (5.1B params) | Fast text NLI, RAG hallucination checks, general tool routing. |
| **Gevva e2b Multimodal** | [`davidburhans/gevva-e2b-multimodal`](https://huggingface.co/davidburhans/gevva-e2b-multimodal) | ~10.2 GB (5.1B params) | Multimodal: Checking charts, invoices, and photos against text claims. |
| **Gevva e4b** | [`davidburhans/gevva-e4b`](https://huggingface.co/davidburhans/gevva-e4b) | 15.88 GB (7.94B params) | Deep reasoning: Multi-hop logic, complex contracts, high-stakes verification. |

---

## 🛠️ Train on Your Own Data in 1 Line

Fine-tune Gevva on proprietary enterprise policies, internal tools, or domain datasets:

```bash
# Auto-detects column names and label formats (.jsonl, .csv, .tsv):
gevva finetune --data my_company_data.jsonl --out-dir ./my_custom_model --epochs 3
```

For advanced options (LoRA vs full fine-tuning, Quantization-Aware Training, token bucketing), see the [Custom Fine-Tuning Guide](docs/CUSTOM_FINETUNING_GUIDE.md).

---

## 📄 License & Citation

Gevva is licensed under the open-source [Apache 2.0 License](LICENSE). Base model weights inherit the [Google Gemma Terms of Use](https://ai.google.dev/gemma/terms).

```bibtex
@software{gevva2026,
  author = {Burhans, Dave and Contributors},
  title = {Gevva: Multimodal System 1 Decision Engine},
  year = {2026},
  publisher = {GitHub / Hugging Face},
  url = {https://github.com/davidburhans/gevva}
}
```
