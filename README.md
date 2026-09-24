<div align="center">

```
 ██████╗ ███████╗██╗   ██╗██╗   ██╗ █████╗ 
██╔════╝ ██╔════╝██║   ██║██║   ██║██╔══██╗
██║  ███╗█████╗  ██║   ██║██║   ██║███████║
██║   ██║██╔══╝  ╚██╗ ██╔╝╚██╗ ██╔╝██╔══██║
╚██████╔╝███████╗ ╚████╔╝  ╚████╔╝ ██║  ██║
 ╚═════╝ ╚══════╝  ╚═══╝    ╚═══╝  ╚═╝  ╚═╝
```

# Gevva: SOTA Multimodal 128K System 1 Decision Engine

### *Ultra-fast, non-autoregressive categorical decisions, verification & candidate ranking in 14–16 ms.*
**#1 Global Rank on JevBench (76.95 Harmonic / 77.54 Geometric)**

[![Leaderboard](https://img.shields.io/badge/JevBench%20v1.4-%231%20Global%20(76.95%20Harmonic)-gold.svg?style=for-the-badge)](results/jevbench_public_gevva_e2b_summary.json)
[![Latency](https://img.shields.io/badge/Latency%20(GPU)-14.3--16.5%20ms-blue.svg?style=for-the-badge)](results/benchmark_comparison_100.json)
[![CPU Latency](https://img.shields.io/badge/Latency%20(CPU)-147%20ms%20(No%20GPU!)-teal.svg?style=for-the-badge)](scratch/benchmark_cpu.py)
[![Context](https://img.shields.io/badge/Context%20Window-128K%20Tokens-purple.svg?style=for-the-badge)](https://github.com/davidburhans/gevva)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg?style=for-the-badge)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-informational.svg?style=for-the-badge)](pyproject.toml)

[The Paradigm](#-the-system-1-paradigm) • [Leaderboard](#-official-jevbench-leaderboard) • [CPU Performance](#-cpu-performance-no-gpu-required) • [Quickstart](#-quickstart) • [Use Cases](#-primary-use-cases) • [Model Zoo](#-model-zoo) • [Fine-Tuning](#-custom-data-fine-tuning) • [Citation](#-citation)

---

</div>

## 📖 The System 1 Paradigm: Why Gevva?

Generative Large Language Models (LLMs) are brilliant at synthesis and creative writing, but they are **catastrophically over-engineered for categorical decisions**. 

When an AI system needs to verify a RAG citation, choose an agent tool, detect hallucination, or route an intent, asking a generative model to output text forces the GPU into an autoregressive decoding loop across dozens of tokens. This introduces **500–3,000 ms of latency**, inflates token bills, and produces uncalibrated, non-deterministic outputs.

### Enter Gevva
Inspired by Daniel Kahneman's cognitive framework (*Thinking, Fast and Slow*), **Gevva** is a high-throughput, non-autoregressive **System 1 Decision Engine**. 

Instead of generating text, **Gevva evaluates inputs in a single forward pass (~14–16 ms on GPU, ~147 ms on CPU)**, emitting calibrated probability distributions across three foundational semantic states:

$$\text{Class} \in \{\text{Contradiction (0)}, \text{Entailment (1)}, \text{Neutral (2)}\}$$

```
Traditional Autoregressive LLM:
[Premise + Query] ──> Autoregressive Generation (50-200 tokens) ──> 800 - 3,000 ms (High Cost / Variable Latency)

Gevva System 1 Decision Engine:
[Premise + Query] ──> Single Forward Pass (Calibrated Head)     ──> 14.3 - 16.5 ms (Fixed VRAM / Calibrated Probs)
```

> **Note on Architecture**: Gevva utilizes Google's lightweight multimodal Gemma 4 architecture as its foundational transformer backbone, but augments it with Gevva's specialized non-autoregressive decision head, group-atomic ranking loss, multi-judge synthetic curriculum, and mathematical calibration engine.

---

## 🏆 Official JevBench Leaderboard

On the official **JevBench** global benchmark, **Gevva e2b holds the #1 rank in the world across both the latest v1.4.0 release and earlier versions**:

### JevBench v1.4.0 Leaderboard (Current Official Release)
*Scored via the v1.4 equal-weight harmonic mean composite over Intelligence, Calibration, Speed, and Cost:*

| Rank | Model | Architecture / Backbone | Parameters | JevBench v1.4 Score | Intelligence | Calibration | Speed ($p_{50}$) | Cost / 1k | Open Source? |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 **#1** | **Gevva e2b** (Ours) | **Gevva Engine (Gemma 4 E2B-it Backbone)** | **2.3B** | **`76.95`** | **`73.91`** | **`86.90`** | **`16.5 ms`** | **`$0.0149`** | **Yes (Apache 2.0)** |
| 🥈 #2 | Jev 1.13.0 | Proprietary Commercial API | Closed | 63.29 | 53.06 | 76.34 | 236.0 ms | $0.0399 | No |
| 🥉 #3 | JevK5 v0.2.0 | Qwen 2.5 7B | 7.0B | 62.04 | 48.89 | 74.53 | 48.0 ms | $0.0210 | Yes |
| #4 | Hopper | Custom Decision Transformer | 1.8B | 59.43 | 48.00 | 79.06 | 62.0 ms | $0.0180 | Yes |
| #5 | Winnow-12B Q8 | Mistral NeMo 12B | 12.0B | 55.58 | 48.30 | 64.81 | 142.0 ms | $0.0310 | Yes |
| #6 | reflex 4B | Qwen 2.5 4B | 4.0B | 53.99 | 45.20 | 68.10 | 58.0 ms | $0.0220 | Yes |

> **Why Gevva's Lead Expanded under v1.4**: JevBench v1.4 replaced geometric weighting with an equal-weight harmonic mean, penalizing entrants with lagging latency or uncalibrated distributions. Because Gevva e2b delivers balanced, world-class performance across all four axes (Intelligence 73.91, Calibration 86.90, Speed 86.86, Cost 64.80), Gevva's lead over commercial Jev 1.13.0 grew from **+2.13 points** to **+13.66 points**!

### Gevva e2b Tier Breakdown:
- **Easy Tier Accuracy**: **`100.0%`** (48/48)
- **Standard Tier Accuracy**: **`88.89%`** (64/72)
- **Hard Tier Accuracy**: **`47.75%`** (53/111) — *massive gains in complex multi-hop, policy, and tradeoff reasoning*
- **Overall Accuracy**: **`71.43%`** (165/231 public items across all 18 evaluation families)
- **Hard-Tier Calibration (ECE)**: **`0.0655`** (*with $T^* = 1.60$ post-hoc temperature calibration*)
- **JevBench v1.2 Geometric Composite Score**: **`77.54`** (#1 Global)

### 📋 Leaderboard Methodology & Disclosures
- **Harmonic Composite (v1.4.0)**: Evaluated under JevBench v1.4's equal-weight harmonic mean composite over the 4 axes. Gevva scores **`76.95`** (leading commercial Jev 1.13.0 at 63.29 by +13.66 points).
- **Geometric Composite (v1.2/v1.3)**: Evaluated under the earlier 4-axis geometric mean, yielding **`77.54`** (leading Jev 1.13.0 at 75.41 by +2.13 points).
- **Full Public Benchmark Suite (231/231 Items - 100% Evaluation)**: Evaluated across 100% of all public items in the official JevBench suite (48 easy, 72 standard, 111 hard), with zero sampling, skipping, or cherry-picking.
- **Calibrated Temperature**: Scores reflect the model's shipped calibration configuration ($T^* = 1.60$), optimizing probability fidelity without altering discrete choice rankings. At raw uncalibrated temperature ($T=1.00$), Gevva e2b scores **72.94** (#4 globally).
- **Speed Axis**: JevBench applies a $\times 2 + 0.15\text{s}$ penalty to self-hosted models to simulate production network overhead. Gevva's raw hardware latency on RTX 5090 is **16.5 ms** ($p_{50}$).

---

## 💻 CPU Performance (No GPU Required!)

While Gevva achieves blazing **14.3–16.5 ms** latency on NVIDIA GPUs, it is natively engineered to run on commodity CPUs without any dedicated accelerator.

Because Gevva operates non-autoregressively, **running Gevva on a CPU is actually faster than running commercial Jev (236 ms) or LLaMA 8B (651 ms) on datacenter GPUs!**

### Measured CPU Benchmarks (AMD Ryzen 16-Core / AVX-512 BF16)
- **Host RAM Required**: **~4.8 GB** (comfortably runs on standard laptops, MacBooks, Mac Minis, or cloud CPU VMs)
- **Model Load Time**: **1.24 seconds**

| Workload Mode | Batch Size | Total Latency | Per-Decision Latency | Throughput | Real-World Application |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Single-Decision ($p_{50}$)** | 1 | **147.7 ms** | **147.7 ms** | **6.7 decisions/s** | Real-time chat & agent tool routing |
| **Single-Decision ($p_{95}$)** | 1 | **156.5 ms** | **156.5 ms** | **6.4 decisions/s** | High-reliability SLA endpoints |
| **Batched Decisions** | 4 | **244.5 ms** | **61.1 ms** | **16.4 decisions/s** | Document RAG paragraph scanning |
| **4-Candidate Reranking** | 4 | **210.5 ms** | **52.6 ms** | **19.0 options/s** | Zero-shot search candidate rerank |

---

## 🌟 Key Highlights of Gevva

- **⚡ 14.3–16.5 ms Forward Latency**: Up to **38× faster** than competing System 1 models (`system-one-open` at 651 ms, commercial Jev at 236 ms).
- **🎯 World-Class Calibration**: Expected Calibration Error (ECE) of **0.0107** on validation and **0.0655** on JevBench Hard tier. Predictions represent genuine Bayesian probabilities, not uncalibrated overconfident logits.
- **📚 Native 128K Context Window**: Seamlessly ingests full multi-page PDFs, clinical trials, legal documents, or complete codebases without chunking artifacts.
- **👁️ Multimodal Vision Support**: Evaluates visual inputs (charts, UI wireframes, documents) alongside text queries through Gevva's frozen SigLIP vision tower.
- **🔄 100% Drop-In Jev & OpenJEV API Parity**: Full signature and return type compatibility with [`AlexWortega/openjev`](https://huggingface.co/AlexWortega/openjev) (`predict`, `rerank`, `grade`, `latents`, `LatentMLPHead`, `OpenJevCrossEncoder`).
- **🛠️ 1-Line Turnkey Fine-Tuning**: Auto-detects data formats (`.jsonl`, `.csv`, `.tsv`, `.parquet`), normalizes label schemas, automatically maps columns, and runs stratified splitting with optional 4-bit QAT or Full Fine-Tuning.

---

## 🚀 Quickstart

### Installation

```bash
# Install directly from GitHub
git clone https://github.com/davidburhans/gevva.git
cd gevva
pip install -e .

# Or using uv (recommended for speed)
uv sync
```

### 1. Basic 3-Class NLI Prediction

```python
import gevva

# Loads the champion Gevva e2b model directly into VRAM (or CPU)
model = gevva.load("ckpt/gevva-e2b", device="auto")

# Predict semantic relationship
probs = model.predict([
    ("The Apollo 11 mission landed astronauts on the Moon in July 1969.", "Apollo 11 was an autumn mission.")
])

# Output: [P(contradiction), P(entailment), P(neutral)]
print(probs)  # [[0.9426, 0.0324, 0.0250]] -> Contradiction!
```

### 2. Zero-Shot Candidate Reranking

Rerank search results, multi-choice candidates, or retrieval passages:

```python
import gevva

model = gevva.load("ckpt/gevva-e2b")

query = "Which gas do plants primarily absorb from the atmosphere during photosynthesis?"
candidates = [
    "Nitrogen gas (N2)",
    "Carbon dioxide (CO2)",
    "Oxygen gas (O2)",
    "Argon gas (Ar)"
]

best_idx, scores = model.rerank(query, candidates)
print(f"Top Option: {candidates[best_idx]} (Score: {scores[best_idx]:.4f})")
# Top Option: Carbon dioxide (CO2) (Score: 0.9812)
```

### 3. Reference-Based Answer Grading (Rubric Scoring)

Evaluate an LLM's response against a reference answer or rubric standard:

```python
import gevva

model = gevva.load("ckpt/gevva-e2b")

grade = model.grade(
    question="What causes the seasons to change on Earth?",
    reference="The 23.5-degree axial tilt of the Earth relative to its orbital plane around the Sun.",
    candidate="Earth's seasons are caused by the tilt of its rotational axis as it orbits the sun."
)

print(f"Is Correct: {grade.is_correct}")  # True
print(f"Confidence: {grade.score * 100:.2f}%")  # 96.4%
```

### 4. 100% Drop-In OpenJEV Compatibility

Existing code using `openjev` works without any refactoring:

```python
from gevva import OpenJevCrossEncoder, LatentMLPHead

# Drop-in replacement for OpenJEV
jev = OpenJevCrossEncoder("ckpt/gevva-e2b")

# Standard OpenJEV rerank call
best_idx = jev.rerank("What is the capital of Japan?", ["Kyoto", "Tokyo", "Osaka"])
assert best_idx == 1  # Standard int indexing works out of the box!

# Extract pooled latent vectors for downstream probe heads:
latents = jev.latents([("Evidence document...", "Hypothesis statement...")])
print("Latents shape:", latents.shape)  # (1, 2048)
```

---

## 💡 Primary Use Cases

### 1. RAG Hallucination Detection & Document Grounding
Traditional RAG pipelines rely on slow, expensive LLM calls to check whether extracted answers are hallucinated. Gevva verifies claims against up to **128,000 tokens** of source text in a single forward pass:

```python
premise = full_retrieved_pdf_text  # Up to 128K tokens
claim = "The clinical trial demonstrated a 34% reduction in primary cardiac events."

probs = model.predict([(premise, claim)])[0]
if probs[0] > 0.80:
    alert_hallucination("Claim contradicts source documentation!")
elif probs[1] > 0.70:
    proceed("Claim is directly entailed by the source.")
```

### 2. High-Throughput Intent & Tool Routing
Route user requests to the correct API endpoint or tool in **16 ms**, eliminating prompt-tuning and output-parsing latency:

```python
tools = [
    "Process credit card payment and checkout order",
    "Check inventory stock level for SKU",
    "Track shipment and delivery status",
    "Initiate return or customer support refund"
]

best_idx, scores = model.rerank("Where is package #94001234?", tools)
# Immediately triggers shipment tracker API without waiting for LLM token generation
```

### 3. Multimodal Visual Claim Verification
Pass an image alongside textual context to verify visual claims:

```python
from PIL import Image

image = Image.open("quarterly_revenue_chart.png")
claim = "Q3 revenue grew by 18% quarter-over-quarter."

# Visual entailment checks the image and text simultaneously
probs = model.predict([(image, claim)])
```

---

## 📦 The Gevva Model Family

| Model | Checkpoint | Backbone Architecture | Parameters | Context | JevBench Score | Latency ($p_{50}$) | Primary Application |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **`Gevva e2b`** | [`ckpt/gevva-e2b`](ckpt/gevva-e2b) | `google/gemma-4-E2B-it` | 2.3B | 128K | **`76.95`** (#1 Global) | **16.5 ms** (147 ms CPU) | Real-time production serving, edge & mobile |
| **`Gevva e2b (W4A16)`** | `ckpt/gevva-e2b-w4a16` | Merged INT4 Group-32 | 2.3B | 128K | **`76.80`** | **14.3 ms** | Ultra-low VRAM (<5.2 GB), maximum throughput |
| **`Gevva e4b`** | *In Staging* | `google/gemma-4-E4B-it` | 4.5B | 128K | *Targeting 80+* | ~28.0 ms | Complex legal/medical reasoning & deep documents |

---

## 🛠️ Custom Data Fine-Tuning

Adapt Gevva to your proprietary domain with **zero boilerplate**:

```bash
# 1-line command with automatic column mapping and stratified train/val split:
gevva finetune --data my_domain_data.jsonl --out-dir ./ckpt/my_domain_gevva

# Or run via Python runner with Full Fine-Tuning:
python finetune.py \
    --data enterprise_cases.jsonl \
    --base-model ckpt/gevva-e2b \
    --out-dir ./ckpt/enterprise_gevva \
    --full-fine-tune \
    --epochs 2
```

For complete recipes, supported column variations, and QAT options, see the [Custom Fine-Tuning Guide](docs/CUSTOM_FINETUNING_GUIDE.md).

---

## 🔬 Architecture & Methodology

### 1. The Gevva Training Curriculum
Gevva was trained across a rigorous **3-stage curriculum** encompassing **243,916 curated pairs** across 41 diverse sources:
1. **Core NLI & Foundation**: SNLI, MNLI, ANLI (R1, R2, R3), WANLI, FEVER, XNLI (15 languages).
2. **System 1 Enterprise Decisions**: 82,000+ multi-choice workflow routing problems from `n4ze3m/typed-decisions-synth`.
3. **Hard-Tier Reasoning & Parity Data**: Multi-hop deduction (ReClor, LogiQA 2.0), legal precedence (CaseHOLD), long policy comprehension (RACE), temporal/arithmetic constraints (AQuA-RAT), and committee-validated SDK parity pairs.

### 2. Group-Atomic Cross-Option Ranking Loss
Unlike traditional cross-encoders trained only on isolated binary pairs, Gevva trains directly on the **served option distribution**:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{served\_dist}} + 0.5 \mathcal{L}_{\text{cross\_option}} + 0.15 \mathcal{L}_{\text{nli\_aux}} + 0.5 \mathcal{L}_{\text{Brier}}$$
This aligns training dynamics exactly with serving behavior, eliminating confirmation asymmetry and decision drift.

### 3. Temperature Calibration
To ensure model confidence matches empirical truth, Gevva models include post-hoc validation temperature scaling ($T^* = 1.60$), compressing Expected Calibration Error (ECE) from 0.1604 down to **0.0655** on hard reasoning.

---

## 💻 Command-Line Interface (CLI)

Gevva includes a comprehensive CLI:

```bash
# Check version and hardware capabilities
gevva version

# Evaluate premise-hypothesis pair
gevva predict \
    --model ckpt/gevva-e2b \
    --premise "Company revenue was $4.2B in 2025." \
    --hypothesis "Revenue exceeded four billion dollars."

# Rerank multiple choices
gevva rerank \
    --model ckpt/gevva-e2b \
    --query "Select the correct protocol for encrypted web traffic:" \
    --options "HTTP" "HTTPS" "FTP" "Telnet"

# Run JevBench evaluation suite
gevva eval --suite jevbench
```

---

## 📄 License & Attribution

This project is licensed under the [Apache 2.0 License](LICENSE). Base model weights inherit the [Google Gemma Terms of Use](https://ai.google.dev/gemma/terms).

### Citation

If you use Gevva in your research, systems, or products, please cite:

```bibtex
@software{gevva2026,
  author = {Burhans, Dave and Contributors},
  title = {Gevva: State-of-the-Art Multimodal 128K System 1 Decision Engine},
  year = {2026},
  url = {https://github.com/davidburhans/gevva},
  note = {Rank 1 on Global JevBench Leaderboard}
}
```

---

<div align="center">
  <sub>Built with ❤️ by the Gevva Team. Inspired by Kahneman's System 1 cognitive framework.</sub>
</div>
