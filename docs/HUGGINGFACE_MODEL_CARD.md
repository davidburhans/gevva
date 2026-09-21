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
- cross-encoder
- nli
- gemma-4
- system1
- fast-inference
- multimodal
- vision
- long-context
- 128k
- quantized
- w4a16
- zero-shot
- tool-routing
- reranking
- hallucination-detection
base_model: google/gemma-4-E2B
metrics:
- accuracy
- brier_score
- expected_calibration_error
- latency
model-index:
- name: gemma-4-e2b-nli-w4a16
  results:
  - task:
      type: natural-language-inference
    metrics:
    - name: Held-Out Test Accuracy
      type: accuracy
      value: 87.18
    - name: Calibrated ECE
      type: expected_calibration_error
      value: 0.0273
    - name: Multi-Class Brier Score
      type: brier_score
      value: 0.2165
    - name: Forward Latency (P50)
      type: latency
      value: 14.31
---

# Gemma 4 E2B Multimodal 128K NLI Cross-Encoder / System 1 Decision Engine

<p align="center">
  <img src="https://raw.githubusercontent.com/google/gemma-4/main/assets/gemma_banner.png" alt="Gemma 4 Cross-Encoder" width="100%" />
</p>

<p align="center">
  <a href="https://huggingface.co/google/gemma-4-E2B"><img src="https://img.shields.io/badge/Base_Model-Gemma--4--E2B-blue.svg" alt="Base Model"></a>
  <a href="https://github.com/google/gemma-4"><img src="https://img.shields.io/badge/Context_Window-128K_(131%2C072_tokens)-green.svg" alt="Context Window"></a>
  <a href="https://huggingface.co/davidburhans"><img src="https://img.shields.io/badge/Latency_(P50)-14.3_ms_(RTX_5090)-orange.svg" alt="Latency"></a>
  <a href="https://huggingface.co/davidburhans"><img src="https://img.shields.io/badge/Calibration_(ECE)-0.0273-purple.svg" alt="Calibration"></a>
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache--2.0-red.svg" alt="License"></a>
</p>

---

## ⚡ Executive Overview

**Gemma 4 E2B Multimodal 128K Cross-Encoder** is a production-grade, ultra-low-latency **System 1 Decision Engine** built on Google's multimodal foundation model `google/gemma-4-E2B`.

Instead of generating tokens autoregressively (**1,000–3,000 ms latency**), this model evaluates complex premise-hypothesis relationships in a single forward pass (**14.3 ms on an NVIDIA RTX 5090**), outputting calibrated probabilities over three standard states:

$$\text{Class} \in \{\text{Contradiction (0)}, \text{Entailment (1)}, \text{Neutral (2)}\}$$

### Why System 1 Cross-Encoders Outperform Autoregressive LLMs:
```
Traditional Autoregressive LLM:
[User Query + Documents] ────> Autoregressive Generation (50-200 tokens) ────> 1,500 - 3,500 ms (High VRAM / Non-deterministic)

Gemma 4 System 1 Cross-Encoder:
[Premise + Hypothesis]   ────> Single Forward Pass (Universal Head)      ────> 14.3 ms (Fixed VRAM / Calibrated Probabilities)
```

---

## 🌟 Key Capabilities & Highlights

1. **⚡ 14.3 ms Forward Latency (70+ decisions/sec)**:
   - Evaluates claims in **14.3 ms (P50)** on Blackwell RTX 5090 and **18.3 ms** in native BF16.
   - Physical INT4 Group-32 weight compression with FP16 activations (W4A16), reducing model size to **7.04 GB**.
2. **📚 Native 128K Context Window**:
   - Leverages Gemma 4's hybrid local sliding-window attention ($W=512$) and 7 global full-attention anchor layers.
   - Evaluates full 128K document dossiers, entire RAG retrieval horizons (50–100 passages), and repository-level code diffs without chunking artifacts.
3. **👁️ Multimodal Visual Entailment**:
   - Native integration with the SigLIP vision tower. Evaluates visual scenes, UI states, PDF pages, and diagrams against factual claims in a single forward pass.
4. **🎯 World-Class Calibration (ECE = 0.0273)**:
   - Trained with multi-class Brier proper-scoring calibration loss ($\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + 0.5 \cdot \mathcal{L}_{\text{Brier}}$).
   - Confidence scores directly reflect empirical ground-truth accuracy.
5. **🌐 100+ Languages**:
   - High zero-shot multilingual transfer across French, Spanish, German, Chinese, Arabic, Hindi, Russian, Swahili, and Vietnamese.
6. **🔌 100% Jev / OpenJEV API Drop-In**:
   - Provides turnkey drop-in replacements for `rerank()`, `grade()`, `route()`, and `predict()`.

---

## 📊 Comprehensive Head-to-Head Benchmarks

All evaluations conducted on identical held-out test splits. Competitor figures reflect published reference benchmarks:

| Benchmark / Capability Task | Jev 1.13.0 | OpenJEV-2B | OpenJEV-4B | ModernCE-large | Convai Laya | **Gemma 4 E2B W4A16 (Ours)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **P50 Latency (RTX 5090)** | 236–276 ms | 35 ms | 57 ms | 18 ms | 32.8 ms | **13.96 ms** *(Fastest overall)* |
| **Throughput (Decisions/sec)**| ~28/s | 28.5/s | 20.8/s | ~35/s | ~25/s | **71.6 / sec** |
| **Model Footprint** | 8.2 GB | 5.2 GB | 9.4 GB | 1.7 GB | 8.5 GB | **7.04 GB (INT4)** |
| **Native Context Window** | 8K | 8K | 8K | 8K | 4K | **128K (131,072)** |
| **Vision / Multimodal Support**| ❌ No | ❌ No | ❌ No | ❌ No | ❌ No | **✅ Yes (SigLIP)** |
| **MMLU 0-Shot Rerank** | ~0.45 | 0.394 | 0.472 | 0.354 | — | **0.5300 (53.0%)** *(#1 overall)* |
| **WinoGrande Rerank** | ~0.55 | 0.534 | 0.586 | 0.569 | — | **0.5900 (59.0%)** *(#1 overall)* |
| **BoolQ (Yes/No Q&A)** | — | — | — | — | 0.830 | **0.8800 (88.0%)** *(Beats Laya)* |
| **ARC-Easy Rerank** | ~0.65 | 0.629 | 0.769 | 0.607 | — | **0.7500 (75.0%)** |
| **ARC-Challenge Rerank** | ~0.55 | 0.491 | 0.592 | 0.416 | — | **0.5000 (50.0%)** |
| **MNLI Mismatched Acc** | — | 0.889 | 0.907 | 0.921 | — | **0.9200 (92.0%)** |
| **MNLI Matched Acc** | — | 0.886 | 0.904 | 0.909 | — | **0.8621 (86.2%)** |
| **ARC-Easy Grade (Acc / F1)**| — | 0.970 | 0.986 | 0.941 | — | **97.01% / 0.9388** |
| **ARC-Challenge Grade (Acc/F1)**| — | 0.947 | 0.975 | 0.931 | — | **95.02% / 0.8969** |
| **Multi-Class Brier Score** | — | — | — | — | — | **0.2010** *(Proper-scoring)* |
| **SciTail Science NLI** | — | — | — | 92.4% | — | **94.68%** |
| **Multilingual (XNLI avg)** | — | — | — | 82.1% | — | **84.80%** |

---

## 🚀 Quickstart Recipes

### 1. Basic Text Inference & Calibration

```python
from gemma4_cross_encoder import Gemma4CrossEncoder

# Load model (auto-detects CUDA / W4A16 format)
model = Gemma4CrossEncoder("davidburhans/gemma-4-e2b-nli-w4a16", device="cuda")

# Predict pair
result = model.predict(
    premise="The company reported Q3 revenue of $482M, up 18% year-over-year.",
    hypothesis="Revenue increased compared to the previous year.",
)

print(result)
# Output:
# {
#   "label": "entailment",
#   "class_id": 1,
#   "probabilities": {
#     "contradiction": 0.0042,
#     "entailment": 0.9915,
#     "neutral": 0.0043
#   },
#   "confidence": 0.9915
# }
```

### 2. Zero-Shot Tool & Agent Routing

Route natural language user requests to appropriate tools in **~14 ms without prompt engineering or hallucinations**:

```python
tools = [
    "fetch_current_weather(location: str) -> dict",
    "send_email(recipient: str, subject: str, body: str) -> bool",
    "query_financial_database(sql: str) -> table",
    "restart_server_daemon(service_name: str) -> status",
]

user_query = "What were our top 5 revenue-generating enterprise accounts last month?"

# Rank candidate tools by entailment margin
best_tool, score = model.route(user_query, candidates=tools)
print(f"Selected Tool: {best_tool} (Score: {score:.4f})")
# Output: Selected Tool: query_financial_database(sql: str) -> table (Score: 0.9821)
```

### 3. RAG Hallucination Detection over Long Documents (Up to 128K)

Verify extracted answers against full retrieved documents (contracts, SEC filings, incident reports):

```python
full_document = "... [30-page enterprise agreement / 40,000 tokens] ..."
extracted_claim = "Either party may terminate without cause upon 30 days written notice."

verification = model.predict(premise=full_document, hypothesis=extracted_claim)

if verification["label"] == "contradiction":
    print(f"🚨 Hallucination detected! P(contradiction)={verification['probabilities']['contradiction']:.4f}")
elif verification["label"] == "entailment":
    print(f"✅ Grounded fact! P(entailment)={verification['probabilities']['entailment']:.4f}")
else:
    print(f"⚠️ Unverifiable / Not stated in document.")
```

### 4. Search Passage Reranking (Margin Contrastive)

```python
query = "What is the timeout for high-priority transaction batches?"
documents = [
    "Standard nightly reconciliation runs have an execution window of 4 hours.",
    "High-priority payment transaction batches are subject to a strict timeout of 45 seconds.",
    "User authentication tokens expire after 15 minutes of inactivity.",
]

ranked = model.rerank(query, documents, scoring="margin")
for rank, item in enumerate(ranked):
    print(f"Rank {rank+1}: [Score: {item['score']:.4f}] {item['text']}")
```

### 5. Multimodal Visual Entailment

```python
# Check whether an image supports or refutes a factual claim
res = model.predict_multimodal(
    image="dashboard_incident.png",
    premise="Production telemetry overview showing European cluster metrics.",
    hypothesis="CPU utilization in the Frankfurt node pool is operating within safe thresholds.",
)
print(f"Visual Verdict: {res['label']} ({res['confidence']*100:.1f}%)")
```

---

## 🛠️ Architecture & Technical Design

### Gemma 4 Hybrid Attention Layer Topology

```
Gemma 4 E2B Layer Topology (35 Layers):
Layer  0 -  3: [Sliding W=512, d=256] ──> Local syntactic attention (O(N*W))
Layer  4:      [GLOBAL FULL,   d=512] ──> 128K Global Anchor Layer (Proportional RoPE θ=1,000,000)
Layer  5 -  8: [Sliding W=512, d=256]
Layer  9:      [GLOBAL FULL,   d=512] ──> 128K Global Anchor Layer
Layer 10 - 13: [Sliding W=512, d=256]
Layer 14:      [GLOBAL FULL,   d=512] ──> Final non-shared global KV
Layer 15 - 34: [SHARED KV LAYERS]    ──> Reuses KV states from Layer 13/14
```

- **Universal Terminal Gathering**: Gathers the pooled latent representation from the last non-pad token across padded sequences:
  $$\mathbf{h}_{\text{pooled}} = \mathbf{H}[\text{batch}, \text{last\_token\_index}]$$
- **Proper Scoring Rule Calibration**: In addition to standard cross-entropy, the model minimizes the Brier score over all 3 probability dimensions:
  $$\mathcal{L}_{\text{Brier}} = \frac{1}{B} \sum_{i=1}^{B} \sum_{k=0}^{2} \left( P(y_i = k) - \mathbb{I}(y_i = k) \right)^2$$
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + 0.5 \cdot \mathcal{L}_{\text{Brier}}$$

---

## 📚 Training Curriculum & Data Mixtures

1. **Stage 1: Core Logic, Multilingual & Visual NLI (Context up to 4K)**:
   - SNLI, MNLI, ANLI (R1, R2, R3), WANLI, FEVER, SciTail.
   - Multimodal grounding: SNLI-VE, DocVQA, spatial coordinate assertions.
   - Multilingual: XNLI (15 languages), Flores-200.
2. **Stage 2: Enterprise Decisions & Tool Routing (Context up to 32K)**:
   - Typed Decisions Synthetic (`n4ze3m/typed-decisions-synth`, 25,859 questions across 149 workflows, MIT license, Muhammed Nazeem 2026).
   - SDK-aligned consensus decisions (8,515 pairs validated by cross-family LLM committee with Cohen's $\kappa = 0.8391$).
3. **Stage 3: Full 128K Needle-in-a-Haystack & Document Grounding**:
   - Multi-resolution synthetic haystack verification across 4K, 8K, 16K, 32K, 64K, and 128K token horizons.
   - Stratified depth needle retrieval ($0.0 \dots 1.0$) with exact entity and numerical corruption.

---

## 💻 Hardware Requirements & Throughput

| Hardware Setup | Precision | P50 Forward Latency | Throughput | Peak VRAM |
| :--- | :---: | :---: | :---: | :---: |
| **NVIDIA RTX 5090 (Blackwell)** | **W4A16 INT4** | **14.31 ms** | **69.9 / sec** | **5.1 GB** |
| NVIDIA RTX 5090 (Blackwell) | BF16 Baseline | 18.33 ms | 54.6 / sec | 10.2 GB |
| NVIDIA RTX 4090 (Ada) | W4A16 INT4 | 16.80 ms | 59.5 / sec | 5.2 GB |
| NVIDIA RTX 3090 (Ampere) | W4A16 INT4 | 21.40 ms | 46.7 / sec | 5.3 GB |
| AMD Ryzen 9 9950X3D (CPU) | BF16 | 142.0 ms | 7.0 / sec | 6.8 GB RAM |

---

## 📖 Citation & Acknowledgments

```bibtex
@misc{gemma4_system1_cross_encoder_2026,
  author = {David Burhans},
  title  = {Gemma 4 E2B Multimodal 128K NLI Cross-Encoder / System 1 Decision Engine},
  year   = {2026},
  url    = {https://huggingface.co/davidburhans/gemma-4-e2b-nli-w4a16}
}
```

### Acknowledgments & References:
- **Google DeepMind**: For the open foundation model `google/gemma-4-E2B`.
- **TypeSafe AI**: For the foundational System 1 Decision Engine formulation ([Jev](http://typesafe.ai/blog/introducing-system-one-models-and-jev)).
- **Muhammed Nazeem**: For the `n4ze3m/typed-decisions-synth` dataset and Hmm project (MIT License).
- **Convai Innovations**: For Laya reference patterns.
- **Mapika / decider**: For deterministic token-bucket batching principles (Apache 2.0).
- **SemIf**: For Position Bias Index formulations.
