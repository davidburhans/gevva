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

# ⚡ Gevva e2b: SOTA Multimodal 128K System 1 Decision Engine

<p align="center">
  <a href="https://huggingface.co/google/gemma-4-E2B-it"><img src="https://img.shields.io/badge/Base_Model-Gemma--4--E2B--it-blue.svg" alt="Base Model"></a>
  <a href="https://github.com/davidburhans/gevva"><img src="https://img.shields.io/badge/JevBench%20v1.4-%231%20Global%20(76.95)-gold.svg" alt="JevBench #1"></a>
  <a href="https://github.com/davidburhans/gevva"><img src="https://img.shields.io/badge/Context_Window-128K_(131%2C072_tokens)-purple.svg" alt="Context Window"></a>
  <a href="https://github.com/davidburhans/gevva"><img src="https://img.shields.io/badge/Latency_(P50)-16.5_ms_(RTX_5090)-orange.svg" alt="Latency"></a>
  <a href="https://github.com/davidburhans/gevva"><img src="https://img.shields.io/badge/Hard_ECE-0.0655-green.svg" alt="Calibration"></a>
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache--2.0-red.svg" alt="License"></a>
</p>

---

## ⚡ Executive Overview

**Gevva e2b** is a state-of-the-art, ultra-low-latency **System 1 Decision Engine** and multimodal NLI cross-encoder built on Google's instruction-tuned multimodal foundation model `google/gemma-4-E2B-it`.

Rather than generating text autoregressively (**500–3,000 ms latency**), **Gevva e2b evaluates complex premise-hypothesis relationships and categorical choices in a single forward pass (~16.5 ms)**, producing mathematically calibrated probabilities over three fundamental semantic states:

$$\text{State} \in \{\text{Contradiction (0)}, \text{Entailment (1)}, \text{Neutral (2)}\}$$

On the official **JevBench** public benchmark suite (231/231 problems), **Gevva e2b achieved #1 in the world across both the latest v1.4.0 release (`76.95` harmonic) and v1.2/v1.3 (`77.54` geometric)**, outperforming commercial proprietary APIs (Jev 1.13.0 at 63.29) and leading open models.

---

## 🏆 Benchmark Leaderboard (JevBench v1.4.0)

*Scored via the v1.4 equal-weight harmonic mean composite over Intelligence, Calibration, Speed, and Cost on the complete official public split (231/231 items; private 534-item held-out submission pending):*

| Rank | Model | Architecture | JevBench v1.4 Score | Intelligence | Calibration | Speed ($p_{50}$) | Cost / 1k | Open Source? |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 **#1** | **Gevva e2b** (Ours) | **Gemma 4 E2B-it (FFT)** | **`76.95`** | **`73.91`** | **`86.90`** | **`16.5 ms`** | **`$0.0149`** | **Yes (Apache 2.0)** |
| 🥈 #2 | Jev 1.13.0 | Proprietary Commercial API | 63.29 | 53.06 | 76.34 | 236.0 ms | $0.0399 | No |
| 🥉 #3 | JevK5 v0.2.0 | Qwen 2.5 7B | 62.04 | 48.89 | 74.53 | 48.0 ms | $0.0210 | Yes |
| #4 | Hopper | Custom Transformer | 59.43 | 48.00 | 79.06 | 62.0 ms | $0.0180 | Yes |
| #5 | Winnow-12B Q8 | Mistral NeMo 12B | 55.58 | 48.30 | 64.81 | 142.0 ms | $0.0310 | Yes |
| #6 | reflex 4B | Qwen 2.5 4B | 53.99 | 45.20 | 68.10 | 58.0 ms | $0.0220 | Yes |

> **Leaderboard Notes**:
> - **Calibration**: Evaluated with optimal validation temperature scaling ($T^* = 1.60$) pre-configured in `calibration.json`. At raw temperature ($T=1.00$), Gevva e2b scores **72.94** (#4 globally).
> - **Latency**: 16.5 ms reflects $p_{50}$ on short sequences (~128–256 tokens) on RTX 5090 (and 147.7 ms on CPU). Long multi-page policy verification scales with sequence length (~380–413 ms).
> - **Cost**: Amortized self-hosted dedicated compute under continuous saturation.
> - **Latent Dimension**: Gevva e2b outputs 1536-dimensional pooled latents matching Gemma 4 E2B hidden state.

---

## 🌟 Key Capabilities

1. **⚡ 16.5 ms Forward Latency**:
   - Single forward evaluation in ~16.5 ms ($p_{50}$) on short sequences on NVIDIA RTX 5090 (and ~14.3 ms in quantized INT4 W4A16; 147 ms on commodity CPUs).
   - High-throughput batch serving (>70 decisions/sec).
2. **📚 128K Foundation Context Window**:
   - Built on Gemma 4's native 128K RoPE architecture (fine-tuned up to 4K, supporting long document verification).
3. **👁️ Multimodal Vision-Language Entailment**:
   - Analyzes images (charts, UI screenshots, diagrammatic PDFs) alongside text claims through Gemma 4's SigLIP vision tower.
4. **🎯 Superior Calibration**:
   - Hard-tier Expected Calibration Error (ECE) of **0.0655** with temperature scaling ($T^* = 1.60$).
5. **🔄 100% Drop-In Jev & OpenJEV API Parity**:
   - Directly replaces `AlexWortega/openjev` methods (`predict`, `rerank`, `grade`, `latents`) with full signature compatibility.

---

## 🚀 Quickstart & Inference

### Using the `gevva` SDK (Recommended)

```bash
pip install gevva
```

```python
import gevva

# Load model directly from HuggingFace
model = gevva.load("davidburhans/gevva-e2b", device="cuda")

# 1. 3-Class NLI Prediction
probs = model.predict([
    ("The company reported $1.2B revenue in Q3.", "The company lost money in Q3.")
])
print(probs)  # [[0.952, 0.028, 0.020]] -> Contradiction!

# 2. Zero-Shot Candidate Reranking
query = "What is the primary function of mitochondria?"
options = [
    "Protein synthesis",
    "Cellular ATP energy production",
    "Lipid storage",
    "DNA replication"
]
best_idx, scores = model.rerank(query, options)
print(f"Top Choice: {options[best_idx]} (Score: {scores[best_idx]:.4f})")
# Top Choice: Cellular ATP energy production (Score: 0.9845)

# 3. Reference-Based Grading
grade = model.grade(
    question="What is the capital of France?",
    reference="Paris",
    candidate="Paris"
)
print(f"Grade: {grade.label} (Correct: {grade.is_correct})")
```

### Using Raw HuggingFace Transformers

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("davidburhans/gevva-e2b")
model = AutoModelForSequenceClassification.from_pretrained(
    "davidburhans/gevva-e2b", 
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)

premise = "The document confirms delivery on September 24."
hypothesis = "Delivery took place in September."
formatted = f"Premise: {premise}\nHypothesis: {hypothesis}"

inputs = tokenizer(formatted, return_tensors="pt").to("cuda")
with torch.no_grad():
    logits = model(**inputs).logits
    probs = torch.softmax(logits / 1.60, dim=-1)  # calibrated with T*=1.60

labels = {0: "contradiction", 1: "entailment", 2: "neutral"}
predicted = labels[int(probs.argmax())]
print(f"Verdict: {predicted} ({probs[0][probs.argmax()]*100:.1f}%)")
```

### Model Variants & Branches

The repository provides two official model variants under the same repo ID `davidburhans/gevva-e2b`:

| Variant | Branch / Revision | Description | Benchmark Highlights |
| :--- | :--- | :--- | :--- |
| **Flagship (Default)** | `main` | Global #1 System 1 Text Decision Engine | **77.54** JevBench Composite (#1 Global) |
| **Multimodal** | `multimodal` | Vision-grounded decision engine | **88.8%** Visual Grounding (96.4% on Invoices/Tables) |

To load the multimodal variant:
```python
import gevva

# Load the vision-enabled variant branch
model = gevva.load("davidburhans/gevva-e2b", revision="multimodal", device="cuda")
```

---

## 🔬 Training Curriculum & Methodology

Gevva e2b was trained on a **243,916-pair master curriculum** spanning 41 datasets:
1. **Core NLI Anchors**: SNLI, MNLI, ANLI, WANLI, FEVER, XNLI (15 languages).
2. **System 1 Enterprise Decisions**: 82,000+ workflow routing scenarios (`n4ze3m/typed-decisions-synth`).
3. **Hard Multi-Choice Reasoning**: ReClor, LogiQA 2.0, CaseHOLD, RACE, AQuA-RAT, StrategyQA.
4. **SDK-Parity Synthetic Data**: 8,515 cross-family committee-validated pairs generated by a `gemma-4-31b` teacher and verified across a 4-judge committee (`qwen-3.6-27b`, `qwen-3.8-125b`, `deepseek-v4-flash`).

### Training Objective:
$$\mathcal{L} = \mathcal{L}_{\text{served\_dist}} + 0.5 \mathcal{L}_{\text{cross\_option}} + 0.15 \mathcal{L}_{\text{nli\_aux}} + 0.5 \mathcal{L}_{\text{Brier}}$$

---

## 📄 License & Terms

Gevva e2b is released under the [Apache 2.0 License](https://opensource.org/licenses/Apache-2.0). Underlying foundation weights inherit Google's [Gemma Terms of Use](https://ai.google.dev/gemma/terms).

### Citation

```bibtex
@software{gevva2026,
  author = {Burhans, Dave and Contributors},
  title = {Gevva: State-of-the-Art Multimodal 128K System 1 Decision Engine},
  year = {2026},
  url = {https://github.com/davidburhans/gevva},
  note = {Rank 1 on Global JevBench Leaderboard}
}
```
