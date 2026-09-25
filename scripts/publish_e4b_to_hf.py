#!/usr/bin/env python3
"""scripts/publish_e4b_to_hf.py
==============================
Publishes the Gevva E4B Flagship checkpoint to the Hugging Face Hub.
Uses HF_WRITE_TOKEN from the environment.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder

E4B_MODEL_CARD = """---
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
base_model: google/gemma-4-E4B-it
metrics:
- accuracy
- brier_score
- expected_calibration_error
- latency
model-index:
- name: gevva-e4b
  results:
  - task:
      type: natural-language-inference
    metrics:
    - name: JevBench Public Accuracy
      type: accuracy
      value: 76.19
    - name: ARC-Challenge Rerank
      type: accuracy
      value: 84.00
    - name: ARC-Easy Rerank
      type: accuracy
      value: 92.00
    - name: Forward Latency (P50)
      type: latency
      value: 17.8
---

# ⚡ Gevva e4b Flagship: Deep Reasoning AI Decision Engine

<p align="center">
  <a href="https://huggingface.co/spaces/davidburhans/gevva-demo"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Space-Interactive%20Demo-blue.svg" alt="Interactive Demo"></a>
  <a href="https://pypi.org/project/gevva/"><img src="https://img.shields.io/pypi/v/gevva.svg?logo=pypi&logoColor=white" alt="PyPI"></a>
  <a href="https://github.com/davidburhans/gevva"><img src="https://img.shields.io/badge/Latency-17.8ms%20(RTX%205090)-orange.svg" alt="Latency"></a>
  <a href="https://github.com/davidburhans/gevva"><img src="https://img.shields.io/badge/ARC--Challenge-84%25%20Accuracy-brightgreen.svg" alt="ARC-Challenge"></a>
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache--2.0-green.svg" alt="License"></a>
</p>

---

### *17-millisecond deep reasoning, high-stakes fact checking, multi-tool routing & multi-image verification.*
**Built on Google's Gemma 4 E4B (4.5B parameters, 42 layers, 128K context) • 84% on ARC-Challenge • 76.19% on JevBench**

---

## 🤔 What is Gevva e4b? (The 30-Second Explainer)

When you ask ChatGPT or Claude to verify a document or route a customer request, it generates words **one token at a time**, like a human slowly typing out an explanation. That takes **2 to 5 seconds** and burns expensive GPU compute.

That is fine for writing stories, but it is **painfully slow and expensive for decisions**:
- *"Did the AI hallucinate this medical summary, or is it grounded in the research paper?"*
- *"Which of these 10 enterprise APIs should execute this user workflow?"*
- *"Did this UI screenshot change in an unexpected way after deployment?"*
- *"Did the student's solution satisfy all 5 steps in the grading rubric?"*

### The Solution: An Instant "Reflex Engine" with Deep Reasoning
Psychologist Daniel Kahneman famously described human thought in two modes:
- **System 1 (Fast Reflexes)**: Instant decisions in 15–20 milliseconds (dodging a ball, recognizing a face).
- **System 2 (Slow Reasoning)**: Writing essays or working out long equations step-by-step.

**Gevva e4b** is the deep reasoning flagship of the Gevva family. In a single **17.8 millisecond forward pass**, it evaluates evidence and outputs confident, mathematically calibrated decision probabilities.

```
┌────────────────────────────────────────┐       ┌────────────────────────────────────────┐
│         SYSTEM 1: GEVVA e4b            │       │       SYSTEM 2: CHATGPT / CLAUDE       │
│  ⚡ Evaluates decisions in 17.8 ms     │  vs   │  🐢 Generates text word-by-word        │
│  🎯 84% ARC-Challenge accuracy        │       │  ⏳ Takes 2,000 - 5,000 milliseconds   │
│  💰 90%+ cheaper compute cost          │       │  💸 Expensive GPU server bills         │
│  🔍 Best for: Fact-checks, routing,    │       │  ✍️ Best for: Creative writing, long   │
│     guardrails, and visual audit       │       │     essays, and coding from scratch    │
└────────────────────────────────────────┘       └────────────────────────────────────────┘
```

---

## 🚀 3-Line Quickstart

```python
from gevva import load

# 1. Load the flagship decision engine (GPU or CPU)
engine = load("davidburhans/gevva-e4b")

# 2. Instant Fact Check / Hallucination Detection
verdict, probs = engine.grade(
    premise="Albert Einstein won the Nobel Prize in Physics in 1921 for his discovery of the photoelectric effect.",
    hypothesis="Einstein won the Nobel Prize for his theory of General Relativity."
)

print(verdict)
# Output: 'contradiction' (p_contradiction = 0.96)
```

### Smart Tool & API Routing

```python
tools = [
    "send_email(to, subject, body): Sends an email message to a contact",
    "search_database(query): Queries internal company records and docs",
    "refund_charge(charge_id, reason): Issues a payment refund to a customer",
]

best_idx, scores = engine.rerank(
    premise="A customer says they were double-billed and wants their money back.",
    options=tools
)

print(f"Selected: {tools[best_idx]}")
# Output: Selected: refund_charge(...)
```

---

## 🏆 Benchmark Highlights: Dominating the 4B Class

Gevva e4b was evaluated on standard reasoning batteries on an **NVIDIA GeForce RTX 5090**:

| Benchmark | Gevva e4b Flagship | OpenJEV-4B | Decider 4B | TypeSafe Jev 1.13 |
| :--- | :---: | :---: | :---: | :---: |
| **ARC-Challenge (Reasoning)** | **84.00%** | 59.20% | ~55% | ~55% |
| **ARC-Easy (Knowledge)** | **92.00%** | 76.90% | ~65% | ~65% |
| **WinoGrande (Commonsense)** | **75.00%** | 58.60% | ~58% | ~55% |
| **MMLU (General Knowledge)** | **59.00%** | 47.20% | ~47% | ~45% |
| **BoolQ (Fact Decisions)** | **91.00%** | — | — | — |
| **Forward Latency ($p_{50}$)** | **17.83 ms** | 57.0 ms | 23.4 ms | 236 ms |
| **Context Window** | **128,000 tokens** | 4,096 tokens | 4,096 tokens | 8,192 tokens |

---

## 🛠️ Architecture Details

- **Foundation Backbone**: Google's `gemma-4-E4B-it` (4.5B parameters, 42 layers).
- **Classification Head**: Last-token pooled representation normalized with Gemma4RMSNorm into a 3-class linear head.
- **Label Convention**:
  - `0`: Contradiction / Refutes / Invalid
  - `1`: Entailment / Supports / Valid
  - `2`: Neutral / Unverifiable / Inconclusive
- **Calibration**: Trained with strictly proper Brier score calibration ($ECE = 0.0224$ on validation).

---

## 📄 License & Citations

Released under the **Apache 2.0 License**.

```bibtex
@software{gevva2026,
  author = {David Burhans},
  title = {Gevva: Multimodal 128K System 1 Decision Engine},
  year = {2026},
  url = {https://github.com/davidburhans/gevva}
}
```
"""


def main():
    token = os.environ.get("HF_WRITE_TOKEN")
    if not token:
        print("Error: HF_WRITE_TOKEN environment variable not set.")
        sys.exit(1)

    repo_id = "davidburhans/gevva-e4b"
    checkpoint_dir = Path("ckpt/gevva-e4b-flagship/best")

    if not checkpoint_dir.exists():
        print(f"Error: Checkpoint directory not found at {checkpoint_dir}")
        sys.exit(1)

    api = HfApi(token=token)

    print(f"Ensuring repository exists: {repo_id}")
    create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, token=token)

    # Write model card README.md into checkpoint dir
    readme_path = checkpoint_dir / "README.md"
    readme_path.write_text(E4B_MODEL_CARD, encoding="utf-8")
    print(f"Written model card to {readme_path}")

    print(f"Uploading {checkpoint_dir} to https://huggingface.co/{repo_id}...")
    upload_folder(
        folder_path=str(checkpoint_dir),
        repo_id=repo_id,
        repo_type="model",
        token=token,
        commit_message="Initial release of Gevva e4b Flagship Decision Engine",
    )
    print(f"Successfully published Gevva e4b: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
