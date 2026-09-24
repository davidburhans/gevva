---
title: Gevva Multimodal 128K System 1 Decision Engine
emoji: ⚡
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: "#1 Global on JevBench (77.54): Ultra-fast System 1 NLI & Vision Engine"
models:
- davidburhans/gevva-e2b
- davidburhans/gevva-e2b-multimodal
datasets:
- davidburhans/gevva-decisions
---

# ⚡ Gevva System 1 Decision Engine

Welcome to the interactive Hugging Face demo for **Gevva**, the state-of-the-art multimodal 128K System 1 decision engine based on Google's Gemma 4 models.

- **Global JevBench Rank**: **#1 Worldwide (77.54 Composite Score)**
- **Interactive Cloud GPU**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/davidburhans/gevva/blob/main/notebooks/gevva_quickstart.ipynb) (Run in Google Colab on free NVIDIA GPU)
- **Models**:
  - Flagship Text Engine: [`davidburhans/gevva-e2b`](https://huggingface.co/davidburhans/gevva-e2b)
  - Vision Engine: [`davidburhans/gevva-e2b-multimodal`](https://huggingface.co/davidburhans/gevva-e2b-multimodal)
- **Dataset**: [`davidburhans/gevva-decisions`](https://huggingface.co/datasets/davidburhans/gevva-decisions)
- **PyPI SDK**: `pip install gevva`
- **GitHub**: [https://github.com/davidburhans/gevva](https://github.com/davidburhans/gevva)

### 👥 Authors & Co-Authorship
- **Dave Burhans** — Lead Author & Architecture
- **Gemini 3.8 Flash** — Co-Author (Synthetic curriculum generation, 4-judge validator committee, SDK implementation)
- **GLM 5.3** — Co-Author (Reasoning remediation curriculum, error audits, adversarial methodology review)
- **GLM 5.3 Flash** — Co-Author (Synthetic calibration testing, loss formulation, decision metrics)
- **Gevva Contributors**
