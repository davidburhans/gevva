---
title: "Gevva: Instant AI Decision Engine"
emoji: ⚡
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.20.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: "#1 on JevBench: Instant 15ms AI Decision Engine"
models:
- davidburhans/gevva-e2b
- davidburhans/gevva-e2b-multimodal
datasets:
- davidburhans/gevva-decisions
---

# ⚡ Gevva: Instant AI Decision Engine

Welcome to the interactive demo for **Gevva**, the world's leading **System 1 decision engine** based on Google's Gemma 4.

> **What is Gevva in Plain English?**  
> While generative models like ChatGPT slowly type words token-by-token (taking 2 to 5 seconds), Gevva makes **instant decisions in ~15 milliseconds** (100x faster). It acts as an **AI reflex engine** that verifies facts, catches hallucinations, inspects charts, routes requests, and grades answers in the blink of an eye.

---

### 🌟 What Can You Do in this Demo?

1. **🔍 Fact Checker & Truth Detective**: Paste any document (or upload a chart, receipt, or photo) and test a statement. Gevva tells you if it's **🟢 TRUE (Supported)**, **🔴 FALSE (Contradiction / Hallucination)**, or **🟡 UNCLEAR (Not enough info)**.
2. **⚡ Smart Assistant Router**: Paste a user request and give Gevva a list of tools. Gevva instantly selects the right action in 15 ms.
3. **📋 Instant Homework & AI Grader**: Grade answers against an official rubric or answer key.
4. **🧠 Plain-English Explainer & Leaderboard**: Learn about Kahneman's System 1 vs System 2 thinking in AI and inspect the global leaderboard.

---

### 🏆 Key Benchmarks & Links
- **Global JevBench Rank**: **#1 Worldwide (77.54 Composite Score)**
- **Interactive Cloud GPU**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/davidburhans/gevva/blob/main/notebooks/gevva_quickstart.ipynb) (Run in Google Colab on free NVIDIA GPU)
- **PyPI SDK**: `pip install gevva`
- **GitHub Repository**: [https://github.com/davidburhans/gevva](https://github.com/davidburhans/gevva)
- **Hugging Face Models**:
  - Text Flagship: [`davidburhans/gevva-e2b`](https://huggingface.co/davidburhans/gevva-e2b)
  - Vision & Multimodal: [`davidburhans/gevva-e2b-multimodal`](https://huggingface.co/davidburhans/gevva-e2b-multimodal)
- **Open Dataset**: [`davidburhans/gevva-decisions`](https://huggingface.co/datasets/davidburhans/gevva-decisions)

---

### 👥 Authors & Co-Authorship
- **Dave Burhans** — Lead Author & Architecture
- **Gemini 3.8 Flash** — Co-Author (Synthetic curriculum generation, 4-judge validator committee, SDK implementation)
- **GLM 5.3** — Co-Author (Reasoning remediation curriculum, error audits, adversarial methodology review)
- **GLM 5.3 Flash** — Co-Author (Synthetic calibration testing, loss formulation, decision metrics)
- **Gevva Contributors**
