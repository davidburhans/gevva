#!/usr/bin/env python3
import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# ⚡ Gevva: SOTA Multimodal 128K System 1 Decision Engine\n",
                "\n",
                "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/davidburhans/gevva/blob/main/notebooks/gevva_quickstart.ipynb)\n",
                "[![PyPI version](https://img.shields.io/pypi/v/gevva.svg)](https://pypi.org/project/gevva/)\n",
                "[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/davidburhans/gevva-e2b)\n",
                "[![JevBench](https://img.shields.io/badge/JevBench%20v1.4-%231%20Global%20(77.54)-gold.svg)](https://github.com/davidburhans/gevva)\n",
                "\n",
                "**Gevva** is the world-champion System 1 decision engine based on Google's lightweight Gemma 4 multimodal models. Rather than generating text autoregressively (500–3,000 ms), Gevva evaluates pairs in a **single non-autoregressive forward pass (~14.3–16.5 ms on GPU)** with calibrated probabilities.\n",
                "\n",
                "### Models covered in this notebook:\n",
                "- 🥇 **`davidburhans/gevva-e2b`**: #1 Global on JevBench (77.54 Composite Score) — flagship text decision engine.\n",
                "- 👁️ **`davidburhans/gevva-e2b-multimodal`**: 88.8% visual grounding accuracy (96.4% on tables & invoices).\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Install & Hardware Setup\n",
                "Make sure your runtime has a GPU allocated (**Runtime > Change runtime type > T4 GPU**)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Check GPU availability\n",
                "!nvidia-smi\n",
                "\n",
                "# Install official Gevva SDK from PyPI\n",
                "!pip install -q gevva pillow"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Text System 1 Decision (Flagship #1 Global Champion)\n",
                "Evaluate premise-hypothesis factual entailment in a single forward pass (~14 ms)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import time\n",
                "from gevva import GevvaCrossEncoder\n",
                "\n",
                "# Load the flagship text decision engine on GPU\n",
                "model = GevvaCrossEncoder(\"davidburhans/gevva-e2b\", device=\"cuda\")\n",
                "\n",
                "premise = \"The company reported $4.2B in Q3 revenue with a net profit margin of 18%.\"\n",
                "hypothesis = \"The company was profitable in Q3.\"\n",
                "\n",
                "t0 = time.perf_counter()\n",
                "pred = model.predict([(premise, hypothesis)])[0]\n",
                "elapsed_ms = (time.perf_counter() - t0) * 1000\n",
                "\n",
                "print(f\"Verdict:        {pred.predicted_label.upper()}\")\n",
                "print(f\"Confidence:     {pred.confidence*100:.2f}%\")\n",
                "print(f\"Latency:        {elapsed_ms:.1f} ms\")\n",
                "print(f\"Probabilities:  Contradiction={pred.probabilities[0]:.4f}, Entailment={pred.probabilities[1]:.4f}, Neutral={pred.probabilities[2]:.4f}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Multimodal Visual Entailment\n",
                "Verify visual claims against charts, financial diagrams, invoices, and receipts."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import urllib.request\n",
                "from PIL import Image\n",
                "\n",
                "# Load the multimodal decision engine on GPU\n",
                "mm_model = GevvaCrossEncoder(\"davidburhans/gevva-e2b-multimodal\", device=\"cuda\")\n",
                "\n",
                "# Download sample financial chart\n",
                "chart_url = \"https://raw.githubusercontent.com/davidburhans/gevva/main/spaces/gevva-demo/examples/sample_chart.jpg\"\n",
                "urllib.request.urlretrieve(chart_url, \"chart.jpg\")\n",
                "image = Image.open(\"chart.jpg\").convert(\"RGB\")\n",
                "\n",
                "claim = \"The Q3 value is higher than the Q4 value.\"\n",
                "\n",
                "t0 = time.perf_counter()\n",
                "pred = mm_model.predict(\n",
                "    pairs=[(\"A quarterly financial bar chart is shown.\", claim)],\n",
                "    images=[image]\n",
                ")[0]\n",
                "elapsed_ms = (time.perf_counter() - t0) * 1000\n",
                "\n",
                "print(f\"Visual Claim:   {claim}\")\n",
                "print(f\"Verdict:        {pred.predicted_label.upper()} ({pred.confidence*100:.1f}% confidence)\")\n",
                "print(f\"Latency:        {elapsed_ms:.1f} ms\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Zero-Shot Tool & Intent Routing\n",
                "Route incoming customer or agent queries to appropriate tools or API handlers instantly."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "query = \"I was billed twice for subscription sub_8912, please refund my credit card.\"\n",
                "tools = [\n",
                "    \"process_refund: Refund payment to customer payment method\",\n",
                "    \"track_package: Query logistics carrier for tracking status\",\n",
                "    \"reset_password: Send password reset link to user email\",\n",
                "    \"search_faq: Search knowledge base articles\",\n",
                "]\n",
                "\n",
                "result = model.route(query, tools)\n",
                "print(f\"Query:         \\\"{query}\\\"\")\n",
                "print(f\"Routed Tool:   {result.selected_tool}\")\n",
                "print(f\"Confidence:    {result.top_score*100:.2f}%\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Automated Response & Rubric Grading\n",
                "Check whether candidate LLM answers satisfy an exact gold answer or grading rubric."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "question = \"What is the capital of Australia?\"\n",
                "reference = \"Canberra\"\n",
                "candidate = \"The federal capital of Australia is Canberra, located in the ACT.\"\n",
                "\n",
                "grade = model.grade(question=question, reference=reference, candidate=candidate)\n",
                "print(f\"Is Correct:    {grade.is_correct}\")\n",
                "print(f\"Confidence:    {grade.score*100:.2f}%\")\n",
                "print(f\"Label:         {grade.label}\")"
            ]
        }
    ],
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "provenance": []
        },
        "kernelspec": {
            "display_name": "Python 3",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 0
}

with open("notebooks/gevva_quickstart.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print("Created notebooks/gevva_quickstart.ipynb successfully!")
