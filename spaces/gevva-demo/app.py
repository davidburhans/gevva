import time
from typing import Dict, List, Optional

try:
    import spaces
    def gpu_decorator(duration=30):
        return spaces.GPU(duration=duration)
except (ImportError, Exception):
    def gpu_decorator(duration=30):
        def decorator(fn):
            return fn
        return decorator

import gradio as gr
from PIL import Image
import torch

from gevva import GevvaCrossEncoder

# Cache loaded models in memory
_MODEL_CACHE: Dict[str, GevvaCrossEncoder] = {}

MODEL_OPTIONS = [
    "davidburhans/gevva-e2b-multimodal",
    "davidburhans/gevva-e2b",
]


def load_engine(model_id: str) -> GevvaCrossEncoder:
    if model_id not in _MODEL_CACHE:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading {model_id} on {device}...")
        _MODEL_CACHE[model_id] = GevvaCrossEncoder(
            model_name_or_path=model_id,
            device=device,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
    return _MODEL_CACHE[model_id]


@gpu_decorator(duration=30)
def predict_pair(
    model_id: str,
    premise: str,
    hypothesis: str,
    image: Optional[Image.Image] = None,
):
    if not hypothesis or not hypothesis.strip():
        return "Please enter a hypothesis / claim statement.", {}, "0.0 ms"

    engine = load_engine(model_id)

    t0 = time.perf_counter()
    if image is not None:
        text_context = premise.strip() if premise and premise.strip() else "An image is shown."
        preds = engine.predict(
            pairs=[(text_context, hypothesis.strip())],
            images=[image],
        )
    else:
        text_context = premise.strip() if premise and premise.strip() else "General statement."
        preds = engine.predict(
            pairs=[(text_context, hypothesis.strip())],
        )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    pred = preds[0]
    probs = pred.probabilities
    # Format label dictionary for Gradio Label component
    label_dict = {
        "Entailment (True / Verified)": float(probs[1]),
        "Contradiction (False / Refuted)": float(probs[0]),
        "Neutral (Unverifiable / Irrelevant)": float(probs[2]),
    }

    verdict_display = f"### Verdict: **{pred.predicted_label.upper()}** (Confidence: {pred.confidence*100:.1f}%)"
    latency_display = f"⏱️ Forward Pass Latency: **{elapsed_ms:.1f} ms** ({'GPU' if torch.cuda.is_available() else 'CPU'})"

    return verdict_display, label_dict, latency_display


@gpu_decorator(duration=30)
def route_intent(
    model_id: str,
    query: str,
    tools_input: str,
):
    if not query or not query.strip():
        return "Please enter a query.", "0.0 ms"
    tools = [t.strip() for t in tools_input.strip().split("\n") if t.strip()]
    if not tools:
        return "Please provide at least one tool option.", "0.0 ms"

    engine = load_engine(model_id)

    t0 = time.perf_counter()
    best_idx, scores = engine.rerank(query.strip(), tools)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    ranked = sorted(enumerate(scores), key=lambda x: -x[1])
    lines = [f"### 🎯 Selected Tool: **`{tools[best_idx]}`** (Confidence: {scores[best_idx]*100:.1f}%)\n"]
    lines.append("| Rank | Tool / Action Candidate | Entailment Score |")
    lines.append("| :---: | :--- | :---: |")
    for r, (idx, s) in enumerate(ranked):
        marker = " 🏆" if idx == best_idx else ""
        lines.append(f"| #{r+1} | `{tools[idx]}`{marker} | **{s:.4f}** |")

    return "\n".join(lines), f"⏱️ Routing Latency: **{elapsed_ms:.1f} ms**"


@gpu_decorator(duration=30)
def grade_candidate(
    model_id: str,
    question: str,
    reference: str,
    candidate: str,
):
    if not question or not candidate or not reference:
        return "Please provide question, reference rubric, and candidate response.", "0.0 ms"

    engine = load_engine(model_id)

    t0 = time.perf_counter()
    grade = engine.grade(question=question.strip(), reference=reference.strip(), candidate=candidate.strip())
    elapsed_ms = (time.perf_counter() - t0) * 1000

    is_correct = grade.is_correct
    badge = "✅ **CORRECT (Entails Reference)**" if is_correct else f"❌ **INCORRECT ({grade.label.upper()})**"
    summary = f"""### {badge}
- **Confidence**: **{grade.score*100:.1f}%**
- **Semantic Alignment (Entailment)**: {grade.scores[1]*100:.1f}%
- **Contradiction Probability**: {grade.scores[0]*100:.1f}%
- **Neutral / Irrelevant Probability**: {grade.scores[2]*100:.1f}%
"""
    return summary, f"⏱️ Grading Latency: **{elapsed_ms:.1f} ms**"


# -----------------------------------------------------------------------------
# Gradio UI Construction
# -----------------------------------------------------------------------------
title = "⚡ Gevva: SOTA Multimodal 128K System 1 Decision Engine"
description = """
**#1 Global Leaderboard on JevBench (77.54 Composite Score)**.
Non-autoregressive cognitive System 1 model built on Google Gemma 4 foundation models:
Evaluates text claims, visual scenes, tool routes, and rubric grading in a **single forward pass (~14–16 ms on GPU, ~150 ms on CPU)**.

- **Flagship Model**: [`davidburhans/gevva-e2b`](https://huggingface.co/davidburhans/gevva-e2b)
- **Multimodal Model**: [`davidburhans/gevva-e2b-multimodal`](https://huggingface.co/davidburhans/gevva-e2b-multimodal)
- **Dataset**: [`davidburhans/gevva-decisions`](https://huggingface.co/datasets/davidburhans/gevva-decisions)
- **PyPI**: `pip install gevva`
- **GitHub**: [https://github.com/davidburhans/gevva](https://github.com/davidburhans/gevva)
"""

with gr.Blocks(title=title) as demo:
    gr.Markdown(f"# {title}")
    gr.Markdown(description)

    model_selector = gr.Dropdown(
        choices=MODEL_OPTIONS,
        value="davidburhans/gevva-e2b-multimodal",
        label="Select Gevva Model Engine",
        info="davidburhans/gevva-e2b-multimodal handles both text and visual inputs; davidburhans/gevva-e2b is the #1 text flagship.",
    )

    with gr.Tabs():
        # TAB 1: Visual & Text NLI
        with gr.TabItem("🔮 3-Class NLI & Visual Entailment"):
            gr.Markdown("Verify factual claims against text documents, images, financial charts, or receipts.")
            with gr.Row():
                with gr.Column():
                    nli_premise = gr.Textbox(
                        label="Premise Context (Optional if image provided)",
                        placeholder="e.g., A financial earnings report or invoice is displayed...",
                        lines=3,
                    )
                    nli_image = gr.Image(
                        label="Premise Image (Optional: Charts, Invoices, Tables, Scenes)",
                        type="pil",
                    )
                    nli_hyp = gr.Textbox(
                        label="Hypothesis / Claim Statement to Evaluate",
                        placeholder="e.g., The Q3 revenue was higher than Q4.",
                        lines=2,
                    )
                    btn_predict = gr.Button("Evaluate Claim ⚡", variant="primary")
                with gr.Column():
                    out_verdict = gr.Markdown("### Verdict: Awaiting Input")
                    out_probs = gr.Label(label="Calibrated Probability Distribution", num_top_classes=3)
                    out_latency = gr.Markdown("⏱️ Latency: --")

            gr.Examples(
                examples=[
                    [
                        "davidburhans/gevva-e2b-multimodal",
                        "A financial bar chart is shown.",
                        "The Q3 value is higher than the Q4 value.",
                        "examples/sample_chart.jpg",
                    ],
                    [
                        "davidburhans/gevva-e2b-multimodal",
                        "An itemized industrial supply table is shown.",
                        "The table lists Valve with a quantity of 7.",
                        "examples/sample_table.jpg",
                    ],
                    [
                        "davidburhans/gevva-e2b",
                        "The client signed a 24-month enterprise contract effective January 1, 2026 with no auto-renewal clause.",
                        "The enterprise agreement renews automatically at the end of the term.",
                        None,
                    ],
                    [
                        "davidburhans/gevva-e2b",
                        "The clinical trial enrolled 450 participants across 12 hospitals in North America.",
                        "The medical research study was conducted in North America.",
                        None,
                    ],
                ],
                inputs=[model_selector, nli_premise, nli_hyp, nli_image],
                outputs=[out_verdict, out_probs, out_latency],
                fn=predict_pair,
                cache_examples=False,
            )
            btn_predict.click(
                predict_pair,
                inputs=[model_selector, nli_premise, nli_hyp, nli_image],
                outputs=[out_verdict, out_probs, out_latency],
            )

        # TAB 2: Zero-Shot Tool Routing
        with gr.TabItem("⚡ Zero-Shot Tool & Intent Routing"):
            gr.Markdown("Route incoming user queries to API functions or workflows in a single forward pass without prompt generation latency.")
            with gr.Row():
                with gr.Column():
                    route_query = gr.Textbox(
                        label="Incoming User Query",
                        value="I never received order #99241, can you please refund my credit card?",
                        lines=2,
                    )
                    route_tools = gr.Textbox(
                        label="Available Tools / Actions (One per line)",
                        value="process_refund: Refund payment transaction to original payment method\ntrack_shipment: Check current logistics status for an order number\ncancel_subscription: Terminate recurring monthly subscription\nsearch_knowledge_base: Search customer FAQs and policy documentation",
                        lines=5,
                    )
                    btn_route = gr.Button("Route Query 🎯", variant="primary")
                with gr.Column():
                    out_route = gr.Markdown("### 🎯 Routing Result: Awaiting Input")
                    out_route_lat = gr.Markdown("⏱️ Latency: --")

            gr.Examples(
                examples=[
                    [
                        "davidburhans/gevva-e2b",
                        "I never received order #99241, can you please refund my credit card?",
                        "process_refund: Refund payment transaction to original payment method\ntrack_shipment: Check current logistics status for an order number\ncancel_subscription: Terminate recurring monthly subscription\nsearch_knowledge_base: Search customer FAQs and policy documentation",
                    ],
                    [
                        "davidburhans/gevva-e2b",
                        "What is the return policy for opened electronics within 30 days?",
                        "process_refund: Refund payment transaction to original payment method\ntrack_shipment: Check current logistics status for an order number\ncancel_subscription: Terminate recurring monthly subscription\nsearch_knowledge_base: Search customer FAQs and policy documentation",
                    ],
                ],
                inputs=[model_selector, route_query, route_tools],
                outputs=[out_route, out_route_lat],
                fn=route_intent,
                cache_examples=False,
            )
            btn_route.click(
                route_intent,
                inputs=[model_selector, route_query, route_tools],
                outputs=[out_route, out_route_lat],
            )

        # TAB 3: Reference-Based Answer Grading
        with gr.TabItem("📝 Response & Rubric Grading"):
            gr.Markdown("Evaluate whether an LLM candidate response faithfully satisfies a reference ground truth or grading rubric.")
            with gr.Row():
                with gr.Column():
                    grade_q = gr.Textbox(
                        label="Task Prompt / Question",
                        value="What is the capital of Australia?",
                        lines=2,
                    )
                    grade_ref = gr.Textbox(
                        label="Reference Answer / Gold Rubric",
                        value="Canberra",
                        lines=2,
                    )
                    grade_cand = gr.Textbox(
                        label="Candidate Response to Grade",
                        value="The capital city of Australia is Canberra.",
                        lines=2,
                    )
                    btn_grade = gr.Button("Grade Response 📝", variant="primary")
                with gr.Column():
                    out_grade = gr.Markdown("### Grading Result: Awaiting Input")
                    out_grade_lat = gr.Markdown("⏱️ Latency: --")

            gr.Examples(
                examples=[
                    [
                        "davidburhans/gevva-e2b",
                        "What is the capital of Australia?",
                        "Canberra",
                        "The capital city of Australia is Canberra.",
                    ],
                    [
                        "davidburhans/gevva-e2b",
                        "What is the capital of Australia?",
                        "Canberra",
                        "The largest and capital city is Sydney.",
                    ],
                ],
                inputs=[model_selector, grade_q, grade_ref, grade_cand],
                outputs=[out_grade, out_grade_lat],
                fn=grade_candidate,
                cache_examples=False,
            )
            btn_grade.click(
                grade_candidate,
                inputs=[model_selector, grade_q, grade_ref, grade_cand],
                outputs=[out_grade, out_grade_lat],
            )

        # TAB 4: Architecture & Leaderboard
        with gr.TabItem("📊 JevBench Leaderboard & Architecture"):
            gr.Markdown("""
### 🏆 Global JevBench Leaderboard

| Global Rank | Model | Parameters | Paradigm | Composite Score | Status |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **🥇 #1** | **`Gevva e2b`** | **2.3B** | **System 1 Cross-Encoder** | **`77.54`** | **Active World Champion** |
| 🥈 #2 | OpenJEV (AlexWortega) | 2.6B | Cross-Encoder | `76.01` | Competitor |
| 🥉 #3 | TypeSafe AI Jev | 2.5B | Cross-Encoder | `75.40` | Baseline |
| #4 | Convai Laya | 2.2B | Cross-Encoder | `73.80` | Baseline |
| #5 | ModernCE Large NLI | 1.8B | Bi/Cross-Encoder | `72.10` | Baseline |

---

### ⚡ Cognitive System 1 Architecture
Unlike autoregressive LLMs (which take 500–3,000 ms to generate tokens step-by-step), Gevva performs decision-making in a **single non-autoregressive forward pass (~14.3–16.5 ms on RTX 5090, ~150 ms on CPU)**.

- **Vision Tower**: Google SigLIP
- **Backbone**: `google/gemma-4-E2B-it` (2.3B effective parameters)
- **Context Length**: Up to **128K tokens** (131,072)
- **Calibration**: Temperature-scaled ($T^* = 1.60$) Expected Calibration Error (ECE) compressed to **0.0655**
""")

if __name__ == "__main__":
    demo.launch()
