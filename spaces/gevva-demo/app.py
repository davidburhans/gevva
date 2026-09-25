import os
import time
from typing import Dict, List, Optional

import numpy as np
from PIL import Image
import torch

try:
    import spaces
    if hasattr(spaces, "GPU"):
        def gpu_decorator(duration=30):
            return spaces.GPU(duration=duration)
    else:
        def gpu_decorator(duration=30):
            def decorator(fn):
                return fn
            return decorator
except Exception:
    def gpu_decorator(duration=30):
        def decorator(fn):
            return fn
        return decorator

import gradio as gr
import gradio_client.utils as client_utils

# Patch gradio_client bug where boolean additionalProperties causes TypeError in Pydantic 2.11+
_orig_json_schema_to_python_type = client_utils._json_schema_to_python_type


def _safe_json_schema_to_python_type(schema, defs=None):
    if isinstance(schema, bool):
        return "Any"
    return _orig_json_schema_to_python_type(schema, defs)


client_utils._json_schema_to_python_type = _safe_json_schema_to_python_type

from gevva import GevvaCrossEncoder

# Cache loaded models in memory
_MODEL_CACHE: Dict[str, GevvaCrossEncoder] = {}

MODEL_OPTIONS = [
    "davidburhans/gevva-e2b-multimodal",
    "davidburhans/gevva-e2b",
    "davidburhans/gevva-e4b",
]


def load_engine(model_id: str) -> GevvaCrossEncoder:
    if model_id not in _MODEL_CACHE:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading {model_id} on {device}...")
        resolved_path = model_id
        # In local development, resolve to local checkpoints if present
        if model_id == "davidburhans/gevva-e2b-multimodal" and os.path.isdir("ckpt/gevva-e2b-phase4/best"):
            resolved_path = "ckpt/gevva-e2b-phase4/best"
        elif model_id == "davidburhans/gevva-e2b" and os.path.isdir("ckpt/gevva-e2b"):
            resolved_path = "ckpt/gevva-e2b"
        elif model_id == "davidburhans/gevva-e4b" and os.path.isdir("ckpt/gevva-e4b-flagship/best"):
            resolved_path = "ckpt/gevva-e4b-flagship/best"

        _MODEL_CACHE[model_id] = GevvaCrossEncoder(
            model_name_or_path=resolved_path,
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

    # Defensively normalize image format if provided
    if image is not None:
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif hasattr(image, "convert"):
            image = image.convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")

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

    # preds is np.ndarray of shape (1, 3): [p_contradiction, p_entailment, p_neutral]
    probs = preds[0]
    class_idx = int(np.argmax(probs))
    labels = ["contradiction", "entailment", "neutral"]
    predicted_label = labels[class_idx]
    confidence = float(probs[class_idx])

    # Plain-English human explanation
    if predicted_label == "entailment":
        icon = "🟢"
        verdict_text = "VERIFIED TRUE (ENTAILMENT)"
        meaning = "The provided document or image directly supports and proves this statement."
    elif predicted_label == "contradiction":
        icon = "🔴"
        verdict_text = "FALSE / REFUTED (CONTRADICTION)"
        meaning = "This statement directly contradicts the provided document or image (hallucination or factual error detected)."
    else:
        icon = "🟡"
        verdict_text = "UNCLEAR / NOT ENOUGH INFO (NEUTRAL)"
        meaning = "The provided document or image does not contain enough evidence to confirm or deny this statement."

    label_dict = {
        "Entailment (True / Verified)": float(probs[1]),
        "Contradiction (False / Refuted)": float(probs[0]),
        "Neutral (Unverifiable / Irrelevant)": float(probs[2]),
    }

    verdict_display = f"""### {icon} Verdict: **{verdict_text}** (Confidence: {confidence*100:.1f}%)

> **In Plain English**: {meaning}
"""
    latency_display = f"⏱️ Forward Pass Latency: **{elapsed_ms:.1f} ms** ({'GPU accelerated' if torch.cuda.is_available() else 'CPU mode'})"

    return verdict_display, label_dict, latency_display


@gpu_decorator(duration=30)
def route_intent(
    model_id: str,
    query: str,
    tools_input: str,
    image: Optional[Image.Image] = None,
):
    if not query or not query.strip():
        return "Please enter a query.", "0.0 ms"
    tools = [t.strip() for t in tools_input.strip().split("\n") if t.strip()]
    if not tools:
        return "Please provide at least one tool option.", "0.0 ms"

    engine = load_engine(model_id)

    # Defensively normalize image format if provided
    if image is not None:
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif hasattr(image, "convert"):
            image = image.convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")

    t0 = time.perf_counter()
    best_idx, scores = engine.rerank(
        premise=query.strip(),
        options=tools,
        image=image,
        temperature=1.0,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    ranked = sorted(enumerate(scores), key=lambda x: -x[1])
    winning_tool = tools[best_idx]
    winning_score = scores[best_idx] * 100

    lines = [
        f"### 🎯 Winning Action: **`{winning_tool}`** ({winning_score:.1f}% Match)\n",
        f"> **Plain-English Meaning**: Gevva instantly selected `{winning_tool}` as the best tool to handle this request out of {len(tools)} candidates in {elapsed_ms:.1f} ms.\n",
        "| Rank | Tool / Action Candidate | Match Probability |",
        "| :---: | :--- | :---: |",
    ]
    for r, (idx, s) in enumerate(ranked):
        marker = " 🏆" if idx == best_idx else ""
        lines.append(f"| #{r+1} | `{tools[idx]}`{marker} | **{s*100:.1f}%** |")

    return "\n".join(lines), f"⏱️ Routing Latency: **{elapsed_ms:.1f} ms** ({'GPU accelerated' if torch.cuda.is_available() else 'CPU mode'})"


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
    p_dict = grade.probabilities
    p_ent = float(p_dict.get("entailment", 0.0))
    p_con = float(p_dict.get("contradiction", 0.0))
    p_neu = float(p_dict.get("neutral", 0.0))

    if is_correct:
        badge = "✅ CORRECT (Entails Reference)"
        explanation = "The student / AI response accurately satisfies the reference answer key."
    else:
        badge = f"❌ INCORRECT ({grade.label.upper()})"
        if grade.label == "contradiction":
            explanation = "The response directly contradicts the answer key (contains factual errors or wrong facts)."
        else:
            explanation = "The response is incomplete, irrelevant, or fails to satisfy the answer key."

    summary = f"""### {badge}
> **In Plain English**: {explanation}

- **Overall Grade Confidence**: **{grade.score*100:.1f}%**
- **Semantic Alignment (Entailment)**: {p_ent*100:.1f}%
- **Contradiction Probability**: {p_con*100:.1f}%
- **Neutral / Irrelevant Probability**: {p_neu*100:.1f}%
"""
    return summary, f"⏱️ Grading Latency: **{elapsed_ms:.1f} ms**"


# -----------------------------------------------------------------------------
# Gradio UI Construction & Styling
# -----------------------------------------------------------------------------
title = "⚡ Gevva: Instant AI Decision Engine"

custom_css = """
.hero-box {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 20px;
    color: #f8fafc;
}
.hero-title {
    font-size: 26px;
    font-weight: 700;
    margin-bottom: 8px;
    color: #38bdf8;
}
.hero-subtitle {
    font-size: 15px;
    line-height: 1.5;
    color: #cbd5e1;
    margin-bottom: 16px;
}
.cards-row {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin-top: 12px;
}
.card-item {
    flex: 1;
    min-width: 220px;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 8px;
    padding: 12px 16px;
}
.card-item h4 {
    margin: 0 0 4px 0;
    font-size: 14px;
    color: #f1f5f9;
}
.card-item p {
    margin: 0;
    font-size: 12px;
    color: #94a3b8;
}
.tip-banner {
    background-color: #f8fafc;
    border-left: 4px solid #3b82f6;
    padding: 10px 14px;
    border-radius: 4px;
    margin-bottom: 14px;
    font-size: 14px;
}
"""

with gr.Blocks(title=title) as demo:
    # HERO HEADER & STYLING
    gr.HTML(f"""
    <style>{custom_css}</style>
    <div class="hero-box">
        <div class="hero-title">⚡ Gevva: Instant AI Decision Engine</div>
        <div class="hero-subtitle">
            Think of Gevva as an <strong>instant reflex engine</strong> for AI. While generative models like ChatGPT slowly type words token-by-token (taking seconds), Gevva makes <strong>split-second decisions in ~15 milliseconds</strong> (100x faster). It verifies facts, catches hallucinations, inspects charts, routes user requests, and grades answers with calibrated certainty.
        </div>
        <div class="cards-row">
            <div class="card-item">
                <h4>⚡ 15 Millisecond Reflex</h4>
                <p>Non-autoregressive forward pass. No typing delay or token stutter.</p>
            </div>
            <div class="card-item">
                <h4>👁️ Vision & Text Combined</h4>
                <p>Verifies financial charts, supply tables, and invoices alongside text.</p>
            </div>
            <div class="card-item">
                <h4>🏆 77.54 on JevBench Public</h4>
                <p>Scored 77.54 composite score on the open JevBench public evaluation suite.</p>
            </div>
        </div>
    </div>
    """)

    with gr.Row():
        model_selector = gr.Dropdown(
            choices=MODEL_OPTIONS,
            value="davidburhans/gevva-e2b-multimodal",
            label="🤖 Model Engine",
            info="Choose 'gevva-e2b-multimodal' for text + images/charts, or 'gevva-e2b' for text-only.",
            scale=3,
        )

    with gr.Tabs():
        # TAB 1: Fact Checker & Truth Detective
        with gr.TabItem("🔍 Fact Checker & Truth Detective"):
            gr.HTML("""
            <div class="tip-banner">
                <strong>How this works:</strong> Provide a source document or upload an image (chart, receipt, invoice, photo). Then enter a statement to test. Gevva will instantly tell you if the statement is <strong>🟢 TRUE (Supported)</strong>, <strong>🔴 FALSE (Contradicted / Hallucinated)</strong>, or <strong>🟡 UNCLEAR (Not enough info)</strong>.
            </div>
            """)
            with gr.Row():
                with gr.Column(scale=5):
                    nli_premise = gr.Textbox(
                        label="1. Source Evidence / Document (Text)",
                        placeholder="Paste an article paragraph, contract clause, medical trial note, or company policy...",
                        lines=3,
                    )
                    nli_image = gr.Image(
                        label="Context Image (Optional: Chart, Invoice, Table, or Scene)",
                        type="pil",
                    )
                    nli_hyp = gr.Textbox(
                        label="2. Statement to Verify (Is this True or False based on the evidence?)",
                        placeholder="e.g. 'The contract renews automatically' or 'Q3 revenue was higher than Q4'",
                        lines=2,
                    )
                    btn_predict = gr.Button("⚡ Check Truth & Facts", variant="primary", size="lg")
                with gr.Column(scale=5):
                    out_verdict = gr.Markdown("### 🔍 Verdict: Ready for input — click an example below to try!")
                    out_probs = gr.Label(label="Confidence Breakdown", num_top_classes=3)
                    out_latency = gr.Markdown("⏱️ Latency: --")

            gr.Markdown("#### 💡 Click any real-world example to test instantly:")
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

        # TAB 2: Smart Assistant: Action Routing
        with gr.TabItem("⚡ Smart Assistant: Which Action to Take?"):
            gr.HTML("""
            <div class="tip-banner">
                <strong>How this works:</strong> AI agents often need to decide what to do next when a user talks to them. Instead of waiting for a slow generative model to write out thoughts, Gevva looks at the incoming request and instantly picks the best tool or action in <strong>15 milliseconds</strong>.
            </div>
            """)
            with gr.Row():
                with gr.Column(scale=5):
                    route_query = gr.Textbox(
                        label="User Request or Customer Message",
                        value="The package never arrived at my doorstep, give me back the money you charged me.",
                        lines=2,
                    )
                    route_image = gr.Image(
                        label="Context Image (Optional)",
                        type="pil",
                    )
                    route_tools = gr.Textbox(
                        label="Available Tools / Actions (One action per line)",
                        value="reverse_settled_transaction: Issue credit adjustment and wire funds to cardholder bank account\ntrack_carrier_dispatch: Query live GPS coordinates and logistics milestone for courier vehicle\nterminate_membership: End active recurring subscription privileges at end of current billing cycle\nquery_knowledge_base: Search customer help documentation and policy articles",
                        lines=5,
                    )
                    btn_route = gr.Button("🎯 Choose Best Action", variant="primary", size="lg")
                with gr.Column(scale=5):
                    out_route = gr.Markdown("### 🎯 Routing Result: Ready for input")
                    out_route_lat = gr.Markdown("⏱️ Latency: --")

            gr.Markdown("#### 💡 Click any real-world scenario to test:")
            gr.Examples(
                examples=[
                    [
                        "davidburhans/gevva-e2b",
                        "The package never arrived at my doorstep, give me back the money you charged me.",
                        "reverse_settled_transaction: Issue credit adjustment and wire funds to cardholder bank account\ntrack_carrier_dispatch: Query live GPS coordinates and logistics milestone for courier vehicle\nterminate_membership: End active recurring subscription privileges at end of current billing cycle\nquery_knowledge_base: Search customer help documentation and policy articles",
                        None,
                    ],
                    [
                        "davidburhans/gevva-e2b-multimodal",
                        "Did our performance improve towards the end of the year or drop off?",
                        "evaluate_growth_trajectory: Assess historical performance deltas and period trends from visual reports\nlog_warehouse_stock: Record manufactured unit counts and physical equipment quantities\nissue_account_credit: Reimburse disputed billing adjustments to a client balance\ndispatch_field_agent: Book an on-site hardware repair visit for technical issues",
                        "examples/sample_chart.jpg",
                    ],
                    [
                        "davidburhans/gevva-e2b-multimodal",
                        "Check whether we have enough replacement pipe fittings and valves on hand for dispatch.",
                        "audit_supply_stock: Record component quantities, verify available hardware units, and update enterprise depot levels\nforecast_quarterly_revenue: Model predictive financial trajectories and commercial earnings\nprocess_chargeback: Reclaim unauthorized customer payments through banking network\nreset_access_credentials: Send one-time authentication link to user workstation",
                        "examples/sample_table.jpg",
                    ],
                    [
                        "davidburhans/gevva-e2b-multimodal",
                        "Identify the arrangement of objects on the table and measure how far apart they are.",
                        "spatial_layout_perception: Calculate Euclidean distances, relative coordinates, and geometric boundaries from visual feed\nreimburse_travel_stipend: Review lodging receipts and wire approved expense disbursements to employee\ngenerate_sql_migration: Create database schema alterations and index definitions for PostgreSQL\nrenew_domain_registration: Check DNS records and execute SSL renewal with registrar",
                        "examples/sample_scene.jpg",
                    ],
                    [
                        "davidburhans/gevva-e2b",
                        "Can a customer bring back an unsealed stereo gadget four weeks after purchase?",
                        "lookup_post_sale_guidelines: Retrieve consumer eligibility terms and timeframe allowances for opened merchandise\nreverse_settled_transaction: Issue credit adjustment and wire funds to cardholder bank account\ntrack_carrier_dispatch: Query live GPS coordinates and logistics milestone for courier vehicle\nterminate_membership: End active recurring subscription privileges at end of current billing cycle",
                        None,
                    ],
                ],
                inputs=[model_selector, route_query, route_tools, route_image],
                outputs=[out_route, out_route_lat],
                fn=route_intent,
                cache_examples=False,
            )
            btn_route.click(
                route_intent,
                inputs=[model_selector, route_query, route_tools, route_image],
                outputs=[out_route, out_route_lat],
            )

        # TAB 3: Instant Homework & AI Grader
        with gr.TabItem("📋 Instant Homework & AI Grader"):
            gr.HTML("""
            <div class="tip-banner">
                <strong>How this works:</strong> Did a student (or another AI) answer a test question correctly? Paste the question, the official answer key, and the student's answer. Gevva evaluates whether the response is factually accurate or contains mistakes in <strong>15 milliseconds</strong>.
            </div>
            """)
            with gr.Row():
                with gr.Column(scale=5):
                    grade_q = gr.Textbox(
                        label="Question / Problem Prompt",
                        value="What is the capital of Australia?",
                        lines=2,
                    )
                    grade_ref = gr.Textbox(
                        label="Official Answer Key / Reference Rubric",
                        value="Canberra",
                        lines=2,
                    )
                    grade_cand = gr.Textbox(
                        label="Candidate Answer to Grade",
                        value="The capital city of Australia is Canberra.",
                        lines=2,
                    )
                    btn_grade = gr.Button("📝 Grade Answer", variant="primary", size="lg")
                with gr.Column(scale=5):
                    out_grade = gr.Markdown("### 📝 Grading Result: Ready for input")
                    out_grade_lat = gr.Markdown("⏱️ Latency: --")

            gr.Markdown("#### 💡 Click an example to test correct vs incorrect grading:")
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
                    [
                        "davidburhans/gevva-e2b",
                        "What causes ocean tides on Earth?",
                        "The gravitational pull of the Moon and the Sun.",
                        "Ocean tides are primarily created by the gravitational pull exerted by the Moon.",
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

        # TAB 4: How It Works & Leaderboard
        with gr.TabItem("🧠 How It Works & Leaderboard"):
            gr.Markdown("""
### 🧠 The "Gut Reflex" (System 1) vs "Deep Thought" (System 2)

Psychologist Daniel Kahneman demonstrated that human thinking operates in two modes:
- **System 1 (Fast & Intuitive)**: Recognizing a friend's face, dodging an incoming ball, or knowing 2+2=4 instantly in a few milliseconds.
- **System 2 (Slow & Deliberate)**: Solving complex calculus, writing an essay, or planning an itinerary step-by-step.

```
+------------------------------------+    +------------------------------------+
|         SYSTEM 1: GEVVA            |    |       SYSTEM 2: CHATGPT / CLAUDE   |
|  - Non-autoregressive decision     |    |  - Autoregressive text generation  |
|  - Evaluates everything in 1 pass  |    |  - Generates token-by-token        |
|  - Latency: ~15 milliseconds       |    |  - Latency: 2,000 - 5,000 ms       |
|  - Cost: 95% cheaper compute       |    |  - Cost: High GPU usage            |
|  - Use: Routing, Hallucination     |    |  - Use: Long essays, creative text |
|         Guardrails, Fact Checking  |    |         and multi-step reasoning   |
+------------------------------------+    +------------------------------------+
```

Most AI agent systems waste huge amounts of time and money calling expensive System 2 models just to make simple decisions (like *"Should I refund this user?"* or *"Did the answer match the source document?"*). **Gevva solves this by acting as the AI's instant reflex.**

---

### 📊 JevBench Public Benchmark Results

Evaluated locally against the open **JevBench Public Dataset** (231 evaluation tasks):

| Model | Parameters | Decision Latency | Public Composite Score | Status |
| :--- | :---: | :---: | :---: | :---: |
| **`Gevva e2b`** | **2.3B** | **14.3 ms** | **`77.54`** | **Open Source (Apache 2.0)** |
| **`Gevva e4b`** | **4.5B** | **17.8 ms** | **`77.28`** | **Open Source (Apache 2.0)** |
| OpenJEV (AlexWortega) | 2.6B | 18.2 ms | `76.01` | Competitor |
| TypeSafe AI Jev | 2.5B | 15.0 ms | `75.40` | Baseline |
| Convai Laya | 2.2B | 18.4 ms | `73.80` | Baseline |

> *Note: Evaluated against the open 231-item public split of JevBench. Not an official claim on the full private benchmark suite until verified.*

---
""")

            with gr.Accordion("🔬 For Machine Learning Engineers & Data Scientists (Technical Specs & SDK)", open=False):
                gr.Markdown("""
#### Technical Architecture
- **Backbone Architecture**: Google Gemma 4 (`google/gemma-4-E2B-it`), 2.3B effective parameters.
- **Vision Encoder**: Google SigLIP tower (frozen during cross-encoder classification tuning).
- **Context Budget**: Native 128,000 token Rotary Position Embeddings (RoPE).
- **Pooling & Classification**: Last non-pad token pooling over backbone hidden state with linear projection head to 3 calibrated logits (0=Contradiction, 1=Entailment, 2=Neutral).
- **Calibration**: Temperature-scaled ($T^* = 1.60$) Expected Calibration Error (ECE) compressed to **0.0655**.

#### Python SDK Usage
Install the official package from PyPI:
```bash
pip install gevva
```

Run instant predictions in Python:
```python
from gevva import load

# Load the champion model (runs on GPU or CPU)
engine = load("davidburhans/gevva-e2b")

# 1. Fact checking / Hallucination verification
probs = engine.predict([("The sky is blue today.", "The sky is blue.")])
print(probs)  # [p_contradiction, p_entailment, p_neutral]

# 2. Tool & workflow routing
best_idx, scores = engine.rerank(
    premise="Please refund my credit card.",
    options=["process_refund", "track_shipment", "cancel_account"]
)
print("Chosen tool:", ["process_refund", "track_shipment", "cancel_account"][best_idx])

# 3. Answer grading
grade = engine.grade(
    question="What is 2+2?",
    reference="4",
    candidate="The answer is 4."
)
print("Is Correct:", grade.is_correct, "Score:", grade.score)
```

- **Hugging Face Model Hub**:
  - Text Flagship: [`davidburhans/gevva-e2b`](https://huggingface.co/davidburhans/gevva-e2b)
  - Vision & Multimodal: [`davidburhans/gevva-e2b-multimodal`](https://huggingface.co/davidburhans/gevva-e2b-multimodal)
  - Training Dataset: [`davidburhans/gevva-decisions`](https://huggingface.co/datasets/davidburhans/gevva-decisions)
- **GitHub Repository**: [https://github.com/davidburhans/gevva](https://github.com/davidburhans/gevva)
""")

if __name__ == "__main__":
    demo.launch()
