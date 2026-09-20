# Vision-Enabled NLI: Dataset Engineering, Task Formulations, and Gemma 4 Token Budgeting

**Author:** Vision NLI Specialist  
**Target Architecture:** Google Gemma 4 (E2B / E4B Edge Backbones, 128K Context Window)  
**Report Series:** Multimodal Cross-Encoder Research — Report 02  
**Date:** September 2026  
**Reference Code:** [`research/adapters/multimodal_nli_adapter.py`](file:///home/dave/workspaces/nli-cross-encoder/research/adapters/multimodal_nli_adapter.py)  

---

## Executive Summary

Natural Language Inference (NLI) cross-encoders have emerged as the foundational primitive for fast, calibrated, non-autoregressive **System 1 decision-making**. While first-generation cross-encoders (such as `ModernCE-large-nli` and `convaiinnovations/laya`) established sub-40ms text classification over 512–1024 tokens, and OpenJEV demonstrated the feasibility of turning decoder backbones (`Qwen3.5-0.8B/2B/4B`) into multimodal game-playing cross-encoders, the next frontier requires:
1. **Long-context multi-page reasoning** (scaling beyond short text snippets to entire 50-page financial reports, legal filings, and technical manuals).
2. **Native multilingual multimodal alignment** (evaluating visual and textual claims seamlessly across 100+ languages).
3. **Flexible vision token budgeting with aspect-ratio preservation**, eliminating square distortion while controlling computational overhead.

This technical report provides the comprehensive mathematical, algorithmic, and data-engineering specification for building the **Vision-Enabled NLI Cross-Encoder based on Google Gemma 4** (specifically the edge-optimized E2B and E4B tiers). We establish the formal multimodal NLI framework, deconstruct OpenJEV's vision implementation, detail the conversion of existing multimodal corpora (SNLI-VE, VQAv2, GQA, DocVQA, ChartQA), engineer negative and neutral mining pipelines, analyze Gemma 4's 150M vision encoder and 128K token context window economics, and deliver a production-grade reference data adapter in [`research/adapters/multimodal_nli_adapter.py`](file:///home/dave/workspaces/nli-cross-encoder/research/adapters/multimodal_nli_adapter.py).

---

## 1. Multimodal NLI Task Formulation

### 1.1 Mathematical Framework

In standard textual NLI, a model evaluates the directional relationship between a textual premise $T_P$ and a textual hypothesis $T_H$. In **Visual / Multimodal NLI (V-NLI)**, the premise is generalized to include arbitrary visual and textual context:

$$\mathcal{P} = \left( \{\mathcal{I}_k\}_{k=1}^K, \mathcal{T}_P \right)$$

where:
- $\mathcal{I}_k \in \mathbb{R}^{3 \times H_k \times W_k}$ represents image frame or document page $k$ ($K \ge 1$).
- $\mathcal{T}_P$ is an optional textual context (e.g., surrounding conversation turns, extracted OCR layout text, document metadata, or agent state logs).
- The **Hypothesis** $\mathcal{H} = \mathcal{T}_H$ is strictly a declarative natural-language proposition whose factual truth value is conditioned on $\mathcal{P}$.

The cross-encoder maps $(\mathcal{P}, \mathcal{H})$ into a calibrated probability distribution over the discrete 3-class label space $\mathcal{Y} \in \{0, 1, 2\}$:

$$P(Y = y \mid \mathcal{P}, \mathcal{H}) = \text{softmax}\left( \mathbf{W}_s \cdot \mathbf{h}_{\text{pooled}} + \mathbf{b}_s \right)$$

where $\mathbf{h}_{\text{pooled}} \in \mathbb{R}^d$ is the pooled representation of the joint sequence (typically extracted from the final non-pad token or a dedicated classification marker).

```
+-----------------------------------------------------------------------------------+
|                                   PREMISE P                                       |
|  +---------------------------+  +-----------------------------------------------+ |
|  | Image(s) I_1 ... I_K      |  | Text Context T_P                              | |
|  | [Page 1 / Chart / Frame]  |  | "Invoice #9821, issued March 12, 2026..."     | |
|  +---------------------------+  +-----------------------------------------------+ |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                                 HYPOTHESIS H                                      |
|  "The total amount due on the invoice is $1,450.00."                              |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                    GEMMA 4 MULTIMODAL CROSS-ENCODER BACKBONE                      |
|                  (Joint Multimodal Attention over 128K Tokens)                    |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                         CALIBRATED 3-WAY DECISION HEAD                            |
|       Contradiction (0)        |        Entailment (1)        |    Neutral (2)    |
|             0.02               |             0.97             |       0.01        |
+-----------------------------------------------------------------------------------+
```

### 1.2 Standardized Label Space and Semantic Definitions

To ensure complete interoperability across ModernCE, OpenJEV, and Laya, we adopt the standard label ordering:

$$\mathcal{Y} = \{0: \text{Contradiction}, \; 1: \text{Entailment}, \; 2: \text{Neutral}\}$$

The semantics of these three classes in multimodal domains are defined as follows:

| Label ID | Name | Semantic Definition in Multimodal Reasoning | Visual Grounding Condition |
| :---: | :---: | :--- | :--- |
| **0** | **Contradiction** | The hypothesis is **provably false** given $\mathcal{P}$. The visual or textual premise directly refutes the claim. | Conflicting attributes (e.g. car is blue, claim says red), count mismatch ($n \ne n_{\text{claim}}$), spatial impossibility, arithmetic discrepancy in invoices. |
| **1** | **Entailment** | The hypothesis is **provably true** given $\mathcal{P}$. The visual and textual evidence is sufficient to establish truth beyond reasonable doubt. | Visual entities, spatial relationships, tabular cells, or textual extractions directly corroborate the proposition. |
| **2** | **Neutral** | The hypothesis is **neither provably true nor provably false** given $\mathcal{P}$. The claim is possible or plausible, but information is missing, occluded, or unstated. | Visual attributes outside the camera frustum, historical/future events, internal beliefs, occluded document clauses, ambiguous low-resolution details. |

> [!IMPORTANT]
> **The Neutral vs. Contradiction Boundary in Vision:**  
> A frequent failure mode in multimodal dataset conversion is collapsing "unseen" into "contradiction". For instance, if an image shows a dog running in a park and the hypothesis is *"The dog's owner is an architect"*, this must **never** be labeled Contradiction. The claim is plausible but unverifiable from pixels alone, making it strictly **Neutral**. Conversely, *"The dog is an adult elephant"* is a **Contradiction** because the visual evidence directly refutes elephant morphology.

---

### 1.3 Deep Investigation of OpenJEV's Vision Implementation

In OpenJEV (`research/openjev/data_mix.py`, `eval_image_nli.py`, and `train.py`), vision was integrated into a Qwen3.5 decoder backbone to evaluate zero-shot game policies and visual entailment. Analyzing its architecture, conversion heuristics, and runtime optimizations provides crucial engineering lessons for our Gemma 4 system.

#### 1. VQA-to-Declarative Statement Engine (`q_to_statement`)
In [`research/openjev/data_mix.py` (lines 181–210)](file:///home/dave/workspaces/nli-cross-encoder/research/openjev/data_mix.py#L181-L210), OpenJEV transformed (Question, Answer, Answer_Type) triples into declarative hypotheses:
- **Color Queries:** `"what colour is the car"` + `"red"` $\to$ `"The car is red."` (regex pattern: `^what colou?r (?:is|are) (?:the )?(.+)$`).
- **Counting Queries:** `"how many dogs are there"` + `"2"` $\to$ `"There are two dogs in the image."` using a word-mapping table (`NUM_WORDS = {"0": "no", "1": "one", ...}`).
- **Auxiliary Verb Yes/No Queries:** `"is the man riding a bike"` + `"yes"` $\to$ `"The man is riding a bike."` (with subject-auxiliary inversion).
- **Polarity Inversion (`flip_ok`):** For boolean yes/no queries, if the answer is `"no"`, the exact same declarative sentence is assigned label `0` (**Contradiction**), providing natural hard negatives.

#### 2. Annotator Disagreement Heuristic for Genuine Neutrality
A major challenge in VQA conversion is that standard VQA has no native "Neutral" category. OpenJEV solved this by mining annotator entropy:
```python
# data_mix.py line 307:
answers = [x["answer"] if isinstance(x, dict) else x for x in (ex.get("answers") or [])]
agree = sum(1 for x in answers if str(x).strip().lower() == a.lower())
if agree and agree < 6:  # Fewer than 6 of 10 annotators agree
    p.add(f"{lead} {IMG}", stmt, OURS["neutral"], "vqa_disagree", rel)
```
When human annotators disagree on a visual fact, the image contains genuine visual ambiguity (e.g. heavy shadows, occlusion, distance), making the claim objectively unverifiable.

#### 3. Spatial Reasoning and Bounding Box Invariants
In [`research/openjev/data_mix.py` (lines 212–269)](file:///home/dave/workspaces/nli-cross-encoder/research/openjev/data_mix.py#L212-L269), OpenJEV ingested DETA bounding boxes (`[x1, y1, x2, y2]`) and synthesized fine-grained spatial and count claims:
- **Third-of-Image Partitioning:**
  $$\text{third}(c_x) = \begin{cases} \text{left} & \text{if } c_x < W/3 \\ \text{middle} & \text{if } W/3 \le c_x \le 2W/3 \\ \text{right} & \text{if } c_x > 2W/3 \end{cases}$$
  Generates paired hypotheses: *"There is a bicycle in the left third"* (**Entailment**) vs. *"There is a bicycle in the right third"* (**Contradiction**).
- **Pixel-Coordinate Framing:** Used in the zero-shot Doom game policy:
  - $c_x < 110 \implies$ *"The monster is at x < 110, on the left."* (**Entailment**)
  - $c_x > 210 \implies$ *"The monster is at x > 210, on the right."* (**Entailment**)
  - Center around $x=160 \implies$ *"The monster is near the centre, around x = 160."* (**Entailment**)
- **Object Counting:** Actual count $n$ generates Entailment; synthetic distractor $n + 2$ generates Contradiction.
- **Absent Object Distractors:** Sampling non-existent classes from COCO (e.g. `giraffe`, `helicopter`, `traffic light`) to create absolute visual contradictions.
- **Unverifiable Attributes:** Synthetic hypotheses like *"The bicycle was bought last week"* were assigned **Neutral**, training the model not to hallucinate visual support for temporal claims.

#### 4. Critical Engineering Optimizations in Training & Evaluation
OpenJEV revealed three vital engineering requirements for multimodal cross-encoders:

```
FastPatchEmbed Workaround:
-------------------------
Standard Conv3d in cuDNN under bf16: ~2,000 ms / frame (cuDNN kernel search bug)
FastPatchEmbed in float32 without autocast: ~0.3 ms / frame (6,600x speedup!)
```

- **`FastPatchEmbed` (`train.py` lines 65–78):** Qwen3.5's 3D convolution patch embed layer suffers from a cuDNN pathology under `bfloat16` autocast, inflating latency to ~2 seconds per image. Casting weights and inputs to `float32` during convolution reduced latency to **0.3 ms per image**—a **6,600× speedup**.
- **Vision Tower Freezing (`train.py` line 80):** Keeping the visual backbone frozen and wrapping its forward pass in `torch.no_grad()` eliminated gradient checkpointing overhead on vision blocks, saving over 40% VRAM during training.
- **`add_smoke_callback` Blank-Pixel Stress Test (`train.py` lines 158–217):** To verify genuine multimodal integration and detect cross-sample attention leakage, OpenJEV ran an active diagnostic callback:
  1. Forward pass with original image batch $\to \mathbf{L}$.
  2. Forward pass with all-zero blank pixels $\to \mathbf{L}_0$.
  3. Image rows must shift significantly: $\Delta_{\text{img}} = \max |\mathbf{L}_{\text{img}} - \mathbf{L}_{0,\text{img}}| > 10^{-2}$.
  4. Pure-text rows in the same batch must remain identical: $\Delta_{\text{txt}} = \max |\mathbf{L}_{\text{txt}} - \mathbf{L}_{0,\text{txt}}| < 10^{-3}$.

#### 5. Why State Verification Wins: The Flappy Bird and Doom Breakthrough
OpenJEV demonstrated that cross-encoders fail when asked to *generate actions* (e.g., hypothesis *"The correct action is: turn left"* yielded 1.0 kills in Doom, identical to random chance). But when asked to *verify physical state claims* (hypothesis *"The nearest enemy is to the left of the crosshair"* $\to$ bind to `turn left`), the zero-shot cross-encoder achieved **11.0 kills** from text state and **5.2 kills directly from raw pixels** without any game-specific fine-tuning.

This demonstrates that **NLI cross-encoders are natural state verifiers, not generative deciders**.

---

## 2. Multimodal Datasets for NLI

To construct a robust, production-grade training mixture for Gemma 4, we examine four core multimodal dataset families and formulate explicit transformation algorithms.

```
+----------------------------------------------------------------------------------------------------+
|                                    MULTIMODAL DATASET INGESTION                                    |
|                                                                                                    |
|  +--------------------+   +--------------------+   +---------------------+   +-------------------+ |
|  |     SNLI-VE        |   |    VQA v2 / GQA    |   |  DocVQA / InfoVQA   |   | ChartQA / PlotQA  | |
|  | (565k Pairs, Flickr|   | (Real-World Scenes |   | (Invoices, Tables,  |   | (Data Trends,     | |
|  |  Direct Visual NLI)|   |   & Scene Graphs)  |   |  Financial Reports) |   |  Visual Statistics| |
|  +--------------------+   +--------------------+   +---------------------+   +-------------------+ |
|            |                        |                         |                        |           |
|            v                        v                         v                        v           |
|    Label Normalizer         Declarative Parser        Document OCR Merger       Numerical Mutator  |
|  {E:1, C:0, N:2}        + Negative Sampler        + Tabular Formatter       + Trend Inverter   |
|            |                        |                         |                        |           |
+------------+------------------------+-------------------------+------------------------+-----------+
                                                  |
                                                  v
+----------------------------------------------------------------------------------------------------+
|                         UNIFIED MULTIMODAL NLI TRAINING MIXTURE (BALANCED 1:1:1)                   |
|                       {Premise: <<IMG>>, Hypothesis: "...", Label: {0, 1, 2}}                      |
+----------------------------------------------------------------------------------------------------+
```

### 2.1 SNLI-VE (Visual Entailment)

SNLI-VE (Xie et al., 2019) is the foundational visual entailment benchmark, replacing the text premise of SNLI with Flickr30k photographs while retaining the human-annotated hypotheses.

#### Dataset Statistics and Splits

| Split | Images Count | Entailment (1) | Contradiction (0) | Neutral (2) | Total Instances |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Train** | 29,783 | 176,932 (33.4%) | 176,550 (33.3%) | 176,045 (33.2%) | **529,527** |
| **Dev / Val** | 1,000 | 5,959 (33.4%) | 5,939 (33.3%) | 5,960 (33.4%) | **17,858** |
| **Test** | 1,000 | 5,973 (33.4%) | 5,964 (33.3%) | 5,964 (33.3%) | **17,901** |
| **Total** | **31,783** | **188,864** | **188,453** | **187,969** | **565,286** |

#### Label Normalization Mapping
SNLI-VE records labels as string literals `"entailment"`, `"contradiction"`, and `"neutral"`.
```python
SNLI_VE_MAP = {
    "contradiction": 0,  # False claim
    "entailment": 1,     # Supported by image
    "neutral": 2         # Cannot be determined
}
```

#### Known Vulnerabilities and Quality Guardrails
- **Linguistic Bias Shortcuts:** Early models achieved ~68% accuracy on SNLI-VE without ever looking at the image, relying on hypothesis lexical cues (e.g. *"nobody"* $\to$ Contradiction, *"outside"* $\to$ Entailment).
- **Mitigation:** During training, we enforce **Hypothesis-Only Regularization** (computing loss on hypothesis-only inputs and penalizing confident predictions when premise is stripped) and apply **Image Shuffling** (pairing the hypothesis with an unrelated image; if the model still predicts Entailment, a high penalty is incurred).

---

### 2.2 Transforming VQA Corpora to NLI (VQAv2, GQA, InfoVQA)

Visual Question Answering corpora contain rich multimodal interactions but are structured as $(I, Q, A)$. Converting them into NLI pairs requires translating interrogative triples into declarative claims while synthesizing negative and neutral counterparts.

#### 1. Rule-Based vs. LLM-Based Transformation Pipeline

```
Rule-Based vs. LLM Synthesis Trade-offs:
----------------------------------------
Rule-Based:
  [+] Zero latency, zero cost, deterministic, no hallucination risk.
  [-] Grammatical rigidity; fails on complex compositional questions ("Why is the man...").
LLM-Based (e.g. Gemma 4 / Gemini batch distillation):
  [+] Natural phrasing, handles complex clauses, preserves subtle semantics.
  [-] Requires batch inference compute; risk of ungrounded factual hallucinations.
```

Our recommended strategy is a **hybrid cascade**:
1. Run high-precision **rule-based transformations** on regex-matched question templates (~65% of VQAv2/GQA).
2. Route long-tail or compositional queries through an **offline LLM batch converter** using strict few-shot verification prompts.

#### 2. Declarative Conversion Prompt Template for LLM Distillation
```markdown
You are a precise multimodal dataset converter. Transform the following (Question, Answer) pair 
into a single, self-contained declarative statement about the image.
Rules:
1. Do not use pronouns referring to unstated entities (e.g., use "the athlete" instead of "he").
2. State the fact positively and neutrally.
3. Keep the statement atomic and factual.

Question: {question}
Answer: {answer}
Declarative Claim:
```

#### 3. GQA Compositional Scene-Graph Conversion
GQA provides clean functional scene-graph representations. We exploit these graph relations to build rigorous entailment claims:
- **Attribute Grounding:** `(car, color, red)` $\to$ *"The car in the driveway is red."* (**Entailment**) vs. *"The car in the driveway is silver."* (**Contradiction**).
- **Spatial Relations:** `(chair, left_of, table)` $\to$ *"The wooden chair is to the left of the dining table."* (**Entailment**).
- **Object Counting:** Ground-truth instance IDs allow exact counting without detection ambiguity.

---

### 2.3 Document Understanding & Chart QA as NLI

For enterprise workflows (invoices, receipts, SEC 10-K filings, scientific charts), NLI cross-encoders provide rapid fact-checking and automated compliance grading.

#### Document Claim Taxonomy

```
+-------------------------------------------------------------------------------------------------+
|                                 DOCUMENT NLI CLAIM CATEGORIES                                   |
|                                                                                                 |
|  +--------------------+   +--------------------+   +---------------------+   +----------------+ |
|  |  Key-Value Lookups |   |   Table Cross-Check|   | Chart Trends & Extr |   | UI / App State | |
|  | "Invoice date is   |   | "Total equals sum  |   | "Q3 revenue grew by |   | "Submit button | |
|  |  2026-03-12."      |   |  of line items."   |   |  15% over Q2."      |   |  is disabled." | |
|  +--------------------+   +--------------------+   +---------------------+   +----------------+ |
+-------------------------------------------------------------------------------------------------+
```

1. **Key-Value Extractions (DocVQA / SROIE / FUNSD):**
   - Premise: Document image $\mathcal{I}_0$ + OCR bounding box text.
   - Hypothesis: *"The vendor name on the receipt is Starbucks Coffee."*
   - Label: Entailment (if matches), Contradiction (if vendor altered to Pete's Coffee).
2. **Tabular Arithmetic & Bounds (TAT-QA / FinQA):**
   - Premise: High-resolution crop of financial table.
   - Hypothesis: *"Operating expenses for FY2025 exceeded $4.2M."*
   - Label: Entailment / Contradiction based on exact cell calculations.
3. **Chart QA (ChartQA / PlotQA):**
   - Premise: Rendered chart image.
   - Hypothesis: *"The blue bar representing 2024 is taller than the orange bar representing 2023."*
   - Evaluates visual trend extraction, axis interpolation, and relative scale comparison.
4. **UI Screenshot State Verification (Mind2Web / Screen2Words):**
   - Premise: Mobile/Web UI screenshot.
   - Hypothesis: *"The password input field is currently focused and empty."*
   - Directly usable as a visual policy verifier for web and desktop automation agents.

---

### 2.4 Negative Mining and Genuine Neutral Formulation

A naive VQA conversion generates exclusively positive (Entailment) pairs. To build an un-biased 3-way cross-encoder, we must synthesize rigorous **Contradictions** and genuine **Neutrals**.

```
+---------------------------------------------------------------------------------------------------+
|                                    DATA AUGMENTATION STRATEGIES                                   |
|                                                                                                   |
|  [ CONTRADICTION GENERATION ]                                     [ NEUTRAL GENERATION ]          |
|  * Semantic Distractor Sampling (pool of same question type)      * Unverifiable Temporal/Past    |
|  * Numerical & Currency Perturbation (amount * 1.25)              * Annotator Entropy (agree < 6) |
|  * Spatial Inversion ("left" -> "right")                          * Unstated Document Clauses     |
|  * Hard Negative Image Swapping (CLIP cosine sim > 0.85)          * Field-of-View Occlusion       |
+---------------------------------------------------------------------------------------------------+
```

#### Mining Hard Contradictions
1. **Semantic In-Domain Distractors:** In VQAv2, sample answers from the same `answer_type` pool (e.g. for color question with answer `"yellow"`, sample `"purple"`).
2. **Numerical and Currency Perturbations:** For invoices and charts, mutate amounts by $\pm 15\%$, swap currencies ($\$ \to €$), or transpose digits (`$1,450.00` $\to$ `$1,540.00`).
3. **Spatial and Relational Flipping:** Invert spatial propositions: swap *"left of"* $\leftrightarrow$ *"right of"*, *"above"* $\leftrightarrow$ *"below"*, *"inside"* $\leftrightarrow$ *"outside"*.
4. **Hard Negative Cross-Sample Retrieval:** Using a multimodal embedding model (e.g. SigLIP), retrieve images with high visual similarity ($\cos \theta > 0.85$) that depict different entities, producing challenging visual negatives.

#### Formulating Genuine Neutrals
1. **Unverifiable Real-World Attributes:** Attributes that cannot be established from a static 2D photograph (e.g., owner identity, purchase date, price, internal thoughts):
   - Hypothesis: *"The laptop was purchased at Best Buy last November."* $\to$ **Neutral**.
2. **Annotator Entropy (`agree < 6 / 10`):** Reflects ambiguous visual signals, low contrast, or subjective interpretation.
3. **Unmentioned Document Clauses:** In invoices and receipts, generate hypotheses regarding unstated terms (e.g., payment via cryptocurrency, board approval sign-off, delivery method).
4. **Field-of-View / Occlusion Truncation:** If an object is partially cut off at the edge of the frame, claims about its truncated portion are strictly **Neutral**.

---

## 3. Resolution, Aspect Ratio, and Token Budgeting in Gemma 4

### 3.1 Gemma 4 Vision Encoder Architecture

The Gemma 4 architecture introduces significant multimodal upgrades tailored for efficient edge deployment and long-context processing:

```
Gemma 4 Vision Processing Pipeline:
===================================
Input Image (W_orig x H_orig)
       |
       v
[Aspect-Ratio Preserving Grid Search] ---> (h_grid, w_grid) s.t. h_grid * w_grid <= Budget
       |
       v
[Bicubic Resampling] --------------------> Resized Image: (w_grid * 48) x (h_grid * 48) px
       |
       v
[Patch Projection (16x16 Conv)] ---------> Spatial Feature Map (H/16 x W/16)
       |
       v
[Spatial Pooling (3x3 Kernel, s=3)] -----> Soft Visual Tokens: (h_grid x w_grid) Tokens
       |
       v
[Linear Projection to Backbone Dim] -----> Injected into Gemma 4 128K Context Stream
```

#### Technical Specifications

| Parameter | Specification | Engineering Implication |
| :--- | :--- | :--- |
| **Vision Tower Size** | ~150M Parameters (SigLIP-derived) | Lightweight forward pass; easily frozen or quantized. |
| **Base Patch Size** | $16 \times 16$ pixels | Fine-grained initial spatial receptive field. |
| **Spatial Pooling Kernel** | $3 \times 3$ with stride 3 | $9\times$ spatial compression factor. |
| **Effective Block Size** | **$48 \times 48$ pixels** ($16 \times 3$) | Both $H$ and $W$ of resized image **must be divisible by 48**. |
| **Image Normalization** | Internal scaling to $[-1.0, 1.0]$ | No external ImageNet mean/std required; handles dark/bright modes robustly. |
| **Soft Token Budgets** | **70, 140, 280, 560, 1120 tokens** | Configurable per-task balance between resolution and compute. |
| **Context Window** | **128,000 Tokens (E2B / E4B)** | Native support for multi-image and long-document reasoning. |

---

### 3.2 Dynamic Aspect-Ratio Grid Calculations

Rather than squashing non-square images into a fixed square (which distorts text aspect ratios and introduces spatial artifacts), Gemma 4 allocates visual tokens across a 2D grid $(h_{\text{grid}}, w_{\text{grid}})$ that preserves the natural aspect ratio $r = W_{\text{orig}} / H_{\text{orig}}$.

#### Mathematical Formulation
Given target token budget $N_{\text{budget}} \in \{70, 140, 280, 560, 1120\}$:
1. Ideal unconstrained grid dimensions:
   $$h_{\text{ideal}} = \sqrt{\frac{N_{\text{budget}}}{r}}, \quad w_{\text{ideal}} = \sqrt{N_{\text{budget}} \cdot r}$$
2. Optimal integer grid $(h^*, w^*)$ search:
   $$\arg\min_{h, w} \left| \ln\left(\frac{w / h}{r}\right) \right| - \lambda \frac{h \cdot w}{N_{\text{budget}}} \quad \text{s.t.} \quad h \cdot w \le N_{\text{budget}}$$
   where $\lambda \approx 0.25$ encourages full token budget utilization.
3. Model input pixel dimensions:
   $$H_{\text{px}} = 48 \cdot h^*, \quad W_{\text{px}} = 48 \cdot w^*$$

#### Resolution & Token Grid Reference Matrix

The following table demonstrates the exact grids, token allocations, and pixel resolutions computed by our reference engine for standard image aspect ratios:

| Image Format | Aspect Ratio ($W:H$) | Token Budget ($N$) | Grid ($h^* \times w^*$) | Actual Soft Tokens | Input Resolution ($W_{\text{px}} \times H_{\text{px}}$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Square (1:1)** | 1.00 | **70** | $8 \times 8$ | 64 | $384 \times 384$ px |
| **Square (1:1)** | 1.00 | **140** | $11 \times 11$ | 121 | $528 \times 528$ px |
| **Square (1:1)** | 1.00 | **280 (Default)** | $16 \times 16$ | 256 | $768 \times 768$ px |
| **Square (1:1)** | 1.00 | **560** | $23 \times 23$ | 529 | $1104 \times 1104$ px |
| **Square (1:1)** | 1.00 | **1120** | $33 \times 33$ | 1089 | $1584 \times 1584$ px |
| **Landscape (4:3)** | 1.33 | **280** | $14 \times 19$ | 266 | $912 \times 672$ px |
| **Widescreen (16:9)** | 1.78 | **280** | $12 \times 22$ | 264 | $1056 \times 576$ px |
| **Widescreen (16:9)** | 1.78 | **560** | $17 \times 30$ | 510 | $1440 \times 816$ px |
| **Portrait (9:16)** | 0.56 | **560** | $30 \times 17$ | 510 | $816 \times 1440$ px |
| **Document Page (3:4)** | 0.75 | **280** | $19 \times 14$ | 266 | $672 \times 912$ px |
| **Dense Document (3:4)**| 0.75 | **1120** | $38 \times 29$ | 1102 | $1392 \times 1824$ px |

---

### 3.3 Context Window Economics & Long-Context Scaling

Gemma 4's **128K context window** transforms the operational capabilities of cross-encoders, eliminating the severe context truncation that restricted earlier models.

```
Gemma 4 128K Context Window Allocation Scenarios:
=================================================

1. Single-Image VQA / Visual NLI:
   [Img: 280 tok] [Premise Text: 120 tok] [Hyp: 30 tok] 
   Total: 430 tokens  ==>  0.33% of 128K window!

2. 10-Page Document with Extracted OCR:
   [10 Pages x 280 tok = 2,800 tok] [OCR Text: 12,000 tok] [Hyp: 50 tok]
   Total: 14,850 tokens  ==>  11.6% of 128K window!

3. 50-Page Complex Technical Manual / Financial 10-K:
   [50 Pages x 560 tok = 28,000 tok] [Full Document Text: 45,000 tok] [Hyp: 60 tok]
   Total: 73,060 tokens  ==>  57.1% of 128K window! (55,000 tokens remaining)
```

#### Memory & Latency Analysis for Cross-Encoder Inference
Unlike autoregressive generation (where KV cache must be sustained across hundreds of decode steps), a cross-encoder performs a **single non-autoregressive forward pass**.
- **Attention Complexity:** Gemma 4 employs a hybrid attention architecture (alternating sliding-window local attention and global dense attention).
- **VRAM Footprint on RTX 5090 (32GB VRAM):**
  - Batch size 1 at 16K tokens: ~4.2 GB VRAM.
  - Batch size 1 at 64K tokens: ~11.8 GB VRAM.
  - Batch size 1 at 128K tokens (full window with FlashAttention-3 / Flash-Linear-Attention): ~23.5 GB VRAM.
- This allows full 50-page multimodal cross-encoder grading to execute comfortably on a single workstation GPU.

---

## 4. Master Training Mixture & Curriculum Strategy

### 4.1 Recommended Data Mixture Composition

To prevent catastrophic forgetting of logical reasoning while mastering multimodal and document comprehension, we propose a **balanced multi-source mixture**:

| Data Category | Sources | Share | Sampling Rationale |
| :--- | :--- | :---: | :--- |
| **Core Text NLI** | SNLI, MNLI, ANLI (R1–R3), WANLI, LingNLI | **30%** | Anchors robust propositional logic, negation, and linguistic calibration. |
| **Long-Document Haystack** | Synthetic Haystack (evidence needle at random positions + 30% unstated drops) | **10%** | Trains the cross-encoder to find small needles in 10k–50k token contexts and predict Neutral when absent. |
| **Natural Visual NLI** | SNLI-VE, VQAv2 Declarative, GQA Scene Graphs | **25%** | Real-world visual reasoning, attribute binding, and spatial localization. |
| **Document & Chart NLI** | DocVQA, TAT-QA, ChartQA, PlotQA, Invoices/Receipts | **20%** | OCR alignment, tabular arithmetic, numeric mutation, and unstated clause detection. |
| **Agentic & UI State Verification** | Mind2Web, Screen2Words, When2Call, AgentTraj | **15%** | Verifying interactive UI state, button affordances, and tool parameter validity. |

### 4.2 Three-Stage Training Curriculum

```
+---------------------------------------------------------------------------------------------------+
|                                  THREE-STAGE TRAINING CURRICULUM                                  |
|                                                                                                   |
|   STAGE 1: Text-Only Warmup & Calibration (1 Epoch)                                               |
|   * Train on 100% Text NLI + Haystack                                                             |
|   * Objectives: Align classification head, establish ModernCE-level text baseline (MNLI > 90%)   |
|                                         |                                                         |
|                                         v                                                         |
|   STAGE 2: Multimodal Cross-Modal Alignment (Frozen Vision Tower, 1 Epoch)                        |
|   * Freeze Gemma 4 150M Vision Tower; train projection adapter and backbone cross-attention       |
|   * Run `add_smoke_callback` blank-pixel check to guarantee multimodal gradient flow              |
|                                         |                                                         |
|                                         v                                                         |
|   STAGE 3: Full End-to-End Long-Context Fine-Tuning (LoRA / Full FT, 1 Epoch)                     |
|   * Unfreeze backbone with sliding window attention; train on full 128K multimodal mixture        |
|   * Class balance enforced: 33.3% Contradiction, 33.3% Entailment, 33.3% Neutral                  |
+---------------------------------------------------------------------------------------------------+
```

---

## 5. Reference Implementation Walkthrough

The companion script [`research/adapters/multimodal_nli_adapter.py`](file:///home/dave/workspaces/nli-cross-encoder/research/adapters/multimodal_nli_adapter.py) provides an end-to-end, verified pipeline implementing these capabilities.

### 5.1 Architecture of the Adapter Module

```
research/adapters/multimodal_nli_adapter.py
├── LABEL2ID / ID2LABEL / LABEL_SYNONYMS (Canonical 0=Con, 1=Ent, 2=Neu)
├── Gemma 4 Grid Engine:
│   ├── compute_gemma4_grid(w, h, budget) -> (hg, wg, n_tok, rh, rw)
│   └── preprocess_image_for_gemma4(img, budget) -> (PIL.Image, metadata)
├── Domain Adapters:
│   ├── SNLIVEAdapter: Flickr30k visual grounding & 3-way mapping
│   ├── VQAToNLIAdapter: Question inversion, yes/no polarity, annotator entropy
│   ├── SpatialAndCountAdapter: Thirds, coordinates, counting, absent objects
│   ├── DocVQAAndChartAdapter: Key-value facts, currency mutation, unstated facts
│   └── InterleavedMultiImageAdapter: Multi-page document interleaving
├── PyTorch Gemma4MultimodalNLICollator:
│   ├── Expands <<IMG_k>> with calculated soft tokens (<|image|>*N)
│   ├── Dynamic batch right-padding and attention masking
│   └── Pixel tensor normalisation to [-1, 1]
└── Verification Suite:
    └── run_adapter_self_test(): Validates grid math, sample generation, collation
```

### 5.2 Verification and Self-Test Results

The adapter was executed and verified on our local environment using PyTorch 2.10.0+cu128 and Transformers 5.1.0 on an NVIDIA RTX 5090 GPU:

```bash
/home/dave/workspaces/agent-pump/.venv/bin/python3 research/adapters/multimodal_nli_adapter.py
```

```
======================================================================
RUNNING MULTIMODAL NLI ADAPTER & GEMMA 4 GRID SELF-TEST
======================================================================

--- 1. Gemma 4 Aspect-Ratio Grid Calculations ---
[Square 1:1          ] Orig:  640x640  | Budget:  280 -> Grid: (16, 16) =  256 tokens | Resized:  768x768  px
[Landscape 4:3       ] Orig:  800x600  | Budget:  280 -> Grid: (14, 19) =  266 tokens | Resized:  912x672  px
[Widescreen 16:9     ] Orig: 1920x1080 | Budget:  560 -> Grid: (17, 30) =  510 tokens | Resized: 1440x816  px
[Portrait 9:16       ] Orig: 1080x1920 | Budget:  560 -> Grid: (30, 17) =  510 tokens | Resized:  816x1440 px
[Dense Document 3:4  ] Orig: 1200x1600 | Budget: 1120 -> Grid: (38, 29) = 1102 tokens | Resized: 1392x1824 px
[Thumbnail preview   ] Orig:  320x240  | Budget:   70 -> Grid: ( 7,  9) =   63 tokens | Resized:  432x336  px

--- 2. Generated Multimodal NLI Dataset Samples: 11 ---
Label Distribution:
  0 (contradiction): 4 samples
  1 (entailment   ): 5 samples
  2 (neutral      ): 2 samples

[ID: snli_01] (Source: snli_ve, Label: entailment)
  Premise:    Visual context: <<IMG_0>>...
  Hypothesis: Two dogs are playing in the grass.

[ID: vqa_01_entailment] (Source: vqa_answer_positive, Label: entailment)
  Premise:    A photo: <<IMG_0>>...
  Hypothesis: The car is red.

[ID: spatial_01_spatial_third_ent] (Source: spatial_third, Label: entailment)
  Premise:    Visual premise: <<IMG_0>>...
  Hypothesis: There is a bicycle in the left third of the image.

[ID: spatial_01_spatial_third_con] (Source: spatial_third, Label: contradiction)
  Premise:    Visual premise: <<IMG_0>>...
  Hypothesis: There is a bicycle in the right third of the image.

[ID: spatial_01_count_ent] (Source: object_count, Label: entailment)
  Premise:    Visual premise: <<IMG_0>>...
  Hypothesis: There are two bicycles in the image.

[ID: spatial_01_count_con] (Source: object_count, Label: contradiction)
  Premise:    Visual premise: <<IMG_0>>...
  Hypothesis: There are three bicycles in the image.

--- 3. Testing Gemma 4 Multimodal Collator ---
Collation Output Keys: ['input_ids', 'attention_mask', 'labels', 'pixel_values', 'image_grids']
input_ids shape:      torch.Size([4, 16])
attention_mask shape: torch.Size([4, 16])
labels tensor:        tensor([1, 1, 1, 0]) (['entailment', 'entailment', 'entailment', 'contradiction'])
num image tensors:    4
image_grids shape:    torch.Size([4, 3])
  Image 0: grid (14, 19), tokens = 266
  Image 1: grid (14, 19), tokens = 266
  Image 2: grid (14, 19), tokens = 266
  Image 3: grid (14, 19), tokens = 266

[SUCCESS] All Multimodal NLI Adapter checks passed!
======================================================================
```

---

## 6. Actionable Next Steps & Engineering Roadmap

1. **Pre-Tokenized Dataset Generation:**  
   Deploy `multimodal_nli_adapter.py` on a distributed ray/multiprocessing cluster to ingest full splits of SNLI-VE, VQAv2, GQA, DocVQA, and ChartQA, writing sharded Arrow/Parquet datasets with pre-computed Gemma 4 grids.
2. **Gemma 4 Sequence Classification Model Definition (`modeling_gemma4_nli.py`):**  
   Implement `Gemma4ForSequenceClassification` wrapping the Gemma 4 multimodal backbone with a linear 3-way score head on the final hidden state, integrating `FastPatchEmbed` for instant vision convolution.
3. **Training Launch with Length Grouping:**  
   Execute the Stage 1 $\to$ Stage 2 $\to$ Stage 3 curriculum using `MixTrainer` with native Python `_train_lengths` to prevent Arrow IPC sorting bottlenecks over large mixtures.
4. **Active Smoke Callback Integration:**  
   Incorporate the `add_smoke_callback` diagnostic into early training steps to guarantee visual gradient propagation before executing large-scale runs.

---
*Report completed and filed to `research/reports/02_vision_nli_data_and_tasks.md`.*
