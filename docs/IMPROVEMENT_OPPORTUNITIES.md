# Gevva Architectural & Curriculum Improvement Opportunities

> **Status**: Active Engineering & Research Backlog  
> **Origin**: Empirical audit and benchmark profiling on Decision Index 0.2 (RTX 5090 & Apple M4 Pro).  
> **Last Updated**: 2026-09-25

---

## 1. Executive Summary

This document formalizes the prioritized architectural, algorithmic, and curriculum improvements identified during the evaluation of **Gevva e4b** and **Gevva e2b** across the 44-benchmark **Decision Index 0.2** suite.

While Gevva achieved **#1 among all Gemma 4-based models** on single-turn tool selection (**BFCL: 92.62%**) and tool retrieval (**ToolRet: 43.85% nDCG@10**), long-context and multi-candidate benchmarks (such as **API-Bank: 55.66%**) revealed two critical optimization vectors:
1. **Inference Latency & VRAM**: Causal prefix KV redundancy in multi-candidate scoring.
2. **Task Generalization**: Gap on multi-turn conversational dialogue state tracking compared to dialog-specialized baselines (e.g. Winnow-E4B).

---

## 2. Priority 1: Shared Prefix KV Caching for Multi-Candidate Scoring

### The Problem
In multi-candidate decision problems (e.g., API-Bank with $K=53$ tools, search reranking with $K=100$ items, intent routing across 77 classes), standard cross-encoders construct $K$ distinct premise-hypothesis pairs:
$$\{(\text{Premise}, \text{Option}_1), (\text{Premise}, \text{Option}_2), \dots, (\text{Premise}, \text{Option}_K)\}$$

When the premise is long (in API-Bank, dialogue history + environment averages **6,786 tokens**, truncated to `max_length = 4096`), standard batching evaluates full concatenated sequences:
* At `batch_size = 16`, each batch contains $16 \times 4,096 = 65,536$ tokens, consuming **~14.5 GB of activation memory** and **30.34 GB total VRAM** (98% of the 32 GB RTX 5090).
* A naive batch size of $K=53$ would require $53 \times 4,096 = 217,088$ tokens, demanding **~62 GB VRAM** and causing an immediate CUDA Out-of-Memory (`OOM`) crash.
* Consequently, the engine must split the 53 options into 4 sequential chunks, re-encoding the exact same 4,096-token premise 53 times, inflating request latency to **16.95 seconds**.

### Architectural Insight
Gemma 4 is a **causal decoder transformer**. Attention is lower-triangular:
* Premise tokens at positions $0 \dots L_{\text{premise}}$ only attend backward to preceding premise tokens.
* Premise tokens **never attend forward** to candidate hypothesis tokens.
* Therefore, the Key and Value representations of the premise across all 42 transformer layers are **100% invariant across all $K$ candidates**.

### Target Implementation (`predict_candidates`)
Instead of standard sequence concatenation:
1. **Stage 1 (Single-Pass Premise Pre-Fill)**:
   * Pass the premise through the model once ($B=1, L=4096$) with `use_cache=True`.
   * Cache `past_key_values` (taking ~150 ms and negligible activation memory).
2. **Stage 2 (Parallel Candidate Evaluation)**:
   * Expand the cached KV prefix along the batch dimension to size $K$.
   * Each candidate option is short ($L_{\text{hyp}} \approx 20\text{--}30$ tokens).
   * Forward-pass all $K$ candidates in parallel: total active tokens = $K \times L_{\text{hyp}} \approx 53 \times 25 = \mathbf{1,325\text{ tokens}}$.
   * Activation memory for 1,325 tokens is **< 150 MB** (vs 48 GB).

### Expected Impact
* **VRAM**: Decreases from 48 GB to < 150 MB, easily fitting within 17 GB total footprint.
* **Latency**: Drops from **16.95 seconds down to ~180 ms** (**~90× speedup**).
* **Throughput**: Enables evaluating up to 256 candidate tools simultaneously in a single pass.

---

## 3. Priority 2: Multi-Turn Conversational Dialogue Curriculum

### The Problem
On the Decision Index 0.2 board:
* **Single-Turn Function Calling (BFCL)**: Gevva e4b (**92.62%**) beats Winnow-E4B (**91.44%**).
* **Tool Retrieval (ToolRet)**: Gevva e4b (**43.85%**) beats Winnow-E4B (**41.30%**).
* **Multi-Turn Dialogue (API-Bank)**: Gevva e4b (**55.66%**) trails Winnow-E4B (**73.03%**).

### Root Cause
Gevva's pre-training curriculum was predominantly focused on single-turn factual claims, NLI pairs, document verification, and search queries. In contrast, Winnow-E4B's curriculum included substantial dialogue state tracking (DST) corpora. When faced with 10 back-and-forth conversational turns, Gevva must perform zero-shot conversational history resolution.

### Action Plan & Datasets
Incorporate conversational dialogue state tracking into the Phase 5 training mixture:
1. **Schema-Guided Dialogue (SGD / SGD-X)**: 20,000+ multi-turn dialogues with explicit service/API selection annotations.
2. **MultiWOZ 2.4**: Dialogue state belief tracking across restaurant, hotel, taxi, and train domains.
3. **API-Bank (Training Split)**: Dialogue-grounded API retrieval and parameter identification pairs.
4. **Formatting Alignment**: Use Gemma 4 standard role delimiters:
   ```
   <start_of_turn>user
   Can you book a meeting with John tomorrow at 2pm?<end_of_turn>
   <start_of_turn>model
   Checking calendar availability...<end_of_turn>
   ```

---

## 4. Priority 3: Dynamic Token-Budget Inference Batching

### The Problem
`GevvaEngine` currently uses a static default `batch_size = 16`:
* For long sequences ($L=4096$), $B=16$ pushes VRAM close to the 32 GB limit.
* For short sequences (e.g., $L=200$ tokens in BFCL or intent routing), $B=16$ vastly underutilizes the RTX 5090 (using only 3,200 tokens out of a 65,536-token capacity).

### Action Plan
Implement dynamic token-budget batching in `GevvaEngine` during inference (mirroring the training collator):
$$\text{BatchSize}(L) = \min\left(B_{\max}, \left\lfloor \frac{\text{TokenBudget}}{L} \right\rfloor\right)$$
Where `TokenBudget = 65,536` on 32GB GPUs:
* For $L \le 512$: `batch_size = 128` (4× faster on short benchmarks).
* For $L = 1024$: `batch_size = 64`.
* For $L = 4096$: `batch_size = 16`.

---

## 5. Priority 4: Kernel & Attention Optimizations for RTX 5090 (Blackwell)

1. **FlashAttention-3 / PyTorch SDPA**: Ensure PyTorch 2.5+ SDPA automatically selects the optimal cuDNN / FlashAttention kernel on Blackwell architecture (SM 12.0) to eliminate any memory overhead in long-context classification.
2. **FP8 / W4A16 Quantized Inference Pipeline**: Complete integration of the W4A16 engine into `GevvaEngine` so that Gevva e4b can run at ~4.5 GB VRAM with sub-10ms latency for deployment on edge devices and Apple Silicon Macs.

---

## 6. Priority 5: Thresholded Out-of-Scope (OOS) Intent Routing

For benchmarks and applications involving unknown intents (e.g., Catalog 5: CLINC150+OOS):
* Leverage Gevva's calibrated 3-class distribution:
  $$\text{Class} \in \{\text{Contradiction (0)}, \text{Entailment (1)}, \text{Neutral (2)}\}$$
* When all candidate intents yield $P(\text{entailment}) < \tau$ or $P(\text{neutral}) > \theta$, route directly to `OUT_OF_SCOPE` without requiring a synthetic fallback class.

---

## 7. Empirical Benchmark Variance Analysis

The 12 completed catalogs on Gevva e4b reveal significant variance across problem types:
* In structured schema evaluation, single-turn tool calling, and model routing, Gevva is **world-class (including #1 worldwide on SGD/SGD-X)**.
* In adversarial negation (ANLI), discrete state-machine transitions (Home Appliance), dense legal reasoning (ContractNLI), and dense intent overlapping (BANKING77), performance drops significantly.

### The Divergence Map

| Performance Tier | Benchmark (Catalog) | Gevva e4b Result | Competitor Ceiling | Diagnosis / Missing Muscle |
| :--- | :--- | :---: | :---: | :--- |
| **World Class (#1)** | **SGD / SGD-X (10)** | **65.16%** *(#1 of 58)* | 64.97% *(Jevfire)* | Slot-filling & parameter schemas align natively with factual NLI. |
| **Top Tier (Top 25%)** | **BFCL (1)** | **92.62%** *(#18 of 58)* | 97.99% *(Solomon)* | Excellent single-turn function calling; #1 among all Gemma 4 models. |
| **Top Tier (Top 25%)** | **ToolRet (2)** | **43.85%** *(#15 of 58)* | 46.18% *(AutoJev)* | Strong candidate reranking (+5.3% over BM25 baseline). |
| **Top Tier (Top 25%)** | **RouterBench (6)** | **79.86%** *(#16 of 58)* | 80.07% *(Kev 4B)* | 83.1% Oracle Optimal model selection rate across 10,000 queries. |
| **Mid Tier** | **CLINC150+OOS (5)** | **71.74%** *(#19 of 58)* | 94.68% *(Decider)* | Good zero-shot intent routing across 151 classes; trails specialized intent models. |
| **Mid Tier** | **Humicroedit (21)** | **59.82%** *(#21 of 58)* | 67.88% *(AutoJev)* | Nuanced humor change detection; solid semantic comparison. |
| **Mid Tier** | **BPoMP (20)** | **72.14%** *(#28 of 58)* | 83.84% *(AutoJev)* | Pragmatic metaphor interpretation. |
| **Trailing** | **BANKING77 (4)** | **65.58%** *(#27 of 58)* | 89.79% *(Decider)* | Highly overlapping intents (77 classes); near-synonyms receive high false scores. |
| **Trailing** | **API-Bank (3)** | **49.61%** *(#25 of 58)* | 88.19% *(Jev Ref)* | Multi-turn conversation history (turns 6–10) with authentication token tracking. |
| **Weak Gap** | **ContractNLI (11)** | **54.47%** *(#32 of 58)* | 78.03% *(AutoJev)* | Multi-page dense legal document reasoning; clause cross-referencing. |
| **Severe Blind Spot** | **Home Appliance (9)** | **15.00%** *(#20 of 58)* | 75.00% *(Decider)* | Discrete state-machine transition modeling (FSM states, events, guards). |
| **Severe Blind Spot** | **ANLI R1–R3 (12)** | **31.01%** | 61.05% *(AutoJev)* | Adversarial negations and deceptive word-swap traps; near random chance (33%). |

---

## 8. Prescribed Training Remediations

To systematically eliminate this variance and bring all categories to frontier parity, five specific training interventions are scheduled for the next curriculum iteration:

### TR-01: Adversarial Hard-Anchor Replay & Anti-Negation Contrastive Loss (Fixing ANLI 31.0%)
* **Root Cause**: Catastrophic forgetting of adversarial edge cases during Phase 3/4 decision fine-tuning. The model relies on lexical correlation shortcuts that fail when adversarial negations or deceptive distractors are introduced.
* **Curriculum Fix**:
  1. Mandate a permanent, protected **15% anchor slice** in all fine-tuning mixtures consisting of **ANLI (R1, R2, R3)**, **WANLI**, and **Counterfactually Augmented Data (CAD)**.
  2. Implement **Minimal-Pair Contrastive Loss**: For each premise-hypothesis pair $(P, H)$, generate a counterfactual minimal pair $(P, H')$ where a single token modification (e.g. inserting/removing "not", antonym swap) flips the ground-truth from Entailment $\to$ Contradiction. Penalize representations that assign high similarity to both:
     $$\mathcal{L}_{\text{anti-shortcut}} = \max\left(0, \gamma - |s(P, H) - s(P, H')|\right)$$

### TR-02: Discrete State-Machine Transition Modeling (Fixing Home Appliance 15.0%)
* **Root Cause**: Gevva has seen extensive static premise-hypothesis claims, but zero dynamic state-transition graphs. In Home Appliance Simulation, the model must evaluate whether an event $e$ validly triggers a transition from state $S_{\text{current}} \to S_{\text{next}}$ given guard conditions $G$.
* **Curriculum Fix**:
  1. Programmatically generate 25,000 synthetic state-machine verification pairs covering:
     * IoT appliance state charts (ovens, thermostats, washing machines).
     * Network protocol state transitions (TCP handshakes, HTTP connection pools).
     * Game state / turn-based rule engines.
  2. Input framing format:
     ```
     Current State: {mode: "preheating", target_temp: 350, door: "closed"}
     Event: user_opens_door
     Guards: if door opens during preheating, heating element pauses and safety alert activates.
     Hypothesis: Next state is {mode: "preheat_paused", alert: "active"}. -> ENTAILMENT
     ```

### TR-03: Dense Legal Clause Grounding (Fixing ContractNLI 54.5% vs 78.0%)
* **Root Cause**: Gevva's long-context training relied predominantly on needle-in-a-haystack verification rather than dense, multi-page legal clause cross-referencing.
* **Curriculum Fix**:
  1. Ingest real-world contract and regulatory compliance corpora:
     * **ContractNLI** (full training split of Non-Disclosure Agreements with 11 standard NDAs clauses).
     * **CUAD** (Contract Understanding Atticus Dataset, 510 contracts with 41 clause types).
     * **CaseHOLD** (judicial holding legal reasoning pairs).
  2. Train with multi-page premises (1,000–4,096 tokens) requiring negative clause identification (e.g. determining whether an NDA lacks a non-compete clause or an explicit jurisdiction waiver).

### TR-04: In-Batch Hard Negative Mining for Dense Intent Catalogs (Fixing BANKING77 65.6% vs 89.8%)
* **Root Cause**: BANKING77 features 77 closely related intent classes (e.g., `card_arrival`, `card_delivery_estimate`, `card_linking`, `card_not_working`). When trained with isolated pointwise cross-entropy, the model predicts high entailment scores for multiple near-synonym intents.
* **Curriculum Fix**:
  1. Implement **In-Batch Hard Negative Mining**: During intent fine-tuning, dynamically identify the top-3 highest-scoring false intent hypotheses for each query and apply a margin ranking penalty:
     $$\mathcal{L}_{\text{intent-margin}} = \sum_{j \in \text{hard negatives}} \max\left(0, \gamma - (s_{\text{gold}} - s_j)\right)$$
  2. Ingest dense intent benchmarks with confusion matrix awareness: BANKING77, HWU64, and CLINC150 training splits.

### TR-05: Multi-Turn Conversation History & Token Authorization (Fixing API-Bank 49.6% vs 88.2%)
* **Root Cause**: Gevva evaluated single-turn requests well (BFCL 92.6%), but struggled when the target tool depended on a conversation history 6–10 turns long involving user authentication tokens (`GetUserToken`), confirmation steps, and error corrections.
* **Curriculum Fix**:
  1. Ingest multi-turn conversational datasets: **MultiWOZ 2.4**, **Taskmaster-1/2/3**, and the **API-Bank Level 2/3 training split**.
  2. Format premises using Gemma 4 native conversation role delimiters:
     ```
     <start_of_turn>user\n...\n<end_of_turn>\n<start_of_turn>model\n...\n<end_of_turn>
     ```
  3. Formulate hypotheses as specific state assertions: `"The assistant's next action must be: GetUserToken(username='...')"` to directly mirror API-Bank and SGD evaluation patterns.

---

## 9. Training Dataset Availability & Sourcing Inventory

Before executing Phase 5 training, the following inventory categorizes existing open-source assets vs gaps requiring synthetic generation:

| Target Gap / Benchmark | Existing Open-Source Datasets | Source Location / Status | Synthetic GenAI Required? |
| :--- | :--- | :--- | :---: |
| **Dense Legal Clause Reasoning** *(ContractNLI: 54.5%)* | • **CaseHOLD** (100k+ judicial holdings)<br>• **ContractNLI** (NDA training split)<br>• **CUAD** (510 contracts, 41 clause types) | **Ready on disk**: `data/staged/mc_qa/casehold_train.jsonl` (364 MB) & `work/.../contractnli`. CUAD on HF (`theatticusproject/cuad`). | Optional (open data sufficient for baseline) |
| **Adversarial Negations & Traps** *(ANLI: 31.0%)* | • **ANLI R1–R3** (162k adversarial pairs)<br>• **WANLI** (102k worker-AI adversarial pairs)<br>• **Counterfactually Augmented Data (CAD)** | **Ready on Hugging Face**: `facebook/anli` and `alisawuffles/WANLI` (permissive open-source). | **YES**: Controlled minimal pairs via teacher LLM |
| **Dense Intent Classification** *(BANKING77: 65.6%)* | • **BANKING77** (13k queries, 77 intents)<br>• **HWU64** (64 intents across 21 domains)<br>• **CLINC150** (150 intents + OOS) | **Ready on Hugging Face**: `PolyAI/banking77` and `clinc_oos` have standardized train splits. | **YES**: Boundary paraphrases for near-synonyms |
| **Multi-Turn Dialogue Tracking** *(API-Bank: 49.6%)* | • **API-Bank** (Level 1–3 train dialogues)<br>• **Schema-Guided Dialogue (SGD)**<br>• **MultiWOZ 2.4** (Belief tracking) | **Ready on disk & HF**: `work/.../apibank` & `work/.../sgd` on disk; MultiWOZ 2.4 on HF (`multiwoz_v22`). | Supplementary |
| **Discrete State Machines** *(Home Appliance: 15.0%)* | • **None available in standard NLI format.** Public benchmarks only contain raw test instances without modular NLI training splits. | **MISSING FROM OPEN WEB**. | **CRITICAL**: Pure programmatic + GenAI synthesis |

---

## 10. Generative AI Synthetic Data Generation Strategy

Because open-source web scrapes suffer from uncontrolled reporting bias and lexical shortcuts, Generative AI will be deployed as a primary data engineering instrument for the Phase 5 mixture:

### A. The Hybrid Programmatic-FSM + Generative Narrative Architecture
To fix the severe blind spot in state-machine transitions (Home Appliance: 15.0%):
1. **Symbolic FSM Generator**: Programmatically instantiate 50 distinct domain state charts (smart home appliances, network protocols, cloud resource lifecycles, workflow approvals). Each state chart defines explicit states $S$, alphabet events $E$, guard conditions $G$, and deterministic transitions $\delta: S \times E \times G \to S'$.
2. **GenAI Narrative Wrapper**: Pass the symbolic execution trace through a local frontier teacher LLM (via `llm_client.py`) with strict prompt constraints:
   * Convert the telemetry/event log into realistic natural-language assistant dialogues or system execution logs.
   * Generate three balanced, unambiguous hypotheses:
     - **Entailment**: Legal transition to valid next state.
     - **Hard Contradiction**: State mutation violating an explicit guard condition.
     - **Neutral**: Missing prerequisite sensor data; outcome cannot be deduced from context.
3. **Volume Target**: 25,000 verified state-machine triplets.

### B. Controlled Counterfactual Minimal-Pair Engine
To fix the ANLI negation and lexical shortcut collapse (31.0%):
1. Take verified premise-claim pairs from Stage 1 anchor data.
2. Prompt the teacher LLM to generate **exact minimal pairs** by applying atomic logical transformations:
   * **Scope Particle Inversion**: Swap *"only authorized users"* $\leftrightarrow$ *"any user"*.
   * **Quantifier Perturbation**: Swap *"all servers were patched"* $\leftrightarrow$ *"at least one server was patched"*.
   * **Negation Insertion**: Insert subtle grammatical negations (*"failed to detect"*, *"neither...nor"*).
3. **Training Objective**: The cross-encoder is trained with symmetric contrastive regularization, penalizing any model that assigns similar representations to minimal pairs with inverted truth values.
4. **Volume Target**: 30,000 balanced counterfactual minimal pairs.

### C. Boundary Paraphraser for Dense Intent Disambiguation
To close the gap on dense intent catalogs (BANKING77: 65.6% vs 89.8%):
1. Identify confusable intent pairs using the empirical confusion matrix (e.g., `card_arrival` vs `card_delivery_estimate`).
2. Prompt the teacher LLM to generate boundary queries designed to sit on the exact semantic edge between the two intents, explicitly highlighting distinguishing parameters (e.g. asking for a tracking number vs reporting that a physical envelope has not arrived).
3. **Volume Target**: 15,000 hard-negative intent pairs.

### D. Quality Control & Multi-Judge Consensus Verification
All synthetic data generated via GenAI must satisfy the pre-registered quality gates before inclusion into the master training mixture:
1. **Transport**: Executed through [`llm_client.py`](file:///home/dave/workspaces/nli-cross-encoder/llm_client.py) with GBNF grammar constraints to enforce schema validity.
2. **Committee Validation**: Every synthetic pair must pass the 4-judge committee in [`validator_committee.py`](file:///home/dave/workspaces/nli-cross-encoder/validator_committee.py) requiring $\ge 75\%$ consensus.
3. **Decontamination Gate**: Every generated pair is filtered against all 151,034 Decision Index 0.2 requests using an 8-gram rolling hash to guarantee zero test leakage.
4. **Metrics Audit**: All verdicts, consensus scores, and judge agreement latencies are logged idempotently into SQLite ([`validation_metrics.db`](file:///home/dave/workspaces/nli-cross-encoder/validation_metrics.db)).

---

## 11. Comprehensive Tracking Matrix

| ID | Initiative | Category | Target Problem / Benchmark | Complexity | Expected Impact | Target Release |
| :---: | :--- | :---: | :--- | :---: | :--- | :---: |
| **OPT-01** | Prefix KV Caching (`predict_candidates`) | Architecture | Multi-candidate latency & VRAM | Medium | **~90× speedup on multi-candidate tasks (16.9s $\to$ 180ms)** | gevva 1.1.0 |
| **OPT-02** | Dynamic Token-Budget Inference Batching | Architecture | Short-context throughput | Low | **2–4× speedup on short-context benchmarks** | gevva 1.1.0 |
| **OPT-03** | Adaptive Batch Slicing on OOM | Architecture | Memory fragmentation resilience | Low | **Zero OOM crashes during long evaluation runs** | gevva 1.1.0 |
| **OPT-04** | W4A16 Quantized Inference Pipeline | Serving | Low-VRAM / Edge serving | Low | **Reduces RAM/VRAM footprint to 4.5 GB** | gevva 1.1.0 |
| **OPT-05** | Native OOS Routing via Neutral Mass | Architecture | Out-of-scope intent rejection | Low | **Zero-shot out-of-domain rejection without fallback classes** | gevva 1.1.0 |
| **TR-01** | Adversarial Hard-Anchor Replay & Anti-Shortcut Loss | Curriculum | **ANLI R1–R3 (31.0% $\to$ 60%+)** | Medium | **Eliminates negation and word-swap vulnerability** | Gevva Phase 5 |
| **TR-02** | State-Machine Transition Modeling | Curriculum | **Home Appliance (15.0% $\to$ 70%+)** | Medium | **Enables dynamic state-chart and transition verification** | Gevva Phase 5 |
| **TR-03** | Dense Legal Clause & Contract Grounding | Curriculum | **ContractNLI (54.5% $\to$ 75%+)** | Medium | **Enables multi-page dense clause cross-referencing** | Gevva Phase 5 |
| **TR-04** | In-Batch Hard Negative Intent Mining | Curriculum | **BANKING77 (65.6% $\to$ 85%+)** | Low | **Disambiguates dense, near-synonym intent classes** | Gevva Phase 5 |
| **TR-05** | Multi-Turn Dialogue State Curriculum | Curriculum | **API-Bank (49.6% $\to$ 75%+)** | Medium | **Enables multi-turn conversational tool tracking** | Gevva Phase 5 |
| **SYN-01** | Synthetic FSM State Transition Generator | GenAI Data | **Home Appliance (15.0% $\to$ 70%+)** | Medium | **25k FSM transitions with programmatic ground-truth** | Gevva Phase 5 |
| **SYN-02** | Counterfactual Minimal-Pair Synthesizer | GenAI Data | **ANLI R1–R3 (31.0% $\to$ 60%+)** | Medium | **30k atomic scope & polarity perturbations** | Gevva Phase 5 |
| **SYN-03** | Hard-Negative Intent Boundary Paraphraser | GenAI Data | **BANKING77 (65.6% $\to$ 85%+)** | Low | **15k borderline confusion queries for near-synonyms** | Gevva Phase 5 |
| **SYN-04** | Multi-Judge Consensus Verification Pipeline | Data Quality | All Phase 5 Synthetic Data | Low | **Zero label noise; 100% committee verification** | Gevva Phase 5 |


