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

## 7. Tracking Matrix

| ID | Initiative | Target Area | Complexity | Expected Impact | Target Release |
| :---: | :--- | :--- | :---: | :--- | :---: |
| **OPT-01** | Prefix KV Caching (`predict_candidates`) | Inference Engine | Medium | **~90× speedup on multi-candidate tasks (16.9s $\to$ 180ms)** | gevva 1.1.0 |
| **OPT-02** | Dynamic Token-Budget Inference Batching | Inference Engine | Low | **2–4× speedup on short-context benchmarks** | gevva 1.1.0 |
| **OPT-03** | Multi-Turn Dialogue State Curriculum | Training / Data | Medium | **Closes API-Bank gap (+15–20% accuracy)** | Gevva Phase 5 |
| **OPT-04** | W4A16 Engine in `decision_index` | Export / Serving | Low | **Reduces RAM/VRAM footprint to 4.5 GB** | gevva 1.1.0 |
| **OPT-05** | Native OOS Routing via Neutral Mass | SDK / Routing | Low | **Robust zero-shot out-of-domain rejection** | gevva 1.1.0 |
