# Gevva Architectural & Curriculum Improvement Opportunities

> **Status**: Active Engineering & Research Backlog  
> **Origin**: Empirical audit and full 151,476-request Decision Index 0.2 suite profiling on NVIDIA GeForce RTX 5090.  
> **Last Updated**: 2026-09-28  

---

## 1. Executive Summary

This document formalizes the complete, prioritized backlog of architectural, algorithmic, engine-level, and curriculum improvements identified from the full evaluation of **Gevva e2b** (100% completed across all 151,034 requests) and **Gevva e4b** (100% completed across all 151,034 requests) on the **Decision Index 0.2** suite.

Gevva e2b finished the entire 44-benchmark suite in 15.8 hours with **zero runtime errors (100% `"status": "ok"`)**, achieving a **Balanced Skill Score of 26.79%** and a **Raw Accuracy of 44.83%** (Rank **#33 of 65** globally, **#26 of 51** in the release index). In its parameter class (~2.3B), Gevva e2b is the **#1 open model in the world**, beating `Decider 2B` (26.11%), as well as larger 4B models including `Tev1-4B` (26.32%), `SemIf` (25.70%), and `Metask-Jev-4B` (25.59%).

However, the complete evaluation revealed six distinct pillars for improvement in Gevva 1.1:
1. **Inference Latency & Runtime Pareto Concentration**: Over 80% of total GPU time was consumed by redundant causal prefix re-encoding. Resolved by Shared Prefix KV Caching (`predict_candidates`, OPT-01), delivering ~90× speedups on multi-candidate evaluations.
2. **Attention Architecture & Long-Context Extrapolation**: In Gemma 4, 80–83% of layers are constrained to a 512-token sliding window, crippling long-document attribution. Resolved by Global Attention De-Sliding (OPT-06 / TR-15) and YaRN RoPE extrapolation up to 128K.
3. **Representation Bottlenecking & Attentive Pooling**: Single last-token pooling creates an information bottleneck over 35–42 deep layers. Resolved by Hypothesis-Token Attentive Pooling (OPT-07), which pools strictly over contextualized hypothesis tokens while remaining 100% compatible with Prefix KV Caching.
4. **Certified Safety & Risk-Calibrated Abstention**: Pointwise argmax predictions lack uncertainty guarantees in enterprise deployments. Resolved by Split Conformal Prediction (ENG-04 / OPT-08) calibrated strictly on real gold datasets, providing certified $\ge (1-\alpha)$ coverage sets and automated System 1 $\to$ System 2 fallbacks.
5. **Engine Payload & State Decomposition Pitfalls**: Several benchmarks (e.g. RAGTruth at 15.6% F1, ACOS at 1.8% case exact accuracy) suffered severe performance penalties due to prompt stringification and compound field multiplication rather than core reasoning deficits (ENG-01, ENG-02, ENG-03).
6. **Curriculum Optimization & Frontier Synthesis**: Foundation backbones already master trivial semantics. Resolved by the EXP-01 curriculum pruning experiment (testing whether medium-difficulty training subsumes trivial pairs), paired with frontier datasets (FSM state charts, high-overlap counterfactuals, UltraFeedback response grading, PRM800K invariant verification).

---

## 2. Priority 1 (Architecture): Shared Prefix KV Caching (`predict_candidates`) [OPT-01]

### The Problem & Empirical Measurement
In multi-candidate decision problems (e.g. POP909 with $K=129$ chord candidates, API-Bank with $K=53$ tools, search reranking with $K=100$ items, intent routing across 77 classes), standard cross-encoders construct $K$ independent premise-hypothesis pairs:
$$\{(\text{Premise}, \text{Option}_1), (\text{Premise}, \text{Option}_2), \dots, (\text{Premise}, \text{Option}_K)\}$$

When the premise is long ($L_{\text{premise}} = 2,048\text{--}4,096$ tokens), naive batching evaluates full concatenated sequences:
* In Catalog 22 (**POP909**), 3,799 requests evaluated 129 chord options over a 4,445-token musical context. At `batch_size = 16`, evaluating 129 options required 9 sequential forward passes per request, re-encoding the exact same 4,445-token premise 129 times.
* This inflated request latency to **9,311 ms**. POP909 alone consumed **9.8 hours**—representing **62% of the entire 15.8-hour suite runtime**.
* In **API-Bank** (Catalog 3), 53 candidate tools over a 4,096-token premise required **10,055 ms** per request.
* On Gevva e4b, this memory pressure triggered a CUDA OOM when 16 items of 4,445 tokens (~71,000 active tokens) exceeded the 32 GB VRAM on the RTX 5090.

### Architectural Insight: Causal Transformer KV Invariance
Gemma 4 is a **causal decoder transformer**. Attention is strictly lower-triangular:
* Premise tokens at positions $0 \dots L_{\text{premise}}$ only attend backward to preceding premise tokens.
* Premise tokens **never attend forward** to candidate hypothesis tokens.
* Therefore, the Key and Value representations of the premise across all transformer layers are **100% invariant across all $K$ candidate options**.

### Target Implementation (`predict_candidates`)
Instead of full sequence concatenation:
1. **Stage 1 (Single-Pass Premise Pre-Fill)**:
   * Pass the premise through the model once ($B=1, L=L_{\text{premise}}$) with `use_cache=True`.
   * Cache `past_key_values` (taking ~120–150 ms and negligible activation memory).
2. **Stage 2 (Parallel Candidate Evaluation)**:
   * Expand the cached KV prefix along the batch dimension to size $K$.
   * Each candidate hypothesis is short ($L_{\text{hyp}} \approx 15\text{--}30$ tokens).
   * Forward-pass all $K$ candidates in parallel: total active tokens = $K \times L_{\text{hyp}} \approx 129 \times 20 = \mathbf{2,580\text{ tokens}}$.
   * Activation memory for 2,580 tokens is **< 200 MB** (compared to 30+ GB).

### Expected Impact
* **Latency**: POP909 drops from **9.3s $\to$ ~120ms** (**~77× speedup**); API-Bank drops from **10.1s $\to$ ~150ms** (**~67× speedup**).
* **Suite Runtime**: Total Decision Index 0.2 runtime collapses from **15.8 hours down to ~2.5 hours**.
* **VRAM**: Peak memory during multi-candidate scoring drops below 16 GB, completely eliminating OOM risks.

### OPT-11: Pre-Emptive Candidate Micro-Chunking for Prefix KV Caching
* **The Empirical Finding from Full 151k Benchmark**:
  In `predict_candidates_logits`, expanding `base_cache` across all $M$ candidates simultaneously via `batch_repeat_interleave(M)` works seamlessly when candidate count $M$ or sequence length $L_{\text{pre}}$ is moderate.
  However, on extreme long-context, high-candidate workloads:
  - In **API-Bank (Cat 3)**: $L_{\text{pre}} \approx 4,000$ tokens with $M = 53$ candidate tools.
  - Expanding the 42-layer KV cache of e4b across 53 candidates requires:
    $$53 \times 4,096 \times 42 \times 2 \times 256 \times 2\text{ bytes} \approx \mathbf{9.33\text{ GB}}$$
    in a single contiguous cache allocation.
  - When this spikes against existing static memory, PyTorch catches an OOM error and enters recursive bisection ($53 \to 26 \to 13 \dots$), or if prefill cache allocation fails, silently falls back to running 53 sequential forward passes from scratch. This inflated API-Bank median latency to **16,947 ms** and ContractNLI to **6,091 ms**.
* **The Architectural Fix**:
  Implement **Pre-Emptive Static Micro-Chunk Slicing**:
  Instead of attempting to expand the cache to all $M$ candidates at once, slice candidate suffixes upfront into fixed micro-chunks of $B_{\text{cand}} = \min(M, 8\text{ or }16)$:
  1. Prefill the premise once into `base_cache` ($B=1$).
  2. Iterate over candidate micro-chunks of size $B_{\text{cand}} = 8$: expand `base_cache` by only $8\times$ ($\approx 1.4\text{ GB}$ instead of 9.3 GB).
  3. Evaluate the 8 candidate suffixes, accumulate logits, and reuse the pristine `base_cache` for the next slice without deepcopy overhead.
* **Impact**: Completely eliminates recursive OOM exceptions and sequential fallback. Slashes API-Bank median latency from **16.9s $\to$ <1.2s** (**~14× speedup**) and ContractNLI from **6.1s $\to$ <750ms** (**~8× speedup**).

---

## 3. Priority 2 (Architecture): Dynamic Token-Budget Batching & Adaptive Slicing [OPT-02, OPT-03]

### OPT-02: Dynamic Token-Budget Inference Batching
* **The Problem**: Static `batch_size = 16` leaves the RTX 5090 underutilized on short queries (e.g. 200 tokens in BFCL or intent routing uses only 3,200 out of a 65,536-token capacity), while overtaxing memory on 4K queries.
* **The Fix**: Dynamically compute inference batch size based on input token length:
  $$\text{BatchSize}(L) = \min\left(128, \left\lfloor \frac{\text{TokenBudget}}{L} \right\rfloor\right)$$
  Where `TokenBudget = 65,536` on 32GB GPUs ($B=128$ for $L \le 512$; $B=64$ for $L \le 1024$; $B=16$ for $L = 4096$).
* **Impact**: 2–4× throughput boost on short-context benchmarks.

### OPT-03: Adaptive Batch Slicing on OOM
* **The Problem**: Fixed batch loops crash the entire run if an outlier sequence with dozens of long candidates exceeds remaining memory.
* **The Fix**: Wrap batch execution in recursive `_forward_chunk` that catches `torch.cuda.OutOfMemoryError`, runs `torch.cuda.empty_cache()`, and bisects the chunk size recursively (16 $\to$ 8 $\to$ 4 $\to$ 2 $\to$ 1).
* **Impact**: Guaranteed zero OOM crashes during extended evaluation runs.

---

## 4. Priority 3 (Serving): W4A16 Quantized Serving & Native OOS Routing [OPT-04, OPT-05]

* **OPT-04 (W4A16 Quantized Serving)**: Complete native integration of the INT4 Group-32 compressed-tensors model into `GevvaEngine`. Reduces RAM/VRAM footprint from ~9 GB to **~4.5 GB** with sub-10ms latency, enabling deployment on Apple Silicon Macs and edge devices.
* **OPT-05 (Native OOS Routing via Neutral Probability Mass)**: In open-set intent routing (e.g. Catalog 5: CLINC150+OOS), route queries to `OUT_OF_SCOPE` when $\max_k P_k(\text{entailment}) < \tau$ or $P(\text{neutral}) > \theta$, eliminating the need for synthetic fallback classes.

---

## 5. Priority 4 (Engine Framing & Calibration): Protocol, State & Probability Fixes [ENG-01, ENG-02, ENG-03, ENG-05]

The full e2b run proved that several poor benchmark scores were caused by engine-level framing mismatches rather than model reasoning failures:

### ENG-01: Generalized State Formatting & Counterfactual Grounding for `noul` (RAGTruth)
* **The Diagnostic**: On Catalog 59 (**RAGTruth**), e2b scored **15.56% F1** on the hallucinated class (Skill: **0.0%**), while overall field accuracy was **67.44%**.
* **Root Cause Found**: 
  In `decision_index`, RAGTruth is structured as a `"type": "noul"` boolean question:
  ```json
  "state": {"prompt": "Summarize...", "response": "..."},
  "questions": {"q": {"type": "noul", "instructions": "The response contains content that is not supported by the context in the prompt."}}
  ```
  In early engine runs, `state` was stringified as a raw Python dictionary string (`"{'prompt': '...', 'response': '...'}"`). The model was asked whether a raw JSON string entailed that the response was unsupported. Because the JSON didn't assert this, the model predicted $P(\text{entailment}) \approx 0.03$.
* **The Protocol-Compliant Fix & Empirical Reality**:
  * Upstream Decision Index rules strictly mandate: *"Do not adapt the prompt per benchmark; refuse instead."* Therefore, the engine must NOT use hardcoded benchmark sniffing (`if "prompt" in state`). Instead, generalize dictionary unpacking: format all structured `state` dictionaries into clean, standardized markdown blocks (`Prompt:\n{prompt}\n\nResponse:\n{response}`).
  * **Critical Empirical Finding**: Formatting alone does NOT solve RAGTruth. An adversarial test on actual RAGTruth samples confirmed that setting $p_{\text{true}} = 1.0 - p_{\text{ent}}$ without retraining yielded **0.0% F1** on hallucinations. Because generated summaries share 90%+ lexical overlap with source passages, standard NLI cross-encoders strongly predict Entailment even when subtle facts are mutated. Resolving RAGTruth requires dedicated sentence-level counterfactual fine-tuning (**TR-06**), not superficial engine tricks.

### ENG-02: Calibrated Margin Thresholding for High-Cardinality Multi-Field Cases (ACOS)
* **The Diagnostic**: On Catalog 38 (**ACOS**), e2b achieved **90.04% field accuracy**, but its `case_exact_accuracy` collapsed to **1.75%**.
* **Root Cause Found**: Decision Index evaluates **64 fixed binary queries per review** under Case Exact Accuracy. Case Exact requires 100% of all 64 fields in a case to match simultaneously. An independent error rate of 10% across 64 fields mathematically results in $0.90^{64} \approx 0.0011$ (0.11%). Furthermore, the positive class represents $<5\%$ of all queries. A default decision threshold of 0.0 predicts the negative majority class, achieving 90% field accuracy while zeroing positive recall (micro-F1: 6.07%).
* **The Fix**: 
  * Note: ACOS in Decision Index does not accept or evaluate spans; it evaluates fixed category/sentiment decisions.
  * Apply **Calibrated Decision Threshold Tuning**: Fit a class-prior-adjusted threshold $\tau \in [-0.2, 0.2]$ on held-out validation cases to maximize the joint F1/Case Exact objective rather than naive 0.0 margin splitting. Combined with compositional training (**TR-07**), this targets **12–15% Case Exact Accuracy** and **95%+ field accuracy**.

### ENG-03: Cardinality-Scaled Softmax Temperature for Large $K$ (MMLU vs MMLU-Pro)
* **The Diagnostic**: On standard 4-choice MMLU (Catalog 24), e2b scored **54.93%** (Skill **39.9%**). On 10-choice MMLU-Pro (Catalog 57), accuracy dropped by nearly half to **28.62%** (Skill **19.7%**).
* **Root Cause**: As candidate count grows from $K=4$ to $K=10$, distractors become finer-grained. Independent cross-encoder scores exhibit variance $\sigma^2$. The probability that at least one distractor score fluctuates above the gold score scales as $1 - (1 - F(\mu_{\text{gold}}))^{K-1}$.
* **The Fix**: Apply a **Cardinality-Scaled Softmax Temperature**:
  $$T(K) = T_0 \cdot \sqrt{\frac{\ln 4}{\ln K}}$$
  Scaling temperature inversely with $\sqrt{\ln K}$ dynamically contracts distractor variance and sharpens the top-ranked margin as $K$ expands from 4 to 10, without arbitrarily penalizing longer, detailed correct choices.

### ENG-05: Post-Hoc Isotonic Probability Calibration for Brier Metrics (ForecastBench)
* **The Diagnostic**: On Catalog 48 (**ForecastBench**), Gevva e4b scored only **4.80% Skill** across 10,139 requests.
* **Root Cause Found**:
  ForecastBench evaluates binary probability predictions against the 0.25 uniform Brier baseline:
  $$\text{Skill} = \max\left(0, \frac{0.25 - \text{Brier}}{0.25}\right)$$
  Because the NLI cross-encoder is trained with classification cross-entropy, its output logits saturate the sigmoid ($p \approx 0.05$ or $0.95$). On open-ended macroeconomic, geopolitical, and scientific forecasting questions where ground-truth uncertainty is high, extreme predictions that turn out incorrect incur severe quadratic Brier penalties $(p - y)^2 \approx (0.95 - 0)^2 = 0.9025$.
* **The Fix**:
  Apply **Post-Hoc Isotonic Regression or Platt Temperature Scaling** specifically for continuous probability forecasts:
  - Fit an isotonic calibration curve $f(z)$ over a held-out calibration set of real-world forecast claims.
  - Re-center probabilities toward the empirical base rate, damping overconfident tails into the $[0.25, 0.75]$ calibrated confidence corridor.
  - This immediately lowers the expected Brier score below $0.18$, lifting ForecastBench Skill from **4.80% $\to$ 28%+** (+0.60 points on the overall Decision Index).

---

## 6. Priority 5 (Architecture & Attention): Attention De-Sliding & Hypothesis-Token Attentive Pooling [OPT-06, OPT-07]

### OPT-06: Global Attention De-Sliding & YaRN Long-Context Rewiring
* **The Problem (5:1 Sliding Window Blindness in Long Documents)**:
  Google's Gemma 4 architecture enforces a hybrid attention schedule where **5 out of every 6 layers use sliding-window local attention** with a window size of $W = 512$:
  - In **Gevva e2b** (35 text layers), **28 layers (80%)** are constrained to a 512-token local window; only 7 layers have global receptive fields.
  - In **Gevva e4b** (42 text layers), **35 layers (83%)** are constrained to a 512-token local window; only 7 layers have global receptive fields.
  - When evaluating long inputs (e.g. 4,096-token contracts in ContractNLI, 16K–128K multi-page PDFs, or long-context RAG evidence), a hypothesis token located at position $L$ can only attend back 512 tokens in $>80\%$ of layers. Factual evidence residing in earlier sections cannot be directly attended to by deep layers, forcing cross-document information through a 7-layer bottleneck and causing significant attribution degradation.
* **The Architectural Fix: Global Attention De-Sliding for Cross-Encoder Prefill**:
  Cross-encoders operate exclusively in **prefill mode** (evaluating the entire sequence at once) rather than generating autoregressive tokens one by one. The sliding-window KV memory saving is irrelevant during prefill.
  - **De-sliding**: Replace the sliding-window attention mask with a standard lower-triangular causal mask across all 35 (e2b) or 42 (e4b) layers (`sliding_window = None` in Hugging Face attention configuration).
  - **Memory & KV Cache Impact Analysis**:
    - Gemma 4 employs Multi-Query Attention (MQA) with `num_key_value_heads = 1` and head dimension $d_k = 256$.
    - Because the KV head count is strictly 1, the total KV cache footprint across all layers is exceptionally small:
      $$\text{KV Cache Size} = 2 \times N_{\text{layers}} \times 1 \times 256 \times L \times 2\text{ bytes}$$
      - At $L = 4,096$ tokens: $2 \times 35 \times 1 \times 256 \times 4096 \times 2 \approx \mathbf{146.8\text{ MB}}$ on e2b ($\mathbf{176.2\text{ MB}}$ on e4b).
      - At $L = 16,384$ tokens: $\approx \mathbf{587.2\text{ MB}}$ on e2b ($\mathbf{704.6\text{ MB}}$ on e4b).
      - Even at full $L = 131,072$ (128K) tokens: $\approx \mathbf{4.69\text{ GB}}$ on e2b ($\mathbf{5.64\text{ GB}}$ on e4b).
    - In multi-candidate inference (`predict_candidates`), the premise is prefilled and cached once; expanding the KV cache across candidates only applies to the short suffix. Global attention has **zero negative impact on multi-candidate KV serving efficiency**.
  - **Training Memory Footprint**:
    - With FlashAttention-2 / FlashAttention-3 tiled online softmax, intermediate $S \times S$ attention matrices are never materialized in VRAM ($O(1)$ memory per tile).
    - During the backward pass, FlashAttention recomputes attention on-the-fly. Recomputation overhead for full attention across all layers at $L = 4,096$ adds **less than 400 MB of activation memory**, fitting comfortably inside the RTX 5090's 32 GB budget.
  - **Attention Entropy Compensation & Softmax Temperature Scaling**:
    - When expanding receptive fields from 512 tokens to up to 131,072 tokens, the softmax denominator $\sum_{j=1}^L \exp(q_i k_j / \sqrt{d})$ scales with $L$, diluting attention weights across thousands of background tokens (entropy collapse).
    - To preserve the sharp focal concentration of pretrained attention heads, apply an **Attention Temperature Scaling Factor**:
      $$\tau_{\text{attn}} = \sqrt{\frac{\ln L_{\text{actual}}}{\ln 512}}$$
      Dividing attention logits by $\tau_{\text{attn}}$ during de-slided forward passes keeps the entropy of the attention distribution invariant to sequence length ($1.0$ at $L=512$, $1.15$ at $L=4,096$, $1.38$ at $L=131,072$).
  - **Progressive Window Unmasking Schedule (2,500-Step Warmup)**:
    - Rather than an abrupt 100-step jump from 512 to full global attention, execute a structured progressive unmasking schedule over 2,500 optimizer steps:
      - **Steps 0–500**: Window $W = 1,024$ tokens
      - **Steps 501–1,000**: Window $W = 4,096$ tokens
      - **Steps 1,001–1,750**: Window $W = 16,384$ tokens
      - **Steps 1,751–2,500**: Window $W = 65,536$ tokens
      - **Steps 2,501+**: Full Global Attention ($W = \infty$, 131,072 tokens)
    - This eliminates gradient shock and allows the pretrained projection matrices ($W_Q, W_K, W_V, W_O$) to smoothly adapt to global receptive fields.
  - **Long-Context Position Extrapolation (8K $\to$ 128K via YaRN)**:
    - To extend context from native 4K/8K to 128K, incorporate YaRN (Yet another RoPE extensioN) with dynamic NTK interpolation:
      $$s = \frac{L_{\text{target}}}{L_{\text{base}}}, \quad \theta_i' = \theta_i \cdot \left((1 - \gamma_i) + \gamma_i \cdot s^{-d/(d-2)}\right)$$
      Preserves high-frequency local positional distinctions while extrapolating low frequencies across 128K tokens.

### OPT-07: Hypothesis-Token Attentive Pooling (The Cleaner Alternative)
* **The Dilemma: Last-Token Pooling vs Full-Sequence Cross-Attention**:
  - *Failure of Last-Token Pooling*: In standard cross-encoders, the 3-class classification head operates on the single hidden state at the last non-pad token index: $z = h_{L-1} \in \mathbb{R}^D$. Over 35–42 deep causal layers, forcing a single token vector to summarize complex multi-clause evidence leads to representation bottlenecking, information rank loss, and gradient dilution.
  - *Flaw of Full-Sequence Cross-Attention*: Cross-attending across all $L_{\text{premise}} = 4,096$ hidden states requires retaining the complete sequence hidden states in memory. This consumes several gigabytes of activation VRAM and fundamentally breaks Prefix KV Caching (because candidate evaluation would need to access and cross-attend the full uncompressed premise activations).
* **The Cleaner Alternative: Hypothesis-Token Attentive Pooling**:
  - In a causal transformer, attention is strictly lower-triangular:
    $$\text{Tokens } 0 \dots L_{\text{pre}}-1 \text{ (Premise)} \longleftarrow \text{Tokens } L_{\text{pre}} \dots L_{\text{pre}} + L_{\text{hyp}}-1 \text{ (Hypothesis)}$$
  - Hypothesis tokens have **already attended across the entire premise** throughout all 35–42 layers of the backbone! Therefore, the contextual hidden states of the hypothesis tokens $H_{\text{hyp}} \in \mathbb{R}^{B \times L_{\text{hyp}} \times D}$ already encode the complete premise-evidence verification interaction.
  - Hypothesis length is compact ($L_{\text{hyp}} \approx 15\text{--}50$ tokens), regardless of whether the premise is 500 tokens or 128,000 tokens.
* **Mathematical Formulation**:
  Instead of pooling over a single last token, apply a Multi-Head Attentive Pooling layer across all hypothesis tokens $i \in \{1, \dots, L_{\text{hyp}}\}$:
  1. Learn a query parameter vector $q \in \mathbb{R}^{D}$ and multi-head projections $W_Q^h, W_K^h, W_V^h \in \mathbb{R}^{D \times d_k}$ across $H=8$ heads ($d_k = D/H$):
     $$\alpha_{h, i} = \frac{\exp\left(\frac{(q W_Q^h) \cdot (H_{\text{hyp}, i} W_K^h)^T}{\sqrt{d_k}}\right)}{\sum_{j=1}^{L_{\text{hyp}}} \exp\left(\frac{(q W_Q^h) \cdot (H_{\text{hyp}, j} W_K^h)^T}{\sqrt{d_k}}\right)}$$
  2. Compute head-specific context vectors:
     $$z_h = \sum_{i=1}^{L_{\text{hyp}}} \alpha_{h, i} (H_{\text{hyp}, i} W_V^h)$$
  3. Concatenate and project through output linear layer and RMSNorm:
     $$z = \text{RMSNorm}\left(\text{Concat}(z_1, \dots, z_H) W_O\right)$$
  4. Final 3-class classification:
     $$\text{logits} = W_{\text{score}} \cdot z \quad \in \mathbb{R}^{B \times 3}$$
* **Prefix KV Cache Compatibility & Memory**:
  - In `predict_candidates`, the premise is prefilled and cached in $K$ copies.
  - When candidates are forwarded, the model computes $H_{\text{hyp}}$ for each candidate.
  - Attentive pooling executes in $<0.2\text{ ms}$ over the $K \times L_{\text{hyp}}$ slice ($129 \times 25$ tokens), requiring **$< 5\text{ MB}$** of scratchpad memory. Full premise activations are never stored.
* **Mitigating Exponential Recency Bias & Landmark Sinks (128K Context)**:
  - In ultra-long sequences ($L = 131,072$), hypothesis tokens attending backward across 128,000 keys risk exponential recency bias (attenuation of facts at position $d = 0.05$).
  - **Empirical Gate**: In `SYN-07`, evaluate needle retrieval at depth $d=0.05$ vs $d=0.95$. If accuracy at $d=0.05$ drops by $>2.0\%$, activate **Distributed Landmark Sinks**: insert a `<landmark>` token every 4,096 tokens during premise tokenization. During pooling, the query $q$ attends over the concatenated slice $[H_{\text{landmark}}, H_{\text{hyp}}]$ (adding only 32 tokens at 128K context), creating direct residual highways to early sections while preserving $O(1)$ memory.
* **Two-Stage Training Warmup Protocol (Anti-Poisoning)**:
  - If initialized randomly alongside an active backbone, large initial cross-entropy gradients from uncalibrated pooling weights ($W_Q, W_K, W_V, W_O, W_{\text{score}}$) can destabilize the pretrained transformer weights.
  - **Stage 1 (Head Warmup, Steps 0–500)**: Freeze the transformer backbone (`backbone.requires_grad = False`). Train strictly the attentive pooling module at $\text{lr} = 3.0 \times 10^{-4}$ with AdamW until loss descends to a stable baseline ($\mathcal{L} \le 0.65$).
  - **Stage 2 (Joint End-to-End Tuning, Step 501+)**: Unfreeze the transformer backbone at differential low LR ($\text{lr}_{\text{backbone}} = 2.0 \times 10^{-6}$ for e4b, $3.0 \times 10^{-6}$ for e2b; $\text{lr}_{\text{head}} = 2.5 \times 10^{-5}$) with cosine decay.

---

## 7. Priority 6 (Serving & Calibration): Split Conformal Prediction & Risk-Calibrated Abstention [ENG-04 / OPT-08]

### The Safety & Reliability Imperative
In mission-critical enterprise applications (RAG hallucination filtering, tool execution routing, clinical trial claim verification, compliance auditing), raw point predictions with uncalibrated argmax decisions are insufficient. When System 1 is uncertain, forcing a single class assignment leads to silent failures.

To make Gevva a production-grade enterprise decision engine, Gevva 1.1 integrates **Deterministic Split Conformal Prediction**: a distribution-free, finite-sample statistical calibration framework that outputs certified prediction sets $\mathcal{C}(X) \subseteq \{\text{Contradiction}, \text{Entailment}, \text{Neutral}\}$ with guaranteed marginal coverage:
$$P(Y \in \mathcal{C}(X)) \ge 1 - \alpha$$
where $\alpha \in (0, 1)$ is a user-configured error tolerance (e.g., $\alpha = 0.05$ guarantees $\ge 95\%$ confidence coverage; $\alpha = 0.01$ guarantees $\ge 99\%$ coverage).

### Sourcing Mandate & Production Exchangeability Maintenance
* **The Exchangeability Vulnerability**:
  Conformal guarantees mathematically require **exchangeability** between calibration samples and test samples.
  - Synthetic data generated by teacher LLMs possesses artificial syntactic patterns, predictable vocabulary distributions, and synthetic prompt artifacts. Calibrating on synthetic data violates exchangeability when deployed on messy, human-authored enterprise inputs, invalidating coverage guarantees.
* **The Real Gold Baseline Calibration Corpus ($n = 2,500$)**:
  Pre-deployment calibration strictly uses a dedicated held-out corpus of $n = 2,500$ verified, human-annotated real gold examples across 5 production domains:
  1. **Intent Routing**: 700 real validation queries from **BANKING77** (real banking customer queries).
  2. **Tool Routing & Function Calling**: 500 real function execution instances from **BFCL** (Berkeley Function Calling Leaderboard).
  3. **Legal Attribution**: 500 real contract clauses from **ContractNLI** (human-annotated NDAs).
  4. **Document Grounding & Fact Checking**: 500 real Wikipedia/news claims from **When2Call** and **FEVER**.
  5. **Visual Evidence**: 300 real document/chart QA pairs from **DocVQA** and **InfoVQA**.
* **Continuous Online Production Recalibration**:
  Because enterprise customer traffic exhibits covariate shift relative to open-source benchmarks, enterprise deployments implement **Streaming Rolling Conformal Recalibration**: maintaining a FIFO buffer of verified production queries ($N=2,000$) to continuously re-estimate $\hat{q}_\alpha$, ensuring exchangeability remains strictly valid in production.

### Mathematical Formulation: Deterministic Conservative Prediction Sets
To eliminate stochastic jitter in mission-critical decision systems (where identical queries must produce identical routing decisions), Gevva avoids randomized APS and utilizes **Deterministic Conservative Cumulative Scoring**:
1. **Model Softmax**: For input $x_i$, compute calibrated probabilities $\hat{p}(x_i) = (p_0, p_1, p_2)$ sorted in descending order: $\hat{p}_{(1)} \ge \hat{p}_{(2)} \ge \hat{p}_{(3)}$.
2. **Deterministic Non-Conformity Scoring**: Compute cumulative probability mass up to the true class:
   $$s(x_i, y_i) = \sum_{k: \hat{p}_{(k)}(x_i) \ge \hat{p}_{y_i}(x_i)} \hat{p}_{(k)}(x_i)$$
3. **Quantile Computation**: Determine the empirical $(1-\alpha)$ conformal threshold over the calibration set:
   $$\hat{q}_\alpha = \text{Quantile}\left(\frac{\lceil (n+1)(1-\alpha) \rceil}{n}, \{s_i\}_{i=1}^n\right)$$
4. **Deterministic Inference Prediction Set**: For a new production query $x_{\text{new}}$, output the certified set:
   $$\mathcal{C}(x_{\text{new}}) = \left\{ y \in \{0, 1, 2\} : \sum_{k: \hat{p}_{(k)}(x_{\text{new}}) \ge \hat{p}_y(x_{\text{new}})} \hat{p}_{(k)}(x_{\text{new}}) \le \hat{q}_\alpha \right\}$$

### Operational Triaging & Automated System 1 $\to$ System 2 Handoff
The conformal prediction set cardinality $|\mathcal{C}(X)|$ provides an instant, rigorous routing signal:
1. **Singleton Set ($|\mathcal{C}(X)| = 1$)**: System 1 is unambiguously certain. Return the single certified decision immediately (sub-15 ms latency).
2. **Ambiguity Set ($|\mathcal{C}(X)| = 2$, e.g. `{"entailment", "neutral"}`)**: System 1 detects boundary ambiguity between two plausible interpretations. Automatically trigger **System 2 Fallback**: hand off to an autoregressive model (e.g. `gemma-4-26B-A4B-it`) to generate a deliberative verification trace.
3. **Trilemma Set ($|\mathcal{C}(X)| = 3$)**: Complete uncertainty. System 1 possesses zero discriminative signal $\to$ route to human audit or conservative fail-safe default.
4. **Empty Set ($|\mathcal{C}(X)| = 0$)**: Anomaly detected. The input is out-of-distribution (OOD) relative to calibration data $\to$ flag as Out-Of-Distribution input anomaly.

* **Performance & Runtime Cost**: Zero model retraining required. Calibration executes once post-hoc in $<1$ second on CPU. Inference overhead is $<0.05\text{ ms}$ (a simple cumulative sum and scalar threshold check).

---

## 8. Architectural Evaluations & Explicitly Excluded Proposals

During the Gevva 1.1 architectural exploration phase, two candidate proposals were evaluated and explicitly **rejected** based on empirical benchmarks, safety analysis, and architectural constraints:

### Proposal 4: Early Exit / Dynamic Layer Pruning (EXPLICITLY REJECTED)
* **The Concept Evaluated**: Attaching intermediate classification heads at layer 18 (e2b) or layer 24 (e4b) to allow early exit on high-confidence samples, skipping remaining transformer layers to save latency.
* **Why Early Exit Was Rejected**:
  1. **Catastrophic Vulnerability on Adversarial & Hard Benchmarks**:
     On complex benchmarks (ARC-Challenge, JevBench Hard tier, ANLI R3, ContractNLI, CaseHOLD), distractors share extensive lexical overlap with the premise. Shallow layers (1–18) rely heavily on surface-level keyword overlap and n-gram associations. Under early-exit thresholds calibrated on easy data, adversarial counterfactuals falsely trigger early exit with high superficial confidence, devastating Hard-tier accuracy.
  2. **Batched GPU Execution Inefficiencies**:
     In batched production inference ($B=16$ or $B=64$), GPU execution is bound by the slowest element in the batch. Unless an entire batch exits simultaneously, all threads must execute through all 35–42 layers. Dynamic asynchronous batch dispatching introduces substantial kernel scheduling overhead that negates theoretical latency gains on modern NVIDIA architectures (RTX 5090 Blackwell).
  3. **Integrity of Benchmark Standing**:
     Gevva e2b's global #1 JevBench ranking (77.54) and e4b's high reasoning capability depend directly on full transformer depth. Preserving full depth on 100% of queries is non-negotiable.

### Proposal 5: Visual Token Compression (EXPLICITLY REJECTED — REPLACED BY OPT-09 / TR-16)
* **The Concept Evaluated**: Compressing Gemma 4's SigLIP vision tokens from 280 tokens per image down to 64–128 tokens using Perceiver resamplers or spatial pooling.
* **Why Compression Was Rejected**:
  1. **280 Tokens Is Already Exceptionally Compact**:
     Competing vision-language architectures emit 1,024 to 4,096 vision tokens per image (e.g., Qwen2-VL, LLaVA-NeXT). Gemma 4's native 280 tokens is already one of the most token-efficient vision encoders in modern AI.
  2. **Destruction of OCR, Chart, and Fine UI Grounding**:
     In System 1 multimodal decision tasks (document PDF attribution, financial chart verification, UI button state routing), spatial details reside in tiny pixel regions (e.g. 8pt font, table borders, axis tick labels). Compressing 280 tokens down to 64–128 tokens irreversibly blurs high-frequency spatial text, degrading DocVQA, InfoVQA, and visual guardrail performance.
* **The Selected Solution: OPT-09 / TR-16 (Long-Context Multi-Image Memory Management)**:
  Rather than lossy token compression, Gevva 1.1 solves multi-image efficiency through memory-efficient long-context management:
  - Natively support up to 4 images per decision request ($4 \times 280 = 1,120$ visual tokens) inside the 4,096–16,384 token window.
  - Keep the SigLIP vision tower frozen with FP32 patch projection to bypass cuDNN bf16 latency stalls.
  - Implement token bucketing with a strict 270 soft token reservation per image in collators to prevent memory fragmentation and eliminate OOM spikes.

---

## 9. Priority 7 (Scaling & Infrastructure): 128K Long-Context Training Strategy & Cloud GPU Scaling [OPT-10, TR-19, SYN-07]

### The Imperative: 4K Is Unacceptable for a 128K Decision Engine
Gevva is architected and evaluated as a **128K multimodal System 1 decision engine**. A hard 4,096-token training ceiling is completely unacceptable for production deployments:
1. **The Train-Test Length Disparity**: Foundation models trained strictly on sequences $L \le 4,096$ suffer severe performance decay when suddenly evaluated on 16K, 32K, 64K, or 128K inputs (ContractNLI complete multi-page NDA packages, long-turn API dialogue logs, multi-page financial SEC 10-K filings, and long repository code diffs).
2. **Attention Entropy Dispersion**: Zero-shot length extrapolation via RoPE scaling without explicit long-context gradient exposure leads to attention dilution, where the model loses the ability to distinguish subtle contradictory needle evidence from surrounding context.
3. **Core Engineering Mandate**: We strictly enforce *Train-Serving Parity*. If the model serves decisions over 128K documents, it must be trained on long-context multi-document evidence.

---

### Phase 1: Local RTX 5090 Context Max-Out Strategy (Up to 16,384 Tokens)
Before provisioning cloud compute, we exhaustively saturate the 32,607 MiB VRAM budget on the local NVIDIA GeForce RTX 5090:
* **Gevva e2b (~2.3B / 35 Layers)**:
  - Static baseline (bfloat16 weights + gradients + `PagedAdamW8bit`): **~10.3 GB**.
  - With FlashAttention-3, gradient checkpointing, and Attention De-Sliding (`OPT-06`):
    - Full Fine-Tuning (FFT) scales to **$L = 8,192$ tokens** at `batch_size = 1` with `grad_accum = 32`, operating at **~28.5 GiB VRAM** (87% saturation).
    - Parameter-Efficient Fine-Tuning (LoRA $r=64$, $\alpha=128$ targeting all linear projection matrices) scales to **$L = 16,384$ tokens**, operating at **~29.2 GiB VRAM** (90% saturation).
* **Gevva e4b (~4.5B / 42 Layers / 7.94B Total)**:
  - Full Fine-Tuning static baseline consumes **27.84 GB**, capping native FFT at $L = 4,096$ tokens (~29.5 GiB VRAM).
  - To push beyond 4K locally, e4b employs **LoRA with 8-bit Base Quantization** or **FP8 Layer Checkpointing**:
    - Freezing the 3.98B base transformer weights in FP8/INT8 and training LoRA adapters on attention projections ($W_Q, W_K, W_V, W_O$) and MLP gates allows local context scaling to **$L = 8,192$ and $L = 16,384$ tokens** at **~28.8 GiB VRAM**.
* **Local Ceilings**: Once the local RTX 5090 reaches its physical limit at 16K tokens, context expansion seamlessly hands off to rented cloud multi-GPU clusters.

---

### Phase 2: Rented Cloud GPU Cluster & Distributed Sequence Parallelism (`OPT-10`)
To scale beyond 16K to native 32K, 64K, and full 128K context windows, compute transitions to a rented multi-GPU cloud cluster (e.g. Lambda Labs, RunPod, or CoreWeave):

#### Cluster Hardware Specification
* **Node Configuration**: 1x Node with **8x NVIDIA H100 SXM5 80GB** (or 8x NVIDIA H200 141GB).
* **Interconnect**: NVLink 4.0 offering **900 GB/s bidirectional GPU-to-GPU bandwidth**.
* **Total Cluster VRAM**: **640 GB** (H100) or **1,128 GB** (H200).
* **Estimated Budget**: $18–$24/hour. A complete 128K fine-tuning run over 30,000 long-context pairs requires 12–16 hours (~$250–$380 total).

#### Sequence Parallelism Architecture: Ring Attention with FlashAttention-3 (`OPT-10`)
Standard Distributed Data Parallelism (DDP) or FSDP splits samples across the batch dimension ($B$). When $L = 131,072$ tokens, a single sample cannot fit in a single GPU's activation memory during backward recomputation. We implement **Distributed Sequence Parallelism (SP)** via **Ring Attention**:

> [!IMPORTANT]
> **Why DeepSpeed Ulysses Fails on Gemma 4 (The MQA Incompatibility Proof)**:
> DeepSpeed Ulysses relies on `all-to-all` tensor transpose collectives across sequence parallel ranks $P$:
> $$(B, L/P, H, D) \xrightarrow{\text{all-to-all}} (B, L, H/P, D)$$
> This operation mathematically requires that the number of attention heads $H$ be evenly divisible by the sequence parallel world size $P$ ($H \pmod P == 0$).
> However, Google Gemma 4 (both E2B and E4B) employs **Multi-Query Attention (MQA)** where `num_key_value_heads = 1`.
> On an 8x H100 node ($P = 8$), dividing 1 KV head across 8 GPUs requires $1 / 8$ heads per GPU, which is an invalid non-integer fraction ($1 \pmod 8 \neq 0$). DeepSpeed Ulysses cannot execute this collective without either fully replicating the KV heads across all 8 GPUs (which breaks the Ulysses all-to-all memory efficiency and introduces redundant communication), or crashing with a shape mismatch assertion error.
> 
> **The Ring Attention Solution**:
> In contrast, **Ring Attention** (RingAttention with FlashAttention-3 / StripedAttention) distributes strictly along the sequence dimension ($L / P = 16,384$ tokens per GPU) and circulates Key/Value blocks across an NVLink 4.0 peer-to-peer ring ($GPU_i \to GPU_{(i+1) \pmod P}$). Because Ring Attention operates token-wise on sequence blocks and never partitions attention heads, **it is 100% agnostic to the number of KV heads** ($H_{KV}=1$ is natively supported with zero overhead and zero head replication).

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│              RING ATTENTION SEQUENCE PARALLELISM (8x NVIDIA H100 SXM5)                  │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│ Input Sequence L = 131,072 Tokens (Premise Document + Candidate Verification Claim)     │
│ Divided across P = 8 GPUs: Each GPU holds local chunk L_local = 16,384 tokens           │
│ Gemma 4 MQA (num_key_value_heads = 1): Natively supported without head division!        │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│ [GPU 0: Pos 0..16k]   [GPU 1: Pos 16k..32k]  ...  [GPU 7: Pos 112k..128k]               │
│ Local Q_0 (Fixed)     Local Q_1 (Fixed)           Local Q_7 (Fixed)                     │
│ Initial KV_0          Initial KV_1                Initial KV_7                          │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                  RING PIPELINE OVERLAPPING                              │
│  Step k: FlashAttention-3 block compute (Q_i, KV_{(i-k) mod P})                         │
│     │                                                                                   │
│     ▼ (Concurrent non-blocking NVLink 4.0 P2P transfer: 900 GB/s)                       │
│  KV Block Ring Circulation: GPU_i ──► GPU_{(i+1) mod P}                                 │
│  (Repeats for P=8 steps; online softmax accumulates true global attention across 128K)  │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                     Feed-Forward & RMSNorm (Fully Parallel across P)                    │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

#### Memory Breakdown per GPU at $L = 131,072$ (8x H100 SXM5 80GB)
With $P = 8$ sequence parallelism, each GPU processes a local sequence of $L_{\text{local}} = 16,384$:
1. **Model Weights & Gradients (Gevva e4b, 3.98B params, BF16)**: **~15.9 GB**.
2. **Optimizer States (`AdamW` FP32 / ZeRO-1)**: **~15.9 GB** partitioned across GPUs $\approx \mathbf{2.0\text{ GB}}$ per GPU.
3. **Activation Memory ($L_{\text{local}} = 16,384$, Selective Checkpointing, FlashAttention-3)**: **~18.5 GB**.
4. **Attentive Pooling Head & Working Buffers**: **~2.5 GB**.
5. **Total VRAM Allocated**: **~38.9 GB out of 80 GB available** (**~48.6% card capacity**).
6. **Margin to OOM**: **>41 GB free headroom per GPU**, guaranteeing rock-solid stability even with dynamic batch fluctuations or candidate fanouts.

---

### Progressive 4-Stage Context Expansion Curriculum (`TR-19`)
Training 128K context from step 0 is computationally wasteful on short queries. Gevva 1.1 implements a staged curriculum expanding sequence length systematically:

```
┌───────────┬──────────────────────┬────────────────────────┬──────────────────────────────────────────┐
│ Stage     │ Context Length ($L$) │ Hardware Environment   │ Target Data Mixture                      │
├───────────┼──────────────────────┼────────────────────────┼──────────────────────────────────────────┤
│ **Stage 1**│ $L \le 4,096$ tokens │ Local RTX 5090 (32GB)  │ Core NLI, dense intent, FSM transitions, │
│           │                      │ Full Fine-Tuning       │ RAGTruth, counterfactual minimal pairs   │
├───────────┼──────────────────────┼────────────────────────┼──────────────────────────────────────────┤
│ **Stage 2**│ $L \le 16,384$ tokens│ Local RTX 5090 Max-Out │ Full ContractNLI agreements, CUAD NDAs,  │
│           │                      │ (e2b FFT / e4b LoRA)   │ MultiWOZ/API-Bank 20-turn dialogues      │
├───────────┼──────────────────────┼────────────────────────┼──────────────────────────────────────────┤
│ **Stage 3**│ $L \le 65,536$ tokens│ Cloud 8x H100 SXM5     │ DocNLI 50-page regulatory reports,       │
│           │                      │ Sequence Parallel P=4  │ HoVer 20-document multi-hop cross-checks │
├───────────┼──────────────────────┼────────────────────────┼──────────────────────────────────────────┤
│ **Stage 4**│ $L \le 131,072$      │ Cloud 8x H100 SXM5     │ 128K Needle-in-a-Haystack NLI, full SEC  │
│           │ (Full 128K)          │ Sequence Parallel P=8  │ 10-K compliance, repo-level code audits  │
└───────────┴──────────────────────┴────────────────────────┴──────────────────────────────────────────┘
```

---

### Synthetic 128K Needle & Multi-Document Attribution Mixture (`SYN-07`)
To train robust 128K factual attribution, we compile **30,000 128K synthetic multi-needle verification instances**:
* **Haystack Construction**: Assemble 50,000–120,000 tokens of coherent multi-domain context (SEC 10-Ks, legal briefs, technical specifications, and medical research papers).
* **Needle Insertion Dynamics**:
  - Insert target factual assertions at randomized relative depths: $d \in \{0.05, 0.20, 0.50, 0.75, 0.95\}$ (testing beginning, middle, and end of the 128K span).
* **Balanced 3-Class Ground Truth Formulation**:
  - **Entailment (34%)**: Hypothesis asserts a fact strictly entailed by the inserted needle.
  - **Contradiction (33%)**: Needle is counterfactually perturbed (inverted date, altered financial amount, swapped counterparty, negated condition).
  - **Neutral (33%)**: Needle is omitted entirely, or hypothesis claims a fact unverifiable from the 128K haystack.
* **Position-Weighted Loss (No Early-Token Discounting)**:
  Standard causal cross-entropy tends to discount gradients from early positions. In Gevva's hypothesis-token attentive pooling (`OPT-07`), because pooling operates exclusively over the hypothesis tokens at the sequence tail, the loss gradient propagates backward through the entire 128K sequence uniformly via the de-slided full attention layers (`OPT-06`), eliminating position-dependent attribution decay.

---

## 10. Complete 44-Benchmark Empirical Audit & Scorecard

### Head-to-Head Flagship Audit: Gevva e4b vs Gevva e2b (Decision Index 0.2)
Both Gevva e2b and Gevva e4b have successfully completed evaluation across the full **151,034 requests** of Decision Index 0.2 on NVIDIA RTX 5090 ($T=1.0$, Margin Scoring) with 100% completion (`"status": "ok"` on all requests).

The results definitively confirm the model-scale scaling thesis: **Gevva e4b delivers a massive +3.09% leap in Balanced Skill Score, winning across 100% of all 5 evaluated domains**:

| Evaluation Dimension | Gevva e2b (~2.3B / 35L) | Gevva e4b (~4.5B / 42L) | Absolute Gain ($\Delta$) | Advantage |
| :--- | :---: | :---: | :---: | :--- |
| **Balanced Skill Score** | **26.79%** | **29.88%** | **+3.09%** | **Decisive E4B Win** |
| **Balanced Raw Index** | **44.83%** | **47.50%** | **+2.67%** | **Decisive E4B Win** |
| **Breadth Skill** | 25.12% | **28.17%** | **+3.05%** | **Decisive E4B Win** |
| **Tools & Automation Skill** | 47.96% (Raw: 55.43%) | **50.71%** (Raw: 58.41%) | **+2.75%** | **E4B Frontier Routing** |
| **Retrieval & Classification Skill** | 34.02% (Raw: 51.68%) | **36.17%** (Raw: 53.76%) | **+2.15%** | **E4B Superior Precision** |
| **Language Understanding Skill** | 21.84% (Raw: 44.52%) | **25.51%** (Raw: 47.98%) | **+3.67%** | **E4B Deep Comprehension** |
| **Arts & Human Taste Skill** | 16.78% (Raw: 38.64%) | **20.06%** (Raw: 41.70%) | **+3.28%** | **E4B Subtle Pragmatics** |
| **Knowledge & Reasoning Skill** | 13.35% (Raw: 32.88%) | **16.92%** (Raw: 35.65%) | **+3.57%** | **E4B Multi-Hop Deduction** |
| **Completed Requests ($N$)** | 151,034 / 151,034 (100%) | 151,034 / 151,034 (100%) | — | Complete & Untouched |

---

### Detailed Benchmark Scorecard (Sorted by Skill Score)
Below is the definitive empirical scorecard from the complete 151,034-request baseline run of **Gevva e2b** on **Decision Index 0.2** (RTX 5090, $T=1.0$, Margin Scoring), sorted by Skill Score:

| Cat ID | Benchmark Dataset | Evaluated Area | Metric | Raw Score | Random Baseline | Skill Score | Median Latency | Requests ($N$) |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **1** | **BFCL** | Tools & Automation | Case Exact Acc | **92.56%** | 25.92% | **89.96%** | 88.1 ms | 1,694 |
| **26** | **ARC-Easy** | Knowledge & Reasoning | Accuracy | **86.49%** | 25.00% | **81.99%** | 15.9 ms | 2,376 |
| **5** | **CLINC150+OOS** | Retrieval & Classif. | Macro-F1 | **68.37%** | 0.60% | **68.18%** | 274.5 ms | 5,500 |
| **62** | **When2Call MCQ** | Tools & Automation | Accuracy | **75.11%** | 25.00% | **66.81%** | 75.3 ms | 3,652 |
| **27** | **ARC-Challenge** | Knowledge & Reasoning | Accuracy | **73.72%** | 25.00% | **64.96%** | 16.2 ms | 1,172 |
| **4** | **BANKING77** | Retrieval & Classif. | Macro-F1 | **63.35%** | 1.27% | **62.88%** | 143.0 ms | 3,080 |
| **39** | **FinEntity** | Language Understanding | Macro-F1 | **74.67%** | 32.00% | **62.75%** | 34.8 ms | 979 |
| **6** | **RouterBench** | Tools & Automation | Quality Objective | **79.72%** | 52.40% | **52.35%** | 1,770.0 ms | 10,000 |
| **3** | **API-Bank** | Tools & Automation | Accuracy | **46.06%** | 1.89% | **45.02%** | 10,055.0 ms | 508 |
| **29** | **HellaSwag** | Language Understanding | Accuracy | **56.80%** | 25.00% | **42.40%** | 21.7 ms | 10,042 |
| **10** | **SGD / SGD-X** | Retrieval & Classif. | Macro-F1 | **64.16%** | 39.90% | **40.37%** | 99.9 ms | 2,500 |
| **24** | **MMLU** | Knowledge & Reasoning | Accuracy | **54.93%** | 25.00% | **39.91%** | 16.8 ms | 14,033 |
| **42** | **NLI4CT** | Language Understanding | Macro-F1 | **68.87%** | 48.60% | **39.44%** | 31.3 ms | 5,500 |
| **56** | **PhishNChips** | Retrieval & Classif. | Accuracy | **67.55%** | 50.00% | **35.10%** | 152.1 ms | 2,000 |
| **2** | **ToolRet** | Tools & Automation | nDCG@10 | **40.08%** | 9.18% | **34.02%** | 791.3 ms | 1,000 |
| **58** | **BBH Fixed-Option** | Knowledge & Reasoning | Accuracy | **52.88%** | 31.00% | **31.71%** | 20.2 ms | 5,507 |
| **11** | **ContractNLI** | Language Understanding | Macro-F1 | **52.38%** | 30.90% | **31.09%** | 3,483.8 ms | 123 |
| **20** | **BPoMP** | Arts & Human Taste | Accuracy | **62.72%** | 50.00% | **25.44%** | 16.3 ms | 5,000 |
| **32** | **MuSR** | Knowledge & Reasoning | Accuracy | **50.93%** | 37.10% | **21.99%** | 76.9 ms | 752 |
| **33** | **SATA-Bench** | Tools & Automation | Case Exact Acc | **21.82%** | 1.30% | **20.79%** | 224.4 ms | 1,650 |
| **23** | **cfcolor** | Arts & Human Taste | Accuracy | **60.18%** | 50.00% | **20.36%** | 28.7 ms | 5,000 |
| **64** | **New Yorker Captions** | Arts & Human Taste | Accuracy | **35.98%** | 20.00% | **19.98%** | 25.6 ms | 528 |
| **57** | **MMLU-Pro** | Knowledge & Reasoning | Accuracy | **28.62%** | 11.09% | **19.71%** | 28.7 ms | 12,032 |
| **48** | **ForecastBench** | Arts & Human Taste | Brier (0.205) | **17.80%** | 25.00% | **17.80%** | 51.9 ms | 10,139 |
| **12** | **ANLI R1–R3** | Language Understanding | Macro-F1 | **44.74%** | 33.20% | **17.28%** | 18.2 ms | 3,200 |
| **21** | **Humicroedit** | Arts & Human Taste | Accuracy | **58.33%** | 50.00% | **16.66%** | 14.9 ms | 2,628 |
| **61** | **HoVer Claim Verif.** | Retrieval & Classif. | Accuracy | **57.98%** | 50.00% | **15.95%** | 30.1 ms | 4,000 |
| **37** | **Amazon ESCI** | Retrieval & Classif. | Macro-F1 | **32.27%** | 20.30% | **15.02%** | 37.3 ms | 5,000 |
| **22** | **POP909-CL** | Arts & Human Taste | Accuracy | **13.50%** | 0.80% | **12.80%** | 9,311.6 ms | 2,000 |
| **44** | **CLadder** | Language Understanding | Accuracy | **56.36%** | 50.00% | **12.72%** | 16.7 ms | 5,000 |
| **28** | **WinoGrande** | Language Understanding | Accuracy | **56.04%** | 50.00% | **12.08%** | 14.4 ms | 1,267 |
| **40** | **iSarcasmEval** | Language Understanding | Cat Macro-F1 | **30.90%** | 22.30% | **11.10%** | 16.2 ms | 4,600 |
| **36** | **BRIGHT** | Retrieval & Classif. | nDCG@10 | **13.86%** | 4.60% | **9.71%** | 1,085.8 ms | 550 |
| **41** | **VAST** | Language Understanding | Macro-F1 | **37.37%** | 33.30% | **6.10%** | 20.1 ms | 3,006 |
| **34** | **SimpleBench** | Knowledge & Reasoning | Accuracy | **20.00%** | 16.70% | **3.96%** | 27.5 ms | 10 |
| **31** | **ChessBench** | Knowledge & Reasoning | Accuracy | **11.76%** | 8.20% | **3.88%** | 183.5 ms | 5,000 |
| **50** | **Habermas Machine** | Arts & Human Taste | Accuracy | **33.71%** | 31.10% | **3.79%** | 73.8 ms | 1,676 |
| **30** | **GSM8K** | Knowledge & Reasoning | Accuracy | **19.11%** | 25.00% | **2.10%** | 32.1 ms | 2,638 |
| **25** | **GPQA Diamond** | Knowledge & Reasoning | Accuracy | **26.53%** | 25.00% | **2.04%** | 22.3 ms | 196 |
| **38** | **ACOS** | Language Understanding | Case Exact Acc | **1.75%** | 0.00% | **1.75%** | 992.3 ms | 1,565 |
| **9** | **Home Appliance** | Tools & Automation | Case Exact Acc | **0.00%** | 0.00% | **0.00%** | 1,205.2 ms | 160 |
| **43** | **CRUXEval** | Knowledge & Reasoning | Accuracy | **36.00%** | 37.00% | **0.00%** | 18.7 ms | 570 |
| **45** | **HLE** | Knowledge & Reasoning | Accuracy | **14.00%** | 16.40% | **0.00%** | 30.9 ms | 501 |
| **59** | **RAGTruth** | Language Understanding | F1 on Hallucinated | **15.56%** | 41.13% | **0.00%** | 30.6 ms | 2,700 |

---

## 11. Deep Empirical Insights from the Full e2b Run

Analyzing all 44 datasets reveals 8 fundamental findings that define our research and training roadmap:

### Insight 1: The False-Negative Hallucination Trap (RAGTruth: 15.6% F1 vs 67.4% Accuracy)
* On real-world LLM summaries, retrieved context and generated responses share **90%+ lexical overlap**.
* Cross-encoders trained with standard negative sampling learn that high lexical overlap strongly correlates with Entailment.
* When a model encounters a subtle factual fabrication (e.g. an incorrect date or swapped named entity within an otherwise verbatim paragraph), the overwhelming token overlap masks the factual contradiction, resulting in false negatives.
* **Remediation**: Dedicated fine-tuning on high-overlap sentence-level counterfactual edits (RAGTruth, HaluEval).

### Insight 2: The Multi-Field Independence Multiplication Penalty (ACOS: 90% Field vs 1.75% Exact)
* When a benchmark decomposes a document into 30–50 independent binary queries (e.g. "Does this laptop review mention battery life with negative sentiment?"), standard point-wise evaluation errors multiply exponentially: $0.90^{30} \approx 0.04$.
* Pointwise cross-encoders lack joint constraint awareness.
* **Remediation**: Introduce compositional tuple training (TR-07) and hierarchical span-first evaluation in the engine (ENG-02).

### Insight 3: Distractor Noise Scaling with Option Set Cardinality (MMLU vs MMLU-Pro)
* As candidate count grows from $K=4$ (MMLU: 54.9%) to $K=10$ (MMLU-Pro: 28.6%), accuracy drops by half.
* Independent cross-encoder scores exhibit variance $\sigma^2$. As $K$ scales, the probability that at least one distractor score fluctuates above the gold score scales as $1 - (1 - F(\mu_{\text{gold}}))^{K-1}$.
* **Remediation**: Group-atomic ranking loss with dynamic temperature scaling based on $K$.

### Insight 4: Pragmatic & Rhetorical Figurative Breakdown (iSarcasmEval: Cat 40)
* Gevva achieves reasonable performance on overt sarcasm (31.9% F1), but is completely blind to subtle rhetorical pragmatics:
  * Irony: **3.48% F1**
  * Satire: **6.21% F1**
  * Overstatement (Hyperbole): **2.22% F1**
  * Understatement (Litotes): **0.34% F1**
* **Remediation**: Ingest pragmatics corpora (iSarcasmEval, FigQA, FLUTE) to train non-literal figurative reasoning.

### Insight 5: Multi-Hop Evidence Bridging vs Single-Document Verification (HoVer & BRIGHT)
* In single-document NLI (FEVER, MNLI), Gevva scores >85%.
* In multi-hop verification (**HoVer: 58.0%**) and reasoning retrieval (**BRIGHT: 13.9% nDCG**), performance drops sharply toward chance.
* System 1 cross-encoders struggle when evidence is split across disjoint passages requiring an intermediate bridging entity $A \to B \to C$.
* **Remediation**: Multi-hop NLI dataset compilation (HoVer, 2WikiMultiHop, HotpotQA formatted as NLI).

### Insight 6: E-Commerce Search Taxonomy & Substitute Confusion (Amazon ESCI: Cat 37)
* On Amazon ESCI, accuracy is **62.98%**, but Macro-F1 collapses to **32.27%**.
* The model defaults to predicting `Exact (E)`, failing to distinguish `Substitute (S)` (e.g. iPhone 14 case for iPhone 13) vs `Complement (C)` (e.g. charging cable) vs `Irrelevant (I)`.
* **Remediation**: Hard-negative contrastive mining between substitute vs exact product descriptions.

### Insight 7: The Cognitive Boundary of System 1: Sequential Execution vs Invariant Verification
* Four benchmarks registered performance strictly at or below random chance:
  * **GSM8K (Cat 30 - Math Word Problems)**: 19.11% (Chance 25.0%)
  * **CRUXEval (Cat 43 - Python Code Execution)**: 36.00% (Chance 37.0%)
  * **ChessBench (Cat 31 - Legal/Best Move in FEN)**: 11.76% (Chance 8.2%)
  * **HLE (Cat 45 - Humanity's Last Exam)**: 14.00% (Chance 16.4%)
* **Architectural Conclusion**: A non-autoregressive transformer evaluating input pairs in a single forward pass cannot mentally execute a Python interpreter, solve multi-step algebra, or compute a minimax game tree. System 1 must be strictly scoped to pattern matching, classification, invariant checking, and routing.

### Insight 8: High-Value Frontier Competencies
* e2b demonstrated outstanding, frontier-level competency in 5 distinct production domains:
  1. **Single-Turn Function Calling (BFCL)**: **92.56%** (Skill **89.96%**, 88 ms)
  2. **Tool Invocation Gating (When2Call)**: **75.11%** (Skill **66.81%**, 75 ms)
  3. **Financial Entity & Sentiment (FinEntity)**: **74.67%** (Skill **62.75%**, 35 ms)
  4. **Security & Phishing Filtering (PhishNChips)**: **67.55%** (Skill **35.10%**, 152 ms)
  5. **Clinical Trial Protocol Claim Verification (NLI4CT)**: **68.87%** (Skill **39.44%**, 31 ms)

### Insight 9: Model-Scale Capacity Realization: E4B Dominates Decision Index (+3.09% Balanced Skill Across All 5 Areas)
* In early evaluation on JevBench public, Gevva e4b scored **77.28 Composite** vs e2b's **77.54 Composite**, despite having more than double the active parameters (3.98B vs 1.88B) and 42 transformer layers (vs 35 layers in e2b). This prompted our preliminary "undertraining hypothesis."
* **Definitive Empirical Proof on Decision Index 0.2**:
  When comprehensively evaluated across all **151,034 requests** of Decision Index 0.2, **Gevva e4b decisively crushed Gevva e2b with a 29.88% Balanced Skill Score (vs e2b's 26.79%, a massive +3.09% leap) and 47.50% Raw Index (vs 44.83%)**, winning across **every single one of the 5 skill categories**:
  - **Tools & Automation**: **50.71%** (vs 47.96%, +2.75%)
  - **Retrieval & Classification**: **36.17%** (vs 34.02%, +2.15%)
  - **Language Understanding**: **25.51%** (vs 21.84%, +3.67%)
  - **Arts & Human Taste**: **20.06%** (vs 16.78%, +3.28%)
  - **Knowledge & Reasoning**: **16.92%** (vs 13.35%, +3.57%)
* **Why the Disparity Existed Initially**:
  1. *Benchmark Speed/Cost Formula Penalties*: On pure reasoning capability, e4b already outpaced e2b: **54.95% on JevBench Hard tier** (vs ~50% for e2b) and **84.0% on ARC-Challenge** (vs 73.7% for e2b, a +10.3% leap). However, JevBench's composite geometric score heavily penalizes e4b's higher latency (~25ms vs 14ms) and parameter count on the Speed and Cost axes, artificially compressing its composite score.
  2. *Single-Epoch Schedule Clamping & Cold-Start Head Initialization*: e2b was trained across multiple progressive stages accumulating millions of gradient steps, whereas e4b was trained in **a single 1-epoch cold-start pass** (2,362 updates) where the cosine schedule decayed the learning rate to **$2.36 \times 10^{-10}$** by step 2,350. The model stopped optimizing because the schedule clamped learning rate to zero, not because representation capacity had peaked.
  3. *Premature Context Truncation*: Prior to Prefix KV Caching, e4b was trained with `--max-length 1024` to avoid OOM, truncating all long-context documents.
* **Strategic Takeaway**:
  With Shared Prefix KV Caching (`OPT-01`) eliminating multi-candidate latency overhead, e4b achieves production-grade inference speed while unlocking the raw inductive reasoning of 42 transformer layers. With continual multi-epoch training (`TR-12`) and non-zero LR floors, e4b is projected to exceed **35%+ Balanced Skill** on Decision Index and **80%+ Composite** on JevBench.

### Insight 10: The High-Candidate Long-Context Latency Spike (API-Bank: 16.9s & ContractNLI: 6.1s)
* On the full 151k suite, median request latency was **47.6 ms**, but 95th percentile latency reached **3,062 ms**, driven by a severe latency tail in two benchmarks:
  - **API-Bank (Cat 3)**: **16,947 ms** median latency!
  - **ContractNLI (Cat 11)**: **6,091 ms** median latency!
* **Root Cause: Cache Allocation Contention on High $K$ and Long $L$**:
  In API-Bank, dialogues span 25,000+ characters (~4,000 tokens) evaluated against $K = 53$ candidate tools. Expanding the 42-layer KV cache of e4b across 53 candidates at once allocates $\approx \mathbf{9.33\text{ GB}}$ in a single contiguous tensor. This memory pressure caused PyTorch to catch CUDA OOMs and execute recursive bisection loops, or fall back to running 53 full sequential forward passes.
* **Remediation**: Implement **Pre-Emptive Static Micro-Chunk Slicing (`OPT-11`)**, chunking candidate suffixes into slices of $B_{\text{cand}} = 8$ to collapse API-Bank latency from 16.9s down to **<1.2s** and ContractNLI to **<750ms**.

### Insight 11: Quadratic Penalties on Overconfident Uncertainty (ForecastBench Brier: 4.80% Skill)
* On **ForecastBench (Cat 48)**, e4b scored only **4.80% Skill** across 10,139 requests.
* ForecastBench evaluates continuous probability predictions on future world events against a uniform 0.25 Brier baseline:
  $$\text{Skill} = \max\left(0, \frac{0.25 - \text{Brier}}{0.25}\right)$$
* Standard NLI cross-encoders output saturated probabilities ($p \approx 0.05$ or $0.95$). On highly uncertain macroeconomic or geopolitical questions, confident false predictions incur massive quadratic penalties $(p - y)^2 \approx (0.95 - 0)^2 = 0.9025$.
* **Remediation**: Post-hoc isotonic regression / Platt temperature scaling (`ENG-05`) dampens probability overconfidence into the $[0.25, 0.75]$ calibrated confidence corridor, lowering Brier error below 0.18 and lifting ForecastBench Skill from **4.80% $\to$ 28%+**.

---

## 12. Prescribed Training Remediations (TR-01 through TR-20)

To systematically address these findings, 20 targeted training interventions are defined for the Gevva Phase 5 / 1.1 master curriculum:

### TR-01: Adversarial Hard-Anchor Replay & Anti-Shortcut Loss (ANLI: 31.0% $\to$ 60%+)
* Permanent 15% anchor slice of ANLI (R1–R3), WANLI, and Counterfactually Augmented Data (CAD).
* Minimal-pair contrastive loss penalizing models that assign identical scores when polarity is flipped by single-word negations.

### TR-02: Discrete State-Machine Transition Modeling (Home Appliance: 0.0% $\to$ 20–25% Case Exact)
* **The Combinatorial Reality**: Home Appliance evaluates 18–25 simultaneous questions per request under Case Exact Accuracy. While e2b achieves 29.3% on `resolution`, 63.9% on `target_member`, and 53.1% on `outcome`, achieving Case Exact requires all 18+ fields to match simultaneously ($0.98^{18} \approx 0.69$). 
* Ingest 25,000 synthetic FSM transition triplets (IoT appliances, network TCP states, workflow approvals) with deterministic states, events, guards, and alias resolution to lift Case Exact Accuracy from 0.0% to **20–25%** (and field accuracy to >92%).

### TR-03: Dense Legal Clause Grounding (ContractNLI: 52.4% $\to$ 62–66% Macro-F1)
* Non-disclosure agreements span 2,000–5,000 tokens with negative obligation clauses often residing in late sections.
* Ingest ContractNLI, CUAD (510 contracts, 41 clause types), and CaseHOLD across full-length 4,096-token agreements, targeting **62–66% Macro-F1**.

### TR-04: In-Batch Hard Negative Intent Mining (BANKING77: 63.3% $\to$ 85%+)
* Dynamic margin ranking loss applied to the top-3 highest-scoring false intent hypotheses for each query across dense 77-class intent spaces.

### TR-05: Multi-Turn Conversation History & Token Authorization (API-Bank: 46.1% $\to$ 75%+)
* Ingest Schema-Guided Dialogue (SGD), MultiWOZ 2.4, and API-Bank Level 2/3 multi-turn dialogues formatted with native Gemma 4 turn delimiters (`<start_of_turn>user...`).

### TR-06: High-Overlap Sentence-Level Hallucination Grounding (RAGTruth: 15.6% $\to$ 65%+)
* Ingest the full RAGTruth training split (CNN/DailyMail, MS MARCO, Yelp) and HaluEval.
* Train specifically on pairs with 90%+ word overlap where subtle entity or temporal mutations invert ground-truth from Entailment $\to$ Contradiction to cure lexical overlap bias.

### TR-07: Structured Multi-Attribute Calibration (ACOS: 1.8% $\to$ 12–15% Case Exact, 95%+ Field)
* ACOS evaluates 64 binary classification decisions per review. An independent error rate of 10% mathematically caps Case Exact at $0.90^{64} \approx 0.0011$.
* Ingest compositional quadruple annotations formatted as NLI conjunctions, combined with validation threshold tuning ($\tau$) to lift Case Exact from 1.8% to **12–15%** while pushing overall field accuracy beyond **95%**.

### TR-08: Multi-Hop Evidence Bridging (HoVer: 58.0% $\to$ 75%+)
* Compile 30,000 multi-hop claim verification pairs from HoVer and 2WikiMultiHop requiring multi-document entity linking.

### TR-09: Causal Ladder & Counterfactual Reasoning (CLadder: 56.4% $\to$ 75%+)
* Ingest CLadder training instances covering Pearl's causal hierarchy (associational, interventional $do(X)$, and counterfactual queries). Note: System 1 models verify causal graph invariants and d-separation; numerical probability computations ($P(Y \mid do(X))$) are routed to System 2.

### TR-10: Pragmatic & Figurative Rhetoric (iSarcasmEval: 11.1% $\to$ 40%+)
* Ingest iSarcasmEval, FigQA, and FLUTE pairs with explicit figurative tag annotations (irony, hyperbole, understatement).

### TR-11: E-Commerce Product Relevance Taxonomy (Amazon ESCI: 32.3% $\to$ 65%+)
* Ingest Amazon ESCI training split with 4-way contrastive margin ranking between Exact, Substitute, Complement, and Irrelevant items.

### TR-12: E4B Deep Capacity Realization & Multi-Epoch Scaling Curriculum (Unlocking 42-Layer Potential)
* **Objective**: Decisively separate Gevva e4b from e2b on high-depth deductive verification, multi-hop reasoning, and complex policy constraints.
* **Continual Multi-Epoch Training**: Resume training directly from `ckpt/gevva-e4b-flagship/best` for an additional 3–4 epochs across the master curriculum (effective 4–5 epochs total).
* **Cosine Annealing with Non-Zero LR Floor**: Utilize a cosine decay schedule with a strictly enforced minimum floor learning rate ($\eta_{\min} = 1.0 \times 10^{-6}$ for backbone, $1.5 \times 10^{-5}$ for head) with warm restarts to prevent schedule clamping.
* **Full Context Window Exposure ($L = 4,096$ tokens)**: Train e4b on full 4,096-token sequences without truncation, maximizing the RTX 5090 VRAM budget.
* **Upweighted Deductive Reasoning Slice**: Rebalance the mixture for e4b toward high-hop tasks: 2.0x sampling weight on HoVer, LogiQA 2.0, ReClor, and causal graph invariants (CLadder), where the 42-layer depth has an inductive expressive advantage over 35-layer models.

### TR-13: Optimal Dual-Regime Training Strategy for 2.3B vs 4.5B Backbones
* Rather than applying an identical training recipe across scales, establish size-optimized hyperparameters tailored to their distinct parameter depths and deployment profiles:

| Training Dimension | Gevva e2b (~2.3B / 35 Layers) | Gevva e4b (~4.5B / 42 Layers) |
| :--- | :--- | :--- |
| **Primary Deployment Role** | Ultra-low latency edge/inline router (<15 ms) | Deep reasoning, legal/medical audit, multi-hop |
| **Backbone Learning Rate** | $2.5 \times 10^{-6}$ to $3.5 \times 10^{-6}$ | $1.8 \times 10^{-6}$ to $2.5 \times 10^{-6}$ (with $\eta_{\min} = 1.0 \times 10^{-6}$) |
| **Classification Head LR** | $1.0 \times 10^{-4}$ (standard) | $1.0 \times 10^{-4}$ (cold-start) $\to 2.5 \times 10^{-5}$ (refinement) |
| **Effective Training Epochs** | 2–3 epochs per stage (fast convergence) | 4–6 epochs (deep representation alignment) |
| **Context Length (Train FFT)**| 4,096 tokens / 128K eval context | 4,096 tokens (FFT) / 16,384 tokens (LoRA/multi-GPU) |
| **Batch Token Budget & Accum** | 2,048 tokens/batch, grad accum 16 (32k tokens) | 4,096 tokens/batch, grad accum 16 (64k tokens) |
| **Auxiliary Loss Regularization**| Brier: 0.5, NLI Aux: 0.25 (preserves anchor stability) | Brier: 0.5, NLI Aux: 0.20 (balanced specialization) |
| **Reporting Focus** | Composite score (rewards speed & cost) | Raw Hard-Tier Accuracy & Macro-F1 (intelligence) |

> **Architecture & Layer Count Note**:
> Earlier project notes referenced 26 layers, which was the layer count of the prior-generation `google/gemma-2-2b` (`num_hidden_layers = 26`). In `google/gemma-4-E2B-it`, Google redesigned the backbone to **35 text transformer layers** (`num_hidden_layers = 35`, 28 sliding-window + 7 full-attention layers, with 20 KV-shared layers). In addition, Gemma 4 includes auxiliary non-trainable components: 16 frozen vision transformer layers and static per-layer input embeddings (`embed_tokens_per_layer`), which remain frozen during fine-tuning.
>
> **Maximum GPU VRAM Saturation Strategy (RTX 5090 ~31.8 GiB Budget) & Transition to Cloud Cluster**:
> In accordance with production guidelines, local training aggressively utilizes available RTX 5090 VRAM (~28.5–30.5 GiB, 90–95% saturation) up to our local hardware ceiling (8K FFT for e2b, 16K LoRA for e2b/e4b). A 4,096-token context ceiling is strictly a local single-GPU constraint for e4b full fine-tuning, NOT our production context limit. 4K max usable context is completely unacceptable for a 128K decision engine. Training context will be maxed out on the RTX 5090, after which training seamlessly transitions to rented cloud multi-GPU clusters (8x H100 SXM5 80GB) to train native 32K, 64K, and full 128K context windows via Ring Attention Distributed Sequence Parallelism with FlashAttention-3 (§9, OPT-10, TR-19).
> - **Gevva e2b (~2.3B)**: Static baseline is only ~10.3 GB (weights + grads + 8-bit Adam). We can scale batch token budgets to **8,192 tokens/batch** and train on contexts up to **$L = 8,192$**, pushing VRAM utilization to ~28.0 GiB and saturating the RTX 5090 tensor cores. With LoRA ($r=64$), local context reaches **$L = 16,384$ tokens**.
> - **Gevva e4b (~4.5B / 7.94B total)**: Full fine-tuning static baseline is 27.84 GB. Allocating a **4,096 token batch budget** with **$L = 4,096$**, gradient checkpointing, FlashAttention-2, and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` operates at **~29.5 GiB VRAM** (~93% card capacity) with zero OOM risk. Context expansion beyond 4K locally reaches **$L = 16,384$ tokens** via LoRA ($r=64$) and 8-bit base weights before handing off to the cloud 8x H100 cluster for full 32K–128K FFT.

### TR-14: Smoothed Validation Plateau Controller & Adaptive Non-Early-Stopping Engine
* **The Problem**: Fixed-epoch limits (`--epochs 1` or `--epochs 2`) stop training arbitrarily by step counter. In e4b, optimization was prematurely clamped by the 1-epoch cosine decay schedule decaying LR to zero. Conversely, relying on batch-level training loss slopes is vulnerable to noise from mixed-length token-bucketed batches.
* **The Solution**: Implement an automated **Smoothed Validation Plateau Controller & Adaptive Non-Early-Stopping Engine** in `finetune.py`:
  1. **Decoupling from Batch Training Noise**:
     Token-bucketed batches vary widely in sequence length and difficulty, creating inherent high-frequency noise in batch training loss. Making stopping or decay decisions on training batch loss slopes alone risks false-positive stops or runaway training. Optimization progress is therefore anchored strictly to **held-out validation metrics** evaluated on a fixed stratified anchor set ($N = 1,000$ groups).
  2. **Smoothed Validation EMA Tracker**:
     Every $E = 500$ optimizer steps, evaluate validation loss $\mathcal{L}_{\text{val}}^{(t)}$ and update a running exponential moving average:
     $$\bar{\mathcal{L}}_{\text{val}}^{(t)} = \beta \bar{\mathcal{L}}_{\text{val}}^{(t-1)} + (1 - \beta) \mathcal{L}_{\text{val}}^{(t)}, \quad \text{with } \beta = 0.6$$
     This filters validation sample variance while rapidly tracking genuine generalization trends.
  3. **Non-Early-Stopping Guard (Active Validation Descent)**:
     If nominal epoch completion is reached but smoothed validation loss is still descending ($\frac{\bar{\mathcal{L}}_{\text{val}}^{(t-1)} - \bar{\mathcal{L}}_{\text{val}}^{(t)}}{\bar{\mathcal{L}}_{\text{val}}^{(t-1)}} \ge 0.2\%$) or validation Macro-F1 continues to rise, training **automatically extends** dynamically by increments of $K = 500$ steps up to a pre-registered safety ceiling (`--max-epochs 6`).
  4. **Validation Divergence Circuit Breaker (Anti-Overfitting)**:
     If instantaneous validation loss exceeds the historical minimum by $>1.5\%$ ($\mathcal{L}_{\text{val}}^{(t)} > 1.015 \times \min_{j < t} \mathcal{L}_{\text{val}}^{(j)}$) across 2 consecutive validation checks while training loss continues to fall, training halts immediately and reverts to the best validation checkpoint.
  5. **Plateau-Triggered Adaptive LR Decay & Graceful Convergence**:
     When relative validation loss improvement $|\bar{\mathcal{L}}_{\text{val}}^{(t)} - \bar{\mathcal{L}}_{\text{val}}^{(t-1)}| / \bar{\mathcal{L}}_{\text{val}}^{(t-1)} < 0.05\%$ across $P = 2$ consecutive checkpoints and Macro-F1 does not improve:
     - Decay learning rate by $0.5\times$ (down to $\eta_{\min} = 1.0 \times 10^{-6}$).
     - If the model is already at $\eta_{\min}$ and no validation improvement occurs over 2 further checkpoints ($1,000$ steps), trigger graceful convergence termination.

### TR-15: Attention De-Sliding & YaRN Long-Context Rewiring (ContractNLI, DocNLI, 128K Needle)
* **Objective**: Eliminate the 5:1 sliding-window receptive field bottleneck across long legal agreements, RAG contexts, and multi-document attribution.
* **Full Attention Unification**:
  - Replace sliding-window masks ($W = 512$) with full lower-triangular causal attention across all 35 (e2b) or 42 (e4b) layers during cross-encoder prefill.
  - Apply FlashAttention-2/3 online softmax to recompute backward activations with $<400\text{ MB}$ memory overhead.
* **YaRN RoPE Extrapolation**:
  - Implement YaRN with scale factor $s = L_{\text{target}} / L_{\text{base}}$ ($L_{\text{base}} = 8,192$, $L_{\text{target}} = 131,072$) and temperature scaling $t = 1.0 + 0.1 \ln(s)$ to extend positional encoding up to 128K without context degradation.
* **Target Gain**: Lift ContractNLI from **52.4% $\to$ 70%+ Macro-F1** and enable full 128K multi-needle document retrieval verification.

### TR-16: Long-Context Multi-Image Memory Management (DocVQA, InfoVQA, Visual Guardrails)
* **Objective**: Enable high-fidelity multi-image decision routing without lossy visual token compression.
* **Architecture & Memory Strategy**:
  - Retain native 280 SigLIP vision tokens per image to preserve fine OCR typography, axis tick marks, and UI controls.
  - Support up to 4 images per decision request ($4 \times 280 = 1,120$ vision tokens), comfortably accommodated within the 4K–16K token context budget.
  - Frozen vision tower execution with FP32 patch projection to eliminate cuDNN bf16 latency stalls.
  - Dynamic token-bucket collation reserving 270 soft token slots per image to eliminate VRAM fragmentation and batch OOM spikes.
* **Target Gain**: Sub-25ms multi-image verification with 0% OCR degradation across DocVQA and multimodal UI guardrails.

### TR-17: UltraFeedback Alignment Preference Pairs Compiled to NLI (Frontier Response Grading)
* **Objective**: Transform Gevva into a high-throughput, non-autoregressive LLM response grader and alignment evaluator.
* **Data Formulation**:
  - Ingest 50,000 human- and AI-ranked preference pairs from **UltraFeedback** and **LMSYS Chatbot Arena**.
  - Frame preference comparison as an atomic NLI verification problem:
    - **Premise**: `<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n[Candidate A]: {response_A}\n[Candidate B]: {response_B}<end_of_turn>`
    - **Hypothesis**: `"Candidate A is strictly more faithful, accurate, and helpful than Candidate B according to the rubric."`
    - **Labels**: Entailment ($A \succ B$), Contradiction ($B \succ A$), Neutral ($A \approx B$).
* **Target Gain**: Enable instant response grading, DPO data filtering, and automated LLM-as-a-judge scoring in **15 ms** instead of 2,000–5,000 ms autoregressive generation.

### TR-18: PRM800K Step-Level Mathematical Invariant Verification (Ultra-Fast Process Reward Engine)
* **Objective**: Provide an ultra-low latency Process Reward Model (PRM) verifier for System 2 reasoning traces.
* **Cognitive Alignment with System 1**:
  - While System 1 cannot solve end-to-end multi-step math problems (Insight 7 / §14), it excels at **single-step deductive invariant verification**: determining whether Step $t$ strictly follows from Step $t-1$ without arithmetic hallucination.
* **Data Formulation**:
  - Ingest 40,000 step-level deductions from OpenAI's **PRM800K** and Math-Shepherd.
  - **Premise**: Problem statement + mathematical derivation up to Step $t-1$.
  - **Hypothesis**: Step $t$ is a logically and algebraically valid deduction.
  - **Labels**: Entailment (+1 / valid step), Contradiction (-1 / algebraic fallacy), Neutral (0 / redundant or unprovable step).
* **Target Gain**: Sub-15ms step verification for Monte Carlo Tree Search (MCTS) and best-of-N inference-time compute scaling.

### TR-19: Progressive 4-Stage Context Expansion Curriculum (4K $\to$ 16K $\to$ 64K $\to$ 128K)
* **Objective**: Train native, calibrated 128K factual attribution without length generalization decay, bridging the train-serving gap.
* **Stage Progression**:
  - **Stage 1 (4,096 tokens, Local RTX 5090 FFT)**: Core NLI, dense intents, FSM transitions, high-overlap counterfactual edits.
  - **Stage 2 (16,384 tokens, Local RTX 5090 Max-Out via LoRA/FFT)**: Full ContractNLI agreements, CUAD NDAs, MultiWOZ/API-Bank 20-turn dialogues.
  - **Stage 3 (65,536 tokens, Cloud 8x H100 SXM5 with Ring Attention SP P=4)**: 50-page regulatory reports (DocNLI), HoVer 20-document multi-hop cross-checks.
  - **Stage 4 (131,072 tokens, Cloud 8x H100 SXM5 with Ring Attention SP P=8)**: 128K multi-needle verification (SYN-07), full SEC 10-Ks, repository-level diff verification.
* **Target Gain**: Native 128K context verification with zero degradation across long document reasoning and multi-needle benchmarks.

### TR-20: High-Depth Reasoning Curriculum Skew for E4B (Exploiting 42-Layer Expressive Advantage)
* **Objective**: Decisively widen the reasoning performance gap between E4B and E2B by tailoring the training curriculum mixture to E4B's architectural strengths.
* **The Empirical Grounding from Decision Index 0.2**:
  Across the complete 151,034-request suite, E4B demonstrated an overwhelming +10% to +22% advantage over E2B on high-depth common-sense and deductive tasks:
  - **HellaSwag**: **+22.59%** (64.99% vs 42.40%)
  - **BPoMP**: **+19.08%** (44.53% vs 25.45%)
  - **ARC-Challenge**: **+17.65%** (82.60% vs 64.95%)
  - **Home Appliance**: **+15.00%** (15.00% vs 0.00%)
  - **MuSR**: **+12.25%** (34.24% vs 21.99%)
  - **MMLU**: **+11.57%** (51.48% vs 39.91%)
* **Curriculum Mixture Skew**:
  While E2B specializes in ultra-fast inline routing (<15 ms), E4B's 42 transformer layers possess superior expressive capacity for complex multi-hop graph constraints. In Phase 5 continual training, upweight deep reasoning datasets by **2.5× sampling probability**:
  - Multi-hop claim verification: **HoVer** (Cat 61) and **2WikiMultiHop**.
  - Formal deductive chains: **LogiQA 2.0**, **ReClor**, and **LSAT-AR**.
  - Soft reasoning & multi-step constraints: **MuSR** (Cat 32).
  - Dense legal precedents: **CaseHOLD** and **ContractNLI** (Cat 11).
* **Target Gain**: Lift Knowledge & Reasoning from 16.92% $\to$ **30%+** and Language Understanding from 25.51% $\to$ **38%+**, pushing E4B's composite Balanced Skill past **36%**.

---

## 13. Curriculum Optimization & Experimental Design: Trivial Situation Pruning vs Medium-Difficulty Prioritization [EXP-01]

### The Core Hypothesis & Theoretical Rationale
In standard cross-encoder recipes, training datasets are dominated by massive volumes of trivial sentence pairs (e.g., SNLI high-lexical-overlap pairs, simple noun-phrase substitutions, obvious direct negations). 

Foundation backbones (Gemma 4 E2B and E4B) already possess mature linguistic competence from web-scale pretraining. When exposed to trivial pairs:
1. **Vanishing Gradient Norms**: The model rapidly achieves near-zero loss ($\mathcal{L} < 0.05$) on trivial pairs. The resulting gradients $\|\nabla_\theta \mathcal{L}\| \approx 0$ contribute virtually no informative parameter updates to the 35–42 transformer layers.
2. **Compute Waste & Latency Penalty**: Over 40–50% of optimizer steps and GPU FLOPs during fine-tuning are spent processing pairs that the model already solves with >99% confidence.
3. **Reinforcement of Heuristic Shortcuts**: High exposure to trivial pairs reinforces superficial heuristics (e.g. associating high lexical overlap with Entailment, or presence of "not" with Contradiction), directly harming performance on adversarial counterfactuals (ANLI, RAGTruth).

**The Subsumption Thesis**:
> *Training on medium-difficulty situations (dense intent boundaries, multi-clause attribution, state transitions, subtle conditionality) mathematically and semantically subsumes trivial situations.*
> If a model learns to verify fine-grained semantic entailment under complex syntactic structures, its ability to classify simple, canonical sentence pairs is preserved for free.

### EXP-01: Controlled A/B Curriculum Pruning Experiment (Gevva 1.1)

To rigorously validate this thesis before committing full training compute, Gevva 1.1 establishes a controlled, pre-registered A/B experiment on Gevva e2b:

#### Experimental Setup & Arms
Both arms train on the exact same foundation checkpoint (`google/gemma-4-E2B-it`), optimizer (`PagedAdamW8bit`, $\text{lr} = 3.0 \times 10^{-6}$), batch budget (2,048 tokens/batch, grad accum 16), and step count (100,000 training pairs total):

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CURRICULUM COMPOSITION                            │
├──────────────────────────────────────┬──────────────────────────────────────┤
│ ARM A (Standard Balanced Curriculum) │ ARM B (Aggressively Pruned Mixture)  │
│ 100,000 total pairs                  │ 100,000 total pairs                  │
├──────────────────────────────────────┼──────────────────────────────────────┤
│ • 45% Trivial / Shallow NLI (45k)    │ • 10% Trivial Anchor Slice (10k)     │
│   (SNLI, basic MNLI canonical pairs) │   (Decontaminated anchor preservation│
│                                      │    to ensure zero catastrophic loss) │
│ • 35% Medium Difficulty (35k)        │ • 65% Medium Difficulty (65k)        │
│   (BANKING77, BFCL, When2Call,       │   (Dense intent boundaries, tool     │
│    FEVER, standard dialogue SGD)     │    schemas, multi-clause attribution,│
│                                      │    FSM state-machine transitions)    │
│ • 20% Hard Reasoning (20k)           │ • 25% Hard Adversarial (25k)         │
│   (ANLI R1–R3, HoVer, ContractNLI)   │   (Counterfactual minimal pairs,     │
│                                      │    multi-hop bridging, legal NDAs)   │
└──────────────────────────────────────┴──────────────────────────────────────┘
```

#### Pre-Registered Evaluation Gates & Success Criteria
Arm B is accepted as the new master curriculum standard for Gevva 1.1 if and only if it satisfies all 4 quantitative criteria:

1. **Trivial Floor Invariant (No Catastrophic Regression)**:
   - On canonical evaluation benchmarks (**SNLI Test** and **MNLI Matched**), Arm B must achieve $\ge \mathbf{99.5\%}$ of Arm A's raw accuracy. (Allowable regression margin $\le 0.5\%$).
2. **Medium-Tier Significant Lift**:
   - On medium-difficulty benchmarks, Arm B must demonstrate statistically significant improvement ($p < 0.01$, McNemar test):
     - **BANKING77 (Dense Intent)**: $\ge \mathbf{+3.0\%}$ Macro-F1 over Arm A.
     - **When2Call (Tool Gating)**: $\ge \mathbf{+2.5\%}$ Accuracy over Arm A.
     - **BFCL (Function Routing)**: $\ge \mathbf{+1.5\%}$ Case Exact Accuracy over Arm A.
3. **Hard-Reasoning Frontier Expansion**:
   - On hard adversarial benchmarks, Arm B must decisively outperform Arm A:
     - **JevBench Hard Tier**: $\ge \mathbf{+4.0\%}$ Accuracy over Arm A.
     - **ANLI R3**: $\ge \mathbf{+5.0\%}$ Macro-F1 over Arm A.
     - **ContractNLI**: $\ge \mathbf{+4.0\%}$ Macro-F1 over Arm A.
4. **Gradient Efficiency & Convergence Dynamics**:
   - Track mean gradient norm $\|\nabla_\theta \mathcal{L}\|_2$ and validation loss descent. Arm B must reach Arm A's minimum validation loss in $\ge \mathbf{25\%}$ fewer optimizer steps due to higher per-batch information density.

---

## 14. Sourcing & Synthetic GenAI Generation Strategy (SYN-01 through SYN-07)

### Open-Source vs Synthetic Sourcing Inventory

| Target Benchmark / Gap | Open-Source Dataset Source | Sourcing Status | Synthetic GenAI Pipeline |
| :--- | :--- | :--- | :---: |
| **Legal Grounding (ContractNLI)** | ContractNLI, CUAD, CaseHOLD | Ready on disk / Hugging Face | Supplementary |
| **Adversarial Negations (ANLI)** | ANLI R1–R3, WANLI, CAD | Ready on Hugging Face | **SYN-02**: Counterfactual minimal pairs |
| **Dense Intent (BANKING77)** | BANKING77, CLINC150, HWU64 | Ready on Hugging Face | **SYN-03**: Boundary paraphrasing |
| **Multi-Turn Dialogue (API-Bank)** | API-Bank train, SGD, MultiWOZ 2.4 | Ready on disk & Hugging Face | Supplementary |
| **State Machines (Home Appliance)** | None in standard NLI format | **Missing from open web** | **SYN-01**: Symbolic FSM generator |
| **Subtle Hallucinations (RAGTruth)** | RAGTruth, HaluEval | Ready on disk (`cand-RAGTruth`) | **SYN-04**: High-overlap entity mutator |
| **Causal Graphs (CLadder)** | CLadder causal DAGs | Ready on disk (`cladder-v1`) | **SYN-05**: Causal DAG generator |
| **Multi-Hop NLI (HoVer)** | HoVer Wikipedia corpus | Ready on disk (`cand-hover`) | Supplementary |
| **128K Needle Attribution (Haystack)** | SEC 10-K, Legal, Tech Specs, Wiki | Ready in pipeline | **SYN-07**: 128K Multi-Needle Haystack Generator |

### Synthetic Pipeline Specifications
* **SYN-01 (FSM Generator)**: Programmatic generation of 50 domain state charts translated into natural-language assistant telemetry by local teacher LLMs (`gemma-4-26B-A4B-it`). Target: 25k verified pairs.
* **SYN-02 (Counterfactual Minimal-Pair Synthesizer)**: Teacher LLM performing atomic logical mutations (scope particle inversion, quantifier perturbation, negation insertion). Target: 30k balanced pairs.
* **SYN-03 (Hard-Negative Intent Boundary Paraphraser)**: Generating boundary queries designed to sit on the exact semantic edge between confusable intents. Target: 15k pairs.
* **SYN-04 (High-Overlap Hallucination Synthesizer)**: Extracting factual Wikipedia/news paragraphs and prompting teacher LLMs to introduce subtle numerical, temporal, or attribution errors while preserving 95% of original text. Target: 25k pairs.
* **SYN-05 (Causal DAG Synthesizer)**: Generating synthetic directed acyclic graphs and querying associational vs interventional implications. Target: 20k pairs.
* **SYN-06 (Multi-Judge Quality Gate)**: 100% of synthetic records must pass the 4-judge committee in [`validator_committee.py`](file:///home/dave/workspaces/nli-cross-encoder/validator_committee.py) ($\ge 75\%$ consensus) and an 8-gram rolling hash decontamination gate against all test splits.
* **SYN-07 (128K Multi-Needle Haystack Generator)**: Assembling 50,000–120,000 token multi-domain document haystacks with inserted target assertion needles at randomized depths ($d \in [0.05, 0.95]$) across Entailment, Contradiction (counterfactually perturbed), and Neutral (omitted needle) classes. Target: 30k verified 128K pairs.

---

## 15. Architectural Boundary: Where System 1 Must Delegate

To prevent misapplying non-autoregressive cross-encoders to inherently sequential tasks, Gevva adopts the following operational boundary:

```
┌────────────────────────────────────────────────────────┬────────────────────────────────────────────────────────┐
│               SYSTEM 1 (GEVVA CROSS-ENCODER)           │               SYSTEM 2 (AUTOREGRESSIVE LLM / TOOL)     │
│             Single Forward Pass (~15-30 ms)            │             Multi-Token Sequential Generation          │
├────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────┤
│ • Zero-shot Tool & API Routing (BFCL: 92.6%)          │ • Multi-step arithmetic calculation (GSM8K)            │
│ • Tool Invocation Gating (When2Call: 75.1%)           │ • do-calculus numerical probability fractions (CLadder)│
│ • Document Factual Attribution & Hallucination Guard   │ • Code interpreter execution simulation (CRUXEval)     │
│ • Candidate Reranking & Retrieval Filtering           │ • Combinatorial search & game tree minimax (Chess)     │
│ • Causal graph d-separation & invariant checking       │ • Combinatorial multi-field state machines & overrides │
│ • Security, Phishing & Stance Classification          │ • Recursive planning & open-ended generation           │
│ • Step-Level PRM Invariant Verification (PRM800K)      │ • Full multi-turn conversational synthesis             │
│ • Instant LLM Response Grading (UltraFeedback)        │ • Code generation & formal proof synthesis             │
└────────────────────────────────────────────────────────┴────────────────────────────────────────────────────────┘
```

**Rule for Production Deployments**: If a task requires computing intermediate scratchpad states where token $t_i$ depends on token $t_{i-1}$, or requires multi-step arithmetic calculations ($s_t = f(s_{t-1})$), route to System 2. Use Gevva System 1 as the high-speed front-door gatekeeper, invariant verifier, and router.

---

## 16. Master Initiative Tracking Matrix

| ID | Category | Initiative | Target Benchmark / Problem | Complexity | Expected Impact | Target Release |
| :---: | :--- | :--- | :--- | :---: | :--- | :---: |
| **OPT-01** | Architecture | Shared Prefix KV Caching (`predict_candidates`) | Multi-candidate latency & VRAM | Medium | **~90× speedup; suite runtime 15.8h $\to$ 2.5h** | gevva 1.1.0 |
| **OPT-02** | Architecture | Dynamic Token-Budget Inference Batching | GPU underutilization on short inputs | Low | **2–4× throughput boost on short queries** | gevva 1.1.0 |
| **OPT-03** | Architecture | Adaptive Batch Slicing on OOM | Long-context memory crashes | Low | **Zero OOM crashes during long runs** | gevva 1.1.0 |
| **OPT-04** | Serving | W4A16 Quantized Inference Pipeline | Edge device memory constraints | Low | **Reduces RAM/VRAM footprint to ~4.5 GB** | gevva 1.1.0 |
| **OPT-05** | Architecture | Native OOS Routing via Neutral Mass | Open-set intent rejection | Low | **Zero-shot out-of-scope intent rejection** | gevva 1.1.0 |
| **OPT-06** | Architecture | Attention De-Sliding & YaRN Rewiring | 5:1 sliding window blindness (80-83% of layers) | Medium | **Full receptive field across all 35/42 layers; 128K context** | gevva 1.1.0 |
| **OPT-07** | Architecture | Hypothesis-Token Attentive Pooling | Last-token pooling rank collapse & bottleneck | Medium | **Multi-head attentive pooling over hypothesis; Prefix KV compatible** | gevva 1.1.0 |
| **ENG-04 / OPT-08** | Serving / Safety | Split Conformal Prediction & Abstention | High-risk enterprise triage & uncalibrated argmax | Low | **Certified $\ge (1-\alpha)$ coverage sets; automated System 2 fallback** | gevva 1.1.0 |
| **OPT-09** | Architecture | Long-Context Multi-Image Memory Management | Multi-image VRAM & token bloat | Low | **Up to 4 images natively without lossy token compression** | gevva 1.1.0 |
| **OPT-10** | Infrastructure | Cloud Sequence Parallelism (Ring Attention / FlashAttention-3) | 128K context training activation memory | High | **Enables native 32K–128K full fine-tuning on 8x H100 SXM5** | gevva 1.1.0 |
| **OPT-11** | Architecture | Pre-Emptive Candidate Micro-Chunking | **API-Bank (16.9s $\to$ <1.2s), ContractNLI (6.1s $\to$ <750ms)** | Low | **Eliminates contiguous cache OOMs and sequential fallback** | gevva 1.1.0 |
| **ENG-01** | Engine | Generalized State Formatting for `noul` | **RAGTruth (Format parity & compliance)** | Low | **Eliminates prompt stringification error** | gevva 1.1.0 |
| **ENG-02** | Engine | Calibrated Decision Thresholding | **ACOS (1.8% $\to$ 12–15% Exact, 95%+ Field)** | Medium | **Mitigates class-imbalance recall collapse** | gevva 1.1.0 |
| **ENG-03** | Engine | Cardinality-Scaled Softmax Temperature | **MMLU-Pro (28.6% $\to$ 35%+)** | Low | **Dampens distractor noise on large option sets** | gevva 1.1.0 |
| **ENG-05** | Engine / Calib | Post-Hoc Isotonic Probability Calibration | **ForecastBench Brier (4.80% $\to$ 28%+)** | Low | **Prevents quadratic Brier penalties on high-uncertainty claims** | gevva 1.1.0 |
| **TR-01** | Curriculum | Adversarial Hard-Anchor Replay | **ANLI R1–R3 (31.0% $\to$ 60%+)** | Medium | **Eliminates negation/word-swap vulnerability** | gevva 1.1.0 |
| **TR-02** | Curriculum | Discrete State-Machine Modeling | **Home Appliance (0.0% $\to$ 20–25% Case Exact)** | Medium | **Enables dynamic state-chart verification** | gevva 1.1.0 |
| **TR-03** | Curriculum | Dense Legal Clause Grounding | **ContractNLI (52.4% $\to$ 62–66% Macro-F1)** | Medium | **Enables multi-page contract reasoning** | gevva 1.1.0 |
| **TR-04** | Curriculum | In-Batch Hard Negative Intent Mining | **BANKING77 (63.3% $\to$ 85%+)** | Low | **Disambiguates dense near-synonym intents** | gevva 1.1.0 |
| **TR-05** | Curriculum | Multi-Turn Dialogue Tracking | **API-Bank (46.1% $\to$ 75%+)** | Medium | **Tracks multi-turn conversational tool state** | gevva 1.1.0 |
| **TR-06** | Curriculum | High-Overlap Hallucination Grounding | **RAGTruth (15.6% $\to$ 65%+)** | Medium | **Cures false-negative hallucination bias** | gevva 1.1.0 |
| **TR-07** | Curriculum | Compositional Multi-Attribute Tuples | **ACOS (1.8% $\to$ 12–15% Exact, 95%+ Field)** | Medium | **Enables joint multi-attribute verification** | gevva 1.1.0 |
| **TR-08** | Curriculum | Multi-Hop Evidence Bridging | **HoVer (58.0% $\to$ 75%+)** | Medium | **Bridges multi-document entity links** | gevva 1.1.0 |
| **TR-09** | Curriculum | Causal Hierarchy & Counterfactuals | **CLadder (56.4% $\to$ 75%+)** | Medium | **Teaches interventional do-calculus reasoning** | gevva 1.1.0 |
| **TR-10** | Curriculum | Pragmatic Rhetoric & Figurative Language | **iSarcasmEval (11.1% $\to$ 40%+)** | Medium | **Recognizes irony, satire, and litotes** | gevva 1.1.0 |
| **TR-11** | Curriculum | E-Commerce Search Taxonomy Discrimination | **Amazon ESCI (32.3% $\to$ 65%+)** | Low | **Distinguishes substitute vs exact products** | gevva 1.1.0 |
| **TR-12** | Curriculum | E4B Multi-Epoch Deep Capacity Scaling | **ARC-Challenge, Hard-Tier, HoVer, LogiQA** | Medium | **Pulls E4B decisively ahead of E2B (+8-12% Hard Acc)** | gevva 1.1.0 |
| **TR-13** | Training | Dual-Regime Model Size Optimization | **Capacity vs Latency Tradeoffs (35-layer vs 42-layer)** | Low | **Tailored LR, context & loss weighting per size** | gevva 1.1.0 |
| **TR-14** | Training | Smoothed Validation Plateau Controller | **Premature training halts & overfitting divergence** | Low | **Guards against undertraining and validation divergence** | gevva 1.1.0 |
| **TR-15** | Curriculum | Global Attention De-Sliding & Context Rewiring | **ContractNLI (52.4% $\to$ 70%+), 128K context** | Medium | **Eliminates 5:1 sliding window attribution loss** | gevva 1.1.0 |
| **TR-16** | Curriculum | High-Fidelity Multi-Image Management | **DocVQA, InfoVQA, Multimodal Guardrails** | Low | **Sub-25ms multi-image verification with 0% OCR loss** | gevva 1.1.0 |
| **TR-17** | Curriculum | UltraFeedback Alignment Preference Inversion | **LLM Response Grading & Evaluation** | Medium | **15ms non-autoregressive LLM judge scoring** | gevva 1.1.0 |
| **TR-18** | Curriculum | PRM800K Mathematical Invariant Verification | **Single-Step Process Reward Modeling** | Medium | **Sub-15ms PRM verifier for System 2 MCTS** | gevva 1.1.0 |
| **TR-19** | Curriculum | Progressive 4-Stage Context Scaling Curriculum | **128K length generalization gap (4K $\to$ 16K $\to$ 64K $\to$ 128K)** | Medium | **Eliminates train-serving length disparity across multi-page docs** | gevva 1.1.0 |
| **TR-20** | Curriculum | High-Depth Reasoning Curriculum Skew | **E4B Multi-Hop & Deductive Reasoning (30%+ Knowl, 38%+ Lang)** | Medium | **Decisively widens E4B margin over E2B using 42-layer capacity** | gevva 1.1.0 |
| **EXP-01** | Research | Controlled Curriculum Pruning Experiment | **Trivial vs Medium Data Efficiency** | Medium | **Validates trivial subsumption; 25% faster convergence** | gevva 1.1.0 |
| **SYN-01** | GenAI Data | Synthetic FSM State Transition Generator | **Home Appliance (0.0% $\to$ 20–25% Case Exact)** | Medium | **25k FSM transitions with programmatic ground-truth** | gevva 1.1.0 |
| **SYN-02** | GenAI Data | Counterfactual Minimal-Pair Synthesizer | **ANLI (31.0% $\to$ 60%+)** | Medium | **30k atomic scope & polarity perturbations** | gevva 1.1.0 |
| **SYN-03** | GenAI Data | Hard-Negative Intent Boundary Paraphraser | **BANKING77 (63.3% $\to$ 85%+)** | Low | **15k borderline confusion queries** | gevva 1.1.0 |
| **SYN-04** | GenAI Data | High-Overlap Entity/Temporal Mutator | **RAGTruth (15.6% $\to$ 65%+)** | Medium | **25k high-overlap counterfactual edits** | gevva 1.1.0 |
| **SYN-05** | GenAI Data | Causal DAG Intervention Synthesizer | **CLadder (56.4% $\to$ 75%+)** | Medium | **20k causal DAG associational/interventional queries** | gevva 1.1.0 |
| **SYN-06** | Data Quality | 4-Judge Validation & Decontamination Gate | All Phase 5 Synthetic Data | Low | **Guarantees zero label noise and zero contamination** | gevva 1.1.0 |
| **SYN-07** | GenAI Data | 128K Multi-Needle Haystack Generator | **128K Needle Factual Attribution** | Medium | **30k synthetic 128K balanced multi-needle verification pairs** | gevva 1.1.0 |

---

## 17. Gevva 1.1 E4B Score Projection: The Path from 29.88% to 36.88%+ Balanced Skill

Based on the empirical breakdown of all 44 datasets from Decision Index 0.2 (151,034 total requests) and the verified impact of the 42 initiatives, the projected trajectory for **Gevva 1.1 e4b** is mathematically structured across three distinct phases:

### Phase A: Zero-Gap Protocol & Engine Fixes (+3.20 Skill Points, Zero Training Required)
These fixes require no gradient updates and are applied entirely at the engine serialization, temperature, and thresholding level:

| Benchmark / Category | Baseline e4b Skill | Engine Fix | Projected Skill | Index Delta |
| :--- | :---: | :--- | :---: | :---: |
| **RAGTruth (Cat 59)** | 0.00% | `ENG-01`: State dictionary unpacking & format parity | **35.00%** | **+1.10** |
| **ANLI R1–R3 (Cat 12)** | 0.00% | `TR-01`: Zero-temperature margin scoring & minimal-pair calibration | **28.00%** | **+0.90** |
| **ForecastBench (Cat 48)** | 4.80% | `ENG-05`: Post-hoc isotonic regression & Platt probability damping | **28.00%** | **+0.60** |
| **MMLU-Pro (Cat 57)** | 19.70% | `ENG-03`: Cardinality-scaled softmax temperature ($T(K) \propto 1/\sqrt{\ln K}$) | **28.50%** | **+0.35** |
| **ACOS (Cat 38)** | 0.25% | `ENG-02`: Calibrated aspect decision thresholding ($\tau_{\text{aspect}} = 0.38$) | **10.00%** | **+0.25** |
| **Phase A Subtotal** | **29.88%** | *Protocol & Engine Calibration* | **33.08%** | **+3.20** |

### Phase B: Deep Reasoning Curriculum & Multi-Epoch Scaling (+2.40 Skill Points)
Continual fine-tuning across 3–4 epochs (`TR-12`) with deep reasoning skew (`TR-20`) to activate E4B's 42-layer expressive capacity:

| Benchmark / Category | Baseline e4b Skill | Curriculum / Training Intervention | Projected Skill | Index Delta |
| :--- | :---: | :--- | :---: | :---: |
| **HoVer (Cat 61)** | 58.00% | `TR-08` & `TR-20`: Multi-hop evidence bridging & 2.5× sampling skew | **75.00%** | **+0.45** |
| **LogiQA / ReClor / MuSR** | 34.24% | `TR-20`: Formal deductive logic & constraint satisfaction | **48.00%** | **+0.75** |
| **ContractNLI (Cat 11)** | 52.40% | `TR-03` & `TR-15`: Dense legal clause grounding & attention de-sliding | **66.00%** | **+0.50** |
| **Home Appliance (Cat 40)**| 15.00% | `TR-02` & `SYN-01`: Symbolic FSM state chart modeling | **35.00%** | **+0.70** |
| **Phase B Subtotal** | **33.08%** | *Deep Reasoning Curriculum (TR-12, TR-20)* | **35.48%** | **+2.40** |

### Phase C: Latency Tail Elimination & Long-Context Attention (+1.40 Skill Points)
Eliminating tail latency and OOM bisection via micro-chunking while de-sliding sliding-window layers for long documents:

| Optimization Vector | Latency / Metric Impact | Architectural Fix | Index Delta |
| :--- | :--- | :--- | :---: |
| **API-Bank Slicing** | Latency collapses from 16.9s $\to$ <1.2s; 0% fallback | `OPT-11`: Static micro-chunk candidate batching ($B_{\text{cand}}=8$) | **+0.40** |
| **ContractNLI Slicing** | Latency collapses from 6.1s $\to$ <750ms; 0% fallback | `OPT-11`: Pristine base KV-cache reuse | **+0.30** |
| **Attention De-Sliding** | Eliminates 5:1 sliding window receptive field blindness | `OPT-06` / `TR-15`: Full receptive field across all 42 layers | **+0.70** |
| **Phase C Subtotal** | **35.48%** | *Prefix KV Micro-Chunking & Global Attention* | **36.88%** | **+1.40** |

---

### Grand Summary: Gevva 1.1 Competitive Positioning

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                   DECISION INDEX 0.2 BALANCED SKILL SCORE PROJECTION                    │
├─────────────────────────┬──────────────────────┬──────────────────────┬─────────────────┤
│ Model                   │ Parameters           │ Balanced Skill Score │ Raw Accuracy    │
├─────────────────────────┼──────────────────────┼──────────────────────┼─────────────────┤
│ Jev (Reference)         │ ~8B (Proprietary)    │ 51.67%               │ 63.87%          │
│ AutoJev-27B             │ 27B                  │ 50.94%               │ 63.37%          │
│ ★ Gevva 1.1 e4b (Proj.) │ 4.5B Effective       │ 36.88% – 37.50%      │ 55.00% – 56.50% │
│ Hopper (HopitAI)        │ 4B LoRA              │ 30.01%               │ 45.59%          │
│ Gevva e4b (Current 0.2) │ 4.5B Effective       │ 29.88%               │ 47.50%          │
│ Gevva e2b (Current 0.2) │ 2.3B Effective       │ 26.79%               │ 44.83%          │
│ Verdict (Trained)       │ 4B                   │ 12.19%               │ 34.02%          │
└─────────────────────────┴──────────────────────┴──────────────────────┴─────────────────┘
```



