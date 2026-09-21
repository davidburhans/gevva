# Resources & Foundation References: Multimodal 128K NLI Cross-Encoder

A comprehensive registry of foundational architectures, external models, training datasets, and reference implementations informing this project.

---

## 1. Foundation Models

- **`google/gemma-4-E2B`**: https://huggingface.co/google/gemma-4-E2B
  - ~2.3B effective parameter multimodal base model (vision + text).
  - Native 128K context window with sliding window + global RoPE attention.
  - Primary backbone for our lightweight, edge-optimized cross-encoder.
- **`google/gemma-4-E2B-it-qat-w4a16-ct`**: https://huggingface.co/google/gemma-4-E2B-it-qat-w4a16-ct
  - Google official QAT compressed-tensors checkpoint.
- **`google/gemma-4-E4B`**: https://huggingface.co/google/gemma-4-E4B
  - ~4.5B effective parameter high-capacity multimodal model.
- **`google/gemma-4-E4B-it-qat-w4a16-ct`**: https://huggingface.co/google/gemma-4-E4B-it-qat-w4a16-ct

---

## 2. System 1 Decision Engines & Reference Frameworks

- **TypeSafe AI Jev**: http://typesafe.ai/blog/introducing-system-one-models-and-jev
  - Architectural pioneer for non-autoregressive "System 1" decision engines.
  - Replaces slow token generation (~1–5s) with single-pass sequence classification logits (~25–40ms).
- **ModernCE**: https://huggingface.co/blog/dleemiller/nli-xenc-ways-to-use
  - Modern cross-encoder formulation using standard 3-state NLI (0=contradiction, 1=entailment, 2=neutral).
- **Convai Laya**: https://huggingface.co/convaiinnovations/laya
  - Low-latency conversational decision-making and tool-routing cross-encoder.
- **AlexWortega / OpenJEV**: https://huggingface.co/AlexWortega/openjev
  - Open replication of the Jev System 1 API.
- **Hmm (Muhammed Nazeem)**: https://github.com/n4ze3m/hmm | https://huggingface.co/n4ze3m/Qwen3.5-4B-Hmm
  - Open model for typed decisions with calibrated probability outputs.
- **Mapika / Decider**: https://github.com/Mapika/decider
  - Token-bucket batching, abstention augmentation, and multiple-choice decision routing.
- **SemIf**: https://github.com/TheoLeeCJ/SemIf
  - Semantic if-then routing and cyclic position-debiasing algorithms.
- **Bespoke Labs Nimble**: https://huggingface.co/bespokelabs/Bespoke-Nimble-9B
  - Sub-50ms reasoning and judgment engine.

---

## 3. Training & Evaluation Datasets

### A. System 1 Structured Decision Datasets
- **`n4ze3m/typed-decisions-synth`**: https://huggingface.co/datasets/n4ze3m/typed-decisions-synth
  - **Author**: Muhammed Nazeem (2026)
  - **License**: MIT License
  - **Citation**:
    ```bibtex
    @misc{nazeem2026hmm,
      author = {Muhammed Nazeem},
      title  = {Hmm: a small open model for typed decisions},
      year   = {2026},
      url    = {https://github.com/n4ze3m/hmm}
    }
    ```
  - **Description**: 7,414 cases with 25,859 structured questions across 149 enterprise domains and workflows.
  - **Question Types**:
    - `noul` (10,192 questions): Boolean policy / verification checks.
    - `choice` (10,179 questions): Multi-choice triage, category selection, and tool routing.
    - `score` (5,488 questions): Multi-level ordered rubric evaluation.
  - **Soft Labels**: DeepSeek V4.1 Flash averaged over 3 passes to provide soft probability distributions.
  - **Benchmark Hygiene**: The 4 core benchmark workflows from `LocalLLaMA/typed-decisions` (customer service, invoice processing, security incidents, agent traces) were deliberately held out, preventing evaluation leakage.
  - **Adapter**: Implemented in [`research/adapters/typed_decisions_adapter.py`](file:///home/dave/workspaces/nli-cross-encoder/research/adapters/typed_decisions_adapter.py) with full soft-label mapping and inverted polarity controls.

### B. Core Text & Multimodal NLI
- **Stanford SNLI**: https://huggingface.co/datasets/stanfordnlp/snli
- **NYU MultiNLI (MNLI)**: https://huggingface.co/datasets/nyu-mll/multi_nli
- **Adversarial NLI (ANLI)**: https://huggingface.co/datasets/facebook/anli
- **FEVER Fact Verification**: https://huggingface.co/datasets/fever/fever
- **SciTail**: https://huggingface.co/datasets/allenai/scitail
- **QNLI**: Stanford Question Natural Language Inference.
- **XNLI (Multilingual 15 Languages)**: https://huggingface.co/datasets/facebook/xnli
- **Multimodal Grounding (Synthetic Spatial & Visual)**: Generated via PIL bounding boxes and geometric property verification.

### C. Long-Context & Document Verification
- **Synthetic Needle-in-a-Haystack**: Programmatic multi-depth needle insertion (0% to 100% depth), corrupted needle contradiction mutations, and dropped-needle unanswerable neutral states (up to 128K context).
- **DocNLI**: https://huggingface.co/datasets/DocNLI
  - Multi-paragraph document-level NLI (average 800–3,500 tokens).
- **QASPER-NLI**: Long-form multi-paragraph academic paper citation grounding.
