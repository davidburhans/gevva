# Pull Request: Add GevvaEngine for Gemma 4 System 1 Decision Models

**Target Repository**: [https://github.com/apolinario/decision-index](https://github.com/apolinario/decision-index)  
**Target Branch**: `main`  
**PR Title**: `feat(engines): add GevvaEngine for Gemma 4 System 1 decision models`

---

## Pull Request Description

### Summary of Changes

This PR adds native support for **Gevva** (`GevvaEngine`), enabling the evaluation of Google Gemma 4-based System 1 decision cross-encoders on the Decision Index benchmark suite:

1. **Engine Adapter** ([`decision_index/engines/gevva_engine.py`](file:///home/dave/workspaces/nli-cross-encoder/research/decision_index_upstream/decision_index/engines/gevva_engine.py)):
   - Subclasses `decision_index.engines.base.Engine` with device-synchronized latency measurement.
   - Evaluates multiple-choice (`choice`) questions by scoring candidate options as NLI hypothesis statements against the premise/instructions via calibrated entailment-contradiction margin.
   - Evaluates boolean/yes-no (`noul`) questions with calibrated probability estimates.
   - Outputs strict, valid probability distributions conforming to `validate(questions, response)`.
   - Returns standard `(response, raw)` tuples for seamless integration with `decision_index.runner.run`.

2. **Registry Integration** ([`decision_index/engines/__init__.py`](file:///home/dave/workspaces/nli-cross-encoder/research/decision_index_upstream/decision_index/engines/__init__.py)):
   - Registers `"gevva": "decision_index.engines.gevva_engine:GevvaEngine"` in `REGISTRY`.

3. **Packaging** ([`pyproject.toml`](file:///home/dave/workspaces/nli-cross-encoder/research/decision_index_upstream/pyproject.toml)):
   - Adds `gevva = ["gevva>=1.0.0"]` under `[project.optional-dependencies]`.

4. **Unit Tests** ([`tests/test_gevva_engine.py`](file:///home/dave/workspaces/nli-cross-encoder/research/decision_index_upstream/tests/test_gevva_engine.py)):
   - Full mock test suite verifying `load_engine("gevva", ...)`, `choice` probability distributions, `noul` probabilities, warmup, and runtime introspection.
   - Includes `pytest.importorskip("torch")` and graceful module mocking to pass cleanly in minimal environments.

---

### What is Gevva?

**Gevva** is an open-source (Apache 2.0) family of System 1 decision engines built on Google's Gemma 4:
- **`davidburhans/gevva-e2b`**: 2.3B parameters, ~14.3 ms latency, 128K context window.
- **`davidburhans/gevva-e4b`**: 4.5B parameters, ~17.8 ms latency, 84.0% ARC-Challenge, 128K context window.

Rather than generating tokens autoregressively (which takes 2–5 seconds and burns high GPU memory), Gevva evaluates candidate decisions in a single forward pass and outputs calibrated class probabilities (`Contradiction: 0, Entailment: 1, Neutral: 2`).

- **Weights on Hugging Face**: [https://huggingface.co/davidburhans/gevva-e4b](https://huggingface.co/davidburhans/gevva-e4b) & [https://huggingface.co/davidburhans/gevva-e2b](https://huggingface.co/davidburhans/gevva-e2b)
- **Code & Specs**: [https://github.com/davidburhans/gevva](https://github.com/davidburhans/gevva)

---

### How to Run

Install with the optional `gevva` dependency:
```bash
pip install -e ".[gevva]"
```

Run either Gevva model directly through the pipeline:
```bash
# Evaluate Gevva e4b
decision-index run \
    --engine gevva \
    --option model=davidburhans/gevva-e4b \
    --rows datasets/suite.jsonl.gz \
    --out runs/gevva-e4b

# Evaluate Gevva e2b
decision-index run \
    --engine gevva \
    --option model=davidburhans/gevva-e2b \
    --rows datasets/suite.jsonl.gz \
    --out runs/gevva-e2b
```

---

### Verification & Test Suite

All 40 unit tests pass locally:
```bash
pytest tests/
# ============================== 40 passed in 0.67s ==============================
```

---

## How to Submit This PR to GitHub

### Option A: Via GitHub Web / Fork (Fastest)

1. Navigate to [https://github.com/apolinario/decision-index](https://github.com/apolinario/decision-index) and click **Fork** (top-right) to create `github.com/<your-username>/decision-index`.
2. Push the local branch to your fork:
   ```bash
   cd /home/dave/workspaces/nli-cross-encoder/research/decision_index_upstream
   git remote add myfork git@github.com:<your-username>/decision-index.git
   git push -u myfork add-gevva-engine
   ```
3. Open your browser to:
   ```
   https://github.com/apolinario/decision-index/compare/main...<your-username>:decision-index:add-gevva-engine
   ```
4. Copy and paste the PR title and description above, and click **Create Pull Request**.

### Option B: Using the Generated Patch File

A complete Git patch file has been generated at:
`docs/0001-feat-engines-add-GevvaEngine-for-Gemma-4-System-1-de.patch`

You can apply it to any clone of `decision-index` with:
```bash
git apply docs/0001-feat-engines-add-GevvaEngine-for-Gemma-4-System-1-de.patch
```
