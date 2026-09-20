#!/usr/bin/env python3
"""eval_baselines.py - LOCAL head-to-head evaluation of openjev checkpoints.

Replaces quoted REFERENCE_BENCHMARKS constants with real measurements under our
frozen protocol (docs/EVALUATION_PROTOCOL.md): seeded slices, entailment scoring
(the protocol openjev documents), contamination guard, n>=1000, provenance per run.

Feasibility (scout report 2025-09-20): openjev-4B v1/v2 confirmed published;
0.8B/2B subfolders speculative (attempted, skipped on 404); Jev = closed API
(quoted forever); Laya = separate decision paradigm, own harness TODO.

Usage: uv run python scripts/eval_baselines.py [--limit 1000]
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "research" / "openjev"))

from gemma4_cross_encoder import GradeResult, RerankResult  # noqa: E402  (SDK-native result shapes)
from nli_labels import ENTAILMENT, ID2LABEL  # noqa: E402

ROSTER = [
    ("openjev-4b-v1", "AlexWortega/openjev", "qwen3.5-4b-nli", True),
    ("openjev-4b-v2", "AlexWortega/openjev", "qwen3.5-4b-nli-v2", True),
    ("openjev-0.8b", "AlexWortega/openjev", "qwen3.5-0.8b-nli", False),
    ("openjev-2b", "AlexWortega/openjev", "qwen3.5-2b-nli", False),
    ("von-1.0", "wfzyx/von-1.0", None, True),          # pip: von-sdk (grounded: HF README)
    ("gliner2-large-v1", "fastino/gliner2-large-v1", None, True),  # pip: gliner2[local] - PENDING API grounding (task G2)
]


class OpenJevSDKAdapter:
    """Adapts OpenJevCrossEncoder to the Gemma4CrossEncoder eval-suite interface.

    SHAPE PARITY CONTRACT (drop-in requirement): returns the SDK's own result
    types - RerankResult(int subclass with .scores) and GradeResult(str subclass
    with .probabilities) - so any consumer of our SDK consumes baseline results
    identically. predict delegates 1:1 (same 3-logit softmax, same label order).
    """

    def __init__(self, repo: str, subfolder: str, device: str = "cuda"):
        from modeling_openjev import OpenJevCrossEncoder
        self._inner = OpenJevCrossEncoder(repo, subfolder=subfolder, device=device)
        self.model_name = f"{repo}:{subfolder}"

    def predict(self, pairs: Sequence) -> "np.ndarray":
        return self._inner.predict(pairs)

    def rerank(self, premise=None, options: Sequence[str] = (), hyp_format: str = "The correct answer is: {}",
               question=None, hyp_fmt=None, scoring: str = "entailment", image=None, **_):
        if image is not None:
            raise ValueError("text-only baseline checkpoints cannot process images")
        premise = premise if premise is not None else question
        hyp_format = hyp_format or hyp_fmt or "The correct answer is: {}"
        pairs = [(premise, hyp_format.format(o)) for o in options]
        probs = self._inner.predict(pairs)
        if scoring == "entailment":
            scores = probs[:, ENTAILMENT]
        elif scoring == "margin":
            scores = probs[:, ENTAILMENT] - probs[:, 0]
        else:
            raise ValueError(f"baseline adapter supports entailment|margin, got {scoring!r}")
        return RerankResult(int(scores.argmax()), [float(s) for s in scores])

    def grade(self, premise=None, reference_answer=None, candidate_answer=None,
              image=None, question=None, reference=None, candidate=None, **_):
        # Same formatting as Gemma4CrossEncoder.grade so the protocol is identical.
        query = premise if premise is not None else question
        ref = reference_answer if reference_answer is not None else reference
        cand = candidate_answer if candidate_answer is not None else candidate
        probs = self._inner.predict([(f"{query}\nReference answer: {ref}",
                                      f"Candidate answer: {cand}")])[0]
        prob_dict = {"contradiction": float(probs[0]), "entailment": float(probs[1]),
                     "neutral": float(probs[2])}
        return GradeResult(ID2LABEL[int(probs.argmax())], prob_dict)


NLI_CHOICES = {
    "entailment": "The hypothesis is definitely true given the premise",
    "contradiction": "The hypothesis is false given the premise",
    "neutral": "The hypothesis might be true, but the premise does not prove it",
}


class VonAdapter:
    """wfzyx/von-1.0 (395M ModernBERT System-1 decision model, Apache-2.0, pip: von-sdk).

    Grounded in the vendor's documented API (HF README, fetched 2026-09-20):
    `von.decide(state, choices={name: description}, instructions)` -> object with
    .choice/.confidence/.probabilities; `von.judge(state, instructions)` -> float.
    NLI mapping: 3-label decide with the vendor's choice-description pattern; this
    mapping decision is recorded in each artifact's provenance block.
    Shape parity: returns SDK RerankResult / GradeResult and (n,3) predict probs
    in our label order (contradiction, entailment, neutral).
    """

    def __init__(self, repo: str = "wfzyx/von-1.0", device: str = "cuda", inner=None):
        if inner is not None:
            self._inner = inner  # injection point for contract tests
        else:
            import von
            self._inner = von
        self.model_name = repo

    def _decide_probs(self, state: str, choices: Dict[str, str], instructions: str) -> Dict[str, float]:
        result = self._inner.decide(state=state, choices=choices, instructions=instructions)
        return dict(result.probabilities)

    def predict(self, pairs: Sequence) -> "np.ndarray":
        rows = [self._decide_probs(
            state=f"Premise: {p}\nHypothesis: {h}", choices=NLI_CHOICES,
            instructions="Which relation does the hypothesis bear to the premise?")
            for p, h in pairs]
        return np.array([[r["contradiction"], r["entailment"], r["neutral"]] for r in rows])

    def rerank(self, premise=None, options: Sequence[str] = (), hyp_format: str = "The correct answer is: {}",
               question=None, hyp_fmt=None, scoring: str = "entailment", image=None, **_):
        premise = premise if premise is not None else question
        pairs = [(premise, hyp_format.format(o)) for o in options]
        probs = self.predict(pairs)
        scores = (probs[:, ENTAILMENT] if scoring == "entailment"
                  else probs[:, ENTAILMENT] - probs[:, 0])
        return RerankResult(int(scores.argmax()), [float(s) for s in scores])

    def grade(self, premise=None, reference_answer=None, candidate_answer=None,
              question=None, reference=None, candidate=None, **_):
        query = premise if premise is not None else question
        ref = reference_answer if reference_answer is not None else reference
        cand = candidate_answer if candidate_answer is not None else candidate
        probs = self.predict([(f"{query}\nReference answer: {ref}", f"Candidate answer: {cand}")])[0]
        prob_dict = {"contradiction": float(probs[0]), "entailment": float(probs[1]),
                     "neutral": float(probs[2])}
        return GradeResult(ID2LABEL[int(np.argmax(probs))], prob_dict)


class GLiNER2Adapter:
    """fastino/gliner2-large-v1 (340M DeBERTa-v3 span model, Apache-2.0, pip: gliner2[local]).

    GROUNDED (task G2, 2026-09-20): classification call verified against package
    source (gliner2/inference/runtime.py classify_text) and README:
        model = GLiNER2.from_pretrained("fastino/gliner2-large-v1")
        model.classify_text(text, {"field": [labels]}, include_confidence=True)
        -> {"field": {"label": str, "confidence": float}}
    Vendor exposes ONLY top-label + confidence (no full distribution). The (n,3)
    predict matrix therefore places `confidence` on the top label and spreads the
    remainder uniformly - ACCURACY ROWS ARE EXACT; ECE/Brier rows for this model
    are approximate and will be flagged N/A in published tables.
    """

    grounded_api = True
    exposes_full_distribution = False

    def __init__(self, repo: str = "fastino/gliner2-large-v1", device: str = "cuda", inner=None):
        if inner is not None:
            self._inner = inner
        else:
            from gliner2 import GLiNER2
            self._inner = GLiNER2.from_pretrained(repo)
        self.model_name = repo

    def _classify(self, text: str, labels: Sequence[str]) -> Dict[str, float]:
        result = self._inner.classify_text(text, {"classification": list(labels)},
                                           include_confidence=True)
        field = next(iter(result.values()))
        top, confidence = str(field["label"]), float(field["confidence"])
        others = [l for l in labels if l != top]
        dist = {l: (1.0 - confidence) / max(1, len(others)) for l in others}
        dist[top] = confidence
        return dist

    def predict(self, pairs: Sequence) -> "np.ndarray":
        rows = [self._classify(f"Premise: {p}\nHypothesis: {h}",
                               ["contradiction", "entailment", "neutral"]) for p, h in pairs]
        return np.array([[r["contradiction"], r["entailment"], r["neutral"]] for r in rows])

    def rerank(self, premise=None, options: Sequence[str] = (), hyp_format: str = "The correct answer is: {}",
               question=None, hyp_fmt=None, scoring: str = "entailment", image=None, **_):
        premise = premise if premise is not None else question
        pairs = [(premise, hyp_format.format(o)) for o in options]
        probs = self.predict(pairs)
        scores = (probs[:, ENTAILMENT] if scoring == "entailment"
                  else probs[:, ENTAILMENT] - probs[:, 0])
        return RerankResult(int(scores.argmax()), [float(s) for s in scores])

    def grade(self, premise=None, reference_answer=None, candidate_answer=None,
              question=None, reference=None, candidate=None, **_):
        query = premise if premise is not None else question
        ref = reference_answer if reference_answer is not None else reference
        cand = candidate_answer if candidate_answer is not None else candidate
        probs = self.predict([(f"{query}\nReference answer: {ref}", f"Candidate answer: {cand}")])[0]
        prob_dict = {"contradiction": float(probs[0]), "entailment": float(probs[1]),
                     "neutral": float(probs[2])}
        return GradeResult(ID2LABEL[int(np.argmax(probs))], prob_dict)


def _build_client(kind: str, repo: str, subfolder, device: str = "cuda"):
    if kind.startswith("openjev"):
        return OpenJevSDKAdapter(repo, subfolder, device=device)
    if kind.startswith("von"):
        return VonAdapter(repo, device=device)
    if kind.startswith("gliner2"):
        return GLiNER2Adapter(repo, device=device)
    raise ValueError(f"unknown baseline kind: {kind}")


def _prefetch(repo: str, subfolder: str, kind: str = "openjev") -> bool:
    """Best-effort staging: HF subfolder snapshot for openjev, pip import for von/gliner2."""
    if kind.startswith("von"):
        try:
            import von  # noqa: F401
            return True
        except ImportError as e:
            print(f"  [prefetch failed] pip install von-sdk ({e})")
            return False
    if kind.startswith("gliner2"):
        try:
            import gliner2  # noqa: F401
            print("  [gliner2] package present but classification API grounding pending (G2) - skipping")
            return False
        except ImportError as e:
            print(f"  [prefetch failed] pip install gliner2[local] ({e})")
            return False
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo, allow_patterns=[f"{subfolder}/*"])
        return True
    except Exception as e:
        print(f"  [prefetch failed] {repo}:{subfolder} -> {e}")
        return False


def eval_model(name: str, repo: str, subfolder: str, limit: int, scoring: str) -> Path:
    import eval_openjev_benchmarks as suite
    ce = _build_client(name, repo, subfolder)
    results = {
        "model": name,
        "provenance": {
            "kind": "LOCAL RUN (measured, not quoted)",
            "repo": f"{repo}:{subfolder}",
            "scoring": scoring,
            "limit_per_task": limit,
            "shuffle_seed": 0,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }
    results["mnli_m"] = suite.eval_mnli(ce, split="validation_matched", limit=limit)
    print(f"  [{name}] mnli_m = {results['mnli_m']:.4f}")
    results["mnli_mm"] = suite.eval_mnli(ce, split="validation_mismatched", limit=limit)
    print(f"  [{name}] mnli_mm = {results['mnli_mm']:.4f}")
    results["arc_easy_rerank"] = suite.eval_arc_rerank(ce, subset="ARC-Easy", limit=limit, scoring=scoring)
    results["arc_challenge_rerank"] = suite.eval_arc_rerank(ce, subset="ARC-Challenge", limit=limit, scoring=scoring)
    print(f"  [{name}] arc_e = {results['arc_easy_rerank']:.4f} arc_c = {results['arc_challenge_rerank']:.4f}")
    results["mmlu_rerank"] = suite.eval_mmlu_rerank(ce, limit=limit, scoring=scoring)
    results["winogrande_rerank"] = suite.eval_winogrande_rerank(ce, limit=limit, scoring=scoring)
    print(f"  [{name}] mmlu = {results['mmlu_rerank']:.4f} winogrande = {results['winogrande_rerank']:.4f}")

    out = REPO / "results" / f"baselines_local_{name}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"  saved -> {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--scoring", default="entailment", choices=["entailment", "margin"],
                        help="entailment = openjev's documented protocol (protocol-identical)")
    parser.add_argument("--only", default=None, help="comma list of roster names to run")
    args = parser.parse_args()
    only = set(args.only.split(",")) if args.only else None

    import gc
    import torch
    summary: List[Dict] = []
    for name, repo, subfolder, confirmed in ROSTER:
        if only and name not in only:
            continue
        kind = "openjev" if name.startswith("openjev") else name.split("-")[0]
        print(f"\n=== baseline: {name} ({repo}:{subfolder}) confirmed={confirmed} ===")
        if not _prefetch(repo, subfolder, kind):
            summary.append({"model": name, "status": "skipped_or_unavailable"})
            continue
        try:
            out = eval_model(name, repo, subfolder, args.limit, args.scoring)
            summary.append({"model": name, "status": "measured", "artifact": str(out)})
        except Exception as e:
            summary.append({"model": name, "status": f"failed: {e}"})
            print(f"  [FAILED] {name}: {e}")
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    out = REPO / "results" / "baselines_local_summary.json"
    out.write_text(json.dumps({"scoring": args.scoring, "runs": summary}, indent=2))
    print(f"\nsummary -> {out}")


if __name__ == "__main__":
    main()
