#!/usr/bin/env python3
"""tests/test_sdk_parity.py - Drop-in contract tests for SDK result shapes.

Guards the "drop-in replacement" promise: the baseline adapter (and by extension
every consumer in the eval suite) must exchange the SDK's own result types -
RerankResult(int + .scores) and GradeResult(str + .probabilities).

Run: uv run python tests/test_sdk_parity.py
"""

import inspect
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from gemma4_cross_encoder import GradeResult, RerankResult  # noqa: E402
from scripts.eval_baselines import OpenJevSDKAdapter  # noqa: E402


class FakeInner:
    """Deterministic 3-logit stand-in for OpenJevCrossEncoder (no GPU, no download)."""

    def __init__(self, probs: np.ndarray):
        self._probs = probs

    def predict(self, pairs):
        return np.array([self._probs[i % len(self._probs)] for i in range(len(pairs))])


def _adapter(rows: np.ndarray) -> OpenJevSDKAdapter:
    adapter = OpenJevSDKAdapter.__new__(OpenJevSDKAdapter)  # skip __init__ (no download)
    adapter._inner = FakeInner(rows)
    adapter.model_name = "fake"
    return adapter


def test_rerank_returns_sdk_rerankresult_with_entailment_scoring():
    # Options: ent 0.7 / 0.2 / 0.5 -> argmax option 0
    rows = np.array([[0.1, 0.7, 0.2], [0.5, 0.2, 0.3], [0.2, 0.5, 0.3]])
    result = _adapter(rows).rerank(premise="p", options=["a", "b", "c"],
                                   hyp_format="{}", scoring="entailment")
    assert isinstance(result, RerankResult), "must be the SDK's RerankResult type"
    assert int(result) == 0 and result.index == 0
    assert result.scores == [0.7, 0.2, 0.5]
    idx, scores = result  # Jev-style tuple unpacking
    assert idx == 0 and len(scores) == 3


def test_rerank_margin_scoring_matches_sdk_rule():
    # margin = P(ent) - P(con): opt0 0.4, opt1 0.5 -> margin picks opt1,
    # while entailment argmax (0.6 vs 0.5) picks opt0 - the rules genuinely differ.
    rows = np.array([[0.2, 0.6, 0.2], [0.0, 0.5, 0.5]])
    result = _adapter(rows).rerank(premise="p", options=["a", "b"], hyp_format="{}", scoring="margin")
    assert isinstance(result, RerankResult) and int(result) == 1
    assert all(abs(s - e) < 1e-9 for s, e in zip(result.scores, [0.4, 0.5]))
    result_ent = _adapter(rows).rerank(premise="p", options=["a", "b"], hyp_format="{}", scoring="entailment")
    assert int(result_ent) == 0


def test_grade_returns_sdk_graderesult_with_probabilities():
    rows = np.array([[0.1, 0.8, 0.1]])
    result = _adapter(rows).grade(question="q", reference="ref", candidate="cand")
    assert isinstance(result, GradeResult), "must be the SDK's GradeResult type"
    assert str(result) == "entailment"
    assert set(result.probabilities) == {"contradiction", "entailment", "neutral"}
    assert abs(result["entailment"] - 0.8) < 1e-9
    label, probs = result  # Jev-style tuple unpacking
    assert label == "entailment"


def test_eval_suite_call_shapes_are_accepted():
    adapter = _adapter(np.array([[0.2, 0.6, 0.2], [0.3, 0.4, 0.3]]))
    # Exactly the call shapes used inside eval_openjev_benchmarks.py:
    pred = adapter.rerank(premise="Statement verification:", options=["a", "b"],
                          hyp_format="This statement is correct and logical: {}", scoring="entailment")
    assert int(pred) == 0
    graded = adapter.grade(question="q", reference="r", candidate="c")
    assert str(graded) == "entailment"


def test_adapter_covers_sdk_public_parameter_surface():
    """Every named parameter the SDK rerank/grade expose must be accepted by the adapter."""
    import gemma4_cross_encoder as sdk

    def _covers(adapter_params: dict, sdk_params: dict, label: str) -> None:
        has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in adapter_params.values())
        for name in sdk_params:
            if name == "self":
                continue
            assert name in adapter_params or has_var_kw, f"{label}: adapter missing SDK param {name!r}"

    _covers(inspect.signature(OpenJevSDKAdapter.rerank).parameters,
            inspect.signature(sdk.Gemma4CrossEncoder.rerank).parameters, "rerank")
    _covers(inspect.signature(OpenJevSDKAdapter.grade).parameters,
            inspect.signature(sdk.Gemma4CrossEncoder.grade).parameters, "grade")


def test_von_adapter_maps_decide_probs_to_our_label_order():
    """Von returns named probabilities; predict must emit OUR order (con, ent, neu)."""
    from scripts.eval_baselines import VonAdapter

    class FakeVon:
        class _R:
            choice = "entailment"
            confidence = 0.3
            probabilities = {"entailment": 0.6, "contradiction": 0.3, "neutral": 0.1}
        def decide(self, state, choices, instructions):
            assert set(choices) == {"entailment", "contradiction", "neutral"}
            assert "Premise:" in state and "Hypothesis:" in state
            return self._R()

    adapter = VonAdapter(inner=FakeVon())
    probs = adapter.predict([("p1", "h1"), ("p2", "h2")])
    assert probs.shape == (2, 3)
    assert probs[0].tolist() == [0.3, 0.6, 0.1], "order must be contradiction, entailment, neutral"
    result = adapter.rerank(premise="q", options=["a", "b"], hyp_format="{}", scoring="entailment")
    assert isinstance(result, RerankResult) and int(result) == 0
    graded = adapter.grade(question="q", reference="r", candidate="c")
    assert isinstance(graded, GradeResult) and str(graded) == "entailment"


def test_gliner2_adapter_grounded_classification_mapping():
    """Grounded (task G2): classify_text with include_confidence=True -> top label + conf.
    The (n,3) matrix places confidence on the top label, remainder split evenly."""
    from scripts.eval_baselines import GLiNER2Adapter

    class FakeGLiNER2:
        def classify_text(self, text, tasks, include_confidence=False):
            assert include_confidence is True, "grounded call must request confidence"
            assert list(tasks["classification"]) == ["contradiction", "entailment", "neutral"]
            assert "Premise:" in text and "Hypothesis:" in text
            return {"classification": {"label": "entailment", "confidence": 0.8}}

    adapter = GLiNER2Adapter(inner=FakeGLiNER2())
    assert adapter.grounded_api is True and adapter.exposes_full_distribution is False
    probs = adapter.predict([("p1", "h1")])
    assert probs.shape == (1, 3)
    assert np.allclose(probs[0], [0.1, 0.8, 0.1]), "top label 0.8, remainder split evenly"
    result = adapter.rerank(premise="q", options=["a", "b"], hyp_format="{}", scoring="entailment")
    assert isinstance(result, RerankResult) and int(result) == 0


TESTS = [test_rerank_returns_sdk_rerankresult_with_entailment_scoring,
         test_rerank_margin_scoring_matches_sdk_rule,
         test_grade_returns_sdk_graderesult_with_probabilities,
         test_eval_suite_call_shapes_are_accepted,
         test_adapter_covers_sdk_public_parameter_surface,
         test_von_adapter_maps_decide_probs_to_our_label_order,
         test_gliner2_adapter_grounded_classification_mapping]


def main() -> int:
    failures = 0
    for test in TESTS:
        try:
            test()
            print(f"PASS  {test.__name__}")
        except Exception:
            import traceback
            failures += 1
            print(f"FAIL  {test.__name__}")
            traceback.print_exc()
    print(f"\n{len(TESTS) - failures}/{len(TESTS)} tests passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
