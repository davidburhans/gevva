#!/usr/bin/env python3
"""tests/test_system1_engines.py - Offline CPU tests for the Gemma4CrossEncoder System1Engine integration.

Covers the adversarial-review fixes:
1. rerank() accepts System1Engine ABC aliases (query / documents) with LSP precedence.
2. decide() uses the rerank softmax-normalized scores (probabilities sum to ~1.0).
3. decide() mode handling: fast / robust / ValueError on unknown modes.
4. Duplicate option / rubric-level texts raise instead of collapsing text-keyed dicts.
5. rate() level_values length validation; judge() empty-claim rejection.
6. option_keys serving-parity hypothesis formatting ('The correct answer is: {key}: {text}').

Run: CUDA_VISIBLE_DEVICES= uv run python tests/test_system1_engines.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from decision_engine import DecisionResult, JudgeResult, RateResult
from gemma4_cross_encoder import Gemma4CrossEncoder, RerankResult


class FakeCrossEncoderTokenizer:
    """Deterministic char-level tokenizer (pattern: DummyTokenizer in tests/test_grouped_collator.py)."""

    pad_token: str = "<pad>"
    pad_token_id: int = 0
    eos_token_id: int = 1
    bos_token_id: int = 2
    padding_side: str = "right"

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        return [ord(char) % 251 + 2 for char in text]


class ContentBasedNliModel(nn.Module):
    """Fake 3-way NLI head whose logits depend only on token content.

    WHY: content-based logits make option scores permutation-invariant, so the
    robust (cyclic-ensemble) mode must return exactly the fast-mode decision.
    Logits order is (contradiction, entailment, neutral) per ID2LABEL.
    """

    def __init__(self) -> None:
        super().__init__()
        # The engine reads boi/eoi token ids off model.config via getattr defaults.
        self.config = SimpleNamespace()
        self.forward_calls: int = 0

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> SimpleNamespace:
        self.forward_calls += 1
        content: torch.Tensor = input_ids.sum(dim=1).to(torch.float32)
        entailment_logit = torch.fmod(content, 9.0) - 4.0
        zeros = torch.zeros_like(entailment_logit)
        logits = torch.stack([zeros, entailment_logit, zeros], dim=1)  # (B, 3)
        return SimpleNamespace(logits=logits)


def build_offline_engine() -> Gemma4CrossEncoder:
    """Builds a fully offline engine via the model= / tokenizer= injection seam of __init__."""
    return Gemma4CrossEncoder(
        model=ContentBasedNliModel(),
        tokenizer=FakeCrossEncoderTokenizer(),
        device="cpu",
        max_length=512,
        batch_size=8,
    )


def argmax_index(values: Sequence[float]) -> int:
    """First-argmax over a sequence, matching np.argmax tie semantics."""
    best = 0
    for i in range(1, len(values)):
        if values[i] > values[best]:
            best = i
    return best


class TestDecide(unittest.TestCase):

    def setUp(self) -> None:
        self.engine = build_offline_engine()
        self.options = ["Refund the customer.", "Escalate to a human agent.", "Close the ticket."]

    def test_decide_fast_returns_normalized_decision(self) -> None:
        res = self.engine.decide(
            context="The user demands a refund for a broken item.",
            question="What is the next best action?",
            options=self.options,
        )
        self.assertIsInstance(res, DecisionResult)
        self.assertEqual(len(res.scores), len(self.options))
        # rerank is called with temperature=1.0 -> .scores are softmax-normalized probabilities
        self.assertAlmostEqual(sum(res.probabilities.values()), 1.0, places=5)
        self.assertAlmostEqual(sum(res.scores), 1.0, places=5)
        self.assertEqual(list(res.probabilities.keys()), self.options)
        best_idx = argmax_index(res.scores)
        self.assertEqual(res.best_index, best_idx)
        self.assertEqual(res.best_option, self.options[best_idx])
        self.assertEqual(res.probabilities[res.best_option], max(res.probabilities.values()))

    def test_decide_robust_matches_fast_on_content_based_fake(self) -> None:
        fast = self.engine.decide(context="ctx text", question="q?", options=self.options, mode="fast")
        robust = self.engine.decide(context="ctx text", question="q?", options=self.options, mode="robust")
        self.assertEqual(robust.best_index, fast.best_index)
        self.assertEqual(robust.best_option, fast.best_option)
        for fast_score, robust_score in zip(fast.scores, robust.scores):
            self.assertAlmostEqual(fast_score, robust_score, places=6)

    def test_decide_rejects_unknown_mode(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            self.engine.decide(context="c", question="q", options=["a", "b"], mode="bogus")
        self.assertIn("bogus", str(ctx.exception))
        self.assertIn("fast", str(ctx.exception))
        self.assertIn("robust", str(ctx.exception))

    def test_decide_rejects_duplicate_options(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            self.engine.decide(context="c", question="q", options=["Approve", "Approve", "Reject"])
        self.assertIn("Approve", str(ctx.exception))

    def test_decide_rejects_mismatched_option_keys(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            self.engine.decide(context="c", question="q", options=["a", "b"], option_keys=["A"])
        self.assertIn("2", str(ctx.exception))


class TestOptionKeysServingParity(unittest.TestCase):

    def test_option_keys_format_hypotheses_with_original_texts_in_result(self) -> None:
        engine = build_offline_engine()
        captured_pairs: List[Tuple[str, str]] = []
        original_prepare = engine._prepare_batch

        def spy_prepare(pairs: Sequence[Tuple[str, str]], images: Any = None) -> Any:
            captured_pairs.extend(pairs)
            return original_prepare(pairs, images)

        engine._prepare_batch = spy_prepare  # spy on the exact premise/hypothesis pairs scored

        options = ["Deny the claim.", "Pay in full.", "Request more documents."]
        res = engine.decide(
            context="Policy section 4.2 governs settlements.",
            question="Which action applies?",
            options=options,
            option_keys=["A", "B", "C"],
        )

        expected_premise = "Policy section 4.2 governs settlements.\nQuestion: Which action applies?"
        # Must match SERVING_CHOICE_TEMPLATE = "The correct answer is: {key}: {desc}"
        expected_hypotheses = [
            "The correct answer is: A: Deny the claim.",
            "The correct answer is: B: Pay in full.",
            "The correct answer is: C: Request more documents.",
        ]
        self.assertEqual(len(captured_pairs), 3)
        self.assertEqual([premise for premise, _ in captured_pairs], [expected_premise] * 3)
        self.assertEqual([hyp for _, hyp in captured_pairs], expected_hypotheses)
        # DecisionResult reports the ORIGINAL option texts, never the keyed rerank strings
        self.assertEqual(list(res.probabilities.keys()), options)
        self.assertIn(res.best_option, options)


class TestJudge(unittest.TestCase):

    def setUp(self) -> None:
        self.engine = build_offline_engine()

    def test_judge_returns_normalized_three_class_distribution(self) -> None:
        res = self.engine.judge(
            context="The policy covers water damage in the basement.",
            claim="Water damage in the basement is covered.",
        )
        self.assertIsInstance(res, JudgeResult)
        self.assertIn(res.verdict, ("contradiction", "entailment", "neutral"))
        self.assertEqual(len(res.probabilities), 3)
        self.assertAlmostEqual(sum(res.probabilities.values()), 1.0, places=5)

    def test_judge_rejects_empty_claim(self) -> None:
        with self.assertRaises(ValueError):
            self.engine.judge(context="some context", claim="")


class TestRate(unittest.TestCase):

    def setUp(self) -> None:
        self.engine = build_offline_engine()
        self.levels = ["poor", "adequate", "excellent"]
        self.values = [0.0, 1.0, 2.0]

    def test_rate_expected_score_equals_value_probability_sum(self) -> None:
        res = self.engine.rate(
            context="The essay cites four primary sources with a full bibliography.",
            rubric_levels=self.levels,
            level_values=self.values,
        )
        self.assertIsInstance(res, RateResult)
        expected = sum(
            value * res.level_probabilities[level] for level, value in zip(self.levels, self.values)
        )
        self.assertAlmostEqual(res.expected_score, expected, places=6)
        best_level = max(self.levels, key=lambda level: res.level_probabilities[level])
        self.assertEqual(res.predicted_level, best_level)

    def test_rate_rejects_mismatched_level_values_length(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            self.engine.rate(context="c", rubric_levels=self.levels, level_values=[1.0, 2.0])
        message = str(ctx.exception)
        self.assertIn("2", message)
        self.assertIn("3", message)

    def test_rate_rejects_duplicate_rubric_levels(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            self.engine.rate(context="c", rubric_levels=["good", "good", "bad"], level_values=[0.0, 1.0, 2.0])
        self.assertIn("good", str(ctx.exception))


class TestRerankAbcAliases(unittest.TestCase):
    """LSP seam: decision_engine.System1Engine.rerank(query, documents) on the flagship engine."""

    def setUp(self) -> None:
        self.engine = build_offline_engine()
        self.documents = ["The sky is blue.", "Cats are liquid.", "Rust compiles fast."]
        self.query = "What is the sky like?"

    def test_query_documents_aliases_match_premise_options(self) -> None:
        via_abc = self.engine.rerank(query=self.query, documents=self.documents)
        via_jev = self.engine.rerank(premise=self.query, options=self.documents)
        self.assertIsInstance(via_abc, RerankResult)
        self.assertEqual(int(via_abc), int(via_jev))
        self.assertEqual(via_abc.scores, via_jev.scores)

    def test_premise_wins_over_query_alias(self) -> None:
        with_premise = self.engine.rerank(premise="Premise A", options=["alpha", "beta", "gamma"])
        with_both = self.engine.rerank(premise="Premise A", query="Different text", options=["alpha", "beta", "gamma"])
        self.assertEqual(int(with_both), int(with_premise))
        self.assertEqual(with_both.scores, with_premise.scores)

    def test_nonempty_options_win_over_documents_alias(self) -> None:
        via_options = self.engine.rerank(premise="q", options=["a", "b"])
        via_both = self.engine.rerank(premise="q", options=["a", "b"], documents=["ignored"])
        self.assertEqual(int(via_both), int(via_options))
        self.assertEqual(via_both.scores, via_options.scores)

    def test_missing_premise_and_query_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            self.engine.rerank(options=["only-option"])
        self.assertIn("premise", str(ctx.exception))

    def test_query_alias_works_with_robust_debias(self) -> None:
        # Regression: the cyclic-permutation recursion must carry the aliased query through.
        via_abc = self.engine.rerank(query=self.query, documents=self.documents, debias_position=True)
        via_jev = self.engine.rerank(premise=self.query, options=self.documents, debias_position=True)
        self.assertEqual(int(via_abc), int(via_jev))
        for abc_score, jev_score in zip(via_abc.scores, via_jev.scores):
            self.assertAlmostEqual(abc_score, jev_score, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
