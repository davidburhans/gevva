#!/usr/bin/env python3
"""test_system1_framework.py
=============================
Comprehensive unit and regression test suite for the Unified System 1 Decision Framework.

Tests run strictly on CPU (no network / GPU required):
1. Canonical Data Representation & Serialization
2. Cyclic Permutation Option Rotation & Label Remapping
3. In-Context Causal Collator Token-Masking and Prompt Formatting
4. Mathematical Proof of Permutation Equivariance for Set-Attention Head
5. System1Engine Protocol Compliance across Cross-Encoder, Causal, and Set-Attention models
6. Position Bias Index (PBI) and Decision Flip Rate Arithmetic
7. Evaluation Suite End-to-End Smoke Test and Pareto Table Generation

Run: CUDA_VISIBLE_DEVICES= uv run python tests/test_system1_framework.py
"""

from __future__ import annotations

import sys
import unittest
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_engine import (
    CanonicalDecision,
    InContextPermutationCollator,
    STANDARD_OPTION_KEYS,
)
from decision_engine import (
    DecisionResult,
    JudgeResult,
    RateResult,
    RerankResult,
    System1Engine,
)
from eval_system1_suite import (
    compute_pbi_and_flip_rate,
    format_pareto_markdown_table,
    run_full_suite,
    MockInvariantEngine,
)
from gemma4_cross_encoder import Gemma4CrossEncoder
from gemma4_set_attention import (
    Gemma4SetAttentionDecider,
    PermutationEquivariantSetHead,
)


class MockTokenizer:
    """Lightweight deterministic mock tokenizer for offline testing."""
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = 2
        self.padding_side = "right"
        self._vocab = {"<pad>": 0, "</s>": 1, "<s>": 2}
        # Populate letter tokens
        for k in STANDARD_OPTION_KEYS:
            self._vocab[f" {k}"] = len(self._vocab) + 10
            self._vocab[k] = len(self._vocab) + 10

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        tokens = [self.bos_token_id] if add_special_tokens else []
        words = text.replace("\n", " ").split()
        for w in words:
            if w in self._vocab:
                tokens.append(self._vocab[w])
            elif f" {w}" in self._vocab:
                tokens.append(self._vocab[f" {w}"])
            else:
                tokens.append(abs(hash(w)) % 1000 + 50)
        return tokens

    def __call__(self, text: str, return_tensors: str = None) -> Dict[str, Any]:
        ids = self.encode(text)
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor([ids], dtype=torch.long),
                "attention_mask": torch.ones((1, len(ids)), dtype=torch.long),
            }
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


class MockBiasedEngine(System1Engine):
    """Mock engine that always prefers the first option (severe position bias)."""
    @property
    def model_name(self) -> str:
        return "mock-biased-engine"

    @property
    def paradigm(self) -> str:
        return "mock"

    def decide(self, context: str, question: str, options: Sequence[str], mode: str = "fast", **kwargs: Any) -> DecisionResult:
        k = len(options)
        # 80% confidence on slot 0
        probs = {opt: (0.8 if i == 0 else 0.2 / max(1, k - 1)) for i, opt in enumerate(options)}
        return DecisionResult(
            best_option=options[0],
            best_index=0,
            probabilities=probs,
            scores=[1.0 if i == 0 else 0.0 for i in range(k)],
        )

    def judge(self, context: str, claim: str, **kwargs: Any) -> JudgeResult:
        return JudgeResult(verdict="entailment", probabilities={"entailment": 1.0, "contradiction": 0.0, "neutral": 0.0})

    def rate(self, context: str, rubric_levels: Sequence[str], level_values: Sequence[float] = None, **kwargs: Any) -> RateResult:
        return RateResult(expected_score=0.0, level_probabilities={l: 1.0/len(rubric_levels) for l in rubric_levels}, predicted_level=rubric_levels[0])

    def rerank(self, query: str, documents: Sequence[str], **kwargs: Any) -> RerankResult:
        return RerankResult(ranked_indices=list(range(len(documents))), items=[])


class UniformRoundRobinEngine(System1Engine):
    """Deterministic engine whose base selections cycle uniformly across slots.

    WHY a first-sight registry keyed by (context, question) instead of a global call
    counter: the PBI harness evaluates each item once per cyclic shift, so with fixed
    K every item's base call lands on the same global-counter residue and a naive
    round-robin looks maximally biased. First-sight assignment per option-count
    makes base selections round-robin across items, while shifted re-evaluations of
    the same item stay on its assigned slot.
    """

    def __init__(self) -> None:
        self._slot_by_item: Dict[Tuple[str, str], int] = {}
        self._first_sight_count_by_k: Dict[int, int] = defaultdict(int)

    @property
    def model_name(self) -> str:
        return "uniform-round-robin"

    @property
    def paradigm(self) -> str:
        return "mock"

    def decide(self, context: str, question: str, options: Sequence[str], mode: str = "fast", **kwargs: Any) -> DecisionResult:
        k = len(options)
        item_key = (context, question)
        if item_key not in self._slot_by_item:
            self._slot_by_item[item_key] = self._first_sight_count_by_k[k] % k
            self._first_sight_count_by_k[k] += 1
        best_idx = self._slot_by_item[item_key]
        probs = {opt: 1.0 / k for opt in options}
        return DecisionResult(
            best_option=options[best_idx],
            best_index=best_idx,
            probabilities=probs,
            scores=[1.0 / k] * k,
        )

    def judge(self, context: str, claim: str, **kwargs: Any) -> JudgeResult:
        uniform_third = 1.0 / 3.0
        return JudgeResult(verdict="neutral", probabilities={"entailment": uniform_third, "contradiction": uniform_third, "neutral": uniform_third})

    def rate(self, context: str, rubric_levels: Sequence[str], level_values: Sequence[float] = None, **kwargs: Any) -> RateResult:
        return RateResult(expected_score=0.0, level_probabilities={l: 1.0/len(rubric_levels) for l in rubric_levels}, predicted_level=rubric_levels[0])

    def rerank(self, query: str, documents: Sequence[str], **kwargs: Any) -> RerankResult:
        return RerankResult(ranked_indices=list(range(len(documents))), items=[])


# -----------------------------------------------------------------------------
# Test Suite
# -----------------------------------------------------------------------------
class TestSystem1Framework(unittest.TestCase):

    def setUp(self):
        self.sample_decision = CanonicalDecision(
            id="test_decision_001",
            domain="insurance_policy",
            context="Policy section 4.2 states that vacancy exclusions apply after 60 days of non-occupancy.",
            question="Which clause governs the settlement of claim #889?",
            options=[
                {"key": "A", "text": "Deny via vacancy exclusion."},
                {"key": "B", "text": "Pay under occupied dwelling endorsement."},
                {"key": "C", "text": "Pay replacement cost subject to sublimit."},
            ],
            gold_index=1,
            gold_key="B",
            soft_distribution=[0.15, 0.80, 0.05],
            is_trap=True,
        )

    def _distinct_decision(self, scenario_idx: int, num_options: int) -> CanonicalDecision:
        """Builds a deterministic decision with unique context per (index, K) pair."""
        options = [
            {"key": STANDARD_OPTION_KEYS[j], "text": f"Choice {scenario_idx}.{j} for the scenario."}
            for j in range(num_options)
        ]
        return CanonicalDecision(
            id=f"uniform_probe_k{num_options}_{scenario_idx}",
            domain="bias_probe",
            context=f"Unique context {scenario_idx} for the {num_options}-option scenario.",
            question=f"Which option holds for scenario {scenario_idx}?",
            options=options,
            gold_index=scenario_idx % num_options,
        )

    def test_canonical_serialization(self):
        """Verifies round-trip serialization of CanonicalDecision."""
        d = self.sample_decision.to_dict()
        reconstructed = CanonicalDecision.from_dict(d)
        self.assertEqual(reconstructed.id, self.sample_decision.id)
        self.assertEqual(reconstructed.num_options, 3)
        self.assertEqual(reconstructed.gold_index, 1)
        self.assertEqual(reconstructed.gold_key, "B")
        self.assertTrue(reconstructed.is_trap)

    def test_cyclic_permutation_rotation(self):
        """Verifies cyclic option rotation, key re-indexing, and gold tracking."""
        perms = self.sample_decision.to_cyclic_permutations()
        self.assertEqual(len(perms), 3)

        # Shift 0: original order
        self.assertEqual(perms[0].options[0]["text"], "Deny via vacancy exclusion.")
        self.assertEqual(perms[0].gold_index, 1)
        self.assertEqual(perms[0].gold_key, "B")

        # Shift 1: rotated by 1
        self.assertEqual(perms[1].options[0]["text"], "Pay under occupied dwelling endorsement.")
        self.assertEqual(perms[1].gold_index, 0)
        self.assertEqual(perms[1].gold_key, "A")
        # Verify soft distribution rotated with options
        self.assertAlmostEqual(perms[1].soft_distribution[0], 0.80)

        # Shift 2: rotated by 2
        self.assertEqual(perms[2].options[0]["text"], "Pay replacement cost subject to sublimit.")
        self.assertEqual(perms[2].gold_index, 2)
        self.assertEqual(perms[2].gold_key, "C")

    def test_in_context_collator_token_masking(self):
        """Verifies that InContextPermutationCollator sets prompt tokens to -100 and targets decision slot."""
        tok = MockTokenizer()
        collator = InContextPermutationCollator(tokenizer=tok, max_length=512, permute_training=False)
        batch = collator([self.sample_decision])

        input_ids = batch["input_ids"]
        labels = batch["labels"]

        self.assertEqual(input_ids.shape, labels.shape)
        # All tokens except the final token should be masked with -100
        non_masked = (labels[0] != -100).nonzero(as_tuple=False).squeeze(-1)
        self.assertEqual(len(non_masked), 1, "Exactly one token must have a valid loss target")

        # The unmasked token ID must match the gold token ID
        gold_char = self.sample_decision.gold_key
        expected_token_id = collator.key_to_token_id[gold_char]
        self.assertEqual(labels[0, non_masked[0]].item(), expected_token_id)

    def test_set_attention_permutation_equivariance_math_proof(self):
        """Mathematical proof: Permuting candidate inputs permutes output scores identically."""
        torch.manual_seed(42)
        hidden_size = 128
        head = PermutationEquivariantSetHead(hidden_size=hidden_size, num_heads=4, num_layers=2)
        head.eval()

        for k in [2, 3, 5, 8]:
            cand_embeds = torch.randn(1, k, hidden_size)

            # Standard order
            with torch.no_grad():
                scores = head(cand_embeds)  # (1, K)

            # Test multiple random permutations
            for _ in range(3):
                perm = torch.randperm(k).tolist()
                cand_perm = cand_embeds[:, perm, :]

                with torch.no_grad():
                    scores_perm = head(cand_perm)

                # Expected: scores[:, perm] == scores_perm
                max_diff = torch.max(torch.abs(scores[:, perm] - scores_perm)).item()
                self.assertLess(
                    max_diff,
                    1e-5,
                    f"SetHead failed permutation equivariance at K={k}: delta={max_diff}",
                )

    def test_system1_engine_interface_contract(self):
        """Verifies that Gemma4CrossEncoder and Gemma4SetAttentionDecider implement System1Engine."""
        self.assertTrue(issubclass(Gemma4CrossEncoder, System1Engine))
        self.assertTrue(issubclass(Gemma4SetAttentionDecider, System1Engine))

        # Instantiate mock Gemma4SetAttentionDecider (embed_fallback_for_tests=True is
        # mandatory offline: the constructor now rejects backbone=None otherwise)
        engine = Gemma4SetAttentionDecider(
            model_path_or_name="dummy",
            tokenizer=MockTokenizer(),
            hidden_size=64,
            embed_fallback_for_tests=True,
        )
        self.assertEqual(engine.paradigm, "set_attention")

        res = engine.decide(
            context="The policy pays on proof of loss.",
            question="Does the policy pay?",
            options=["Yes, it pays.", "No, it is excluded."],
        )
        self.assertIsInstance(res, DecisionResult)
        self.assertEqual(len(res.probabilities), 2)
        self.assertIn(res.best_option, ["Yes, it pays.", "No, it is excluded."])

        # Test judge
        j_res = engine.judge(context="The policy pays on proof of loss.", claim="Losses are covered.")
        self.assertIsInstance(j_res, JudgeResult)
        self.assertIn(j_res.verdict, ["entailment", "contradiction", "neutral"])

    def test_pbi_and_flip_rate_arithmetic(self):
        """Verifies that PBI and flip rate detect bias and confirm invariance."""
        items = [self.sample_decision] * 10

        # Biased engine always picks slot 0 -> high PBI and non-zero flip rate
        pbi_biased, flip_biased, _ = compute_pbi_and_flip_rate(MockBiasedEngine(), items, max_items=10)
        self.assertGreater(pbi_biased, 0.20, f"Biased engine should have high PBI, got {pbi_biased}")
        self.assertGreater(flip_biased, 0.50, f"Biased engine should flip on cyclic shifts, got {flip_biased}")

        # Invariant engine picks content -> zero flip rate
        pbi_inv, flip_inv, _ = compute_pbi_and_flip_rate(MockInvariantEngine(), items, max_items=10)
        self.assertEqual(flip_inv, 0.0, "Content-invariant engine must have 0.0% flip rate")

    def test_uniform_engine_exact_zero_pbi_fixed_k3(self):
        """Regression (finding 6): a perfectly uniform engine must score PBI == 0.0 exactly.

        The old metric compared against a max(4, ...) slot baseline, so K=3 items paid
        an unreachable 4th slot and PBI never reached 0.
        """
        items = [self._distinct_decision(i, 3) for i in range(30)]
        pbi, flip_rate, meta = compute_pbi_and_flip_rate(UniformRoundRobinEngine(), items, max_items=30)
        self.assertEqual(pbi, 0.0, f"Uniform engine on fixed K=3 must score exactly 0.0 PBI, got {pbi}")
        self.assertEqual(meta["num_eligible"], 30)
        self.assertEqual(meta["num_tested"], 30)

    def test_uniform_engine_exact_zero_pbi_mixed_k(self):
        """Regression (finding 6): the PBI baseline must match the evaluated K mixture.

        With 12 items per K in {2, 3, 4}, a uniform round-robin picks each slot s of a
        K-option item exactly 12/K times, which equals the mixed-K expectation
        sum(12/k for eligible k > s) — so the normalized gap must be exactly zero.
        """
        items = [self._distinct_decision(i, k) for k in (2, 3, 4) for i in range(12)]
        pbi, _, meta = compute_pbi_and_flip_rate(UniformRoundRobinEngine(), items, max_items=len(items))
        self.assertEqual(pbi, 0.0, f"Uniform engine on mixed K in {{2,3,4}} must score exactly 0.0 PBI, got {pbi}")
        self.assertEqual(meta["num_eligible"], 36)
        # Slots outside Kmax-1 are never expected or observed
        self.assertEqual(sorted(meta["slot_distribution"]), [0, 1, 2, 3])

    def test_flip_rate_ignores_single_option_items(self):
        """Regression (finding 6): k<=1 items are excluded from the flip-rate denominator.

        Two K=3 items that always flip plus one K=1 item that cannot flip must yield
        flip_rate == 1.0 (2 flips / 2 eligible), not 2/3 over the raw sample size.
        """
        single_option = CanonicalDecision(
            id="flip_probe_k1",
            domain="bias_probe",
            context="Only one candidate exists.",
            question="Which option holds?",
            options=[{"key": "A", "text": "The only option."}],
            gold_index=0,
        )
        items = [single_option, self.sample_decision, self.sample_decision]
        _, flip_rate, meta = compute_pbi_and_flip_rate(MockBiasedEngine(), items, max_items=3)
        self.assertEqual(meta["num_tested"], 3)
        self.assertEqual(meta["num_eligible"], 2)
        self.assertEqual(flip_rate, 1.0, f"Flip rate must use only eligible (k>1) items, got {flip_rate}")

    def test_eval_suite_smoke_run(self):
        """Runs full evaluation suite on mock invariant engine."""
        items = [self.sample_decision, self.sample_decision]
        engine = MockInvariantEngine()

        res = run_full_suite(engine, items, limit=2, modes=["fast", "robust"])
        self.assertIn("fast", res["modes"])
        self.assertIn("robust", res["modes"])

        table_md = format_pareto_markdown_table([res])
        self.assertIn("| Model Architecture |", table_md)
        self.assertIn("mock-invariant-engine", table_md)


if __name__ == "__main__":
    unittest.main(verbosity=2)
