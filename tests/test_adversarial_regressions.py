#!/usr/bin/env python3
"""tests/test_adversarial_regressions.py
========================================
Permanent regression suite for the adversarial-review fixes (findings 5, 6, 15, 18).

Replaces the exploratory script tests/probe_adversarial.py (deleted): every probe
that used to PRINT a confirmed defect is now a hard assertion.

P1  decide() on an offline-constructed engine returns a valid DecisionResult (fast + robust).
P2  Gemma4CrossEncoder.rerank exposes the System1Engine ABC aliases query / documents.
P3  Gemma4SetAttentionDecider without a backbone raises ValueError (no silent hash embeddings).
P4  Duplicate option texts raise ValueError on every engine (no probability-dict collapse).
P5  CanonicalDecision.from_dict with 25 options raises ValueError mentioning the 20-key limit.
P6  A perfectly uniform round-robin engine scores PBI == 0.0 exactly (mixed-K-correct baseline).
P7  convert_case_to_nli_pairs(serving_parity=False) keeps the question text in the premise.
P8  GroupedTokenBucketBatchSampler len() equals the number of batches __iter__ yields.

Run: CUDA_VISIBLE_DEVICES= uv run python tests/test_adversarial_regressions.py
"""

from __future__ import annotations

import inspect
import json
import sys
import unittest
from pathlib import Path
from typing import Dict, List, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = Path(__file__).resolve().parent
# WHY tests dir: P1/P4/P6 reuse mocks from sibling test modules, which only resolves
# when the tests directory itself is importable regardless of the launch command.
for _path in (str(REPO_ROOT), str(TESTS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from data_engine import CanonicalDecision, CanonicalDecisionDataset
from decision_engine import DecisionResult
from eval_system1_suite import compute_pbi_and_flip_rate
from gemma4_cross_encoder import Gemma4CrossEncoder
from gemma4_set_attention import Gemma4SetAttentionDecider
from research.adapters.grouped_decision_collator import GroupedTokenBucketBatchSampler
from research.adapters.typed_decisions_adapter import convert_case_to_nli_pairs

from test_aux_engines import (
    CharLevelTokenizer,
    OptionAwareCausalLM,
    build_causal_decider,
    build_set_attention_engine,
)
from test_system1_engines import build_offline_engine
from test_system1_framework import UniformRoundRobinEngine


class TestAdversarialRegressions(unittest.TestCase):
    """Each test locks in one adversarial-review finding as a permanent assertion."""

    def test_p1_decide_on_offline_engine_returns_valid_decision_result(self):
        """P1: decide() must serve a normalized DecisionResult in fast and robust modes."""
        engine = build_offline_engine()
        options = ["Refund the customer.", "Escalate to a human agent.", "Close the ticket."]
        for mode in ("fast", "robust"):
            with self.subTest(mode=mode):
                res = engine.decide(context="ctx text", question="q?", options=options, mode=mode)
                self.assertIsInstance(res, DecisionResult)
                self.assertEqual(list(res.probabilities.keys()), options)
                self.assertTrue(0 <= res.best_index < len(options))
                self.assertEqual(res.best_option, options[res.best_index])
                self.assertAlmostEqual(sum(res.probabilities.values()), 1.0, places=5)

    def test_p2_rerank_signature_has_query_and_documents_aliases(self):
        """P2: the flagship engine must satisfy System1Engine.rerank(query, documents)."""
        params = set(inspect.signature(Gemma4CrossEncoder.rerank).parameters)
        self.assertIn("query", params)
        self.assertIn("documents", params)

        # Polymorphic dispatch through the ABC must reach the subclass implementation
        engine = build_offline_engine()
        result = engine.rerank(query="policy payout", documents=["doc one", "doc two"])
        # gemma4's RerankResult is a Jev-compat int (best index) with .scores per document
        self.assertEqual(len(result.scores), 2)
        self.assertIn(int(result), (0, 1))

    def test_p3_set_attention_without_backbone_raises_valueerror(self):
        """P3: backbone=None must raise instead of silently serving hash embeddings."""
        with self.assertRaises(ValueError) as ctx:
            Gemma4SetAttentionDecider(
                model_path_or_name="offline-fake",
                tokenizer=CharLevelTokenizer(),
                hidden_size=32,
                device="cpu",
            )
        self.assertIn("backbone", str(ctx.exception))
        self.assertIn("embed_fallback_for_tests", str(ctx.exception))

    def test_p4_duplicate_options_raise_valueerror_on_any_engine(self):
        """P4: duplicate option texts must raise, not collapse the probability dict."""
        engines = {
            "cross_encoder": build_offline_engine(),
            "set_attention": build_set_attention_engine(embed_fallback_for_tests=True),
            "causal": build_causal_decider(OptionAwareCausalLM(winner_option="beta")),
        }
        for engine_name, engine in engines.items():
            with self.subTest(engine=engine_name):
                with self.assertRaises(ValueError) as ctx:
                    engine.decide(context="c", question="q", options=["same", "same", "other"])
                self.assertIn("same", str(ctx.exception))

    def test_p5_from_dict_with_25_options_raises_mentioning_limit(self):
        """P5: K=25 must fail loudly on the 20 single-letter key inventory limit."""
        with self.assertRaises(ValueError) as ctx:
            CanonicalDecision.from_dict({
                "id": "big",
                "context": "c",
                "question": "q",
                "options": [f"opt_{i}" for i in range(25)],
                "gold_index": 24,
            })
        self.assertIn("20", str(ctx.exception))

    def test_p6_uniform_round_robin_scores_exact_zero_pbi(self):
        """P6: the mixed-K-correct PBI baseline scores exactly 0.0 for uniform engines.

        The old num_slots = max(4, ...) baseline charged K=3 items an unreachable
        fourth slot, so a perfectly uniform engine scored PBI > 0.
        """
        items = [
            CanonicalDecision.from_dict({
                "id": f"uniform_{i}",
                "context": f"Unique context {i} for the uniform probe.",
                "question": "Which option holds?",
                "options": [f"alpha choice {i}", f"beta choice {i}", f"gamma choice {i}"],
                "gold_index": 0,
            })
            for i in range(9)
        ]
        pbi, _, meta = compute_pbi_and_flip_rate(UniformRoundRobinEngine(), items, max_items=9)
        self.assertEqual(pbi, 0.0, f"Uniform round-robin must score exactly 0.0 PBI, got {pbi}")
        self.assertEqual(meta["num_eligible"], 9)

    def test_p7_serving_parity_false_keeps_question_in_premise(self):
        """P7: without serving parity the premise must still carry the question text.

        The old `q_premise if serving_parity else base_premise` dropped the question
        instructions from the premise in the training-parity layout.
        """
        raw_case = {
            "state_id": "adv_p7_case",
            "domain": "it_triage",
            "state": "Server CPU is at 99% due to unbounded memory leak in worker daemon.",
            "state_is_json": False,
            "questions": json.dumps({
                "severity": {
                    "type": "choice",
                    "instructions": "Identify the incident severity level.",
                    "criteria": {
                        "P0_critical": "System down or unresponsive",
                        "P1_high": "Major degradation but operational",
                    },
                },
            }),
            "gold": json.dumps({"severity": "P0_critical"}),
            "teacher": json.dumps({
                "severity": {"probabilities": {"P0_critical": 0.9, "P1_high": 0.1}},
            }),
        }
        pairs = convert_case_to_nli_pairs(raw_case, serving_parity=False)
        self.assertTrue(pairs, "choice question must convert to at least one pair")
        for pair in pairs:
            self.assertIn("Identify the incident severity level.", pair["premise"])

    def test_p8_sampler_len_equals_yielded_batch_count(self):
        """P8: GroupedTokenBucketBatchSampler len() must match yielded batches.

        The old sampler drew batch composition inside __iter__, so shuffle epochs
        yielded a different batch count than len() reported.
        """
        lengths: List[int] = [512, 512, 512, 512, 512, 300, 300, 300]
        group_ids: Sequence[object] = ["g1", "g1", "g2", "g2", None, None, None, None]
        for shuffle in (False, True):
            sampler = GroupedTokenBucketBatchSampler(
                lengths,
                group_ids=group_ids,
                max_tokens_per_batch=1024,
                shuffle=shuffle,
                seed=7,
            )
            batches = list(sampler)
            self.assertEqual(
                len(sampler),
                len(batches),
                f"len={len(sampler)} != yielded={len(batches)} (shuffle={shuffle})",
            )
            # Composition is fixed at construction; epochs only reorder batches
            sampler.set_epoch(3)
            self.assertEqual(len(sampler), len(list(sampler)))


class TestDatasetSchemaError(unittest.TestCase):
    """Regression for the CanonicalDecisionDataset loader: pairwise NLI files must
    fail with a ValueError naming the file, the 1-based line, and the canonical schema
    (tested on a /tmp file; the data/ symlink is never touched)."""

    def test_pairwise_nli_record_names_file_line_and_schema(self):
        import os
        import tempfile

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", suffix=".jsonl", delete=False, dir="/tmp", encoding="utf-8"
            ) as tmp_file:
                tmp_file.write('{"id": "pair_1", "premise": "A man reads a scroll.", "hypothesis": "He is outdoors.", "label": 0}\n')
                tmp_file.write('{"id": "pair_2", "premise": "Dogs run in a park.", "hypothesis": "Animals are outside.", "label": 2}\n')
                tmp_path = tmp_file.name

            with self.assertRaises(ValueError) as ctx:
                CanonicalDecisionDataset(tmp_path)
            message = str(ctx.exception)
            self.assertIn(tmp_path, message, "error must name the file")
            self.assertIn("line 1", message, "error must name the 1-based line number")
            self.assertIn("options", message, "error must state the expected canonical schema")
            self.assertIn("premise", message, "error must list the offending record's actual keys")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
