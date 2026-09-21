#!/usr/bin/env python3
"""tests/test_typed_decisions_adapter.py - Unit tests for Typed Decisions dataset adapter.

Run: uv run python tests/test_typed_decisions_adapter.py
"""

import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nli_labels import CONTRADICTION, ENTAILMENT
from research.adapters.typed_decisions_adapter import (
    convert_case_to_nli_pairs,
    _format_state_premise,
    DATASET_NAME,
    DATASET_LICENSE,
)


class TestTypedDecisionsAdapter(unittest.TestCase):

    def test_format_state_premise(self):
        prem = _format_state_premise("e-commerce", '{"order_id": 123}', True)
        self.assertIn("Domain: e-commerce", prem)
        self.assertIn('"order_id": 123', prem)

    def test_convert_case_noul(self):
        raw_case = {
            "state_id": "test_case_01",
            "domain": "customer_support",
            "state": "Customer requested a full refund for an opened software license.",
            "state_is_json": False,
            "questions": json.dumps({
                "is_software": {
                    "type": "noul",
                    "instructions": "Determine whether the product in question is digital software.",
                    "criteria": {"true": "Software item", "false": "Physical hardware"},
                },
                "is_hardware": {
                    "type": "noul",
                    "instructions": "Determine whether the product is a physical hardware device.",
                    "criteria": {"true": "Physical item", "false": "Software item"},
                },
            }),
            "gold": json.dumps({
                "is_software": True,
                "is_hardware": False,
            }),
            "teacher": json.dumps({
                "is_software": {"noul": 0.98},
                "is_hardware": {"noul": 0.02},
            }),
        }

        pairs = convert_case_to_nli_pairs(raw_case, include_inverted_noul=False)
        self.assertEqual(len(pairs), 2)

        # Check is_software (True -> Entailment)
        p_soft = next(p for p in pairs if p["metadata"]["question_id"] == "is_software")
        self.assertEqual(p_soft["label"], ENTAILMENT)
        self.assertEqual(p_soft["soft_labels"][1], 0.98)
        self.assertEqual(p_soft["metadata"]["attribution"]["dataset"], DATASET_NAME)
        self.assertEqual(p_soft["metadata"]["attribution"]["license"], DATASET_LICENSE)

        # Check is_hardware (False -> Contradiction)
        p_hard = next(p for p in pairs if p["metadata"]["question_id"] == "is_hardware")
        self.assertEqual(p_hard["label"], CONTRADICTION)
        self.assertEqual(p_hard["soft_labels"][1], 0.02)

    def test_convert_case_choice(self):
        raw_case = {
            "state_id": "test_case_02",
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
                        "P2_low": "Minor cosmetic defect",
                    },
                },
            }),
            "gold": json.dumps({
                "severity": "P0_critical",
            }),
            "teacher": json.dumps({
                "severity": {"probabilities": {"P0_critical": 0.94, "P1_high": 0.05, "P2_low": 0.01}},
            }),
        }

        pairs = convert_case_to_nli_pairs(raw_case, max_negatives_per_choice=2)
        self.assertEqual(len(pairs), 3)

        gold_pair = next(p for p in pairs if p["metadata"]["is_gold"] is True)
        self.assertEqual(gold_pair["label"], ENTAILMENT)
        self.assertIn("P0_critical", gold_pair["hypothesis"])
        self.assertEqual(gold_pair["soft_labels"][1], 0.94)

        neg_pairs = [p for p in pairs if p["metadata"]["is_gold"] is False]
        self.assertEqual(len(neg_pairs), 2)
        for neg in neg_pairs:
            self.assertEqual(neg["label"], CONTRADICTION)

    def test_convert_case_score(self):
        raw_case = {
            "state_id": "test_case_03",
            "domain": "essay_grading",
            "state": "The essay cites 4 primary academic sources with full bibliography.",
            "state_is_json": False,
            "questions": json.dumps({
                "citation_quality": {
                    "type": "score",
                    "instructions": "Rate citation quality from 0 to 2.",
                    "criteria": [
                        "0 citations",
                        "1-2 citations",
                        "3 or more valid academic citations",
                    ],
                },
            }),
            "gold": json.dumps({
                "citation_quality": 2,
            }),
            "teacher": json.dumps({
                "citation_quality": {"probabilities": {"0": 0.0, "1": 0.05, "2": 0.95}},
            }),
        }

        pairs = convert_case_to_nli_pairs(raw_case)
        self.assertGreaterEqual(len(pairs), 2)

        gold_pair = next(p for p in pairs if p["metadata"]["is_gold"] is True)
        self.assertEqual(gold_pair["label"], ENTAILMENT)
        self.assertEqual(gold_pair["metadata"]["level"], 2)
        self.assertIn("3 or more valid academic citations", gold_pair["hypothesis"])

        alt_pair = next(p for p in pairs if p["metadata"]["is_gold"] is False)
        self.assertEqual(alt_pair["label"], CONTRADICTION)
        self.assertIn(alt_pair["metadata"]["level"], (0, 1))

    def test_convert_case_grouped(self):
        raw_case = {
            "state_id": "case_grp_01",
            "domain": "customer_support",
            "state": "User requested cancellation within 14 days cooling-off period.",
            "state_is_json": False,
            "questions": json.dumps({
                "action": {
                    "type": "choice",
                    "instructions": "Select the correct support action.",
                    "criteria": {
                        "full_refund": "Process complete refund under statutory cooling-off",
                        "partial_refund": "Deduct admin fee",
                        "deny_request": "Refuse refund",
                        "escalate": "Escalate to manager",
                    },
                },
            }),
            "gold": json.dumps({"action": "full_refund"}),
            "teacher": json.dumps({"action": {"probabilities": {"full_refund": 0.92, "partial_refund": 0.05, "deny_request": 0.02, "escalate": 0.01}}}),
        }

        pairs = convert_case_to_nli_pairs(raw_case, grouped=True, serving_parity=True)
        self.assertEqual(len(pairs), 4)

        # All 4 options share the same group_id
        gids = {p["group_id"] for p in pairs}
        self.assertEqual(len(gids), 1)
        self.assertEqual(list(gids)[0], "td_case_grp_01_action")

        # Exactly 1 gold
        golds = [p for p in pairs if p["is_gold"]]
        self.assertEqual(len(golds), 1)
        self.assertEqual(golds[0]["label"], ENTAILMENT)
        self.assertEqual(golds[0]["soft_target"], 0.92)

        # Premise has serving parity
        self.assertTrue(pairs[0]["premise"].startswith("Select the correct support action."))

    def test_pointwise_premise_keeps_question_without_serving_parity(self):
        """Regression: pointwise branches used to fall back to base_premise when
        serving_parity=False, silently dropping the question instructions the
        hypothesis still referenced."""
        instructions_by_qid = {
            "q_bool": "BOOLQ Determine packet loss on edge-7.",
            "q_choice": "CHOICEQ Choose a mitigation action.",
            "q_score": "SCOREQ Rate the incident severity.",
        }
        raw_case = {
            "state_id": "case_noparity_01",
            "domain": "net_triage",
            "state": "Gateway latency at 800ms with packet loss on edge-7.",
            "state_is_json": False,
            "questions": json.dumps({
                "q_bool": {
                    "type": "noul",
                    "instructions": instructions_by_qid["q_bool"],
                    "criteria": {"true": "Lossy link", "false": "Stable link"},
                },
                "q_choice": {
                    "type": "choice",
                    "instructions": instructions_by_qid["q_choice"],
                    "criteria": {"rollback": "Roll back the release", "restart": "Restart the gateway"},
                },
                "q_score": {
                    "type": "score",
                    "instructions": instructions_by_qid["q_score"],
                    "criteria": ["sev3 minor", "sev1 major"],
                },
            }),
            "gold": json.dumps({"q_bool": True, "q_choice": "rollback", "q_score": 0}),
            "teacher": json.dumps({
                "q_bool": {"noul": 0.9},
                "q_choice": {"probabilities": {"rollback": 0.9, "restart": 0.1}},
                "q_score": {"probabilities": {"0": 0.9, "1": 0.1}},
            }),
        }

        for serving_parity in (False, True):
            pairs = convert_case_to_nli_pairs(
                raw_case,
                serving_parity=serving_parity,
                include_inverted_noul=False,
            )
            # 1 noul + (1 gold + 1 neg) choice + (1 gold + 1 alt) score
            self.assertEqual(len(pairs), 5, f"serving_parity={serving_parity}")
            for pair in pairs:
                expected_instruction = instructions_by_qid[pair["metadata"]["question_id"]]
                self.assertIn(
                    expected_instruction,
                    pair["premise"],
                    f"premise lost the question (serving_parity={serving_parity}): {pair['id']}",
                )
                if not serving_parity:
                    # Non-parity premise embeds the question with an explicit marker
                    self.assertIn("Question:", pair["premise"])

    def test_inverted_noul_pair_premise_keeps_question_without_serving_parity(self):
        raw_case = {
            "state_id": "case_noparity_inv",
            "domain": "policy",
            "state": "Refunds are allowed within 30 days of purchase.",
            "state_is_json": False,
            "questions": json.dumps({
                "q_bool": {
                    "type": "noul",
                    "instructions": "INVPOLQ Determine whether the refund window is open.",
                    "criteria": {"true": "Window open", "false": "Window closed"},
                },
            }),
            "gold": json.dumps({"q_bool": True}),
            "teacher": json.dumps({"q_bool": {"noul": 0.95}}),
        }

        pairs = convert_case_to_nli_pairs(raw_case, serving_parity=False, include_inverted_noul=True)
        self.assertEqual(len(pairs), 2)
        for pair in pairs:
            self.assertIn("INVPOLQ Determine whether the refund window is open.", pair["premise"])


if __name__ == "__main__":
    unittest.main()
