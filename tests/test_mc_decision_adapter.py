#!/usr/bin/env python3
"""tests/test_mc_decision_adapter.py - Unit tests for MC Decision Adapter.

Run: uv run python tests/test_mc_decision_adapter.py
"""

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nli_labels import CONTRADICTION, ENTAILMENT, NEUTRAL
from research.adapters.mc_decision_adapter import (
    convert_aqua_rat_record,
    convert_casehold_record,
    convert_csqa_record,
    convert_logiqa_record,
    convert_lsat_ar_record,
    convert_pubmedqa_record,
    convert_race_record,
    convert_reclor_record,
    convert_strategy_qa_record,
    extract_ngrams,
    normalize_text,
    SERVING_CHOICE_TEMPLATE,
)


class TestMCDecisionAdapter(unittest.TestCase):

    def test_convert_casehold_record(self):
        sample = {
            "context": "Courts have found possession of a bomb to be a crime of violence (<HOLDING>); United States v. Dodge",
            "endings": [
                "holding that possession of a pipe bomb is a crime of violence",
                "holding that bank robbery is a crime of violence",
                "holding that sexual assault is a crime of violence",
                "holding that a firearm is not a violent felony",
                "holding that a court must only look to statutory definition",
            ],
            "label": 0,
        }
        pairs = convert_casehold_record(sample, index=42, split="train")
        self.assertEqual(len(pairs), 5)

        # Check group consistency
        group_ids = {p["group_id"] for p in pairs}
        self.assertEqual(len(group_ids), 1)
        self.assertEqual(list(group_ids)[0], "casehold_train_42")

        # Check labels & targets
        gold_pairs = [p for p in pairs if p["is_gold"]]
        neg_pairs = [p for p in pairs if not p["is_gold"]]
        self.assertEqual(len(gold_pairs), 1)
        self.assertEqual(len(neg_pairs), 4)

        gold = gold_pairs[0]
        self.assertEqual(gold["label"], ENTAILMENT)
        self.assertEqual(gold["soft_target"], 1.0)
        self.assertEqual(gold["soft_labels"], [0.0, 1.0, 0.0])
        self.assertEqual(
            gold["hypothesis"],
            "The correct answer is: holding that possession of a pipe bomb is a crime of violence",
        )
        self.assertIn("Courts have found possession of a bomb", gold["premise"])

        for neg in neg_pairs:
            self.assertEqual(neg["label"], CONTRADICTION)
            self.assertEqual(neg["soft_target"], 0.0)
            self.assertEqual(neg["soft_labels"], [1.0, 0.0, 0.0])

        # Test invalid record returns empty
        invalid = {"context": "", "endings": [], "label": 10}
        self.assertEqual(convert_casehold_record(invalid, index=0), [])

    def test_convert_race_record(self):
        sample = {
            "example_id": "high19088.txt",
            "article": "Given that I teach students who are training to be doctors, I was surprised...",
            "question": "We can know from the passage that the author works as a_.",
            "options": ["doctor", "model", "teacher", "reporter"],
            "answer": "C",
        }
        pairs = convert_race_record(sample, index=1, split="train")
        self.assertEqual(len(pairs), 4)

        # 'C' is index 2
        gold = [p for p in pairs if p["is_gold"]][0]
        self.assertEqual(gold["metadata"]["option_idx"], 2)
        self.assertEqual(gold["hypothesis"], "The correct answer is: teacher")
        self.assertEqual(gold["label"], ENTAILMENT)
        self.assertIn("Question: We can know from the passage", gold["premise"])
        self.assertEqual(gold["group_id"], "race_train_high19088.txt")

        # Test numeric answer fallback
        sample_num = dict(sample, answer=0)
        pairs_num = convert_race_record(sample_num, index=2, split="train")
        self.assertEqual(pairs_num[0]["label"], ENTAILMENT)
        self.assertEqual(pairs_num[0]["hypothesis"], "The correct answer is: doctor")

    def test_convert_reclor_record(self):
        sample = {
            "id_string": "train_101",
            "context": "All birds fly. Tweety is a bird.",
            "question": "Which of the following follows?",
            "answers": ["Tweety flies", "Tweety swims", "Tweety runs", "Tweety talks"],
            "label": 0,
        }
        pairs = convert_reclor_record(sample, index=0, split="train")
        self.assertEqual(len(pairs), 4)
        self.assertEqual(pairs[0]["label"], ENTAILMENT)
        self.assertEqual(pairs[0]["group_id"], "reclor_train_train_101")
        self.assertEqual(pairs[1]["label"], CONTRADICTION)

    def test_convert_logiqa_record(self):
        sample = {
            "id": 999,
            "text": "If condition A holds, B holds. Condition A holds.",
            "question": "What holds?",
            "options": ["B", "Not B", "C", "D"],
            "answer": 0,
        }
        pairs = convert_logiqa_record(sample, index=0, split="train")
        self.assertEqual(len(pairs), 4)
        self.assertEqual(pairs[0]["label"], ENTAILMENT)
        self.assertEqual(pairs[0]["hypothesis"], "The correct answer is: B")
        self.assertEqual(pairs[0]["group_id"], "logiqa_train_999")

    def test_convert_pubmedqa_grouped(self):
        sample = {
            "pubid": 12345,
            "question": "Does treatment X cure disease Y?",
            "context": {"contexts": ["Clinical trial showed efficacy."]},
            "final_decision": "yes",
            "long_answer": "Yes it cures it.",
        }
        pairs = convert_pubmedqa_record(sample, index=0, split="train", native_3class=False)
        self.assertEqual(len(pairs), 3)  # yes, no, maybe
        self.assertEqual(pairs[0]["hypothesis"], "The correct answer is: yes")
        self.assertEqual(pairs[0]["label"], ENTAILMENT)
        self.assertEqual(pairs[1]["hypothesis"], "The correct answer is: no")
        self.assertEqual(pairs[1]["label"], CONTRADICTION)
        self.assertEqual(pairs[2]["hypothesis"], "The correct answer is: maybe")
        self.assertEqual(pairs[2]["label"], CONTRADICTION)

    def test_convert_pubmedqa_native_3class(self):
        # Test 'maybe' -> NEUTRAL
        sample_maybe = {
            "pubid": 54321,
            "question": "Is drug Z effective in elderly cohorts?",
            "context": {"contexts": ["Data was inconclusive due to small cohort size."]},
            "final_decision": "maybe",
            "long_answer": "Inconclusive results.",
        }
        pairs = convert_pubmedqa_record(sample_maybe, index=0, split="train", native_3class=True)
        self.assertEqual(len(pairs), 1)
        p = pairs[0]
        self.assertEqual(p["label"], NEUTRAL)
        self.assertEqual(p["soft_labels"], [0.0, 0.0, 1.0])
        self.assertEqual(p["group_id"], -1)
        self.assertFalse(p["is_gold"])

    def test_convert_csqa_record(self):
        sample = {
            "id": "csqa_001",
            "question": "Where do people keep books?",
            "question_concept": "book",
            "choices": {
                "label": ["A", "B", "C", "D", "E"],
                "text": ["library", "oven", "river", "cloud", "tree"],
            },
            "answerKey": "A",
        }
        pairs = convert_csqa_record(sample, index=0, split="train")
        self.assertEqual(len(pairs), 5)
        self.assertEqual(pairs[0]["label"], ENTAILMENT)
        self.assertEqual(pairs[0]["hypothesis"], "The correct answer is: library")
        self.assertEqual(pairs[1]["label"], CONTRADICTION)

    def test_convert_lsat_ar_record(self):
        sample = {
            "id_string": "game1_q1",
            "context": "Six speakers give presentations: P, Q, R, S, T, U. P speaks before Q.",
            "question": "Which of the following could be the speaking order?",
            "answers": ["P, Q, R, S, T, U", "Q, P, R, S, T, U", "R, Q, P, S, T, U", "S, T, U, Q, P, R", "T, U, S, Q, P, R"],
            "label": 0,
        }
        pairs = convert_lsat_ar_record(sample, index=0, split="train")
        self.assertEqual(len(pairs), 5)
        self.assertEqual(pairs[0]["label"], ENTAILMENT)
        self.assertEqual(pairs[0]["hypothesis"], "The correct answer is: P, Q, R, S, T, U")
        self.assertEqual(pairs[0]["group_id"], "lsat_ar_train_game1_q1")
        self.assertEqual(pairs[1]["label"], CONTRADICTION)

    def test_convert_strategy_qa_record(self):
        sample = {
            "qid": "strat_001",
            "question": "Can an adult human fit inside a typical commercial washing machine drum?",
            "facts": ["A typical commercial washing machine drum has a volume of 3 to 4 cubic feet.", "An adult human body has an average volume of 2.3 cubic feet."],
            "answer": True,
        }
        pairs = convert_strategy_qa_record(sample, index=0, split="train")
        self.assertEqual(len(pairs), 2)  # Yes and No
        self.assertEqual(pairs[0]["hypothesis"], "The correct answer is: Yes")
        self.assertEqual(pairs[0]["label"], ENTAILMENT)
        self.assertEqual(pairs[1]["hypothesis"], "The correct answer is: No")
        self.assertEqual(pairs[1]["label"], CONTRADICTION)
        self.assertEqual(pairs[0]["group_id"], "strategy_qa_train_strat_001")

    def test_convert_aqua_rat_record(self):
        sample = {
            "question": "If a train travels 60 miles in 1 hour and 30 minutes, what is its average speed in miles per hour?",
            "options": ["A)30", "B)35", "C)40", "D)45", "E)50"],
            "correct": "C",
        }
        pairs = convert_aqua_rat_record(sample, index=5, split="train")
        self.assertEqual(len(pairs), 5)
        # 'C' is index 2, answer is 40
        self.assertEqual(pairs[2]["label"], ENTAILMENT)
        self.assertEqual(pairs[2]["hypothesis"], "The correct answer is: 40")
        self.assertEqual(pairs[0]["label"], CONTRADICTION)
        self.assertEqual(pairs[0]["hypothesis"], "The correct answer is: 30")
        self.assertEqual(pairs[2]["group_id"], "aqua_rat_train_5")

    def test_decontamination_ngram_extraction(self):
        text = "This is a simple test sentence for checking ngrams extraction functionality."
        grams = extract_ngrams(text, n=4)
        self.assertIn("this is a simple", grams)
        self.assertIn("checking ngrams extraction functionality.", grams)


if __name__ == "__main__":
    unittest.main()
