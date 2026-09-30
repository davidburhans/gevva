#!/usr/bin/env python3
"""tests/test_decision_index_adapter.py
=======================================
Unit tests for the Gevva Decision Index Engine adapter.
"""

import math
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from research.adapters.gevva_decision_index_engine import (
    GevvaDecisionIndexEngine,
    _softmax,
    _text,
)


def validate_decision_index_response(questions, response):
    """Reference validator from decision_index/engines/base.py."""
    if set(response.get("answers", {})) != set(questions):
        raise ValueError("Question keys mismatch")
    for k, q in questions.items():
        a = response["answers"][k]
        if a.get("type") != q["type"]:
            raise ValueError(f"Type mismatch on {k}")
        if q["type"] == "choice":
            if a.get("choice") not in q["criteria"]:
                raise ValueError(f"Invalid chosen answer for {k}")
            p = a.get("probabilities", {})
            if set(p) != set(q["criteria"]) or any(not math.isfinite(v) or not 0 <= v <= 1 for v in p.values()) or abs(sum(p.values()) - 1) > 0.01:
                raise ValueError(f"Incomplete/invalid probability distribution for {k}")
        elif q["type"] == "noul":
            if not math.isfinite(a["noul"]) or not 0 <= a["noul"] <= 1:
                raise ValueError(f"Invalid noul probability for {k}")
        else:
            raise ValueError("Unsupported question type " + str(q["type"]))


class TestDecisionIndexAdapter(unittest.TestCase):
    def test_text_helper(self):
        self.assertEqual(_text(None), "")
        self.assertEqual(_text("  hello  "), "hello")
        self.assertEqual(_text({"key": "val"}), '{"key":"val"}')

    def test_softmax_helper(self):
        probs = _softmax([1.0, 2.0, 3.0])
        self.assertEqual(len(probs), 3)
        self.assertAlmostEqual(sum(probs), 1.0, places=5)
        self.assertTrue(probs[2] > probs[1] > probs[0])

    @patch("research.adapters.gevva_decision_index_engine.Gemma4CrossEncoder")
    def test_engine_call_choice_and_noul(self, mock_encoder_cls):
        mock_encoder = MagicMock()
        mock_encoder_cls.return_value = mock_encoder

        # Setup mock predictions
        # predict returns [p_con, p_ent, p_neu]
        mock_encoder.predict.side_effect = [
            # For 3 choice options:
            np.array([
                [0.1, 0.2, 0.7],  # A
                [0.05, 0.85, 0.1], # B
                [0.8, 0.1, 0.1],  # C
            ]),
            # For 1 noul question:
            np.array([
                [0.1, 0.9, 0.0],
            ]),
        ]

        engine = GevvaDecisionIndexEngine(model_name_or_path="mock-model", device="cpu")

        questions = {
            "q1": {
                "type": "choice",
                "instructions": "Which city is the capital of France?",
                "criteria": {
                    "A": "London",
                    "B": "Paris",
                    "C": "Berlin",
                },
            },
            "q2": {
                "type": "noul",
                "instructions": "Paris is located in Europe.",
                "criteria": {
                    "false": "False",
                    "true": "True",
                },
            },
        }

        response = engine("Geography context.", questions)

        # Validate against official Decision Index rules
        validate_decision_index_response(questions, response)

        # Check answers
        q1_ans = response["answers"]["q1"]
        self.assertEqual(q1_ans["type"], "choice")
        self.assertEqual(q1_ans["choice"], "B")
        self.assertTrue(q1_ans["probabilities"]["B"] > q1_ans["probabilities"]["A"])

        q2_ans = response["answers"]["q2"]
        self.assertEqual(q2_ans["type"], "noul")
        self.assertAlmostEqual(q2_ans["noul"], 0.9, places=2)


if __name__ == "__main__":
    unittest.main()


class TestFormatState(unittest.TestCase):
    """ENG-01 fix: dict states render as readable blocks, never raw JSON."""

    def test_dict_state_renders_labeled_blocks(self):
        from research.adapters.gevva_decision_index_engine import _format_state
        out = _format_state({"prompt": "Summarize this.", "response": "A summary."})
        self.assertEqual(out, "Prompt: Summarize this.\n\nResponse: A summary.")

    def test_nested_value_falls_back_to_json(self):
        from research.adapters.gevva_decision_index_engine import _format_state
        out = _format_state({"events": [{"a": 1}]})
        self.assertIn("Events:", out)
        self.assertIn('{"a":1}', out)

    def test_string_and_empty_pass_through(self):
        from research.adapters.gevva_decision_index_engine import _format_state
        self.assertEqual(_format_state("plain"), "plain")
        self.assertEqual(_format_state({}), "")
        self.assertEqual(_format_state(None), "")


class TestNoulScoreMapping(unittest.TestCase):
    """Binary score = p_ent + 0.5*p_neu (neutral split, not discarded)."""

    def test_neutral_mass_is_split(self):
        from research.adapters.gevva_decision_index_engine import GevvaDecisionIndexEngine
        # p_con=0.17, p_ent=0.28, p_neu=0.55 -> p_true = 0.28 + 0.275 = 0.555
        # (old renormalization would give 0.28/0.45 = 0.622 - inverted verdict)
        class _Stub:
            def predict(self, pairs):
                return [(0.17, 0.28, 0.55)]
        engine = GevvaDecisionIndexEngine.__new__(GevvaDecisionIndexEngine)
        engine.cross_encoder = _Stub()
        engine.temperature = 1.0
        engine.scoring = "margin"
        answers = engine(
            {"prompt": "p", "response": "r"},
            {"q": {"type": "noul", "instructions": "The response contains content that is not supported."}},
        )
        self.assertAlmostEqual(answers["answers"]["q"]["noul"], 0.555, places=6)
