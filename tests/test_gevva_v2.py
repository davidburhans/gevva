import json
import tempfile
import unittest
from pathlib import Path

from train_gevva_v2 import format_systemone_sample, get_option_code
from scripts.export_gevva_v2_gguf import inject_decision_metadata_to_config


class TestGevvaV2Pipeline(unittest.TestCase):
    def test_get_option_code_coverage(self):
        # Index 0..25 should be A..Z
        self.assertEqual(get_option_code(0), "A")
        self.assertEqual(get_option_code(25), "Z")
        # Index 26..254 should be two-letter codes
        self.assertEqual(get_option_code(26), "AA")
        self.assertEqual(get_option_code(27), "AB")
        self.assertEqual(get_option_code(51), "AZ")
        self.assertEqual(get_option_code(52), "BA")
        self.assertEqual(get_option_code(254), "IU")

    def test_format_systemone_sample_canonical(self):
        sample = {
            "state": "Document text",
            "instructions": "Is this an invoice?",
            "options": [
                {"label": "A", "key": "yes", "description": "It is an invoice"},
                {"label": "B", "key": "no", "description": "It is not an invoice"},
            ],
            "gold_code": "A",
            "soft_probability": 0.95,
        }
        prompt, completion, active_codes, gold_code, soft_probs = format_systemone_sample(sample, permute=False)
        self.assertTrue(prompt.startswith("<start_of_turn>user\n"))
        self.assertTrue(prompt.endswith("<start_of_turn>model\n"))
        self.assertEqual(completion, "A<end_of_turn>")
        self.assertEqual(active_codes, ["A", "B"])
        self.assertEqual(gold_code, "A")
        self.assertAlmostEqual(soft_probs[0], 0.95)
        self.assertAlmostEqual(soft_probs[1], 0.05)

    def test_format_systemone_sample_permutation(self):
        sample = {
            "state": "Ticket text",
            "instructions": "Route ticket",
            "options": [
                {"label": "A", "key": "billing"},
                {"label": "B", "key": "support"},
                {"label": "C", "key": "sales"},
            ],
            "gold_code": "B",
            "soft_probabilities": [0.1, 0.8, 0.1],
        }
        # Run 20 times to verify permutation logic preserves semantics
        for _ in range(20):
            prompt, completion, active_codes, gold_code, soft_probs = format_systemone_sample(sample, permute=True)
            self.assertEqual(active_codes, ["A", "B", "C"])
            self.assertEqual(completion, f"{gold_code}<end_of_turn>")
            gold_idx = active_codes.index(gold_code)
            # The soft probability corresponding to gold option 'support' (0.8) must track with gold_idx
            self.assertAlmostEqual(soft_probs[gold_idx], 0.8)

    def test_export_metadata_injection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            cfg_file = model_dir / "config.json"
            cfg_file.write_text(json.dumps({"architectures": ["Gemma4ForConditionalGeneration"]}))
            inject_decision_metadata_to_config(model_dir)

            with open(cfg_file) as f:
                saved_cfg = json.load(f)
            self.assertEqual(saved_cfg["architectures"], ["Gemma4ForCausalLM"])
            self.assertEqual(saved_cfg["decision"]["type"], "lev")
            self.assertEqual(saved_cfg["decision"]["max_options"], 255)
            self.assertIn("choice.large", saved_cfg["decision"]["temperatures"])


if __name__ == "__main__":
    unittest.main()
