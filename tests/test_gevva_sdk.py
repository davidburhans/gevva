"""tests/test_gevva_sdk.py
========================
Verification of the Gevva top-level SDK, aliases, and runtime contracts.
"""

import unittest


class TestGevvaSDK(unittest.TestCase):
    def test_imports_and_version(self):
        import gevva
        self.assertEqual(gevva.__version__, "1.0.0")
        self.assertTrue(hasattr(gevva, "GevvaCrossEncoder"))
        self.assertTrue(hasattr(gevva, "Gevva"))
        self.assertTrue(hasattr(gevva, "GevvaForSequenceClassification"))
        self.assertTrue(hasattr(gevva, "OpenJevCrossEncoder"))
        self.assertTrue(hasattr(gevva, "Gemma4CrossEncoder"))
        self.assertTrue(hasattr(gevva, "System1Engine"))
        self.assertTrue(hasattr(gevva, "load"))

    def test_class_aliasing_identities(self):
        import gevva
        self.assertIs(gevva.Gevva, gevva.GevvaCrossEncoder)
        self.assertIs(gevva.OpenJevCrossEncoder, gevva.GevvaCrossEncoder)
        self.assertIs(gevva.Gemma4CrossEncoder, gevva.GevvaCrossEncoder)

    def test_constants_and_label_mappings(self):
        import gevva
        self.assertEqual(gevva.CONTRADICTION, 0)
        self.assertEqual(gevva.ENTAILMENT, 1)
        self.assertEqual(gevva.NEUTRAL, 2)
        self.assertEqual(gevva.ID2LABEL[0], "contradiction")
        self.assertEqual(gevva.ID2LABEL[1], "entailment")
        self.assertEqual(gevva.ID2LABEL[2], "neutral")
        self.assertEqual(gevva.LABEL2ID["contradiction"], 0)
        self.assertEqual(gevva.LABEL2ID["entailment"], 1)
        self.assertEqual(gevva.LABEL2ID["neutral"], 2)

    def test_data_models_instantiation(self):
        import gevva
        item = gevva.RerankItem(index=0, document="doc", score=0.9, probability=0.9)
        self.assertEqual(item.index, 0)
        self.assertEqual(item.score, 0.9)

        dec = gevva.DecisionResult(
            best_option="A",
            best_index=0,
            probabilities={"A": 0.8, "B": 0.2},
            scores=[1.5, -0.5],
        )
        self.assertEqual(dec.best_option, "A")
        self.assertAlmostEqual(dec.top_confidence, 0.8)

        judge = gevva.JudgeResult(
            verdict="entailment",
            probabilities={"entailment": 0.95, "contradiction": 0.03, "neutral": 0.02},
        )
        self.assertTrue(judge.is_entailed)

    def test_default_model_and_cli(self):
        import inspect
        import gevva
        from gevva import cli

        self.assertEqual(gevva.DEFAULT_MODEL_ID, "davidburhans/gevva-e2b")

        # Verify gevva.load default argument
        sig = inspect.signature(gevva.load)
        self.assertEqual(sig.parameters["model_name_or_path"].default, "davidburhans/gevva-e2b")

        # Verify CLI parser defaults to HuggingFace repo
        parser = cli.build_parser()
        pred_parser = parser._subparsers._actions[1].choices["predict"]
        rerank_parser = parser._subparsers._actions[1].choices["rerank"]
        grade_parser = parser._subparsers._actions[1].choices["grade"]

        self.assertEqual(pred_parser.get_default("model"), "davidburhans/gevva-e2b")
        self.assertEqual(rerank_parser.get_default("model"), "davidburhans/gevva-e2b")
        self.assertEqual(grade_parser.get_default("model"), "davidburhans/gevva-e2b")


if __name__ == "__main__":
    unittest.main()
