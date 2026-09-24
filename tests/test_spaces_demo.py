"""Tests for spaces/gevva-demo/app.py logic and Gradio UI contracts."""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from PIL import Image

# Ensure repository root and spaces/gevva-demo are in path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SPACES_DIR = os.path.join(REPO_ROOT, "spaces", "gevva-demo")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if SPACES_DIR not in sys.path:
    sys.path.insert(0, SPACES_DIR)

import app
from gemma4_cross_encoder import GradeResult, RerankResult


class TestSpacesDemoUnit(unittest.TestCase):
    """Unit tests for app.py functions with mocked engine outputs to verify return type handling."""

    def setUp(self):
        self.mock_engine = MagicMock()
        app._MODEL_CACHE["test-model"] = self.mock_engine

    def tearDown(self):
        if "test-model" in app._MODEL_CACHE:
            del app._MODEL_CACHE["test-model"]

    def test_predict_pair_text(self):
        # engine.predict returns np.ndarray of shape (1, 3): [con, ent, neu]
        self.mock_engine.predict.return_value = np.array([[0.05, 0.90, 0.05]], dtype=np.float32)

        verdict, labels, latency = app.predict_pair(
            model_id="test-model",
            premise="The sky is blue today.",
            hypothesis="The sky is blue.",
            image=None,
        )

        self.assertIn("ENTAILMENT", verdict)
        self.assertIn("90.0%", verdict)
        self.assertIn("Entailment (True / Verified)", labels)
        self.assertAlmostEqual(labels["Entailment (True / Verified)"], 0.90, places=2)
        self.assertAlmostEqual(labels["Contradiction (False / Refuted)"], 0.05, places=2)
        self.assertAlmostEqual(labels["Neutral (Unverifiable / Irrelevant)"], 0.05, places=2)
        self.assertIn("Latency", latency)

    def test_predict_pair_image(self):
        self.mock_engine.predict.return_value = np.array([[0.85, 0.10, 0.05]], dtype=np.float32)
        img = Image.new("RGB", (64, 64), color="red")

        verdict, labels, latency = app.predict_pair(
            model_id="test-model",
            premise="A red square is shown.",
            hypothesis="The image is completely green.",
            image=img,
        )

        self.assertIn("CONTRADICTION", verdict)
        self.assertIn("85.0%", verdict)
        self.assertAlmostEqual(labels["Contradiction (False / Refuted)"], 0.85, places=2)
        self.mock_engine.predict.assert_called_once()

    def test_predict_pair_empty_hypothesis(self):
        verdict, labels, latency = app.predict_pair(
            model_id="test-model",
            premise="Some premise",
            hypothesis="",
        )
        self.assertIn("Please enter a hypothesis", verdict)
        self.assertEqual(labels, {})

    def test_route_intent(self):
        # Rerank returns RerankResult with scores
        scores = [0.85, 0.10, 0.05]
        docs = ["refund", "tracking", "cancel"]
        rerank_res = RerankResult(0, scores, documents=docs)
        self.mock_engine.rerank.return_value = rerank_res

        out, latency = app.route_intent(
            model_id="test-model",
            query="Please refund my money",
            tools_input="refund\ntracking\ncancel",
        )

        self.assertIn("refund", out)
        self.assertIn("85.0%", out)
        self.assertIn("| #1 | `refund` 🏆 | **85.0%** |", out)
        self.assertIn("Latency", latency)

    def test_route_intent_empty_tools(self):
        out, latency = app.route_intent(
            model_id="test-model",
            query="test",
            tools_input="",
        )
        self.assertIn("Please provide at least one tool option", out)

    def test_grade_candidate_correct(self):
        grade_res = GradeResult("entailment", {"entailment": 0.94, "contradiction": 0.04, "neutral": 0.02})
        self.mock_engine.grade.return_value = grade_res

        summary, latency = app.grade_candidate(
            model_id="test-model",
            question="What is 2+2?",
            reference="4",
            candidate="The answer is 4.",
        )

        self.assertIn("CORRECT (Entails Reference)", summary)
        self.assertIn("94.0%", summary)
        self.assertIn("**Semantic Alignment (Entailment)**: 94.0%", summary)
        self.assertIn("Latency", latency)

    def test_grade_candidate_incorrect(self):
        grade_res = GradeResult("contradiction", {"entailment": 0.05, "contradiction": 0.90, "neutral": 0.05})
        self.mock_engine.grade.return_value = grade_res

        summary, latency = app.grade_candidate(
            model_id="test-model",
            question="What is 2+2?",
            reference="4",
            candidate="The answer is 5.",
        )

        self.assertIn("INCORRECT (CONTRADICTION)", summary)
        self.assertIn("90.0%", summary)
        self.assertIn("**Contradiction Probability**: 90.0%", summary)


class TestSpacesDemoLive(unittest.TestCase):
    """Live integration tests using local weights if available and test Gradio app startup."""

    def test_demo_blocks_definition(self):
        """Verifies that demo Blocks are configured properly with title and tabs."""
        self.assertIsNotNone(app.demo)
        self.assertIn("Gevva", app.title)

    def test_real_inference_with_local_model(self):
        """Tests live model inference if local checkpoint exists."""
        local_model_path = os.path.join(REPO_ROOT, "ckpt", "gevva-e2b")
        if not os.path.isdir(local_model_path):
            self.skipTest(f"Local checkpoint {local_model_path} not found; skipping live test.")

        engine = app.load_engine(local_model_path)
        self.assertIsNotNone(engine)

        verdict, labels, latency = app.predict_pair(
            model_id=local_model_path,
            premise="The capital of France is Paris.",
            hypothesis="Paris is the capital of France.",
        )
        self.assertIn("ENTAILMENT", verdict)
        self.assertGreater(labels["Entailment (True / Verified)"], 0.5)

        # Test rerank / route intent
        out_route, lat_route = app.route_intent(
            model_id=local_model_path,
            query="Where can I find the user manual?",
            tools_input="search_knowledge_base: Search articles and documentation\nprocess_refund: Refund payment",
        )
        self.assertIn("search_knowledge_base", out_route)

        # Test grade
        out_grade, lat_grade = app.grade_candidate(
            model_id=local_model_path,
            question="What is the capital of France?",
            reference="Paris",
            candidate="Paris",
        )
    def test_real_multimodal_inference(self):
        """Tests live multimodal inference with an image and local multimodal checkpoint."""
        local_model_path = os.path.join(REPO_ROOT, "ckpt", "gevva-e2b-phase4", "best")
        if not os.path.isdir(local_model_path):
            self.skipTest(f"Local multimodal checkpoint {local_model_path} not found.")

        img_path = os.path.join(SPACES_DIR, "examples", "sample_chart.jpg")
        if not os.path.isfile(img_path):
            self.skipTest(f"Example image {img_path} not found.")

        img = Image.open(img_path)
        verdict, labels, latency = app.predict_pair(
            model_id=local_model_path,
            premise="A financial bar chart is shown.",
            hypothesis="The chart displays quarterly data.",
            image=img,
        )
        self.assertTrue(any(k in verdict for k in ["ENTAILMENT", "CONTRADICTION", "NEUTRAL"]))
        self.assertIn("Entailment (True / Verified)", labels)
        self.assertIn("Contradiction (False / Refuted)", labels)
        self.assertIn("Neutral (Unverifiable / Irrelevant)", labels)

    def test_gradio_server_http_launch(self):
        """Tests that Gradio demo launches cleanly on a socket and responds to HTTP requests."""
        import urllib.request
        import time

        port = 7895
        try:
            app.demo.launch(server_port=port, prevent_thread_lock=True)
            time.sleep(1.0)
            url = f"http://127.0.0.1:{port}/"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=5) as response:
                status = response.getcode()
                content = response.read().decode("utf-8")
                self.assertEqual(status, 200)
                self.assertIn("Gevva", content)
        finally:
            try:
                app.demo.close()
            except Exception:
                pass


if __name__ == "__main__":
    unittest.main()

