"""tests/test_adaptive_forward.py
==============================
Verifies that GevvaCrossEncoder._forward_adaptive recursively bisects chunks
and clears cache on OutOfMemoryError.
"""

import unittest
from unittest.mock import MagicMock, patch
import torch
import numpy as np
from gemma4_cross_encoder import GevvaCrossEncoder


class TestAdaptiveForward(unittest.TestCase):
    def test_adaptive_bisection_on_oom(self):
        """Simulate an OOM on batch size 4, verifying it bisects to batch size 2 and succeeds."""
        # Create a mock cross encoder instance
        encoder = object.__new__(GevvaCrossEncoder)
        encoder.device = "cpu"
        encoder.batch_size = 4
        encoder.calibrated_temperature = 1.0

        # Mock model that fails on batch_size >= 4 with OutOfMemoryError, but succeeds on batch_size < 4
        mock_model = MagicMock()
        def side_effect(**batch):
            bs = batch["input_ids"].shape[0]
            if bs >= 4:
                raise torch.cuda.OutOfMemoryError("CUDA out of memory simulation")
            # Return dummy logits (bs, 3)
            res = MagicMock()
            res.logits = torch.ones((bs, 3), dtype=torch.float32) * float(bs)
            return res

        mock_model.side_effect = side_effect
        encoder.model = mock_model

        # Mock _prepare_batch to return a dummy dict with input_ids shaped (len(pairs), 10)
        def mock_prepare(pairs, images=None):
            return {"input_ids": torch.zeros((len(pairs), 10), dtype=torch.long)}

        encoder._prepare_batch = mock_prepare

        pairs = [("premise", f"hyp_{i}") for i in range(4)]
        with patch("torch.cuda.empty_cache") as mock_empty_cache:
            logits = encoder._forward_adaptive(pairs)
            self.assertEqual(logits.shape, (4, 3))
            # Verify empty_cache was called when OOM occurred
            self.assertTrue(mock_empty_cache.called)

    def test_single_item_oom_raises(self):
        """Verify that if a single item still OOMs, the error is re-raised."""
        encoder = object.__new__(GevvaCrossEncoder)
        encoder.device = "cpu"
        encoder.batch_size = 1

        mock_model = MagicMock()
        mock_model.side_effect = torch.cuda.OutOfMemoryError("CUDA out of memory simulation")
        encoder.model = mock_model

        def mock_prepare(pairs, images=None):
            return {"input_ids": torch.zeros((len(pairs), 10), dtype=torch.long)}

        encoder._prepare_batch = mock_prepare

        pairs = [("premise", "hyp")]
        with self.assertRaises(torch.cuda.OutOfMemoryError):
            encoder._forward_adaptive(pairs)


if __name__ == "__main__":
    unittest.main()
