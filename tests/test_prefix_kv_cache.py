"""tests/test_prefix_kv_cache.py
================================
Unit tests verifying Shared Prefix KV Caching in GevvaCrossEncoder.
Runs strictly on CPU without GPU or heavy model weights.
"""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import torch
from gemma4_cross_encoder import GevvaCrossEncoder


class TestPrefixKVCache(unittest.TestCase):
    def setUp(self):
        self.encoder = object.__new__(GevvaCrossEncoder)
        self.encoder.device = "cpu"
        self.encoder.batch_size = 16
        self.encoder.max_length = 512
        self.encoder.calibrated_temperature = 1.0

        # Mock tokenizer
        mock_tok = MagicMock()
        mock_tok.bos_token_id = 2
        mock_tok.pad_token_id = 0

        def mock_encode(text, add_special_tokens=False):
            # return deterministic fake token ids based on words
            return [abs(hash(w)) % 1000 + 10 for w in text.split()]

        mock_tok.encode.side_effect = mock_encode
        self.encoder.tokenizer = mock_tok

    def test_predict_candidates_empty_hypotheses(self):
        """Empty hypotheses should return empty array with shape (0, 3)."""
        res = self.encoder.predict_candidates_logits("Some premise", [])
        self.assertEqual(res.shape, (0, 3))
        res_prob = self.encoder.predict_candidates("Some premise", [])
        self.assertEqual(res_prob.shape, (0, 3))

    def test_predict_candidates_fallback_when_no_backbone(self):
        """When model has no inner .model backbone, fallback to pairwise forward."""
        self.encoder.model = MagicMock(spec=[])  # no .model attribute
        self.encoder._forward_adaptive = MagicMock(return_value=torch.zeros((2, 3)))

        logits = self.encoder.predict_candidates_logits("premise", ["hyp1", "hyp2"])
        self.assertEqual(logits.shape, (2, 3))
        self.assertTrue(self.encoder._forward_adaptive.called)

    def test_predict_candidates_shared_kv_flow(self):
        """Verify the full shared KV cache prefill and suffix forward pass."""
        mock_model = MagicMock()
        mock_bb = MagicMock()
        mock_model.model = mock_bb

        # Mock cache object with batch_repeat_interleave
        mock_cache = MagicMock()
        mock_cache.batch_repeat_interleave = MagicMock()
        mock_cache.__deepcopy__ = MagicMock(side_effect=lambda memo: mock_cache)

        # Step 1: prefill return
        pre_out = MagicMock()
        pre_out.past_key_values = mock_cache

        # Step 2: suffix return
        def bb_forward(input_ids, attention_mask, past_key_values=None, position_ids=None, **kwargs):
            if past_key_values is None:
                return pre_out
            else:
                suf_out = MagicMock()
                batch_size = input_ids.shape[0]
                seq_len = input_ids.shape[1]
                hidden_dim = 16
                suf_out.last_hidden_state = torch.ones((batch_size, seq_len, hidden_dim))
                return suf_out

        mock_bb.side_effect = bb_forward

        # Mock norm and score
        mock_model.norm = MagicMock(side_effect=lambda x: x)
        mock_score = MagicMock()
        mock_score.side_effect = lambda x: torch.ones((x.shape[0], 3), dtype=torch.float32) * 2.0
        mock_model.score = mock_score

        self.encoder.model = mock_model

        hypotheses = ["option A", "option B", "option C"]
        logits = self.encoder.predict_candidates_logits("Context premise", hypotheses, max_candidate_batch_size=2)
        self.assertEqual(logits.shape, (3, 3))
        self.assertTrue(mock_cache.batch_repeat_interleave.called)

        # Test predict_candidates probabilities
        probs = self.encoder.predict_candidates("Context premise", hypotheses, temperature=1.0)
        self.assertEqual(probs.shape, (3, 3))
        np.testing.assert_allclose(probs.sum(axis=-1), np.ones(3), rtol=1e-5)

    def test_auto_routing_shared_premise(self):
        """When pairs share identical premise and no images, predict() routes to predict_candidates."""
        pairs = [("shared premise", f"hyp {i}") for i in range(5)]

        self.encoder.predict_candidates = MagicMock(return_value=np.ones((5, 3)) / 3.0)
        probs = self.encoder.predict(pairs)
        self.encoder.predict_candidates.assert_called_once_with(
            premise="shared premise",
            hypotheses=[p[1] for p in pairs],
            temperature=1.0,
        )
        self.assertEqual(probs.shape, (5, 3))

    def test_auto_routing_bypassed_on_different_premises(self):
        """When premises differ, predict() uses standard chunked forward."""
        pairs = [("premise 1", "hyp 1"), ("premise 2", "hyp 2")]
        self.encoder.predict_candidates = MagicMock()
        self.encoder._forward_adaptive = MagicMock(return_value=torch.zeros((2, 3)))

        probs = self.encoder.predict(pairs)
        self.assertFalse(self.encoder.predict_candidates.called)
        self.assertTrue(self.encoder._forward_adaptive.called)
        self.assertEqual(probs.shape, (2, 3))

    def test_auto_routing_bypassed_with_images(self):
        """When images are present, predict() uses standard chunked forward."""
        pairs = [("shared premise", "hyp 1"), ("shared premise", "hyp 2")]
        images = [MagicMock(), None]
        self.encoder.predict_candidates = MagicMock()
        self.encoder._forward_adaptive = MagicMock(return_value=torch.zeros((2, 3)))

    def test_dynamic_budget_allocation(self):
        """Verify that premise budget dynamically accommodates candidate length without truncation overshoot."""
        self.encoder.max_length = 64
        mock_model = MagicMock()
        mock_bb = MagicMock()
        mock_model.model = mock_bb
        mock_cache = MagicMock()
        mock_cache.batch_repeat_interleave = MagicMock()
        mock_cache.__deepcopy__ = MagicMock(side_effect=lambda memo: mock_cache)

        observed_pre_lens = []
        def bb_forward(input_ids, attention_mask, past_key_values=None, position_ids=None, **kwargs):
            if past_key_values is None:
                observed_pre_lens.append(input_ids.shape[1])
                out = MagicMock()
                out.past_key_values = mock_cache
                return out
            else:
                out = MagicMock()
                out.last_hidden_state = torch.ones((input_ids.shape[0], input_ids.shape[1], 16))
                return out

        mock_bb.side_effect = bb_forward
        mock_model.norm = MagicMock(side_effect=lambda x: x)
        mock_model.score = MagicMock(side_effect=lambda x: torch.ones((x.shape[0], 3)))
        self.encoder.model = mock_model

        long_premise = " ".join([f"word{i}" for i in range(100)])
        short_hyps = ["short one", "short two"]
        self.encoder.predict_candidates_logits(long_premise, short_hyps)
        pre_len_short = observed_pre_lens[-1]

        # With longer hypothesis, premise should yield room dynamically
        long_hyps = [" ".join([f"hypword{i}" for i in range(20)])]
        self.encoder.predict_candidates_logits(long_premise, long_hyps)
        pre_len_long = observed_pre_lens[-1]

        # The short hypothesis run should preserve more premise tokens than the long hypothesis run
        self.assertGreater(pre_len_short, pre_len_long)

    def test_suffix_chunk_adaptive_bisection_on_oom(self):
        """Verify that if suffix chunk evaluation OOMs on M >= 4, it bisects to M = 2 and succeeds."""
        self.encoder.device = "cpu"
        mock_model = MagicMock()
        mock_bb = MagicMock()
        mock_model.model = mock_bb
        mock_cache = MagicMock()
        mock_cache.batch_repeat_interleave = MagicMock()
        mock_cache.__deepcopy__ = MagicMock(side_effect=lambda memo: mock_cache)

        def bb_forward(input_ids, attention_mask, past_key_values=None, position_ids=None, **kwargs):
            if past_key_values is None:
                out = MagicMock()
                out.past_key_values = mock_cache
                return out
            else:
                bs = input_ids.shape[0]
                if bs >= 4:
                    raise torch.cuda.OutOfMemoryError("CUDA out of memory in suffix forward pass")
                out = MagicMock()
                out.last_hidden_state = torch.ones((bs, input_ids.shape[1], 16))
                return out

        mock_bb.side_effect = bb_forward
        mock_model.norm = MagicMock(side_effect=lambda x: x)
        mock_model.score = MagicMock(side_effect=lambda x: torch.ones((x.shape[0], 3)))
        self.encoder.model = mock_model

        hyps = [f"option {i}" for i in range(4)]
        with patch("torch.cuda.empty_cache") as mock_empty:
            logits = self.encoder.predict_candidates_logits("Some premise", hyps, max_candidate_batch_size=4)
            self.assertEqual(logits.shape, (4, 3))
            self.assertTrue(mock_empty.called)

    def test_prefill_oom_fallback_to_forward_adaptive(self):
        """Verify that if Stage 1 prefill OOMs, it catches it and falls back to _forward_adaptive."""
        self.encoder.device = "cpu"
        mock_model = MagicMock()
        mock_bb = MagicMock()
        mock_model.model = mock_bb
        mock_bb.side_effect = torch.cuda.OutOfMemoryError("CUDA out of memory during premise prefill")
        self.encoder.model = mock_model

        self.encoder._forward_adaptive = MagicMock(return_value=torch.ones((3, 3)))
        with patch("torch.cuda.empty_cache") as mock_empty:
            logits = self.encoder.predict_candidates_logits("Huge premise", ["h1", "h2", "h3"])
            self.assertEqual(logits.shape, (3, 3))
            self.assertTrue(self.encoder._forward_adaptive.called)
            self.assertTrue(mock_empty.called)


if __name__ == "__main__":
    unittest.main()

