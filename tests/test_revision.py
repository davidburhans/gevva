"""Unit tests for revision resolution and variant loading in Gevva."""

import os
import unittest
from unittest.mock import MagicMock, patch

from gevva import DEFAULT_MODEL_ID, load
from gemma4_cross_encoder import Gemma4CrossEncoder


class TestRevisionResolution(unittest.TestCase):
    def test_default_model_id_resolution(self):
        # When revision is None and ckpt/gevva-e2b exists, should point to ckpt/gevva-e2b
        if os.path.isdir("ckpt/gevva-e2b"):
            with patch("gemma4_cross_encoder.AutoTokenizer.from_pretrained") as mock_tok, \
                 patch("gemma4_cross_encoder.Gemma4ForSequenceClassification.from_pretrained") as mock_model:
                mock_tok.return_value = MagicMock(pad_token=None)
                mock_model.return_value = MagicMock()
                enc = Gemma4CrossEncoder(model_name_or_path=DEFAULT_MODEL_ID)
                self.assertIsNone(enc.revision)
                # Verify that it resolved to the local flagship checkpoint
                mock_tok.assert_called_once()
                called_path = mock_tok.call_args[0][0]
                self.assertEqual(called_path, "ckpt/gevva-e2b")

    def test_multimodal_revision_resolution(self):
        # When revision is 'multimodal' and ckpt/gevva-e2b-phase4/best exists, should point to ckpt/gevva-e2b-phase4/best
        if os.path.isdir("ckpt/gevva-e2b-phase4/best"):
            with patch("gemma4_cross_encoder.AutoTokenizer.from_pretrained") as mock_tok, \
                 patch("gemma4_cross_encoder.Gemma4ForSequenceClassification.from_pretrained") as mock_model:
                mock_tok.return_value = MagicMock(pad_token=None)
                mock_model.return_value = MagicMock()
                enc = Gemma4CrossEncoder(model_name_or_path=DEFAULT_MODEL_ID, revision="multimodal")
                self.assertEqual(enc.revision, "multimodal")
                mock_tok.assert_called_once()
                called_path = mock_tok.call_args[0][0]
                self.assertEqual(called_path, "ckpt/gevva-e2b-phase4/best")

    def test_gevva_load_multimodal_revision(self):
        if os.path.isdir("ckpt/gevva-e2b-phase4/best"):
            with patch("gemma4_cross_encoder.AutoTokenizer.from_pretrained") as mock_tok, \
                 patch("gemma4_cross_encoder.Gemma4ForSequenceClassification.from_pretrained") as mock_model:
                mock_tok.return_value = MagicMock(pad_token=None)
                mock_model.return_value = MagicMock()
                enc = load(revision="multimodal")
                self.assertEqual(enc.revision, "multimodal")
                called_path = mock_tok.call_args[0][0]
                self.assertEqual(called_path, "ckpt/gevva-e2b-phase4/best")

    def test_hf_revision_kwarg_passed_when_remote(self):
        with patch("gemma4_cross_encoder.AutoTokenizer.from_pretrained") as mock_tok, \
             patch("gemma4_cross_encoder.Gemma4ForSequenceClassification.from_pretrained") as mock_model:
            mock_tok.return_value = MagicMock(pad_token=None)
            mock_model.return_value = MagicMock()
            enc = Gemma4CrossEncoder(model_name_or_path="davidburhans/gevva-remote-test", revision="multimodal")
            self.assertEqual(enc.revision, "multimodal")
            mock_tok.assert_called_once_with("davidburhans/gevva-remote-test", revision="multimodal")
            mock_model.assert_called_once()
    def test_multimodal_dedicated_repo_id_resolution(self):
        if os.path.isdir("ckpt/gevva-e2b-phase4/best"):
            with patch("gemma4_cross_encoder.AutoTokenizer.from_pretrained") as mock_tok, \
                 patch("gemma4_cross_encoder.Gemma4ForSequenceClassification.from_pretrained") as mock_model:
                mock_tok.return_value = MagicMock(pad_token=None)
                mock_model.return_value = MagicMock()
                enc = Gemma4CrossEncoder(model_name_or_path="davidburhans/gevva-e2b-multimodal")
                mock_tok.assert_called_once()
                called_path = mock_tok.call_args[0][0]
                self.assertEqual(called_path, "ckpt/gevva-e2b-phase4/best")

                # Also test via gevva.load
                mock_tok.reset_mock()
                enc2 = load(model_name_or_path="davidburhans/gevva-e2b-multimodal")
                called_path2 = mock_tok.call_args[0][0]
                self.assertEqual(called_path2, "ckpt/gevva-e2b-phase4/best")


if __name__ == "__main__":
    unittest.main()
