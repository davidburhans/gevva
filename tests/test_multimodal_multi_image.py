#!/usr/bin/env python3
"""tests/test_multimodal_multi_image.py - Unit test suite for multi-image cross-encoder inputs.

Tests:
1. tokenize_nli_pair_safe with 0, 1, 2, and 3 image token budgets.
2. CustomNLICollator on mixed batches (text-only, single-image, multi-image).
3. GroupedDecisionCollator on group-atomic multi-image decision candidate sets.
4. Gemma 4 forward pass with multi-image inputs.
"""

import os
import tempfile
import unittest
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoTokenizer

from gemma4_cross_encoder import (
    tokenize_nli_pair_safe,
    Gemma4CrossEncoder,
    CONTRADICTION,
    ENTAILMENT,
    NEUTRAL,
)
from finetune import CustomNLICollator
from research.adapters.grouped_decision_collator import GroupedDecisionCollator


class TestMultimodalMultiImage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = AutoTokenizer.from_pretrained("google/gemma-4-E2B-it")
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.img_dir = Path(cls.temp_dir.name)

        # Create dummy images
        cls.img1_path = str(cls.img_dir / "red.png")
        cls.img2_path = str(cls.img_dir / "blue.png")
        cls.img3_path = str(cls.img_dir / "green.png")

        Image.new("RGB", (64, 64), color="red").save(cls.img1_path)
        Image.new("RGB", (64, 64), color="blue").save(cls.img2_path)
        Image.new("RGB", (64, 64), color="green").save(cls.img3_path)

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_tokenize_zero_image(self):
        prem = "The quick brown fox"
        hyp = "A fox jumps"
        ids = tokenize_nli_pair_safe(self.tok, prem, hyp, max_length=128, image_soft_tokens=0)
        text = self.tok.decode(ids)
        self.assertIn("Premise: The quick brown fox", text)
        self.assertIn("Hypothesis: A fox jumps", text)
        self.assertIn("\nPrediction:", text)
        self.assertNotIn("<|image|>", text)

    def test_tokenize_single_image(self):
        prem = "This is a single image scene"
        hyp = "The image is red"
        ids = tokenize_nli_pair_safe(self.tok, prem, hyp, max_length=512, image_soft_tokens=10)
        img_id = self.tok.convert_tokens_to_ids("<|image|>")
        self.assertEqual(ids.count(img_id), 10)
        text = self.tok.decode(ids)
        self.assertIn("<|image>", text)
        self.assertIn("<image|>", text)
        self.assertNotIn("Image 1:", text)  # single image does not add redundant prefix

    def test_tokenize_multi_image(self):
        prem = "Comparison between two scenes"
        hyp = "Image 1 is red and Image 2 is blue"
        ids = tokenize_nli_pair_safe(self.tok, prem, hyp, max_length=1024, image_soft_tokens=[15, 20])
        img_id = self.tok.convert_tokens_to_ids("<|image|>")
        self.assertEqual(ids.count(img_id), 35)
        text = self.tok.decode(ids)
        self.assertIn("Image 1: ", text)
        self.assertIn("Image 2: ", text)
        self.assertIn("Hypothesis: Image 1 is red and Image 2 is blue", text)

    def test_custom_nli_collator_mixed_batch(self):
        collator = CustomNLICollator(
            tokenizer=self.tok,
            max_length=1024,
            image_root=str(self.img_dir),
        )

        batch_records = [
            # 1. Multi-image row (2 images)
            {
                "premise": "Two visual charts",
                "hypothesis": "Both charts show data",
                "label": ENTAILMENT,
                "images": [self.img1_path, self.img2_path],
                "source": "multi_chart",
            },
            # 2. Text-only row
            {
                "premise": "A sunny day outside",
                "hypothesis": "It is cloudy",
                "label": CONTRADICTION,
                "source": "snli",
            },
            # 3. Single-image row
            {
                "premise": "A single photograph",
                "hypothesis": "The photo contains green",
                "label": ENTAILMENT,
                "image": self.img3_path,
                "source": "single_photo",
            },
        ]

        batch = collator(batch_records)

        self.assertIn("input_ids", batch)
        self.assertIn("attention_mask", batch)
        self.assertIn("pixel_values", batch)
        self.assertIn("image_position_ids", batch)

        # Batch size is 3
        self.assertEqual(batch["input_ids"].shape[0], 3)
        self.assertEqual(batch["attention_mask"].shape[0], 3)
        self.assertEqual(batch["labels"].shape[0], 3)

        # Total images in batch = 2 (sample 1) + 0 (sample 2) + 1 (sample 3) = 3 images
        self.assertEqual(batch["pixel_values"].shape[0], 3)
        self.assertEqual(batch["image_position_ids"].shape[0], 3)

    def test_grouped_decision_collator_multi_image(self):
        collator = GroupedDecisionCollator(
            tokenizer=self.tok,
            max_length=1024,
            image_root=str(self.img_dir),
        )

        group_records = [
            {
                "group_id": "decision_001",
                "premise": "Compare Image 1 and Image 2",
                "hypothesis": "Option A: Image 1 is red and Image 2 is blue",
                "is_gold": True,
                "label": ENTAILMENT,
                "images": [self.img1_path, self.img2_path],
            },
            {
                "group_id": "decision_001",
                "premise": "Compare Image 1 and Image 2",
                "hypothesis": "Option B: Image 1 is green and Image 2 is yellow",
                "is_gold": False,
                "label": CONTRADICTION,
                "images": [self.img1_path, self.img2_path],
            },
        ]

        batch = collator(group_records)

        self.assertEqual(batch["input_ids"].shape[0], 2)
        # 2 options * 2 images = 4 image instances
        self.assertEqual(batch["pixel_values"].shape[0], 4)
        self.assertEqual(batch["group_ids"].tolist(), [0, 0])
        self.assertEqual(batch["is_gold"].tolist(), [1.0, 0.0])


    def test_prepare_batch_multi_image(self):
        # Verify that Gemma4CrossEncoder._prepare_batch does not drop subsequent images
        class DummyEncoder:
            pass

        encoder = DummyEncoder()
        encoder.tokenizer = self.tok
        encoder.max_length = 512
        encoder.boi_token = "<|image>"
        encoder.image_token = "<|image|>"
        encoder.eoi_token = "<image|>"
        encoder.device = "cpu"
        encoder.dtype = torch.float32

        # Mock image processor
        class DummyFeat(dict):
            pass

        class DummyProc:
            def __call__(self, imgs, return_tensors="pt"):
                n = len(imgs)
                return {
                    "num_soft_tokens_per_image": [10] * n,
                    "pixel_values": torch.zeros((n, 20, 64)),
                    "image_position_ids": torch.zeros((n, 20, 2), dtype=torch.long),
                }

        encoder.image_processor = DummyProc()
        # Bind the real method to the dummy instance
        encoder._prepare_batch = Gemma4CrossEncoder._prepare_batch.__get__(encoder)

        img1 = Image.new("RGB", (32, 32), color="red")
        img2 = Image.new("RGB", (32, 32), color="blue")

        pairs = [
            ("Compare these two", "First is red, second is blue"),
            ("Text only pair", "Some hypothesis"),
        ]
        images = [
            [img1, img2],  # 2 images for pair 0
            None,          # 0 images for pair 1
        ]

        batch = encoder._prepare_batch(pairs, images)
        self.assertEqual(batch["input_ids"].shape[0], 2)
        # Exactly 2 images should be in pixel_values (not dropped to 1!)
        self.assertIn("pixel_values", batch)
        self.assertEqual(batch["pixel_values"].shape[0], 2)
        self.assertEqual(batch["image_position_ids"].shape[0], 2)


if __name__ == "__main__":
    unittest.main()

