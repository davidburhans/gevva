"""Tests for the checkpoint disk-space guard added after the 2026-09-29 r3
disk-full crash (15GB e4b shards filled a 3.6T disk during step checkpointing).
"""

import unittest

from finetune import _free_gb


class TestFreeGb(unittest.TestCase):
    def test_free_gb_returns_positive_float_for_existing_path(self):
        """_free_gb('.') must return a sane positive number of GB."""
        value = _free_gb(".")
        self.assertIsInstance(value, float)
        self.assertGreater(value, 0.0)

    def test_free_gb_never_raises_on_bad_path(self):
        """A failing stat must degrade to +inf (never block checkpointing)."""
        value = _free_gb("/nonexistent/path/that/cannot/be/statted")
        self.assertEqual(value, float("inf"))


if __name__ == "__main__":
    unittest.main()
