"""Tests for the local JevBench v1.2 composite implementation.

Anchor values are reverse-verified against published official numbers where the
formula is reproducible locally (e.g. the e2b champion Calibration 86.90 from
ECE 0.0655).
"""

import math
import unittest

from scripts.jevbench_composite import (
    calibration,
    cost,
    intelligence,
    jevbench_score,
    speed,
)


class TestJevbenchComposite(unittest.TestCase):
    def test_calibration_matches_published_champion_value(self):
        """e2b champion: ECE 0.0655 -> official Calibration 86.90."""
        self.assertAlmostEqual(calibration(0.0655), 86.90, places=2)

    def test_calibration_clips_at_zero(self):
        self.assertEqual(calibration(0.9), 0.0)

    def test_intelligence_renormalizes_missing_judge(self):
        """Public split has no judge tier; remaining weights renormalize."""
        tiers = {"easy": 1.0, "standard": 1.0, "hard": 0.5}
        expect = (0.14 * 100 + 0.28 * 100 + 0.30 * 50) / 0.72
        self.assertAlmostEqual(intelligence(tiers), expect, places=6)

    def test_speed_uses_own_gpu_adjustment(self):
        """s_adj = raw*2 + 0.15; champion p50 16.5ms -> 0.183s -> 94.75."""
        self.assertAlmostEqual(speed(0.0165, 0.0165), 100 - 20 * math.log10(0.183 / 0.1), places=6)

    def test_cost_matches_published_champion_tariff(self):
        """e2b champion $0.0149/1k -> official Cost 64.80."""
        self.assertAlmostEqual(cost(0.0149), 64.80, places=2)

    def test_jevbench_score_is_geometric_mean(self):
        """exp(mean ln) with a floor of 1 per axis; weak axes pull hard."""
        axes = {"intelligence": 73.91, "calibration": 86.90, "speed": 86.86, "cost": 64.80}
        expect = math.exp(0.25 * sum(math.log(v) for v in axes.values()))
        self.assertAlmostEqual(jevbench_score(axes), expect, places=6)
        # champion sanity: the four published axes reproduce 77.54
        self.assertAlmostEqual(jevbench_score(axes), 77.54, places=1)

    def test_missing_axis_returns_none(self):
        self.assertIsNone(jevbench_score({"intelligence": 50.0, "calibration": None,
                                           "speed": 50.0, "cost": 50.0}))


if __name__ == "__main__":
    unittest.main()
