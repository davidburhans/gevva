#!/usr/bin/env python3
"""jevbench_composite.py - Official JevBench v1.2 composite from local public-split results.

Implements the published v1.2 formula (~/workspaces/jevbench/jevbench/composite_v12.py)
for artifacts produced by scripts/eval_jevbench_public.py:

  Intelligence  100 x weighted tier accuracy (easy .14 / standard .28 / judge .28 / hard .30).
                Our local public split has NO judge tier (sealed), so weights renormalize
                over easy/standard/hard per the v1.2 partial-run rule -> NOT directly
                comparable to official 4-tier scores (e.g. the published 77.28/77.54).
  Calibration   100 x (1 - ECE_hard / 0.5), clipped at 0 (matches the published e2b
                champion Calibration 86.90 from ECE 0.0655 exactly).
  Speed         mean(score(p50), score(p95)), score(s) = clamp(100 - 20 log10(s / 0.1 s)),
                with the v1.2 own-GPU-server adjustment: s_adj = s_raw x 2 + 0.15 s.
  Cost          clamp(100 - 30 log10(usd_per_1000 / $0.001), 0, 100).
  JevBench Score  geometric mean of the four axes: exp(sum 0.25 ln(max(axis, 1))).

Usage (library):
    from scripts.jevbench_composite import composite_from_result
    comp = composite_from_result(result_dict, usd_per_1000=0.0365)

Usage (CLI):
    uv run python scripts/jevbench_composite.py --result results/jevbench_public_best.json --usd-per-1000 0.0365
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Optional

TIER_WEIGHTS = {"easy": 0.14, "standard": 0.28, "judge": 0.28, "hard": 0.30}
SPEED_BEST_S = 0.1
SPEED_PER_DECADE = 20.0
OWN_GPU_LOAD_FACTOR = 2.0
OWN_GPU_ADD_S = 0.15
COST_BEST_USD = 0.001
COST_PER_DECADE = 30.0


def clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


def intelligence(tier_accuracies: Dict[str, Optional[float]]) -> Optional[float]:
    """v1.2 weighted accuracy; missing tiers renormalize (partial-run rule)."""
    num = den = 0.0
    for tier, weight in TIER_WEIGHTS.items():
        acc = tier_accuracies.get(tier)
        if acc is not None:
            num += weight * 100.0 * acc
            den += weight
    return num / den if den > 0 else None


def calibration(ece_hard: Optional[float]) -> Optional[float]:
    if ece_hard is None:
        return None
    return clamp(100.0 * (1.0 - ece_hard / 0.5))


def _speed_score(s_raw_s: Optional[float], own_gpu: bool = True) -> Optional[float]:
    if s_raw_s is None:
        return None
    s = s_raw_s * OWN_GPU_LOAD_FACTOR + OWN_GPU_ADD_S if own_gpu else s_raw_s
    return clamp(100.0 - SPEED_PER_DECADE * math.log10(s / SPEED_BEST_S))


def speed(p50_s: Optional[float], p95_s: Optional[float], own_gpu: bool = True) -> Optional[float]:
    a, b = _speed_score(p50_s, own_gpu), _speed_score(p95_s, own_gpu)
    return None if a is None or b is None else (a + b) / 2.0


def cost(usd_per_1000: Optional[float]) -> Optional[float]:
    if usd_per_1000 is None or usd_per_1000 <= 0:
        return None
    return clamp(100.0 - COST_PER_DECADE * math.log10(usd_per_1000 / COST_BEST_USD))


def jevbench_score(axes: Dict[str, Optional[float]]) -> Optional[float]:
    """Geometric mean of the four axes, v1.2 weighting (25:25:25:25)."""
    vals = [axes.get(k) for k in ("intelligence", "calibration", "speed", "cost")]
    if any(v is None for v in vals):
        return None
    return math.exp(sum(0.25 * math.log(max(v, 1.0)) for v in vals))


def composite_from_result(result: Dict, usd_per_1000: float) -> Dict:
    tiers = {t: v.get("accuracy") for t, v in (result.get("tiers") or {}).items()}
    overall = result.get("overall") or {}
    axes = {
        "intelligence": intelligence(tiers),
        "calibration": calibration(result.get("renormalized_ece_hard")),
        "speed": speed(
            (overall.get("p50_ms") or 0) / 1000.0,
            (overall.get("p95_ms") or 0) / 1000.0,
        ),
        "cost": cost(usd_per_1000),
    }
    axes["jevbench_score_local_3tier"] = jevbench_score(axes)
    axes["note"] = ("local 3-tier footing (no judge tier in public split); "
                    "not comparable to official 4-tier scores")
    return axes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True, help="eval_jevbench_public.py artifact JSON")
    parser.add_argument("--usd-per-1000", type=float, required=True,
                        help="Tariff assumption ($/1k decisions); use one consistent basis for compared models")
    args = parser.parse_args()
    result = json.loads(Path(args.result).read_text())
    print(json.dumps(composite_from_result(result, args.usd_per_1000), indent=2))


if __name__ == "__main__":
    main()
