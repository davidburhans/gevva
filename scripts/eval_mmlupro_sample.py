#!/usr/bin/env python3
"""eval_mmlupro_sample.py - K-robustness gate: MMLU-Pro accuracy by option count.

Samples N items from the Decision Index MMLU-Pro shard (cat 57, K=9-10 options),
scores them with predict_candidates (shared prefix KV), and reports overall
accuracy plus accuracy-by-K. Baselines: e2b v1 28.6% raw, e4b v1 36.4% raw.

Usage:
    uv run python scripts/eval_mmlupro_sample.py --model-path ckpt/X/best --out results/X_mmlupro.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import random

import numpy as np
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SHARDS = [
    REPO_ROOT / "suite-0.2" / "selected-rows.jsonl.gz",
    REPO_ROOT / "suite-0.2" / "added-rows.jsonl.gz",
]


def load_items() -> list:
    items = []
    for shard in SHARDS:
        if not shard.exists():
            continue
        with gzip.open(shard, "rt", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r.get("_evaluation", {}).get("catalog_id") != 57:
                    continue
                q = list(r["questions"].values())[0]
                criteria = q.get("criteria", {})
                gold = (r.get("expected") or {}).get("q")
                if not criteria or gold not in criteria:
                    continue
                items.append({
                    "premise": str(r.get("state", "")),
                    "options": list(criteria.items()),  # [(key, text)]
                    "gold_key": gold,
                })
    return items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from gemma4_cross_encoder import Gemma4CrossEncoder

    items = load_items()
    rng = random.Random(args.seed)
    sample = rng.sample(items, min(args.n, len(items)))

    enc = Gemma4CrossEncoder(args.model_path, device=args.device)
    correct = 0
    by_k = defaultdict(lambda: [0, 0])
    for item in sample:
        keys = [k for k, _ in item["options"]]
        texts = [t for _, t in item["options"]]
        best = enc.predict_candidates(item["premise"], texts)
        # predict_candidates returns an int-like index OR a 1-element array depending on path
        best_idx = int(np.asarray(best).reshape(-1)[0])
        pred_key = keys[best_idx]
        ok = pred_key == item["gold_key"]
        correct += int(ok)
        k = len(keys)
        by_k[k][0] += int(ok)
        by_k[k][1] += 1

    report = {
        "model_path": args.model_path,
        "n": len(sample),
        "accuracy": round(correct / len(sample), 4),
        "accuracy_by_k": {str(k): {"acc": v[0] / v[1], "n": v[1]} for k, v in sorted(by_k.items())},
        "baselines": {"e2b_v1_raw": 0.2862, "e4b_v1_raw": 0.3641},
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
