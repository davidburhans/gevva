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
                    "state": r.get("state", ""),
                    "instructions": q.get("instructions", "Which option is the correct answer?"),
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
    if len(items) < 100:
        raise SystemExit(f"ABORT: loaded only {len(items)} MMLU-Pro items (expected thousands) - suite shard moved?")
    rng = random.Random(args.seed)
    sample = rng.sample(items, min(args.n, len(items)))

    enc = Gemma4CrossEncoder(args.model_path, device=args.device)
    correct = 0
    by_k = defaultdict(lambda: [0, 0])
    pred_counter = Counter()
    pos_counter = Counter()
    for item in sample:
        keys = [k for k, _ in item["options"]]
        # EXACT engine-served framing (review C1/m7): premise = state + question,
        # hypothesis = "The correct answer is {k}: {desc}.", argmax over the
        # entailment-minus-contradiction logit margin - identical to the
        # Decision Index adapter's choice() path. (The 2026-09-30 morning
        # 'fix' never landed: only the import line was applied, and the
        # finalizer ran the always-option-A script - caught by the validated
        # report showing 0.1320 for every model.)
        premise = f"{item['state']}\n\nQuestion: {item['instructions']}"
        hyps = []
        for k, desc in item["options"]:
            if desc and desc.lower() != str(k).lower():
                hyps.append(f"The correct answer is {k}: {desc}.")
            else:
                hyps.append(f"The correct answer is: {k}.")
        logits = np.asarray(enc.predict_candidates_logits(premise, hyps))
        if logits.ndim != 2 or logits.shape != (len(keys), 3):
            raise SystemExit(f"ABORT: predict_candidates_logits shape {logits.shape} != ({len(keys)}, 3)")
        margins = logits[:, 1] - logits[:, 0]
        best_idx = int(np.argmax(margins))
        pred_key = keys[best_idx]
        pred_counter[pred_key] += 1
        pos_counter[best_idx] += 1
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
        "top_predicted_key": pred_counter.most_common(1)[0],
        "top_predicted_position": pos_counter.most_common(1)[0],
        "framing_note": "engine-served parity: premise=state+question, hypothesis='The correct answer is {k}: {desc}...', "
                        "argmax over ent-minus-contra logit margin (matches adapter choice())",
        "baselines": {"e2b_v1_raw": 0.2862, "e4b_v1_raw": 0.3641},
    }
    # Degenerate guards (review C1): key VALUES vary per item ("A".."E" vs "1".."9"),
    # so a value-count check alone misses always-position-0 predictions. Check both.
    top_key, top_n = pred_counter.most_common(1)[0]
    top_pos, top_pos_n = pos_counter.most_common(1)[0]
    if top_n > 0.9 * len(sample) or top_pos_n > 0.9 * len(sample):
        raise SystemExit(
            f"ABORT: degenerate predictions - key {top_key} x{top_n}, position {top_pos} x{top_pos_n} of {len(sample)}")
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
