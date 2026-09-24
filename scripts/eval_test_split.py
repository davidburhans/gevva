#!/usr/bin/env python3
"""scripts/eval_test_split.py - Standardized evaluation on frozen test splits.

Evaluates an NLI Cross-Encoder model on data/test.jsonl (3,113 items) and/or
data/test_v2.jsonl, computing accuracy, per-class F1, ECE, and Brier score.

Usage:
  uv run python scripts/eval_test_split.py --model-path ckpt/gemma-4-e2b-nli-phase3/best
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from gemma4_cross_encoder import Gemma4CrossEncoder
from train_cross_encoder import compute_calibration_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate_split(
    enc: Gemma4CrossEncoder,
    data_path: str,
    batch_size: int = 64,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    logger.info(f"Loaded {len(rows):,} items from {data_path}")

    pairs = [(r["premise"], r["hypothesis"]) for r in rows]
    golds = np.array([int(r["label"]) for r in rows], dtype=int)

    probs = enc.predict(pairs)
    preds = np.argmax(probs, axis=-1)

    acc = float(np.mean(preds == golds))
    calib = compute_calibration_metrics(probs, golds)

    # Per-class metrics
    labels = [0, 1, 2]
    class_names = ["contradiction", "entailment", "neutral"]
    per_class = {}
    for l, name in zip(labels, class_names):
        mask_gold = golds == l
        mask_pred = preds == l
        tp = int(np.sum(mask_gold & mask_pred))
        fp = int(np.sum((~mask_gold) & mask_pred))
        fn = int(np.sum(mask_gold & (~mask_pred)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[name] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": int(np.sum(mask_gold)),
        }

    # Language breakdown if available
    by_lang = {}
    for i, r in enumerate(rows):
        lang = r.get("language", "unknown")
        if lang not in by_lang:
            by_lang[lang] = {"correct": 0, "total": 0}
        by_lang[lang]["total"] += 1
        if preds[i] == golds[i]:
            by_lang[lang]["correct"] += 1

    lang_accs = {
        lang: round(v["correct"] / v["total"], 4)
        for lang, v in sorted(by_lang.items())
    }

    return {
        "data_path": data_path,
        "n_samples": len(rows),
        "accuracy": round(acc, 4),
        "ece": round(calib["ece"], 4),
        "brier": round(calib["brier"], 4),
        "per_class": per_class,
        "language_accuracy": lang_accs,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate on frozen test split")
    parser.add_argument("--model-path", default="ckpt/gemma-4-e2b-nli-phase3/best", help="Model checkpoint path")
    parser.add_argument("--test-file", default="data/test.jsonl", help="Primary test JSONL path")
    parser.add_argument("--test-v2-file", default="data/test_v2.jsonl", help="Test v2 JSONL path (optional)")
    parser.add_argument("--out-dir", default=None, help="Output directory (defaults to model-path)")
    parser.add_argument("--device", default="cuda", help="Inference device")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.model_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading model from {args.model_path}...")
    enc = Gemma4CrossEncoder(args.model_path, device=args.device)

    results = {}
    if os.path.exists(args.test_file):
        logger.info(f"\n--- Evaluating on primary test split: {args.test_file} ---")
        res_primary = evaluate_split(enc, args.test_file)
        results["test_primary"] = res_primary
        logger.info(f"Primary Test Acc: {res_primary['accuracy']*100:.2f}% | ECE: {res_primary['ece']:.4f} | Brier: {res_primary['brier']:.4f}")
        for c, m in res_primary["per_class"].items():
            logger.info(f"  {c:<14s} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} (support={m['support']})")

    if args.test_v2_file and os.path.exists(args.test_v2_file):
        logger.info(f"\n--- Evaluating on restated test split: {args.test_v2_file} ---")
        res_v2 = evaluate_split(enc, args.test_v2_file)
        results["test_v2"] = res_v2
        logger.info(f"Test v2 Acc: {res_v2['accuracy']*100:.2f}% | ECE: {res_v2['ece']:.4f} | Brier: {res_v2['brier']:.4f}")

    out_file = out_dir / "test_evaluation_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved full test report to {out_file}")


if __name__ == "__main__":
    main()
