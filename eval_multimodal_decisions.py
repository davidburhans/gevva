#!/usr/bin/env python3
"""eval_multimodal_decisions.py - Genuine Multimodal & Multi-Image Capability Benchmark.

Measures the 4 core System 1 visual decision muscles:
1. Multi-Image Before/After State Attribution (Action effects, state progression)
2. Adversarial Distractor Resistance (Subtle visual mutations, hard negatives)
3. Visual Neutral Calibration (Pixel-unverifiable properties, ECE, Brier score)
4. Cross-Chart & Quantitative Visual Reasoning (Numeric comparisons, table facts)

Usage:
  uv run python eval_multimodal_decisions.py --model-path ckpt/gevva-e2b
  uv run python eval_multimodal_decisions.py --model-path ckpt/gevva-e2b --out-dir results/multimodal_eval
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from gemma4_cross_encoder import (
    Gemma4CrossEncoder,
    CONTRADICTION,
    ENTAILMENT,
    NEUTRAL,
    ID2LABEL,
    LABEL2ID,
)


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Computes Expected Calibration Error over top-1 predictions."""
    if len(probs) == 0:
        return 0.0
    confidences = np.max(probs, axis=-1)
    predictions = np.argmax(probs, axis=-1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_total = len(confidences)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        mask = (confidences > bin_lower) & (confidences <= bin_upper) if i > 0 else (confidences >= bin_lower) & (confidences <= bin_upper)
        bin_count = np.sum(mask)
        if bin_count > 0:
            bin_acc = np.mean(accuracies[mask])
            bin_conf = np.mean(confidences[mask])
            ece += (bin_count / n_total) * abs(bin_acc - bin_conf)

    return float(ece)


def compute_brier(probs: np.ndarray, labels: np.ndarray) -> float:
    """Computes multi-class Brier score."""
    if len(probs) == 0:
        return 0.0
    n_classes = probs.shape[-1]
    one_hot = np.eye(n_classes)[labels]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=-1)))


def evaluate_dataset(
    encoder: Gemma4CrossEncoder,
    data_path: Path,
    image_root: Optional[Path] = None,
    temperature: float = 1.0,
) -> Dict[str, Any]:
    """Runs cross-encoder evaluation over a JSONL benchmark dataset with multi-image support."""
    if not data_path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {data_path}")

    if image_root is None:
        image_root = data_path.parent / "images"
        if not image_root.exists():
            image_root = data_path.parent

    records = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        return {"error": "empty_dataset", "count": 0}

    pairs = []
    batch_images = []
    gold_labels = []
    families = []

    for r in records:
        pairs.append((r["premise"], r["hypothesis"]))
        
        # Parse label
        lbl = r["label"]
        if isinstance(lbl, str):
            gold_labels.append(LABEL2ID.get(lbl.lower(), 1))
        else:
            gold_labels.append(int(lbl))

        families.append(r.get("task_family", r.get("family", "general")))

        # Resolve image(s)
        raw_imgs = r.get("images")
        if raw_imgs is None:
            single = r.get("image")
            raw_imgs = single if isinstance(single, list) else ([single] if single else [])
        elif isinstance(raw_imgs, str):
            raw_imgs = [raw_imgs]

        loaded_imgs = []
        for img_p in raw_imgs:
            p = Path(img_p)
            if not p.is_absolute():
                p = image_root / p
            if p.exists():
                try:
                    loaded_imgs.append(Image.open(p).convert("RGB"))
                except Exception:
                    pass
        
        batch_images.append(loaded_imgs if loaded_imgs else None)

    # Predict in batch
    t0 = time.perf_counter()
    probs = encoder.predict(pairs, images=batch_images, temperature=temperature)
    latency_ms = ((time.perf_counter() - t0) / len(pairs)) * 1000.0

    preds = np.argmax(probs, axis=-1)
    golds = np.array(gold_labels)

    acc = float(np.mean(preds == golds))
    ece = compute_ece(probs, golds)
    brier = compute_brier(probs, golds)

    # Breakdown by task family
    family_metrics = {}
    for fam in sorted(set(families)):
        fam_mask = np.array([f == fam for f in families])
        if np.sum(fam_mask) > 0:
            fam_acc = float(np.mean(preds[fam_mask] == golds[fam_mask]))
            fam_ece = compute_ece(probs[fam_mask], golds[fam_mask])
            fam_brier = compute_brier(probs[fam_mask], golds[fam_mask])
            family_metrics[fam] = {
                "count": int(np.sum(fam_mask)),
                "accuracy": round(fam_acc * 100, 2),
                "ece": round(fam_ece, 4),
                "brier": round(fam_brier, 4),
            }

    # Breakdown by label
    label_metrics = {}
    for lbl_id, lbl_name in ID2LABEL.items():
        lbl_mask = golds == lbl_id
        if np.sum(lbl_mask) > 0:
            lbl_acc = float(np.mean(preds[lbl_mask] == lbl_id))
            label_metrics[lbl_name] = {
                "count": int(np.sum(lbl_mask)),
                "recall": round(lbl_acc * 100, 2),
            }

    return {
        "total_samples": len(pairs),
        "overall_accuracy": round(acc * 100, 2),
        "ece": round(ece, 4),
        "brier": round(brier, 4),
        "p50_latency_ms": round(latency_ms, 2),
        "family_breakdown": family_metrics,
        "label_breakdown": label_metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Gevva on Multimodal Decisions")
    parser.add_argument("--model-path", default="ckpt/gevva-e2b", help="Model checkpoint path")
    parser.add_argument("--data", default="data/visual_synth/val.jsonl", help="Evaluation JSONL path")
    parser.add_argument("--image-root", default=None, help="Root folder for images")
    parser.add_argument("--temperature", type=float, default=1.0, help="Inference temperature")
    parser.add_argument("--out-dir", default="results/multimodal_eval", help="Directory to save report")
    args = parser.parse_args()

    print(f"Loading Gevva Cross-Encoder from: {args.model_path}...")
    encoder = Gemma4CrossEncoder(model_name_or_path=args.model_path, device="cuda")

    data_p = Path(args.data)
    img_root = Path(args.image_root) if args.image_root else None

    print(f"Evaluating benchmark: {data_p}...")
    report = evaluate_dataset(encoder, data_p, image_root=img_root, temperature=args.temperature)

    print("\n" + "=" * 65)
    print("MULTIMODAL DECISION EVALUATION REPORT")
    print("=" * 65)
    print(f"Total Samples:      {report.get('total_samples', 0):,}")
    print(f"Overall Accuracy:   {report.get('overall_accuracy', 0.0)}%")
    print(f"ECE Calibration:    {report.get('ece', 0.0)}")
    print(f"Brier Score:        {report.get('brier', 0.0)}")
    print(f"Per-Pair Latency:   {report.get('p50_latency_ms', 0.0)} ms")
    print("\nFamily Breakdown:")
    for fam, m in report.get("family_breakdown", {}).items():
        print(f"  {fam:<25} ({m['count']:>4} samples): Acc = {m['accuracy']:>5.2f}%, ECE = {m['ece']:.4f}")
    print("=" * 65)

    out_p = Path(args.out_dir) / "eval_report.json"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved evaluation report to: {out_p}")


if __name__ == "__main__":
    main()
