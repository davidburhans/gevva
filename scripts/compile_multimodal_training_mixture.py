#!/usr/bin/env python3
"""scripts/compile_multimodal_training_mixture.py - Compiles Multimodal Multi-Image Training Mixture.

Recipe:
1. Multi-Image Before/After Pairs: data/multimodal_multi_image/train.jsonl (2,400 rows, 3x oversampled -> 7,200 rows)
2. Single-Image Visual Reasoning: data/visual_synth/train.jsonl (2,872 rows)
3. Balanced Clean NLI Anchors: data/train.jsonl (5,000 rows, group_id: -1, 1:1:1 class balance)
4. Enterprise Decision Anchors: data/train_phase4_mixture.jsonl (5,000 rows)
5. Zero Contamination: Strict filtering against data/test.jsonl.

Output:
  data/train_multimodal_mixture.jsonl (~20,072 rows)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nli_labels import CONTRADICTION, ENTAILMENT, NEUTRAL, ID2LABEL

logger = logging.getLogger(__name__)


def load_test_pairs(test_path: Path) -> Set[Tuple[str, str]]:
    test_pairs = set()
    if test_path.exists():
        with open(test_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    test_pairs.add((r["premise"].strip(), r["hypothesis"].strip()))
    return test_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile Multimodal Multi-Image Training Mixture")
    parser.add_argument("--multi-image-train", default="data/multimodal_multi_image/train.jsonl")
    parser.add_argument("--visual-synth-train", default="data/visual_synth/train.jsonl")
    parser.add_argument("--clean-nli", default="data/train.jsonl")
    parser.add_argument("--decision-anchors", default="data/train_phase4_mixture.jsonl")
    parser.add_argument("--out-file", default="data/train_multimodal_mixture.jsonl")
    parser.add_argument("--multi-image-repeat", type=int, default=3, help="Oversampling factor for multi-image data")
    parser.add_argument("--nli-anchor-count", type=int, default=5000)
    parser.add_argument("--decision-anchor-count", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    rng = random.Random(args.seed)

    test_pairs = load_test_pairs(REPO_ROOT / "data" / "test.jsonl")
    logger.info("Loaded %d test pairs for strict decontamination.", len(test_pairs))

    mixture_rows: List[Dict[str, Any]] = []
    source_stats = Counter()

    # 1. Ingest Multi-Image Pairs
    multi_p = REPO_ROOT / args.multi_image_train
    if multi_p.exists():
        loaded_multi = []
        with open(multi_p, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                # Ensure image paths are relative to REPO_ROOT
                fixed_images = []
                for img in r.get("images", []):
                    if img.startswith("images/"):
                        fixed_images.append(f"data/multimodal_multi_image/{img}")
                    else:
                        fixed_images.append(img)
                r["images"] = fixed_images
                r["group_id"] = -1
                r["source"] = f"multimodal_{r.get('task_family', 'multi_image')}"
                loaded_multi.append(r)
        
        # Apply oversampling
        for repeat_idx in range(args.multi_image_repeat):
            for r in loaded_multi:
                r_copy = dict(r)
                if repeat_idx > 0:
                    r_copy["id"] = f"{r['id']}_rep{repeat_idx}"
                mixture_rows.append(r_copy)
                source_stats["multimodal_multi_image"] += 1
        logger.info("-> Ingested %d multi-image samples (repeated %dx -> %d rows).", len(loaded_multi), args.multi_image_repeat, len(loaded_multi) * args.multi_image_repeat)

    # 2. Ingest Single-Image Visual Reasoning
    vis_p = REPO_ROOT / args.visual_synth_train
    if vis_p.exists():
        vis_count = 0
        with open(vis_p, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("image") and str(r["image"]).startswith("images/"):
                    r["image"] = f"data/visual_synth/{r['image']}"
                r["group_id"] = -1
                r["source"] = "visual_single_image"
                mixture_rows.append(r)
                source_stats["visual_single_image"] += 1
                vis_count += 1
        logger.info("-> Ingested %d single-image visual synthetic samples.", vis_count)

    # 3. Ingest Balanced Clean NLI Anchors
    clean_p = REPO_ROOT / args.clean_nli
    if clean_p.exists():
        by_label = defaultdict(list)
        with open(clean_p, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                lbl = r.get("label")
                if lbl in (CONTRADICTION, ENTAILMENT, NEUTRAL):
                    key = (r["premise"].strip(), r["hypothesis"].strip())
                    if key in test_pairs:
                        continue
                    by_label[lbl].append(r)

        target_per_lbl = args.nli_anchor_count // 3
        anchors = []
        for lbl in (CONTRADICTION, ENTAILMENT, NEUTRAL):
            pool = by_label[lbl]
            sampled = rng.sample(pool, min(len(pool), target_per_lbl))
            for r in sampled:
                anchors.append({
                    "premise": r["premise"],
                    "hypothesis": r["hypothesis"],
                    "label": lbl,
                    "group_id": -1,
                    "is_gold": (lbl == ENTAILMENT),
                    "source": "anchor_clean_nli",
                })
        rng.shuffle(anchors)
        mixture_rows.extend(anchors)
        source_stats["anchor_clean_nli"] += len(anchors)
        logger.info("-> Ingested %d balanced clean NLI anchor rows.", len(anchors))

    # 4. Ingest Enterprise Decision Anchors
    dec_p = REPO_ROOT / args.decision_anchors
    if dec_p.exists():
        dec_candidates = []
        with open(dec_p, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                key = (r.get("premise", "").strip(), r.get("hypothesis", "").strip())
                if key in test_pairs:
                    continue
                # Exclude visual rows from this pool to prevent duplicates
                if r.get("image") or r.get("images"):
                    continue
                dec_candidates.append(r)
        
        sample_dec = rng.sample(dec_candidates, min(len(dec_candidates), args.decision_anchor_count))
        for r in sample_dec:
            r_copy = dict(r)
            r_copy["source"] = "anchor_decision"
            mixture_rows.append(r_copy)
            source_stats["anchor_decision"] += 1
        logger.info("-> Ingested %d enterprise decision anchor rows.", len(sample_dec))

    # Shuffle the final mixture
    rng.shuffle(mixture_rows)

    out_p = REPO_ROOT / args.out_file
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        for r in mixture_rows:
            f.write(json.dumps(r) + "\n")

    logger.info("==================================================")
    logger.info("COMPILED MULTIMODAL TRAINING MIXTURE: %s", out_p.name)
    logger.info("Total Rows: %d", len(mixture_rows))
    for src, count in sorted(source_stats.items(), key=lambda kv: -kv[1]):
        logger.info("  %-30s: %6d (%5.1f%%)", src, count, 100.0 * count / len(mixture_rows))
    logger.info("==================================================")


if __name__ == "__main__":
    main()
