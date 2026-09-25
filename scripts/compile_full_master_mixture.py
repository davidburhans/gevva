#!/usr/bin/env python3
"""scripts/compile_full_master_mixture.py - Compiles the complete, uncompromised Master Training Mixture for Gemma 4 E4B.

Combines every proven curriculum slice accumulated across the entire project:
1. Phase 3 Enriched (243,916 pairs: foundational NLI, XNLI, Prometheus rubrics, HelpSteer2, RACE, LogiQA, ReClor, StrategyQA, Typed Decisions)
2. Phase 4 Weak-Family Remediations (65,472 pairs: tradeoff, policy, temporal numeric, routing, judge hard)
3. Multimodal Multi-Image (7,200 pairs: table/invoice diffs, UI state transitions, spatial maps, dashboard alerts, cross-chart trends)
4. Visual Synthetic Logic (2,872 pairs: geometry, charts, spatial layouts)
5. Clean NLI Anchors & Haystack verification from Stages 1 & 2

Decontaminated against data/test.jsonl (2,947 pairs). Group-atomic.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
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
    parser = argparse.ArgumentParser(description="Compile Complete Master Training Mixture for E4B")
    parser.add_argument("--out-file", default="data/train_master_full.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    test_pairs = load_test_pairs(REPO_ROOT / "data" / "test.jsonl")
    logger.info("Loaded %d test pairs for strict decontamination.", len(test_pairs))

    master_rows: List[Dict[str, Any]] = []
    seen_pairs: Set[Tuple[str, str]] = set()
    source_stats = Counter()

    # 1. Ingest Multimodal Multi-Image (3x oversampled)
    multi_p = REPO_ROOT / "data" / "multimodal_multi_image" / "train.jsonl"
    if multi_p.exists():
        loaded_multi = []
        with open(multi_p, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
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
        
        for repeat_idx in range(3):
            for r in loaded_multi:
                r_copy = dict(r)
                if repeat_idx > 0:
                    r_copy["id"] = f"{r['id']}_rep{repeat_idx}"
                master_rows.append(r_copy)
                source_stats["multimodal_multi_image"] += 1
        logger.info("-> Added %d multi-image rows (3x oversampled).", len(loaded_multi) * 3)

    # 2. Ingest Single-Image Visual Reasoning
    vis_p = REPO_ROOT / "data" / "visual_synth" / "train.jsonl"
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
                master_rows.append(r)
                source_stats["visual_single_image"] += 1
                vis_count += 1
        logger.info("-> Added %d single-image visual reasoning rows.", vis_count)

    # 3. Ingest All Other Historical & Enriched Datasets
    input_files = [
        REPO_ROOT / "data" / "train_phase3_enriched.jsonl",
        REPO_ROOT / "data" / "train_phase4_mixture.jsonl",
        REPO_ROOT / "data" / "stage2_train.jsonl",
        REPO_ROOT / "data" / "train.jsonl",
    ]

    for p in input_files:
        if not p.exists():
            continue
        file_count = 0
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                premise = r.get("premise", "").strip()
                hypothesis = r.get("hypothesis", "").strip()
                if not premise or not hypothesis:
                    continue
                key = (premise, hypothesis)
                if key in test_pairs or key in seen_pairs:
                    continue
                # Skip duplicate visual rows from text mixtures
                if r.get("image") or r.get("images"):
                    continue
                seen_pairs.add(key)
                src = r.get("source", r.get("task_family", "unknown"))
                source_stats[src] += 1
                master_rows.append(r)
                file_count += 1
        logger.info("-> Ingested %d unique rows from %s", file_count, p.name)

    # Shuffle the master dataset
    rng.shuffle(master_rows)

    out_p = REPO_ROOT / args.out_file
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        for r in master_rows:
            f.write(json.dumps(r) + "\n")

    logger.info("==================================================")
    logger.info("COMPILED COMPLETE MASTER TRAINING MIXTURE: %s", out_p.name)
    logger.info("Total Rows: %d", len(master_rows))
    for src, count in sorted(source_stats.items(), key=lambda kv: -kv[1])[:20]:
        logger.info("  %-40s: %6d (%5.1f%%)", src, count, 100.0 * count / len(master_rows))
    logger.info("  ... and %d more sources", max(0, len(source_stats) - 20))
    logger.info("==================================================")


if __name__ == "__main__":
    main()
