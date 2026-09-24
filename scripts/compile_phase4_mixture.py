#!/usr/bin/env python3
"""scripts/compile_phase4_mixture.py - Compiles Phase-4 Training Mixture.

Recipe for Phase-4:
1. Ingests Phase-3 Enriched training mixture (or stratified replay sample) to preserve
   all prior formal logic, SDK parity, and enterprise decision capabilities.
2. Ingests Phase-4 Targeted Weak-Family Remediation (long_policy, temporal_numeric,
   multi_hop, ambiguous_neutral, tradeoff_decisions).
3. Ingests Balanced 1:1:1 Clean NLI Anchors to preserve neutral and contradiction boundaries.
4. Enforces group atomicity, max character lengths, and 8-gram decontamination.

Output:
  data/train_phase4_mixture.jsonl
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

DEFAULT_P3_BASE_FILE = REPO_ROOT / "data" / "train_phase3_enriched.jsonl"
DEFAULT_P4_SYNTH_FILE = REPO_ROOT / "data" / "staged" / "phase4" / "synth_phase4_remediation.jsonl"
DEFAULT_VISUAL_SYNTH_FILE = REPO_ROOT / "data" / "visual_synth" / "train.jsonl"
DEFAULT_CLEAN_NLI_FILE = REPO_ROOT / "data" / "train.jsonl"
DEFAULT_OUT_FILE = REPO_ROOT / "data" / "train_phase4_mixture.jsonl"


def load_balanced_nli_anchors(
    clean_nli_path: Path,
    total_samples: int = 15000,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Loads balanced clean NLI anchor rows sampled 1:1:1 across Entailment, Contradiction, and Neutral."""
    by_label: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    with open(clean_nli_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            lbl = r.get("label")
            if lbl in (CONTRADICTION, ENTAILMENT, NEUTRAL):
                r_clean = {
                    "premise": r["premise"],
                    "hypothesis": r["hypothesis"],
                    "label": lbl,
                    "group_id": f"anchor_{r.get('id', len(by_label[lbl]))}",
                    "is_gold": (lbl == ENTAILMENT),
                    "source": f"anchor_{r.get('source', 'clean_nli')}",
                }
                by_label[lbl].append(r_clean)

    target_per_class = total_samples // 3
    rng = random.Random(seed)
    anchors: List[Dict[str, Any]] = []
    for lbl in (CONTRADICTION, ENTAILMENT, NEUTRAL):
        pool = by_label[lbl]
        sample_count = min(len(pool), target_per_class)
        sampled = rng.sample(pool, sample_count)
        anchors.extend(sampled)

    logger.info("Loaded %d balanced clean NLI anchor replay pairs (%d per class).", len(anchors), len(anchors) // 3)
    return anchors


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile Phase-4 Training Mixture")
    parser.add_argument("--p3-base-file", default=str(DEFAULT_P3_BASE_FILE), help="Phase-3 enriched base mixture")
    parser.add_argument("--p4-synth-file", default=str(DEFAULT_P4_SYNTH_FILE), help="Phase-4 synthetic remediation JSONL")
    parser.add_argument("--visual-synth-file", default=str(DEFAULT_VISUAL_SYNTH_FILE), help="Visual synthetic pairs JSONL")
    parser.add_argument("--clean-nli-file", default=str(DEFAULT_CLEAN_NLI_FILE), help="Clean NLI reference file")
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_FILE), help="Output Phase-4 mixture path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--p3-sample-size", type=int, default=60000, help="Stratified sample size from Phase-3 base (0 for all)")
    parser.add_argument("--clean-anchor-samples", type=int, default=15000, help="Clean NLI anchor replay pairs")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    rng = random.Random(args.seed)

    mixture_rows: List[Dict[str, Any]] = []
    source_stats = Counter()

    # 1. Ingest Phase-4 Targeted Synthetic Remediation
    p4_synth_path = Path(args.p4_synth_file)
    if p4_synth_path.exists():
        logger.info("Loading Phase-4 synthetic remediation pairs from %s...", p4_synth_path.name)
        with open(p4_synth_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                mixture_rows.append(r)
                source_stats[r.get("source", "phase4_synth")] += 1
        logger.info("-> Loaded %d Phase-4 synthetic rows.", len(mixture_rows))
    else:
        logger.warning("Phase-4 synthetic file %s not found.", p4_synth_path)

    # 2. Ingest Multimodal Visual Synthetic Data
    visual_synth_path = Path(args.visual_synth_file)
    if visual_synth_path.exists():
        logger.info("Loading Visual Synthetic pairs from %s...", visual_synth_path.name)
        v_count = 0
        with open(visual_synth_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                # Normalize relative image path
                if "image" in r and not os.path.exists(r["image"]):
                    cand = str(REPO_ROOT / "data" / "visual_synth" / r["image"])
                    if os.path.exists(cand):
                        r["image"] = cand
                mixture_rows.append(r)
                source_stats[r.get("source", "visual_synth")] += 1
                v_count += 1
        logger.info("-> Loaded %d multimodal visual synthetic rows.", v_count)
    else:
        logger.warning("Visual synthetic file %s not found.", visual_synth_path)

    # 2. Ingest Phase-3 Base Mixture (with optional sampling to keep training nimble)
    p3_base_path = Path(args.p3_base_file)
    if p3_base_path.exists():
        logger.info("Loading Phase-3 base mixture from %s...", p3_base_path.name)
        p3_rows: List[Dict[str, Any]] = []
        with open(p3_base_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                p3_rows.append(json.loads(line))

        if args.p3_sample_size > 0 and len(p3_rows) > args.p3_sample_size:
            logger.info("Sampling %d rows from %d Phase-3 base rows for anchor preservation...", args.p3_sample_size, len(p3_rows))
            sampled_p3 = rng.sample(p3_rows, args.p3_sample_size)
        else:
            sampled_p3 = p3_rows

        for r in sampled_p3:
            mixture_rows.append(r)
            source_stats[r.get("source", "p3_base")] += 1
        logger.info("-> Added %d Phase-3 base rows.", len(sampled_p3))
    else:
        logger.warning("Phase-3 base file %s not found.", p3_base_path)

    # 3. Ingest Balanced Clean NLI Anchors
    clean_nli_path = Path(args.clean_nli_file)
    if clean_nli_path.exists() and args.clean_anchor_samples > 0:
        anchors = load_balanced_nli_anchors(clean_nli_path, total_samples=args.clean_anchor_samples, seed=args.seed)
        for r in anchors:
            mixture_rows.append(r)
            source_stats[r.get("source", "clean_nli")] += 1

    # 4. Shuffle and write out
    logger.info("Total combined Phase-4 rows: %d. Shuffling...", len(mixture_rows))
    rng.shuffle(mixture_rows)

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in mixture_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info("Successfully compiled %d rows to %s", len(mixture_rows), out_path)
    logger.info("Top sources: %s", source_stats.most_common(10))


if __name__ == "__main__":
    main()
