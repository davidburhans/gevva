#!/usr/bin/env python3
"""scripts/compile_e4b_overnight_mixture.py - Compiles flagship multimodal & deep reasoning training mixture for Gemma 4 E4B.

Comprehensive Overnight Curriculum (~110,000 rows):
1. Multimodal Multi-Image Reasoning:
   - 2,400 multi-image before/after pairs (3x oversampled -> 7,200 rows)
   - 2,872 single-image visual reasoning pairs
   -> Total Multimodal: 10,072 rows (9.1%)
2. Balanced Foundational NLI Anchors:
   - 36,000 clean NLI rows (12,000 per class: contradiction, entailment, neutral, group_id: -1) (32.7%)
3. Rubric Evaluation & Response Grading:
   - Prometheus rubric evaluation & HelpSteer fine-grained critiques: 16,000 rows (14.5%)
4. Long-Context & Multi-Hop Reasoning:
   - RACE, LogiQA, ReClor, StrategyQA: 18,000 rows (16.4%)
5. Enterprise Decisions & Weak-Family Remediations:
   - Phase 4 remediated decision groups (tradeoff, policy, temporal_numeric, routing, judge_hard) +
     typed decisions (choice, score, noul): 30,000 rows (27.3%)

Total: ~110,000 rows, strictly group-atomic, decontaminated against data/test.jsonl.
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

from nli_labels import CONTRADICTION, ENTAILMENT, NEUTRAL, ID2LABEL

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
    parser = argparse.ArgumentParser(description="Compile Comprehensive E4B Flagship Training Mixture")
    parser.add_argument("--out-file", default="data/train_e4b_overnight.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    test_pairs = load_test_pairs(REPO_ROOT / "data" / "test.jsonl")
    logger.info("Loaded %d test pairs for strict decontamination.", len(test_pairs))

    mixture_rows: List[Dict[str, Any]] = []
    source_stats = Counter()

    # ---------------------------------------------------------
    # 1. Multimodal Multi-Image (3x oversampled = 7,200 rows)
    # ---------------------------------------------------------
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
                mixture_rows.append(r_copy)
                source_stats["multimodal_multi_image"] += 1
        logger.info("-> Added %d multi-image rows (3x oversampled).", len(loaded_multi) * 3)

    # ---------------------------------------------------------
    # 2. Single-Image Visual Reasoning (2,872 rows)
    # ---------------------------------------------------------
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
                mixture_rows.append(r)
                source_stats["visual_single_image"] += 1
                vis_count += 1
        logger.info("-> Added %d single-image visual reasoning rows.", vis_count)

    # ---------------------------------------------------------
    # 3. Balanced Foundational NLI Anchors (36,000 rows, 12k per class, group_id: -1)
    # ---------------------------------------------------------
    anchor_candidates = defaultdict(list)
    seen_anchors = set()

    for p in [REPO_ROOT / "data" / "train.jsonl", REPO_ROOT / "data" / "train_phase3_enriched.jsonl"]:
        if not p.exists():
            continue
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("group_id") not in (None, -1):
                    continue
                lbl = r.get("label")
                if lbl not in (CONTRADICTION, ENTAILMENT, NEUTRAL):
                    continue
                key = (r["premise"].strip(), r["hypothesis"].strip())
                if key in test_pairs or key in seen_anchors:
                    continue
                seen_anchors.add(key)
                anchor_candidates[lbl].append(r)

    target_per_lbl = 12000
    total_anchors = 0
    for lbl in (CONTRADICTION, ENTAILMENT, NEUTRAL):
        pool = anchor_candidates[lbl]
        sampled = rng.sample(pool, min(len(pool), target_per_lbl))
        for r in sampled:
            mixture_rows.append({
                "premise": r["premise"],
                "hypothesis": r["hypothesis"],
                "label": lbl,
                "group_id": -1,
                "is_gold": (lbl == ENTAILMENT),
                "source": "anchor_foundational_nli",
            })
            total_anchors += 1
            source_stats["anchor_foundational_nli"] += 1
    logger.info("-> Added %d balanced foundational NLI anchor rows (12k/class, group_id: -1).", total_anchors)

    # ---------------------------------------------------------
    # 4. Enterprise Decisions & Weak Families (from Phase 4, group-atomic ~30,000 rows)
    # ---------------------------------------------------------
    p4_p = REPO_ROOT / "data" / "train_phase4_mixture.jsonl"
    if p4_p.exists():
        p4_groups = defaultdict(list)
        with open(p4_p, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                key = (r.get("premise", "").strip(), r.get("hypothesis", "").strip())
                if key in test_pairs:
                    continue
                gid = r.get("group_id")
                if gid not in (None, -1):
                    p4_groups[gid].append(r)
        
        all_gids = list(p4_groups.keys())
        rng.shuffle(all_gids)
        
        p4_selected = 0
        target_p4 = 30000
        for gid in all_gids:
            group_rows = p4_groups[gid]
            if p4_selected + len(group_rows) > target_p4:
                break
            for r in group_rows:
                r_copy = dict(r)
                src = r_copy.get("source", "phase4_decision")
                mixture_rows.append(r_copy)
                source_stats[f"p4_{src}"] += 1
                p4_selected += 1
        logger.info("-> Added %d Phase 4 grouped decision rows across %d groups.", p4_selected, len(p4_groups))

    # ---------------------------------------------------------
    # 5. Deep Reasoning & Long Context (from Phase 3 enriched, group-atomic ~34,000 rows)
    # ---------------------------------------------------------
    p3_p = REPO_ROOT / "data" / "train_phase3_enriched.jsonl"
    if p3_p.exists():
        eval_groups = defaultdict(list)
        reason_groups = defaultdict(list)
        
        eval_sources = {"prometheus_feedback", "helpsteer2_subtle_flaw"}
        reason_sources = {"race_mc_grouped", "strategy_qa_mc_grouped", "logiqa_mc_grouped", "reclor_mc_grouped"}
        
        with open(p3_p, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                key = (r.get("premise", "").strip(), r.get("hypothesis", "").strip())
                if key in test_pairs:
                    continue
                src = r.get("source", "")
                gid = r.get("group_id")
                if not gid or gid == -1:
                    continue
                
                if src in eval_sources:
                    eval_groups[gid].append(r)
                elif src in reason_sources:
                    reason_groups[gid].append(r)

        # Sample evaluation groups (~16,000 rows)
        eval_gids = list(eval_groups.keys())
        rng.shuffle(eval_gids)
        eval_selected = 0
        for gid in eval_gids:
            group_rows = eval_groups[gid]
            if eval_selected + len(group_rows) > 16000:
                break
            for r in group_rows:
                mixture_rows.append(dict(r))
                source_stats[f"eval_{r.get('source')}"] += 1
                eval_selected += 1

        # Sample reasoning / long context groups (~18,000 rows)
        reason_gids = list(reason_groups.keys())
        rng.shuffle(reason_gids)
        reason_selected = 0
        for gid in reason_gids:
            group_rows = reason_groups[gid]
            if reason_selected + len(group_rows) > 18000:
                break
            for r in group_rows:
                mixture_rows.append(dict(r))
                source_stats[f"reason_{r.get('source')}"] += 1
                reason_selected += 1

        logger.info("-> Added %d evaluation/grading rows and %d long-context reasoning rows from Phase 3.", eval_selected, reason_selected)

    # Shuffle the final mixture
    rng.shuffle(mixture_rows)

    out_p = REPO_ROOT / args.out_file
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        for r in mixture_rows:
            f.write(json.dumps(r) + "\n")

    logger.info("==================================================")
    logger.info("COMPILED COMPREHENSIVE E4B FLAGSHIP MIXTURE: %s", out_p.name)
    logger.info("Total Rows: %d", len(mixture_rows))
    for src, count in sorted(source_stats.items(), key=lambda kv: -kv[1]):
        logger.info("  %-40s: %6d (%5.1f%%)", src, count, 100.0 * count / len(mixture_rows))
    logger.info("==================================================")


if __name__ == "__main__":
    main()
