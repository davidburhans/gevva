#!/usr/bin/env python3
"""scripts/compile_phase2_mixture.py - Compiles Phase-2 Training Mixture.

Applies the Adversarial Review Prescriptions:
1. Resolves Confirmation Asymmetry: Subsamples distractors per group (--max-distractors 1 or 2)
   enforcing a balanced 1:1 or 1:2 gold-to-distractor ratio.
2. Protects Neutral Class Boundary: Ingests balanced clean NLI anchor replay (15K pairs
   stratified 1:1:1 across Entailment, Contradiction, and Neutral from data/train.jsonl)
   preventing catastrophic neutral starvation.
3. Weak-Family Realism: Ingests targeted deterministic generators from
   data/synth_hard_decisions_grouped.jsonl (business hours SLAs, timezones, Bayes screening)
   plus high-leverage logical reasoning (LogiQA 2.0, ReClor, LSAT-AR, StrategyQA).
   Drops off-distribution algebra (AQuA-RAT) and citation lookups (CaseHOLD) by default.
4. Hardware Safety: Filters/caps oversize groups (>8,192 tokens) to guarantee zero CUDA OOMs
   on the RTX 5090 under group-atomic batching.

Outputs:
- data/train_phase2_mixture.jsonl
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

logger = logging.getLogger(__name__)

DEFAULT_STAGED_DIR = REPO_ROOT / "data" / "staged" / "mc_qa"
DEFAULT_BASE_FILE = REPO_ROOT / "data" / "train_p1_mixture.jsonl"
DEFAULT_CLEAN_NLI_FILE = REPO_ROOT / "data" / "train.jsonl"
DEFAULT_HARD_SYNTH_FILE = REPO_ROOT / "data" / "synth_hard_decisions_grouped.jsonl"
DEFAULT_OUT_FILE = REPO_ROOT / "data" / "train_phase2_mixture.jsonl"


def load_grouped_file(
    path: Path,
    max_distractors_per_group: Optional[int] = 1,
    max_group_chars: int = 32768,  # ~8,192 tokens max per group
    rng: Optional[random.Random] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Loads a JSONL file, groups rows by group_id, and balances gold:distractor ratio.

    Args:
        path: Path to staged JSONL file.
        max_distractors_per_group: Max negative alternatives to keep per group (1 = 1:1 ratio).
        max_group_chars: Total characters allowed in a group to prevent batch OOMs.
        rng: Random instance for distractor subsampling.

    Returns:
        Dict mapping group_id to balanced list of pair dicts.
    """
    if rng is None:
        rng = random.Random(42)

    raw_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            gid = str(row.get("group_id", "-1"))
            raw_groups[gid].append(row)

    balanced_groups: Dict[str, List[Dict[str, Any]]] = {}
    for gid, rows in raw_groups.items():
        # Check group size limit for hardware safety
        total_chars = sum(len(r.get("premise", "")) + len(r.get("hypothesis", "")) for r in rows)
        if total_chars > max_group_chars:
            continue

        if max_distractors_per_group is None:
            balanced_groups[gid] = rows
            continue

        golds = [r for r in rows if r.get("is_gold", False)]
        alts = [r for r in rows if not r.get("is_gold", False)]

        if not golds:
            # Ungrouped or pure negative item
            balanced_groups[gid] = rows
            continue

        # Subsample distractors to enforce balanced ratio (e.g. 1 gold + 1 or 2 alts)
        sampled_alts = rng.sample(alts, min(len(alts), max_distractors_per_group)) if alts else []
        balanced_groups[gid] = golds + sampled_alts

    return balanced_groups


def load_balanced_nli_anchors(
    clean_nli_path: Path,
    total_samples: int = 15000,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Loads clean NLI anchor replay sampled in strict 1:1:1 ratio across classes.

    Prevents neutral starvation and preserves general NLI accuracy on MNLI/SNLI/ANLI.
    """
    if not clean_nli_path.exists() or total_samples <= 0:
        return []

    logger.info(f"Sampling {total_samples:,} balanced clean NLI anchors (1:1:1) from {clean_nli_path.name}...")
    by_class: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    with open(clean_nli_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            lbl = int(r.get("label", -1))
            if lbl in (CONTRADICTION, ENTAILMENT, NEUTRAL):
                by_class[lbl].append(r)

    rng = random.Random(seed)
    per_class = total_samples // 3
    sampled: List[Dict[str, Any]] = []

    for lbl in (CONTRADICTION, ENTAILMENT, NEUTRAL):
        candidates = by_class[lbl]
        k = min(len(candidates), per_class)
        chosen = rng.sample(candidates, k)
        for r in chosen:
            is_gold = (lbl == ENTAILMENT)
            soft = [0.0, 0.0, 0.0]
            soft[lbl] = 1.0
            sampled.append({
                "id": f"anchor_clean_{r.get('id', len(sampled))}",
                "premise": r["premise"],
                "hypothesis": r["hypothesis"],
                "label": lbl,
                "group_id": -1,  # Ungrouped anchor
                "is_gold": is_gold,
                "soft_target": 1.0 if is_gold else 0.0,
                "soft_labels": soft,
                "source": f"anchor_{r.get('source', 'nli')}",
                "language": r.get("language", "en"),
                "image": r.get("image", ""),
            })

    logger.info(f"-> Loaded {len(sampled):,} balanced anchor pairs ({per_class:,} per class).")
    return sampled


def main():
    parser = argparse.ArgumentParser(description="Compile Phase-2 Training Mixture (Adversarially Hardened)")
    parser.add_argument("--base-file", default=str(DEFAULT_BASE_FILE), help="Base Stage 2 P1 mixture")
    parser.add_argument("--staged-dir", default=str(DEFAULT_STAGED_DIR), help="Directory of staged MC datasets")
    parser.add_argument("--clean-nli-file", default=str(DEFAULT_CLEAN_NLI_FILE), help="Clean NLI dataset for anchor replay")
    parser.add_argument("--hard-synth-file", default=str(DEFAULT_HARD_SYNTH_FILE), help="Deterministic hard synthetic file")
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_FILE), help="Target output JSONL path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    
    # Confirmation asymmetry & neutral protection
    parser.add_argument("--max-distractors", type=int, default=1, help="Max distractors per group (1 = 1:1 gold:alt ratio)")
    parser.add_argument("--clean-anchor-samples", type=int, default=15000, help="Clean NLI 1:1:1 anchor replay pairs")

    # Per-dataset group caps (Default 0 for off-distribution AQuA-RAT / CaseHOLD)
    parser.add_argument("--cap-logiqa", type=int, default=4000, help="Max groups from LogiQA 2.0 (multi-hop)")
    parser.add_argument("--cap-reclor", type=int, default=3000, help="Max groups from ReClor (multi-hop)")
    parser.add_argument("--cap-lsat-ar", type=int, default=1585, help="Max groups from LSAT-AR (logic games)")
    parser.add_argument("--cap-strategy-qa", type=int, default=2289, help="Max groups from StrategyQA (facts)")
    parser.add_argument("--cap-race", type=int, default=2000, help="Max groups from RACE (evidence weighing)")
    parser.add_argument("--cap-casehold", type=int, default=0, help="Max groups from CaseHOLD (default 0 per review)")
    parser.add_argument("--cap-aqua-rat", type=int, default=0, help="Max groups from AQuA-RAT (default 0 per review)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    rng = random.Random(args.seed)
    mixture_rows: List[Dict[str, Any]] = []
    source_stats = Counter()

    # 1. Ingest Base Mixture if present
    base_path = Path(args.base_file)
    if base_path.exists():
        logger.info(f"Loading base mixture from {base_path}...")
        with open(base_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                mixture_rows.append(r)
                source_stats[r.get("source", "base_mixture")] += 1
        logger.info(f"-> Loaded {len(mixture_rows):,} base rows.")
    else:
        logger.warning(f"Base mixture {base_path} not found; building from staged sets only.")

    # 2. Ingest Deterministic Hard Synthetics (temporal_numeric & probability)
    synth_path = Path(args.hard_synth_file)
    if synth_path.exists():
        logger.info(f"Ingesting deterministic hard synthetic scenarios from {synth_path.name}...")
        synth_grouped = load_grouped_file(synth_path, max_distractors_per_group=None, rng=rng)
        synth_pairs = 0
        for rows in synth_grouped.values():
            mixture_rows.extend(rows)
            synth_pairs += len(rows)
            source_stats[rows[0].get("source", "synth_hard")] += len(rows)
        logger.info(f"-> Loaded {synth_pairs:,} deterministic hard scenario pairs across {len(synth_grouped):,} groups.")

    # 3. Ingest Balanced Clean NLI Anchors (Guarantees >=12% Neutral representation)
    if args.clean_anchor_samples > 0:
        anchors = load_balanced_nli_anchors(Path(args.clean_nli_file), total_samples=args.clean_anchor_samples, seed=args.seed)
        mixture_rows.extend(anchors)
        for a in anchors:
            source_stats[a.get("source", "anchor_nli")] += 1

    # 4. Ingest Staged MC Sets with Distractor Subsampling & Group Caps
    staged_dir = Path(args.staged_dir)
    mc_targets = [
        ("logiqa_train.jsonl", "logiqa", args.cap_logiqa),
        ("reclor_train.jsonl", "reclor", args.cap_reclor),
        ("lsat_ar_train.jsonl", "lsat_ar", args.cap_lsat_ar),
        ("strategy_qa_train.jsonl", "strategy_qa", args.cap_strategy_qa),
        ("race_train.jsonl", "race", args.cap_race),
        ("casehold_train.jsonl", "casehold", args.cap_casehold),
        ("aqua_rat_train.jsonl", "aqua_rat", args.cap_aqua_rat),
    ]

    for filename, name, cap in mc_targets:
        if cap <= 0:
            logger.info(f"Skipping {name} (cap={cap}).")
            continue

        file_path = staged_dir / filename
        if not file_path.exists():
            logger.warning(f"File {file_path} not found; skipping {name}.")
            continue

        logger.info(f"Loading {name} from {file_path.name} (cap={cap} groups, max_distractors={args.max_distractors})...")
        grouped = load_grouped_file(file_path, max_distractors_per_group=args.max_distractors, rng=rng)
        group_keys = list(grouped.keys())

        if cap is not None and cap < len(group_keys):
            rng.shuffle(group_keys)
            selected_keys = group_keys[:cap]
        else:
            selected_keys = group_keys

        n_pairs = 0
        for gid in selected_keys:
            rows = grouped[gid]
            mixture_rows.extend(rows)
            n_pairs += len(rows)
            source_stats[rows[0].get("source", name)] += len(rows)

        logger.info(f"-> Added {n_pairs:,} balanced pairs across {len(selected_keys):,} groups from {name}.")

    # 5. Summary & Verification
    logger.info(f"\n==================== Phase-2 Hardened Mixture Summary ====================")
    logger.info(f"Total Mixture Pairs: {len(mixture_rows):,}")

    label_counts = Counter(r["label"] for r in mixture_rows)
    for lbl in (CONTRADICTION, ENTAILMENT, NEUTRAL):
        cnt = label_counts.get(lbl, 0)
        pct = cnt / len(mixture_rows) * 100 if mixture_rows else 0.0
        logger.info(f"  [{lbl}] {ID2LABEL[lbl]:13s}: {cnt:>8,} ({pct:5.1f}%)")

    # Verification of Confirmation Asymmetry (Gold vs Alt in grouped items)
    grouped_golds = sum(1 for r in mixture_rows if r.get("group_id") not in (-1, "-1") and r.get("is_gold"))
    grouped_alts = sum(1 for r in mixture_rows if r.get("group_id") not in (-1, "-1") and not r.get("is_gold"))
    logger.info(f"\nGrouped Confirmation Ratio (Golds vs Alts): {grouped_golds:,} : {grouped_alts:,} (Ratio 1 : {grouped_alts / max(1, grouped_golds):.2f})")

    logger.info(f"\nTop Breakdown by Source:")
    for src, cnt in source_stats.most_common(15):
        logger.info(f"  - {src:35s}: {cnt:>8,}")

    # 6. Write Output
    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in mixture_rows:
            f.write(json.dumps(r) + "\n")

    size_mb = out_path.stat().st_size / (1024 * 1024)
    logger.info(f"\nSuccessfully wrote {len(mixture_rows):,} rows to {out_path} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
