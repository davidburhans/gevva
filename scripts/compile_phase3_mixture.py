#!/usr/bin/env python3
"""scripts/compile_phase3_mixture.py - Compiles Phase-3 Rubric & Judge Training Mixture.

Recipe for Phase-3 (Rubric Grounding & Subtle-Flaw Remediation):
1. Ingests Phase-2 reasoning base (or high-value subset) to preserve formal logic gains.
2. Ingests Prometheus Feedback Collection (rubric-grounded grading for `judge_hard` & `adequacy`).
3. Ingests NVIDIA HelpSteer2 subtle-flaw contrastives (eliminates Mode B polished-misinformation traps).
4. Ingests scaled Clean NLI Anchors (1:1:1 stratified) to prevent neutral boundary drift.
5. Applies group atomicity, distractor balancing, and hardware safety caps (<= 8,192 tokens/group).

Outputs:
- data/train_phase3_mixture.jsonl
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

DEFAULT_P2_BASE_FILE = REPO_ROOT / "data" / "train_phase2_mixture.jsonl"
DEFAULT_PROMETHEUS_FILE = REPO_ROOT / "data" / "staged" / "phase3" / "prometheus_feedback_train.jsonl"
DEFAULT_HELPSTEER2_FILE = REPO_ROOT / "data" / "staged" / "phase3" / "helpsteer2_train.jsonl"
DEFAULT_CLEAN_NLI_FILE = REPO_ROOT / "data" / "train.jsonl"
DEFAULT_OUT_FILE = REPO_ROOT / "data" / "train_phase3_mixture.jsonl"


def load_grouped_file(
    path: Path,
    max_distractors_per_group: Optional[int] = 1,
    max_group_chars: int = 32768,
    rng: Optional[random.Random] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Loads a JSONL file, groups rows by group_id, and balances gold:distractor ratio."""
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
        total_chars = sum(len(r.get("premise", "")) + len(r.get("hypothesis", "")) for r in rows)
        if total_chars > max_group_chars:
            continue

        if max_distractors_per_group is None:
            balanced_groups[gid] = rows
            continue

        golds = [r for r in rows if r.get("is_gold", False)]
        alts = [r for r in rows if not r.get("is_gold", False)]

        if not golds:
            balanced_groups[gid] = rows
            continue

        sampled_alts = rng.sample(alts, min(len(alts), max_distractors_per_group)) if alts else []
        balanced_groups[gid] = golds + sampled_alts

    return balanced_groups


def load_balanced_nli_anchors(
    clean_nli_path: Path,
    total_samples: int = 20000,
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

    logger.info(f"Loaded {len(anchors):,} balanced clean NLI anchor replay pairs ({len(anchors)//3:,} per class).")
    return anchors


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile Phase-3 Rubric & Judge Training Mixture")
    parser.add_argument("--base-file", default=str(DEFAULT_P2_BASE_FILE), help="Phase-2 base training mixture")
    parser.add_argument("--prometheus-file", default=str(DEFAULT_PROMETHEUS_FILE), help="Staged Prometheus feedback JSONL")
    parser.add_argument("--helpsteer2-file", default=str(DEFAULT_HELPSTEER2_FILE), help="Staged HelpSteer2 subtle-flaw JSONL")
    parser.add_argument("--clean-nli-file", default=str(DEFAULT_CLEAN_NLI_FILE), help="Clean NLI reference file")
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_FILE), help="Output Phase-3 mixture path")
    parser.add_argument("--seed", type=int, default=42)

    # Caps for dynamic tuning post-Phase 2
    parser.add_argument("--cap-prometheus", type=int, default=25000, help="Max Prometheus pairs to ingest")
    parser.add_argument("--cap-helpsteer2", type=int, default=15000, help="Max HelpSteer2 pairs to ingest")
    parser.add_argument("--clean-anchor-samples", type=int, default=20000, help="Clean NLI 1:1:1 anchor replay pairs")
    parser.add_argument("--remediation-file", default=str(REPO_ROOT / "data" / "staged" / "synth_targeted_remediation.jsonl"), help="Targeted remediation pairs")
    parser.add_argument("--max-distractors", type=int, default=1, help="Max distractors per group (1 = 1:1 gold:alt ratio)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    rng = random.Random(args.seed)
    mixture_rows: List[Dict[str, Any]] = []
    source_stats = Counter()

    # 1. Ingest Phase-2 Base Mixture
    base_path = Path(args.base_file)
    if base_path.exists():
        logger.info(f"Loading Phase-2 base mixture from {base_path}...")
        with open(base_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                mixture_rows.append(r)
                source_stats[r.get("source", "p2_base")] += 1
        logger.info(f"-> Loaded {len(mixture_rows):,} base Phase-2 rows.")
    else:
        logger.warning(f"Phase-2 base file {base_path} not found; building from Phase-3 staged sets only.")

    # 2. Ingest Prometheus Feedback Collection (Rubric Grounding)
    prom_path = Path(args.prometheus_file)
    if prom_path.exists() and args.cap_prometheus > 0:
        logger.info(f"Loading Prometheus Feedback from {prom_path.name} (cap={args.cap_prometheus})...")
        prom_rows: List[Dict[str, Any]] = []
        with open(prom_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                prom_rows.append(json.loads(line))
        if args.cap_prometheus < len(prom_rows):
            rng.shuffle(prom_rows)
            prom_rows = prom_rows[: args.cap_prometheus]
        mixture_rows.extend(prom_rows)
        source_stats["prometheus_feedback"] += len(prom_rows)
        logger.info(f"-> Added {len(prom_rows):,} Prometheus rubric pairs.")

    # 3. Ingest HelpSteer2 Subtle Flaws
    hs_path = Path(args.helpsteer2_file)
    if hs_path.exists() and args.cap_helpsteer2 > 0:
        logger.info(f"Loading HelpSteer2 subtle flaws from {hs_path.name} (cap={args.cap_helpsteer2})...")
        hs_rows: List[Dict[str, Any]] = []
        with open(hs_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                hs_rows.append(json.loads(line))
        if args.cap_helpsteer2 < len(hs_rows):
            rng.shuffle(hs_rows)
            hs_rows = hs_rows[: args.cap_helpsteer2]
        mixture_rows.extend(hs_rows)
        source_stats["helpsteer2_subtle_flaws"] += len(hs_rows)
        logger.info(f"-> Added {len(hs_rows):,} HelpSteer2 subtle-flaw pairs.")

    # 4. Ingest Scaled Clean NLI Anchor Replay
    if args.clean_anchor_samples > 0:
        anchors = load_balanced_nli_anchors(Path(args.clean_nli_file), total_samples=args.clean_anchor_samples, seed=args.seed)
        mixture_rows.extend(anchors)
        for a in anchors:
            source_stats[a.get("source", "anchor_nli")] += 1

    # 5. Ingest Targeted Weak-Family Remediation Pairs
    remed_path = Path(args.remediation_file) if args.remediation_file else None
    if remed_path and remed_path.exists():
        logger.info(f"Loading targeted remediation pairs from {remed_path.name}...")
        remed_rows: List[Dict[str, Any]] = []
        with open(remed_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                remed_rows.append(json.loads(line))
        mixture_rows.extend(remed_rows)
        for r in remed_rows:
            source_stats[r.get("source", "synth_remed")] += 1
        logger.info(f"-> Added {len(remed_rows):,} targeted remediation pairs.")

    # 6. Summary & Verification
    logger.info(f"\n==================== Phase-3 Master Mixture Summary ====================")
    logger.info(f"Total Mixture Pairs: {len(mixture_rows):,}")

    label_counts = Counter(r["label"] for r in mixture_rows)
    for lbl in (CONTRADICTION, ENTAILMENT, NEUTRAL):
        cnt = label_counts.get(lbl, 0)
        pct = cnt / len(mixture_rows) * 100 if mixture_rows else 0.0
        logger.info(f"  [{lbl}] {ID2LABEL[lbl]:13s}: {cnt:>8,} ({pct:5.1f}%)")

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
