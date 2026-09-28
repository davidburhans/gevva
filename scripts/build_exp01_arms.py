#!/usr/bin/env python3
"""build_exp01_arms.py - Builds the EXP-01 curriculum-pruning A/B training arms.

Pre-registered design (docs/IMPROVEMENT_OPPORTUNITIES.md §13):
    Arm A: 45% trivial / 35% medium / 20% hard   (standard balanced curriculum)
    Arm B: 10% trivial / 65% medium / 25% hard   (aggressively pruned mixture)

Adaptations recorded in the manifest (honest deviations from the §13 text):
- Trivial pool (~36k unique pairs) is smaller than 45k, so both arms are sized
  N = trivial_pool / 0.45 (identical step counts, identical hard share).
- Dataset substitution: the §13 example datasets not present on disk
  (BANKING77/BFCL/When2Call/SGD medium; HoVer/ContractNLI hard) are replaced by
  on-disk sources in the same difficulty tier (see BUCKETS below).
- Text-only: multimodal sources are excluded from both arms.

Decontamination: every arm row is deduped against the gate eval set
(data/exp01_gate_eval.jsonl), data/test_v2.jsonl and data/val.jsonl pair keys.

Usage:
    uv run python scripts/build_exp01_arms.py [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set

REPO_ROOT = Path(__file__).resolve().parent.parent
POOL_PATHS = [
    REPO_ROOT / "data" / "train_master_full.jsonl",
    REPO_ROOT / "data" / "hf_dataset_staging" / "data" / "flagship_train.jsonl",
]
GATE_EVAL_PATH = REPO_ROOT / "data" / "exp01_gate_eval.jsonl"
TEST_V2_PATH = REPO_ROOT / "data" / "test_v2.jsonl"
VAL_PATH = REPO_ROOT / "data" / "val.jsonl"
OUT_DIR = REPO_ROOT / "data"
MANIFEST_PATH = OUT_DIR / "exp01_arms_manifest.json"

TRIVIAL_SOURCES = {
    "anchor_snli_train", "snli_train", "anchor_mnli_train", "mnli_train",
    "anchor_scitail_train", "scitail_train", "anchor_qnli_train", "qnli_train",
    "anchor_xnli_es", "anchor_xnli_fr", "anchor_xnli_de", "anchor_xnli_zh",
    "anchor_xnli_ru", "anchor_xnli_ar", "anchor_xnli_sw", "anchor_xnli_vi",
    "anchor_xnli_hi", "xnli_es", "xnli_fr", "xnli_de", "xnli_zh", "xnli_ru",
    "xnli_ar", "xnli_sw", "xnli_vi", "xnli_hi",
}
MEDIUM_SOURCES = {
    "typed_decisions_synth_choice_grouped", "typed_decisions_synth_score_grouped",
    "typed_decisions_synth_noul_grouped", "typed_decisions_synth_choice_neg",
    "typed_decisions_synth_choice_gold", "typed_decisions_synth_noul",
    "typed_decisions_synth_score_alt", "typed_decisions_synth_score_gold",
    "typed_decisions_synth_noul_inverted",
    "prometheus_feedback", "helpsteer2_subtle_flaw",
    "synth_remed_routing", "synth_remed_tradeoff",
    "race_mc_grouped", "logiqa_mc_grouped", "strategy_qa_mc_grouped",
    "anchor_fever_train", "fever_train",
    "sdk_tool_routing", "sdk_rubric_grading", "sdk_cloze_reasoning",
    "sdk_search_reranking", "sdk_abstention_augmentation",
}
HARD_SOURCES = {
    "anchor_anli_train_r1", "anli_train_r1",
    "synth_remed_judge_hard", "synth_remed_policy",
    "synth_p4_ambiguous", "synth_p4_multi_hop", "synth_p4_long_policy",
    "synth_p4_temporal_numeric", "synth_p4_tradeoff",
    "synth_hard_temporal_numeric", "synth_hard_probability",
    "synth_hard_tradeoff", "synth_hard_multi_hop", "synth_hard_long_policy",
    "synth_hard_long_policy_multisection", "synth_hard_trap",
    "synth_hard_adversarial",
    "anchor_haystack_embedded", "anchor_haystack_drop_neutral",
    "anchor_haystack_corrupted_con",
    "haystack_embedded", "haystack_drop_neutral", "haystack_corrupted_con",
    "sdk_counterfactual_inversion", "sdk_rag_hallucination",
    "reclor_mc_grouped", "lsat_ar_mc_grouped",
    "teacher_enterprise_security", "teacher_financial_fraud",
    "teacher_database_replication", "teacher_cloud_kubernetes",
    "teacher_pharmacovigilance",
}

ARM_A_RATIOS = {"trivial": 0.45, "medium": 0.35, "hard": 0.20}
ARM_B_RATIOS = {"trivial": 0.10, "medium": 0.65, "hard": 0.25}


def _pair_key(premise: str, hypothesis: str) -> str:
    return f"{premise.strip()[:512]}||{hypothesis.strip()[:512]}".lower()


def _load_pools() -> Dict[str, Dict[str, List[Dict]]]:
    """Union-dedup the pool files into buckets: {bucket: {source: [rows]}}."""
    seen: Set[str] = set()
    buckets: Dict[str, Dict[str, List[Dict]]] = {
        "trivial": {}, "medium": {}, "hard": {},
    }
    unmapped = Counter()
    for path in POOL_PATHS:
        with open(path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                source = str(row.get("source", "?"))
                if source in TRIVIAL_SOURCES:
                    bucket = "trivial"
                elif source in MEDIUM_SOURCES:
                    bucket = "medium"
                elif source in HARD_SOURCES:
                    bucket = "hard"
                else:
                    unmapped[source] += 1
                    continue
                key = _pair_key(row.get("premise", ""), row.get("hypothesis", ""))
                if key in seen:
                    continue
                seen.add(key)
                clean = {
                    "premise": row["premise"],
                    "hypothesis": row["hypothesis"],
                    "label": int(row["label"]),
                    "source": source,
                }
                if row.get("group_id") is not None:
                    clean["group_id"] = row["group_id"]
                buckets[bucket].setdefault(source, []).append(clean)
    if unmapped:
        print(f"  (excluded unmapped sources: {dict(unmapped.most_common(6))} ...)")
    return buckets


def _forbidden_keys() -> Set[str]:
    keys: Set[str] = set()
    for path in (GATE_EVAL_PATH, TEST_V2_PATH, VAL_PATH):
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                keys.add(_pair_key(row.get("premise", ""), row.get("hypothesis", "")))
    return keys


def _flatten(bucket: Dict[str, List[Dict]], forbidden: Set[str]) -> List[Dict]:
    rows = [row for rows in bucket.values() for row in rows]
    clean = [r for r in rows if _pair_key(r["premise"], r["hypothesis"]) not in forbidden]
    return clean


def _sample(rows: List[Dict], n: int, rng: random.Random) -> List[Dict]:
    if n >= len(rows):
        return list(rows)
    return rng.sample(rows, n)


def _write_rows(path: Path, rows: List[Dict], rng: random.Random) -> None:
    rng.shuffle(rows)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  {path.name}: {len(rows)} rows")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    print("Loading pools...")
    buckets = _load_pools()
    forbidden = _forbidden_keys()
    pools = {name: _flatten(bucket, forbidden) for name, bucket in buckets.items()}
    for name, rows in pools.items():
        label_dist = Counter(r["label"] for r in rows)
        print(f"  {name} pool: {len(rows)} rows (labels {dict(sorted(label_dist.items()))})")

    trivial_pool = len(pools["trivial"])
    n_total = int(trivial_pool / ARM_A_RATIOS["trivial"] / 1000) * 1000
    print(f"Sizing: trivial pool {trivial_pool} -> N={n_total} per arm")

    arm_a: List[Dict] = []
    arm_a += _sample(pools["trivial"], int(n_total * ARM_A_RATIOS["trivial"]), rng)
    arm_a += _sample(pools["medium"], int(n_total * ARM_A_RATIOS["medium"]), rng)
    arm_a += _sample(pools["hard"], int(n_total * ARM_A_RATIOS["hard"]), rng)

    arm_b: List[Dict] = []
    arm_b += _sample(pools["trivial"], int(n_total * ARM_B_RATIOS["trivial"]), rng)
    arm_b += _sample(pools["medium"], int(n_total * ARM_B_RATIOS["medium"]), rng)
    arm_b += _sample(pools["hard"], int(n_total * ARM_B_RATIOS["hard"]), rng)

    print("Writing arms...")
    _write_rows(OUT_DIR / "exp01_armA_train.jsonl", arm_a, rng)
    _write_rows(OUT_DIR / "exp01_armB_train.jsonl", arm_b, rng)

    manifest = {
        "experiment": "EXP-01 curriculum pruning A/B",
        "seed": args.seed,
        "n_per_arm": n_total,
        "armA_ratios": ARM_A_RATIOS,
        "armB_ratios": ARM_B_RATIOS,
        "pool_sizes": {k: len(v) for k, v in pools.items()},
        "decontamination": {
            "against": ["exp01_gate_eval.jsonl", "test_v2.jsonl", "val.jsonl"],
            "method": "normalized (premise[:512]||hypothesis[:512]).lower() pair keys",
        },
        "deviations_from_section13": [
            "N sized to trivial pool / 0.45 (both arms identical step count)",
            "on-disk sources substituted for BANKING77/BFCL/When2Call/SGD/HoVer/ContractNLI",
            "multimodal sources excluded (text-only arms)",
        ],
        "armA_source_counts": dict(Counter(r["source"] for r in arm_a).most_common()),
        "armB_source_counts": dict(Counter(r["source"] for r in arm_b).most_common()),
    }
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
