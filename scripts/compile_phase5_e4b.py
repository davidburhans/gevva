#!/usr/bin/env python3
"""compile_phase5_e4b.py - Compiles the Gevva 1.1 Phase 5 e4b continual mixture.

Evidence base (2026-09-28):
- EXP-01: anchor-heavy curriculum beat the pruned arm on every tier
  (§13.5); foundational anchors are load-bearing -> maximize anchor share.
- F-07 investigation: the e4b first round had ZERO adversarial-NLI rows,
  causing ANLI neutral-class collapse (2.9% recall) -> this mixture asserts
  a large ANLI R1-R3 + WANLI anchor slice at compile time.
- TR-20: deep-reasoning MC sources (LogiQA/ReClor/RACE/StrategyQA/LSAT-AR)
  included in full for the e4b 42-layer skew.

Composition (~130k rows):
  trivial anchors (all)         ~28k   21.5%
  adversarial anchors (ANLI+WANLI) ~40k   31%   <- the F-07 fix
  medium supplements            ~45k   34.5%
  hard supplements              ~17k   13%

Decontamination: pair-key dedup vs exp01_gate_eval / test_v2 / val /
JevBench items is enforced; label-0 downsampling keeps medium <= 45% c-label.

Usage:
    uv run python scripts/compile_phase5_e4b.py [--total 130000]
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
NEW_ANCHOR_PATHS = [
    REPO_ROOT / "data" / "anli_r2r3_train.jsonl",
    REPO_ROOT / "data" / "wanli_train.jsonl",
]
FORBIDDEN_PATHS = [
    REPO_ROOT / "data" / "exp01_gate_eval.jsonl",
    REPO_ROOT / "data" / "test_v2.jsonl",
    REPO_ROOT / "data" / "val.jsonl",
    REPO_ROOT / "data" / "test.jsonl",
]
OUT_PATH = REPO_ROOT / "data" / "train_phase5_e4b.jsonl"
MANIFEST_PATH = REPO_ROOT / "data" / "phase5_e4b_manifest.json"

TRIVIAL_SOURCES = {
    "anchor_snli_train", "snli_train", "anchor_mnli_train", "mnli_train",
    "anchor_scitail_train", "scitail_train", "anchor_qnli_train", "qnli_train",
    "anchor_xnli_es", "anchor_xnli_fr", "anchor_xnli_de", "anchor_xnli_zh",
    "anchor_xnli_ru", "anchor_xnli_ar", "anchor_xnli_sw", "anchor_xnli_vi",
    "anchor_xnli_hi", "xnli_es", "xnli_fr", "xnli_de", "xnli_zh", "xnli_ru",
    "xnli_ar", "xnli_sw", "xnli_vi", "xnli_hi",
}
R1_SOURCES = {"anchor_anli_train_r1", "anli_train_r1"}
MEDIUM_SOURCES = {
    "typed_decisions_synth_choice_grouped", "typed_decisions_synth_score_grouped",
    "typed_decisions_synth_noul_grouped", "typed_decisions_synth_choice_neg",
    "typed_decisions_synth_choice_gold", "typed_decisions_synth_noul",
    "typed_decisions_synth_score_alt", "typed_decisions_synth_score_gold",
    "typed_decisions_synth_noul_inverted",
    "prometheus_feedback", "helpsteer2_subtle_flaw",
    "synth_remed_routing", "synth_remed_tradeoff",
    "anchor_fever_train", "fever_train",
    "sdk_tool_routing", "sdk_rubric_grading", "sdk_cloze_reasoning",
    "sdk_search_reranking", "sdk_abstention_augmentation",
}
# TR-20 deep-reasoning MC sources: included in full (hard/medium boundary).
DEEP_REASONING_SOURCES = {
    "race_mc_grouped", "logiqa_mc_grouped", "strategy_qa_mc_grouped",
    "reclor_mc_grouped", "lsat_ar_mc_grouped",
}
HARD_SOURCES = {
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
    "teacher_enterprise_security", "teacher_financial_fraud",
    "teacher_database_replication", "teacher_cloud_kubernetes",
    "teacher_pharmacovigilance",
}

# Anchor sampling plan: keep R1 in full (hardest curated), cap R2/R3/WANLI.
ANCHOR_PLAN = {
    "anchor_anli_train_r1": None,        # all
    "anli_train_r1": None,               # all
    "anchor_anli_train_r2": 12000,
    "anchor_anli_train_r3": 12000,
    "anchor_wanli_train": 12000,
}
MEDIUM_LABEL0_CAP = 0.45
MIN_ANLI_ROWS = 25000
MIN_TRIVIAL_ROWS = 20000


def _pair_key(premise: str, hypothesis: str) -> str:
    return f"{premise.strip()[:512]}||{hypothesis.strip()[:512]}".lower()


def _forbidden_keys() -> Set[str]:
    keys: Set[str] = set()
    for path in FORBIDDEN_PATHS:
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                keys.add(_pair_key(row.get("premise", ""), row.get("hypothesis", "")))
    return keys


def _load_pools(forbidden: Set[str]) -> Dict[str, List[Dict]]:
    pools: Dict[str, List[Dict]] = {
        "trivial": [], "anchors_new": [], "r1": [],
        "medium": [], "deep": [], "hard": [],
    }
    seen: Set[str] = set()

    def _keep(row: Dict, bucket: str) -> None:
        key = _pair_key(row.get("premise", ""), row.get("hypothesis", ""))
        if key in seen or key in forbidden:
            return
        seen.add(key)
        clean = {
            "premise": row["premise"],
            "hypothesis": row["hypothesis"],
            "label": int(row["label"]),
            "source": str(row.get("source", "?")),
        }
        if row.get("group_id") is not None:
            clean["group_id"] = row["group_id"]
        pools[bucket].append(clean)

    for path in POOL_PATHS:
        with open(path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                source = str(row.get("source", "?"))
                if source in TRIVIAL_SOURCES:
                    _keep(row, "trivial")
                elif source in R1_SOURCES:
                    _keep(row, "r1")
                elif source in MEDIUM_SOURCES:
                    _keep(row, "medium")
                elif source in DEEP_REASONING_SOURCES:
                    _keep(row, "deep")
                elif source in HARD_SOURCES:
                    _keep(row, "hard")

    for path in NEW_ANCHOR_PATHS:
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                _keep(json.loads(line), "anchors_new")
    return pools


def _balanced_sample(rows: List[Dict], n: int, rng: random.Random) -> List[Dict]:
    """Sample n rows with per-class balance when the pool is imbalanced."""
    if n >= len(rows):
        return list(rows)
    by_label: Dict[int, List[Dict]] = {}
    for row in rows:
        by_label.setdefault(row["label"], []).append(row)
    per_class = {lab: min(len(v), n // 3 + 1) for lab, v in by_label.items()}
    chosen: List[Dict] = []
    for lab, v in by_label.items():
        chosen.extend(rng.sample(v, per_class[lab]))
    if len(chosen) < n:
        rest = [r for r in rows if r not in chosen]
        chosen.extend(rng.sample(rest, min(len(rest), n - len(chosen))))
    return chosen[:n]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total", type=int, default=130000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--anli-r2-cap", type=int, default=12000)
    parser.add_argument("--anli-r3-cap", type=int, default=12000)
    parser.add_argument("--wanli-cap", type=int, default=12000)
    parser.add_argument("--ragtruth", default=None, help="Optional RAGTruth NLI jsonl (label 0=hallucinated, 1=supported)")
    parser.add_argument("--avoid-file", default=None, help="Prior training jsonl; its pair keys are excluded from anchor/medium/hard pools (freshness)")
    parser.add_argument("--out", default=None)
    parser.add_argument("--manifest", default=None)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    anchor_plan = {
        "anchor_anli_train_r1": None,
        "anli_train_r1": None,
        "anchor_anli_train_r2": args.anli_r2_cap,
        "anchor_anli_train_r3": args.anli_r3_cap,
        "anchor_wanli_train": args.wanli_cap,
    }

    forbidden = _forbidden_keys()
    pools = _load_pools(forbidden)

    # Round-3 freshness: drop pair keys already trained on (adversarial/medium/hard only;
    # trivial anchors are intentionally re-seen as load-bearing replay).
    avoid: Set[str] = set()
    if args.avoid_file and Path(args.avoid_file).exists():
        with open(args.avoid_file, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                avoid.add(_pair_key(row.get("premise", ""), row.get("hypothesis", "")))
        pools["anchors_new"] = [r for r in pools["anchors_new"] if _pair_key(r["premise"], r["hypothesis"]) not in avoid]
        print(f"  freshness: avoided {len(avoid)} prior pair keys; anchors_new pool now {len(pools['anchors_new'])}")
    for name, rows in pools.items():
        print(f"  pool {name}: {len(rows)}")

    # --- RAGTruth (TR-06): high-overlap hallucination grounding ---
    ragtruth_rows: List[Dict] = []
    if args.ragtruth and Path(args.ragtruth).exists():
        import re
        with open(args.ragtruth, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                m = re.match(r"Premise: (.*?)Hypothesis: (.*)", row.get("text", ""), re.S)
                if not m:
                    continue
                lab = {0: 0, 1: 1}.get(row.get("label"))  # 0=hallucinated->contradiction, 1=supported->entailment
                if lab is None:
                    continue
                ragtruth_rows.append({
                    "premise": m.group(1).strip(), "hypothesis": m.group(2).strip(),
                    "label": lab, "source": "ragtruth_train",
                })
        print(f"  ragtruth: {len(ragtruth_rows)}")

    # --- F-07 assertions (fail loudly; this bug cost e4b its ANLI score) ---
    r1_all = pools["r1"]
    anchor_rows = list(r1_all)
    for source, cap in anchor_plan.items():
        rows = [r for r in pools["anchors_new"] if r["source"] == source]
        if cap is None or len(rows) <= cap:
            anchor_rows.extend(rows)
        else:
            anchor_rows.extend(rng.sample(rows, cap))
    anchor_rows = list({id(r): r for r in anchor_rows}.values())
    anli_total = sum(1 for r in anchor_rows if "anli" in r["source"] and "wanli" not in r["source"])
    assert anli_total >= MIN_ANLI_ROWS, (
        f"F-07 assertion failed: only {anli_total} true ANLI rows in mixture "
        f"(required >={MIN_ANLI_ROWS}); the first e4b round shipped with zero"
    )
    assert len(pools["trivial"]) >= MIN_TRIVIAL_ROWS, (
        f"EXP-01 lesson violated: trivial anchor pool {len(pools['trivial'])} < {MIN_TRIVIAL_ROWS}"
    )

    # --- medium with label-0 downsampling ---
    medium = pools["medium"]
    label0 = [r for r in medium if r["label"] == 0]
    others = [r for r in medium if r["label"] != 0]
    rng.shuffle(label0)
    max_l0 = int(MEDIUM_LABEL0_CAP / (1 - MEDIUM_LABEL0_CAP) * len(others))
    medium_balanced = others + label0[:max_l0]

    n_medium = int(0.345 * args.total)
    chosen_medium = _balanced_sample(medium_balanced, n_medium, rng)
    chosen_hard = _balanced_sample(pools["hard"] + pools["deep"], int(0.13 * args.total), rng)

    mixture = pools["trivial"] + anchor_rows + ragtruth_rows + chosen_medium + chosen_hard
    rng.shuffle(mixture)

    label_dist = Counter(r["label"] for r in mixture)
    source_counts = Counter(r["source"] for r in mixture)
    out_path = Path(args.out) if args.out else OUT_PATH
    manifest_path = Path(args.manifest) if args.manifest else MANIFEST_PATH
    with open(out_path, "w", encoding="utf-8") as f:
        for row in mixture:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "mixture": "phase5_e4b_continual",
        "rows": len(mixture),
        "label_dist": dict(sorted(label_dist.items())),
        "composition": {
            "trivial_anchors": len(pools["trivial"]),
            "adversarial_anchors": len(anchor_rows),
            "anli_rows": anli_total,
            "wanli_rows": sum(1 for r in anchor_rows if "wanli" in r["source"]),
            "ragtruth_rows": len(ragtruth_rows),
            "medium_supplements": len(chosen_medium),
            "hard_and_deep_reasoning": len(chosen_hard),
        },
        "assertions": {"anli_min": MIN_ANLI_ROWS, "trivial_min": MIN_TRIVIAL_ROWS},
        "evidence": [
            "EXP-01 §13.5: anchor-heavy beats pruned on all tiers",
            "F-07: first e4b round had zero ANLI rows -> neutral collapse",
            "TR-20: deep-reasoning MC included for 42-layer skew",
        ],
        "source_counts": dict(source_counts.most_common()),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(mixture)} rows -> {out_path}")
    print(f"Labels: {dict(sorted(label_dist.items()))}")
    print(f"Composition: {json.dumps(manifest['composition'], indent=2)}")


if __name__ == "__main__":
    main()
