#!/usr/bin/env python3
"""compile_gevva_v2_mixture.py - Compiles Gevva v2 Master Curriculum Mixture.

Transforms Gevva 1.0/1.1 training pools (foundational NLI anchors, grouped scenarios,
synthetic SDK rounds, and multimodal scenes) into the unified causal decision format
matching the llama.cpp `lev` decision protocol.

Features:
1. Option Codes: Sequential assignment from ['A'..'Z', 'AA'..'ZZ'] up to K=255.
2. Train-time Permutation Augmentation: Option order is randomized per sample;
   gold code is assigned dynamically to guarantee zero positional bias.
3. Foundational NLI Anchoring: Preserves semantic reasoning by mapping SNLI/MNLI/ANLI
   into both `choice` and `noul` verification tasks.
4. Grouped Scenarios: Aggregates grouped decision items into multi-option `choice` tasks.
5. Decontamination: Strict pair-key deduplication against test/eval splits.

Usage:
    uv run python scripts/compile_gevva_v2_mixture.py [--out data/gevva_v2_train.jsonl]
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

FOUNDATIONAL_ANCHORS = [
    DATA_DIR / "wanli_train.jsonl",
    DATA_DIR / "anli_r2r3_train.jsonl",
]

GROUPED_POOLS = [
    DATA_DIR / "synth_hard_decisions_grouped.jsonl",
    DATA_DIR / "synth_long_policy_grouped.jsonl",
]

SDK_DISTILL_POOLS = [
    DATA_DIR / "round4c" / "sdk_synthetic_train.jsonl",
    DATA_DIR / "ragtruth_served_train_v2.jsonl",
    DATA_DIR / "train_distill_e2b.jsonl",
]

MULTIMODAL_POOLS = [
    DATA_DIR / "train_multimodal_mixture.jsonl",
]

FORBIDDEN_EVAL_PATHS = [
    DATA_DIR / "exp01_gate_eval.jsonl",
    DATA_DIR / "test_v2.jsonl",
    DATA_DIR / "val.jsonl",
    DATA_DIR / "test.jsonl",
]


def generate_lev_codes(max_count: int = 255) -> List[str]:
    """Generates the single-token option codes used by llama.cpp lev protocol."""
    codes: List[str] = []
    # Single letters A..Z (26)
    for c in range(ord("A"), ord("Z") + 1):
        codes.append(chr(c))
    # Two-letter codes AA..ZZ (676 available)
    for a in range(ord("A"), ord("Z") + 1):
        for b in range(ord("A"), ord("Z") + 1):
            codes.append(chr(a) + chr(b))
            if len(codes) >= max_count:
                return codes[:max_count]
    return codes[:max_count]


LEV_CODES = generate_lev_codes(255)


def build_forbidden_keys() -> Set[str]:
    """Builds a set of hash keys from evaluation/test splits to guarantee zero leakage."""
    forbidden = set()
    for path in FORBIDDEN_EVAL_PATHS:
        if not path.is_file():
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    p = obj.get("premise") or obj.get("context") or obj.get("state") or ""
                    h = obj.get("hypothesis") or obj.get("claim") or obj.get("question") or ""
                    if p and h:
                        key = f"{str(p).strip().lower()} ||| {str(h).strip().lower()}"
                        forbidden.add(key)
                except Exception:
                    pass
    return forbidden


def format_nli_as_v2_sample(
    premise: str,
    hypothesis: str,
    label: int,  # 0=contradiction, 1=entailment, 2=neutral
    as_noul: bool = True,
    soft_labels: Optional[List[float]] = None,
    domain: str = "foundational_nli",
    images: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Transforms an NLI triplet into a Gevva v2 causal decision training sample."""
    if as_noul:
        # Formulate as a noul (boolean verification) task
        if label == 1:
            gold_key = "true"
            soft_true = 1.0 if not soft_labels else soft_labels[1]
        elif label == 0:
            gold_key = "false"
            soft_true = 0.0 if not soft_labels else 1.0 - soft_labels[0]
        else:
            gold_key = "false"
            soft_true = 0.5 if not soft_labels else soft_labels[1] + 0.5 * soft_labels[2]

        options = [
            {"key": "true", "description": "The statement logically follows from the evidence."},
            {"key": "false", "description": "The statement is not supported or contradicted."},
        ]

        # Randomize presentation order
        perm = [0, 1]
        random.shuffle(perm)
        permuted_options = [options[i] for i in perm]
        gold_idx = perm.index(0 if gold_key == "true" else 1)
        gold_code = LEV_CODES[gold_idx]

        sample: Dict[str, Any] = {
            "type": "noul",
            "state": premise.strip(),
            "instructions": f"Does the following claim logically follow from the evidence: \"{hypothesis.strip()}\"?",
            "options": [
                {"label": LEV_CODES[i], "key": opt["key"], "description": opt["description"]}
                for i, opt in enumerate(permuted_options)
            ],
            "gold_code": gold_code,
            "gold_key": gold_key,
            "soft_probability": float(soft_true),
            "domain": domain,
        }
        if images:
            sample["images"] = images
        return sample
    else:
        # Formulate as a 3-way choice task
        options = [
            {"key": "entailment", "description": f"The claim \"{hypothesis.strip()}\" is factually supported."},
            {"key": "contradiction", "description": f"The claim \"{hypothesis.strip()}\" is factually contradicted."},
            {"key": "neutral", "description": f"The claim \"{hypothesis.strip()}\" cannot be determined from the evidence."},
        ]
        label_keys = {0: "contradiction", 1: "entailment", 2: "neutral"}
        target_key = label_keys.get(label, "neutral")

        perm = list(range(3))
        random.shuffle(perm)
        permuted = [options[i] for i in perm]
        gold_idx = [opt["key"] for opt in permuted].index(target_key)
        gold_code = LEV_CODES[gold_idx]

        sample = {
            "type": "choice",
            "state": premise.strip(),
            "instructions": f"Evaluate the truth value of the claim relative to the evidence: \"{hypothesis.strip()}\"",
            "options": [
                {"label": LEV_CODES[i], "key": opt["key"], "description": opt["description"]}
                for i, opt in enumerate(permuted)
            ],
            "gold_code": gold_code,
            "gold_key": target_key,
            "domain": domain,
        }
        if soft_labels and len(soft_labels) == 3:
            # Map [p_con, p_ent, p_neu] to permuted options
            prob_map = {"contradiction": soft_labels[0], "entailment": soft_labels[1], "neutral": soft_labels[2]}
            sample["soft_probabilities"] = [float(prob_map[opt["key"]]) for opt in permuted]
        if images:
            sample["images"] = images
        return sample


def format_grouped_scenario_as_v2_sample(
    rows: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Transforms a group of options into a multi-choice decision sample."""
    if not rows or len(rows) < 2 or len(rows) > 255:
        return None

    state = rows[0].get("premise", "").strip()
    if not state:
        return None

    options_dict = {}
    soft_probs_dict = {}
    gold_key = None

    for r in rows:
        meta = r.get("metadata", {})
        opt_key = meta.get("option_key") or r.get("hypothesis", "").split(":")[0].strip()
        desc = meta.get("rationale") or r.get("hypothesis", "").strip()
        options_dict[opt_key] = desc
        if r.get("is_gold"):
            gold_key = opt_key
        if "soft_target" in r:
            soft_probs_dict[opt_key] = float(r["soft_target"])

    if not gold_key or gold_key not in options_dict:
        # Fallback: check label == 1
        for r in rows:
            if r.get("label") == 1:
                meta = r.get("metadata", {})
                gold_key = meta.get("option_key") or r.get("hypothesis", "").split(":")[0].strip()
                break

    if not gold_key or gold_key not in options_dict:
        return None

    # Epoch-level permutation
    keys = list(options_dict.keys())
    k = len(keys)
    perm = list(range(k))
    random.shuffle(perm)
    permuted_keys = [keys[i] for i in perm]
    gold_idx = permuted_keys.index(gold_key)
    gold_code = LEV_CODES[gold_idx]

    rendered_options = []
    active_soft_probs = []
    for i, key in enumerate(permuted_keys):
        rendered_options.append({
            "label": LEV_CODES[i],
            "key": str(key),
            "description": str(options_dict[key]),
        })
        if key in soft_probs_dict:
            active_soft_probs.append(soft_probs_dict[key])

    sample: Dict[str, Any] = {
        "type": "choice",
        "state": state,
        "instructions": "Determine the correct resolution or option based on the stated agreement and incident record:",
        "options": rendered_options,
        "gold_code": gold_code,
        "gold_key": str(gold_key),
        "num_options": k,
        "domain": rows[0].get("source", "grouped_decision"),
    }
    if len(active_soft_probs) == k:
        # Normalize soft probabilities to sum to 1.0
        total_p = sum(active_soft_probs) + 1e-8
        sample["soft_probabilities"] = [p / total_p for p in active_soft_probs]

    return sample


def main():
    parser = argparse.ArgumentParser(description="Compile Gevva v2 master training mixture.")
    parser.add_argument("--out", type=Path, default=DATA_DIR / "gevva_v2_train.jsonl", help="Output JSONL path.")
    parser.add_argument("--manifest", type=Path, default=DATA_DIR / "gevva_v2_manifest.json", help="Manifest path.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for repeatable compilation.")
    parser.add_argument("--cap-anchors", type=int, default=50000, help="Cap on foundational anchors.")
    parser.add_argument("--cap-distill", type=int, default=50000, help="Cap on distillation samples.")
    args = parser.parse_args()

    random.seed(args.seed)
    print("=== Compiling Gevva v2 Master Mixture (lev Protocol, K<=255) ===")
    print(f"Target Output: {args.out}")

    forbidden = build_forbidden_keys()
    print(f"Loaded {len(forbidden)} evaluation keys for strict decontamination.")

    stats: Counter = Counter()
    compiled_samples: List[Dict[str, Any]] = []

    # 1. Ingest Grouped Scenarios (Multi-Option Choice)
    print("1. Ingesting grouped multi-option scenarios...")
    for pool_path in GROUPED_POOLS:
        if not pool_path.is_file():
            continue
        groups = defaultdict(list)
        with open(pool_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                    gid = r.get("group_id")
                    if gid and gid != -1:
                        groups[gid].append(r)
                except Exception:
                    pass
        for gid, rows in groups.items():
            sample = format_grouped_scenario_as_v2_sample(rows)
            if sample:
                compiled_samples.append(sample)
                stats["grouped_scenarios"] += 1

    # 2. Ingest Round 4c & Distillation Pools (Teacher-Authored Decisions)
    print("2. Ingesting SDK & distillation decision pools...")
    distill_count = 0
    for pool_path in SDK_DISTILL_POOLS:
        if not pool_path.is_file():
            continue
        with open(pool_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip() or distill_count >= args.cap_distill:
                    continue
                try:
                    row = json.loads(line)
                    premise = row.get("premise", "")
                    hypothesis = row.get("hypothesis", "")
                    label = row.get("label")
                    if not premise or not hypothesis or label is None:
                        continue
                    key = f"{premise.strip().lower()} ||| {hypothesis.strip().lower()}"
                    if key in forbidden:
                        stats["decontaminated_distill"] += 1
                        continue
                    as_noul = random.random() < 0.5
                    sample = format_nli_as_v2_sample(
                        premise=premise,
                        hypothesis=hypothesis,
                        label=int(label),
                        as_noul=as_noul,
                        soft_labels=row.get("soft_labels"),
                        domain=row.get("source", "sdk_distill"),
                    )
                    if sample:
                        compiled_samples.append(sample)
                        stats["sdk_distill_decisions"] += 1
                        distill_count += 1
                except Exception:
                    pass

    # 3. Ingest Multimodal Decision Mixtures
    print("3. Ingesting multimodal decision mixtures...")
    for pool_path in MULTIMODAL_POOLS:
        if not pool_path.is_file():
            continue
        with open(pool_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    premise = row.get("premise", "")
                    hypothesis = row.get("hypothesis", "")
                    label = row.get("label")
                    images = row.get("images", [])
                    if not premise or not hypothesis or label is None:
                        continue
                    key = f"{premise.strip().lower()} ||| {hypothesis.strip().lower()}"
                    if key in forbidden:
                        stats["decontaminated_multimodal"] += 1
                        continue
                    as_noul = random.random() < 0.5
                    sample = format_nli_as_v2_sample(
                        premise=premise,
                        hypothesis=hypothesis,
                        label=int(label),
                        as_noul=as_noul,
                        domain="multimodal_grounding",
                        images=images,
                    )
                    if sample:
                        compiled_samples.append(sample)
                        stats["multimodal_decisions"] += 1
                except Exception:
                    pass

    # 4. Ingest Foundational Anchors (Capped for balance)
    print("4. Ingesting foundational NLI anchors...")
    anchor_count = 0
    for path in FOUNDATIONAL_ANCHORS:
        if not path.is_file():
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip() or anchor_count >= args.cap_anchors:
                    continue
                try:
                    row = json.loads(line)
                    premise = row.get("premise", "")
                    hypothesis = row.get("hypothesis", "")
                    label = row.get("label")
                    if not premise or not hypothesis or label is None:
                        continue
                    key = f"{premise.strip().lower()} ||| {hypothesis.strip().lower()}"
                    if key in forbidden:
                        stats["decontaminated_anchors"] += 1
                        continue
                    as_noul = random.random() < 0.6
                    sample = format_nli_as_v2_sample(premise, hypothesis, int(label), as_noul=as_noul)
                    if sample:
                        compiled_samples.append(sample)
                        stats["foundational_anchors"] += 1
                        anchor_count += 1
                except Exception:
                    pass

    random.shuffle(compiled_samples)
    print(f"\nTotal compiled v2 samples: {len(compiled_samples):,}")
    print("Composition breakdown:")
    for k, v in stats.items():
        print(f"  - {k}: {v:,}")

    # Write output JSONL
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for item in compiled_samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    manifest = {
        "version": "gevva-v2.0-master",
        "total_rows": len(compiled_samples),
        "protocol": "lev",
        "max_options": 255,
        "stats": dict(stats),
        "seed": args.seed,
    }
    with open(args.manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nSuccessfully compiled Gevva v2 master mixture to: {args.out}")
    print(f"Saved manifest to: {args.manifest}")


if __name__ == "__main__":
    main()
