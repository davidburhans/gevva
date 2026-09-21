#!/usr/bin/env python3
"""compile_p1_training_mixture.py
=================================
Compiles the comprehensive Stage 2+ training dataset uniting:
1. Synthesized Hard Decisions (temporal_numeric, probability, tradeoff, long_policy, trap, multi_hop, adversarial)
2. Grouped Typed Decisions Synth (n4ze3m/typed-decisions-synth, 10,179 choice & score workflows)
3. Anchor Clean NLI Replay (SNLI, MNLI, ANLI, FEVER) to prevent catastrophic forgetting (P6).

Outputs:
- data/train_p1_mixture.jsonl (Grouped training mixture ready for finetune.py)
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from typing import Any, Dict, List

from nli_labels import CONTRADICTION, ENTAILMENT, NEUTRAL, ID2LABEL
from research.adapters.typed_decisions_adapter import convert_typed_decisions_to_nli


def main():
    parser = argparse.ArgumentParser(description="Compile Stage 2+ P1/P2 Decision Mixture")
    parser.add_argument("--hard-synth-file", default="./data/synth_hard_decisions_grouped.jsonl")
    parser.add_argument("--long-policy-file", default="./data/synth_long_policy_grouped.jsonl")
    parser.add_argument("--clean-nli-file", default="./data/train.jsonl")
    parser.add_argument("--clean-nli-sample", type=int, default=5000, help="Number of clean NLI pairs to include as anchor")
    parser.add_argument("--max-typed-cases", type=int, default=None, help="Max cases from typed-decisions-synth (None = all)")
    parser.add_argument("--out-file", default="./data/train_p1_mixture.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    mixture: List[Dict[str, Any]] = []
    source_counts = Counter()

    # 1. Ingest Hard Synthesized Scenarios
    for path, desc in [(args.hard_synth_file, "hard synthetic scenarios"), (args.long_policy_file, "long-policy grounding scenarios")]:
        if os.path.exists(path):
            print(f"Loading {desc} from {path}...")
            n_items = 0
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        mixture.append(item)
                        source_counts[item.get("source", "hard_synth")] += 1
                        n_items += 1
            print(f"-> Loaded {n_items:,} items from {path}.")
        else:
            print(f"Note: {path} not found; skipping.")

    # 2. Ingest Grouped Typed Decisions Synth
    print(f"\nIngesting 'n4ze3m/typed-decisions-synth' in grouped mode (max_cases={args.max_typed_cases})...")
    typed_pairs = convert_typed_decisions_to_nli(
        split="train",
        max_cases=args.max_typed_cases,
        seed=args.seed,
        grouped=True,
    )
    for p in typed_pairs:
        mixture.append(p)
        source_counts[p.get("source", "typed_decisions")] += 1
    print(f"-> Converted {len(typed_pairs):,} grouped NLI items from typed-decisions-synth.")

    # 3. Ingest Anchor Clean NLI Replay (P6 Anchor Core)
    if os.path.exists(args.clean_nli_file) and args.clean_nli_sample > 0:
        print(f"\nSampling {args.clean_nli_sample:,} clean NLI anchor pairs from {args.clean_nli_file}...")
        clean_rows = []
        with open(args.clean_nli_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    clean_rows.append(row)
        
        # Filter for text-only clean NLI
        nli_candidates = [
            r for r in clean_rows
            if r.get("source") in ("snli", "mnli", "anli", "fever", "wanli", "xnli")
        ]
        if not nli_candidates:
            nli_candidates = clean_rows
            
        sampled_nli = rng.sample(nli_candidates, min(len(nli_candidates), args.clean_nli_sample))
        for r in sampled_nli:
            is_gold = (r.get("label") == ENTAILMENT)
            mixture.append({
                "id": f"anchor_nli_{r.get('id', len(mixture))}",
                "premise": r["premise"],
                "hypothesis": r["hypothesis"],
                "label": int(r["label"]),
                "group_id": -1,  # Un-grouped item (participates in NLI cross-entropy, ignored by xopt)
                "is_gold": is_gold,
                "soft_target": 1.0 if is_gold else 0.0,
                "soft_labels": r.get("soft_labels", [1.0, 0.0, 0.0] if r["label"] == CONTRADICTION else [0.0, 1.0, 0.0] if r["label"] == ENTAILMENT else [0.0, 0.0, 1.0]),
                "source": f"anchor_{r.get('source', 'nli')}",
                "language": r.get("language", "en"),
                "image": r.get("image", ""),
            })
            source_counts[f"anchor_{r.get('source', 'nli')}"] += 1
        print(f"-> Added {len(sampled_nli):,} clean NLI anchor pairs.")

    # 4. Shuffle at Group Level
    print(f"\n--- Final Mixture Assembly ({len(mixture):,} total rows) ---")
    
    # Verify group atomicity: all items with same group_id stay together
    groups = Counter(r.get("group_id") for r in mixture)
    n_grouped_items = sum(c for gid, c in groups.items() if gid is not None and gid != -1 and gid != "-1")
    n_unique_groups = len([gid for gid in groups if gid is not None and gid != -1 and gid != "-1"])
    print(f"Unique Decision Groups: {n_unique_groups:,} (containing {n_grouped_items:,} competing options)")
    print(f"Un-grouped Anchor Pairs: {groups.get(-1, 0) + groups.get('-1', 0):,}")

    label_dist = Counter(r["label"] for r in mixture)
    print("\nLabel Distribution:")
    for l_id in (CONTRADICTION, ENTAILMENT, NEUTRAL):
        count = label_dist.get(l_id, 0)
        pct = count / len(mixture) * 100 if mixture else 0.0
        print(f"  [{l_id}] {ID2LABEL[l_id]:13s}: {count:,} ({pct:.1f}%)")

    print("\nTop Data Sources:")
    for src, count in source_counts.most_common(12):
        print(f"  - {src:35s}: {count:,}")

    # Write output
    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    with open(args.out_file, "w", encoding="utf-8") as f:
        for r in mixture:
            f.write(json.dumps(r) + "\n")
    print(f"\nSuccessfully written to {args.out_file} ({os.path.getsize(args.out_file) / (1024*1024):.2f} MB)")


if __name__ == "__main__":
    main()
