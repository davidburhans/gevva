#!/usr/bin/env python3
"""compile_cal2.py - Round-2 calibration mixture for e2b (committed builder; fixes review M8).

Surgical design: identical to data/train_cal_e2b.jsonl (round 1) EXCEPT the
RAGTruth slice is swapped for the anti-collapse v2 slice
(data/ragtruth_served_train_v2.jsonl). Every other row is carried over
byte-identical, so any outcome delta vs round 1 isolates the slice + recipe
changes (brier 0.4, cross-option 0.0, phase5 base - recorded in the chain).

Hygiene: full-hash (sha256 of premise+hypothesis) dedup against the gate-eval
file - stronger than the 512-char prefix keys criticized in review C3.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def pair_key(premise: str, hypothesis: str) -> str:
    return hashlib.sha256((premise + "\x00" + hypothesis).encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=str(REPO_ROOT / "data" / "train_cal_e2b.jsonl"))
    parser.add_argument("--v2", default=str(REPO_ROOT / "data" / "ragtruth_served_train_v2.jsonl"))
    parser.add_argument("--out", default=str(REPO_ROOT / "data" / "train_cal2_e2b.jsonl"))
    parser.add_argument("--seed", type=int, default=47)
    args = parser.parse_args()

    forbidden = set()
    gate = REPO_ROOT / "data" / "exp01_gate_eval.jsonl"
    for line in open(gate):
        r = json.loads(line)
        forbidden.add(pair_key(r["premise"], r["hypothesis"]))

    kept, dropped_v1 = [], 0
    for line in open(args.base):
        r = json.loads(line)
        if r.get("source") == "ragtruth_served_train":
            dropped_v1 += 1
            continue
        if pair_key(r["premise"], r["hypothesis"]) in forbidden:
            continue
        kept.append(r)

    v2 = [json.loads(line) for line in open(args.v2)]
    v2 = [r for r in v2 if pair_key(r["premise"], r["hypothesis"]) not in forbidden]
    rows = kept + v2
    random.Random(args.seed).shuffle(rows)

    anli = sum(1 for r in rows if "anli" in r.get("source", ""))
    # Manifest v1 counts trivial_anchors=28003 as the SNLI/MNLI/QNLI/SciTail/FEVER
    # entailment-anchor blocks (source names verified 2026-09-30).
    trivial = sum(1 for r in rows if any(t in r.get("source", "")
                for t in ("snli", "mnli", "qnli", "scitail", "fever")))
    rag2 = sum(1 for r in rows if r.get("source") == "ragtruth_served_v2")
    # Assertions (review rule 1 + EXP-01 anchor evidence)
    if anli < 25000:
        raise SystemExit(f"ABORT: ANLI anchors {anli} < 25000 (EXP-01: anchors are load-bearing)")
    if trivial < 20000:
        raise SystemExit(f"ABORT: trivial anchors {trivial} < 20000")
    if rag2 != len(v2) or len(v2) < 7000:
        raise SystemExit(f"ABORT: ragtruth v2 slice {rag2} (loaded {len(v2)}) unexpected")
    if dropped_v1 != 15090:
        raise SystemExit(f"ABORT: expected to drop 15090 v1 ragtruth rows, dropped {dropped_v1}")

    Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in rows))
    import subprocess
    sha = subprocess.run(["sha256sum", args.base, args.v2], capture_output=True, text=True).stdout
    stem = Path(args.out).stem.replace("train_", "")
    manifest = {
        "file": Path(args.out).name, "rows": len(rows), "seed": args.seed,
        "carried_from_cal1": len(kept), "ragtruth_v2": len(v2), "dropped_v1_ragtruth": dropped_v1,
        "anli": anli, "trivial": trivial, "gate_overlap_removed": len(kept) + len(v2) - len(rows),
        "input_hashes": sha.strip(),
        "recipe_deltas_vs_cal1": ["ragtruth slice: v1 15090 constant-hyp -> v2 7545 templated 35%",
                                   "brier 0.8 -> 0.4", "cross-option 0.5 -> 0.0 (e2b) / 0.5 (e4b)",
                                   "seed 45/46 -> 47"],
    }
    (REPO_ROOT / "data" / f"{stem}_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
