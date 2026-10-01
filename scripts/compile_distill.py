#!/usr/bin/env python3
"""compile_distill.py - Round-4 distillation mixture (committed builder, review M8 rule).

Base: data/train_cal2_e2b.jsonl (the validated round-2 recipe). Addition: the
fresh committee-validated teacher slice from the round-4 generation run, capped
at --dose (fraction of final rows, default 0.10 of ~143k). Hygiene: full-hash
dedup against the gate file and the OLD SDK slice (no exact retraining rows),
anchor assertions per EXP-01.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def pair_key(premise: str, hypothesis: str) -> str:
    return hashlib.sha256((premise + "\x00" + hypothesis).encode("utf-8")).hexdigest()


def load_keys(path: Path) -> set:
    keys = set()
    if not path.exists():
        return keys
    for line in open(path):
        r = json.loads(line)
        keys.add(pair_key(r["premise"], r["hypothesis"]))
    return keys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=str(REPO_ROOT / "data" / "train_cal2_e2b.jsonl"))
    parser.add_argument("--distill", default=str(REPO_ROOT / "data" / "round4" / "sdk_synthetic_train.jsonl"))
    parser.add_argument("--old-sdk", default=str(REPO_ROOT / "data" / "staged" / "sdk_synthetic_train.jsonl"))
    parser.add_argument("--out", default=str(REPO_ROOT / "data" / "train_distill_e2b.jsonl"))
    parser.add_argument("--dose", type=float, default=0.10, help="Max fraction of final rows from distill slice")
    parser.add_argument("--seed", type=int, default=48)
    args = parser.parse_args()

    distill_path = Path(args.distill)
    if not distill_path.exists():
        raise SystemExit(f"ABORT: distill slice missing: {distill_path} (generation failed?)")
    raw = [json.loads(line) for line in open(distill_path)]
    clean = [r for r in raw if not r.get("needs_review")]
    if len(clean) < 4000:
        raise SystemExit(f"ABORT: distill slice too small after review filter: {len(clean)} clean of {len(raw)}")
    # Provenance assertion (2026-09-30 incident: committee silently skipped when
    # --validator-url was unset; template-only rows passed every count assertion).
    # Committee-validated rows carry per-judge rationales - require them on >=95%.
    with_rationales = sum(1 for r in clean if r.get("committee_rationales"))
    if with_rationales < 0.95 * len(clean):
        raise SystemExit(
            f"ABORT: distill slice lacks committee provenance: only {with_rationales}/{len(clean)} "
            f"rows carry committee_rationales (unvalidated teacher-less output?)")

    forbidden = load_keys(REPO_ROOT / "data" / "exp01_gate_eval.jsonl") | load_keys(Path(args.old_sdk))
    base_rows = [json.loads(line) for line in open(args.base)]
    base_keys = {pair_key(r["premise"], r["hypothesis"]) for r in base_rows}

    fresh = [r for r in clean if pair_key(r["premise"], r["hypothesis"]) not in forbidden
             and pair_key(r["premise"], r["hypothesis"]) not in base_keys]
    if len(fresh) < 3000:
        raise SystemExit(f"ABORT: fresh distill rows after dedup too small: {len(fresh)}")

    cap = int(args.dose * (len(base_rows) + len(fresh)))
    if len(fresh) > cap:
        fresh = random.Random(args.seed).sample(fresh, cap)
    rows = base_rows + [
        {"premise": r["premise"], "hypothesis": r["hypothesis"], "label": int(r["label"]),
         "source": "round4_distill", "group_id": r.get("id", "")}
        for r in fresh
    ]
    random.Random(args.seed).shuffle(rows)

    anli = sum(1 for r in rows if "anli" in r.get("source", ""))
    trivial = sum(1 for r in rows if any(t in r.get("source", "") for t in ("snli", "mnli", "qnli", "scitail", "fever")))
    rag2 = sum(1 for r in rows if r.get("source") == "ragtruth_served_v2")
    if anli < 25000 or trivial < 20000:
        raise SystemExit(f"ABORT: anchors degraded anli={anli} trivial={trivial}")
    if rag2 != 7545:
        raise SystemExit(f"ABORT: ragtruth_v2 slice changed: {rag2} != 7545")

    Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in rows))
    sha = subprocess.run(["sha256sum", args.base, str(distill_path)], capture_output=True, text=True).stdout
    stem = Path(args.out).stem.replace("train_", "")
    manifest = {
        "file": Path(args.out).name, "rows": len(rows), "seed": args.seed,
        "base_rows": len(base_rows), "distill_rows": len(fresh), "distill_raw": len(raw),
        "distill_clean": len(clean), "distill_dose": round(len(fresh) / len(rows), 4),
        "anli": anli, "trivial": trivial, "ragtruth_v2": rag2,
        "input_hashes": sha.strip(),
        "recipe": "cal2 base + round4 committee distill slice (cascade panel, early exit)",
    }
    (REPO_ROOT / "data" / f"{stem}_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
