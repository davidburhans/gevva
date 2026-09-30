#!/usr/bin/env python3
"""audit_ragtruth_leakage.py - Prove RAGTruth train/eval disjointness (review C3).

Exact sha256 (prompt, response) overlap between the training slice and the
Decision Index cat-59 suite items. Exits non-zero on any overlap > tolerance
so it can gate future compiles.
"""
import gzip, hashlib, json, re, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

def h(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def suite_pairs() -> set:
    pairs = set()
    for shard in ("selected-rows.jsonl.gz", "added-rows.jsonl.gz"):
        p = REPO / "suite-0.2" / shard
        if not p.exists():
            continue
        with gzip.open(p, "rt") as f:
            for line in f:
                r = json.loads(line)
                if r.get("_evaluation", {}).get("catalog_id") != 59:
                    continue
                st = r.get("state", {})
                pairs.add((h(st.get("prompt", "")), h(st.get("response", ""))))
    return pairs

def train_pairs() -> set:
    pairs = set()
    with open(REPO / "data" / "ragtruth_train.jsonl") as f:
        for line in f:
            r = json.loads(line)
            m = re.match(r"Premise: (.*?)Hypothesis: (.*)", r.get("text", ""), re.S)
            if m:
                pairs.add((h(m.group(1).strip()), h(m.group(2).strip())))
    return pairs

if __name__ == "__main__":
    sp, tp = suite_pairs(), train_pairs()
    overlap = sp & tp
    print(f"suite pairs: {len(sp)} | train pairs: {len(tp)} | overlap: {len(overlap)}")
    out = {"suite_pairs": len(sp), "train_pairs": len(tp), "overlap": len(overlap),
           "audited": __import__("time").strftime("%Y-%m-%d %H:%M:%S")}
    Path(REPO / "results" / "ragtruth_leakage_audit.json").write_text(json.dumps(out, indent=2))
    sys.exit(1 if len(overlap) > 0 else 0)
