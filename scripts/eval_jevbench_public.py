#!/usr/bin/env python3
"""scripts/eval_jevbench_public.py - Local JevBench public-split evaluation harness.

Scores any checkpoint under the EXACT frozen mapping from the jevbench repo
(docs/cross-encoder-mapping.md): the mapping helpers and dataset loader are
imported from the sibling jevbench checkout, and scoring uses the wrapper's own
rerank(scoring='entailment') - identity with the scored-run protocol is by
construction, not re-implementation.

Reports: accuracy by family and tier, gold-ranked-2nd rate on errors (the Phase-1
audit metric), renormalized-distribution ECE, Wilson CIs, latency.

Usage:
  uv run python scripts/eval_jevbench_public.py --model-path ckpt/gevva-e2b
  uv run python scripts/eval_jevbench_public.py --temperature 1.6
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

JEVBENCH_DEFAULT = Path(__file__).resolve().parents[2] / "jevbench"
TIER_FILES = {"easy": "easy.jsonl", "standard": "original.jsonl", "hard": "hard.jsonl"}


def wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return center - half, center + half


def main() -> int:
    parser = argparse.ArgumentParser(description="JevBench public-split evaluation for Gevva")
    parser.add_argument("--model-path", default="ckpt/gevva-e2b", help="Path to checkpoint (default: ckpt/gevva-e2b)")
    parser.add_argument("--jevbench-dir", default=str(JEVBENCH_DEFAULT))
    parser.add_argument("--limit", type=int, default=None, help="Cap items per tier (quick runs)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-len", type=int, default=16384, help="Context budget (declare trained length)")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Override the shipped calibration temperature (1.0 = score the untempered "
                             "served distribution; default = the artifact's calibration.json T*)")
    parser.add_argument("--out", default=None, help="JSON artifact path (default results/jevbench_public_<name>.json)")
    args = parser.parse_args()

    jb = Path(args.jevbench_dir)
    sys.path.insert(0, str(jb))
    from jevbench.adapters.cross_encoder_local import (  # noqa: E402
        ce_option_text,
        ce_premise,
        ce_options,
        renormalize_entailments,
    )
    from jevbench.tasks import Task, load_jsonl  # noqa: E402

    from gemma4_cross_encoder import Gemma4CrossEncoder  # noqa: E402

    # Build a tiny shim so adapter helpers see the same task shape they were written for.
    def as_task(item: dict) -> Task:
        return Task(**item) if not isinstance(item, Task) else item

    tiers: dict[str, list] = {}
    for tier, fname in TIER_FILES.items():
        items = [as_task(d) for d in load_jsonl(str(jb / "datasets" / "public" / fname))]
        if args.limit:
            items = items[: args.limit]
        tiers[tier] = items

    print(f"Loading {args.model_path} (max_len={args.max_len})...")
    enc = Gemma4CrossEncoder(args.model_path, device=args.device, max_len=args.max_len)
    if args.temperature is not None:
        enc.calibrated_temperature = float(args.temperature)  # gate must score T=1 AND shipped T* (review F08/N9)

    per_item = []
    for tier, items in tiers.items():
        for t in items:
            premise = ce_premise(t)
            options = ce_options(t)
            t0 = time.perf_counter()
            scores = enc.rerank(premise, [options[l] for l in t.labels],
                                hyp_format="The correct answer is: {}", scoring="entailment").scores
            lat = time.perf_counter() - t0
            s_map = {l: float(s) for l, s in zip(t.labels, scores)}
            probs = renormalize_entailments(s_map)
            ranked = sorted(t.labels, key=lambda l: -probs[l])
            pred = ranked[0]
            is_corr = str(pred) == str(t.expected)
            gold2nd = (str(ranked[1]) == str(t.expected)) if len(ranked) > 1 and not is_corr else False
            per_item.append({
                "id": t.id, "tier": tier, "family": t.family, "type": t.question.get("type"),
                "gold": t.expected, "pred": pred, "correct": is_corr,
                "gold_rank2_on_error": bool(gold2nd), "latency_s": round(lat, 4),
                "probs": {k: round(v, 4) for k, v in probs.items()},
            })

    def summarize(items):
        n = len(items)
        if not n:
            return {}
        acc = sum(i["correct"] for i in items) / n
        lo, hi = wilson_ci(acc, n)
        errs = [i for i in items if not i["correct"]]
        lat = sorted(i["latency_s"] for i in items)
        return {"n": n, "accuracy": round(acc, 4), "ci95": [round(lo, 4), round(hi, 4)],
                "gold_rank2_on_error": f"{sum(e['gold_rank2_on_error'] for e in errs)}/{len(errs)}",
                "p50_ms": round(lat[n // 2] * 1000, 1), "p95_ms": round(lat[int(n * .95)] * 1000, 1)}

    by_family = defaultdict(list)
    for i in per_item:
        by_family[i["family"]].append(i)
    # hard-tier ECE on the renormalized distribution
    def dist_ece(items, bins=10):
        data = [(max(i["probs"].values()), i["correct"]) for i in items]
        out = 0.0
        for b in range(bins):
            grp = [(c, ok) for c, ok in data if b / bins <= c < (b + 1) / bins or (b == bins - 1 and c == 1.0)]
            if grp:
                out += len(grp) / len(data) * abs(sum(ok for _, ok in grp) / len(grp) - sum(c for c, _ in grp) / len(grp))
        return round(out, 4)

    hard = [i for i in per_item if i["tier"] == "hard"]
    tier_items = {t: [i for i in per_item if i["tier"] == t] for t in tiers}
    artifact = {
        "provenance": {
            "model_path": args.model_path, "max_len": args.max_len, "device": args.device,
            "temperature": args.temperature if args.temperature is not None else enc.calibrated_temperature,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mapping": "jevbench docs/cross-encoder-mapping.md (helpers imported from jevbench repo)",
            "limit_per_tier": args.limit,
        },
        "tiers": {t: summarize(v) for t, v in tier_items.items() if v},
        "families": {f: summarize(v) for f, v in sorted(by_family.items())},
        "renormalized_ece_hard": dist_ece(hard),
        "overall": summarize(per_item),
    }

    name = Path(args.model_path).name
    out = args.out or f"results/jevbench_public_{name}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    with open(out.replace(".json", "_items.jsonl"), "w", encoding="utf-8") as f:
        for i in per_item:
            f.write(json.dumps(i) + "\n")

    print(json.dumps({k: artifact[k] for k in ("tiers", "renormalized_ece_hard", "overall")}, indent=1))
    print(f"\nArtifact: {out} (+ per-item jsonl)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
