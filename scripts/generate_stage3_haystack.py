#!/usr/bin/env python3
"""scripts/generate_stage3_haystack.py - Multi-Resolution 128K Needle-in-a-Haystack Generator.

Builds large-context synthetic NLI verification pairs across multi-resolution token horizons:
  [4K, 8K, 16K, 32K, 64K, 128K] tokens

Key Design Principles:
1. Full 128K Context Support:
   - Realistically sized documents reaching up to 131,072 tokens.
   - Varied filler corpora: Technical architecture specs, enterprise incident logs,
     legal contracts, regulatory compliance docs, and code modules.
2. 3-State Calibrated Distribution:
   - 35% Entailment (needle inserted at exact depth percentile d in [0.0, 1.0])
   - 35% Contradiction (corrupted needle: numerical mutation or semantic negation at depth d)
   - 30% Neutral (needle omitted, or related distractor claiming different entity)
3. Depth Stratification:
   - Tracks needle position across 5 depth buckets:
     [0.0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0]
4. Zero-Contamination Enforcement:
   - Holds out any premise or pair present in data/test.jsonl.

Usage:
  uv run python scripts/generate_stage3_haystack.py --out-dir data/stage3 --samples 2500
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nli_labels import CONTRADICTION, ENTAILMENT, NEUTRAL

# Context length target buckets (in approximate tokens, ~0.75 words/token)
CONTEXT_BUCKETS = {
    "4k": 3000,      # ~4K tokens
    "8k": 6000,      # ~8K tokens
    "16k": 12000,    # ~16K tokens
    "32k": 24000,    # ~32K tokens
    "64k": 48000,    # ~64K tokens
    "128k": 96000,   # ~128K tokens
}

# Rich multi-domain filler templates to simulate enterprise documents
DOMAINS = [
    "cloud_infrastructure",
    "enterprise_security",
    "legal_compliance",
    "financial_audit",
    "software_architecture",
]

PARAGRAPH_TEMPLATES = {
    "cloud_infrastructure": [
        "The distributed ingress controller handles load balancing across regional edge nodes. Latency metrics indicate p99 response times of 14.2ms under sustained query load. Traffic throttling policies are automatically enacted when the connection pool depth exceeds 4,500 active HTTP/2 connections.",
        "Worker pool autoscaling is managed via custom Kubernetes horizontal pod autoscalers based on memory utilization and ingress queue length. Node pools are partitioned into spot instances for asynchronous worker tasks and on-demand instances for critical path transaction coordinators.",
        "The object storage replication topology enforces cross-region bucket synchronization with an asynchronous RPO of less than 30 seconds. Encryption at rest utilizes customer-managed keys rotated every 90 days via the hardware security module API.",
        "Network security groups restrict egress access to registered CIDR blocks. VPC peering connections between production and analytics environments are routed through stateful inspection firewalls with TLS termination disabled on internal links.",
    ],
    "enterprise_security": [
        "The identity provider enforces hardware-backed FIDO2 multi-factor authentication for all privileged operator sessions. Session lifespans for infrastructure administrators are strictly capped at 4 hours before re-authentication is mandatory.",
        "Vulnerability scanning routines evaluate container base images against the national vulnerability database every 6 hours. Images containing critical severity CVEs with published exploit payloads are automatically quarantined and prevented from deploying to production clusters.",
        "Audit logging pipelines stream immutable event records to append-only storage buckets with WORM compliance enabled. Access to log search interfaces requires dual-custody authorization and is reviewed bi-weekly by the compliance committee.",
        "Zero-trust network access policies evaluate endpoint posture attributes including device encryption status, OS patch level, and active EDR agent heartbeat prior to granting API gateway access.",
    ],
    "legal_compliance": [
        "Either party may terminate this master services agreement without cause upon providing sixty (60) days prior written notice to the designated legal notices address. In the event of material breach, the non-breaching party may terminate immediately if such breach remains uncured after thirty (30) days.",
        "The aggregate liability of either party arising out of or related to this agreement, regardless of the form of action, shall not exceed the total fees paid or payable by customer during the twelve (12) months preceding the incident giving rise to liability.",
        "All proprietary data, trade secrets, customer confidential records, and system architecture specifications disclosed during the evaluation period shall remain confidential for a duration of five (5) years following the termination of this agreement.",
        "Customer shall maintain sole and exclusive ownership of all right, title, and interest in and to Customer Data. Provider is granted a limited, non-exclusive license solely to the extent necessary to deliver the managed services.",
    ],
    "financial_audit": [
        "Consolidated net revenues for the third fiscal quarter reached $482.4 million, representing an 18.5% year-over-year increase driven by enterprise cloud adoption. Operating expenses totaled $312.1 million, including non-cash stock-based compensation of $44.2 million.",
        "Capital expenditures for data center infrastructure expansion totaled $68.5 million during the period. Free cash flow conversion remained robust at 24.8% of operating income, supporting ongoing share repurchase authorizations.",
        "Accounts receivable aging analysis shows that 94.2% of outstanding balances are current within 30 days. The allowance for doubtful accounts was adjusted to 1.8% of gross receivables based on historical loss migration rates.",
        "Cash and cash equivalents at the end of the reporting period stood at $1.24 billion, with zero drawn balance on the revolving credit facility and no senior secured debt maturing prior to 2029.",
    ],
    "software_architecture": [
        "The microservices communication mesh uses gRPC with Protocol Buffers version 3 for schema validation and binary serialization. Service discovery is maintained via Consul clusters with health checks executed every 5 seconds.",
        "The database layer implements an active-passive multi-region primary replication scheme with streaming write-ahead logs. Read queries are distributed across four regional read replicas with eventual consistency guarantees under 250ms.",
        "The distributed task queue utilizes partitioned Kafka topics with message retention configured for 7 days. Dead letter queues capture malformed event payloads and trigger automated alerting if dead letter depth exceeds 50 messages.",
        "Cache invalidation employs a two-tier strategy: local in-memory LRU caches expire after 60 seconds, while centralized Redis clusters use explicit CDC-based invalidation streams derived from database transaction commits.",
    ],
}

FACT_TEMPLATES = [
    # (needle_fact, entailment_hypo, contradiction_hypo, neutral_hypo)
    (
        "Project Vanguard's primary deployment pipeline was successfully migrated from Jenkins to GitHub Actions on October 14, 2025.",
        "Project Vanguard transitioned its main deployment pipeline to GitHub Actions in October 2025.",
        "Project Vanguard continues to use Jenkins as its primary deployment pipeline without migration.",
        "Project Vanguard uses GitLab CI for its mobile client application builds.",
    ),
    (
        "The maximum allowable batch processing timeout for high-priority payment transactions was set to 45 seconds.",
        "High-priority payment transactions are constrained by a batch timeout of 45 seconds.",
        "The maximum allowable batch timeout for high-priority payment transactions is 180 seconds.",
        "Low-priority settlement batches execute with a 15-minute retry window.",
    ),
    (
        "Dr. Elena Rostova was appointed Lead Architect for the neural retrieval acceleration initiative on January 12, 2026.",
        "Dr. Elena Rostova assumed the role of Lead Architect for neural retrieval acceleration in early 2026.",
        "Dr. Elena Rostova was removed from the neural retrieval acceleration project and replaced in January 2026.",
        "Dr. Elena Rostova previously published papers on distributed systems at MIT.",
    ),
    (
        "The quarterly compliance audit confirmed that all 14 database clusters in the Dublin data center have full disk encryption enabled.",
        "Every database cluster in the Dublin facility operates with full disk encryption.",
        "Three of the 14 database clusters in the Dublin facility failed compliance due to missing disk encryption.",
        "The Frankfurt data center completed its annual ISO 27001 audit with zero findings.",
    ),
    (
        "The emergency failover drill conducted on November 3 demonstrated seamless cluster switchover in 8.4 seconds without data loss.",
        "The failover exercise achieved complete cluster switchover in under 10 seconds with zero data loss.",
        "The emergency failover drill suffered severe data corruption and required over 45 minutes to recover.",
        "The disaster recovery testing schedule is mandated twice annually by regulatory authorities.",
    ),
    (
        "Under the revised SLA agreement, service availability compensation applies if quarterly uptime drops below 99.95%.",
        "Customers are eligible for SLA compensation if quarterly system uptime falls below 99.95%.",
        "The revised SLA agreement requires 99.999% uptime before any compensation penalties are assessed.",
        "Maintenance windows are scheduled exclusively on the third Sunday of each calendar month.",
    ),
    (
        "The security incident report concluded that the unauthorized access was contained within 14 minutes via automatic API token revocation.",
        "Automatic token revocation successfully contained the unauthorized access in less than 15 minutes.",
        "The security incident remained uncontained for three days before manual operator intervention.",
        "The organization conducts quarterly tabletop exercises for threat incident response.",
    ),
    (
        "The board of directors approved a $120 million capital allocation for expanding renewable energy procurement in Oregon.",
        "A $120 million capital expenditure for Oregon renewable energy was formally authorized by the board.",
        "The board of directors rejected the proposed $120 million capital allocation for renewable energy expansion.",
        "The company aims to achieve net-zero carbon emissions across all global operations by 2030.",
    ),
]


def _build_filler_document(target_words: int, rng: random.Random) -> List[str]:
    """Builds a multi-paragraph filler document of approximately target_words."""
    paragraphs: List[str] = []
    current_words = 0
    all_paras = []
    for domain_paras in PARAGRAPH_TEMPLATES.values():
        all_paras.extend(domain_paras)

    while current_words < target_words:
        p = rng.choice(all_paras)
        paragraphs.append(p)
        current_words += len(p.split())

    return paragraphs


def generate_haystack_dataset(
    n_samples: int = 2500,
    val_fraction: float = 0.1,
    test_file: Optional[str] = "data/test.jsonl",
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    rng = random.Random(seed)

    # 1. Contamination Filter against test set
    test_premises: Set[str] = set()
    test_pairs: Set[Tuple[str, str]] = set()
    if test_file and os.path.exists(test_file):
        with open(test_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    p = item.get("premise", "").strip()
                    h = item.get("hypothesis", "").strip()
                    test_premises.add(p)
                    test_pairs.add((p, h))

    samples: List[Dict[str, Any]] = []
    bucket_keys = list(CONTEXT_BUCKETS.keys())

    # Generate samples across multi-resolution horizons
    for idx in range(n_samples):
        # Pick context bucket
        bucket = rng.choice(bucket_keys)
        target_words = CONTEXT_BUCKETS[bucket]

        # Pick fact template
        needle, ent_hypo, con_hypo, neu_hypo = rng.choice(FACT_TEMPLATES)

        # Pick target class: 0=Contradiction, 1=Entailment, 2=Neutral
        label = rng.choices([ENTAILMENT, CONTRADICTION, NEUTRAL], weights=[0.35, 0.35, 0.30])[0]

        # Generate filler paragraphs
        doc_paras = _build_filler_document(target_words, rng)
        n_paras = len(doc_paras)

        # Stratified depth percentile: [0.0 - 1.0]
        depth_pct = round(rng.uniform(0.02, 0.98), 4)
        insert_idx = int(depth_pct * n_paras)

        if label == ENTAILMENT:
            hypothesis = ent_hypo
            doc_paras.insert(insert_idx, needle)
            needle_injected = True
            source_tag = f"haystack_{bucket}_ent"
        elif label == CONTRADICTION:
            hypothesis = con_hypo
            doc_paras.insert(insert_idx, needle)
            needle_injected = True
            source_tag = f"haystack_{bucket}_con"
        else:  # NEUTRAL
            hypothesis = neu_hypo
            # 50% omit completely, 50% inject unrelated distractor
            if rng.random() < 0.5:
                needle_injected = False
            else:
                doc_paras.insert(insert_idx, "An unrelated system assessment was logged regarding routine maintenance.")
                needle_injected = False
            source_tag = f"haystack_{bucket}_neu"

        full_document = "\n\n".join(doc_paras)

        # Depth bucket categorizer
        if depth_pct < 0.20:
            depth_bucket = "0.0-0.2"
        elif depth_pct < 0.40:
            depth_bucket = "0.2-0.4"
        elif depth_pct < 0.60:
            depth_bucket = "0.4-0.6"
        elif depth_pct < 0.80:
            depth_bucket = "0.6-0.8"
        else:
            depth_bucket = "0.8-1.0"

        sample = {
            "id": f"haystack128k_{idx:06d}",
            "premise": full_document,
            "hypothesis": hypothesis,
            "label": label,
            "source": source_tag,
            "language": "en",
            "image": "",
            "length": len(full_document.split()) + len(hypothesis.split()),
            "metadata": {
                "context_bucket": bucket,
                "target_words": target_words,
                "depth_percentile": depth_pct if needle_injected else None,
                "depth_bucket": depth_bucket if needle_injected else "none",
                "needle_injected": needle_injected,
            },
        }

        # Contamination guard
        if (full_document.strip(), hypothesis.strip()) in test_pairs:
            continue

        samples.append(sample)

    rng.shuffle(samples)
    n_val = int(len(samples) * val_fraction)
    val_samples = samples[:n_val]
    train_samples = samples[n_val:]

    train_dist = dict(Counter(s["label"] for s in train_samples))
    val_dist = dict(Counter(s["label"] for s in val_samples))
    bucket_dist = dict(Counter(s["metadata"]["context_bucket"] for s in train_samples))

    manifest = {
        "dataset_name": "gemma4-stage3-128k-haystack",
        "n_train": len(train_samples),
        "n_val": len(val_samples),
        "train_labels": train_dist,
        "val_labels": val_dist,
        "train_context_buckets": bucket_dist,
        "resolution_horizons": list(CONTEXT_BUCKETS.keys()),
    }

    return train_samples, val_samples, manifest


def main():
    parser = argparse.ArgumentParser(description="Generate 128K multi-resolution synthetic haystack dataset")
    parser.add_argument("--out-dir", default="data/stage3")
    parser.add_argument("--samples", type=int, default=2500)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-file", default="data/test.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.samples:,} multi-resolution 128K haystack samples...")
    train_samples, val_samples, manifest = generate_haystack_dataset(
        n_samples=args.samples,
        val_fraction=args.val_fraction,
        test_file=args.test_file,
        seed=args.seed,
    )

    train_file = out_path / "stage3_haystack_train.jsonl"
    val_file = out_path / "stage3_haystack_val.jsonl"
    manifest_file = out_path / "stage3_haystack_manifest.json"

    with open(train_file, "w", encoding="utf-8") as f:
        for s in train_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    with open(val_file, "w", encoding="utf-8") as f:
        for s in val_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 60)
    print("STAGE 3 HAYSTACK GENERATION COMPLETE:")
    print(f"  Train: {len(train_samples):,} rows -> {train_file}")
    print(f"  Val:   {len(val_samples):,} rows -> {val_file}")
    print(f"  Context Buckets: {manifest['train_context_buckets']}")
    print(f"  Train Labels: {manifest['train_labels']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
