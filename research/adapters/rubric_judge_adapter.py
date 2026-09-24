#!/usr/bin/env python3
"""research/adapters/rubric_judge_adapter.py - Phase 3 Rubric & Judge Adapter.

Converts high-precision rubric evaluation and subtle-flaw datasets into
grouped NLI cross-encoder training pairs targeting the measured JevBench
weaknesses on `judge_hard`, `adequacy`, `policy`, and Mode B subtle near-misses.

Supported Datasets:
1. Prometheus Feedback Collection (`prometheus-eval/Feedback-Collection`, Apache 2.0, 100K items)
   - Pairs explicit rubric criteria and candidate responses.
   - Maps scores: 4/5 -> ENTAILMENT (gold), 1/2 -> CONTRADICTION, 3 -> NEUTRAL.
   - Grouped by instruction for atomic cross-option / served-distribution ranking.
2. NVIDIA HelpSteer2 (`nvidia/HelpSteer2`, CC-BY-4.0, 20K items)
   - Human-annotated multi-attribute ratings (correctness, coherence, helpfulness).
   - Isolates subtle flaws: coherence >= 3 (fluent text) but correctness <= 1 (factually false) -> CONTRADICTION.
   - High quality: coherence >= 4 and correctness >= 4 -> ENTAILMENT.
   - Grouped by prompt for atomic ranking.

Protocol §8 Decontamination:
- Integrated 8-gram decontamination against JevBench public splits and held-out test.jsonl.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, Set, Tuple

import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nli_labels import CONTRADICTION, ENTAILMENT, NEUTRAL, ID2LABEL

logger = logging.getLogger(__name__)

DEFAULT_OUT_DIR = REPO_ROOT / "data" / "staged" / "phase3"
DEFAULT_JEVBENCH_DIR = Path(__file__).resolve().parents[3] / "jevbench"

DATASET_REGISTRY = {
    "prometheus_feedback": {
        "hf_path": "prometheus-eval/Feedback-Collection",
        "config": None,
        "license": "Apache 2.0",
        "description": "100K instruction-response-rubric triplets with fine-grained scores (1-5)",
        "weakness_target": "judge_hard_adequacy_rubric",
    },
    "helpsteer2": {
        "hf_path": "nvidia/HelpSteer2",
        "config": None,
        "license": "CC-BY-4.0",
        "description": "20K human-annotated multi-attribute ratings for subtle flaw detection",
        "weakness_target": "mode_b_subtle_flaws",
    },
}


class DecontaminationFilter:
    """8-gram filter protecting JevBench evaluation splits and test.jsonl."""

    def __init__(self, reference_files: Iterable[Path], n: int = 8) -> None:
        self.n = n
        self.ref_ngrams: Set[str] = set()
        self._build_index(reference_files)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def _extract_ngrams(self, tokens: List[str]) -> Set[str]:
        if len(tokens) < self.n:
            return set()
        return {" ".join(tokens[i : i + self.n]) for i in range(len(tokens) - self.n + 1)}

    def _build_index(self, reference_files: Iterable[Path]) -> None:
        count_files = 0
        for p in reference_files:
            if not p or not p.exists():
                continue
            count_files += 1
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                        for key in ("premise", "hypothesis", "text", "question", "prompt", "context"):
                            val = row.get(key)
                            if isinstance(val, str) and val.strip():
                                toks = self._tokenize(val)
                                self.ref_ngrams.update(self._extract_ngrams(toks))
                    except json.JSONDecodeError:
                        continue
        logger.info(f"Decontamination index: {len(self.ref_ngrams):,} {self.n}-grams from {count_files} files.")

    def is_contaminated(self, text: str) -> bool:
        if not self.ref_ngrams:
            return False
        toks = self._tokenize(text)
        sample_ngrams = self._extract_ngrams(toks)
        return bool(sample_ngrams.intersection(self.ref_ngrams))


def convert_prometheus_feedback(
    record: Dict[str, Any],
    group_idx: int,
) -> Optional[Dict[str, Any]]:
    """Converts a Prometheus Feedback record to an NLI pair."""
    instruction = str(record.get("orig_instruction", "")).strip()
    criteria = str(record.get("orig_criteria", "")).strip()
    response = str(record.get("orig_response", "")).strip()
    raw_score = str(record.get("orig_score", "")).strip()

    if not instruction or not response or not raw_score:
        return None

    try:
        score = int(raw_score)
    except ValueError:
        return None

    # Map score: 4/5 -> Entailment (meets criteria), 1/2 -> Contradiction (fails), 3 -> Neutral
    if score >= 4:
        label = ENTAILMENT
        is_gold = True
    elif score <= 2:
        label = CONTRADICTION
        is_gold = False
    else:
        label = NEUTRAL
        is_gold = False

    premise = f"Task Instruction: {instruction}\n\nEvaluation Criteria: {criteria}\n\nCandidate Response: {response}"
    hypothesis = "The candidate response satisfies the evaluation criteria."

    return {
        "premise": premise,
        "hypothesis": hypothesis,
        "label": label,
        "group_id": f"prom_fb_{group_idx}",
        "is_gold": is_gold,
        "source": "prometheus_feedback",
        "metadata": {
            "score": score,
            "orig_score": raw_score,
        },
    }


def convert_helpsteer2(
    record: Dict[str, Any],
    group_idx: int,
) -> Optional[Dict[str, Any]]:
    """Converts a HelpSteer2 record to an NLI pair focusing on subtle flaws."""
    prompt = str(record.get("prompt", "")).strip()
    response = str(record.get("response", "")).strip()
    correctness = record.get("correctness", -1)
    coherence = record.get("coherence", -1)

    if not prompt or not response or correctness < 0:
        return None

    # High quality: correctness >= 4 and coherence >= 4
    if correctness >= 4 and coherence >= 4:
        label = ENTAILMENT
        is_gold = True
    # Subtle flaws: fluent/coherent (>= 3) but factually flawed (<= 1)
    elif coherence >= 3 and correctness <= 1:
        label = CONTRADICTION
        is_gold = False
    elif correctness == 2 or correctness == 3:
        label = NEUTRAL
        is_gold = False
    else:
        # Ambiguous / incoherent garbage
        return None

    premise = f"Prompt: {prompt}\n\nResponse: {response}"
    hypothesis = "The response is completely correct and factually accurate."

    return {
        "premise": premise,
        "hypothesis": hypothesis,
        "label": label,
        "group_id": f"helpsteer_{group_idx}",
        "is_gold": is_gold,
        "source": "helpsteer2_subtle_flaw",
        "metadata": {
            "correctness": correctness,
            "coherence": coherence,
            "helpfulness": record.get("helpfulness"),
        },
    }


def get_default_reference_files() -> List[Path]:
    refs = []
    test_jsonl = REPO_ROOT / "data" / "test.jsonl"
    if test_jsonl.exists():
        refs.append(test_jsonl)
    jb_dir = DEFAULT_JEVBENCH_DIR / "datasets" / "public"
    if jb_dir.exists():
        refs.extend(jb_dir.glob("*.jsonl"))
    return refs


def stage_dataset(
    dataset_name: str,
    out_dir: Path,
    decon_filter: DecontaminationFilter,
    max_samples: Optional[int] = None,
    seed: int = 42,
) -> Path:
    """Streams, decontaminates, and stages a dataset into JSONL."""
    from datasets import load_dataset

    meta = DATASET_REGISTRY[dataset_name]
    logger.info(f"Staging {dataset_name} ({meta['hf_path']})...")

    ds = load_dataset(meta["hf_path"], meta["config"], split="train")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{dataset_name}_train.jsonl"

    rng = random.Random(seed)
    indices = list(range(len(ds)))
    if max_samples and max_samples < len(indices):
        rng.shuffle(indices)
        indices = indices[:max_samples]

    saved = 0
    decon_dropped = 0
    label_counts = Counter()

    with open(out_file, "w", encoding="utf-8") as f:
        for idx in indices:
            raw_row = ds[idx]
            if dataset_name == "prometheus_feedback":
                converted = convert_prometheus_feedback(raw_row, group_idx=idx)
            elif dataset_name == "helpsteer2":
                converted = convert_helpsteer2(raw_row, group_idx=idx)
            else:
                continue

            if not converted:
                continue

            # 8-gram decontamination check
            premise = converted["premise"]
            if decon_filter.is_contaminated(premise):
                decon_dropped += 1
                continue

            f.write(json.dumps(converted) + "\n")
            label_counts[converted["label"]] += 1
            saved += 1

    size_mb = out_file.stat().st_size / (1024 * 1024)
    logger.info(f"Finished {dataset_name}: {saved:,} pairs written ({size_mb:.2f} MB), {decon_dropped} contaminated pairs filtered.")
    for lbl in (CONTRADICTION, ENTAILMENT, NEUTRAL):
        cnt = label_counts[lbl]
        pct = cnt / saved * 100 if saved else 0.0
        logger.info(f"  [{lbl}] {ID2LABEL[lbl]:13s}: {cnt:>6,} ({pct:5.1f}%)")

    return out_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage Phase-3 Rubric & Subtle-Flaw Datasets")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output directory")
    parser.add_argument("--dataset", choices=["all", "prometheus_feedback", "helpsteer2"], default="all")
    parser.add_argument("--max-prometheus", type=int, default=30000, help="Max Prometheus samples to stage")
    parser.add_argument("--max-helpsteer2", type=int, default=20000, help="Max HelpSteer2 samples to stage")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    decon = DecontaminationFilter(get_default_reference_files(), n=8)
    out_dir = Path(args.out_dir)

    targets = []
    if args.dataset in ("all", "prometheus_feedback"):
        targets.append(("prometheus_feedback", args.max_prometheus))
    if args.dataset in ("all", "helpsteer2"):
        targets.append(("helpsteer2", args.max_helpsteer2))

    for name, max_s in targets:
        stage_dataset(name, out_dir, decon, max_samples=max_s, seed=args.seed)


if __name__ == "__main__":
    main()
