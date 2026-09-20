#!/usr/bin/env python3
"""eval_openjev_benchmarks.py
==============================
Direct Head-to-Head Capability Comparison Benchmark Suite.

Benchmarks our Gemma 4 E2B NLI Cross-Encoder.

PROVENANCE NOTE: competitor numbers printed alongside ours are quoted published
reference constants (REFERENCE_BENCHMARKS below), NOT local runs - protocols and
hardware may differ. Our default rerank scoring is `margin`, a post-hoc variant;
openjev's documented protocol is raw entailment (`--scoring entailment`). MNLI/ECE
slices are shuffled with a fixed seed and exclude any pair present in the training
or checkpoint-selection data.

Supported Benchmarks:
1. NLI Sanity (Accuracy on MNLI-m and MNLI-mm)
2. Zero-Shot Multiple Choice Reranking without Reference (Blog #3 protocol):
   - ARC-Easy, ARC-Challenge, MMLU, WinoGrande
3. Zero-Shot Multiple Choice Grading with Reference (Blog #6 protocol):
   - ARC-Easy, ARC-Challenge, MMLU (Accuracy & F1)
4. Laya System 1 Decision Benchmarks:
   - BoolQ (Yes/No Question Answering)
   - AG News (4-class Topic Routing)
   - DAIR Emotion (6-class Emotion Classification)
5. Decision Latency & Calibration:
   - P50, P90, P99 Latency (ms) and Expected Calibration Error (ECE), Brier Score

Usage:
    # Run quick evaluation (50 examples per task) on W4A16 production model:
    uv run python eval_openjev_benchmarks.py --limit 50

    # Run comprehensive evaluation on specific tasks:
    uv run python eval_openjev_benchmarks.py --tasks rerank,grading,mnli --limit 200

    # Run full evaluation on full test sets:
    uv run python eval_openjev_benchmarks.py --full --out-file results/benchmarks_full.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset

from gemma4_cross_encoder import (
    CONTRADICTION,
    ENTAILMENT,
    NEUTRAL,
    ID2LABEL,
    LABEL2ID,
    Gemma4CrossEncoder,
)

# Reference published benchmarks from OpenJEV and Laya reports
REFERENCE_BENCHMARKS = {
    "mnli_m": {
        "Jev 1.13.0": "-",
        "openjev-4B": 0.904,
        "openjev-2B": 0.886,
        "openjev-0.8B": 0.869,
        "ModernCE": 0.909,
        "Laya": "-",
    },
    "mnli_mm": {
        "Jev 1.13.0": "-",
        "openjev-4B": 0.907,
        "openjev-2B": 0.889,
        "openjev-0.8B": 0.874,
        "ModernCE": 0.921,
        "Laya": "-",
    },
    "arc_easy_rerank": {
        "Jev 1.13.0": "~0.65",
        "openjev-4B": 0.769,
        "openjev-2B": 0.629,
        "openjev-0.8B": 0.555,
        "ModernCE": 0.607,
        "Laya": "-",
    },
    "arc_challenge_rerank": {
        "Jev 1.13.0": "~0.55",
        "openjev-4B": 0.592,
        "openjev-2B": 0.491,
        "openjev-0.8B": 0.375,
        "ModernCE": 0.416,
        "Laya": "-",
    },
    "mmlu_rerank": {
        "Jev 1.13.0": "~0.45",
        "openjev-4B": 0.472,
        "openjev-2B": 0.394,
        "openjev-0.8B": 0.351,
        "ModernCE": 0.354,
        "Laya": "-",
    },
    "winogrande_rerank": {
        "Jev 1.13.0": "~0.55",
        "openjev-4B": 0.586,
        "openjev-2B": 0.534,
        "openjev-0.8B": 0.504,
        "ModernCE": 0.569,
        "Laya": "-",
    },
    "arc_easy_grade_f1": {
        "Jev 1.13.0": "-",
        "openjev-4B": 0.986,
        "openjev-2B": 0.970,
        "openjev-0.8B": 0.951,
        "ModernCE": 0.941,
        "Laya": "-",
    },
    "arc_challenge_grade_f1": {
        "Jev 1.13.0": "-",
        "openjev-4B": 0.975,
        "openjev-2B": 0.947,
        "openjev-0.8B": 0.922,
        "ModernCE": 0.931,
        "Laya": "-",
    },
    "mmlu_grade_f1": {
        "Jev 1.13.0": "-",
        "openjev-4B": 0.949,
        "openjev-2B": 0.940,
        "openjev-0.8B": 0.909,
        "ModernCE": 0.912,
        "Laya": "-",
    },
    "ag_news": {
        "Jev 1.13.0": 0.910,
        "openjev-4B": "-",
        "openjev-2B": "-",
        "openjev-0.8B": "-",
        "ModernCE": "-",
        "Laya": 0.950,
    },
    "boolq": {
        "Jev 1.13.0": "-",
        "openjev-4B": "-",
        "openjev-2B": "-",
        "openjev-0.8B": "-",
        "ModernCE": "-",
        "Laya": 0.830,
    },
    "dair_emotion": {
        "Jev 1.13.0": 0.480,
        "openjev-4B": "-",
        "openjev-2B": "-",
        "openjev-0.8B": "-",
        "ModernCE": "-",
        "Laya": 0.595,
    },
    "p50_latency_ms": {
        "Jev 1.13.0": "236-276 ms",
        "openjev-4B": "57 ms",
        "openjev-2B": "35 ms",
        "openjev-0.8B": "22 ms",
        "ModernCE": "18 ms",
        "Laya": "32.8 ms",
    },
    "ece_score": {
        "Jev 1.13.0": 0.246,
        "openjev-4B": "~0.08",
        "openjev-2B": "~0.09",
        "openjev-0.8B": "~0.10",
        "ModernCE": "~0.07",
        "Laya": 0.081,
    },
}


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Computes Expected Calibration Error over predicted probabilities."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == labels

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(labels)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * (np.sum(in_bin) / total_samples)

    return float(ece)


def compute_brier(probs: np.ndarray, labels: np.ndarray) -> float:
    """Computes multi-class Brier score."""
    one_hot = np.zeros_like(probs)
    for i, l in enumerate(labels):
        one_hot[i, l] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def _seeded_slice(ds: Any, limit: Optional[int], seed: int = 0) -> Any:
    """Deterministic shuffled slice.

    WHY: first-N slices pick wrong distributions (MMLU's test set is subject-ordered
    so the first 100 rows are Abstract Algebra only; AG News CSVs are class-grouped).

    Example:
        ds = _seeded_slice(load_dataset("nyu-mll/multi_nli", split="validation_matched"), 100)
    """
    shuffled = ds.shuffle(seed=seed)
    if limit is None or limit >= len(shuffled):
        return shuffled
    return shuffled.select(range(limit))


def _pair_key(premise: str, hypothesis: str) -> str:
    """Content key matching data_pipeline's normalization (contamination checks)."""
    import hashlib
    norm = lambda t: " ".join(str(t).lower().split())
    return hashlib.sha1(f"{norm(premise)}\x1f{norm(hypothesis)}".encode("utf-8")).hexdigest()


_FORBIDDEN_PAIR_KEYS = None


def load_forbidden_nli_keys(data_dir: str = "data") -> set:
    """Every (premise, hypothesis) pair compiled into train/val, for benchmark exclusion.

    Example:
        forbidden = load_forbidden_nli_keys("data")
    """
    global _FORBIDDEN_PAIR_KEYS
    if _FORBIDDEN_PAIR_KEYS is None:
        keys: set = set()
        for name in ("train.jsonl", "val.jsonl"):
            path = os.path.join(data_dir, name)
            if not os.path.exists(path):
                continue
            with open(path, encoding="utf-8") as f:
                for line in f:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if row.get("premise") and row.get("hypothesis"):
                        keys.add(_pair_key(row["premise"], row["hypothesis"]))
        _FORBIDDEN_PAIR_KEYS = keys
    return _FORBIDDEN_PAIR_KEYS


def eval_mnli(ce: Gemma4CrossEncoder, split: str = "validation_matched", limit: Optional[int] = 100) -> float:
    """Evaluates NLI sanity on MNLI matched or mismatched split."""
    ds = load_dataset("nyu-mll/multi_nli", split=split)
    if limit is not None:
        ds = _seeded_slice(ds, limit)
    forbidden = load_forbidden_nli_keys()

    # Native MNLI mapping: 0=entailment, 1=neutral, 2=contradiction
    # Our schema: 0=contradiction, 1=entailment, 2=neutral
    NATIVE2OURS = {0: 1, 1: 2, 2: 0}

    pairs = []
    gold_labels = []
    excluded = 0
    for row in ds:
        p = row["premise"]
        h = row["hypothesis"]
        lbl = row["label"]
        if lbl in NATIVE2OURS:
            # WHY: never benchmark on pairs the model trained on or that were part of
            # the checkpoint-selection set (audit blocker).
            if _pair_key(p, h) in forbidden:
                excluded += 1
                continue
            pairs.append((p, h))
            gold_labels.append(NATIVE2OURS[lbl])
    if excluded:
        print(f"  [contamination-guard] excluded {excluded} {split} rows seen in train/val")

    probs = ce.predict(pairs)
    preds = np.argmax(probs, axis=1)
    acc = float(np.mean(preds == np.array(gold_labels)))
    return acc


def eval_arc_rerank(ce: Gemma4CrossEncoder, subset: str = "ARC-Easy", limit: Optional[int] = 100, scoring: str = "entailment") -> float:
    """Evaluates multiple-choice reranking without reference (Blog #3 protocol)."""
    ds = load_dataset("allenai/ai2_arc", subset, split="test")
    if limit is not None:
        ds = _seeded_slice(ds, limit)

    correct = 0
    total = 0

    for row in ds:
        question = row["question"]
        choices = row["choices"]["text"]
        labels = row["choices"]["label"]
        gold_key = row["answerKey"]

        if gold_key not in labels:
            continue

        gold_idx = labels.index(gold_key)
        pred_idx = ce.rerank(
            premise=question,
            options=choices,
            hyp_format="The correct answer is: {}",
            scoring=scoring,
        )
        if int(pred_idx) == gold_idx:
            correct += 1
        total += 1

    return float(correct / total) if total > 0 else 0.0


def eval_mmlu_rerank(ce: Gemma4CrossEncoder, limit: Optional[int] = 100, scoring: str = "entailment") -> float:
    """Evaluates MMLU multiple-choice reranking without reference."""
    ds = load_dataset("cais/mmlu", "all", split="test")
    if limit is not None:
        ds = _seeded_slice(ds, limit)

    correct = 0
    total = 0
    for row in ds:
        question = row["question"]
        choices = row["choices"]
        gold_idx = int(row["answer"])

        pred_idx = ce.rerank(
            premise=question,
            options=choices,
            hyp_format="The correct answer is: {}",
            scoring=scoring,
        )
        if int(pred_idx) == gold_idx:
            correct += 1
        total += 1

    return float(correct / total) if total > 0 else 0.0


def eval_winogrande_rerank(ce: Gemma4CrossEncoder, limit: Optional[int] = 100, scoring: str = "entailment") -> float:
    """Evaluates WinoGrande debiased reranking."""
    ds = load_dataset("allenai/winogrande", "winogrande_debiased", split="validation")
    if limit is not None:
        ds = _seeded_slice(ds, limit)

    correct = 0
    total = 0
    for row in ds:
        sentence = row["sentence"]
        opt1 = row["option1"]
        opt2 = row["option2"]
        answer = row["answer"]

        gold_idx = 0 if answer == "1" else 1
        cand1 = sentence.replace("_", opt1)
        cand2 = sentence.replace("_", opt2)

        pred_idx = ce.rerank(
            premise="Statement verification:",
            options=[cand1, cand2],
            hyp_format="This statement is correct and logical: {}",
            scoring=scoring,
        )
        if int(pred_idx) == gold_idx:
            correct += 1
        total += 1

    return float(correct / total) if total > 0 else 0.0


def eval_grading_with_reference(
    ce: Gemma4CrossEncoder, dataset_name: str = "ARC-Easy", limit: Optional[int] = 100
) -> Tuple[float, float]:
    """Evaluates grading with reference (Blog #6 protocol):
    Premise: {question}\nReference answer: {gold}
    Hypothesis: Answer: {option}
    Entailment <=> option is gold.
    Returns: (Accuracy, F1).
    """
    if dataset_name == "ARC-Easy":
        ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    elif dataset_name == "ARC-Challenge":
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    elif dataset_name == "MMLU":
        ds = load_dataset("cais/mmlu", "all", split="test")
    else:
        raise ValueError(f"Unknown grading dataset: {dataset_name}")

    if limit is not None:
        ds = _seeded_slice(ds, limit)

    y_true = []
    y_pred = []

    pairs = []
    ground_truths = []

    for row in ds:
        question = row["question"]
        if "choices" in row and isinstance(row["choices"], dict):
            choices = row["choices"]["text"]
            labels = row["choices"]["label"]
            gold_key = row["answerKey"]
            if gold_key not in labels:
                continue
            gold_idx = labels.index(gold_key)
        else:
            choices = row["choices"]
            gold_idx = int(row["answer"])

        gold_text = choices[gold_idx]
        ref_premise = f"{question}\nReference answer: {gold_text}"

        for idx, option in enumerate(choices):
            pairs.append((ref_premise, f"Answer: {option}"))
            ground_truths.append(1 if idx == gold_idx else 0)

    # Batched inference
    probs = ce.predict(pairs)
    # Binary classification: Entailment (1) vs Non-entailment (0 or 2)
    # Model predicts positive if P(entailment) is highest or > threshold
    preds = np.argmax(probs, axis=1)
    binary_preds = [1 if p == ENTAILMENT else 0 for p in preds]

    y_true = np.array(ground_truths)
    y_pred = np.array(binary_preds)

    acc = float(np.mean(y_true == y_pred))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return acc, float(f1)


def eval_boolq(ce: Gemma4CrossEncoder, limit: Optional[int] = 100, scoring: str = "entailment") -> float:
    """Evaluates BoolQ binary question answering as zero-shot reranking."""
    ds = load_dataset("google/boolq", split="validation")
    if limit is not None:
        ds = _seeded_slice(ds, limit)

    correct = 0
    total = 0
    for row in ds:
        question = row["question"]
        passage = row["passage"]
        answer = row["answer"]  # True or False

        gold_idx = 0 if answer is True else 1
        options = ["Yes, this is true.", "No, this is false."]
        pred_idx = ce.rerank(
            premise=f"Passage: {passage}\nQuestion: {question}?",
            options=options,
            hyp_format="The factual answer is: {}",
            scoring=scoring,
        )
        if int(pred_idx) == gold_idx:
            correct += 1
        total += 1

    return float(correct / total) if total > 0 else 0.0


def eval_ag_news(ce: Gemma4CrossEncoder, limit: Optional[int] = 100, scoring: str = "entailment") -> float:
    """Evaluates AG News 4-topic classification."""
    ds = load_dataset("fancyzhx/ag_news", split="test")
    if limit is not None:
        ds = _seeded_slice(ds, limit)

    options = [
        "World news, foreign affairs, and international politics.",
        "Sports competitions, athletic teams, and games.",
        "Business, corporate earnings, stock markets, and economy.",
        "Science, computer technology, software, and engineering.",
    ]

    correct = 0
    total = 0
    for row in ds:
        text = row["text"]
        gold_idx = int(row["label"])

        pred_idx = ce.rerank(
            premise=f"News Article: {text}",
            options=options,
            hyp_format="The primary topic of this article is: {}",
            scoring=scoring,
        )
        if int(pred_idx) == gold_idx:
            correct += 1
        total += 1

    return float(correct / total) if total > 0 else 0.0


def eval_dair_emotion(ce: Gemma4CrossEncoder, limit: Optional[int] = 100, scoring: str = "entailment") -> float:
    """Evaluates DAIR Emotion 6-class classification."""
    ds = load_dataset("dair-ai/emotion", split="test")
    if limit is not None:
        ds = _seeded_slice(ds, limit)

    # Labels: 0: sadness, 1: joy, 2: love, 3: anger, 4: fear, 5: surprise
    options = [
        "sadness and sorrow",
        "joy and happiness",
        "love and affection",
        "anger and irritation",
        "fear and anxiety",
        "surprise and astonishment",
    ]

    correct = 0
    total = 0
    for row in ds:
        text = row["text"]
        gold_idx = int(row["label"])

        pred_idx = ce.rerank(
            premise=f"Statement: {text}",
            options=options,
            hyp_format="The author feels {}.",
            scoring=scoring,
        )
        if int(pred_idx) == gold_idx:
            correct += 1
        total += 1

    return float(correct / total) if total > 0 else 0.0


def eval_latency_and_calibration(
    ce: Gemma4CrossEncoder, n_samples: int = 100
) -> Tuple[Dict[str, float], float, float]:
    """Measures single-decision latency percentiles (P50, P90, P99) and calibration on MNLI."""
    ds = load_dataset("nyu-mll/multi_nli", split="validation_matched")
    ds = _seeded_slice(ds, n_samples)
    forbidden = load_forbidden_nli_keys()

    NATIVE2OURS = {0: 1, 1: 2, 2: 0}
    pairs = []
    gold_labels = []
    excluded = 0
    for row in ds:
        lbl = row["label"]
        if lbl in NATIVE2OURS:
            if _pair_key(row["premise"], row["hypothesis"]) in forbidden:
                excluded += 1
                continue
            pairs.append((row["premise"], row["hypothesis"]))
            gold_labels.append(NATIVE2OURS[lbl])
    if excluded:
        print(f"  [contamination-guard] excluded {excluded} latency/ECE rows seen in train/val")

    # Warmup
    for _ in range(5):
        _ = ce.predict([pairs[0]])

    latencies = []
    all_probs = []

    for pair in pairs:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        prob = ce.predict([pair])
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)
        all_probs.append(prob[0])

    probs_arr = np.array(all_probs)
    labels_arr = np.array(gold_labels)

    latencies.sort()
    latency_stats = {
        "p50_ms": float(np.percentile(latencies, 50)),
        "p90_ms": float(np.percentile(latencies, 90)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "mean_ms": float(np.mean(latencies)),
    }

    ece = compute_ece(probs_arr, labels_arr)
    brier = compute_brier(probs_arr, labels_arr)

    return latency_stats, ece, brier


def print_comparison_table(results: Dict[str, Any]):
    """Formats and prints a comprehensive comparison table against Jev, OpenJEV, Laya, and ModernCE."""
    print("\n" + "=" * 115)
    print("DIRECT HEAD-TO-HEAD CAPABILITY BENCHMARK: GEMMA 4 E2B W4A16 vs JEV / OPENJEV / LAYA")
    print("=" * 115)

    header = f"{'Benchmark Task / Metric':<32} | {'Jev 1.13.0':<12} | {'openjev-4B':<12} | {'openjev-2B':<12} | {'ModernCE':<10} | {'Laya':<10} | {'Gemma 4 W4A16':<12}"
    print(header)
    print("-" * 115)

    display_names = [
        ("mnli_m", "MNLI Matched (Accuracy)"),
        ("mnli_mm", "MNLI Mismatched (Accuracy)"),
        ("arc_easy_rerank", "ARC-Easy Rerank (0-shot)"),
        ("arc_challenge_rerank", "ARC-Challenge Rerank"),
        ("mmlu_rerank", "MMLU Rerank (0-shot)"),
        ("winogrande_rerank", "WinoGrande Rerank"),
        ("arc_easy_grade_f1", "ARC-Easy Grade (F1)"),
        ("arc_challenge_grade_f1", "ARC-Challenge Grade (F1)"),
        ("mmlu_grade_f1", "MMLU Grade (F1)"),
        ("ag_news", "AG News (4 topics)"),
        ("boolq", "BoolQ (Yes/No Q&A)"),
        ("dair_emotion", "DAIR Emotion (6 classes)"),
        ("p50_latency_ms", "Single-Forward Latency P50"),
        ("ece_score", "ECE Calibration (lower better)"),
    ]

    for key, name in display_names:
        refs = REFERENCE_BENCHMARKS.get(key, {})
        jev_val = str(refs.get("Jev 1.13.0", "-"))
        oj4_val = str(refs.get("openjev-4B", "-"))
        oj2_val = str(refs.get("openjev-2B", "-"))
        mod_val = str(refs.get("ModernCE", "-"))
        laya_val = str(refs.get("Laya", "-"))

        our_raw = results.get(key, None)
        if our_raw is None:
            our_val = "N/A"
        elif isinstance(our_raw, float):
            our_val = f"{our_raw:.4f}"
        else:
            our_val = str(our_raw)

        print(f"{name:<32} | {jev_val:<12} | {oj4_val:<12} | {oj2_val:<12} | {mod_val:<10} | {laya_val:<10} | \033[1;32m{our_val:<12}\033[0m")

    print("=" * 115)


def main():
    parser = argparse.ArgumentParser(description="Direct Capability Benchmark: Gemma 4 vs Jev / OpenJEV / Laya")
    parser.add_argument("--model-path", default="./ckpt/gemma-4-e2b-nli-w4a16", help="Model checkpoint path")
    parser.add_argument("--limit", type=int, default=50, help="Number of examples per task (default: 50)")
    parser.add_argument("--full", action="store_true", help="Run full evaluation without limits")
    parser.add_argument("--scoring", default="margin", choices=["entailment", "margin", "contrastive"], help="Scoring rule: margin (recommended for 0-shot reranking) or entailment (raw Jev)")
    parser.add_argument("--tasks", default="all", help="Comma-separated list of tasks to run or 'all'")
    parser.add_argument("--out-file", default="results/benchmark_comparison.json", help="Path to save results JSON")
    args = parser.parse_args()

    limit = None if args.full else args.limit
    task_filter = set(args.tasks.split(",")) if args.tasks != "all" else None

    print(f"Loading cross-encoder from: {args.model_path}")
    print(f"Scoring rule for reranking: {args.scoring}")
    ce = Gemma4CrossEncoder(args.model_path, device="cuda")

    results: Dict[str, Any] = {}
    results["scoring_rule"] = args.scoring
    results["provenance"] = {
        "model_path": args.model_path,
        "limit_per_task": limit,
        "shuffle_seed": 0,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "competitor_columns": "quoted published reference constants (REFERENCE_BENCHMARKS), not local runs",
        "scoring_note": "margin/contrastive are post-hoc variants; openjev's documented protocol is raw entailment",
        "contamination_guard": "MNLI/latency/ECE slices exclude pairs present in data/train.jsonl or data/val.jsonl",
    }

    def should_run(task_name: str) -> bool:
        if task_filter is None:
            return True
        return task_name in task_filter or "all" in task_filter

    # 1. NLI Sanity
    if should_run("mnli") or should_run("mnli_m"):
        print(f"\n[1/5] Evaluating MNLI Matched (limit={limit})...")
        results["mnli_m"] = eval_mnli(ce, split="validation_matched", limit=limit)
        print(f" -> MNLI Matched Accuracy: {results['mnli_m']*100:.2f}%")

    if should_run("mnli") or should_run("mnli_mm"):
        print(f"[1/5] Evaluating MNLI Mismatched (limit={limit})...")
        results["mnli_mm"] = eval_mnli(ce, split="validation_mismatched", limit=limit)
        print(f" -> MNLI Mismatched Accuracy: {results['mnli_mm']*100:.2f}%")

    # 2. Multiple Choice Reranking (without reference)
    if should_run("rerank") or should_run("arc_easy"):
        print(f"\n[2/5] Evaluating ARC-Easy Rerank (limit={limit}, scoring={args.scoring})...")
        results["arc_easy_rerank"] = eval_arc_rerank(ce, subset="ARC-Easy", limit=limit, scoring=args.scoring)
        print(f" -> ARC-Easy Accuracy: {results['arc_easy_rerank']*100:.2f}%")

    if should_run("rerank") or should_run("arc_challenge"):
        print(f"[2/5] Evaluating ARC-Challenge Rerank (limit={limit}, scoring={args.scoring})...")
        results["arc_challenge_rerank"] = eval_arc_rerank(ce, subset="ARC-Challenge", limit=limit, scoring=args.scoring)
        print(f" -> ARC-Challenge Accuracy: {results['arc_challenge_rerank']*100:.2f}%")

    if should_run("rerank") or should_run("mmlu"):
        print(f"[2/5] Evaluating MMLU Rerank (limit={limit}, scoring={args.scoring})...")
        results["mmlu_rerank"] = eval_mmlu_rerank(ce, limit=limit, scoring=args.scoring)
        print(f" -> MMLU Accuracy: {results['mmlu_rerank']*100:.2f}%")

    if should_run("rerank") or should_run("winogrande"):
        print(f"[2/5] Evaluating WinoGrande Rerank (limit={limit}, scoring={args.scoring})...")
        results["winogrande_rerank"] = eval_winogrande_rerank(ce, limit=limit, scoring=args.scoring)
        print(f" -> WinoGrande Accuracy: {results['winogrande_rerank']*100:.2f}%")

    # 3. Grading with Reference (with gold answer)
    if should_run("grading") or should_run("arc_easy_grade"):
        print(f"\n[3/5] Evaluating ARC-Easy Grading with Reference (limit={limit})...")
        acc, f1 = eval_grading_with_reference(ce, "ARC-Easy", limit=limit)
        results["arc_easy_grade_acc"] = acc
        results["arc_easy_grade_f1"] = f1
        print(f" -> ARC-Easy Grade Acc: {acc*100:.2f}%, F1: {f1*100:.2f}%")

    if should_run("grading") or should_run("arc_challenge_grade"):
        print(f"[3/5] Evaluating ARC-Challenge Grading with Reference (limit={limit})...")
        acc, f1 = eval_grading_with_reference(ce, "ARC-Challenge", limit=limit)
        results["arc_challenge_grade_acc"] = acc
        results["arc_challenge_grade_f1"] = f1
        print(f" -> ARC-Challenge Grade Acc: {acc*100:.2f}%, F1: {f1*100:.2f}%")

    if should_run("grading") or should_run("mmlu_grade"):
        print(f"[3/5] Evaluating MMLU Grading with Reference (limit={limit})...")
        acc, f1 = eval_grading_with_reference(ce, "MMLU", limit=limit)
        results["mmlu_grade_acc"] = acc
        results["mmlu_grade_f1"] = f1
        print(f" -> MMLU Grade Acc: {acc*100:.2f}%, F1: {f1*100:.2f}%")

    # 4. Laya Decision Benchmarks
    if should_run("laya") or should_run("boolq"):
        print(f"\n[4/5] Evaluating BoolQ Decision (limit={limit}, scoring={args.scoring})...")
        results["boolq"] = eval_boolq(ce, limit=limit, scoring=args.scoring)
        print(f" -> BoolQ Accuracy: {results['boolq']*100:.2f}%")

    if should_run("laya") or should_run("ag_news"):
        print(f"[4/5] Evaluating AG News 4-class routing (limit={limit}, scoring={args.scoring})...")
        results["ag_news"] = eval_ag_news(ce, limit=limit, scoring=args.scoring)
        print(f" -> AG News Accuracy: {results['ag_news']*100:.2f}%")

    if should_run("laya") or should_run("dair_emotion"):
        print(f"[4/5] Evaluating DAIR Emotion 6-class routing (limit={limit}, scoring={args.scoring})...")
        results["dair_emotion"] = eval_dair_emotion(ce, limit=limit, scoring=args.scoring)
        print(f" -> DAIR Emotion Accuracy: {results['dair_emotion']*100:.2f}%")

    # 5. Latency & Calibration
    if should_run("latency") or should_run("calibration"):
        print(f"\n[5/5] Measuring Forward Latency & Calibration on RTX 5090...")
        lat_stats, ece, brier = eval_latency_and_calibration(ce, n_samples=min(100, limit or 100))
        results["p50_latency_ms"] = f"{lat_stats['p50_ms']:.2f} ms"
        results["latency_stats"] = lat_stats
        results["ece_score"] = ece
        results["brier_score"] = brier
        print(f" -> Latency P50: {lat_stats['p50_ms']:.2f} ms, P90: {lat_stats['p90_ms']:.2f} ms, P99: {lat_stats['p99_ms']:.2f} ms")
        print(f" -> ECE: {ece:.4f}, Brier: {brier:.4f}")

    # Print comparative table
    print_comparison_table(results)

    # Save to JSON
    os.makedirs(os.path.dirname(args.out_file) or ".", exist_ok=True)
    with open(args.out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nBenchmark results saved to: {args.out_file}")


if __name__ == "__main__":
    main()
