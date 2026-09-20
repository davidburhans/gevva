#!/usr/bin/env python3
"""eval_downstream_decisions.py
==============================
Comprehensive evaluation and demonstration of the fine-tuned Gemma 4 Cross-Encoder
as a high-throughput System 1 Decision Engine.

Evaluates 4 core capabilities:
1. Standard 3-Class NLI Calibration & Accuracy
2. Zero-Shot Tool & Intent Routing (Non-autoregressive candidate selection)
3. RAG Hallucination Detection & Document Grounding
4. Multilingual Zero-Shot Transfer across 5 distinct language families
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer
from peft import PeftModel

from gemma4_cross_encoder import (
    CONTRADICTION,
    ENTAILMENT,
    NEUTRAL,
    ID2LABEL,
    LABEL2ID,
    Gemma4CrossEncoder,
    Gemma4ForSequenceClassification,
)


import os

def load_fine_tuned_cross_encoder(
    base_model_id: str,
    adapter_path: str,
    qat: bool = False,
    qat_bits: int = 4,
    qat_group_size: int = 32,
    device: str = "cuda",
):
    print(f"Loading base model {base_model_id}...")
    config = AutoConfig.from_pretrained(base_model_id)
    config.num_labels = 3
    base_model = Gemma4ForSequenceClassification.from_pretrained(
        base_model_id,
        config=config,
        torch_dtype=torch.bfloat16,
    )

    if qat:
        from gemma4_cross_encoder import apply_quantization_aware_training
        print(f"Applying QAT {qat_bits}-bit (group_size={qat_group_size}) to base model for QAT evaluation...")
        base_model = apply_quantization_aware_training(
            base_model, num_bits=qat_bits, group_size=qat_group_size
        )

    print(f"Loading LoRA adapter from {adapter_path}...")
    peft_model = PeftModel.from_pretrained(base_model, adapter_path)

    # Restore explicit classification head and norm weights if saved
    head_weights_path = os.path.join(adapter_path, "head_weights.pt")
    if os.path.exists(head_weights_path):
        print(f"Loading explicit classification head weights from {head_weights_path}...")
        head_dict = torch.load(head_weights_path, map_location="cpu", weights_only=True)
        raw = peft_model.base_model.model if hasattr(peft_model, "base_model") else peft_model
        if "score" in head_dict and hasattr(raw, "score"):
            target_score = raw.score.modules_to_save["default"] if (hasattr(raw.score, "modules_to_save") and "default" in raw.score.modules_to_save) else raw.score
            if "weight" in head_dict["score"]:
                target_score.load_state_dict(head_dict["score"])
            else:
                raw.score.load_state_dict(head_dict["score"])
        if "norm" in head_dict and hasattr(raw, "norm"):
            target_norm = raw.norm.modules_to_save["default"] if (hasattr(raw.norm, "modules_to_save") and "default" in raw.norm.modules_to_save) else raw.norm
            if "weight" in head_dict["norm"]:
                target_norm.load_state_dict(head_dict["norm"])
            else:
                raw.norm.load_state_dict(head_dict["norm"])

    peft_model.to(device)
    peft_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    return Gemma4CrossEncoder(model=peft_model, tokenizer=tokenizer, device=device)


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Computes Expected Calibration Error (ECE) across confidence bins."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            acc_in_bin = np.mean(accuracies[in_bin])
            conf_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(acc_in_bin - conf_in_bin) * prop_in_bin
    return float(ece)


def compute_multiclass_brier(probs: np.ndarray, labels: np.ndarray) -> float:
    """Computes Multi-Class Brier Score: mean squared probability error across all classes."""
    n_classes = probs.shape[1]
    one_hot = np.eye(n_classes)[labels]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def evaluate_standard_nli_calibration(
    encoder: Gemma4CrossEncoder, val_path: str = "data/val.jsonl", max_samples: Optional[int] = 1000
):
    print("\n" + "=" * 60)
    print("0. Standard 3-Class NLI Calibration & Accuracy Evaluation")
    print("=" * 60)

    if not os.path.exists(val_path):
        print(f"Validation file {val_path} not found. Skipping standard NLI evaluation.")
        return

    records = []
    with open(val_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
            if max_samples and len(records) >= max_samples:
                break

    print(f"Evaluating {len(records):,} validation samples from {val_path}...")
    pairs = [(r["premise"], r["hypothesis"]) for r in records]
    golds = np.array([r["label"] for r in records])
    sources = [r.get("source", "unknown") for r in records]

    probs = encoder.predict(pairs)
    preds = np.argmax(probs, axis=-1)

    acc = float(np.mean(preds == golds))
    ece = compute_ece(probs, golds)
    brier = compute_multiclass_brier(probs, golds)

    print(f"Overall Validation Accuracy: {acc * 100:.2f}%")
    print(f"Expected Calibration Error (ECE): {ece:.4f}")
    print(f"Multi-Class Brier Score:          {brier:.4f}")

    # Per-Class Precision, Recall, F1
    from collections import Counter, defaultdict
    print("\nPer-Class Breakdown:")
    for c_id, c_name in ID2LABEL.items():
        tp = int(np.sum((preds == c_id) & (golds == c_id)))
        fp = int(np.sum((preds == c_id) & (golds != c_id)))
        fn = int(np.sum((preds != c_id) & (golds == c_id)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        support = int(np.sum(golds == c_id))
        print(f"  [{c_name:13s}] Prec: {prec:.3f} | Rec: {rec:.3f} | F1: {f1:.3f} (support={support})")

    # Source breakdown
    by_src = defaultdict(list)
    for p, g, s in zip(preds, golds, sources):
        by_src[s].append(p == g)
    print("\nAccuracy by Data Source:")
    for s, accs in sorted(by_src.items()):
        print(f"  - {s:25s}: {np.mean(accs) * 100:.2f}% (n={len(accs)})")


def benchmark_system_1_latency(encoder: Gemma4CrossEncoder, n_runs: int = 50):
    print("\n" + "=" * 60)
    print("1. Benchmarking System 1 Decision Latency (GPU Forward Pass)")
    print("=" * 60)

    pairs = [("The weather today in Seattle is rainy and 55 degrees.", "Seattle is experiencing precipitation.")]

    # Warmup with explicit CUDA synchronization
    for _ in range(5):
        encoder.predict(pairs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    latencies_e2e = []
    gpu_forward_times = []

    # Prepare tokenized batch once to isolate pure GPU forward pass
    batch = encoder._prepare_batch(pairs)
    input_ids = batch["input_ids"].to(encoder.device)
    attn_mask = batch["attention_mask"].to(encoder.device)

    for _ in range(n_runs):
        # 1. Pure GPU forward pass using CUDA events
        if torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            with torch.no_grad():
                encoder.model(input_ids=input_ids, attention_mask=attn_mask)
            end_event.record()
            torch.cuda.synchronize()
            gpu_forward_times.append(start_event.elapsed_time(end_event))

        # 2. End-to-End latency (Text in -> Softmax probabilities out)
        t0 = time.perf_counter()
        encoder.predict(pairs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latencies_e2e.append((time.perf_counter() - t0) * 1000)

    p50_gpu = np.percentile(gpu_forward_times, 50) if gpu_forward_times else 0.0
    p50_e2e = np.percentile(latencies_e2e, 50)
    p95_e2e = np.percentile(latencies_e2e, 95)
    print(f"Pure GPU Forward Pass Latency:  P50 = {p50_gpu:.2f} ms")
    print(f"End-to-End Inference Latency:   P50 = {p50_e2e:.2f} ms | P95 = {p95_e2e:.2f} ms")
    print(f"Throughput: {1000 / p50_e2e:.1f} decisions/second per GPU stream")


def evaluate_zero_shot_tool_routing(encoder: Gemma4CrossEncoder):
    print("\n" + "=" * 60)
    print("2. Zero-Shot Intent & Tool Routing (1-Pass Classification)")
    print("=" * 60)

    tools = [
        "Execute a wire transfer or bank transaction.",
        "Check weather forecast for a location.",
        "Search web for latest news articles.",
        "Query code repository and symbol index.",
        "Generate or edit an image from text description."
    ]

    test_queries = [
        ("Send $350 from my checking account to Bob for dinner.", 0),
        ("What will the temperature be like tomorrow in Tokyo?", 1),
        ("Who won the men 100m sprint in the recent world athletics championship?", 2),
        ("Where is the Gemma4ClippableLinear class defined in the codebase?", 3),
        ("Render an oil painting of an astronaut riding a red bicycle on Mars.", 4),
    ]

    correct = 0
    for query, expected_tool_idx in test_queries:
        best_idx, scores = encoder.rerank(
            premise=f"User request: {query}",
            options=tools,
            hyp_format="The appropriate tool to handle this request is: {}",
        )
        is_correct = (best_idx == expected_tool_idx)
        if is_correct:
            correct += 1
        print(f"\nQuery: \"{query}\"")
        print(f"  Selected Tool: [{best_idx}] {tools[best_idx]} (Entailment P: {scores[best_idx]:.4f})")
        print(f"  Expected Tool: [{expected_tool_idx}] {tools[expected_tool_idx]} -> {'PASS' if is_correct else 'FAIL'}")

    acc = correct / len(test_queries)
    print(f"\nTool Routing Accuracy: {acc * 100:.1f}% ({correct}/{len(test_queries)})")


def evaluate_rag_hallucination_detection(encoder: Gemma4CrossEncoder):
    print("\n" + "=" * 60)
    print("3. RAG Hallucination Detection & Citation Grounding")
    print("=" * 60)

    test_cases = [
        {
            "context": "The Apollo 11 mission was launched on July 16, 1969, carrying commander Neil Armstrong, command module pilot Michael Collins, and lunar module pilot Buzz Aldrin.",
            "claim": "Neil Armstrong was the commander of the Apollo 11 mission.",
            "expected": "entailment",
            "desc": "Factual Claim Supported by Document",
        },
        {
            "context": "The Apollo 11 mission was launched on July 16, 1969, carrying commander Neil Armstrong, command module pilot Michael Collins, and lunar module pilot Buzz Aldrin.",
            "claim": "The Apollo 11 mission was launched in August 1972.",
            "expected": "contradiction",
            "desc": "Hallucinated Date (Direct Contradiction)",
        },
        {
            "context": "The Apollo 11 mission was launched on July 16, 1969, carrying commander Neil Armstrong, command module pilot Michael Collins, and lunar module pilot Buzz Aldrin.",
            "claim": "Neil Armstrong had oatmeal for breakfast before the launch.",
            "expected": "neutral",
            "desc": "Unverifiable Claim (Neutral Evidence)",
        },
    ]

    correct = 0
    for tc in test_cases:
        probs = encoder.predict([(tc["context"], tc["claim"])])[0]
        pred_label = ID2LABEL[int(np.argmax(probs))]
        is_correct = (pred_label == tc["expected"])
        if is_correct:
            correct += 1
        print(f"\nScenario: {tc['desc']}")
        print(f"  Claim: \"{tc['claim']}\"")
        print(f"  Expected: {tc['expected'].upper()} | Predicted: {pred_label.upper()} ({'PASS' if is_correct else 'FAIL'})")
        print(f"  Probabilities: Con={probs[0]:.4f}, Ent={probs[1]:.4f}, Neu={probs[2]:.4f}")

    print(f"\nHallucination Detection Accuracy: {correct / len(test_cases) * 100:.1f}%")


def evaluate_multilingual_grounding(encoder: Gemma4CrossEncoder):
    print("\n" + "=" * 60)
    print("4. Multilingual Zero-Shot Grounding")
    print("=" * 60)

    multilingual_tests = [
        # German
        ("Ein Hund läuft über die grüne Wiese.", "Ein Tier bewegt sich im Freien.", "entailment", "German (DE)"),
        ("Ein Hund läuft über die grüne Wiese.", "Die Katze schläft auf dem Sofa.", "contradiction", "German (DE)"),
        # Spanish
        ("Dos niños juegan al fútbol en el parque.", "Los niños están jugando deportes al aire libre.", "entailment", "Spanish (ES)"),
        ("Dos niños juegan al fútbol en el parque.", "El parque está completamente vacío.", "contradiction", "Spanish (ES)"),
        # French
        ("Le chef prépare un repas gastronomique dans la cuisine.", "Quelqu'un cuisine de la nourriture.", "entailment", "French (FR)"),
        # Chinese
        ("一位年轻的女子在图书馆里看书。", "有人在室内阅读。", "entailment", "Chinese (ZH)"),
        ("一位年轻的女子在图书馆里看书。", "图书馆里没有任何人。", "contradiction", "Chinese (ZH)"),
    ]

    correct = 0
    for premise, hyp, expected, lang in multilingual_tests:
        probs = encoder.predict([(premise, hyp)])[0]
        pred = ID2LABEL[int(np.argmax(probs))]
        is_correct = (pred == expected)
        if is_correct:
            correct += 1
        print(f"[{lang}] P: \"{premise}\" | H: \"{hyp}\"")
        print(f"  -> Predicted: {pred.upper()} (Expected: {expected.upper()}) | {'PASS' if is_correct else 'FAIL'}")

    print(f"\nMultilingual Accuracy: {correct / len(multilingual_tests) * 100:.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=None, help="Direct path to model directory (e.g. W4A16 packed model or adapter directory)")
    parser.add_argument("--base-model", default="google/gemma-4-E2B", help="Base model ID")
    parser.add_argument("--adapter-path", default="./ckpt/gemma-4-e2b-nli-qat-stage1/best", help="Trained adapter path")
    parser.add_argument("--val-path", default="data/val.jsonl", help="Validation dataset path for calibration benchmark")
    parser.add_argument("--max-val-samples", type=int, default=1000, help="Max validation samples to evaluate")
    parser.add_argument("--qat", action="store_true", default=False, help="Simulate QAT quantization on base model")
    parser.add_argument("--qat-bits", type=int, default=4, help="QAT weight bit-width (default: 4)")
    parser.add_argument("--qat-group-size", type=int, default=32, help="QAT group size (default: 32)")
    args = parser.parse_args()

    if args.model_path:
        print(f"Loading cross-encoder directly from {args.model_path}...")
        encoder = Gemma4CrossEncoder(args.model_path, device="cuda")
    else:
        encoder = load_fine_tuned_cross_encoder(
            args.base_model,
            args.adapter_path,
            qat=args.qat,
            qat_bits=args.qat_bits,
            qat_group_size=args.qat_group_size,
        )
    evaluate_standard_nli_calibration(encoder, val_path=args.val_path, max_samples=args.max_val_samples)
    benchmark_system_1_latency(encoder)
    evaluate_zero_shot_tool_routing(encoder)
    evaluate_rag_hallucination_detection(encoder)
    evaluate_multilingual_grounding(encoder)


if __name__ == "__main__":
    main()
