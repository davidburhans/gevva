#!/usr/bin/env python3
"""eval_system1_suite.py
========================
Unified Evaluation and Pareto-Frontier Benchmark Suite for System 1 Decision Engines.

Evaluates any `System1Engine` across:
1. Decision Accuracy (Overall, Clean, and Trap/Tradeoff subsets).
2. Position Bias Index (PBI) and Decision Flip Rate (measuring sensitivity to option permutation).
3. Expected Calibration Error (ECE) and Multi-Class Brier Score.
4. Pure Forward Latency Profiling (P50, P90, P99, Throughput).
5. Fast Mode (single pass) vs Robust Mode (cyclic ensembled).

Outputs:
- JSON artifact with full provenance and per-task metrics.
- Comparative Markdown table displaying the Quality-vs-Latency Pareto Frontier.

CLI (fully offline by default):
    CUDA_VISIBLE_DEVICES= uv run python eval_system1_suite.py
    CUDA_VISIBLE_DEVICES= uv run python eval_system1_suite.py --engine cross_encoder \
        --model-path ./ckpt/gemma-4-e2b-nli-w4a16 --test-file tests/fixtures/system1_sample_decisions.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from data_engine import CanonicalDecision, CanonicalDecisionDataset
from decision_engine import DecisionResult, JudgeResult, RateResult, RerankResult, System1Engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_TEST_FILE = "tests/fixtures/system1_sample_decisions.jsonl"
DEFAULT_OUTPUT_PATH = "results/system1_eval_results.json"
ENGINE_CHOICES = ("mock-invariant", "cross_encoder", "causal", "set_attention")


class MockInvariantEngine(System1Engine):
    """Offline content-based mock engine that is invariant to option order.

    Heuristic: always predicts the option with the longest text; ties at the maximum
    length are broken by the lexicographically smallest TEXT, never by position. The
    positional first-max tie-break (np.argmax) used to leak presentation order into
    best_option whenever several options tied at the maximum length. Fully
    deterministic (no RNG) and permutation-invariant, so the default CLI run
    exercises the full evaluation suite (PBI, flip rate, calibration, latency)
    without any network or GPU.
    """

    @property
    def model_name(self) -> str:
        return "mock-invariant-engine"

    @property
    def paradigm(self) -> str:
        return "mock"

    def decide(self, context: str, question: str, options: Sequence[str], mode: str = "fast", **kwargs: Any) -> DecisionResult:
        opt_texts = list(options)
        k = len(opt_texts)
        lens = [len(opt) for opt in opt_texts]
        # Content-based tie-break: among maximum-length options, lexicographically
        # smallest text wins. Rotating the presentation then cannot change best_option,
        # because the winning (length, text) pair is a property of the option SET.
        max_len = max(lens)
        best_text = min(opt for opt, length in zip(opt_texts, lens) if length == max_len)
        best_idx = opt_texts.index(best_text)
        probs = {opt: (0.9 if i == best_idx else 0.1 / max(1, k - 1)) for i, opt in enumerate(opt_texts)}
        return DecisionResult(
            best_option=opt_texts[best_idx],
            best_index=best_idx,
            probabilities=probs,
            scores=[float(l) for l in lens],
        )

    def judge(self, context: str, claim: str, **kwargs: Any) -> JudgeResult:
        return JudgeResult(verdict="entailment", probabilities={"entailment": 0.9, "contradiction": 0.05, "neutral": 0.05})

    def rate(self, context: str, rubric_levels: Sequence[str], level_values: Optional[Sequence[float]] = None, **kwargs: Any) -> RateResult:
        return RateResult(expected_score=1.0, level_probabilities={l: 1.0 / len(rubric_levels) for l in rubric_levels}, predicted_level=rubric_levels[0])

    def rerank(self, query: str, documents: Sequence[str], **kwargs: Any) -> RerankResult:
        return RerankResult(ranked_indices=list(range(len(documents))), items=[])


def _position_bias_index(
    slot_counts: Counter,
    eligible_ks: Counter,
    eligible: int,
) -> Tuple[float, Dict[int, float]]:
    """RMS gap between observed slot shares and the mixed-K expected shares.

    expected_count[slot] = sum over evaluated items with k > slot of 1/k; both the
    observed and expected counts are normalized over slots 0..Kmax-1.

    WHY exact integer expected counts: accumulating 1/k per item in floats leaves
    rounding dust (e.g. 30 x (1/3) != 10.0), so a perfectly uniform engine would
    score PBI ~ 1e-17 instead of exactly 0.0. Scaling every 1/k by the LCM of all
    eligible K keeps the expectation integral until the single final normalization
    division (IEEE division is correctly rounded, so equal rationals give equal
    doubles and the uniform case collapses to an exact 0.0).
    """
    if eligible == 0:
        return 0.0, {}
    kmax = max(eligible_ks)
    common = math.lcm(*sorted(eligible_ks))
    denom = common * eligible
    pbi_squared_sum = 0.0
    expected_distribution: Dict[int, float] = {}
    for slot in range(kmax):
        expected_num = sum(count * (common // k) for k, count in eligible_ks.items() if k > slot)
        observed_share = slot_counts[slot] / eligible
        expected_share = expected_num / denom
        gap = observed_share - expected_share
        pbi_squared_sum += gap * gap
        expected_distribution[slot] = expected_share
    pbi = math.sqrt(pbi_squared_sum / kmax)
    return float(pbi), expected_distribution


def compute_pbi_and_flip_rate(
    engine: System1Engine,
    items: Sequence[CanonicalDecision],
    max_items: int = 50,
    mode: str = "fast",
) -> Tuple[float, float, Dict[str, Any]]:
    """Calculates the Position Bias Index (PBI) and Decision Flip Rate.

    Flip Rate: fraction of ELIGIBLE items (num_options > 1) whose predicted option
               changes when the presentation order is cyclically shifted. Items with
               k <= 1 have no permutation to flip and are excluded from both the slot
               statistics and the flip-rate denominator.
    PBI: RMS gap between the observed base-presentation slot distribution and the
         mixed-K correct expectation (see `_position_bias_index`).
         Perfectly uniform engine -> PBI == 0.0 exactly.
         Skewed distribution (e.g. always picking 'A') -> PBI is high (>0.15).

    Serving parity: every decide() call forwards option_keys taken from the
    presentation's OWN option dicts — the base presentation uses the item's original
    keys, and each cyclic shift uses the positionally reassigned keys from
    `CanonicalDecision.to_cyclic_permutations`, mirroring the training permutation
    curriculum whose choice hypotheses are formatted as
    'The correct answer is: {key}: {desc}' (SERVING_CHOICE_TEMPLATE,
    research/adapters/typed_decisions_adapter.py). Engines key result probabilities
    on the ORIGINAL option texts, so best_option / slot statistics remain valid.
    """
    sample = items[:max_items]
    if not sample:
        return 0.0, 0.0, {}

    flips = 0
    eligible = 0
    slot_counts: Counter = Counter()
    eligible_ks: Counter = Counter()

    for item in sample:
        k = item.num_options
        if k <= 1:
            continue
        eligible += 1
        eligible_ks[k] += 1

        base_res = engine.decide(
            context=item.context,
            question=item.question,
            options=[o["text"] for o in item.options],
            option_keys=[o["key"] for o in item.options],
            mode=mode,
        )
        base_choice = base_res.best_option
        slot_counts[base_res.best_index] += 1

        # Test on cyclic shifts (shift 0 is the base presentation itself)
        item_flipped = False
        for shifted_item in item.to_cyclic_permutations(num_shifts=k)[1:]:
            s_res = engine.decide(
                context=shifted_item.context,
                question=shifted_item.question,
                options=[o["text"] for o in shifted_item.options],
                option_keys=[o["key"] for o in shifted_item.options],
                mode=mode,
            )
            # Check if chosen option text differs from base choice
            if s_res.best_option != base_choice:
                item_flipped = True
                break

        if item_flipped:
            flips += 1

    flip_rate = flips / eligible if eligible else 0.0
    pbi, expected_distribution = _position_bias_index(slot_counts, eligible_ks, eligible)

    meta = {
        "num_tested": len(sample),
        "num_eligible": eligible,
        "flips": flips,
        "flip_rate": flip_rate,
        "pbi": pbi,
        "slot_distribution": dict(sorted(slot_counts.items())),
        "expected_slot_distribution": {str(k): v for k, v in sorted(expected_distribution.items())},
    }
    return pbi, flip_rate, meta


def evaluate_decision_dataset(
    engine: System1Engine,
    dataset: Sequence[CanonicalDecision],
    mode: str = "fast",
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Runs standard evaluation over a CanonicalDecision dataset.

    Serving parity: each decide() call forwards option_keys from the item's OWN
    option dicts so cross-encoder hypotheses are formatted as
    'The correct answer is: {key}: {desc}' — the SERVING_CHOICE_TEMPLATE pattern the
    training curriculum actually teaches — instead of the unrepresented
    'The correct answer is: {text}'. Result probabilities stay keyed on the original
    option texts, which the accuracy / Brier bookkeeping below relies on.
    """
    items = dataset[:limit] if limit is not None else dataset
    total = len(items)
    if total == 0:
        return {"accuracy": 0.0, "total": 0}

    correct = 0
    trap_correct = 0
    trap_total = 0
    clean_correct = 0
    clean_total = 0

    confidences = []
    accuracies = []
    brier_scores = []
    latencies = []

    for item in items:
        opts = [o["text"] for o in item.options]
        t0 = time.perf_counter()
        res = engine.decide(
            context=item.context,
            question=item.question,
            options=opts,
            option_keys=[o["key"] for o in item.options],
            mode=mode,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(elapsed_ms)

        is_hit = (res.best_index == item.gold_index)
        if is_hit:
            correct += 1

        if item.is_trap:
            trap_total += 1
            if is_hit:
                trap_correct += 1
        else:
            clean_total += 1
            if is_hit:
                clean_correct += 1

        top_conf = res.top_confidence
        confidences.append(top_conf)
        accuracies.append(1.0 if is_hit else 0.0)

        # Multi-class Brier score
        prob_vec = np.array([res.probabilities.get(o["text"], 0.0) for o in item.options])
        if prob_vec.sum() > 0:
            prob_vec = prob_vec / prob_vec.sum()
        gold_one_hot = np.zeros_like(prob_vec)
        if 0 <= item.gold_index < len(gold_one_hot):
            gold_one_hot[item.gold_index] = 1.0
        brier = float(np.sum((prob_vec - gold_one_hot) ** 2))
        brier_scores.append(brier)

    acc = correct / total
    trap_acc = (trap_correct / trap_total) if trap_total > 0 else None
    clean_acc = (clean_correct / clean_total) if clean_total > 0 else None
    mean_brier = float(np.mean(brier_scores)) if brier_scores else 0.0

    # Expected Calibration Error (10 bins)
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    conf_arr = np.array(confidences)
    acc_arr = np.array(accuracies)

    for i in range(n_bins):
        b_low, b_high = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (conf_arr > b_low) & (conf_arr <= b_high)
        if np.any(mask):
            bin_acc = np.mean(acc_arr[mask])
            bin_conf = np.mean(conf_arr[mask])
            prop = np.mean(mask)
            ece += np.abs(bin_conf - bin_acc) * prop

    lat_p50 = float(np.percentile(latencies, 50)) if latencies else 0.0
    lat_p90 = float(np.percentile(latencies, 90)) if latencies else 0.0
    lat_p99 = float(np.percentile(latencies, 99)) if latencies else 0.0

    return {
        "accuracy": acc,
        "trap_accuracy": trap_acc,
        "clean_accuracy": clean_acc,
        "brier_score": mean_brier,
        "ece": float(ece),
        "total_samples": total,
        "trap_samples": trap_total,
        "latency_p50_ms": lat_p50,
        "latency_p90_ms": lat_p90,
        "latency_p99_ms": lat_p99,
        "throughput_per_sec": 1000.0 / max(1e-3, lat_p50),
    }


def run_full_suite(
    engine: System1Engine,
    dataset: Sequence[CanonicalDecision],
    limit: Optional[int] = 100,
    modes: Sequence[str] = ("fast", "robust"),
) -> Dict[str, Any]:
    """Executes full evaluation across fast and robust modes and calculates Pareto metrics."""
    logger.info(f"Running System 1 Evaluation Suite on [{engine.model_name}] (Paradigm: {engine.paradigm})")
    results = {
        "model_name": engine.model_name,
        "paradigm": engine.paradigm,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "modes": {},
    }

    for mode in modes:
        logger.info(f"Evaluating mode='{mode}'...")
        eval_metrics = evaluate_decision_dataset(engine, dataset, mode=mode, limit=limit)
        pbi, flip_rate, pbi_meta = compute_pbi_and_flip_rate(engine, dataset, max_items=min(50, len(dataset)), mode=mode)

        eval_metrics["position_bias_index"] = pbi
        eval_metrics["decision_flip_rate"] = flip_rate
        eval_metrics["pbi_meta"] = pbi_meta

        results["modes"][mode] = eval_metrics
        logger.info(
            f"[{mode.upper()}] Acc: {eval_metrics['accuracy']*100:.2f}% | "
            f"Trap Acc: {(eval_metrics['trap_accuracy'] or 0.0)*100:.2f}% | "
            f"PBI: {pbi:.4f} | Flip Rate: {flip_rate*100:.1f}% | "
            f"P50 Latency: {eval_metrics['latency_p50_ms']:.2f} ms"
        )

    return results


def format_pareto_markdown_table(suite_results: List[Dict[str, Any]]) -> str:
    """Formats comparative results into a clean markdown table showing Quality vs Latency."""
    lines = [
        "| Model Architecture | Mode | Accuracy | Trap Acc | Flip Rate | PBI | P50 Latency | Throughput | ECE |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in suite_results:
        m_name = f"{r['model_name']} ({r['paradigm']})"
        for mode_name, m in r.get("modes", {}).items():
            t_acc = f"{m['trap_accuracy']*100:.1f}%" if m["trap_accuracy"] is not None else "N/A"
            lines.append(
                f"| {m_name} | `{mode_name}` | **{m['accuracy']*100:.1f}%** | {t_acc} | "
                f"{m['decision_flip_rate']*100:.1f}% | {m['position_bias_index']:.4f} | "
                f"{m['latency_p50_ms']:.1f} ms | {m['throughput_per_sec']:.1f}/s | {m['ece']:.4f} |"
            )
    return "\n".join(lines)


def build_engine(engine_name: str, model_path: Optional[str]) -> System1Engine:
    """Builds the requested engine for the CLI.

    The mock engine is fully offline. Real engines import lazily and load strictly
    from a LOCAL checkpoint directory: `--model-path` is validated to exist on disk
    before any construction, so the suite can never trigger a Hugging Face Hub
    download (offline-safe failure instead).
    """
    if engine_name == "mock-invariant":
        return MockInvariantEngine()

    if not model_path:
        raise SystemExit(
            f"--engine {engine_name!r} requires --model-path pointing to a local checkpoint "
            f"directory; got model_path={model_path!r}"
        )
    if not os.path.isdir(model_path):
        raise SystemExit(
            f"--model-path {model_path!r} is not a local directory; real engines load weights "
            "from disk and this suite runs offline (no Hugging Face Hub downloads)"
        )

    try:
        if engine_name == "cross_encoder":
            from gemma4_cross_encoder import Gemma4CrossEncoder

            return Gemma4CrossEncoder(model_name_or_path=model_path, device="cpu")
        if engine_name == "causal":
            from gemma4_causal_decider import Gemma4CausalDecider

            return Gemma4CausalDecider(model_path_or_name=model_path, device="cpu")
        if engine_name == "set_attention":
            from gemma4_cross_encoder import Gemma4CrossEncoder
            from gemma4_set_attention import Gemma4SetAttentionDecider

            backbone = Gemma4CrossEncoder(model_name_or_path=model_path, device="cpu")
            return Gemma4SetAttentionDecider(model_path_or_name=model_path, device="cpu", backbone=backbone)
    except Exception as exc:
        raise SystemExit(
            f"Failed to build engine={engine_name!r} from model_path={model_path!r}: {exc}"
        ) from exc
    raise SystemExit(f"Unknown engine {engine_name!r}; expected one of {list(ENGINE_CHOICES)}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point: evaluate, write the JSON artifact, print the markdown table."""
    parser = argparse.ArgumentParser(description="System 1 Decision Engine Evaluation Suite")
    parser.add_argument("--engine", choices=ENGINE_CHOICES, default="mock-invariant",
                        help="Engine paradigm to evaluate (default: offline mock-invariant)")
    parser.add_argument("--model-path", default=None,
                        help="Local checkpoint directory (required for real engines; never downloads)")
    parser.add_argument("--test-file", default=DEFAULT_TEST_FILE,
                        help=f"Path to a CanonicalDecision JSONL dataset (default: {DEFAULT_TEST_FILE})")
    parser.add_argument("--limit", type=int, default=None, help="Max test items per evaluation (default: all)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH,
                        help=f"Output JSON path (default: {DEFAULT_OUTPUT_PATH})")
    args = parser.parse_args(argv)

    engine = build_engine(args.engine, args.model_path)
    dataset = CanonicalDecisionDataset(args.test_file)

    results = run_full_suite(engine, dataset, limit=args.limit)

    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(format_pareto_markdown_table([results]))
    print(f"\nResults written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
