#!/usr/bin/env python3
"""tests/test_token_bucket_lengths.py - Regression tests for the stage-3 OOM fixes.

Covers the 2026-09-21 incident: chars//4 length estimates undercounted dense
synthetic haystack text 2-4x, and unbucketed val/test loaders packed 16
truncated 128K-haystack rows (262K tokens) into single eval forwards.

Run: uv run python tests/test_token_bucket_lengths.py
"""

import json
import random
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from finetune import TokenBucketBatchSampler  # noqa: E402
from train_cross_encoder import (  # noqa: E402
    TEMPLATE_TOKEN_OVERHEAD,
    compute_token_lengths,
)

TOKENIZER_DIR = Path(__file__).resolve().parent.parent / "ckpt" / "gemma-4-e2b-nli-w4a16-stage2"

# Dense synthetic haystack noise: citation ids, serial numbers, decimals.
# ~2-3 chars/token for Gemma's tokenizer vs the 4 chars/token heuristic.
DENSE_PREMISE = (
    "REF-2026-114592; batch #77-0031-a; vals 3.14159, 2.71828, 1.61803; "
    "ids QK7-XJ2/9L4, MN-3310/88214; 2024-09-21T04:17:03Z; 41.8781N 87.6298W "
) * 20
PLAIN_PREMISE = "The quarterly report was published after the board meeting concluded. "

_TOKENIZER = None


def get_tokenizer():
    """Lazily load the offline Gemma tokenizer from the exported W4A16 checkpoint."""
    global _TOKENIZER
    if _TOKENIZER is None:
        if not (TOKENIZER_DIR / "tokenizer.json").exists():
            raise RuntimeError(f"local tokenizer fixture missing: {TOKENIZER_DIR} (export W4A16 stage2 first)")
        from transformers import AutoTokenizer

        _TOKENIZER = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    return _TOKENIZER


def test_dense_text_exceeds_chars_per_4_heuristic():
    """The old estimate must undercount dense synthetic text (regression motive)."""
    tokenizer = get_tokenizer()
    row = {"premise": DENSE_PREMISE, "hypothesis": "claim " * 10}
    true_len = compute_token_lengths([row], tokenizer, 1_000_000)[0]
    old_heuristic = (len(row["premise"]) + len(row["hypothesis"])) // 4 + 16
    assert true_len > old_heuristic * 1.5, (
        f"dense text should tokenize >1.5x the chars//4 estimate: true={true_len} old={old_heuristic}"
    )


def test_plain_text_estimate_is_close():
    """Natural prose stays near the old heuristic (sanity bound, not brittle)."""
    tokenizer = get_tokenizer()
    row = {"premise": PLAIN_PREMISE, "hypothesis": "A report exists."}
    true_len = compute_token_lengths([row], tokenizer, 1_000_000)[0]
    old_heuristic = (len(row["premise"]) + len(row["hypothesis"])) // 4 + 16
    assert abs(true_len - old_heuristic) < max(64, true_len * 0.5)


def test_lengths_capped_at_max_length():
    tokenizer = get_tokenizer()
    rows = [{"premise": DENSE_PREMISE, "hypothesis": "h" * 500}]
    lengths = compute_token_lengths(rows, tokenizer, 512)
    assert lengths == [512]


def test_overhead_bound_covers_template():
    """TEMPLATE_TOKEN_OVERHEAD must upper-bound the real wrapper tokens."""
    tokenizer = get_tokenizer()
    row = {"premise": "x", "hypothesis": "y"}
    true_len = compute_token_lengths([row], tokenizer, 1_000_000)[0]
    n_premise = len(tokenizer("x", add_special_tokens=False)["input_ids"])
    n_hypothesis = len(tokenizer("y", add_special_tokens=False)["input_ids"])
    assert true_len == TEMPLATE_TOKEN_OVERHEAD + n_premise + n_hypothesis
    assert TEMPLATE_TOKEN_OVERHEAD >= 16  # "Premise: \nHypothesis: \nPrediction:"


def _make_rows(n):
    rng = random.Random(0)
    rows = []
    for i in range(n):
        text = DENSE_PREMISE if rng.choice(["dense", "plain"]) == "dense" else PLAIN_PREMISE
        rows.append({"premise": text[: rng.randint(100, 4000)], "hypothesis": f"claim {i}"})
    return rows


def test_bucket_batches_respect_token_budget():
    """Every batch fits the budget unless a single row exceeds it (bucket cap > budget)."""
    tokenizer = get_tokenizer()
    rows = _make_rows(300)
    lengths = compute_token_lengths(rows, tokenizer, 16_384)
    budget = 8192
    sampler = TokenBucketBatchSampler(lengths, max_tokens_per_batch=budget, shuffle=True, seed=42)
    for batch in sampler:
        total = sum(lengths[i] for i in batch)
        if len(batch) == 1 and lengths[batch[0]] > budget:
            continue  # single over-budget row: unavoidable, gets its own batch
        assert total <= budget, f"batch of {len(batch)} rows holds {total} tokens > {budget}"


def test_eval_sampler_is_deterministic_without_shuffle():
    """shuffle=False iteration order must be identical across passes (val/test)."""
    tokenizer = get_tokenizer()
    rows = _make_rows(120)
    lengths = compute_token_lengths(rows, tokenizer, 16_384)
    batches_a = [list(b) for b in TokenBucketBatchSampler(lengths, max_tokens_per_batch=4096, shuffle=False)]
    batches_b = [list(b) for b in TokenBucketBatchSampler(lengths, max_tokens_per_batch=4096, shuffle=False)]
    assert batches_a == batches_b
    assert sorted(i for b in batches_a for i in b) == list(range(len(rows)))


def test_collator_id_fallback_unique_under_reordering():
    """Rows lacking an 'id' must get dataset-stable fallback ids (batch-local i collides)."""
    from train_cross_encoder import DataCollatorNLI, NLIDataset

    tmp = Path(__file__).parent / "fixtures" / "tmp_no_id_rows.jsonl"
    tmp.parent.mkdir(exist_ok=True)
    rows = [{"premise": f"p{i}", "hypothesis": f"h{i}", "label": 0, "source": "t"} for i in range(6)]
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    ds = NLIDataset(str(tmp))
    collator = DataCollatorNLI(get_tokenizer(), max_length=64)
    # Bucket-style reordering: batch rows [5, 2] then [0, 1, 3, 4].
    batch_a = collator([ds[5], ds[2]])
    batch_b = collator([ds[0], ds[1], ds[3], ds[4]])
    ids = list(batch_a["ids"]) + list(batch_b["ids"])
    assert len(set(ids)) == 6, f"fallback ids must be unique across reordered batches, got {ids}"
    tmp.unlink()


TESTS = [
    test_dense_text_exceeds_chars_per_4_heuristic,
    test_plain_text_estimate_is_close,
    test_lengths_capped_at_max_length,
    test_overhead_bound_covers_template,
    test_bucket_batches_respect_token_budget,
    test_eval_sampler_is_deterministic_without_shuffle,
    test_collator_id_fallback_unique_under_reordering,
]


def main() -> int:
    failures = 0
    for test in TESTS:
        try:
            test()
            print(f"PASS  {test.__name__}")
        except Exception:
            failures += 1
            print(f"FAIL  {test.__name__}")
            traceback.print_exc()
    print(f"\n{len(TESTS) - failures}/{len(TESTS)} tests passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
