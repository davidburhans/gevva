import hashlib
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List

import torch

from data_engine import (
    MAX_STANDARD_OPTIONS,
    CanonicalDecision,
    InContextPermutationCollator,
)
from research.adapters.grouped_decision_collator import (
    GroupedDecisionCollator,
    GroupedTokenBucketBatchSampler,
    compute_cross_option_loss,
)


class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    bos_token_id = 2
    padding_side = "right"

    def __call__(self, text, add_special_tokens=True, return_attention_mask=False):
        # Deterministic dummy tokenization
        tokens = [ord(c) % 100 + 3 for c in text[:64]]
        return {"input_ids": tokens}

    def encode(self, text, add_special_tokens=True):
        return [ord(c) % 100 + 3 for c in text[:64]]


def test_compute_cross_option_loss_hard():
    # 2 groups: Group 0 has 3 options (gold = option 1), Group 1 has 2 options (gold = option 0)
    # 1 un-grouped item: Group -1
    group_ids = torch.tensor([0, 0, 0, 1, 1, -1])
    is_gold = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 0.0])

    # Case A: Model predicts gold with high confidence
    scores_good = torch.tensor([0.1, 5.0, 0.2, 4.0, 0.1, 1.0], requires_grad=True)
    loss_good, m_good = compute_cross_option_loss(scores_good, group_ids, is_gold)
    assert loss_good.item() < 0.2
    assert m_good["xopt_accuracy"] == 1.0
    assert m_good["n_groups"] == 2

    # Gradients flow back to scores
    loss_good.backward()
    assert scores_good.grad is not None
    assert scores_good.grad[1] < 0.0  # Gold logit pushed UP
    assert scores_good.grad[0] > 0.0  # Distractor pushed DOWN
    assert scores_good.grad[5] == 0.0  # Un-grouped item has zero gradient from xopt

    # Case B: Model predicts wrong distractor
    scores_bad = torch.tensor([5.0, 0.1, 0.2, 0.1, 4.0, 1.0], requires_grad=True)
    loss_bad, m_bad = compute_cross_option_loss(scores_bad, group_ids, is_gold)
    assert loss_bad.item() > 3.0
    assert m_bad["xopt_accuracy"] == 0.0


def test_compute_cross_option_loss_soft_targets():
    # 1 group of 3 options with soft teacher distribution [0.1, 0.7, 0.2]
    group_ids = torch.tensor([0, 0, 0])
    is_gold = torch.tensor([0.0, 1.0, 0.0])
    soft_targets = torch.tensor([0.1, 0.7, 0.2])

    scores = torch.tensor([1.0, 2.0, 1.5], requires_grad=True)
    loss, m = compute_cross_option_loss(scores, group_ids, is_gold, soft_targets=soft_targets)
    assert loss.item() > 0.0
    assert m["n_groups"] == 1
    loss.backward()
    assert scores.grad is not None


def _decision_options(k: int) -> List[dict]:
    """k options with unique keys and unique texts."""
    return [{"key": f"K{i}", "text": f"option text {i}"} for i in range(k)]


def test_canonical_decision_rejects_too_many_options():
    opts = _decision_options(MAX_STANDARD_OPTIONS + 1)
    try:
        CanonicalDecision(id="overfull", domain="t", context="c", question="q", options=opts, gold_index=0)
    except ValueError as exc:
        assert "21" in str(exc) and str(MAX_STANDARD_OPTIONS) in str(exc), str(exc)
    else:
        raise AssertionError(f"expected ValueError for {len(opts)} options")


def test_canonical_decision_rejects_gold_index_out_of_bounds():
    opts = _decision_options(3)
    for bad_gold in (-1, 3, 7):
        try:
            CanonicalDecision(id="oob", domain="t", context="c", question="q", options=opts, gold_index=bad_gold)
        except ValueError as exc:
            assert str(bad_gold) in str(exc) and "[0, 2]" in str(exc), str(exc)
        else:
            raise AssertionError(f"expected ValueError for gold_index={bad_gold}")


def test_canonical_decision_rejects_duplicate_option_texts():
    opts = [
        {"key": "K0", "text": "same text"},
        {"key": "K1", "text": "same text"},
        {"key": "K2", "text": "different text"},
    ]
    try:
        CanonicalDecision(id="dupes", domain="t", context="c", question="q", options=opts, gold_index=0)
    except ValueError as exc:
        assert "same text" in str(exc), str(exc)
    else:
        raise AssertionError("expected ValueError for duplicate option texts")


def test_canonical_decision_gold_key_derived_safely():
    # gold_key derived from the gold option's own key, never raw key-table indexing
    opts = [{"key": "X1", "text": "a"}, {"key": "X2", "text": "b"}]
    dec = CanonicalDecision(id="safekey", domain="t", context="c", question="q", options=opts, gold_index=1)
    assert dec.gold_key == "X2"


def test_from_dict_id_is_deterministic():
    d = {
        "domain": "t",
        "context": "c",
        "question": "q",
        "options": [{"key": "A", "text": "a"}, {"key": "B", "text": "b"}],
        "gold_index": 1,
    }
    first = CanonicalDecision.from_dict(d)
    second = CanonicalDecision.from_dict(dict(d))
    expected_hash = hashlib.sha1(json.dumps(d, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:10]
    assert first.id == second.id == f"decision_{expected_hash}"
    # Explicit ids still win over the content hash
    assert CanonicalDecision.from_dict({**d, "id": "custom"}).id == "custom"


def test_to_cyclic_permutations_rejects_non_positive_shifts():
    dec = CanonicalDecision(
        id="shifts", domain="t", context="c", question="q",
        options=_decision_options(3), gold_index=0,
    )
    try:
        dec.to_cyclic_permutations(num_shifts=0)
    except ValueError as exc:
        assert "num_shifts=0" in str(exc), str(exc)
    else:
        raise AssertionError("expected ValueError for num_shifts=0")


class BosAwareTokenizer:
    """Deterministic mock tokenizer that prepends BOS like real causal tokenizers."""

    pad_token_id = 0
    eos_token_id = 1
    bos_token_id = 2
    padding_side = "right"

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        tokens = [ord(c) % 100 + 3 for c in text]
        return ([self.bos_token_id] + tokens) if add_special_tokens else tokens


def _collator_decision(context: str) -> CanonicalDecision:
    return CanonicalDecision(
        id="collator_dec", domain="t", context=context, question="Which option holds?",
        options=[{"key": "A", "text": "first"}, {"key": "B", "text": "second"}],
        gold_index=1,
    )


def test_in_context_collator_same_epoch_determinism():
    tok = BosAwareTokenizer()
    items = [_collator_decision(f"context number {i} " * 3) for i in range(4)]
    collator = InContextPermutationCollator(tok, max_length=256, permute_training=True, seed=11)

    collator.set_epoch(3)
    first = collator(items)
    collator.set_epoch(3)  # reseed to the same epoch -> identical permutation stream
    second = collator(items)

    assert torch.equal(first["input_ids"], second["input_ids"])
    assert torch.equal(first["labels"], second["labels"])
    assert first["gold_indices"].tolist() == second["gold_indices"].tolist()


def test_in_context_collator_preserves_bos_after_truncation():
    tok = BosAwareTokenizer()
    long_decision = _collator_decision("long context token " * 200)
    collator = InContextPermutationCollator(tok, max_length=64, permute_training=False)
    out = collator([long_decision])

    ids = out["input_ids"][0].tolist()
    labels = out["labels"][0].tolist()

    assert len(ids) <= 64, f"sequence exceeded max_length: {len(ids)}"
    assert ids[0] == tok.bos_token_id, f"BOS was stripped by left-truncation, first id={ids[0]}"

    supervised = [l for l in labels if l != -100]
    assert len(supervised) == 1, f"expected exactly one supervised token, got {supervised}"
    # The surviving supervised token is the gold decision slot 'B' (gold_index=1)
    assert supervised[0] == collator.key_to_token_id["B"]
    assert out["gold_indices"].tolist() == [1]


def test_grouped_token_bucket_sampler():
    records = [
        {"group_id": "q1", "premise": "p1", "hypothesis": "h1"},
        {"group_id": "q1", "premise": "p1", "hypothesis": "h2"},
        {"group_id": "q1", "premise": "p1", "hypothesis": "h3"},
        {"group_id": "q2", "premise": "p2", "hypothesis": "h1"},
        {"group_id": "q2", "premise": "p2", "hypothesis": "h2"},
        {"group_id": "", "premise": "p_single", "hypothesis": "h_single"},
    ]
    lengths = [100, 100, 100, 200, 200, 50]

    # Max tokens = 500: group q1 (300 tokens) cannot be split, group q2 (400 tokens) cannot be split
    sampler = GroupedTokenBucketBatchSampler(records, lengths, max_tokens_per_batch=450, shuffle=False)
    batches = list(sampler)

    # Verify that all indices of q1 [0, 1, 2] are in the exact same batch
    for b in batches:
        q1_in_b = [idx in b for idx in [0, 1, 2]]
        assert all(q1_in_b) or not any(q1_in_b), f"q1 was split across batches! Batch: {b}"

        q2_in_b = [idx in b for idx in [3, 4]]
        assert all(q2_in_b) or not any(q2_in_b), f"q2 was split across batches! Batch: {b}"


def _sampler_fixture():
    records = [
        {"group_id": "q1", "premise": "p1", "hypothesis": "h1"},
        {"group_id": "q1", "premise": "p1", "hypothesis": "h2"},
        {"group_id": "q1", "premise": "p1", "hypothesis": "h3"},
        {"group_id": "q2", "premise": "p2", "hypothesis": "h1"},
        {"group_id": "q2", "premise": "p2", "hypothesis": "h2"},
        {"group_id": "", "premise": "p_single", "hypothesis": "h_single"},
        {"group_id": "q3", "premise": "p3", "hypothesis": "h1"},
        {"group_id": "q3", "premise": "p3", "hypothesis": "h2"},
        {"group_id": "q3", "premise": "p3", "hypothesis": "h3"},
    ]
    lengths = [100, 100, 100, 200, 200, 50, 80, 80, 80]
    return records, lengths


def _assert_group_atomicity(batches, records):
    """Every record index appears exactly once and groups are never split."""
    flat = sorted(idx for batch in batches for idx in batch)
    assert flat == list(range(len(records))), f"coverage mismatch: {flat}"
    groups = {}
    for idx, r in enumerate(records):
        gid = r["group_id"]
        if gid:
            groups.setdefault(gid, []).append(idx)
    for batch in batches:
        batch_set = set(batch)
        for gid, indices in groups.items():
            inside = sum(1 for i in indices if i in batch_set)
            assert inside in (0, len(indices)), f"group {gid} split across batches: {batch}"


def test_sampler_len_matches_yielded_batch_count():
    records, lengths = _sampler_fixture()
    for shuffle in (False, True):
        sampler = GroupedTokenBucketBatchSampler(
            records, lengths, max_tokens_per_batch=450, shuffle=shuffle, seed=7,
        )
        batches = list(sampler)
        assert len(sampler) == len(batches), \
            f"len={len(sampler)} != yielded={len(batches)} (shuffle={shuffle})"
        _assert_group_atomicity(batches, records)


def test_sampler_batch_composition_is_epoch_invariant():
    records, lengths = _sampler_fixture()
    sampler = GroupedTokenBucketBatchSampler(
        records, lengths, max_tokens_per_batch=450, shuffle=True, seed=7,
    )
    sampler.set_epoch(0)
    epoch0 = list(sampler)
    sampler.set_epoch(1)
    epoch1 = list(sampler)

    # Same composition every epoch; only the order of batches may differ
    assert sorted(map(sorted, epoch0)) == sorted(map(sorted, epoch1))
    assert len(epoch0) == len(epoch1) == len(sampler)
    _assert_group_atomicity(epoch1, records)


def test_sampler_never_splits_oversized_group():
    # One group whose token footprint alone exceeds the bucket: it must still
    # arrive as a single atomic batch (atomicity outranks the token budget).
    records = [{"group_id": "big", "premise": "p"} for _ in range(4)]
    lengths = [500, 500, 500, 500]
    sampler = GroupedTokenBucketBatchSampler(records, lengths, max_tokens_per_batch=1000, shuffle=False)
    batches = list(sampler)
    assert batches == [[0, 1, 2, 3]], f"oversized group was split: {batches}"
    assert len(sampler) == 1


def test_grouped_decision_collator():
    tok = DummyTokenizer()
    collator = GroupedDecisionCollator(tok, max_length=128, is_gemma=False)

    batch_items = [
        {"group_id": "qA", "premise": "The car is blue.", "hypothesis": "The correct answer is: blue", "is_gold": True, "label": 1},
        {"group_id": "qA", "premise": "The car is blue.", "hypothesis": "The correct answer is: red", "is_gold": False, "label": 0},
        {"group_id": "qB", "premise": "It rained today.", "hypothesis": "The correct answer is: yes", "is_gold": True, "label": 1},
        {"group_id": "qB", "premise": "It rained today.", "hypothesis": "The correct answer is: no", "is_gold": False, "label": 0},
        {"group_id": None, "premise": "A man sleeps.", "hypothesis": "A person rests.", "is_gold": True, "label": 1},
    ]

    out = collator(batch_items)
    assert out["input_ids"].shape[0] == 5
    assert out["attention_mask"].shape[0] == 5
    assert out["labels"].shape[0] == 5
    assert out["group_ids"].shape[0] == 5
    assert out["is_gold"].shape[0] == 5

    # Check that qA mapped to same ID, qB to another ID, and None to -1
    g_ids = out["group_ids"].tolist()
    assert g_ids[0] == g_ids[1] >= 0
    assert g_ids[2] == g_ids[3] >= 0
    assert g_ids[0] != g_ids[2]
    assert g_ids[4] == -1


if __name__ == "__main__":
    test_compute_cross_option_loss_hard()
    test_compute_cross_option_loss_soft_targets()
    test_grouped_token_bucket_sampler()
    test_sampler_len_matches_yielded_batch_count()
    test_sampler_batch_composition_is_epoch_invariant()
    test_sampler_never_splits_oversized_group()
    test_grouped_decision_collator()
    test_canonical_decision_rejects_too_many_options()
    test_canonical_decision_rejects_gold_index_out_of_bounds()
    test_canonical_decision_rejects_duplicate_option_texts()
    test_canonical_decision_gold_key_derived_safely()
    test_from_dict_id_is_deterministic()
    test_to_cyclic_permutations_rejects_non_positive_shifts()
    test_in_context_collator_same_epoch_determinism()
    test_in_context_collator_preserves_bos_after_truncation()
    print("All grouped decision collator tests passed successfully!")
