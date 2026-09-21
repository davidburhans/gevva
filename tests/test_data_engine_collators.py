"""Offline CPU tests for data_engine.py collators.

Covers:
1. CanonicalDecision.shifted() single-shift equivalence vs to_cyclic_permutations().
2. SetAttentionCollator set_adjacency_mask semantics (grouped batch).
3. SetAttentionCollator rejects all-ungrouped batches (fully-masked attention NaN guard).
4. SetAttentionCollator isolates ungrouped rows in mixed batches.

Run: CUDA_VISIBLE_DEVICES= uv run python tests/test_data_engine_collators.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Any, Dict, List

import torch

from data_engine import CanonicalDecision, SetAttentionCollator


class DummyTokenizer:
    """Same deterministic mock pattern as tests/test_grouped_collator.py."""

    pad_token_id = 0
    eos_token_id = 1
    bos_token_id = 2
    padding_side = "right"

    def __call__(self, text, add_special_tokens=True, return_attention_mask=False):
        tokens = [ord(c) % 100 + 3 for c in text[:64]]
        return {"input_ids": tokens}

    def encode(self, text, add_special_tokens=True):
        return [ord(c) % 100 + 3 for c in text[:64]]


def _soft_decision(k: int = 3) -> CanonicalDecision:
    """Decision with a soft teacher distribution so shifted() must rotate it."""
    options = [{"key": f"K{i}", "text": f"option text {i}"} for i in range(k)]
    return CanonicalDecision(
        id="soft_shift_dec", domain="t", context="c", question="q",
        options=options, gold_index=1,
        soft_distribution=[0.1, 0.7, 0.2][:k],
    )


def test_shifted_matches_to_cyclic_permutations_for_every_shift():
    """Regression: InContextPermutationCollator used to materialize all K
    cyclic permutations just to index one; shifted() must be byte-identical
    to to_cyclic_permutations()[shift] on the semantic fields."""
    dec = _soft_decision(k=3)
    k = dec.num_options
    perms = dec.to_cyclic_permutations(num_shifts=k)
    assert len(perms) == k, f"expected {k} permutations, got {len(perms)}"

    for shift in range(k):
        direct = dec.shifted(shift)
        expected = perms[shift]
        assert direct.options == expected.options, \
            f"shift={shift}: options {direct.options} != {expected.options}"
        assert direct.gold_index == expected.gold_index, \
            f"shift={shift}: gold_index {direct.gold_index} != {expected.gold_index}"
        assert direct.gold_key == expected.gold_key, \
            f"shift={shift}: gold_key {direct.gold_key!r} != {expected.gold_key!r}"
        assert direct.soft_distribution == expected.soft_distribution, \
            f"shift={shift}: soft_distribution {direct.soft_distribution} != {expected.soft_distribution}"
        assert direct.id == expected.id, f"shift={shift}: id {direct.id!r} != {expected.id!r}"
        assert direct.metadata == expected.metadata, \
            f"shift={shift}: metadata {direct.metadata} != {expected.metadata}"


def test_shifted_rejects_out_of_range_shift():
    dec = _soft_decision(k=3)
    for bad_shift in (-1, 3, 7):
        try:
            dec.shifted(bad_shift)
        except ValueError as exc:
            assert str(bad_shift) in str(exc) and "[0, 2]" in str(exc), str(exc)
        else:
            raise AssertionError(f"expected ValueError for shift={bad_shift}")


def _pair_record(group_id: Any, tag: str) -> Dict[str, Any]:
    return {"group_id": group_id, "premise": f"premise {tag}", "hypothesis": f"hypothesis {tag}", "is_gold": True}


def _expected_adjacency(gids: List[int]) -> torch.Tensor:
    """mask[i, j] = (gids[i] == gids[j]) and both >= 0."""
    n = len(gids)
    return torch.tensor(
        [[gids[i] >= 0 and gids[i] == gids[j] for j in range(n)] for i in range(n)],
        dtype=torch.bool,
    )


def test_set_attention_adjacency_mask_on_grouped_batch():
    tok = DummyTokenizer()
    collator = SetAttentionCollator(tok, max_length=64)
    batch = [
        _pair_record("qA", "a gold"),
        _pair_record("qA", "a distractor"),
        _pair_record("qB", "b gold"),
        _pair_record("qB", "b distractor"),
    ]
    out = collator(batch)

    mask = out["set_adjacency_mask"]
    n = len(batch)
    assert mask.shape == (n, n), f"expected ({n}, {n}) mask, got {tuple(mask.shape)}"
    assert mask.dtype == torch.bool, f"expected bool mask, got {mask.dtype}"

    gids = out["group_ids"].tolist()
    assert gids[0] == gids[1] >= 0 and gids[2] == gids[3] >= 0 and gids[0] != gids[2], f"gids={gids}"
    assert torch.equal(mask, _expected_adjacency(gids)), \
        f"mask mismatch for gids={gids}:\n{mask.to(dtype=torch.int)}"
    # Group-mates (including the diagonal) see each other; other groups do not
    assert mask[0, 1] and mask[1, 0] and mask[2, 3] and mask[3, 2]
    assert not mask[0, 2] and not mask[1, 3]


def test_set_attention_rejects_all_ungrouped_batch():
    """Regression: an all-ungrouped batch yields an all-False adjacency mask,
    and an nn.TransformerEncoder consuming a fully-masked src_key_padding_mask
    produces NaN scores — the collator must reject the batch up front."""
    tok = DummyTokenizer()
    collator = SetAttentionCollator(tok, max_length=64)
    batch = [
        _pair_record(None, "clean nli"),
        _pair_record(-1, "another clean nli"),
    ]
    try:
        collator(batch)
    except ValueError as exc:
        assert "at least one grouped decision" in str(exc), str(exc)
    else:
        raise AssertionError(f"expected ValueError for all-ungrouped batch, got keys={batch}")


def test_set_attention_isolates_ungrouped_rows_in_mixed_batch():
    tok = DummyTokenizer()
    collator = SetAttentionCollator(tok, max_length=64)
    batch = [
        _pair_record("qA", "a gold"),
        _pair_record("qA", "a distractor"),
        _pair_record(None, "clean nli"),
        _pair_record("qB", "b gold"),
        _pair_record(-1, "another clean nli"),
    ]
    out = collator(batch)

    mask = out["set_adjacency_mask"]
    gids = out["group_ids"].tolist()
    assert torch.equal(mask, _expected_adjacency(gids)), \
        f"mask mismatch for gids={gids}:\n{mask.to(dtype=torch.int)}"
    # Each ungrouped row must be fully isolated: no True in its row or column
    for i, gid in enumerate(gids):
        if gid == -1:
            assert not mask[i, :].any(), f"ungrouped row {i} has adjacency: {mask[i, :].tolist()}"
            assert not mask[:, i].any(), f"ungrouped column {i} has adjacency: {mask[:, i].tolist()}"
    # Grouped blocks remain fully connected
    assert mask[0, 1] and mask[1, 0]
    assert mask[3, 3], "single-item grouped row must still see itself"


if __name__ == "__main__":
    test_shifted_matches_to_cyclic_permutations_for_every_shift()
    test_shifted_rejects_out_of_range_shift()
    test_set_attention_adjacency_mask_on_grouped_batch()
    test_set_attention_rejects_all_ungrouped_batch()
    test_set_attention_isolates_ungrouped_rows_in_mixed_batch()
    print("All data_engine collator tests passed successfully!")
