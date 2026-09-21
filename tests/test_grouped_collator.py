import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer

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
    test_grouped_decision_collator()
    print("All grouped decision collator tests passed successfully!")
