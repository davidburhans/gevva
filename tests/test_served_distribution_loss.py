#!/usr/bin/env python3
"""tests/test_served_distribution_loss.py - Phase-1 train-serving parity loss tests.

The 2026-09-22 JevBench audit: serving scores options by renormalized
P(entailment), training ranked by softmax(z_ent - z_con) - different distributions
(60% of benchmark errors had gold ranked 2nd under the served one). These tests
pin the served-distribution loss: hand-computed CE, the disagreement case,
optimization, exclusions, and soft targets.

Run: uv run python tests/test_served_distribution_loss.py
"""

import math
import sys
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gemma4_cross_encoder import CONTRADICTION, ENTAILMENT, NEUTRAL  # noqa: E402
from research.adapters.grouped_decision_collator import (  # noqa: E402
    compute_cross_option_loss,
    compute_served_distribution_loss,
)


def _logits(*triples):
    """Build (N,3) logits from (z_con, z_ent, z_neu) triples."""
    return torch.tensor(triples, dtype=torch.float32)


def test_hand_computed_cross_entropy():
    logits = _logits((0.0, 2.0, 0.0), (0.0, 1.0, 0.0))
    gid = torch.tensor([0, 0]); gold = torch.tensor([1.0, 0.0])  # gold = option 0
    loss, meta = compute_served_distribution_loss(logits, gid, gold)
    p = torch.softmax(logits, dim=-1)[:, ENTAILMENT]
    expected = -math.log(float(p[0] / p.sum()))  # option 0 is gold
    assert abs(meta["served_loss"] - expected) < 1e-5, f"{meta['served_loss']} != {expected}"
    assert meta["served_groups"] == 1 and meta["served_accuracy"] == 1.0


def test_served_and_margin_rankings_disagree():
    """The parity gap, constructed: A wins the margin ranking, B wins the served one.

    A: high z_ent but a NEUTRAL logit (z_neu=5) inflates the 3-class normalizer -> P_ent moderate.
    B: slightly lower z_ent with all other logits very negative -> P_ent ~0.99.
    """
    logits = _logits(
        (-4.0, 4.0, 5.0),    # option A: margin 8.0, P_ent ~0.27
        (-3.0, 3.8, -8.0),  # option B: margin 6.8, P_ent ~0.99
    )
    gid = torch.tensor([0, 0])
    margin_scores = logits[:, ENTAILMENT] - logits[:, CONTRADICTION]
    p_ent = torch.softmax(logits, dim=-1)[:, ENTAILMENT]
    assert margin_scores[0] > margin_scores[1], "fixture must let A win the margin ranking"
    assert p_ent[1] > p_ent[0], "fixture must let B win the served ranking"

    gold_A = torch.tensor([1.0, 0.0]); gold_B = torch.tensor([0.0, 1.0])
    served_A, _ = compute_served_distribution_loss(logits, gid, gold_A)
    served_B, _ = compute_served_distribution_loss(logits, gid, gold_B)
    assert served_B < served_A, "served loss must prefer gold=B (the served ranking)"

    margin_A, _ = compute_cross_option_loss(margin_scores, gid, gold_A)
    margin_B, _ = compute_cross_option_loss(margin_scores, gid, gold_B)
    assert margin_A < margin_B, "margin loss must prefer gold=A (the training ranking)"


def test_optimization_pushes_served_distribution_to_gold():
    torch.manual_seed(0)
    logits = (0.5 * torch.randn(4, 3)).requires_grad_(True)
    gid = torch.tensor([0, 0, 0, 0])
    gold = torch.tensor([0.0, 0.0, 1.0, 0.0])
    opt = torch.optim.SGD([logits], lr=0.5)
    for _ in range(200):
        opt.zero_grad()
        loss, meta = compute_served_distribution_loss(logits, gid, gold)
        loss.backward()
        opt.step()
    _, meta = compute_served_distribution_loss(logits.detach(), gid, gold)
    p = torch.softmax(logits.detach(), dim=-1)[:, ENTAILMENT]
    assert meta["served_accuracy"] == 1.0, "gold must win the served ranking after training"
    assert p[2] / p.sum() > 0.9, f"served gold prob {float(p[2]/p.sum()):.3f} must be sharp"


def test_ungrouped_and_singletons_excluded():
    logits = _logits((0, 1, 0), (0, 1, 0), (0, 1, 0))
    loss, meta = compute_served_distribution_loss(logits, torch.tensor([-1, -1, -1]),
                                                  torch.tensor([1.0, 0.0, 0.0]))
    assert meta["served_groups"] == 0 and float(loss) == 0.0
    loss, meta = compute_served_distribution_loss(logits, torch.tensor([0, -1, -1]),
                                                  torch.tensor([1.0, 0.0, 0.0]))
    assert meta["served_groups"] == 0, "singleton group must be skipped"


def test_soft_targets_renormalized():
    logits = _logits((0.0, 2.0, 0.0), (0.0, 1.0, 0.0))
    gid = torch.tensor([0, 0])
    soft = torch.tensor([0.75, 0.25])  # already a distribution
    loss, _ = compute_served_distribution_loss(logits, gid, None, soft_targets=soft)
    p = torch.softmax(logits, dim=-1)[:, ENTAILMENT]
    dist = p / p.sum()
    expected = -(0.75 * math.log(float(dist[0])) + 0.25 * math.log(float(dist[1])))
    assert abs(float(loss) - expected) < 1e-5


def test_all_zero_entailment_group_skipped():
    logits = _logits((0.0, -200.0, 0.0), (0.0, -200.0, 0.0))  # P(ent) underflows to exact 0
    gid = torch.tensor([0, 0]); gold = torch.tensor([1.0, 0.0])
    loss, meta = compute_served_distribution_loss(logits, gid, gold)
    assert meta["served_groups"] == 0, "all-underflow entailment group must be skipped (adapter serves uniform)"


def test_finetune_served_loss_wiring_is_complete():
    """Regression (review F02/F03, 2026-09-22): the served-distribution loss once
    shipped with a CLI flag and call-site argument but NO signature parameter and
    NO call inside the training loop - argparse would crash, then the run would
    silently train plain CE+Brier. Pins the full wiring chain and the launcher's
    flags against finetune.py's actual CLI."""
    import inspect
    import re
    import subprocess
    import sys as _sys

    import finetune

    sig = inspect.signature(finetune.finetune_custom_data)
    assert "served_dist_weight" in sig.parameters, "signature must accept served_dist_weight"
    src = inspect.getsource(finetune.finetune_custom_data)
    assert "compute_served_distribution_loss(" in src, \
        "the training loop must CALL compute_served_distribution_loss"
    assert "served_dist_weight > 0.0" in src, "the call must be gated on served_dist_weight"
    assert "served_dist_weight * served_loss" in src, "the loss must enter total_loss weighted"
    # Group-atomic batching is required for the loss to see whole option sets.
    assert "GroupedTokenBucketBatchSampler(" in src, "grouped batches must use the group sampler"

    launcher = Path(finetune.__file__).parent / "scripts" / "launch_phase1_served.py"
    launcher_src = launcher.read_text()
    cmd_block = launcher_src[launcher_src.index("TRAIN_CMD"):launcher_src.index("]", launcher_src.index("TRAIN_CMD"))]
    cmd_flags = set(re.findall(r'"--([a-z0-9-]+)"', cmd_block))
    help_text = subprocess.run(
        [_sys.executable, str(Path(finetune.__file__)), "--help"],
        capture_output=True, text=True, timeout=120,
    ).stdout
    allowed = set(re.findall(r'--([a-z0-9-]+)', help_text))
    unknown = {f for f in cmd_flags if f not in allowed}
    assert not unknown, f"launcher passes flags finetune.py does not define: {unknown}"


TESTS = [
    test_hand_computed_cross_entropy,
    test_served_and_margin_rankings_disagree,
    test_optimization_pushes_served_distribution_to_gold,
    test_ungrouped_and_singletons_excluded,
    test_soft_targets_renormalized,
    test_all_zero_entailment_group_skipped,
    test_finetune_served_loss_wiring_is_complete,
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
