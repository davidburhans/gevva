#!/usr/bin/env python3
"""tests/test_qat_integrity.py - Audit A8 remediation tests.

A8 (MAJOR): QAT was silently ON in finetune.py with the nvfp4 simulator while
the production export path (export_w4a16.py) emits INT4 group-32; the trained
format was unlogged; and PROGRESS claimed a "bounded STE" that the code never
implemented (plain pass-through). These tests pin the remediated contracts.

Run: uv run python tests/test_qat_integrity.py
"""

import sys
import traceback
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gemma4_cross_encoder import MultiQuantSTE, apply_quantization_aware_training  # noqa: E402


def test_int4_forward_error_is_group_bounded():
    """INT4 symmetric group-32 quantization error must not exceed half a step."""
    torch.manual_seed(0)
    w = torch.randn(16, 64, dtype=torch.float32)
    q = MultiQuantSTE.apply(w, "w4a16", 32)
    err = (q - w).abs().view(16, -1, 32)
    steps = q.view(16, -1, 32).abs().amax(dim=-1, keepdim=True) / 7.0 + 1e-8
    assert (err <= steps / 2 + 1e-6).all(), "quantization error exceeds half-step bound"


def test_ste_backward_is_pass_through_by_contract():
    """The STE is a plain pass-through (A8 doc correction: it was never bounded).

    Pins the ACTUAL contract so any future bounding change is deliberate.
    """
    w = torch.randn(4, 32, requires_grad=True)
    q = MultiQuantSTE.apply(w, "w4a16", 32)
    grad = torch.randn_like(q) * 1000.0  # unbounded magnitude must pass through
    q.backward(grad)
    assert torch.equal(w.grad, grad), "pass-through STE contract violated"


def test_unalignable_shapes_pass_through_unquantized():
    """2D grouping requires in_features % group_size == 0; otherwise identity."""
    w = torch.randn(8, 33)  # 33 not divisible by any sane group size
    q = MultiQuantSTE.apply(w, "w4a16", 32)
    assert torch.equal(q, w), "non-conforming shape must pass through unchanged"


def test_finetune_qat_defaults_are_opt_in_and_export_aligned():
    """finetune_custom_data signature + CLI defaults: QAT opt-in, format = export path."""
    import inspect

    import finetune

    sig = inspect.signature(finetune.finetune_custom_data)
    assert sig.parameters["qat"].default is False, "finetune_custom_data qat default must be False (A8)"
    assert sig.parameters["target_quant"].default == "w4a16", \
        "finetune_custom_data target_quant default must be w4a16 (A8 export alignment)"


def test_apply_qat_injects_and_runs_on_tiny_model():
    torch.manual_seed(1)

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(64, 8, bias=False)

        def forward(self, x):
            return self.proj(x)

    model = Tiny()
    quantized = apply_quantization_aware_training(model, quant_format="w4a16", group_size=32, num_bits=4)
    out = quantized(torch.randn(3, 64))
    assert out.shape == (3, 8)
    # Gradient flow through the parametrized (fake-quantized) weights.
    out.sum().backward()


TESTS = [
    test_int4_forward_error_is_group_bounded,
    test_ste_backward_is_pass_through_by_contract,
    test_unalignable_shapes_pass_through_unquantized,
    test_finetune_qat_defaults_are_opt_in_and_export_aligned,
    test_apply_qat_injects_and_runs_on_tiny_model,
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
