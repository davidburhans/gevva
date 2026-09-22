#!/usr/bin/env python3
"""tests/test_warm_start.py - Warm-start adapter initialization tests.

Closes the stage-3 regression vector: long-context stages trained from the base
model lose core NLI accuracy (MNLI 73% vs stage-2's 86%) because the mixture is
too small to relearn what the previous stage already knew. --warm-start
continues from the prior adapter instead. CPU-only via a tiny PEFT-wrapped model.

Run: uv run python tests/test_warm_start.py
"""

import sys
import tempfile
import traceback
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from peft import LoraConfig, get_peft_model  # noqa: E402
from train_cross_encoder import (  # noqa: E402
    _extract_head_state,
    warm_start_from_adapter,
)


class TinyClassifier(nn.Module):
    """Mirrors the real layout: LoRA targets backbone projections; score/norm are
    modules_to_save (PEFT forbids a module being both)."""

    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(8, 8)
        self.score = nn.Linear(8, 3, bias=False)
        self.norm = nn.LayerNorm(8)


def _peft_model():
    cfg = LoraConfig(r=4, lora_alpha=8, target_modules=["backbone"],
                     modules_to_save=["score"], lora_dropout=0.0, bias="none")
    return get_peft_model(TinyClassifier(), cfg)


def _save_checkpoint(model, adapter_dir):
    model.save_pretrained(adapter_dir)
    torch.save(_extract_head_state(model), str(Path(adapter_dir) / "head_weights.pt"))


def test_warm_start_restores_adapter_and_head():
    with tempfile.TemporaryDirectory() as tmp:
        adapter_dir = str(Path(tmp) / "prev_stage_best")
        teacher = _peft_model()
        # Give LoRA + head non-default values worth restoring.
        with torch.no_grad():
            for name, p in teacher.named_parameters():
                if "lora_" in name:
                    p.add_(0.25)
            teacher.base_model.model.score.weight.add_(0.5)
        _save_checkpoint(teacher, adapter_dir)

        student = _peft_model()
        differs = any(
            not torch.equal(p, q)
            for (_, p), (_, q) in zip(student.named_parameters(), teacher.named_parameters())
        )
        assert differs, "fresh model must differ from teacher before warm start"

        n_loaded = warm_start_from_adapter(student, adapter_dir)
        assert n_loaded > 0, "no adapter tensors loaded"

        # Adapter checkpoints carry LoRA params + modules_to_save (+ head_weights.pt
        # for score/norm) — NOT the frozen base weights, which stay randomly init.
        restored = [(n, p, q) for (n, p), (_, q) in zip(student.named_parameters(), teacher.named_parameters())
                    if ("lora_" in n or "modules_to_save" in n or ".norm." in n)]
        assert restored, "no adapter-carried parameters found to compare"
        for name, p, q in restored:
            assert torch.equal(p, q), f"parameter {name} not restored by warm start"


def test_warm_start_without_head_file_still_loads_adapter():
    with tempfile.TemporaryDirectory() as tmp:
        adapter_dir = str(Path(tmp) / "prev_stage_best")
        teacher = _peft_model()
        _save_checkpoint(teacher, adapter_dir)
        Path(adapter_dir, "head_weights.pt").unlink()  # some checkpoints lack it

        student = _peft_model()
        n_loaded = warm_start_from_adapter(student, adapter_dir)
        assert n_loaded > 0, "adapter tensors must load even without head_weights.pt"


def test_warm_start_missing_dir_raises_with_path():
    try:
        warm_start_from_adapter(_peft_model(), "/nonexistent/adapter/dir")
        raise AssertionError("missing adapter dir must raise")
    except Exception as exc:
        message = str(exc)
        assert "/nonexistent/adapter/dir" in message, f"error must include offending path: {message}"


TESTS = [
    test_warm_start_restores_adapter_and_head,
    test_warm_start_without_head_file_still_loads_adapter,
    test_warm_start_missing_dir_raises_with_path,
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
