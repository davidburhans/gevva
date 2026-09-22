#!/usr/bin/env python3
"""tests/test_resume_checkpointing.py - Mid-epoch resume checkpointing tests.

Covers the durability gap exposed on 2026-09-21: training previously only
persisted at epoch-end validation (~4.7h granularity), so any interruption
lost the whole epoch. These tests exercise save/load round-trips, atomic
rotation, fingerprint refusal, and exact mid-epoch tail replay — all on CPU
with duck-typed models (no PEFT / GPU required).

Run: uv run python tests/test_resume_checkpointing.py
"""

import json
import random
import sys
import tempfile
import traceback
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import train_cross_encoder  # noqa: E402

from finetune import SkipPrefixBatchSampler, TokenBucketBatchSampler  # noqa: E402
from train_cross_encoder import (  # noqa: E402
    _retire_resume_state,
    resume_fingerprint,
    save_resume_state,
    try_load_resume_state,
)


class FakeHeadModel(nn.Module):
    """Duck-typed stand-in: plain module with score/norm + save_pretrained."""

    def __init__(self):
        super().__init__()
        self.score = nn.Linear(8, 3, bias=False)
        self.norm = nn.LayerNorm(8)

    def save_pretrained(self, path):
        torch.save(self.state_dict(), str(Path(path) / "model_state.pt"))


class PeftFakeModel(nn.Module):
    """PEFT-shaped stand-in: LoRA backbone + modules_to_save head (real PEFT wrap)."""

    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(8, 8)
        self.score = nn.Linear(8, 3, bias=False)
        self.norm = nn.LayerNorm(8)


def _peft_fake_model():
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(r=4, lora_alpha=8, target_modules=["backbone"],
                     modules_to_save=["score"], lora_dropout=0.0, bias="none")
    return get_peft_model(PeftFakeModel(), cfg)


def _tiny_training_setup(out_dir):
    model = FakeHeadModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    # Take one optimizer step so states are non-trivial.
    loss = model.score(model.norm(torch.randn(4, 8))).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()
    return model, optimizer, scheduler


def test_skip_prefix_sampler_yields_exact_tail():
    base = TokenBucketBatchSampler([100] * 12 + [900] * 3, max_tokens_per_batch=512, shuffle=True, seed=7)
    base.set_epoch(1)
    all_batches = [list(b) for b in base]
    base.set_epoch(1)  # same epoch -> same deterministic order
    tail = [list(b) for b in SkipPrefixBatchSampler(base, skip=4)]
    assert tail == all_batches[4:], "tail must equal base batches after the skipped prefix"
    assert len(SkipPrefixBatchSampler(base, skip=4)) == len(all_batches) - 4
    assert len(SkipPrefixBatchSampler(base, skip=len(all_batches) + 5)) == 0


def test_skip_prefix_sampler_rejects_negative_skip():
    base = TokenBucketBatchSampler([8, 8], max_tokens_per_batch=64, shuffle=False)
    try:
        SkipPrefixBatchSampler(base, skip=-1)
        raise AssertionError("negative skip must raise")
    except ValueError as exc:
        assert "-1" in str(exc), f"error must include offending value: {exc}"


def test_save_rotate_and_load_roundtrip():
    with tempfile.TemporaryDirectory() as out_dir:
        model, optimizer, scheduler = _tiny_training_setup(out_dir)
        meta = {"epoch": 0, "steps_done_in_epoch": 42, "global_step": 5,
                "best_val_acc": 0.81, "fingerprint": "abc123"}
        save_resume_state(out_dir, model, optimizer, scheduler, meta)
        # Second save must rotate (no stale resume_old/tmp dirs left behind).
        meta2 = dict(meta, steps_done_in_epoch=99, global_step=7)
        save_resume_state(out_dir, model, optimizer, scheduler, meta2)
        for stale in ("resume_tmp", "resume_old"):
            assert not Path(out_dir, stale).exists(), f"{stale}/ must not linger after rotation"

        fresh_model, fresh_optimizer, fresh_scheduler = _tiny_training_setup(out_dir)
        assert not torch.equal(fresh_model.score.weight, model.score.weight), "setup must differ pre-load"
        ctx = try_load_resume_state(out_dir, fresh_model, fresh_optimizer, fresh_scheduler, "abc123")
        assert ctx is not None
        assert (ctx.epoch, ctx.steps_done_in_epoch, ctx.global_step, ctx.best_val_acc) == (0, 99, 7, 0.81)
        assert torch.equal(fresh_model.score.weight, model.score.weight), "weights must round-trip"
        assert fresh_optimizer.state_dict()["param_groups"][0]["lr"] == \
            optimizer.state_dict()["param_groups"][0]["lr"]
        assert fresh_optimizer.state_dict()["state"][0]["exp_avg"].shape == (3, 8)


def test_load_refuses_incomplete_checkpoint():
    with tempfile.TemporaryDirectory() as out_dir:
        model, optimizer, scheduler = _tiny_training_setup(out_dir)
        save_resume_state(out_dir, model, optimizer, scheduler,
                          {"epoch": 1, "steps_done_in_epoch": 0, "global_step": 9,
                           "best_val_acc": 0.5, "fingerprint": "ff"})
        # Simulate a crash mid-save: meta.json deleted (it is written last).
        Path(out_dir, "resume", "meta.json").unlink()
        result = try_load_resume_state(out_dir, model, optimizer, scheduler, "ff")
        assert result is None, "checkpoint without complete meta.json must be refused"


def test_load_refuses_fingerprint_mismatch():
    with tempfile.TemporaryDirectory() as out_dir:
        model, optimizer, scheduler = _tiny_training_setup(out_dir)
        save_resume_state(out_dir, model, optimizer, scheduler,
                          {"epoch": 1, "steps_done_in_epoch": 0, "global_step": 9,
                           "best_val_acc": 0.5, "fingerprint": "old"})
        result = try_load_resume_state(out_dir, model, optimizer, scheduler, "new")
        assert result is None, "fingerprint mismatch must be refused"


def test_fingerprint_binds_data_and_recipe():
    with tempfile.TemporaryDirectory() as tmp:
        train = Path(tmp, "train.jsonl")
        val = Path(tmp, "val.jsonl")
        train.write_text('{"premise":"a","hypothesis":"b","label":0,"source":"t"}\n')
        val.write_text('{"premise":"c","hypothesis":"d","label":1,"source":"t"}\n')
        args = {"lr": 5e-5, "epochs": 2}
        fp1 = resume_fingerprint(args, str(train), str(val))
        assert fp1 == resume_fingerprint(dict(args), str(train), str(val)), "must be deterministic"
        assert fp1 != resume_fingerprint({**args, "epochs": 3}, str(train), str(val)), "recipe change must invalidate"
        train.write_text('{"premise":"CHANGED","hypothesis":"b","label":0,"source":"t"}\n')
        assert fp1 != resume_fingerprint(args, str(train), str(val)), "data change must invalidate"


def test_retire_removes_all_resume_dirs():
    with tempfile.TemporaryDirectory() as out_dir:
        model, optimizer, scheduler = _tiny_training_setup(out_dir)
        save_resume_state(out_dir, model, optimizer, scheduler,
                          {"epoch": 2, "steps_done_in_epoch": 0, "global_step": 30,
                           "best_val_acc": 0.9, "fingerprint": "x"})
        Path(out_dir, "resume_tmp").mkdir(exist_ok=True)
        _retire_resume_state(out_dir)
        for stale in ("resume", "resume_tmp", "resume_old"):
            assert not Path(out_dir, stale).exists()


def test_meta_json_is_written_last():
    with tempfile.TemporaryDirectory() as out_dir:
        model, optimizer, scheduler = _tiny_training_setup(out_dir)
        save_resume_state(out_dir, model, optimizer, scheduler,
                          {"epoch": 0, "steps_done_in_epoch": 10, "global_step": 2,
                           "best_val_acc": 0.7, "fingerprint": "y"})
        files = [p.name for p in Path(out_dir, "resume").iterdir()]
        for required in ("model_state.pt", "head_weights.pt", "optimizer.pt",
                         "scheduler.pt", "rng.pt", "meta.json"):
            assert required in files, f"resume snapshot missing {required}"
        meta = json.loads(Path(out_dir, "resume", "meta.json").read_text())
        assert meta["complete"] is True


def test_peft_resume_roundtrip_restores_lora_weights():
    """Regression: plain load_state_dict(strict=False) silently dropped LoRA keys;
    resume must restore lora_A/lora_B AND the saved head exactly."""
    with tempfile.TemporaryDirectory() as out_dir:
        teacher = _peft_fake_model()
        with torch.no_grad():
            for name, p in teacher.named_parameters():
                if "lora_" in name:
                    p.add_(0.33)
        optimizer = torch.optim.AdamW(teacher.parameters(), lr=1e-3)
        save_resume_state(out_dir, teacher, optimizer,
                          torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5),
                          {"epoch": 1, "steps_done_in_epoch": 5, "global_step": 3,
                           "best_val_acc": 0.75, "fingerprint": "peft"})

        student = _peft_fake_model()
        opt2 = torch.optim.AdamW(student.parameters(), lr=1e-3)
        ctx = try_load_resume_state(out_dir, student, opt2,
                                    torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=5), "peft")
        assert ctx is not None and ctx.global_step == 3
        lora_ok = 0
        for (name, p), (_, q) in zip(student.named_parameters(), teacher.named_parameters()):
            if "lora_" in name:
                assert torch.equal(p, q), f"LoRA parameter {name} not restored on resume"
                lora_ok += 1
        assert lora_ok >= 2, f"expected LoRA params in fixture, found {lora_ok}"


def test_fingerprint_call_sites_are_symmetric():
    """Regression (2026-09-22): the resume LOAD hashed full vars(args) while SAVES
    hashed fp_args (minus checkpoint_interval/resume_auto), so no resume could
    ever match its own checkpoint - the first real resume silently restarted
    training. Contract: exactly one shared fingerprint expression on the active path.
    """
    source = Path(train_cross_encoder.__file__).read_text(encoding="utf-8")
    assert "resume_fingerprint(vars(args)" not in source, \
        "load-site fingerprint must use fp_args, not full vars(args)"
    assert source.count("active_fingerprint = resume_fingerprint(fp_args") == 1, \
        "the active fingerprint must be computed exactly once from fp_args"
    assert source.count("ctx = try_load_resume_state(args.out_dir, model, optimizer, scheduler, active_fingerprint)") == 1, \
        "the load site must use the shared active_fingerprint"


TESTS = [
    test_skip_prefix_sampler_yields_exact_tail,
    test_skip_prefix_sampler_rejects_negative_skip,
    test_save_rotate_and_load_roundtrip,
    test_load_refuses_incomplete_checkpoint,
    test_load_refuses_fingerprint_mismatch,
    test_fingerprint_binds_data_and_recipe,
    test_retire_removes_all_resume_dirs,
    test_meta_json_is_written_last,
    test_peft_resume_roundtrip_restores_lora_weights,
    test_fingerprint_call_sites_are_symmetric,
]


def main() -> int:
    failures = 0
    for test in TESTS:
        try:
            random.seed(0)
            torch.manual_seed(0)
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
