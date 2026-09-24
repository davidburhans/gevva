#!/usr/bin/env python3
"""export_w4a16.py
==================
Standalone W4A16 (INT4 Group-32 Weights) Checkpoint Exporter for Gevva Decision Engine.

Takes a base Gemma 4 model and fine-tuned LoRA adapter, merges them,
and packs target linear layers into an INT4 (Group-32 symmetric) compressed weight format.

Format Specification:
1. Merges LoRA adapters directly into base weights before discretization.
2. In-Features 2D Group Partitioning (dim=1) with group_size=32.
3. Offset-Binary [0, 15] INT4 Packing: 8 signed 4-bit integers with +8 bias packed into 1 INT32 word.
4. Preserves 16-bit fidelity on MQA projections (k_proj, v_proj), SigLIP vision tower, and classification head.
5. Reduces checkpoint disk footprint by ~72% (from ~18 GB to ~5.1 GB).
6. Note: Dequantizes to BF16 in VRAM at inference time via this repository's native loader;
   it is an optimized weight-storage format, not an external runtime kernel (vLLM/TensorRT).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file
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


def pack_weights_w4a16(
    weight: torch.Tensor,
    group_size: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantizes and packs a 2D weight matrix into INT4 Group-32 format.

    Args:
        weight: Float/Bfloat tensor of shape `(out_features, in_features)`.
        group_size: Group size along in_features (dim=1). Must divide in_features.

    Returns:
        packed_weight: INT32 tensor of shape `(out_features, in_features // 8)`.
        scales: BF16 tensor of shape `(out_features, in_features // group_size)`.
    """
    weight = torch.nan_to_num(weight, nan=0.0, posinf=7.0, neginf=-8.0)
    out_features, in_features = weight.shape
    assert in_features % group_size == 0, f"in_features ({in_features}) must be divisible by group_size ({group_size})"
    assert group_size % 8 == 0, f"group_size ({group_size}) must be divisible by 8 for INT32 packing"

    # 1. Group-wise symmetric scale calculation (dim=1)
    w_grouped = weight.view(out_features, in_features // group_size, group_size)
    max_abs = torch.amax(torch.abs(w_grouped), dim=-1) + 1e-8  # (out_features, n_groups)
    q_max = 7.0  # 4-bit signed symmetric range [-8, 7]
    scales = (max_abs / q_max).to(torch.bfloat16)

    # 2. Integer quantization: clamp to [-8, 7]
    scales_expanded = scales.unsqueeze(-1).expand(-1, -1, group_size).reshape(out_features, in_features)
    scales_expanded = torch.clamp_min(scales_expanded, 1e-7)
    w_int = torch.clamp(torch.round(weight / scales_expanded), -8, 7).to(torch.int32)

    # 3. Standard Offset-Binary [0, 15] for compressed-tensors / vLLM (Offset +8)
    w_u4 = (w_int + 8).to(torch.int32)
    w_chunks = w_u4.view(out_features, in_features // 8, 8)
    packed_weight = torch.zeros((out_features, in_features // 8), dtype=torch.int32, device=weight.device)
    for k in range(8):
        packed_weight |= ((w_chunks[:, :, k].to(torch.int32) & 0x0F) << (4 * k))

    return packed_weight, scales


def unpack_weights_w4a16(
    packed_weight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = 32,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Unpacks INT4 Group-32 packed tensor back to reconstructed floating-point weights.

    Args:
        packed_weight: INT32 tensor of shape `(out_features, in_features // 8)`.
        scales: Float/Bfloat tensor of shape `(out_features, in_features // group_size)`.
        group_size: Quantization group size.
        dtype: Output reconstructed tensor dtype.

    Returns:
        reconstructed: Dequantized tensor of shape `(out_features, in_features)`.
    """
    out_features, packed_in = packed_weight.shape
    in_features = packed_in * 8

    # Unpack 8 int4 words from each int32 word
    unpacked_u4 = torch.zeros((out_features, packed_in, 8), dtype=torch.int32, device=packed_weight.device)
    for k in range(8):
        unpacked_u4[:, :, k] = (packed_weight >> (4 * k)) & 0x0F

    # Invert offset-binary (+8) to recover signed [-8, 7]
    unpacked_signed = (unpacked_u4 - 8).to(dtype)
    w_reconstructed = unpacked_signed.view(out_features, in_features)

    # Apply per-group scaling
    scales_expanded = scales.unsqueeze(-1).expand(-1, -1, group_size).reshape(out_features, in_features).to(dtype)
    return w_reconstructed * scales_expanded


def export_model_w4a16(
    base_model_id: str,
    adapter_path: str,
    output_dir: str,
    group_size: int = 32,
    preserve_mqa: bool = True,
    verify: bool = True,
) -> None:
    """Loads base model + adapter, merges LoRA, quantizes linear layers, and saves W4A16 safetensors.

    Args:
        base_model_id: HuggingFace foundation model identifier.
        adapter_path: Directory containing fine-tuned PEFT adapter.
        output_dir: Output directory for packed checkpoint.
        group_size: Weight grouping size (default: 32).
        preserve_mqa: Keep k_proj and v_proj in 16-bit for single-KV-head models.
        verify: Run verification on reconstructed model.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 65)
    print("W4A16 Model Exporter (compressed-tensors format)")
    print("=" * 65)
    print(f"Base Model:       {base_model_id}")
    print(f"Adapter Path:     {adapter_path}")
    print(f"Output Directory: {output_dir}")
    print(f"Group Size:       {group_size}")
    print(f"Preserve MQA:     {preserve_mqa}")
    print(f"Device:           {device}")

    # 1. Load Base Model and LoRA Adapter
    print("\n1. Loading base architecture & LoRA weights...")
    config = AutoConfig.from_pretrained(base_model_id)
    config.num_labels = 3
    base_model = Gemma4ForSequenceClassification.from_pretrained(
        base_model_id,
        config=config,
        torch_dtype=torch.bfloat16,
    )

    peft_model = PeftModel.from_pretrained(base_model, adapter_path)

    # Restore explicit classification head weights if available
    head_weights_path = os.path.join(adapter_path, "head_weights.pt")
    if os.path.exists(head_weights_path):
        print(f"Restoring classification head from {head_weights_path}...")
        hw = torch.load(head_weights_path, map_location="cpu", weights_only=True)
        raw = peft_model.base_model.model if hasattr(peft_model, "base_model") else peft_model
        if "score" in hw and hasattr(raw, "score"):
            target_score = raw.score.modules_to_save["default"] if (hasattr(raw.score, "modules_to_save") and "default" in raw.score.modules_to_save) else raw.score
            if "weight" in hw["score"]:
                target_score.load_state_dict(hw["score"])
            else:
                raw.score.load_state_dict(hw["score"])
        if "norm" in hw and hasattr(raw, "norm"):
            target_norm = raw.norm.modules_to_save["default"] if (hasattr(raw.norm, "modules_to_save") and "default" in raw.norm.modules_to_save) else raw.norm
            if "weight" in hw["norm"]:
                target_norm.load_state_dict(hw["norm"])
            else:
                raw.norm.load_state_dict(hw["norm"])

    # 2. Merge LoRA weights into base parameters
    print("2. Merging LoRA adapters into base model weights...")
    merged_model = peft_model.merge_and_unload()

    # Clean up any active PyTorch parametrizations to restore clean state_dict keys
    import torch.nn.utils.parametrize as parametrize
    for name, mod in merged_model.named_modules():
        if hasattr(mod, "parametrizations") and "weight" in mod.parametrizations:
            parametrize.remove_parametrizations(mod, "weight", leave_parametrized=True)

    merged_model.eval()

    # 3. Quantize and pack linear layers
    print("\n3. Quantizing target layers to 4-bit INT4 Group-32...")
    state_dict = merged_model.state_dict()
    packed_state_dict: Dict[str, torch.Tensor] = {}
    quantized_modules: List[str] = []
    preserved_modules: List[str] = []

    for key, tensor in state_dict.items():
        is_quantizable = (
            tensor.ndim == 2
            and "layers" in key
            and "weight" in key
            and any(proj in key for proj in ("self_attn", "mlp"))
            and not any(skip in key for skip in ("vision_tower", "embed_vision", "score", "norm"))
        )

        if preserve_mqa and any(k in key for k in ("k_proj", "v_proj")):
            is_quantizable = False

        if is_quantizable and tensor.shape[1] % group_size == 0:
            packed_w, scales = pack_weights_w4a16(tensor.cpu(), group_size=group_size)
            # compressed-tensors naming convention:
            # weight -> weight_packed (int32), weight_scale (bfloat16)
            base_key = key[: -len(".weight")]
            packed_state_dict[f"{base_key}.weight_packed"] = packed_w
            packed_state_dict[f"{base_key}.weight_scale"] = scales
            packed_state_dict[f"{base_key}.weight_shape"] = torch.tensor(
                [tensor.shape[0], tensor.shape[1]], dtype=torch.int32
            )
            quantized_modules.append(base_key)
        else:
            packed_state_dict[key] = tensor.to(torch.bfloat16).cpu()
            preserved_modules.append(key)

    print(f"Quantized and packed: {len(quantized_modules)} linear layers into W4A16.")
    print(f"Preserved in 16-bit:   {len(preserved_modules)} parameters (MQA, ViT, Head, Norms, Biases).")

    # 4. Save Safetensors Checkpoint
    output_weights_file = os.path.join(output_dir, "model.safetensors")
    print(f"\n4. Saving packed weights to {output_weights_file}...")
    save_file(packed_state_dict, output_weights_file)

    # Calculate compressed file size
    size_mb = os.path.getsize(output_weights_file) / (1024 * 1024)
    print(f"Model saved successfully! Total size: {size_mb:.1f} MB (~{size_mb/1024:.2f} GB)")

    # 5. Write Quantization Configuration (compressed-tensors metadata)
    quant_config = {
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "config_groups": {
            "group_0": {
                "weights": {
                    "num_bits": 4,
                    "type": "int",
                    "symmetric": True,
                    "strategy": "group",
                    "group_size": group_size,
                    "actorder": None,
                },
                "targets": ["Linear"],
            }
        },
        "ignore": [
            "*vision_tower*",
            "*embed_vision*",
            "*k_proj*",
            "*v_proj*",
            "*score*",
            "*norm*",
        ],
    }
    with open(os.path.join(output_dir, "quantization_config.json"), "w") as f:
        json.dump(quant_config, f, indent=2)

    # Copy tokenizer and model config
    print("5. Saving tokenizer and configuration files...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    tokenizer.save_pretrained(output_dir)

    config.num_labels = 3
    config.id2label = ID2LABEL
    config.label2id = LABEL2ID
    config.quantization_config = quant_config
    config.save_pretrained(output_dir)

    # Copy calibration and evaluation metadata if present
    for meta_file in ["calibration.json", "eval_metrics.json", "train_config.json"]:
        src_meta = os.path.join(adapter_path, meta_file)
        if os.path.exists(src_meta):
            shutil.copy2(src_meta, os.path.join(output_dir, meta_file))
            print(f"  Copied {meta_file} from adapter to {output_dir}.")

    # 6. Verification & Dequantization Self-Test
    if verify:
        print("\n6. Running dequantization reconstruction verification...")
        sample_layer = quantized_modules[0]
        packed_w = packed_state_dict[f"{sample_layer}.weight_packed"]
        scales = packed_state_dict[f"{sample_layer}.weight_scale"]
        original_w = state_dict[f"{sample_layer}.weight"]

        reconstructed_w = unpack_weights_w4a16(packed_w, scales, group_size=group_size)
        abs_diff = torch.abs(original_w - reconstructed_w)
        mean_abs_err = torch.mean(abs_diff).item()
        max_abs_err = torch.max(abs_diff).item()
        scales_exp = scales.unsqueeze(-1).expand(-1, -1, group_size).reshape(original_w.shape)
        max_step_err = torch.max(abs_diff / (scales_exp + 1e-8)).item()

        print(f"Layer [{sample_layer}]:")
        print(f"  Mean Absolute Quantization Error: {mean_abs_err:.6f}")
        print(f"  Max Absolute Quantization Error:  {max_abs_err:.6f}")
        print(f"  Max Relative Step Error:          {max_step_err:.4f} steps (bound <= 0.6)")
        assert max_step_err <= 0.6, f"Quantization step error exceeded bound: {max_step_err}"
        print("Reconstruction Verification: PASSED!")

    print("\n" + "=" * 65)
    print(f"Export Complete! W4A16 Artifact ready at: {output_dir}")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Export Gemma 4 Cross-Encoder to W4A16 compressed format.")
    parser.add_argument("--base-model", default="google/gemma-4-E2B", help="Base HuggingFace model")
    parser.add_argument("--adapter-path", default="./ckpt/gemma-4-e2b-nli-qat-stage1/best", help="Fine-tuned LoRA adapter path")
    parser.add_argument("--output-dir", default="./ckpt/gemma-4-e2b-nli-w4a16", help="Output directory")
    parser.add_argument("--group-size", type=int, default=32, help="Quantization group size (default: 32)")
    parser.add_argument("--no-preserve-mqa", action="store_false", dest="preserve_mqa", help="Quantize MQA projections as well")
    parser.add_argument("--skip-verify", action="store_false", dest="verify", help="Skip reconstruction verification")
    args = parser.parse_args()

    export_model_w4a16(
        base_model_id=args.base_model,
        adapter_path=args.adapter_path,
        output_dir=args.output_dir,
        group_size=args.group_size,
        preserve_mqa=args.preserve_mqa,
        verify=args.verify,
    )


if __name__ == "__main__":
    main()
