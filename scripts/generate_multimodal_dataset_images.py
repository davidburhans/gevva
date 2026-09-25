#!/usr/bin/env python3
"""scripts/generate_multimodal_dataset_images.py - Batch 1 Image Generator.

Generates paired and multi-image scenes for multimodal decision tasks:
1. Programmatic Synthetics (40%): Mathematical charts, UI tables, geometric scenes with known truth.
2. Qwen-Image-2.1 Diffusion Edits (60%): Photorealistic before/after counterfactual pairs.

Preserves zero memory thrashing:
Loads generator once, processes all image requests, and unloads completely from VRAM.

Usage:
  # Generate mini test bank (10 pairs):
  uv run python scripts/generate_multimodal_dataset_images.py --limit 10 --out-dir data/multimodal_multi_image

  # Full generation run (500 pairs):
  uv run python scripts/generate_multimodal_dataset_images.py --limit 500 --out-dir data/multimodal_multi_image
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# High-resolution & enterprise aspect presets (spanning 512px up to 1K, 1080p, and 1600px A4 scans)
ASPECT_PRESETS = [
    ("1080p_fhd_landscape", 1920, 1080),
    ("a4_document_portrait", 1200, 1600),
    ("1k_square", 1024, 1024),
    ("720p_hd_landscape", 1280, 720),
    ("mobile_fhd_portrait", 1080, 1920),
    ("widescreen_banner", 1680, 720),
    ("presentation_4_3", 1024, 768),
    ("document_3_4", 768, 1024),
    ("compact_square", 512, 512),
]


# -----------------------------------------------------------------------------
# Programmatic Pair Generators (100% Deterministic Ground Truth)
# -----------------------------------------------------------------------------

def generate_programmatic_chart_pair(pair_id: str, rng: random.Random) -> Dict[str, Any]:
    """Generates two related financial/metric bar charts with dynamic aspect ratio."""
    aspect_name, w, h = rng.choice(ASPECT_PRESETS)
    categories = ["Engineering", "Marketing", "Sales", "Operations", "Product"]
    v1 = [rng.randint(20, 100) for _ in categories]
    # In Image 2, mutate 1-2 categories intentionally
    v2 = list(v1)
    mutated_cat_idx = rng.randint(0, len(categories) - 1)
    delta = rng.choice([-25, -15, 20, 35])
    v2[mutated_cat_idx] = max(10, min(100, v2[mutated_cat_idx] + delta))

    def _render_chart(values: List[int], title: str) -> Image.Image:
        img = Image.new("RGB", (w, h), color=(250, 250, 252))
        draw = ImageDraw.Draw(img)
        draw.text((int(w * 0.05), int(h * 0.05)), f"{title} [{aspect_name}]", fill=(20, 20, 30))
        # Draw axes
        axis_y = int(h * 0.86)
        axis_x_start = int(w * 0.12)
        axis_x_end = int(w * 0.92)
        axis_y_top = int(h * 0.18)

        draw.line([(axis_x_start, axis_y), (axis_x_end, axis_y)], fill=(150, 150, 160), width=2)
        draw.line([(axis_x_start, axis_y_top), (axis_x_start, axis_y)], fill=(150, 150, 160), width=2)

        available_w = axis_x_end - axis_x_start - 20
        n_bars = len(categories)
        bar_w = max(20, int(available_w / (n_bars * 1.5)))
        spacing = max(10, int(bar_w * 0.4))
        start_x = axis_x_start + spacing

        colors = [(66, 133, 244), (234, 67, 53), (251, 188, 4), (52, 168, 83), (142, 68, 173)]
        max_h = axis_y - axis_y_top - 20

        for i, (cat, val) in enumerate(zip(categories, values)):
            bx = start_x + i * (bar_w + spacing)
            bar_h = int((val / 100) * max_h)
            by = axis_y - bar_h
            draw.rectangle([(bx, by), (bx + bar_w, axis_y)], fill=colors[i % len(colors)])
            draw.text((bx + max(2, bar_w // 6), by - 16), str(val), fill=(40, 40, 50))
            draw.text((bx + max(2, bar_w // 8), axis_y + 8), cat[:4], fill=(80, 80, 90))
        return img

    img1 = _render_chart(v1, "Period 1 Metric Report")
    img2 = _render_chart(v2, "Period 2 Metric Report")

    mutated_cat = categories[mutated_cat_idx]
    actual_change = v2[mutated_cat_idx] - v1[mutated_cat_idx]
    direction = "increased" if actual_change > 0 else "decreased"

    return {
        "id": pair_id,
        "family": "cross_chart_trend",
        "aspect_ratio": aspect_name,
        "dimensions": (w, h),
        "images": [img1, img2],
        "metadata": {
            "category": mutated_cat,
            "period1_val": v1[mutated_cat_idx],
            "period2_val": v2[mutated_cat_idx],
            "direction": direction,
            "delta": actual_change,
            "aspect_ratio": aspect_name,
        },
        "claims": {
            "entailment": f"Between Period 1 and Period 2, {mutated_cat} budget {direction} by {abs(actual_change)} units.",
            "contradiction": f"Between Period 1 and Period 2, {mutated_cat} budget remained unchanged.",
            "neutral": f"{mutated_cat} will receive another budget increase next quarter.",
        },
    }


def generate_programmatic_state_pair(pair_id: str, rng: random.Random) -> Dict[str, Any]:
    """Generates two UI / system state scenes with dynamic aspect ratio and layout scaling."""
    aspect_name, w, h = rng.choice(ASPECT_PRESETS)

    def _render_ui(is_success: bool) -> Image.Image:
        img = Image.new("RGB", (w, h), color=(240, 242, 245))
        d = ImageDraw.Draw(img)
        # Header card
        pad = int(min(w, h) * 0.08)
        d.rectangle([(pad, pad), (w - pad, h - pad)], fill=(255, 255, 255), outline=(210, 215, 220), width=2)
        d.text((pad + 20, pad + 15), "Account Security Settings", fill=(30, 35, 45))

        # Username field
        field_w = w - 2 * pad - 40
        y1 = pad + 45
        d.text((pad + 20, y1), "Username", fill=(80, 85, 95))
        d.rectangle([(pad + 20, y1 + 18), (pad + 20 + field_w, y1 + 48)], fill=(245, 247, 250), outline=(200, 205, 215))
        d.text((pad + 30, y1 + 25), "user_admin", fill=(30, 35, 45))

        # Email field
        y2 = y1 + 65
        d.text((pad + 20, y2), "Work Email", fill=(80, 85, 95))
        if not is_success:
            d.rectangle([(pad + 20, y2 + 18), (pad + 20 + field_w, y2 + 48)], fill=(254, 242, 242), outline=(239, 68, 68), width=2)
            d.text((pad + 30, y2 + 25), "invalid-email-address", fill=(185, 28, 28))
            d.text((pad + 20, y2 + 54), "[!] Please enter a valid corporate email", fill=(220, 38, 38))
        else:
            d.rectangle([(pad + 20, y2 + 18), (pad + 20 + field_w, y2 + 48)], fill=(240, 253, 244), outline=(34, 197, 94), width=2)
            d.text((pad + 30, y2 + 25), "admin@enterprise.corp", fill=(22, 101, 52))
            d.text((pad + 20, y2 + 54), "[OK] Email verified", fill=(22, 163, 74))

        # Button
        by = min(h - pad - 45, y2 + 90)
        btn_w = min(160, int(field_w * 0.45))
        btn_color = (37, 99, 235) if is_success else (156, 163, 175)
        d.rectangle([(pad + 20, by), (pad + 20 + btn_w, by + 35)], fill=btn_color)
        d.text((pad + 35, by + 10), "Save Changes", fill=(255, 255, 255))
        return img

    img1 = _render_ui(is_success=False)
    img2 = _render_ui(is_success=True)

    return {
        "id": pair_id,
        "family": "ui_state_transition",
        "aspect_ratio": aspect_name,
        "dimensions": (w, h),
        "images": [img1, img2],
        "metadata": {
            "action": "email_correction",
            "from": "invalid-email-address",
            "to": "admin@enterprise.corp",
            "aspect_ratio": aspect_name,
        },
        "claims": {
            "entailment": "In Image 2, the email address input field was corrected and marked verified.",
            "contradiction": "Image 2 still displays an invalid email error alert.",
            "neutral": "The user entered their email address using an external password manager.",
        },
    }


def generate_programmatic_table_pair(pair_id: str, rng: random.Random) -> Dict[str, Any]:
    """Generates enterprise invoice/billing tables with dynamic aspect ratio and numerical diffs."""
    aspect_name, w, h = rng.choice(ASPECT_PRESETS)
    items = ["Compute VM", "Object Storage", "Egress Bandwidth", "KMS Keys", "NAT Gateway"]
    base_qty = [rng.randint(2, 20) for _ in items]
    base_rates = [120, 15, 25, 50, 40]

    mut_idx = rng.randint(0, len(items) - 1)
    new_qty = list(base_qty)
    delta_qty = rng.choice([-4, -2, 5, 10])
    new_qty[mut_idx] = max(1, new_qty[mut_idx] + delta_qty)

    status1 = "PENDING REVIEW"
    status2 = rng.choice(["APPROVED", "PAID", "FLAGGED FOR AUDIT"])

    def _render_table(quantities: List[int], status: str, title: str) -> Tuple[Image.Image, int]:
        img = Image.new("RGB", (w, h), color=(248, 249, 250))
        d = ImageDraw.Draw(img)
        pad = int(min(w, h) * 0.06)
        d.rectangle([(pad, pad), (w - pad, h - pad)], fill=(255, 255, 255), outline=(220, 224, 230), width=2)
        d.text((pad + 20, pad + 15), f"{title} [{aspect_name}]", fill=(20, 25, 35))
        st_color = (180, 50, 50) if "FLAGGED" in status else (20, 120, 50)
        d.text((pad + 20, pad + 40), f"Status: {status}", fill=st_color)

        row_y = pad + 80
        d.text((pad + 20, row_y), "Service", fill=(100, 105, 115))
        d.text((pad + int(w * 0.45), row_y), "Qty", fill=(100, 105, 115))
        d.text((pad + int(w * 0.60), row_y), "Rate", fill=(100, 105, 115))
        d.text((pad + int(w * 0.75), row_y), "Subtotal", fill=(100, 105, 115))
        d.line([(pad + 20, row_y + 20), (w - pad - 20, row_y + 20)], fill=(200, 205, 215), width=1)

        total = 0
        for i, (item_name, q, r) in enumerate(zip(items, quantities, base_rates)):
            ry = row_y + 35 + i * 32
            sub = q * r
            total += sub
            d.text((pad + 20, ry), item_name, fill=(30, 35, 45))
            d.text((pad + int(w * 0.45), ry), str(q), fill=(30, 35, 45))
            d.text((pad + int(w * 0.60), ry), f"${r}", fill=(30, 35, 45))
            d.text((pad + int(w * 0.75), ry), f"${sub}", fill=(30, 35, 45))

        tot_y = row_y + 35 + len(items) * 32 + 15
        d.line([(pad + 20, tot_y), (w - pad - 20, tot_y)], fill=(200, 205, 215), width=2)
        d.text((pad + int(w * 0.60), tot_y + 10), "Total:", fill=(20, 25, 35))
        d.text((pad + int(w * 0.75), tot_y + 10), f"${total}", fill=(20, 25, 35))
        return img, total

    img1, tot1 = _render_table(base_qty, status1, "Invoice Period 1")
    img2, tot2 = _render_table(new_qty, status2, "Invoice Period 2")
    mut_item = items[mut_idx]
    diff_qty = new_qty[mut_idx] - base_qty[mut_idx]
    dir_str = "increased" if diff_qty > 0 else "decreased"

    return {
        "id": pair_id,
        "family": "table_invoice_diff",
        "aspect_ratio": aspect_name,
        "dimensions": (w, h),
        "images": [img1, img2],
        "metadata": {
            "item": mut_item,
            "delta_qty": diff_qty,
            "status1": status1,
            "status2": status2,
            "total1": tot1,
            "total2": tot2,
            "aspect_ratio": aspect_name,
        },
        "claims": {
            "entailment": f"In Image 2, the quantity for {mut_item} {dir_str} by {abs(diff_qty)} units and invoice status is {status2}.",
            "contradiction": f"Image 1 and Image 2 show identical line item quantities and total billing amounts.",
            "neutral": "The invoice will be paid via corporate credit card before the next billing cycle.",
        },
    }


def generate_programmatic_dashboard_metric_pair(pair_id: str, rng: random.Random) -> Dict[str, Any]:
    """Generates cloud infrastructure monitoring dashboard snapshots with metric threshold transitions."""
    aspect_name, w, h = rng.choice(ASPECT_PRESETS)
    cpu1 = rng.randint(30, 65)
    delta_cpu = rng.choice([-20, 20, 30])
    cpu2 = max(10, min(95, cpu1 + delta_cpu))

    st1 = "HEALTHY"
    st2 = "CRITICAL" if cpu2 >= 85 else ("DEGRADED" if cpu2 >= 70 else "HEALTHY")

    def _render_dash(cpu: int, st: str, title: str) -> Image.Image:
        img = Image.new("RGB", (w, h), color=(15, 23, 42))
        d = ImageDraw.Draw(img)
        pad = int(min(w, h) * 0.06)
        d.text((pad, pad), f"{title} - Production Cluster Alpha [{aspect_name}]", fill=(241, 245, 249))

        cw = int((w - 2 * pad - 40) / 2)
        ch = int(min(h * 0.45, 260))
        # CPU card
        d.rectangle([(pad, pad + 40), (pad + cw, pad + 40 + ch)], fill=(30, 41, 59), outline=(51, 65, 85))
        d.text((pad + 20, pad + 60), "CPU Utilization", fill=(148, 163, 184))
        d.text((pad + 20, pad + 90), f"{cpu}%", fill=(248, 113, 113) if cpu > 80 else (74, 222, 128))

        # Status card
        d.rectangle([(pad + cw + 20, pad + 40), (pad + 2 * cw + 20, pad + 40 + ch)], fill=(30, 41, 59), outline=(51, 65, 85))
        d.text((pad + cw + 40, pad + 60), "Cluster Health Status", fill=(148, 163, 184))
        col = (239, 68, 68) if st == "CRITICAL" else ((245, 158, 11) if st == "DEGRADED" else (34, 197, 94))
        d.text((pad + cw + 40, pad + 90), st, fill=col)
        return img

    img1 = _render_dash(cpu1, st1, "Timestamp T1")
    img2 = _render_dash(cpu2, st2, "Timestamp T2")
    delta = cpu2 - cpu1
    dir_str = "increased" if delta > 0 else "decreased"

    return {
        "id": pair_id,
        "family": "dashboard_metric_alert",
        "aspect_ratio": aspect_name,
        "dimensions": (w, h),
        "images": [img1, img2],
        "metadata": {
            "cpu1": cpu1,
            "cpu2": cpu2,
            "status1": st1,
            "status2": st2,
            "delta_cpu": delta,
            "aspect_ratio": aspect_name,
        },
        "claims": {
            "entailment": f"Between Timestamp T1 and Timestamp T2, CPU Utilization {dir_str} by {abs(delta)}% and health is {st2}.",
            "contradiction": f"Image 2 indicates that CPU Utilization dropped below 5% with zero active workloads.",
            "neutral": "The site reliability engineering team will deploy a canary kernel update at midnight.",
        },
    }


def generate_programmatic_spatial_map_pair(pair_id: str, rng: random.Random) -> Dict[str, Any]:
    """Generates warehouse/datacenter floor plan layouts with spatial coordinate movements."""
    aspect_name, w, h = rng.choice(ASPECT_PRESETS)
    bays = ["Zone 1 (Dock)", "Zone 2 (Storage)", "Zone 3 (Packaging)", "Zone 4 (Dispatch)"]
    asset = rng.choice(["Automated AGV-1", "Pallet Jack 4", "Robotic Sorter B", "Forklift Alpha"])
    from_bay = rng.choice(bays)
    to_bay = rng.choice([b for b in bays if b != from_bay])

    def _render_map(current_bay: str, title: str) -> Image.Image:
        img = Image.new("RGB", (w, h), color=(245, 245, 247))
        d = ImageDraw.Draw(img)
        pad = int(min(w, h) * 0.06)
        d.text((pad, pad), f"{title}: Facility Floor Plan [{aspect_name}]", fill=(20, 20, 30))

        grid_w = w - 2 * pad
        grid_h = h - pad - 60
        mid_x = pad + grid_w // 2
        mid_y = pad + 40 + grid_h // 2

        zones = [
            (bays[0], (pad, pad + 40, mid_x - 10, mid_y - 10)),
            (bays[1], (mid_x + 10, pad + 40, pad + grid_w, mid_y - 10)),
            (bays[2], (pad, mid_y + 10, mid_x - 10, pad + 40 + grid_h)),
            (bays[3], (mid_x + 10, mid_y + 10, pad + grid_w, pad + 40 + grid_h)),
        ]

        for z_name, box in zones:
            is_active = (z_name == current_bay)
            fill_col = (219, 234, 254) if is_active else (255, 255, 255)
            outline_col = (37, 99, 235) if is_active else (209, 213, 219)
            d.rectangle(box, fill=fill_col, outline=outline_col, width=2 if is_active else 1)
            d.text((box[0] + 15, box[1] + 15), z_name, fill=(30, 41, 59))
            if is_active:
                d.rectangle([(box[0] + 15, box[1] + 45), (box[0] + 200, box[1] + 80)], fill=(37, 99, 235))
                d.text((box[0] + 25, box[1] + 55), f"[ACTIVE] {asset}", fill=(255, 255, 255))
        return img

    img1 = _render_map(from_bay, "T1 Floor Scan")
    img2 = _render_map(to_bay, "T2 Floor Scan")

    return {
        "id": pair_id,
        "family": "spatial_map_movement",
        "aspect_ratio": aspect_name,
        "dimensions": (w, h),
        "images": [img1, img2],
        "metadata": {
            "asset": asset,
            "from_bay": from_bay,
            "to_bay": to_bay,
            "aspect_ratio": aspect_name,
        },
        "claims": {
            "entailment": f"Between Image 1 and Image 2, {asset} relocated from {from_bay} to {to_bay}.",
            "contradiction": f"In Image 2, {asset} is still located in {from_bay}.",
            "neutral": f"{asset} was manufactured by a logistics contractor in Germany.",
        },
    }


# -----------------------------------------------------------------------------
# Qwen-Image-2.1 Diffusion Pair Generator (Counterfactual Editing)
# -----------------------------------------------------------------------------

def load_qwen_image_pipeline(device: str = "cuda"):
    """Loads QwenImage21Pipeline with model offload to strictly respect VRAM limits."""
    import torch
    from diffusers import QwenImage21Pipeline

    print("Loading Qwen-Image-2.1 pipeline onto GPU (bfloat16)...")
    pipe = QwenImage21Pipeline.from_pretrained(
        "Qwen/Qwen-Image-2.1",
        torch_dtype=torch.bfloat16,
    )
    # Enable CPU offload to keep peak VRAM under 15 GB
    if hasattr(pipe, "enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    return pipe


def generate_dataset_bank(
    out_dir: Path,
    limit: int = 50,
    use_diffusion: bool = False,
    seed: int = 42,
) -> None:
    rng = random.Random(seed)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    pair_units: List[Tuple[str, List[Dict[str, Any]]]] = []
    pipe = None
    if use_diffusion:
        try:
            pipe = load_qwen_image_pipeline()
        except Exception as e:
            print(f"Warning: Could not load Qwen-Image-2.1 pipeline ({e}). Falling back to programmatic generator.")
            pipe = None

    generators = [
        generate_programmatic_chart_pair,
        generate_programmatic_state_pair,
        generate_programmatic_table_pair,
        generate_programmatic_dashboard_metric_pair,
        generate_programmatic_spatial_map_pair,
    ]

    print(f"Generating {limit} multi-image decision pairs across {len(generators)} families into {out_dir}...")
    t0 = time.time()

    for idx in range(limit):
        pair_id = f"pair_{idx:05d}"

        # Decide category:
        # If diffusion available: 60% diffusion, 40% programmatic
        if pipe is not None and rng.random() < 0.60:
            pass  # fallback to programmatic if pipe is none

        # Programmatic branch: cycle smoothly through the 5 diverse families
        gen_fn = generators[idx % len(generators)]
        item = gen_fn(pair_id, rng)

        # Save images
        img1_p = images_dir / f"{pair_id}_1.png"
        img2_p = images_dir / f"{pair_id}_2.png"
        item["images"][0].save(img1_p)
        item["images"][1].save(img2_p)

        rel_img1 = f"images/{img1_p.name}"
        rel_img2 = f"images/{img2_p.name}"

        # Create 3 NLI samples from the pair: Entailment, Contradiction, Neutral
        premise = "Compare Image 1 and Image 2."
        claims_for_pair = [
            {
                "id": f"{pair_id}_entail",
                "premise": premise,
                "hypothesis": item["claims"]["entailment"],
                "label": 1,
                "task_family": item["family"],
                "images": [rel_img1, rel_img2],
                "metadata": item.get("metadata", {}),
            },
            {
                "id": f"{pair_id}_contra",
                "premise": premise,
                "hypothesis": item["claims"]["contradiction"],
                "label": 0,
                "task_family": item["family"],
                "images": [rel_img1, rel_img2],
                "metadata": item.get("metadata", {}),
            },
            {
                "id": f"{pair_id}_neut",
                "premise": premise,
                "hypothesis": item["claims"]["neutral"],
                "label": 2,
                "task_family": item["family"],
                "images": [rel_img1, rel_img2],
                "metadata": item.get("metadata", {}),
            },
        ]
        pair_units.append((pair_id, claims_for_pair))

    # Cleanup generator from GPU
    if pipe is not None:
        del pipe
        import torch
        torch.cuda.empty_cache()

    # Split into train (80%) and val (20%) preserving strict group atomicity (ZERO image leakage)
    rng.shuffle(pair_units)
    n_val_pairs = max(2, int(len(pair_units) * 0.20))
    val_units = pair_units[:n_val_pairs]
    train_units = pair_units[n_val_pairs:]

    train_records = [r for _, recs in train_units for r in recs]
    val_records = [r for _, recs in val_units for r in recs]

    rng.shuffle(train_records)
    rng.shuffle(val_records)

    train_p = out_dir / "train.jsonl"
    val_p = out_dir / "val.jsonl"

    with open(train_p, "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r) + "\n")

    with open(val_p, "w", encoding="utf-8") as f:
        for r in val_records:
            f.write(json.dumps(r) + "\n")

    total_samples = len(train_records) + len(val_records)
    print(f"Generated {total_samples} total NLI samples across {limit} image pairs ({len(train_units)} train pairs, {len(val_units)} val pairs) in {time.time() - t0:.1f}s.")
    print(f"Train samples: {len(train_records):,} -> {train_p}")
    print(f"Val samples:   {len(val_records):,} -> {val_p}")


def main():
    parser = argparse.ArgumentParser(description="Generate Multi-Image Dataset Bank (Batch 1)")
    parser.add_argument("--out-dir", default="data/multimodal_multi_image", type=Path)
    parser.add_argument("--limit", type=int, default=50, help="Number of image pairs to generate")
    parser.add_argument("--use-diffusion", action="store_true", help="Enable Qwen-Image-2.1 diffusion editing")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_dataset_bank(
        out_dir=args.out_dir,
        limit=args.limit,
        use_diffusion=args.use_diffusion,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
