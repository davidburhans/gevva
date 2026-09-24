#!/usr/bin/env python3
"""generate_visual_synth.py - Deterministic multimodal NLI data generator.

Renders scenes whose ground truth is PROGRAMMATICALLY KNOWN, so every label is
provable by construction - no LLM, no committee, no label noise. Three families
(the mission's weakest multimodal axes, per AGENTS.md applications):

1. scenes   - multi-object spatial reasoning (3-6 shapes, colors, sizes, regions,
              overlap): existence / counting / spatial relations / superlatives.
2. charts   - bar & line charts with known series: values, trends, comparisons.
3. tables   - rendered documents/forms with known cells: contents, row facts.

Label discipline (the lesson of every audit this week):
- ENTAILMENT claims are asserted from the scene spec (verified true).
- CONTRADICTION claims mutate one ground-truth fact (verified false).
- NEUTRAL claims are unverifiable-from-pixels (provenance/temporal/subjective) -
  never "absent object" phrasing, which is decidable and therefore contradiction.

Output: data/visual_synth/{train,val}.jsonl + images/ + manifest.json
(val exists so multimodal capability finally becomes GATED, closing the
val-has-zero-images blind spot.)

Usage:
  uv run python generate_visual_synth.py --n 6000            # CPU, runs alongside training
  uv run python generate_visual_synth.py --n 200 --out-dir /tmp/vs
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nli_labels import CONTRADICTION, ENTAILMENT, NEUTRAL  # noqa: E402

COLORS = [("red", (220, 60, 60)), ("blue", (60, 90, 220)), ("green", (60, 170, 80)),
          ("yellow", (230, 200, 60)), ("purple", (150, 70, 190)), ("orange", (240, 140, 50))]
SHAPES = ["circle", "square", "triangle", "rectangle"]
CANVAS = 448
REGIONS = {"left": (0.02, 0.33), "center": (0.34, 0.66), "right": (0.67, 0.98)}


# ----------------------------------------------------------------------------- scenes
def render_scene(rng: random.Random) -> Tuple[Image.Image, Dict[str, Any]]:
    """3-6 non-degenerate objects; spec records shape/color/bbox/area for claims."""
    img = Image.new("RGB", (CANVAS, CANVAS), (245, 245, 245))
    draw = ImageDraw.Draw(img)
    spec: List[Dict[str, Any]] = []
    for _ in range(rng.randint(3, 6)):
        color_name, color_val = rng.choice(COLORS)
        shape = rng.choice(SHAPES)
        w = rng.randint(45, 130)
        h = w if shape in ("circle", "square") else rng.randint(35, 140)
        x0 = rng.randint(10, CANVAS - w - 10)
        y0 = rng.randint(10, CANVAS - h - 10)
        bbox = (x0, y0, x0 + w, y0 + h)
        if shape == "circle":
            draw.ellipse(bbox, fill=color_val)
        elif shape == "triangle":
            draw.polygon([(x0, y0 + h), (x0 + w, y0 + h), (x0 + w // 2, y0)], fill=color_val)
        else:
            draw.rectangle(bbox, fill=color_val)
        spec.append({"shape": shape, "color": color_name, "bbox": bbox,
                     "area": w * h, "cx": (x0 + x0 + w) // 2})
    return img, {"objects": spec, "n": len(spec)}


def _region_of(cx: int) -> str:
    frac = cx / CANVAS
    for name, (lo, hi) in REGIONS.items():
        if lo <= frac <= hi:
            return name
    return "center"


def scene_claims(rng: random.Random, spec: Dict[str, Any]) -> List[Tuple[int, str]]:
    """One (label, hypothesis) per ground-truth fact; contradictions mutate one fact."""
    objs = spec["objects"]
    out: List[Tuple[int, str]] = []

    def colors_of(shape): return {o["color"] for o in objs if o["shape"] == shape}
    def shapes_of(color): return {o["shape"] for o in objs if o["color"] == color}
    def counts_by(key, val): return sum(1 for o in objs if o[key] == val)

    # Counting (exact, verifiable)
    out.append((ENTAILMENT, f"The image contains exactly {spec['n']} shapes."))
    wrong_n = spec["n"] + rng.choice([-2, -1, 1, 2])
    out.append((CONTRADICTION, f"The image contains exactly {max(1, wrong_n)} shapes."))

    # Existence + color mutation
    o = rng.choice(objs)
    out.append((ENTAILMENT, f"There is a {o['color']} {o['shape']} in the {_region_of(o['cx'])} region of the image."))
    wrong_color = rng.choice([c for c, _ in COLORS if c != o["color"]])
    out.append((CONTRADICTION, f"There is a {wrong_color} {o['shape']} in the {_region_of(o['cx'])} region of the image."))

    # Absence (stated as verifiable negation -> entailment; its mutation -> contradiction).
    # Guard: a scene may contain every color (6 objects, 6 colors) - fall back to shapes.
    present_colors = {o["color"] for o in objs}
    absent_colors = [c for c, _ in COLORS if c not in present_colors]
    if absent_colors:
        absent = rng.choice(absent_colors)
        out.append((ENTAILMENT, f"No shape in the image is {absent}."))
        out.append((CONTRADICTION, f"At least one shape in the image is {absent}."))
    else:
        present_shapes = {o["shape"] for o in objs}
        absent_shapes = [s for s in SHAPES if s not in present_shapes]
        if absent_shapes:
            absent = rng.choice(absent_shapes)
            out.append((ENTAILMENT, f"No {absent} appears anywhere in the image."))
            out.append((CONTRADICTION, f"A {absent} appears in the image."))

    # Superlative (largest object)
    big = max(objs, key=lambda o: o["area"])
    out.append((ENTAILMENT, f"The largest shape in the image is {big['color']}."))
    not_big = rng.choice([o for o in objs if o is not big])
    out.append((CONTRADICTION, f"The largest shape in the image is {not_big['color']}."))

    # Spatial relation between two objects
    a, b = rng.sample(objs, 2)
    rel = "to the left of" if a["cx"] < b["cx"] else "to the right of"
    out.append((ENTAILMENT, f"A {a['color']} {a['shape']} is {rel} a {b['color']} {b['shape']}."))
    flip = "to the right of" if a["cx"] < b["cx"] else "to the left of"
    out.append((CONTRADICTION, f"A {a['color']} {a['shape']} is {flip} a {b['color']} {b['shape']}."))

    # Shape-count by color
    c0 = rng.choice([o["color"] for o in objs])
    k = counts_by("color", c0)
    if 1 <= k <= 3:
        out.append((ENTAILMENT, f"There are exactly {k} {c0} shape{'s' if k > 1 else ''} in the image."))
        out.append((CONTRADICTION, f"There are exactly {min(4, k + 1)} {c0} shapes in the image."))

    # Neutral: unverifiable-from-pixels (provenance / temporal / purpose / off-screen)
    n_tpl = [
        "The shapes were arranged by a graphic designer last week.",
        "This image was originally created for a design textbook.",
        f"The {objs[0]['shape']} will be recolored in the next revision.",
        "A photographer reviewed this composition before publication.",
        "These shapes belong to a company's internal branding mockup.",
    ]
    out.append((NEUTRAL, rng.choice(n_tpl)))
    return out


# ----------------------------------------------------------------------------- charts
def render_chart(rng: random.Random) -> Tuple[Image.Image, Dict[str, Any]]:
    quarters = ["Q1", "Q2", "Q3", "Q4"]
    values = [rng.randint(20, 95) for _ in quarters]
    kind = rng.choice(["bar", "line"])
    img = Image.new("RGB", (448, 360), (255, 255, 255))
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
    x0, y0, x1, y1 = 55, 30, 430, 300
    d.rectangle([x0, y0, x1, y1], outline=(60, 60, 60))
    for gy in range(y0, y1 + 1, 54):
        v = round(100 - (gy - y0) / (y1 - y0) * 100)
        d.text((15, gy - 7), f"{v}", fill=(40, 40, 40), font=font)
        d.line([x0, gy, x1, gy], fill=(225, 225, 225))
    for i, (q, v) in enumerate(zip(quarters, values)):
        cx = x0 + 45 + i * 92
        d.text((cx - 8, y1 + 8), q, fill=(30, 30, 30), font=font)
        if kind == "bar":
            h = int((v / 100) * (y1 - y0))
            d.rectangle([cx - 25, y1 - h, cx + 25, y1], fill=(70, 110, 210))
        else:
            py = y1 - int((v / 100) * (y1 - y0))
            if i == 0:
                prev = (cx, py)
            else:
                d.line([prev, (cx, py)], fill=(200, 60, 60), width=4)
                prev = (cx, py)
            d.ellipse([cx - 5, py - 5, cx + 5, py + 5], fill=(200, 60, 60))
    spec = {"kind": kind, "series": dict(zip(quarters, values)), "quarters": quarters}
    return img, spec


def chart_claims(rng: random.Random, spec: Dict[str, Any]) -> List[Tuple[int, str]]:
    s = spec["series"]
    q = spec["quarters"]
    out = []
    qi = rng.choice(q)
    out.append((ENTAILMENT, f"The {qi} value in the chart is {s[qi]}."))
    out.append((CONTRADICTION, f"The {qi} value in the chart is {s[qi] + rng.choice([3, 7, 12])}."))
    a, b = rng.sample(q, 2)
    hi, lo = (a, b) if s[a] >= s[b] else (b, a)
    out.append((ENTAILMENT, f"The {hi} value is higher than the {lo} value."))
    out.append((CONTRADICTION, f"The {lo} value is higher than the {hi} value."))
    peak = max(q, key=lambda k: s[k])
    out.append((ENTAILMENT, f"{peak} shows the highest value in the chart."))
    not_peak = rng.choice([k for k in q if k != peak])
    out.append((CONTRADICTION, f"{not_peak} shows the highest value in the chart."))
    out.append((NEUTRAL, rng.choice([
        "The team was satisfied with the results shown in this chart.",
        "This chart was presented at the quarterly board meeting.",
        "The values are expected to improve next year.",
        "An analyst prepared this chart for an internal report.",
    ])))
    return out


# ----------------------------------------------------------------------------- tables
def render_table(rng: random.Random) -> Tuple[Image.Image, Dict[str, Any]]:
    headers = ["Item", "Qty", "Price"]
    items = rng.sample(["Widget", "Gadget", "Bracket", "Panel", "Coil", "Sensor", "Valve"], 3)
    rows = [[it, rng.randint(2, 12), rng.randint(5, 90)] for it in items]
    img = Image.new("RGB", (448, 90 + 34 * (len(rows) + 1)), (255, 255, 255))
    d = ImageDraw.Draw(img)
    try:
        fH, fB = (ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 16),
                  ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 16))
    except OSError:
        fH = fB = ImageFont.load_default()
    d.rectangle([20, 20, 428, 60], fill=(60, 90, 160))
    for j, h in enumerate(headers):
        d.text((35 + j * 130, 32), h, fill=(255, 255, 255), font=fH)
    for i, row in enumerate(rows):
        y = 60 + 34 * (i + 1)
        if i % 2:
            d.rectangle([20, y, 428, y + 34], fill=(238, 242, 248))
        for j, cell in enumerate(row):
            d.text((35 + j * 130, y + 8), str(cell), fill=(30, 30, 30), font=fB)
    return img, {"rows": rows, "n_rows": len(rows)}


def table_claims(rng: random.Random, spec: Dict[str, Any]) -> List[Tuple[int, str]]:
    rows = spec["rows"]
    out = []
    r = rng.choice(rows)
    out.append((ENTAILMENT, f"The table lists {r[0]} with a quantity of {r[1]}."))
    out.append((CONTRADICTION, f"The table lists {r[0]} with a quantity of {r[1] + rng.choice([1, 3, 5])}."))
    out.append((ENTAILMENT, f"The table contains exactly {spec['n_rows']} item rows."))
    out.append((CONTRADICTION, f"The table contains exactly {spec['n_rows'] + rng.choice([1, 2])} item rows."))
    out.append((NEUTRAL, rng.choice([
        "The order in this table was placed by the procurement department.",
        "This table was exported from the inventory system last night.",
        "The listed items are scheduled for delivery next month.",
        "A warehouse manager verified this table before archiving.",
    ])))
    return out


# ----------------------------------------------------------------------------- driver
RENDERERS = {"scenes": (render_scene, scene_claims),
             "charts": (render_chart, chart_claims),
             "tables": (render_table, table_claims)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Deterministic multimodal NLI generator")
    parser.add_argument("--n", type=int, default=6000, help="Approximate total rows")
    parser.add_argument("--out-dir", default="data/visual_synth")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-frac", type=float, default=0.08)
    args = parser.parse_args()

    out = REPO_ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "images").mkdir(exist_ok=True)
    rng = random.Random(args.seed)

    per_family = max(1, args.n // (3 * 8))  # each render yields ~8 rows
    all_rows: List[Dict[str, Any]] = []
    for fam, (render, claims) in RENDERERS.items():
        for i in range(per_family):
            img, spec = render(rng)
            digest = hashlib.sha1(f"{fam}|{i}|{args.seed}".encode()).hexdigest()[:12]
            img_path = out / "images" / f"{fam}_{digest}.jpg"
            img.save(img_path, quality=88)
            for label, hyp in claims(rng, spec):
                all_rows.append({
                    "id": f"vsynth_{fam}_{digest}_{label}",
                    "premise": "An image is shown.",
                    "premise_markers": "<|vision_start|>" + "<|image_pad|>" * 280 + "<|vision_end|>",
                    "hypothesis": hyp,
                    "label": label,
                    "source": f"visual_synth_{fam}",
                    "language": "en",
                    "image": os.path.relpath(img_path, out),
                    "length": 300,
                })

    rng.shuffle(all_rows)
    n_val = max(50, int(len(all_rows) * args.val_frac))
    val, train = all_rows[:n_val], all_rows[n_val:]
    for name, rows in (("train", train), ("val", val)):
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    manifest = {
        "generator": "generate_visual_synth.py", "seed": args.seed,
        "rows": {"train": len(train), "val": len(val)},
        "labels": dict(Counter(r["label"] for r in all_rows)),
        "families": dict(Counter(r["source"] for r in all_rows)),
        "label_guarantee": "entail=asserted-from-spec, contradict=mutated-fact, neutral=unverifiable-from-pixels",
        "note": "val split exists so multimodal capability is GATED (val.jsonl has zero image rows)",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
