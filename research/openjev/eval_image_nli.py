#!/usr/bin/env python
"""Image-premise NLI accuracy on a held-out jsonl written by `data_mix.py images` (e.g. from VQAv2_validation, never trained).

    python eval_image_nli.py --models ckpt/qwen3.5-4b-nli /mnt/qwen_nli_ckpt/qwen3.5-4b-nli-v2 \
        --data /mnt/nli_eval_vqa/parts/vqa_val.jsonl --image-root /mnt/nli_eval_vqa --out results/image_nli.json
Reports 3-class accuracy overall and per source (answer / yes-no / disagreement / spatial claims).
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForSequenceClassification, AutoTokenizer

from train import IMG_MARK, TEMPLATE, DataCollatorNLIMM, freeze_vision


@torch.no_grad()
def score(model_path, rows, root, bs, ip_name):
    tok = AutoTokenizer.from_pretrained(model_path); tok.padding_side = "right"
    model = AutoModelForSequenceClassification.from_pretrained(model_path, dtype=torch.bfloat16).cuda().eval()
    model.config.get_text_config().pad_token_id = tok.pad_token_id
    freeze_vision(model, fast_patch=True)
    ip = AutoImageProcessor.from_pretrained(ip_name)
    img_id = tok.convert_tokens_to_ids("<|image_pad|>")
    from PIL import Image
    n_img = int(ip(images=[Image.new("RGB", (320, 240))], return_tensors="pt")["image_grid_thw"].prod()) // ip.merge_size ** 2
    block = "<|vision_start|>" + "<|image_pad|>" * n_img + "<|vision_end|>"
    template = getattr(model.config, "nli_template", None) or TEMPLATE
    coll = DataCollatorNLIMM(tok, ip, img_id, image_root=root)
    preds = []
    for s in range(0, len(rows), bs):
        feats = []
        for r in rows[s:s + bs]:
            text = template.format(premise=r["premise"].replace(IMG_MARK, block).strip(), hypothesis=r["hypothesis"].strip())
            ids = tok(text, add_special_tokens=False)["input_ids"]
            feats.append({"input_ids": ids, "attention_mask": [1] * len(ids), "labels": r["label"], "image": r["image"]})
        b = coll(feats)
        b = {k: v.cuda() for k, v in b.items() if k != "labels"}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**b).logits.float()
        preds.append(logits.argmax(-1).cpu().numpy())
    del model; torch.cuda.empty_cache()
    return np.concatenate(preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--image-processor", default="Qwen/Qwen3.5-4B")
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.data)]
    gold = np.array([r["label"] for r in rows])
    src = [r["source"] for r in rows]
    print(f"{len(rows)} rows; labels {np.bincount(gold, minlength=3).tolist()}")
    res = {}
    for m in args.models:
        pred = score(m, rows, args.image_root, args.bs, args.image_processor)
        by = defaultdict(list)
        for p, g, s in zip(pred, gold, src):
            by[s].append(p == g)
        res[m] = {"acc": float((pred == gold).mean()), "n": len(rows),
                  "pred_dist": np.bincount(pred, minlength=3).tolist(),
                  "by_source": {k: round(float(np.mean(v)), 4) for k, v in sorted(by.items())}}
        print(m, json.dumps(res[m]), flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
