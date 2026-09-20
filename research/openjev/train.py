#!/usr/bin/env python
"""Fine-tune a Qwen3.5 model as a 3-way NLI cross-encoder on AllNLI (SNLI + MNLI).

Label order follows dleemiller/ModernCE-large-nli: 0=contradiction, 1=entailment, 2=neutral.
Usage:
    python train.py --model Qwen/Qwen3.5-0.8B --out ckpt/qwen3.5-0.8b-nli
    python train.py --model Qwen/Qwen3.5-9B  --out ckpt/qwen3.5-9b-nli --lora --grad-ckpt
"""
import argparse
import json
import math
import os
import random

import functools

import numpy as np
import torch
from datasets import concatenate_datasets, load_dataset, load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

ID2LABEL = {0: "contradiction", 1: "entailment", 2: "neutral"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}
# SNLI / MNLI native: 0=entailment, 1=neutral, 2=contradiction
NATIVE2OURS = {0: 1, 1: 2, 2: 0}
TEMPLATE = "Premise: {premise}\nHypothesis: {hypothesis}"
IMG_MARK = "<<IMG>>"  # data_mix.py puts this where the image-token block must go


def format_pair(premise: str, hypothesis: str) -> str:
    return TEMPLATE.format(premise=premise.strip(), hypothesis=hypothesis.strip())


def load_allnli(n_train: int, n_val: int, seed: int):
    snli = load_dataset("stanfordnlp/snli", split="train")
    mnli = load_dataset("nyu-mll/multi_nli", split="train")
    cols = ["premise", "hypothesis", "label"]
    train = concatenate_datasets([snli.select_columns(cols), mnli.select_columns(cols)])
    train = train.filter(lambda x: x["label"] in (0, 1, 2), num_proc=8)
    train = train.shuffle(seed=seed).select(range(min(n_train, len(train))))
    val = load_dataset("nyu-mll/multi_nli", split="validation_matched").select_columns(cols)
    val = val.filter(lambda x: x["label"] in (0, 1, 2)).shuffle(seed=seed).select(range(n_val))
    return train, val


def load_mix(path, n_train, n_val, seed):
    """Mixture rows carry labels ALREADY in our space (0=con,1=ent,2=neu) -- NATIVE2OURS is never applied here."""
    ds = load_from_disk(os.path.join(path, "mix"))
    train, val = ds["train"], ds["val"]
    if n_train and n_train < len(train):
        train = train.shuffle(seed=seed).select(range(n_train))
    if n_val and n_val < len(val):
        val = val.select(range(n_val))
    labels = set(train.unique("label"))
    assert labels <= {0, 1, 2}, f"mixture labels out of range: {labels}"
    return train, val


class FastPatchEmbed(torch.nn.Module):
    """Qwen3.5's vision patch embed is a Conv3d whose bf16 cuDNN path costs ~2 s/frame; fp32 costs 0.3 ms.
    `autocast(enabled=False)` is mandatory under Trainer: accelerate wraps forward in bf16 autocast, which would cast
    the inputs straight back and restore the slow path. `weight` stays bf16 because the caller reads its dtype."""

    def __init__(self, conv):
        super().__init__()
        self.weight, self.bias, self.stride = conv.weight, conv.bias, conv.stride

    def forward(self, x):
        with torch.autocast("cuda", enabled=False):
            y = torch.nn.functional.conv3d(x.float(), self.weight.float(), self.bias.float(), stride=self.stride)
        return y.to(self.weight.dtype)


def freeze_vision(model, fast_patch=True):
    """Freeze the vision tower, make its patch embed fast, and run it under no_grad (it is frozen, so nothing flows;
    this also skips the pointless checkpoint wrapper gradient_checkpointing_enable() installs on each vision block)."""
    n_vis = 0
    for n, p in model.named_parameters():
        if "visual" in n:
            p.requires_grad = False
            n_vis += p.numel()
    vis = getattr(getattr(model, "model", model), "visual", None)
    if vis is not None and fast_patch:
        vis.patch_embed.proj = FastPatchEmbed(vis.patch_embed.proj)
        orig = vis.forward

        @functools.wraps(orig)
        def fwd(*a, **k):
            with torch.no_grad():
                return orig(*a, **k)

        vis.forward = fwd
    print(f"frozen visual params: {n_vis/1e6:.1f}M" + (" (fp32 patch-embed, no_grad forward)" if vis is not None and fast_patch else ""))
    return model


class DataCollatorNLIMM:
    """Pads pre-tokenized rows and, for rows that carry a JPEG, adds pixel_values / image_grid_thw / mm_token_type_ids.
    A batch with no image rows gets none of the three keys, so it is byte-identical to DataCollatorWithPadding."""

    KEYS = ("input_ids", "attention_mask", "labels")

    def __init__(self, tok, image_processor, img_id, image_root="", pad_to_multiple_of=8):
        self.tok, self.ip, self.img_id, self.root = tok, image_processor, img_id, image_root
        self.pad_to_multiple_of = pad_to_multiple_of
        self.last = None

    def __call__(self, features):
        from PIL import Image
        paths = [f.get("image") or "" for f in features]
        batch = self.tok.pad([{k: f[k] for k in self.KEYS if k in f} for f in features],
                             padding=True, pad_to_multiple_of=self.pad_to_multiple_of, return_tensors="pt")
        imgs = [p for p in paths if p]
        if not imgs:
            return batch
        pil = []
        for p in imgs:
            try:
                pil.append(Image.open(os.path.join(self.root, p)).convert("RGB"))
            except Exception as e:  # noqa: BLE001
                print(f"[collator] bad image {p}: {type(e).__name__}", flush=True)
                pil.append(Image.new("RGB", (320, 240), (128, 128, 128)))
        vis = self.ip(images=pil, return_tensors="pt")
        batch["pixel_values"] = vis["pixel_values"]
        batch["image_grid_thw"] = vis["image_grid_thw"]
        batch["mm_token_type_ids"] = (batch["input_ids"] == self.img_id).long()
        n_tok = int(batch["mm_token_type_ids"].sum())
        n_feat = int((vis["image_grid_thw"].prod(-1) // self.ip.merge_size ** 2).sum())
        if n_tok != n_feat:
            raise ValueError(f"image-pad tokens {n_tok} != merged patches {n_feat} (rows={len(features)}, imgs={imgs[:3]})")
        self.last = batch
        return batch


class MixTrainer(Trainer):
    """Trainer whose length-grouped sampler gets lengths as a plain Python list. datasets 5.0 returns a lazy Arrow
    `Column` for ds["length"]; LengthGroupedSampler sorts megabatches with `lengths[i]` (~32 us per access through
    Arrow), which for 1.26M rows stalls the first step for 10+ minutes on every rank."""

    def __init__(self, *a, train_lengths=None, **k):
        super().__init__(*a, **k)
        self._train_lengths = train_lengths

    def _get_train_sampler(self, train_dataset=None):
        if self.args.train_sampling_strategy == "group_by_length" and self._train_lengths is not None:
            from transformers.trainer_pt_utils import LengthGroupedSampler
            return LengthGroupedSampler(self.args.train_batch_size * self.args.gradient_accumulation_steps,
                                        lengths=self._train_lengths)
        return super()._get_train_sampler(train_dataset)


def add_smoke_callback(trainer, collator, tok, img_id, n_img_tokens):
    """One-shot check that the image path is real: token/patch counts line up, blanking the pixels moves the image
    rows' logits and leaves the text rows bit-identical, and the vision tower is frozen and fast."""
    from transformers import TrainerCallback

    class Smoke(TrainerCallback):
        """The real collator runs inside dataloader workers, so the check builds its own batch in-process."""
        t = {}

        def on_step_end(self, a, state, control, model=None, **kw):
            import time
            if state.global_step in (5, 35):
                torch.cuda.synchronize(); self.t[state.global_step] = time.perf_counter()
                if state.global_step == 35:
                    print(f"[smoke] step time {(self.t[35]-self.t[5])/30:.3f} s/step (steps 5..35)", flush=True)
            if state.global_step != 1:
                return control
            ds = trainer.train_dataset
            scan = range(min(4000, len(ds)))
            img_idx = [i for i in scan if ds[i]["image"]][:6]
            txt_idx = [i for i in scan if not ds[i]["image"]][:6]
            assert img_idx, "no image rows in the first 4000 training rows"
            b = collator([ds[i] for i in txt_idx[:3] + img_idx + txt_idx[3:]])
            b = {k: (v.cuda() if hasattr(v, "cuda") else v) for k, v in b.items()}
            n_tok = int((b["input_ids"] == img_id).sum())
            n_rows = int(b["image_grid_thw"].shape[0])
            assert n_tok == n_img_tokens * n_rows, (n_tok, n_img_tokens, n_rows)
            assert b["mm_token_type_ids"].shape == b["input_ids"].shape
            assert (b["attention_mask"].diff(dim=1) <= 0).all(), "batch is not right-padded"
            m = model.module if hasattr(model, "module") else model
            assert not any(p.requires_grad for n, p in m.named_parameters() if "visual" in n), "vision tower is trainable"
            feed = {k: v for k, v in b.items() if k != "labels"}
            was_training = m.training
            m.eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                lg = m(**feed).logits.float()
                blank = dict(feed); blank["pixel_values"] = torch.zeros_like(feed["pixel_values"])
                lg0 = m(**blank).logits.float()
            rows = b["mm_token_type_ids"].sum(1) > 0
            d_img = float((lg[rows] - lg0[rows]).abs().max())
            d_txt = float((lg[~rows] - lg0[~rows]).abs().max()) if (~rows).any() else 0.0
            assert d_img > 1e-2, f"blanking the image moved image-row logits by only {d_img} -- the tower is dead"
            assert d_txt < 1e-3, f"blanking the image moved TEXT rows by {d_txt} -- cross-row leakage"
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                m.model.visual(b["pixel_values"], grid_thw=b["image_grid_thw"])
                torch.cuda.synchronize(); t0 = time.perf_counter()
                for _ in range(5):
                    m.model.visual(b["pixel_values"], grid_thw=b["image_grid_thw"])
                torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / 5
            if was_training:
                m.train()
            assert dt < 0.2, f"vision forward {dt*1e3:.0f} ms under autocast -- FastPatchEmbed lost to autocast"
            print(f"[smoke] OK: image rows {n_rows}/{b['input_ids'].shape[0]}, shape {tuple(b['input_ids'].shape)}, "
                  f"d_img {d_img:.3f} d_text {d_txt:.5f}, vision fwd {dt*1e3:.1f} ms for {n_rows} images, "
                  f"peak {torch.cuda.max_memory_allocated()/2**30:.1f} GiB", flush=True)
            return control

    trainer.add_callback(Smoke())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=200_000)
    ap.add_argument("--n-val", type=int, default=2000)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-targets", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,in_proj_qkv,in_proj_z,in_proj_a,in_proj_b,out_proj")
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--head-only", action="store_true", help="freeze the backbone, train only the `score` head")
    ap.add_argument("--eval-steps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-steps", type=int, default=-1, help="debug: stop early")
    ap.add_argument("--device-map", default=None, help='e.g. "auto" to shard a big model over all visible GPUs')
    ap.add_argument("--data", choices=["allnli", "mix"], default="allnli")
    ap.add_argument("--mix-dir", default=None, help="dir written by data_mix.py build (holds mix/ and images/)")
    ap.add_argument("--image-processor", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--group-by-length", action="store_true")
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="assert the image path really works, then keep training")
    ap.add_argument("--eval-mnli", action="store_true", default=True, help="keep MNLI-m as a second eval set")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if "moe" in args.model.lower() or "a3b" in args.model.lower():
        import modeling_qwen35_moe_seqcls  # noqa: F401  registers Qwen3_5MoeForSequenceClassification
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    def encode(batch):
        texts = [format_pair(p, h) for p, h in zip(batch["premise"], batch["hypothesis"])]
        enc = tok(texts, truncation=True, max_length=args.max_len)
        enc["labels"] = [NATIVE2OURS[l] for l in batch["label"]]
        return enc

    if args.data == "allnli":
        train, val = load_allnli(args.n_train, args.n_val, args.seed)
        print(f"train={len(train)} val={len(val)}")
        train = train.map(encode, batched=True, remove_columns=train.column_names, num_proc=8)
        val = val.map(encode, batched=True, remove_columns=val.column_names)
        eval_sets = val
    else:
        assert args.mix_dir, "--data mix needs --mix-dir"
        img_id = tok.convert_tokens_to_ids("<|image_pad|>")
        img_block = None  # filled once the image processor tells us the token count

        from transformers import AutoImageProcessor
        image_processor = AutoImageProcessor.from_pretrained(args.image_processor)
        from PIL import Image as _PILImage
        _probe = image_processor(images=[_PILImage.new("RGB", (320, 240))], return_tensors="pt")
        n_img_tokens = int(_probe["image_grid_thw"].prod()) // image_processor.merge_size ** 2
        img_block = "<|vision_start|>" + "<|image_pad|>" * n_img_tokens + "<|vision_end|>"
        print(f"image block: {n_img_tokens} tokens, pixel_values {tuple(_probe['pixel_values'].shape)}")

        def encode_mix(batch):
            """Labels pass through untouched (already ours). Over-long rows are truncated on the PREMISE side, so the
            hypothesis -- and the image block that sits inside the premise -- always survive."""
            ids_out, am_out, lab, ln, im_out = [], [], [], [], []
            for pr, hy, y, im in zip(batch["premise"], batch["hypothesis"], batch["label"], batch["image"]):
                pr = pr.replace(IMG_MARK, img_block) if im else pr
                ids = tok(format_pair(pr, hy), add_special_tokens=False)["input_ids"]
                if len(ids) > args.max_len:
                    left = tok("Premise: " + pr.strip(), add_special_tokens=False)["input_ids"]
                    right = tok("\nHypothesis: " + hy.strip(), add_special_tokens=False)["input_ids"][: args.max_len - 16]
                    keep = max(args.max_len - len(right), 16)
                    if im:  # never cut into the image block: keep its head, drop text after it
                        first = next((i for i, t in enumerate(left) if t == img_id), None)
                        if first is not None:
                            keep = max(keep, first + n_img_tokens + 2)
                    ids = (left[:keep] + right)[: max(args.max_len, keep + len(right))]
                if im:
                    n = sum(1 for t in ids if t == img_id)
                    if n != n_img_tokens:
                        continue  # never feed a row whose image span got clipped
                ids_out.append(ids); am_out.append([1] * len(ids))
                lab.append(int(y)); ln.append(len(ids)); im_out.append(im or "")
            return {"input_ids": ids_out, "attention_mask": am_out, "labels": lab, "length": ln, "image": im_out}

        train, val = load_mix(args.mix_dir, args.n_train if args.n_train != 200_000 else 0, args.n_val, args.seed)
        print(f"train={len(train)} val={len(val)} (mixture)")
        cols = train.column_names
        from accelerate import PartialState
        _state = PartialState()  # under torchrun: rank 0 tokenizes and writes the cache, the others then read it
        with _state.main_process_first():
            train = train.map(encode_mix, batched=True, remove_columns=cols, num_proc=8)
            val = val.map(encode_mix, batched=True, remove_columns=cols, num_proc=4)
        eval_sets = {"mix": val}
        if args.eval_mnli:
            with _state.main_process_first():
                _, mnli_val = load_allnli(1, args.n_val, args.seed)
            mnli_val = mnli_val.map(
                lambda b: {**encode(b), "length": [len(x) for x in encode(b)["input_ids"]], "image": [""] * len(b["label"])},
                batched=True, remove_columns=mnli_val.column_names)
            eval_sets["mnli"] = mnli_val
        L = np.array(train["length"])
        print(f"[mix] tokens: mean {L.mean():.0f} p50 {np.percentile(L,50):.0f} p90 {np.percentile(L,90):.0f} "
              f"max {L.max()} | total {L.sum()/1e6:.0f}M | image rows {sum(1 for x in train['image'] if x)}")

    cls = AutoModelForSequenceClassification
    if "moe" in args.model.lower() or "a3b" in args.model.lower():
        from modeling_qwen35_moe_seqcls import Qwen3_5MoeForSequenceClassification as cls
    model = cls.from_pretrained(
        args.model,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        dtype=torch.bfloat16,
        device_map=args.device_map,
    )
    # Qwen3.5 config is composite (text_config inside); the seq-cls head reads get_text_config().pad_token_id
    model.config.get_text_config().pad_token_id = tok.pad_token_id
    model.config.pad_token_id = tok.pad_token_id
    model.config.nli_template = TEMPLATE  # consumed by eval.py
    model.config.use_cache = False
    # Qwen3.5 checkpoints carry a vision tower; it stays frozen in both paths (the mix path also makes it fast).
    freeze_vision(model, fast_patch=(args.data == "mix"))
    if args.head_only:
        for n, p in model.named_parameters():
            p.requires_grad = n.startswith("score")
        n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"head-only: trainable params {n_tr/1e3:.1f}K")

    if args.lora:
        from peft import LoraConfig, TaskType, get_peft_model

        lcfg = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=args.lora_r,
            lora_alpha=2 * args.lora_r,
            lora_dropout=0.05,
            target_modules=args.lora_targets.split(","),
            modules_to_save=["score"],
        )
        model = get_peft_model(model, lcfg)
        model.print_trainable_parameters()

    def compute_metrics(p):
        logits = p.predictions[0] if isinstance(p.predictions, (tuple, list)) else p.predictions
        preds = logits.argmax(-1)
        return {"accuracy": float((preds == p.label_ids).mean())}

    world = int(os.environ.get("WORLD_SIZE", 1))
    total_steps = args.max_steps if args.max_steps > 0 else int(
        math.ceil(len(train) / (args.bs * args.grad_accum * world)) * args.epochs)
    targs = TrainingArguments(
        output_dir=args.out + "_trainer",
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=64,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=max(1, int(0.03 * total_steps)),  # warmup_ratio was removed in transformers 5.15
        weight_decay=0.01,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        bf16=True,
        gradient_checkpointing=args.grad_ckpt,
        logging_steps=10 if args.smoke else 25,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 500,
        save_total_limit=1,
        report_to="none",
        dataloader_num_workers=6 if args.data == "mix" else 4,
        seed=args.seed,
        remove_unused_columns=False,
        label_names=["labels"],  # transformers 5.x leaves this empty -> no eval loss/metrics otherwise
        ddp_broadcast_buffers=False,
        **({"train_sampling_strategy": "group_by_length", "length_column_name": "length"} if args.group_by_length else {}),
    )
    collator = (DataCollatorNLIMM(tok, image_processor, img_id, image_root=args.mix_dir)
                if args.data == "mix" else DataCollatorWithPadding(tok))
    trainer = MixTrainer(
        model=model,
        args=targs,
        train_dataset=train,
        eval_dataset=eval_sets,
        data_collator=collator,
        compute_metrics=compute_metrics,
        train_lengths=[int(x) for x in np.asarray(train["length"])] if (args.data == "mix" and args.group_by_length) else None,
    )
    if args.smoke:
        add_smoke_callback(trainer, collator, tok, img_id, n_img_tokens)
    trainer.train()
    final = trainer.evaluate()
    print("final eval:", final)

    if args.lora:
        model = model.merge_and_unload()
    model.config.nli_template = TEMPLATE
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    with open(os.path.join(args.out, "train_result.json"), "w") as f:
        json.dump({"args": vars(args), "final_eval": final}, f, indent=2)
    print("saved to", args.out)


if __name__ == "__main__":
    main()
