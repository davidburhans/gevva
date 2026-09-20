#!/usr/bin/env python
"""Build the openjev v2 training mixture: hard NLI + long documents + image premises + agentic traces.

Every adapter emits rows in ONE schema, with labels already in our space (0=contradiction, 1=entailment, 2=neutral):

    {"premise": str, "hypothesis": str, "label": int, "source": str, "image": str}

`image` is "" for text rows, else a path relative to the mixture dir. When an image is present the premise carries the
literal marker `<<IMG>>` at the position where the `<|vision_start|><|image_pad|>*n<|vision_end|>` block must go
(train.py substitutes it, because only the tokenizer knows the real token count).

Subcommands (each writes parts/<name>.jsonl incrementally, so a crash never loses finished work):

    python data_mix.py images  --out /mnt/nli_mix --n-rows 120000     # VQAv2 stream -> jpegs + claims
    python data_mix.py text    --out /mnt/nli_mix                     # SNLI/MNLI/ANLI/WANLI/FEVER/... + haystack
    python data_mix.py agentic --out /mnt/nli_mix                     # xlam / AgentTraj-L / AgentInstruct / When2Call / synth
    python data_mix.py build   --out /mnt/nli_mix                     # merge, balance, leakage-check, save_to_disk
"""
import argparse
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

OURS = {"contradiction": 0, "entailment": 1, "neutral": 2}
SYN = {
    "entailment": "entailment", "entails": "entailment", "entailed": "entailment", "supports": "entailment",
    "contradiction": "contradiction", "contradicts": "contradiction", "refutes": "contradiction",
    "neutral": "neutral", "not_entailment": "neutral", "not-entailed": "neutral", "not entailment": "neutral",
    "not enough info": "neutral", "not_enough_info": "neutral", "nei": "neutral",
}
IMG = "<<IMG>>"

# Datasets/splits that eval.py reports on. Never train on them, in any split (see plan: strict zero-shot).
BANNED = {
    "nyu-mll/multi_nli": {"validation_matched", "validation_mismatched"},
    "cais/mmlu": "*", "allenai/ai2_arc": "*", "allenai/winogrande": "*", "Rowan/hellaswag": "*",
    "openai/gsm8k": "*", "Idavidrein/gpqa": "*",
}


def norm_label(value, feature=None):
    """Resolve any source's label to our id, by NAME (never by position)."""
    name = value
    if isinstance(value, int) and feature is not None and hasattr(feature, "names"):
        name = feature.names[value]
    if not isinstance(name, str):
        raise ValueError(f"cannot resolve label {value!r} (feature={feature})")
    key = SYN.get(name.strip().lower())
    if key is None:
        raise ValueError(f"unknown label name {name!r}")
    return OURS[key]


def check_not_banned(repo, split):
    banned = BANNED.get(repo)
    if banned == "*" or (banned and split in banned):
        raise RuntimeError(f"LEAKAGE: {repo}:{split} is an eval set and must never be used for training")


class Part:
    """Incremental jsonl writer; a teardown crash keeps everything already written."""

    def __init__(self, out, name):
        os.makedirs(os.path.join(out, "parts"), exist_ok=True)
        self.path = os.path.join(out, "parts", f"{name}.jsonl")
        self.f = open(self.path, "w")
        self.n = 0
        self.by_label = Counter()

    def add(self, premise, hypothesis, label, source, image=""):
        premise, hypothesis = premise.strip(), hypothesis.strip()
        if not premise or not hypothesis:
            return
        self.f.write(json.dumps({"premise": premise, "hypothesis": hypothesis, "label": int(label),
                                 "source": source, "image": image}, ensure_ascii=False) + "\n")
        self.n += 1
        self.by_label[int(label)] += 1
        if self.n % 20000 == 0:
            self.f.flush()
            print(f"  {os.path.basename(self.path)}: {self.n} rows", flush=True)

    def close(self):
        self.f.close()
        print(f"[part] {os.path.basename(self.path)}: {self.n} rows, labels {dict(self.by_label)}", flush=True)


# --------------------------------------------------------------------------------------- text NLI
def take(ds, n, seed):
    if n and len(ds) > n:
        ds = ds.shuffle(seed=seed).select(range(n))
    return ds


def build_text(args):
    from datasets import concatenate_datasets, get_dataset_config_names, load_dataset
    seed = args.seed
    p = Part(args.out, "text")
    pool = []  # filler sentences for the haystack block
    pairs_for_haystack = []

    def emit(ds, src, prem_col="premise", hyp_col="hypothesis", lab_col="label", keep_for_haystack=False):
        feat = ds.features.get(lab_col)
        kept = 0
        for ex in ds:
            v = ex[lab_col]
            if isinstance(v, int) and v < 0:
                continue
            try:
                y = norm_label(v, feat)
            except ValueError:
                continue
            pr, hy = ex[prem_col], ex[hyp_col]
            if not pr or not hy:
                continue
            p.add(pr, hy, y, src)
            kept += 1
            if len(pool) < 200_000:
                pool.append(pr.strip())
            if keep_for_haystack and len(pairs_for_haystack) < 250_000:
                pairs_for_haystack.append((pr.strip(), hy.strip(), y))
        print(f"[text] {src}: {kept}", flush=True)

    # core NLI
    for repo, n in [("stanfordnlp/snli", args.n_snli), ("nyu-mll/multi_nli", args.n_mnli)]:
        check_not_banned(repo, "train")
        emit(take(load_dataset(repo, split="train"), n, seed), repo.split("/")[-1], keep_for_haystack=True)

    # adversarial
    anli = load_dataset("facebook/anli")
    emit(concatenate_datasets([anli[f"train_r{i}"] for i in (1, 2, 3)]), "anli", keep_for_haystack=True)

    wanli = load_dataset("alisawuffles/WANLI", split="train")
    emit(wanli, "wanli", lab_col="gold")

    # evidence-grounded
    fever = take(load_dataset("pietrolesci/nli_fever", split="train"), args.n_fever, seed)
    emit(fever, "nli_fever", keep_for_haystack=True)

    emit(load_dataset("tasksource/lingnli", split="train"), "lingnli")
    emit(load_dataset("tasksource/ConTRoL-nli", split="train"), "control")

    sci = load_dataset("allenai/scitail", "snli_format", split="train")
    emit(sci, "scitail", prem_col="sentence1", hyp_col="sentence2", lab_col="gold_label")

    qnli = take(load_dataset("nyu-mll/glue", "qnli", split="train"), args.n_qnli, seed)
    emit(qnli, "qnli", prem_col="sentence", hyp_col="question")

    for cfg in get_dataset_config_names("tasksource/babi_nli"):
        try:
            emit(load_dataset("tasksource/babi_nli", cfg, split="train"), f"babi:{cfg}")
        except Exception as e:  # noqa: BLE001
            print(f"[text] babi:{cfg} skipped: {type(e).__name__}", flush=True)

    p.close()

    # ------------------------------------------------------------------ haystack (long-document entailment)
    rng = random.Random(seed)
    h = Part(args.out, "haystack")
    rng.shuffle(pairs_for_haystack)
    for prem, hyp, y in pairs_for_haystack[: args.n_haystack]:
        filler = rng.sample(pool, rng.randint(15, 40))
        drop = rng.random() < 0.30  # 30%: the evidence is NOT in the document -> "not stated" == neutral
        if drop:
            doc, label = filler, OURS["neutral"]
        else:
            pos = rng.randrange(len(filler) + 1)
            doc, label = filler[:pos] + [prem] + filler[pos:], y
        h.add("\n\n".join(doc), hyp, label, "haystack_drop" if drop else "haystack")
    h.close()


# --------------------------------------------------------------------------------------- images (VQAv2)
NUM_WORDS = {"0": "no", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five"}
LEAD_INS = ["A photograph:", "A photo:", "A picture:", "An image:", "A photograph of a scene:"]
LEAD_PIX = "A photograph, 320 pixels wide, the centre at x = 160:"


def q_to_statement(q, a, atype):
    """VQA question + answer -> a declarative claim. Returns (statement, flip_ok) where flip_ok means
    a yes/no question whose 'no' answer turns the same statement into a contradiction."""
    q = q.strip().rstrip("?").strip()
    a = a.strip().lower()
    ql = q.lower()
    m = re.match(r"^what colou?r (?:is|are) (?:the )?(.+)$", ql)
    if m:
        return f"The {m.group(1)} is {a}.", False
    m = re.match(r"^how many (.+)$", ql)
    if m:
        noun = m.group(1)
        noun = re.sub(r"\b(are|is|can|do|does|there|in|on|the|this|that|photo|picture|image)\b.*$", "", noun).strip()
        if noun:
            return f"There are {NUM_WORDS.get(a, a)} {noun} in the image.", False
    if atype == "yes/no":
        m = re.match(r"^(is|are|was|were|does|do|did|can|has|have|will)\s+(.+)$", ql)
        if m:
            aux, rest = m.group(1), m.group(2)
            if aux in ("is", "are", "was", "were"):
                parts = rest.split(" ", 1)
                if len(parts) == 2:
                    subj, tail = parts
                    return f"{subj.capitalize()} {aux} {tail}.", True
            else:
                return f"{rest.capitalize()} ({aux}).".replace(" ().", "."), True
    if atype == "yes/no":
        return f'The answer to "{q}?" is {a}.', False
    return f'In this image, the answer to "{q}?" is {a}.', False


def spatial_claims(dets, W, H, rng):
    """Templated claims from DETA detections, in the coordinates of the RESIZED 320x240 frame.
    Returns a list of (hypothesis, label)."""
    out = []
    if not dets:
        return out
    sx, sy = 320.0 / max(W, 1), 240.0 / max(H, 1)
    boxes = []
    for d in dets:
        b = d.get("box")
        lab = (d.get("label") or "").strip().lower()
        if not b or len(b) != 4 or not lab:
            continue
        boxes.append((lab, [b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy]))
    if not boxes:
        return out
    present = Counter(l for l, _ in boxes)
    labels = sorted(present)

    def third(cx):
        return "left" if cx < 320 / 3 else ("right" if cx > 2 * 320 / 3 else "middle")

    lab, box = rng.choice(boxes)
    cx = (box[0] + box[2]) / 2
    t = third(cx)
    out.append((f"There is a {lab} in the {t} third of the image.", OURS["entailment"]))
    wrong = rng.choice([x for x in ("left", "middle", "right") if x != t])
    out.append((f"There is a {lab} in the {wrong} third of the image.", OURS["contradiction"]))
    # pixel-coordinate form: mirrors the `pixels` variant of doom_vision.py, the best one on Doom
    if cx < 110:
        out.append((f"The {lab} is at x < 110, on the left.", OURS["entailment"]))
        out.append((f"The {lab} is at x > 210, on the right.", OURS["contradiction"]))
    elif cx > 210:
        out.append((f"The {lab} is at x > 210, on the right.", OURS["entailment"]))
        out.append((f"The {lab} is at x < 110, on the left.", OURS["contradiction"]))
    else:
        out.append((f"The {lab} is near the centre, around x = 160.", OURS["entailment"]))
        out.append((f"The {lab} is at x < 110, on the left.", OURS["contradiction"]))
    # counting
    n = present[lab]
    out.append((f"There are {NUM_WORDS.get(str(n), n)} {lab}s in the image.", OURS["entailment"]))
    out.append((f"There are {NUM_WORDS.get(str(n + 2), n + 2)} {lab}s in the image.", OURS["contradiction"]))
    # relative position
    if len(labels) >= 2:
        a, b = rng.sample(labels, 2)
        ca = sum((bx[0] + bx[2]) / 2 for l, bx in boxes if l == a) / present[a]
        cb = sum((bx[0] + bx[2]) / 2 for l, bx in boxes if l == b) / present[b]
        rel, anti = ("left", "right") if ca < cb else ("right", "left")
        out.append((f"The {a} is to the {rel} of the {b}.", OURS["entailment"]))
        out.append((f"The {a} is to the {anti} of the {b}.", OURS["contradiction"]))
    # absent object -> contradiction; unverifiable attribute -> neutral
    for cand in ("giraffe", "helicopter", "piano", "traffic light", "zebra"):
        if cand not in present:
            out.append((f"There is a {cand} in the image.", OURS["contradiction"]))
            break
    out.append((f"The {lab} was bought last week.", OURS["neutral"]))
    return out


def build_images(args):
    from datasets import load_dataset
    from PIL import Image

    rng = random.Random(args.seed)
    img_dir = os.path.join(args.out, "images")
    os.makedirs(img_dir, exist_ok=True)
    p = Part(args.out, args.part_name)
    ds = load_dataset(args.vqa_repo, split=args.vqa_split, streaming=True)
    ds = ds.shuffle(seed=args.seed, buffer_size=10_000)
    seen_images, seen_answers, n_rows = {}, defaultdict(list), 0

    for ex in ds:
        if n_rows >= args.n_rows or p.n >= args.n_claims:
            break
        n_rows += 1
        try:
            iid = str(ex["id_image"])
            rel = f"images/{iid}.jpg"
            if iid not in seen_images:
                if len(seen_images) >= args.max_images:
                    continue
                im = ex["image"].convert("RGB")
                seen_images[iid] = im.size
                im.resize((320, 240)).save(os.path.join(img_dir, f"{iid}.jpg"), quality=88)
            W, H = seen_images[iid]

            q, a = ex["question"], (ex["multiple_choice_answer"] or "").strip()
            atype = ex.get("answer_type") or ""
            if not a:
                continue
            answers = [x["answer"] if isinstance(x, dict) else x for x in (ex.get("answers") or [])]
            agree = sum(1 for x in answers if str(x).strip().lower() == a.lower())
            stmt, flip_ok = q_to_statement(q, a, atype)
            lead = rng.choice(LEAD_INS)

            if agree and agree < 6:                      # annotators disagree -> not determinable from the image
                p.add(f"{lead} {IMG}", stmt, OURS["neutral"], "vqa_disagree", rel)
            elif atype == "yes/no" and flip_ok:
                p.add(f"{lead} {IMG}", stmt, OURS["entailment"] if a == "yes" else OURS["contradiction"], "vqa_yesno", rel)
            else:
                p.add(f"{lead} {IMG}", stmt, OURS["entailment"], "vqa_answer", rel)
                pool = seen_answers[atype]
                if pool and rng.random() < args.answer_neg_prob:
                    wrong = rng.choice(pool)
                    if wrong.lower() != a.lower():
                        bad, _ = q_to_statement(q, wrong, atype)
                        p.add(f"{lead} {IMG}", bad, OURS["contradiction"], "vqa_answer_neg", rel)
                if len(pool) < 5000:
                    pool.append(a)

            dets = ex.get("DETA_detections_deta_swin_large_o365_coco_classes") or []
            claims = spatial_claims(dets, W, H, rng)
            rng.shuffle(claims)
            for hyp, y in claims[: args.spatial_per_image]:
                p.add(f"{LEAD_PIX} {IMG}", hyp, y, "vqa_spatial", rel)
        except Exception as e:  # noqa: BLE001
            print(f"[vqa] row skipped: {type(e).__name__}: {str(e)[:80]}", flush=True)
    p.close()
    print(f"[vqa] read {n_rows} rows, {len(seen_images)} images saved", flush=True)
    sys.stdout.flush()
    os._exit(0)  # datasets 5.0 segfaults tearing down the parquet stream generator


# --------------------------------------------------------------------------------------- agentic
def _tools_str(tools, limit=1200):
    s = tools if isinstance(tools, str) else json.dumps(tools, ensure_ascii=False)
    return s[:limit]


def build_agentic(args):
    from datasets import load_dataset
    rng = random.Random(args.seed)
    p = Part(args.out, "agentic")

    # ---- xlam function calling (gated upstream; download it yourself to data/xlam.jsonl)
    xlam_path = os.path.join(args.data_dir, "xlam.jsonl")
    if os.path.exists(xlam_path):
        rows = [json.loads(l) for l in open(xlam_path)]
        alt_names = []
        for r in rows:
            try:
                calls = json.loads(r["answers"]) if isinstance(r["answers"], str) else r["answers"]
            except Exception:  # noqa: BLE001
                continue
            if not calls:
                continue
            for c in calls[:2]:
                alt_names.append(c.get("name", ""))
        for r in rows:
            try:
                calls = json.loads(r["answers"]) if isinstance(r["answers"], str) else r["answers"]
            except Exception:  # noqa: BLE001
                continue
            if not calls:
                continue
            call = calls[0]
            name, argd = call.get("name", ""), call.get("arguments", {}) or {}
            if not name:
                continue
            prem = f"User request: {r['query'].strip()}\nAvailable tools: {_tools_str(r.get('tools'))}"
            fmt = lambda n, d: f"The correct call is {n}(" + ", ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in d.items()) + ")."
            p.add(prem, fmt(name, argd), OURS["entailment"], "xlam", "")
            bad = rng.choice(alt_names)
            if bad and bad != name:
                p.add(prem, fmt(bad, argd), OURS["contradiction"], "xlam_neg_name", "")
            if argd:
                k = rng.choice(list(argd))
                mut = dict(argd)
                v = mut[k]
                mut[k] = (v + 7) if isinstance(v, (int, float)) and not isinstance(v, bool) else (str(v) + "_x")
                p.add(prem, fmt(name, mut), OURS["contradiction"], "xlam_neg_arg", "")
                drop = {kk: vv for kk, vv in argd.items() if kk != k}
                p.add(prem, f"The value of `{k}` cannot be determined from the user request.",
                      OURS["neutral"] if len(argd) > 1 else OURS["contradiction"], "xlam_underspec", "")
                if drop:
                    p.add(prem, fmt(name, drop), OURS["contradiction"], "xlam_neg_missing", "")
    else:
        print(f"[agentic] {xlam_path} missing -> xlam skipped", flush=True)

    # ---- trajectory datasets: premise = task + last observation, hypothesis = next action
    def from_conversations(repo, cfgs, src, cap):
        n = 0
        for cfg in cfgs:
            try:
                ds = load_dataset(repo, split=cfg) if cfg else load_dataset(repo, split="train")
            except Exception as e:  # noqa: BLE001
                print(f"[agentic] {repo}:{cfg} skipped: {type(e).__name__}", flush=True)
                continue
            all_actions = []
            convs = []
            for ex in ds:
                turns = ex.get("conversations") or []
                msgs = [(t.get("from") or t.get("role") or "", (t.get("value") or t.get("content") or "").strip()) for t in turns]
                acts = [m for who, m in msgs if who in ("gpt", "assistant") and m]
                if len(msgs) >= 3 and acts:
                    convs.append(msgs)
                    all_actions.extend(acts)
            for msgs in convs:
                if n >= cap:
                    break
                idxs = [i for i, (who, _) in enumerate(msgs) if who in ("gpt", "assistant") and i > 0]
                for i in rng.sample(idxs, min(2, len(idxs))):
                    if n >= cap:
                        break
                    ctx = "\n".join(m for _, m in msgs[max(0, i - 2):i])[-1500:]
                    act = msgs[i][1][:400]
                    p.add(f"Agent task and last observations:\n{ctx}", f"The next action is: {act}", OURS["entailment"], src, "")
                    other = rng.choice(all_actions)[:400]
                    if other != act:
                        p.add(f"Agent task and last observations:\n{ctx}", f"The next action is: {other}", OURS["contradiction"], f"{src}_neg", "")
                    n += 2
        print(f"[agentic] {src}: {n}", flush=True)

    from_conversations("AgentGym/AgentTraj-L", [None], "agenttraj", args.n_traj)
    from_conversations("THUDM/AgentInstruct", ["os", "db", "alfworld", "webshop", "kg", "mind2web"], "agentinstruct", 8000)

    # ---- When2Call: call a tool / ask a clarifying question / answer directly
    try:
        w2c = load_dataset("nvidia/When2Call", split="mcq")
        for ex in w2c:
            q = (ex.get("question") or "").strip()
            if not q:
                continue
            prem = f"{q[:2500]}\nAvailable tools: {_tools_str(ex.get('tools'))}"
            correct = str(ex.get("correct_answer", "")).strip()
            answers = ex.get("answers")
            opts = answers if isinstance(answers, list) else [answers]
            for o in opts:
                o = str(o).strip()
                if not o:
                    continue
                y = OURS["entailment"] if o == correct else OURS["contradiction"]
                p.add(prem, f"The assistant should: {o[:300]}", y, "when2call", "")
            if ex.get("held_out_param"):
                p.add(prem, f"The value of `{ex['held_out_param']}` is stated in the user request.",
                      OURS["contradiction"], "when2call_param", "")
    except Exception as e:  # noqa: BLE001
        print(f"[agentic] When2Call skipped: {type(e).__name__}: {str(e)[:80]}", flush=True)

    # ---- synthetic state predicates (the Minecraft-scaffold format), fully generated here
    if not args.no_synth_state:
        items = ["oak log", "oak planks", "stick", "crafting table", "cobblestone", "raw iron", "iron ingot",
                 "furnace", "coal", "wooden pickaxe", "stone pickaxe", "iron pickaxe", "bucket", "torch", "apple"]
        places = ["a crafting table", "a furnace", "a chest", "water", "lava"]
        for _ in range(args.n_synth):
            inv = {it: rng.randint(1, 40) for it in rng.sample(items, rng.randint(2, 7))}
            near = rng.sample(places, rng.randint(0, 2))
            health = rng.randint(1, 20)
            lines = [f"{k}: {v}" for k, v in inv.items()] or ["(empty)"]
            prem = ("Agent state.\nInventory:\n" + "\n".join(lines)
                    + f"\nNearby: {', '.join(near) if near else 'nothing'}\nHealth: {health}/20")
            kind = rng.randrange(4)
            if kind == 0:
                it = rng.choice(items)
                have, thr = inv.get(it, 0), rng.randint(1, 20)
                p.add(prem, f"The number of {it} in the inventory is {thr} or more.",
                      OURS["entailment"] if have >= thr else OURS["contradiction"], "synth_count", "")
            elif kind == 1:
                it = rng.choice(items)
                p.add(prem, f"There are no {it} in the inventory.",
                      OURS["entailment"] if inv.get(it, 0) == 0 else OURS["contradiction"], "synth_absent", "")
            elif kind == 2:
                pl = rng.choice(places)
                p.add(prem, f"The agent is near {pl}.",
                      OURS["entailment"] if pl in near else OURS["contradiction"], "synth_near", "")
            else:
                p.add(prem, rng.choice([f"The agent has been playing for {rng.randint(2, 90)} minutes.",
                                        "The agent intends to build a shelter next.",
                                        f"The {rng.choice(items)} was crafted by another player."]),
                      OURS["neutral"], "synth_unknown", "")
    p.close()


# --------------------------------------------------------------------------------------- merge
def build_final(args):
    from datasets import Dataset
    rng = random.Random(args.seed)
    rows, by_source, by_label = [], Counter(), Counter()
    drop = set(x for x in args.drop.split(",") if x)
    cap_prefixes = tuple(x for x in args.cap_prefixes.split(",") if x) or ("\0",)
    part_dir = os.path.join(args.out, "parts")
    for fn in sorted(os.listdir(part_dir)):
        if not fn.endswith(".jsonl"):
            continue
        with open(os.path.join(part_dir, fn)) as f:
            for line in f:
                r = json.loads(line)
                if r["source"] in drop:
                    continue
                if (args.cap_per_source and r["source"].startswith(cap_prefixes)
                        and by_source[r["source"]] >= args.cap_per_source):
                    continue
                rows.append(r)
                by_source[r["source"]] += 1
                by_label[r["label"]] += 1
    print(f"[build] {len(rows)} rows from {len(by_source)} sources; labels {dict(by_label)}", flush=True)

    # class balance: downsample the dominant class to at most 1.15x the smallest
    target = int(min(by_label.values()) * 1.15)
    keep, per = [], Counter()
    rng.shuffle(rows)
    for r in rows:
        if per[r["label"]] < target:
            keep.append(r)
            per[r["label"]] += 1
    rows = keep
    print(f"[build] after balancing: {len(rows)}; labels {dict(per)}", flush=True)

    # leakage check against the MNLI validation splits that eval.py reports
    from datasets import load_dataset
    banned_pairs = set()
    for split in ("validation_matched", "validation_mismatched"):
        for ex in load_dataset("nyu-mll/multi_nli", split=split):
            banned_pairs.add((ex["premise"].strip().lower()[:200], ex["hypothesis"].strip().lower()[:200]))
    before = len(rows)
    rows = [r for r in rows if (r["premise"].strip().lower()[:200], r["hypothesis"].strip().lower()[:200]) not in banned_pairs]
    print(f"[build] leakage filter dropped {before - len(rows)} rows overlapping MNLI val", flush=True)

    rng.shuffle(rows)
    n_val = args.n_val
    val, train = rows[:n_val], rows[n_val:]
    ds = {"train": Dataset.from_list(train), "val": Dataset.from_list(val)}
    from datasets import DatasetDict
    DatasetDict(ds).save_to_disk(os.path.join(args.out, "mix"))

    comp = {"n_train": len(train), "n_val": len(val), "by_source": dict(by_source),
            "by_label_final": dict(per), "images": sum(1 for r in rows if r["image"])}
    json.dump(comp, open(os.path.join(args.out, "composition.json"), "w"), indent=2)
    print(json.dumps(comp, indent=2)[:2000])
    print("\n| source | rows |\n|---|---|")
    for s, n in by_source.most_common():
        print(f"| {s} | {n} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["text", "images", "agentic", "build"])
    ap.add_argument("--out", default="/mnt/nli_mix")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--seed", type=int, default=0)
    # text
    ap.add_argument("--n-snli", type=int, default=120_000)
    ap.add_argument("--n-mnli", type=int, default=120_000)
    ap.add_argument("--n-fever", type=int, default=150_000)
    ap.add_argument("--n-qnli", type=int, default=60_000)
    ap.add_argument("--n-haystack", type=int, default=80_000)
    # images
    ap.add_argument("--n-rows", type=int, default=200_000)
    ap.add_argument("--n-claims", type=int, default=220_000)
    ap.add_argument("--max-images", type=int, default=60_000)
    ap.add_argument("--spatial-per-image", type=int, default=1)
    ap.add_argument("--answer-neg-prob", type=float, default=0.5)
    ap.add_argument("--vqa-repo", default="Multimodal-Fatima/VQAv2_train")
    ap.add_argument("--vqa-split", default="train")
    ap.add_argument("--part-name", default="vqa")
    # agentic
    ap.add_argument("--n-traj", type=int, default=40_000)
    ap.add_argument("--n-synth", type=int, default=50_000)
    ap.add_argument("--no-synth-state", action="store_true")
    # build
    ap.add_argument("--n-val", type=int, default=4000)
    ap.add_argument("--drop", default="xlam_underspec", help="comma-separated sources to exclude (xlam_underspec is mislabeled)")
    ap.add_argument("--cap-per-source", type=int, default=40000)
    ap.add_argument("--cap-prefixes", default="xlam", help="the per-source cap applies only to sources with these prefixes")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    {"text": build_text, "images": build_images, "agentic": build_agentic, "build": build_final}[args.cmd](args)


if __name__ == "__main__":
    main()
