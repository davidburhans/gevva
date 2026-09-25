#!/usr/bin/env python3
"""eval_minecraft_gevva.py - Minecraft Long-Horizon Crafting Benchmark for Gevva.

Direct port of AlexWortega/openjev's code/minecraft.py adapted for Gevva Cross-Encoder.
The cross-encoder never outputs text: every decision is a non-autoregressive System 1
entailment check over a natural language rendering of the Minecraft game state.

Planners evaluated:
- random: uniform skill selection (baseline)
- oracle: ground-truth state predicates
- chain:  backward chaining over the Minecraft tech tree (log -> planks -> ... -> iron pickaxe)
- flat:   direct argmax over all candidate skill hypotheses

Usage:
  uv run python eval_minecraft_gevva.py \
      --model-path ckpt/gevva-e2b-multimodal/best \
      --planners random oracle chain \
      --episodes 5 \
      --out results/minecraft_gevva_e2b.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from gevva import load

# -----------------------------------------------------------------------------
# Minecraft Tech Tree & Skill Definitions (Parity with OpenJEV)
# -----------------------------------------------------------------------------
TREE = {
    "log":            {"skill": ("collect", "log"), "batch": 4, "reqs": [("visible", "log")]},
    "planks":         {"skill": ("craft", "planks"), "reqs": [("have", "log", 1)]},
    "stick":          {"skill": ("craft", "stick"), "reqs": [("have", "planks", 2)]},
    "crafting_table": {"skill": ("craft", "crafting_table"), "reqs": [("have", "planks", 4)]},
    "wooden_pickaxe": {"skill": ("craft", "wooden_pickaxe"), "reqs": [("have", "stick", 2), ("have", "planks", 3), ("near", "crafting_table")]},
    "cobblestone":    {"skill": ("collect", "stone"), "reqs": [("have", "wooden_pickaxe", 1), ("visible", "stone")]},
    "stone_pickaxe":  {"skill": ("craft", "stone_pickaxe"), "reqs": [("have", "stick", 2), ("have", "cobblestone", 3), ("near", "crafting_table")]},
    "furnace":        {"skill": ("craft", "furnace"), "reqs": [("have", "cobblestone", 8), ("near", "crafting_table")]},
    "raw_iron":       {"skill": ("collect", "iron_ore"), "reqs": [("have", "stone_pickaxe", 1), ("visible", "iron_ore")]},
    "iron_ingot":     {"skill": ("smelt", "raw_iron"), "reqs": [("have", "raw_iron", 3), ("have", "planks", 2), ("near", "furnace")]},
    "iron_pickaxe":   {"skill": ("craft", "iron_pickaxe"), "reqs": [("have", "iron_ingot", 3), ("have", "stick", 2), ("near", "crafting_table")]},
}
RECIPE_OUT = {"planks": 4, "stick": 4}
CONSUMES = {
    "planks": {"log": 1},
    "stick": {"planks": 2},
    "crafting_table": {"planks": 4},
    "wooden_pickaxe": {"stick": 2, "planks": 3},
    "stone_pickaxe": {"stick": 2, "cobblestone": 3},
    "furnace": {"cobblestone": 8},
    "iron_pickaxe": {"iron_ingot": 3, "stick": 2},
}
MILESTONES = list(TREE)
NAME = {
    "log": ("log", "logs"),
    "planks": ("plank", "planks"),
    "stick": ("stick", "sticks"),
    "crafting_table": ("crafting table", "crafting tables"),
    "wooden_pickaxe": ("wooden pickaxe", "wooden pickaxes"),
    "cobblestone": ("cobblestone block", "cobblestone blocks"),
    "stone_pickaxe": ("stone pickaxe", "stone pickaxes"),
    "furnace": ("furnace", "furnaces"),
    "raw_iron": ("raw iron chunk", "raw iron chunks"),
    "iron_ingot": ("iron ingot", "iron ingots"),
    "iron_pickaxe": ("iron pickaxe", "iron pickaxes"),
    "stone": ("stone", "stone"),
    "iron_ore": ("iron ore", "iron ore"),
}
nm = lambda item, n=1: NAME.get(item, (item.replace("_", " "),) * 2)[0 if n == 1 else 1]
VERB = {"collect": "mine", "craft": "craft", "smelt": "smelt", "place": "place"}

SKILLS = [TREE[i]["skill"] + (TREE[i].get("batch", 1),) for i in TREE] + [
    ("place", "crafting_table", 1),
    ("place", "furnace", 1),
    ("explore", None, 1),
]


def skill_text(sk):
    if sk[0] == "explore":
        return "walk away to explore a new area"
    if sk[0] == "collect":
        return f"mine {nm(sk[1])} blocks to get {nm(next(i for i in TREE if TREE[i]['skill'] == sk[:2]), 2)}"
    if sk[0] == "smelt":
        return f"smelt {nm(sk[1])} into iron ingots in the furnace"
    return f"{VERB[sk[0]]} {'' if sk[1] == 'planks' else ('an ' if nm(sk[1])[0] in 'aeiou' else 'a ')}{nm(sk[1], 2 if sk[1] == 'planks' else 1)}"


# -----------------------------------------------------------------------------
# Symbolic Environment (Deterministic Game Physics)
# -----------------------------------------------------------------------------
class SimEnv:
    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        self.inv = {}
        self.near = {"crafting_table": False, "furnace": False}
        self.visible = {
            "log": self.rng.random() < 0.8,
            "stone": self.rng.random() < 0.8,
            "iron_ore": self.rng.random() < 0.4,
        }
        return self.state()

    def state(self):
        return {
            "inventory": {k: v for k, v in self.inv.items() if v > 0},
            "near": dict(self.near),
            "visible": dict(self.visible),
            "pos": [0, 64, 0],
        }

    def act(self, sk):
        kind, arg, n = sk
        have = lambda i, k=1: self.inv.get(i, 0) >= k
        if kind == "explore":
            self.near = {k: False for k in self.near}
            self.visible = {
                "log": self.rng.random() < 0.8,
                "stone": self.rng.random() < 0.8,
                "iron_ore": self.rng.random() < 0.5,
            }
            return {"ok": True, "msg": "walked 40 blocks"}
        if kind == "place":
            if not have(arg):
                return {"ok": False, "msg": f"no {arg} in the inventory"}
            self.inv[arg] -= 1
            self.near[arg] = True
            return {"ok": True, "msg": f"placed {arg}"}
        item = next(i for i in TREE if TREE[i]["skill"] == (kind, arg))
        for r in TREE[item]["reqs"]:
            ok = have(r[1], r[2]) if r[0] == "have" else (self.near[r[1]] if r[0] == "near" else self.visible[r[1]])
            if not ok:
                return {"ok": False, "msg": f"cannot {VERB[kind]} {nm(arg)}: requirement not met ({r[0]} {nm(r[1])})"}
        if kind == "collect":
            self.inv[item] = self.inv.get(item, 0) + n
            return {"ok": True, "msg": f"collected {n} {item}"}
        if kind == "smelt":
            k = self.inv.get("raw_iron", 0)
            self.inv["raw_iron"] = 0
            self.inv["planks"] = max(0, self.inv.get("planks", 0) - 2)
            self.inv["iron_ingot"] = self.inv.get("iron_ingot", 0) + k
            return {"ok": True, "msg": f"smelted {k} iron_ingot"}
        for i, k in CONSUMES[item].items():
            self.inv[i] -= k
        self.inv[item] = self.inv.get(item, 0) + RECIPE_OUT.get(item, 1)
        return {"ok": True, "msg": f"crafted {item}"}


# -----------------------------------------------------------------------------
# Text Grounding & Natural Language State Rendering
# -----------------------------------------------------------------------------
def render_state(s, goal=None, last=None, recipes=False):
    inv = ", ".join(f"{s['inventory'].get(i, 0)} {nm(i, s['inventory'].get(i, 0))}" for i in TREE)
    near = " ".join(f"A placed {nm(k)} {'stands' if v else 'does not stand'} nearby." for k, v in s["near"].items())
    vis = " ".join(f"{nm(k).capitalize()} blocks {'are' if v else 'are not'} visible nearby." for k, v in s["visible"].items())
    t = "Minecraft survival. "
    if goal:
        t += f"The goal is to craft {'an' if nm(goal)[0] in 'aeiou' else 'a'} {nm(goal)}. "
    if recipes:
        t += (
            "Recipes: planks = 1 log; sticks = 2 planks; crafting table = 4 planks; wooden pickaxe = 3 planks + 2 sticks; "
            "stone pickaxe = 3 cobblestone + 2 sticks; furnace = 8 cobblestone; iron pickaxe = 3 iron ingots + 2 sticks. "
            "Pickaxes and the furnace are crafted at a placed crafting table. Stone needs a wooden pickaxe, "
            "iron ore needs a stone pickaxe, iron ingots are smelted from raw iron in a placed furnace with planks as fuel. "
        )
    t += f"Inventory counts: {inv}. {near} {vis}"
    if last:
        t += f" Last action: {last}"
    return t


def req_hyps(r, negated=False):
    if r[0] == "have":
        i, n = r[1], r[2]
        return [
            f"The player has {'fewer than' if negated else 'at least'} {n} {nm(i, n)}.",
            f"The {nm(i)} count in the inventory is {'smaller than' if negated else 'greater than or equal to'} {n}.",
            f"The number of {nm(i, 2)} in the inventory is {'less than ' + str(n) if negated else str(n) + ' or more'}.",
        ]
    if r[0] == "near":
        return [f"The answer to whether a {nm(r[1])} is placed nearby is {'no' if negated else 'yes'}."]
    return [f"{'No' if negated else 'A'} {nm(r[1])} block is visible nearby."]


req_hyp = lambda r: req_hyps(r)[0]
req_truth = lambda r, s: s["inventory"].get(r[1], 0) >= r[2] if r[0] == "have" else (s["near"][r[1]] if r[0] == "near" else s["visible"][r[1]])


# -----------------------------------------------------------------------------
# Gevva Cross-Encoder Scorer Wrapper
# -----------------------------------------------------------------------------
class GevvaMinecraftScorer:
    def __init__(self, model_path: str):
        print(f"Loading Gevva Cross-Encoder for Minecraft Benchmark: {model_path}...")
        self.engine = load(model_path)
        self.calls = 0
        self.total_time = 0.0

    def probs(self, premise: str, hyps: List[str]) -> np.ndarray:
        t0 = time.perf_counter()
        pairs = [(premise, h) for h in hyps]
        probs = self.engine.predict(pairs)  # (len(hyps), 3) -> [contra, entail, neut]
        self.calls += 1
        self.total_time += time.perf_counter() - t0
        return probs


# -----------------------------------------------------------------------------
# Backward-Chaining Planner (System 1 Tech Tree Search)
# -----------------------------------------------------------------------------
class Chain:
    def __init__(self, goal, judge):
        self.goal, self.judge, self.fails = goal, judge, {}

    def decide(self, s, trace, failed=()):
        self.fails = {k: v for k, v in self.fails.items() if k in failed}
        for sk in failed:
            self.fails.setdefault(sk, 0)
        trace.append({"node": "goal reached?"})
        if self.judge(s, [("have", self.goal, 1)], trace)[0] > 0:
            return None
        item, n = self.goal, 1
        for _ in range(12):
            trace.append({"node": item})
            reqs = TREE[item]["reqs"]
            m = list(self.judge(s, reqs, trace))
            sk = TREE[item]["skill"]
            sk = sk + (max(n, TREE[item].get("batch", 1)) if sk[0] == "collect" else 1,)
            if all(x > 0 for x in m) and sk in failed:
                self.fails[sk] += 1
                for k in np.argsort(m)[: self.fails[sk]]:
                    m[k] = -1.0
                    trace.append({"node": f"action failed -> doubting: {req_hyp(reqs[k])}"})
            miss = next((r for r, x in zip(reqs, m) if x <= 0), None)
            if miss is None:
                return sk
            if miss[0] == "visible":
                return ("explore", None, 1)
            if miss[0] == "near":
                trace.append({"node": "placed " + miss[1]})
                if self.judge(s, [("have", miss[1], 1)], trace)[0] > 0:
                    return ("place", miss[1], 1)
                item, n = miss[1], 1
            else:
                item, n = miss[1], miss[2]
        return ("explore", None, 1)


def make_planner(name, scorer, goal, rng, rule="pair"):
    if name == "random":
        return lambda s, last, failed, trace: rng.choice(SKILLS)
    if name == "oracle":
        ch = Chain(goal, lambda s, reqs, trace: [1.0 if req_truth(r, s) else -1.0 for r in reqs])
        return lambda s, last, failed, trace: ch.decide(s, trace, failed)
    if name == "chain":
        def judge(s, reqs, trace):
            hyps, owner = [], []
            for k, r in enumerate(reqs):
                for sign, neg in ((1, False), (-1, True)):
                    for h in req_hyps(r, neg):
                        hyps.append(h)
                        owner.append((k, sign / len(req_hyps(r))))
            p = scorer.probs(render_state(s), hyps)[:, 1]  # P(entailment)
            pos, neg = np.zeros(len(reqs)), np.zeros(len(reqs))
            for (k, w), pi in zip(owner, p):
                (pos if w > 0 else neg)[k] += abs(w) * pi
            margin = [float(a - b) if rule == "pair" else float(a - 0.5) for a, b in zip(pos, neg)]
            for r, a, b, mg in zip(reqs, pos, neg, margin):
                trace.append({
                    "hyp": req_hyp(r),
                    "p_ent": float(a),
                    "p_ent_neg": float(b),
                    "judged": mg > 0,
                    "truth": bool(req_truth(r, s)),
                })
            return margin

        ch = Chain(goal, judge)
        return lambda s, last, failed, trace: ch.decide(s, trace, failed)
    raise ValueError(name)


def run_episode(env, planner, goal, max_steps, verbose=False):
    s = env.reset()
    reached, steps, failed, last, t0 = set(i for i in MILESTONES if s["inventory"].get(i, 0) > 0), [], set(), None, time.time()
    for t in range(max_steps):
        trace, ts = [], time.time()
        sk = planner(s, last, failed, trace)
        ts_act = time.time()
        if sk is None or s["inventory"].get(goal, 0) > 0:
            break
        r = env.act(sk)
        s2 = env.state()
        failed = failed | {sk} if (not r["ok"] and s2["inventory"] == s["inventory"] and s2["near"] == s["near"]) else set()
        last = f"{skill_text(sk)} - {'succeeded' if r['ok'] else 'failed'}: {r['msg']}."
        steps.append({
            "t": t,
            "state": s,
            "skill": list(sk),
            "ok": r["ok"],
            "msg": r["msg"],
            "trace": trace,
            "ts": ts,
            "ts_act": ts_act,
            "ts_end": time.time(),
            "state_after": s2,
        })
        if verbose:
            print(f"  [{t:3d}] {skill_text(sk):55s} {'ok  ' if r['ok'] else 'FAIL'} {r['msg']}   inv={s2['inventory']}", flush=True)
        s = s2
        reached |= {i for i in MILESTONES if s["inventory"].get(i, 0) > 0} | {k for k, v in s["near"].items() if v}
    return {
        "success": s["inventory"].get(goal, 0) > 0,
        "milestones": [m for m in MILESTONES if m in reached],
        "n_steps": len(steps),
        "steps": steps,
        "final_inventory": s["inventory"],
        "seconds": time.time() - t0,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Gevva on Minecraft Tech Tree Crafting")
    parser.add_argument("--model-path", default="ckpt/gevva-e2b-multimodal/best")
    parser.add_argument("--planners", nargs="+", default=["random", "oracle", "chain"])
    parser.add_argument("--goal", default="iron_pickaxe", choices=list(TREE))
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="results/minecraft_gevva_e2b.json")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    scorer = GevvaMinecraftScorer(args.model_path) if "chain" in args.planners else None

    results = {}
    print(f"\nEvaluating Minecraft Crafting Benchmark (Goal: {args.goal}, Episodes: {args.episodes})...")

    for planner_name in args.planners:
        rng = random.Random(args.seed)
        planner = make_planner(planner_name, scorer, args.goal, rng)
        ep_results = []
        for ep in range(args.episodes):
            env = SimEnv(seed=args.seed + ep * 7919)
            res = run_episode(env, planner, args.goal, args.max_steps, verbose=args.verbose)
            ep_results.append(res)
            succ_str = "SUCCESS" if res["success"] else "FAIL"
            print(f"[{planner_name:8s}] Episode {ep+1}/{args.episodes}: {succ_str} in {res['n_steps']} steps ({len(res['milestones'])} milestones: {res['milestones']}) [{res['seconds']:.1f}s]")

        n_succ = sum(1 for r in ep_results if r["success"])
        avg_steps = np.mean([r["n_steps"] for r in ep_results])
        avg_miles = np.mean([len(r["milestones"]) for r in ep_results])
        results[planner_name] = {
            "success_rate": n_succ / args.episodes,
            "avg_steps": float(avg_steps),
            "avg_milestones": float(avg_miles),
            "episodes": ep_results,
        }
        print(f"--> {planner_name.upper()} Summary: Success Rate = {100.0 * n_succ / args.episodes:.1f}% | Avg Steps = {avg_steps:.1f} | Avg Milestones = {avg_miles:.1f}\n")

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_p}")


if __name__ == "__main__":
    main()
