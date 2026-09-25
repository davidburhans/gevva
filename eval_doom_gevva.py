#!/usr/bin/env python3
"""eval_doom_gevva.py - Doom "Defend the Center" Benchmark for Gevva.

Direct port and enhancement of AlexWortega/openjev's code/doom.py and doom_vision.py.
The Gevva Cross-Encoder controls the player in real time (non-autoregressive System 1).

Modes:
- text: Game state (visible enemies, offsets, health, ammo) rendered as natural language.
- vision: Raw RGB game frames fed directly to Gevva's vision tower.

Actions:
0: turn left
1: turn right
2: attack (shoot pistol)

Usage:
  # Text-based policy:
  uv run python eval_doom_gevva.py --mode text --episodes 3

  # Vision-based policy (pure pixels):
  uv run python eval_doom_gevva.py --mode vision --episodes 3
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
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import vizdoom as vzd
from gevva import load

ACTIONS = ["turn left", "turn right", "attack"]
BUTTONS = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
FRAME_SKIP = 4

ENEMY_NAMES = {
    "Zombieman": "zombie soldier",
    "ShotgunGuy": "shotgun guard",
    "Imp": "imp",
    "Demon": "pinky demon",
    "MarineChainsaw": "chainsaw marine",
    "MarineChainsawVzd": "chainsaw marine",
    "ChaingunGuy": "chaingunner",
    "HellKnight": "hell knight",
    "Cacodemon": "cacodemon",
    "LostSoul": "lost soul",
    "Revenant": "revenant",
    "BaronOfHell": "baron of hell",
}

VISION_HYPS = [
    "The marine should turn left.",
    "The marine should turn right.",
    "The marine should fire now.",
]

TEXT_HYPS = [
    "The correct action is: turn left",
    "The correct action is: turn right",
    "The correct action is: attack",
]


def make_game(res=vzd.ScreenResolution.RES_640X480) -> vzd.DoomGame:
    game = vzd.DoomGame()
    game.load_config(os.path.join(vzd.scenarios_path, "defend_the_center.cfg"))
    game.set_screen_resolution(res)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_labels_buffer_enabled(True)
    game.set_window_visible(False)
    game.set_mode(vzd.Mode.PLAYER)
    game.set_episode_timeout(2100)
    game.init()
    return game


def parse_state(game: vzd.DoomGame) -> Optional[Dict[str, Any]]:
    st = game.get_state()
    if st is None:
        return None
    H, W = st.screen_buffer.shape[0], st.screen_buffer.shape[1]
    enemies = []
    for lab in st.labels:
        if lab.object_name not in ENEMY_NAMES or lab.width == 0:
            continue
        cx = (lab.x + lab.width / 2) / W - 0.5
        enemies.append({
            "name": ENEMY_NAMES.get(lab.object_name, lab.object_name.lower()),
            "off": float(cx),
            "size": float(lab.height / H),
        })
    enemies.sort(key=lambda e: abs(e["off"]))
    ammo = game.get_game_variable(vzd.GameVariable.AMMO2)
    health = game.get_game_variable(vzd.GameVariable.HEALTH)
    kills = game.get_game_variable(vzd.GameVariable.KILLCOUNT)
    return {
        "enemies": enemies,
        "ammo": int(ammo),
        "health": int(health),
        "kills": int(kills),
        "frame": st.screen_buffer,
    }


def oracle_policy(s: Dict[str, Any], tol: float = 0.035) -> int:
    if not s["enemies"]:
        return 0  # Scan left
    e = s["enemies"][0]
    if abs(e["off"]) < tol:
        return 2 if s["ammo"] > 0 else (0 if e["off"] < 0 else 1)
    return 0 if e["off"] < 0 else 1


def render_text_state(s: Dict[str, Any]) -> str:
    if s["enemies"]:
        parts = []
        for e in s["enemies"][:4]:
            side = "right of" if e["off"] > 0.015 else ("left of" if e["off"] < -0.015 else "exactly on")
            dist = "very close" if e["size"] > 0.45 else ("close" if e["size"] > 0.25 else "far")
            parts.append(f"a {e['name']} {abs(e['off']):.2f} to the {side} the crosshair ({dist})".replace("to the exactly on", "exactly on"))
        seen = "Visible enemies: " + "; ".join(parts) + "."
    else:
        seen = "No enemies are visible right now."
    return (
        f"Doom, Defend the Center. You stand in the middle of a circular arena with a pistol ({s['ammo']} bullets, health {s['health']}). "
        f"Enemies walk toward you from all sides and attack when close; you can only turn left, turn right, or fire. "
        f"Screen offsets are fractions of the screen width (0 = crosshair, 0.5 = screen edge); one turn step moves the view by about 0.05. "
        f"{seen} A shot hits only if an enemy is within about 0.03 of the crosshair."
    )


def play_episode(
    game: vzd.DoomGame,
    policy_fn,
    seed: int,
) -> Dict[str, Any]:
    game.set_seed(seed)
    game.new_episode()
    steps = 0
    latencies = []
    oracle_agreements = 0

    while not game.is_episode_finished():
        s = parse_state(game)
        if s is None:
            break

        t0 = time.perf_counter()
        action = policy_fn(s)
        lat = time.perf_counter() - t0
        latencies.append(lat)

        oracle_act = oracle_policy(s)
        if action == oracle_act:
            oracle_agreements += 1

        game.make_action(BUTTONS[action], FRAME_SKIP)
        steps += 1

    kills = int(game.get_game_variable(vzd.GameVariable.KILLCOUNT))
    reward = float(game.get_total_reward())
    avg_lat = float(np.mean(latencies) * 1000) if latencies else 0.0
    agreement_rate = float(oracle_agreements / max(1, steps))

    return {
        "kills": kills,
        "reward": reward,
        "steps": steps,
        "avg_latency_ms": avg_lat,
        "oracle_agreement": agreement_rate,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Gevva Cross-Encoder on ViZDoom Defend the Center")
    parser.add_argument("--model-path", default="ckpt/gevva-e2b-multimodal/best")
    parser.add_argument("--mode", default="text", choices=["text", "vision"])
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="results/doom_gevva_e2b.json")
    args = parser.parse_args()

    print(f"Loading Gevva Cross-Encoder: {args.model_path}...")
    engine = load(args.model_path)
    game = make_game()

    def gevva_text_policy(s: Dict[str, Any]) -> int:
        premise = render_text_state(s)
        pairs = [(premise, h) for h in TEXT_HYPS]
        probs = engine.predict(pairs)  # (3, 3) -> [contra, entail, neut]
        p_ent = probs[:, 1]
        return int(np.argmax(p_ent))

    def gevva_vision_policy(s: Dict[str, Any]) -> int:
        frame_rgb = s["frame"]  # (H, W, 3)
        pil_img = Image.fromarray(frame_rgb)
        premise = "Doom, Defend the Center, seen from the player's eyes. You can only turn left, turn right, or fire the pistol; a shot hits only if an enemy is on the crosshair."
        pairs = [(premise, h) for h in VISION_HYPS]
        probs = engine.predict(pairs, images=[pil_img, pil_img, pil_img])
        p_ent = probs[:, 1]
        return int(np.argmax(p_ent))

    def random_policy(s: Dict[str, Any]) -> int:
        return random.randint(0, 2)

    policies = {
        "oracle": oracle_policy,
        "random": random_policy,
        "gevva": gevva_vision_policy if args.mode == "vision" else gevva_text_policy,
    }

    all_results = {}
    print(f"\nEvaluating Doom Benchmark (Mode: {args.mode.upper()}, Episodes: {args.episodes})...")

    for pol_name, pol_fn in policies.items():
        ep_stats = []
        for ep in range(args.episodes):
            ep_res = play_episode(game, pol_fn, seed=args.seed + ep * 100)
            ep_stats.append(ep_res)
            print(f"[{pol_name:7s}] Episode {ep+1}/{args.episodes}: Kills = {ep_res['kills']} | Steps = {ep_res['steps']} | Latency = {ep_res['avg_latency_ms']:.1f}ms | Oracle Match = {100*ep_res['oracle_agreement']:.1f}%")

        avg_kills = float(np.mean([r["kills"] for r in ep_stats]))
        avg_steps = float(np.mean([r["steps"] for r in ep_stats]))
        avg_lat = float(np.mean([r["avg_latency_ms"] for r in ep_stats]))
        avg_agree = float(np.mean([r["oracle_agreement"] for r in ep_stats]))

        all_results[pol_name] = {
            "avg_kills": avg_kills,
            "avg_steps": avg_steps,
            "avg_latency_ms": avg_lat,
            "avg_oracle_agreement": avg_agree,
            "episodes": ep_stats,
        }
        print(f"--> {pol_name.upper()} Summary: Avg Kills = {avg_kills:.1f} | Avg Steps = {avg_steps:.1f} | Latency = {avg_lat:.1f}ms | Oracle Agreement = {100*avg_agree:.1f}%\n")

    game.close()

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"Doom benchmark results saved to {out_p}")


if __name__ == "__main__":
    main()
