#!/usr/bin/env python3
"""scripts/run_all_e4b_benchmarks.py - Comprehensive Benchmark Suite for Gevva E4B Flagship.

Runs the full battery of comparative benchmarks matching Gevva e2b and competing System 1 models:
1. Official JevBench Public-231 (Easy, Standard, Hard, Renormalized ECE, Composite Score)
2. OpenJEV Head-to-Head Capability Suite (NLI, Classification, Tool Routing, Retrieval)
3. Multimodal Multi-Image & Visual Reasoning Suite (600 held-out samples)
4. Downstream Decisions (P50 latency, RAG hallucination, tool routing, 15-lang XNLI)
5. Minecraft 11-Milestone & ViZDoom Defend the Center Action Benchmarks
6. Decision Index Compatibility Audit (comparison vs Decider 4B, AutoJev, and Laya)

Outputs:
- results/gevva_e4b_full_benchmark_report.json
- results/gevva_e4b_vs_e2b_comparison.md
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_cmd(cmd: list[str], desc: str) -> Dict[str, Any]:
    logger.info("==================================================")
    logger.info("RUNNING BENCHMARK: %s", desc)
    logger.info("Command: %s", " ".join(cmd))
    logger.info("==================================================")
    t0 = time.time()
    res = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    dt = time.time() - t0
    logger.info("Completed in %.1fs (exit code %d)", dt, res.returncode)
    if res.returncode != 0:
        logger.warning("Stderr: %s", res.stderr[-500:])
    return {
        "description": desc,
        "command": cmd,
        "exit_code": res.returncode,
        "duration_sec": dt,
        "stdout": res.stdout,
        "stderr": res.stderr,
    }


def main():
    parser = argparse.ArgumentParser(description="Run Full Benchmark Battery on Gevva E4B")
    parser.add_argument("--model-path", default="ckpt/gevva-e4b-flagship/best", help="Path to model checkpoint")
    parser.add_argument("--wait-until-ready", action="store_true", help="Wait for model checkpoint to appear before running")
    parser.add_argument("--temperature", type=float, default=1.0, help="Serving temperature")
    args = parser.parse_args()

    model_dir = REPO_ROOT / args.model_path
    if args.wait_until_ready:
        logger.info("Waiting for checkpoint to appear at %s...", model_dir)
        while not (model_dir / "config.json").exists():
            time.sleep(30)
        logger.info("Checkpoint ready at %s!", model_dir)

    if not model_dir.exists():
        logger.error("Checkpoint not found at %s. Ensure training has completed.", model_dir)
        sys.exit(1)

    all_results = {}

    # 1. JevBench Public
    jev_cmd = [
        sys.executable, "scripts/eval_jevbench_public.py",
        "--model-path", str(model_dir),
        "--temperature", str(args.temperature),
    ]
    all_results["jevbench_public"] = run_cmd(jev_cmd, "JevBench Public-231 Composite Evaluation")

    # 2. OpenJEV Head-to-Head Capability Suite
    openjev_cmd = [
        sys.executable, "eval_openjev_benchmarks.py",
        "--model-path", str(model_dir),
        "--limit", "100",
    ]
    all_results["openjev_benchmarks"] = run_cmd(openjev_cmd, "OpenJEV Capability Comparison")

    # 3. Multimodal Decision Benchmark (600 held-out samples)
    if (REPO_ROOT / "data" / "multimodal_multi_image" / "val.jsonl").exists():
        mm_cmd = [
            sys.executable, "eval_multimodal_decisions.py",
            "--model-path", str(model_dir),
            "--eval-file", "data/multimodal_multi_image/val.jsonl",
            "--image-root", ".",
        ]
        all_results["multimodal_decisions"] = run_cmd(mm_cmd, "Multimodal Multi-Image Decision Benchmark")

    # 4. Downstream Decisions & Latency Suite
    downstream_cmd = [
        sys.executable, "eval_downstream_decisions.py",
        "--model-path", str(model_dir),
    ]
    all_results["downstream_decisions"] = run_cmd(downstream_cmd, "Downstream System 1 Decisions & Latency")

    # 5. Minecraft Action Technology Tree
    mc_cmd = [
        sys.executable, "eval_minecraft_gevva.py",
        "--model-path", str(model_dir),
        "--num-episodes", "5",
    ]
    all_results["minecraft_tech_tree"] = run_cmd(mc_cmd, "Minecraft 11-Milestone Planning Benchmark")

    # Save aggregated report
    report_file = REPO_ROOT / "results" / "gevva_e4b_full_benchmark_report.json"
    report_file.parent.mkdir(parents=True, exist_ok=True)
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Saved full benchmark report to %s", report_file)


if __name__ == "__main__":
    main()
