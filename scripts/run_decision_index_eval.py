#!/usr/bin/env python3
"""scripts/run_decision_index_eval.py
=====================================
Automated runner for Decision Index 0.2 benchmark suite evaluation on RTX 5090.
Imports the rebuilt suite and runs GevvaEngine across both Gevva e4b and Gevva e2b.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Add decision-index repo to sys.path
DECISION_INDEX_DIR = Path("research/decision_index_upstream").resolve()
sys.path.insert(0, str(DECISION_INDEX_DIR))
sys.path.insert(0, str(Path(".").resolve()))


def run_cmd(cmd: list[str], cwd: Path | None = None) -> int:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{DECISION_INDEX_DIR}:{Path('.').resolve()}"
    print(f"\n[EXEC] {' '.join(cmd)}")
    t0 = time.perf_counter()
    ret = subprocess.run(cmd, cwd=cwd, env=env)
    elapsed = time.perf_counter() - t0
    print(f"[DONE] Exit code {ret.returncode} in {elapsed:.1f}s")
    return ret.returncode


def main():
    parser = argparse.ArgumentParser(description="Run Decision Index evaluation on Gevva models.")
    parser.add_argument("--models", nargs="+", default=["e4b", "e2b"], help="Models to evaluate: e4b, e2b, or custom path")
    parser.add_argument("--work-dir", type=str, default="work", help="Work directory for rebuild")
    parser.add_argument("--suite-dir", type=str, default="suite-0.2", help="Suite directory")
    parser.add_argument("--device", type=str, default="cuda", help="Inference device")
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    suite_dir = Path(args.suite_dir)
    selected_rows = work_dir / "artifacts/benchmark-suite/release-v2-rebuilt/selected-rows.jsonl.gz"
    added_rows = work_dir / "artifacts/benchmark-suite/release-v2-rebuilt/added-rows.jsonl.gz"

    # Step 1: Ensure suite is imported
    if not (suite_dir / "selected-rows.jsonl.gz").exists():
        if not selected_rows.exists():
            print(f"Error: {selected_rows} not found. Rebuild suite first with 'python -m decision_index suite rebuild --work {work_dir}'")
            sys.exit(1)

        print(f"Importing suite from {work_dir} into {suite_dir}...")
        code = run_cmd([
            sys.executable, "-m", "decision_index", "suite", "import",
            "--rows", str(selected_rows),
            "--added-rows", str(added_rows),
            "--dir", str(suite_dir),
        ])
        if code != 0:
            print("Failed to import suite.")
            sys.exit(code)

    # Step 2: Run pipeline for each model
    model_paths = {
        "e4b": "ckpt/gevva-e4b-flagship/best",
        "e2b": "ckpt/gevva-e2b",
    }

    for m in args.models:
        resolved_path = model_paths.get(m, m)
        out_dir = Path(f"runs/gevva-{m}-0.2")
        print(f"\n=======================================================")
        print(f"Evaluating Gevva model '{m}' ({resolved_path})")
        print(f"Output directory: {out_dir}")
        print(f"=======================================================")

        cmd = [
            sys.executable, "-m", "decision_index", "pipeline",
            "--engine", "gevva",
            "--suite-dir", str(suite_dir),
            "--option", f"model={resolved_path}",
            "--option", f"device={args.device}",
            "--out", str(out_dir),
        ]
        ret = run_cmd(cmd)
        if ret != 0:
            print(f"Evaluation for {m} exited with code {ret}")

        # Check for results
        scores_file = out_dir / "scores.json"
        index_file = out_dir / "index.json"
        if scores_file.exists():
            with open(scores_file, "r") as f:
                data = json.load(f)
            index_val = data.get("decision_index") or (data.get("index", {}).get("index") if isinstance(data.get("index"), dict) else data.get("index"))
            print(f"\n🎯 [RESULT] Gevva {m} Decision Index 0.2 Score: {index_val}")

    print("\nAll requested evaluations complete!")


if __name__ == "__main__":
    main()
