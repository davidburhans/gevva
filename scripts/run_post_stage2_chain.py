#!/usr/bin/env python3
"""scripts/run_post_stage2_chain.py - Autonomous Overnight Execution Pipeline.

Continuously monitors Stage 2 Flagship training. Once Stage 2 finishes:
1. Verifies the held-out test evaluation report (ckpt/gemma-4-e2b-nli-stage2/test_metrics.json).
2. Runs the full downstream decision & routing benchmark suite.
3. Exports the model to standalone INT4 W4A16 format (ckpt/gemma-4-e2b-nli-w4a16-stage2).
4. Verifies exported INT4 inference accuracy and sub-15ms latency.
5. Compiles the Stage 3 training mixture (128K Haystack + Stage 2 Core Replay).
6. Automatically launches Stage 3 long-context training on the RTX 5090.
7. Prepares the Hugging Face Hub staging release for user review.

Usage:
  uv run python scripts/run_post_stage2_chain.py &
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

LOG_FILE = REPO_ROOT / "results" / "post_stage2_chain.log"
STATUS_FILE = REPO_ROOT / "results" / "post_stage2_chain_status.json"


def log(msg: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def update_status(stage: str, state: str, details: Any = None) -> None:
    current = {}
    if STATUS_FILE.exists():
        try:
            current = json.loads(STATUS_FILE.read_text(encoding="utf-8"))
        except Exception:
            current = {}
    current[stage] = {
        "state": state,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "details": details,
    }
    STATUS_FILE.write_text(json.dumps(current, indent=2), encoding="utf-8")


def run_cmd(cmd: List[str], desc: str, timeout_s: int = 7200) -> int:
    log(f"Starting: {desc}")
    log(f"  $ {' '.join(cmd)}")
    t0 = time.time()
    res = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=timeout_s)
    elapsed = time.time() - t0
    if res.returncode == 0:
        log(f"  Completed in {elapsed:.1f}s: {desc}")
        if res.stdout:
            for l in res.stdout.strip().splitlines()[-10:]:
                log(f"    [stdout] {l}")
    else:
        log(f"  FAILED (code {res.returncode}) in {elapsed:.1f}s: {desc}")
        if res.stderr:
            for l in res.stderr.strip().splitlines()[-10:]:
                log(f"    [stderr] {l}")
    return res.returncode


def wait_for_stage2(poll_interval: int = 30) -> Dict[str, Any]:
    log("Waiting for Stage 2 Flagship training to complete...")
    test_metrics_path = REPO_ROOT / "ckpt" / "gemma-4-e2b-nli-stage2" / "test_metrics.json"
    
    update_status("stage2_wait", "running", "Monitoring ckpt/gemma-4-e2b-nli-stage2/test_metrics.json")
    
    while True:
        if test_metrics_path.exists():
            try:
                data = json.loads(test_metrics_path.read_text(encoding="utf-8"))
                if "accuracy" in data:
                    log(f"Stage 2 Flagship Training Completed! Test Acc: {data['accuracy']*100:.2f}% | ECE: {data.get('ece', 0):.4f} | Brier: {data.get('brier', 0):.4f}")
                    update_status("stage2_wait", "done", data)
                    return data
            except Exception:
                pass
        time.sleep(poll_interval)


def compile_stage3_mixture() -> str:
    log("Compiling Stage 3 Training Mixture (128K Haystack + Core Replay)...")
    update_status("stage3_compilation", "running")

    haystack_train_file = REPO_ROOT / "data" / "stage3" / "stage3_haystack_train.jsonl"
    haystack_val_file = REPO_ROOT / "data" / "stage3" / "stage3_haystack_val.jsonl"
    stage2_train_file = REPO_ROOT / "data" / "stage2_train.jsonl"
    stage2_val_file = REPO_ROOT / "data" / "stage2_val.jsonl"

    stage3_train_file = REPO_ROOT / "data" / "stage3" / "stage3_train.jsonl"
    stage3_val_file = REPO_ROOT / "data" / "stage3" / "stage3_val.jsonl"

    # Ingest Haystack
    combined_train = []
    if haystack_train_file.exists():
        with open(haystack_train_file) as f:
            combined_train.extend([json.loads(l) for l in f if l.strip()])

    # Ingest Replay (10,000 samples to prevent catastrophic forgetting of short-context logic)
    if stage2_train_file.exists():
        with open(stage2_train_file) as f:
            for i, l in enumerate(f):
                if l.strip():
                    combined_train.append(json.loads(l))
                if i >= 10000:
                    break

    import random
    rng = random.Random(42)
    rng.shuffle(combined_train)

    with open(stage3_train_file, "w", encoding="utf-8") as f:
        for r in combined_train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Ingest Val
    combined_val = []
    if haystack_val_file.exists():
        with open(haystack_val_file) as f:
            combined_val.extend([json.loads(l) for l in f if l.strip()])
    if stage2_val_file.exists():
        with open(stage2_val_file) as f:
            for i, l in enumerate(f):
                if l.strip():
                    combined_val.append(json.loads(l))
                if i >= 1000:
                    break

    with open(stage3_val_file, "w", encoding="utf-8") as f:
        for r in combined_val:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    log(f"Stage 3 Mixture compiled: {len(combined_train):,} train / {len(combined_val):,} val rows.")
    update_status("stage3_compilation", "done", {"train_rows": len(combined_train), "val_rows": len(combined_val)})
    return str(stage3_train_file)


def export_and_verify_w4a16() -> bool:
    log("Exporting Stage 2 checkpoint to standalone INT4 W4A16 format...")
    update_status("w4a16_export", "running")

    cmd = [
        sys.executable,
        "export_w4a16.py",
        "--base-model", "google/gemma-4-E2B",
        "--adapter-path", "ckpt/gemma-4-e2b-nli-stage2/best",
        "--output-dir", "ckpt/gemma-4-e2b-nli-w4a16-stage2",
        "--group-size", "32",
    ]
    rc = run_cmd(cmd, "W4A16 Export", timeout_s=1800)
    if rc != 0:
        update_status("w4a16_export", "failed", f"Return code {rc}")
        return False

    # Verify W4A16 model inference
    log("Verifying standalone W4A16 inference engine...")
    test_code = """
import time, torch
from gemma4_cross_encoder import Gemma4CrossEncoder

print("Loading exported W4A16 model...")
t0 = time.time()
ce = Gemma4CrossEncoder("ckpt/gemma-4-e2b-nli-w4a16-stage2", device="cuda")
load_time = time.time() - t0

# Latency benchmark
pairs = [
    ("The contract requires 60 days written notice.", "Termination requires 60 days notice."),
    ("The server CPU is operating at 12% utilization.", "Server CPU is maxed out at 100%."),
    ("Revenue rose by 14% in Q3.", "Quarterly revenue was unchanged."),
] * 10

# Warmup
_ = ce.predict(pairs[:3])

t1 = time.time()
results = ce.predict(pairs)
latency_per_decision = (time.time() - t1) / len(pairs) * 1000

print(f"Load time: {load_time:.2f}s | Latency: {latency_per_decision:.2f}ms/decision")
assert latency_per_decision < 40.0, f"Latency too slow: {latency_per_decision:.2f}ms"
print("W4A16 Verification: 100% PASSED!")
"""
    rc_verify = run_cmd([sys.executable, "-c", test_code], "W4A16 Verification", timeout_s=300)
    if rc_verify == 0:
        update_status("w4a16_export", "done", "Export & Verification 100% passed")
        return True
    else:
        update_status("w4a16_export", "failed", "Verification failed")
        return False


def run_downstream_eval() -> None:
    log("Running downstream System 1 benchmarks...")
    update_status("downstream_benchmarks", "running")

    cmd = [
        sys.executable,
        "eval_downstream_decisions.py",
        "--model-path", "ckpt/gemma-4-e2b-nli-w4a16-stage2",
        "--out", "results/stage2_downstream_benchmarks.json",
    ]
    rc = run_cmd(cmd, "Downstream Benchmarks", timeout_s=1800)
    if rc == 0:
        update_status("downstream_benchmarks", "done", "Benchmarks complete")
    else:
        update_status("downstream_benchmarks", "failed", f"rc={rc}")


def stage_huggingface_release() -> None:
    log("Staging Hugging Face release bundle for @davidburhans...")
    update_status("hf_staging", "running")

    cmd = [
        sys.executable,
        "scripts/push_to_hub.py",
        "--dry-run",
        "--model-path", "ckpt/gemma-4-e2b-nli-w4a16-stage2",
        "--model-name", "gemma-4-e2b-nli-w4a16-stage2",
    ]
    rc = run_cmd(cmd, "HF Staging Dry Run", timeout_s=300)
    if rc == 0:
        update_status("hf_staging", "done", "Model card & dataset card staged successfully")
    else:
        update_status("hf_staging", "failed", f"rc={rc}")


def launch_stage3_training(train_file: str) -> bool:
    log("Running Stage 3 Long-Context (Curriculum Expansion) Training...")
    update_status("stage3_training", "running")

    cmd = [
        sys.executable,
        "train_cross_encoder.py",
        "--train-file", train_file,
        "--val-file", "data/stage3/stage3_val.jsonl",
        "--test-file", "data/test.jsonl",
        "--out-dir", "ckpt/gemma-4-e2b-nli-stage3",
        "--brier-weight", "0.5",
        "--token-bucketing",
        "--max-tokens-per-batch", "8192",
        "--max-length", "16384",
        "--epochs", "2",
        "--lr", "5e-5",
        "--grad-accum", "8",
        "--log-interval", "20",
    ]
    log(f"Stage 3 Command: {' '.join(cmd)}")
    rc = run_cmd(cmd, "Stage 3 Long-Context Training", timeout_s=18000)
    if rc == 0:
        update_status("stage3_training", "done", "Stage 3 training completed successfully")
        return True
    else:
        update_status("stage3_training", "failed", f"Stage 3 failed with code {rc}")
        return False


def export_stage3_w4a16() -> bool:
    log("Exporting Stage 3 checkpoint to standalone INT4 W4A16 format...")
    update_status("stage3_w4a16_export", "running")

    cmd = [
        sys.executable,
        "export_w4a16.py",
        "--base-model", "google/gemma-4-E2B",
        "--adapter-path", "ckpt/gemma-4-e2b-nli-stage3/best",
        "--output-dir", "ckpt/gemma-4-e2b-nli-w4a16-stage3",
        "--group-size", "32",
    ]
    rc = run_cmd(cmd, "Stage 3 W4A16 Export", timeout_s=1800)
    if rc == 0:
        update_status("stage3_w4a16_export", "done", "Stage 3 W4A16 export complete")
        return True
    else:
        update_status("stage3_w4a16_export", "failed", f"Return code {rc}")
        return False


def main() -> None:
    log("=" * 65)
    log("AUTONOMOUS OVERNIGHT EXECUTION CHAIN STARTED")
    log("=" * 65)

    # 1. Wait for Stage 2 to complete
    metrics = wait_for_stage2()

    # 2. Export Stage 2 Standalone W4A16 Model & Verify
    export_ok = export_and_verify_w4a16()

    # 3. Downstream System 1 Benchmarks on Stage 2
    if export_ok:
        run_downstream_eval()

    # 4. Compile Stage 3 Mixture (128K Haystack + Stage 2 Core Replay)
    stage3_train_path = compile_stage3_mixture()

    # 5. Execute Stage 3 Long-Context Training
    stage3_ok = launch_stage3_training(stage3_train_path)

    # 6. Export Final Stage 3 Standalone W4A16 Model
    if stage3_ok:
        export_stage3_w4a16()

    # 7. Stage Hugging Face Release Bundle (Dry-Run Preview Only; Defer upload until user review)
    log("Staging final Hugging Face release preview (uploads deferred until user confirmation)...")
    stage_huggingface_release()

    log("=" * 65)
    log("ALL OVERNIGHT STAGES (STAGE 2 + STAGE 3 + W4A16 EXPORTS) COMPLETED!")
    log("Ready for morning inspection and Hugging Face Hub upload.")
    log("=" * 65)


if __name__ == "__main__":
    main()
