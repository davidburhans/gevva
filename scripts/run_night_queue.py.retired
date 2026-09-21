#!/usr/bin/env python3
"""run_night_queue.py - Autonomous overnight GPU queue (pre-registered, logged, gated).

Stages (each logged under results/night_queue/, status in results/night_queue_status.json):
  1. wait     : wait for the committee run (sdk_synthetic_run.pid) to exit
  2. stage_sdk: move fresh sdk_synthetic_*.jsonl into data/staged/ so the compiler
                emits a CLEAN train set (arm mixing is build_arms' job, not the compiler's)
  3. compile  : data_pipeline --quick  (train/val/test regenerated, contamination guards on)
  4. arms     : build_arms (A = clean, B = clean + <=12.5% validated synthetic)
  5. free_gpu : unload every llama-swap model so training owns the 5090
  6. train_A  : shakedown arm A (BF16 LoRA, no QAT, 3 epochs)
  7. train_B  : shakedown arm B
  8. baselines: scripts/eval_baselines.py if present (best-effort, never fatal)
  9. gate     : scripts/gate_decision.py  ->  results/gate_decision.json
 10. stage2   : data_pipeline --stage2, then flagship training with the gated recipe

Usage: uv run python scripts/run_night_queue.py [--committee-pid-file results/sdk_synthetic_run.pid]
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Callable, Dict, List, Optional

REPO = Path(__file__).resolve().parent.parent
QUEUE_DIR = REPO / "results" / "night_queue"
STATUS_FILE = REPO / "results" / "night_queue_status.json"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def set_status(stage: str, state: str, detail: str = "") -> None:
    status: Dict = {}
    if STATUS_FILE.exists():
        status = json.loads(STATUS_FILE.read_text())
    history: List = status.get("history", [])
    history.append({"stage": stage, "state": state, "detail": detail,
                    "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    status["history"] = history
    status["current_stage"] = stage
    status["state"] = state
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATUS_FILE.write_text(json.dumps(status, indent=2))


def run_cmd(cmd: List[str], log_name: str, timeout_s: int = 48 * 3600) -> int:
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    log_path = QUEUE_DIR / log_name
    log(f"$ {' '.join(cmd)}  (log: {log_path})")
    with open(log_path, "w") as f:
        proc = subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT, timeout=timeout_s)
    if proc.returncode != 0:
        tail = log_path.read_text()[-2000:]
        log(f"STAGE FAILED rc={proc.returncode}. Log tail:\n{tail}")
    return proc.returncode


def wait_for_committee(pid_file: Path, timeout_s: int = 36 * 3600) -> None:
    pid = pid_file.read_text().strip() if pid_file.exists() else ""
    if not pid or not Path(f"/proc/{pid}").exists():
        log(f"No live committee process (pid='{pid}'); skipping wait.")
        return
    log(f"Waiting for committee run pid={pid} to finish...")
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if not Path(f"/proc/{pid}").exists():
            log("Committee process exited.")
            return
        time.sleep(60)
    raise TimeoutError("committee run did not finish within timeout")


def stage_sdk_files(data_dir: Path) -> str:
    """Moves fresh sdk files to data/staged/ so the compiler emits a clean train set."""
    staged = data_dir / "staged"
    staged.mkdir(exist_ok=True)
    moved = []
    for name in ("sdk_synthetic_train.jsonl", "sdk_synthetic_val.jsonl"):
        src = data_dir / name
        if src.exists():
            src.rename(staged / name)
            moved.append(name)
    if not (staged / "sdk_synthetic_train.jsonl").exists():
        return "no fresh sdk synthetic data found - arms will train clean-only"
    return f"staged: {', '.join(moved)}"


def free_gpu() -> str:
    """Unloads every llama-swap model so training owns the GPU (best effort)."""
    freed = []
    try:
        with urllib.request.urlopen("http://localhost:8080/v1/models", timeout=10) as r:
            models = json.loads(r.read().decode()).get("data", [])
        for m in models:
            req = urllib.request.Request(
                "http://localhost:8080/models/unload",
                data=json.dumps({"model": m["id"]}).encode(),
                headers={"Content-Type": "application/json"})
            try:
                urllib.request.urlopen(req, timeout=15)
                freed.append(m["id"])
            except Exception:
                pass
    except Exception as e:
        return f"unload skipped ({e})"
    return f"unloaded: {freed or 'nothing resident'}"


def stage2_recipe(gate_pass: bool) -> Dict:
    return {
        "compile": [sys.executable, "-u", "data_pipeline.py", "--out-dir", "./data", "--stage2"],
        "synthetic_mixed": gate_pass,
        "train_cmd": ["--out-dir", "ckpt/gemma-4-e2b-nli-stage2-clean" + ("-synthetic" if gate_pass else ""),
                      "--target-quant", "none", "--epochs", "3"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--committee-pid-file", default="results/sdk_synthetic_run.pid")
    parser.add_argument("--skip-wait", action="store_true", help="skip committee wait (debug)")
    parser.add_argument("--through", default="stage2",
                        help="last stage to run: wait,stage_sdk,compile,arms,train_A,train_B,baselines,gate,stage2")
    args = parser.parse_args()
    order = ["wait", "stage_sdk", "compile", "arms", "free_gpu", "train_A", "train_B",
             "baselines", "gate", "stage2"]
    last = order.index(args.through)
    data_dir = REPO / "data"
    set_status("queue", "started")

    if last >= 0:
        if not args.skip_wait:
            set_status("wait", "running")
            wait_for_committee(REPO / args.committee_pid_file)
        set_status("wait", "done")
    if last >= order.index("stage_sdk"):
        set_status("stage_sdk", "running")
        detail = stage_sdk_files(data_dir)
        set_status("stage_sdk", "done", detail)
        log(detail)
    if last >= order.index("compile"):
        set_status("compile", "running")
        rc = run_cmd([sys.executable, "-u", "data_pipeline.py", "--out-dir", "./data", "--quick"],
                     "02_compile.log")
        if rc != 0:
            set_status("compile", "failed", f"rc={rc}")
            raise SystemExit(1)
        set_status("compile", "done")
    if last >= order.index("arms"):
        set_status("arms", "running")
        rc = run_cmd([sys.executable, "-u", "scripts/build_arms.py", "--data-dir", "./data"],
                     "03_arms.log")
        set_status("arms", "done" if rc == 0 else "failed", f"rc={rc}")
        if rc != 0:
            raise SystemExit(1)
    if last >= order.index("free_gpu"):
        set_status("free_gpu", "running")
        detail = free_gpu()
        set_status("free_gpu", "done", detail)
        log(detail)
    for arm in ("A", "B"):
        if last >= order.index(f"train_{arm}"):
            set_status(f"train_{arm}", "running")
            rc = run_cmd([sys.executable, "-u", "train_cross_encoder.py",
                          "--train-file", f"data/arm{arm}_train.jsonl",
                          "--data-dir", "./data", "--test-file", "data/test.jsonl",
                          "--out-dir", f"ckpt/shakedown_{arm}",
                          "--target-quant", "none", "--epochs", "3"],
                         f"05_train_{arm}.log", timeout_s=24 * 3600)
            set_status(f"train_{arm}", "done" if rc == 0 else "failed", f"rc={rc}")
            if rc != 0:
                raise SystemExit(1)
    if last >= order.index("baselines"):
        set_status("baselines", "running")
        baseline_script = REPO / "scripts" / "eval_baselines.py"
        if baseline_script.exists():
            rc = run_cmd([sys.executable, "-u", str(baseline_script)], "07_baselines.log",
                         timeout_s=12 * 3600)
            set_status("baselines", "done" if rc == 0 else "failed", f"rc={rc}")
        else:
            set_status("baselines", "skipped", "scripts/eval_baselines.py not present yet")
            log("baselines skipped (script not present)")
    gate_pass = False
    if last >= order.index("gate"):
        set_status("gate", "running")
        rc = run_cmd([sys.executable, "-u", "scripts/gate_decision.py",
                      "--arm-a", "ckpt/shakedown_A", "--arm-b", "ckpt/shakedown_B",
                      "--out", "results/gate_decision.json"], "08_gate.log")
        gate_path = REPO / "results" / "gate_decision.json"
        gate_pass = gate_path.exists() and json.loads(gate_path.read_text()).get("passed", False)
        set_status("gate", "done" if rc in (0, 3) else "failed",
                   f"passed={gate_pass} (rc={rc}; rc3 = legit FAIL)")
    if last >= order.index("stage2"):
        recipe = stage2_recipe(gate_pass)
        set_status("stage2", "running", json.dumps(recipe))
        rc = run_cmd(recipe["compile"], "09_stage2_compile.log", timeout_s=8 * 3600)
        if rc != 0:
            set_status("stage2", "failed", "compile")
            raise SystemExit(1)
        free_gpu()
        if recipe["synthetic_mixed"]:
            run_cmd([sys.executable, "-u", "scripts/build_arms.py", "--data-dir", "./data"],
                    "10_stage2_arms.log")
            train_file = "data/armB_train.jsonl"
        else:
            train_file = "data/train.jsonl"
        rc = run_cmd([sys.executable, "-u", "train_cross_encoder.py",
                      "--train-file", train_file, "--data-dir", "./data",
                      "--test-file", "data/test.jsonl", *recipe["train_cmd"]],
                     "11_stage2_train.log", timeout_s=96 * 3600)
        set_status("stage2", "done" if rc == 0 else "failed", f"rc={rc}")
    set_status("queue", "completed")


if __name__ == "__main__":
    main()
