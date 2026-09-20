#!/usr/bin/env python3
"""tonight_chain.py - Sequenced single-GPU pipeline (strictly serial, no contention).

 1. wait-judge1     : poll until qwen-3.6-27b-q4 has 7515 ok verdicts (committee live)
 2. pause-committee : kill committee process group, purge non-ok rows, unload qwen
 3. train-armA      : clean-only baseline model (the synthetic gate's baseline)
 4. export          : qwen-validated synthetic rows -> data/staged/
 5. arms            : build_arms (A=clean, B=clean + <=12.5% validated synthetic)
 6. train-armB      : synthetic arm
 7. resume-committee: judges 2-4 in owner order (q3 -> q4 -> deepseek LAST), watchdog re-armed

Every stage logs to results/tonight_chain.log; status in results/tonight_chain_status.json.
"""

import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from build_arms import build_arms, export_validated_synthetic  # noqa: E402

RUN_ID = "run_20260919_223701"
JUDGE1 = "qwen-3.6-27b-q4"
TOTAL = 7515
QUEUE_LOG = REPO / "results" / "tonight_chain.log"
STATUS = REPO / "results" / "tonight_chain_status.json"


def log(msg: str) -> None:
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(QUEUE_LOG, "a") as f:
        f.write(line + "\n")


def set_stage(stage: str, state: str, detail: str = "") -> None:
    data = json.loads(STATUS.read_text()) if STATUS.exists() else {"history": []}
    data["history"] = data.get("history", []) + [{"stage": stage, "state": state,
                                                  "detail": detail, "at": time.strftime("%H:%M:%S")}]
    data["current_stage"], data["state"] = stage, state
    STATUS.write_text(json.dumps(data, indent=2))
    log(f"[{stage}] {state} {detail}")


def run(cmd: list, log_name: str, timeout_s: int = 10 * 3600) -> int:
    out = REPO / "results" / "tonight_chain" / log_name
    out.parent.mkdir(parents=True, exist_ok=True)
    log(f"$ {' '.join(cmd)}")
    with open(out, "w") as f:
        rc = subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT, timeout=timeout_s).returncode
    if rc != 0:
        log(f"  FAILED rc={rc}; tail:\n" + "\n".join(out.read_text().splitlines()[-15:]))
    return rc


def committee_pids() -> list:
    out = subprocess.run(["pgrep", "-f", "python3 -u generate_sdk_syn"],
                         capture_output=True, text=True).stdout.split()
    return [int(p) for p in out]


def kill_committee() -> None:
    pids = committee_pids()
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    # uv wrapper parents too (they do not forward signals to the python child)
    out = subprocess.run(["pgrep", "-f", "uv run python -u generate_sdk_syn"],
                         capture_output=True, text=True).stdout.split()
    for pid in [int(p) for p in out]:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    time.sleep(2)
    log(f"committee pids killed: {pids}")


def unload_gpu_models() -> None:
    try:
        with urllib.request.urlopen("http://localhost:8080/v1/models", timeout=10) as r:
            models = json.loads(r.read().decode()).get("data", [])
        for m in models:
            if m.get("status", {}).get("value") != "unloaded":
                req = urllib.request.Request("http://localhost:8080/models/unload",
                                             data=json.dumps({"model": m["id"]}).encode(),
                                             headers={"Content-Type": "application/json"})
                urllib.request.urlopen(req, timeout=30)
                log(f"unloaded {m['id']}")
    except Exception as e:
        log(f"unload skipped: {e}")


def purge_non_ok(judge: str) -> int:
    conn = sqlite3.connect(REPO / "data" / "validation_metrics.db")
    cur = conn.execute("DELETE FROM sample_verdicts WHERE judge_model = ? AND status != 'ok'", (judge,))
    conn.commit()
    n = cur.rowcount
    conn.close()
    return n


def wait_judge1(db_path: Path, timeout_s: int = 4 * 3600) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        conn = sqlite3.connect(db_path)
        n = conn.execute("SELECT COUNT(*) FROM sample_verdicts WHERE judge_model=? AND status='ok'",
                         (JUDGE1,)).fetchone()[0]
        conn.close()
        if n >= TOTAL:
            log(f"judge 1 complete: {n}/{TOTAL} ok verdicts")
            return
        log(f"judge 1 at {n}/{TOTAL} ok verdicts; waiting...")
        time.sleep(120)
    raise TimeoutError("judge 1 did not complete in time")


def main() -> None:
    db_path = REPO / "data" / "validation_metrics.db"
    set_stage("wait-judge1", "running")
    wait_judge1(db_path)

    set_stage("pause-committee", "running")
    kill_committee()
    purged = purge_non_ok(JUDGE1)
    unload_gpu_models()
    set_stage("pause-committee", "done", f"purged {purged} non-ok rows")

    set_stage("train-armA", "running")
    rc = run([sys.executable, "-u", "train_cross_encoder.py", "--data-dir", "./data",
              "--test-file", "data/test.jsonl", "--out-dir", "ckpt/shakedown_A",
              "--target-quant", "none", "--epochs", "3"], "armA_train.log", timeout_s=12 * 3600)
    if rc != 0:
        set_stage("train-armA", "failed", f"rc={rc}")
        raise SystemExit(1)
    set_stage("train-armA", "done")

    set_stage("export", "running")
    raw = REPO / "data" / "sdk_synthetic_raw.jsonl"
    staged = REPO / "data" / "staged" / "sdk_synthetic_train.jsonl"
    stats = export_validated_synthetic(str(raw), str(db_path), str(staged), JUDGE1)
    set_stage("export", "done", json.dumps(stats))
    log(f"exported validated synthetic: {stats}")

    set_stage("arms", "running")
    manifest = build_arms(str(REPO / "data"), 0.125, 42)
    set_stage("arms", "done", json.dumps(manifest))

    set_stage("train-armB", "running")
    rc = run([sys.executable, "-u", "train_cross_encoder.py",
              "--train-file", "data/armB_train.jsonl", "--data-dir", "./data",
              "--test-file", "data/test.jsonl", "--out-dir", "ckpt/shakedown_B",
              "--target-quant", "none", "--epochs", "3"], "armB_train.log", timeout_s=12 * 3600)
    if rc != 0:
        set_stage("train-armB", "failed", f"rc={rc}")
        raise SystemExit(1)
    set_stage("train-armB", "done")

    set_stage("resume-committee", "running")
    validators = ["qwen-3.6-27b-q4", "qwen-3.8-125b-q3", "qwen-3.8-125b-q4", "deepseek-v4-flash-q3"]
    cmd = [sys.executable, "-u", "generate_sdk_synthetic_data.py",
           "--teacher-url", "http://localhost:8080/v1", "--teacher-model", "gemma-4-31b-q4",
           "--validator-url", "http://localhost:8080/v1", "--validators", ",".join(validators),
           "--out-dir", "./data", "--resume-run", RUN_ID, "--validator-timeout", "600"]
    proc = subprocess.Popen(cmd, cwd=REPO, start_new_session=True,
                            stdout=open(REPO / "results" / "tonight_chain" / "committee_resume.log", "w"),
                            stderr=subprocess.STDOUT)
    (REPO / "results" / "sdk_synthetic_run.pid").write_text(str(proc.pid))
    log(f"committee resumed (pid {proc.pid}) with judge order: {validators} - deepseek LAST")
    # Re-arm the committee health watchdog for the multi-day judges 2-4 tail.
    watchdog = subprocess.Popen([sys.executable, "-u", str(REPO / "scripts" / "committee_watchdog.py")],
                                cwd=REPO, start_new_session=True,
                                stdout=open(REPO / "results" / "tonight_chain" / "watchdog.log", "a"),
                                stderr=subprocess.STDOUT)
    (REPO / "results" / "committee_watchdog.pid").write_text(str(watchdog.pid))
    log(f"watchdog re-armed (pid {watchdog.pid})")
    set_stage("resume-committee", "done", f"pid={proc.pid}")
    set_stage("chain", "completed")


if __name__ == "__main__":
    main()
