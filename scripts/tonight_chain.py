#!/usr/bin/env python3
"""tonight_chain.py - Sequenced single-GPU pipeline (strictly serial, no contention).

 1. ensure-committee : launch the committee if not alive (resume, explicit run id)
 2. wait-judge1      : poll run-scoped ok verdicts; liveness-checked, auto-relaunch <=3x
 3. pause            : kill committee group, purge ALL run non-ok rows, unload, GPU drain-wait
 4. compile          : data_pipeline --quick (sanitized vision premises, fresh train/val/test)
 5. train-armA       : clean-only baseline (the synthetic gate's baseline model)
 6. export           : qwen-validated synthetic rows, 10% val holdback (forward-contamination guard)
 7. arms             : build_arms (A=clean, B=clean + <=12.5% validated synthetic)
 8. train-armB       : synthetic arm
 9. gate             : gate_decision (rc 3 = legitimate FAIL, recorded)
10. resume-committee : judges 2-4 in owner order (q3 -> q4 -> deepseek LAST) + watchdog re-arm

SAFETY: any failure after pause writes results/committee_alert.json and relaunches the
committee from the failure handler before re-raising - validation never dies overnight
(audit NIGHT-WASTER 3). Exactly ONE committee launch per path: failure -> handler
resume; success -> resume after the gate, then scripts/committee_watchdog.py re-armed
detached alongside it.
The legacy run_night_queue driver must not be running (startup guard kills it).
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
from typing import Dict

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from build_arms import build_arms, export_validated_synthetic  # noqa: E402

RUN_ID = "run_20260919_223701"
JUDGE1 = "qwen-3.6-27b-q4"
TOTAL = 7515  # Target ok-verdict threshold for Judge 1 pilot gate (raw queue may exceed this as SOTA modes expand)
QUEUE_LOG = REPO / "results" / "tonight_chain.log"
STATUS = REPO / "results" / "tonight_chain_status.json"
ALERT = REPO / "results" / "committee_alert.json"
PID_FILE = REPO / "results" / "sdk_synthetic_run.pid"
COMMITTEE_LOG = REPO / "results" / "tonight_chain" / "committee_resume.log"
WATCHDOG_PID_FILE = REPO / "results" / "committee_watchdog.pid"
WATCHDOG_STDOUT = REPO / "results" / "tonight_chain" / "committee_watchdog_stdout.log"
MAX_COMMITTEE_RESTARTS = 3


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


def write_alert(alert: str, detail: Dict) -> None:
    ALERT.parent.mkdir(parents=True, exist_ok=True)
    ALERT.write_text(json.dumps({"alert": alert, "detail": detail,
                                 "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
                                indent=2))
    log(f"ALERT written: {alert}")


def run(cmd: list, log_name: str, timeout_s: int = 12 * 3600) -> int:
    out = REPO / "results" / "tonight_chain" / log_name
    out.parent.mkdir(parents=True, exist_ok=True)
    log(f"$ {' '.join(cmd)}")
    with open(out, "w") as f:
        rc = subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT, timeout=timeout_s).returncode
    if rc != 0:
        log(f"  FAILED rc={rc}; tail:\n" + "\n".join(out.read_text().splitlines()[-15:]))
    return rc


def committee_pids() -> list:
    # WHY data\.py suffix: interpreter-agnostic (matches .venv/bin/python children too)
    out = subprocess.run(["pgrep", "-f", "generate_sdk_synthetic_data\\.py"],
                         capture_output=True, text=True).stdout.split()
    return [int(p) for p in out]


def kill_committee() -> None:
    for round_no in range(3):
        pids = committee_pids()
        if not pids:
            return
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        time.sleep(2)
    log(f"committee kill rounds done; remaining: {committee_pids()}")


def gpu_memory_used_mb() -> float:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip().splitlines()[0]
    return float(out)


def wait_gpu_drain(threshold_mb: float = 2500.0, timeout_s: int = 180) -> bool:
    """WHY: /models/unload is asynchronous - today train_A OOMed 8s after unload."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if gpu_memory_used_mb() < threshold_mb:
            return True
        time.sleep(5)
    return gpu_memory_used_mb() < threshold_mb


def unload_gpu_models() -> None:
    errors = []
    try:
        with urllib.request.urlopen("http://localhost:8080/v1/models", timeout=10) as r:
            models = json.loads(r.read().decode()).get("data", [])
    except Exception as e:
        log(f"model list unavailable: {e}")
        return
    for m in models:
        if m.get("status", {}).get("value") == "unloaded":
            continue
        req = urllib.request.Request("http://localhost:8080/models/unload",
                                     data=json.dumps({"model": m["id"]}).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(req, timeout=60)
            log(f"unloaded {m['id']}")
        except Exception as e:
            errors.append(f"{m['id']}: {e}")
    if errors:
        log(f"unload errors: {errors}")


def resume_cmd() -> list:
    validators = ["qwen-3.6-27b-q4", "qwen-3.8-125b-q3", "qwen-3.8-125b-q4", "deepseek-v4-flash-q3"]
    return [sys.executable, "-u", "generate_sdk_synthetic_data.py",
            "--teacher-url", "http://localhost:8080/v1", "--teacher-model", "gemma-4-31b-q4",
            "--validator-url", "http://localhost:8080/v1", "--validators", ",".join(validators),
            "--out-dir", "./data", "--resume-run", RUN_ID, "--validator-timeout", "600"]


def ensure_committee() -> subprocess.Popen:
    """Launches the committee (resume, explicit run id) and verifies liveness + progress log."""
    COMMITTEE_LOG.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(resume_cmd(), cwd=REPO, start_new_session=True,
                            stdout=open(COMMITTEE_LOG, "a"), stderr=subprocess.STDOUT)
    PID_FILE.write_text(str(proc.pid))
    time.sleep(45)
    if proc.poll() is not None:
        raise RuntimeError(f"committee died within 45s rc={proc.poll()}")
    tail = COMMITTEE_LOG.read_text()[-2000:] if COMMITTEE_LOG.exists() else ""
    if "Resuming validation run" not in tail:
        raise RuntimeError(f"committee log lacks resume marker; tail:\n{tail}")
    log(f"committee live (pid {proc.pid}), liveness verified")
    return proc


def ensure_or_await_committee() -> subprocess.Popen:
    """Returns a live committee handle; relaunches (<=3x) if dead or unresumable."""
    for attempt in range(MAX_COMMITTEE_RESTARTS):
        try:
            return ensure_committee()
        except RuntimeError as e:
            log(f"committee start failed (attempt {attempt + 1}/{MAX_COMMITTEE_RESTARTS}): {e}")
            write_alert("committee_start_failed", {"attempt": attempt + 1, "error": str(e)})
            time.sleep(60)
    raise RuntimeError("committee could not be started after max restarts")


def wait_judge1(db_path: Path, proc: subprocess.Popen) -> None:
    """Polls run-scoped ok verdicts with committee liveness + bounded auto-restart."""
    restarts = 0
    deadline = time.time() + 6 * 3600
    while time.time() < deadline:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA busy_timeout=30000")
        n = conn.execute(
            "SELECT COUNT(*) FROM sample_verdicts WHERE run_id=? AND judge_model=? AND status='ok'",
            (RUN_ID, JUDGE1)).fetchone()[0]
        conn.close()
        if n >= TOTAL:
            log(f"judge 1 complete: {n}/{TOTAL} ok verdicts")
            return
        if proc.poll() is not None:
            restarts += 1
            if restarts > MAX_COMMITTEE_RESTARTS:
                raise RuntimeError("committee keeps dying during judge 1 - giving up")
            log(f"committee died mid-judge-1 (restart {restarts}/{MAX_COMMITTEE_RESTARTS}); relaunching")
            write_alert("committee_died_during_judge1", {"restart": restarts})
            proc = ensure_or_await_committee()
        time.sleep(120)
    raise TimeoutError("judge 1 did not complete in time")


def purge_run_non_ok(db_path: Path) -> int:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA busy_timeout=30000")
    cur = conn.execute("DELETE FROM sample_verdicts WHERE run_id = ? AND status != 'ok'", (RUN_ID,))
    conn.commit()
    n = cur.rowcount
    conn.close()
    return n


def main() -> None:
    db_path = REPO / "data" / "validation_metrics.db"

    # LAUNCH GUARD (audit NIGHT-WASTER 4): the legacy driver fires a stale stage list
    # on committee exit and targets the same ckpt dirs. Retire it hard.
    stale = subprocess.run(["pgrep", "-f", "run_night_queue\\.py"],
                           capture_output=True, text=True).stdout.split()
    for pid in [int(p) for p in stale]:
        try:
            os.kill(pid, signal.SIGKILL)
            log(f"killed stale night-queue driver pid {pid}")
        except ProcessLookupError:
            pass
    legacy = REPO / "scripts" / "run_night_queue.py"
    if legacy.exists():
        legacy.rename(legacy.with_suffix(".py.retired"))
        log("retired scripts/run_night_queue.py (superseded by tonight_chain)")

    set_stage("ensure-committee", "running")
    proc = ensure_or_await_committee()
    set_stage("ensure-committee", "done", f"pid={proc.pid}")

    set_stage("wait-judge1", "running")
    wait_judge1(db_path, proc)

    # From here the committee is intentionally stopped; EVERY failure path must
    # resume it so the multi-day validation tail never dies silently
    # (audit NIGHT-WASTER 3). Exactly ONE committee launch per path:
    #   failure -> kill leftovers + alert + relaunch inside the handler, re-raise
    #   success -> relaunch after the gate, then re-arm the watchdog.
    def launch_committee_detached() -> subprocess.Popen:
        proc2 = subprocess.Popen(resume_cmd(), cwd=REPO, start_new_session=True,
                                 stdout=open(COMMITTEE_LOG, "a"), stderr=subprocess.STDOUT)
        PID_FILE.write_text(str(proc2.pid))
        return proc2

    try:
        set_stage("pause", "running")
        kill_committee()
        purged = purge_run_non_ok(db_path)
        unload_gpu_models()
        if not wait_gpu_drain():
            raise RuntimeError("GPU did not drain after unload (possible foreign process)")
        set_stage("pause", "done", f"purged {purged} non-ok rows; GPU drained")

        set_stage("compile", "running")
        rc = run([sys.executable, "-u", "data_pipeline.py", "--out-dir", "./data", "--quick"],
                 "02_compile.log", timeout_s=3600)
        if rc != 0:
            set_stage("compile", "failed", f"rc={rc}")
            raise SystemExit(1)
        set_stage("compile", "done")

        set_stage("train-armA", "running")
        rc = run([sys.executable, "-u", "train_cross_encoder.py", "--data-dir", "./data",
                  "--test-file", "data/test.jsonl", "--out-dir", "ckpt/shakedown_A",
                  "--target-quant", "none", "--epochs", "3"], "armA_train.log", timeout_s=12 * 3600)
        if rc != 0:
            set_stage("train-armA", "failed", f"rc={rc}")
            raise SystemExit(1)
        set_stage("train-armA", "done")

        set_stage("export", "running")
        staged_dir = REPO / "data" / "staged"
        staged_dir.mkdir(exist_ok=True)
        raw = REPO / "data" / "sdk_synthetic_raw.jsonl"
        stats = export_validated_synthetic(str(raw), str(db_path),
                                           str(staged_dir / "sdk_synthetic_train.jsonl"),
                                           str(staged_dir / "sdk_synthetic_val.jsonl"),
                                           run_id=RUN_ID, judge_model=JUDGE1, val_frac=0.1, seed=42)
        if stats["validated_rows"] < 0.9 * TOTAL:
            raise RuntimeError(f"validated synthetic too thin for a meaningful arm: {stats}")
        set_stage("export", "done", json.dumps(stats))
        log(f"exported validated synthetic: {stats}")

        set_stage("arms", "running")
        manifest = build_arms(str(REPO / "data"), 0.125, 42)
        if manifest.get("armB_synthetic_rows", 0) <= 0:
            raise RuntimeError("arm B has no synthetic rows - gate would be meaningless")
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

        set_stage("gate", "running")
        rc = run([sys.executable, "-u", "scripts/gate_decision.py",
                  "--arm-a", "ckpt/shakedown_A", "--arm-b", "ckpt/shakedown_B",
                  "--out", "results/gate_decision.json"], "08_gate.log")
        gate_path = REPO / "results" / "gate_decision.json"
        gate_pass = gate_path.exists() and json.loads(gate_path.read_text()).get("passed", False)
        set_stage("gate", "done" if rc in (0, 3) else "failed",
                  f"passed={gate_pass} (single-judge pilot gate; flagship mixing requires the multi-judge gate)")
    except BaseException:
        # WHY BaseException: stage failures raise SystemExit(1), and Ctrl-C during a
        # multi-hour training stage must still keep the validation tail alive.
        try:
            kill_committee()
            proc2 = launch_committee_detached()
            write_alert("chain_failed_committee_resumed",
                        {"note": "validation tail continues; chain stage failed - see tonight_chain logs"})
            log(f"chain failed: committee resumed (pid {proc2.pid}) so judges 2-4 continue")
        except Exception as resume_err:
            log(f"chain failed AND committee resume FAILED: {resume_err}")
        raise

    # SUCCESS path: judges 2-4 tail continues for days; keep a health watchdog on it.
    set_stage("resume-committee", "running")
    proc2 = launch_committee_detached()
    watchdog = subprocess.Popen(
        [sys.executable, "-u", "scripts/committee_watchdog.py",
         "--pid-file", "results/sdk_synthetic_run.pid"],
        cwd=REPO, start_new_session=True, stdout=open(WATCHDOG_STDOUT, "a"),
        stderr=subprocess.STDOUT)
    WATCHDOG_PID_FILE.write_text(str(watchdog.pid))
    log(f"committee resumed (judges 2-4, deepseek LAST) pid={proc2.pid}; "
        f"watchdog re-armed pid={watchdog.pid}")
    set_stage("resume-committee", "done", f"committee pid={proc2.pid}; watchdog pid={watchdog.pid}")
    set_stage("chain", "completed")


if __name__ == "__main__":
    main()
