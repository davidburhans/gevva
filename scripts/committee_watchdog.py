#!/usr/bin/env python3
"""committee_watchdog.py - Health monitor for the running validation committee.

Polls the metrics DB every poll_seconds and writes results/committee_alert.json when:
  - failure rate (status != ok) over the most recent `window` verdicts exceeds max_failure_rate, OR
  - verdict throughput falls below min_per_hour over the last hour.

ALERT-ONLY by design: it never kills the committee (the night-queue driver fires on
committee exit, so a kill would prematurely trigger the downstream stages).

Usage: uv run python scripts/committee_watchdog.py [--pid-file results/sdk_synthetic_run.pid]
"""

import argparse
import json
import sqlite3
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DB = REPO / "data" / "validation_metrics.db"
ALERT = REPO / "results" / "committee_alert.json"


def check(window: int = 40, max_failure_rate: float = 0.30, min_per_hour: float = 20.0) -> dict:
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT status, created_at FROM sample_verdicts ORDER BY id DESC LIMIT ?", (window,)).fetchall()
    hour_rows = conn.execute(
        "SELECT COUNT(*) AS n FROM sample_verdicts WHERE created_at > datetime('now', '-1 hour')").fetchone()
    conn.close()
    if not rows:
        return {"alert": None, "detail": "no verdicts yet"}
    failures = sum(1 for r in rows if r["status"] != "ok")
    rate = failures / len(rows)
    per_hour = hour_rows["n"]
    alert = None
    if rate > max_failure_rate:
        alert = f"failure rate {rate:.0%} over last {len(rows)} verdicts (> {max_failure_rate:.0%})"
    elif per_hour < min_per_hour:
        alert = f"throughput {per_hour}/hr (< {min_per_hour}/hr)"
    return {"alert": alert, "recent_failures": failures, "window": len(rows),
            "failure_rate": round(rate, 3), "verdicts_last_hour": per_hour}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid-file", default="results/sdk_synthetic_run.pid")
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--max-failure-rate", type=float, default=0.30)
    parser.add_argument("--min-per-hour", type=float, default=20.0)
    args = parser.parse_args()
    log = REPO / "results" / "committee_watchdog.log"

    def emit(msg: str) -> None:
        line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log, "a") as f:
            f.write(line + "\n")

    while True:
        pid = Path(args.pid_file).exists() and Path(args.pid_file).read_text().strip()
        if not pid or not Path(f"/proc/{pid}").exists():
            emit("committee process gone - watchdog exiting")
            return
        try:
            health = check(max_failure_rate=args.max_failure_rate, min_per_hour=args.min_per_hour)
        except sqlite3.Error as e:
            health = {"alert": f"db error: {e}"}
        if health.get("alert"):
            health["at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            ALERT.parent.mkdir(parents=True, exist_ok=True)
            ALERT.write_text(json.dumps(health, indent=2))
            emit(f"ALERT: {health['alert']} (written to {ALERT})")
        else:
            emit(f"healthy: {json.dumps({k: v for k, v in health.items() if k != 'alert'})}")
            if ALERT.exists():
                ALERT.unlink()  # recovered
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
