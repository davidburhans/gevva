#!/usr/bin/env python3
"""validation_metrics_db.py - SQLite persistence for judge verdicts & run bookkeeping.

Split out of validator_committee.py (SRP): this module owns ALL storage concerns —
schema (validator_schema.sql), run rows, per-batch verdict writes (idempotent via a
UNIQUE index so resumed runs never duplicate or skew judge statistics), resume
lookups, and the `judge_performance` analysis view.

Usage example:
    from validation_metrics_db import RunSpec, ValidationMetricsDB
    db = ValidationMetricsDB("data/validation_metrics.db")
    run_id = db.start_run(RunSpec("gemma-4-31b-q4", ["qwen-3.6-27b-q4"], 10, 42))
    db.finish_run(run_id, "completed", total_samples=60)
"""

import json
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# WHY: DDL lives in validator_schema.sql, reviewable independently of wrapper code (SRP).
DB_SCHEMA = (Path(__file__).parent / "validator_schema.sql").read_text(encoding="utf-8")

# SQLite's default max variables per query is 999; stay safely below it for IN clauses.
_SQL_CHUNK = 500


def utc_now() -> str:
    """UTC ISO-8601 timestamp for DB rows."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass
class RunSpec:
    """Parameters describing one pipeline invocation, persisted in the `runs` table."""
    teacher_model: str
    validator_models: List[str]
    samples_per_mode: int
    seed: int


class ValidationMetricsDB:
    """Thin SQLite wrapper persisting judge verdicts & batch telemetry for later analysis.

    Inject an alternate `conn` (e.g. `:memory:`) in tests.

    Example:
        db = ValidationMetricsDB("data/validation_metrics.db")
        run_id = db.start_run(RunSpec("gemma-4-31b-q4", ["qwen-3.6-27b-q4"], 10, 42))
        db.finish_run(run_id, "completed", total_samples=60)
    """

    def __init__(self, db_path: str, conn: Optional[sqlite3.Connection] = None):
        self.db_path = db_path
        if conn is not None:
            self.conn = conn
        else:
            os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
            self.conn = sqlite3.connect(db_path)
        # WHY: name-addressable rows (sqlite3.Row) so analysis queries read r["judge_model"].
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(DB_SCHEMA)
        self.conn.commit()

    def __enter__(self) -> "ValidationMetricsDB":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        self.conn.close()

    def start_run(self, spec: RunSpec, run_id: Optional[str] = None) -> str:
        """Inserts a running run row; returns the run_id (timestamp-based by default)."""
        rid = run_id or f"run_{time.strftime('%Y%m%d_%H%M%S')}"
        self.conn.execute(
            "INSERT INTO runs (run_id, started_at, teacher_model, validator_models,"
            " samples_per_mode, seed, status) VALUES (?,?,?,?,?,?, 'running')",
            (rid, utc_now(), spec.teacher_model, json.dumps(spec.validator_models),
             spec.samples_per_mode, spec.seed),
        )
        self.conn.commit()
        return rid

    def run_exists(self, run_id: str) -> bool:
        row = self.conn.execute("SELECT 1 FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return row is not None

    def latest_resumable_run(self, validator_models: List[str]) -> Optional[str]:
        """Newest run (any non-completed status) with the exact same judge lineup.

        Example:
            run_id = db.latest_resumable_run(["qwen-3.6-27b-q4", "deepseek-v4-flash-q3"])
        """
        row = self.conn.execute(
            "SELECT run_id FROM runs WHERE validator_models = ? AND status != 'completed'"
            " ORDER BY started_at DESC LIMIT 1",
            (json.dumps(validator_models),),
        ).fetchone()
        return row["run_id"] if row else None

    def count_verdicts(self, run_id: str, judge_model: Optional[str] = None) -> int:
        """Number of persisted verdict rows for a run (optionally one judge)."""
        if judge_model is None:
            row = self.conn.execute(
                "SELECT COUNT(*) AS n FROM sample_verdicts WHERE run_id = ?", (run_id,)).fetchone()
        else:
            row = self.conn.execute(
                "SELECT COUNT(*) AS n FROM sample_verdicts WHERE run_id = ? AND judge_model = ?",
                (run_id, judge_model)).fetchone()
        return int(row["n"])

    def persisted_verdicts(self, run_id: str, judge_model: str,
                           sample_ids: Sequence[str]) -> Dict[str, Any]:
        """Maps sample_id -> persisted verdict row for one judge (chunked IN queries).

        Example:
            done = db.persisted_verdicts(run_id, "qwen-3.6-27b-q4", [s["id"] for s in samples])
        """
        done: Dict[str, Any] = {}
        ids = list(sample_ids)
        for start in range(0, len(ids), _SQL_CHUNK):
            chunk = ids[start:start + _SQL_CHUNK]
            placeholders = ",".join("?" * len(chunk))
            rows = self.conn.execute(
                f"SELECT * FROM sample_verdicts WHERE run_id = ? AND judge_model = ?"
                f" AND sample_id IN ({placeholders})",
                [run_id, judge_model, *chunk],
            ).fetchall()
            for row in rows:
                done[row["sample_id"]] = row
        return done

    def finish_run(self, run_id: str, status: str = "completed", total_samples: Optional[int] = None) -> None:
        self.conn.execute(
            "UPDATE runs SET finished_at = ?, status = ?, total_samples = ? WHERE run_id = ?",
            (utc_now(), status, total_samples, run_id),
        )
        self.conn.commit()

    def record_batch(self, run_id: str, judge_model: str, batch_index: int,
                     batch_size: int, parse_ok: bool, latency_ms: float) -> None:
        self.conn.execute(
            "INSERT INTO judge_batches (run_id, judge_model, batch_index, batch_size,"
            " parse_ok, latency_ms, created_at) VALUES (?,?,?,?,?,?,?)",
            (run_id, judge_model, batch_index, batch_size, int(parse_ok), latency_ms, utc_now()),
        )
        self.conn.commit()

    def record_verdicts(self, rows: Sequence[Tuple]) -> None:
        # WHY: INSERT OR REPLACE + the UNIQUE(run_id, sample_id, judge_model) index makes
        # re-validating a partially-persisted batch on resume idempotent - judge statistics
        # can never be double-counted.
        self.conn.executemany(
            "INSERT OR REPLACE INTO sample_verdicts (run_id, sample_id, source, judge_model,"
            " generator_label, verdict_label, status, rationale, agreed_with_generator,"
            " latency_ms, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )
        self.conn.commit()

    def record_final_labels(self, rows: Sequence[Tuple]) -> None:
        self.conn.executemany(
            "INSERT OR REPLACE INTO final_labels (run_id, sample_id, generator_label,"
            " final_label, confidence, needs_review, disagreement_type) VALUES (?,?,?,?,?,?,?)",
            rows,
        )
        self.conn.commit()

    def judge_summary(self, run_id: str) -> List[sqlite3.Row]:
        cur = self.conn.execute(
            "SELECT * FROM judge_performance WHERE run_id = ? ORDER BY judge_model", (run_id,))
        return cur.fetchall()
