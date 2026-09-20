CREATE TABLE IF NOT EXISTS runs (
    run_id           TEXT PRIMARY KEY,
    started_at       TEXT NOT NULL,
    finished_at      TEXT,
    teacher_model    TEXT,
    validator_models TEXT NOT NULL,
    samples_per_mode INTEGER,
    seed             INTEGER,
    total_samples    INTEGER,
    status           TEXT NOT NULL DEFAULT 'running'
);
CREATE TABLE IF NOT EXISTS judge_batches (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT NOT NULL REFERENCES runs(run_id),
    judge_model TEXT NOT NULL,
    batch_index INTEGER NOT NULL,
    batch_size  INTEGER NOT NULL,
    parse_ok    INTEGER NOT NULL,
    latency_ms  REAL NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sample_verdicts (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id               TEXT NOT NULL REFERENCES runs(run_id),
    sample_id            TEXT NOT NULL,
    source               TEXT NOT NULL,
    judge_model          TEXT NOT NULL,
    generator_label      INTEGER,
    verdict_label        INTEGER,
    status               TEXT NOT NULL,
    rationale            TEXT,
    agreed_with_generator INTEGER,
    latency_ms           REAL NOT NULL DEFAULT 0.0,
    created_at           TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS final_labels (
    run_id            TEXT NOT NULL REFERENCES runs(run_id),
    sample_id         TEXT NOT NULL,
    generator_label   INTEGER NOT NULL,
    final_label       INTEGER NOT NULL,
    confidence        REAL,
    needs_review      INTEGER NOT NULL DEFAULT 0,
    disagreement_type TEXT,
    PRIMARY KEY (run_id, sample_id)
);
CREATE INDEX IF NOT EXISTS idx_sv_run_judge ON sample_verdicts(run_id, judge_model);
CREATE INDEX IF NOT EXISTS idx_jb_run_judge ON judge_batches(run_id, judge_model);
-- WHY: idempotent verdict writes - resumed runs re-validate at most the in-flight
-- batch, and INSERT OR REPLACE against this index keeps judge stats double-count-free.
CREATE UNIQUE INDEX IF NOT EXISTS idx_sv_unique ON sample_verdicts(run_id, sample_id, judge_model);
-- Per-judge scoring: success rate, generator agreement, consensus agreement, latency.
CREATE VIEW IF NOT EXISTS judge_performance AS
SELECT
    v.run_id,
    v.judge_model,
    COUNT(*)                                                          AS total_verdicts,
    SUM(v.status = 'ok')                                              AS ok_votes,
    ROUND(1.0 * SUM(v.status = 'ok') / COUNT(*), 4)                   AS success_rate,
    ROUND(AVG(v.agreed_with_generator), 4)                            AS generator_agreement,
    ROUND(AVG(CASE WHEN f.final_label IS NOT NULL AND v.verdict_label IS NOT NULL
                   THEN (v.verdict_label = f.final_label) END), 4)    AS consensus_agreement,
    ROUND(AVG(v.latency_ms), 1)                                       AS avg_latency_ms
FROM sample_verdicts v
LEFT JOIN final_labels f
       ON f.run_id = v.run_id AND f.sample_id = v.sample_id
GROUP BY v.run_id, v.judge_model;
