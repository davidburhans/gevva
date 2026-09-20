#!/usr/bin/env python3
"""tests/test_night_stats.py - Tests for the pre-registered gate statistics.

Run: uv run python tests/test_night_stats.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from gate_decision import ece_from_items, evaluate_gate, mcnemar_p  # noqa: E402


def test_mcnemar_balanced_discordance_is_not_significant():
    assert mcnemar_p(0, 0) == 1.0
    # Continuity correction leaves chi2=0.05 for perfectly balanced discordance:
    # far from significance, but not exactly 1.0.
    assert mcnemar_p(10, 10) > 0.5


def test_mcnemar_extreme_imbalance_is_significant():
    p = mcnemar_p(0, 20)  # chi2 = 361/20 = 18.05 -> p ~ 2.1e-5
    assert p < 0.001, f"0/20 discordance must be significant, got {p}"


def test_mcnemar_known_value_range():
    p = mcnemar_p(1, 9)  # chi2 = 4.9
    assert 0.02 <= p <= 0.035, f"mcnemar_p(1,9)={p} outside known range"


def test_ece_perfectly_calibrated_is_near_zero():
    # 30 items: confidence 0.5, 50% correct in one bin range -> near-zero contribution.
    items = [{"confidence": 0.5, "correct": i % 2 == 0} for i in range(30)]
    assert ece_from_items(items) < 0.01


def test_ece_overconfident_is_large():
    items = [{"confidence": 0.95, "correct": i % 2 == 0} for i in range(30)]
    assert ece_from_items(items) > 0.30


def test_evaluate_gate_passes_only_on_rule():
    import tempfile
    import os
    import json as _json
    with tempfile.TemporaryDirectory() as tmp:
        # A: 85% acc at 0.85 confidence (perfectly calibrated) -> ECE 0
        a = [{"id": f"i{i}", "correct": i >= 30, "confidence": 0.85} for i in range(200)]
        # B: 97.5% acc at 0.975 confidence (perfectly calibrated) -> ECE ~0, higher acc
        b = [{"id": f"i{i}", "correct": i >= 5, "confidence": 0.975} for i in range(200)]
        for arm, rows in (("A", a), ("B", b)):
            d = os.path.join(tmp, arm)
            os.makedirs(d)
            with open(os.path.join(d, "test_items.jsonl"), "w") as f:
                for r in rows:
                    f.write(_json.dumps(r) + "\n")
        verdict = evaluate_gate(os.path.join(tmp, "A"), os.path.join(tmp, "B"))
        assert verdict["passed"] is True, verdict
        assert verdict["n_paired"] == 200
        # Break rule 3 (calibration regression): same accuracy but overconfident 0.99
        b2 = [dict(r, confidence=0.99) for r in b]
        d = os.path.join(tmp, "B2")
        os.makedirs(d)
        with open(os.path.join(d, "test_items.jsonl"), "w") as f:
            for r in b2:
                f.write(_json.dumps(r) + "\n")
        verdict2 = evaluate_gate(os.path.join(tmp, "A"), os.path.join(tmp, "B2"))
        assert verdict2["passed"] is False, "calibration regression must fail the gate"


def test_export_validated_synthetic_excludes_failures():
    """Only ok-status verdicts may become synthetic training rows (failures -> review queue)."""
    import json as _json
    import tempfile
    from pathlib import Path
    from build_arms import export_validated_synthetic
    from validation_metrics_db import ValidationMetricsDB

    with tempfile.TemporaryDirectory() as tmp:
        db = ValidationMetricsDB(str(Path(tmp) / "m.db"))
        raw = Path(tmp) / "raw.jsonl"
        rows = [{"id": f"s{i}", "premise": f"p{i}", "hypothesis": f"h{i}", "label": 1}
                for i in range(3)]
        raw.write_text("\n".join(_json.dumps(r) for r in rows), encoding="utf-8")
        db.record_verdicts([
            ("r1", "s0", "src", "qwen-j", 1, 0, "ok", "r", 0, 10.0, "now"),
            ("r1", "s1", "src", "qwen-j", 1, 2, "ok", "r", 0, 10.0, "now"),
            ("r1", "s2", "src", "qwen-j", 1, 1, "parse_error", "r", None, 10.0, "now"),
        ])
        out_train = Path(tmp) / "staged" / "syn_train.jsonl"
        out_val = Path(tmp) / "staged" / "syn_val.jsonl"
        stats = export_validated_synthetic(str(raw), str(db.db_path), str(out_train),
                                           str(out_val), run_id="r1", judge_model="qwen-j")
        assert stats["validated_rows"] == 2 and stats["excluded_rows"] == 1
        assert stats["val_holdback_rows"] == 0 and stats["train_rows"] == 2
        kept = [_json.loads(l) for l in open(out_train, encoding="utf-8")]
        assert {(r["id"], r["label"]) for r in kept} == {("s0", 0), ("s1", 2)}
        assert not out_val.exists() or out_val.stat().st_size == 0
        db.close()


TESTS = [test_mcnemar_balanced_discordance_is_not_significant,
         test_mcnemar_extreme_imbalance_is_significant,
         test_mcnemar_known_value_range,
         test_ece_perfectly_calibrated_is_near_zero,
         test_ece_overconfident_is_large,
         test_evaluate_gate_passes_only_on_rule,
         test_export_validated_synthetic_excludes_failures]


def main() -> int:
    failures = 0
    for test in TESTS:
        try:
            test()
            print(f"PASS  {test.__name__}")
        except Exception:
            import traceback
            failures += 1
            print(f"FAIL  {test.__name__}")
            traceback.print_exc()
    print(f"\n{len(TESTS) - failures}/{len(TESTS)} tests passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
