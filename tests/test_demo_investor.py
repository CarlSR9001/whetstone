from __future__ import annotations

import json
from pathlib import Path

from bcv.demo_investor import (
    DemoConfig,
    LEAKED_ORIGINALS,
    build_training_buffer,
    run_demo,
)
from bcv.examiner import training_originals


def test_training_buffer_rows_parse_as_leakage_set(tmp_path):
    path = build_training_buffer(tmp_path / "buffer.jsonl")
    originals = training_originals([path])
    assert set(LEAKED_ORIGINALS) <= originals


def test_demo_end_to_end(tmp_path):
    report = run_demo(
        DemoConfig(
            root=tmp_path / "demo",
            max_repair_items=6,
            max_game_items=2,
            quiet=True,
        )
    )
    # The leakage quarantine fired for real and nothing quarantined was promoted.
    assert report["quarantined"] >= 1
    assert report["promoted"] == report["minted"] - report["quarantined"]
    assert report["total_items"] == report["promoted"]

    # The gate separated the two systems, and the decision follows the rule.
    assert report["candidate_score"] > report["base_score"]
    assert report["gains"] >= 1
    expected = "PASS" if report["gains"] and not report["regressions"] else "BLOCK"
    assert report["decision"] == expected

    # The ledger and bank buckets are real files.
    ledger_path = tmp_path / "demo" / "ledger.jsonl"
    lines = [json.loads(l) for l in ledger_path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == report["ledger_events"]
    assert any(row["event"] == "promotion_decision" for row in lines)
    bank_dir = tmp_path / "demo" / "bank"
    assert (bank_dir / "private_promotion_exam.jsonl").exists()
    assert (bank_dir / "quarantined.jsonl").exists()

    # Retired items flow downward into trainable rows, never back up.
    assert report["trainable_rows"] == report["retired"]
