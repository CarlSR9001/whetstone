from __future__ import annotations

import json

from bcv import demo_stage


def test_story_compiles_from_committed_receipts():
    story = demo_stage.compile_story()
    d = story["decision"]
    # The three-card spine must be populated from real receipts.
    assert d["base"]["total"] == "22/48"
    assert d["gen2"]["verdict"] == "BLOCK"
    assert d["gen2"]["regressions"] == 2
    assert d["gen3"]["verdict"] == "PASS"
    assert d["gen3"]["regressions"] == 0
    assert d["gen3"]["p"] == 0.03125
    assert d["gen3"]["repeats"]["scores"] == [28, 28, 28]
    # The stress wall must have all four exhibits.
    assert story["ladder"]["rungs"]
    assert story["attack"]["rows"]
    assert story["incident"]["audits"]
    assert story["bakeoff"]["gate_strict"]["verdict"] == "BLOCK"
    assert len(story["receipts_on_disk"]) >= 5


def test_story_never_leaks_item_content():
    """The stage is a public-facing surface: no exam prompts or item ids."""
    blob = json.dumps(demo_stage.compile_story())
    assert "item_id" not in blob
    assert "fen" not in blob
    assert '"moves"' not in blob
    assert "repair_expression" not in blob


def test_live_run_executes_real_engine(tmp_path, monkeypatch):
    monkeypatch.chdir(demo_stage.RESULTS.parent if demo_stage.RESULTS.exists() else tmp_path)
    report = demo_stage.run_live()
    # It ran the real demo: quarantine fired, a decision came out, ledger written.
    assert report["quarantined"] >= 1
    assert report["decision"] in ("PASS", "HOLD", "BLOCK")
    assert report["total_items"] >= 1
