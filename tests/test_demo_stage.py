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


def test_story_never_leaks_item_content_or_local_paths():
    """The stage is a public-facing surface: no exam content, no personal paths."""
    blob = json.dumps(demo_stage.compile_story())
    assert "item_id" not in blob
    assert "fen" not in blob
    assert '"moves"' not in blob
    assert "repair_expression" not in blob
    assert "Users" not in blob and "AI Arch" not in blob


def test_story_carries_live_reference_fallback():
    """The button must never dead-end: a committed real run is always available."""
    ref = demo_stage.compile_story()["live_reference"]
    assert ref.get("decision") in ("PASS", "HOLD", "BLOCK")
    assert ref.get("cached") is True
    assert ref.get("quarantined", 0) >= 1


def test_live_run_streams_real_phase_events():
    """The step rail is honest: events arrive in pipeline order with the
    engine's own numbers, and the terminal event carries the full report."""
    import time

    runner = demo_stage._LiveRun()
    assert runner.start() == {"started": True}
    assert runner.start() == {"error": "a live run is already in progress"}
    deadline = time.time() + 90
    while runner.snapshot()["running"] and time.time() < deadline:
        time.sleep(0.5)
    events = runner.snapshot()["events"]
    phases = [event["phase"] for event in events]
    assert phases[-1] in ("done", "fallback")
    if phases[-1] == "done":
        # Real pipeline order, real numbers.
        assert phases.index("buffer") < phases.index("mint_start") < phases.index("mint")
        assert phases.index("quarantine") < phases.index("grade_base") < phases.index("grade_candidate")
        assert phases.index("decision") < phases.index("retire")
        quarantine = next(e for e in events if e["phase"] == "quarantine")
        assert quarantine["data"]["quarantined"] >= 1
        report = events[-1]["data"]
        assert report["decision"] in ("PASS", "HOLD", "BLOCK")
        assert report["cached"] is False
