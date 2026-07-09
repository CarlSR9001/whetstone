from __future__ import annotations

import json

from bcv.examiner import ExamItem, ExaminerBank
from bcv.metabolism import metabolism_report, metabolism_summary, write_metabolism_report


def _item(item_id: str) -> ExamItem:
    return ExamItem(
        item_id=item_id, domain="code", kind="code", payload={}, oracle="test",
        source="test", horizon="test", lineage=[],
    )


def test_lifecycle_events_produce_a_safe_supply_history(tmp_path):
    bank = ExaminerBank(tmp_path)
    bank.add(_item("retired"))
    assert bank.promote("retired")
    bank.retire("retired")
    bank.add(_item("burned"))
    assert bank.promote("burned")
    bank.burn("burned", provider="test", reason="exposed")
    bank.save()

    report = metabolism_report(tmp_path)
    assert report["totals"] == {"burned": 1, "minted": 2, "promoted": 2, "retired": 1}
    assert report["current_promoted_supply"] == 0
    assert report["consumed_from_supply"] == 2
    assert report["mint_to_consumption_ratio"] == 1.0
    assert report["promotion_to_consumption_ratio"] == 1.0
    assert report["sustainability"] == "break-even: every promoted item has been consumed"
    summary = metabolism_summary(tmp_path)
    assert "supply_series" not in summary
    assert summary["current_promoted_supply"] == 0

    json_path, html_path = write_metabolism_report(tmp_path, tmp_path / "report")
    assert json.loads(json_path.read_text(encoding="utf-8"))["events_total"] == 6
    assert "Promoted supply over lifecycle events" in html_path.read_text(encoding="utf-8")
