from __future__ import annotations

import json

from bcv import cli
from bcv.examiner import ExaminerBank


def test_sweep_retires_saturated_items(tmp_path, capsys):
    bank_root = tmp_path / "bank"
    assert cli.main(["--root", str(bank_root), "mint", "--domain", "code", "--max-items", "2"]) == 0
    capsys.readouterr()

    # two systems pass the same single item; the other stays undiscriminating
    bank = ExaminerBank(bank_root)
    items = bank.promoted_items()
    saturated = {items[0].item_id: True, items[1].item_id: False}
    bank.record_grades("system_a", saturated)
    bank.record_grades("system_b", saturated)
    bank.save()

    # first sweep marks staleness, second retires
    assert cli.main(["--root", str(bank_root), "sweep"]) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["retired_this_sweep"] == []
    assert cli.main(["--root", str(bank_root), "sweep"]) == 0
    second = json.loads(capsys.readouterr().out)
    assert second["retired_this_sweep"] == [items[0].item_id]
    assert second["trainable_rows_available"] == 1
    assert second["promoted_remaining"] == 1


def test_redteam_command_reports_no_escapes(tmp_path, capsys):
    assert cli.main(["redteam", "--out", str(tmp_path / "redteam")]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["paraphrase_attack"]["promotion_allowed"] is False
    assert report["inflation_attack"]["caught"] is True
