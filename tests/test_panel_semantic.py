from __future__ import annotations

import json
from pathlib import Path

from bcv.panel import calibration_admissible
from bcv.panel_semantic import semantic_support_panel


class FakeJudge:
    model = "fake"

    def judge(self, case, answer):
        return "fail" if "unsupported" in answer else "pass"


def test_semantic_panel_vetoes_local_judge_failure_without_weakening_other_checks():
    panel = semantic_support_panel(FakeJudge())
    case = {"source": "Returns are allowed after inspection.", "question": "Can I return?", "forbidden": []}
    assert panel.grade(case, "Returns are allowed after inspection.").passed is True
    report = panel.grade(case, "Returns are allowed after inspection, unsupported promise.")
    assert report.passed is False
    assert "semantic_source_verdict" in report.failed_checks


def test_semantic_baseline_is_explicitly_not_admissible_with_one_false_accept():
    path = Path(__file__).resolve().parents[1] / "results" / "support_hard_semantic_qwen3_baseline.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["calibration"]["agreement"] > 0.9
    assert report["calibration"]["false_accepts"] == 1
    assert calibration_admissible(report["calibration"]) is False
