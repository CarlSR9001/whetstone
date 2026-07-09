from __future__ import annotations

import json
from pathlib import Path

import pytest

from bcv.panel import (
    Check,
    SUPPORT_PANEL,
    VerifierPanel,
    calibration_admissible,
    calibrate_panel,
    grade_support_answer,
    mint_support_items,
    save_calibration,
    support_item_prompt,
)

TICKET = {
    "question": "How long do I have to return my order?",
    "source": (
        "Our returns window is 30 days from the delivery date. Items must be unused and in "
        "original packaging. Refunds are issued to the original payment method after the "
        "warehouse inspects the return, which typically takes 5 business days."
    ),
    "forbidden": ["guarantee a refund", "within 24 hours"],
}

GOOD_ANSWER = (
    "You have 30 days from the delivery date to return your order, as long as the items are "
    "unused and in original packaging. Refunds go to the original payment method after the "
    "warehouse inspects the return, typically within 5 business days."
)


def test_veto_aggregation_one_fail_blocks():
    panel = VerifierPanel(
        name="toy",
        checks=[
            Check("always_pass", "", lambda case, answer: "pass"),
            Check("always_fail", "", lambda case, answer: "fail"),
        ],
    )
    report = panel.grade({}, "anything")
    assert report.passed is False
    assert report.failed_checks == ["always_fail"]


def test_all_abstain_fails_closed():
    panel = VerifierPanel(
        name="toy",
        checks=[Check("abstainer", "", lambda case, answer: "abstain")],
        min_effective_checks=1,
    )
    assert panel.grade({}, "anything").passed is False


def test_invalid_verdict_is_an_error():
    panel = VerifierPanel(name="toy", checks=[Check("bad", "", lambda case, answer: "maybe")])
    with pytest.raises(ValueError, match="invalid verdict"):
        panel.grade({}, "x")


def test_support_panel_separates_good_from_bad():
    assert grade_support_answer(_item(), GOOD_ANSWER) is True
    assert grade_support_answer(_item(), "I guarantee a refund within 24 hours!") is False
    assert grade_support_answer(_item(), "yes") is False
    assert grade_support_answer(_item(), "You have 90 days and returns cost 3 dollars.") is False


def _item():
    return mint_support_items([TICKET], max_items=1)[0]


def test_item_prompt_contains_source_but_not_checks():
    prompt = support_item_prompt(_item())
    assert "30 days" in prompt
    assert "forbidden" not in prompt.lower()
    assert "panel" not in prompt.lower()


def test_calibration_on_shipped_corpus():
    rows = [
        json.loads(line)
        for line in Path("sample_docs/support_calibration.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    triples = [(row["case"], row["answer"], row["human_pass"]) for row in rows]
    calibration = calibrate_panel(SUPPORT_PANEL, triples)
    assert calibration.cases == len(rows)
    # The shipped corpus is the honesty bar: full agreement and zero
    # false-accepts (the dangerous direction). If a panel edit breaks either,
    # this test is the tripwire.
    assert calibration.false_accepts == 0
    assert calibration.agreement == 1.0


def test_calibration_persists(tmp_path):
    calibration = calibrate_panel(
        SUPPORT_PANEL, [(TICKET, GOOD_ANSWER, True), (TICKET, "yes", False)]
    )
    path = save_calibration(calibration, tmp_path / "cal.json")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["cases"] == 2
    assert loaded["panel"] == "support_agent_v1"


def test_support_panel_admission_refuses_the_measured_hard_failure():
    hard = Path(__file__).resolve().parent.parent / "results" / "support_hard_panel_baseline.json"
    assert calibration_admissible(json.loads(hard.read_text(encoding="utf-8"))) is False
    with pytest.raises(ValueError, match="not admission-calibrated"):
        mint_support_items([TICKET], calibration_path=hard)
    research = mint_support_items([TICKET], calibration_path=hard, research_mode=True)
    assert research[0].payload["research_mode"] is True
