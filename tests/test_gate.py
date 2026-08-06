from __future__ import annotations

import pytest

from bcv.examiner import ExamItem, ExaminerBank
from bcv.gate import (
    GatePolicy, build_gate_report, exact_mcnemar_p_value, latest_grade_event_results, power_statement, write_gate_report,
)


def _bank(tmp_path, count: int = 6) -> ExaminerBank:
    bank = ExaminerBank(tmp_path)
    for index in range(count):
        bank.add(ExamItem(
            item_id=f"i{index}", domain="coloring" if index % 2 else "mis", kind="repair",
            payload={}, oracle="test", source="test", horizon="test", lineage=[], status="promoted",
        ))
    return bank


def _retained(base: int = 0, candidate: int = 6, items: int = 8) -> dict:
    return {"eval": {"base_verified": base, "adapter_verified": candidate, "eval_examples": items}}


def test_exact_mcnemar_is_two_sided():
    assert exact_mcnemar_p_value(4, 0) == 0.125
    assert exact_mcnemar_p_value(6, 0) == 0.03125
    assert exact_mcnemar_p_value(0, 0) == 1.0


def test_power_statement_declares_the_current_bank_resolution():
    resolution = power_statement(4, 1, 0.05)
    assert resolution["minimum_clean_discordant_wins"] == 6
    assert resolution["observed_discordant_items"] == 5
    assert resolution["best_case_p_if_observed_discordants_were_all_clean"] == 0.0625
    assert resolution["additional_clean_discriminators_needed_if_current_outcomes_were_clean"] == 1


def test_gate_passes_only_with_paired_confidence_and_retained_probe(tmp_path):
    bank = _bank(tmp_path)
    baseline = {item_id: False for item_id in bank.items}
    candidate = {item_id: True for item_id in bank.items}
    report = build_gate_report(
        bank, "base", "candidate", baseline, candidate, _retained(),
        grade_runs={"baseline": {"adapter": "fixture"}, "candidate": {"adapter": "fixture", "seed": 7}},
    )
    assert report["verdict"] == "PASS"
    assert report["paired_evidence"]["exact_mcnemar_two_sided_p"] == 0.03125
    json_path, html_path = write_gate_report(report, tmp_path / "report")
    assert json_path.exists() and html_path.exists()
    assert "Private bank SHA-256" in html_path.read_text(encoding="utf-8")
    assert report["grade_runs"]["candidate"]["seed"] == 7
    assert "Grading-run provenance" in html_path.read_text(encoding="utf-8")


def test_gate_holds_weak_gain_and_blocks_regression(tmp_path):
    bank = _bank(tmp_path, count=3)
    baseline = {item_id: False for item_id in bank.items}
    candidate = {item_id: True for item_id in bank.items}
    weak = build_gate_report(bank, "base", "candidate", baseline, candidate, _retained())
    assert weak["verdict"] == "HOLD"
    candidate["i0"] = False
    baseline["i0"] = True
    blocked = build_gate_report(bank, "base", "candidate", baseline, candidate, _retained())
    assert blocked["verdict"] == "BLOCK"


def test_gate_rejects_partial_cohorts_instead_of_silently_intersecting_them(tmp_path):
    bank = _bank(tmp_path, count=3)
    baseline = {"i0": False, "i1": False}
    candidate = {"i0": True}
    with pytest.raises(ValueError, match="same item ids"):
        build_gate_report(bank, "base", "candidate", baseline, candidate, policy=GatePolicy(require_retained_probe=False))


def test_burned_items_never_return_to_promotion_or_training(tmp_path):
    bank = _bank(tmp_path, count=1)
    bank.burn("i0", provider="example-api", reason="external grading")
    bank.save()
    reloaded = ExaminerBank(tmp_path)
    assert reloaded.items["i0"].status == "burned"
    assert not reloaded.promoted_items()
    assert not reloaded.trainable_rows()
    assert not reloaded.promote("i0")


def test_latest_grade_event_is_used_instead_of_aggregate_history(tmp_path):
    events = tmp_path / "grade_events.jsonl"
    events.write_text(
        '{"event":"grade","system":"base","results":{"old":false}}\n'
        '{"event":"grade","system":"base","results":{"new":true}}\n',
        encoding="utf-8",
    )
    assert latest_grade_event_results(events, "base") == {"new": True}


def test_reliability_aware_gate_blocks_stable_regression_but_holds_unknown_one(tmp_path):
    bank = _bank(tmp_path, count=6)
    baseline = {item_id: False for item_id in bank.items}
    candidate = {item_id: True for item_id in bank.items}
    baseline["i0"] = True
    candidate["i0"] = False
    bank.items["i0"].graded["repeat"] = {"pass": 10, "fail": 0}
    policy = GatePolicy(require_retained_probe=False, regression_policy="reliability_aware")
    stable = build_gate_report(bank, "base", "candidate", baseline, candidate, policy=policy)
    assert stable["verdict"] == "BLOCK"
    bank.items["i0"].graded = {}
    unknown = build_gate_report(bank, "base", "candidate", baseline, candidate, policy=policy)
    assert unknown["verdict"] == "HOLD"


def test_reliability_aware_pass_reason_describes_allowed_noisy_regression(tmp_path):
    bank = _bank(tmp_path, count=11)
    baseline = {item_id: False for item_id in bank.items}
    candidate = {item_id: True for item_id in bank.items}
    baseline["i0"] = True
    candidate["i0"] = False
    bank.items["i0"].graded["repeat"] = {"pass": 5, "fail": 5}
    policy = GatePolicy(require_retained_probe=False, regression_policy="reliability_aware")

    report = build_gate_report(bank, "base", "candidate", baseline, candidate, policy=policy)

    assert report["verdict"] == "PASS"
    assert report["paired_evidence"]["exact_mcnemar_two_sided_p"] == 0.01171875
    reason = report["reasons"][0]
    assert "1 historically noisy regression(s) stayed within the policy budget of 1" in reason
    assert "retained probe was not required" in reason
    assert "no regressions" not in reason
    assert "retained probe held" not in reason
