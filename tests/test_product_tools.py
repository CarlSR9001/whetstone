from __future__ import annotations

import hashlib

import pytest

from bcv.product_tools import (
    ProductInputError,
    audit_leakage,
    bank_health,
    examples,
    gate_results,
    hunt_counterexample,
    inspect_promotion,
    memory_relevance,
    replay_trace,
    safe_patch,
)


def test_leakage_audit_matches_declared_id_and_exact_content():
    report = audit_leakage({
        "exam": [
            {"item_id": "a", "prompt": "one"},
            {"item_id": "b", "prompt": "two"},
            {"item_id": "c", "prompt": "three"},
        ],
        "exposure": [
            {"id": "a", "source": "ids.jsonl:1"},
            {"content_hash": hashlib.sha256(b"two").hexdigest(), "source": "hashes.jsonl:2"},
        ],
    })
    assert report["quarantined_items"] == 2
    assert [row["item_id"] for row in report["clean_exam"]] == ["c"]
    assert report["exact_identity_only"] is True
    assert "semantic" in report["claim_boundary"]


def test_gate_pass_hold_and_block_are_paired_not_aggregate():
    sample = examples()["gate"]
    passed = gate_results(sample)
    assert passed["verdict"] == "PASS"
    assert passed["paired_evidence"]["gains"] == 6
    assert passed["paired_evidence"]["exact_mcnemar_two_sided_p"] == 0.03125

    held = gate_results({"baseline": {"a": False}, "candidate": {"a": True}})
    assert held["verdict"] == "HOLD"

    baseline = {"a": True, "b": False, "c": False}
    candidate = {"a": False, "b": True, "c": True}
    blocked = gate_results({"baseline": baseline, "candidate": candidate})
    assert sum(candidate.values()) > sum(baseline.values())
    assert blocked["paired_evidence"]["gains"] == 2
    assert blocked["paired_evidence"]["regressions"] == 1
    assert blocked["verdict"] == "BLOCK"


def test_gate_refuses_silent_cohort_intersection():
    with pytest.raises(ProductInputError, match="identical item cohort"):
        gate_results({"baseline": {"a": False}, "candidate": {"a": True, "b": True}})


def test_inspector_quarantines_before_gate_and_holds_incomplete_cohort():
    sample = examples()["inspector"]
    report = inspect_promotion(sample)
    repeated = inspect_promotion(sample)
    assert report["audit"]["quarantined_items"] == 1
    assert report["cohort"]["complete"] is True
    assert report["gate"]["verdict"] == "PASS"
    assert report["gate"]["paired_evidence"]["items"] == 7
    assert repeated["receipt_sha256"] == report["receipt_sha256"]

    incomplete = {**sample, "candidate": {"item-1": True}}
    held = inspect_promotion(incomplete)
    assert held["cohort"]["complete"] is False
    assert held["gate"]["verdict"] == "HOLD"
    assert held["gate"]["paired_evidence"]["items"] == 0


def test_bank_health_finds_frontier_saturation_flakiness_and_gaps():
    report = bank_health(examples()["health"])
    classes = {row["item_id"]: row["classification"] for row in report["items_detail"]}
    assert classes == {
        "frontier": "discriminating",
        "hard": "too_hard",
        "noisy": "flaky",
        "stable": "saturated",
    }
    assert report["retirement_candidates"] == ["stable"]
    assert report["frontier_gaps"] == ["support"]


def test_safepatch_applies_targeted_edit_and_rejects_protected_removal():
    accepted = safe_patch(examples()["safepatch"])
    assert accepted["accepted"] is True
    assert "Release 4 ships July 12, 2026." in accepted["updated_document"]
    assert "The draft is concise." in accepted["updated_document"]
    with pytest.raises(ProductInputError, match="protected token removed"):
        safe_patch({
            "document": "# Terms\nPay Alice 17 credits.\n",
            "operations": [{"target_heading": "Terms", "find": "Alice 17", "replace": "Bob"}],
        })


def test_memory_debugger_separates_relevance_from_salience():
    report = memory_relevance(examples()["memory"])
    assert report["ranking"][0]["content"].startswith("STATE: Ivo")
    assert report["ranking"][0]["relevance"] > report["ranking"][-1]["relevance"]
    assert len(report["shiny_traps"]) == 1
    assert len(report["boring_but_decisive"]) == 1
    assert report["selected_by_relevance"]


def test_agent_replay_reconstructs_controls_and_rewinds():
    report = replay_trace(examples()["replay"])
    assert report["controls"] == {"ANSWER": 1, "CHECK": 2, "LOAD": 1, "SAVE": 1}
    assert report["checkpoints"] == [{"name": "first_try", "step": 1}]
    assert report["rewinds"][0]["target_step"] == 1
    assert "tree hypothesis failed" in report["notes"]


def test_counterexample_hunter_returns_exact_witness_under_public_budget():
    report = hunt_counterexample(examples()["counterexample"])
    assert report["status"] == "FALSIFIED"
    assert report["find"]["greedy_colors"] > report["find"]["chromatic_number"]
    assert report["certificate_sha256"]


def test_public_budget_limits_fail_closed():
    with pytest.raises(ProductInputError, match="1-6 restarts"):
        hunt_counterexample({"expression": "is_tree", "restarts": 100, "steps": 100})
