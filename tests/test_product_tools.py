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


def test_leakage_layers_behavioral_quarantine_and_text_review_without_conflating_them():
    report = audit_leakage(examples()["leakage"])
    assert report["schema_version"] == 2
    assert report["tier_counts"] == {"behavioral": 1, "exact": 1}
    assert {row["item_id"] for row in report["quarantined"]} == {"exact-row", "behavioral-row"}
    assert [row["item_id"] for row in report["review_queue"]] == ["review-row"]
    assert {row["item_id"] for row in report["clean_exam"]} == {"review-row", "clean-row"}
    assert report["analysis_tiers"]["behavioral_fingerprint"]["observations"] == 75
    assert report["analysis_tiers"]["text_similarity"]["action"] == "human_review_only"


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


def test_gate_explains_the_exact_additional_evidence_path():
    report = gate_results({"baseline": {"a": False}, "candidate": {"a": True}})
    assert report["verdict"] == "HOLD"
    assert report["paired_evidence"]["additional_clean_gains_needed"] == 5
    assert "Add 5 clean paired gain" in report["next_action"]
    assert [check["check"] for check in report["decision_path"]] == [
        "regression_budget", "retained_probe", "minimum_gains", "paired_exact_test",
    ]


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
    assert report["readiness"]["index"] == 62.5
    assert report["system_ladder"][-1]["system"] == "large"
    assert report["action_queue"][0]["action"] == "review_or_retire"
    assert {row["recommended_action"] for row in report["items_detail"]} >= {"retain", "review_or_retire"}


def test_safepatch_applies_targeted_edit_and_rejects_protected_removal():
    accepted = safe_patch(examples()["safepatch"])
    assert accepted["accepted"] is True
    assert "Release 4 ships July 12, 2026." in accepted["updated_document"]
    assert "The draft is concise." in accepted["updated_document"]
    assert accepted["diff_stats"]["sections_changed"] == 1
    assert accepted["diff_stats"]["sections_untouched"] == 1
    assert accepted["section_receipts"][0]["protected_tokens_removed"] == []
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
    assert report["selected_by_relevance"] != report["selected_by_salience"]
    assert report["budget_comparison"]["attention_waste_avoided_tokens"] == 8
    assert report["ranking"][0]["selected_by_relevance"] is True


def test_agent_replay_reconstructs_controls_and_rewinds():
    report = replay_trace(examples()["replay"])
    assert report["controls"] == {"ANSWER": 1, "CHECK": 2, "LOAD": 1, "SAVE": 1}
    assert report["checkpoints"] == [{"name": "first_try", "step": 1}]
    assert report["rewinds"][0]["target_step"] == 1
    assert "tree hypothesis failed" in report["notes"]
    assert [branch["id"] for branch in report["branches"]] == [0, 1]
    assert report["critical_path"] == [0, 1]
    assert report["verifier_summary"] == {"accepts": 1, "rejects": 1, "accept_rate": 0.5}


def test_counterexample_hunter_returns_exact_witness_under_public_budget():
    report = hunt_counterexample(examples()["counterexample"])
    assert report["status"] == "FALSIFIED"
    assert report["find"]["greedy_colors"] > report["find"]["chromatic_number"]
    assert report["certificate_sha256"]
    assert len(report["witness"]["nodes"]) == report["find"]["n"]
    assert report["witness"]["gap"] == 1
    assert all(report["witness"]["checks"].values())


def test_public_budget_limits_fail_closed():
    with pytest.raises(ProductInputError, match="1-6 restarts"):
        hunt_counterexample({"expression": "is_tree", "restarts": 100, "steps": 100})
