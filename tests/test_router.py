from __future__ import annotations

from bcv.router import compile_task_contract, route_task, run_router_probe


def test_router_answers_simple_low_risk_task_directly():
    decision = route_task(compile_task_contract("Summarize this short paragraph."))

    assert decision.actions == ("answer_directly",)


def test_router_sends_document_work_to_artifact_parser_and_branch():
    decision = route_task(
        compile_task_contract("Edit contract.md and preserve all payment numbers.")
    )

    assert "use_artifact_parser" in decision.actions
    assert "branch" in decision.actions


def test_router_requires_evidence_for_research_claims():
    decision = route_task(
        compile_task_contract("Research this claim and cite the latest evidence.")
    )

    assert "retrieve_evidence" in decision.actions
    assert "run_verifier" in decision.actions


def test_router_requires_approval_for_irreversible_actions():
    decision = route_task(
        compile_task_contract("Delete the production branch after checking it.")
    )

    assert "require_user_approval" in decision.actions


def test_router_probe_covers_all_current_action_classes():
    results = run_router_probe()
    flattened = {action for decision in results.values() for action in decision.actions}

    assert "answer_directly" in flattened
    assert "use_artifact_parser" in flattened
    assert "retrieve_evidence" in flattened
    assert "require_user_approval" in flattened
