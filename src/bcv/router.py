from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


RouteAction = Literal[
    "answer_directly",
    "ask_for_missing_input",
    "retrieve_evidence",
    "branch",
    "use_artifact_parser",
    "run_verifier",
    "require_user_approval",
]


@dataclass(frozen=True)
class TaskContract:
    objective: str
    constraints: tuple[str, ...] = ()
    artifact_refs: tuple[str, ...] = ()
    factual_claims_required: bool = False
    irreversible_actions: tuple[str, ...] = ()
    ambiguity_score: float = 0.0
    cost_of_error: float = 0.0
    missing_inputs: tuple[str, ...] = ()
    needs_long_context: bool = False


@dataclass(frozen=True)
class RouteDecision:
    actions: tuple[RouteAction, ...]
    reasons: tuple[str, ...]
    estimated_cost_units: int


def compile_task_contract(request: str) -> TaskContract:
    lowered = request.lower()
    artifact_refs = tuple(re.findall(r"[\w .#-]+\.(?:md|docx|pdf|xlsx|csv|py|ts|js)", request))
    factual_claims_required = any(
        marker in lowered
        for marker in ("cite", "source", "latest", "research", "verify", "evidence", "claim")
    )
    irreversible_actions = tuple(
        action
        for action in ("send", "delete", "publish", "merge", "push", "commit", "buy")
        if re.search(rf"\b{action}\b", lowered)
    )
    ambiguity_score = 0.7 if any(word in lowered for word in ("maybe", "somehow", "figure out", "whatever")) else 0.2
    cost_of_error = 0.8 if any(word in lowered for word in ("contract", "legal", "medical", "money", "production")) else 0.3
    missing_inputs = ("artifact",) if "edit" in lowered and not artifact_refs else ()
    needs_long_context = any(word in lowered for word in ("long", "many", "entire", "full document", "repo"))

    return TaskContract(
        objective=request.strip(),
        artifact_refs=artifact_refs,
        factual_claims_required=factual_claims_required,
        irreversible_actions=irreversible_actions,
        ambiguity_score=ambiguity_score,
        cost_of_error=cost_of_error,
        missing_inputs=missing_inputs,
        needs_long_context=needs_long_context,
    )


def route_task(contract: TaskContract) -> RouteDecision:
    actions: list[RouteAction] = []
    reasons: list[str] = []
    cost = 1

    if contract.missing_inputs:
        actions.append("ask_for_missing_input")
        reasons.append(f"missing inputs: {', '.join(contract.missing_inputs)}")
        return RouteDecision(tuple(actions), tuple(reasons), cost)

    if contract.irreversible_actions:
        actions.append("require_user_approval")
        reasons.append(f"irreversible actions requested: {', '.join(contract.irreversible_actions)}")
        cost += 1

    if contract.artifact_refs:
        actions.append("use_artifact_parser")
        reasons.append("artifact references present")
        cost += 2

    if contract.factual_claims_required:
        actions.append("retrieve_evidence")
        actions.append("run_verifier")
        reasons.append("factual or source-grounded claims required")
        cost += 3

    if contract.ambiguity_score >= 0.6 or contract.cost_of_error >= 0.7 or contract.needs_long_context:
        actions.append("branch")
        reasons.append("ambiguity, error cost, or long context justifies branch isolation")
        cost += 2

    if not actions:
        actions.append("answer_directly")
        reasons.append("low ambiguity and low cost of error")

    deduped_actions = tuple(dict.fromkeys(actions))
    return RouteDecision(deduped_actions, tuple(reasons), cost)


def run_router_probe() -> dict[str, RouteDecision]:
    requests = {
        "simple": "Summarize this short paragraph in one sentence.",
        "document": "Edit agreement.md but preserve all numbers and citations.",
        "research": "Research the latest evidence and cite sources for this claim.",
        "dangerous": "Delete the production branch after you verify the repo.",
    }
    return {
        name: route_task(compile_task_contract(request))
        for name, request in requests.items()
    }

