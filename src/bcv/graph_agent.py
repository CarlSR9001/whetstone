from __future__ import annotations

import argparse
import ast
import json
import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from bcv.discovery import CandidateRule, GraphObservation, RuleResult, enumerate_graphs, evaluate_rule, observe_graph
from bcv.local_model import LocalModelClient, LocalModelError, auto_local_client
from bcv.schema import Event, TestResult
from bcv.store import CognitiveStore


ALLOWED_FEATURES = {
    "n",
    "m",
    "density",
    "max_degree",
    "min_degree",
    "is_connected",
    "is_complete",
    "is_forest",
    "is_tree",
    "is_bipartite",
    "is_triangle_free",
    "max_degree_le_2",
    "has_universal_vertex",
    "has_isolated_vertex",
    "is_regular",
    "num_components",
    "clique_number",
    "girth",
}


@dataclass(frozen=True)
class ProposedRule:
    name: str
    description: str
    expression: str


@dataclass(frozen=True)
class ProposalEvaluation:
    max_n: int
    graphs_checked: int
    backend: str
    model: str
    proposed_rules: tuple[ProposedRule, ...]
    accepted_rules: tuple[RuleResult, ...]
    rejected_rules: tuple[RuleResult, ...]
    invalid_rules: tuple[dict[str, str], ...]
    repair_suggestions: tuple[dict[str, object], ...]


def compile_feature_expression(expression: str):
    tree = ast.parse(expression, mode="eval")
    _validate_expression(tree)

    def predicate(observation) -> bool:
        value = _eval_node(tree.body, observation.features)
        return bool(value)

    return predicate


def load_proposals(path: str | Path) -> tuple[ProposedRule, ...]:
    raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    return parse_proposals(raw)


def parse_proposals(raw: Any) -> tuple[ProposedRule, ...]:
    items = raw.get("rules", raw) if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        raise ValueError("proposal JSON must be a list or an object with a rules list")
    proposals: list[ProposedRule] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"proposal {index} is not an object")
        name = str(item.get("name", f"rule_{index}"))
        description = str(item.get("description", name))
        expression = str(item["expression"])
        proposals.append(
            ProposedRule(
                name=_slug(name),
                description=description,
                expression=expression,
            )
        )
    return tuple(proposals)


def propose_with_model(
    client: LocalModelClient,
    max_rules: int = 6,
    temperature: float = 0.2,
    feedback: str = "",
) -> tuple[ProposedRule, ...]:
    prompt = _proposal_prompt(max_rules, feedback)
    raw = client.generate_json(prompt, temperature=temperature)
    return parse_proposals(raw)


def evaluate_proposals(
    proposals: tuple[ProposedRule, ...],
    max_n: int = 6,
    backend: str = "static",
    model: str = "proposal_file",
    root: str | Path = ".bcv_runs/graph_agent",
) -> ProposalEvaluation:
    observations = list(_observations_for(max_n))
    accepted: list[RuleResult] = []
    rejected: list[RuleResult] = []
    invalid: list[dict[str, str]] = []
    repairs: list[dict[str, object]] = []

    for proposal in proposals:
        try:
            predicate = compile_feature_expression(proposal.expression)
            result = evaluate_rule(
                CandidateRule(
                    proposal.name,
                    f"{proposal.description} Predicate: {proposal.expression}",
                    predicate,
                ),
                observations,
            )
        except (SyntaxError, ValueError, TypeError, KeyError) as exc:
            invalid.append(
                {
                    "name": proposal.name,
                    "expression": proposal.expression,
                    "error": str(exc),
                }
            )
            continue
        if result.promoted:
            accepted.append(result)
        else:
            rejected.append(result)
            repairs.extend(_suggest_repairs(proposal, observations))

    evaluation = ProposalEvaluation(
        max_n=max_n,
        graphs_checked=len(observations),
        backend=backend,
        model=model,
        proposed_rules=proposals,
        accepted_rules=tuple(accepted),
        rejected_rules=tuple(rejected),
        invalid_rules=tuple(invalid),
        repair_suggestions=tuple(repairs),
    )
    _write_evaluation(Path(root), evaluation)
    return evaluation


@lru_cache(maxsize=8)
def _observations_for(max_n: int) -> tuple[GraphObservation, ...]:
    return tuple(observe_graph(graph) for graph in enumerate_graphs(max_n))


def run_model_conjecture_loop(
    max_n: int = 6,
    root: str | Path = ".bcv_runs/graph_agent",
    client: LocalModelClient | None = None,
    max_rules: int = 6,
    feedback: str = "",
) -> ProposalEvaluation:
    client = client or auto_local_client()
    proposals = propose_with_model(client, max_rules=max_rules, feedback=feedback)
    return evaluate_proposals(
        proposals,
        max_n=max_n,
        backend=client.backend,
        model=client.model,
        root=root,
    )


def _validate_expression(tree: ast.AST) -> None:
    allowed_nodes = (
        ast.Expression,
        ast.BoolOp,
        ast.UnaryOp,
        ast.Compare,
        ast.Name,
        ast.Load,
        ast.Constant,
        ast.And,
        ast.Or,
        ast.Not,
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
    )
    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            raise ValueError(f"unsupported expression node: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in ALLOWED_FEATURES:
            raise ValueError(f"unknown feature: {node.id}")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (bool, int, float)):
            raise ValueError("only boolean and numeric constants are allowed")


def _eval_node(node: ast.AST, features: dict[str, bool | int | float]) -> bool | int | float:
    if isinstance(node, ast.Name):
        return features[node.id]
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BoolOp):
        values = [_eval_node(value, features) for value in node.values]
        if isinstance(node.op, ast.And):
            return all(bool(value) for value in values)
        return any(bool(value) for value in values)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return not bool(_eval_node(node.operand, features))
    if isinstance(node, ast.Compare):
        left = _eval_node(node.left, features)
        for operator, comparator_node in zip(node.ops, node.comparators):
            right = _eval_node(comparator_node, features)
            if not _compare(left, operator, right):
                return False
            left = right
        return True
    raise ValueError(f"unsupported expression node: {type(node).__name__}")


def _compare(left: Any, operator: ast.cmpop, right: Any) -> bool:
    if isinstance(operator, ast.Eq):
        return left == right
    if isinstance(operator, ast.NotEq):
        return left != right
    if isinstance(operator, ast.Lt):
        return left < right
    if isinstance(operator, ast.LtE):
        return left <= right
    if isinstance(operator, ast.Gt):
        return left > right
    if isinstance(operator, ast.GtE):
        return left >= right
    raise ValueError(f"unsupported comparison operator: {type(operator).__name__}")


def _suggest_repairs(proposal: ProposedRule, observations: list[Any]) -> list[dict[str, object]]:
    suggestions: list[dict[str, object]] = []
    base_predicate = compile_feature_expression(proposal.expression)
    base_matches = [obs for obs in observations if base_predicate(obs)]
    false_matches = [obs for obs in base_matches if not obs.greedy_is_optimal]
    if not false_matches:
        return suggestions
    for atom in _atomic_refinements(observations):
        expression = f"({proposal.expression}) and ({atom})"
        try:
            atom_predicate = compile_feature_expression(atom)
        except (SyntaxError, ValueError, TypeError, KeyError):
            continue
        if any(atom_predicate(obs) for obs in false_matches):
            continue
        support = sum(1 for obs in base_matches if atom_predicate(obs))
        if support:
            suggestions.append(
                {
                    "original_name": proposal.name,
                    "original_expression": proposal.expression,
                    "repair_expression": expression,
                    "added_constraint": atom,
                    "support": support,
                    "precision": 1.0,
                    "counterexamples": (),
                }
            )
    suggestions.sort(key=lambda item: (-int(item["support"]), str(item["repair_expression"])))
    return suggestions[:3]


def _atomic_refinements(observations: list[Any]) -> tuple[str, ...]:
    atoms: set[str] = set()
    boolean_features = sorted(
        feature
        for feature in ALLOWED_FEATURES
        if isinstance(observations[0].features.get(feature), bool)
    )
    numeric_features = ("n", "m", "max_degree", "min_degree", "num_components", "clique_number", "girth")
    for feature in boolean_features:
        atoms.add(feature)
        atoms.add(f"not {feature}")
    for feature in numeric_features:
        values = sorted({int(obs.features[feature]) for obs in observations})
        for value in values:
            atoms.add(f"{feature} == {value}")
            atoms.add(f"{feature} <= {value}")
            atoms.add(f"{feature} >= {value}")
    return tuple(sorted(atoms))


def _proposal_prompt(max_rules: int, feedback: str = "") -> str:
    feature_list = ", ".join(sorted(ALLOWED_FEATURES))
    feedback_block = ""
    if feedback.strip():
        feedback_block = f"""

Verifier feedback from previous attempts:
{feedback.strip()[:4000]}

Use the counterexamples to narrow or replace failed rules. Do not repeat a rejected
predicate unless you have made it strictly narrower.
"""
    return f"""You are proposing finite graph conjectures for a verifier.

Objective: find graph feature predicates where degree-descending greedy coloring is always optimal.

Return only JSON with this shape:
{{"rules":[{{"name":"short_name","description":"plain English claim","expression":"boolean feature predicate"}}]}}

Expression grammar:
- Feature names: {feature_list}
- Operators: and, or, not, ==, !=, <, <=, >, >=
- Constants: booleans and numbers only
- No functions, arithmetic, strings, lists, or prose inside expression.

Propose at most {max_rules} rules. Prefer narrow, falsifiable rules over broad vague rules.
{feedback_block}
"""


def _write_evaluation(root: Path, evaluation: ProposalEvaluation) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "proposal_evaluation.json").write_text(
        json.dumps(asdict(evaluation), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (root / "repair_sft.jsonl").write_text(
        "\n".join(json.dumps(item, sort_keys=True) for item in _repair_examples(evaluation)) + "\n",
        encoding="utf-8",
    )
    store = CognitiveStore(root / "ledger")
    store.init()
    branch = "experiment/graph-agent"
    if not store.branch_exists(branch):
        store.create_branch(branch, from_branch="main")
    for rule in (*evaluation.accepted_rules, *evaluation.rejected_rules):
        store.commit(
            branch,
            f"model proposal check: {rule.name}",
            [
                Event(
                    event_type="model_conjecture_verifier_result",
                    actor="verifier",
                    message=rule.description,
                    output_refs=(f"rule:{rule.name}",),
                    tests=(
                        TestResult(
                            "finite_graph_model_proposal_check",
                            "pass" if rule.promoted else "fail",
                            json.dumps(asdict(rule), sort_keys=True),
                        ),
                    ),
                )
            ],
        )


def _repair_examples(evaluation: ProposalEvaluation) -> list[dict[str, object]]:
    examples: list[dict[str, object]] = []
    for rule in evaluation.rejected_rules:
        examples.append(
            {
                "messages": [
                    {
                        "role": "system",
                        "content": "Revise model-generated graph conjectures after verifier failure.",
                    },
                    {
                        "role": "user",
                        "content": json.dumps(asdict(rule), sort_keys=True),
                    },
                    {
                        "role": "assistant",
                        "content": (
                            f"Reject `{rule.name}` as stated. It matched {rule.support} graphs "
                            f"but had {rule.false_positives} counterexamples: "
                            f"{', '.join(rule.counterexamples[:3])}."
                        ),
                    },
                ]
            }
        )
    for rule in evaluation.invalid_rules:
        examples.append(
            {
                "messages": [
                    {
                        "role": "system",
                        "content": "Repair invalid graph-conjecture DSL output.",
                    },
                    {"role": "user", "content": json.dumps(rule, sort_keys=True)},
                    {
                        "role": "assistant",
                        "content": (
                            "Return a predicate using only known graph features, boolean operators, "
                            "comparisons, and numeric or boolean constants."
                        ),
                    },
                ]
            }
        )
    for repair in evaluation.repair_suggestions:
        examples.append(
            {
                "messages": [
                    {
                        "role": "system",
                        "content": "Convert verifier-mined graph repair suggestions into stronger conjectures.",
                    },
                    {"role": "user", "content": json.dumps(repair, sort_keys=True)},
                    {
                        "role": "assistant",
                        "content": (
                            f"Replace `{repair['original_expression']}` with "
                            f"`{repair['repair_expression']}`. The repaired predicate had "
                            f"support {repair['support']} and no finite counterexamples."
                        ),
                    },
                ]
            }
        )
    return examples


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_]+", "_", value.strip().lower()).strip("_")
    return slug or "rule"


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify local-model graph conjectures.")
    parser.add_argument("--max-n", type=int, default=6)
    parser.add_argument("--root", default=".bcv_runs/graph_agent")
    parser.add_argument("--proposal-file")
    parser.add_argument("--max-rules", type=int, default=6)
    parser.add_argument("--use-model", action="store_true")
    parser.add_argument("--feedback-file")
    args = parser.parse_args()

    if args.proposal_file:
        proposals = load_proposals(args.proposal_file)
        result = evaluate_proposals(proposals, max_n=args.max_n, root=args.root)
    elif args.use_model:
        feedback = ""
        if args.feedback_file:
            feedback = _load_feedback(args.feedback_file)
        result = run_model_conjecture_loop(
            args.max_n,
            args.root,
            max_rules=args.max_rules,
            feedback=feedback,
        )
    else:
        raise LocalModelError("pass --proposal-file or --use-model")
    print(json.dumps(asdict(result), indent=2, sort_keys=True))


def _load_feedback(path: str | Path) -> str:
    text = Path(path).read_text(encoding="utf-8-sig")
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        return text
    lines: list[str] = []
    for rule in raw.get("rejected_rules", []):
        lines.append(
            "REJECTED "
            f"name={rule.get('name')} "
            f"precision={rule.get('precision')} "
            f"false_positives={rule.get('false_positives')} "
            f"predicate={_extract_predicate(str(rule.get('description', '')))} "
            f"counterexamples={rule.get('counterexamples', [])[:3]}"
        )
    for rule in raw.get("invalid_rules", []):
        lines.append(
            "INVALID "
            f"name={rule.get('name')} "
            f"expression={rule.get('expression')} "
            f"error={rule.get('error')}"
        )
    for rule in raw.get("accepted_rules", []):
        lines.append(
            "ACCEPTED "
            f"name={rule.get('name')} "
            f"predicate={_extract_predicate(str(rule.get('description', '')))} "
            f"support={rule.get('support')}"
        )
    for repair in raw.get("repair_suggestions", []):
        lines.append(
            "REPAIR_SUGGESTION "
            f"original={repair.get('original_expression')} "
            f"repair={repair.get('repair_expression')} "
            f"support={repair.get('support')} "
            f"added_constraint={repair.get('added_constraint')}"
        )
    return "\n".join(lines) or text


def _extract_predicate(description: str) -> str:
    marker = "Predicate:"
    if marker not in description:
        return ""
    return description.split(marker, 1)[1].strip()


if __name__ == "__main__":
    main()
