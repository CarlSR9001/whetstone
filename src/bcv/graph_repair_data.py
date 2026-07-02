from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from bcv.graph_agent import (
    ProposedRule,
    _atomic_refinements,
    _observations_for,
    compile_feature_expression,
    evaluate_proposals,
)


BASE_EXPRESSIONS = (
    "is_tree",
    "is_forest",
    "is_bipartite",
    "is_triangle_free",
    "max_degree_le_2",
    "is_connected and is_bipartite",
    "is_bipartite and max_degree_le_2",
    "is_forest and max_degree_le_2",
    "is_triangle_free and is_regular",
    "is_connected and is_triangle_free",
    "is_connected and max_degree_le_2",
    "is_bipartite and is_triangle_free",
    "is_forest and not has_isolated_vertex",
    "is_bipartite and not has_isolated_vertex",
    "is_triangle_free and not has_isolated_vertex",
)

# Appended after every original candidate so runs capped at --max-proposals 48
# keep their historical candidate set; raise the cap to reach these.
EXTENDED_EXPRESSIONS = (
    "girth >= 5",
    "girth >= 5 and is_connected",
    "girth >= 4 and is_connected",
    "clique_number <= 2 and is_connected",
    "num_components == 1 and is_regular",
    "num_components >= 2 and is_forest",
    "girth == 999 and num_components <= 2",
    "clique_number >= 3 and is_connected",
)


@dataclass(frozen=True)
class GraphRepairDatasetResult:
    proposal_count: int
    accepted_rules: int
    rejected_rules: int
    repair_suggestions: int
    sft_examples: int
    sft_path: str
    json_sft_examples: int
    json_sft_path: str
    evaluation_path: str


def build_graph_repair_dataset(
    root: str | Path = ".bcv_runs/graph_repair_data",
    max_n: int = 6,
    max_proposals: int | None = 48,
) -> GraphRepairDatasetResult:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    expressions = _candidate_expressions()
    if max_proposals is not None:
        expressions = expressions[:max_proposals]
    proposals = tuple(
        ProposedRule(
            name=f"candidate_{index:03d}",
            description=f"Candidate graph class `{expression}` should be exact.",
            expression=expression,
        )
        for index, expression in enumerate(expressions)
    )
    evaluation = evaluate_proposals(
        proposals,
        max_n=max_n,
        backend="synthetic",
        model="graph_repair_data",
        root=root,
    )
    sft_path = root / "repair_sft.jsonl"
    json_sft_path = root / "repair_json_sft.jsonl"
    _write_json_sft(json_sft_path, evaluation.repair_suggestions)
    result = GraphRepairDatasetResult(
        proposal_count=len(proposals),
        accepted_rules=len(evaluation.accepted_rules),
        rejected_rules=len(evaluation.rejected_rules),
        repair_suggestions=len(evaluation.repair_suggestions),
        sft_examples=_count_jsonl(sft_path),
        sft_path=str(sft_path),
        json_sft_examples=_count_jsonl(json_sft_path),
        json_sft_path=str(json_sft_path),
        evaluation_path=str(root / "proposal_evaluation.json"),
    )
    (root / "dataset_result.json").write_text(
        json.dumps(asdict(result), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return result


HARD_SYSTEM_PROMPT = (
    "You repair rejected graph conjectures about degree-descending greedy coloring. "
    "The original predicate matched counterexample graphs where greedy was not optimal. "
    "Return only JSON with one key: repair_expression. The value must be a strictly "
    "narrower verifier DSL predicate that keeps positive graphs and excludes every "
    "counterexample. Allowed features: n, m, density, max_degree, min_degree, "
    "is_connected, is_complete, is_forest, is_tree, is_bipartite, is_triangle_free, "
    "max_degree_le_2, has_universal_vertex, has_isolated_vertex, is_regular, "
    "num_components, clique_number, girth (999 means acyclic). "
    "Allowed operators: and, or, not, ==, !=, <, <=, >, >=."
)

EVIDENCE_FEATURES = (
    "n",
    "m",
    "max_degree",
    "min_degree",
    "is_connected",
    "is_bipartite",
    "is_triangle_free",
    "is_regular",
    "has_isolated_vertex",
    "max_degree_le_2",
    "num_components",
    "clique_number",
    "girth",
)


@dataclass(frozen=True)
class HardGraphRepairDatasetResult:
    proposal_count: int
    rejected_rules: int
    repair_groups: int
    train_examples: int
    heldout_examples: int
    train_path: str
    heldout_path: str
    all_path: str
    evaluation_path: str
    stress_ns: tuple[int, ...] = ()
    dropped_groups: int = 0


def build_hard_graph_repair_dataset(
    root: str | Path = ".bcv_runs/graph_repair_hard",
    max_n: int = 6,
    max_proposals: int | None = 48,
    heldout_groups: int = 8,
    evidence_examples: int = 4,
    stress_ns: tuple[int, ...] = (),
    stress_samples_per_np: int = 120,
    stress_seed: int = 0,
) -> HardGraphRepairDatasetResult:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    expressions = _candidate_expressions()
    if max_proposals is not None:
        expressions = expressions[:max_proposals]
    proposals = tuple(
        ProposedRule(
            name=f"candidate_{index:03d}",
            description=f"Candidate graph class `{expression}` should be exact.",
            expression=expression,
        )
        for index, expression in enumerate(expressions)
    )
    evaluation = evaluate_proposals(
        proposals,
        max_n=max_n,
        backend="synthetic",
        model="graph_repair_hard",
        root=root,
    )
    best_repairs: dict[str, dict[str, object]] = {}
    for repair in evaluation.repair_suggestions:
        original = str(repair["original_expression"])
        current = best_repairs.get(original)
        if current is None or int(repair["support"]) > int(current["support"]):
            best_repairs[original] = repair

    observations = _observations_for(max_n)
    dropped_groups = 0
    if stress_ns:
        from bcv.graph_generalize import build_observation_pool

        pool, _, _ = build_observation_pool(
            ns=stress_ns,
            samples_per_np=stress_samples_per_np,
            seed=stress_seed,
        )
        targets: dict[str, str] = {}
        for original in sorted(best_repairs):
            stressed = _stress_best_repair(original, observations, pool)
            if stressed is None:
                dropped_groups += 1
            else:
                targets[original] = stressed
    else:
        targets = {
            original: str(best_repairs[original]["repair_expression"])
            for original in sorted(best_repairs)
        }
    examples = [
        _hard_example(original, targets[original], observations, evidence_examples)
        for original in sorted(targets)
    ]
    heldout_groups = min(heldout_groups, len(examples))
    stride = max(1, len(examples) // heldout_groups) if heldout_groups else len(examples) + 1
    heldout_indices = set(range(0, len(examples), stride)[:heldout_groups] if heldout_groups else [])
    heldout = [example for index, example in enumerate(examples) if index in heldout_indices]
    train = [example for index, example in enumerate(examples) if index not in heldout_indices]

    train_path = root / "hard_train.jsonl"
    heldout_path = root / "hard_heldout.jsonl"
    all_path = root / "hard_all.jsonl"
    _write_jsonl(train_path, train)
    _write_jsonl(heldout_path, heldout)
    _write_jsonl(all_path, examples)
    result = HardGraphRepairDatasetResult(
        proposal_count=len(proposals),
        rejected_rules=len(evaluation.rejected_rules),
        repair_groups=len(examples),
        train_examples=len(train),
        heldout_examples=len(heldout),
        train_path=str(train_path),
        heldout_path=str(heldout_path),
        all_path=str(all_path),
        evaluation_path=str(root / "proposal_evaluation.json"),
        stress_ns=tuple(stress_ns),
        dropped_groups=dropped_groups,
    )
    (root / "hard_dataset_result.json").write_text(
        json.dumps(asdict(result), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return result


def _stress_best_repair(
    original_expression: str,
    observations,
    pool,
) -> str | None:
    """Re-mine the best conjunctive repair whose full expression also survives the stress pool.

    Unlike the plain miner, a candidate is kept only if it has no counterexamples on the
    exhaustive small-n universe AND no counterexamples among the larger-n pool graphs it
    matches. Vacuous pool survival (matching nothing at larger n) is allowed but ranked
    below repairs that keep matching real larger graphs.
    """
    predicate = compile_feature_expression(original_expression)
    base_matches = [obs for obs in observations if predicate(obs)]
    false_matches = [obs for obs in base_matches if not obs.greedy_is_optimal]
    if not false_matches:
        return None
    best: tuple[int, int, str] | None = None
    for atom in _atomic_refinements(observations):
        try:
            atom_predicate = compile_feature_expression(atom)
        except (SyntaxError, ValueError, TypeError, KeyError):
            continue
        if any(atom_predicate(obs) for obs in false_matches):
            continue
        support = sum(1 for obs in base_matches if atom_predicate(obs))
        if not support:
            continue
        expression = f"({original_expression}) and ({atom})"
        repaired = compile_feature_expression(expression)
        pool_matches = [obs for obs in pool if repaired(obs)]
        if any(not obs.greedy_is_optimal for obs in pool_matches):
            continue
        key = (min(len(pool_matches), 1), support, expression)
        if best is None or (key[0], key[1]) > (best[0], best[1]):
            best = key
    return best[2] if best else None


def _hard_example(
    original_expression: str,
    repair_expression: str,
    observations,
    evidence_examples: int,
) -> dict[str, object]:
    predicate = compile_feature_expression(original_expression)
    matches = [obs for obs in observations if predicate(obs)]
    false_positives = [obs for obs in matches if not obs.greedy_is_optimal]
    true_positives = [obs for obs in matches if obs.greedy_is_optimal]
    return {
        "messages": [
            {"role": "system", "content": HARD_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "original_expression": original_expression,
                        "false_positive_count": len(false_positives),
                        "counterexamples": _distinct_evidence(false_positives, evidence_examples, with_coloring=True),
                        "kept_examples": _distinct_evidence(true_positives, 2),
                    },
                    sort_keys=True,
                ),
            },
            {
                "role": "assistant",
                "content": json.dumps({"repair_expression": repair_expression}, sort_keys=True),
            },
        ]
    }


def _distinct_evidence(observations, limit: int, with_coloring: bool = False) -> list[dict[str, object]]:
    """First `limit` observations with pairwise-distinct feature summaries.

    Relabelings of the same graph produce identical summaries; duplicated evidence
    wastes prompt tokens without adding information.
    """
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for observation in observations:
        summary = _evidence(observation, with_coloring=with_coloring)
        key = json.dumps(summary, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        rows.append(summary)
        if len(rows) >= limit:
            break
    return rows


def _evidence(observation, with_coloring: bool = False) -> dict[str, object]:
    summary: dict[str, object] = {feature: observation.features[feature] for feature in EVIDENCE_FEATURES}
    if with_coloring:
        summary["chromatic_number"] = observation.chromatic_number
        summary["greedy_colors"] = observation.greedy_degree_desc_colors
    return summary


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _candidate_expressions() -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()

    def add(expression: str) -> None:
        if expression not in seen:
            seen.add(expression)
            ordered.append(expression)

    for expression in BASE_EXPRESSIONS:
        add(expression)

    boolean_constraints = (
        "is_connected",
        "not is_connected",
        "has_isolated_vertex",
        "not has_isolated_vertex",
        "is_regular",
        "not is_regular",
    )
    numeric_constraints = (
        "m >= 3",
        "m >= 4",
        "m >= 5",
        "max_degree >= 2",
        "max_degree >= 3",
        "max_degree_le_2",
    )
    for base in BASE_EXPRESSIONS:
        for constraint in (*boolean_constraints, *numeric_constraints):
            if constraint in base:
                continue
            add(f"{base} and {constraint}")
    for expression in EXTENDED_EXPRESSIONS:
        add(expression)
    return tuple(ordered)


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _write_json_sft(path: Path, repair_suggestions: tuple[dict[str, object], ...]) -> None:
    rows = []
    for repair in repair_suggestions:
        rows.append(
            {
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Return only JSON with one key: repair_expression. "
                            "The value must be a verifier DSL predicate."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "original_expression": repair["original_expression"],
                                "added_constraint": repair["added_constraint"],
                                "support": repair["support"],
                                "precision": repair["precision"],
                            },
                            sort_keys=True,
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps(
                            {"repair_expression": repair["repair_expression"]},
                            sort_keys=True,
                        ),
                    },
                ]
            }
        )
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate graph verifier-repair SFT data.")
    parser.add_argument("--root", default=".bcv_runs/graph_repair_data")
    parser.add_argument("--max-n", type=int, default=6)
    parser.add_argument("--max-proposals", type=int, default=48)
    parser.add_argument("--hard", action="store_true", help="Build the no-leak hard dataset with group split.")
    parser.add_argument("--heldout-groups", type=int, default=8)
    parser.add_argument("--evidence-examples", type=int, default=4)
    parser.add_argument(
        "--stress-ns",
        type=int,
        nargs="*",
        default=[],
        help="Mine only repair targets that also survive sampled graphs at these larger n.",
    )
    args = parser.parse_args()
    if args.hard:
        result = asdict(
            build_hard_graph_repair_dataset(
                args.root,
                args.max_n,
                args.max_proposals,
                heldout_groups=args.heldout_groups,
                evidence_examples=args.evidence_examples,
                stress_ns=tuple(args.stress_ns),
            )
        )
    else:
        result = asdict(build_graph_repair_dataset(args.root, args.max_n, args.max_proposals))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
