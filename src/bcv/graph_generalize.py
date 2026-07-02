"""Out-of-range stress test for graph conjectures accepted at small n.

Every rule and repair the pipeline promotes carries `precision 1.0` measured on the
exhaustive n <= 6 universe. This module re-checks those expressions on larger graphs
(random G(n, p) samples plus structured adversarial families such as crown graphs,
which are the classic worst case for greedy coloring) and reports which conjectures
survive and which are falsified once n grows past the training verifier's horizon.

A failure here is sound: any counterexample found is a real counterexample. Survival
is not a proof, only sampled evidence.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from bcv.discovery import Graph, GraphObservation, observe_graph
from bcv.graph_agent import compile_feature_expression


@dataclass(frozen=True)
class ExpressionGeneralization:
    expression: str
    source: str
    parseable: bool
    graphs_checked: int
    matched: int
    false_positives: int
    counterexamples: tuple[str, ...]
    survived: bool
    vacuous: bool


@dataclass(frozen=True)
class GeneralizationReport:
    ns: tuple[int, ...]
    graphs_checked: int
    random_graphs: int
    structured_graphs: int
    expressions_checked: int
    survived: int
    survived_vacuously: int
    falsified: int
    unparseable: int
    results: tuple[ExpressionGeneralization, ...]


def random_graph(n: int, p: float, rng: random.Random) -> Graph:
    edges = tuple(
        (u, v)
        for u in range(n)
        for v in range(u + 1, n)
        if rng.random() < p
    )
    return Graph(n=n, edges=edges)


def relabeled(graph: Graph, rng: random.Random) -> Graph:
    mapping = list(range(graph.n))
    rng.shuffle(mapping)
    edges = tuple(tuple(sorted((mapping[u], mapping[v]))) for u, v in graph.edges)
    return Graph(n=graph.n, edges=tuple(sorted(edges)))


def crown_graph(k: int) -> Graph:
    """Complete bipartite K_{k,k} minus a perfect matching: the greedy killer."""
    edges = tuple(
        (u, k + v)
        for u in range(k)
        for v in range(k)
        if u != v
    )
    return Graph(n=2 * k, edges=edges)


def crown_graph_interleaved(k: int) -> Graph:
    """Crown graph labeled so vertices 2i and 2i+1 are the removed matching pair.

    All degrees are equal, so degree-descending greedy falls back to index order and
    walks the matching: every pair is forced onto a fresh color, using k colors on a
    2-chromatic graph. This is the deterministic worst case; random relabelings only
    find it by luck.
    """
    edges = tuple(
        tuple(sorted((2 * i, 2 * j + 1)))
        for i in range(k)
        for j in range(k)
        if i != j
    )
    return Graph(n=2 * k, edges=tuple(sorted(edges)))


def greedy_adversarial_tree(n: int) -> Graph:
    """Tree on n >= 8 vertices where degree-descending greedy needs 3 colors.

    Two leaf-heavy hubs (labeled 0 and 1) sit at odd distance, joined through a
    2-vertex path (labeled 2 and 3). The hubs have the highest degree, get colored
    first with the same color, and the path interior is then forced onto colors 1
    and 2 — three colors on a 2-chromatic graph.
    """
    if n < 8:
        raise ValueError("greedy adversarial tree needs n >= 8")
    leaf_budget = n - 4
    left_leaves = leaf_budget // 2
    edges = [(0, 2), (2, 3), (1, 3)]
    next_vertex = 4
    for _ in range(left_leaves):
        edges.append((0, next_vertex))
        next_vertex += 1
    while next_vertex < n:
        edges.append((1, next_vertex))
        next_vertex += 1
    return Graph(n=n, edges=tuple(sorted(edges)))


BAD_PATH_6_EDGES = ((0, 3), (0, 5), (1, 2), (1, 4), (2, 3))
"""A 6-vertex path labeled so degree-descending greedy uses 3 colors on chi=2."""

BAD_CYCLE_6_EDGES = ((0, 4), (0, 5), (1, 2), (1, 3), (2, 5), (3, 4))
"""A 6-vertex even cycle labeled so degree-descending greedy uses 3 colors."""


def adversarial_path_union(n: int) -> Graph:
    """The bad 6-vertex path plus a disjoint matching: kills `num_components >= 2`
    conjectures that random sampling misses (adversarial labelings of disconnected
    graphs are vanishingly rare in G(n,p) draws)."""
    if n < 8:
        raise ValueError("adversarial path union needs n >= 8")
    edges = list(BAD_PATH_6_EDGES)
    vertex = 6
    while vertex + 1 < n:
        edges.append((vertex, vertex + 1))
        vertex += 2
    return Graph(n=n, edges=tuple(edges))


def adversarial_cycle_union(n: int) -> Graph:
    """The bad 6-vertex even cycle plus a disjoint cycle on the remaining vertices:
    2-regular, disconnected, chi=2 when the filler cycle is even, greedy 3."""
    if n < 9:
        raise ValueError("adversarial cycle union needs n >= 9")
    filler = tuple((vertex, vertex + 1) for vertex in range(6, n - 1)) + ((6, n - 1),)
    return Graph(n=n, edges=BAD_CYCLE_6_EDGES + filler)


def cycle_graph(n: int) -> Graph:
    return Graph(n=n, edges=tuple((i, (i + 1) % n) for i in range(n - 1)) + ((0, n - 1),))


def path_graph(n: int) -> Graph:
    return Graph(n=n, edges=tuple((i, i + 1) for i in range(n - 1)))


def complete_bipartite(a: int, b: int) -> Graph:
    return Graph(n=a + b, edges=tuple((u, a + v) for u in range(a) for v in range(b)))


def star_graph(n: int) -> Graph:
    return Graph(n=n, edges=tuple((0, i) for i in range(1, n)))


def structured_graphs(n: int) -> list[Graph]:
    graphs: list[Graph] = [cycle_graph(n), path_graph(n), star_graph(n)]
    if n % 2 == 0 and n >= 6:
        graphs.append(crown_graph(n // 2))
        graphs.append(crown_graph_interleaved(n // 2))
    if n >= 8:
        graphs.append(greedy_adversarial_tree(n))
        graphs.append(adversarial_path_union(n))
    if n >= 9:
        graphs.append(adversarial_cycle_union(n))
    for a in range(1, n // 2 + 1):
        graphs.append(complete_bipartite(a, n - a))
    return graphs


def build_observation_pool(
    ns: tuple[int, ...] = (7, 8),
    samples_per_np: int = 120,
    ps: tuple[float, ...] = (0.2, 0.35, 0.5, 0.65, 0.8),
    relabels: int = 3,
    seed: int = 0,
    library_path: str | Path | None = None,
) -> tuple[list[GraphObservation], int, int]:
    rng = random.Random(seed)
    seen: set[str] = set()
    graphs: list[Graph] = []

    def add(graph: Graph) -> None:
        key = graph.graph_id()
        if key not in seen:
            seen.add(key)
            graphs.append(graph)

    for n in ns:
        for base in structured_graphs(n):
            add(base)
            for _ in range(relabels):
                add(relabeled(base, rng))
    if library_path is not None:
        from bcv.graph_adversary import library_graphs

        for graph in library_graphs(library_path):
            add(graph)
    structured_count = len(graphs)
    for n in ns:
        for p in ps:
            for _ in range(samples_per_np):
                add(random_graph(n, p, rng))
    random_count = len(graphs) - structured_count
    observations = [observe_graph(graph) for graph in graphs]
    return observations, random_count, structured_count


def check_expressions(
    expressions: dict[str, str],
    observations: list[GraphObservation],
) -> list[ExpressionGeneralization]:
    results: list[ExpressionGeneralization] = []
    for expression, source in sorted(expressions.items()):
        try:
            predicate = compile_feature_expression(expression)
        except (SyntaxError, ValueError, TypeError, KeyError):
            results.append(
                ExpressionGeneralization(
                    expression=expression,
                    source=source,
                    parseable=False,
                    graphs_checked=len(observations),
                    matched=0,
                    false_positives=0,
                    counterexamples=(),
                    survived=False,
                    vacuous=True,
                )
            )
            continue
        matches = [obs for obs in observations if predicate(obs)]
        false_positives = [obs for obs in matches if not obs.greedy_is_optimal]
        results.append(
            ExpressionGeneralization(
                expression=expression,
                source=source,
                parseable=True,
                graphs_checked=len(observations),
                matched=len(matches),
                false_positives=len(false_positives),
                counterexamples=tuple(obs.graph.graph_id() for obs in false_positives[:5]),
                survived=not false_positives,
                vacuous=not matches,
            )
        )
    return results


def collect_expressions_from_evaluation(path: str | Path) -> dict[str, str]:
    raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    expressions: dict[str, str] = {}
    for rule in raw.get("accepted_rules", []):
        predicate = _extract_predicate(str(rule.get("description", "")))
        if predicate:
            expressions[predicate] = "accepted_rule"
    for repair in raw.get("repair_suggestions", []):
        expression = repair.get("repair_expression")
        if isinstance(expression, str) and expression.strip():
            expressions.setdefault(expression.strip(), "repair_suggestion")
    return expressions


def _extract_predicate(description: str) -> str:
    marker = "Predicate:"
    if marker not in description:
        return ""
    return description.split(marker, 1)[1].strip()


def run_generalization(
    expressions: dict[str, str],
    ns: tuple[int, ...] = (7, 8),
    samples_per_np: int = 120,
    relabels: int = 3,
    seed: int = 0,
    root: str | Path = ".bcv_runs/graph_generalize",
    library_path: str | Path | None = None,
) -> GeneralizationReport:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    observations, random_count, structured_count = build_observation_pool(
        ns=ns,
        samples_per_np=samples_per_np,
        relabels=relabels,
        seed=seed,
        library_path=library_path,
    )
    results = check_expressions(expressions, observations)
    report = GeneralizationReport(
        ns=tuple(ns),
        graphs_checked=len(observations),
        random_graphs=random_count,
        structured_graphs=structured_count,
        expressions_checked=len(results),
        survived=sum(1 for result in results if result.parseable and result.survived),
        survived_vacuously=sum(1 for result in results if result.parseable and result.survived and result.vacuous),
        falsified=sum(1 for result in results if result.parseable and not result.survived),
        unparseable=sum(1 for result in results if not result.parseable),
        results=tuple(results),
    )
    (root / "generalization_report.json").write_text(
        json.dumps(asdict(report), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Stress-test accepted graph conjectures at larger n.")
    parser.add_argument("--evaluation-file", help="proposal_evaluation.json with accepted rules and repairs")
    parser.add_argument("--expression", action="append", default=[], help="extra expression to check (repeatable)")
    parser.add_argument("--ns", type=int, nargs="+", default=[7, 8])
    parser.add_argument("--samples-per-np", type=int, default=120)
    parser.add_argument("--relabels", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--root", default=".bcv_runs/graph_generalize")
    parser.add_argument(
        "--library",
        default=None,
        help="adversary_library.jsonl of discovered counterexamples to add to the pool",
    )
    args = parser.parse_args()

    expressions: dict[str, str] = {}
    if args.evaluation_file:
        expressions.update(collect_expressions_from_evaluation(args.evaluation_file))
    for expression in args.expression:
        expressions[expression] = "cli"
    if not expressions:
        raise SystemExit("pass --evaluation-file or --expression")

    report = run_generalization(
        expressions,
        ns=tuple(args.ns),
        samples_per_np=args.samples_per_np,
        relabels=args.relabels,
        seed=args.seed,
        root=args.root,
        library_path=args.library,
    )
    print(json.dumps(asdict(report), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
