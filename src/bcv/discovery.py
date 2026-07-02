from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from bcv.schema import Event, TestResult
from bcv.store import CognitiveStore


@dataclass(frozen=True)
class Graph:
    n: int
    edges: tuple[tuple[int, int], ...]

    def adjacency(self) -> tuple[tuple[int, ...], ...]:
        adj = [set() for _ in range(self.n)]
        for u, v in self.edges:
            adj[u].add(v)
            adj[v].add(u)
        return tuple(tuple(sorted(neighbors)) for neighbors in adj)

    def degrees(self) -> tuple[int, ...]:
        adj = self.adjacency()
        return tuple(len(neighbors) for neighbors in adj)

    def edge_count(self) -> int:
        return len(self.edges)

    def graph_id(self) -> str:
        return f"n{self.n}:" + ",".join(f"{u}-{v}" for u, v in self.edges)


@dataclass(frozen=True)
class GraphObservation:
    graph: Graph
    chromatic_number: int
    greedy_degree_desc_colors: int
    greedy_is_optimal: bool
    features: dict[str, bool | int | float]


@dataclass(frozen=True)
class CandidateRule:
    name: str
    description: str
    predicate: Callable[[GraphObservation], bool]


@dataclass(frozen=True)
class RuleResult:
    name: str
    description: str
    support: int
    precision: float
    false_positives: int
    counterexamples: tuple[str, ...]
    promoted: bool


@dataclass(frozen=True)
class DiscoveryResult:
    graphs_checked: int
    max_n: int
    baseline_optimal_rate: float
    promoted_rules: tuple[RuleResult, ...]
    rejected_rules: tuple[RuleResult, ...]


def enumerate_graphs(max_n: int) -> list[Graph]:
    graphs: list[Graph] = []
    for n in range(1, max_n + 1):
        possible_edges = tuple(itertools.combinations(range(n), 2))
        for mask in range(1 << len(possible_edges)):
            edges = tuple(edge for bit, edge in enumerate(possible_edges) if mask & (1 << bit))
            graphs.append(Graph(n=n, edges=edges))
    return graphs


def observe_graph(graph: Graph) -> GraphObservation:
    chromatic = exact_chromatic_number(graph)
    greedy = greedy_degree_desc_coloring(graph)
    features = graph_features(graph)
    return GraphObservation(
        graph=graph,
        chromatic_number=chromatic,
        greedy_degree_desc_colors=greedy,
        greedy_is_optimal=greedy == chromatic,
        features=features,
    )


def exact_chromatic_number(graph: Graph) -> int:
    if graph.n == 0:
        return 0
    for colors in range(1, graph.n + 1):
        if _can_color(graph, colors):
            return colors
    return graph.n


def greedy_degree_desc_coloring(graph: Graph) -> int:
    adj = graph.adjacency()
    degrees = graph.degrees()
    order = sorted(range(graph.n), key=lambda node: (-degrees[node], node))
    assignment: dict[int, int] = {}
    for node in order:
        used = {assignment[neighbor] for neighbor in adj[node] if neighbor in assignment}
        color = 0
        while color in used:
            color += 1
        assignment[node] = color
    return max(assignment.values(), default=-1) + 1


ACYCLIC_GIRTH = 999
"""Sentinel girth for forests. Mathematically girth is infinite when no cycle
exists, so a large sentinel keeps `girth >= k` predicates aligned with the usual
convention (forests satisfy every finite lower bound)."""


def graph_features(graph: Graph) -> dict[str, bool | int | float]:
    degrees = graph.degrees()
    edge_count = graph.edge_count()
    max_edges = graph.n * (graph.n - 1) / 2
    return {
        "n": graph.n,
        "m": edge_count,
        "density": edge_count / max_edges if max_edges else 0.0,
        "max_degree": max(degrees, default=0),
        "min_degree": min(degrees, default=0),
        "is_connected": is_connected(graph),
        "is_complete": edge_count == max_edges,
        "is_forest": is_forest(graph),
        "is_tree": is_connected(graph) and edge_count == graph.n - 1,
        "is_bipartite": is_bipartite(graph),
        "is_triangle_free": is_triangle_free(graph),
        "max_degree_le_2": max(degrees, default=0) <= 2,
        "has_universal_vertex": any(degree == graph.n - 1 for degree in degrees),
        "has_isolated_vertex": any(degree == 0 for degree in degrees),
        "is_regular": len(set(degrees)) <= 1,
        "num_components": num_components(graph),
        "clique_number": clique_number(graph),
        "girth": girth(graph),
    }


def candidate_rules() -> tuple[CandidateRule, ...]:
    return (
        CandidateRule(
            "complete_graphs",
            "Degree-descending greedy is optimal on complete graphs.",
            lambda obs: bool(obs.features["is_complete"]),
        ),
        CandidateRule(
            "forests",
            "Degree-descending greedy is optimal on forests.",
            lambda obs: bool(obs.features["is_forest"]),
        ),
        CandidateRule(
            "trees",
            "Degree-descending greedy is optimal on trees.",
            lambda obs: bool(obs.features["is_tree"]),
        ),
        CandidateRule(
            "max_degree_le_2",
            "Degree-descending greedy is optimal on graphs with maximum degree at most 2.",
            lambda obs: bool(obs.features["max_degree_le_2"]),
        ),
        CandidateRule(
            "bipartite_max_degree_le_2",
            "Degree-descending greedy is optimal on bipartite graphs with maximum degree at most 2.",
            lambda obs: bool(obs.features["is_bipartite"]) and bool(obs.features["max_degree_le_2"]),
        ),
        CandidateRule(
            "has_universal_vertex",
            "Degree-descending greedy is optimal on graphs with a universal vertex.",
            lambda obs: bool(obs.features["has_universal_vertex"]),
        ),
        CandidateRule(
            "triangle_free_regular",
            "Degree-descending greedy is optimal on triangle-free regular graphs.",
            lambda obs: bool(obs.features["is_triangle_free"]) and bool(obs.features["is_regular"]),
        ),
        CandidateRule(
            "connected_bipartite",
            "Degree-descending greedy is optimal on connected bipartite graphs.",
            lambda obs: bool(obs.features["is_connected"]) and bool(obs.features["is_bipartite"]),
        ),
    )


def run_graph_discovery(max_n: int = 6, root: str | Path = ".bcv_runs/discovery") -> DiscoveryResult:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    observations = [observe_graph(graph) for graph in enumerate_graphs(max_n)]
    rule_results = tuple(evaluate_rule(rule, observations) for rule in candidate_rules())
    promoted = tuple(result for result in rule_results if result.promoted)
    rejected = tuple(result for result in rule_results if not result.promoted)
    baseline_rate = sum(1 for obs in observations if obs.greedy_is_optimal) / len(observations)
    result = DiscoveryResult(
        graphs_checked=len(observations),
        max_n=max_n,
        baseline_optimal_rate=baseline_rate,
        promoted_rules=promoted,
        rejected_rules=rejected,
    )
    _write_discovery(root, observations, result)
    return result


def evaluate_rule(rule: CandidateRule, observations: list[GraphObservation]) -> RuleResult:
    matches = [obs for obs in observations if rule.predicate(obs)]
    false_positives = [obs for obs in matches if not obs.greedy_is_optimal]
    precision = (
        (len(matches) - len(false_positives)) / len(matches)
        if matches
        else 0.0
    )
    return RuleResult(
        name=rule.name,
        description=rule.description,
        support=len(matches),
        precision=precision,
        false_positives=len(false_positives),
        counterexamples=tuple(obs.graph.graph_id() for obs in false_positives[:5]),
        promoted=len(matches) > 0 and not false_positives,
    )


def is_connected(graph: Graph) -> bool:
    if graph.n <= 1:
        return True
    adj = graph.adjacency()
    seen = {0}
    stack = [0]
    while stack:
        node = stack.pop()
        for neighbor in adj[node]:
            if neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return len(seen) == graph.n


def is_bipartite(graph: Graph) -> bool:
    adj = graph.adjacency()
    colors: dict[int, int] = {}
    for start in range(graph.n):
        if start in colors:
            continue
        colors[start] = 0
        stack = [start]
        while stack:
            node = stack.pop()
            for neighbor in adj[node]:
                if neighbor not in colors:
                    colors[neighbor] = 1 - colors[node]
                    stack.append(neighbor)
                elif colors[neighbor] == colors[node]:
                    return False
    return True


def is_forest(graph: Graph) -> bool:
    parent = list(range(graph.n))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for u, v in graph.edges:
        root_u = find(u)
        root_v = find(v)
        if root_u == root_v:
            return False
        parent[root_u] = root_v
    return True


def num_components(graph: Graph) -> int:
    parent = list(range(graph.n))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    components = graph.n
    for u, v in graph.edges:
        root_u = find(u)
        root_v = find(v)
        if root_u != root_v:
            parent[root_u] = root_v
            components -= 1
    return components


def clique_number(graph: Graph) -> int:
    if graph.n == 0:
        return 0
    adj = [set(neighbors) for neighbors in graph.adjacency()]
    upper = min(graph.n, max(graph.degrees(), default=0) + 1)
    for size in range(upper, 1, -1):
        for candidate in itertools.combinations(range(graph.n), size):
            if all(v in adj[u] for u, v in itertools.combinations(candidate, 2)):
                return size
    return 1


def girth(graph: Graph) -> int:
    """Length of the shortest cycle; ACYCLIC_GIRTH when the graph is a forest."""
    adj = graph.adjacency()
    best = ACYCLIC_GIRTH
    for start in range(graph.n):
        distance = {start: 0}
        parent = {start: -1}
        queue = [start]
        head = 0
        while head < len(queue):
            node = queue[head]
            head += 1
            for neighbor in adj[node]:
                if neighbor not in distance:
                    distance[neighbor] = distance[node] + 1
                    parent[neighbor] = node
                    queue.append(neighbor)
                elif parent[node] != neighbor:
                    best = min(best, distance[node] + distance[neighbor] + 1)
    return best


def is_triangle_free(graph: Graph) -> bool:
    adj = graph.adjacency()
    for a, b, c in itertools.combinations(range(graph.n), 3):
        if b in adj[a] and c in adj[a] and c in adj[b]:
            return False
    return True


def _can_color(graph: Graph, color_count: int) -> bool:
    adj = graph.adjacency()
    order = sorted(range(graph.n), key=lambda node: (-len(adj[node]), node))
    colors = [-1] * graph.n

    def backtrack(index: int) -> bool:
        if index == len(order):
            return True
        node = order[index]
        used = {colors[neighbor] for neighbor in adj[node] if colors[neighbor] != -1}
        for color in range(color_count):
            if color in used:
                continue
            colors[node] = color
            if backtrack(index + 1):
                return True
            colors[node] = -1
        return False

    return backtrack(0)


def _write_discovery(root: Path, observations: list[GraphObservation], result: DiscoveryResult) -> None:
    (root / "discovery_result.json").write_text(
        json.dumps(asdict(result), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (root / "observations.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "graph_id": obs.graph.graph_id(),
                    "chromatic_number": obs.chromatic_number,
                    "greedy_degree_desc_colors": obs.greedy_degree_desc_colors,
                    "greedy_is_optimal": obs.greedy_is_optimal,
                    "features": obs.features,
                },
                sort_keys=True,
            )
            for obs in observations
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "repair_sft.jsonl").write_text(
        "\n".join(json.dumps(item, sort_keys=True) for item in _repair_examples(result)) + "\n",
        encoding="utf-8",
    )
    store = CognitiveStore(root / "ledger")
    store.init()
    branch = "experiment/graph-discovery"
    if not store.branch_exists(branch):
        store.create_branch(branch, from_branch="main")
    for rule in (*result.promoted_rules, *result.rejected_rules):
        store.commit(
            branch,
            f"evaluate rule: {rule.name}",
            [
                Event(
                    event_type="verifier_result",
                    actor="verifier",
                    message=rule.description,
                    output_refs=(f"rule:{rule.name}",),
                    tests=(
                        TestResult(
                            "finite_graph_rule_check",
                            "pass" if rule.promoted else "fail",
                            json.dumps(asdict(rule), sort_keys=True),
                        ),
                    ),
                )
            ],
        )


def _repair_examples(result: DiscoveryResult) -> list[dict[str, object]]:
    examples: list[dict[str, object]] = []
    for rule in result.rejected_rules:
        examples.append(
            {
                "messages": [
                    {
                        "role": "system",
                        "content": "Repair overbroad graph conjectures using finite verifier counterexamples. Do not preserve a conjecture contradicted by evidence.",
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "conjecture": rule.description,
                                "support": rule.support,
                                "precision": rule.precision,
                                "counterexamples": rule.counterexamples,
                            },
                            sort_keys=True,
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": (
                            f"Reject or weaken `{rule.name}`. The finite verifier found "
                            f"{rule.false_positives} counterexamples among {rule.support} matching graphs; "
                            f"first counterexamples: {', '.join(rule.counterexamples[:3])}."
                        ),
                    },
                ]
            }
        )
    return examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-n", type=int, default=6)
    parser.add_argument("--root", default=".bcv_runs/discovery")
    args = parser.parse_args()
    print(json.dumps(asdict(run_graph_discovery(args.max_n, args.root)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
