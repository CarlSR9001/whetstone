"""Pluggable conjecture domains: greedy heuristic vs exact optimum.

Everything downstream (exhaustive small-n verification, atomic repair mining, the
stress pool, the annealing falsifier, the novelty hull) only ever touches three
things: a graph's feature dict, a "claim holds" bit, and a failure gap. So a domain
is just an observe function. `GraphObservation.chromatic_number` and
`greedy_degree_desc_colors` are reused as generic (exact value, greedy value) slots;
`greedy_is_optimal` is the claim bit. The feature DSL is shared across domains, which
means the novelty hull and the miner's atom vocabulary transfer for free.

Domains:
- coloring: degree-descending greedy coloring uses exactly chi colors.
- mis: degree-ascending greedy independent set reaches the independence number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from bcv.discovery import Graph, GraphObservation, graph_features, observe_graph


@dataclass(frozen=True)
class Domain:
    name: str
    claim: str
    observe: Callable[[Graph], GraphObservation]

    def gap(self, observation: GraphObservation) -> int:
        """How badly the greedy heuristic failed on this observation (0 = claim holds)."""
        return abs(observation.greedy_degree_desc_colors - observation.chromatic_number)


def greedy_independent_set_size(graph: Graph) -> int:
    """Degree-ascending greedy MIS with index tie-breaks."""
    adj = graph.adjacency()
    degrees = graph.degrees()
    order = sorted(range(graph.n), key=lambda node: (degrees[node], node))
    chosen: set[int] = set()
    for node in order:
        if all(neighbor not in chosen for neighbor in adj[node]):
            chosen.add(node)
    return len(chosen)


def independence_number(graph: Graph) -> int:
    """Exact maximum independent set size by branch and bound."""
    adj = [set(neighbors) for neighbors in graph.adjacency()]

    def solve(candidates: frozenset[int], size: int, best: int) -> int:
        if size + len(candidates) <= best:
            return best
        if not candidates:
            return max(best, size)
        node = max(candidates, key=lambda v: len(adj[v] & candidates))
        # Branch 1: include node.
        best = solve(candidates - adj[node] - {node}, size + 1, best)
        # Branch 2: exclude node.
        best = solve(candidates - {node}, size, best)
        return best

    return solve(frozenset(range(graph.n)), 0, 0)


def observe_graph_mis(graph: Graph) -> GraphObservation:
    exact = independence_number(graph)
    greedy = greedy_independent_set_size(graph)
    return GraphObservation(
        graph=graph,
        chromatic_number=exact,
        greedy_degree_desc_colors=greedy,
        greedy_is_optimal=greedy == exact,
        features=graph_features(graph),
    )


COLORING = Domain(
    name="coloring",
    claim="degree-descending greedy coloring uses exactly the chromatic number",
    observe=observe_graph,
)

MIS = Domain(
    name="mis",
    claim="degree-ascending greedy independent set reaches the independence number",
    observe=observe_graph_mis,
)

DOMAINS = {domain.name: domain for domain in (COLORING, MIS)}
