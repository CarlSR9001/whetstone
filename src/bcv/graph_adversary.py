"""Optimization-driven counterexample search for graph conjectures.

The generalization pool (random G(n,p) plus hand-built adversary families) samples a
hostile distribution, but every family in it was curated by a human after a failure
was already suspected. This module replaces curation with search: simulated annealing
over edge flips, constrained to the conjecture's predicate class, climbing toward
graphs where degree-descending greedy coloring beats exact chromatic number.

Key economy: the annealing objective is the *greedy* color count (cheap, O(E)); exact
chromatic number is only computed when a state is a plausible counterexample —
bipartite states with greedy >= 3 (chi is 2, gap certain) or non-bipartite states
with greedy >= 4 (chi >= 3, gap possible). A confirmed find is exact, never sampled.

Every confirmed counterexample is appended to a persistent adversary library
(JSONL). `bcv.graph_generalize.build_observation_pool` can load the library, so each
discovery permanently raises the hostility of every future verification pool: the
harness's search frontier grows instead of being re-curated.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

from bcv.discovery import Graph, exact_chromatic_number, greedy_degree_desc_coloring
from bcv.graph_agent import compile_feature_expression


DEFAULT_LIBRARY = Path(".bcv_runs/adversary_library.jsonl")


@dataclass(frozen=True)
class AdversaryFind:
    expression: str
    graph_id: str
    n: int
    edges: tuple[tuple[int, int], ...]
    chromatic_number: int
    greedy_colors: int
    method: str
    found_on: str
    domain: str = "coloring"


@dataclass(frozen=True)
class AttackResult:
    expression: str
    falsified: bool
    find: AdversaryFind | None
    restarts_used: int
    steps_per_restart: int
    exact_checks: int
    best_greedy_seen: int


def attack_expression(
    expression: str,
    ns: tuple[int, ...] = (8, 9, 10, 11, 12),
    restarts: int = 24,
    steps: int = 4000,
    seed: int = 0,
    initial_temperature: float = 2.0,
    library_path: str | Path | None = DEFAULT_LIBRARY,
) -> AttackResult:
    predicate = compile_feature_expression(expression)
    rng = random.Random(seed)
    exact_checks = 0
    best_greedy = 0

    for restart in range(restarts):
        n = ns[restart % len(ns)]
        graph = _seed_graph(predicate, n, rng)
        edges = {tuple(sorted(edge)) for edge in graph.edges}
        score = _score(Graph(n, tuple(sorted(edges))), predicate)
        temperature = initial_temperature
        cooling = 0.999

        for _ in range(steps):
            u = rng.randrange(n)
            v = rng.randrange(n)
            if u == v:
                continue
            edge = (min(u, v), max(u, v))
            if edge in edges:
                edges.discard(edge)
            else:
                edges.add(edge)
            candidate = Graph(n, tuple(sorted(edges)))
            candidate_score, matches, greedy, bipartite = _score_detail(candidate, predicate)
            best_greedy = max(best_greedy, greedy if matches else 0)

            if matches and ((bipartite and greedy >= 3) or (not bipartite and greedy >= 4)):
                exact_checks += 1
                chromatic = exact_chromatic_number(candidate)
                if greedy > chromatic:
                    find = AdversaryFind(
                        expression=expression,
                        graph_id=candidate.graph_id(),
                        n=n,
                        edges=candidate.edges,
                        chromatic_number=chromatic,
                        greedy_colors=greedy,
                        method="anneal",
                        found_on=date.today().isoformat(),
                    )
                    if library_path is not None:
                        record_find(find, library_path)
                    return AttackResult(
                        expression=expression,
                        falsified=True,
                        find=find,
                        restarts_used=restart + 1,
                        steps_per_restart=steps,
                        exact_checks=exact_checks,
                        best_greedy_seen=best_greedy,
                    )

            delta = candidate_score - score
            if delta >= 0 or rng.random() < _acceptance(delta, temperature):
                score = candidate_score
            else:
                # revert the flip
                if edge in edges:
                    edges.discard(edge)
                else:
                    edges.add(edge)
            temperature *= cooling

    return AttackResult(
        expression=expression,
        falsified=False,
        find=None,
        restarts_used=restarts,
        steps_per_restart=steps,
        exact_checks=exact_checks,
        best_greedy_seen=best_greedy,
    )


def _acceptance(delta: float, temperature: float) -> float:
    import math

    if temperature <= 0:
        return 0.0
    return math.exp(delta / temperature)


def _score(graph: Graph, predicate) -> float:
    score, _, _, _ = _score_detail(graph, predicate)
    return score


def _score_detail(graph: Graph, predicate) -> tuple[float, bool, int, bool]:
    observation = observe_graph_cheap(graph)
    matches = _safe_match(predicate, observation)
    greedy = observation.greedy_colors
    score = float(greedy) + (0.0 if matches else -10.0)
    return score, matches, greedy, observation.bipartite


@dataclass(frozen=True)
class _CheapObservation:
    features: dict[str, bool | int | float]
    greedy_colors: int
    bipartite: bool


def observe_graph_cheap(graph: Graph) -> _CheapObservation:
    """Features + greedy colors without the exact chromatic number."""
    from bcv.discovery import graph_features

    features = graph_features(graph)
    return _CheapObservation(
        features=features,
        greedy_colors=greedy_degree_desc_coloring(graph),
        bipartite=bool(features["is_bipartite"]),
    )


def _safe_match(predicate, observation) -> bool:
    try:
        return bool(predicate(observation))
    except Exception:
        return False


def _seed_graph(predicate, n: int, rng: random.Random) -> Graph:
    """Start from a matching graph when possible: structured seeds, then random."""
    from bcv.graph_generalize import structured_graphs

    candidates = list(structured_graphs(n))
    rng.shuffle(candidates)
    for candidate in candidates:
        if _safe_match(predicate, observe_graph_cheap(candidate)):
            return candidate
    for _ in range(200):
        p = rng.choice((0.15, 0.3, 0.5))
        edges = tuple(
            (u, v)
            for u in range(n)
            for v in range(u + 1, n)
            if rng.random() < p
        )
        candidate = Graph(n, edges)
        if _safe_match(predicate, observe_graph_cheap(candidate)):
            return candidate
    return Graph(n, ())


def attack_with_model(
    client,
    expression: str,
    tries: int = 10,
    temperature: float = 0.7,
    library_path: str | Path | None = DEFAULT_LIBRARY,
) -> AttackResult:
    """Ask a model for counterexample GRAPHS (edge lists), verify each exactly.

    This is the frontier version of the repair task: instead of choosing a predicate
    the miner could enumerate, the model must construct an adversarial labeled
    instance. Feedback across tries tells it why the last attempt failed (wrong
    class, or greedy happened to be optimal)."""
    from bcv.local_model import LocalModelError

    predicate = compile_feature_expression(expression)
    feedback = ""
    exact_checks = 0
    best_greedy = 0
    for attempt in range(tries):
        prompt = _model_attack_prompt(expression, feedback)
        try:
            payload = client.generate_json(prompt, temperature=temperature)
        except LocalModelError as exc:
            feedback = f"Your previous reply was not valid JSON ({str(exc)[:120]}). Reply with JSON only."
            continue
        graph = _graph_from_payload(payload)
        if graph is None:
            feedback = (
                'Your previous JSON was malformed. Reply exactly as {"n": <int>, "edges": [[u, v], ...]} '
                "with 0 <= u < v < n and n between 5 and 14."
            )
            continue
        observation = observe_graph_cheap(graph)
        if not _safe_match(predicate, observation):
            relevant = {
                key: observation.features[key]
                for key in ("n", "m", "is_connected", "is_bipartite", "is_triangle_free", "max_degree")
            }
            feedback = (
                f"Your graph did not satisfy the class predicate `{expression}`. "
                f"Its features were {json.dumps(relevant, sort_keys=True)}. Fix the class membership first."
            )
            continue
        greedy = observation.greedy_colors
        best_greedy = max(best_greedy, greedy)
        exact_checks += 1
        chromatic = exact_chromatic_number(graph)
        if greedy > chromatic:
            find = AdversaryFind(
                expression=expression,
                graph_id=graph.graph_id(),
                n=graph.n,
                edges=graph.edges,
                chromatic_number=chromatic,
                greedy_colors=greedy,
                method="model",
                found_on=date.today().isoformat(),
            )
            if library_path is not None:
                record_find(find, library_path)
            return AttackResult(
                expression=expression,
                falsified=True,
                find=find,
                restarts_used=attempt + 1,
                steps_per_restart=1,
                exact_checks=exact_checks,
                best_greedy_seen=best_greedy,
            )
        feedback = (
            f"Your graph was in the class, but greedy used {greedy} colors and the chromatic number is "
            f"{chromatic}, so greedy was optimal. You need greedy to use MORE colors than the chromatic "
            "number. Remember: greedy processes vertices by descending degree, breaking ties by LOWER "
            "vertex index first, and assigns the smallest color not used by already-colored neighbors. "
            "The vertex NUMBERING is your weapon: label the graph so early-processed vertices force bad colors."
        )
    return AttackResult(
        expression=expression,
        falsified=False,
        find=None,
        restarts_used=tries,
        steps_per_restart=1,
        exact_checks=exact_checks,
        best_greedy_seen=best_greedy,
    )


def _model_attack_prompt(expression: str, feedback: str) -> str:
    feedback_block = f"\nFeedback on your previous attempt:\n{feedback}\n" if feedback else ""
    return f"""You are attacking a graph coloring conjecture.

The conjecture claims: for every simple undirected graph satisfying `{expression}`,
greedy coloring in degree-descending order (ties broken by lower vertex index first,
each vertex gets the smallest color unused by its already-colored neighbors) uses
exactly the chromatic number of colors.

Construct a COUNTEREXAMPLE: a labeled graph that satisfies `{expression}` but where
this greedy procedure uses MORE colors than the chromatic number. The vertex labels
matter because they decide the processing order.

Feature meanings: n = vertex count, m = edge count, is_connected, is_bipartite,
is_triangle_free, max_degree, girth (999 = acyclic), clique_number, num_components.
{feedback_block}
Reply with JSON only, no prose: {{"n": <int between 5 and 14>, "edges": [[u, v], ...]}}
"""


def _graph_from_payload(payload) -> Graph | None:
    if not isinstance(payload, dict):
        return None
    n = payload.get("n")
    raw_edges = payload.get("edges")
    if not isinstance(n, int) or not isinstance(raw_edges, list) or not 4 <= n <= 16:
        return None
    edges: set[tuple[int, int]] = set()
    for item in raw_edges:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            return None
        u, v = item
        if not isinstance(u, int) or not isinstance(v, int) or u == v:
            return None
        if not (0 <= u < n and 0 <= v < n):
            return None
        edges.add((min(u, v), max(u, v)))
    return Graph(n, tuple(sorted(edges)))


def record_find(find: AdversaryFind, library_path: str | Path) -> None:
    library_path = Path(library_path)
    library_path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_library(library_path)
    if any(entry.graph_id == find.graph_id and entry.expression == find.expression for entry in existing):
        return
    with library_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(find), sort_keys=True) + "\n")


def load_library(library_path: str | Path = DEFAULT_LIBRARY) -> tuple[AdversaryFind, ...]:
    library_path = Path(library_path)
    if not library_path.exists():
        return ()
    finds: list[AdversaryFind] = []
    for line in library_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        raw["edges"] = tuple(tuple(edge) for edge in raw["edges"])
        finds.append(AdversaryFind(**raw))
    return tuple(finds)


def library_graphs(library_path: str | Path = DEFAULT_LIBRARY, domain: str | None = None) -> list[Graph]:
    return [
        Graph(find.n, find.edges)
        for find in load_library(library_path)
        if domain is None or find.domain == domain
    ]


def closure_variants(graph: Graph) -> list[tuple[str, Graph]]:
    """Structural transforms that tend to preserve greedy failure.

    Every confirmed counterexample can spawn a family: append an isolated vertex
    (processed last, changes nothing upstream), append a universal vertex (shifts
    every color/choice by one level), or disjoint-union with an edge. Each variant
    must still be re-verified in its domain before entering the library."""
    n = graph.n
    variants: list[tuple[str, Graph]] = []
    variants.append(("pad_isolated", Graph(n + 1, graph.edges)))
    universal_edges = graph.edges + tuple((v, n) for v in range(n))
    variants.append(("pad_universal", Graph(n + 1, tuple(sorted(universal_edges)))))
    variants.append(("union_edge", Graph(n + 2, graph.edges + ((n, n + 1),))))
    return variants


def expand_library_closure(
    library_path: str | Path = DEFAULT_LIBRARY,
    domain=None,
    max_n: int = 14,
) -> int:
    """Grow the library with verified transforms of every find. Returns additions."""
    from bcv.domains import COLORING

    domain = domain or COLORING
    added = 0
    for find in load_library(library_path):
        if find.domain != domain.name:
            continue
        base = Graph(find.n, find.edges)
        for method, variant in closure_variants(base):
            if variant.n > max_n:
                continue
            observation = domain.observe(variant)
            if observation.greedy_is_optimal:
                continue
            new_find = AdversaryFind(
                expression=find.expression,
                graph_id=variant.graph_id(),
                n=variant.n,
                edges=variant.edges,
                chromatic_number=observation.chromatic_number,
                greedy_colors=observation.greedy_degree_desc_colors,
                method=f"closure_{method}",
                found_on=date.today().isoformat(),
                domain=domain.name,
            )
            before = len(load_library(library_path))
            record_find(new_find, library_path)
            if len(load_library(library_path)) > before:
                added += 1
    return added


def attack_survivors(
    evaluation_or_report: str | Path,
    ns: tuple[int, ...] = (8, 9, 10, 11, 12),
    restarts: int = 24,
    steps: int = 4000,
    seed: int = 0,
    library_path: str | Path = DEFAULT_LIBRARY,
    root: str | Path = ".bcv_runs/graph_adversary",
) -> list[AttackResult]:
    """Attack every surviving expression from a generalization report."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    raw = json.loads(Path(evaluation_or_report).read_text(encoding="utf-8-sig"))
    expressions = [
        result["expression"]
        for result in raw.get("results", [])
        if result.get("parseable") and result.get("survived") and not result.get("vacuous")
    ]
    results = []
    for index, expression in enumerate(expressions):
        results.append(
            attack_expression(
                expression,
                ns=ns,
                restarts=restarts,
                steps=steps,
                seed=seed + index,
                library_path=library_path,
            )
        )
    (root / "attack_results.json").write_text(
        json.dumps([asdict(result) for result in results], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Anneal for counterexamples to graph conjectures.")
    parser.add_argument("--expression", action="append", default=[])
    parser.add_argument("--report-file", help="generalization_report.json whose survivors get attacked")
    parser.add_argument("--ns", type=int, nargs="+", default=[8, 9, 10, 11, 12])
    parser.add_argument("--restarts", type=int, default=24)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--library", default=str(DEFAULT_LIBRARY))
    parser.add_argument("--root", default=".bcv_runs/graph_adversary")
    args = parser.parse_args()

    results: list[AttackResult] = []
    if args.report_file:
        results.extend(
            attack_survivors(
                args.report_file,
                ns=tuple(args.ns),
                restarts=args.restarts,
                steps=args.steps,
                seed=args.seed,
                library_path=args.library,
                root=args.root,
            )
        )
    for index, expression in enumerate(args.expression):
        results.append(
            attack_expression(
                expression,
                ns=tuple(args.ns),
                restarts=args.restarts,
                steps=args.steps,
                seed=args.seed + 1000 + index,
                library_path=args.library,
            )
        )
    print(json.dumps([asdict(result) for result in results], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
