"""The conjecture refinery: one command from candidate space to a theorem ledger.

Pipeline, per domain (coloring, mis, ...):

  1. enumerate candidate expressions over the shared feature DSL,
  2. verify each exhaustively on the n <= max_n universe (exact, not sampled),
  3. re-check every small-n acceptance on a hostile stress pool (random G(n,p),
     deterministic adversary families, and the domain's adversary library),
  4. mine stress-surviving repairs for everything that fell,
  5. attack every remaining survivor with simulated annealing inside its own
     predicate class (a counterexample found here is exact),
  6. expand the adversary library by closure operators over all finds,
  7. emit THEOREMS_<domain>.md — surviving conjectures with survival certificates —
     and a falsification museum recording what died where.

Nothing in the ledger is proved; a certificate records exactly how much adversarial
pressure a conjecture survived. The library persists, so every run starts from all
previous falsification knowledge.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

from bcv.discovery import Graph, GraphObservation
from bcv.domains import DOMAINS, Domain
from bcv.graph_adversary import (
    AdversaryFind,
    expand_library_closure,
    library_graphs,
    record_find,
)
from bcv.graph_agent import _atomic_refinements, compile_feature_expression
from bcv.graph_repair_data import _candidate_expressions


@dataclass(frozen=True)
class Certificate:
    expression: str
    origin: str  # candidate | repair
    support_small_n: int
    small_n: int
    stress_pool_size: int
    stress_matched: int
    anneal_restarts: int
    anneal_steps: int
    anneal_exact_checks: int
    vacuous_at_stress: bool


@dataclass(frozen=True)
class Falsification:
    expression: str
    origin: str
    stage: str  # small_n | stress_pool | anneal
    counterexample: str
    exact_value: int
    greedy_value: int


@dataclass(frozen=True)
class RefineryResult:
    domain: str
    claim: str
    candidates: int
    accepted_small_n: int
    repairs_mined: int
    theorems: int
    falsified: int
    library_additions: int
    theorems_path: str
    museum_path: str


def run_refinery(
    domain: Domain,
    max_n: int = 6,
    stress_ns: tuple[int, ...] = (7, 8, 10),
    samples_per_np: int = 80,
    restarts: int = 10,
    steps: int = 1500,
    anneal_ns: tuple[int, ...] = (8, 9, 10, 11),
    seed: int = 0,
    max_candidates: int = 64,
    root: str | Path = ".bcv_runs/refinery",
    library_path: str | Path | None = None,
) -> RefineryResult:
    root = Path(root) / domain.name
    root.mkdir(parents=True, exist_ok=True)
    library_path = Path(library_path) if library_path else root / "adversary_library.jsonl"

    small_observations = _observe_all(domain, max_n, root)
    stress_pool = _stress_pool(domain, stress_ns, samples_per_np, seed, library_path)

    candidates = list(_candidate_expressions()[:max_candidates])
    certificates: list[Certificate] = []
    museum: list[Falsification] = []
    accepted_small = 0
    repairs_mined = 0

    survivors: list[tuple[str, str]] = []  # (expression, origin)
    for expression in candidates:
        verdict = _verify(expression, small_observations)
        if verdict is None:
            continue
        matched, failure = verdict
        if failure is not None:
            museum.append(_falsification(expression, "candidate", "small_n", failure))
            repair = _mine_stress_repair(expression, small_observations, stress_pool)
            if repair is not None:
                repairs_mined += 1
                survivors.append((repair, "repair"))
            continue
        if matched == 0:
            continue
        accepted_small += 1
        survivors.append((expression, "candidate"))

    theorems: list[Certificate] = []
    rng = random.Random(seed)
    for expression, origin in survivors:
        stress = _check_pool(expression, stress_pool)
        if stress is None:
            continue
        stress_matched, stress_failure = stress
        if stress_failure is not None:
            museum.append(_falsification(expression, origin, "stress_pool", stress_failure))
            continue
        attack = _anneal_attack(
            domain,
            expression,
            ns=anneal_ns,
            restarts=restarts,
            steps=steps,
            rng=rng,
            library_path=library_path,
        )
        if attack is not None:
            museum.append(_falsification(expression, origin, "anneal", attack))
            continue
        small_support = sum(
            1 for obs in small_observations if _matches(expression, obs)
        )
        theorems.append(
            Certificate(
                expression=expression,
                origin=origin,
                support_small_n=small_support,
                small_n=max_n,
                stress_pool_size=len(stress_pool),
                stress_matched=stress_matched,
                anneal_restarts=restarts,
                anneal_steps=steps,
                anneal_exact_checks=restarts * steps,
                vacuous_at_stress=stress_matched == 0,
            )
        )

    library_additions = expand_library_closure(library_path, domain=domain, max_n=max(anneal_ns) + 2)

    theorems_path = root / f"THEOREMS_{domain.name}.md"
    museum_path = root / "falsification_museum.json"
    _write_theorems(theorems_path, domain, theorems, max_n, stress_ns)
    museum_path.write_text(
        json.dumps([asdict(item) for item in museum], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    result = RefineryResult(
        domain=domain.name,
        claim=domain.claim,
        candidates=len(candidates),
        accepted_small_n=accepted_small,
        repairs_mined=repairs_mined,
        theorems=len(theorems),
        falsified=len(museum),
        library_additions=library_additions,
        theorems_path=str(theorems_path),
        museum_path=str(museum_path),
    )
    (root / "refinery_result.json").write_text(
        json.dumps(asdict(result), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return result


# --------------------------------------------------------------------- stages

_OBSERVATION_CACHE: dict[tuple[str, int], list[GraphObservation]] = {}


def _observe_all(domain: Domain, max_n: int, root: Path) -> list[GraphObservation]:
    key = (domain.name, max_n)
    if key not in _OBSERVATION_CACHE:
        from bcv.discovery import enumerate_graphs

        _OBSERVATION_CACHE[key] = [domain.observe(graph) for graph in enumerate_graphs(max_n)]
    return _OBSERVATION_CACHE[key]


def _stress_pool(
    domain: Domain,
    stress_ns: tuple[int, ...],
    samples_per_np: int,
    seed: int,
    library_path: Path,
) -> list[GraphObservation]:
    from bcv.graph_generalize import random_graph, relabeled, structured_graphs

    rng = random.Random(seed)
    seen: set[str] = set()
    graphs: list[Graph] = []

    def add(graph: Graph) -> None:
        key = graph.graph_id()
        if key not in seen:
            seen.add(key)
            graphs.append(graph)

    for n in stress_ns:
        for base in structured_graphs(n):
            add(base)
            for _ in range(3):
                add(relabeled(base, rng))
        for p in (0.2, 0.35, 0.5, 0.65, 0.8):
            for _ in range(samples_per_np):
                add(random_graph(n, p, rng))
    for graph in library_graphs(library_path, domain=domain.name):
        add(graph)
    return [domain.observe(graph) for graph in graphs]


def _matches(expression: str, observation: GraphObservation) -> bool:
    try:
        return bool(compile_feature_expression(expression)(observation))
    except Exception:
        return False


def _verify(expression: str, observations) -> tuple[int, GraphObservation | None] | None:
    try:
        predicate = compile_feature_expression(expression)
    except (SyntaxError, ValueError, TypeError, KeyError):
        return None
    matched = 0
    for observation in observations:
        if predicate(observation):
            matched += 1
            if not observation.greedy_is_optimal:
                return matched, observation
    return matched, None


def _check_pool(expression: str, pool) -> tuple[int, GraphObservation | None] | None:
    return _verify(expression, pool)


def _mine_stress_repair(expression: str, observations, pool) -> str | None:
    """Best conjunctive repair with no counterexample at small n or in the pool."""
    try:
        predicate = compile_feature_expression(expression)
    except (SyntaxError, ValueError, TypeError, KeyError):
        return None
    base_matches = [obs for obs in observations if predicate(obs)]
    false_matches = [obs for obs in base_matches if not obs.greedy_is_optimal]
    if not false_matches:
        return None
    best: tuple[int, str] | None = None
    for atom in _atomic_refinements(list(observations)):
        try:
            atom_predicate = compile_feature_expression(atom)
        except (SyntaxError, ValueError, TypeError, KeyError):
            continue
        if any(atom_predicate(obs) for obs in false_matches):
            continue
        support = sum(1 for obs in base_matches if atom_predicate(obs))
        if not support or (best is not None and support <= best[0]):
            continue
        repaired = f"({expression}) and ({atom})"
        combined = compile_feature_expression(repaired)
        if any(combined(obs) and not obs.greedy_is_optimal for obs in pool):
            continue
        best = (support, repaired)
    return best[1] if best else None


def _anneal_attack(
    domain: Domain,
    expression: str,
    ns: tuple[int, ...],
    restarts: int,
    steps: int,
    rng: random.Random,
    library_path: Path,
) -> GraphObservation | None:
    """Domain-generic annealing: score = predicate match + failure gap, exact per step."""
    try:
        predicate = compile_feature_expression(expression)
    except (SyntaxError, ValueError, TypeError, KeyError):
        return None
    import math

    for restart in range(restarts):
        n = ns[restart % len(ns)]
        edges = set(_seed_edges(predicate, domain, n, rng))
        observation = domain.observe(Graph(n, tuple(sorted(edges))))
        score = _attack_score(domain, predicate, observation)
        temperature = 2.0
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
            candidate = domain.observe(Graph(n, tuple(sorted(edges))))
            if not candidate.greedy_is_optimal and _safe(predicate, candidate):
                find = AdversaryFind(
                    expression=expression,
                    graph_id=candidate.graph.graph_id(),
                    n=n,
                    edges=candidate.graph.edges,
                    chromatic_number=candidate.chromatic_number,
                    greedy_colors=candidate.greedy_degree_desc_colors,
                    method="anneal",
                    found_on=date.today().isoformat(),
                    domain=domain.name,
                )
                record_find(find, library_path)
                return candidate
            candidate_score = _attack_score(domain, predicate, candidate)
            delta = candidate_score - score
            if delta >= 0 or rng.random() < math.exp(delta / max(temperature, 1e-6)):
                score = candidate_score
            else:
                if edge in edges:
                    edges.discard(edge)
                else:
                    edges.add(edge)
            temperature *= 0.999
    return None


def _attack_score(domain: Domain, predicate, observation: GraphObservation) -> float:
    return float(domain.gap(observation)) * 5.0 + (0.0 if _safe(predicate, observation) else -10.0)


def _safe(predicate, observation) -> bool:
    try:
        return bool(predicate(observation))
    except Exception:
        return False


def _seed_edges(predicate, domain: Domain, n: int, rng: random.Random):
    from bcv.graph_generalize import structured_graphs

    candidates = list(structured_graphs(n))
    rng.shuffle(candidates)
    for candidate in candidates:
        if _safe(predicate, domain.observe(candidate)):
            return candidate.edges
    for _ in range(100):
        p = rng.choice((0.15, 0.3, 0.5))
        edges = tuple((u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p)
        if _safe(predicate, domain.observe(Graph(n, edges))):
            return edges
    return ()


def _falsification(expression: str, origin: str, stage: str, observation: GraphObservation) -> Falsification:
    return Falsification(
        expression=expression,
        origin=origin,
        stage=stage,
        counterexample=observation.graph.graph_id(),
        exact_value=observation.chromatic_number,
        greedy_value=observation.greedy_degree_desc_colors,
    )


def _write_theorems(path: Path, domain: Domain, theorems: list[Certificate], max_n: int, stress_ns) -> None:
    lines = [
        f"# Machine-validated conjectures: {domain.name}",
        "",
        f"Claim template: for every graph satisfying the predicate, {domain.claim}.",
        "",
        f"Nothing here is proved. Each entry survived: exhaustive verification on all",
        f"graphs with n <= {max_n}, a hostile stress pool at n in {tuple(stress_ns)} (random,",
        "deterministic adversary families, adversary library), and simulated-annealing",
        "counterexample search inside its own predicate class.",
        "",
        "| predicate | origin | small-n support | stress matches | anneal budget |",
        "| --- | --- | --- | --- | --- |",
    ]
    for cert in sorted(theorems, key=lambda c: -c.stress_matched):
        vacuous = " (vacuous)" if cert.vacuous_at_stress else ""
        lines.append(
            f"| `{cert.expression}` | {cert.origin} | {cert.support_small_n} "
            f"| {cert.stress_matched}{vacuous} | {cert.anneal_restarts}x{cert.anneal_steps} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the conjecture refinery for a domain.")
    parser.add_argument("--domain", choices=sorted(DOMAINS), default="coloring")
    parser.add_argument("--max-n", type=int, default=6)
    parser.add_argument("--stress-ns", type=int, nargs="+", default=[7, 8, 10])
    parser.add_argument("--samples-per-np", type=int, default=80)
    parser.add_argument("--restarts", type=int, default=10)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--anneal-ns", type=int, nargs="+", default=[8, 9, 10, 11])
    parser.add_argument("--max-candidates", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--root", default=".bcv_runs/refinery")
    parser.add_argument("--library", default=None)
    args = parser.parse_args()
    result = run_refinery(
        DOMAINS[args.domain],
        max_n=args.max_n,
        stress_ns=tuple(args.stress_ns),
        samples_per_np=args.samples_per_np,
        restarts=args.restarts,
        steps=args.steps,
        anneal_ns=tuple(args.anneal_ns),
        seed=args.seed,
        max_candidates=args.max_candidates,
        root=args.root,
        library_path=args.library,
    )
    print(json.dumps(asdict(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
