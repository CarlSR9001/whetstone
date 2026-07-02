from __future__ import annotations

import json

from bcv.discovery import Graph
from bcv.domains import COLORING, MIS, greedy_independent_set_size, independence_number
from bcv.refinery import run_refinery


def test_independence_number_exact():
    pentagon = Graph(5, ((0, 1), (1, 2), (2, 3), (3, 4), (0, 4)))
    clique = Graph(4, ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)))
    path = Graph(4, ((0, 1), (1, 2), (2, 3)))
    empty = Graph(4, ())

    assert independence_number(pentagon) == 2
    assert independence_number(clique) == 1
    assert independence_number(path) == 2
    assert independence_number(empty) == 4


def test_mis_domain_observation_consistency():
    for graph in (
        Graph(5, ((0, 1), (1, 2), (2, 3), (3, 4))),
        Graph(6, ((0, 1), (0, 2), (0, 3), (4, 5))),
    ):
        obs = MIS.observe(graph)
        assert obs.greedy_degree_desc_colors == greedy_independent_set_size(graph)
        assert obs.chromatic_number == independence_number(graph)
        # Greedy independent sets are never larger than the optimum.
        assert obs.greedy_degree_desc_colors <= obs.chromatic_number
        assert obs.greedy_is_optimal == (obs.greedy_degree_desc_colors == obs.chromatic_number)


def test_refinery_smoke_both_domains(tmp_path):
    for domain in (COLORING, MIS):
        result = run_refinery(
            domain,
            max_n=5,
            stress_ns=(6,),
            samples_per_np=8,
            restarts=2,
            steps=150,
            anneal_ns=(7,),
            max_candidates=16,
            seed=3,
            root=tmp_path,
        )
        assert result.domain == domain.name
        assert result.candidates == 16
        assert result.theorems + result.falsified > 0
        theorems_md = (tmp_path / domain.name / f"THEOREMS_{domain.name}.md").read_text(encoding="utf-8")
        assert domain.claim in theorems_md
        museum = json.loads((tmp_path / domain.name / "falsification_museum.json").read_text(encoding="utf-8"))
        for item in museum:
            assert item["stage"] in ("small_n", "stress_pool", "anneal")
