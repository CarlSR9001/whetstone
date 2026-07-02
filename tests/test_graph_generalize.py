from __future__ import annotations

import random

from bcv.discovery import observe_graph
from bcv.graph_generalize import (
    build_observation_pool,
    check_expressions,
    crown_graph,
    crown_graph_interleaved,
    cycle_graph,
    random_graph,
    relabeled,
    run_generalization,
)


def test_crown_graph_shape():
    crown = crown_graph(4)
    assert crown.n == 8
    degrees = crown.degrees()
    assert all(degree == 3 for degree in degrees)
    obs = observe_graph(crown)
    assert obs.chromatic_number == 2


def test_interleaved_crown_defeats_greedy_deterministically():
    crown = crown_graph_interleaved(4)
    obs = observe_graph(crown)
    assert obs.chromatic_number == 2
    assert obs.greedy_degree_desc_colors == 4
    assert obs.greedy_is_optimal is False


def test_adversarial_path_union_defeats_greedy():
    from bcv.graph_generalize import adversarial_path_union

    for n in (8, 9, 10):
        union = adversarial_path_union(n)
        obs = observe_graph(union)
        assert obs.features["max_degree_le_2"] is True
        assert obs.features["num_components"] >= 2
        assert obs.chromatic_number == 2
        assert obs.greedy_degree_desc_colors == 3
        assert obs.greedy_is_optimal is False


def test_adversarial_cycle_union_defeats_greedy():
    from bcv.graph_generalize import adversarial_cycle_union

    union = adversarial_cycle_union(10)  # bad C6 + C4, both even
    obs = observe_graph(union)
    assert obs.features["is_regular"] is True
    assert obs.features["max_degree_le_2"] is True
    assert obs.features["num_components"] == 2
    assert obs.chromatic_number == 2
    assert obs.greedy_degree_desc_colors == 3
    assert obs.greedy_is_optimal is False


def test_adversarial_tree_defeats_greedy_deterministically():
    from bcv.discovery import is_forest, is_connected
    from bcv.graph_generalize import greedy_adversarial_tree

    for n in (8, 9, 12):
        tree = greedy_adversarial_tree(n)
        assert is_forest(tree) and is_connected(tree)
        obs = observe_graph(tree)
        assert obs.chromatic_number == 2
        assert obs.greedy_degree_desc_colors == 3
        assert obs.greedy_is_optimal is False


def test_relabeled_preserves_structure():
    rng = random.Random(7)
    graph = cycle_graph(7)
    shuffled = relabeled(graph, rng)
    assert shuffled.n == graph.n
    assert len(shuffled.edges) == len(graph.edges)
    assert observe_graph(shuffled).chromatic_number == observe_graph(graph).chromatic_number


def test_random_graph_edge_probability_extremes():
    rng = random.Random(0)
    empty = random_graph(6, 0.0, rng)
    full = random_graph(6, 1.0, rng)
    assert empty.edge_count() == 0
    assert full.edge_count() == 15


def test_check_expressions_flags_unparseable_and_survivors():
    observations, random_count, structured_count = build_observation_pool(
        ns=(7,), samples_per_np=10, relabels=1, seed=1
    )
    assert random_count > 0
    assert structured_count > 0
    results = check_expressions(
        {
            "is_complete": "accepted_rule",
            "not_a_feature > 3": "repair_suggestion",
        },
        observations,
    )
    by_expression = {result.expression: result for result in results}
    assert by_expression["is_complete"].parseable is True
    assert by_expression["is_complete"].survived is True
    assert by_expression["not_a_feature > 3"].parseable is False


def test_run_generalization_writes_report(tmp_path):
    report = run_generalization(
        {"is_complete": "accepted_rule"},
        ns=(7,),
        samples_per_np=5,
        relabels=1,
        seed=2,
        root=tmp_path,
    )
    assert (tmp_path / "generalization_report.json").exists()
    assert report.expressions_checked == 1
    assert report.graphs_checked == report.results[0].graphs_checked
