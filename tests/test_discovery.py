from __future__ import annotations

from bcv.discovery import (
    ACYCLIC_GIRTH,
    Graph,
    clique_number,
    exact_chromatic_number,
    girth,
    greedy_degree_desc_coloring,
    num_components,
    run_graph_discovery,
)


def test_new_structural_features():
    triangle_plus_isolated = Graph(4, ((0, 1), (0, 2), (1, 2)))
    assert num_components(triangle_plus_isolated) == 2
    assert clique_number(triangle_plus_isolated) == 3
    assert girth(triangle_plus_isolated) == 3

    path = Graph(4, ((0, 1), (1, 2), (2, 3)))
    assert num_components(path) == 1
    assert clique_number(path) == 2
    assert girth(path) == ACYCLIC_GIRTH

    square = Graph(4, ((0, 1), (1, 2), (2, 3), (0, 3)))
    assert girth(square) == 4
    assert clique_number(square) == 2

    pentagon = Graph(5, ((0, 1), (1, 2), (2, 3), (3, 4), (0, 4)))
    assert girth(pentagon) == 5

    empty = Graph(3, ())
    assert num_components(empty) == 3
    assert clique_number(empty) == 1
    assert girth(empty) == ACYCLIC_GIRTH


def test_exact_chromatic_number_small_graphs():
    empty = Graph(3, ())
    triangle = Graph(3, ((0, 1), (0, 2), (1, 2)))
    path = Graph(3, ((0, 1), (1, 2)))

    assert exact_chromatic_number(empty) == 1
    assert exact_chromatic_number(path) == 2
    assert exact_chromatic_number(triangle) == 3


def test_greedy_degree_desc_coloring_triangle():
    triangle = Graph(3, ((0, 1), (0, 2), (1, 2)))

    assert greedy_degree_desc_coloring(triangle) == 3


def test_graph_discovery_promotes_and_rejects_rules(tmp_path):
    result = run_graph_discovery(max_n=6, root=tmp_path)

    assert result.graphs_checked > 0
    assert result.promoted_rules
    assert result.rejected_rules
    assert all(rule.precision == 1.0 for rule in result.promoted_rules)
    assert any(rule.counterexamples for rule in result.rejected_rules)
