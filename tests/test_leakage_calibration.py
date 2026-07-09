from __future__ import annotations

from types import SimpleNamespace

from bcv.leakage_calibration import build_oracle_labeled_pairs, equivalent_rewrites


def test_equivalent_rewrites_are_explicit_and_valid():
    double_negation, redundant_atom = equivalent_rewrites("is_tree")
    assert double_negation == "not (not (is_tree))"
    assert redundant_atom == "(is_tree) and (n >= 1)"


def test_pair_builder_uses_oracle_behavior_to_admit_distinct_pairs():
    observations = [
        SimpleNamespace(graph=SimpleNamespace(n=1), features={"is_tree": True, "is_bipartite": True, "n": 1}),
        SimpleNamespace(graph=SimpleNamespace(n=2), features={"is_tree": False, "is_bipartite": True, "n": 2}),
    ]
    pairs = build_oracle_labeled_pairs(("is_tree", "is_bipartite"), observations, per_kind=1)
    assert pairs[0][2] is True
    assert pairs[-1][2] is False
