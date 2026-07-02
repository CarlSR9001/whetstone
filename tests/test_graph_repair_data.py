from __future__ import annotations

import json

from bcv.graph_agent import _observations_for, compile_feature_expression
from bcv.graph_generalize import build_observation_pool
from bcv.graph_repair_data import _stress_best_repair, build_hard_graph_repair_dataset


def test_hard_dataset_has_no_leaked_constraint_and_disjoint_groups(tmp_path):
    result = build_hard_graph_repair_dataset(
        root=tmp_path,
        max_n=6,
        max_proposals=24,
        heldout_groups=3,
        evidence_examples=3,
    )
    assert result.repair_groups > 0
    assert result.train_examples > 0
    assert result.heldout_examples > 0
    assert result.train_examples + result.heldout_examples == result.repair_groups

    def load(path):
        return [json.loads(line) for line in (tmp_path / path).read_text(encoding="utf-8").splitlines() if line.strip()]

    train = load("hard_train.jsonl")
    heldout = load("hard_heldout.jsonl")

    def originals(rows):
        return {json.loads(row["messages"][1]["content"])["original_expression"] for row in rows}

    assert originals(train).isdisjoint(originals(heldout))

    for row in train + heldout:
        user_payload = json.loads(row["messages"][1]["content"])
        assistant_payload = json.loads(row["messages"][2]["content"])
        # The answer constraint must not be leaked in the prompt.
        assert "added_constraint" not in user_payload
        assert "repair_expression" not in user_payload
        assert user_payload["false_positive_count"] > 0
        assert user_payload["counterexamples"]
        # Evidence rows must be pairwise distinct, not relabelings of one graph.
        signatures = [json.dumps(item, sort_keys=True) for item in user_payload["counterexamples"]]
        assert len(signatures) == len(set(signatures))
        assert assistant_payload["repair_expression"].startswith(f"({user_payload['original_expression']})")


def test_stress_best_repair_survives_pool():
    observations = _observations_for(6)
    pool, _, _ = build_observation_pool(ns=(7,), samples_per_np=20, relabels=1, seed=3)
    repair = _stress_best_repair("is_tree", observations, pool)
    assert repair is not None
    assert repair.startswith("(is_tree)")
    predicate = compile_feature_expression(repair)
    small_matches = [obs for obs in observations if predicate(obs)]
    assert small_matches
    assert all(obs.greedy_is_optimal for obs in small_matches)
    assert all(obs.greedy_is_optimal for obs in pool if predicate(obs))
