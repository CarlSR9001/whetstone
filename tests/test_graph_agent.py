from __future__ import annotations

import pytest

from bcv.discovery import Graph, observe_graph
from bcv.graph_agent import ProposedRule, compile_feature_expression, evaluate_proposals


def test_compile_feature_expression_accepts_boolean_and_comparison():
    obs = observe_graph(Graph(3, ((0, 1), (1, 2))))
    predicate = compile_feature_expression("is_tree and max_degree <= 2 and not is_complete")

    assert predicate(obs) is True


def test_compile_feature_expression_rejects_unknown_features_and_calls():
    with pytest.raises(ValueError, match="unknown feature"):
        compile_feature_expression("made_up_feature")

    with pytest.raises(ValueError, match="unsupported expression node"):
        compile_feature_expression("is_tree.__class__")


def test_evaluate_proposals_splits_accepted_rejected_and_invalid(tmp_path):
    result = evaluate_proposals(
        (
            ProposedRule(
                "complete_graphs",
                "Complete graphs should be exact for degree-descending greedy.",
                "is_complete",
            ),
            ProposedRule(
                "all_trees",
                "All trees should be exact for degree-descending greedy.",
                "is_tree",
            ),
            ProposedRule(
                "invalid",
                "Bad expression.",
                "unknown_feature",
            ),
        ),
        max_n=6,
        root=tmp_path,
    )

    assert [rule.name for rule in result.accepted_rules] == ["complete_graphs"]
    assert [rule.name for rule in result.rejected_rules] == ["all_trees"]
    assert result.rejected_rules[0].false_positives > 0
    assert result.repair_suggestions
    assert result.repair_suggestions[0]["original_name"] == "all_trees"
    assert result.repair_suggestions[0]["support"] > 0
    assert result.invalid_rules[0]["name"] == "invalid"
    assert (tmp_path / "proposal_evaluation.json").exists()
    assert (tmp_path / "repair_sft.jsonl").exists()
