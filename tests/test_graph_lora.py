from __future__ import annotations

import json

from bcv.graph_lora import (
    _analyze_expression,
    _extract_expression,
    _original_expression,
    _verify_expression,
)


def test_extract_expression_prefers_repaired_predicate():
    output = "Replace `is_tree` with `(is_tree) and (max_degree == 3)`."

    assert _extract_expression(output) == "(is_tree) and (max_degree == 3)"


def test_extract_expression_parses_json_first():
    output = '{"repair_expression": "(is_tree) and (max_degree == 3)"}'

    assert _extract_expression(output) == "(is_tree) and (max_degree == 3)"


def test_extract_expression_rejects_json_echo_without_repair_key():
    output = '{"added_constraint": "max_degree == 3", "original_expression": "is_tree"}'

    assert _extract_expression(output) is None


def test_analyze_expression_requires_refinement_of_original():
    # Verified but unrelated to the original: not a refinement.
    unrelated = _analyze_expression("(is_tree) and (max_degree == 3)", "is_bipartite and m >= 5", max_n=5)
    assert unrelated.verified is True
    assert unrelated.refines_original is False

    # Verified and implies the original: a genuine repair.
    refinement = _analyze_expression("(is_tree) and (max_degree == 3)", "is_tree", max_n=5)
    assert refinement.verified is True
    assert refinement.refines_original is True
    assert refinement.support > 0
    assert refinement.original_true_positives >= refinement.support


def test_original_expression_reads_user_json():
    example = {
        "messages": [
            {"role": "system", "content": "repair"},
            {"role": "user", "content": json.dumps({"original_expression": "is_tree"})},
            {"role": "assistant", "content": json.dumps({"repair_expression": "is_tree and n <= 2"})},
        ]
    }

    assert _original_expression(example) == "is_tree"


def test_verify_expression_distinguishes_false_and_repaired_rules():
    parseable, verified = _verify_expression("is_tree", max_n=6)
    assert parseable is True
    assert verified is False

    parseable, verified = _verify_expression("(is_tree) and (max_degree == 3)", max_n=6)
    assert parseable is True
    assert verified is True

    parseable, verified = _verify_expression("is_tree.__class__", max_n=6)
    assert parseable is False
    assert verified is False
