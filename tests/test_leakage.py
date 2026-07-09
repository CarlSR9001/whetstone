from __future__ import annotations

from types import SimpleNamespace

from bcv.leakage import behavioral_fingerprint, calibrate_behavioral_fingerprints


def _observations():
    return [
        SimpleNamespace(features={"is_tree": True, "is_bipartite": True, "max_degree": 2}),
        SimpleNamespace(features={"is_tree": False, "is_bipartite": True, "max_degree": 4}),
        SimpleNamespace(features={"is_tree": False, "is_bipartite": False, "max_degree": 4}),
    ]


def test_behavioral_fingerprint_catches_reordered_dsl_without_collapsing_distinct_rules():
    observations = _observations()
    left = "is_tree and is_bipartite"
    reordered = "is_bipartite and is_tree"
    distinct = "is_bipartite and max_degree >= 3"
    assert behavioral_fingerprint(left, observations) == behavioral_fingerprint(reordered, observations)
    assert behavioral_fingerprint(left, observations) != behavioral_fingerprint(distinct, observations)


def test_calibration_reports_measured_error_without_guessing_labels():
    result = calibrate_behavioral_fingerprints(
        [
            ("is_tree and is_bipartite", "is_bipartite and is_tree", True),
            ("is_tree", "is_bipartite", False),
        ],
        _observations(),
    )
    assert result.to_dict() == {
        "pairs": 2, "expected_duplicates": 1, "expected_distinct": 1,
        "true_positives": 1, "false_positives": 0, "true_negatives": 1,
        "false_negatives": 0, "false_positive_rate": 0.0, "false_negative_rate": 0.0,
    }
