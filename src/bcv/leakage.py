"""Leakage fingerprints for verifier-grounded DSL items.

These fingerprints are intentionally not sold as an embedding-based semantic
oracle. They compare behavior over a supplied, versioned oracle corpus. That
makes the signal reproducible and permits an honest false-positive measurement
against labeled equivalence/non-equivalence pairs.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Iterable

from bcv.graph_agent import compile_feature_expression


@dataclass(frozen=True)
class LeakageAssessment:
    risk: float
    match: str
    matched_training_expressions: tuple[str, ...]


def behavioral_fingerprint(expression: str, observations: Iterable) -> str:
    """Hash the predicate's truth vector over a fixed oracle+stress corpus."""
    predicate = compile_feature_expression(expression)
    bits = bytearray()
    for observation in observations:
        bits.append(1 if predicate(observation) else 0)
    return hashlib.sha256(bytes(bits)).hexdigest()


def assess_dsl_leakage(
    expression: str, training_expressions: Iterable[str], observations: Iterable
) -> LeakageAssessment:
    """Layer row identity over reproducible finite-corpus behavioral matching."""
    training = set(training_expressions)
    if expression in training:
        return LeakageAssessment(1.0, "row_identity", (expression,))
    observations = tuple(observations)
    candidate = behavioral_fingerprint(expression, observations)
    matches = []
    for trained in sorted(training):
        try:
            if behavioral_fingerprint(trained, observations) == candidate:
                matches.append(trained)
        except (SyntaxError, ValueError, TypeError, KeyError):
            continue
    if matches:
        return LeakageAssessment(0.5, "behavioral_fingerprint", tuple(matches))
    return LeakageAssessment(0.0, "", ())


@dataclass(frozen=True)
class FingerprintCalibration:
    pairs: int
    expected_duplicates: int
    expected_distinct: int
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int

    @property
    def false_positive_rate(self) -> float:
        return self.false_positives / self.expected_distinct if self.expected_distinct else 0.0

    @property
    def false_negative_rate(self) -> float:
        return self.false_negatives / self.expected_duplicates if self.expected_duplicates else 0.0

    def to_dict(self) -> dict:
        result = asdict(self)
        result["false_positive_rate"] = self.false_positive_rate
        result["false_negative_rate"] = self.false_negative_rate
        return result


def calibrate_behavioral_fingerprints(
    labeled_pairs: Iterable[tuple[str, str, bool]], observations: Iterable
) -> FingerprintCalibration:
    """Measure error on caller-supplied labeled expression pairs.

    Each pair is ``(left_expression, right_expression, expected_duplicate)``.
    The caller owns those labels: unlabeled non-collisions are never converted
    into a fictional false-positive estimate.
    """
    observations = tuple(observations)
    tp = fp = tn = fn = expected_duplicate = expected_distinct = 0
    for left, right, duplicate in labeled_pairs:
        same = behavioral_fingerprint(left, observations) == behavioral_fingerprint(right, observations)
        if duplicate:
            expected_duplicate += 1
            tp += same
            fn += not same
        else:
            expected_distinct += 1
            fp += same
            tn += not same
    return FingerprintCalibration(
        pairs=expected_duplicate + expected_distinct,
        expected_duplicates=expected_duplicate,
        expected_distinct=expected_distinct,
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
    )
