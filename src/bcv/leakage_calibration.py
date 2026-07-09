"""CPU-only calibration of behavioral leakage fingerprints.

Labels are not hand-waved: equivalent pairs are mechanical valid rewrites, and
distinct pairs are admitted only after the larger exhaustive oracle corpus finds
a behavioral difference.  Smaller corpora can then be measured for the exact
failure we care about: distinct expressions that accidentally collapse to the
same finite-horizon fingerprint.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from bcv.domains import COLORING
from bcv.graph_repair_data import _candidate_expressions
from bcv.leakage import behavioral_fingerprint, calibrate_behavioral_fingerprints
from bcv.refinery import _observe_all


def equivalent_rewrites(expression: str) -> tuple[str, str]:
    """Two grammar-valid rewrites that preserve every graph-domain truth value."""
    return (
        f"not (not ({expression}))",
        f"({expression}) and (n >= 1)",
    )


def build_oracle_labeled_pairs(expressions: tuple[str, ...], observations, per_kind: int = 48):
    """Return mechanically equivalent and oracle-distinct expression pairs."""
    observations = tuple(observations)
    fingerprints = {expression: behavioral_fingerprint(expression, observations) for expression in expressions}
    pairs: list[tuple[str, str, bool]] = []
    for expression in expressions:
        for rewrite in equivalent_rewrites(expression):
            if behavioral_fingerprint(rewrite, observations) != fingerprints[expression]:
                raise AssertionError(f"rewrite was not equivalent on oracle corpus: {expression}")
            pairs.append((expression, rewrite, True))
            if sum(duplicate for _, _, duplicate in pairs) >= per_kind:
                break
        if sum(duplicate for _, _, duplicate in pairs) >= per_kind:
            break

    distinct = 0
    for index, left in enumerate(expressions):
        for right in expressions[index + 1:]:
            if fingerprints[left] == fingerprints[right]:
                continue
            pairs.append((left, right, False))
            distinct += 1
            if distinct >= per_kind:
                return tuple(pairs)
    if distinct < per_kind:
        raise RuntimeError(f"only found {distinct} oracle-distinct pairs")
    return tuple(pairs)


def run_fingerprint_calibration(
    max_n: int = 6,
    min_n: int = 3,
    pairs_per_kind: int = 48,
    root: str | Path = ".bcv_runs/fingerprint_calibration",
) -> dict:
    """Measure fingerprint calibration as the exhaustive corpus grows by n."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    gold = tuple(_observe_all(COLORING, max_n, root / "oracle"))
    expressions = _candidate_expressions()
    pairs = build_oracle_labeled_pairs(expressions, gold, pairs_per_kind)
    curve = []
    for horizon in range(min_n, max_n + 1):
        corpus = tuple(observation for observation in gold if observation.graph.n <= horizon)
        result = calibrate_behavioral_fingerprints(pairs, corpus).to_dict()
        # Every expected-distinct pair differs somewhere in the max_n oracle.
        # A collision here is therefore a finite-horizon false quarantine, not
        # an ambiguous human label.
        result.update(
            {
                "max_n": horizon,
                "observations": len(corpus),
                "finite_horizon_collision_rate": result["false_positive_rate"],
            }
        )
        curve.append(result)
    report = {
        "schema_version": 1,
        "domain": COLORING.name,
        "gold_oracle_max_n": max_n,
        "gold_observations": len(gold),
        "pair_labeling": {
            "equivalent": "mechanical double-negation or n>=1 rewrite",
            "distinct": "different truth vector on the exhaustive gold oracle",
            "pairs_per_kind": pairs_per_kind,
        },
        "curve": curve,
    }
    (root / "fingerprint_calibration.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
