"""Evaluate the 32B-generated paraphrase attack against the fingerprint tier.

The question: how often can an adversarial rewrite COLLIDE with its original
on the fingerprint corpus while genuinely differing on larger graphs? That is
the fingerprint's corpus-size blind spot, measured with a real language-model
adversary instead of mechanical rewrites — and measured as a CURVE over
corpus horizon (n<=4, n<=5, n<=6), because the blind spot is a function of
how much behavior the corpus can see.

Ground truth for "genuinely differs": disagreement anywhere on the exhaustive
n<=6 enumeration plus the n in {7,8} stress pool. A pair that agrees on all of
that is treated as equivalent at our verification horizon (stated, not hidden).

Run: $env:PYTHONPATH='src'; python scripts/eval_paraphrase_attack.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, "src")

from bcv.domains import COLORING
from bcv.graph_agent import compile_feature_expression
from bcv.leakage import behavioral_fingerprint
from bcv.refinery import _observe_all, _stress_pool

CORPUS_PATH = ".bcv_runs/pod_sync/paraphrase_attack_corpus.jsonl"
RECEIPT = "results/paraphrase_attack_receipt.json"


def truth_vector(expression: str, observations) -> tuple[bool, ...] | None:
    try:
        predicate = compile_feature_expression(expression)
        return tuple(bool(predicate(obs)) for obs in observations)
    except Exception:
        return None


def main() -> None:
    rows = [
        json.loads(line)
        for line in Path(CORPUS_PATH).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(f"attack rows: {len(rows)}")

    horizons = {n: _observe_all(COLORING, n, Path(".bcv_runs/attack_eval_tmp")) for n in (4, 5, 6)}
    stress = _stress_pool(COLORING, (7, 8), 40, 0, Path(".bcv_runs/attack_eval_tmp/nolib.jsonl"))
    truth_set = list(horizons[6]) + list(stress)

    invalid = 0
    per_horizon = {n: {"collisions": 0, "false_merges": 0} for n in horizons}
    truly_equivalent = truly_different = 0
    claim_correct = {"equivalent": [0, 0], "near_miss": [0, 0]}  # [correct, total]
    blind_spot_examples = 0

    for row in rows:
        original, rewrite, claim = row["original"], row["rewrite"], row["model_claim"]
        rewrite_truth = truth_vector(rewrite, truth_set)
        original_truth = truth_vector(original, truth_set)
        if rewrite_truth is None or original_truth is None:
            invalid += 1
            continue
        equivalent = rewrite_truth == original_truth
        truly_equivalent += equivalent
        truly_different += not equivalent
        bucket = claim_correct.setdefault(claim, [0, 0])
        bucket[1] += 1
        bucket[0] += (claim == "equivalent") == equivalent

        for n, observations in horizons.items():
            collide = behavioral_fingerprint(original, observations) == behavioral_fingerprint(
                rewrite, observations
            )
            if collide:
                per_horizon[n]["collisions"] += 1
                if not equivalent:
                    per_horizon[n]["false_merges"] += 1
                    if n == 6:
                        blind_spot_examples += 1

    valid = len(rows) - invalid
    receipt = {
        "evidence_scope": "LLM paraphrase attack vs behavioral fingerprints, "
        + datetime.now(timezone.utc).date().isoformat(),
        "attacker": "unsloth/Qwen2.5-32B-Instruct-bnb-4bit on a rented A40 (controlled infra)",
        "corpus": {
            "rows": len(rows),
            "invalid_rewrites_discarded": invalid,
            "valid": valid,
            "truly_equivalent_at_horizon": truly_equivalent,
            "truly_different": truly_different,
            "truth_horizon": "exhaustive n<=6 + stress pool n in {7,8}",
        },
        "model_claim_accuracy": {
            claim: {"correct": correct, "total": total, "rate": round(correct / total, 3) if total else None}
            for claim, (correct, total) in sorted(claim_correct.items())
        },
        "fingerprint_blind_spot_curve": {
            f"n<={n}": {
                "collisions": stats["collisions"],
                "false_merges_vs_truth_horizon": stats["false_merges"],
                "false_merge_rate_of_truly_different": round(
                    stats["false_merges"] / truly_different, 4
                )
                if truly_different
                else None,
            }
            for n, stats in sorted(per_horizon.items())
        },
        "interpretation": "a false merge at n<=6 is an attack that would evade the deployed "
        "quarantine tier while genuinely differing on larger graphs; the curve shows how the "
        "blind spot shrinks as the fingerprint corpus grows",
        "note": "expressions are from the public DSL list; no exam bank content involved",
    }
    Path(RECEIPT).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
