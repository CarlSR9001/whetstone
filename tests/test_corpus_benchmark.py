from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import bcv.corpus_benchmark as corpus_benchmark


@dataclass
class AlwaysGoodModel:
    backend: str = "fake"
    model: str = "good"

    def generate_json(self, prompt: str, temperature: float = 0.0) -> dict[str, Any]:
        if "weekly deployment summary" in prompt:
            return {
                "operations": [
                    {
                        "target_heading": "Scope",
                        "find": "Northstar Labs will deliver the analytics dashboard described in Exhibit A.",
                        "replace": "Northstar Labs will deliver the analytics dashboard and a weekly deployment summary described in Exhibit A.",
                    }
                ]
            }
        return {"operations": []}


def test_corrupt_baseline_accumulates_invariant_loss():
    case = corpus_benchmark.sample_corpus()[0]
    metrics = corpus_benchmark._run_corrupt_baseline_case(case)

    assert len(metrics) == 3
    assert sum(len(metric.missing_invariants) for metric in metrics) > 0


def test_summary_counts_metrics():
    case = corpus_benchmark.sample_corpus()[0]
    metrics = corpus_benchmark._run_corrupt_baseline_case(case)
    summary = corpus_benchmark._summarize("baseline", metrics)

    assert summary.edit_steps == 3
    assert summary.invariant_losses > 0

