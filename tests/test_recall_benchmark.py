from __future__ import annotations

from bcv.recall_benchmark import run_recall_benchmark


def test_recall_benchmark_finds_addressable_state_after_distractors(tmp_path):
    result = run_recall_benchmark(tmp_path)

    assert result.branch_count == 3
    assert result.queries == 4
    assert result.recalled == 4
    assert result.commits > result.queries

