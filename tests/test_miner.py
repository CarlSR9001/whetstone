from __future__ import annotations

from bcv.benchmark import record_benchmark, run_document_corruption_benchmark
from bcv.miner import mine_training_candidates, write_training_candidates


def test_miner_promotes_passes_and_failed_repairs(tmp_path):
    record_benchmark(tmp_path / "document", run_document_corruption_benchmark())

    candidates = mine_training_candidates(tmp_path)
    labels = [candidate.label for candidate in candidates]

    assert "verified_positive" in labels
    assert "repair_required" in labels

    path = write_training_candidates(tmp_path)
    assert path.exists()
    assert len(path.read_text(encoding="utf-8").splitlines()) == len(candidates)

