from __future__ import annotations

import json

from bcv.benchmark import record_benchmark, run_document_corruption_benchmark
from bcv.miner import write_training_candidates
from bcv.sft_export import export_training_datasets


def test_sft_export_writes_controller_datasets(tmp_path):
    record_benchmark(tmp_path / "document", run_document_corruption_benchmark())
    write_training_candidates(tmp_path)

    result = export_training_datasets(tmp_path)

    sft_lines = [
        json.loads(line)
        for line in open(result.sft_path, encoding="utf-8").read().splitlines()
        if line.strip()
    ]
    preference_lines = [
        json.loads(line)
        for line in open(result.preference_path, encoding="utf-8").read().splitlines()
        if line.strip()
    ]

    assert result.sft_examples == 2
    assert result.preference_examples == 1
    assert sft_lines[0]["messages"][0]["role"] == "system"
    assert {"prompt", "chosen", "rejected"} <= set(preference_lines[0])

