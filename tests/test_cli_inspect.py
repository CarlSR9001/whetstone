from __future__ import annotations

import json

from bcv.cli import main
from bcv.product_tools import examples


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def test_inspect_cli_writes_receipt_and_clean_exam(tmp_path, capsys):
    sample = examples()["inspector"]
    paths = {}
    for key in ("exam", "exposure", "baseline", "candidate"):
        paths[key] = tmp_path / f"{key}.json"
        write_json(paths[key], sample[key])
    receipt = tmp_path / "receipt.json"
    clean = tmp_path / "clean.jsonl"
    code = main([
        "inspect",
        "--exam", str(paths["exam"]),
        "--exposure", str(paths["exposure"]),
        "--baseline", str(paths["baseline"]),
        "--candidate", str(paths["candidate"]),
        "--out", str(receipt),
        "--clean-exam-out", str(clean),
    ])
    assert code == 0
    output = capsys.readouterr().out
    assert "QUARANTINED: 1" in output
    assert "DECISION: PASS" in output
    assert json.loads(receipt.read_text())["gate"]["verdict"] == "PASS"
    assert len(clean.read_text().splitlines()) == 7


def test_inspect_cli_exit_code_blocks_higher_aggregate_regression(tmp_path):
    exam = [{"item_id": item} for item in ("a", "b", "c")]
    values = {
        "exam": exam,
        "exposure": [],
        "baseline": {"a": True, "b": False, "c": False},
        "candidate": {"a": False, "b": True, "c": True},
    }
    paths = {}
    for key, value in values.items():
        paths[key] = tmp_path / f"{key}.json"
        write_json(paths[key], value)
    code = main([
        "inspect",
        "--exam", str(paths["exam"]),
        "--exposure", str(paths["exposure"]),
        "--baseline", str(paths["baseline"]),
        "--candidate", str(paths["candidate"]),
    ])
    assert code == 3
