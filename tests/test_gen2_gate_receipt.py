from __future__ import annotations

import json
from pathlib import Path


def test_gen2_gate_receipt_closes_retained_probe_caveat_without_leaking_private_artifacts():
    path = Path(__file__).resolve().parents[1] / "results" / "gen2_gate_receipt.json"
    receipt = json.loads(path.read_text(encoding="utf-8"))
    assert receipt["retained_coloring_probe"]["no_regression"] is True
    assert receipt["retained_coloring_probe"]["adapter_verified"] >= receipt["retained_coloring_probe"]["base_verified"]
    assert receipt["full_promotion_gate"]["verdict"] == "BLOCK"
    serialized = json.dumps(receipt).lower()
    assert '"prompt"' not in serialized
    assert '"adapter_path"' not in serialized
