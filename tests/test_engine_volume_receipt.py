from __future__ import annotations

import json
from pathlib import Path


def test_engine_volume_receipt_records_private_bank_capacity_without_positions():
    path = Path(__file__).resolve().parents[1] / "results" / "engine_volume_receipt.json"
    receipt = json.loads(path.read_text(encoding="utf-8"))
    bank = receipt["isolated_bank"]
    assert bank["promoted_total"] == bank["chess_promoted"] + bank["go_promoted"]
    assert bank["burned"] == 0
    assert receipt["milling"]["chess"][1]["milled"] == 500
    assert receipt["milling"]["go"][0]["promoted"] == 47
    serialized = json.dumps(receipt).lower()
    for private_field in ('"fen"', '"moves"', '"oracle_move"', '"item_id"'):
        assert private_field not in serialized
