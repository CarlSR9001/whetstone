from __future__ import annotations

import json
from pathlib import Path


def test_redteam_receipt_records_caught_attacks_without_private_payloads():
    path = Path(__file__).resolve().parents[1] / "results" / "redteam_gate_receipt.json"
    receipt = json.loads(path.read_text(encoding="utf-8"))
    assert receipt["paraphrase_attack"]["row_identity_evaded"] is True
    assert receipt["paraphrase_attack"]["behavioral_fingerprint_matched"] is True
    assert receipt["paraphrase_attack"]["promotion_allowed"] is False
    assert receipt["inflation_attack"]["caught"] is True
    assert receipt["inflation_attack"]["unsafe_counterfactual_verdict"] == "PASS"
    assert receipt["inflation_attack"]["protected_bank_verdict"] != "PASS"
    serialized = json.dumps(receipt).lower()
    assert '"prompt"' not in serialized
    assert '"payload"' not in serialized
