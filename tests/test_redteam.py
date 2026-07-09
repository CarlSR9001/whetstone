from __future__ import annotations

from types import SimpleNamespace

from bcv.redteam import run_redteam


def test_redteam_catches_paraphrase_and_prevents_memorization_inflation(tmp_path):
    observations = [
        SimpleNamespace(graph=SimpleNamespace(n=1), features={"is_bipartite": True, "n": 1}),
        SimpleNamespace(graph=SimpleNamespace(n=2), features={"is_bipartite": False, "n": 2}),
    ]
    report = run_redteam(tmp_path, observations)
    assert report["paraphrase_attack"]["row_identity_evaded"]
    assert report["paraphrase_attack"]["leakage_match"] == "behavioral_fingerprint"
    assert not report["paraphrase_attack"]["promotion_allowed"]
    assert report["inflation_attack"] == {
        "leaked_items": 6,
        "unsafe_counterfactual_verdict": "PASS",
        "protected_verdict": "HOLD",
        "caught": True,
    }
