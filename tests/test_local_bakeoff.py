from __future__ import annotations

import json

import pytest

from bcv.gate import GatePolicy
from bcv.local_bakeoff import run_local_code_bakeoff


class BlankLocalCandidate:
    provider = "fixture/local"
    is_external = False
    backend = "fixture"

    def generate_text(self, prompt: str, temperature: float = 0.0) -> str:
        return ""


class ExternalCandidate(BlankLocalCandidate):
    is_external = True


def test_local_bakeoff_mints_fresh_seeded_cohort_and_writes_gate_receipt(tmp_path):
    summary = run_local_code_bakeoff(
        tmp_path / "run", "small", BlankLocalCandidate(), "large", BlankLocalCandidate(),
        items=3, seed=7, candidate_repeats=2, policy=GatePolicy(require_retained_probe=False),
    )
    assert summary["mode"] == "local_only"
    assert summary["mint"]["promoted"] == 3
    assert len(summary["candidate_runs"]) == 2
    assert summary["gate"]["verdict"] == "HOLD"
    assert len(summary["gate"]["item_set_sha256"]) == 64
    saved = json.loads((tmp_path / "run" / "local_bakeoff.json").read_text(encoding="utf-8"))
    assert saved["selection_seed"] == 7
    assert (tmp_path / "run" / "gate" / "promotion_gate.html").exists()
    with pytest.raises(ValueError, match="not empty"):
        run_local_code_bakeoff(tmp_path / "run", "small", BlankLocalCandidate(), "large", BlankLocalCandidate())


def test_local_bakeoff_refuses_external_candidates(tmp_path):
    with pytest.raises(ValueError, match="in-boundary"):
        run_local_code_bakeoff(tmp_path / "run", "small", ExternalCandidate(), "large", BlankLocalCandidate())
