from __future__ import annotations

from bcv.bisect_probe import run_bisect_probe


def test_bisect_probe_identifies_unsupported_claim_commit(tmp_path):
    result = run_bisect_probe(tmp_path)

    assert result.accepted is True
    assert result.bad_object_ref == "claim:unsupported-generalization"
    assert result.commit_message == "add unsupported generalization"

