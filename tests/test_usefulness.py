from __future__ import annotations

from bcv.usefulness import run_usefulness_benchmark


def test_usefulness_benchmark_covers_accept_recover_and_block(tmp_path):
    results = {case.name: case for case in run_usefulness_benchmark(tmp_path)}

    assert results["immediate_accept"].accepted is True
    assert results["immediate_accept"].useful_change_present is True
    assert results["reject_then_recover"].accepted is True
    assert results["reject_then_recover"].attempts == 2
    assert results["blocked_bad_model"].accepted is False
    assert results["blocked_bad_model"].citation_preserved is True

