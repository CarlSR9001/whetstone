from __future__ import annotations

from bcv.research import Claim, ClaimGraph, Source, run_research_contradiction_probe


def test_claim_graph_accepts_supported_noncontradictory_claim():
    graph = ClaimGraph()
    graph.add_source(Source("source:1", "Memo", "The launch date is 2026-07-15."))
    graph.add_claim(
        Claim(
            claim_id="claim:date",
            text="The launch date is 2026-07-15.",
            source_ids=("source:1",),
        )
    )

    result = graph.verify(("claim:date",))
    assert result.accepted is True
    assert graph.support_map(("claim:date",)) == {"claim:date": ("source:1",)}


def test_research_probe_rejects_unsupported_and_contradictory_claims():
    result = run_research_contradiction_probe()

    assert result.accepted is False
    assert "claim:unverified-cause" in result.unsupported_claims
    assert ("claim:rollout-july", "claim:rollout-june") in result.contradictions

