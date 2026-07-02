from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4


class ResearchError(ValueError):
    pass


@dataclass(frozen=True)
class Source:
    source_id: str
    title: str
    text: str


@dataclass(frozen=True)
class Claim:
    text: str
    claim_id: str = field(default_factory=lambda: f"claim_{uuid4().hex}")
    source_ids: tuple[str, ...] = ()
    contradicts: tuple[str, ...] = ()
    confidence: float = 0.5


@dataclass(frozen=True)
class ResearchVerification:
    accepted: bool
    unsupported_claims: tuple[str, ...]
    contradictions: tuple[tuple[str, str], ...]
    final_claim_ids: tuple[str, ...]


class ClaimGraph:
    def __init__(self) -> None:
        self.sources: dict[str, Source] = {}
        self.claims: dict[str, Claim] = {}

    def add_source(self, source: Source) -> None:
        if source.source_id in self.sources:
            raise ResearchError(f"duplicate source: {source.source_id}")
        self.sources[source.source_id] = source

    def add_claim(self, claim: Claim) -> None:
        if claim.claim_id in self.claims:
            raise ResearchError(f"duplicate claim: {claim.claim_id}")
        missing_sources = [source_id for source_id in claim.source_ids if source_id not in self.sources]
        if missing_sources:
            raise ResearchError(f"claim references unknown sources: {missing_sources}")
        self.claims[claim.claim_id] = claim

    def verify(self, final_claim_ids: tuple[str, ...] | None = None) -> ResearchVerification:
        selected_ids = final_claim_ids or tuple(self.claims)
        unsupported: list[str] = []
        contradictions: list[tuple[str, str]] = []

        for claim_id in selected_ids:
            claim = self.claims[claim_id]
            if not claim.source_ids:
                unsupported.append(claim_id)
            for other_id in claim.contradicts:
                if other_id in selected_ids:
                    contradictions.append(tuple(sorted((claim_id, other_id))))

        unique_contradictions = tuple(sorted(set(contradictions)))
        return ResearchVerification(
            accepted=not unsupported and not unique_contradictions,
            unsupported_claims=tuple(unsupported),
            contradictions=unique_contradictions,
            final_claim_ids=tuple(selected_ids),
        )

    def support_map(self, claim_ids: tuple[str, ...] | None = None) -> dict[str, tuple[str, ...]]:
        selected_ids = claim_ids or tuple(self.claims)
        return {
            claim_id: self.claims[claim_id].source_ids
            for claim_id in selected_ids
        }


def run_research_contradiction_probe() -> ResearchVerification:
    graph = ClaimGraph()
    graph.add_source(
        Source(
            source_id="source:a",
            title="Deployment Memo A",
            text="The rollout completed on 2026-06-20 with no payment change.",
        )
    )
    graph.add_source(
        Source(
            source_id="source:b",
            title="Deployment Memo B",
            text="The rollout moved to 2026-07-15 after payment terms changed.",
        )
    )

    claim_a = Claim(
        claim_id="claim:rollout-june",
        text="The rollout completed on 2026-06-20.",
        source_ids=("source:a",),
        contradicts=("claim:rollout-july",),
        confidence=0.8,
    )
    claim_b = Claim(
        claim_id="claim:rollout-july",
        text="The rollout moved to 2026-07-15.",
        source_ids=("source:b",),
        contradicts=("claim:rollout-june",),
        confidence=0.8,
    )
    claim_c = Claim(
        claim_id="claim:unverified-cause",
        text="The delay happened because the vendor forgot the launch checklist.",
        source_ids=(),
        confidence=0.4,
    )
    graph.add_claim(claim_a)
    graph.add_claim(claim_b)
    graph.add_claim(claim_c)
    return graph.verify(
        (
            "claim:rollout-june",
            "claim:rollout-july",
            "claim:unverified-cause",
        )
    )

