from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from bcv.schema import Event, TestResult
from bcv.store import CognitiveStore


@dataclass(frozen=True)
class BisectProbeResult:
    identified_commit_id: str | None
    bad_object_ref: str
    commit_message: str | None
    accepted: bool


def run_bisect_probe(root: str | Path) -> BisectProbeResult:
    store = CognitiveStore(root)
    store.init()
    branch = "experiment/bisect"
    if not store.branch_exists(branch):
        store.create_branch(branch, from_branch="main")

    store.commit(
        branch,
        "add supported premise",
        [
            Event(
                event_type="claim_added",
                actor="model",
                message="Supported premise entered.",
                output_refs=("claim:supported-premise",),
                evidence_refs=("source:known-good",),
                tests=(TestResult("claim_has_evidence", "pass"),),
            )
        ],
    )
    bad = store.commit(
        branch,
        "add unsupported generalization",
        [
            Event(
                event_type="claim_added",
                actor="model",
                message="Unsupported generalization entered.",
                output_refs=("claim:unsupported-generalization",),
                tests=(TestResult("claim_has_evidence", "fail", "missing evidence_refs"),),
            )
        ],
    )
    store.commit(
        branch,
        "build final answer from claims",
        [
            Event(
                event_type="verifier_result",
                actor="verifier",
                message="Final answer failed support check.",
                input_refs=("claim:supported-premise", "claim:unsupported-generalization"),
                output_refs=("answer:final",),
                tests=(TestResult("answer_claims_supported", "fail"),),
            )
        ],
    )

    def fails_after(commits):
        return any(
            "claim:unsupported-generalization" in event.output_refs
            for commit in commits
            for event in commit.events
        )

    identified = store.bisect(branch, fails_after)
    result = BisectProbeResult(
        identified_commit_id=identified.commit_id if identified else None,
        bad_object_ref="claim:unsupported-generalization",
        commit_message=identified.message if identified else None,
        accepted=identified is not None and identified.commit_id == bad.commit_id,
    )
    store.commit(
        branch,
        "record bisect result",
        [
            Event(
                event_type="verifier_result",
                actor="runtime",
                message=result.commit_message or "no bad commit identified",
                output_refs=(result.bad_object_ref,),
                tests=(
                    TestResult(
                        "bisect_identifies_bad_commit",
                        "pass" if result.accepted else "fail",
                        str(asdict(result)),
                    ),
                ),
            )
        ],
    )
    return result

