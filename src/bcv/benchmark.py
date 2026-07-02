from __future__ import annotations

import json
from pathlib import Path
from dataclasses import asdict, dataclass

from bcv.markdown_editor import (
    MarkdownPatch,
    PatchError,
    PatchOperation,
    apply_markdown_patch,
    verify_conservation,
)
from bcv.schema import Event, TestResult
from bcv.store import CognitiveStore


SAMPLE_DOCUMENT = """# Vendor Agreement

## Parties

This agreement is between Northstar Labs and Meridian Health.

## Payment

Meridian Health will pay invoice INV-2049 within 30 days of receipt.

## Scope

Northstar Labs will deliver the analytics dashboard described in Exhibit A.

## Term

The agreement starts on 2026-07-01 and ends on 2026-12-31.

## Citation

The privacy obligations follow the Data Processing Addendum [DPA-17].
"""


@dataclass(frozen=True)
class BenchmarkResult:
    candidate: str
    accepted: bool
    failure: str | None
    accidental_deletions: int
    number_drift: int
    section_drift: int


def run_document_corruption_benchmark() -> list[BenchmarkResult]:
    patch = MarkdownPatch(
        operations=(
            PatchOperation(
                target_heading="Scope",
                find="Northstar Labs will deliver the analytics dashboard described in Exhibit A.",
                replace="Northstar Labs will deliver the analytics dashboard and a weekly deployment summary described in Exhibit A.",
            ),
        ),
        reason="Add weekly deployment summary to scope.",
    )
    clean = apply_markdown_patch(SAMPLE_DOCUMENT, patch)
    corrupt = clean.replace("30 days", "45 days").replace("[DPA-17]", "")
    corrupt = corrupt.replace("## Term", "## Duration")

    return [
        evaluate_candidate("patch_only_editor", SAMPLE_DOCUMENT, clean, {"Scope"}),
        evaluate_candidate("corrupt_full_rewrite", SAMPLE_DOCUMENT, corrupt, {"Scope"}),
    ]


def record_benchmark(root: str | Path, results: list[BenchmarkResult]) -> CognitiveStore:
    store = CognitiveStore(root)
    store.init()
    branch = "experiment/document-conservation"
    if not store.branch_exists(branch):
        store.create_branch(branch, from_branch="main")

    for result in results:
        store.commit(
            branch,
            f"record benchmark result: {result.candidate}",
            [
                Event(
                    event_type="verifier_result",
                    actor="verifier",
                    message=result.failure or "candidate accepted",
                    output_refs=(f"candidate:{result.candidate}",),
                    tests=(
                        TestResult(
                            test_id="document_conservation",
                            result="pass" if result.accepted else "fail",
                            details=json.dumps(asdict(result), sort_keys=True),
                        ),
                    ),
                    confidence_before=0.5,
                    confidence_after=1.0 if result.accepted else 0.0,
                )
            ],
        )
    return store


def evaluate_candidate(
    candidate: str,
    original: str,
    updated: str,
    changed_headings: set[str],
) -> BenchmarkResult:
    failure = None
    accepted = True
    try:
        verify_conservation(original, updated, changed_headings)
    except PatchError as exc:
        accepted = False
        failure = str(exc)

    return BenchmarkResult(
        candidate=candidate,
        accepted=accepted,
        failure=failure,
        accidental_deletions=_count_missing_markers(original, updated),
        number_drift=_count_number_drift(original, updated),
        section_drift=_count_section_drift(original, updated),
    )


def _count_missing_markers(original: str, updated: str) -> int:
    markers = ["[DPA-17]", "INV-2049", "Northstar Labs", "Meridian Health"]
    return sum(1 for marker in markers if marker in original and marker not in updated)


def _count_number_drift(original: str, updated: str) -> int:
    pairs = [("30 days", "45 days"), ("2026-07-01", "2026-07-02"), ("2026-12-31", "2027-12-31")]
    return sum(1 for before, after in pairs if before in original and after in updated)


def _count_section_drift(original: str, updated: str) -> int:
    headings = ["## Parties", "## Payment", "## Scope", "## Term", "## Citation"]
    return sum(1 for heading in headings if heading in original and heading not in updated)


def main() -> None:
    results = run_document_corruption_benchmark()
    record_benchmark(Path(".bcv_runs") / "document_conservation", results)
    print(json.dumps([asdict(result) for result in results], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
