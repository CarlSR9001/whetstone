from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from bcv.schema import Event, TestResult
from bcv.store import CognitiveStore


@dataclass(frozen=True)
class RecallQuery:
    name: str
    object_ref: str
    expected_message_fragment: str


@dataclass(frozen=True)
class RecallMetric:
    name: str
    object_ref: str
    recalled: bool
    commit_message: str | None
    event_message: str | None


@dataclass(frozen=True)
class RecallBenchmarkResult:
    branch_count: int
    commits: int
    queries: int
    recalled: int
    metrics: tuple[RecallMetric, ...]


def run_recall_benchmark(root: str | Path = ".bcv_runs/recall_benchmark") -> RecallBenchmarkResult:
    root = Path(root)
    store = CognitiveStore(root)
    if root.exists():
        # Keep reruns deterministic without touching anything outside this generated run root.
        for child in root.rglob("*"):
            if child.is_file():
                child.unlink()
        for child in sorted((path for path in root.rglob("*") if path.is_dir()), reverse=True):
            child.rmdir()
    store.init()

    branches = {
        "hypothesis/runtime-learning": (
            ("claim:learning-external-first", "Durable learning starts external before weight updates."),
            ("constraint:no-unsourced-memory", "No memory can become high confidence without a source."),
        ),
        "implementation/document-agent": (
            ("artifact:patch-editor", "Markdown edits must be patches, not full rewrites."),
            ("test:document-invariants", "Document verifier checks protected tokens and section drift."),
        ),
        "critique/verifier-risk": (
            ("risk:verifier-gaming", "Weak verifiers can be gamed by generated traces."),
            ("mitigation:heldout-checks", "Held-out checks and adversarial branches reduce verifier gaming."),
        ),
    }

    for branch, objects in branches.items():
        store.create_branch(branch, from_branch="main")
        for object_ref, message in objects:
            store.commit(
                branch,
                f"add {object_ref}",
                [
                    Event(
                        event_type="claim_added",
                        actor="model",
                        message=message,
                        output_refs=(object_ref,),
                        tests=(TestResult("has_object_ref", "pass"),),
                    )
                ],
            )
        for index in range(8):
            store.commit(
                branch,
                f"distractor {index}",
                [
                    Event(
                        event_type="claim_added",
                        actor="model",
                        message=f"Distractor note {index} for {branch}.",
                        output_refs=(f"distractor:{branch}:{index}",),
                    )
                ],
            )

    queries = (
        RecallQuery("external learning", "claim:learning-external-first", "external before weight"),
        RecallQuery("document patching", "artifact:patch-editor", "patches, not full rewrites"),
        RecallQuery("verifier risk", "risk:verifier-gaming", "gamed"),
        RecallQuery("heldout mitigation", "mitigation:heldout-checks", "Held-out checks"),
    )
    metrics: list[RecallMetric] = []
    for query in queries:
        found = None
        for branch in branches:
            found = store.blame(branch, query.object_ref)
            if found:
                break
        commit, event = found if found else (None, None)
        event_message = event.message if event else None
        metrics.append(
            RecallMetric(
                name=query.name,
                object_ref=query.object_ref,
                recalled=bool(event_message and query.expected_message_fragment in event_message),
                commit_message=commit.message if commit else None,
                event_message=event_message,
            )
        )

    result = RecallBenchmarkResult(
        branch_count=len(branches),
        commits=sum(len(store.log(branch)) for branch in branches),
        queries=len(queries),
        recalled=sum(1 for metric in metrics if metric.recalled),
        metrics=tuple(metrics),
    )
    (root / "recall_result.json").write_text(
        json.dumps(asdict(result), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return result


def main() -> None:
    print(json.dumps(asdict(run_recall_benchmark()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

