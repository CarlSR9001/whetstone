from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from bcv.schema import Commit


@dataclass(frozen=True)
class TrainingCandidate:
    candidate_id: str
    source_commit_id: str
    branch_id: str
    label: str
    event_type: str
    object_refs: tuple[str, ...]
    details: str

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "source_commit_id": self.source_commit_id,
            "branch_id": self.branch_id,
            "label": self.label,
            "event_type": self.event_type,
            "object_refs": list(self.object_refs),
            "details": self.details,
        }


def mine_training_candidates(run_root: str | Path) -> list[TrainingCandidate]:
    run_root = Path(run_root)
    candidates: list[TrainingCandidate] = []
    for episodes_path in sorted(run_root.rglob("experience/episodes.jsonl")):
        for line in episodes_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            commit = Commit.from_dict(json.loads(line))
            for event in commit.events:
                for test in event.tests:
                    if test.result == "pass":
                        label = "verified_positive"
                    elif test.result == "fail":
                        label = "repair_required"
                    else:
                        label = "needs_review"
                    candidates.append(
                        TrainingCandidate(
                            candidate_id=f"candidate:{commit.commit_id}:{event.event_id}:{test.test_id}",
                            source_commit_id=commit.commit_id,
                            branch_id=commit.branch_id,
                            label=label,
                            event_type=event.event_type,
                            object_refs=event.output_refs,
                            details=test.details,
                        )
                    )
    return candidates


def write_training_candidates(run_root: str | Path) -> Path:
    run_root = Path(run_root)
    output_path = run_root / "experience" / "training_candidates.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    candidates = mine_training_candidates(run_root)
    with output_path.open("w", encoding="utf-8") as handle:
        for candidate in candidates:
            handle.write(json.dumps(candidate.to_dict(), sort_keys=True) + "\n")
    return output_path

