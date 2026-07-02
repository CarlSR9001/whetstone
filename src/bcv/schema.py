from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, ClassVar, Literal
from uuid import uuid4


EventType = Literal[
    "claim_added",
    "artifact_patch",
    "verifier_result",
    "memory_update",
    "model_delta",
    "conflict",
    "branch_merge",
]


Actor = Literal["model", "tool", "verifier", "user", "trainer", "runtime"]


@dataclass(frozen=True)
class TestResult:
    __test__: ClassVar[bool] = False

    test_id: str
    result: Literal["pass", "fail", "inconclusive"]
    details: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Event:
    event_type: EventType
    actor: Actor
    message: str
    event_id: str = field(default_factory=lambda: f"evt_{uuid4().hex}")
    episode_id: str = "ep_default"
    branch_id: str = "main"
    input_refs: tuple[str, ...] = ()
    output_refs: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    artifact_refs: tuple[str, ...] = ()
    tests: tuple[TestResult, ...] = ()
    confidence_before: float | None = None
    confidence_after: float | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["tests"] = [test.to_dict() for test in self.tests]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Event":
        tests = tuple(TestResult(**item) for item in data.get("tests", ()))
        clean = dict(data)
        clean["tests"] = tests
        clean["input_refs"] = tuple(clean.get("input_refs", ()))
        clean["output_refs"] = tuple(clean.get("output_refs", ()))
        clean["evidence_refs"] = tuple(clean.get("evidence_refs", ()))
        clean["artifact_refs"] = tuple(clean.get("artifact_refs", ()))
        return cls(**clean)


@dataclass(frozen=True)
class Commit:
    branch_id: str
    message: str
    events: tuple[Event, ...]
    commit_id: str = field(default_factory=lambda: f"commit_{uuid4().hex}")
    parent_ids: tuple[str, ...] = ()
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "commit_id": self.commit_id,
            "branch_id": self.branch_id,
            "message": self.message,
            "parent_ids": list(self.parent_ids),
            "timestamp": self.timestamp,
            "events": [event.to_dict() for event in self.events],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Commit":
        return cls(
            commit_id=data["commit_id"],
            branch_id=data["branch_id"],
            message=data["message"],
            parent_ids=tuple(data.get("parent_ids", ())),
            timestamp=data["timestamp"],
            events=tuple(Event.from_dict(item) for item in data.get("events", ())),
        )
