from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4


MemoryKind = Literal["observation", "preference", "fact", "inference", "joke", "instruction"]
UsePolicy = Literal["silent", "ask-before-use", "never-use-without-surfacing"]


class MemoryError(ValueError):
    pass


@dataclass(frozen=True)
class Memory:
    content: str
    source: str | None
    confidence: float
    kind: MemoryKind
    memory_id: str = field(default_factory=lambda: f"memory_{uuid4().hex}")
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    last_verified_at: str | None = None
    expiry_policy: str | None = None
    contradiction_refs: tuple[str, ...] = ()
    use_policy: UsePolicy = "ask-before-use"
    supersedes: tuple[str, ...] = ()


@dataclass(frozen=True)
class MemoryVerification:
    accepted: bool
    memory_id: str
    failures: tuple[str, ...]
    effective_use_policy: UsePolicy
    effective_confidence: float


def verify_memory(memory: Memory) -> MemoryVerification:
    failures: list[str] = []
    effective_confidence = memory.confidence
    effective_use_policy = memory.use_policy

    if memory.confidence > 0.7 and not memory.source:
        failures.append("high_confidence_memory_without_source")
        effective_confidence = 0.7

    if memory.contradiction_refs and memory.use_policy == "silent":
        failures.append("contradicted_memory_cannot_be_used_silently")
        effective_use_policy = "ask-before-use"

    if memory.kind in {"fact", "inference"} and not memory.last_verified_at:
        failures.append("fact_or_inference_missing_last_verified_at")
        effective_confidence = min(effective_confidence, 0.6)

    if memory.expiry_policy == "expired" and memory.use_policy == "silent":
        failures.append("expired_memory_cannot_be_used_silently")
        effective_use_policy = "ask-before-use"
        effective_confidence = min(effective_confidence, 0.5)

    return MemoryVerification(
        accepted=not failures,
        memory_id=memory.memory_id,
        failures=tuple(failures),
        effective_use_policy=effective_use_policy,
        effective_confidence=effective_confidence,
    )


def run_memory_hygiene_probe() -> tuple[MemoryVerification, ...]:
    clean = Memory(
        memory_id="memory:technical-depth",
        content="User prefers direct technical depth over beginner explanations.",
        source="conversation:2026-06-30",
        confidence=0.9,
        kind="preference",
        last_verified_at="2026-06-30T00:00:00+00:00",
        use_policy="silent",
    )
    unsourced = Memory(
        memory_id="memory:favorite-stack",
        content="User prefers Rust for every new prototype.",
        source=None,
        confidence=0.95,
        kind="inference",
        use_policy="silent",
    )
    contradicted = Memory(
        memory_id="memory:timezone",
        content="User is always in Pacific time.",
        source="old-session",
        confidence=0.8,
        kind="fact",
        last_verified_at="2025-01-01T00:00:00+00:00",
        contradiction_refs=("environment:America/Chicago",),
        use_policy="silent",
    )
    return tuple(verify_memory(memory) for memory in (clean, unsourced, contradicted))

