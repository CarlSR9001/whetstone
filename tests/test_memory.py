from __future__ import annotations

from bcv.memory import Memory, run_memory_hygiene_probe, verify_memory


def test_memory_hygiene_accepts_sourced_preference():
    memory = Memory(
        memory_id="memory:style",
        content="User prefers concise technical answers.",
        source="conversation:1",
        confidence=0.9,
        kind="preference",
        last_verified_at="2026-06-30T00:00:00+00:00",
        use_policy="silent",
    )

    result = verify_memory(memory)
    assert result.accepted is True
    assert result.effective_use_policy == "silent"


def test_memory_hygiene_downgrades_unsourced_high_confidence_inference():
    memory = Memory(
        memory_id="memory:bad",
        content="User prefers all prototypes in Rust.",
        source=None,
        confidence=0.95,
        kind="inference",
        use_policy="silent",
    )

    result = verify_memory(memory)
    assert result.accepted is False
    assert "high_confidence_memory_without_source" in result.failures
    assert "fact_or_inference_missing_last_verified_at" in result.failures
    assert result.effective_confidence == 0.6


def test_memory_probe_has_clean_and_rejected_cases():
    results = {result.memory_id: result for result in run_memory_hygiene_probe()}

    assert results["memory:technical-depth"].accepted is True
    assert results["memory:favorite-stack"].accepted is False
    assert results["memory:timezone"].effective_use_policy == "ask-before-use"

