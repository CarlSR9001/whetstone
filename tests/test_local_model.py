from __future__ import annotations

import pytest

from bcv.local_model import LocalModelClient, LocalModelError
from bcv.model_probe import _evaluate_model_patch, _evaluate_model_research


def test_model_patch_evaluator_accepts_valid_patch():
    client = LocalModelClient("ollama", "test")
    result = _evaluate_model_patch(
        client,
        {
            "mode": "patch",
            "operations": [
                {
                    "target_heading": "Scope",
                    "find": "Northstar Labs will deliver the analytics dashboard described in Exhibit A.",
                    "replace": "Northstar Labs will deliver the analytics dashboard and a weekly deployment summary described in Exhibit A.",
                }
            ],
        },
    )

    assert result.accepted is True


def test_model_patch_evaluator_rejects_full_rewrite_mode():
    client = LocalModelClient("ollama", "test")
    result = _evaluate_model_patch(
        client,
        {
            "mode": "rewrite",
            "document": "bad",
        },
    )

    assert result.accepted is False
    assert "unsupported mode" in result.failure


def test_model_research_evaluator_accepts_sourced_contradiction_graph():
    client = LocalModelClient("ollama", "test")
    result = _evaluate_model_research(
        client,
        {
            "sources": [
                {"source_id": "source:a", "title": "A", "text": "Done June 20."},
                {"source_id": "source:b", "title": "B", "text": "Moved July 15."},
            ],
            "claims": [
                {
                    "claim_id": "claim:june",
                    "text": "The rollout completed on 2026-06-20.",
                    "source_ids": ["source:a"],
                    "contradicts": ["claim:july"],
                },
                {
                    "claim_id": "claim:july",
                    "text": "The rollout moved to 2026-07-15.",
                    "source_ids": ["source:b"],
                    "contradicts": ["claim:june"],
                },
            ],
            "final_claim_ids": ["claim:june", "claim:july"],
        },
    )

    assert result.accepted is True
    assert result.contradictions == (("claim:july", "claim:june"),)


def test_local_model_error_is_runtime_error():
    assert issubclass(LocalModelError, RuntimeError)
