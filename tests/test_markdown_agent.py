from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from bcv.markdown_agent import edit_markdown_text


DOC = """# Vendor Agreement

## Parties

This agreement is between Northstar Labs and Meridian Health.

## Scope

Northstar Labs will deliver the analytics dashboard described in Exhibit A.

## Payment

Meridian Health will pay invoice INV-2049 within 30 days of receipt.

## Citation

The privacy obligations follow the Data Processing Addendum [DPA-17].
"""


@dataclass
class FakeJsonModel:
    responses: list[dict[str, Any]]
    backend: str = "fake"
    model: str = "fake-model"

    def generate_json(self, prompt: str, temperature: float = 0.0) -> dict[str, Any]:
        return self.responses.pop(0)


def test_markdown_agent_accepts_verified_patch(tmp_path):
    client = FakeJsonModel(
        [
            {
                "operations": [
                    {
                        "target_heading": "Scope",
                        "find": "Northstar Labs will deliver the analytics dashboard described in Exhibit A.",
                        "replace": "Northstar Labs will deliver the analytics dashboard and a weekly deployment summary described in Exhibit A.",
                    }
                ]
            }
        ]
    )

    result, updated = edit_markdown_text(
        DOC,
        "Add weekly deployment summary to Scope.",
        run_root=tmp_path,
        client=client,
    )

    assert result.accepted is True
    assert result.attempts == 1
    assert "weekly deployment summary" in updated
    assert "30 days" in updated


def test_markdown_agent_retries_after_verifier_failure(tmp_path):
    client = FakeJsonModel(
        [
                {
                    "operations": [
                        {
                            "target_heading": "Citation",
                            "find": "The privacy obligations follow the Data Processing Addendum [DPA-17].",
                            "replace": "The privacy obligations follow the Data Processing Addendum.",
                        }
                    ]
                },
            {
                "operations": [
                    {
                        "target_heading": "Scope",
                        "find": "Northstar Labs will deliver the analytics dashboard described in Exhibit A.",
                        "replace": "Northstar Labs will deliver the analytics dashboard and a weekly deployment summary described in Exhibit A.",
                    }
                ]
            },
        ]
    )

    result, updated = edit_markdown_text(
        DOC,
        "Add weekly deployment summary to Scope.",
        run_root=tmp_path,
        client=client,
    )

    assert result.accepted is True
    assert result.attempts == 2
    assert "weekly deployment summary" in updated


def test_markdown_agent_retries_after_missing_required_phrase(tmp_path):
    client = FakeJsonModel(
        [
            {
                "operations": [
                    {
                        "target_heading": "Parties",
                        "find": "This agreement is between Northstar Labs and Meridian Health.",
                        "replace": "This agreement is between Northstar Labs and Meridian Health as the client.",
                    }
                ]
            },
            {
                "operations": [
                    {
                        "target_heading": "Parties",
                        "find": "This agreement is between Northstar Labs and Meridian Health.",
                        "replace": "This agreement is between Northstar Labs and Meridian Health as the customer.",
                    }
                ]
            },
        ]
    )

    result, updated = edit_markdown_text(
        DOC,
        "Clarify Meridian Health is the customer.",
        run_root=tmp_path,
        client=client,
        required_phrases=("customer",),
    )

    assert result.accepted is True
    assert result.attempts == 2
    assert "customer" in updated
