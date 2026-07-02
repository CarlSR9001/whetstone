from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from bcv.markdown_agent import edit_markdown_text


DOC = """# Vendor Agreement

## Parties

This agreement is between Northstar Labs and Meridian Health.

## Payment

Meridian Health will pay invoice INV-2049 within 30 days of receipt.

## Scope

Northstar Labs will deliver the analytics dashboard described in Exhibit A.

## Citation

The privacy obligations follow the Data Processing Addendum [DPA-17].
"""


@dataclass
class ScriptedModel:
    responses: list[dict[str, Any]]
    backend: str = "scripted"
    model: str = "scripted-model"

    def generate_json(self, prompt: str, temperature: float = 0.0) -> dict[str, Any]:
        if not self.responses:
            return {"operations": []}
        return self.responses.pop(0)


@dataclass(frozen=True)
class UsefulnessCase:
    name: str
    accepted: bool
    attempts: int
    payment_preserved: bool
    citation_preserved: bool
    useful_change_present: bool
    failure: str | None


def run_usefulness_benchmark(root: str | Path = ".bcv_runs/usefulness") -> list[UsefulnessCase]:
    root = Path(root)
    cases: list[UsefulnessCase] = []

    immediate = ScriptedModel([_good_patch()])
    result, updated = edit_markdown_text(
        DOC,
        "Add weekly deployment summary to Scope.",
        run_root=root / "immediate",
        client=immediate,
    )
    cases.append(_case("immediate_accept", result.accepted, result.attempts, updated, result.failure))

    retry = ScriptedModel([_bad_patch_removes_citation(), _good_patch()])
    result, updated = edit_markdown_text(
        DOC,
        "Add weekly deployment summary to Scope.",
        run_root=root / "retry",
        client=retry,
    )
    cases.append(_case("reject_then_recover", result.accepted, result.attempts, updated, result.failure))

    blocked = ScriptedModel([_bad_patch_removes_citation(), _bad_patch_removes_citation()])
    result, updated = edit_markdown_text(
        DOC,
        "Add weekly deployment summary to Scope.",
        run_root=root / "blocked",
        client=blocked,
        max_attempts=2,
    )
    cases.append(_case("blocked_bad_model", result.accepted, result.attempts, updated, result.failure))

    output_path = root / "usefulness_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps([asdict(case) for case in cases], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return cases


def _good_patch() -> dict[str, Any]:
    return {
        "operations": [
            {
                "target_heading": "Scope",
                "find": "Northstar Labs will deliver the analytics dashboard described in Exhibit A.",
                "replace": "Northstar Labs will deliver the analytics dashboard and a weekly deployment summary described in Exhibit A.",
            }
        ]
    }


def _bad_patch_removes_citation() -> dict[str, Any]:
    return {
        "operations": [
            {
                "target_heading": "Citation",
                "find": "The privacy obligations follow the Data Processing Addendum [DPA-17].",
                "replace": "The privacy obligations follow the Data Processing Addendum.",
            }
        ]
    }


def _case(
    name: str,
    accepted: bool,
    attempts: int,
    document: str,
    failure: str | None,
) -> UsefulnessCase:
    return UsefulnessCase(
        name=name,
        accepted=accepted,
        attempts=attempts,
        payment_preserved="30 days" in document and "INV-2049" in document,
        citation_preserved="[DPA-17]" in document,
        useful_change_present="weekly deployment summary" in document,
        failure=failure,
    )


def main() -> None:
    print(json.dumps([asdict(case) for case in run_usefulness_benchmark()], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

