from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetExportResult:
    sft_path: str
    preference_path: str
    sft_examples: int
    preference_examples: int


def export_training_datasets(run_root: str | Path) -> DatasetExportResult:
    run_root = Path(run_root)
    candidates_path = run_root / "experience" / "training_candidates.jsonl"
    if not candidates_path.exists():
        raise FileNotFoundError(f"missing training candidates: {candidates_path}")

    candidates = [
        json.loads(line)
        for line in candidates_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    dataset_dir = run_root / "datasets"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    sft_path = dataset_dir / "controller_sft.jsonl"
    preference_path = dataset_dir / "controller_preferences.jsonl"

    with sft_path.open("w", encoding="utf-8") as handle:
        for candidate in candidates:
            handle.write(json.dumps(_sft_example(candidate), sort_keys=True) + "\n")

    preference_examples = _preference_examples(candidates)
    with preference_path.open("w", encoding="utf-8") as handle:
        for example in preference_examples:
            handle.write(json.dumps(example, sort_keys=True) + "\n")

    return DatasetExportResult(
        sft_path=str(sft_path),
        preference_path=str(preference_path),
        sft_examples=len(candidates),
        preference_examples=len(preference_examples),
    )


def _sft_example(candidate: dict[str, Any]) -> dict[str, Any]:
    label = candidate["label"]
    decision = "promote" if label == "verified_positive" else "repair"
    response = {
        "decision": decision,
        "label": label,
        "source_commit_id": candidate["source_commit_id"],
        "object_refs": candidate["object_refs"],
    }
    if decision == "repair":
        response["repair_reason"] = _compact_details(candidate.get("details", ""))

    return {
        "messages": [
            {
                "role": "system",
                "content": "You are the merge controller for a verifier-backed cognitive branch runtime. Decide whether a trace should be promoted or routed to repair.",
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "branch_id": candidate["branch_id"],
                        "event_type": candidate["event_type"],
                        "object_refs": candidate["object_refs"],
                        "verifier_details": _compact_details(candidate.get("details", "")),
                    },
                    sort_keys=True,
                ),
            },
            {
                "role": "assistant",
                "content": json.dumps(response, sort_keys=True),
            },
        ]
    }


def _preference_examples(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    positives = [candidate for candidate in candidates if candidate["label"] == "verified_positive"]
    repairs = [candidate for candidate in candidates if candidate["label"] == "repair_required"]
    examples: list[dict[str, Any]] = []
    for positive, repair in zip(positives, repairs):
        prompt = json.dumps(
            {
                "positive_trace": positive["details"],
                "failed_trace": repair["details"],
                "instruction": "Choose the trace that should be merged into main.",
            },
            sort_keys=True,
        )
        examples.append(
            {
                "prompt": prompt,
                "chosen": json.dumps(
                    {
                        "decision": "promote",
                        "candidate_id": positive["candidate_id"],
                        "object_refs": positive["object_refs"],
                    },
                    sort_keys=True,
                ),
                "rejected": json.dumps(
                    {
                        "decision": "promote",
                        "candidate_id": repair["candidate_id"],
                        "object_refs": repair["object_refs"],
                    },
                    sort_keys=True,
                ),
            }
        )
    return examples


def _compact_details(details: str) -> str:
    if len(details) <= 1200:
        return details
    return details[:1200] + "...[truncated]"


def main() -> None:
    print(json.dumps(asdict(export_training_datasets(".bcv_runs/all")), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

