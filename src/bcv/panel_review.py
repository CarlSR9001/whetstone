"""Blind, auditable human-label intake for fuzzy-domain panel calibration.

This module deliberately separates *review distribution* from *adjudication*:
the exported queue contains no existing label, while consensus output carries
the reviewer identities and disagreement record needed to audit provenance.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


VALID_VERDICTS = {"pass", "fail"}


def _jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: expected an object")
        rows.append(row)
    return rows


def review_id(case: dict, answer: str) -> str:
    """Stable identifier over the exact record a reviewer is shown."""
    canonical = json.dumps({"case": case, "answer": answer}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]


def export_blind_review_queue(source: str | Path, output: str | Path) -> dict:
    """Strip labels/provenance and write the cases a reviewer may inspect."""
    exported: list[dict] = []
    seen: set[str] = set()
    for row in _jsonl(source):
        if not isinstance(row.get("case"), dict) or not isinstance(row.get("answer"), str):
            raise ValueError("review source rows require object 'case' and string 'answer'")
        identifier = review_id(row["case"], row["answer"])
        if identifier in seen:
            raise ValueError(f"duplicate review record: {identifier}")
        seen.add(identifier)
        exported.append({"review_id": identifier, "case": row["case"], "answer": row["answer"]})
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in exported), encoding="utf-8")
    return {"queue": str(destination), "cases": len(exported), "queue_sha256": _sha256(destination)}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_votes(paths: Iterable[str | Path], queue_ids: set[str]) -> tuple[dict[str, list[dict]], list[str]]:
    votes: dict[str, list[dict]] = defaultdict(list)
    reviewers: list[str] = []
    for path in paths:
        rows = _jsonl(path)
        names = {row.get("reviewer_id") for row in rows}
        if len(names) != 1 or not isinstance(next(iter(names), None), str) or not next(iter(names)).strip():
            raise ValueError(f"{path}: each vote file must contain exactly one non-empty reviewer_id")
        reviewer = next(iter(names)).strip()
        if reviewer in reviewers:
            raise ValueError(f"duplicate reviewer_id across vote files: {reviewer}")
        reviewers.append(reviewer)
        submitted: set[str] = set()
        for row in rows:
            identifier, verdict = row.get("review_id"), row.get("verdict")
            if identifier not in queue_ids:
                raise ValueError(f"{path}: unknown review_id {identifier!r}")
            if identifier in submitted:
                raise ValueError(f"{path}: duplicate vote for {identifier}")
            if verdict not in VALID_VERDICTS:
                raise ValueError(f"{path}: verdict must be one of {sorted(VALID_VERDICTS)}")
            submitted.add(identifier)
            votes[identifier].append({"reviewer_id": reviewer, "verdict": verdict})
    return votes, sorted(reviewers)


def adjudicate_review_labels(
    queue: str | Path,
    vote_paths: Iterable[str | Path],
    output: str | Path,
    disagreements: str | Path,
    min_reviewers: int = 2,
) -> dict:
    """Create calibration triples only where distinct reviewers unanimously agree.

    Missing votes and split votes are retained in ``disagreements``; they never
    silently become a positive or negative calibration label.
    """
    if min_reviewers < 2:
        raise ValueError("min_reviewers must be at least 2")
    queue_rows = _jsonl(queue)
    queue_by_id = {row.get("review_id"): row for row in queue_rows}
    if len(queue_by_id) != len(queue_rows) or None in queue_by_id:
        raise ValueError("queue contains missing or duplicate review_id values")
    for identifier, row in queue_by_id.items():
        if review_id(row.get("case"), row.get("answer")) != identifier:
            raise ValueError(f"queue record {identifier!r} does not match its content")
    votes, reviewers = _load_votes(vote_paths, set(queue_by_id))
    if len(reviewers) < min_reviewers:
        raise ValueError(f"need at least {min_reviewers} distinct reviewers; got {len(reviewers)}")

    agreed: list[dict] = []
    unresolved: list[dict] = []
    for identifier, row in queue_by_id.items():
        item_votes = votes.get(identifier, [])
        counts = Counter(vote["verdict"] for vote in item_votes)
        unanimous = len(item_votes) >= min_reviewers and len(counts) == 1
        if unanimous:
            agreed.append({
                "case": row["case"],
                "answer": row["answer"],
                "human_pass": item_votes[0]["verdict"] == "pass",
                "label_source": "two_reviewer_agreement",
                "review_id": identifier,
                "reviewer_ids": sorted(vote["reviewer_id"] for vote in item_votes),
            })
        else:
            unresolved.append({
                "review_id": identifier,
                "reason": "split_votes" if len(counts) > 1 else "insufficient_votes",
                "votes": item_votes,
            })

    output_path, disagreements_path = Path(output), Path(disagreements)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    disagreements_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in agreed), encoding="utf-8")
    disagreements_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in unresolved), encoding="utf-8")
    return {
        "queue": str(queue),
        "queue_sha256": _sha256(Path(queue)),
        "reviewers": reviewers,
        "cases": len(queue_rows),
        "agreed_cases": len(agreed),
        "unresolved_cases": len(unresolved),
        "calibration_rows": str(output_path),
        "disagreements": str(disagreements_path),
    }
