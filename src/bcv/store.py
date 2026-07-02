from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from pathlib import Path
from urllib.parse import quote, unquote

from bcv.schema import Commit, Event


class CognitiveStore:
    """JSONL-backed semantic branch store.

    This intentionally uses boring files first. Git can version those files, while
    this layer keeps state addressable by branch, commit, event, claim, and artifact.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.branches_dir = self.root / "state" / "branches"
        self.refs_dir = self.root / "state" / "refs"
        self.experience_dir = self.root / "experience"

    def init(self) -> None:
        self.branches_dir.mkdir(parents=True, exist_ok=True)
        self.refs_dir.mkdir(parents=True, exist_ok=True)
        self.experience_dir.mkdir(parents=True, exist_ok=True)
        if not self.branch_exists("main"):
            self.create_branch("main")

    def branch_exists(self, branch_id: str) -> bool:
        return self._branch_path(branch_id).exists()

    def create_branch(self, branch_id: str, from_branch: str | None = None) -> None:
        branch_path = self._branch_path(branch_id)
        if branch_path.exists():
            raise ValueError(f"branch already exists: {branch_id}")
        branch_path.mkdir(parents=True)
        self._commit_path(branch_id).write_text("", encoding="utf-8")
        if from_branch is not None:
            for commit in self.log(from_branch):
                self._append_commit(branch_id, commit)
        self._write_branch_index()

    def commit(
        self,
        branch_id: str,
        message: str,
        events: Iterable[Event],
        parent_ids: Iterable[str] = (),
    ) -> Commit:
        self._require_branch(branch_id)
        normalized_events = tuple(
            Event(
                event_type=event.event_type,
                actor=event.actor,
                message=event.message,
                event_id=event.event_id,
                episode_id=event.episode_id,
                branch_id=branch_id,
                input_refs=event.input_refs,
                output_refs=event.output_refs,
                evidence_refs=event.evidence_refs,
                artifact_refs=event.artifact_refs,
                tests=event.tests,
                confidence_before=event.confidence_before,
                confidence_after=event.confidence_after,
                timestamp=event.timestamp,
            )
            for event in events
        )
        commit = Commit(
            branch_id=branch_id,
            message=message,
            events=normalized_events,
            parent_ids=tuple(parent_ids),
        )
        self._append_commit(branch_id, commit)
        self._append_experience(commit)
        return commit

    def log(self, branch_id: str) -> list[Commit]:
        self._require_branch(branch_id)
        path = self._commit_path(branch_id)
        commits: list[Commit] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                commits.append(Commit.from_dict(json.loads(line)))
        return commits

    def checkout(self, branch_id: str) -> dict[str, object]:
        commits = self.log(branch_id)
        events = [event for commit in commits for event in commit.events]
        return {
            "branch_id": branch_id,
            "commit_count": len(commits),
            "event_count": len(events),
            "latest_commit": commits[-1].commit_id if commits else None,
            "latest_message": commits[-1].message if commits else None,
        }

    def diff(self, source_branch: str, target_branch: str) -> dict[str, list[str]]:
        source_events = {event.event_id for event in self._events(source_branch)}
        target_events = {event.event_id for event in self._events(target_branch)}
        return {
            "only_in_source": sorted(source_events - target_events),
            "only_in_target": sorted(target_events - source_events),
        }

    def merge(self, source_branch: str, target_branch: str, message: str) -> Commit:
        self._require_branch(source_branch)
        self._require_branch(target_branch)
        target_event_ids = {event.event_id for event in self._events(target_branch)}
        new_events = [
            event
            for event in self._events(source_branch)
            if event.event_id not in target_event_ids
        ]
        merge_event = Event(
            event_type="branch_merge",
            actor="runtime",
            message=f"merged {source_branch} into {target_branch}",
            input_refs=(source_branch,),
            output_refs=(target_branch,),
        )
        return self.commit(target_branch, message, [*new_events, merge_event])

    def blame(self, branch_id: str, object_ref: str) -> tuple[Commit, Event] | None:
        for commit in self.log(branch_id):
            for event in commit.events:
                refs = (
                    *event.input_refs,
                    *event.output_refs,
                    *event.evidence_refs,
                    *event.artifact_refs,
                )
                if object_ref in refs:
                    return commit, event
        return None

    def bisect(
        self,
        branch_id: str,
        fails_after: Callable[[list[Commit]], bool],
    ) -> Commit | None:
        commits = self.log(branch_id)
        low = 0
        high = len(commits) - 1
        first_bad: Commit | None = None
        while low <= high:
            mid = (low + high) // 2
            prefix = commits[: mid + 1]
            if fails_after(prefix):
                first_bad = commits[mid]
                high = mid - 1
            else:
                low = mid + 1
        return first_bad

    def _events(self, branch_id: str) -> list[Event]:
        return [event for commit in self.log(branch_id) for event in commit.events]

    def _append_commit(self, branch_id: str, commit: Commit) -> None:
        with self._commit_path(branch_id).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(commit.to_dict(), sort_keys=True) + "\n")

    def _append_experience(self, commit: Commit) -> None:
        path = self.experience_dir / "episodes.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(commit.to_dict(), sort_keys=True) + "\n")

    def _write_branch_index(self) -> None:
        branches = sorted(
            unquote(path.name)
            for path in self.branches_dir.iterdir()
            if path.is_dir()
        )
        (self.refs_dir / "branch_index.json").write_text(
            json.dumps({"branches": branches}, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _branch_path(self, branch_id: str) -> Path:
        if branch_id in {"", ".", ".."} or "\x00" in branch_id:
            raise ValueError(f"invalid branch id: {branch_id}")
        return self.branches_dir / quote(branch_id, safe="")

    def _commit_path(self, branch_id: str) -> Path:
        return self._branch_path(branch_id) / "commits.jsonl"

    def _require_branch(self, branch_id: str) -> None:
        if not self.branch_exists(branch_id):
            raise ValueError(f"unknown branch: {branch_id}")
