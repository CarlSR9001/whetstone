from __future__ import annotations

from bcv.schema import Event, TestResult
from bcv.store import CognitiveStore


def test_store_commits_blames_and_merges(tmp_path):
    store = CognitiveStore(tmp_path)
    store.init()
    store.create_branch("hypothesis/runtime-learning", from_branch="main")

    event = Event(
        event_type="claim_added",
        actor="model",
        message="Added sourced claim.",
        output_refs=("claim:runtime-learning",),
        evidence_refs=("source:conversation",),
        tests=(TestResult("claim_has_source", "pass"),),
    )
    commit = store.commit("hypothesis/runtime-learning", "add runtime-learning claim", [event])

    blame = store.blame("hypothesis/runtime-learning", "claim:runtime-learning")
    assert blame is not None
    blamed_commit, blamed_event = blame
    assert blamed_commit.commit_id == commit.commit_id
    assert blamed_event.message == "Added sourced claim."

    diff_before = store.diff("hypothesis/runtime-learning", "main")
    assert event.event_id in diff_before["only_in_source"]

    store.merge("hypothesis/runtime-learning", "main", "merge verified claim")
    diff_after = store.diff("hypothesis/runtime-learning", "main")
    assert event.event_id not in diff_after["only_in_source"]


def test_store_bisects_first_bad_commit(tmp_path):
    store = CognitiveStore(tmp_path)
    store.init()
    store.commit(
        "main",
        "good",
        [Event(event_type="claim_added", actor="model", message="good", output_refs=("claim:good",))],
    )
    bad_commit = store.commit(
        "main",
        "bad",
        [Event(event_type="claim_added", actor="model", message="bad", output_refs=("claim:bad",))],
    )
    store.commit(
        "main",
        "later",
        [Event(event_type="claim_added", actor="model", message="later", output_refs=("claim:later",))],
    )

    def fails_after(commits):
        refs = {
            ref
            for commit in commits
            for event in commit.events
            for ref in event.output_refs
        }
        return "claim:bad" in refs

    assert store.bisect("main", fails_after).commit_id == bad_commit.commit_id
