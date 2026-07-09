import json

import pytest

from bcv.panel_review import adjudicate_review_labels, export_blind_review_queue, write_vote_templates


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_blind_export_and_two_reviewer_adjudication_preserves_disagreements(tmp_path):
    source = tmp_path / "source.jsonl"
    _write_jsonl(source, [
        {"case": {"source": "A", "question": "Q"}, "answer": "Answer A", "human_pass": True, "label_source": "secret"},
        {"case": {"source": "B", "question": "Q"}, "answer": "Answer B", "human_pass": False, "label_source": "secret"},
    ])
    queue = tmp_path / "queue.jsonl"
    exported = export_blind_review_queue(source, queue)
    queue_rows = [json.loads(line) for line in queue.read_text(encoding="utf-8").splitlines()]
    assert exported["cases"] == 2
    assert all("human_pass" not in row and "label_source" not in row for row in queue_rows)

    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    _write_jsonl(a, [
        {"reviewer_id": "reviewer-a", "review_id": queue_rows[0]["review_id"], "verdict": "pass"},
        {"reviewer_id": "reviewer-a", "review_id": queue_rows[1]["review_id"], "verdict": "fail"},
    ])
    _write_jsonl(b, [
        {"reviewer_id": "reviewer-b", "review_id": queue_rows[0]["review_id"], "verdict": "pass"},
        {"reviewer_id": "reviewer-b", "review_id": queue_rows[1]["review_id"], "verdict": "pass"},
    ])
    result = adjudicate_review_labels(queue, [a, b], tmp_path / "agreed.jsonl", tmp_path / "unresolved.jsonl")
    agreed = [json.loads(line) for line in (tmp_path / "agreed.jsonl").read_text(encoding="utf-8").splitlines()]
    unresolved = [json.loads(line) for line in (tmp_path / "unresolved.jsonl").read_text(encoding="utf-8").splitlines()]
    assert result["agreed_cases"] == 1
    assert agreed[0]["human_pass"] is True
    assert agreed[0]["label_source"] == "two_reviewer_agreement"
    assert unresolved == [{"reason": "split_votes", "review_id": queue_rows[1]["review_id"], "votes": [
        {"reviewer_id": "reviewer-a", "verdict": "fail"}, {"reviewer_id": "reviewer-b", "verdict": "pass"}
    ]}]


def test_adjudication_rejects_duplicate_reviewer_identity(tmp_path):
    source = tmp_path / "source.jsonl"
    _write_jsonl(source, [{"case": {"source": "A", "question": "Q"}, "answer": "Answer A"}])
    queue = tmp_path / "queue.jsonl"
    export_blind_review_queue(source, queue)
    identifier = json.loads(queue.read_text(encoding="utf-8"))["review_id"]
    first, second = tmp_path / "first.jsonl", tmp_path / "second.jsonl"
    _write_jsonl(first, [{"reviewer_id": "same", "review_id": identifier, "verdict": "pass"}])
    _write_jsonl(second, [{"reviewer_id": "same", "review_id": identifier, "verdict": "pass"}])
    with pytest.raises(ValueError, match="duplicate reviewer_id"):
        adjudicate_review_labels(queue, [first, second], tmp_path / "out", tmp_path / "bad")


def test_vote_templates_are_blank_and_bound_to_distinct_reviewers(tmp_path):
    source = tmp_path / "source.jsonl"
    _write_jsonl(source, [{"case": {"source": "A", "question": "Q"}, "answer": "Answer A", "human_pass": True}])
    queue = tmp_path / "queue.jsonl"
    export_blind_review_queue(source, queue)
    result = write_vote_templates(queue, ["reviewer-a", "reviewer-b"], tmp_path / "ballots")
    first = [json.loads(line) for line in (tmp_path / "ballots" / "reviewer-a.votes.jsonl").read_text(encoding="utf-8").splitlines()]
    assert result["cases"] == 1
    assert first == [{"review_id": first[0]["review_id"], "reviewer_id": "reviewer-a", "verdict": ""}]
    with pytest.raises(ValueError, match="distinct"):
        write_vote_templates(queue, ["same", "same"], tmp_path / "bad")
