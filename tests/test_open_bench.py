from __future__ import annotations

import copy
import json
import random
import subprocess

import pytest

from bcv.open_bench import OpenBenchError, OpenPromotionBench
from bcv.receipts import receipt_key_bundle, verify_receipt


def make_bench(tmp_path, *, ttl: float = 1800.0) -> OpenPromotionBench:
    return OpenPromotionBench(
        ledger_path=tmp_path / "public.jsonl",
        session_ttl=ttl,
        rng_factory=lambda: random.Random(17),
        enforce_limits=False,
    )


def enable_signing(tmp_path, monkeypatch) -> None:
    key = tmp_path / "receipt-key"
    subprocess.run(
        ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "test", "-f", str(key)],
        check=True,
    )
    monkeypatch.setenv("WHETSTONE_RECEIPT_SIGNING_KEY", str(key))


def answers_for(bench: OpenPromotionBench, session_id: str, *, solved: bool) -> dict:
    tasks = bench.sessions[session_id]["tasks"]
    if solved:
        return {task.public["item_id"]: copy.deepcopy(task.reference_patch) for task in tasks}
    return {task.public["item_id"]: {"writes": {}, "deletes": []} for task in tasks}


def payload(session: dict, baseline: dict, candidate: dict, *, publish: bool = False) -> dict:
    return {
        "session_id": session["session_id"],
        "baseline_manifest": {"name": "Baseline v1", "model": "local model"},
        "candidate_manifest": {"name": "Candidate v2", "model": "local model", "harness": "agent runner"},
        "baseline_answers": baseline,
        "candidate_answers": candidate,
        "publish": publish,
        "attestation": publish,
    }


def test_paired_gain_flow_publishes_only_sanitized_receipt(tmp_path, monkeypatch):
    enable_signing(tmp_path, monkeypatch)
    bench = make_bench(tmp_path)
    session = bench.start_session("198.51.100.1")
    baseline = answers_for(bench, session["session_id"], solved=False)
    candidate = answers_for(bench, session["session_id"], solved=True)

    receipt = bench.submit(payload(session, baseline, candidate, publish=True), "198.51.100.1")

    assert receipt["verdict"] == "PASS"
    assert receipt["baseline_passed"] == 0
    assert receipt["candidate_passed"] == receipt["total"] == 6
    assert receipt["gains"] == 6
    assert receipt["regressions"] == 0
    assert receipt["publication"]["status"] == "published"
    assert len(receipt["receipt_sha256"]) == 64
    assert len(receipt["answers_sha256"]) == 64

    raw_ledger = (tmp_path / "public.jsonl").read_text(encoding="utf-8")
    assert "trace_sampling" not in raw_ledger
    assert "DENIED_ACTIONS" not in raw_ledger
    assert "full replacement text" not in raw_ledger
    board = bench.leaderboard()
    assert board["entry_count"] == 1
    public_receipt = board["entries"][0]
    assert verify_receipt(
        public_receipt,
        receipt_key_bundle(),
        expected_challenge=session["challenge"],
    )["valid"] is True
    assert verify_receipt(
        receipt,
        receipt_key_bundle(),
        expected_challenge=session["challenge"],
    )["valid"] is True
    assert public_receipt["source_evidence_sha256"] == receipt["grading_evidence_sha256"]
    assert public_receipt["receipt_sha256"] == receipt["publication"]["public_receipt_sha256"]
    assert public_receipt["receipt_sha256"] != receipt["receipt_sha256"]
    assert board["raw_tasks_persisted"] is False
    assert board["raw_answers_persisted"] is False
    public_id = receipt["publication"]["public_id"]
    assert bench.receipt(public_id)["public_id"] == public_id
    assert bench.status()["publication_ledger_parent_writable"] is True

    with pytest.raises(OpenBenchError, match="one-shot"):
        bench.submit(payload(session, baseline, candidate), "198.51.100.1")


def test_regression_blocks_even_when_other_candidate_items_pass(tmp_path):
    bench = make_bench(tmp_path)
    session = bench.start_session("198.51.100.2")
    baseline = answers_for(bench, session["session_id"], solved=True)
    candidate = answers_for(bench, session["session_id"], solved=True)
    first_id = next(iter(candidate))
    candidate[first_id] = {"writes": {"security/forbidden.txt": "disabled\n"}, "deletes": []}

    receipt = bench.submit(payload(session, baseline, candidate), "198.51.100.2")

    assert receipt["verdict"] == "BLOCK"
    assert receipt["regressions"] == 1
    assert receipt["candidate_passed"] == 5
    failed = next(row for row in receipt["items"] if row["item_id"] == first_id)
    assert failed["transition"] == "regression"
    assert failed["candidate"]["failure_codes"][0].startswith("out_of_scope_write:")


def test_public_receipt_removes_answer_derived_failure_details(tmp_path, monkeypatch):
    enable_signing(tmp_path, monkeypatch)
    bench = make_bench(tmp_path)
    session = bench.start_session("198.51.100.9")
    baseline = answers_for(bench, session["session_id"], solved=True)
    candidate = answers_for(bench, session["session_id"], solved=True)
    first_id = next(iter(candidate))
    candidate[first_id] = {"writes": {"private/submitted-path.txt": "secret answer fragment\n"}, "deletes": []}

    receipt = bench.submit(payload(session, baseline, candidate, publish=True), "198.51.100.9")

    assert receipt["verdict"] == "BLOCK"
    assert "private/submitted-path.txt" in json.dumps(receipt)
    raw_ledger = (tmp_path / "public.jsonl").read_text(encoding="utf-8")
    assert "private/submitted-path.txt" not in raw_ledger
    assert "secret answer fragment" not in raw_ledger
    public_item = bench.leaderboard()["entries"][0]["items"][0]
    assert set(public_item["candidate"]) == {"passed", "changed_files"}


def test_no_outcome_change_holds(tmp_path):
    bench = make_bench(tmp_path)
    session = bench.start_session("198.51.100.3")
    baseline = answers_for(bench, session["session_id"], solved=False)
    candidate = answers_for(bench, session["session_id"], solved=False)

    receipt = bench.submit(payload(session, baseline, candidate), "198.51.100.3")

    assert receipt["verdict"] == "HOLD"
    assert receipt["gains"] == receipt["regressions"] == 0
    assert receipt["tie_fail"] == 6


def test_invalid_manifest_does_not_spend_session(tmp_path):
    bench = make_bench(tmp_path)
    session = bench.start_session("198.51.100.4")
    baseline = answers_for(bench, session["session_id"], solved=False)
    candidate = answers_for(bench, session["session_id"], solved=True)
    bad = payload(session, baseline, candidate)
    bad["candidate_manifest"]["name"] = "<script>alert(1)</script>"

    with pytest.raises(OpenBenchError, match="candidate_manifest.name"):
        bench.submit(bad, "198.51.100.4")
    assert session["session_id"] in bench.sessions


def test_publish_requires_explicit_attestation(tmp_path):
    bench = make_bench(tmp_path)
    session = bench.start_session("198.51.100.5")
    baseline = answers_for(bench, session["session_id"], solved=False)
    candidate = answers_for(bench, session["session_id"], solved=True)
    request = payload(session, baseline, candidate, publish=True)
    request["attestation"] = False

    with pytest.raises(OpenBenchError, match="attestation"):
        bench.submit(request, "198.51.100.5")
    assert session["session_id"] in bench.sessions


def test_publication_failure_returns_private_receipt(tmp_path, monkeypatch):
    enable_signing(tmp_path, monkeypatch)
    ledger_path = tmp_path / "public.jsonl"
    ledger_path.mkdir()
    bench = OpenPromotionBench(
        ledger_path=ledger_path,
        rng_factory=lambda: random.Random(17),
        enforce_limits=False,
    )
    session = bench.start_session("198.51.100.8")
    baseline = answers_for(bench, session["session_id"], solved=False)
    candidate = answers_for(bench, session["session_id"], solved=True)

    receipt = bench.submit(payload(session, baseline, candidate, publish=True), "198.51.100.8")

    assert receipt["verdict"] == "PASS"
    assert receipt["publication"] == {"status": "unavailable", "public_id": None}
    assert receipt["session_spent"] is True


def test_expired_session_is_destroyed(tmp_path, monkeypatch):
    bench = make_bench(tmp_path, ttl=1.0)
    monkeypatch.setattr("bcv.open_bench.time.time", lambda: 100.0)
    session = bench.start_session("198.51.100.6")
    baseline = answers_for(bench, session["session_id"], solved=False)
    candidate = answers_for(bench, session["session_id"], solved=True)
    monkeypatch.setattr("bcv.open_bench.time.time", lambda: 102.0)

    with pytest.raises(OpenBenchError, match="expired"):
        bench.submit(payload(session, baseline, candidate), "198.51.100.6")


def test_public_task_contract_is_bounded_and_hashable(tmp_path):
    bench = make_bench(tmp_path)
    session = bench.start_session("198.51.100.7", "caller-12345678")

    assert len(session["tasks"]) == 6
    assert len(session["cohort_sha256"]) == 64
    assert session["track"] == "self_attested_procedural"
    assert session["challenge"] == "caller-12345678"
    for task in session["tasks"]:
        assert set(task) == {"item_id", "title", "request", "repository", "scope"}
        assert task["scope"]["rule"].startswith("Every unlisted path")
        assert all(".." not in path and "\\" not in path for path in task["repository"])
    json.dumps(session)
