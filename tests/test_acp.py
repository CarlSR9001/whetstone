from __future__ import annotations

import sys
from pathlib import Path

import pytest

from bcv.acp import ACPCandidate
from bcv.candidates import CandidateError

FAKE_AGENT = str(Path(__file__).resolve().parent / "fake_acp_agent.py")


def _candidate(*extra: str) -> ACPCandidate:
    return ACPCandidate([sys.executable, FAKE_AGENT, *extra], timeout_seconds=30)


def test_prompt_round_trip_reassembles_chunks():
    candidate = _candidate()
    try:
        assert candidate.generate_text("repair is_tree") == "parrot:repair is_tree"
        # session survives across prompts: one process, sequential turns
        assert candidate.generate_text("second") == "parrot:second"
    finally:
        candidate.close()


def test_fixed_answer_mode():
    candidate = _candidate("--answer", '{"repair_expression": "(is_tree) and (m <= 4)"}')
    try:
        assert candidate.generate_text("anything") == '{"repair_expression": "(is_tree) and (m <= 4)"}'
    finally:
        candidate.close()


def test_permission_requests_are_cancelled():
    candidate = _candidate("--ask-permission")
    try:
        reply = candidate.generate_text("try something")
        assert reply.endswith("|permission:cancelled")
    finally:
        candidate.close()


def test_unsupported_protocol_version_is_loud():
    candidate = _candidate("--wrong-version")
    try:
        with pytest.raises(CandidateError, match="unsupported ACP version"):
            candidate.generate_text("x")
    finally:
        candidate.close()


def test_agent_crash_is_loud():
    candidate = ACPCandidate([sys.executable, "-c", "import sys; sys.exit(2)"], timeout_seconds=10)
    try:
        with pytest.raises(CandidateError, match="agent exited"):
            candidate.generate_text("x")
    finally:
        candidate.close()


def test_acp_candidate_grades_through_registry(tmp_path):
    from bcv.examiner import ExaminerBank
    from bcv.registry import grade_bank, mint_domain

    bank = ExaminerBank(tmp_path / "bank")
    mint_domain(bank, "code", max_items=1)
    solution = (
        "```python\\n"
        "def rle_encode(s):\\n"
        "    pairs = []\\n"
        "    for ch in s:\\n"
        "        if pairs and pairs[-1][0] == ch:\\n"
        "            pairs[-1] = (ch, pairs[-1][1] + 1)\\n"
        "        else:\\n"
        "            pairs.append((ch, 1))\\n"
        "    return pairs\\n"
        "def rle_decode(pairs):\\n"
        "    return ''.join(ch * n for ch, n in pairs)\\n"
        "```"
    )
    candidate = _candidate("--answer", solution.replace("\\n", "\n"))
    try:
        report = grade_bank(bank, "acp_agent", candidate)
    finally:
        candidate.close()
    assert report["items"] == 1
    assert report["passed"] == 1
    assert report["burned"] == []  # ACP agents run inside the trust boundary
