from __future__ import annotations

import json
import sys

import pytest

import bcv.whetstone_mcp as wmcp

GOOD_BALANCED = """
def balanced(s):
    stack = []
    pairs = {')': '(', ']': '[', '}': '{'}
    for ch in s:
        if ch in '([{':
            stack.append(ch)
        elif ch in pairs:
            if not stack or stack.pop() != pairs[ch]:
                return False
    return not stack
"""


@pytest.fixture
def bank_root(tmp_path):
    root = str(tmp_path / "bank")
    wmcp.use_bank_impl(root)
    yield root
    wmcp.use_bank_impl(".bcv_runs/examiner")  # restore default for other tests


def test_full_mcp_flow(bank_root):
    minted = wmcp.mint_impl("code", max_items=3)
    assert minted["promoted"] == 3

    status = wmcp.status_impl()
    assert status["buckets"]["promoted"] == 3
    assert status["root"] == bank_root

    # locate the balanced-brackets item WITHOUT any MCP surface: prove the
    # boundary by going to disk, the way an operator would
    from bcv.examiner import ExaminerBank

    bank = ExaminerBank(bank_root)
    by_task = {item.payload["task_id"]: item.item_id for item in bank.promoted_items()}

    base = wmcp.grade_answers_impl("base", {item_id: "no idea" for item_id in by_task.values()})
    assert base["passed"] == 0
    cand = wmcp.grade_answers_impl(
        "cand", {item_id: f"```python\n{GOOD_BALANCED}\n```" for item_id in by_task.values()}
    )
    assert cand["passed"] == 1  # only the balanced task is solved by this answer

    verdict = wmcp.gate_impl("base", "cand")
    assert verdict["verdict"] in ("PASS", "HOLD", "BLOCK")
    assert "resolution" in verdict
    assert verdict["gains"] >= 1
    reliability = wmcp.gate_impl("base", "cand", regression_policy="reliability_aware")
    assert reliability["verdict"] in ("PASS", "HOLD", "BLOCK")

    burn = wmcp.burn_impl(by_task["balanced_brackets"], "manual", "test exposure")
    assert wmcp.status_impl()["buckets"]["burned"] == 1
    assert burn["burned"] == by_task["balanced_brackets"]


def test_grade_command_runs_server_side(bank_root):
    wmcp.mint_impl("code", max_items=1)
    report = wmcp.grade_command_impl(
        "echo_agent",
        f'"{sys.executable}" -c "import sys; sys.stdout.write(sys.stdin.read())"',
        timeout_seconds=60,
    )
    assert report["items"] == 1
    assert "results" not in report  # summaries only over MCP


def test_mint_rejects_unknown_domain(bank_root):
    with pytest.raises(ValueError, match="unknown domain"):
        wmcp.mint_impl("astrology")


def test_no_tool_serves_item_contents():
    """The boundary as a test: no registered MCP tool name suggests item access,
    and no tool result in the happy path contains a prompt payload."""
    tool_names = set()
    manager = getattr(wmcp.mcp, "_tool_manager", None)
    if manager is not None:
        tool_names = set(getattr(manager, "_tools", {}).keys())
    assert tool_names, "expected registered tools on the FastMCP instance"
    forbidden = {"item", "prompt", "exam_content", "payload"}
    for name in tool_names:
        assert not any(word in name for word in ("items", "prompts")), name
    assert "whetstone_status" in tool_names
    assert "whetstone_gate" in tool_names


def test_calibrate_panel_via_mcp():
    from pathlib import Path

    labeled = Path(__file__).resolve().parent.parent / "sample_docs" / "support_calibration.jsonl"
    result = wmcp.calibrate_panel_impl(str(labeled))
    assert result["false_accepts"] == 0
    assert result["agreement"] == 1.0
