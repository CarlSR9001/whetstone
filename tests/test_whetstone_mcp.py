from __future__ import annotations

import asyncio
import json
import sys
import tomllib
from pathlib import Path

import pytest
from mcp import Client

import bcv.whetstone_mcp as wmcp
from bcv._version import __version__

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


def test_no_tool_serves_item_contents(bank_root):
    """The boundary as a test: no registered MCP tool name suggests item access,
    and no tool result in the happy path contains a prompt payload."""
    async def inspect_surface():
        async with Client(wmcp.mcp) as client:
            tools = await client.list_tools()
            status = await client.call_tool("whetstone_status", {})
            return tools, status

    tools, status = asyncio.run(inspect_surface())
    tool_names = {tool.name for tool in tools.tools}
    assert tools.meta["io.modelcontextprotocol/serverInfo"] == {
        "name": "whetstone",
        "version": __version__,
    }
    assert status.is_error is False
    assert json.loads(status.content[0].text)["root"] == bank_root
    for name in tool_names:
        assert not any(word in name for word in ("items", "prompts")), name
    assert "whetstone_status" in tool_names
    assert "whetstone_gate" in tool_names


def test_reasoning_emulator_registers_with_mcp_v2():
    from bcv import emulator_mcp

    tools = asyncio.run(emulator_mcp.mcp.list_tools())
    tool_names = {tool.name for tool in tools}
    assert {"emu_start", "emu_step", "emu_status", "emu_check", "emu_result"} <= tool_names


def test_package_requires_mcp_v2():
    project = tomllib.loads((Path(__file__).resolve().parent.parent / "pyproject.toml").read_text(encoding="utf-8"))
    extras = project["project"]["optional-dependencies"]
    for extra in ("agents", "all", "test"):
        assert [requirement for requirement in extras[extra] if requirement.startswith("mcp")] == ["mcp>=2,<3"]


def test_calibrate_panel_via_mcp():
    labeled = Path(__file__).resolve().parent.parent / "sample_docs" / "support_calibration.jsonl"
    result = wmcp.calibrate_panel_impl(str(labeled))
    assert result["false_accepts"] == 0
    assert result["agreement"] == 1.0
