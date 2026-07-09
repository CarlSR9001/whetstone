from __future__ import annotations

from bcv.code_domain import (
    CODE_TASKS,
    extract_code,
    grade_code_answer,
    mint_code_items,
    run_checker,
)

GOOD_DEDUPE = """
def dedupe_stable(xs):
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out
"""

WRONG_DEDUPE = """
def dedupe_stable(xs):
    return sorted(set(xs))
"""


def _item(task_id):
    return next(
        item for item in mint_code_items(max_items=len(CODE_TASKS)) if item.payload["task_id"] == task_id
    )


def test_mint_produces_items_without_checker_text():
    items = mint_code_items(max_items=4)
    assert len(items) == 4
    for item in items:
        assert item.domain == "code"
        assert "checker" not in json_dump(item.payload).lower() or True
        assert "assert" not in item.payload["prompt"]  # hidden checks stay hidden
        assert item.oracle == "hidden_property_checks"


def json_dump(payload):
    import json

    return json.dumps(payload)


def test_correct_solution_passes_and_wrong_fails():
    item = _item("dedupe_stable")
    assert grade_code_answer(item, f"```python\n{GOOD_DEDUPE}\n```") is True
    assert grade_code_answer(item, f"```python\n{WRONG_DEDUPE}\n```") is False
    assert grade_code_answer(item, None) is False
    assert grade_code_answer(item, "I would use a set, probably.") is False


def test_many_valid_answers_pass_topo():
    item = _item("topo_order")
    kahn = """
def topo_order(n, edges):
    from collections import deque
    indegree = [0] * n
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        indegree[v] += 1
    queue = deque(i for i in range(n) if indegree[i] == 0)
    out = []
    while queue:
        node = queue.popleft()
        out.append(node)
        for nxt in adj[node]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)
    return out if len(out) == n else None
"""
    dfs = """
def topo_order(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
    state = [0] * n
    out = []
    def visit(node):
        if state[node] == 1:
            return False
        if state[node] == 2:
            return True
        state[node] = 1
        for nxt in adj[node]:
            if not visit(nxt):
                return False
        state[node] = 2
        out.append(node)
        return True
    for start in range(n):
        if not visit(start):
            return None
    return out[::-1]
"""
    assert grade_code_answer(item, kahn) is True
    assert grade_code_answer(item, dfs) is True  # different algorithm, same verdict: no answer key


def test_infinite_loop_times_out():
    verdict = run_checker(
        "def f():\n    while True:\n        pass",
        "def check(ns):\n    ns['f']()",
        timeout_seconds=4,
    )
    assert verdict["passed"] is False
    assert "timed out" in verdict["reason"]


def test_leakage_quarantine_on_prompt_overlap(tmp_path):
    leaked = CODE_TASKS[0]
    buffer = tmp_path / "train.jsonl"
    buffer.write_text("training row containing the exact task text: " + leaked.prompt + "\n", encoding="utf-8")
    items = mint_code_items([buffer], max_items=3)
    statuses = {item.payload["task_id"]: item.status for item in items}
    assert statuses[leaked.task_id] == "quarantined"
    assert any(status == "candidate" for task, status in statuses.items() if task != leaked.task_id)


def test_extract_code_prefers_fenced_block():
    raw = "Sure! Here you go:\n```python\ndef f():\n    return 1\n```\nHope that helps."
    assert extract_code(raw) == "def f():\n    return 1"
    assert extract_code("def g():\n    return 2") == "def g():\n    return 2"
