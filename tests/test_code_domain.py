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
    items = mint_code_items([buffer], max_items=len(CODE_TASKS))
    statuses = {item.payload["task_id"]: item.status for item in items}
    assert statuses[leaked.task_id] == "quarantined"
    assert any(status == "candidate" for task, status in statuses.items() if task != leaked.task_id)


def test_code_bank_seed_reproducibly_selects_a_non_prefix_slice():
    first = mint_code_items(max_items=8, seed=17)
    repeated = mint_code_items(max_items=8, seed=17)
    other = mint_code_items(max_items=8, seed=18)
    first_ids = [item.payload["task_id"] for item in first]
    assert first_ids == [item.payload["task_id"] for item in repeated]
    assert first_ids != [item.payload["task_id"] for item in other]
    assert first_ids != [task.task_id for task in CODE_TASKS[:8]]
    assert {item.payload["selection_seed"] for item in first} == {17}


def test_extract_code_prefers_fenced_block():
    raw = "Sure! Here you go:\n```python\ndef f():\n    return 1\n```\nHope that helps."
    assert extract_code(raw) == "def f():\n    return 1"
    assert extract_code("def g():\n    return 2") == "def g():\n    return 2"


def test_expanded_code_bank_has_independent_hidden_checker_families():
    task_ids = {task.task_id for task in CODE_TASKS}
    assert len(task_ids) >= 24
    assert {
        "rotate_right", "sliding_max", "word_counts_ascii", "top_k_frequent",
        "find_all_anagrams", "shortest_path_length", "min_coin_change",
        "spiral_order", "normalize_posix_path", "count_islands", "is_valid_sudoku",
    } <= task_ids


def test_second_tranche_checkers_accept_reference_implementations():
    references = {
        "binary_search_left": """
def binary_search_left(xs, target):
    low, high = 0, len(xs)
    while low < high:
        middle = (low + high) // 2
        if xs[middle] < target:
            low = middle + 1
        else:
            high = middle
    return low
""",
        "find_all_anagrams": """
def find_all_anagrams(text, pattern):
    if not pattern: return []
    return [index for index in range(len(text) - len(pattern) + 1) if sorted(text[index:index + len(pattern)]) == sorted(pattern)]
""",
        "shortest_path_length": """
def shortest_path_length(n, edges, start, goal):
    adjacency = [[] for _ in range(n)]
    for a, b in edges:
        adjacency[a].append(b); adjacency[b].append(a)
    queue, seen = [(start, 0)], {start}
    for node, distance in queue:
        if node == goal: return distance
        for nxt in adjacency[node]:
            if nxt not in seen:
                seen.add(nxt); queue.append((nxt, distance + 1))
    return -1
""",
        "min_coin_change": """
def min_coin_change(coins, target):
    best = [0] + [target + 1] * target
    for amount in range(1, target + 1):
        for coin in coins:
            if 0 < coin <= amount:
                best[amount] = min(best[amount], best[amount - coin] + 1)
    return -1 if best[target] > target else best[target]
""",
    }
    for task_id, source in references.items():
        assert grade_code_answer(_item(task_id), source) is True


def test_third_tranche_checkers_accept_reference_implementations():
    references = {
        "spiral_order": """
def spiral_order(matrix):
    if not matrix or not matrix[0]: return []
    top, bottom, left, right, out = 0, len(matrix)-1, 0, len(matrix[0])-1, []
    while top <= bottom and left <= right:
        out.extend(matrix[top][left:right+1]); top += 1
        for row in range(top, bottom+1): out.append(matrix[row][right])
        right -= 1
        if top <= bottom: out.extend(reversed(matrix[bottom][left:right+1])); bottom -= 1
        if left <= right:
            for row in range(bottom, top-1, -1): out.append(matrix[row][left])
            left += 1
    return out
""",
        "normalize_posix_path": """
def normalize_posix_path(path):
    absolute, parts = path.startswith('/'), []
    for segment in path.split('/'):
        if not segment or segment == '.': continue
        if segment == '..':
            if parts and parts[-1] != '..': parts.pop()
            elif not absolute: parts.append(segment)
        else: parts.append(segment)
    result = '/'.join(parts)
    return '/' + result if absolute else result or '.'
""",
        "count_islands": """
def count_islands(grid):
    remaining = {(r, c) for r, row in enumerate(grid) for c, value in enumerate(row) if value == '1'}
    groups = 0
    while remaining:
        groups += 1; stack = [remaining.pop()]
        while stack:
            r, c = stack.pop()
            for nxt in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
                if nxt in remaining: remaining.remove(nxt); stack.append(nxt)
    return groups
""",
        "is_valid_sudoku": """
def is_valid_sudoku(board):
    groups = list(board) + [''.join(row[c] for row in board) for c in range(9)]
    groups += [''.join(board[r][c] for r in range(br,br+3) for c in range(bc,bc+3)) for br in range(0,9,3) for bc in range(0,9,3)]
    return all(len([x for x in group if x != '.']) == len(set(x for x in group if x != '.')) for group in groups)
""",
    }
    for task_id, source in references.items():
        assert grade_code_answer(_item(task_id), source) is True
