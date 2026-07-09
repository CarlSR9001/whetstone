"""Code tasks: the first non-toy exam domain, graded by hidden property checks.

The whetstone rule, applied to code: the system under exam sees a task
description; the checker it must satisfy stays private. Checkers verify
PROPERTIES — round-trips, invariants, brute-force oracles on small inputs —
so there is no canonical solution string anywhere in the system. A topological
sort task has exponentially many correct answers; all of them pass, none of
them are stored.

Submitted code runs in an isolated subprocess (``python -I``, no inherited
site-packages or environment, hard wall-clock timeout). That is process
isolation for honest measurement, not a security sandbox for hostile code —
production deployments should run graders inside their existing containment.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

from bcv.examiner import ExamItem


@dataclass(frozen=True)
class CodeTask:
    task_id: str
    prompt: str  # everything the system under exam sees
    checker_source: str  # private: defines check(namespace) raising AssertionError


CODE_TASKS: tuple[CodeTask, ...] = (
    CodeTask(
        task_id="rle_roundtrip",
        prompt=(
            "Define two functions. rle_encode(s: str) -> list[tuple[str, int]] compresses a "
            "string into maximal (character, run_length) pairs. rle_decode(pairs) -> str inverts it."
        ),
        checker_source="""
import random
def check(ns):
    encode, decode = ns["rle_encode"], ns["rle_decode"]
    rng = random.Random(7)
    cases = ["", "a", "aaabbbcccd", "abababab", "zzzzzzzzzz"]
    cases += ["".join(rng.choice("abc") for _ in range(rng.randrange(0, 40))) for _ in range(30)]
    for s in cases:
        pairs = encode(s)
        assert decode(pairs) == s, f"round trip failed for {s!r}"
        assert all(count >= 1 for _, count in pairs)
        assert all(pairs[i][0] != pairs[i + 1][0] for i in range(len(pairs) - 1)), "runs not maximal"
""",
    ),
    CodeTask(
        task_id="merge_intervals",
        prompt=(
            "Define merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]] that "
            "merges overlapping or touching closed integer intervals and returns them sorted."
        ),
        checker_source="""
import random
def check(ns):
    merge = ns["merge_intervals"]
    rng = random.Random(11)
    for _ in range(40):
        raw = [(a, a + rng.randrange(0, 6)) for a in (rng.randrange(0, 30) for _ in range(rng.randrange(0, 8)))]
        merged = merge(list(raw))
        assert merged == sorted(merged), "output not sorted"
        assert all(a <= b for a, b in merged)
        assert all(merged[i + 1][0] > merged[i][1] + 1 for i in range(len(merged) - 1)), "not fully merged"
        covered = {x for a, b in raw for x in range(a, b + 1)}
        covered_out = {x for a, b in merged for x in range(a, b + 1)}
        assert covered == covered_out, "coverage changed"
""",
    ),
    CodeTask(
        task_id="balanced_brackets",
        prompt=(
            "Define balanced(s: str) -> bool for strings of ()[]{} deciding whether every bracket "
            "is closed by the matching kind in the right order."
        ),
        checker_source="""
import random
def check(ns):
    balanced = ns["balanced"]
    def oracle(s):
        stack = []
        pairs = {')': '(', ']': '[', '}': '{'}
        for ch in s:
            if ch in '([{':
                stack.append(ch)
            elif not stack or stack.pop() != pairs[ch]:
                return False
        return not stack
    rng = random.Random(3)
    cases = ["", "()", "([]{})", "(]", "([)]", "((("]
    cases += ["".join(rng.choice("()[]{}") for _ in range(rng.randrange(0, 14))) for _ in range(60)]
    for s in cases:
        assert balanced(s) == oracle(s), f"disagrees with oracle on {s!r}"
""",
    ),
    CodeTask(
        task_id="topo_order",
        prompt=(
            "Define topo_order(n: int, edges: list[tuple[int, int]]) -> list[int] | None returning "
            "any topological order of nodes 0..n-1 for directed edges (u, v) meaning u before v, "
            "or None when the graph has a cycle."
        ),
        checker_source="""
import random
def check(ns):
    topo = ns["topo_order"]
    rng = random.Random(5)
    for _ in range(30):
        n = rng.randrange(1, 9)
        nodes = list(range(n)); rng.shuffle(nodes)
        dag = [(nodes[i], nodes[j]) for i in range(n) for j in range(i + 1, n) if rng.random() < 0.3]
        order = topo(n, list(dag))
        assert order is not None and sorted(order) == list(range(n)), "not a permutation"
        position = {node: k for k, node in enumerate(order)}
        assert all(position[u] < position[v] for u, v in dag), "edge violated"
    assert topo(2, [(0, 1), (1, 0)]) is None, "cycle not detected"
    assert topo(3, [(0, 1), (1, 2), (2, 0)]) is None, "cycle not detected"
""",
    ),
    CodeTask(
        task_id="mode_stable",
        prompt=(
            "Define mode_stable(xs: list[int]) -> int | None returning the most frequent value; "
            "ties break toward the value whose FIRST occurrence is earliest. Empty list returns None."
        ),
        checker_source="""
import random
def check(ns):
    mode = ns["mode_stable"]
    def oracle(xs):
        if not xs:
            return None
        counts = {}
        for x in xs:
            counts[x] = counts.get(x, 0) + 1
        best = max(counts.values())
        for x in xs:
            if counts[x] == best:
                return x
    rng = random.Random(13)
    cases = [[], [4], [1, 2, 2, 1], [3, 1, 3, 1, 2]]
    cases += [[rng.randrange(0, 5) for _ in range(rng.randrange(0, 20))] for _ in range(50)]
    for xs in cases:
        assert mode(list(xs)) == oracle(xs), f"wrong on {xs}"
""",
    ),
    CodeTask(
        task_id="base_digits",
        prompt=(
            "Define base_digits(n: int, base: int) -> list[int] returning the digits of a "
            "non-negative integer in the given base (2..16), most significant first, with no "
            "leading zeros; zero is [0]."
        ),
        checker_source="""
import random
def check(ns):
    digits_fn = ns["base_digits"]
    rng = random.Random(17)
    cases = [(0, 2), (1, 2), (255, 16), (100, 10)]
    cases += [(rng.randrange(0, 10**6), rng.randrange(2, 17)) for _ in range(60)]
    for n, base in cases:
        digits = digits_fn(n, base)
        assert digits and all(0 <= d < base for d in digits), f"digit range wrong for {n} base {base}"
        assert digits == [0] or digits[0] != 0, "leading zero"
        value = 0
        for d in digits:
            value = value * base + d
        assert value == n, f"value mismatch for {n} base {base}"
""",
    ),
    CodeTask(
        task_id="dedupe_stable",
        prompt=(
            "Define dedupe_stable(xs: list) -> list removing duplicates while keeping the first "
            "occurrence of each value in the original order. Values may be unhashable-free "
            "(assume hashable)."
        ),
        checker_source="""
import random
def check(ns):
    dedupe = ns["dedupe_stable"]
    rng = random.Random(19)
    cases = [[], [1], [1, 1, 1], [2, 1, 2, 3, 1]]
    cases += [[rng.randrange(0, 6) for _ in range(rng.randrange(0, 25))] for _ in range(50)]
    for xs in cases:
        out = dedupe(list(xs))
        assert len(set(out)) == len(out), "still has duplicates"
        assert set(out) == set(xs), "value set changed"
        seen = []
        for x in xs:
            if x not in seen:
                seen.append(x)
        assert out == seen, f"order not first-occurrence-stable on {xs}"
""",
    ),
    CodeTask(
        task_id="longest_common_prefix",
        prompt=(
            "Define longest_common_prefix(strs: list[str]) -> str returning the longest string "
            "that is a prefix of every input; empty list returns ''."
        ),
        checker_source="""
import random
def check(ns):
    lcp = ns["longest_common_prefix"]
    rng = random.Random(23)
    cases = [[], ["abc"], ["abc", "abd"], ["", "a"], ["same", "same"]]
    cases += [["pre" + "".join(rng.choice("xy") for _ in range(rng.randrange(0, 5))) for _ in range(rng.randrange(1, 5))] for _ in range(40)]
    for strs in cases:
        out = lcp(list(strs))
        assert all(s.startswith(out) for s in strs), f"not a common prefix on {strs}"
        if strs:
            longer_exists = all(len(s) > len(out) for s in strs) and len({s[len(out)] for s in strs}) == 1
            assert not longer_exists, f"prefix not maximal on {strs}"
        else:
            assert out == ""
""",
    ),
)

TASKS_BY_ID = {task.task_id: task for task in CODE_TASKS}


# ------------------------------------------------------------------ minting


def training_overlap(buffer_paths: list[str | Path], text: str) -> bool:
    """Row-level leakage: the exact task prompt appears in a training buffer."""
    for path in buffer_paths:
        path = Path(path)
        if path.exists() and text in path.read_text(encoding="utf-8"):
            return True
    return False


def mint_code_items(
    buffer_paths: list[str | Path] | None = None,
    max_items: int = 8,
    seed: int = 0,
) -> list[ExamItem]:
    buffer_paths = buffer_paths or []
    items: list[ExamItem] = []
    for task in CODE_TASKS[:max_items]:
        leaked = training_overlap(buffer_paths, task.prompt)
        item = ExamItem(
            item_id=f"code_{uuid.uuid4().hex[:8]}",
            domain="code",
            kind="code_task",
            payload={"task_id": task.task_id, "prompt": task.prompt},
            oracle="hidden_property_checks",
            source="task_library_v1",
            horizon="property_checks_v1",
            lineage=[task.task_id],
            leakage_risk=1.0 if leaked else 0.0,
        )
        if leaked:
            item.status = "quarantined"
        items.append(item)
    return items


# ------------------------------------------------------------------ grading


def code_item_prompt(item: ExamItem) -> str:
    return (
        f"{item.payload['prompt']}\n\n"
        "Reply with a single Python code block containing only the required definition(s). "
        "No commentary outside the block."
    )


def extract_code(raw: str) -> str:
    fenced = re.findall(r"```(?:python)?\s*(.*?)```", raw, re.DOTALL)
    if fenced:
        return max(fenced, key=len).strip()
    return raw.strip()


_HARNESS = """
import json, sys
namespace = {}
try:
    exec(compile(CANDIDATE_SOURCE, "<candidate>", "exec"), namespace)
    exec(compile(CHECKER_SOURCE, "<checker>", "exec"), namespace)
    namespace["check"](namespace)
except AssertionError as error:
    print(json.dumps({"passed": False, "reason": f"check failed: {error}"}))
except BaseException as error:
    print(json.dumps({"passed": False, "reason": f"{type(error).__name__}: {error}"}))
else:
    print(json.dumps({"passed": True, "reason": "all property checks passed"}))
"""


def run_checker(candidate_source: str, checker_source: str, timeout_seconds: float = 15.0) -> dict:
    """Run candidate + private checker in an isolated interpreter with a hard timeout."""
    program = (
        f"CANDIDATE_SOURCE = {candidate_source!r}\n"
        f"CHECKER_SOURCE = {checker_source!r}\n"
        f"{_HARNESS}"
    )
    with tempfile.TemporaryDirectory() as scratch:
        path = Path(scratch) / "graded_run.py"
        path.write_text(program, encoding="utf-8")
        try:
            completed = subprocess.run(
                [sys.executable, "-I", str(path)],
                capture_output=True,
                timeout=timeout_seconds,
                cwd=scratch,
            )
        except subprocess.TimeoutExpired:
            return {"passed": False, "reason": f"timed out after {timeout_seconds}s"}
    for line in reversed(completed.stdout.decode("utf-8", "replace").splitlines()):
        if line.strip().startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return {
        "passed": False,
        "reason": f"no verdict emitted (exit {completed.returncode}): "
        f"{completed.stderr.decode('utf-8', 'replace')[:300]}",
    }


def grade_code_answer(item: ExamItem, raw_answer: str | None, timeout_seconds: float = 15.0) -> bool:
    if not raw_answer:
        return False
    task = TASKS_BY_ID.get(item.payload.get("task_id", ""))
    if task is None:
        return False
    source = extract_code(str(raw_answer))
    if not source:
        return False
    return bool(run_checker(source, task.checker_source, timeout_seconds).get("passed"))
