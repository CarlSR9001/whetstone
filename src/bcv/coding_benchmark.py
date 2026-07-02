from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from bcv.local_model import LocalModelClient, LocalModelError, auto_local_client
from bcv.schema import Event, TestResult
from bcv.store import CognitiveStore


@dataclass(frozen=True)
class CodingTask:
    name: str
    prompt: str
    tests: str


@dataclass(frozen=True)
class CodingAttempt:
    task: str
    editor: str
    attempt: int
    passed: bool
    failure: str | None
    code_chars: int


@dataclass(frozen=True)
class CodingSummary:
    editor: str
    tasks: int
    passed: int
    total_attempts: int


@dataclass(frozen=True)
class CodingBenchmarkResult:
    attempts: tuple[CodingAttempt, ...]
    summaries: tuple[CodingSummary, ...]


def coding_tasks() -> tuple[CodingTask, ...]:
    return (
        CodingTask(
            name="slugify",
            prompt=(
                "Implement function slugify(text: str) -> str. It should lowercase text, replace runs of "
                "non-alphanumeric characters with one hyphen, strip leading/trailing hyphens, preserve digits, "
                "and return 'untitled' if the result is empty."
            ),
            tests="""
from solution import slugify

def test_slugify_basic():
    assert slugify("Hello, World!") == "hello-world"
    assert slugify("  Build   Verifier #42 ") == "build-verifier-42"
    assert slugify("!!!") == "untitled"
""",
        ),
        CodingTask(
            name="merge_intervals",
            prompt=(
                "Implement function merge_intervals(intervals). Each interval is a tuple (start, end). Return a "
                "list of merged inclusive intervals sorted by start. Adjacent intervals should merge when the next "
                "start is less than or equal to the current end."
            ),
            tests="""
from solution import merge_intervals

def test_merge_intervals():
    assert merge_intervals([(5, 7), (1, 3), (2, 4)]) == [(1, 4), (5, 7)]
    assert merge_intervals([(1, 2), (2, 3), (10, 12)]) == [(1, 3), (10, 12)]
    assert merge_intervals([]) == []
""",
        ),
        CodingTask(
            name="parse_duration",
            prompt=(
                "Implement function parse_duration(text: str) -> int. Parse compact duration strings like '2h 5m 3s', "
                "'90s', and '1h30m'. Return total seconds. Valid units are h, m, s. Ignore extra whitespace. "
                "Raise ValueError if no duration tokens are present."
            ),
            tests="""
import pytest
from solution import parse_duration

def test_parse_duration():
    assert parse_duration("2h 5m 3s") == 7503
    assert parse_duration("1h30m") == 5400
    assert parse_duration("90s") == 90
    with pytest.raises(ValueError):
        parse_duration("nothing")
""",
        ),
        CodingTask(
            name="toposort",
            prompt=(
                "Implement function topo_sort(edges). edges is an iterable of (before, after) dependency pairs. "
                "Return a list containing every node exactly once in a valid topological order. Include nodes that "
                "appear only as dependencies. If there is a cycle, raise ValueError. When multiple nodes are available, "
                "return them in lexicographic order for deterministic output."
            ),
            tests="""
import pytest
from solution import topo_sort

def test_topo_sort_order_and_cycle():
    assert topo_sort([("cook", "eat"), ("shop", "cook"), ("plan", "shop")]) == ["plan", "shop", "cook", "eat"]
    assert topo_sort([("a", "c"), ("b", "c")]) == ["a", "b", "c"]
    with pytest.raises(ValueError):
        topo_sort([("a", "b"), ("b", "a")])
""",
        ),
        CodingTask(
            name="mini_csv",
            prompt=(
                "Implement function parse_csv_line(line: str) -> list[str]. Parse one CSV record with comma separators, "
                "double-quoted fields, escaped quotes represented by two double quotes, empty fields, and commas inside "
                "quoted fields. Do not use the csv module."
            ),
            tests="""
from solution import parse_csv_line

def test_parse_csv_line():
    assert parse_csv_line('a,b,c') == ['a', 'b', 'c']
    assert parse_csv_line('"a,b",c,"d""e"') == ['a,b', 'c', 'd"e']
    assert parse_csv_line(',"",tail,') == ['', '', 'tail', '']
""",
        ),
        CodingTask(
            name="rename_function",
            prompt=(
                "Implement function rename_function(source: str, old: str, new: str) -> str. Rename a Python function "
                "definition and calls to that function using the ast module. Do not rename strings, comments, attributes, "
                "or unrelated variables. Return source code as a string."
            ),
            tests="""
from solution import rename_function

def test_rename_function_ast_safe():
    src = '''
def target(x):
    return x + 1

def wrapper():
    msg = "target should stay in this string"
    return target(4)
'''
    out = rename_function(src, 'target', 'renamed')
    assert 'def renamed' in out
    assert 'renamed(4)' in out
    assert 'target should stay in this string' in out
    assert 'def target' not in out
""",
        ),
    )


def run_coding_benchmark(
    root: str | Path = ".bcv_runs/coding_benchmark",
    client: LocalModelClient | None = None,
    task_limit: int | None = None,
    branch_attempts: int = 3,
    model_timeout_s: int = 30,
) -> CodingBenchmarkResult:
    root = Path(root)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    client = client or auto_local_client()
    client.timeout_s = min(client.timeout_s, model_timeout_s)
    attempts: list[CodingAttempt] = []
    tasks = coding_tasks()[:task_limit] if task_limit is not None else coding_tasks()
    for task in tasks:
        attempts.extend(_run_direct_task(root / "direct" / task.name, task, client))
        attempts.extend(_run_branch_task(root / "branch" / task.name, task, client, max_attempts=branch_attempts))

    summaries = tuple(
        _summary(editor, [attempt for attempt in attempts if attempt.editor == editor])
        for editor in ("direct", "branch_agent")
    )
    result = CodingBenchmarkResult(tuple(attempts), summaries)
    (root / "coding_result.json").write_text(
        json.dumps(asdict(result), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return result


def _run_direct_task(root: Path, task: CodingTask, client: LocalModelClient) -> list[CodingAttempt]:
    root.mkdir(parents=True, exist_ok=True)
    code = _generate_solution(client, _coding_prompt(task), temperature=0.0)
    passed, failure = _run_task_tests(root, code, task.tests)
    return [
        CodingAttempt(
            task=task.name,
            editor="direct",
            attempt=1,
            passed=passed,
            failure=failure,
            code_chars=len(code),
        )
    ]


def _run_branch_task(root: Path, task: CodingTask, client: LocalModelClient, max_attempts: int = 3) -> list[CodingAttempt]:
    root.mkdir(parents=True, exist_ok=True)
    store = CognitiveStore(root / "ledger")
    store.init()
    branch = "experiment/coding-agent"
    store.create_branch(branch, from_branch="main")
    attempts: list[CodingAttempt] = []
    previous_code = ""
    previous_failure = ""
    for attempt in range(1, max_attempts + 1):
        prompt = _coding_prompt(task, previous_code=previous_code, previous_failure=previous_failure)
        code = _generate_solution(client, prompt, temperature=0.0)
        passed, failure = _run_task_tests(root / f"attempt_{attempt}", code, task.tests)
        metric = CodingAttempt(
            task=task.name,
            editor="branch_agent",
            attempt=attempt,
            passed=passed,
            failure=failure,
            code_chars=len(code),
        )
        attempts.append(metric)
        store.commit(
            branch,
            f"{task.name} attempt {attempt}",
            [
                Event(
                    event_type="verifier_result",
                    actor="verifier",
                    message="tests passed" if passed else (failure or "tests failed"),
                    output_refs=(f"code:{task.name}",),
                    tests=(
                        TestResult(
                            "python_tests",
                            "pass" if passed else "fail",
                            json.dumps(asdict(metric), sort_keys=True),
                        ),
                    ),
                )
            ],
        )
        if passed:
            break
        previous_code = code
        previous_failure = failure or ""
    return attempts


def _coding_prompt(
    task: CodingTask,
    previous_code: str = "",
    previous_failure: str = "",
) -> str:
    repair = ""
    if previous_failure:
        repair = f"""
Previous code failed tests.

Previous code:
```python
{previous_code}
```

Test failure:
{previous_failure}

Return a corrected solution.py.
"""
    return f"""/no_think
You are writing a single Python module named solution.py.

Task:
{task.prompt}

Tests that will be run:
```python
{task.tests}
```

{repair}

Return only complete Python source for solution.py.
Do not include markdown fences or commentary. Do not omit imports needed by the solution.
"""


def _generate_solution(client: LocalModelClient, prompt: str, temperature: float) -> str:
    try:
        response = client.generate(prompt, temperature=temperature, json_mode=False)
        text = response.text
        fenced = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            return fenced.group(1).strip() + "\n"
        if "def " in text or "import " in text or "from " in text:
            return _strip_non_code_prefix(text)
    except LocalModelError:
        return "raise RuntimeError('local model generation failed')\n"
    return "raise RuntimeError('model did not return code')\n"


def _strip_non_code_prefix(text: str) -> str:
    lines = text.strip().splitlines()
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(("import ", "from ", "def ", "class ")):
            return "\n".join(lines[index:]).strip() + "\n"
    return text.strip() + "\n"


def _run_task_tests(root: Path, code: str, tests: str) -> tuple[bool, str | None]:
    root.mkdir(parents=True, exist_ok=True)
    (root / "solution.py").write_text(code, encoding="utf-8")
    (root / "test_solution.py").write_text(textwrap.dedent(tests), encoding="utf-8")
    try:
        completed = subprocess.run(
            ["python", "-m", "pytest", "-q", "test_solution.py"],
            cwd=root,
            text=True,
            capture_output=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired as exc:
        return False, f"pytest timed out after {exc.timeout} seconds"
    if completed.returncode == 0:
        return True, None
    failure = (completed.stdout + "\n" + completed.stderr).strip()
    return False, failure[-4000:]


def _summary(editor: str, attempts: list[CodingAttempt]) -> CodingSummary:
    tasks = {attempt.task for attempt in attempts}
    passed_tasks = {
        task
        for task in tasks
        if any(attempt.task == task and attempt.passed for attempt in attempts)
    }
    return CodingSummary(
        editor=editor,
        tasks=len(tasks),
        passed=len(passed_tasks),
        total_attempts=len(attempts),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run one bounded smoke task.")
    parser.add_argument("--model-timeout-s", type=int, default=30)
    args = parser.parse_args()
    task_limit = 1 if args.quick else None
    branch_attempts = 2 if args.quick else 3
    print(
        json.dumps(
            asdict(
                run_coding_benchmark(
                    task_limit=task_limit,
                    branch_attempts=branch_attempts,
                    model_timeout_s=args.model_timeout_s,
                )
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
