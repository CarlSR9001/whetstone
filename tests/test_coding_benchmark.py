from __future__ import annotations

from bcv.coding_benchmark import _run_task_tests, coding_tasks


def test_coding_tasks_have_tests():
    tasks = coding_tasks()

    assert len(tasks) >= 3
    assert all("from solution import" in task.tests or "import pytest" in task.tests for task in tasks)


def test_run_task_tests_executes_solution(tmp_path):
    code = """
def slugify(text):
    return "untitled"
"""
    passed, failure = _run_task_tests(tmp_path, code, coding_tasks()[0].tests)

    assert passed is False
    assert failure

