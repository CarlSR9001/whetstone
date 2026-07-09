from __future__ import annotations

import json
from dataclasses import dataclass

from bcv.candidates import StoredAnswerCandidate
from bcv.examiner import ExaminerBank
from bcv.registry import MINTABLE_DOMAINS, REGISTRY, grade_bank, mint_domain

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


def test_every_mintable_domain_is_registered():
    for domain in MINTABLE_DOMAINS:
        assert domain in REGISTRY


def test_mint_and_grade_code_domain_with_stored_answers(tmp_path):
    bank = ExaminerBank(tmp_path / "bank")
    report = mint_domain(bank, "code", max_items=3)
    assert report["promoted"] == 3

    by_task = {item.payload["task_id"]: item.item_id for item in bank.promoted_items()}
    answers_path = tmp_path / "answers.jsonl"
    answers_path.write_text(
        json.dumps({"item_id": by_task["balanced_brackets"], "answer": f"```python\n{GOOD_BALANCED}\n```"}) + "\n",
        encoding="utf-8",
    )
    result = grade_bank(bank, "stored_system", StoredAnswerCandidate(answers_path))
    assert result["items"] == 3
    assert result["passed"] == 1  # one right answer, two missing -> graded false
    assert result["burned"] == []
    assert result["run_manifest"]["answers_file"] == "answers.jsonl"
    assert len(result["run_manifest"]["item_set_sha256"]) == 64
    assert GOOD_BALANCED not in json.dumps(result["run_manifest"])

    reloaded = ExaminerBank(tmp_path / "bank")
    graded_item = reloaded.items[by_task["balanced_brackets"]]
    assert graded_item.graded["stored_system"]["pass"] == 1


@dataclass
class FakeExternalCandidate:
    provider: str = "api.example.com/frontier-model"
    is_external: bool = True

    def generate_text(self, prompt: str, temperature: float = 0.0) -> str:
        return "no idea"


def test_external_grading_burns_every_exposed_item(tmp_path):
    bank = ExaminerBank(tmp_path / "bank")
    mint_domain(bank, "code", max_items=2)
    result = grade_bank(bank, "frontier", FakeExternalCandidate())
    assert result["external"] is True
    assert len(result["burned"]) == 2

    reloaded = ExaminerBank(tmp_path / "bank")
    assert len(reloaded.promoted_items()) == 0
    burned = [item for item in reloaded.items.values() if item.status == "burned"]
    assert len(burned) == 2
    for item in burned:
        assert item.exposures and item.exposures[0]["provider"] == "api.example.com/frontier-model"


def test_external_burn_can_only_be_skipped_explicitly(tmp_path):
    bank = ExaminerBank(tmp_path / "bank")
    mint_domain(bank, "code", max_items=2)
    result = grade_bank(bank, "frontier", FakeExternalCandidate(), burn_external=False)
    assert result["burned"] == []
    reloaded = ExaminerBank(tmp_path / "bank")
    assert len(reloaded.promoted_items()) == 2
