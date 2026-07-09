from __future__ import annotations

import json

from bcv.examiner import (
    ExamItem,
    ExaminerBank,
    grade_game_answer,
    grade_repair_answer,
    training_originals,
)


def _repair_item(original="is_tree", status="candidate"):
    return ExamItem(
        item_id="t1",
        domain="coloring",
        kind="repair",
        payload={"original_expression": original, "counterexample": "n6:...", "claim": "greedy optimal"},
        oracle="exact_verifier+stress_pool",
        source="test",
        horizon="n<=6",
        lineage=[],
        status=status,
    )


def test_downward_only_flow(tmp_path):
    bank = ExaminerBank(tmp_path)
    item = _repair_item()
    bank.add(item)
    assert bank.promote("t1")
    assert bank.promoted_items()[0].item_id == "t1"
    bank.retire("t1")
    assert item.status == "retired"
    # No path back up: promote() only lifts candidates.
    assert not bank.promote("t1")
    assert bank.trainable_rows()[0]["item_id"] == "t1"
    bank.save()
    reloaded = ExaminerBank(tmp_path)
    assert reloaded.items["t1"].status == "retired"


def test_leakage_quarantine_blocks_promotion(tmp_path):
    bank = ExaminerBank(tmp_path)
    item = _repair_item()
    item.leakage_risk = 1.0
    item.status = "quarantined"
    bank.add(item)
    assert not bank.promote("t1")


def test_training_originals_extraction(tmp_path):
    buffer = tmp_path / "buffer.jsonl"
    buffer.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "system", "content": "x"},
                    {"role": "user", "content": json.dumps({"original_expression": "is_tree"})},
                    {"role": "assistant", "content": "{}"},
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert training_originals([buffer]) == {"is_tree"}


def test_grade_events_preserve_item_level_paired_evidence(tmp_path):
    bank = ExaminerBank(tmp_path)
    item = _repair_item(status="promoted")
    bank.add(item)
    bank.record_grades("base", {"t1": False})
    event = json.loads((tmp_path / "grade_events.jsonl").read_text(encoding="utf-8"))
    assert event["system"] == "base"
    assert event["results"] == {"t1": False}


def test_checker_spec_grading_no_answer_key():
    item = _repair_item(original="is_tree")
    # Any verified strict refinement passes; a wrong one fails; garbage fails.
    assert grade_repair_answer(item, "(is_tree) and (max_degree >= 3)")
    assert not grade_repair_answer(item, "(is_tree) and (max_degree_le_2)")
    assert not grade_repair_answer(item, "is_bipartite and max_degree >= 3")  # not a refinement
    assert not grade_repair_answer(item, "not_a_feature > 1")
    assert not grade_repair_answer(item, None)


def test_game_grading_and_saturation_sweep(tmp_path):
    bank = ExaminerBank(tmp_path)
    game = ExamItem(
        item_id="g1",
        domain="playground",
        kind="game_move",
        payload={"acceptable": [["place", 3], ["place", 5]], "state": [0] * 7, "player": 0, "rules": {}},
        oracle="simulator",
        source="test",
        horizon="depth2",
        lineage=[],
        status="promoted",
    )
    bank.add(game)
    assert grade_game_answer(game, ("place", 3))
    assert not grade_game_answer(game, ("place", 0))
    # Two systems both pass twice -> staleness accrues -> retirement.
    for _ in range(2):
        bank.record_grades("base", {"g1": True})
        bank.record_grades("adapter", {"g1": True})
        bank.sweep_saturation()
    assert bank.items["g1"].status == "retired"
    assert bank.items["g1"].discrimination() == 0.0
