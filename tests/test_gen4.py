from __future__ import annotations

import json
from pathlib import Path

import pytest

from bcv.examiner import ExamItem, ExaminerBank
from bcv.gen4 import Gen4DataError, prepare_engine_data, trajectory_bucket
from scripts.run_gen4_engine_student import _pregrade_identity


def _trajectory(prefix: str, bucket: str) -> str:
    for value in range(10_000):
        identifier = f"{prefix}_{value:032x}"
        if trajectory_bucket(identifier, 25) == bucket:
            return identifier
    raise AssertionError("no trajectory id found")


def _bank(tmp_path) -> ExaminerBank:
    bank = ExaminerBank(tmp_path / "bank")
    bank.add(
        ExamItem(
            item_id="chess_private",
            domain="chess",
            kind="game_move",
            payload={"fen": "private-fen", "acceptable": [["a1a2"]]},
            oracle="stockfish",
            source="test",
            horizon="d12",
            lineage=[],
        )
    )
    bank.promote("chess_private")
    bank.save()
    return bank


def test_prepare_engine_data_enforces_position_and_trajectory_disjointness(tmp_path) -> None:
    train_id = _trajectory("chess", "train")
    holdout_id = _trajectory("go", "holdout")
    chess = [
        {
            "game": "chess",
            "trajectory_id": train_id,
            "trajectory_ply": 8,
            "fen": "private-fen",
            "oracle_move": "a1a2",
            "shallow_move": "a1b1",
        },
        {
            "game": "chess",
            "trajectory_id": train_id,
            "trajectory_ply": 10,
            "fen": "fresh-fen",
            "oracle_move": "b1b2",
            "shallow_move": "b1c1",
        },
    ]
    go = [
        {
            "game": "go9",
            "trajectory_id": holdout_id,
            "trajectory_ply": 6,
            "moves": ["D4", "F6"],
            "to_move": "black",
            "oracle_move": "E5",
            "shallow_move": "C3",
        }
    ]
    split = prepare_engine_data(chess, go, _bank(tmp_path), holdout_percent=25)
    assert len(split.train_examples) == 1
    assert len(split.holdout_rows) == 1
    assert "fresh-fen" in split.train_examples[0]["messages"][1]["content"]
    assert split.manifest["deduplication"]["promoted_bank_collisions_removed"] == 1
    assert split.manifest["invariants"]["train_holdout_trajectory_overlap"] == 0
    assert split.manifest["invariants"]["train_promoted_exact_position_overlap"] == 0
    public_manifest = json.dumps(split.manifest)
    assert "fresh-fen" not in public_manifest
    assert "D4" not in public_manifest


def test_prepare_engine_data_rejects_legacy_rows_without_trajectory_ids(tmp_path) -> None:
    row = {
        "game": "chess",
        "fen": "fresh-fen",
        "oracle_move": "b1b2",
        "shallow_move": "b1c1",
    }
    with pytest.raises(Gen4DataError, match="trajectory_id"):
        prepare_engine_data([row], [], _bank(tmp_path))


def test_committed_gen4_evaluation_receipt_is_blocked_and_sanitized() -> None:
    path = Path(__file__).resolve().parents[1] / "results" / "gen4_engine_student_evaluation_receipt.json"
    receipt = json.loads(path.read_text(encoding="utf-8"))

    assert receipt["artifact_type"] == "evaluation_receipt"
    assert receipt["promotion_claim"] is False
    assert receipt["training"]["accepted"] is True
    assert receipt["training"]["steps_completed"] == 1313
    assert receipt["training"]["skipped_steps"] == 0
    assert receipt["training_data"]["train"]["rows"] == 1313
    assert receipt["training_data"]["invariants"]["train_holdout_trajectory_overlap"] == 0
    assert receipt["training_data"]["invariants"]["train_promoted_exact_position_overlap"] == 0
    assert receipt["repeated_grading"]["baseline"]["scores"] == [7, 7, 7]
    assert receipt["repeated_grading"]["candidate"]["scores"] == [8, 8, 8]
    assert receipt["gate_strict"]["gains"] == 3
    assert receipt["gate_strict"]["regressions"] == 2
    assert receipt["gate_strict"]["exact_mcnemar_two_sided_p"] == 1.0
    assert receipt["gate_strict"]["verdict"] == "BLOCK"
    assert receipt["gate_reliability_aware"]["verdict"] == "BLOCK"

    serialized = json.dumps(receipt).lower()
    for private_marker in ("/home/", "c:\\", '"item_id"', '"fen"', '"moves"'):
        assert private_marker not in serialized


def test_pregrade_identity_survives_grade_history_and_rejects_item_set_drift(tmp_path) -> None:
    bank = _bank(tmp_path)
    identity = _pregrade_identity(tmp_path / "run", bank)
    before = identity["before_grading_sha256"]

    bank.items["chess_private"].graded["candidate"] = {"pass": 1, "fail": 0}
    bank.save()
    assert _pregrade_identity(tmp_path / "run", ExaminerBank(bank.root))["before_grading_sha256"] == before

    bank.add(
        ExamItem(
            item_id="chess_private_2",
            domain="chess",
            kind="game_move",
            payload={"fen": "another-private-fen", "acceptable": [["b1b2"]]},
            oracle="stockfish",
            source="test",
            horizon="d12",
            lineage=[],
        )
    )
    bank.promote("chess_private_2")
    bank.save()
    with pytest.raises(SystemExit, match="item count changed"):
        _pregrade_identity(tmp_path / "run", ExaminerBank(bank.root))
