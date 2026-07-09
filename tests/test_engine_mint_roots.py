from __future__ import annotations

from bcv.baduk import mint_go_exam_items
from bcv.examiner import ExaminerBank
from bcv.grandmaster import mint_chess_exam_items


def test_chess_mint_can_target_an_isolated_bank_without_engine_runtime(tmp_path):
    rows = [
        {"fen": "startpos", "oracle_move": "e2e4", "shallow_move": "d2d4"},
        {"fen": "other", "oracle_move": "g1f3", "shallow_move": "b1c3"},
    ]
    root = tmp_path / "chess_bank"
    assert mint_chess_exam_items(rows, per_bank=2, bank_root=root) == 2
    assert len(ExaminerBank(root).promoted_items("chess")) == 2


def test_go_mint_can_target_an_isolated_bank_without_engine_runtime(tmp_path):
    rows = [
        {"moves": ["D4"], "to_move": "white", "oracle_move": "Q16", "shallow_move": "D16"},
        {"moves": ["C3"], "to_move": "white", "oracle_move": "R17", "shallow_move": "C17"},
    ]
    root = tmp_path / "go_bank"
    assert mint_go_exam_items(rows, per_bank=2, bank_root=root) == 2
    assert len(ExaminerBank(root).promoted_items("go")) == 2
