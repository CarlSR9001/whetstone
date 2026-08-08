from __future__ import annotations

import pytest

from bcv.grandmaster import STOCKFISH

needs_engine = pytest.mark.skipif(not STOCKFISH.exists(), reason="stockfish binary not installed")


@needs_engine
def test_engine_ladder_and_play():
    import random

    import chess

    from bcv.grandmaster import move_depth, move_random, open_engine, play_game

    engine = open_engine()
    try:
        board = chess.Board()
        move = move_depth(4)(board, engine, random.Random(0))
        assert move in board.legal_moves
        score = play_game(move_depth(1), move_random, engine, random.Random(1))
        assert 0.0 <= score <= 1.0
    finally:
        engine.quit()


@needs_engine
def test_mill_produces_engine_labeled_rows():
    from bcv.grandmaster import mill_positions

    rows = mill_positions(3, oracle_depth=6, shallow_depth=1, seed=2)
    assert len(rows) == 3
    for row in rows:
        assert row["fen"] and len(row["oracle_move"]) >= 4
        assert row["trajectory_id"].startswith("chess_")
        assert isinstance(row["trajectory_ply"], int)


def test_chess_exam_item_grading_shape():
    from bcv.examiner import ExamItem, grade_game_answer

    item = ExamItem(
        item_id="c1",
        domain="chess",
        kind="game_move",
        payload={"rules": {"game": "chess"}, "fen": "startpos", "acceptable": [["e2e4"]]},
        oracle="stockfish_d12",
        source="test",
        horizon="d12_vs_d2",
        lineage=[],
        status="promoted",
    )
    assert grade_game_answer(item, ["e2e4"])
    assert not grade_game_answer(item, ["d2d4"])
