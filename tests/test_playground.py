from __future__ import annotations

import random

from bcv.emulator import _split_at_first_control
from bcv.memstore import MemoryStore
from bcv.playground import (
    GameRules,
    evaluate_game,
    game_grammar,
    legal_moves,
    mill_experience,
    play_episode,
    policy_depth2,
    policy_greedy,
    policy_random,
    run_playground,
    transfer_test,
    winner,
)


def test_mechanics_k_in_row():
    rules = GameRules("t", board_size=7, max_turns=21, win_kind="k_in_row", k=3)
    state = (1, 1, 0, 0, 0, 0, 0)
    assert winner((1, 1, 1, 0, 0, 0, 0), 0, 3, rules) == 0
    assert winner(state, 0, 2, rules) is None
    assert ("place", 2) in legal_moves(state, 0, rules)


def test_meta_verifier_rejects_trivial_and_certifies_real_games():
    # k=2 on a small board is decided nearly instantly: should fail certification.
    trivial = evaluate_game(GameRules("trivial", 7, 21, "k_in_row", k=2), episodes=80, seed=1)
    assert not trivial.certified
    # Classic 3-in-a-row on 9 cells: skill and depth should both matter.
    classic = evaluate_game(GameRules("classic", 9, 27, "k_in_row", k=3), episodes=120, seed=1)
    assert classic.skill_winrate > 0.5
    # The grammar must certify SOME games and reject others (the gates bite).
    reports = [evaluate_game(rules, episodes=60, seed=2) for rules in game_grammar()[:24]]
    assert any(report.certified for report in reports)
    assert any(not report.certified for report in reports)


def test_experience_mill_and_transfer():
    train = [
        GameRules("a", 9, 27, "k_in_row", k=3),
        GameRules("b", 11, 33, "k_in_row", k=3),
    ]
    heldout = GameRules("c", 9, 27, "k_in_row", k=4)
    rows = mill_experience(train[0], episodes=5, seed=3)
    assert rows and all("state" in row and "move" in row for row in rows)
    result = transfer_test(train, heldout, seed=3)
    assert result["experience_rows"] > 50
    assert 0.0 <= result["skill_agreement"] <= 1.0


def test_goal_ledger():
    store = MemoryStore()
    store.set_goal("find a game where depth-3 beats depth-2", ("depth", "lookahead"), step=1)
    goals = store.active_goals()
    assert len(goals) == 1
    assert store.goal_entities() == ("depth", "lookahead")


def test_structured_control_packets():
    segment, control, argument = _split_at_first_control(
        'thinking...\n{"control": "CHECK", "arg": "(is_tree) and (max_degree >= 3)"}\ntrailing'
    )
    assert control == "CHECK"
    assert argument == "(is_tree) and (max_degree >= 3)"
    _, control, argument = _split_at_first_control(
        '{"control": "load", "arg": "base", "note": "wrong branch"}'
    )
    assert control == "LOAD"
    assert argument == "base :: wrong branch"
