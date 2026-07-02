from __future__ import annotations

import random

from bcv.arcade import GAMES, Connect4, Hex7, Othello6, mc_policy, play, rollout, winrate


def test_connect4_gravity_and_win():
    game = Connect4()
    state = game.initial()
    for _ in range(3):
        state = game.apply(state, 0)  # p0 stacks col 0
        assert game.winner(state) is None
        state = game.apply(state, 1)  # p1 stacks col 1
    state = game.apply(state, 0)
    assert game.winner(state) == 0
    # Gravity: pieces sit at the bottom rows of column 0.
    bottom = state.board[(game.height - 1) * game.width + 0]
    assert bottom == 1


def test_othello_flipping_and_pass():
    game = Othello6()
    state = game.initial()
    moves = game.legal_moves(state)
    assert len(moves) == 4  # standard opening mobility
    before = sum(1 for v in state.board if v == 1)
    state = game.apply(state, moves[0])
    after = sum(1 for v in state.board if v == 1)
    assert after >= before + 2  # placed one, flipped at least one


def test_hex_has_no_draws_and_detects_connection():
    game = Hex7()
    state = game.initial()
    # Fill column 0 for player 0 top-to-bottom: connected.
    board = list(state.board)
    for row in range(7):
        board[row * 7] = 1
    from bcv.arcade import ArcadeState

    connected = ArcadeState("hex7", tuple(board), 7, 7, 1)
    assert game.winner(connected) == 0
    # Random self-play never draws.
    rng = random.Random(0)
    for _ in range(5):
        assert play(game, mc_policy(0), mc_policy(0), rng) in (0, 1)


def test_rollout_terminates_all_games():
    rng = random.Random(1)
    for game in GAMES.values():
        assert rollout(game, game.initial(), rng) in (0, 1, -1)


def test_ladder_strength_ordering():
    # More rollouts must not be weaker — checked on the fastest game.
    game = GAMES["connect4"]
    rng = random.Random(2)
    skill = winrate(game, mc_policy(4), mc_policy(0), 16, rng)
    assert skill >= 0.6
