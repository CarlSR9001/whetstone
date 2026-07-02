"""The arcade: real games, exact simulators, a rollout ladder, a skill ledger.

The playground proved that game-invention can bootstrap verifiers; the arcade
replaces its 1D toys with the classics that train real cognitive functions:

  connect4   6x7, gravity — threats and forced lines (lookahead under constraint)
  gomoku     7x7 four-in-row, free placement — open pattern threats
  othello6   6x6 with real flipping and pass rules — long-horizon evaluation,
             where piece-count greed famously LOSES (a salience trap as a game)
  hex7       7x7 connection game — topological reasoning, provably drawless

One uniform strength ladder, no hand-tuned evaluations to argue about: Monte-Carlo
move choice with R rollouts per move. R=0 is random; more rollouts is strictly
more compute and (verifiably, via the meta-verifier) more strength. The ladder is
the oracle for milling supervision, the frontier detector for exam items, and the
measuring stick for the SKILL LEDGER — an append-only growth curve of every system
generation's winrate per game per rung. "Getting smarter" becomes a curve in a
file, not a feeling.
"""

from __future__ import annotations

import argparse
import json
import random
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class ArcadeState:
    game: str
    board: tuple[int, ...]  # 0 empty, 1 player0, 2 player1 (row-major)
    width: int
    height: int
    to_move: int
    passes: int = 0


class ArcadeGame:
    name = "base"
    width = 0
    height = 0

    def initial(self) -> ArcadeState:
        return ArcadeState(self.name, tuple([0] * (self.width * self.height)), self.width, self.height, 0)

    def legal_moves(self, state: ArcadeState) -> list[int]:
        raise NotImplementedError

    def apply(self, state: ArcadeState, move: int) -> ArcadeState:
        raise NotImplementedError

    def winner(self, state: ArcadeState) -> int | None:
        """0/1 winner, -1 draw, None ongoing."""
        raise NotImplementedError


def _lines(width: int, height: int, k: int):
    for row in range(height):
        for col in range(width):
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                end_r, end_c = row + dr * (k - 1), col + dc * (k - 1)
                if 0 <= end_r < height and 0 <= end_c < width:
                    yield tuple((row + dr * i) * width + (col + dc * i) for i in range(k))


class Connect4(ArcadeGame):
    name, width, height, k = "connect4", 7, 6, 4

    def __init__(self):
        self.line_cache = list(_lines(self.width, self.height, self.k))

    def legal_moves(self, state):
        return [col for col in range(self.width) if state.board[col] == 0]

    def apply(self, state, move):
        board = list(state.board)
        for row in range(self.height - 1, -1, -1):
            index = row * self.width + move
            if board[index] == 0:
                board[index] = state.to_move + 1
                break
        return ArcadeState(state.game, tuple(board), state.width, state.height, 1 - state.to_move)

    def winner(self, state):
        for line in self.line_cache:
            values = {state.board[i] for i in line}
            if len(values) == 1 and 0 not in values:
                return values.pop() - 1
        if all(value != 0 for value in state.board):
            return -1
        return None


class Gomoku(ArcadeGame):
    name, width, height, k = "gomoku", 7, 7, 4

    def __init__(self):
        self.line_cache = list(_lines(self.width, self.height, self.k))

    def legal_moves(self, state):
        return [i for i, value in enumerate(state.board) if value == 0]

    def apply(self, state, move):
        board = list(state.board)
        board[move] = state.to_move + 1
        return ArcadeState(state.game, tuple(board), state.width, state.height, 1 - state.to_move)

    def winner(self, state):
        for line in self.line_cache:
            values = {state.board[i] for i in line}
            if len(values) == 1 and 0 not in values:
                return values.pop() - 1
        if all(value != 0 for value in state.board):
            return -1
        return None


class Othello6(ArcadeGame):
    name, width, height = "othello6", 6, 6
    PASS = -1

    def initial(self):
        board = [0] * 36
        board[2 * 6 + 2], board[3 * 6 + 3] = 2, 2
        board[2 * 6 + 3], board[3 * 6 + 2] = 1, 1
        return ArcadeState(self.name, tuple(board), 6, 6, 0)

    def _flips(self, state, move):
        mine, theirs = state.to_move + 1, 2 - state.to_move
        row, col = divmod(move, 6)
        flips = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == dc == 0:
                    continue
                run = []
                r, c = row + dr, col + dc
                while 0 <= r < 6 and 0 <= c < 6 and state.board[r * 6 + c] == theirs:
                    run.append(r * 6 + c)
                    r, c = r + dr, c + dc
                if run and 0 <= r < 6 and 0 <= c < 6 and state.board[r * 6 + c] == mine:
                    flips.extend(run)
        return flips

    def legal_moves(self, state):
        moves = [i for i, v in enumerate(state.board) if v == 0 and self._flips(state, i)]
        return moves or [self.PASS]

    def apply(self, state, move):
        if move == self.PASS:
            return ArcadeState(state.game, state.board, 6, 6, 1 - state.to_move, state.passes + 1)
        board = list(state.board)
        board[move] = state.to_move + 1
        for i in self._flips(state, move):
            board[i] = state.to_move + 1
        return ArcadeState(state.game, tuple(board), 6, 6, 1 - state.to_move, 0)

    def winner(self, state):
        if state.passes < 2 and any(value == 0 for value in state.board):
            return None
        p0 = sum(1 for value in state.board if value == 1)
        p1 = sum(1 for value in state.board if value == 2)
        if p0 == p1:
            return -1
        return 0 if p0 > p1 else 1


class Hex7(ArcadeGame):
    """7x7 Hex: player 0 connects top-bottom, player 1 left-right. No draws."""

    name, width, height = "hex7", 7, 7

    def legal_moves(self, state):
        return [i for i, value in enumerate(state.board) if value == 0]

    def apply(self, state, move):
        board = list(state.board)
        board[move] = state.to_move + 1
        return ArcadeState(state.game, tuple(board), 7, 7, 1 - state.to_move)

    def _connected(self, state, player):
        size = 7
        mine = player + 1
        if player == 0:
            frontier = [i for i in range(size) if state.board[i] == mine]
            goal = set(range(size * (size - 1), size * size))
        else:
            frontier = [r * size for r in range(size) if state.board[r * size] == mine]
            goal = {r * size + size - 1 for r in range(size)}
        seen = set(frontier)
        while frontier:
            cell = frontier.pop()
            if cell in goal:
                return True
            row, col = divmod(cell, size)
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)):
                r, c = row + dr, col + dc
                index = r * size + c
                if 0 <= r < size and 0 <= c < size and index not in seen and state.board[index] == mine:
                    seen.add(index)
                    frontier.append(index)
        return False

    def winner(self, state):
        for player in (0, 1):
            if self._connected(state, player):
                return player
        if all(value != 0 for value in state.board):
            return -1  # unreachable in Hex, kept for interface safety
        return None


GAMES: dict[str, ArcadeGame] = {g.name: g for g in (Connect4(), Gomoku(), Othello6(), Hex7())}
MAX_PLIES = 140


# ------------------------------------------------------------------ policies


def rollout(game: ArcadeGame, state: ArcadeState, rng) -> int:
    for _ in range(MAX_PLIES):
        result = game.winner(state)
        if result is not None:
            return result
        moves = game.legal_moves(state)
        state = game.apply(state, rng.choice(moves))
    return -1


def mc_policy(rollouts: int):
    def policy(game: ArcadeGame, state: ArcadeState, rng) -> int:
        moves = game.legal_moves(state)
        if rollouts == 0 or len(moves) == 1:
            return rng.choice(moves)
        player = state.to_move
        best_move, best_value = moves[0], -1.0
        for move in moves:
            after = game.apply(state, move)
            wins = 0.0
            for _ in range(rollouts):
                result = rollout(game, after, rng)
                if result == player:
                    wins += 1
                elif result == -1:
                    wins += 0.5
            value = wins / rollouts
            if value > best_value:
                best_move, best_value = move, value
        return best_move

    policy.name = f"mc{rollouts}" if rollouts else "random"
    return policy


def spam_policy(game, state, rng):
    return game.legal_moves(state)[0]


spam_policy.name = "spam"


def play(game: ArcadeGame, policy0, policy1, rng) -> int:
    state = game.initial()
    for _ in range(MAX_PLIES):
        result = game.winner(state)
        if result is not None:
            return result
        policy = policy0 if state.to_move == 0 else policy1
        state = game.apply(state, policy(game, state, rng))
    return -1


def winrate(game, policy_a, policy_b, episodes, rng) -> float:
    wins = 0.0
    for episode in range(episodes):
        if episode % 2 == 0:
            result = play(game, policy_a, policy_b, rng)
            wins += 1.0 if result == 0 else (0.5 if result == -1 else 0.0)
        else:
            result = play(game, policy_b, policy_a, rng)
            wins += 1.0 if result == 1 else (0.5 if result == -1 else 0.0)
    return wins / episodes


# ------------------------------------------------------------- certification


def certify_game(name: str, episodes: int = 24, seed: int = 0) -> dict:
    game = GAMES[name]
    rng = random.Random(seed)
    random_p, mc4, mc16 = mc_policy(0), mc_policy(4), mc_policy(16)
    skill = winrate(game, mc4, random_p, episodes, rng)
    depth = winrate(game, mc16, mc4, episodes, rng)
    spam = winrate(game, spam_policy, mc4, episodes, rng)
    balance = winrate(game, random_p, random_p, episodes, rng)
    report = {
        "game": name,
        "skill_mc4_vs_random": round(skill, 3),
        "depth_mc16_vs_mc4": round(depth, 3),
        "spam_vs_mc4": round(spam, 3),
        "first_seat_random": round(balance, 3),
        "certified": skill >= 0.6 and depth >= 0.5 and spam <= 0.5,
        "trains": (["tactics"] if skill >= 0.6 else []) + (["search_depth"] if depth >= 0.55 else []),
    }
    return report


# ------------------------------------------------------- mill + exam + ledger


def mill_positions(name: str, count: int, oracle_rollouts: int = 64, seed: int = 0) -> list[dict]:
    """Supervision: positions labeled with the strongest ladder rung's move."""
    game = GAMES[name]
    rng = random.Random(seed)
    oracle = mc_policy(oracle_rollouts)
    weak = mc_policy(4)
    rows: list[dict] = []
    while len(rows) < count:
        state = game.initial()
        for ply in range(MAX_PLIES):
            if game.winner(state) is not None:
                break
            if ply >= 2 and len(rows) < count and rng.random() < 0.4:
                choice = oracle(game, state, rng)
                rows.append(
                    {
                        "game": name,
                        "board": list(state.board),
                        "to_move": state.to_move,
                        "oracle_move": choice,
                        "weak_move": weak(game, state, rng),
                    }
                )
                state = game.apply(state, choice)
            else:
                state = game.apply(state, mc_policy(8)(game, state, rng))
    return rows


def mint_arcade_exam_items(per_game: int = 3, seed: int = 0) -> int:
    """Frontier positions (oracle disagrees with the weak rung) into the exam bank."""
    from bcv.examiner import ExamItem, ExaminerBank

    bank = ExaminerBank()
    added = 0
    for name in GAMES:
        rows = mill_positions(name, per_game * 4, seed=seed)
        frontier = [row for row in rows if row["oracle_move"] != row["weak_move"]][:per_game]
        for row in frontier:
            item = ExamItem(
                item_id=f"arcade_{uuid.uuid4().hex[:8]}",
                domain="playground",
                kind="game_move",
                payload={
                    "rules": {"game": row["game"]},
                    "state": row["board"],
                    "player": row["to_move"],
                    "acceptable": [[row["oracle_move"]]],
                },
                oracle="mc_consensus",
                source="arcade_frontier",
                horizon=f"mc64_vs_mc4",
                lineage=[row["game"]],
            )
            bank.add(item)
            if bank.promote(item.item_id):
                added += 1
    bank.save()
    return added


def update_skill_ledger(system: str, policy, games: list[str], episodes: int = 16, seed: int = 0,
                        root: str | Path = ".bcv_runs/arcade") -> list[dict]:
    """Append the system's winrate vs every ladder rung: the growth curve."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    entries = []
    for name in games:
        game = GAMES[name]
        for rung in (mc_policy(0), mc_policy(4), mc_policy(16)):
            entry = {
                "timestamp": datetime.now().isoformat(),
                "system": system,
                "game": name,
                "opponent": rung.name,
                "winrate": round(winrate(game, policy, rung, episodes, rng), 3),
                "episodes": episodes,
            }
            entries.append(entry)
    with (root / "skill_ledger.jsonl").open("a", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description="Certify real games; mill; mint; ledger.")
    parser.add_argument("--certify", action="store_true")
    parser.add_argument("--mill", type=int, default=0, help="positions per game")
    parser.add_argument("--mint-exams", action="store_true")
    parser.add_argument("--ledger-baselines", action="store_true")
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    root = Path(".bcv_runs/arcade")
    root.mkdir(parents=True, exist_ok=True)
    output: dict = {}
    if args.certify:
        output["certification"] = [certify_game(name, args.episodes, args.seed) for name in GAMES]
        (root / "certification.json").write_text(json.dumps(output["certification"], indent=2), encoding="utf-8")
    if args.mill:
        rows = []
        for name in GAMES:
            rows.extend(mill_positions(name, args.mill, seed=args.seed))
        (root / "experience.jsonl").write_text(
            "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8"
        )
        output["milled"] = len(rows)
    if args.mint_exams:
        output["exam_items_promoted"] = mint_arcade_exam_items(seed=args.seed)
    if args.ledger_baselines:
        entries = []
        for rung_size, rung_name in ((4, "mc4"),):
            entries += update_skill_ledger(rung_name, mc_policy(rung_size), list(GAMES), seed=args.seed)
        output["ledger_entries"] = len(entries)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
