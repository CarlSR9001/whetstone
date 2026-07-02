"""Play as verifier bootstrapping: sticks and stones, literally.

The refinery attacked conjectures under a FIXED verifier. This module searches the
space of verifiers themselves, the way children do: propose a game (a rule-set that
makes activity gradeable), then let a META-VERIFIER decide whether the game is a
good instrument. A candidate game earns certification only if it:

  terminates    every episode reaches a determinate outcome (decidability),
  is balanced   neither seat wins on coin-flip terms under random play,
  rewards skill 1-ply beats random decisively (outcomes respond to strategy),
  rewards depth 2-ply beats 1-ply (the game trains lookahead — the Reactor
                knob readout: WHICH cognition a game trains is measurable),
  resists hacks a degenerate spam policy does not dominate (rule-lawyering test).

"That doesn't count!" -> counterexample; "new rule!" -> repair: the arguing IS
adversarial verifier refinement, and the meta-verifier mechanizes it.

Certified games are verifiers: each one mills unlimited exactly-graded experience
(state -> deeper-search move), which is the missing supply line for continual
learning. And skills consolidated ACROSS games of a family are tested for transfer
to held-out new games — learning that survives leaving the game it came from.

Substrate: a row of cells; player 0 plays sticks, player 1 plays stones. Moves are
place / shift / capture-adjacent, toggled per game. Win conditions come from a
small grammar (k-in-a-row, hold both ends, most pieces, opponent immobilized).
Everything is exact and CPU-cheap by construction.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path


EMPTY, STICK, STONE = 0, 1, 2


@dataclass(frozen=True)
class GameRules:
    name: str
    board_size: int
    max_turns: int
    win_kind: str  # k_in_row | both_ends | most_pieces | immobilize
    k: int = 3
    allow_shift: bool = False
    allow_capture: bool = False


@dataclass(frozen=True)
class GameReport:
    rules: GameRules
    decisive_rate: float
    first_seat_winrate: float
    skill_winrate: float  # greedy-1ply vs random
    depth_winrate: float  # 2-ply vs 1-ply
    spam_winrate: float  # degenerate policy vs 1-ply
    certified: bool
    trains: tuple[str, ...]


# ------------------------------------------------------------------ mechanics


def legal_moves(state: tuple[int, ...], player: int, rules: GameRules) -> list[tuple]:
    mine = STICK if player == 0 else STONE
    theirs = STONE if player == 0 else STICK
    moves: list[tuple] = []
    for cell, value in enumerate(state):
        if value == EMPTY:
            moves.append(("place", cell))
    if rules.allow_shift:
        for cell, value in enumerate(state):
            if value != mine:
                continue
            for neighbor in (cell - 1, cell + 1):
                if 0 <= neighbor < len(state) and state[neighbor] == EMPTY:
                    moves.append(("shift", cell, neighbor))
    if rules.allow_capture:
        for cell, value in enumerate(state):
            if value != mine:
                continue
            for neighbor in (cell - 1, cell + 1):
                if 0 <= neighbor < len(state) and state[neighbor] == theirs:
                    moves.append(("capture", neighbor))
    return moves


def apply_move(state: tuple[int, ...], player: int, move: tuple) -> tuple[int, ...]:
    cells = list(state)
    mine = STICK if player == 0 else STONE
    if move[0] == "place":
        cells[move[1]] = mine
    elif move[0] == "shift":
        cells[move[1]] = EMPTY
        cells[move[2]] = mine
    elif move[0] == "capture":
        cells[move[1]] = EMPTY
    return tuple(cells)


def winner(state: tuple[int, ...], player_just_moved: int, turn: int, rules: GameRules) -> int | None:
    """Returns 0/1 for a winner, -1 for draw, None if the game continues."""
    mine = STICK if player_just_moved == 0 else STONE
    if rules.win_kind == "k_in_row":
        run = 0
        for value in state:
            run = run + 1 if value == mine else 0
            if run >= rules.k:
                return player_just_moved
    elif rules.win_kind == "both_ends":
        if state[0] == mine and state[-1] == mine:
            return player_just_moved
    elif rules.win_kind == "immobilize":
        if not legal_moves(state, 1 - player_just_moved, rules):
            return player_just_moved
    if turn >= rules.max_turns or all(value != EMPTY for value in state):
        if rules.win_kind == "most_pieces":
            sticks = sum(1 for value in state if value == STICK)
            stones = sum(1 for value in state if value == STONE)
            if sticks != stones:
                return 0 if sticks > stones else 1
        return -1
    return None


# ------------------------------------------------------------------- policies


def policy_random(state, player, rules, rng):
    moves = legal_moves(state, player, rules)
    return rng.choice(moves) if moves else None


def _wins_now(state, player, move, turn, rules) -> bool:
    return winner(apply_move(state, player, move), player, turn, rules) == player


def policy_greedy(state, player, rules, rng):
    """1-ply: take an immediate win if one exists, else random."""
    moves = legal_moves(state, player, rules)
    if not moves:
        return None
    for move in moves:
        if _wins_now(state, player, move, 0, rules):
            return move
    return rng.choice(moves)


def policy_depth2(state, player, rules, rng):
    """2-ply: win now if possible; else avoid moves that hand the opponent a win."""
    moves = legal_moves(state, player, rules)
    if not moves:
        return None
    for move in moves:
        if _wins_now(state, player, move, 0, rules):
            return move
    safe = []
    for move in moves:
        following = apply_move(state, player, move)
        opponent_moves = legal_moves(following, 1 - player, rules)
        if not any(_wins_now(following, 1 - player, reply, 0, rules) for reply in opponent_moves):
            safe.append(move)
    return rng.choice(safe or moves)


def policy_spam(state, player, rules, rng):
    """Degenerate rule-lawyer: always the first legal move (stall/spam pattern)."""
    moves = legal_moves(state, player, rules)
    return moves[0] if moves else None


def play_episode(rules: GameRules, policy0, policy1, rng) -> int:
    state = tuple([EMPTY] * rules.board_size)
    for turn in range(1, rules.max_turns + 1):
        player = (turn - 1) % 2
        policy = policy0 if player == 0 else policy1
        move = policy(state, player, rules, rng)
        if move is None:
            return 1 - player if rules.win_kind == "immobilize" else -1
        state = apply_move(state, player, move)
        result = winner(state, player, turn, rules)
        if result is not None:
            return result
    return -1


def _winrate(rules, policy_a, policy_b, episodes, rng) -> tuple[float, float]:
    """Seat-swapped winrate of A vs B and decisive rate."""
    wins = 0.0
    decisive = 0
    for episode in range(episodes):
        if episode % 2 == 0:
            result = play_episode(rules, policy_a, policy_b, rng)
            if result == 0:
                wins += 1
        else:
            result = play_episode(rules, policy_b, policy_a, rng)
            if result == 1:
                wins += 1
        if result in (0, 1):
            decisive += 1
    return wins / episodes, decisive / episodes


# -------------------------------------------------------------- meta-verifier


def evaluate_game(rules: GameRules, episodes: int = 120, seed: int = 0) -> GameReport:
    rng = random.Random(seed)
    first_seat_wins = 0
    decisive = 0
    for _ in range(episodes):
        result = play_episode(rules, policy_random, policy_random, rng)
        if result in (0, 1):
            decisive += 1
            if result == 0:
                first_seat_wins += 1
    decisive_rate = decisive / episodes
    first_seat = first_seat_wins / max(1, decisive)
    skill, _ = _winrate(rules, policy_greedy, policy_random, episodes, rng)
    depth, _ = _winrate(rules, policy_depth2, policy_greedy, episodes, rng)
    spam, _ = _winrate(rules, policy_spam, policy_greedy, episodes, rng)

    trains: list[str] = []
    if skill >= 0.6:
        trains.append("immediate_tactics")
    if depth >= 0.55:
        trains.append("lookahead")
    certified = (
        decisive_rate >= 0.25
        and 0.2 <= first_seat <= 0.8
        and skill >= 0.6
        and depth >= 0.55
        and spam <= 0.5
    )
    return GameReport(
        rules=rules,
        decisive_rate=round(decisive_rate, 3),
        first_seat_winrate=round(first_seat, 3),
        skill_winrate=round(skill, 3),
        depth_winrate=round(depth, 3),
        spam_winrate=round(spam, 3),
        certified=certified,
        trains=tuple(trains),
    )


def game_grammar() -> list[GameRules]:
    """The candidate space: what a kid can invent with a row of dirt and pebbles."""
    games: list[GameRules] = []
    for size in (7, 9, 11):
        for win_kind in ("k_in_row", "both_ends", "most_pieces", "immobilize"):
            for k in ((2, 3, 4) if win_kind == "k_in_row" else (3,)):
                for shift in (False, True):
                    for capture in (False, True):
                        games.append(
                            GameRules(
                                name=f"{win_kind}_k{k}_n{size}{'_sh' if shift else ''}{'_cap' if capture else ''}",
                                board_size=size,
                                max_turns=size * 3,
                                win_kind=win_kind,
                                k=k,
                                allow_shift=shift,
                                allow_capture=capture,
                            )
                        )
    return games


# ------------------------------------------------------ experience + transfer


def mill_experience(rules: GameRules, episodes: int, seed: int) -> list[dict]:
    """Verified supervision from a certified game: states + the 2-ply move.
    This is the Gap-2 supply line: unlimited, exactly graded, free."""
    rng = random.Random(seed)
    rows: list[dict] = []
    for _ in range(episodes):
        state = tuple([EMPTY] * rules.board_size)
        for turn in range(1, rules.max_turns + 1):
            player = (turn - 1) % 2
            move = policy_depth2(state, player, rules, rng)
            if move is None:
                break
            rows.append(
                {"game": rules.name, "state": list(state), "player": player, "move": list(move)}
            )
            state = apply_move(state, player, move)
            if winner(state, player, turn, rules) is not None:
                break
    return rows


def _own_max_run(state: tuple[int, ...], player: int) -> int:
    mine = STICK if player == 0 else STONE
    best = run = 0
    for value in state:
        run = run + 1 if value == mine else 0
        best = max(best, run)
    return best


def consolidate_skill(experience: list[dict]) -> float:
    """Cross-game consolidation: how often did the deeper player's move maximize
    own-run-length? If the fraction is high, 'extend your line' is a portable skill."""
    agree = total = 0
    for row in experience:
        state = tuple(row["state"])
        player = row["player"]
        rules = GameRules("probe", len(state), 99, "k_in_row")
        moves = legal_moves(state, player, rules)
        if len(moves) < 2:
            continue
        best_value = max(_own_max_run(apply_move(state, player, m), player) for m in moves)
        chosen_value = _own_max_run(apply_move(state, player, tuple(row["move"])), player)
        total += 1
        if chosen_value == best_value:
            agree += 1
    return agree / max(1, total)


def policy_skill_prior(state, player, rules, rng):
    """Greedy + the consolidated 'extend your line' prior."""
    moves = legal_moves(state, player, rules)
    if not moves:
        return None
    for move in moves:
        if _wins_now(state, player, move, 0, rules):
            return move
    best = max(_own_max_run(apply_move(state, player, m), player) for m in moves)
    preferred = [m for m in moves if _own_max_run(apply_move(state, player, m), player) == best]
    return rng.choice(preferred)


def transfer_test(train_games: list[GameRules], heldout: GameRules, seed: int = 0) -> dict:
    """Does the skill consolidated from games A,B transfer to unseen game C?"""
    rng = random.Random(seed)
    experience: list[dict] = []
    for rules in train_games:
        experience.extend(mill_experience(rules, episodes=40, seed=seed))
    skill_agreement = consolidate_skill(experience)
    transfer_winrate, _ = _winrate(heldout, policy_skill_prior, policy_greedy, 200, rng)
    return {
        "experience_rows": len(experience),
        "skill_agreement": round(skill_agreement, 3),
        "heldout_game": heldout.name,
        "transfer_winrate_vs_greedy": round(transfer_winrate, 3),
        "transferred": transfer_winrate > 0.55,
    }


# ------------------------------------------------------------------ top level


def run_playground(root: str | Path = ".bcv_runs/playground", seed: int = 0) -> dict:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    reports = [evaluate_game(rules, seed=seed) for rules in game_grammar()]
    certified = [report for report in reports if report.certified]
    (root / "game_reports.json").write_text(
        json.dumps([asdict(report) for report in reports], indent=2, sort_keys=True), encoding="utf-8"
    )
    (root / "certified_games.jsonl").write_text(
        "\n".join(json.dumps(asdict(report.rules), sort_keys=True) for report in certified) + "\n",
        encoding="utf-8",
    )
    # Gap 2: mill experience from every certified verifier.
    experience: list[dict] = []
    for report in certified:
        experience.extend(mill_experience(report.rules, episodes=20, seed=seed))
    (root / "experience.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in experience) + "\n", encoding="utf-8"
    )
    # Gap 5: consolidate across k-in-row games, test transfer on a held-out one.
    k_games = [report.rules for report in certified if report.rules.win_kind == "k_in_row"]
    transfer = None
    if len(k_games) >= 3:
        transfer = transfer_test(k_games[:-1], k_games[-1], seed=seed)
        (root / "transfer.json").write_text(json.dumps(transfer, indent=2), encoding="utf-8")
    summary = {
        "candidate_games": len(reports),
        "certified_games": len(certified),
        "certified_training_lookahead": sum(1 for r in certified if "lookahead" in r.trains),
        "rejected_examples": [
            {"name": r.rules.name, "why": _rejection_reason(r)} for r in reports if not r.certified
        ][:5],
        "experience_rows_milled": len(experience),
        "transfer": transfer,
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def _rejection_reason(report: GameReport) -> str:
    if report.decisive_rate < 0.25:
        return f"indecisive (decisive_rate={report.decisive_rate})"
    if not 0.2 <= report.first_seat_winrate <= 0.8:
        return f"unbalanced (first_seat={report.first_seat_winrate})"
    if report.skill_winrate < 0.6:
        return f"luck-dominated (skill={report.skill_winrate})"
    if report.depth_winrate < 0.55:
        return f"shallow (depth={report.depth_winrate})"
    return f"hackable (spam={report.spam_winrate})"


def main() -> None:
    parser = argparse.ArgumentParser(description="Invent games; certify the ones that are good verifiers.")
    parser.add_argument("--root", default=".bcv_runs/playground")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(run_playground(args.root, args.seed), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
