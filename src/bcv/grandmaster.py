"""Chess with a real oracle: Stockfish as a root oracle in the arcade pipeline.

The rule that kept M.U.G.E.N and Unity out is the rule that lets Stockfish in:
labels must come from something that cannot (practically) be wrong. Stockfish at
depth 12+ against depth-1 self-play positions is that thing for chess at our
scale. The ladder needs no hand-tuning: random-legal < depth 1 < depth 4 < depth 8
is strictly increasing strength by construction.

Same pipeline stages as every other domain:
  fingerprint  the meta-verifier gates run on chess itself (skill, depth
               gradient, spam resistance) — uniform methodology, real game,
  mill         positions from shallow self-play labeled with the deep engine's
               best move (student fuel),
  exam         frontier positions where deep and shallow engines disagree,
               promoted into the examiner bank (leakage-clean: new domain),
  ledger       skill-ledger entries per rung, same append-only growth file.
"""

from __future__ import annotations

import argparse
import json
import random
import uuid
from pathlib import Path

import chess
import chess.engine


STOCKFISH = Path("tools/stockfish/stockfish-windows-x86-64-avx2.exe")
MAX_PLIES = 160


def open_engine(path: str | Path = STOCKFISH):
    engine = chess.engine.SimpleEngine.popen_uci(str(path))
    engine.configure({"Threads": 1, "Hash": 64})
    return engine


# ------------------------------------------------------------------ policies


def move_random(board: chess.Board, engine, rng) -> chess.Move:
    return rng.choice(list(board.legal_moves))


def move_spam(board: chess.Board, engine, rng) -> chess.Move:
    return sorted(board.legal_moves, key=lambda m: m.uci())[0]


def move_depth(depth: int):
    def policy(board, engine, rng):
        return engine.play(board, chess.engine.Limit(depth=depth)).move

    policy.name = f"sf_d{depth}"
    return policy


move_random.name = "random"
move_spam.name = "spam"


def play_game(policy0, policy1, engine, rng, opening_plies: int = 4) -> float:
    """Returns score for side A (policy0 as white). Random opening for variety."""
    board = chess.Board()
    for _ in range(opening_plies):
        board.push(rng.choice(list(board.legal_moves)))
        if board.is_game_over():
            break
    while not board.is_game_over() and board.ply() < MAX_PLIES:
        policy = policy0 if board.turn == chess.WHITE else policy1
        board.push(policy(board, engine, rng))
    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        return 0.5
    return 1.0 if outcome.winner == chess.WHITE else 0.0


def winrate(policy_a, policy_b, engine, episodes, rng) -> float:
    total = 0.0
    for episode in range(episodes):
        if episode % 2 == 0:
            total += play_game(policy_a, policy_b, engine, rng)
        else:
            total += 1.0 - play_game(policy_b, policy_a, engine, rng)
    return total / episodes


def fingerprint(episodes: int = 12, seed: int = 0, engine_path: str | Path = STOCKFISH) -> dict:
    rng = random.Random(seed)
    engine = open_engine(engine_path)
    try:
        skill = winrate(move_depth(1), move_random, engine, episodes, rng)
        depth = winrate(move_depth(4), move_depth(1), engine, episodes, rng)
        spam = winrate(move_spam, move_depth(1), engine, episodes, rng)
    finally:
        engine.quit()
    return {
        "game": "chess",
        "skill_d1_vs_random": round(skill, 3),
        "depth_d4_vs_d1": round(depth, 3),
        "spam_vs_d1": round(spam, 3),
        "certified": skill >= 0.6 and depth >= 0.55 and spam <= 0.5,
        "oracle": "stockfish",
    }


# ------------------------------------------------------- mill + exam + ledger


def mill_positions(
    count: int,
    oracle_depth: int = 12,
    shallow_depth: int = 2,
    seed: int = 0,
    engine_path: str | Path = STOCKFISH,
) -> list[dict]:
    rng = random.Random(seed)
    engine = open_engine(engine_path)
    rows: list[dict] = []
    try:
        while len(rows) < count:
            board = chess.Board()
            for _ in range(rng.randrange(6, 14)):
                if board.is_game_over():
                    break
                board.push(rng.choice(list(board.legal_moves)))
            while not board.is_game_over() and board.ply() < 60 and len(rows) < count:
                if rng.random() < 0.35:
                    oracle_move = engine.play(board, chess.engine.Limit(depth=oracle_depth)).move
                    shallow_move = engine.play(board, chess.engine.Limit(depth=shallow_depth)).move
                    rows.append(
                        {
                            "game": "chess",
                            "fen": board.fen(),
                            "oracle_move": oracle_move.uci(),
                            "shallow_move": shallow_move.uci(),
                        }
                    )
                    board.push(oracle_move)
                else:
                    board.push(engine.play(board, chess.engine.Limit(depth=shallow_depth)).move)
    finally:
        engine.quit()
    return rows


def mint_chess_exam_items(
    rows: list[dict], per_bank: int = 6, bank_root: str | Path | None = None
) -> int:
    from bcv.examiner import ExamItem, ExaminerBank

    bank = ExaminerBank(bank_root) if bank_root else ExaminerBank()
    added = 0
    frontier = [row for row in rows if row["oracle_move"] != row["shallow_move"]]
    for row in frontier[:per_bank]:
        item = ExamItem(
            item_id=f"chess_{uuid.uuid4().hex[:8]}",
            domain="chess",
            kind="game_move",
            payload={
                "rules": {"game": "chess"},
                "fen": row["fen"],
                "acceptable": [[row["oracle_move"]]],
            },
            oracle="stockfish_d12",
            source="engine_frontier",
            horizon="d12_vs_d2",
            lineage=["stockfish"],
        )
        bank.add(item)
        if bank.promote(item.item_id):
            added += 1
    bank.save()
    return added


def ledger_entries(seed: int = 0, episodes: int = 8, engine_path: str | Path = STOCKFISH) -> int:
    from datetime import datetime

    rng = random.Random(seed)
    engine = open_engine(engine_path)
    root = Path(".bcv_runs/arcade")
    root.mkdir(parents=True, exist_ok=True)
    entries = []
    try:
        for system, policy in (("sf_d1", move_depth(1)),):
            for rung in (move_random, move_depth(4)):
                entries.append(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "system": system,
                        "game": "chess",
                        "opponent": rung.name,
                        "winrate": round(winrate(policy, rung, engine, episodes, rng), 3),
                        "episodes": episodes,
                    }
                )
    finally:
        engine.quit()
    with (root / "skill_ledger.jsonl").open("a", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
    return len(entries)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stockfish-anchored chess pipeline.")
    parser.add_argument("--fingerprint", action="store_true")
    parser.add_argument("--mill", type=int, default=0)
    parser.add_argument("--mint-exams", action="store_true")
    parser.add_argument("--ledger", action="store_true")
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--engine-path", default=str(STOCKFISH))
    parser.add_argument("--root", default=".bcv_runs/chess", help="experience output directory")
    parser.add_argument("--bank-root", help="optional isolated bank to receive minted exam items")
    parser.add_argument("--per-bank", type=int, default=6)
    args = parser.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    output: dict = {}
    if args.fingerprint:
        output["fingerprint"] = fingerprint(args.episodes, args.seed, args.engine_path)
    rows: list[dict] = []
    if args.mill:
        rows = mill_positions(args.mill, seed=args.seed, engine_path=args.engine_path)
        (root / "experience.jsonl").write_text(
            "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8"
        )
        output["milled"] = len(rows)
        output["frontier_rate"] = round(
            sum(1 for r in rows if r["oracle_move"] != r["shallow_move"]) / max(1, len(rows)), 3
        )
    if args.mint_exams and rows:
        output["exam_items_promoted"] = mint_chess_exam_items(rows, args.per_bank, args.bank_root)
    if args.ledger:
        output["ledger_entries"] = ledger_entries(args.seed, engine_path=args.engine_path)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
