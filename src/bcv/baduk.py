"""Go via KataGo: the visit ladder as oracle, same pipeline stages as chess.

Two engine processes over the same b6c96 net, differing only in maxVisits: the
shallow engine (visits=2) drives self-play; at sampled positions the oracle engine
(visits=48) is asked for its move (genmove + undo, then re-synced with the move
actually played). Frontier positions — where oracle and shallow disagree — become
exam items. KataGo itself needs no certification; it IS the root oracle for Go.
Fingerprint winrate games are deferred (pass/score adjudication adds complexity
without adding label quality).
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import uuid
from pathlib import Path


KATAGO_DIR = Path("tools/katago")
KATAGO = KATAGO_DIR / "katago.exe"
NET = KATAGO_DIR / "net-b6c96.txt.gz"
CONFIG = KATAGO_DIR / "default_gtp.cfg"


class GTPEngine:
    def __init__(self, max_visits: int, katago_dir: str | Path = KATAGO_DIR):
        katago_dir = Path(katago_dir)
        self.process = subprocess.Popen(
            [
                str(katago_dir / "katago.exe"), "gtp", "-model", str(katago_dir / "net-b6c96.txt.gz"),
                "-config", str(katago_dir / "default_gtp.cfg"),
                "-override-config", f"maxVisits={max_visits}",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            cwd=str(katago_dir.parent.parent),
        )

    def send(self, command: str) -> str:
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()
        lines = []
        while True:
            line = self.process.stdout.readline()
            if line.strip() == "" and lines:
                break
            if line.strip():
                lines.append(line.strip())
        response = lines[-1] if lines else ""
        return response.lstrip("= ").strip()

    def close(self):
        try:
            self.send("quit")
        except Exception:
            pass
        self.process.terminate()


def mill_go_positions(count: int = 12, size: int = 9, shallow_visits: int = 2,
                      oracle_visits: int = 48, seed: int = 0,
                      katago_dir: str | Path = KATAGO_DIR) -> list[dict]:
    rng = random.Random(seed)
    shallow = GTPEngine(shallow_visits, katago_dir)
    oracle = GTPEngine(oracle_visits, katago_dir)
    rows: list[dict] = []
    try:
        for engine in (shallow, oracle):
            engine.send(f"boardsize {size}")
            engine.send("komi 7")
        while len(rows) < count:
            for engine in (shallow, oracle):
                engine.send("clear_board")
            moves: list[str] = []
            color = "b"
            for ply in range(30):
                if len(rows) >= count:
                    break
                sample_here = ply >= 2 and rng.random() < 0.5
                oracle_move = None
                if sample_here:
                    oracle_move = oracle.send(f"genmove {color}")
                    oracle.send("undo")
                shallow_move = shallow.send(f"genmove {color}")
                if shallow_move.lower() in ("pass", "resign"):
                    break
                oracle.send(f"play {color} {shallow_move}")
                if sample_here and oracle_move and oracle_move.lower() not in ("pass", "resign"):
                    rows.append(
                        {
                            "game": "go9",
                            "moves": list(moves),
                            "to_move": "black" if color == "b" else "white",
                            "oracle_move": oracle_move,
                            "shallow_move": shallow_move,
                        }
                    )
                moves.append(shallow_move)
                color = "w" if color == "b" else "b"
    finally:
        shallow.close()
        oracle.close()
    return rows


def mint_go_exam_items(
    rows: list[dict], per_bank: int = 4, bank_root: str | Path | None = None
) -> int:
    from bcv.examiner import ExamItem, ExaminerBank

    bank = ExaminerBank(bank_root) if bank_root else ExaminerBank()
    added = 0
    frontier = [row for row in rows if row["oracle_move"] != row["shallow_move"]]
    for row in frontier[:per_bank]:
        item = ExamItem(
            item_id=f"go_{uuid.uuid4().hex[:8]}",
            domain="go",
            kind="game_move",
            payload={
                "rules": {"game": "go9"},
                "moves": row["moves"],
                "to_move": row["to_move"],
                "acceptable": [[row["oracle_move"]]],
            },
            oracle=f"katago_v48",
            source="engine_frontier",
            horizon="v48_vs_v2",
            lineage=["katago_b6c96"],
        )
        bank.add(item)
        if bank.promote(item.item_id):
            added += 1
    bank.save()
    return added


def main() -> None:
    parser = argparse.ArgumentParser(description="KataGo-anchored Go mill.")
    parser.add_argument("--mill", type=int, default=12)
    parser.add_argument("--mint-exams", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--katago-dir", default=str(KATAGO_DIR))
    parser.add_argument("--root", default=".bcv_runs/go", help="experience output directory")
    parser.add_argument("--bank-root", help="optional isolated bank to receive minted exam items")
    parser.add_argument("--per-bank", type=int, default=4)
    args = parser.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    rows = mill_go_positions(args.mill, seed=args.seed, katago_dir=args.katago_dir)
    (root / "experience.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8"
    )
    output = {
        "milled": len(rows),
        "frontier_rate": round(
            sum(1 for r in rows if r["oracle_move"] != r["shallow_move"]) / max(1, len(rows)), 3
        ),
    }
    if args.mint_exams:
        output["exam_items_promoted"] = mint_go_exam_items(rows, args.per_bank, args.bank_root)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
