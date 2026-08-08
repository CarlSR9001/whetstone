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


def random_opening_vertices(rng: random.Random, size: int, plies: int) -> list[str]:
    """Distinct random vertices for a non-replayable opening (GTP letters skip I)."""
    letters = "ABCDEFGHJKLMNOPQRST".replace("I", "")[:size]
    vertices = [f"{letter}{number}" for letter in letters for number in range(1, size + 1)]
    rng.shuffle(vertices)
    return vertices[:plies]


def mill_go_positions(count: int = 12, size: int = 9, shallow_visits: int = 2,
                      oracle_visits: int = 48, seed: int = 0,
                      katago_dir: str | Path = KATAGO_DIR,
                      opening_plies: tuple[int, int] = (4, 8),
                      opening_rng: random.Random | None = None) -> list[dict]:
    """Mill shallow-vs-oracle frontier positions.

    opening_plies=(lo, hi) prepends a RANDOM opening of lo..hi stones before
    the engine ladder starts. Low-visit ladders from the empty board are
    near-deterministic, so any published transcript makes future identical
    mills reconstructible; a random opening makes every trajectory fresh. The
    opening rng defaults to SystemRandom — deliberately NOT the seeded mill
    rng, so trajectories are unreproducible even with the seed known."""
    rng = random.Random(seed)
    opening_rng = opening_rng or random.SystemRandom()
    shallow = GTPEngine(shallow_visits, katago_dir)
    oracle = GTPEngine(oracle_visits, katago_dir)
    rows: list[dict] = []
    try:
        for engine in (shallow, oracle):
            engine.send(f"boardsize {size}")
            engine.send("komi 7")
        while len(rows) < count:
            trajectory_id = f"go_{opening_rng.getrandbits(128):032x}"
            for engine in (shallow, oracle):
                engine.send("clear_board")
            moves: list[str] = []
            color = "b"
            if opening_plies[1] > 0:
                for vertex in random_opening_vertices(
                    opening_rng, size, opening_rng.randint(*opening_plies)
                ):
                    responses = [engine.send(f"play {color} {vertex}") for engine in (shallow, oracle)]
                    if any(response.startswith("?") for response in responses):
                        continue  # illegal for this position; skip the vertex
                    moves.append(vertex)
                    color = "w" if color == "b" else "b"
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
                            "trajectory_id": trajectory_id,
                            "trajectory_ply": len(moves),
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
    rows: list[dict],
    per_bank: int = 4,
    bank_root: str | Path | None = None,
    check_published: bool = True,
) -> int:
    """Mint frontier rows as exam items. By default every candidate position is
    checked against the KNOWN published GTP transcripts at mint time — a
    collision is quarantined on the spot, not discovered by a later audit."""
    from bcv.examiner import ExamItem, ExaminerBank
    from bcv.exposure_audit import is_known_published_prefix, known_published_prefixes

    prefixes = known_published_prefixes() if check_published else set()
    bank = ExaminerBank(bank_root) if bank_root else ExaminerBank()
    added = 0
    frontier = [row for row in rows if row["oracle_move"] != row["shallow_move"]]
    for row in frontier[:per_bank]:
        published = check_published and (
            tuple(row["moves"]) in prefixes or is_known_published_prefix(row["moves"])
        )
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
            leakage_risk=1.0 if published else 0.0,
            leakage_match="published_gtp_log" if published else "",
        )
        if published:
            item.status = "quarantined"
        bank.add(item)
        if not published and bank.promote(item.item_id):
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
    parser.add_argument(
        "--opening-plies", type=int, nargs=2, default=(4, 8), metavar=("MIN", "MAX"),
        help="unseeded random opening length; defaults to 4-8 to prevent trajectory replay",
    )
    args = parser.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    rows = mill_go_positions(
        args.mill,
        seed=args.seed,
        katago_dir=args.katago_dir,
        opening_plies=tuple(args.opening_plies),
    )
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
