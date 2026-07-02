"""The examiner as the system's second learner — gated harder than the first.

The student loop trains adapters; this module is the coupled loop that trains the
EXAM. Design rules, each load-bearing:

- Root oracles end the regress. Every item's label is grounded in an exact
  verifier (graph/MIS refinement + stress pools, playground simulator), never in
  a model's opinion.
- Grade by checker, not answer key. Repair items carry a CHECKER SPEC (original
  expression + stress horizon); any strict refinement that survives passes. There
  is no answer string to leak into training.
- Semantic leakage checks. An item is quarantined if its original expression
  appears in any training buffer the student will see — row identity, not string
  fuzz.
- Discrimination is learned from use. Each grading run records per-system
  pass/fail on every item; an item's discrimination is the spread between the
  best and worst systems it has seen. Items that stop separating systems gain
  staleness and retire.
- Downward-only flow. private_promotion_exam -> public_regression ->
  trainable_experience. Retired exams become student fuel; nothing exposed to
  training ever climbs back into the private bank.
"""

from __future__ import annotations

import argparse
import json
import random
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from bcv.domains import COLORING, MIS, Domain


DEFAULT_ROOT = Path(".bcv_runs/examiner")


@dataclass
class ExamItem:
    item_id: str
    domain: str  # coloring | mis | playground
    kind: str  # repair | game_move
    payload: dict  # repair: original, evidence; game: rules, state, player
    oracle: str
    source: str
    horizon: str
    lineage: list[str]
    status: str = "candidate"
    novelty: float = 0.0
    leakage_risk: float = 0.0
    staleness: int = 0
    graded: dict = field(default_factory=dict)  # system -> {"pass": int, "fail": int}

    def discrimination(self) -> float:
        rates = []
        for stats in self.graded.values():
            total = stats["pass"] + stats["fail"]
            if total:
                rates.append(stats["pass"] / total)
        return round(max(rates) - min(rates), 3) if len(rates) >= 2 else 0.0


class ExaminerBank:
    def __init__(self, root: str | Path = DEFAULT_ROOT) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.items: dict[str, ExamItem] = {}
        for bucket in ("candidate", "promoted", "retired", "quarantined"):
            path = self._bucket_path(bucket)
            if path.exists():
                for line in path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        raw = json.loads(line)
                        self.items[raw["item_id"]] = ExamItem(**raw)

    def _bucket_path(self, bucket: str) -> Path:
        names = {
            "candidate": "items.jsonl",
            "promoted": "private_promotion_exam.jsonl",
            "retired": "public_regression.jsonl",
            "quarantined": "quarantined.jsonl",
        }
        return self.root / names[bucket]

    def save(self) -> None:
        buckets: dict[str, list[ExamItem]] = {"candidate": [], "promoted": [], "retired": [], "quarantined": []}
        for item in self.items.values():
            buckets[item.status].append(item)
        for bucket, rows in buckets.items():
            self._bucket_path(bucket).write_text(
                "\n".join(json.dumps(asdict(item), sort_keys=True) for item in rows) + ("\n" if rows else ""),
                encoding="utf-8",
            )

    # ------------------------------------------------------------- lifecycle

    def add(self, item: ExamItem) -> None:
        self.items[item.item_id] = item

    def promote(self, item_id: str) -> bool:
        item = self.items[item_id]
        if item.status != "candidate" or item.leakage_risk > 0:
            return False
        item.status = "promoted"
        return True

    def retire(self, item_id: str) -> None:
        """Downward only: promoted -> retired (public). Never back up."""
        item = self.items[item_id]
        if item.status == "promoted":
            item.status = "retired"

    def promoted_items(self, domain: str | None = None) -> list[ExamItem]:
        return [
            item
            for item in self.items.values()
            if item.status == "promoted" and (domain is None or item.domain == domain)
        ]

    def trainable_rows(self) -> list[dict]:
        """Retired exams become student fuel (the downward pipeline's mouth)."""
        return [asdict(item) for item in self.items.values() if item.status == "retired"]

    def record_grades(self, system: str, results: dict[str, bool]) -> None:
        for item_id, passed in results.items():
            item = self.items[item_id]
            stats = item.graded.setdefault(system, {"pass": 0, "fail": 0})
            stats["pass" if passed else "fail"] += 1

    def sweep_saturation(self, min_systems: int = 2) -> list[str]:
        """Items every graded system passes stop discriminating: stale, then retired."""
        retired = []
        for item in self.promoted_items():
            if len(item.graded) < min_systems:
                continue
            all_pass = all(
                stats["pass"] > 0 and stats["fail"] == 0 for stats in item.graded.values()
            )
            if all_pass:
                item.staleness += 1
                if item.staleness >= 2:
                    self.retire(item.item_id)
                    retired.append(item.item_id)
            else:
                item.staleness = 0
        return retired


# ----------------------------------------------------------------- minting


def training_originals(buffer_paths: list[str | Path]) -> set[str]:
    """Row-identity leakage set: original expressions the student trains on."""
    originals: set[str] = set()
    for path in buffer_paths:
        path = Path(path)
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                payload = json.loads(row["messages"][1]["content"])
                originals.add(str(payload.get("original_expression", "")))
            except (KeyError, json.JSONDecodeError, IndexError, TypeError):
                continue
    originals.discard("")
    return originals


def mint_repair_items(
    domain: Domain,
    buffer_paths: list[str | Path],
    max_items: int = 10,
    max_n: int = 6,
    stress_ns: tuple[int, ...] = (7, 8),
    seed: int = 0,
) -> list[ExamItem]:
    """Repair exam items whose checker is the domain verifier + stress pool.
    Leakage: any item whose original the student trains on is quarantined."""
    from bcv.graph_repair_data import _candidate_expressions
    from bcv.novelty import NoveltyJudge
    from bcv.refinery import _mine_stress_repair, _observe_all, _stress_pool, _verify

    observations = _observe_all(domain, max_n, Path(".bcv_runs/examiner_tmp"))
    pool = _stress_pool(domain, stress_ns, 40, seed, Path(".bcv_runs/examiner_tmp/nolib.jsonl"))
    leak_set = training_originals(buffer_paths)
    judge = NoveltyJudge(max_n=max_n)
    items: list[ExamItem] = []
    for expression in _candidate_expressions():
        if len(items) >= max_items:
            break
        verdict = _verify(expression, observations)
        if verdict is None:
            continue
        matched, failure = verdict
        if failure is None:
            continue  # not rejected -> nothing to repair
        repair = _mine_stress_repair(expression, observations, pool)
        if repair is None:
            continue  # no certified repair exists: not a fair exam item
        novelty = judge.judge(repair, base_expression=expression)
        item = ExamItem(
            item_id=f"{domain.name}_{uuid.uuid4().hex[:8]}",
            domain=domain.name,
            kind="repair",
            payload={
                "original_expression": expression,
                "counterexample": failure.graph.graph_id(),
                "claim": domain.claim,
            },
            oracle="exact_verifier+stress_pool",
            source="frontier_mint",
            horizon=f"n<={max_n} certified, stress n in {stress_ns}",
            lineage=[f"mined_repair:{repair}"],
            novelty=1.0 if novelty.semantically_novel else 0.0,
            leakage_risk=1.0 if expression in leak_set else 0.0,
        )
        if item.leakage_risk > 0:
            item.status = "quarantined"
        items.append(item)
    return items


def mint_game_items(max_items: int = 8, seed: int = 0) -> list[ExamItem]:
    """Game-move items: acceptable set = win-now moves, else depth-2-safe moves.
    Certified by the simulator (a root oracle)."""
    from bcv.playground import (
        GameRules, _wins_now, apply_move, legal_moves, policy_depth2, winner, EMPTY,
    )

    certified_path = Path(".bcv_runs/playground/certified_games.jsonl")
    if not certified_path.exists():
        return []
    rng = random.Random(seed)
    games = [
        GameRules(**json.loads(line))
        for line in certified_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    items: list[ExamItem] = []
    for rules in games:
        state = tuple([EMPTY] * rules.board_size)
        for turn in range(1, rules.max_turns + 1):
            player = (turn - 1) % 2
            move = policy_depth2(state, player, rules, rng)
            if move is None:
                break
            if turn >= 3 and len(items) < max_items:
                moves = legal_moves(state, player, rules)
                winning = [m for m in moves if _wins_now(state, player, m, 0, rules)]
                if winning:
                    acceptable = winning
                else:
                    acceptable = []
                    for candidate in moves:
                        after = apply_move(state, player, candidate)
                        replies = legal_moves(after, 1 - player, rules)
                        if not any(_wins_now(after, 1 - player, r, 0, rules) for r in replies):
                            acceptable.append(candidate)
                    acceptable = acceptable or moves
                if 0 < len(acceptable) < len(moves):
                    items.append(
                        ExamItem(
                            item_id=f"game_{uuid.uuid4().hex[:8]}",
                            domain="playground",
                            kind="game_move",
                            payload={
                                "rules": asdict(rules),
                                "state": list(state),
                                "player": player,
                                "acceptable": [list(m) for m in acceptable],
                            },
                            oracle="simulator",
                            source="certified_game",
                            horizon="depth2",
                            lineage=[rules.name],
                        )
                    )
            state = apply_move(state, player, move)
            if winner(state, player, turn, rules) is not None:
                break
    return items[:max_items]


# ----------------------------------------------------------------- grading


def grade_repair_answer(item: ExamItem, answer_expression: str | None, max_n: int = 6, pool=None) -> bool:
    """Checker-spec grading: ANY verified strict refinement that survives the
    stress pool passes. No answer key exists."""
    if not answer_expression:
        return False
    from bcv.graph_agent import compile_feature_expression
    from bcv.refinery import _observe_all
    from bcv.domains import DOMAINS

    domain = DOMAINS[item.domain]
    observations = _observe_all(domain, max_n, Path(".bcv_runs/examiner_tmp"))
    try:
        predicate = compile_feature_expression(answer_expression)
        original = compile_feature_expression(item.payload["original_expression"])
    except Exception:
        return False
    matches = [obs for obs in observations if predicate(obs)]
    if not matches or any(not obs.greedy_is_optimal for obs in matches):
        return False
    if not all(original(obs) for obs in matches):
        return False
    if pool is not None and any(predicate(obs) and not obs.greedy_is_optimal for obs in pool):
        return False
    return True


def grade_game_answer(item: ExamItem, move) -> bool:
    if move is None:
        return False
    if isinstance(move, (str, int)):
        move = [move]
    return list(move) in item.payload["acceptable"]


def repair_item_prompt(item: ExamItem) -> str:
    return (
        f"Repair a rejected conjecture. Claim: {item.payload['claim']}. The predicate "
        f"`{item.payload['original_expression']}` has counterexamples (e.g. "
        f"{item.payload['counterexample']}). Reply only JSON: "
        '{"repair_expression": "<strictly narrower DSL predicate>"}. '
        "Features: n, m, density, max_degree, min_degree, is_connected, is_complete, "
        "is_forest, is_tree, is_bipartite, is_triangle_free, max_degree_le_2, "
        "has_universal_vertex, has_isolated_vertex, is_regular, num_components, "
        "clique_number, girth."
    )


def game_item_prompt(item: ExamItem) -> str:
    payload = item.payload
    if "fen" in payload:
        return (
            "You are a strong chess player. Given the FEN, reply only JSON with the best move "
            f'in UCI notation: {{"move": "<uci>"}}. FEN: {payload["fen"]}'
        )
    if "moves" in payload:
        return (
            "You are a strong Go player on a 9x9 board (komi 7). The game so far as GTP vertices, "
            f'alternating from Black: {json.dumps(payload["moves"])}. Reply only JSON with the best '
            f'next move for {payload.get("to_move", "?")} as a GTP vertex: {{"move": "<vertex>"}}'
        )
    rules = payload.get("rules", {})
    game_name = rules.get("game", "")
    if game_name in ("connect4", "gomoku", "othello6", "hex7"):
        return (
            f"You are playing {game_name} (board row-major, 0 empty, 1 player0, 2 player1). "
            f"Rules family: {game_name}. Reply only JSON with the best move index: "
            f'{{"move": <int>}}. Position: {json.dumps({"board": payload["state"], "player": payload["player"]}, sort_keys=True)}'
        )
    return (
        "You are playing a certified board game on a row of cells (0 empty, 1 stick, 2 stone). "
        f"Rules: {json.dumps(rules, sort_keys=True)}. Player 0 plays sticks, player 1 plays stones. "
        'Reply only JSON: {"move": [...]}. Moves: ["place", cell] or ["shift", from, to] '
        f'or ["capture", cell]. Position: {json.dumps({"state": payload["state"], "player": payload["player"]}, sort_keys=True)}'
    )


def grade_system(
    bank: ExaminerBank,
    system_name: str,
    client,
    max_items: int | None = None,
    stress_ns: tuple[int, ...] = (7, 8),
    seed: int = 0,
) -> dict:
    """Grade a model/adapter on the promoted bank; record item-level results."""
    from bcv.refinery import _stress_pool
    from bcv.domains import DOMAINS
    from bcv.transformers_client import extract_json

    pools: dict[str, list] = {}
    results: dict[str, bool] = {}
    items = bank.promoted_items()
    if max_items:
        items = items[:max_items]
    for item in items:
        if item.kind == "repair":
            if item.domain not in pools:
                pools[item.domain] = _stress_pool(
                    DOMAINS[item.domain], stress_ns, 40, seed,
                    Path(".bcv_runs/examiner_tmp/nolib.jsonl"),
                )
            raw = client.generate_text(repair_item_prompt(item), temperature=0.0)
            parsed = extract_json(raw)
            answer = parsed.get("repair_expression") if isinstance(parsed, dict) else None
            results[item.item_id] = grade_repair_answer(item, answer, pool=pools[item.domain])
        else:
            raw = client.generate_text(game_item_prompt(item), temperature=0.0)
            parsed = extract_json(raw)
            move = parsed.get("move") if isinstance(parsed, dict) else None
            results[item.item_id] = grade_game_answer(item, move)
    bank.record_grades(system_name, results)
    bank.save()
    passed = sum(results.values())
    return {"system": system_name, "items": len(results), "passed": passed, "results": results}


def mint_initial_bank(
    buffer_paths: list[str],
    root: str | Path = DEFAULT_ROOT,
    per_domain: int = 8,
    seed: int = 0,
) -> dict:
    bank = ExaminerBank(root)
    minted = []
    minted += mint_repair_items(COLORING, buffer_paths, max_items=per_domain, seed=seed)
    minted += mint_repair_items(MIS, buffer_paths, max_items=per_domain, seed=seed)
    minted += mint_game_items(max_items=per_domain, seed=seed)
    promoted = quarantined = 0
    for item in minted:
        bank.add(item)
        if item.status == "quarantined":
            quarantined += 1
        elif bank.promote(item.item_id):
            promoted += 1
    bank.save()
    return {
        "minted": len(minted),
        "promoted": promoted,
        "quarantined_for_leakage": quarantined,
        "novel_items": sum(1 for item in minted if item.novelty > 0),
        "by_domain": {
            domain: sum(1 for item in minted if item.domain == domain)
            for domain in ("coloring", "mis", "playground")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Mint and manage the promoted exam bank.")
    parser.add_argument("--mint", action="store_true")
    parser.add_argument("--per-domain", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--buffers",
        nargs="*",
        default=[
            ".bcv_runs/cook/buffer_round_3.jsonl",
            ".bcv_runs/graph_repair_hard_rich/hard_train.jsonl",
            ".bcv_runs/graph_repair_hard_stress/hard_train.jsonl",
            ".bcv_runs/graph_repair_hard/hard_train.jsonl",
        ],
    )
    args = parser.parse_args()
    if args.mint:
        print(json.dumps(mint_initial_bank(args.buffers, per_domain=args.per_domain, seed=args.seed), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
