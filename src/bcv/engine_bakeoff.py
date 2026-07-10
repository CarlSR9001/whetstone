"""Grade bracketed engine systems on the isolated engine bank.

The engine bank (17 chess + 47 Go frontier items) was milled where a shallow
policy disagrees with a deep oracle — capacity, but zero graded evidence. This
module supplies the missing measurement: two COMPOSITE systems, each a chess
engine plus a Go engine at matched strength tiers, graded on the identical
64-item cohort, then pushed through the strict promotion gate.

Why composites: the gate requires identical cohorts (no cherry-picking which
items a system deigns to answer), and a real agent under exam doesn't get to
skip domains. engine_shallow reuses the SAME strength tier that milled the
frontier, so it should score near zero BY CONSTRUCTION — grading it is the
sanity check that the mill did its job. engine_mid (deeper search, more
visits) is the real measurement: whether the bank has the resolution to
certify a genuine capability difference between two honest systems.

Everything runs inside the local trust boundary — no burn, no exposure. The
published receipt carries counts, p-values, and the resolution statement,
never FENs, move histories, or item ids.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from bcv.examiner import ExaminerBank, grade_game_answer

ENGINE_BANK_ROOT = Path(".bcv_runs/engine_volume/bank")


@dataclass
class EngineTier:
    name: str
    chess_depth: int
    go_visits: int


SHALLOW = EngineTier("engine_shallow_d2_v2", chess_depth=2, go_visits=2)
MID = EngineTier("engine_mid_d8_v16", chess_depth=8, go_visits=16)


# ------------------------------------------------------------------ answering


def chess_answers(items: list, depth: int) -> dict[str, str]:
    """One persistent Stockfish process answers every chess item at fixed depth."""
    import chess

    from bcv.grandmaster import open_engine

    answers: dict[str, str] = {}
    engine = open_engine()
    try:
        for item in items:
            board = chess.Board(item.payload["fen"])
            result = engine.play(board, chess.engine.Limit(depth=depth))
            answers[item.item_id] = result.move.uci() if result.move else ""
    finally:
        engine.quit()
    return answers


def go_answers(items: list, visits: int) -> dict[str, str]:
    """One persistent KataGo process answers every Go item at fixed visits."""
    from bcv.baduk import GTPEngine

    answers: dict[str, str] = {}
    engine = GTPEngine(max_visits=visits)
    try:
        for item in items:
            engine.send("boardsize 9")
            engine.send("komi 7")
            engine.send("clear_board")
            color = "b"
            for vertex in item.payload["moves"]:
                engine.send(f"play {color} {vertex}")
                color = "w" if color == "b" else "b"
            to_move = item.payload.get("to_move", "black")[0]
            answers[item.item_id] = engine.send(f"genmove {to_move}").upper()
    finally:
        engine.close()
    return answers


def tier_answers(items: list, tier: EngineTier) -> dict[str, str]:
    chess_items = [item for item in items if item.domain == "chess"]
    go_items = [item for item in items if item.domain == "go"]
    answers = chess_answers(chess_items, tier.chess_depth)
    answers.update(go_answers(go_items, tier.go_visits))
    return answers


# -------------------------------------------------------------------- bakeoff


def grade_tier(bank: ExaminerBank, tier: EngineTier, items: list) -> dict[str, bool]:
    started = time.time()
    answers = tier_answers(items, tier)
    results = {item.item_id: grade_game_answer(item, answers.get(item.item_id, "")) for item in items}
    bank.record_grades(
        tier.name,
        results,
        run_manifest={
            "kind": "engine_bakeoff",
            "chess_engine": "stockfish tools/stockfish (one persistent process, fixed depth)",
            "chess_depth": tier.chess_depth,
            "go_engine": "katago b6c96 gtp (one persistent process, fixed maxVisits)",
            "go_visits": tier.go_visits,
            "items": len(items),
            "elapsed_seconds": round(time.time() - started, 1),
            "trust_boundary": "local only; no burn",
        },
    )
    bank.save()
    return results


def _gate_summary(report: dict) -> dict:
    evidence = report["paired_evidence"]
    summary = {
        "verdict": report["verdict"],
        "reasons": report["reasons"],
        "gains": evidence["gains"],
        "regressions": evidence["regressions"],
        "ties": evidence["ties"],
        "exact_mcnemar_two_sided_p": evidence["exact_mcnemar_two_sided_p"],
        "resolution": evidence["resolution"],
    }
    reliability = evidence.get("regression_reliability")
    if reliability:
        summary["regression_classifications"] = [
            {"domain": row["domain"], "classification": row["classification"],
             "flip_rate": (row.get("reliability") or {}).get("flip_rate")}
            for row in reliability
        ]
    return summary


def run_engine_bakeoff(
    root: str | Path = ENGINE_BANK_ROOT,
    baseline: EngineTier = SHALLOW,
    candidate: EngineTier = MID,
    receipt_path: str | Path | None = "results/engine_bank_bakeoff_receipt.json",
    repeats: int = 3,
) -> dict:
    """Grade both tiers `repeats` times, then gate under BOTH regression
    policies. Repeated grading is what turns a single-flip regression from
    "unknown" into measured flakiness — the reliability-aware verdict is only
    as honest as the repeated-grade evidence behind it."""
    from bcv.gate import GatePolicy, build_gate_report, write_gate_report

    bank = ExaminerBank(root)
    items = bank.promoted_items()
    if not items:
        raise SystemExit(f"no promoted items in {root}")

    run_totals: dict[str, list[str]] = {baseline.name: [], candidate.name: []}
    baseline_results: dict[str, bool] = {}
    candidate_results: dict[str, bool] = {}
    for _ in range(max(1, repeats)):
        baseline_results = grade_tier(bank, baseline, items)
        candidate_results = grade_tier(bank, candidate, items)
        run_totals[baseline.name].append(f"{sum(baseline_results.values())}/{len(items)}")
        run_totals[candidate.name].append(f"{sum(candidate_results.values())}/{len(items)}")

    reports = {}
    for policy_name in ("strict", "reliability_aware"):
        report = build_gate_report(
            bank,
            baseline=baseline.name,
            candidate=candidate.name,
            baseline_results=baseline_results,
            candidate_results=candidate_results,
            retained_probe=None,
            policy=GatePolicy(require_retained_probe=False, regression_policy=policy_name),
        )
        write_gate_report(report, Path(root).parent / f"bakeoff_gate_{policy_name}")
        reports[policy_name] = report

    def by_domain_score(results: dict[str, bool]) -> dict[str, str]:
        scores: dict[str, list[int]] = {}
        for item in items:
            passed, total = scores.setdefault(item.domain, [0, 0])
            scores[item.domain] = [passed + results[item.item_id], total + 1]
        return {domain: f"{passed}/{total}" for domain, (passed, total) in sorted(scores.items())}

    receipt = {
        "evidence_scope": "isolated local engine-bank bakeoff, "
        + datetime.now(timezone.utc).date().isoformat(),
        "bank": {
            "items_graded": len(items),
            "bank_sha256": reports["strict"]["bank"]["sha256"],
            "horizons": sorted({item.horizon for item in items}),
        },
        "repeated_grading": {
            "repeats": max(1, repeats),
            "totals_per_run": run_totals,
            "pairing": "verdicts pair the LATEST run; flakiness uses all recorded observations",
        },
        "systems": {
            "baseline": {
                "name": baseline.name,
                "by_domain": by_domain_score(baseline_results),
                "total": f"{sum(baseline_results.values())}/{len(items)}",
                "note": "same strength tier that milled the frontier; near-zero is the mill working",
            },
            "candidate": {
                "name": candidate.name,
                "by_domain": by_domain_score(candidate_results),
                "total": f"{sum(candidate_results.values())}/{len(items)}",
            },
        },
        "gate_strict": _gate_summary(reports["strict"]),
        "gate_reliability_aware": _gate_summary(reports["reliability_aware"]),
        "note": "First graded evidence on the engine bank. The receipt omits FENs, move "
        "histories, engine moves, and item ids; full reports stay in the private run dir.",
    }
    if receipt_path:
        Path(receipt_path).parent.mkdir(parents=True, exist_ok=True)
        Path(receipt_path).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Grade bracketed engine tiers on the engine bank.")
    parser.add_argument("--root", default=str(ENGINE_BANK_ROOT))
    parser.add_argument("--chess-depths", type=int, nargs=2, default=(2, 8), metavar=("BASE", "CAND"))
    parser.add_argument("--go-visits", type=int, nargs=2, default=(2, 16), metavar=("BASE", "CAND"))
    parser.add_argument("--receipt", default="results/engine_bank_bakeoff_receipt.json")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    baseline = EngineTier(
        f"engine_shallow_d{args.chess_depths[0]}_v{args.go_visits[0]}", args.chess_depths[0], args.go_visits[0]
    )
    candidate = EngineTier(
        f"engine_mid_d{args.chess_depths[1]}_v{args.go_visits[1]}", args.chess_depths[1], args.go_visits[1]
    )
    receipt = run_engine_bakeoff(args.root, baseline, candidate, args.receipt, repeats=args.repeats)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
