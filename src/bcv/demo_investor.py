"""The promotion gate in sixty seconds — an investor-facing tour of the examiner.

This is a demonstration wrapper, not a new mechanism. Everything that matters is
the production code path from bcv.examiner:

- Items are minted live by the real frontier mint (exact verifier + stress pool).
- Leakage is the real row-identity check against a training buffer on disk.
- Grading is the real checker spec: a repair passes only if it is a verified
  strict refinement that survives the stress pool. No answer key exists.
- Discrimination, saturation, retirement, and the downward-only bucket flow are
  the same ExaminerBank methods the research loop uses.

What is staged, and labeled as staged in the output: the two "systems" under
exam are stored answer policies (one that echoes what it memorized, one that
proposes real repairs). They stand in for a base model and a fine-tuned
candidate so the demo needs no GPU and finishes in well under a minute. Every
grade they receive is computed live. The demo bank is a toy minted into its own
root; the production promotion bank stays private, which is the product.

Run: $env:PYTHONPATH='src'; python -m bcv.demo_investor
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from bcv.domains import COLORING, DOMAINS
from bcv.examiner import (
    ExamItem,
    ExaminerBank,
    grade_game_answer,
    grade_repair_answer,
    mint_game_items,
    mint_repair_items,
    repair_item_prompt,
    training_originals,
)


DEFAULT_ROOT = Path(".bcv_runs/demo_investor")

# Originals the toy student "trained on". These are early frontier expressions
# the mint reliably rediscovers, so the row-identity leakage check fires for
# real: the mint sees these rows in the buffer and quarantines the collisions.
LEAKED_ORIGINALS = ("is_tree", "is_forest", "is_bipartite", "is_triangle_free")
DECOY_ORIGINALS = ("is_complete", "has_universal_vertex")


@dataclass
class DemoConfig:
    root: Path = DEFAULT_ROOT
    seed: int = 0
    max_repair_items: int = 12
    max_game_items: int = 4
    stress_ns: tuple[int, ...] = (7, 8)
    # The candidate policy only "knows" a real repair for most items; on every
    # guess_every-th item it guesses a plausible narrowing instead, and the
    # checker judges the guess live like anything else.
    guess_every: int = 4
    quiet: bool = False


@dataclass
class Ledger:
    path: Path
    events: list[dict] = field(default_factory=list)

    def record(self, event: str, **payload) -> None:
        row = {"ts": datetime.now(timezone.utc).isoformat(), "event": event, **payload}
        self.events.append(row)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def build_training_buffer(path: Path) -> Path:
    """A toy student training buffer in the cook's row format. The leakage
    check parses these rows exactly the way it parses real cook buffers."""
    rows = []
    for expression in LEAKED_ORIGINALS + DECOY_ORIGINALS:
        rows.append(
            {
                "messages": [
                    {"role": "system", "content": "You repair rejected graph conjectures."},
                    {"role": "user", "content": json.dumps({"original_expression": expression})},
                    {"role": "assistant", "content": json.dumps({"repair_expression": "..."})},
                ]
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


# ------------------------------------------------------------------ policies
# Stored answer policies standing in for two model versions. Labeled as staged;
# their answers are graded live by the real checkers.


def base_answer(item: ExamItem) -> object:
    """The memorizer: echoes the predicate it was trained on. This is what
    'the student grading its own homework' looks like to a real checker."""
    if item.kind == "repair":
        return item.payload["original_expression"]
    from bcv.playground import GameRules, legal_moves

    rules = GameRules(**item.payload["rules"])
    moves = legal_moves(tuple(item.payload["state"]), item.payload["player"], rules)
    return list(moves[0]) if moves else None


def candidate_answer(item: ExamItem, index: int, config: DemoConfig) -> object:
    """The gate candidate: proposes a certified repair where it has one, and a
    plausible guess where it does not. The checker doesn't care which is which."""
    if item.kind == "repair":
        if config.guess_every and (index + 1) % config.guess_every == 0:
            return f"({item.payload['original_expression']}) and (max_degree >= 3)"
        for entry in item.lineage:
            if entry.startswith("mined_repair:"):
                return entry.split("mined_repair:", 1)[1]
        return None
    acceptable = item.payload.get("acceptable") or []
    return acceptable[0] if acceptable else None


def grade_policy(bank: ExaminerBank, system: str, answers: dict[str, object], pools: dict) -> dict[str, bool]:
    """Grade stored answers through the real checker specs and record them on
    the bank exactly as grade_system would."""
    results: dict[str, bool] = {}
    for item in bank.promoted_items():
        answer = answers.get(item.item_id)
        if item.kind == "repair":
            results[item.item_id] = grade_repair_answer(item, answer, pool=pools.get(item.domain))
        else:
            results[item.item_id] = grade_game_answer(item, answer)
    bank.record_grades(system, results)
    bank.save()
    return results


# ---------------------------------------------------------------------- demo


def run_demo(config: DemoConfig) -> dict:
    say = (lambda *a, **k: None) if config.quiet else print
    started = time.time()
    root = Path(config.root)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    ledger = Ledger(root / "ledger.jsonl")

    say("WHETSTONE - the promotion gate for AI agents, live in one minute")
    say("=" * 68)
    say("Toy bank, real machinery: minting, leakage quarantine, grading,")
    say("discrimination, retirement below are the production code paths.")
    say("The two systems under exam are stored answer policies (exhibits);")
    say("every grade is computed live. The production bank stays private.")
    say("")

    # 1. The student's training buffer exists on disk before the mint runs.
    buffer_path = build_training_buffer(root / "student_training_buffer.jsonl")
    leak_set = training_originals([buffer_path])
    say(f"[1] Student training buffer: {len(leak_set)} distinct originals on disk")
    ledger.record("training_buffer", originals=sorted(leak_set), path=str(buffer_path))

    # 2. Mint candidate exam items at the verifier frontier.
    say(f"[2] Minting exam items (exact verifier at n<=6, stress pool at n in {config.stress_ns})...")
    minted = mint_repair_items(
        COLORING, [buffer_path], max_items=config.max_repair_items,
        stress_ns=config.stress_ns, seed=config.seed,
    )
    games = mint_game_items(max_items=config.max_game_items, seed=config.seed)
    if games:
        minted += games
    else:
        say("    (no certified playground games on this machine; repair items only)")
    say(f"    {len(minted)} candidate items minted "
        f"({sum(1 for i in minted if i.kind == 'repair')} repair, "
        f"{sum(1 for i in minted if i.kind == 'game_move')} game-move)")

    # 3. Quarantine and promote through the real bank.
    bank = ExaminerBank(root / "bank")
    promoted = quarantined = 0
    for item in minted:
        bank.add(item)
        if item.status == "quarantined":
            quarantined += 1
            ledger.record("quarantine", item=item.item_id,
                          original=item.payload.get("original_expression", ""))
        elif bank.promote(item.item_id):
            promoted += 1
            ledger.record("promote_item", item=item.item_id, domain=item.domain)
    bank.save()
    leaked = [i for i in minted if i.status == "quarantined"]
    say(f"[3] Leakage check (row identity vs training buffer): "
        f"{quarantined} collisions QUARANTINED, {promoted} items promoted to the active bank")
    for item in leaked:
        say(f"    quarantined {item.item_id}: student trained on "
            f"`{item.payload['original_expression']}`")
    say(f"    bank on disk: {bank.root / 'private_promotion_exam.jsonl'}")
    say("")

    # What a system actually sees.
    first_repair = next((i for i in bank.promoted_items() if i.kind == "repair"), None)
    if first_repair is not None:
        say("    Example exam prompt (what a system under exam sees):")
        say(f"    | {repair_item_prompt(first_repair)[:120]}...")
        say("    There is no answer string to leak: the checker accepts ANY verified")
        say("    strict refinement that survives the stress pool.")
    say("")

    # 4. Grade two systems.
    from bcv.refinery import _stress_pool

    pools = {}
    for domain in {i.domain for i in bank.promoted_items() if i.kind == "repair"}:
        pools[domain] = _stress_pool(
            DOMAINS[domain], config.stress_ns, 40, config.seed,
            root / "no_library.jsonl",
        )
    items = bank.promoted_items()
    base_answers = {i.item_id: base_answer(i) for i in items}
    cand_answers = {i.item_id: candidate_answer(i, k, config) for k, i in enumerate(items)}

    say("[4] Grading two systems against the private bank (live checker, no answer keys):")
    base_results = grade_policy(bank, "student_v1_memorizer", base_answers, pools)
    cand_results = grade_policy(bank, "student_v2_candidate", cand_answers, pools)
    base_score = sum(base_results.values())
    cand_score = sum(cand_results.values())
    total = len(items)
    say(f"    student_v1 (memorizer, echoes its training data): {base_score}/{total}")
    say(f"    student_v2 (gate candidate, proposes repairs):    {cand_score}/{total}")
    ledger.record("grades", system="student_v1_memorizer", passed=base_score, of=total)
    ledger.record("grades", system="student_v2_candidate", passed=cand_score, of=total)

    # 5. Discrimination learned from use.
    discriminating = [i for i in bank.promoted_items() if i.discrimination() > 0]
    say(f"[5] Discrimination (learned from grading history): "
        f"{len(discriminating)}/{total} items now separate the two systems")

    # 6. The promotion decision.
    gains = [i for i in items if cand_results[i.item_id] and not base_results[i.item_id]]
    regressions = [i for i in items if base_results[i.item_id] and not cand_results[i.item_id]]
    decision = "PASS" if gains and not regressions else "BLOCK"
    say(f"[6] Promotion gate: {len(gains)} gains, {len(regressions)} regressions "
        f"-> decision: {decision}")
    ledger.record("promotion_decision", decision=decision,
                  gains=len(gains), regressions=len(regressions))

    # 7. Saturation and the downward-only flow.
    retired: list[str] = []
    for _ in range(2):  # two consecutive saturated grading rounds retire an item
        retired = bank.sweep_saturation() or retired
    bank.save()
    trainable = bank.trainable_rows()
    say(f"[7] Saturation sweep: {len(retired)} item(s) stopped discriminating -> retired")
    say(f"    downward-only flow: {len(trainable)} retired item(s) now available as")
    say("    student training fuel; nothing exposed to training re-enters the bank")
    for item_id in retired:
        ledger.record("retire", item=item_id)

    elapsed = round(time.time() - started, 1)
    say("")
    say("=" * 68)
    say(f"Ledger: {ledger.path} ({len(ledger.events)} events, append-only)")
    say(f"Done in {elapsed}s. The asymmetry is the product: the exam evolves,")
    say("leaks are quarantined before they can flatter a candidate, and every")
    say("promotion decision is a reproducible artifact, not a vibe.")

    return {
        "minted": len(minted),
        "quarantined": quarantined,
        "promoted": promoted,
        "base_score": base_score,
        "candidate_score": cand_score,
        "total_items": total,
        "discriminating_items": len(discriminating),
        "gains": len(gains),
        "regressions": len(regressions),
        "decision": decision,
        "retired": len(retired),
        "trainable_rows": len(trainable),
        "elapsed_seconds": elapsed,
        "ledger_events": len(ledger.events),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Whetstone investor demo: the promotion gate, live.")
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-repair-items", type=int, default=12)
    parser.add_argument("--max-game-items", type=int, default=4)
    parser.add_argument("--json", action="store_true", help="print the report dict as JSON")
    args = parser.parse_args()
    report = run_demo(
        DemoConfig(
            root=Path(args.root),
            seed=args.seed,
            max_repair_items=args.max_repair_items,
            max_game_items=args.max_game_items,
        )
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
