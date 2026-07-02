"""Cook 2: the multi-domain round. One student, five ladders.

Buffer (asymmetry-enforced — any position/original present in the CURRENT promoted
exam bank is excluded from training rows, in code, before training):

  - round-3 buffer (evidence-rich coloring repairs + playground games)
  - terse exam-format repair rows for coloring AND MIS (attacks the two frontiers
    the examiner exposed: domain shift and format shift)
  - chess: Stockfish-d12-labeled positions (FEN -> UCI)
  - arcade: MC-64-labeled positions (board -> move index)
  - go: KataGo-v48-labeled positions (moves -> vertex), if milled

Then: train gen-2 on the 4B, grade base and gen-2 on the full promoted bank,
sweep saturation, report per-domain deltas.

Run: $env:PYTHONPATH='src'; python scripts/cook2.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

ROOT = Path(".bcv_runs/cook2")
PROBE = Path(".bcv_runs/graph_repair_hard_rich/hard_heldout.jsonl")


def bank_signatures():
    from bcv.examiner import ExaminerBank

    bank = ExaminerBank()
    signatures = {"fen": set(), "arcade": set(), "go": set(), "original": set()}
    for item in bank.promoted_items():
        payload = item.payload
        if "fen" in payload:
            signatures["fen"].add(payload["fen"])
        elif "moves" in payload:
            signatures["go"].add((tuple(payload["moves"]), payload.get("to_move")))
        elif "state" in payload:
            signatures["arcade"].add((tuple(payload["state"]), payload.get("player")))
        if "original_expression" in payload:
            signatures["original"].add(payload["original_expression"])
    return bank, signatures


def sft(system: str, user: str, assistant: str) -> str:
    return json.dumps(
        {"messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]},
        sort_keys=True,
    )


def terse_repair_rows(signatures, max_per_domain: int = 24) -> list[str]:
    """Exam-format repair supervision for coloring + MIS, promoted-originals excluded."""
    from bcv.domains import COLORING, MIS
    from bcv.examiner import repair_item_prompt, ExamItem
    from bcv.graph_repair_data import _candidate_expressions
    from bcv.refinery import _mine_stress_repair, _observe_all, _stress_pool, _verify

    rows: list[str] = []
    for domain in (COLORING, MIS):
        observations = _observe_all(domain, 6, ROOT)
        pool = _stress_pool(domain, (7, 8), 40, 0, ROOT / "nolib.jsonl")
        count = 0
        for expression in _candidate_expressions():
            if count >= max_per_domain:
                break
            if expression in signatures["original"]:
                continue  # hard asymmetry: promoted exam originals are off-limits
            verdict = _verify(expression, observations)
            if verdict is None or verdict[1] is None:
                continue
            repair = _mine_stress_repair(expression, observations, pool)
            if repair is None:
                continue
            fake = ExamItem(
                item_id="x", domain=domain.name, kind="repair",
                payload={
                    "original_expression": expression,
                    "counterexample": verdict[1].graph.graph_id(),
                    "claim": domain.claim,
                },
                oracle="", source="", horizon="", lineage=[],
            )
            rows.append(sft(
                "You repair rejected graph conjectures. Reply only JSON.",
                repair_item_prompt(fake),
                json.dumps({"repair_expression": repair}),
            ))
            count += 1
    return rows


def chess_rows(signatures) -> list[str]:
    path = Path(".bcv_runs/chess/experience.jsonl")
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        if raw["fen"] in signatures["fen"]:
            continue
        rows.append(sft(
            "You are a strong chess player. Reply only JSON.",
            f'Given the FEN, reply only JSON with the best move in UCI notation: {{"move": "<uci>"}}. FEN: {raw["fen"]}',
            json.dumps({"move": raw["oracle_move"]}),
        ))
    return rows


def arcade_rows(signatures) -> list[str]:
    path = Path(".bcv_runs/arcade/experience.jsonl")
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        if (tuple(raw["board"]), raw["to_move"]) in signatures["arcade"]:
            continue
        rows.append(sft(
            "You are playing certified board games. Reply only JSON.",
            f'You are playing {raw["game"]} (board row-major, 0 empty, 1 player0, 2 player1). '
            f'Reply only JSON with the best move index: {{"move": <int>}}. '
            f'Position: {json.dumps({"board": raw["board"], "player": raw["to_move"]}, sort_keys=True)}',
            json.dumps({"move": raw["oracle_move"]}),
        ))
    return rows


def go_rows(signatures) -> list[str]:
    path = Path(".bcv_runs/go/experience.jsonl")
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        if (tuple(raw["moves"]), raw["to_move"]) in signatures["go"]:
            continue
        rows.append(sft(
            "You are a strong Go player. Reply only JSON.",
            "You are a strong Go player on a 9x9 board (komi 7). The game so far as GTP vertices, "
            f'alternating from Black: {json.dumps(raw["moves"])}. Reply only JSON with the best '
            f'next move for {raw["to_move"]} as a GTP vertex: {{"move": "<vertex>"}}',
            json.dumps({"move": raw["oracle_move"]}),
        ))
    return rows


def main() -> None:
    from bcv.examiner import grade_system
    from bcv.graph_lora import train_graph_adapter
    from bcv.transformers_client import TransformersLocalClient

    ROOT.mkdir(parents=True, exist_ok=True)
    bank, signatures = bank_signatures()
    base_buffer = [
        line for line in Path(".bcv_runs/cook/buffer_round_3.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    sections = {
        "round3": base_buffer,
        "terse_repairs": terse_repair_rows(signatures),
        "chess": chess_rows(signatures),
        "arcade": arcade_rows(signatures),
        "go": go_rows(signatures),
    }
    rows = list(dict.fromkeys(row for section in sections.values() for row in section))
    buffer_path = ROOT / "buffer.jsonl"
    buffer_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    composition = {name: len(section) for name, section in sections.items()}
    print(json.dumps({"buffer": len(rows), "composition": composition}, sort_keys=True))

    train = train_graph_adapter(
        dataset_path=buffer_path, output_dir=ROOT / "gen2", max_train_examples=500,
        heldout_examples=0, epochs=1, max_length=1024, lora_r=8, lora_alpha=16,
        heldout_path=PROBE, mask_prompt_loss=True,
    )
    print(json.dumps({"train_accepted": train.accepted, "final_loss": train.final_loss,
                      "rows_trained": train.train_examples}))
    if not train.accepted:
        print(train.failure)
        return

    report = {}
    for system, adapter in (("base_4b_v2", None), ("gen2", train.adapter_path)):
        client = TransformersLocalClient(adapter_path=adapter, max_new_tokens=200)
        graded = grade_system(bank, system, client)
        client.unload()
        by_domain: dict[str, list[int]] = {}
        for item_id, passed in graded["results"].items():
            domain = bank.items[item_id].domain
            by_domain.setdefault(domain, [0, 0])
            by_domain[domain][0] += int(passed)
            by_domain[domain][1] += 1
        report[system] = {
            "passed": graded["passed"], "items": graded["items"],
            "by_domain": {d: f"{p}/{t}" for d, (p, t) in sorted(by_domain.items())},
        }
        print(json.dumps({system: report[system]}, sort_keys=True))
    retired = bank.sweep_saturation()
    bank.save()
    report["retired_saturated"] = retired
    (ROOT / "cook2_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
