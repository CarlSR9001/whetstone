"""The cook: the full loop, unattended, on the 4B.

Round structure (resumable — a crash costs one round):
  round 1  buffer = rich hard repair train set
  round 2  buffer += stress-mined repair train + original hard train (dedup)
  round 3  buffer += playground game experience (certified-game depth-2 moves)

Each round: QLoRA-train the 4B on the cumulative buffer (masked loss), then GATE on
the fixed probe (rich heldout, strict verifier refinement metric). An adapter is
promoted only if it beats the incumbent. Curriculum growth per round doubles as an
experiment: does cross-task verified experience (games) help or hurt the repair
probe? The ledger records the answer either way.

Run: $env:PYTHONPATH='src'; python scripts/cook.py
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, "src")

ROOT = Path(".bcv_runs/cook")
PROBE = Path(".bcv_runs/graph_repair_hard_rich/hard_heldout.jsonl")
STATE = ROOT / "state.json"


def log_gpu(tag: str) -> None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,memory.used,power.draw", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    except Exception:
        return
    with (ROOT / "gpu_log.csv").open("a", encoding="utf-8") as handle:
        handle.write(f"{datetime.now().isoformat()},{tag},{out}\n")


def load_rows(path: str) -> list[str]:
    p = Path(path)
    if not p.exists():
        return []
    return [line for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def game_rows(cap: int = 120, seed: int = 7) -> list[str]:
    experience = load_rows(".bcv_runs/playground/experience.jsonl")
    rng = random.Random(seed)
    rng.shuffle(experience)
    rows = []
    for line in experience[:cap]:
        raw = json.loads(line)
        rows.append(json.dumps({
            "messages": [
                {"role": "system", "content": (
                    "You are playing a certified board game on a row of cells "
                    "(0 empty, 1 stick, 2 stone). Reply only JSON: {\"move\": [...]}. "
                    "Moves: [\"place\", cell] or [\"shift\", from, to] or [\"capture\", cell]."
                )},
                {"role": "user", "content": json.dumps({"game": raw["game"], "state": raw["state"], "player": raw["player"]}, sort_keys=True)},
                {"role": "assistant", "content": json.dumps({"move": raw["move"]})},
            ]
        }, sort_keys=True))
    return rows


def build_buffer(round_index: int) -> Path:
    rows = load_rows(".bcv_runs/graph_repair_hard_rich/hard_train.jsonl")
    if round_index >= 2:
        rows += load_rows(".bcv_runs/graph_repair_hard_stress/hard_train.jsonl")
        rows += load_rows(".bcv_runs/graph_repair_hard/hard_train.jsonl")
    if round_index >= 3:
        rows += game_rows()
    rows = list(dict.fromkeys(rows))
    ROOT.mkdir(parents=True, exist_ok=True)
    path = ROOT / f"buffer_round_{round_index}.jsonl"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def main() -> None:
    from bcv.graph_lora import evaluate_graph_adapter, train_graph_adapter

    ROOT.mkdir(parents=True, exist_ok=True)
    state = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {"rounds": {}, "best": None, "best_score": -1}

    for round_index in (1, 2, 3):
        key = str(round_index)
        if key in state["rounds"]:
            print(f"round {round_index}: resumed (score {state['rounds'][key]['adapter_refined']})")
            continue
        log_gpu(f"round_{round_index}_start")
        buffer = build_buffer(round_index)
        n_rows = len(load_rows(buffer))
        out_dir = ROOT / f"round_{round_index}"
        train = train_graph_adapter(
            dataset_path=buffer, output_dir=out_dir, max_train_examples=200,
            heldout_examples=0, epochs=2, max_length=1024, lora_r=8, lora_alpha=16,
            heldout_path=PROBE, mask_prompt_loss=True,
        )
        if not train.accepted:
            print(f"round {round_index}: TRAIN FAILED {train.failure}"); break
        evaluation = evaluate_graph_adapter(
            adapter_path=train.adapter_path, dataset_path=buffer,
            output_dir=out_dir / "eval", heldout_path=PROBE, max_n=6,
        )
        score = evaluation.adapter_refined
        promoted = score > state["best_score"]
        if promoted:
            state["best"], state["best_score"] = train.adapter_path, score
        state["rounds"][key] = {
            "buffer_rows": n_rows, "final_loss": train.final_loss,
            "adapter_refined": score, "base_refined": evaluation.base_refined,
            "distinct": evaluation.distinct_adapter_expressions,
            "promoted": promoted,
        }
        STATE.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
        log_gpu(f"round_{round_index}_end")
        print(f"round {round_index}: buffer {n_rows} rows, probe {score}/8 (base {evaluation.base_refined}), {'PROMOTED' if promoted else 'held'}")

    (ROOT / "cook_report.json").write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(state, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
