"""Claude-as-pilot experiment: can competence be injected through the poke channel?

Two arms on the six heldout repair problems, Qwen3-1.7B, ANSWER gate on:

  solo   the model alone (gate forces CHECK-before-ANSWER)
  pilot  same, plus a [PILOT] note poked into the rewind-proof notepad at step 2 —
         the pilot's analysis of the counterexample structure, produced by a much
         stronger model (Claude) studying the same evidence. The small model still
         has to read the hint, form the expression, CHECK it, and commit.

This measures channel capacity: not whether the big model can solve the problem
(it can), but whether the harness's memory/notepad channel can carry that
competence into a weak reasoner's loop.

Run: $env:PYTHONPATH='src'; python scripts/gundam_run.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, "src")

from bcv.emulator import ReasoningEmulator, repair_task_text
from bcv.graph_lora import _load_examples
from bcv.transformers_client import TransformersLocalClient


def pilot_note(original: str) -> str:
    return (
        f"[PILOT] Analysis: every counterexample shown has max_degree <= 2 (paths and "
        f"cycles), while kept examples include higher-degree graphs. Excluding the "
        f"low-degree failures should work. CHECK this exact expression next: "
        f"({original}) and (max_degree >= 3)"
    )


def main() -> None:
    examples = _load_examples(Path(".bcv_runs/graph_repair_hard_rich/hard_heldout.jsonl"))[:6]
    client = TransformersLocalClient(model_name="Qwen/Qwen3-1.7B", max_new_tokens=260)
    rows = []
    for index, example in enumerate(examples):
        task, original = repair_task_text(example)
        row: dict = {"index": index, "original": original}
        for arm, piloted in (("solo", False), ("pilot", True)):
            emulator = ReasoningEmulator(
                client, task, original_expression=original, max_steps=8
            )
            while not emulator.finished and emulator.step_index < emulator.max_steps:
                emulator.step()
                if piloted and emulator.step_index == 2 and not emulator.finished:
                    # Transcript-tail injection: the notepad is a low-bandwidth
                    # channel (positionally far from the generation point); the
                    # recency slot is where a poke actually lands.
                    emulator.transcript.append(pilot_note(original))
                    emulator._log("poke_transcript", "pilot hint injected at tail")
            result = emulator.result()
            row[arm] = {
                "solved": result.answer_refines,
                "answer": result.answer,
                "steps": result.steps,
                "controls": result.controls_used,
                "gate_events": sum(1 for e in result.events if e["kind"] == "gate"),
            }
            print(
                f"#{index} {arm:5s} {'SOLVED' if result.answer_refines else 'failed'} "
                f"steps={result.steps} controls={result.controls_used}"
            )
        rows.append(row)
        Path(".bcv_runs/gundam").mkdir(parents=True, exist_ok=True)
        Path(".bcv_runs/gundam/run.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    client.unload()
    solo = sum(1 for r in rows if r["solo"]["solved"])
    pilot = sum(1 for r in rows if r["pilot"]["solved"])
    print(f"\nsolo {solo}/{len(rows)}  |  pilot {pilot}/{len(rows)}")


if __name__ == "__main__":
    main()
