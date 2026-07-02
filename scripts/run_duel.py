"""Model-vs-annealer counterexample duel.

Both attackers get the same task: falsify a conjecture by constructing a labeled
graph in its predicate class where degree-descending greedy beats chi. The annealer
is the symbolic baseline the model must be measured against — if the annealer wins
and the model does not, the model has not yet earned frontier status on this task.

Run with: $env:PYTHONPATH='src'; python scripts/run_duel.py
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, "src")

from bcv.graph_adversary import attack_expression, attack_with_model
from bcv.transformers_client import TransformersLocalClient

CLASSES = (
    "is_bipartite and is_connected and max_degree >= 3",
    "is_connected and is_triangle_free and not is_bipartite",
)

LIBRARY = ".bcv_runs/duel/duel_library.jsonl"


def main() -> None:
    root = Path(".bcv_runs/duel")
    root.mkdir(parents=True, exist_ok=True)
    client = TransformersLocalClient(model_name="Qwen/Qwen3-1.7B", max_new_tokens=384)
    rows = []
    for index, expression in enumerate(CLASSES):
        started = time.perf_counter()
        model_result = attack_with_model(client, expression, tries=8, temperature=0.7, library_path=LIBRARY)
        model_seconds = time.perf_counter() - started

        started = time.perf_counter()
        anneal_result = attack_expression(
            expression,
            ns=(8, 9, 10, 11, 12),
            restarts=8,
            steps=2000,
            seed=100 + index,
            library_path=LIBRARY,
        )
        anneal_seconds = time.perf_counter() - started
        rows.append(
            {
                "expression": expression,
                "model": asdict(model_result),
                "model_seconds": round(model_seconds, 1),
                "anneal": asdict(anneal_result),
                "anneal_seconds": round(anneal_seconds, 1),
            }
        )
        print(
            f"{expression}\n"
            f"  model : {'FALSIFIED in ' + str(model_result.restarts_used) + ' tries' if model_result.falsified else 'failed all tries'}"
            f" ({model_seconds:.0f}s, best greedy in-class {model_result.best_greedy_seen})\n"
            f"  anneal: {'FALSIFIED in ' + str(anneal_result.restarts_used) + ' restarts' if anneal_result.falsified else 'failed'}"
            f" ({anneal_seconds:.0f}s)"
        )
    (root / "duel_results.json").write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
