"""Cross-scale ladder on a rented A40: does bank resolution span model scale?

Runs ON the pod (controlled rented infrastructure — inside the operator trust
boundary, so no burn; the run manifest records where grading happened). Each
rung is graded through the production registry: real stress pools for the
repair items, real subprocess checkers for the code items.

Run: cd /workspace/whetstone && nohup python scripts/pod_ladder.py > ladder.log 2>&1 &
"""

from __future__ import annotations

import json
import sys
import time

sys.path.insert(0, "src")

from bcv.examiner import ExaminerBank
from bcv.registry import grade_bank
from bcv.transformers_client import TransformersLocalClient

BANK = ".bcv_runs/pod_ladder/bank"
LADDER = [
    ("qwen25_3b", "Qwen/Qwen2.5-3B-Instruct"),
    ("fastcontext_4b", "microsoft/FastContext-1.0-4B-RL"),
    ("qwen25_7b", "Qwen/Qwen2.5-7B-Instruct"),
    ("qwen25_14b", "Qwen/Qwen2.5-14B-Instruct"),
    ("qwen25_32b", "Qwen/Qwen2.5-32B-Instruct"),
]


def main() -> None:
    bank = ExaminerBank(BANK)
    items = bank.promoted_items()
    print(f"promoted items: {len(items)}", flush=True)
    for system, model_name in LADDER:
        started = time.time()
        try:
            client = TransformersLocalClient(model_name=model_name, max_new_tokens=384)
            client.is_external = False  # rented pod is operator-controlled infrastructure
            client.provider = f"runpod_a40/{model_name}"
            report = grade_bank(bank, system=system, candidate=client)
            client.unload()
            print(
                json.dumps(
                    {
                        "system": system,
                        "passed": report["passed"],
                        "items": report["items"],
                        "minutes": round((time.time() - started) / 60.0, 1),
                    }
                ),
                flush=True,
            )
        except Exception as error:  # a failed rung must not kill the ladder
            print(json.dumps({"system": system, "error": f"{type(error).__name__}: {error}"[:300]}), flush=True)
    print("LADDER DONE", flush=True)


if __name__ == "__main__":
    main()
