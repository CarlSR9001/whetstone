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
# qwen25_3b (12/48) and qwen25_7b (17/48) already graded. Remaining rungs use
# PRE-QUANTIZED checkpoints: the /workspace volume has a ~20 GB quota, and a
# bnb-on-the-fly "32B 4-bit" load downloads ~65 GB of bf16 shards first.
LADDER = [
    ("fastcontext_4b", "microsoft/FastContext-1.0-4B-RL"),
    ("qwen25_14b", "unsloth/Qwen2.5-14B-Instruct-bnb-4bit"),
    ("qwen25_32b", "unsloth/Qwen2.5-32B-Instruct-bnb-4bit"),
]


def wipe_model_cache() -> None:
    """Per-rung cache wipe: the quota holds one big model at a time."""
    import shutil

    for sub in ("hub", "xet"):
        shutil.rmtree(f"/workspace/hf_cache/{sub}", ignore_errors=True)


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
            client.trust_zone = "operator_controlled_rented_gpu"
            client.infrastructure = "runpod_a40"
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
        finally:
            wipe_model_cache()
    print("LADDER DONE", flush=True)


if __name__ == "__main__":
    main()
