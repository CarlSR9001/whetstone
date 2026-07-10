"""Extended overnight ladder: families, specialists, and a 72B ceiling.

Adds to the cross-scale receipt: a coder-specialist pair (does the bank detect
SPECIALIZATION — coder models beating same-size generalists on code items but
not graph items?), an out-of-family model, and Qwen2.5-72B as the 48GB-fits
ceiling. Cheapest first so a dead pod loses only the biggest rung.

Run: HF_HOME=/workspace/hf_cache python scripts/pod_ladder3.py > ladder3.log 2>&1
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
# Pre-quantized where big (see pod_ladder.py: ~20 GB volume quota). 72B is cut:
# even pre-quantized it is ~39 GB. Ceiling stays 32B; the specialist pair
# (coder vs generalist at 7B and 32B) is the interesting comparison anyway.
LADDER = [
    ("qwen25_1_5b", "Qwen/Qwen2.5-1.5B-Instruct"),
    ("qwen25_coder_7b", "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit"),
    ("phi4_14b", "unsloth/phi-4-bnb-4bit"),
    ("qwen25_coder_32b", "unsloth/Qwen2.5-Coder-32B-Instruct-bnb-4bit"),
]


def wipe_model_cache() -> None:
    import shutil

    for sub in ("hub", "xet"):
        shutil.rmtree(f"/workspace/hf_cache/{sub}", ignore_errors=True)


def main() -> None:
    bank = ExaminerBank(BANK)
    print(f"promoted items: {len(bank.promoted_items())}", flush=True)
    for system, model_name in LADDER:
        started = time.time()
        try:
            client = TransformersLocalClient(model_name=model_name, max_new_tokens=384)
            client.is_external = False
            client.provider = f"runpod_a40/{model_name}"
            client.trust_zone = "operator_controlled_rented_gpu"
            client.infrastructure = "runpod_a40"
            report = grade_bank(bank, system=system, candidate=client)
            client.unload()
            print(json.dumps({"system": system, "passed": report["passed"], "items": report["items"],
                              "minutes": round((time.time() - started) / 60.0, 1)}), flush=True)
        except Exception as error:
            print(json.dumps({"system": system, "error": f"{type(error).__name__}: {error}"[:300]}), flush=True)
        finally:
            wipe_model_cache()
    print("LADDER3 DONE", flush=True)


if __name__ == "__main__":
    main()
