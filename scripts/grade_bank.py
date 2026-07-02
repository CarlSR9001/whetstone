"""Grade base 4B and the cook's promoted adapter on the private exam bank.

The saturated static probe said 8/8; the examiner's job is to restore headroom.
Expected: both systems score below ceiling here (the bank is leakage-clean, so it
is dominated by MIS repairs and game moves the student never saw), and every item
picks up discrimination stats from its first real use.

Run: $env:PYTHONPATH='src'; python scripts/grade_bank.py
"""

from __future__ import annotations

import json
import sys

sys.path.insert(0, "src")

from bcv.examiner import ExaminerBank, grade_system
from bcv.transformers_client import TransformersLocalClient

ADAPTER = ".bcv_runs/cook/round_1/fastcontext_graph_repair_lora"


def main() -> None:
    bank = ExaminerBank()
    print(f"promoted bank: {len(bank.promoted_items())} items")
    for system, adapter in (("base_4b", None), ("cook_round1_adapter", ADAPTER)):
        client = TransformersLocalClient(adapter_path=adapter, max_new_tokens=200)
        report = grade_system(bank, system, client)
        client.unload()
        print(json.dumps({k: v for k, v in report.items() if k != "results"}, sort_keys=True))
    discriminating = [
        {"item": item.item_id, "domain": item.domain, "discrimination": item.discrimination()}
        for item in bank.promoted_items()
        if item.discrimination() > 0
    ]
    print(json.dumps({"discriminating_items": discriminating}, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
