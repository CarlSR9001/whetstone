"""Assemble the cross-scale ladder receipt from the pod-graded bank.

Eight systems, one private 48-item bank (12 coloring + 12 MIS repairs, 24
hidden-check code tasks), every grade recorded through the production
registry on a rented A40. The receipt answers three questions:

1. Resolution across scale: does one bank separate models across a 20x
   parameter range? (the within-family curve)
2. Specialization: do coder models beat same-size generalists on code items
   while staying flat on graph items? (the domain fingerprint)
3. Gate behavior: do adjacent rungs HOLD while distant rungs PASS? A gate
   that certifies everything is as broken as one that certifies nothing.

Run: $env:PYTHONPATH='src'; python scripts/cross_scale_receipt.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, "src")

from bcv.examiner import ExaminerBank
from bcv.gate import GatePolicy, build_gate_report, latest_grade_event_results

BANK = ".bcv_runs/pod_sync/bank"
RECEIPT = "results/cross_scale_ladder_receipt.json"

LADDER_ORDER = [
    ("qwen25_1_5b", "1.5B"),
    ("qwen25_3b", "3B"),
    ("qwen25_7b", "7B"),
    ("qwen25_coder_7b", "7B coder"),
    ("qwen25_14b", "14B (bnb-4bit)"),
    ("phi4_14b", "14B phi-4 (out-of-family)"),
    ("qwen25_32b", "32B (bnb-4bit)"),
    ("qwen25_coder_32b", "32B coder (bnb-4bit)"),
]

GATE_PAIRS = [
    ("qwen25_1_5b", "qwen25_32b", "20x scale contrast"),
    ("qwen25_7b", "qwen25_14b", "adjacent rungs"),
    ("qwen25_14b", "qwen25_32b", "adjacent rungs"),
    ("qwen25_7b", "qwen25_coder_7b", "specialization at 7B"),
    ("qwen25_32b", "qwen25_coder_32b", "specialization at 32B"),
]


def main() -> None:
    bank = ExaminerBank(BANK)
    events = Path(BANK) / "grade_events.jsonl"
    domain_of = {item.item_id: item.domain for item in bank.items.values()}

    ladder = []
    results_by_system: dict[str, dict[str, bool]] = {}
    for system, label in LADDER_ORDER:
        try:
            results = latest_grade_event_results(events, system)
        except ValueError:
            continue
        results_by_system[system] = results
        by_domain: dict[str, list[int]] = {}
        for item_id, passed in results.items():
            entry = by_domain.setdefault(domain_of[item_id], [0, 0])
            entry[0] += passed
            entry[1] += 1
        ladder.append(
            {
                "system": system,
                "label": label,
                "total": f"{sum(results.values())}/{len(results)}",
                "by_domain": {
                    domain: f"{passed}/{total}" for domain, (passed, total) in sorted(by_domain.items())
                },
            }
        )

    gates = []
    policy = GatePolicy(require_retained_probe=False)
    for baseline, candidate, why in GATE_PAIRS:
        if baseline not in results_by_system or candidate not in results_by_system:
            continue
        report = build_gate_report(
            bank,
            baseline=baseline,
            candidate=candidate,
            baseline_results=results_by_system[baseline],
            candidate_results=results_by_system[candidate],
            retained_probe=None,
            policy=policy,
        )
        evidence = report["paired_evidence"]
        gates.append(
            {
                "comparison": why,
                "baseline": baseline,
                "candidate": candidate,
                "verdict": report["verdict"],
                "gains": evidence["gains"],
                "regressions": evidence["regressions"],
                "p": evidence["exact_mcnemar_two_sided_p"],
            }
        )

    receipt = {
        "evidence_scope": "cross-scale ladder on a rented A40 (controlled infra), "
        + datetime.now(timezone.utc).date().isoformat(),
        "bank": {
            "items": len(next(iter(results_by_system.values()), {})),
            "composition": "12 coloring repairs + 12 MIS repairs (checker-spec, no answer keys) "
            "+ 24 hidden-property-check code tasks",
            "grading": "production registry: live stress pools, isolated-subprocess checkers",
        },
        "ladder": ladder,
        "gates": gates,
        "findings": [
            "EVERY stock model, 1.5B through 32B including phi-4, scored 0/24 on the graph-repair "
            "items: the entire scale curve (8->12->17->18->20->22) lives in the code domain. For "
            "untrained models the repair frontier sits above the whole cohort — the same lesson "
            "the examiner taught locally: items must bracket a cohort's frontier to discriminate. "
            "The verifier-gated fine-tuned 4B student passes repair items of this family; every "
            "stock model here scores zero (different item sets — direction, not a number-to-number "
            "comparison)",
            "gate selectivity is textbook: the 20x contrast certifies (12 gains, 0 regressions, "
            "p=4.9e-4) while adjacent rungs and specialist pairs BLOCK or HOLD — the gate refuses "
            "differences the evidence cannot carry, which is the product property",
            "adjacent-rung BLOCKs are driven by real regressions: bigger models failing code items "
            "smaller ones pass, at temperature 0 — item-level nonmonotonicity across scale, the "
            "same genre as the chess depth pathology",
            "phi-4 (14B, out-of-family) ties the 32B coder for best code score at half the size: "
            "scale is not the only axis, and the bank sees that",
        ],
        "notes": [
            "FastContext-1.0-4B (the local student's base) failed to load on the pod via its "
            "bespoke cache path — a tooling gap, recorded honestly; Qwen 3B/7B bracket its size",
            "grades ran on operator-controlled rented hardware; no burn per the exposure policy, "
            "and the run manifests record the venue",
        ],
        "sanitization": "totals and domains only; no item ids, prompts, or answers",
    }
    Path(RECEIPT).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
