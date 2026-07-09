"""One-command, local-only paired code-bank experiments.

This is the reproducible path used for the committed Ollama evidence: mint a
fresh seeded cohort, grade two in-boundary candidates, optionally repeat the
candidate for flakiness evidence, and write the ordinary promotion-gate receipt.
"""

from __future__ import annotations

import json
from pathlib import Path

from bcv.examiner import ExaminerBank
from bcv.gate import GatePolicy, build_gate_report, latest_grade_event, write_gate_report
from bcv.registry import Candidate, grade_bank, mint_domain


def run_local_code_bakeoff(
    root: str | Path,
    baseline_name: str,
    baseline: Candidate,
    candidate_name: str,
    candidate: Candidate,
    *,
    items: int = 12,
    seed: int = 1,
    candidate_repeats: int = 1,
    policy: GatePolicy = GatePolicy(require_retained_probe=False),
) -> dict:
    """Run a fresh private code cohort against two local candidates.

    ``root`` must be empty so a result cannot accidentally blend cohorts. The
    candidate adapters themselves enforce locality; this guard makes the
    no-burn local experiment explicit at the orchestration boundary too.
    """
    if baseline.is_external or candidate.is_external:
        raise ValueError("local bakeoff accepts only in-boundary candidates")
    if candidate_repeats < 1:
        raise ValueError("candidate_repeats must be at least 1")
    bank = ExaminerBank(root)
    if bank.items:
        raise ValueError(f"bakeoff root is not empty: {bank.root}")
    mint = mint_domain(bank, "code", max_items=items, seed=seed)
    baseline_grade = grade_bank(bank, baseline_name, baseline, seed=seed)
    candidate_grades = [
        grade_bank(bank, candidate_name, candidate, seed=seed)
        for _ in range(candidate_repeats)
    ]
    events = bank.root / "grade_events.jsonl"
    baseline_event = latest_grade_event(events, baseline_name)
    candidate_event = latest_grade_event(events, candidate_name)
    baseline_results = baseline_event["results"]
    candidate_results = candidate_event["results"]
    if set(baseline_results) != set(candidate_results):
        raise RuntimeError("bakeoff generated mismatched cohorts")
    report = build_gate_report(
        bank,
        baseline=baseline_name,
        candidate=candidate_name,
        baseline_results=baseline_results,
        candidate_results=candidate_results,
        policy=policy,
        grade_runs={
            "baseline": baseline_event.get("run_manifest", {}),
            "candidate": candidate_event.get("run_manifest", {}),
        },
    )
    gate_json, gate_html = write_gate_report(report, bank.root / "gate")
    summary = {
        "schema_version": 1,
        "mode": "local_only",
        "bank_root": str(bank.root),
        "mint": mint,
        "selection_seed": seed,
        "candidate_repeats": candidate_repeats,
        "baseline": {key: value for key, value in baseline_grade.items() if key != "results"},
        "candidate_runs": [{key: value for key, value in grade.items() if key != "results"} for grade in candidate_grades],
        "gate": {
            "verdict": report["verdict"],
            "gains": report["paired_evidence"]["gains"],
            "regressions": report["paired_evidence"]["regressions"],
            "p_value": report["paired_evidence"]["exact_mcnemar_two_sided_p"],
            "item_set_sha256": report["paired_evidence"]["item_set_sha256"],
            "report_json": str(gate_json),
            "report_html": str(gate_html),
        },
    }
    output = bank.root / "local_bakeoff.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary["summary_json"] = str(output)
    return summary
