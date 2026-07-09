"""Create a non-mutating Whetstone promotion-gate report from paired grades.

The bank is private by design. Point --bank-root at its local location; reports
should be kept in a private run directory too because item-level JSON is audit
evidence. A remote/API-grade item must be burned before its result is admitted.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

from bcv.examiner import ExaminerBank
from bcv.gate import GatePolicy, build_gate_report, latest_grade_event_results, write_gate_report


def historical_single_pass_results(bank: ExaminerBank, system: str) -> dict[str, bool]:
    """Recover old one-pass runs; new runs should pass grade_events directly."""
    results = {}
    for item in bank.promoted_items():
        stats = item.graded.get(system)
        if stats is None:
            raise ValueError(f"{system} has no grade for promoted item {item.item_id}")
        total = stats["pass"] + stats["fail"]
        if total != 1:
            raise ValueError(
                f"{system} has {total} aggregate grades for {item.item_id}; use the append-only grade_events trail instead"
            )
        results[item.item_id] = bool(stats["pass"])
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a paired, confidence-aware Whetstone promotion decision.")
    parser.add_argument("--bank-root", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--retained-eval", required=True, help="graph_lora eval_result.json from the retained probe")
    parser.add_argument("--grade-events", default=None, help="append-only grade_events.jsonl; preferred over aggregate counters")
    parser.add_argument("--output-dir", default=".bcv_runs/promotion_gate")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    bank = ExaminerBank(args.bank_root)
    retained = json.loads(Path(args.retained_eval).read_text(encoding="utf-8"))
    if args.grade_events:
        baseline_results = latest_grade_event_results(args.grade_events, args.baseline)
        candidate_results = latest_grade_event_results(args.grade_events, args.candidate)
    else:
        baseline_results = historical_single_pass_results(bank, args.baseline)
        candidate_results = historical_single_pass_results(bank, args.candidate)
    report = build_gate_report(
        bank=bank,
        baseline=args.baseline,
        candidate=args.candidate,
        baseline_results=baseline_results,
        candidate_results=candidate_results,
        retained_probe=retained,
        policy=GatePolicy(confidence_alpha=args.alpha),
    )
    json_path, html_path = write_gate_report(report, args.output_dir)
    print(json.dumps({"verdict": report["verdict"], "reasons": report["reasons"], "json": str(json_path), "html": str(html_path)}, indent=2))


if __name__ == "__main__":
    main()
