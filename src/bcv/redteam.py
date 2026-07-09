"""Hostile self-tests for Whetstone's leakage quarantine and promotion gate."""

from __future__ import annotations

import json
from pathlib import Path

from bcv.examiner import ExamItem, ExaminerBank
from bcv.gate import GatePolicy, build_gate_report
from bcv.leakage import assess_dsl_leakage


def _item(item_id: str, status: str = "candidate") -> ExamItem:
    return ExamItem(
        item_id=item_id,
        domain="coloring",
        kind="repair",
        payload={"original_expression": "not (not (is_bipartite))"},
        oracle="redteam",
        source="redteam",
        horizon="test",
        lineage=[],
        status=status,
    )


def run_redteam(root: str | Path = ".bcv_runs/redteam", observations=None) -> dict:
    """Try to manufacture a false PASS with a paraphrased leaked item set."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if observations is None:
        from bcv.domains import COLORING
        from bcv.refinery import _observe_all

        observations = _observe_all(COLORING, 6, root / "oracle")
    assessment = assess_dsl_leakage("not (not (is_bipartite))", {"is_bipartite"}, observations)

    # Attack 1: the paraphrase misses literal row identity but must be stopped by
    # the behavioral tier before it can enter a promotion bank.
    paraphrase_bank = ExaminerBank(root / "paraphrase_bank")
    paraphrase = _item("paraphrase")
    paraphrase.leakage_risk = assessment.risk
    paraphrase.leakage_match = assessment.match
    if assessment.risk:
        paraphrase.status = "quarantined"
    paraphrase_bank.add(paraphrase)
    promotion_allowed = paraphrase_bank.promote(paraphrase.item_id)
    paraphrase_bank.save()

    # Attack 2: a toy student has memorized six quarantined items. If a careless
    # bank admitted them, their six wins would clear alpha=.05. The real bank
    # keeps only a clean tie in scope, so inflation cannot promote the student.
    protected = ExaminerBank(root / "protected_bank")
    unsafe = ExaminerBank(root / "unsafe_counterfactual_bank")
    leaked_ids = [f"leaked_{index}" for index in range(6)]
    for item_id in leaked_ids:
        leaked = _item(item_id, status="quarantined")
        leaked.leakage_risk = assessment.risk
        leaked.leakage_match = assessment.match
        protected.add(leaked)
        unsafe.add(_item(item_id, status="promoted"))
    protected.add(_item("clean_tie", status="promoted"))
    unsafe.add(_item("clean_tie", status="promoted"))
    policy = GatePolicy(require_retained_probe=False)
    protected_report = build_gate_report(
        protected, "base", "memorizer", {"clean_tie": False}, {"clean_tie": False}, policy=policy
    )
    unsafe_base = {item_id: False for item_id in [*leaked_ids, "clean_tie"]}
    unsafe_candidate = {item_id: item_id in leaked_ids for item_id in unsafe_base}
    unsafe_report = build_gate_report(unsafe, "base", "memorizer", unsafe_base, unsafe_candidate, policy=policy)
    report = {
        "schema_version": 1,
        "paraphrase_attack": {
            "training_expression": "is_bipartite",
            "candidate_expression": "not (not (is_bipartite))",
            "row_identity_evaded": "not (not (is_bipartite))" != "is_bipartite",
            "leakage_match": assessment.match,
            "promotion_allowed": promotion_allowed,
        },
        "inflation_attack": {
            "leaked_items": len(leaked_ids),
            "unsafe_counterfactual_verdict": unsafe_report["verdict"],
            "protected_verdict": protected_report["verdict"],
            "caught": unsafe_report["verdict"] == "PASS" and protected_report["verdict"] != "PASS",
        },
    }
    (root / "redteam_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
