"""Grade cached FastContext base/adapter systems on the pod ladder's exact bank.

The public receipt contains counts, failure stages, hashes, and paired gate
summaries only. Prompts, item ids, raw outputs, and local adapter paths remain
inside the ignored bank/run directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, "src")

from bcv.examiner import ExaminerBank
from bcv.gate import GatePolicy, bank_hash, build_gate_report, file_sha256, latest_grade_event_results, write_gate_report
from bcv.graph_agent import compile_feature_expression
from bcv.registry import grade_bank
from bcv.transformers_client import TransformersLocalClient, extract_json


DEFAULT_BANK = ".bcv_runs/pod_sync/bank"
DEFAULT_ADAPTER = (
    r"C:\Users\shank\Documents\AI Arch #39\.bcv_runs\cook2\gen2"
    r"\fastcontext_graph_repair_lora"
)


class RecordingCandidate:
    """Transparent candidate wrapper that retains outputs only for this process."""

    def __init__(self, candidate: TransformersLocalClient) -> None:
        object.__setattr__(self, "candidate", candidate)
        object.__setattr__(self, "responses", [])

    def __getattr__(self, name):
        return getattr(self.candidate, name)

    def generate_text(self, prompt: str, temperature: float = 0.0) -> str:
        response = self.candidate.generate_text(prompt, temperature=temperature)
        self.responses.append(response)
        return response


def diagnostics(items, responses: list[str], results: dict[str, bool]) -> dict:
    stages: Counter[str] = Counter()
    by_domain: dict[str, list[int]] = {}
    response_hashes: list[str] = []
    for item, response in zip(items, responses, strict=True):
        passed = bool(results[item.item_id])
        score = by_domain.setdefault(item.domain, [0, 0])
        score[0] += passed
        score[1] += 1
        response_hashes.append(hashlib.sha256(response.encode("utf-8")).hexdigest())
        if passed:
            stages[f"{item.domain}:pass"] += 1
            continue
        if not response.strip():
            stages[f"{item.domain}:empty"] += 1
        elif item.domain in {"coloring", "mis"}:
            parsed = extract_json(response)
            expression = parsed.get("repair_expression") if isinstance(parsed, dict) else None
            if not isinstance(expression, str):
                stages[f"{item.domain}:output_contract_failure"] += 1
            else:
                try:
                    compile_feature_expression(expression)
                except Exception:
                    stages[f"{item.domain}:invalid_dsl"] += 1
                else:
                    stages[f"{item.domain}:verifier_reject"] += 1
        else:
            stages[f"{item.domain}:checker_reject"] += 1
    return {
        "by_domain": {
            domain: f"{passed}/{total}"
            for domain, (passed, total) in sorted(by_domain.items())
        },
        "failure_stages": dict(sorted(stages.items())),
        "response_set_sha256": hashlib.sha256(
            json.dumps(response_hashes, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }


def grade_local(bank: ExaminerBank, system: str, adapter_path: str | None, max_tokens: int) -> tuple[dict, dict]:
    print(f"grading {system} on local GPU", flush=True)
    inner = TransformersLocalClient(adapter_path=adapter_path, max_new_tokens=max_tokens)
    inner.trust_zone = "local_process"
    inner.infrastructure = "rtx_5060"
    candidate = RecordingCandidate(inner)
    items = bank.promoted_items()
    try:
        report = grade_bank(bank, system=system, candidate=candidate)
        detail = diagnostics(items, candidate.responses, report["results"])
        detail.update({
            "total": f"{report['passed']}/{report['items']}",
            "run_manifest": report["run_manifest"],
        })
        print(f"committed {system}: {report['passed']}/{report['items']}", flush=True)
        return report["results"], detail
    finally:
        inner.unload()


def gate_summary(
    bank: ExaminerBank,
    baseline: str,
    candidate: str,
    baseline_results: dict,
    candidate_results: dict,
    out: Path,
    regression_policy: str = "strict",
) -> dict:
    report = build_gate_report(
        bank,
        baseline,
        candidate,
        baseline_results,
        candidate_results,
        retained_probe=None,
        policy=GatePolicy(
            require_retained_probe=False,
            regression_policy=regression_policy,
        ),
    )
    write_gate_report(report, out)
    evidence = report["paired_evidence"]
    return {
        "baseline": baseline,
        "candidate": candidate,
        "regression_policy": regression_policy,
        "verdict": report["verdict"],
        "reasons": report["reasons"],
        "gains": evidence["gains"],
        "regressions": evidence["regressions"],
        "ties": evidence["ties"],
        "exact_mcnemar_two_sided_p": evidence["exact_mcnemar_two_sided_p"],
        "by_domain": evidence["by_domain"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Local FastContext same-bank receipt")
    parser.add_argument("--bank", default=DEFAULT_BANK)
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER)
    parser.add_argument("--base-system", default="fastcontext_4b_base_local")
    parser.add_argument("--adapter-system", default="fastcontext_4b_gen2_local")
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--skip-base", action="store_true")
    parser.add_argument("--skip-adapter", action="store_true")
    parser.add_argument("--receipt", default="results/local_fastcontext_same_bank_receipt.json")
    parser.add_argument("--private-out", default=".bcv_runs/local_same_bank")
    args = parser.parse_args()

    bank = ExaminerBank(args.bank)
    if len(bank.promoted_items()) != 48:
        raise SystemExit(f"expected the retained 48-item ladder bank; found {len(bank.promoted_items())}")
    events = Path(args.bank) / "grade_events.jsonl"
    private_out = Path(args.private_out)
    private_out.mkdir(parents=True, exist_ok=True)

    receipt_path = Path(args.receipt)
    prior_receipt = (
        json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt_path.exists()
        else {}
    )
    systems: dict[str, dict] = dict(prior_receipt.get("systems", {}))
    if args.skip_base:
        base_results = latest_grade_event_results(events, args.base_system)
    else:
        base_results, systems[args.base_system] = grade_local(
            bank, args.base_system, None, args.max_tokens
        )
    if args.skip_adapter:
        adapter_results = latest_grade_event_results(events, args.adapter_system)
    else:
        adapter_results, systems[args.adapter_system] = grade_local(
            bank, args.adapter_system, args.adapter, args.max_tokens
        )

    gates = [
        gate_summary(
            bank,
            args.base_system,
            args.adapter_system,
            base_results,
            adapter_results,
            private_out / "base_vs_gen2",
        ),
        gate_summary(
            bank,
            args.base_system,
            args.adapter_system,
            base_results,
            adapter_results,
            private_out / "base_vs_gen2_reliability_aware",
            regression_policy="reliability_aware",
        ),
    ]
    try:
        qwen32 = latest_grade_event_results(events, "qwen25_32b")
    except ValueError:
        pass
    else:
        gates.append(
            gate_summary(
                bank,
                "qwen25_32b",
                args.adapter_system,
                qwen32,
                adapter_results,
                private_out / "qwen32_vs_gen2",
            )
        )

    receipt = {
        "evidence_scope": "cached FastContext-4B base and gen-2 adapter on the exact 48-item cross-scale bank, "
        + datetime.now(timezone.utc).date().isoformat(),
        "hardware": "NVIDIA GeForce RTX 5060, local trust boundary",
        "bank": {
            "items": len(base_results),
            "bank_sha256": bank_hash(bank),
            "grade_events_sha256": file_sha256(events),
        },
        "systems": systems,
        "gates": gates,
        "sanitization": "counts, failure stages, and hashes only; no item ids, prompts, raw outputs, or adapter paths",
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
