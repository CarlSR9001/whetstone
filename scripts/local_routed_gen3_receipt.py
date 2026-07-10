"""Run the real routed gen-3 candidate on the exact retained 48-item bank."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

from bcv.examiner import ExaminerBank
from bcv.gate import bank_hash, file_sha256, latest_grade_event_results
from bcv.registry import grade_bank
from bcv.transformers_client import RoutedAdapterCandidate
from local_same_bank_receipt import DEFAULT_ADAPTER, RecordingCandidate, diagnostics, gate_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Routed FastContext gen-3 same-bank receipt")
    parser.add_argument("--bank", default=".bcv_runs/pod_sync/bank")
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER)
    parser.add_argument("--system", default="fastcontext_4b_gen3_routed_local")
    parser.add_argument("--base-system", default="fastcontext_4b_base_local")
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--receipt", default="results/local_fastcontext_gen3_routed_receipt.json")
    parser.add_argument("--private-out", default=".bcv_runs/local_same_bank/gen3_routed")
    args = parser.parse_args()

    bank = ExaminerBank(args.bank)
    items = bank.promoted_items()
    if len(items) != 48:
        raise SystemExit(f"expected the retained 48-item ladder bank; found {len(items)}")
    events = Path(args.bank) / "grade_events.jsonl"
    base_results = latest_grade_event_results(events, args.base_system)

    inner = RoutedAdapterCandidate(args.adapter, max_new_tokens=args.max_tokens)
    inner.trust_zone = "local_process"
    inner.infrastructure = "rtx_5060"
    candidate = RecordingCandidate(inner)
    try:
        print(f"grading {args.system} on local GPU", flush=True)
        run = grade_bank(bank, args.system, candidate)
        detail = diagnostics(items, candidate.responses, run["results"])
        detail.update({"total": f"{run['passed']}/{run['items']}", "run_manifest": run["run_manifest"]})
        print(f"committed {args.system}: {run['passed']}/{run['items']}", flush=True)
    finally:
        inner.unload()

    private_out = Path(args.private_out)
    gates = [
        gate_summary(
            bank,
            args.base_system,
            args.system,
            base_results,
            run["results"],
            private_out / "base_vs_routed_strict",
        )
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
                args.system,
                qwen32,
                run["results"],
                private_out / "qwen32_vs_routed_strict",
            )
        )

    routed_events = []
    for line in events.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("event") == "grade" and event.get("system") == args.system:
            routed_events.append(event)

    receipt = {
        "evidence_scope": "task-routed FastContext-4B gen-3 on the exact 48-item cross-scale bank, "
        + datetime.now(timezone.utc).date().isoformat(),
        "hardware": "NVIDIA GeForce RTX 5060, local trust boundary",
        "bank": {
            "items": len(run["results"]),
            "bank_sha256": bank_hash(bank),
            "grade_events_sha256": file_sha256(events),
        },
        "system": detail,
        "repeated_grading": {
            "runs": len(routed_events),
            "scores": [sum(event["results"].values()) for event in routed_events],
            "unique_item_outcome_vectors": len({
                json.dumps(event["results"], sort_keys=True) for event in routed_events
            }),
            "fresh_model_load_per_run": True,
        },
        "gates": gates,
        "mechanism": "one PEFT model; graph-repair prompts enable gen-2, all other prompts disable the adapter",
        "sanitization": "counts, failure stages, and hashes only; no item ids, prompts, raw outputs, or adapter paths",
    }
    path = Path(args.receipt)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
