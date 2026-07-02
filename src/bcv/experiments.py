from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path

from bcv.benchmark import record_benchmark, run_document_corruption_benchmark
from bcv.bisect_probe import run_bisect_probe
from bcv.memory import run_memory_hygiene_probe
from bcv.miner import mine_training_candidates, write_training_candidates
from bcv.model_probe import run_model_document_probe, run_model_research_probe
from bcv.research import run_research_contradiction_probe
from bcv.router import run_router_probe
from bcv.tripwire import train_document_tripwire
from bcv.schema import Event, TestResult
from bcv.sft_export import export_training_datasets
from bcv.store import CognitiveStore


def run_all(root: str | Path = ".bcv_runs/all", reset: bool = True) -> dict[str, object]:
    root = Path(root)
    if reset and root.exists():
        resolved_parts = {part.lower() for part in root.resolve().parts}
        if ".bcv_runs" not in resolved_parts:
            raise ValueError(f"refusing to reset non-generated run root: {root}")
        shutil.rmtree(root)
    document_results = run_document_corruption_benchmark()
    record_benchmark(root / "document_conservation", document_results)

    research_result = run_research_contradiction_probe()
    research_store = CognitiveStore(root / "research_synthesis")
    research_store.init()
    branch = "experiment/research-contradictions"
    research_store.create_branch(branch, from_branch="main")
    research_store.commit(
        branch,
        "record research contradiction probe",
        [
            Event(
                event_type="verifier_result",
                actor="verifier",
                message="research graph accepted" if research_result.accepted else "research graph rejected",
                output_refs=tuple(research_result.final_claim_ids),
                tests=(
                    TestResult(
                        "claim_support_and_contradiction_check",
                        "pass" if research_result.accepted else "fail",
                        json.dumps(asdict(research_result), sort_keys=True),
                    ),
                ),
            )
        ],
    )

    memory_results = run_memory_hygiene_probe()
    memory_store = CognitiveStore(root / "memory_hygiene")
    memory_store.init()
    memory_branch = "experiment/memory-hygiene"
    memory_store.create_branch(memory_branch, from_branch="main")
    for result in memory_results:
        memory_store.commit(
            memory_branch,
            f"record memory hygiene result: {result.memory_id}",
            [
                Event(
                    event_type="verifier_result",
                    actor="verifier",
                    message="memory accepted" if result.accepted else ",".join(result.failures),
                    output_refs=(result.memory_id,),
                    tests=(
                        TestResult(
                            "memory_hygiene",
                            "pass" if result.accepted else "fail",
                            json.dumps(asdict(result), sort_keys=True),
                        ),
                    ),
                )
            ],
        )

    router_results = run_router_probe()
    router_store = CognitiveStore(root / "inference_router")
    router_store.init()
    router_branch = "experiment/inference-router"
    router_store.create_branch(router_branch, from_branch="main")
    for name, decision in router_results.items():
        router_store.commit(
            router_branch,
            f"record router decision: {name}",
            [
                Event(
                    event_type="verifier_result",
                    actor="runtime",
                    message=",".join(decision.actions),
                    output_refs=(f"route:{name}",),
                    tests=(
                        TestResult(
                            "route_has_action",
                            "pass" if decision.actions else "fail",
                            json.dumps(asdict(decision), sort_keys=True),
                        ),
                    ),
                )
            ],
        )

    bisect_result = run_bisect_probe(root / "bisect")
    model_probe_result = run_model_document_probe(root / "local_model_document")
    model_research_result = run_model_research_probe(root / "local_model_research")
    tripwire_result = train_document_tripwire(root / "tripwire")

    training_candidates_path = write_training_candidates(root)
    training_candidates = mine_training_candidates(root)
    dataset_export = export_training_datasets(root)

    return {
        "document_conservation": [asdict(result) for result in document_results],
        "research_synthesis": asdict(research_result),
        "memory_hygiene": [asdict(result) for result in memory_results],
        "training_candidates": {
            "path": str(training_candidates_path),
            "count": len(training_candidates),
            "verified_positive": sum(
                1 for candidate in training_candidates if candidate.label == "verified_positive"
            ),
            "repair_required": sum(
                1 for candidate in training_candidates if candidate.label == "repair_required"
            ),
        },
        "training_datasets": asdict(dataset_export),
        "inference_router": {
            name: asdict(decision)
            for name, decision in router_results.items()
        },
        "bisect": asdict(bisect_result),
        "local_model_document": asdict(model_probe_result),
        "local_model_research": asdict(model_research_result),
        "document_tripwire_training": asdict(tripwire_result),
    }


def main() -> None:
    print(json.dumps(run_all(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
