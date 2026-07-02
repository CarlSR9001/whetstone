from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from bcv.benchmark import SAMPLE_DOCUMENT, evaluate_candidate
from bcv.local_model import LocalModelClient, auto_local_client
from bcv.markdown_editor import MarkdownPatch, PatchError, PatchOperation, apply_markdown_patch
from bcv.research import Claim, ClaimGraph, ResearchError, Source
from bcv.schema import Event, TestResult
from bcv.store import CognitiveStore


@dataclass(frozen=True)
class ModelDocumentProbeResult:
    backend: str
    model: str
    accepted: bool
    mode: str
    failure: str | None
    candidate_text: str


@dataclass(frozen=True)
class ModelResearchProbeResult:
    backend: str
    model: str
    accepted: bool
    unsupported_claims: tuple[str, ...]
    contradictions: tuple[tuple[str, str], ...]
    failure: str | None


def run_model_document_probe(
    root: str | Path,
    client: LocalModelClient | None = None,
) -> ModelDocumentProbeResult:
    client = client or auto_local_client()
    prompt = _document_patch_prompt()
    data = client.generate_json(prompt, temperature=0.0)
    result = _evaluate_model_patch(client, data)
    _record_model_probe(root, result, data)
    return result


def run_model_research_probe(
    root: str | Path,
    client: LocalModelClient | None = None,
) -> ModelResearchProbeResult:
    client = client or auto_local_client()
    data = client.generate_json(_research_prompt(), temperature=0.0)
    result = _evaluate_model_research(client, data)
    _record_research_probe(root, result, data)
    return result


def _document_patch_prompt() -> str:
    return f"""/no_think
You are editing a Markdown contract through a patch-only conservation-law editor.

Return exactly one JSON object with this schema:
{{
  "mode": "patch",
  "operations": [
    {{
      "target_heading": "Scope",
      "find": "Northstar Labs will deliver the analytics dashboard described in Exhibit A.",
      "replace": "Northstar Labs will deliver the analytics dashboard and a weekly deployment summary described in Exhibit A."
    }}
  ]
}}

Do not rewrite the whole document. Do not change payment terms, dates, party names, section headings, invoice IDs, or citations.

Document:
{SAMPLE_DOCUMENT}
"""


def _research_prompt() -> str:
    return """/no_think
You are a claim/evidence extractor for a verifier-backed research synthesizer.

Return exactly one JSON object with this schema:
{
  "sources": [
    {"source_id": "source:a", "title": "Deployment Memo A", "text": "..."},
    {"source_id": "source:b", "title": "Deployment Memo B", "text": "..."}
  ],
  "claims": [
    {"claim_id": "claim:rollout-june", "text": "...", "source_ids": ["source:a"], "contradicts": ["claim:rollout-july"]},
    {"claim_id": "claim:rollout-july", "text": "...", "source_ids": ["source:b"], "contradicts": ["claim:rollout-june"]}
  ],
  "final_claim_ids": ["claim:rollout-june", "claim:rollout-july"]
}

Every factual claim must have source_ids. If two source-backed claims cannot both be true, fill contradicts.

Source A:
The rollout completed on 2026-06-20 with no payment change.

Source B:
The rollout moved to 2026-07-15 after payment terms changed.
"""


def _evaluate_model_patch(client: LocalModelClient, data: dict[str, Any]) -> ModelDocumentProbeResult:
    mode = str(data.get("mode", "missing"))
    try:
        if mode != "patch":
            raise PatchError(f"model returned unsupported mode: {mode}")
        operations = tuple(
            PatchOperation(
                target_heading=str(item["target_heading"]),
                find=str(item["find"]),
                replace=str(item["replace"]),
            )
            for item in data.get("operations", [])
        )
        updated = apply_markdown_patch(SAMPLE_DOCUMENT, MarkdownPatch(operations=operations))
        benchmark = evaluate_candidate("model_patch", SAMPLE_DOCUMENT, updated, {"Scope"})
        accepted = benchmark.accepted
        failure = benchmark.failure
        candidate_text = updated
    except (KeyError, TypeError, PatchError) as exc:
        accepted = False
        failure = str(exc)
        candidate_text = json.dumps(data, sort_keys=True)

    return ModelDocumentProbeResult(
        backend=client.backend,
        model=client.model,
        accepted=accepted,
        mode=mode,
        failure=failure,
        candidate_text=candidate_text,
    )


def _evaluate_model_research(client: LocalModelClient, data: dict[str, Any]) -> ModelResearchProbeResult:
    try:
        graph = ClaimGraph()
        for item in data.get("sources", []):
            graph.add_source(
                Source(
                    source_id=str(item["source_id"]),
                    title=str(item["title"]),
                    text=str(item["text"]),
                )
            )
        for item in data.get("claims", []):
            graph.add_claim(
                Claim(
                    claim_id=str(item["claim_id"]),
                    text=str(item["text"]),
                    source_ids=tuple(item.get("source_ids", ())),
                    contradicts=tuple(item.get("contradicts", ())),
                )
            )
        final_claim_ids = tuple(data.get("final_claim_ids", tuple(graph.claims)))
        verification = graph.verify(final_claim_ids)
        accepted = not verification.unsupported_claims and bool(verification.contradictions)
        return ModelResearchProbeResult(
            backend=client.backend,
            model=client.model,
            accepted=accepted,
            unsupported_claims=verification.unsupported_claims,
            contradictions=verification.contradictions,
            failure=None if accepted else "model did not produce sourced contradictory claim graph",
        )
    except (KeyError, TypeError, ResearchError) as exc:
        return ModelResearchProbeResult(
            backend=client.backend,
            model=client.model,
            accepted=False,
            unsupported_claims=(),
            contradictions=(),
            failure=str(exc),
        )


def _record_model_probe(
    root: str | Path,
    result: ModelDocumentProbeResult,
    raw_model_json: dict[str, Any],
) -> None:
    store = CognitiveStore(root)
    store.init()
    branch = "experiment/local-model-document"
    if not store.branch_exists(branch):
        store.create_branch(branch, from_branch="main")
    store.commit(
        branch,
        f"record local model document probe: {result.model}",
        [
            Event(
                event_type="verifier_result",
                actor="verifier",
                message=result.failure or "model patch accepted",
                input_refs=(f"model:{result.backend}:{result.model}",),
                output_refs=("candidate:local-model-document-patch",),
                evidence_refs=("prompt:document_patch",),
                tests=(
                    TestResult(
                        "local_model_document_patch",
                        "pass" if result.accepted else "fail",
                        json.dumps(
                            {
                                "result": asdict(result),
                                "raw_model_json": raw_model_json,
                            },
                            sort_keys=True,
                        ),
                    ),
                ),
            )
        ],
    )


def _record_research_probe(
    root: str | Path,
    result: ModelResearchProbeResult,
    raw_model_json: dict[str, Any],
) -> None:
    store = CognitiveStore(root)
    store.init()
    branch = "experiment/local-model-research"
    if not store.branch_exists(branch):
        store.create_branch(branch, from_branch="main")
    store.commit(
        branch,
        f"record local model research probe: {result.model}",
        [
            Event(
                event_type="verifier_result",
                actor="verifier",
                message=result.failure or "model surfaced contradiction graph",
                input_refs=(f"model:{result.backend}:{result.model}",),
                output_refs=("candidate:local-model-research-graph",),
                evidence_refs=("prompt:research_contradiction",),
                tests=(
                    TestResult(
                        "local_model_research_contradiction",
                        "pass" if result.accepted else "fail",
                        json.dumps(
                            {
                                "result": asdict(result),
                                "raw_model_json": raw_model_json,
                            },
                            sort_keys=True,
                        ),
                    ),
                ),
            )
        ],
    )


def main() -> None:
    root = Path(".bcv_runs")
    result = {
        "document": asdict(run_model_document_probe(root / "local_model_document")),
        "research": asdict(run_model_research_probe(root / "local_model_research")),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
