from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PlannedEdit:
    instruction: str
    target_heading: str
    expected_phrase: str


@dataclass(frozen=True)
class DocumentCase:
    name: str
    path: Path
    edits: tuple[PlannedEdit, ...]
    invariants: tuple[str, ...]


def sample_corpus(root: str | Path = "sample_docs") -> tuple[DocumentCase, ...]:
    root = Path(root)
    return (
        DocumentCase(
            name="vendor_agreement",
            path=root / "vendor_agreement.md",
            edits=(
                PlannedEdit(
                    instruction="In the Scope section, add that Northstar Labs will provide a weekly deployment summary. Do not change payment terms, dates, parties, invoice IDs, headings, Exhibit references, or citations.",
                    target_heading="Scope",
                    expected_phrase="weekly deployment summary",
                ),
                PlannedEdit(
                    instruction="In the Parties section, add the literal word customer by clarifying that Meridian Health is the customer. Do not change payment terms, dates, invoice IDs, headings, Exhibit references, or citations.",
                    target_heading="Parties",
                    expected_phrase="customer",
                ),
                PlannedEdit(
                    instruction="In the Citation section, add that the DPA citation must remain attached to privacy obligations. Do not remove or rename [DPA-17].",
                    target_heading="Citation",
                    expected_phrase="remain attached",
                ),
            ),
            invariants=(
                "Northstar Labs",
                "Meridian Health",
                "INV-2049",
                "30 days",
                "2026-07-01",
                "2026-12-31",
                "Exhibit A",
                "[DPA-17]",
            ),
        ),
        DocumentCase(
            name="research_memo",
            path=root / "research_memo.md",
            edits=(
                PlannedEdit(
                    instruction="In the Result section, add that the next benchmark must report accidental deletion, number drift, section drift, and retry count.",
                    target_heading="Result",
                    expected_phrase="retry count",
                ),
                PlannedEdit(
                    instruction="In the Evidence section, add that accepted patches are recorded in the JSONL branch ledger.",
                    target_heading="Evidence",
                    expected_phrase="JSONL branch ledger",
                ),
                PlannedEdit(
                    instruction="In the Constraint section, add that unsupported claims should be routed to repair instead of merged.",
                    target_heading="Constraint",
                    expected_phrase="routed to repair",
                ),
            ),
            invariants=(
                "Branching Continual Verification",
                "qwen3:8b",
                "Ollama",
                "[BCV-01]",
            ),
        ),
        DocumentCase(
            name="implementation_plan",
            path=root / "implementation_plan.md",
            edits=(
                PlannedEdit(
                    instruction="In the Next Milestone section, add that the benchmark should compare verified editing against a corrupt rewrite baseline.",
                    target_heading="Next Milestone",
                    expected_phrase="corrupt rewrite baseline",
                ),
                PlannedEdit(
                    instruction="In the Current State section, add that the local Markdown agent writes only verifier-accepted patches.",
                    target_heading="Current State",
                    expected_phrase="verifier-accepted patches",
                ),
                PlannedEdit(
                    instruction="In the Guardrails section, add that failed verifier attempts must be kept in the ledger for repair training.",
                    target_heading="Guardrails",
                    expected_phrase="repair training",
                ),
            ),
            invariants=(
                "NVIDIA GeForce RTX 5060",
                "8 GB VRAM",
                "qwen3:8b",
                "JSONL branch ledger",
                "LoRA",
            ),
        ),
    )
