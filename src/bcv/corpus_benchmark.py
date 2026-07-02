from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from bcv.corpus import DocumentCase, sample_corpus
from bcv.markdown_agent import edit_markdown_text
from bcv.markdown_editor import MarkdownSection, parse_sections
from bcv.schema import Event, TestResult
from bcv.store import CognitiveStore


@dataclass(frozen=True)
class EditMetric:
    document: str
    editor: str
    step: int
    accepted: bool
    attempts: int
    expected_present: bool
    missing_invariants: tuple[str, ...]
    section_count_before: int
    section_count_after: int
    failure: str | None


@dataclass(frozen=True)
class CorpusBenchmarkSummary:
    editor: str
    documents: int
    edit_steps: int
    accepted_steps: int
    expected_hits: int
    blocked_steps: int
    invariant_losses: int
    section_count_drifts: int
    total_attempts: int


def run_corpus_benchmark(root: str | Path = ".bcv_runs/corpus_benchmark") -> dict[str, object]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    all_metrics: list[EditMetric] = []
    for case in sample_corpus():
        all_metrics.extend(_run_verified_case(case, root / "verified" / case.name))
        all_metrics.extend(_run_corrupt_baseline_case(case))

    summaries = {
        editor: _summarize(editor, [metric for metric in all_metrics if metric.editor == editor])
        for editor in ("verified_agent", "corrupt_rewrite_baseline")
    }
    _record_metrics(root, all_metrics, summaries)
    return {
        "metrics": [asdict(metric) for metric in all_metrics],
        "summaries": {name: asdict(summary) for name, summary in summaries.items()},
    }


def _run_verified_case(case: DocumentCase, run_root: Path) -> list[EditMetric]:
    document = case.path.read_text(encoding="utf-8")
    metrics: list[EditMetric] = []
    for index, edit in enumerate(case.edits, start=1):
        before = document
        result, updated = edit_markdown_text(
            document,
            edit.instruction,
            run_root=run_root / f"step_{index}",
            max_attempts=3,
            required_phrases=(edit.expected_phrase,),
        )
        if result.accepted:
            document = updated
        metrics.append(
            _metric(
                case=case,
                editor="verified_agent",
                step=index,
                before=before,
                after=document,
                accepted=result.accepted,
                attempts=result.attempts,
                expected_phrase=edit.expected_phrase,
                failure=result.failure,
            )
        )
    output_path = run_root / "final.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")
    return metrics


def _run_corrupt_baseline_case(case: DocumentCase) -> list[EditMetric]:
    document = case.path.read_text(encoding="utf-8")
    metrics: list[EditMetric] = []
    for index, edit in enumerate(case.edits, start=1):
        before = document
        document = _corrupt_rewrite(document, edit.expected_phrase, case.invariants, index)
        metrics.append(
            _metric(
                case=case,
                editor="corrupt_rewrite_baseline",
                step=index,
                before=before,
                after=document,
                accepted=True,
                attempts=1,
                expected_phrase=edit.expected_phrase,
                failure=None,
            )
        )
    return metrics


def _corrupt_rewrite(document: str, expected_phrase: str, invariants: tuple[str, ...], step: int) -> str:
    sections = parse_sections(document)
    target = _first_content_section(sections)
    rewritten = document.replace(target.text, target.text.rstrip() + f"\n\nAdded: {expected_phrase}.\n", 1)
    if invariants:
        victim = invariants[(step - 1) % len(invariants)]
        rewritten = rewritten.replace(victim, "", 1)
    if step == 2 and "## Citation" in rewritten:
        rewritten = rewritten.replace("## Citation", "## Sources", 1)
    return rewritten


def _first_content_section(sections: list[MarkdownSection]) -> MarkdownSection:
    for section in sections:
        if section.heading != "ROOT":
            return section
    return sections[0]


def _metric(
    case: DocumentCase,
    editor: str,
    step: int,
    before: str,
    after: str,
    accepted: bool,
    attempts: int,
    expected_phrase: str,
    failure: str | None,
) -> EditMetric:
    before_sections = parse_sections(before)
    after_sections = parse_sections(after)
    missing = tuple(token for token in case.invariants if token not in after)
    return EditMetric(
        document=case.name,
        editor=editor,
        step=step,
        accepted=accepted,
        attempts=attempts,
        expected_present=expected_phrase in after,
        missing_invariants=missing,
        section_count_before=len(before_sections),
        section_count_after=len(after_sections),
        failure=failure,
    )


def _summarize(editor: str, metrics: list[EditMetric]) -> CorpusBenchmarkSummary:
    return CorpusBenchmarkSummary(
        editor=editor,
        documents=len({metric.document for metric in metrics}),
        edit_steps=len(metrics),
        accepted_steps=sum(1 for metric in metrics if metric.accepted),
        expected_hits=sum(1 for metric in metrics if metric.expected_present),
        blocked_steps=sum(1 for metric in metrics if not metric.accepted),
        invariant_losses=sum(len(metric.missing_invariants) for metric in metrics),
        section_count_drifts=sum(
            1 for metric in metrics if metric.section_count_before != metric.section_count_after
        ),
        total_attempts=sum(metric.attempts for metric in metrics),
    )


def _record_metrics(
    root: Path,
    metrics: list[EditMetric],
    summaries: dict[str, CorpusBenchmarkSummary],
) -> None:
    (root / "metrics.json").write_text(
        json.dumps([asdict(metric) for metric in metrics], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (root / "summary.json").write_text(
        json.dumps({name: asdict(summary) for name, summary in summaries.items()}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    store = CognitiveStore(root / "ledger")
    store.init()
    branch = "experiment/corpus-benchmark"
    if not store.branch_exists(branch):
        store.create_branch(branch, from_branch="main")
    for name, summary in summaries.items():
        store.commit(
            branch,
            f"record corpus benchmark summary: {name}",
            [
                Event(
                    event_type="verifier_result",
                    actor="verifier",
                    message=f"{name}: {summary.expected_hits}/{summary.edit_steps} expected edits, {summary.invariant_losses} invariant losses",
                    output_refs=(f"corpus_summary:{name}",),
                    tests=(
                        TestResult(
                            "corpus_benchmark_summary",
                            "pass" if summary.invariant_losses == 0 else "fail",
                            json.dumps(asdict(summary), sort_keys=True),
                        ),
                    ),
                )
            ],
        )


def main() -> None:
    print(json.dumps(run_corpus_benchmark(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
