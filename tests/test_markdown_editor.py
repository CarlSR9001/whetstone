from __future__ import annotations

import pytest

from bcv.benchmark import run_document_corruption_benchmark
from bcv.markdown_editor import (
    MarkdownPatch,
    PatchError,
    PatchOperation,
    apply_markdown_patch,
    verify_conservation,
)


DOC = """# Research Note

## Claim

Model Alpha preserved citation [SRC-1] and value 42.

## Background

The experiment started on 2026-07-01 with Northstar Labs.
"""


def test_patch_only_edit_preserves_untouched_sections():
    updated = apply_markdown_patch(
        DOC,
        MarkdownPatch(
            operations=(
                PatchOperation(
                    target_heading="Claim",
                    find="Model Alpha preserved citation [SRC-1] and value 42.",
                    replace="Model Alpha preserved citation [SRC-1] and value 42 after the verifier passed.",
                ),
            )
        ),
    )

    assert "after the verifier passed" in updated
    verify_conservation(DOC, updated, {"Claim"})


def test_patch_rejects_removed_protected_token_in_target_section():
    with pytest.raises(PatchError, match="protected token removed"):
        apply_markdown_patch(
            DOC,
            MarkdownPatch(
                operations=(
                    PatchOperation(
                        target_heading="Claim",
                        find="Model Alpha preserved citation [SRC-1] and value 42.",
                        replace="Model Alpha preserved citation and value.",
                    ),
                )
            ),
        )


def test_patch_rejects_removed_exhibit_label():
    doc = """# Agreement

## Scope

Northstar Labs will deliver the dashboard described in Exhibit A.
"""
    with pytest.raises(PatchError, match="Exhibit A"):
        apply_markdown_patch(
            doc,
            MarkdownPatch(
                operations=(
                    PatchOperation(
                        target_heading="Scope",
                        find="Northstar Labs will deliver the dashboard described in Exhibit A.",
                        replace="Northstar Labs will deliver the dashboard.",
                    ),
                )
            ),
        )


def test_verifier_rejects_untargeted_section_drift():
    corrupted = DOC.replace("2026-07-01", "2026-07-02")
    with pytest.raises(PatchError, match="untargeted section changed"):
        verify_conservation(DOC, corrupted, {"Claim"})


def test_benchmark_accepts_clean_patch_and_rejects_corrupt_rewrite():
    results = {result.candidate: result for result in run_document_corruption_benchmark()}

    assert results["patch_only_editor"].accepted is True
    assert results["corrupt_full_rewrite"].accepted is False
    assert results["corrupt_full_rewrite"].number_drift == 1
    assert results["corrupt_full_rewrite"].section_drift == 1
