from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from bcv.local_model import auto_local_client
from bcv.markdown_editor import MarkdownPatch, PatchError, PatchOperation, apply_markdown_patch
from bcv.schema import Event, TestResult
from bcv.store import CognitiveStore


class JsonModel(Protocol):
    backend: str
    model: str

    def generate_json(self, prompt: str, temperature: float = 0.0) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class MarkdownEditResult:
    accepted: bool
    attempts: int
    output_path: str | None
    failure: str | None
    changed_headings: tuple[str, ...]
    model_backend: str
    model_name: str


def edit_markdown_file(
    input_path: str | Path,
    output_path: str | Path,
    instruction: str,
    run_root: str | Path = ".bcv_runs/markdown_agent",
    client: JsonModel | None = None,
    max_attempts: int = 3,
    required_phrases: tuple[str, ...] = (),
) -> MarkdownEditResult:
    input_path = Path(input_path)
    output_path = Path(output_path)
    document = input_path.read_text(encoding="utf-8")
    result, updated = edit_markdown_text(
        document=document,
        instruction=instruction,
        run_root=run_root,
        client=client,
        max_attempts=max_attempts,
        required_phrases=required_phrases,
    )
    if result.accepted:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(updated, encoding="utf-8")
        result = MarkdownEditResult(
            accepted=result.accepted,
            attempts=result.attempts,
            output_path=str(output_path),
            failure=result.failure,
            changed_headings=result.changed_headings,
            model_backend=result.model_backend,
            model_name=result.model_name,
        )
    return result


def edit_markdown_text(
    document: str,
    instruction: str,
    run_root: str | Path = ".bcv_runs/markdown_agent",
    client: JsonModel | None = None,
    max_attempts: int = 3,
    required_phrases: tuple[str, ...] = (),
) -> tuple[MarkdownEditResult, str]:
    client = client or auto_local_client()
    store = CognitiveStore(run_root)
    store.init()
    branch = "experiment/markdown-agent"
    if not store.branch_exists(branch):
        store.create_branch(branch, from_branch="main")

    failure: str | None = None
    raw_model_json: dict[str, Any] = {}
    for attempt in range(1, max_attempts + 1):
        prompt = _patch_prompt(document, instruction, failure, raw_model_json, required_phrases)
        raw_model_json = client.generate_json(prompt, temperature=0.0)
        try:
            patch = _patch_from_json(raw_model_json)
            updated = apply_markdown_patch(document, patch)
            missing_required = tuple(phrase for phrase in required_phrases if phrase not in updated)
            if missing_required:
                raise PatchError(f"required phrase missing after patch: {missing_required}")
            changed_headings = tuple(dict.fromkeys(op.target_heading for op in patch.operations))
            result = MarkdownEditResult(
                accepted=True,
                attempts=attempt,
                output_path=None,
                failure=None,
                changed_headings=changed_headings,
                model_backend=client.backend,
                model_name=client.model,
            )
            _record_attempt(store, branch, instruction, result, raw_model_json)
            return result, updated
        except (KeyError, TypeError, PatchError, ValueError) as exc:
            failure = str(exc)
            result = MarkdownEditResult(
                accepted=False,
                attempts=attempt,
                output_path=None,
                failure=failure,
                changed_headings=(),
                model_backend=client.backend,
                model_name=client.model,
            )
            _record_attempt(store, branch, instruction, result, raw_model_json)

    return result, document


def _patch_prompt(
    document: str,
    instruction: str,
    previous_failure: str | None,
    previous_json: dict[str, Any],
    required_phrases: tuple[str, ...],
) -> str:
    repair_block = ""
    if previous_failure:
        repair_block = f"""
Previous patch failed verification:
{previous_failure}

Previous JSON:
{json.dumps(previous_json, sort_keys=True)}

Return a corrected patch. Do not repeat the same failing shape.
"""

    return f"""/no_think
You are editing a Markdown document through a conservation-law patch editor.

Instruction:
{instruction}

Return exactly one JSON object:
{{
  "operations": [
    {{
      "target_heading": "exact section heading without # characters",
      "find": "exact contiguous text copied from that section",
      "replace": "replacement text"
    }}
  ]
}}

Rules:
- Use exact substrings from the document in "find".
- Only target sections required by the instruction.
- Do not rewrite the whole document.
- Do not change party names, dates, invoice IDs, payment terms, citations, or section headings unless the instruction explicitly asks.
- Keep operations minimal.
- The final document must include these exact required phrases: {json.dumps(required_phrases)}.
{repair_block}

Document:
{document}
"""


def _patch_from_json(data: dict[str, Any]) -> MarkdownPatch:
    operations = data.get("operations")
    if not isinstance(operations, list) or not operations:
        raise ValueError("model returned no patch operations")
    return MarkdownPatch(
        operations=tuple(
            PatchOperation(
                target_heading=str(item["target_heading"]),
                find=str(item["find"]),
                replace=str(item["replace"]),
            )
            for item in operations
        ),
        reason=str(data.get("reason", "")),
    )


def _record_attempt(
    store: CognitiveStore,
    branch: str,
    instruction: str,
    result: MarkdownEditResult,
    raw_model_json: dict[str, Any],
) -> None:
    store.commit(
        branch,
        f"markdown agent attempt {result.attempts}",
        [
            Event(
                event_type="verifier_result",
                actor="verifier",
                message=result.failure or "markdown patch accepted",
                input_refs=(f"model:{result.model_backend}:{result.model_name}",),
                output_refs=("artifact:markdown-edit",),
                evidence_refs=("instruction:markdown-edit",),
                tests=(
                    TestResult(
                        "markdown_agent_patch",
                        "pass" if result.accepted else "fail",
                        json.dumps(
                            {
                                "instruction": instruction,
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
    parser = argparse.ArgumentParser(description="Edit Markdown through local-model patches and verifier hooks.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--run-root", default=".bcv_runs/markdown_agent")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--require-phrase", action="append", default=[])
    args = parser.parse_args()

    result = edit_markdown_file(
        input_path=args.input,
        output_path=args.output,
        instruction=args.instruction,
        run_root=args.run_root,
        max_attempts=args.max_attempts,
        required_phrases=tuple(args.require_phrase),
    )
    print(json.dumps(asdict(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
