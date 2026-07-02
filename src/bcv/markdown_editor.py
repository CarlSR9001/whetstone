from __future__ import annotations

import re
from dataclasses import dataclass


class PatchError(ValueError):
    pass


@dataclass(frozen=True)
class MarkdownSection:
    heading: str
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class PatchOperation:
    target_heading: str
    find: str
    replace: str
    allow_token_changes: tuple[str, ...] = ()


@dataclass(frozen=True)
class MarkdownPatch:
    operations: tuple[PatchOperation, ...]
    reason: str = ""


PROTECTED_TOKEN_RE = re.compile(
    r"\[[A-Za-z0-9_.:-]+\]|\b\d+(?:[.,:/-]\d+)*\b|\b[A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*)+\b"
)


def apply_markdown_patch(document: str, patch: MarkdownPatch) -> str:
    sections = parse_sections(document)
    updated = document
    offset = 0
    changed_headings: set[str] = set()

    for op in patch.operations:
        section = _find_section(sections, op.target_heading)
        start = section.start + offset
        end = section.end + offset
        live_section_text = updated[start:end]
        count = live_section_text.count(op.find)
        if count != 1:
            raise PatchError(
                f"expected exactly one match in section {op.target_heading!r}, found {count}"
            )
        new_section_text = live_section_text.replace(op.find, op.replace, 1)
        _verify_protected_tokens(live_section_text, new_section_text, op)
        updated = updated[:start] + new_section_text + updated[end:]
        offset += len(new_section_text) - len(live_section_text)
        changed_headings.add(op.target_heading)

    verify_unchanged_sections(document, updated, changed_headings)
    return updated


def parse_sections(document: str) -> list[MarkdownSection]:
    matches = list(re.finditer(r"(?m)^#{1,6}\s+(.+?)\s*$", document))
    if not matches:
        return [MarkdownSection("ROOT", 0, len(document), document)]

    sections: list[MarkdownSection] = []
    if matches[0].start() > 0:
        sections.append(MarkdownSection("ROOT", 0, matches[0].start(), document[: matches[0].start()]))

    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(document)
        heading = match.group(1).strip()
        sections.append(MarkdownSection(heading, start, end, document[start:end]))
    return sections


def verify_unchanged_sections(
    original: str,
    updated: str,
    changed_headings: set[str],
) -> None:
    original_sections = {section.heading: section.text for section in parse_sections(original)}
    updated_sections = {section.heading: section.text for section in parse_sections(updated)}
    if original_sections.keys() != updated_sections.keys():
        missing = sorted(original_sections.keys() - updated_sections.keys())
        added = sorted(updated_sections.keys() - original_sections.keys())
        raise PatchError(f"section set changed; missing={missing}, added={added}")

    for heading, section_text in original_sections.items():
        if heading not in changed_headings and updated_sections[heading] != section_text:
            raise PatchError(f"untargeted section changed: {heading}")


def verify_conservation(
    original: str,
    updated: str,
    changed_headings: set[str],
) -> None:
    verify_unchanged_sections(original, updated, changed_headings)


def protected_tokens(text: str) -> set[str]:
    return {match.group(0) for match in PROTECTED_TOKEN_RE.finditer(text)}


def _verify_protected_tokens(before: str, after: str, op: PatchOperation) -> None:
    allowed = set(op.allow_token_changes)
    removed = protected_tokens(before) - protected_tokens(after) - allowed
    if removed:
        raise PatchError(
            f"protected token removed from section {op.target_heading!r}: {sorted(removed)}"
        )


def _find_section(sections: list[MarkdownSection], heading: str) -> MarkdownSection:
    matches = [section for section in sections if section.heading == heading]
    if len(matches) != 1:
        raise PatchError(f"expected exactly one section {heading!r}, found {len(matches)}")
    return matches[0]
