"""Build a deterministic SPDX 2.3 SBOM for one Whetstone release artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from email import policy
from email.parser import BytesParser
from pathlib import Path
from urllib.parse import quote
from zipfile import ZipFile


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metadata(wheel: Path) -> tuple[str, str, str, list[str]]:
    with ZipFile(wheel) as archive:
        names = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        if len(names) != 1:
            raise ValueError(f"expected one wheel METADATA file, found {len(names)}")
        message = BytesParser(policy=policy.default).parsebytes(archive.read(names[0]))
    name = message.get("Name")
    version = message.get("Version")
    if not name or not version:
        raise ValueError("wheel METADATA is missing Name or Version")
    license_expression = message.get("License-Expression") or "NOASSERTION"
    return name, version, license_expression, list(message.get_all("Requires-Dist", []))


def _spdx_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9.-]+", "-", value).strip("-.")
    return f"SPDXRef-{safe or 'Dependency'}"


def _dependency_name(requirement: str) -> str:
    match = re.match(r"\s*([A-Za-z0-9_.-]+)", requirement)
    if not match:
        raise ValueError(f"cannot parse dependency name from {requirement!r}")
    return match.group(1)


def _spdx_timestamp(value: str) -> str:
    candidate = value.strip()
    if candidate.endswith("Z"):
        candidate = candidate[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise ValueError(f"invalid ISO 8601 creation timestamp: {value!r}") from exc
    if parsed.tzinfo is None:
        raise ValueError("creation timestamp must include a UTC offset")
    return parsed.astimezone(timezone.utc).replace(microsecond=0).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def build_document(
    artifact: Path,
    metadata_wheel: Path,
    *,
    created: str,
    source_commit: str | None = None,
) -> dict:
    name, version, license_expression, requirements = _metadata(metadata_wheel)
    created = _spdx_timestamp(created)
    artifact_hash = _sha256(artifact)
    package_id = _spdx_id(f"Package-{name}")
    packages = [
        {
            "name": name,
            "SPDXID": package_id,
            "versionInfo": version,
            "downloadLocation": "NOASSERTION",
            "filesAnalyzed": False,
            "checksums": [{"algorithm": "SHA256", "checksumValue": artifact_hash}],
            "licenseConcluded": license_expression,
            "licenseDeclared": license_expression,
            "copyrightText": "NOASSERTION",
            "comment": f"Release artifact: {artifact.name}",
            "externalRefs": [
                {
                    "referenceCategory": "PACKAGE-MANAGER",
                    "referenceType": "purl",
                    "referenceLocator": f"pkg:pypi/{quote(name)}@{quote(version)}",
                }
            ],
        }
    ]
    relationships = [
        {
            "spdxElementId": "SPDXRef-DOCUMENT",
            "relationshipType": "DESCRIBES",
            "relatedSpdxElement": package_id,
        }
    ]
    seen: set[str] = set()
    for requirement in sorted(requirements, key=str.casefold):
        dependency = _dependency_name(requirement)
        normalized = dependency.lower().replace("_", "-")
        if normalized in seen:
            continue
        seen.add(normalized)
        dependency_id = _spdx_id(f"Dependency-{normalized}")
        packages.append(
            {
                "name": dependency,
                "SPDXID": dependency_id,
                "downloadLocation": "NOASSERTION",
                "filesAnalyzed": False,
                "licenseConcluded": "NOASSERTION",
                "licenseDeclared": "NOASSERTION",
                "copyrightText": "NOASSERTION",
                "comment": f"Declared requirement: {requirement}",
            }
        )
        relationships.append(
            {
                "spdxElementId": package_id,
                "relationshipType": "DEPENDS_ON",
                "relatedSpdxElement": dependency_id,
            }
        )
    if source_commit:
        packages[0]["externalRefs"].append(
            {
                "referenceCategory": "OTHER",
                "referenceType": "vcs",
                "referenceLocator": (
                    "git+https://github.com/CarlSR9001/whetstone.git@"
                    f"{source_commit}"
                ),
            }
        )
    return {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": f"{name}-{version}-{artifact.name}",
        "documentNamespace": (
            "https://whetstone.cyberelf.link/spdx/"
            f"{quote(version)}/{artifact_hash}"
        ),
        "creationInfo": {
            "created": created,
            "creators": ["Tool: Whetstone deterministic release SBOM builder"],
        },
        "packages": packages,
        "relationships": relationships,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--metadata-wheel", type=Path, required=True)
    parser.add_argument("--created", required=True, help="ISO 8601 timestamp with UTC offset")
    parser.add_argument("--source-commit")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    document = build_document(
        args.artifact,
        args.metadata_wheel,
        created=args.created,
        source_commit=args.source_commit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
