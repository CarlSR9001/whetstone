from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from scripts.build_release_sbom import build_document, main


def _wheel(path: Path) -> None:
    metadata = """Metadata-Version: 2.4
Name: branching-continual-verification
Version: 9.8.7
License-Expression: AGPL-3.0-or-later
Requires-Dist: mcp<3,>=2; extra == "agents"
Requires-Dist: python-chess>=1.999; extra == "engines"

"""
    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        archive.writestr(
            "branching_continual_verification-9.8.7.dist-info/METADATA",
            metadata,
        )
        archive.writestr("bcv/__init__.py", "")


def test_release_sbom_describes_exact_artifact_and_declared_dependencies(tmp_path):
    wheel = tmp_path / "whetstone.whl"
    _wheel(wheel)

    document = build_document(
        wheel,
        wheel,
        created="2026-08-08T12:00:00Z",
        source_commit="a" * 40,
    )

    assert document["spdxVersion"] == "SPDX-2.3"
    assert document["dataLicense"] == "CC0-1.0"
    package = document["packages"][0]
    assert package["versionInfo"] == "9.8.7"
    assert package["licenseDeclared"] == "AGPL-3.0-or-later"
    assert package["checksums"] == [{
        "algorithm": "SHA256",
        "checksumValue": hashlib.sha256(wheel.read_bytes()).hexdigest(),
    }]
    assert {entry["name"] for entry in document["packages"][1:]} == {
        "mcp",
        "python-chess",
    }
    assert len([row for row in document["relationships"] if row["relationshipType"] == "DEPENDS_ON"]) == 2


def test_release_sbom_cli_is_deterministic(tmp_path):
    wheel = tmp_path / "whetstone.whl"
    artifact = tmp_path / "source.tar.gz"
    output = tmp_path / "source.spdx.json"
    _wheel(wheel)
    artifact.write_bytes(b"immutable source archive")
    args = [
        "--artifact", str(artifact),
        "--metadata-wheel", str(wheel),
        "--created", "2026-08-08T12:00:00Z",
        "--source-commit", "b" * 40,
        "--output", str(output),
    ]

    assert main(args) == 0
    first = output.read_bytes()
    assert main(args) == 0
    assert output.read_bytes() == first
    assert json.loads(first)["packages"][0]["comment"].endswith("source.tar.gz")


def test_release_sbom_normalizes_git_commit_timestamp_to_spdx_utc(tmp_path):
    wheel = tmp_path / "whetstone.whl"
    _wheel(wheel)

    document = build_document(
        wheel,
        wheel,
        created="2026-08-05T21:34:58-05:00",
    )

    assert document["creationInfo"]["created"] == "2026-08-06T02:34:58Z"


def test_release_sbom_rejects_naive_creation_timestamp(tmp_path):
    wheel = tmp_path / "whetstone.whl"
    _wheel(wheel)

    with pytest.raises(ValueError, match="must include a UTC offset"):
        build_document(wheel, wheel, created="2026-08-08T12:00:00")
