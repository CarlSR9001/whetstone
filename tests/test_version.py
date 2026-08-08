from __future__ import annotations

from pathlib import Path

import pytest

import bcv._version as version
from bcv._version import ReleaseIdentityError, __version__, build_commit


def test_release_version_and_development_provenance(monkeypatch):
    monkeypatch.delenv("WHETSTONE_BUILD_COMMIT", raising=False)
    assert __version__ == "0.8.0"
    assert build_commit() == "development"


def test_release_provenance_accepts_only_full_commit(monkeypatch):
    commit = "0123456789abcdef0123456789abcdef01234567"
    monkeypatch.setenv("WHETSTONE_BUILD_COMMIT", commit.upper())
    assert build_commit() == commit

    monkeypatch.setenv("WHETSTONE_BUILD_COMMIT", "main")
    assert build_commit() == "development"


def test_release_provenance_fails_closed_on_archive_mismatch(monkeypatch):
    monkeypatch.setattr(version, "_ARCHIVE_COMMIT", "a" * 40)
    monkeypatch.setenv("WHETSTONE_BUILD_COMMIT", "b" * 40)
    with pytest.raises(ReleaseIdentityError):
        build_commit()


def test_inspector_action_does_not_interpolate_inputs_in_shell():
    action = Path(".github/actions/whetstone-inspector/action.yml").read_text(
        encoding="utf-8"
    )

    assert '"' + chr(36) + "{{ inputs." not in action
    for variable in (
        "WHETSTONE_EXAM",
        "WHETSTONE_EXPOSURE",
        "WHETSTONE_BASELINE",
        "WHETSTONE_CANDIDATE",
        "WHETSTONE_ALPHA",
        "WHETSTONE_MIN_GAINS",
        "WHETSTONE_MAX_REGRESSIONS",
    ):
        assert '"' + chr(36) + f'{variable}"' in action
