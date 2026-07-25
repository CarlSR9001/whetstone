"""Single source of truth for the public Whetstone release identity."""

from __future__ import annotations

import os
import re

__version__ = "0.5.0"

# ``git archive`` expands this value when ``export-subst`` is set for this
# file. Wheel/editable installs fall back to an explicit deployment variable.
_ARCHIVE_COMMIT = "$Format:%H$"
_FULL_SHA256 = re.compile(r"^[0-9a-f]{40}$")


class ReleaseIdentityError(RuntimeError):
    """The configured commit disagrees with the immutable source archive."""


def build_commit() -> str:
    """Return the exact source commit, or ``development`` when unavailable."""
    configured = os.environ.get("WHETSTONE_BUILD_COMMIT", "").strip().lower()
    archived = _ARCHIVE_COMMIT.strip().lower()
    configured_valid = bool(_FULL_SHA256.fullmatch(configured))
    archived_valid = bool(_FULL_SHA256.fullmatch(archived))

    if configured_valid and archived_valid and configured != archived:
        raise ReleaseIdentityError(
            f"configured release {configured} does not match archived source {archived}"
        )
    if archived_valid:
        return archived
    if configured_valid:
        return configured

    return "development"
