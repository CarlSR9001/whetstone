"""Live forge-library refresh invariants for the public report-card hatchery."""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import bcv.ephemeral as ephemeral
from bcv.ephemeral import Hatchery, TierError


def _wait_for_revision(hatchery: Hatchery, revision: str) -> None:
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        with hatchery.lock:
            ready = hatchery.ready
            loaded = [
                getattr(item, "revision", None)
                for items in hatchery.master.values()
                for item in items
            ]
            thread = hatchery._thread
        if ready and loaded == [revision]:
            return
        if thread is not None:
            thread.join(timeout=0.1)
    raise AssertionError(f"hatchery did not load revision {revision!r}")


def test_synced_library_refreshes_without_restart_and_pins_live_sessions(tmp_path, monkeypatch):
    library = tmp_path / "adversary_library.jsonl"
    first = json.dumps({"graph_id": "first", "domain": "coloring"}) + "\n"
    second = first + json.dumps({"graph_id": "second", "domain": "coloring"}) + "\n"
    library.write_text(first, encoding="utf-8")
    builds: list[tuple[str, str]] = []

    def fake_pool(domain, stress_ns, samples, seed, library_path):
        revision = Path(library_path).read_text(encoding="utf-8")
        builds.append(("pool", revision))
        return [f"pool:{revision}"]

    def fake_mint(domain, buffers, max_items, max_n, stress_ns, seed, library_path=None):
        revision = Path(library_path).read_text(encoding="utf-8")
        builds.append(("mint", revision))
        return [SimpleNamespace(status="ready", revision=revision)]

    monkeypatch.setattr("bcv.refinery._stress_pool", fake_pool)
    monkeypatch.setattr("bcv.examiner.mint_repair_items", fake_mint)
    monkeypatch.setattr("bcv.examiner.repair_item_prompt", lambda item: item.revision)

    hatchery = Hatchery(
        domains=("coloring",),
        items_per_session=1,
        master_items_per_domain=1,
        library_path=str(library),
    )
    hatchery._warm()
    assert hatchery.ready
    original_pools = hatchery.pools

    client_ip = "198.51.100.247"
    issued = hatchery.start_session(client_ip)
    live_session = hatchery.sessions[issued["session_id"]]
    assert live_session["pools"] is original_pools

    library.write_text(second, encoding="utf-8")
    status = hatchery.status()  # health polling notices the atomic publication
    assert status["forge_library"]["entries"] == 2
    _wait_for_revision(hatchery, second)

    assert hatchery.pools is not original_pools
    assert hatchery.pools["coloring"] == [f"pool:{second}"]
    assert live_session["pools"] is original_pools
    assert builds == [
        ("pool", first),
        ("mint", first),
        ("pool", second),
        ("mint", second),
    ]

    # Repeated health checks do not re-warm an already loaded publication.
    hatchery.status()
    assert len(builds) == 4


def test_refresh_starting_during_rate_limit_check_cannot_issue_stale_session(monkeypatch):
    hatchery = Hatchery(domains=("coloring",), items_per_session=1)
    hatchery._has_started = True
    hatchery.ready = True
    hatchery.master = {"coloring": [SimpleNamespace(status="ready")]}
    hatchery.pools = {"coloring": ["old-pool"]}

    def begin_refresh(_client_ip):
        with hatchery.lock:
            hatchery.ready = False
        return True

    monkeypatch.setattr(ephemeral.REPORT_START_LIMIT, "allow", begin_refresh)
    with pytest.raises(TierError, match="still warming"):
        hatchery.start_session("198.51.100.248")
    assert hatchery.sessions == {}
