"""The forge controller's escalation policy, dedupe, sync, and the hatchery's
library pass-through (exam-hardening loop)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import bcv.forge_controller as fc
from bcv.forge_controller import LADDER, PATIENCE, Controller


class ScriptedRunner:
    """Feeds predetermined (rc, additions) outcomes; records calls."""

    def __init__(self, controller_ref: dict, outcomes: list[tuple[int, int]]):
        self.controller_ref = controller_ref
        self.outcomes = outcomes
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, domain: str, config: dict) -> tuple[int, float]:
        rc, additions = self.outcomes.pop(0)
        self.calls.append((domain, config))
        controller = self.controller_ref["c"]
        if additions:
            with open(controller.library, "a", encoding="utf-8") as handle:
                for index in range(additions):
                    handle.write(json.dumps({
                        "graph_id": f"g{len(self.calls)}_{index}", "domain": domain,
                    }) + "\n")
        return rc, 0.1


def make(tmp_path, outcomes, domains=("coloring",), sync_dir=None):
    ref: dict = {}
    runner = ScriptedRunner(ref, outcomes)
    controller = Controller(tmp_path, domains=domains, runner=runner, sleeper=lambda _s: None, sync_dir=sync_dir)
    ref["c"] = controller
    return controller, runner


def test_release_status_is_atomic_and_machine_readable(tmp_path):
    controller, _ = make(tmp_path, [])
    controller.write_release_status()
    status = json.loads((tmp_path / "forge_release_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "running"
    assert status["version"] == "0.5.1"
    assert status["build_commit"] == "development"
    assert status["library_sync"] == "not_checked"
    assert status["pid"] > 0
    assert not (tmp_path / "forge_release_status.tmp").exists()


def test_zero_streak_escalates_and_success_resets(tmp_path):
    # rung starts at 1; PATIENCE zeros -> rung 2; a find resets the streak.
    controller, runner = make(tmp_path, [(0, 0)] * PATIENCE + [(0, 5), (0, 0)])
    for _ in range(PATIENCE):
        controller.run_cycle("coloring")
    assert controller.state.domain("coloring").rung == 2
    controller.run_cycle("coloring")  # finds 5
    state = controller.state.domain("coloring")
    assert state.zeros_at_rung == 0 and state.finds == 5 and not state.exhausted
    controller.run_cycle("coloring")
    assert controller.state.domain("coloring").zeros_at_rung == 1
    # escalated rung's config was actually used by the runner
    assert runner.calls[PATIENCE][1] == LADDER[2]


def test_top_rung_exhaustion_then_sentinel_revival(tmp_path):
    zeros_to_exhaust = (len(LADDER) - 1) * PATIENCE  # from rung 1 to top, then out
    controller, runner = make(tmp_path, [(0, 0)] * zeros_to_exhaust + [(0, 0), (0, 3)])
    for _ in range(zeros_to_exhaust):
        controller.run_cycle("coloring")
    state = controller.state.domain("coloring")
    assert state.exhausted and "top rung" in state.exhausted_reason
    assert controller.all_exhausted()
    # Sentinel probes run at the top rung; a find revives the domain.
    controller.run_cycle("coloring")
    assert runner.calls[-1][1] == LADDER[-1]
    controller.run_cycle("coloring")
    assert not controller.state.domain("coloring").exhausted
    assert not controller.all_exhausted()


def test_timeout_caps_the_ladder_instead_of_benching(tmp_path):
    """rc=124 is a hardware-budget event: cap max_feasible_rung, demote, and
    do NOT count it as an error or a zero."""
    controller, runner = make(tmp_path, [(124, 0), (0, 0)])
    state = controller.state.domain("coloring")
    state.rung = 3
    controller.state.put("coloring", state)
    controller.run_cycle("coloring")  # times out at rung 3
    state = controller.state.domain("coloring")
    assert state.max_feasible_rung == 2
    assert state.rung == 2
    assert state.consecutive_errors == 0 and state.zeros_at_rung == 0
    assert not state.exhausted
    controller.run_cycle("coloring")  # next cycle runs at the capped rung
    assert runner.calls[-1][1] == LADDER[2]


def test_patience_at_capped_rung_exhausts_cleanly_and_sentinel_respects_cap(tmp_path):
    outcomes = [(124, 0)] + [(0, 0)] * PATIENCE + [(0, 0)]
    controller, runner = make(tmp_path, outcomes)
    state = controller.state.domain("coloring")
    state.rung = 3
    controller.state.put("coloring", state)
    controller.run_cycle("coloring")  # cap at 2
    for _ in range(PATIENCE):
        controller.run_cycle("coloring")  # zeros at rung 2 -> exhausted, never re-escalates
    state = controller.state.domain("coloring")
    assert state.exhausted and "hardware budget" in state.exhausted_reason
    controller.run_cycle("coloring")  # sentinel probe
    assert runner.calls[-1][1] == LADDER[2]  # probes at the cap, not the top


def test_timeout_at_rung_zero_means_nothing_fits(tmp_path):
    controller, _ = make(tmp_path, [(124, 0)])
    state = controller.state.domain("coloring")
    state.rung = 0
    controller.state.put("coloring", state)
    controller.run_cycle("coloring")
    state = controller.state.domain("coloring")
    assert state.exhausted and "no rung fits" in state.exhausted_reason


def test_consecutive_errors_bench_the_domain(tmp_path):
    controller, _ = make(tmp_path, [(1, 0)] * fc.MAX_ERRORS)
    for _ in range(fc.MAX_ERRORS):
        controller.run_cycle("coloring")
    state = controller.state.domain("coloring")
    assert state.exhausted and "errors" in state.exhausted_reason


def test_state_survives_restart(tmp_path):
    controller, _ = make(tmp_path, [(0, 0)] * PATIENCE)
    for _ in range(PATIENCE):
        controller.run_cycle("coloring")
    reloaded = Controller(tmp_path, runner=lambda d, c: (0, 0.0), sleeper=lambda _s: None, sync_dir=None)
    assert reloaded.state.domain("coloring").rung == 2
    assert reloaded.state.total_cycles == PATIENCE


def test_dedupe_keeps_first_per_graph_id(tmp_path):
    controller, _ = make(tmp_path, [])
    rows = [
        {"graph_id": "a", "domain": "coloring", "found_on": "day1"},
        {"graph_id": "a", "domain": "coloring", "found_on": "day2"},  # dupe
        {"graph_id": "a", "domain": "mis"},                            # same id, other domain: kept
        {"graph_id": "b", "domain": "coloring"},
    ]
    controller.library.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    assert controller.dedupe_library() == 1
    kept = [json.loads(l) for l in controller.library.read_text(encoding="utf-8").splitlines()]
    assert len(kept) == 3 and kept[0]["found_on"] == "day1"


def test_sync_publishes_library_atomically(tmp_path):
    sync = tmp_path / "service_state"
    sync.mkdir()
    controller, _ = make(tmp_path, [(0, 2)], sync_dir=str(sync))
    controller.run_cycle("coloring")  # find -> sync
    published = sync / fc.LIBRARY_NAME
    assert published.exists()
    assert len(published.read_text(encoding="utf-8").splitlines()) == 2


def test_sync_skips_byte_identical_publication(tmp_path):
    sync = tmp_path / "service_state"
    sync.mkdir()
    controller, _ = make(tmp_path, [], sync_dir=str(sync))
    content = b'{"domain":"coloring","graph_id":"same"}\n'
    controller.library.write_bytes(content)
    published = sync / fc.LIBRARY_NAME
    published.write_bytes(content)
    before = published.stat()

    assert controller.sync_library() == "unchanged"

    after = published.stat()
    assert (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) == (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )


def test_startup_sync_failure_prevents_release_status(tmp_path):
    invalid_sync_dir = tmp_path / "not-a-directory"
    invalid_sync_dir.write_text("occupied", encoding="utf-8")
    controller, _ = make(tmp_path, [], sync_dir=str(invalid_sync_dir))
    controller.library.write_text('{"domain":"coloring","graph_id":"g1"}\n', encoding="utf-8")

    with pytest.raises(OSError):
        controller.main_loop()

    assert not (tmp_path / fc.RELEASE_STATUS_NAME).exists()


def test_hatchery_passes_library_to_both_mint_and_pools(tmp_path, monkeypatch):
    """Fairness invariant: minting certification and grading pools must use
    the same library path."""
    from bcv.ephemeral import Hatchery

    library = tmp_path / "adversary_library.jsonl"
    library.write_text("", encoding="utf-8")
    captured: dict = {}

    def fake_pool(domain, stress_ns, samples, seed, library_path):
        captured["pool_library"] = str(library_path)
        return []

    def fake_mint(domain, buffers, max_items, max_n, stress_ns, seed, library_path=None):
        captured["mint_library"] = str(library_path)
        return []

    monkeypatch.setattr("bcv.refinery._stress_pool", fake_pool)
    monkeypatch.setattr("bcv.examiner.mint_repair_items", fake_mint)
    hatchery = Hatchery(domains=("coloring",), library_path=str(library))
    hatchery._warm()
    assert captured["pool_library"] == str(library)
    assert captured["mint_library"] == str(library)
    assert hatchery.status()["forge_library"] == {"configured": True, "entries": 0}
