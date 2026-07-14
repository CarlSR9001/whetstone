"""Aggregate metrics: privacy properties, persistence, and the public summary."""

from __future__ import annotations

import json
import threading
import urllib.request

import pytest

from bcv.stats import Stats


def test_clients_are_salted_hashes_and_repeats_are_counted(tmp_path):
    stats = Stats(tmp_path / "stats.json")
    stats.touch("203.0.113.7")
    stats.touch("203.0.113.7")
    stats.touch("198.51.100.2")
    summary = stats.public_summary()
    assert summary["unique_clients"] == 2
    assert summary["repeat_clients"] == 1
    assert summary["requests_total"] == 3
    # No raw address appears anywhere in the summary or on disk.
    stats.flush()
    on_disk = (tmp_path / "stats.json").read_text(encoding="utf-8")
    for blob in (json.dumps(summary), on_disk):
        assert "203.0.113.7" not in blob and "198.51.100.2" not in blob


def test_persistence_round_trip_preserves_salt_and_counts(tmp_path):
    path = tmp_path / "stats.json"
    first = Stats(path)
    first.touch("203.0.113.7")
    first.tool_call("promotion_gate", "mcp", "ok")
    first.bump("report_card.sessions")
    first.flush()

    second = Stats(path)
    second.touch("203.0.113.7")  # same client, same salt -> still one unique
    summary = second.public_summary()
    assert summary["unique_clients"] == 1
    assert summary["requests_total"] == 2
    assert summary["tool_calls"]["promotion_gate"]["mcp"] == 1
    assert summary["report_card"]["sessions"] == 1
    assert summary["persistent"] is True


def test_without_state_dir_stats_stay_in_memory(tmp_path):
    stats = Stats(None)
    stats.touch("203.0.113.7")
    stats.flush()  # must be a no-op, not an error
    assert stats.public_summary()["persistent"] is False


def test_retention_buckets():
    stats = Stats(None)
    for value in (0.01, 0.1, 0.3, 0.9):
        stats.retention_bucket(value)
    buckets = stats.public_summary()["report_card"]
    assert buckets["retention.lt5"] == 1
    assert buckets["retention.lt25"] == 1
    assert buckets["retention.lt50"] == 1
    assert buckets["retention.ge50"] == 1


@pytest.fixture()
def toolbox_url():
    from bcv.toolbox_service import make_server

    server = make_server("127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_stats_endpoint_reports_tool_calls(toolbox_url):
    from bcv.product_tools import examples

    body = json.dumps(examples()["gate"]).encode("utf-8")
    request = urllib.request.Request(
        toolbox_url + "/api/gate", data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    urllib.request.urlopen(request, timeout=10).read()

    stats = json.loads(urllib.request.urlopen(toolbox_url + "/api/stats", timeout=5).read())
    assert stats["tool_calls"]["promotion_gate"]["rest"] >= 1
    assert stats["unique_clients"] >= 1
    assert stats["outcomes"]["ok"] >= 1
    raw = json.dumps(stats)
    assert "127.0.0.1" not in raw  # aggregate only, never addresses
