"""The public MCP endpoint and the Tier 1 disposable report card.

The hatchery fixture performs one real warm-up (coloring only, small master
set) and is shared across the module; every test uses a distinct client IP so
the module-global rate limiters never couple tests together.
"""

from __future__ import annotations

import itertools
import json
import threading
import urllib.error
import urllib.request

import pytest

import bcv.ephemeral as ephemeral
from bcv.ephemeral import Hatchery, TierError
from bcv.mcp_service import handle_mcp
from bcv.product_tools import examples, gate_results
from bcv.toolbox_service import make_server

_IP = (f"10.0.0.{n}" for n in itertools.count(1))


def rpc(method: str, params: dict | None = None, request_id: int | None = 1) -> dict:
    message: dict = {"jsonrpc": "2.0", "method": method}
    if request_id is not None:
        message["id"] = request_id
    if params is not None:
        message["params"] = params
    return message


@pytest.fixture(scope="module")
def warm_hatchery():
    instance = Hatchery(domains=("coloring",), master_items_per_domain=3, items_per_session=2)
    instance._warm()
    assert instance.status()["ready"], "hatchery warm-up failed"
    previous = ephemeral._HATCHERY
    ephemeral._HATCHERY = instance
    yield instance
    ephemeral._HATCHERY = previous


def certified_answers(hatchery: Hatchery, session: dict) -> dict[str, str]:
    """Each item's own mined repair (exists by item-fairness construction)."""
    live = hatchery.sessions[session["session_id"]]["items"]
    return {
        public_id: json.dumps({"repair_expression": item.lineage[0].split(":", 1)[1]})
        for public_id, (_, item) in live.items()
    }


# ------------------------------------------------------------------- protocol


def test_initialize_negotiates_protocol_and_declares_tools():
    status, body = handle_mcp(rpc("initialize", {"protocolVersion": "2025-06-18"}), next(_IP))
    assert status == 200
    result = body["result"]
    assert result["protocolVersion"] == "2025-06-18"
    assert result["serverInfo"]["name"] == "whetstone-tools"
    assert "tools" in result["capabilities"]
    assert "no private exam bank" in result["instructions"].lower()


def test_initialize_falls_back_on_unknown_protocol_version():
    status, body = handle_mcp(rpc("initialize", {"protocolVersion": "1999-01-01"}), next(_IP))
    assert body["result"]["protocolVersion"] == "2025-06-18"


def test_notifications_get_202_and_no_body():
    status, body = handle_mcp({"jsonrpc": "2.0", "method": "notifications/initialized"}, next(_IP))
    assert status == 202
    assert body is None


def test_batches_and_malformed_bodies_are_rejected():
    status, body = handle_mcp([rpc("ping")], next(_IP))
    assert status == 400
    assert "batching" in body["error"]["message"]
    status, body = handle_mcp({"not": "jsonrpc"}, next(_IP))
    assert status == 400


def test_unknown_method_and_unknown_tool():
    status, body = handle_mcp(rpc("resources/list"), next(_IP))
    assert body["error"]["code"] == -32601
    status, body = handle_mcp(rpc("tools/call", {"name": "drain_the_bank", "arguments": {}}), next(_IP))
    assert body["error"]["code"] == -32602


def test_tools_list_has_both_tiers_and_object_schemas():
    status, body = handle_mcp(rpc("tools/list"), next(_IP))
    tools = {tool["name"]: tool for tool in body["result"]["tools"]}
    assert {"promotion_gate", "audit_leakage", "inspect_promotion", "bank_health",
            "safe_patch", "counterexample_hunt", "memory_relevance", "replay_trace",
            "report_card_start", "report_card_submit", "about_whetstone"} <= set(tools)
    for tool in tools.values():
        assert tool["inputSchema"]["type"] == "object"
    # No surface lists or serves private items; the tool names must not suggest one.
    assert not any("item" in name and "list" in name for name in tools)


# ------------------------------------------------------------------- tier 0


def test_gate_tool_matches_rest_result_exactly():
    payload = examples()["gate"]
    direct = gate_results(examples()["gate"])
    status, body = handle_mcp(rpc("tools/call", {"name": "promotion_gate", "arguments": payload}), next(_IP))
    result = body["result"]
    assert result["isError"] is False
    assert result["structuredContent"]["verdict"] == direct["verdict"]
    assert json.loads(result["content"][0]["text"])["verdict"] == direct["verdict"]


def test_bad_tool_input_is_a_tool_failure_not_a_protocol_error():
    status, body = handle_mcp(rpc("tools/call", {"name": "promotion_gate", "arguments": {"baseline": 3}}), next(_IP))
    assert status == 200
    assert body["result"]["isError"] is True
    assert "error" not in body  # JSON-RPC level stays clean


# ------------------------------------------------------------------- tier 1


def test_report_card_full_flow_with_certified_repairs(warm_hatchery):
    ip = next(_IP)
    status, body = handle_mcp(rpc("tools/call", {"name": "report_card_start", "arguments": {}}), ip)
    session = body["result"]["structuredContent"]
    assert len(session["items"]) == 2
    for item in session["items"]:
        assert set(item) == {"item_id", "domain", "prompt"}  # no payload, no lineage, no repair
        assert "mined_repair" not in json.dumps(item)

    answers = certified_answers(warm_hatchery, session)
    status, body = handle_mcp(rpc("tools/call", {
        "name": "report_card_submit",
        "arguments": {"session_id": session["session_id"], "answers": answers},
    }), ip)
    report = body["result"]["structuredContent"]
    assert report["passed"] == report["total"] == 2
    assert report["disposable_cohort"] is True
    assert len(report["item_set_sha256"]) == 64
    assert all(row["support_retention"] > 0 for row in report["items"])
    assert report["median_support_retention"] is not None


def test_sessions_are_one_shot(warm_hatchery):
    ip = next(_IP)
    session = warm_hatchery.start_session(ip)
    warm_hatchery.submit(session["session_id"], {}, ip)
    with pytest.raises(TierError, match="unknown or expired"):
        warm_hatchery.submit(session["session_id"], {}, ip)


def test_unanswered_items_fail_and_garbage_answers_fail(warm_hatchery):
    ip = next(_IP)
    session = warm_hatchery.start_session(ip)
    first_id = session["items"][0]["item_id"]
    report = warm_hatchery.submit(
        session["session_id"], {first_id: "import os; os.system('boom')"}, ip
    )
    assert report["passed"] == 0
    rows = {row["item_id"]: row for row in report["items"]}
    assert rows[first_id]["answered"] is True and rows[first_id]["passed"] is False


def test_expired_sessions_are_swept(warm_hatchery):
    ip = next(_IP)
    session = warm_hatchery.start_session(ip)
    warm_hatchery.sessions[session["session_id"]]["expires_at"] = 0.0
    with pytest.raises(TierError, match="unknown or expired"):
        warm_hatchery.submit(session["session_id"], {}, ip)


def test_session_cap_is_enforced(warm_hatchery):
    original_cap = warm_hatchery.max_active_sessions
    warm_hatchery.max_active_sessions = len(warm_hatchery.sessions) + 1
    try:
        warm_hatchery.start_session(next(_IP))
        with pytest.raises(TierError, match="too many live sessions"):
            warm_hatchery.start_session(next(_IP))
    finally:
        warm_hatchery.max_active_sessions = original_cap


def test_per_ip_start_limit(warm_hatchery):
    ip = next(_IP)
    for _ in range(4):
        session = warm_hatchery.start_session(ip)
        warm_hatchery.submit(session["session_id"], {}, ip)  # keep session count flat
    with pytest.raises(TierError, match="session limit"):
        warm_hatchery.start_session(ip)


def test_cold_hatchery_reports_warming_not_a_crash():
    cold = Hatchery()
    with pytest.raises(TierError, match="warming up"):
        cold.start_session(next(_IP))


# ------------------------------------------------------------------- HTTP shell


@pytest.fixture()
def toolbox_url():
    server = make_server("127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def _post(url: str, payload) -> tuple[int, dict | None]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    response = urllib.request.urlopen(request, timeout=10)
    raw = response.read()
    return response.status, (json.loads(raw) if raw else None)


def test_http_mcp_initialize_and_notification(toolbox_url):
    status, body = _post(toolbox_url + "/mcp", rpc("initialize", {"protocolVersion": "2025-06-18"}))
    assert status == 200
    assert body["result"]["serverInfo"]["name"] == "whetstone-tools"
    status, body = _post(toolbox_url + "/mcp", {"jsonrpc": "2.0", "method": "notifications/initialized"})
    assert status == 202
    assert body is None


def test_agent_docs_are_served(toolbox_url):
    skill = urllib.request.urlopen(toolbox_url + "/skill.md", timeout=5)
    assert skill.status == 200
    assert skill.headers["Content-Type"].startswith("text/markdown")
    body = skill.read().decode("utf-8")
    assert body.startswith("---") and "name: whetstone-tools" in body
    assert "report_card_submit" in body and "degenerate_narrowing" in body

    page = urllib.request.urlopen(toolbox_url + "/for-agents", timeout=5)
    assert page.status == 200
    html = page.read().decode("utf-8")
    assert "report_card_start" in html and "/skill.md" in html

    llms = urllib.request.urlopen(toolbox_url + "/llms.txt", timeout=5).read().decode("utf-8")
    assert "/skill.md" in llms and "/mcp" in llms and "/for-agents" in llms

    try:
        urllib.request.urlopen(toolbox_url + "/not-a-page", timeout=5)
        raise AssertionError("expected 404")
    except urllib.error.HTTPError as error:
        assert error.code == 404


def test_discovery_surfaces(toolbox_url):
    robots = urllib.request.urlopen(toolbox_url + "/robots.txt", timeout=5).read().decode("utf-8")
    assert "Sitemap:" in robots and "/sitemap.xml" in robots

    sitemap = urllib.request.urlopen(toolbox_url + "/sitemap.xml", timeout=5)
    assert sitemap.headers["Content-Type"].startswith("application/xml")
    body = sitemap.read().decode("utf-8")
    assert "<urlset" in body and "/for-agents" in body and "/skill.md" in body

    spec = json.loads(urllib.request.urlopen(toolbox_url + "/openapi.json", timeout=5).read())
    assert spec["openapi"].startswith("3.1")
    assert "/api/gate" in spec["paths"] and "/mcp" in spec["paths"]
    assert spec["paths"]["/api/gate"]["post"]["requestBody"]["content"]["application/json"]["example"]

    for manifest_path in ("/.well-known/mcp.json", "/mcp.json"):
        manifest = json.loads(urllib.request.urlopen(toolbox_url + manifest_path, timeout=5).read())
        assert manifest["transport"] == "streamable-http"
        assert manifest["endpoint"].endswith("/mcp")
        assert manifest["authentication"] == {"type": "none"}

    full = urllib.request.urlopen(toolbox_url + "/llms-full.txt", timeout=5).read().decode("utf-8")
    assert "## Tiers" in full and "name: whetstone-tools" in full  # index + embedded skill.md

    og = urllib.request.urlopen(toolbox_url + "/og.png", timeout=5)
    assert og.headers["Content-Type"] == "image/png"
    assert og.read(8) == b"\x89PNG\r\n\x1a\n"


def test_index_has_structured_data_and_faq(toolbox_url):
    html = urllib.request.urlopen(toolbox_url + "/", timeout=5).read().decode("utf-8")
    assert 'application/ld+json' in html
    assert '"@type": "FAQPage"' in html and '"@type": "SoftwareApplication"' in html
    assert 'property="og:image"' in html and "og.png" in html
    assert "Frequently asked questions" in html
    assert 'rel="canonical"' in html


def test_http_get_mcp_is_405_with_guidance(toolbox_url):
    request = urllib.request.Request(toolbox_url + "/mcp", method="GET")
    try:
        urllib.request.urlopen(request, timeout=5)
        raise AssertionError("expected 405")
    except urllib.error.HTTPError as error:
        assert error.code == 405
        assert "JSON-RPC" in json.loads(error.read())["error"]
