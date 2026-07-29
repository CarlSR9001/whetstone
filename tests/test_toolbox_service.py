from __future__ import annotations

import hashlib
import json
import threading
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from bcv._version import __version__
from bcv.product_tools import examples
from bcv.toolbox_service import _trusted_client_ip, make_server


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


def get_json(url: str) -> tuple[dict, urllib.response.addinfourl]:
    response = urllib.request.urlopen(url, timeout=5)
    return json.loads(response.read()), response


def post_json(url: str, payload: dict, **headers) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    return json.loads(urllib.request.urlopen(request, timeout=10).read())


def test_client_ip_trusts_only_the_loopback_proxy():
    assert _trusted_client_ip(
        "127.0.0.1",
        "198.51.100.7",
        "attacker-controlled, 198.51.100.7",
    ) == "198.51.100.7"
    assert _trusted_client_ip(
        "::1",
        None,
        "203.0.113.99, 198.51.100.8",
    ) == "198.51.100.8"
    assert _trusted_client_ip(
        "192.0.2.10",
        "198.51.100.7",
        "203.0.113.99",
    ) == "192.0.2.10"
    assert _trusted_client_ip("127.0.0.1", "not-an-ip", "also-not-an-ip") == "127.0.0.1"


def test_nginx_overwrites_forwarded_client_identity():
    config = (
        Path(__file__).resolve().parents[1] / "deploy" / "nginx.conf"
    ).read_text(encoding="utf-8")
    assert config.count("proxy_set_header X-Forwarded-For $remote_addr;") == 3
    assert "$proxy_add_x_forwarded_for" not in config


def test_health_exposes_stateless_boundary_and_security_headers(toolbox_url):
    payload, response = get_json(toolbox_url + "/api/health")
    assert payload["status"] == "ok"
    assert payload["stateless"] is True
    assert payload["private_bank_loaded"] is False
    assert payload["tools"] == 8
    assert payload["version"] == __version__ == "0.5.3"
    assert payload["build_commit"] == "development"
    assert payload["mcp_endpoint"] == "/mcp"
    assert payload["report_card"]["ready"] is False  # make_server never warms the hatchery
    assert response.headers["X-Frame-Options"] == "DENY"
    assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]
    assert response.headers["Server"].startswith(f"WhetstoneTools/{__version__} ")


def test_index_and_evidence_are_public_but_sanitized(toolbox_url):
    html = urllib.request.urlopen(toolbox_url + "/", timeout=5).read().decode("utf-8")
    assert "Stop asking whether it got better" in html
    assert "resultViz" in html
    assert "Use disposable or sanitized inputs" in html
    evidence, _ = get_json(toolbox_url + "/api/evidence")
    assert evidence["cross_scale"]["models"] == 8
    assert "item_id" not in json.dumps(evidence)
    assert set(evidence["source_receipts_sha256"]) == {
        "cross_scale_ladder_receipt.json",
        "redteam_gate_receipt.json",
        "relevance_eval_report.json",
    }


def test_public_privacy_claims_match_operational_telemetry(toolbox_url):
    for path in ("/", "/for-agents", "/skill.md", "/llms.txt"):
        body = urllib.request.urlopen(toolbox_url + path, timeout=5).read().decode("utf-8").lower()
        assert "nothing stored" not in body
        assert "writes nothing" not in body
        assert "no persistence" not in body
        assert "access log" in body
        assert "request" in body and ("not persisted" in body or "not retained" in body)


def test_packaged_evidence_matches_source_receipts():
    root = Path(__file__).parents[1]
    evidence = json.loads(
        (root / "src" / "bcv" / "toolbox_static" / "evidence.json").read_text(encoding="utf-8")
    )
    receipts = {
        name: json.loads((root / "results" / name).read_text(encoding="utf-8"))
        for name in evidence["source_receipts_sha256"]
    }
    for name, expected in evidence["source_receipts_sha256"].items():
        # Git stores these text receipts with LF. Windows may materialize a
        # CRLF checkout, so hash canonical repository bytes, not host newlines.
        canonical = (root / "results" / name).read_bytes().replace(b"\r\n", b"\n")
        assert hashlib.sha256(canonical).hexdigest() == expected

    relevance = receipts["relevance_eval_report.json"]
    ladder = receipts["cross_scale_ladder_receipt.json"]
    redteam = receipts["redteam_gate_receipt.json"]
    scale_gate = ladder["gates"][0]
    assert evidence["relevance"]["probes"] == relevance["probes"]
    assert evidence["relevance"]["accuracy"] == relevance["accuracy"]
    assert evidence["cross_scale"]["bank_items"] == ladder["bank"]["items"]
    assert evidence["cross_scale"]["models"] == len(ladder["ladder"])
    assert evidence["cross_scale"]["largest_contrast"] == {
        key: scale_gate[key] for key in ("gains", "p", "regressions", "verdict")
    }
    assert evidence["redteam"]["paraphrase_caught"] is redteam["paraphrase_attack"][
        "behavioral_fingerprint_matched"
    ]
    assert evidence["redteam"]["inflation_caught"] is redteam["inflation_attack"]["caught"]


def test_manifest_openapi_and_package_metadata_share_one_version(toolbox_url):
    manifest, _ = get_json(toolbox_url + "/.well-known/mcp.json")
    openapi, _ = get_json(toolbox_url + "/openapi.json")
    assert manifest["version"] == openapi["info"]["version"] == __version__
    assert manifest["build_commit"] == openapi["info"]["x-build-commit"] == "development"

    pyproject = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["dynamic"] == ["version"]
    assert pyproject["tool"]["setuptools"]["dynamic"]["version"]["attr"] == "bcv._version.__version__"
    assert "toolbox_static/*" in pyproject["tool"]["setuptools"]["package-data"]["bcv"]


def test_gate_and_inspector_post_end_to_end(toolbox_url):
    gate = post_json(toolbox_url + "/api/gate", examples()["gate"])
    assert gate["verdict"] == "PASS"
    assert gate["request_id"]
    inspector = post_json(toolbox_url + "/api/inspect", examples()["inspector"])
    assert inspector["gate"]["verdict"] == "PASS"
    assert inspector["audit"]["quarantined_items"] == 1


def test_cross_origin_clients_are_allowed_with_cors(toolbox_url):
    """No auth, cookies, or ambient authority means no browser CSRF surface.

    Report-card ids are explicit one-shot bearer values. Blocking cross-origin
    only blocks real clients (claude.ai connectors send an Origin header).
    """
    payload = json.dumps(examples()["gate"]).encode("utf-8")
    request = urllib.request.Request(
        toolbox_url + "/api/gate",
        data=payload,
        headers={"Content-Type": "application/json", "Origin": "https://claude.ai"},
        method="POST",
    )
    response = urllib.request.urlopen(request, timeout=10)
    assert response.status == 200
    assert response.headers["Access-Control-Allow-Origin"] == "*"


def test_preflight_is_answered(toolbox_url):
    request = urllib.request.Request(toolbox_url + "/mcp", method="OPTIONS")
    response = urllib.request.urlopen(request, timeout=5)
    assert response.status == 204
    assert response.headers["Access-Control-Allow-Origin"] == "*"
    assert "Content-Type" in response.headers["Access-Control-Allow-Headers"]


def test_json_body_accepted_regardless_of_declared_content_type(toolbox_url):
    """`curl -X POST url -d '{...}'` sends form-urlencoded; it must still work."""
    payload = json.dumps(examples()["gate"]).encode("utf-8")
    request = urllib.request.Request(
        toolbox_url + "/api/gate",
        data=payload,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    assert urllib.request.urlopen(request, timeout=10).status == 200


def test_non_json_body_still_fails_with_a_hint(toolbox_url):
    request = urllib.request.Request(toolbox_url + "/api/gate", data=b"not json", method="POST")
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(request, timeout=5)
    assert error.value.code == 400
    assert "/api/examples" in json.loads(error.value.read())["hint"]


def test_unknown_endpoint_does_not_fall_through_to_bank_content(toolbox_url):
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(toolbox_url + "/api/items", timeout=5)
    assert error.value.code == 404
    assert "private_promotion_exam" not in error.value.read().decode("utf-8")
