from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from bcv.product_tools import examples
from bcv.toolbox_service import make_server


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


def test_health_exposes_stateless_boundary_and_security_headers(toolbox_url):
    payload, response = get_json(toolbox_url + "/api/health")
    assert payload["status"] == "ok"
    assert payload["stateless"] is True
    assert payload["private_bank_loaded"] is False
    assert payload["tools"] == 8
    assert response.headers["X-Frame-Options"] == "DENY"
    assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]


def test_index_and_evidence_are_public_but_sanitized(toolbox_url):
    html = urllib.request.urlopen(toolbox_url + "/", timeout=5).read().decode("utf-8")
    assert "Eight small products" in html
    assert "Use disposable or sanitized inputs" in html
    evidence, _ = get_json(toolbox_url + "/api/evidence")
    assert evidence["cross_scale"]["models"] == 8
    assert "item_id" not in json.dumps(evidence)


def test_gate_and_inspector_post_end_to_end(toolbox_url):
    gate = post_json(toolbox_url + "/api/gate", examples()["gate"])
    assert gate["verdict"] == "PASS"
    assert gate["request_id"]
    inspector = post_json(toolbox_url + "/api/inspect", examples()["inspector"])
    assert inspector["gate"]["verdict"] == "PASS"
    assert inspector["audit"]["quarantined_items"] == 1


def test_cross_origin_and_wrong_content_type_fail_closed(toolbox_url):
    request = urllib.request.Request(
        toolbox_url + "/api/gate",
        data=b"{}",
        headers={"Content-Type": "application/json", "Origin": "https://attacker.invalid"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(request, timeout=5)
    assert error.value.code == 403

    wrong = urllib.request.Request(toolbox_url + "/api/gate", data=b"{}", method="POST")
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(wrong, timeout=5)
    assert error.value.code == 415


def test_unknown_endpoint_does_not_fall_through_to_bank_content(toolbox_url):
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(toolbox_url + "/api/items", timeout=5)
    assert error.value.code == 404
    assert "private_promotion_exam" not in error.value.read().decode("utf-8")
