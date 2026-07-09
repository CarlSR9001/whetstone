from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from bcv.examiner import ExaminerBank
from bcv.panel import mint_support_items
from bcv.service import make_server

TICKET = {
    "question": "How long do I have to return my order?",
    "source": (
        "Our returns window is 30 days from the delivery date. Items must be unused and in "
        "original packaging. Refunds are issued to the original payment method after the "
        "warehouse inspects the return, which typically takes 5 business days."
    ),
    "forbidden": ["guarantee a refund", "within 24 hours"],
}

GOOD_ANSWER = (
    "You have 30 days from the delivery date to return your order, as long as the items are "
    "unused and in original packaging. Refunds go to the original payment method after the "
    "warehouse inspects the return, typically within 5 business days."
)


@pytest.fixture
def server(tmp_path):
    bank = ExaminerBank(tmp_path / "bank")
    for item in mint_support_items([TICKET], max_items=1):
        bank.add(item)
        bank.promote(item.item_id)
    bank.save()
    httpd = make_server(tmp_path / "bank", port=0, token="secret")
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd
    httpd.shutdown()


def _call(httpd, path, payload=None, token="secret"):
    url = f"http://127.0.0.1:{httpd.server_port}{path}"
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(url, data=data, method="POST" if data else "GET")
    request.add_header("Content-Type", "application/json")
    if token:
        request.add_header("X-Whetstone-Token", token)
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def test_auth_is_enforced(server):
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _call(server, "/status", token=None)
    assert excinfo.value.code == 401


def test_status_never_serves_items(server):
    status = _call(server, "/status")
    assert status["buckets"]["promoted"] == 1
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _call(server, "/items")
    assert excinfo.value.code == 404


def test_grade_and_gate_over_http(server, tmp_path):
    bank = ExaminerBank(tmp_path / "bank")
    item_id = bank.promoted_items()[0].item_id

    base = _call(server, "/grade", {"system": "base", "answers": {item_id: "yes"}})
    assert base["passed"] == 0
    cand = _call(server, "/grade", {"system": "cand", "answers": {item_id: GOOD_ANSWER, "bogus": "x"}})
    assert cand["passed"] == 1
    assert cand["ignored_unknown_items"] == ["bogus"]

    gate = _call(server, "/gate", {"baseline": "base", "candidate": "cand"})
    assert gate["verdict"] in ("PASS", "HOLD")  # 1 gain, 0 regressions: HOLD at default alpha
    assert gate["verdict"] == "HOLD"
    assert "items_detail" not in gate["paired_evidence"]  # item ids stay off the wire

    relaxed = _call(server, "/gate", {"baseline": "base", "candidate": "cand", "policy": {"confidence_alpha": 1.0}})
    assert relaxed["verdict"] == "PASS"


def test_grade_rejects_garbage(server):
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _call(server, "/grade", {"system": "x", "answers": {"unknown": "y"}})
    assert excinfo.value.code == 400
