import argparse

from scripts.check_release_state import validate_forge, validate_health, validate_sessions


def _health() -> dict:
    return {
        "status": "ok",
        "version": "0.8.0",
        "build_commit": "a" * 40,
        "tools": 8,
        "stateless": True,
        "private_bank_loaded": False,
        "report_card": {"error": None, "ready": True, "active_sessions": 0},
        "open_bench": {
            "ready": True,
            "publication_ledger_configured": True,
            "publication_ledger_parent_writable": True,
            "raw_tasks_persisted": False,
            "raw_answers_persisted": False,
        },
        "receipt_signing": {"configured": True, "ready": True},
    }


def test_release_health_contract() -> None:
    args = argparse.Namespace(version="0.8.0", commit="a" * 40)
    assert validate_health(_health(), args) == []


def test_release_health_reports_all_mismatches() -> None:
    payload = _health()
    payload["version"] = "0.7.0"
    payload["receipt_signing"]["ready"] = False
    args = argparse.Namespace(version="0.8.0", commit="a" * 40)
    errors = validate_health(payload, args)
    assert errors == [
        "version: expected '0.8.0', got '0.7.0'",
        "receipt_signing.ready: expected True, got False",
    ]


def test_forge_and_session_contracts() -> None:
    forge = {
        "status": "running",
        "version": "0.8.0",
        "build_commit": "a" * 40,
        "library_sync": "unchanged",
        "pid": 123,
    }
    args = argparse.Namespace(version="0.8.0", commit="a" * 40, pid=123)
    assert validate_forge(forge, args) == []
    assert validate_sessions({"report_card": {"active_sessions": 0}}, args) == []
    assert validate_sessions({"report_card": {"active_sessions": 1}}, args)
