from __future__ import annotations

import copy
import shutil
import subprocess

import pytest

import bcv.hosted as hosted
import bcv.receipts as receipts
from bcv.cli import main
from bcv.hosted import verify_hosted_receipt
from bcv.receipts import (
    ReceiptVerificationError,
    attest_receipt,
    key_id,
    receipt_key_bundle,
    verify_receipt,
)


pytestmark = pytest.mark.skipif(shutil.which("ssh-keygen") is None, reason="OpenSSH is required")


def _key(tmp_path, name: str):
    path = tmp_path / name
    subprocess.run(
        ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", name, "-f", str(path)],
        check=True,
    )
    return path


def _receipt() -> dict:
    return {
        "verdict": "PASS",
        "identity_verified": False,
        "answers_sha256": "a" * 64,
        "cohort_sha256": "b" * 64,
        "policy": {"max_regressions": 0},
        "baseline_manifest": {"name": "baseline"},
        "candidate_manifest": {"name": "candidate"},
    }


def test_signed_receipt_verifies_and_keeps_identity_boundary(tmp_path, monkeypatch):
    private_key = _key(tmp_path, "active")
    monkeypatch.setenv("WHETSTONE_RECEIPT_SIGNING_KEY", str(private_key))
    receipt = attest_receipt(_receipt(), "open_bench_private", challenge="run-12345678", now=100)
    bundle = receipt_key_bundle()

    verified = verify_receipt(
        receipt,
        bundle,
        expected_challenge="run-12345678",
        now=101,
    )

    assert receipt["attestation"]["status"] == "signed"
    assert verified["valid"] is True
    assert verified["claims"] == {
        "issuer_authenticated": True,
        "subject_identity_verified": False,
    }
    assert bundle["current_key_id"] == verified["key_id"]


def test_tampering_and_cross_session_replay_are_rejected(tmp_path, monkeypatch):
    private_key = _key(tmp_path, "active")
    monkeypatch.setenv("WHETSTONE_RECEIPT_SIGNING_KEY", str(private_key))
    receipt = attest_receipt(_receipt(), "open_bench_private", challenge="run-12345678", now=100)
    bundle = receipt_key_bundle()

    changed = copy.deepcopy(receipt)
    changed["verdict"] = "BLOCK"
    with pytest.raises(ReceiptVerificationError, match="receipt_sha256"):
        verify_receipt(changed, bundle, now=101)
    changed = copy.deepcopy(receipt)
    changed["publication"] = {"status": "published"}
    with pytest.raises(ReceiptVerificationError, match="receipt_sha256"):
        verify_receipt(changed, bundle, now=101)
    with pytest.raises(ReceiptVerificationError, match="challenge"):
        verify_receipt(receipt, bundle, expected_challenge="run-87654321", now=101)
    with pytest.raises(ReceiptVerificationError, match="expired"):
        verify_receipt(receipt, bundle, now=86_501)
    assert verify_receipt(receipt, bundle, now=86_501, allow_expired=True)["expired"] is True


def test_key_rotation_bundle_verifies_old_and_new_receipts(tmp_path, monkeypatch):
    old_key = _key(tmp_path, "old")
    new_key = _key(tmp_path, "new")
    monkeypatch.setenv("WHETSTONE_RECEIPT_SIGNING_KEY", str(old_key))
    old_receipt = attest_receipt(_receipt(), "report_card", challenge="old-12345678", now=100)
    monkeypatch.setenv("WHETSTONE_RECEIPT_SIGNING_KEY", str(new_key))
    new_receipt = attest_receipt(_receipt(), "report_card", challenge="new-12345678", now=100)
    bundle = receipt_key_bundle()
    old_public = (tmp_path / "old.pub").read_text(encoding="utf-8")
    bundle["keys"].append({
        "key_id": key_id(old_public),
        "algorithm": "ssh-ed25519",
        "public_key": old_public,
        "status": "retired",
    })

    with pytest.raises(ReceiptVerificationError, match="retired"):
        verify_receipt(old_receipt, bundle, now=101)
    assert verify_receipt(old_receipt, bundle, now=101, allow_retired=True)["valid"] is True
    bundle["keys"][-1]["status"] = "revoked"
    with pytest.raises(ReceiptVerificationError, match="revoked"):
        verify_receipt(old_receipt, bundle, now=101, allow_retired=True)
    assert verify_receipt(new_receipt, bundle, now=101)["valid"] is True


def test_missing_key_is_explicitly_unsigned(monkeypatch):
    monkeypatch.delenv("WHETSTONE_RECEIPT_SIGNING_KEY", raising=False)
    receipt = attest_receipt(_receipt(), "report_card", challenge="dev-12345678", now=100)

    assert len(receipt["receipt_sha256"]) == 64
    assert receipt["attestation"]["status"] == "unsigned"
    with pytest.raises(ReceiptVerificationError, match="signed attestation"):
        verify_receipt(receipt, {"keys": []}, now=101)


def test_signing_status_requires_readable_private_key(tmp_path, monkeypatch):
    private_key = _key(tmp_path, "status")
    monkeypatch.setenv("WHETSTONE_RECEIPT_SIGNING_KEY", str(private_key))
    real_access = receipts.os.access
    monkeypatch.setattr(
        receipts.os,
        "access",
        lambda path, mode: False if path == private_key else real_access(path, mode),
    )

    assert receipts.signing_status()["ready"] is False


def test_verify_receipt_cli_uses_explicit_trust_bundle(tmp_path, monkeypatch, capsys):
    private_key = _key(tmp_path, "active")
    monkeypatch.setenv("WHETSTONE_RECEIPT_SIGNING_KEY", str(private_key))
    receipt = attest_receipt(_receipt(), "report_card", challenge="cli-12345678", now=100)
    receipt_path = tmp_path / "receipt.json"
    keys_path = tmp_path / "keys.json"
    receipt_path.write_text(__import__("json").dumps(receipt), encoding="utf-8")
    keys_path.write_text(__import__("json").dumps(receipt_key_bundle()), encoding="utf-8")

    code = main([
        "verify-receipt",
        "--receipt", str(receipt_path),
        "--keys", str(keys_path),
        "--expected-challenge", "cli-12345678",
        "--at-unix", "101",
    ])

    assert code == 0
    assert __import__("json").loads(capsys.readouterr().out)["valid"] is True

    code = main([
        "verify-receipt",
        "--receipt", str(receipt_path),
        "--keys", "http://example.com/keys.json",
        "--at-unix", "101",
    ])
    assert code == 1
    assert "require HTTPS" in __import__("json").loads(capsys.readouterr().out)["error"]


def test_hosted_verification_is_signed_and_origin_pinned(tmp_path, monkeypatch):
    private_key = _key(tmp_path, "hosted")
    monkeypatch.setenv("WHETSTONE_RECEIPT_SIGNING_KEY", str(private_key))
    receipt = attest_receipt(_receipt(), "report_card", challenge="hosted-12345678")
    bundle = receipt_key_bundle()
    monkeypatch.setattr(hosted, "_request_json", lambda *_args, **_kwargs: bundle)

    verified = verify_hosted_receipt(
        "https://whetstone.cyberelf.link",
        receipt,
        expected_challenge="hosted-12345678",
    )
    assert verified["valid"] is True
    with pytest.raises(ReceiptVerificationError, match="key-bundle issuer"):
        verify_hosted_receipt(
            "https://other.example",
            receipt,
            expected_challenge="hosted-12345678",
        )


def test_hosted_verification_rejects_unsigned_and_remote_http():
    unsigned = {"attestation": {"status": "unsigned"}}
    with pytest.raises(ReceiptVerificationError, match="not signed"):
        verify_hosted_receipt(
            "https://whetstone.cyberelf.link",
            unsigned,
            expected_challenge="hosted-12345678",
        )
    assert verify_hosted_receipt(
        "http://127.0.0.1:8988",
        unsigned,
        expected_challenge="hosted-12345678",
        allow_unsigned=True,
    ) is None
    with pytest.raises(ReceiptVerificationError, match="only from a loopback"):
        verify_hosted_receipt(
            "https://whetstone.cyberelf.link",
            unsigned,
            expected_challenge="hosted-12345678",
            allow_unsigned=True,
        )

    signed_shape = {"attestation": {"status": "signed"}}
    with pytest.raises(ReceiptVerificationError, match="require HTTPS"):
        verify_hosted_receipt(
            "http://example.com",
            signed_shape,
            expected_challenge="hosted-12345678",
        )
