"""Canonical, issuer-signed receipt envelopes backed by OpenSSH SSHSIG.

The Python core stays dependency-free: Ed25519 operations are delegated to the
platform OpenSSH implementation. A signature authenticates Whetstone's grading
result and build identity. It does not authenticate a caller's self-attested
model or harness label.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import secrets
import subprocess
import tempfile
import time
from pathlib import Path

from bcv._version import __version__, build_commit


ATTESTATION_FORMAT = "whetstone-sshsig-v1"
SIGNATURE_NAMESPACE = "whetstone-receipt-v1"
SIGNATURE_PRINCIPAL = "whetstone"
DEFAULT_ISSUER = "https://whetstone.cyberelf.link"
DEFAULT_TTL_SECONDS = 86_400
KEY_SET_PATH = "/.well-known/whetstone-receipt-keys.json"
_CHALLENGE = re.compile(r"^[A-Za-z0-9._~:/+-]{8,128}$")
_UNCOMMITTED_FIELDS = {"attestation", "receipt_sha256", "request_id"}


class ReceiptVerificationError(ValueError):
    """A receipt is malformed, altered, expired, or not signed by an allowed key."""


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def receipt_core(receipt: dict) -> dict:
    core = copy.deepcopy(receipt)
    for field in _UNCOMMITTED_FIELDS:
        core.pop(field, None)
    return core


def receipt_sha256(receipt: dict) -> str:
    return sha256_json(receipt_core(receipt))


def session_challenge(value: object = None) -> str:
    if value is None:
        return secrets.token_hex(16)
    if not isinstance(value, str) or not _CHALLENGE.fullmatch(value):
        raise ValueError(
            "challenge must be 8-128 characters using letters, digits, or . _ ~ : / + -"
        )
    return value


def _normalized_public_key(value: str) -> str:
    fields = value.strip().split()
    if len(fields) < 2 or fields[0] != "ssh-ed25519":
        raise ValueError("receipt keys must be OpenSSH Ed25519 public keys")
    return f"{fields[0]} {fields[1]}"


def key_id(public_key: str) -> str:
    normalized = _normalized_public_key(public_key)
    return "sha256:" + hashlib.sha256(normalized.encode("ascii")).hexdigest()


def _private_key_path() -> Path | None:
    raw = os.environ.get("WHETSTONE_RECEIPT_SIGNING_KEY", "").strip()
    return Path(raw) if raw else None


def _public_key_for_private(private_key: Path) -> str:
    public_path = Path(str(private_key) + ".pub")
    if public_path.is_file():
        return _normalized_public_key(public_path.read_text(encoding="utf-8"))
    process = subprocess.run(
        ["ssh-keygen", "-y", "-f", str(private_key)],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return _normalized_public_key(process.stdout)


def _public_keys() -> list[tuple[str, str]]:
    keys: list[str] = []
    private_key = _private_key_path()
    if private_key is not None and private_key.is_file():
        keys.append(_public_key_for_private(private_key))
    history = os.environ.get("WHETSTONE_RECEIPT_TRUSTED_KEY_DIR", "").strip()
    if history:
        for path in sorted(Path(history).glob("*.pub")):
            try:
                keys.append(_normalized_public_key(path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
    unique: dict[str, str] = {}
    for public_key in keys:
        unique[key_id(public_key)] = public_key
    return sorted(unique.items())


def receipt_key_bundle() -> dict:
    issuer = os.environ.get("WHETSTONE_RECEIPT_ISSUER", DEFAULT_ISSUER).rstrip("/")
    current_key = None
    private_key = _private_key_path()
    if private_key is not None and private_key.is_file():
        current_key = key_id(_public_key_for_private(private_key))
    return {
        "schema_version": 1,
        "issuer": issuer,
        "format": ATTESTATION_FORMAT,
        "namespace": SIGNATURE_NAMESPACE,
        "current_key_id": current_key,
        "keys": [
            {
                "key_id": identifier,
                "algorithm": "ssh-ed25519",
                "public_key": public_key,
                "status": "active" if identifier == current_key else "retired",
            }
            for identifier, public_key in _public_keys()
        ],
    }


def signing_status() -> dict:
    private_key = _private_key_path()
    if private_key is None:
        return {"configured": False, "ready": False, "key_id": None, "format": ATTESTATION_FORMAT}
    try:
        public_key = _public_key_for_private(private_key)
    except (OSError, ValueError, subprocess.SubprocessError):
        return {"configured": True, "ready": False, "key_id": None, "format": ATTESTATION_FORMAT}
    ready = private_key.is_file() and os.access(private_key, os.R_OK)
    return {
        "configured": True,
        "ready": ready,
        "key_id": key_id(public_key),
        "format": ATTESTATION_FORMAT,
    }


def _commitments(receipt: dict) -> dict:
    commitments = {"receipt_sha256": receipt["receipt_sha256"]}
    for field in (
        "answers_sha256",
        "cohort_sha256",
        "item_set_sha256",
        "source_evidence_sha256",
        "grading_evidence_sha256",
    ):
        value = receipt.get(field)
        if isinstance(value, str):
            commitments[field] = value
    policy_value = receipt.get("policy", receipt.get("grading_policy"))
    if policy_value is not None:
        commitments["policy_sha256"] = sha256_json(policy_value)
    for field in ("baseline_manifest", "candidate_manifest"):
        value = receipt.get(field)
        if value is not None:
            commitments[f"{field}_sha256"] = sha256_json(value)
    return commitments


def _sign(payload: dict, private_key: Path) -> str:
    process = subprocess.run(
        [
            "ssh-keygen",
            "-Y",
            "sign",
            "-f",
            str(private_key),
            "-n",
            SIGNATURE_NAMESPACE,
            "-",
        ],
        input=canonical_bytes(payload),
        capture_output=True,
        check=True,
        timeout=10,
    )
    return process.stdout.decode("ascii")


def attest_receipt(
    receipt: dict,
    receipt_type: str,
    *,
    challenge: str,
    now: int | None = None,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
) -> dict:
    """Attach a signed envelope in place and return the receipt."""
    challenge = session_challenge(challenge)
    receipt["receipt_sha256"] = receipt_sha256(receipt)
    private_key = _private_key_path()
    issuer = os.environ.get("WHETSTONE_RECEIPT_ISSUER", DEFAULT_ISSUER).rstrip("/")
    if private_key is None:
        receipt["attestation"] = {
            "status": "unsigned",
            "format": ATTESTATION_FORMAT,
            "reason": "receipt signing key is not configured",
        }
        return receipt
    try:
        public_key = _public_key_for_private(private_key)
        identifier = key_id(public_key)
        issued_at = int(time.time()) if now is None else int(now)
        payload = {
            "schema_version": 1,
            "format": ATTESTATION_FORMAT,
            "namespace": SIGNATURE_NAMESPACE,
            "issuer": issuer,
            "key_id": identifier,
            "key_set_url": issuer + KEY_SET_PATH,
            "receipt_type": receipt_type,
            "issued_at_unix": issued_at,
            "expires_at_unix": issued_at + int(ttl_seconds),
            "nonce": secrets.token_hex(16),
            "challenge": challenge,
            "service": {
                "name": "whetstone-tools",
                "version": __version__,
                "build_commit": build_commit(),
            },
            "commitments": _commitments(receipt),
            "claims": {
                "issuer_authenticated": True,
                "subject_identity_verified": bool(receipt.get("identity_verified", False)),
            },
        }
        signature = _sign(payload, private_key)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        receipt["attestation"] = {
            "status": "error",
            "format": ATTESTATION_FORMAT,
            "reason": type(error).__name__,
        }
        return receipt
    receipt["attestation"] = {
        "status": "signed",
        "format": ATTESTATION_FORMAT,
        "payload": payload,
        "signature": signature,
    }
    return receipt


def _key_map(bundle: dict) -> dict[str, tuple[str, str]]:
    rows = bundle.get("keys") if isinstance(bundle, dict) else None
    if not isinstance(rows, list):
        raise ReceiptVerificationError("key bundle must contain a keys array")
    keys: dict[str, tuple[str, str]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        public_key = row.get("public_key")
        identifier = row.get("key_id")
        status = row.get("status")
        if (
            not isinstance(public_key, str)
            or not isinstance(identifier, str)
            or status not in {"active", "retired", "revoked"}
        ):
            continue
        normalized = _normalized_public_key(public_key)
        if key_id(normalized) != identifier:
            raise ReceiptVerificationError(f"key bundle id does not match public key: {identifier}")
        keys[identifier] = (normalized, status)
    return keys


def verify_receipt(
    receipt: dict,
    key_bundle: dict,
    *,
    expected_challenge: str | None = None,
    expected_issuer: str | None = DEFAULT_ISSUER,
    now: int | None = None,
    allow_expired: bool = False,
    allow_retired: bool = False,
) -> dict:
    if not isinstance(receipt, dict):
        raise ReceiptVerificationError("receipt must be a JSON object")
    attestation = receipt.get("attestation")
    if not isinstance(attestation, dict) or attestation.get("status") != "signed":
        raise ReceiptVerificationError("receipt does not contain a signed attestation")
    if attestation.get("format") != ATTESTATION_FORMAT:
        raise ReceiptVerificationError("unsupported receipt attestation format")
    payload = attestation.get("payload")
    signature = attestation.get("signature")
    if not isinstance(payload, dict) or not isinstance(signature, str):
        raise ReceiptVerificationError("signed attestation is missing payload or signature")
    if payload.get("format") != ATTESTATION_FORMAT or payload.get("namespace") != SIGNATURE_NAMESPACE:
        raise ReceiptVerificationError("signed payload format or namespace is invalid")
    if expected_issuer is not None and payload.get("issuer") != expected_issuer.rstrip("/"):
        raise ReceiptVerificationError("receipt issuer does not match the expected issuer")
    if expected_challenge is not None and payload.get("challenge") != session_challenge(expected_challenge):
        raise ReceiptVerificationError("receipt challenge does not match the expected session")
    observed_digest = receipt_sha256(receipt)
    if receipt.get("receipt_sha256") != observed_digest:
        raise ReceiptVerificationError("receipt content does not match receipt_sha256")
    commitments = payload.get("commitments")
    if not isinstance(commitments, dict) or commitments.get("receipt_sha256") != observed_digest:
        raise ReceiptVerificationError("signed payload does not commit to this receipt")
    issued = payload.get("issued_at_unix")
    expires = payload.get("expires_at_unix")
    if not isinstance(issued, int) or not isinstance(expires, int) or expires <= issued:
        raise ReceiptVerificationError("receipt validity window is malformed")
    check_time = int(time.time()) if now is None else int(now)
    if issued > check_time + 300:
        raise ReceiptVerificationError("receipt was issued too far in the future")
    expired = check_time > expires
    if expired and not allow_expired:
        raise ReceiptVerificationError("receipt attestation has expired")
    identifier = payload.get("key_id")
    keys = _key_map(key_bundle)
    key_record = keys.get(identifier)
    if key_record is None:
        raise ReceiptVerificationError(f"receipt key is not present in the key bundle: {identifier}")
    public_key, key_status = key_record
    if key_status == "revoked":
        raise ReceiptVerificationError("receipt key is revoked")
    if key_status == "retired" and not allow_retired:
        raise ReceiptVerificationError("receipt key is retired; archival verification requires allow_retired")
    if key_status == "active" and key_bundle.get("current_key_id") != identifier:
        raise ReceiptVerificationError("active receipt key is not the bundle's current key")
    try:
        with tempfile.TemporaryDirectory(prefix="whetstone-receipt-") as raw:
            root = Path(raw)
            signature_path = root / "receipt.sig"
            allowed_path = root / "allowed_signers"
            signature_path.write_text(signature, encoding="ascii", newline="\n")
            allowed_path.write_text(
                f"{SIGNATURE_PRINCIPAL} {public_key}\n",
                encoding="ascii",
                newline="\n",
            )
            process = subprocess.run(
                [
                    "ssh-keygen",
                    "-Y",
                    "verify",
                    "-f",
                    str(allowed_path),
                    "-I",
                    SIGNATURE_PRINCIPAL,
                    "-n",
                    SIGNATURE_NAMESPACE,
                    "-s",
                    str(signature_path),
                ],
                input=canonical_bytes(payload),
                capture_output=True,
                timeout=10,
            )
    except (OSError, subprocess.SubprocessError) as error:
        raise ReceiptVerificationError(f"signature verifier failed: {type(error).__name__}") from error
    if process.returncode != 0:
        raise ReceiptVerificationError("receipt signature is invalid")
    return {
        "valid": True,
        "expired": expired,
        "issuer": payload["issuer"],
        "receipt_type": payload["receipt_type"],
        "key_id": identifier,
        "key_status": key_status,
        "challenge": payload["challenge"],
        "receipt_sha256": observed_digest,
        "service": payload["service"],
        "claims": payload["claims"],
    }
