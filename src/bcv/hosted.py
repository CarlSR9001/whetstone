"""One-command clients for Whetstone's hosted report-card surfaces."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import urlsplit

from bcv.candidates import CandidateError
from bcv.receipts import KEY_SET_PATH, ReceiptVerificationError, verify_receipt
from bcv.transformers_client import extract_json


class HostedServiceError(RuntimeError):
    def __init__(self, status: int, payload: dict) -> None:
        self.status = status
        self.payload = payload
        super().__init__(str(payload.get("error", f"hosted service returned HTTP {status}")))


def _receipt_issuer(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    parsed = urlsplit(normalized)
    if (
        not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in ("", "/")
    ):
        raise ReceiptVerificationError("hosted URL must be an origin with no credentials, query, or path")
    loopback = parsed.hostname.lower() in {"127.0.0.1", "::1", "localhost"}
    if parsed.scheme != "https" and not (parsed.scheme == "http" and loopback):
        raise ReceiptVerificationError("receipt keys require HTTPS except on loopback")
    return normalized


def _request_json(
    base_url: str,
    path: str,
    *,
    payload: dict | None = None,
    timeout: float = 30.0,
) -> dict:
    url = base_url.rstrip("/") + path
    request = urllib.request.Request(
        url,
        data=None if payload is None else json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read())
    except urllib.error.HTTPError as error:
        try:
            body = json.loads(error.read())
        except (ValueError, json.JSONDecodeError):
            body = {"error": f"hosted service returned HTTP {error.code}"}
        raise HostedServiceError(error.code, body) from error
    except urllib.error.URLError as error:
        raise HostedServiceError(0, {"error": f"hosted service is unreachable: {error.reason}"}) from error
    if not isinstance(body, dict):
        raise HostedServiceError(0, {"error": "hosted service returned non-object JSON"})
    return body


def _submit_with_retry(
    base_url: str,
    path: str,
    payload: dict,
    *,
    timeout: float,
    retries: int,
) -> dict:
    if retries < 0:
        raise ValueError("retries must be non-negative")
    for attempt in range(retries + 1):
        try:
            return _request_json(base_url, path, payload=payload, timeout=timeout)
        except HostedServiceError as error:
            retryable = (
                error.status == 429
                and error.payload.get("retryable") is True
                and isinstance(error.payload.get("retry_after_seconds"), int)
            )
            if not retryable or attempt >= retries:
                raise
            delay = error.payload.get("retry_after_seconds", 1)
            delay = delay if isinstance(delay, int) and 0 <= delay <= 10 else 1
            time.sleep(delay)
    raise AssertionError("retry loop exhausted without returning or raising")


def run_report_card(
    base_url: str,
    candidate: Any,
    *,
    challenge: str,
    request_timeout: float = 30.0,
    start_retries: int = 12,
    submit_retries: int = 5,
) -> dict:
    session = _submit_with_retry(
        base_url,
        "/api/report-card/start",
        {"challenge": challenge},
        timeout=request_timeout,
        retries=start_retries,
    )
    answers: dict[str, str] = {}
    for item in session.get("items", []):
        item_id = item.get("item_id")
        prompt = item.get("prompt")
        if not isinstance(item_id, str) or not isinstance(prompt, str):
            raise HostedServiceError(0, {"error": "report-card session contains a malformed item"})
        answers[item_id] = candidate.generate_text(prompt, temperature=0.0)
    return _submit_with_retry(
        base_url,
        "/api/report-card/submit",
        {"session_id": session["session_id"], "answers": answers},
        timeout=request_timeout,
        retries=submit_retries,
    )


def _task_prompt(task: dict) -> str:
    return (
        "You are editing a virtual repository. Return JSON only with this exact shape: "
        '{"writes":{"path":"full replacement text"},"deletes":["path"]}. '
        "Every write must contain the complete final file text. Follow the declared scope; "
        "unlisted paths are immutable.\n\nTASK:\n"
        + json.dumps(task, indent=2, sort_keys=True)
    )


def _patch_answer(candidate: Any, task: dict) -> dict:
    raw = candidate.generate_text(_task_prompt(task), temperature=0.0)
    parsed = extract_json(raw)
    if not isinstance(parsed, dict):
        raise CandidateError("Open Bench candidate did not return a JSON patch object")
    patch = parsed.get("patch", parsed)
    if not isinstance(patch, dict) or not isinstance(patch.get("writes", {}), dict):
        raise CandidateError("Open Bench candidate patch must contain an object under writes")
    if not isinstance(patch.get("deletes", []), list):
        raise CandidateError("Open Bench candidate patch must contain an array under deletes")
    return {"writes": patch.get("writes", {}), "deletes": patch.get("deletes", [])}


def run_open_bench(
    base_url: str,
    baseline: Any,
    candidate: Any,
    *,
    baseline_manifest: dict,
    candidate_manifest: dict,
    challenge: str,
    publish: bool = False,
    request_timeout: float = 30.0,
) -> dict:
    session = _request_json(
        base_url,
        "/api/open-bench/start",
        payload={"challenge": challenge},
        timeout=request_timeout,
    )
    tasks = session.get("tasks")
    if not isinstance(tasks, list):
        raise HostedServiceError(0, {"error": "Open Bench session contains no task array"})
    baseline_answers = {task["item_id"]: _patch_answer(baseline, task) for task in tasks}
    candidate_answers = {task["item_id"]: _patch_answer(candidate, task) for task in tasks}
    return _request_json(
        base_url,
        "/api/open-bench/submit",
        payload={
            "session_id": session["session_id"],
            "baseline_manifest": baseline_manifest,
            "candidate_manifest": candidate_manifest,
            "baseline_answers": baseline_answers,
            "candidate_answers": candidate_answers,
            "publish": publish,
            "attestation": publish,
        },
        timeout=request_timeout,
    )


def verify_hosted_receipt(
    base_url: str,
    receipt: dict,
    *,
    expected_challenge: str,
    timeout: float = 10.0,
    allow_unsigned: bool = False,
) -> dict | None:
    issuer = _receipt_issuer(base_url)
    attestation = receipt.get("attestation")
    if not isinstance(attestation, dict) or attestation.get("status") != "signed":
        hostname = (urlsplit(issuer).hostname or "").lower()
        if allow_unsigned and hostname in {"127.0.0.1", "::1", "localhost"}:
            return None
        if allow_unsigned:
            raise ReceiptVerificationError("unsigned receipts are allowed only from a loopback service")
        raise ReceiptVerificationError("hosted receipt is not signed")
    bundle = _request_json(base_url, KEY_SET_PATH, timeout=timeout)
    if bundle.get("issuer") != issuer:
        raise ReceiptVerificationError("hosted key-bundle issuer does not match the requested service")
    return verify_receipt(
        receipt,
        bundle,
        expected_challenge=expected_challenge,
        expected_issuer=issuer,
    )
