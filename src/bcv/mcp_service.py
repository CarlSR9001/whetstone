"""Remote MCP endpoint for the public toolbox: stdlib-only Streamable HTTP.

One POST route (`/mcp`) speaks JSON-RPC 2.0 in stateless mode: every response
is a single ``application/json`` body, no SSE stream, no server-issued session
id. That is the smallest spec-compliant surface, it adds zero dependencies to
the VPS, and it inherits the toolbox's trust boundary:

- Tier 0 tools wrap the existing stateless product tools (the caller brings
  every byte of data; nothing persists).
- Tier 1 tools drive the disposable report-card hatchery in bcv.ephemeral.
- No tool can reach a private bank; none is loaded in this process, ever.

Rate limits are shared with the REST surface via bcv.ratelimit, so switching
protocols does not double anyone's budget.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from bcv.ephemeral import TierError, hatchery
from bcv.product_tools import (
    ProductInputError,
    audit_leakage,
    bank_health,
    catalog,
    gate_results,
    hunt_counterexample,
    inspect_promotion,
    memory_relevance,
    replay_trace,
    safe_patch,
)
from bcv.ratelimit import HUNTER_LIMIT, HUNTER_SLOT

SUPPORTED_PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
SERVER_INFO = {"name": "whetstone-tools", "version": "0.3.0"}

INSTRUCTIONS = (
    "Whetstone's public verifier toolbox as MCP tools. Two tiers. Tier 0 is stateless: "
    "you supply the data (exam rows, paired results, documents, event logs) and get back "
    "audits, promotion verdicts, patches, or counterexamples; nothing is stored. Tier 1 is "
    "the disposable report card: report_card_start hands your agent a small graph-repair "
    "exam minted from the repository's public frontier, report_card_submit grades it by "
    "checker spec (any verified strict refinement passes; no answer key exists) and "
    "destroys the session. No private exam bank is loaded in this service, so no tool can "
    "leak one. Complete request/response examples for every Tier 0 tool: GET /api/examples."
)


def _loose_object_schema(description: str) -> dict:
    return {"type": "object", "description": description, "additionalProperties": True}


def _hunt_with_limits(payload: dict, client_ip: str) -> dict:
    if not HUNTER_LIMIT.allow(client_ip):
        raise TierError("counterexample search limit exceeded for this address; try again later")
    if not HUNTER_SLOT.acquire(blocking=False):
        raise TierError("counterexample worker busy; retry shortly", retryable=True)
    try:
        return hunt_counterexample(payload)
    finally:
        HUNTER_SLOT.release()


def _report_card_start(payload: dict, client_ip: str) -> dict:
    return hatchery().start_session(client_ip)


def _report_card_submit(payload: dict, client_ip: str) -> dict:
    session_id = payload.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise TierError("session_id (string) is required")
    return hatchery().submit(session_id, payload.get("answers") or {}, client_ip)


# name -> (handler(payload, client_ip), description, inputSchema)
TOOLS: dict[str, tuple[Callable[[dict, str], dict], str, dict]] = {
    "inspect_promotion": (
        lambda p, ip: inspect_promotion(p),
        "Quarantine declared exposure, compare paired baseline/candidate outcomes on the "
        "clean remainder, and issue a promotion receipt. Bring your own exam rows, exposure "
        "records, and per-item results. Full example: GET /api/examples key 'inspector'.",
        _loose_object_schema("Keys: exam, exposure, baseline, candidate, baseline_name, candidate_name, policy."),
    ),
    "audit_leakage": (
        lambda p, ip: audit_leakage(p),
        "Exact declared-exposure audit over your exam rows: row identity, behavioral "
        "fingerprints for graph-DSL expressions, text-similarity review flags, and a clean "
        "exam export. Full example: GET /api/examples key 'leakage'.",
        _loose_object_schema("Keys: exam, exposure, fingerprint_max_n, similarity_threshold."),
    ),
    "promotion_gate": (
        lambda p, ip: gate_results(p),
        "PASS, HOLD, or BLOCK from paired per-item results: gains, regressions, exact "
        "McNemar p-value, per-domain breakdown. Full example: GET /api/examples key 'gate'.",
        _loose_object_schema("Keys: baseline, candidate, domains, baseline_name, candidate_name, policy."),
    ),
    "bank_health": (
        lambda p, ip: bank_health(p),
        "Item-lifecycle diagnostics over your grading history: discriminators, saturated and "
        "flaky items, frontier gaps. Full example: GET /api/examples key 'health'.",
        _loose_object_schema("Keys: items, history."),
    ),
    "safe_patch": (
        lambda p, ip: safe_patch(p),
        "Apply a section-scoped Markdown patch under conservation checks (untouched sections "
        "stay byte-identical; protected tokens preserved). Full example: GET /api/examples "
        "key 'safepatch'.",
        _loose_object_schema("Keys: document, reason, operations."),
    ),
    "counterexample_hunt": (
        _hunt_with_limits,
        "Bounded simulated-annealing search for a graph counterexample inside a DSL "
        "predicate class, with an exact certificate when found. CPU-bounded and strictly "
        "rate-limited. Full example: GET /api/examples key 'counterexample'.",
        _loose_object_schema("Keys: expression, ns, restarts, steps, seed."),
    ),
    "memory_relevance": (
        lambda p, ip: memory_relevance(p),
        "Compare query-free salience against objective-conditioned relevance for a set of "
        "memories under a token budget. Full example: GET /api/examples key 'memory'.",
        _loose_object_schema("Keys: objective, objective_entities, context_entities, token_budget, memories."),
    ),
    "replay_trace": (
        lambda p, ip: replay_trace(p),
        "Turn reasoning-emulator control events into checkpoints, rewinds, notes, and a "
        "timeline. Full example: GET /api/examples key 'replay'.",
        _loose_object_schema("Keys: events, notes."),
    ),
    "report_card_start": (
        _report_card_start,
        "TIER 1: start a disposable report-card session. Returns exam items (graph-repair "
        "prompts minted from the repository's public frontier) for THIS agent to answer. "
        "Answer every item, then call report_card_submit exactly once. Sessions are "
        "one-shot, expire in 15 minutes, and are strictly rate-limited. This demonstrates "
        "the promotion-gate mechanism on disposable items; it is not a private-bank "
        "credential.",
        {"type": "object", "properties": {}, "additionalProperties": False},
    ),
    "report_card_submit": (
        _report_card_submit,
        "TIER 1: submit answers for a report-card session and receive the graded report "
        "(per-item verdicts, per-domain totals, SHA-256 commitments). Grading is by checker "
        "spec: any verified strict refinement of the item's predicate passes; no answer key "
        "exists. The session is destroyed by this call.",
        {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "answers": {
                    "type": "object",
                    "description": "item_id -> answer (a DSL predicate, or the JSON reply the prompt asked for)",
                    "additionalProperties": {"type": "string"},
                },
            },
            "required": ["session_id", "answers"],
            "additionalProperties": False,
        },
    ),
    "about_whetstone": (
        lambda p, ip: {
            "catalog": catalog(),
            "mcp_tiers": {
                "tier0": "stateless analyses of caller-supplied data",
                "tier1": "disposable report-card sessions (report_card_start / report_card_submit)",
            },
            "source": "https://github.com/CarlSR9001/whetstone",
            "site": "https://whetstone.cyberelf.link/",
        },
        "What this service is: the tool catalog, the tier boundaries, and where the source lives.",
        {"type": "object", "properties": {}, "additionalProperties": False},
    ),
}


def _tool_list() -> list[dict]:
    return [
        {"name": name, "description": description, "inputSchema": schema}
        for name, (_, description, schema) in TOOLS.items()
    ]


def _rpc_error(request_id: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def _rpc_result(request_id: Any, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _tool_failure(request_id: Any, message: str) -> dict:
    # Tool-level failures are results with isError, not protocol errors.
    return _rpc_result(request_id, {
        "content": [{"type": "text", "text": message}],
        "isError": True,
    })


def handle_mcp(payload: Any, client_ip: str) -> tuple[int, dict | None]:
    """Dispatch one JSON-RPC message. Returns (http_status, body_or_None)."""
    if isinstance(payload, list):
        return 400, _rpc_error(None, -32600, "JSON-RPC batching is not supported; send one message per request")
    if not isinstance(payload, dict) or payload.get("jsonrpc") != "2.0":
        return 400, _rpc_error(None, -32600, "body must be a JSON-RPC 2.0 message")

    method = payload.get("method")
    request_id = payload.get("id")
    params = payload.get("params") or {}
    if not isinstance(params, dict):
        return 200, _rpc_error(request_id, -32602, "params must be an object")

    if request_id is None:
        # Notification (e.g. notifications/initialized): accept, no body.
        return 202, None

    if method == "initialize":
        requested = str(params.get("protocolVersion", ""))
        version = requested if requested in SUPPORTED_PROTOCOL_VERSIONS else SUPPORTED_PROTOCOL_VERSIONS[0]
        return 200, _rpc_result(request_id, {
            "protocolVersion": version,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": SERVER_INFO,
            "instructions": INSTRUCTIONS,
        })
    if method == "ping":
        return 200, _rpc_result(request_id, {})
    if method == "tools/list":
        return 200, _rpc_result(request_id, {"tools": _tool_list()})
    if method == "tools/call":
        name = params.get("name")
        entry = TOOLS.get(name) if isinstance(name, str) else None
        if entry is None:
            return 200, _rpc_error(request_id, -32602, f"unknown tool {name!r}")
        arguments = params.get("arguments") or {}
        if not isinstance(arguments, dict):
            return 200, _rpc_error(request_id, -32602, "arguments must be an object")
        handler = entry[0]
        try:
            result = handler(arguments, client_ip)
        except (ProductInputError, TierError) as error:
            return 200, _tool_failure(request_id, str(error))
        except Exception as error:  # fail closed without reflecting internals
            print(f"mcp tool {name} failed: {type(error).__name__}", flush=True)
            return 200, _tool_failure(request_id, "internal processing error")
        return 200, _rpc_result(request_id, {
            "content": [{"type": "text", "text": json.dumps(result, sort_keys=True, indent=1)}],
            "structuredContent": result,
            "isError": False,
        })
    return 200, _rpc_error(request_id, -32601, f"method {method!r} not found")
