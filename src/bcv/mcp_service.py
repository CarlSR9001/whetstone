"""Remote MCP endpoint for the public toolbox: stdlib-only Streamable HTTP.

One POST route (`/mcp`) speaks JSON-RPC 2.0 with single ``application/json``
responses, no SSE stream, and no MCP session header. That is the smallest
spec-compliant surface, adds zero dependencies to the VPS, and inherits the
toolbox's trust boundary:

- Tier 0 tools wrap the existing stateless product tools (the caller brings
  every byte of data; request bodies and results are not persisted).
- Tier 1 tools drive the disposable report-card hatchery in bcv.ephemeral.
- Tier 2 drives a paired procedural benchmark; publication is explicit and
  writes only a sanitized public receipt, never task contents or answers.
- No tool can reach a private bank; none is loaded in this process, ever.

Rate limits are shared with the REST surface via bcv.ratelimit, so switching
protocols does not double anyone's budget.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from bcv._version import __version__, build_commit
from bcv.ephemeral import TierError, hatchery
from bcv.open_bench import OpenBenchError, open_bench
from bcv.product_tools import (
    ProductInputError,
    audit_leakage,
    bank_health,
    catalog,
    gate_results,
    hunt_counterexample,
    input_schemas,
    inspect_promotion,
    memory_relevance,
    replay_trace,
    safe_patch,
)
from bcv.ratelimit import HUNTER_LIMIT, HUNTER_SLOT
from bcv.stats import STATS

SUPPORTED_PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
SERVER_INFO = {"name": "whetstone-tools", "version": __version__}

INSTRUCTIONS = (
    "Whetstone's public verifier toolbox as MCP tools. Three tiers. Tier 0 is stateless: "
    "you supply the data (exam rows, paired results, documents, event logs) and get back "
    "audits, promotion verdicts, patches, or counterexamples; payloads and results are not "
    "persisted, while operational counters and standard access logs are retained. Tier 1 is "
    "the disposable report card: report_card_start hands your agent a small graph-repair "
    "exam minted from the repository's public frontier, report_card_submit grades it by "
    "checker spec (any verified strict refinement passes; no answer key exists) and "
    "destroys the session. Tier 2 is Open Promotion Bench: open_bench_start gives a paired "
    "baseline/candidate scope-integrity cohort, open_bench_submit grades both answer maps, "
    "and open_bench_leaderboard returns opt-in public receipts. Published entries retain only "
    "the self-attested manifests and sanitized receipts, never tasks or answers. No private "
    "exam bank is loaded in this service, so no tool can leak one. Complete request/response "
    "examples for every Tier 0 tool: GET /api/examples. "
    "Full agent documentation: https://whetstone.cyberelf.link/for-agents — installable "
    "skill file: https://whetstone.cyberelf.link/skill.md"
)


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


def _open_bench_start(payload: dict, client_ip: str) -> dict:
    if payload:
        raise OpenBenchError("open_bench_start takes no arguments")
    return open_bench().start_session(client_ip)


def _open_bench_submit(payload: dict, client_ip: str) -> dict:
    return open_bench().submit(payload, client_ip)


def _open_bench_leaderboard(payload: dict, client_ip: str) -> dict:
    if payload:
        raise OpenBenchError("open_bench_leaderboard takes no arguments")
    return open_bench().leaderboard()


# name -> (handler(payload, client_ip), description, inputSchema)
TIER0_INPUT_SCHEMAS = input_schemas()
TOOLS: dict[str, tuple[Callable[[dict, str], dict], str, dict]] = {
    "inspect_promotion": (
        lambda p, ip: inspect_promotion(p),
        "Quarantine declared exposure, compare paired baseline/candidate outcomes on the "
        "clean remainder, and issue a promotion receipt. Bring your own exam rows, exposure "
        "records, and per-item results. Full example: GET /api/examples key 'inspector'.",
        TIER0_INPUT_SCHEMAS["inspect_promotion"],
    ),
    "audit_leakage": (
        lambda p, ip: audit_leakage(p),
        "Exact declared-exposure audit over your exam rows: row identity, behavioral "
        "fingerprints for graph-DSL expressions, text-similarity review flags, and a clean "
        "exam export. Full example: GET /api/examples key 'leakage'.",
        TIER0_INPUT_SCHEMAS["audit_leakage"],
    ),
    "promotion_gate": (
        lambda p, ip: gate_results(p),
        "PASS, HOLD, or BLOCK from paired per-item results: gains, regressions, exact "
        "McNemar p-value, per-domain breakdown. Full example: GET /api/examples key 'gate'.",
        TIER0_INPUT_SCHEMAS["promotion_gate"],
    ),
    "bank_health": (
        lambda p, ip: bank_health(p),
        "Item-lifecycle diagnostics over your grading history: discriminators, saturated and "
        "flaky items, frontier gaps. Full example: GET /api/examples key 'health'.",
        TIER0_INPUT_SCHEMAS["bank_health"],
    ),
    "safe_patch": (
        lambda p, ip: safe_patch(p),
        "Apply a section-scoped Markdown patch under conservation checks (untouched sections "
        "stay byte-identical; protected tokens preserved). Full example: GET /api/examples "
        "key 'safepatch'.",
        TIER0_INPUT_SCHEMAS["safe_patch"],
    ),
    "counterexample_hunt": (
        _hunt_with_limits,
        "Bounded simulated-annealing search for a graph counterexample inside a DSL "
        "predicate class, with an exact certificate when found. CPU-bounded and strictly "
        "rate-limited. Full example: GET /api/examples key 'counterexample'.",
        TIER0_INPUT_SCHEMAS["counterexample_hunt"],
    ),
    "memory_relevance": (
        lambda p, ip: memory_relevance(p),
        "Compare query-free salience against objective-conditioned relevance for a set of "
        "memories under a token budget. Full example: GET /api/examples key 'memory'.",
        TIER0_INPUT_SCHEMAS["memory_relevance"],
    ),
    "replay_trace": (
        lambda p, ip: replay_trace(p),
        "Turn reasoning-emulator control events into checkpoints, rewinds, notes, and a "
        "timeline. Full example: GET /api/examples key 'replay'.",
        TIER0_INPUT_SCHEMAS["replay_trace"],
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
    "open_bench_start": (
        _open_bench_start,
        "TIER 2: start a one-shot Open Promotion Bench session. Returns six fresh virtual-"
        "repository scope-integrity tasks. Run a baseline and candidate independently on the "
        "same cohort, then submit both answer maps with open_bench_submit. This is an open, "
        "procedural, self-attested track rather than a private-bank credential.",
        {"type": "object", "properties": {}, "additionalProperties": False},
    ),
    "open_bench_submit": (
        _open_bench_submit,
        "TIER 2: grade paired baseline and candidate patches, count gains/regressions/ties, "
        "and issue PASS/HOLD/BLOCK. Set publish=true plus attestation=true to append only the "
        "safe manifests and sanitized receipt to the public board; tasks and answers are never "
        "persisted.",
        {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "baseline_manifest": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "model": {"type": "string"},
                        "harness": {"type": "string"},
                        "version": {"type": "string"},
                    },
                    "required": ["name"],
                    "additionalProperties": False,
                },
                "candidate_manifest": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "model": {"type": "string"},
                        "harness": {"type": "string"},
                        "version": {"type": "string"},
                    },
                    "required": ["name"],
                    "additionalProperties": False,
                },
                "baseline_answers": {"type": "object"},
                "candidate_answers": {"type": "object"},
                "publish": {"type": "boolean"},
                "attestation": {"type": "boolean"},
            },
            "required": [
                "session_id", "baseline_manifest", "candidate_manifest",
                "baseline_answers", "candidate_answers"
            ],
            "additionalProperties": False,
        },
    ),
    "open_bench_leaderboard": (
        _open_bench_leaderboard,
        "TIER 2: list the self-attested public Open Promotion Bench receipts. Entries contain "
        "manifests, verdicts, item-level transitions, and commitments but never task contents "
        "or submitted answers.",
        {"type": "object", "properties": {}, "additionalProperties": False},
    ),
    "about_whetstone": (
        lambda p, ip: {
            "version": __version__,
            "build_commit": build_commit(),
            "catalog": catalog(),
            "mcp_tiers": {
                "tier0": "stateless analyses of caller-supplied data",
                "tier1": "disposable report-card sessions (report_card_start / report_card_submit)",
                "tier2": "paired Open Promotion Bench sessions and opt-in sanitized public receipts",
            },
            "source": "https://github.com/CarlSR9001/whetstone",
            "site": "https://whetstone.cyberelf.link/",
            "agent_docs": "https://whetstone.cyberelf.link/for-agents",
            "skill_file": "https://whetstone.cyberelf.link/skill.md",
            "llms_txt": "https://whetstone.cyberelf.link/llms.txt",
            "llms_full_txt": "https://whetstone.cyberelf.link/llms-full.txt",
            "openapi": "https://whetstone.cyberelf.link/openapi.json",
            "open_benchmark": "https://whetstone.cyberelf.link/benchmark",
            "mcp_manifest": "https://whetstone.cyberelf.link/.well-known/mcp.json",
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
        STATS.bump("refused.jsonrpc_batch")
        return 400, _rpc_error(None, -32600, "JSON-RPC batching is not supported; send one message per request")
    if not isinstance(payload, dict) or payload.get("jsonrpc") != "2.0":
        STATS.bump("refused.bad_jsonrpc")
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
        STATS.bump("mcp.initialize")
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
    if method in ("resources/list", "resources/templates/list", "prompts/list"):
        # We declare only the tools capability, but registry crawlers and some
        # clients probe these anyway; an empty result reads as "none", while
        # -32601 reads as "broken".
        STATS.bump("mcp.enumeration_probe")
        key = {
            "resources/list": "resources",
            "resources/templates/list": "resourceTemplates",
            "prompts/list": "prompts",
        }[method]
        return 200, _rpc_result(request_id, {key: []})
    if method == "tools/list":
        return 200, _rpc_result(request_id, {"tools": _tool_list()})
    if method == "tools/call":
        name = params.get("name")
        entry = TOOLS.get(name) if isinstance(name, str) else None
        if entry is None:
            STATS.bump("refused.unknown_tool")
            return 200, _rpc_error(request_id, -32602, f"unknown tool {name!r}")
        arguments = params.get("arguments") or {}
        if not isinstance(arguments, dict):
            return 200, _rpc_error(request_id, -32602, "arguments must be an object")
        handler = entry[0]
        try:
            result = handler(arguments, client_ip)
        except (ProductInputError, TierError, OpenBenchError) as error:
            STATS.tool_call(name, "mcp", "input_error")
            return 200, _tool_failure(request_id, str(error))
        except Exception as error:  # fail closed without reflecting internals
            STATS.tool_call(name, "mcp", "internal_error")
            print(f"mcp tool {name} failed: {type(error).__name__}", flush=True)
            return 200, _tool_failure(request_id, "internal processing error")
        STATS.tool_call(name, "mcp", "ok")
        return 200, _rpc_result(request_id, {
            "content": [{"type": "text", "text": json.dumps(result, sort_keys=True, indent=1)}],
            "structuredContent": result,
            "isError": False,
        })
    STATS.bump("refused.unknown_method")
    return 200, _rpc_error(request_id, -32601, f"method {method!r} not found")
