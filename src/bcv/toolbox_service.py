"""Public, stateless HTTP shell for the Whetstone product tools.

The service binds to loopback by default, accepts bounded JSON requests, writes
nothing, and never opens an examiner bank.  Nginx supplies TLS and an additional
rate limit in production.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import secrets
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from bcv.ephemeral import ensure_started, hatchery
from bcv.mcp_service import handle_mcp
from bcv.ratelimit import GENERAL_LIMIT, HUNTER_LIMIT, HUNTER_SLOT, SlidingWindowLimit  # noqa: F401 (re-exported)
from bcv.stats import STATS
from bcv.product_tools import (
    ProductInputError,
    audit_leakage,
    bank_health,
    catalog,
    examples,
    gate_results,
    hunt_counterexample,
    inspect_promotion,
    memory_relevance,
    replay_trace,
    safe_patch,
)


VERSION = "0.4.1"
CANONICAL = "https://whetstone.cyberelf.link"
MAX_BODY_BYTES = 1_000_000
STATIC_ROOT = Path(__file__).with_name("toolbox_static")
REPO_ROOT = Path(__file__).resolve().parents[2]
STARTED_AT = time.time()


def _load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def evidence() -> dict:
    relevance = _load_json(REPO_ROOT / "results" / "relevance_eval_report.json")
    ladder = _load_json(REPO_ROOT / "results" / "cross_scale_ladder_receipt.json")
    redteam = _load_json(REPO_ROOT / "results" / "redteam_gate_receipt.json")
    scale_gate = (ladder.get("gates") or [{}])[0]
    return {
        "relevance": {
            "probes": relevance.get("probes"),
            "accuracy": relevance.get("accuracy", {}),
            "top1_finds_truly_relevant": (relevance.get("estimator_validation") or {}).get("top1_finds_truly_relevant", {}),
            "quadrants": relevance.get("quadrants", {}),
        },
        "cross_scale": {
            "bank_items": (ladder.get("bank") or {}).get("items"),
            "models": len(ladder.get("ladder") or []),
            "largest_contrast": {
                "gains": scale_gate.get("gains"),
                "regressions": scale_gate.get("regressions"),
                "p": scale_gate.get("p"),
                "verdict": scale_gate.get("verdict"),
            },
        },
        "redteam": {
            "paraphrase_caught": (redteam.get("paraphrase_attack") or {}).get("behavioral_fingerprint_matched"),
            "inflation_caught": (redteam.get("inflation_attack") or {}).get("caught"),
            "scope": redteam.get("note"),
        },
        "source": "sanitized committed receipts; no private item content",
    }


def _llms_body() -> str:
    return (
        "# Whetstone Tools\n\n"
        "> Verifier-grounded evaluation tools for AI systems, callable by agents over MCP or REST. "
        "Stateless, no auth, nothing stored, no private exam bank loaded.\n\n"
        "Whetstone decides whether a new agent/model version genuinely improved: exposure quarantine, "
        "paired promotion gates (PASS/HOLD/BLOCK with an exact McNemar test), leakage audits, bank "
        "health, SafePatch conservation edits, graph counterexample search, memory relevance scoring, "
        "and agent event replay.\n\n"
        "## Use it\n\n"
        f"- MCP endpoint (Streamable HTTP, stateless JSON-RPC): POST {CANONICAL}/mcp\n"
        f"- Claude Code: claude mcp add --transport http whetstone {CANONICAL}/mcp\n"
        "- REST: every tool is POST /api/<name>; GET /api/examples returns a complete valid payload per tool\n"
        f"- OpenAPI spec: {CANONICAL}/openapi.json\n"
        f"- Skill file (Anthropic Skills format, full instructions): {CANONICAL}/skill.md\n"
        f"- Agent documentation page: {CANONICAL}/for-agents\n"
        f"- Tool catalog: {CANONICAL}/api/catalog\n"
        "- Source (AGPL-3.0): https://github.com/CarlSR9001/whetstone\n\n"
        "## Tiers\n\n"
        "- Tier 0 (stateless): you supply exam rows, paired results, documents, or event logs; you get "
        "audits, verdicts, patches, or counterexamples with SHA-256 receipts. Tools: inspect_promotion, "
        "audit_leakage, promotion_gate, bank_health, safe_patch, counterexample_hunt, memory_relevance, "
        "replay_trace.\n"
        "- Tier 1 (report card): report_card_start hands your agent a disposable 6-item graph-repair "
        "exam; report_card_submit grades it by checker spec (any verified strict refinement passes; no "
        "answer key exists), reports support-retention diagnostics, and destroys the one-shot session. "
        "Items are minted from the public frontier: a demonstration, not a credential.\n\n"
        "## Limits\n\n"
        "~60 req/min general; 4 report-card sessions + 8 submits per hour per address; counterexample "
        "hunts 4 per 10 min behind one worker; report card warms ~2 min after restart (GET /api/health "
        "-> report_card.ready).\n"
    )


def _sitemap_xml() -> str:
    pages = ("/", "/for-agents", "/skill.md", "/llms.txt", "/llms-full.txt", "/openapi.json")
    entries = "\n".join(f"  <url><loc>{CANONICAL}{page}</loc></url>" for page in pages)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        f"{entries}\n</urlset>\n"
    )


def _mcp_manifest() -> dict:
    return {
        "name": "whetstone-tools",
        "version": VERSION,
        "description": (
            "Promotion gate for AI agents: leakage audits, PASS/HOLD/BLOCK verdicts with exact "
            "statistics, and a disposable report card that grades the connecting agent by checker spec."
        ),
        "endpoint": f"{CANONICAL}/mcp",
        "transport": "streamable-http",
        "protocol_versions": ["2025-06-18", "2025-03-26", "2024-11-05"],
        "authentication": {"type": "none"},
        "capabilities": {"tools": True, "resources": False, "prompts": False},
        "documentation": f"{CANONICAL}/for-agents",
        "skill_file": f"{CANONICAL}/skill.md",
        "openapi": f"{CANONICAL}/openapi.json",
        "source": "https://github.com/CarlSR9001/whetstone",
        "license": "AGPL-3.0-or-later",
    }


def _openapi_spec() -> dict:
    example_payloads = examples()
    paths: dict = {
        "/api/health": {"get": {"operationId": "health", "summary": "Service health, version, and report-card readiness.", "responses": {"200": {"description": "OK"}}}},
        "/api/catalog": {"get": {"operationId": "listTools", "summary": "Catalog of the eight stateless verifier tools.", "responses": {"200": {"description": "OK"}}}},
        "/api/examples": {"get": {"operationId": "getExamples", "summary": "A complete, valid request payload for every tool.", "responses": {"200": {"description": "OK"}}}},
        "/api/stats": {"get": {"operationId": "getStats", "summary": "Aggregate, privacy-preserving usage counters.", "responses": {"200": {"description": "OK"}}}},
        "/mcp": {"post": {"operationId": "mcp", "summary": "MCP endpoint (Streamable HTTP, stateless JSON-RPC 2.0). Tools include the Tier 0 set plus report_card_start / report_card_submit.", "responses": {"200": {"description": "JSON-RPC response"}}}},
    }
    for tool in catalog():
        example = example_payloads.get(tool["id"])
        operation = {
            "operationId": tool["id"],
            "summary": tool["promise"],
            "requestBody": {
                "required": True,
                "content": {"application/json": ({"example": example} if example else {})},
            },
            "responses": {
                "200": {"description": "Deterministic receipt with item-level decision path and SHA-256 hashes."},
                "400": {"description": "Input error (a hint points at /api/examples)."},
                "429": {"description": "Rate limited."},
            },
        }
        paths[tool["endpoint"]] = {"post": operation}
    return {
        "openapi": "3.1.0",
        "info": {
            "title": "Whetstone Tools",
            "version": VERSION,
            "description": (
                "Verifier-grounded evaluation tools for AI systems. Stateless, unauthenticated, "
                "nothing stored. Every POST tool is also exposed as an MCP tool at /mcp."
            ),
            "license": {"name": "AGPL-3.0-or-later", "url": "https://www.gnu.org/licenses/agpl-3.0.html"},
        },
        "servers": [{"url": CANONICAL}],
        "paths": paths,
    }


POST_ROUTES = {
    "/api/inspect": inspect_promotion,
    "/api/leakage": audit_leakage,
    "/api/gate": gate_results,
    "/api/health-report": bank_health,
    "/api/safepatch": safe_patch,
    "/api/memory": memory_relevance,
    "/api/replay": replay_trace,
}

# REST endpoints counted under their MCP tool names so /api/stats merges cleanly.
REST_TOOL_NAMES = {
    "/api/inspect": "inspect_promotion",
    "/api/leakage": "audit_leakage",
    "/api/gate": "promotion_gate",
    "/api/health-report": "bank_health",
    "/api/safepatch": "safe_patch",
    "/api/memory": "memory_relevance",
    "/api/replay": "replay_trace",
    "/api/counterexample": "counterexample_hunt",
}


class ToolboxHandler(BaseHTTPRequestHandler):
    server_version = "WhetstoneTools/0.2"

    def log_message(self, format: str, *args) -> None:
        # Nginx records method/path/status.  The app deliberately never logs a body.
        pass

    def _client_ip(self) -> str:
        forwarded = self.headers.get("X-Forwarded-For", "").split(",", 1)[0].strip()
        return forwarded or self.client_address[0]

    def _cors(self) -> None:
        """Public, unauthenticated, stateless API: no cookies, no sessions, no
        user state to forge a request against. Allowing any origin costs nothing
        and is the only way browser clients and hosted MCP clients (claude.ai
        connectors send Origin) can reach the endpoint at all. Credentials are
        never allowed, so '*' stays safe."""
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type, Accept, MCP-Protocol-Version, Mcp-Session-Id",
        )
        self.send_header("Access-Control-Max-Age", "86400")

    def _is_machine_path(self) -> bool:
        path = self.path.split("?", 1)[0]
        return path == "/mcp" or path.startswith("/api/")

    def _headers(self, content_type: str, length: int, *, cache: str = "no-store") -> None:
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", cache)
        if self._is_machine_path():
            self._cors()
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Permissions-Policy", "camera=(), microphone=(), geolocation=(), payment=()")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
            "connect-src 'self'; object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'",
        )

    def _bytes(self, code: int, body: bytes, content_type: str, *, cache: str = "no-store") -> None:
        self.send_response(code)
        self._headers(content_type, len(body), cache=cache)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        self._bytes(code, body, "application/json; charset=utf-8")

    def do_OPTIONS(self) -> None:
        if self._is_machine_path():
            self.send_response(204)
            self._cors()
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_response(405)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/api/health":
            return self._json(200, {
                "status": "ok",
                "version": VERSION,
                "stateless": True,
                "private_bank_loaded": False,
                "uptime_seconds": round(time.time() - STARTED_AT, 1),
                "tools": len(catalog()),
                "mcp_endpoint": "/mcp",
                "report_card": hatchery().status(),
            })
        if path == "/mcp":
            return self._json(405, {
                "error": "the MCP endpoint is Streamable HTTP in stateless mode: POST a JSON-RPC 2.0 message",
                "server_info": "whetstone-tools",
            })
        if path == "/api/catalog":
            return self._json(200, {"tools": catalog()})
        if path == "/api/stats":
            return self._json(200, STATS.public_summary())
        if path == "/api/examples":
            return self._json(200, examples())
        if path == "/api/evidence":
            return self._json(200, evidence())
        if path == "/robots.txt":
            body = f"User-agent: *\nAllow: /\n\nSitemap: {CANONICAL}/sitemap.xml\n".encode("utf-8")
            return self._bytes(200, body, "text/plain; charset=utf-8", cache="public, max-age=3600")
        if path == "/sitemap.xml":
            return self._bytes(200, _sitemap_xml().encode("utf-8"), "application/xml; charset=utf-8", cache="public, max-age=3600")
        if path == "/openapi.json":
            body = json.dumps(_openapi_spec(), sort_keys=True, indent=1).encode("utf-8")
            return self._bytes(200, body, "application/json; charset=utf-8", cache="public, max-age=3600")
        if path in {"/.well-known/mcp.json", "/mcp.json"}:
            body = json.dumps(_mcp_manifest(), sort_keys=True, indent=1).encode("utf-8")
            return self._bytes(200, body, "application/json; charset=utf-8", cache="public, max-age=3600")
        if path == "/llms.txt":
            return self._bytes(200, _llms_body().encode("utf-8"), "text/plain; charset=utf-8", cache="public, max-age=3600")
        if path == "/llms-full.txt":
            skill_path = STATIC_ROOT / "skill.md"
            skill = skill_path.read_text(encoding="utf-8") if skill_path.exists() else ""
            body = (_llms_body() + "\n---\n\n" + skill).encode("utf-8")
            return self._bytes(200, body, "text/plain; charset=utf-8", cache="public, max-age=3600")
        static_name = "index.html" if path in {"/", "/index.html"} else path.lstrip("/")
        if static_name == "for-agents":
            static_name = "for-agents.html"
        if static_name not in {"index.html", "app.js", "styles.css", "skill.md", "for-agents.html", "og.png"}:
            return self._json(404, {"error": "not found"})
        file_path = STATIC_ROOT / static_name
        if not file_path.exists():
            return self._json(500, {"error": "static asset unavailable"})
        if static_name.endswith(".md"):
            mime = "text/markdown"
        else:
            mime = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        if not mime.startswith("image/"):
            mime = f"{mime}; charset=utf-8"
        cache = "no-store" if static_name == "index.html" else "public, max-age=300"
        return self._bytes(200, file_path.read_bytes(), mime, cache=cache)

    def do_POST(self) -> None:
        request_id = secrets.token_hex(6)
        path = self.path.split("?", 1)[0]
        client_ip = self._client_ip()
        STATS.touch(client_ip)
        if not GENERAL_LIMIT.allow(client_ip):
            STATS.bump("refused.rate_limited")
            return self._json(429, {"error": "rate limit exceeded", "request_id": request_id})
        # Deliberately lenient about Content-Type: `curl -X POST url -d '{...}'`
        # is the first thing anyone types, and curl defaults to form-urlencoded.
        # The body either parses as JSON or it does not; the declared type adds
        # nothing on an API with no cookies to protect.
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            STATS.bump("refused.bad_request")
            return self._json(400, {"error": "invalid Content-Length", "request_id": request_id})
        if length <= 0:
            STATS.bump("refused.bad_request")
            return self._json(400, {"error": "JSON body required", "request_id": request_id})
        if length > MAX_BODY_BYTES:
            STATS.bump("refused.oversized")
            return self._json(413, {"error": f"body exceeds {MAX_BODY_BYTES} bytes", "request_id": request_id})
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            STATS.bump("refused.bad_request")
            return self._json(400, {
                "error": "body must be valid UTF-8 JSON",
                "hint": "GET /api/examples returns a complete valid payload for every tool",
                "request_id": request_id,
            })
        if path == "/mcp":
            status, body = handle_mcp(payload, client_ip)
            if body is None:
                return self._bytes(status, b"", "application/json; charset=utf-8")
            return self._json(status, body)
        if not isinstance(payload, dict):
            STATS.bump("refused.bad_request")
            return self._json(400, {"error": "JSON body must be an object", "request_id": request_id})

        tool_name = REST_TOOL_NAMES.get(path, "")
        acquired = False
        try:
            if path == "/api/counterexample":
                if not HUNTER_LIMIT.allow(client_ip):
                    STATS.tool_call(tool_name, "rest", "limited")
                    return self._json(429, {"error": "counterexample search limit exceeded", "request_id": request_id})
                acquired = HUNTER_SLOT.acquire(blocking=False)
                if not acquired:
                    STATS.tool_call(tool_name, "rest", "limited")
                    return self._json(429, {"error": "counterexample worker busy; retry shortly", "request_id": request_id})
                result = hunt_counterexample(payload)
            else:
                handler = POST_ROUTES.get(path)
                if handler is None:
                    STATS.bump("refused.unknown_endpoint")
                    return self._json(404, {"error": "unknown endpoint", "request_id": request_id})
                result = handler(payload)
            result["request_id"] = request_id
            STATS.tool_call(tool_name, "rest", "ok")
            return self._json(200, result)
        except ProductInputError as error:
            STATS.tool_call(tool_name, "rest", "input_error")
            return self._json(400, {"error": str(error), "request_id": request_id})
        except Exception as error:  # fail closed without reflecting internals
            STATS.tool_call(tool_name, "rest", "internal_error")
            print(f"toolbox request {request_id} failed: {type(error).__name__}", flush=True)
            return self._json(500, {"error": "internal processing error", "request_id": request_id})
        finally:
            if acquired:
                HUNTER_SLOT.release()


def make_server(host: str = "127.0.0.1", port: int = 8988) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), ToolboxHandler)


def serve(host: str = "127.0.0.1", port: int = 8988) -> None:
    server = make_server(host, port)
    ensure_started()  # warm the report-card hatchery in the background
    STATS.start_flusher()
    print(f"Whetstone Tools {VERSION} on http://{host}:{port} (stateless; no private bank loaded)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        STATS.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the stateless Whetstone product toolbox.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8988)
    args = parser.parse_args()
    serve(args.host, args.port)


if __name__ == "__main__":
    main()
