"""Public HTTP shell for Whetstone's stateless tools and open benchmark.

The service binds to loopback by default, accepts bounded JSON requests,
persists no workbench/report-card request bodies or results, and never opens an
examiner bank. It retains privacy-preserving operational counters; Open
Promotion Bench separately persists only opt-in public manifests and sanitized
receipts, never task contents or answers. Nginx supplies TLS, access logging,
and an additional rate limit in production.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import mimetypes
import os
import secrets
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from bcv._version import __version__, build_commit
from bcv.ephemeral import ensure_started, hatchery
from bcv.mcp_service import SUPPORTED_PROTOCOL_VERSIONS, handle_mcp
from bcv.open_bench import OpenBenchError, open_bench
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
    input_schemas,
    inspect_promotion,
    memory_relevance,
    replay_trace,
    safe_patch,
)

CANONICAL = "https://whetstone.cyberelf.link"
MAX_BODY_BYTES = 1_000_000
STATIC_ROOT = Path(__file__).with_name("toolbox_static")
STARTED_AT = time.time()


def _normalized_ip(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return str(ipaddress.ip_address(value.strip()))
    except ValueError:
        return None


def _trusted_client_ip(peer: str, real_ip: str | None, forwarded_for: str | None) -> str:
    """Resolve the caller only when the immediate peer is the loopback proxy.

    Nginx overwrites both forwarding headers with ``$remote_addr``.  Taking the
    first caller-supplied X-Forwarded-For value would let remote clients mint
    arbitrary rate-limit identities.
    """
    normalized_peer = _normalized_ip(peer)
    if normalized_peer is None:
        return peer
    if not ipaddress.ip_address(normalized_peer).is_loopback:
        return normalized_peer

    normalized_real = _normalized_ip(real_ip)
    if normalized_real is not None:
        return normalized_real

    # Rightmost is the address appended by the nearest trusted proxy.  The
    # production Nginx config overwrites this header, but this remains safe if
    # an older proxy configuration briefly supplies a chain.
    forwarded = (forwarded_for or "").rsplit(",", 1)[-1].strip()
    return _normalized_ip(forwarded) or normalized_peer


def _load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def evidence() -> dict:
    return _load_json(STATIC_ROOT / "evidence.json")


def _llms_body() -> str:
    return (
        "# Whetstone Tools\n\n"
        "> Verifier-grounded evaluation tools for AI systems, callable by agents over MCP or REST. "
        "Workbench and report-card requests are stateless: payloads and results are not persisted. "
        "Open Promotion Bench stores only opt-in public manifests and sanitized receipts, never tasks "
        "or answers. No auth and no private exam bank loaded. "
        "Operational counters and standard access logs are retained.\n\n"
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
        f"- Open Promotion Bench: {CANONICAL}/benchmark\n"
        "- Source (AGPL-3.0): https://github.com/CarlSR9001/whetstone\n\n"
        "## Tiers\n\n"
        "- Tier 0 (stateless): you supply exam rows, paired results, documents, or event logs; you get "
        "audits, verdicts, patches, or counterexamples with SHA-256 receipts. Tools: inspect_promotion, "
        "audit_leakage, promotion_gate, bank_health, safe_patch, counterexample_hunt, memory_relevance, "
        "replay_trace.\n"
        "- Tier 1 (report card): report_card_start hands your agent a disposable 6-item graph-repair "
        "exam; report_card_submit grades it by checker spec (any verified strict refinement passes; no "
        "answer key exists), reports support-retention diagnostics, and destroys the one-shot session. "
        "Items are minted from the public frontier: a demonstration, not a credential.\n"
        "- Tier 2 (open benchmark): open_bench_start issues six fresh scope-integrity tasks for "
        "a baseline and candidate; open_bench_submit counts gains and regressions and may publish "
        "a sanitized self-attested receipt; open_bench_leaderboard reads the public board.\n\n"
        "## Limits\n\n"
        "~60 req/min general; 4 report-card sessions + 8 submits per hour per address; counterexample "
        "hunts 4 per 10 min behind one worker; report card warms ~2 min after restart (GET /api/health "
        "-> report_card.ready).\n"
    )


def _sitemap_xml() -> str:
    pages = ("/", "/benchmark", "/for-agents", "/skill.md", "/llms.txt", "/llms-full.txt", "/openapi.json")
    entries = "\n".join(f"  <url><loc>{CANONICAL}{page}</loc></url>" for page in pages)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        f"{entries}\n</urlset>\n"
    )


def _mcp_manifest() -> dict:
    return {
        "name": "whetstone-tools",
        "version": __version__,
        "build_commit": build_commit(),
        "description": (
            "Promotion gate for AI agents: leakage audits, PASS/HOLD/BLOCK verdicts with exact "
            "statistics, a disposable report card, and a paired public scope-integrity benchmark."
        ),
        "endpoint": f"{CANONICAL}/mcp",
        "transport": "streamable-http",
        "protocol_versions": list(SUPPORTED_PROTOCOL_VERSIONS),
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
    schemas = input_schemas()
    paths: dict = {
        "/api/health": {"get": {"operationId": "health", "summary": "Service health, version, and report-card readiness.", "responses": {"200": {"description": "OK"}}}},
        "/api/catalog": {"get": {"operationId": "listTools", "summary": "Catalog of the eight stateless verifier tools.", "responses": {"200": {"description": "OK"}}}},
        "/api/examples": {"get": {"operationId": "getExamples", "summary": "A complete, valid request payload for every tool.", "responses": {"200": {"description": "OK"}}}},
        "/api/stats": {"get": {"operationId": "getStats", "summary": "Aggregate, privacy-preserving usage counters.", "responses": {"200": {"description": "OK"}}}},
        "/api/open-bench/leaderboard": {"get": {"operationId": "openBenchLeaderboard", "summary": "Opt-in, self-attested public Open Promotion Bench receipts.", "responses": {"200": {"description": "OK"}}}},
        "/api/open-bench/receipt/{public_id}": {"get": {"operationId": "openBenchReceipt", "summary": "One sanitized public benchmark receipt.", "parameters": [{"name": "public_id", "in": "path", "required": True, "schema": {"type": "string"}}], "responses": {"200": {"description": "OK"}, "404": {"description": "Not found"}}}},
        "/api/open-bench/start": {"post": {"operationId": "openBenchStart", "summary": "Start a paired six-item scope-integrity cohort.", "requestBody": {"required": True, "content": {"application/json": {"schema": {"type": "object", "additionalProperties": False}}}}, "responses": {"200": {"description": "One-shot session and tasks"}, "429": {"description": "Rate limited"}}}},
        "/api/open-bench/submit": {"post": {"operationId": "openBenchSubmit", "summary": "Grade paired baseline/candidate patches and optionally publish a sanitized receipt.", "responses": {"200": {"description": "PASS/HOLD/BLOCK receipt"}, "400": {"description": "Input error"}, "429": {"description": "Rate limited"}}}},
        "/mcp": {"post": {"operationId": "mcp", "summary": "MCP endpoint (Streamable HTTP JSON-RPC 2.0). Includes Tier 0 tools, the disposable report card, and Open Promotion Bench.", "responses": {"200": {"description": "JSON-RPC response"}}}},
    }
    for tool in catalog():
        example = example_payloads.get(tool["id"])
        schema = schemas[REST_TOOL_NAMES[tool["endpoint"]]]
        operation = {
            "operationId": tool["id"],
            "summary": tool["promise"],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": schema,
                        **({"example": example} if example else {}),
                    }
                },
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
            "version": __version__,
            "x-build-commit": build_commit(),
            "description": (
                "Verifier-grounded evaluation tools for AI systems. Workbench and report-card "
                "requests are stateless and unauthenticated; payloads and results are not persisted. "
                "Open Promotion Bench may persist only an opt-in public manifest and sanitized receipt, "
                "never task contents or answers. Operational counters and standard access logs are retained."
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
    "/api/open-bench/start": "open_bench_start",
    "/api/open-bench/submit": "open_bench_submit",
}


class ToolboxHandler(BaseHTTPRequestHandler):
    server_version = f"WhetstoneTools/{__version__}"

    def log_message(self, format: str, *args) -> None:
        # Nginx records method/path/status.  The app deliberately never logs a body.
        pass

    def _client_ip(self) -> str:
        return _trusted_client_ip(
            self.client_address[0],
            self.headers.get("X-Real-IP"),
            self.headers.get("X-Forwarded-For"),
        )

    def _cors(self) -> None:
        """Public, unauthenticated API with no cookies or ambient authority.

        Report-card and Open Promotion Bench ids are explicit one-shot bearer
        values, not browser credentials. Allowing any origin is required for
        browser clients and hosted MCP clients (claude.ai connectors send
        Origin); credentials are never allowed, so ``*`` stays safe.
        """
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type, Accept, MCP-Protocol-Version, Mcp-Session-Id, Mcp-Method, Mcp-Name",
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
        if self.command != "HEAD":  # HEAD gets the exact GET headers, no body
            self.wfile.write(body)

    def do_HEAD(self) -> None:
        # Crawlers and link-preview bots probe with HEAD before fetching.
        self.do_GET()

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
                "version": __version__,
                "build_commit": build_commit(),
                "stateless": True,
                "stateless_scope": "workbench and disposable report card",
                "publication_persistence": "opt-in Open Promotion Bench manifests and sanitized receipts only",
                "private_bank_loaded": False,
                "uptime_seconds": round(time.time() - STARTED_AT, 1),
                "tools": len(catalog()),
                "mcp_endpoint": "/mcp",
                "report_card": hatchery().status(),
                "open_bench": open_bench().status(),
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
        if path == "/api/open-bench/leaderboard":
            return self._json(200, open_bench().leaderboard())
        if path.startswith("/api/open-bench/receipt/"):
            public_id = path.rsplit("/", 1)[-1]
            receipt = open_bench().receipt(public_id)
            return self._json(200, receipt) if receipt is not None else self._json(404, {"error": "receipt not found"})
        if path == "/robots.txt":
            body = f"User-agent: *\nAllow: /\n\nSitemap: {CANONICAL}/sitemap.xml\n".encode("utf-8")
            return self._bytes(200, body, "text/plain; charset=utf-8", cache="public, max-age=3600")
        if path == "/sitemap.xml":
            return self._bytes(200, _sitemap_xml().encode("utf-8"), "application/xml; charset=utf-8", cache="public, max-age=3600")
        if path == "/openapi.json":
            body = json.dumps(_openapi_spec(), sort_keys=True, indent=1).encode("utf-8")
            return self._bytes(200, body, "application/json; charset=utf-8", cache="public, max-age=3600")
        if path == "/.well-known/mcp-registry-auth":
            # Domain-ownership proof for the official MCP Registry. The record
            # (a public key) lives in the state dir, not the repo, so rotating
            # it never requires a release.
            raw = os.environ.get("WHETSTONE_STATE_DIR") or os.environ.get("STATE_DIRECTORY") or ""
            raw = raw.split(os.pathsep, 1)[0]  # systemd may pass colon-separated dirs; os.pathsep spares Windows drive letters
            proof = Path(raw) / "mcp-registry-auth" if raw else None
            if proof is not None and proof.exists():
                return self._bytes(200, proof.read_bytes(), "text/plain; charset=utf-8", cache="public, max-age=300")
            return self._json(404, {"error": "not found"})
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
        if static_name == "benchmark":
            static_name = "benchmark.html"
        if static_name not in {"index.html", "benchmark.html", "benchmark.js", "app.js", "styles.css", "skill.md", "for-agents.html", "og.png", "open-bench-og.png"}:
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
        cache = "no-store" if static_name in {"index.html", "benchmark.html"} else "public, max-age=300"
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
            status, body = handle_mcp(
                payload,
                client_ip,
                {
                    "MCP-Protocol-Version": self.headers.get("MCP-Protocol-Version"),
                    "Mcp-Method": self.headers.get("Mcp-Method"),
                    "Mcp-Name": self.headers.get("Mcp-Name"),
                },
            )
            if body is None:
                return self._bytes(status, b"", "application/json; charset=utf-8")
            return self._json(status, body)
        if not isinstance(payload, dict):
            STATS.bump("refused.bad_request")
            return self._json(400, {"error": "JSON body must be an object", "request_id": request_id})

        tool_name = REST_TOOL_NAMES.get(path, "")
        acquired = False
        try:
            if path == "/api/open-bench/start":
                if payload:
                    raise OpenBenchError("open benchmark start takes an empty JSON object")
                result = open_bench().start_session(client_ip)
            elif path == "/api/open-bench/submit":
                result = open_bench().submit(payload, client_ip)
            elif path == "/api/counterexample":
                if not HUNTER_LIMIT.allow(client_ip):
                    STATS.tool_call(tool_name, "rest", "limited", "rate_limit")
                    return self._json(429, {"error": "counterexample search limit exceeded", "request_id": request_id})
                acquired = HUNTER_SLOT.acquire(blocking=False)
                if not acquired:
                    STATS.tool_call(tool_name, "rest", "limited", "busy")
                    return self._json(429, {"error": "counterexample worker busy; retry shortly", "request_id": request_id})
                result = hunt_counterexample(payload)
            else:
                handler = POST_ROUTES.get(path)
                if handler is None:
                    STATS.bump("refused.unknown_endpoint")
                    return self._json(404, {"error": "unknown endpoint", "request_id": request_id})
                result = handler(payload)
            result["request_id"] = request_id
            STATS.tool_call(tool_name, "rest", "ok", "none")
            return self._json(200, result)
        except OpenBenchError as error:
            outcome = "limited" if error.status >= 429 else "input_error"
            reason = "rate_limit" if error.status >= 429 else "bench_refusal"
            STATS.tool_call(tool_name, "rest", outcome, reason)
            return self._json(error.status, {"error": str(error), "retryable": error.retryable, "request_id": request_id})
        except ProductInputError as error:
            STATS.tool_call(tool_name, "rest", "input_error", "invalid_input")
            return self._json(400, {"error": str(error), "request_id": request_id})
        except Exception as error:  # fail closed without reflecting internals
            STATS.tool_call(tool_name, "rest", "internal_error", "internal_exception")
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
    print(f"Whetstone Tools {__version__} on http://{host}:{port} (stateless core + opt-in public receipts; no private bank loaded)")
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
