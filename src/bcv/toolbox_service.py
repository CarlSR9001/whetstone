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
from urllib.parse import urlsplit

from bcv.ephemeral import ensure_started, hatchery
from bcv.mcp_service import handle_mcp
from bcv.ratelimit import GENERAL_LIMIT, HUNTER_LIMIT, HUNTER_SLOT, SlidingWindowLimit  # noqa: F401 (re-exported)
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


VERSION = "0.3.0"
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


POST_ROUTES = {
    "/api/inspect": inspect_promotion,
    "/api/leakage": audit_leakage,
    "/api/gate": gate_results,
    "/api/health-report": bank_health,
    "/api/safepatch": safe_patch,
    "/api/memory": memory_relevance,
    "/api/replay": replay_trace,
}


class ToolboxHandler(BaseHTTPRequestHandler):
    server_version = "WhetstoneTools/0.2"

    def log_message(self, format: str, *args) -> None:
        # Nginx records method/path/status.  The app deliberately never logs a body.
        pass

    def _client_ip(self) -> str:
        forwarded = self.headers.get("X-Forwarded-For", "").split(",", 1)[0].strip()
        return forwarded or self.client_address[0]

    def _headers(self, content_type: str, length: int, *, cache: str = "no-store") -> None:
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", cache)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Permissions-Policy", "camera=(), microphone=(), geolocation=(), payment=()")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
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

    def _same_origin(self) -> bool:
        origin = self.headers.get("Origin")
        if not origin:
            return True
        try:
            return urlsplit(origin).netloc.lower() == self.headers.get("Host", "").lower()
        except ValueError:
            return False

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
        if path == "/api/examples":
            return self._json(200, examples())
        if path == "/api/evidence":
            return self._json(200, evidence())
        if path == "/robots.txt":
            return self._bytes(200, b"User-agent: *\nAllow: /\n", "text/plain; charset=utf-8", cache="public, max-age=3600")
        if path == "/llms.txt":
            body = (
                "# Whetstone Tools\n\n"
                "Stateless public demonstrations of exposure auditing, paired promotion gates, bank health, "
                "SafePatch conservation checks, bounded graph counterexample search, memory relevance scoring, "
                "and agent event replay. No private examiner bank is loaded.\n\n"
                "## MCP\n\n"
                "Remote MCP endpoint (Streamable HTTP, stateless): POST JSON-RPC to /mcp. Tier 0 tools are "
                "the stateless analyses above. Tier 1: report_card_start hands your agent a disposable "
                "graph-repair exam; report_card_submit grades it by checker spec and destroys the session.\n"
            ).encode("utf-8")
            return self._bytes(200, body, "text/plain; charset=utf-8", cache="public, max-age=3600")
        static_name = "index.html" if path in {"/", "/index.html"} else path.lstrip("/")
        if static_name not in {"index.html", "app.js", "styles.css"}:
            return self._json(404, {"error": "not found"})
        file_path = STATIC_ROOT / static_name
        if not file_path.exists():
            return self._json(500, {"error": "static asset unavailable"})
        mime = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        cache = "public, max-age=300" if static_name != "index.html" else "no-store"
        return self._bytes(200, file_path.read_bytes(), f"{mime}; charset=utf-8", cache=cache)

    def do_POST(self) -> None:
        request_id = secrets.token_hex(6)
        path = self.path.split("?", 1)[0]
        client_ip = self._client_ip()
        if not self._same_origin():
            return self._json(403, {"error": "cross-origin request refused", "request_id": request_id})
        if not GENERAL_LIMIT.allow(client_ip):
            return self._json(429, {"error": "rate limit exceeded", "request_id": request_id})
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            return self._json(415, {"error": "Content-Type must be application/json", "request_id": request_id})
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return self._json(400, {"error": "invalid Content-Length", "request_id": request_id})
        if length <= 0:
            return self._json(400, {"error": "JSON body required", "request_id": request_id})
        if length > MAX_BODY_BYTES:
            return self._json(413, {"error": f"body exceeds {MAX_BODY_BYTES} bytes", "request_id": request_id})
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return self._json(400, {"error": "body must be valid UTF-8 JSON", "request_id": request_id})
        if path == "/mcp":
            status, body = handle_mcp(payload, client_ip)
            if body is None:
                return self._bytes(status, b"", "application/json; charset=utf-8")
            return self._json(status, body)
        if not isinstance(payload, dict):
            return self._json(400, {"error": "JSON body must be an object", "request_id": request_id})

        acquired = False
        try:
            if path == "/api/counterexample":
                if not HUNTER_LIMIT.allow(client_ip):
                    return self._json(429, {"error": "counterexample search limit exceeded", "request_id": request_id})
                acquired = HUNTER_SLOT.acquire(blocking=False)
                if not acquired:
                    return self._json(429, {"error": "counterexample worker busy; retry shortly", "request_id": request_id})
                result = hunt_counterexample(payload)
            else:
                handler = POST_ROUTES.get(path)
                if handler is None:
                    return self._json(404, {"error": "unknown endpoint", "request_id": request_id})
                result = handler(payload)
            result["request_id"] = request_id
            return self._json(200, result)
        except ProductInputError as error:
            return self._json(400, {"error": str(error), "request_id": request_id})
        except Exception as error:  # fail closed without reflecting internals
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
    print(f"Whetstone Tools {VERSION} on http://{host}:{port} (stateless; no private bank loaded)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the stateless Whetstone product toolbox.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8988)
    args = parser.parse_args()
    serve(args.host, args.port)


if __name__ == "__main__":
    main()
