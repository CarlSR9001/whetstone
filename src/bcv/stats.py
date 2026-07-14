"""Aggregate service metrics: enough to answer "did anyone real use this?"

Design constraints, in order:

- Privacy first. No request bodies, no prompts, no answers, no raw addresses.
  Clients are counted as salted SHA-256 hashes; the salt never leaves the
  state file. Public output is aggregate counts only.
- The stateless promise stays honest. Metrics are the one deliberate
  exception to "writes nothing", and they live only in the systemd
  StateDirectory (WHETSTONE_STATE_DIR). Without that directory the service
  runs exactly as before with in-memory counters that reset on restart.
- Never in the request path's way: recording is a dict bump under a lock;
  persistence happens on a background flush thread.

Tracked: unique and repeat clients, calls per tool per transport (rest/mcp),
report-card funnel (sessions started / graded / items passed / retention
buckets), error categories, and abuse-shaped refusals (rate limits,
cross-origin, malformed JSON-RPC, unknown tools, oversized bodies).
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import threading
import time
from collections import defaultdict
from pathlib import Path

FLUSH_SECONDS = 60.0
MAX_TRACKED_CLIENTS = 50_000


def _state_dir() -> Path | None:
    raw = os.environ.get("WHETSTONE_STATE_DIR") or os.environ.get("STATE_DIRECTORY")
    if not raw:
        return None
    path = Path(raw.split(":", 1)[0])
    return path if path.is_dir() else None


class Stats:
    def __init__(self, state_path: Path | None) -> None:
        self.lock = threading.Lock()
        self.state_path = state_path
        self.dirty = False
        self.started_at = time.time()
        self.launched_at = time.time()
        self.salt = secrets.token_hex(16)
        self.counters: dict[str, int] = defaultdict(int)
        self.clients: dict[str, int] = {}  # salted hash -> request count
        self._flusher: threading.Thread | None = None
        if state_path is not None and state_path.exists():
            self._load(state_path)

    # ------------------------------------------------------------- recording

    def _hash(self, client_ip: str) -> str:
        return hashlib.sha256(f"{self.salt}:{client_ip}".encode("utf-8")).hexdigest()[:16]

    def touch(self, client_ip: str) -> None:
        with self.lock:
            key = self._hash(client_ip)
            if key in self.clients or len(self.clients) < MAX_TRACKED_CLIENTS:
                self.clients[key] = self.clients.get(key, 0) + 1
            self.counters["requests_total"] += 1
            self.dirty = True

    def bump(self, name: str, amount: int = 1) -> None:
        with self.lock:
            self.counters[name] += amount
            self.dirty = True

    def tool_call(self, tool: str, transport: str, outcome: str) -> None:
        """outcome: ok | input_error | internal_error | limited"""
        with self.lock:
            self.counters[f"tool.{tool}.{transport}"] += 1
            self.counters[f"outcome.{outcome}"] += 1
            self.dirty = True

    def retention_bucket(self, retention: float) -> None:
        bucket = "lt5" if retention < 0.05 else ("lt25" if retention < 0.25 else ("lt50" if retention < 0.5 else "ge50"))
        self.bump(f"report_card.retention.{bucket}")

    # ----------------------------------------------------------- persistence

    def _load(self, path: Path) -> None:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            self.salt = raw.get("salt", self.salt)
            self.launched_at = float(raw.get("launched_at", self.launched_at))
            self.counters.update({str(k): int(v) for k, v in raw.get("counters", {}).items()})
            self.clients.update({str(k): int(v) for k, v in raw.get("clients", {}).items()})
        except (OSError, ValueError, json.JSONDecodeError):
            pass  # corrupt or unreadable state never blocks the service

    def flush(self) -> None:
        if self.state_path is None:
            return
        with self.lock:
            if not self.dirty:
                return
            snapshot = {
                "salt": self.salt,
                "launched_at": self.launched_at,
                "counters": dict(self.counters),
                "clients": self.clients,
            }
            self.dirty = False
        tmp = self.state_path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(snapshot, sort_keys=True), encoding="utf-8")
            tmp.replace(self.state_path)
        except OSError:
            pass

    def start_flusher(self) -> None:
        if self._flusher is not None or self.state_path is None:
            return

        def loop() -> None:
            while True:
                time.sleep(FLUSH_SECONDS)
                self.flush()

        self._flusher = threading.Thread(target=loop, name="whetstone-stats-flush", daemon=True)
        self._flusher.start()

    # ---------------------------------------------------------------- public

    def public_summary(self) -> dict:
        with self.lock:
            tools: dict[str, dict[str, int]] = defaultdict(lambda: {"rest": 0, "mcp": 0})
            other: dict[str, int] = {}
            for name, value in self.counters.items():
                if name.startswith("tool."):
                    _, tool, transport = name.split(".", 2)
                    tools[tool][transport] = value
                else:
                    other[name] = value
            repeat = sum(1 for count in self.clients.values() if count > 1)
            return {
                "since_unix": int(self.launched_at),
                "persistent": self.state_path is not None,
                "unique_clients": len(self.clients),
                "repeat_clients": repeat,
                "requests_total": other.get("requests_total", 0),
                "tool_calls": {name: dict(counts) for name, counts in sorted(tools.items())},
                "outcomes": {k.split(".", 1)[1]: v for k, v in other.items() if k.startswith("outcome.")},
                "report_card": {k.split(".", 1)[1]: v for k, v in other.items() if k.startswith("report_card.")},
                "refusals": {k.split(".", 1)[1]: v for k, v in other.items() if k.startswith("refused.")},
                "mcp": {k.split(".", 1)[1]: v for k, v in other.items() if k.startswith("mcp.")},
            }


STATS = Stats((_state_dir() / "stats.json") if _state_dir() else None)
