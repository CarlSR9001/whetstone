"""An Agent Client Protocol candidate: grade any ACP-speaking agent.

ACP (Agent Client Protocol) is the emerging standard for talking to coding
agents over newline-delimited JSON-RPC on stdio — Claude Code, Gemini CLI, and
a growing set of custom agents expose it. This adapter puts Whetstone on the
CLIENT side of that wire: spawn the agent, open a session, send each exam
prompt as a user turn, collect the agent's message chunks, and hand the final
text to the graders. Any agent that speaks ACP is now a system under exam with
zero integration work.

Deliberate boundaries:
- Whetstone is a grader, not an editor. The client advertises no filesystem
  or terminal capabilities, refuses fs/* requests, and answers every
  permission request with "cancelled". An agent that cannot answer an exam
  without touching the world fails that item honestly.
- One agent process, one session, sequential prompts: turn boundaries make
  chunk attribution unambiguous.
- The whole conversation runs inside the caller's machine; is_external is
  False. If the agent itself calls out to a frontier API, that is the agent
  operator's boundary to declare — the same rule as any local candidate.
"""

from __future__ import annotations

import json
import queue
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from bcv.candidates import CandidateError

PROTOCOL_VERSION = 1


@dataclass
class ACPCandidate:
    argv: list[str]
    timeout_seconds: float = 180.0
    cwd: str | None = None
    backend = "acp"
    is_external = False

    _process: subprocess.Popen | None = field(default=None, repr=False)
    _inbox: "queue.Queue[dict]" = field(default_factory=queue.Queue, repr=False)
    _reader: threading.Thread | None = field(default=None, repr=False)
    _session_id: str | None = field(default=None, repr=False)
    _next_id: int = field(default=0, repr=False)
    _write_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def provider(self) -> str:
        return "acp:" + " ".join(self.argv)

    # ------------------------------------------------------------- transport

    def _ensure_started(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        self._process = subprocess.Popen(
            self.argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self.cwd,
        )
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        self._handshake()

    def _read_loop(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        for raw in self._process.stdout:
            line = raw.decode("utf-8", "replace").strip()
            if not line:
                continue
            try:
                self._inbox.put(json.loads(line))
            except json.JSONDecodeError:
                continue  # agents may log to stdout; ignore non-JSON lines
        self._inbox.put({"_eof": True})

    def _send(self, message: dict) -> None:
        assert self._process is not None and self._process.stdin is not None
        data = (json.dumps(message, separators=(",", ":")) + "\n").encode("utf-8")
        with self._write_lock:
            try:
                self._process.stdin.write(data)
                self._process.stdin.flush()
            except OSError as error:
                raise CandidateError(f"agent stdin closed: {error}") from error

    def _request(self, method: str, params: dict, deadline: float, on_notification=None) -> dict:
        self._next_id += 1
        request_id = self._next_id
        self._send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise CandidateError(f"agent timed out during {method}")
            try:
                message = self._inbox.get(timeout=min(remaining, 1.0))
            except queue.Empty:
                continue
            if message.get("_eof"):
                stderr = b""
                if self._process and self._process.stderr:
                    stderr = self._process.stderr.read() or b""
                raise CandidateError(
                    f"agent exited during {method}: {stderr.decode('utf-8', 'replace')[:400]}"
                )
            if message.get("id") == request_id and ("result" in message or "error" in message):
                if "error" in message:
                    raise CandidateError(f"agent error on {method}: {message['error']}")
                return message.get("result") or {}
            if "method" in message and "id" in message:
                self._answer_agent_request(message)
            elif "method" in message and on_notification is not None:
                on_notification(message)

    def _answer_agent_request(self, message: dict) -> None:
        """The grader's side of agent-initiated requests: deny, honestly."""
        method = message.get("method", "")
        if method == "session/request_permission":
            result: dict = {"outcome": {"outcome": "cancelled"}}
            self._send({"jsonrpc": "2.0", "id": message["id"], "result": result})
            return
        self._send(
            {
                "jsonrpc": "2.0",
                "id": message["id"],
                "error": {
                    "code": -32601,
                    "message": f"whetstone grades answers; {method} is not available during an exam",
                },
            }
        )

    def _handshake(self) -> None:
        deadline = time.monotonic() + self.timeout_seconds
        init = self._request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "clientCapabilities": {"fs": {"readTextFile": False, "writeTextFile": False}},
            },
            deadline,
        )
        if init.get("protocolVersion") not in (PROTOCOL_VERSION, str(PROTOCOL_VERSION)):
            raise CandidateError(f"agent proposed unsupported ACP version: {init.get('protocolVersion')!r}")
        session = self._request(
            "session/new",
            {"cwd": str(Path(self.cwd or ".").resolve()), "mcpServers": []},
            deadline,
        )
        self._session_id = session.get("sessionId")
        if not self._session_id:
            raise CandidateError(f"agent returned no sessionId: {session}")

    # --------------------------------------------------------------- answers

    def generate_text(self, prompt: str, temperature: float = 0.0) -> str:
        self._ensure_started()
        chunks: list[str] = []

        def collect(notification: dict) -> None:
            if notification.get("method") != "session/update":
                return
            update = (notification.get("params") or {}).get("update") or {}
            if update.get("sessionUpdate") == "agent_message_chunk":
                content = update.get("content") or {}
                if content.get("type") == "text":
                    chunks.append(content.get("text", ""))

        result = self._request(
            "session/prompt",
            {"sessionId": self._session_id, "prompt": [{"type": "text", "text": prompt}]},
            time.monotonic() + self.timeout_seconds,
            on_notification=collect,
        )
        stop = result.get("stopReason")
        if stop not in ("end_turn", "max_tokens", None):
            raise CandidateError(f"agent stopped abnormally: {stop}")
        return "".join(chunks)

    def close(self) -> None:
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
        self._process = None

    def __del__(self) -> None:  # best-effort cleanup
        try:
            self.close()
        except Exception:
            pass
