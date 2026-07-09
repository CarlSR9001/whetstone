from __future__ import annotations

import json
import sys
import threading

import pytest

from bcv.candidates import (
    CandidateError,
    CommandCandidate,
    OpenAICompatibleCandidate,
    StoredAnswerCandidate,
)


def test_command_candidate_round_trip():
    candidate = CommandCandidate(
        [sys.executable, "-c", "import sys; sys.stdout.write(sys.stdin.read().upper())"],
        timeout_seconds=30,
    )
    assert candidate.generate_text("hello gate") == "HELLO GATE"
    assert candidate.is_external is False


def test_command_candidate_failure_is_loud():
    candidate = CommandCandidate([sys.executable, "-c", "import sys; sys.exit(3)"], timeout_seconds=30)
    with pytest.raises(CandidateError, match="exited 3"):
        candidate.generate_text("x")


def test_stored_answers_load_and_refuse_free_prompts(tmp_path):
    path = tmp_path / "answers.jsonl"
    path.write_text(
        json.dumps({"item_id": "a1", "answer": "is_tree"}) + "\n"
        + json.dumps({"item_id": "a2", "answer": {"move": 3}}) + "\n",
        encoding="utf-8",
    )
    candidate = StoredAnswerCandidate(path)
    assert candidate.answer_for("a1") == "is_tree"
    assert candidate.answer_for("a2") == {"move": 3}
    assert candidate.answer_for("missing") is None
    with pytest.raises(CandidateError):
        candidate.generate_text("anything")


def test_external_detection_by_host():
    local = OpenAICompatibleCandidate(base_url="http://localhost:1234/v1", model="m")
    remote = OpenAICompatibleCandidate(base_url="https://api.openai.com/v1", model="gpt")
    assert local.is_external is False
    assert remote.is_external is True


def test_openai_compatible_against_stub_server():
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    seen = {}

    class Stub(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen["request"] = body
            reply = json.dumps(
                {"choices": [{"message": {"content": f"echo:{body['messages'][-1]['content']}"}}]}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Stub)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        candidate = OpenAICompatibleCandidate(
            base_url=f"http://127.0.0.1:{server.server_port}/v1", model="stub-model", timeout_seconds=10
        )
        assert candidate.generate_text("ping") == "echo:ping"
        assert seen["request"]["model"] == "stub-model"
        assert seen["request"]["temperature"] == 0.0
    finally:
        server.shutdown()
