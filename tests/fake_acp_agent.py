"""A minimal ACP agent for tests: newline-delimited JSON-RPC 2.0 over stdio.

Behavior knobs via argv:
  --answer TEXT     what the agent replies to every prompt (default: parrot)
  --ask-permission  send a session/request_permission request before answering
                    and echo the outcome into the reply
  --wrong-version   propose an unsupported protocol version
"""

from __future__ import annotations

import json
import sys


def send(message: dict) -> None:
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def main() -> None:
    args = sys.argv[1:]
    answer = None
    if "--answer" in args:
        answer = args[args.index("--answer") + 1]
    ask_permission = "--ask-permission" in args
    wrong_version = "--wrong-version" in args

    next_own_id = 1000
    pending_permission: dict[int, int] = {}  # our request id -> client prompt id

    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        message = json.loads(line)

        if message.get("method") == "initialize":
            send({
                "jsonrpc": "2.0", "id": message["id"],
                "result": {"protocolVersion": 99 if wrong_version else 1, "agentCapabilities": {}},
            })
        elif message.get("method") == "session/new":
            send({"jsonrpc": "2.0", "id": message["id"], "result": {"sessionId": "sess-1"}})
        elif message.get("method") == "session/prompt":
            session_id = message["params"]["sessionId"]
            prompt_text = message["params"]["prompt"][0]["text"]
            reply = answer if answer is not None else f"parrot:{prompt_text}"
            if ask_permission:
                next_own_id += 1
                send({
                    "jsonrpc": "2.0", "id": next_own_id, "method": "session/request_permission",
                    "params": {"sessionId": session_id, "toolCall": {"title": "rm -rf /"},
                               "options": [{"optionId": "yes", "name": "Allow"}]},
                })
                pending_permission[next_own_id] = message["id"]
                # stash reply context; outcome handled when the response arrives
                globals()["_reply_context"] = (session_id, message["id"], reply)
                continue
            _answer(session_id, message["id"], reply)
        elif "result" in message and message.get("id") in pending_permission:
            prompt_id = pending_permission.pop(message["id"])
            session_id, _, reply = globals()["_reply_context"]
            outcome = message["result"].get("outcome", {}).get("outcome", "?")
            _answer(session_id, prompt_id, f"{reply}|permission:{outcome}")


def _answer(session_id: str, prompt_id: int, reply: str) -> None:
    half = max(1, len(reply) // 2)
    for chunk in (reply[:half], reply[half:]):
        if chunk:
            send({
                "jsonrpc": "2.0", "method": "session/update",
                "params": {"sessionId": session_id,
                           "update": {"sessionUpdate": "agent_message_chunk",
                                      "content": {"type": "text", "text": chunk}}},
            })
    send({"jsonrpc": "2.0", "id": prompt_id, "result": {"stopReason": "end_turn"}})


if __name__ == "__main__":
    main()
