"""The examiner as a local JSON service: `whetstone serve`.

A deliberately small stdlib HTTP server — no framework dependency — exposing
the three product verbs to anything on the network that can speak JSON:

    GET  /status                     bank health (same shape as `whetstone status`)
    POST /grade   {system, answers}  grade stored answers keyed by item id
    POST /gate    {baseline, candidate, retained_probe?, policy?}

Design boundaries, stated rather than implied:
- Grading over HTTP accepts ANSWERS, not model endpoints. The service never
  fetches completions itself, so it cannot be tricked into shipping private
  prompts to an arbitrary URL. Callers grade their model on their side and
  submit the answers.
- Item prompts are never served. There is no GET /items. The bank's contents
  do not transit this API in either direction.
- Optional shared-secret auth via the X-Whetstone-Token header. This is a
  localhost/VPC service; put real authn in front of it before exposing it.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from bcv.examiner import ExaminerBank


def status_payload(bank: ExaminerBank) -> dict:
    from bcv.metabolism import metabolism_summary

    statuses: dict[str, int] = {}
    domains: dict[str, int] = {}
    for item in bank.items.values():
        statuses[item.status] = statuses.get(item.status, 0) + 1
        if item.status == "promoted":
            domains[item.domain] = domains.get(item.domain, 0) + 1
    return {
        "buckets": dict(sorted(statuses.items())),
        "promoted_by_domain": dict(sorted(domains.items())),
        "discriminating_items": sum(1 for item in bank.promoted_items() if item.discrimination() > 0),
        "graded_systems": sorted({system for item in bank.items.values() for system in item.graded}),
        "metabolism_history": metabolism_summary(bank.root),
    }


def grade_payload(bank: ExaminerBank, body: dict, config: dict) -> dict:
    from bcv.registry import plugin_for

    system = body.get("system")
    answers = body.get("answers")
    if not isinstance(system, str) or not system or not isinstance(answers, dict):
        raise ValueError("grade requires {'system': str, 'answers': {item_id: answer}}")
    grading = config.get("grading", {})
    context: dict = {
        "stress_ns": tuple(grading.get("stress_ns", (7, 8))),
        "seed": int(grading.get("seed", 0)),
        "scratch": str(bank.root / "registry_tmp"),
    }
    results: dict[str, bool] = {}
    unknown: list[str] = []
    for item in bank.promoted_items():
        if item.item_id not in answers:
            continue
        results[item.item_id] = plugin_for(item).grade(item, answers[item.item_id], context)
    unknown = sorted(set(answers) - set(results))
    if not results:
        raise ValueError("no submitted answers matched promoted items")
    bank.record_grades(system, results)
    bank.save()
    return {
        "system": system,
        "items": len(results),
        "passed": sum(results.values()),
        "results": results,
        "ignored_unknown_items": unknown,
    }


def gate_payload(bank: ExaminerBank, body: dict, config: dict) -> dict:
    from bcv.gate import GatePolicy, build_gate_report, latest_grade_event_results

    baseline = body.get("baseline")
    candidate = body.get("candidate")
    if not baseline or not candidate:
        raise ValueError("gate requires {'baseline': str, 'candidate': str}")
    policy_config = {**config.get("policy", {}), **(body.get("policy") or {})}
    policy = GatePolicy(
        min_gains=int(policy_config.get("min_gains", 1)),
        max_regressions=int(policy_config.get("max_regressions", 0)),
        confidence_alpha=float(policy_config.get("confidence_alpha", 0.05)),
        require_retained_probe=bool(policy_config.get("require_retained_probe", False)),
    )
    events = bank.root / "grade_events.jsonl"
    baseline_results = latest_grade_event_results(events, baseline)
    candidate_results = latest_grade_event_results(events, candidate)
    shared = sorted(set(baseline_results) & set(candidate_results))
    if not shared:
        raise ValueError("baseline and candidate share no graded items")
    report = build_gate_report(
        bank,
        baseline=baseline,
        candidate=candidate,
        baseline_results={item: baseline_results[item] for item in shared},
        candidate_results={item: candidate_results[item] for item in shared},
        retained_probe=body.get("retained_probe"),
        policy=policy,
    )
    report["paired_evidence"].pop("items_detail", None)  # keep item ids off the wire
    return report


class WhetstoneHandler(BaseHTTPRequestHandler):
    bank_root: str = ".bcv_runs/examiner"
    token: str | None = None
    config: dict = {}

    def log_message(self, format: str, *args) -> None:  # quiet by default
        pass

    def _authorized(self) -> bool:
        return self.token is None or self.headers.get("X-Whetstone-Token") == self.token

    def _reply(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, sort_keys=True).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if not self._authorized():
            return self._reply(401, {"error": "missing or wrong X-Whetstone-Token"})
        if self.path.rstrip("/") == "/status" or self.path == "/":
            return self._reply(200, status_payload(ExaminerBank(self.bank_root)))
        return self._reply(404, {"error": f"unknown path {self.path}; the bank's items are never served"})

    def do_POST(self) -> None:
        if not self._authorized():
            return self._reply(401, {"error": "missing or wrong X-Whetstone-Token"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
        except (ValueError, json.JSONDecodeError):
            return self._reply(400, {"error": "body must be JSON"})
        bank = ExaminerBank(self.bank_root)
        try:
            if self.path.rstrip("/") == "/grade":
                return self._reply(200, grade_payload(bank, body, self.config))
            if self.path.rstrip("/") == "/gate":
                return self._reply(200, gate_payload(bank, body, self.config))
        except (ValueError, KeyError, FileNotFoundError) as error:
            return self._reply(400, {"error": str(error)})
        return self._reply(404, {"error": f"unknown path {self.path}"})


def make_server(root: str | Path, port: int, token: str | None, config: dict | None = None) -> ThreadingHTTPServer:
    handler = type(
        "BoundHandler",
        (WhetstoneHandler,),
        {"bank_root": str(root), "token": token, "config": config or {}},
    )
    return ThreadingHTTPServer(("127.0.0.1", port), handler)


def serve(root: str | Path, port: int, token: str | None, config: dict | None = None) -> None:
    server = make_server(root, port, token, config)
    auth = "token-protected" if token else "NO auth (localhost only)"
    print(f"whetstone examiner service on http://127.0.0.1:{port} ({auth}) — bank: {root}")
    print("endpoints: GET /status, POST /grade, POST /gate. Item contents are never served.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
