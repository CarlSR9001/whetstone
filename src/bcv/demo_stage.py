"""The demo stage: a thin presentation shell over the real engine.

Everything on the stage is compiled at request time from the committed
receipts in results/ — no number on screen exists anywhere except in an
artifact the repo already publishes. The one live element is real too: the
"run it now" button executes the actual investor demo (mint, quarantine,
grade, gate) on the presenting machine's CPU and shows whatever it returns.

This is a translator, not a product surface: it serves localhost only, reads
receipts, and exposes nothing an exam bank holds.

Run: $env:PYTHONPATH='src'; python -m bcv.cli stage   (then open the URL)
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

RESULTS = Path("results")
PAGE = Path("docs/stage.html")


def _load(name: str) -> dict:
    path = RESULTS / name
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def compile_story() -> dict:
    """The whole talk, as data, from committed receipts only."""
    same_bank = _load("local_fastcontext_same_bank_receipt.json")
    gen3 = _load("local_fastcontext_gen3_routed_receipt.json")
    ladder = _load("cross_scale_ladder_receipt.json")
    attack = _load("paraphrase_attack_receipt.json")
    exposure = _load("gtp_log_exposure_receipt.json")
    bakeoff = _load("engine_bank_bakeoff_receipt.json")
    replenish = _load("engine_bank_replenishment_receipt.json")

    systems = same_bank.get("systems", {})
    base = systems.get("fastcontext_4b_base_local", {})
    gen2 = systems.get("fastcontext_4b_gen2_local", {})
    gen2_gate = next(
        (g for g in same_bank.get("gates", []) if g.get("regression_policy") == "strict"), {}
    )
    gen3_gates = gen3.get("gates", [])
    gen3_vs_base = next((g for g in gen3_gates if "base" in g.get("baseline", "")), {})
    gen3_vs_32b = next((g for g in gen3_gates if "32b" in g.get("baseline", "")), {})
    qwen32 = next(
        (r for r in ladder.get("ladder", []) if r.get("system") == "qwen25_32b"), {}
    )

    return {
        "decision": {
            "bank_items": gen3.get("bank", {}).get("items"),
            "bank_sha256": gen3.get("bank", {}).get("bank_sha256", "")[:16],
            "hardware": gen3.get("hardware", ""),
            "base": {"total": base.get("total"), "by_domain": base.get("by_domain", {})},
            "gen2": {
                "total": gen2.get("total"),
                "by_domain": gen2.get("by_domain", {}),
                "gains": gen2_gate.get("gains"),
                "regressions": gen2_gate.get("regressions"),
                "p": gen2_gate.get("exact_mcnemar_two_sided_p"),
                "verdict": gen2_gate.get("verdict"),
                "reason": (gen2_gate.get("reasons") or [""])[0],
                "regression_domain": "code",
            },
            "gen3": {
                "total": gen3.get("system", {}).get("by_domain", {})
                and f"{sum(int(v.split('/')[0]) for v in gen3['system']['by_domain'].values())}/48",
                "by_domain": gen3.get("system", {}).get("by_domain", {}),
                "gains": gen3_vs_base.get("gains"),
                "regressions": gen3_vs_base.get("regressions"),
                "p": gen3_vs_base.get("exact_mcnemar_two_sided_p"),
                "verdict": gen3_vs_base.get("verdict"),
                "mechanism": gen3.get("mechanism", ""),
                "repeats": gen3.get("repeated_grading", {}),
            },
            "vs_32b": {
                "qwen_total": qwen32.get("total"),
                "gains": gen3_vs_32b.get("gains"),
                "regressions": gen3_vs_32b.get("regressions"),
                "p": gen3_vs_32b.get("exact_mcnemar_two_sided_p"),
                "verdict": gen3_vs_32b.get("verdict"),
            },
        },
        "ladder": {
            "rungs": [
                {"label": r["label"], "total": r["total"], "by_domain": r["by_domain"]}
                for r in ladder.get("ladder", [])
            ],
            "gates": ladder.get("gates", []),
            "findings": ladder.get("findings", []),
        },
        "attack": {
            "rows": attack.get("corpus", {}).get("rows"),
            "distinct": attack.get("corpus", {}).get("truly_different"),
            "equivalent": attack.get("corpus", {}).get("truly_equivalent_at_horizon"),
            "curve": attack.get("fingerprint_calibration_curve", {}),
            "upper_bound": attack.get("uncertainty", {}).get(
                "one_sided_95pct_upper_bound_on_false_positive_rate_after_zero_errors"
            ),
            "attacker": attack.get("attacker", ""),
        },
        "incident": {
            "what": exposure.get("incident", {}).get("what", ""),
            "amplifier": exposure.get("incident", {}).get("amplifier", ""),
            "audits": exposure.get("remediation", {}).get("audits", []),
            "policy": exposure.get("remediation", {}).get("policy", ""),
            "metabolism": replenish.get("metabolism", {}),
        },
        "bakeoff": {
            "baseline": bakeoff.get("systems", {}).get("baseline", {}),
            "candidate": bakeoff.get("systems", {}).get("candidate", {}),
            "gate_strict": {
                k: v for k, v in bakeoff.get("gate_strict", {}).items() if k != "resolution"
            },
            "regression_classifications": bakeoff.get("gate_strict", {}).get(
                "regression_classifications", []
            ),
        },
        "live_reference": _reference_result(),
        "receipts_on_disk": sorted(p.name for p in RESULTS.glob("*receipt*.json")),
    }


def _reference_result() -> dict:
    """A committed real run, used as a graceful fallback so the button never
    dead-ends in front of a room. Every number in it came from an actual run."""
    ref = _load("stage_live_reference.json")
    ref["cached"] = True
    return ref


class _LiveRun:
    """One live evaluation at a time, streaming REAL phase events.

    The engine calls back at each actual pipeline step (buffer written, mint
    finished, quarantine done, each system graded, decision, retirement) with
    the numbers it just computed; the page polls and renders events as they
    arrive. Nothing is animated on a timer — if the engine is slow, the screen
    is slow, which is exactly the honesty the demo sells."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.events: list[dict] = []
        self.running = False
        self.started_at = 0.0

    def _emit(self, phase: str, data: dict) -> None:
        import time

        with self.lock:
            self.events.append({
                "seq": len(self.events),
                "phase": phase,
                "data": data,
                "t": round(time.time() - self.started_at, 1),
            })

    def start(self) -> dict:
        import time

        with self.lock:
            if self.running:
                return {"error": "a live run is already in progress"}
            self.running = True
            self.events = []
            self.started_at = time.time()
        threading.Thread(target=self._work, daemon=True).start()
        return {"started": True}

    def _work(self) -> None:
        from bcv.demo_investor import DemoConfig, run_demo

        try:
            report = run_demo(DemoConfig(
                root=Path(".bcv_runs/stage_live"), quiet=True, on_phase=self._emit,
            ))
            report["cached"] = False
            self._emit("done", report)
        except Exception as error:
            fallback = _reference_result()
            fallback["engine_error"] = f"{type(error).__name__}"
            self._emit("fallback", fallback)
        finally:
            with self.lock:
                self.running = False

    def snapshot(self, since: int = 0) -> dict:
        with self.lock:
            return {"events": self.events[since:], "running": self.running}


LIVE = _LiveRun()


class StageHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:
        pass

    def _reply(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, payload: dict) -> None:
        self._reply(code, json.dumps(payload, sort_keys=True).encode("utf-8"), "application/json")

    def do_GET(self) -> None:
        raw_path, _, query = self.path.partition("?")
        path = raw_path.rstrip("/") or "/"
        if path == "/":
            if not PAGE.exists():
                return self._reply(500, b"docs/stage.html missing", "text/plain")
            return self._reply(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        if path == "/api/story":
            return self._json(200, compile_story())
        if path == "/api/live_events":
            since = 0
            for part in query.split("&"):
                if part.startswith("since="):
                    try:
                        since = int(part.split("=", 1)[1])
                    except ValueError:
                        pass
            return self._json(200, LIVE.snapshot(since))
        return self._json(404, {"error": "unknown path"})

    def do_POST(self) -> None:
        if self.path.rstrip("/") == "/api/live":
            return self._json(200, LIVE.start())
        return self._json(404, {"error": "unknown path"})


def serve_stage(port: int = 8990, open_browser: bool = False) -> None:
    url = f"http://127.0.0.1:{port}"
    try:
        server = ThreadingHTTPServer(("127.0.0.1", port), StageHandler)
    except OSError:
        # Port already serving: the stage is (almost certainly) up — just open it.
        if open_browser:
            import webbrowser

            webbrowser.open(url)
            print(f"stage already running — opened {url}")
            return
        raise
    print(f"whetstone stage on {url}  (localhost only; live run available)")
    if open_browser:
        import webbrowser

        threading.Timer(0.6, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    # Self-locating so `python src/bcv/demo_stage.py` works from anywhere:
    # receipts and the page are read relative to the repo root.
    import os
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    os.chdir(repo_root)
    sys.path.insert(0, str(repo_root / "src"))
    args = [a for a in sys.argv[1:] if a != "--open"]
    serve_stage(int(args[0]) if args else 8990, open_browser="--open" in sys.argv)
