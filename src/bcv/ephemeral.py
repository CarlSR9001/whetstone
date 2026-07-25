"""Tier 1: disposable report-card sessions over an in-memory master mint.

The public service must never open a private bank, never persist exam/session
data, and never hold the CPU hostage to an anonymous caller. This module
satisfies all three at once:

- A background *hatchery* thread mints one master set of graph-repair items
  (coloring + MIS) when the service boots. Minting and grading share an
  in-memory observation cache plus the public forge library read from disk.
- A session is a random sample of master items behind fresh public ids. The
  caller gets prompts (these items are disposable by construction — the
  frontier they are minted from is public in the repository, so a session is a
  demonstration of the mechanism, not a credential).
- Submit is one-shot: the session is destroyed after grading or on TTL expiry.
  Grading takes a global single slot so two callers cannot stack CPU work.

No tool here can reach a persistent bank: there is none in the process.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import threading
import time

from bcv.ratelimit import GRADE_SLOT, REPORT_START_LIMIT, REPORT_SUBMIT_LIMIT
from bcv.stats import STATS


class TierError(ValueError):
    """User-visible refusal (limits, warm-up, bad session). Safe to echo."""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


SESSION_TTL_SECONDS = 900.0
MAX_ACTIVE_SESSIONS = 64
ITEMS_PER_SESSION = 6
MASTER_ITEMS_PER_DOMAIN = 24


class Hatchery:
    """Owns the master item set, the stress pools, and live sessions."""

    def __init__(
        self,
        domains: tuple[str, ...] = ("coloring", "mis"),
        max_n: int = 6,
        stress_ns: tuple[int, ...] = (7, 8),
        master_items_per_domain: int = MASTER_ITEMS_PER_DOMAIN,
        items_per_session: int = ITEMS_PER_SESSION,
        session_ttl: float = SESSION_TTL_SECONDS,
        max_active_sessions: int = MAX_ACTIVE_SESSIONS,
        library_path: str | None = None,
    ) -> None:
        self.domain_names = domains
        self.max_n = max_n
        self.stress_ns = stress_ns
        self.master_items_per_domain = master_items_per_domain
        self.items_per_session = items_per_session
        self.session_ttl = session_ttl
        self.max_active_sessions = max_active_sessions
        # The forge's mined-counterexample library. When present, both the
        # minting certification pool and the grading pools fold it in, so the
        # public exam hardens as the forge finds new adversaries — while items
        # stay fair (certified against the exact pool they are graded with).
        self.library_path = library_path if library_path is not None else os.environ.get("WHETSTONE_FORGE_LIBRARY")

        self.lock = threading.Lock()
        self.ready = False
        self.error: str | None = None
        self.warm_seconds: float | None = None
        self.master: dict[str, list] = {}
        self.pools: dict[str, list] = {}
        self.sessions: dict[str, dict] = {}
        self._thread: threading.Thread | None = None
        self._has_started = False
        self._observed_library_signature: tuple | None = None

    # ------------------------------------------------------------- warm-up

    def start(self) -> None:
        with self.lock:
            if self._has_started:
                return
            self._has_started = True
            self.ready = False
            self.error = None
            self._observed_library_signature = self._library_signature()
            self._thread = threading.Thread(target=self._warm, name="whetstone-hatchery", daemon=True)
        self._thread.start()

    def _library_signature(self) -> tuple | None:
        """Return the identity of the forge's latest atomic publication."""
        if not self.library_path:
            return None
        try:
            stat = os.stat(self.library_path)
        except FileNotFoundError:
            return ("missing",)
        except OSError as error:
            return ("unreadable", type(error).__name__, error.errno)
        # Controller.sync_library publishes with os.replace. The inode plus
        # nanosecond mtime and size therefore changes on every real sync without
        # hashing the entire library on every report-card request.
        return ("file", stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)

    def _refresh_if_library_changed(self) -> bool:
        """Start one background rebuild when the forge publishes a new file."""
        if not self.library_path:
            return False
        signature = self._library_signature()
        with self.lock:
            if (
                not self._has_started
                or self._thread is not None
                or signature == self._observed_library_signature
            ):
                return False
            self.ready = False
            self.error = None
            self._observed_library_signature = signature
            self._thread = threading.Thread(target=self._warm, name="whetstone-hatchery", daemon=True)
            thread = self._thread
        thread.start()
        return True

    def _warm(self) -> None:
        from pathlib import Path

        from bcv.domains import DOMAINS
        from bcv.examiner import mint_repair_items
        from bcv.refinery import _stress_pool

        started = time.perf_counter()
        with self.lock:
            self._has_started = True
            self.ready = False
            self.error = None
        try:
            # A forge sync can land while a rebuild is running. Discard that
            # mixed attempt and retry so minting and grading always consume one
            # stable library publication.
            for attempt in range(3):
                signature_before = self._library_signature()
                # Same library for pools and minting: item fairness requires
                # that a mined repair is certified against the pool it is
                # graded with.
                library = Path(self.library_path) if self.library_path else Path(".does_not_exist.jsonl")
                master: dict[str, list] = {}
                pools: dict[str, list] = {}
                for name in self.domain_names:
                    domain = DOMAINS[name]
                    pools[name] = _stress_pool(domain, self.stress_ns, 40, 0, library)
                    items = mint_repair_items(
                        domain,
                        [],
                        max_items=self.master_items_per_domain,
                        max_n=self.max_n,
                        stress_ns=self.stress_ns,
                        seed=0,
                        library_path=self.library_path,
                    )
                    master[name] = [item for item in items if item.status != "quarantined"]
                signature_after = self._library_signature()
                if signature_before == signature_after:
                    break
                if attempt == 2:
                    raise RuntimeError("forge library changed repeatedly during hatchery warm-up")
            with self.lock:
                self.master = master
                self.pools = pools
                self.ready = True
                self.error = None
                self.warm_seconds = round(time.perf_counter() - started, 1)
                self._observed_library_signature = signature_after
                if self._thread is threading.current_thread():
                    self._thread = None
        except Exception as error:  # surface in /api/health, never crash the service
            with self.lock:
                self.error = f"{type(error).__name__}"
                self.ready = False
                self._observed_library_signature = self._library_signature()
                if self._thread is threading.current_thread():
                    self._thread = None

    def status(self) -> dict:
        self._refresh_if_library_changed()
        library_entries = 0
        if self.library_path:
            try:
                from pathlib import Path as _Path

                text = _Path(self.library_path).read_text(encoding="utf-8")
                library_entries = sum(1 for line in text.splitlines() if line.strip())
            except OSError:
                library_entries = 0
        with self.lock:
            self._sweep_locked(time.time())
            return {
                "ready": self.ready,
                "error": self.error,
                "warm_seconds": self.warm_seconds,
                "master_items": {name: len(items) for name, items in self.master.items()},
                "active_sessions": len(self.sessions),
                "forge_library": {"configured": bool(self.library_path), "entries": library_entries},
            }

    # ------------------------------------------------------------- sessions

    def _sweep_locked(self, now: float) -> None:
        expired = [sid for sid, session in self.sessions.items() if session["expires_at"] <= now]
        for sid in expired:
            del self.sessions[sid]

    def _require_ready_locked(self) -> None:
        if self.error:
            raise TierError("report cards are unavailable: warm-up failed; the operator has been notified")
        if not self.ready:
            STATS.bump("report_card.refused_warming")
            raise TierError("the exam hatchery is still warming up; retry in about a minute", retryable=True)

    def start_session(self, client_ip: str) -> dict:
        self._refresh_if_library_changed()
        with self.lock:
            self._require_ready_locked()
        if not REPORT_START_LIMIT.allow(client_ip):
            STATS.bump("report_card.refused_limit")
            raise TierError("report-card session limit reached for this address; try again later")
        now = time.time()
        with self.lock:
            # A health poll can notice a forge sync and start a refresh between
            # the first readiness check and rate limiting.
            self._require_ready_locked()
            self._sweep_locked(now)
            if len(self.sessions) >= self.max_active_sessions:
                raise TierError("too many live sessions right now; retry shortly", retryable=True)
            import random

            rng = random.Random(secrets.randbits(64))
            chosen: list[tuple[str, object]] = []
            per_domain = max(1, self.items_per_session // max(1, len(self.master)))
            for name, items in self.master.items():
                take = min(per_domain, len(items))
                chosen.extend((name, item) for item in rng.sample(items, take))
            rng.shuffle(chosen)
            chosen = chosen[: self.items_per_session]
            if not chosen:
                raise TierError("no exam items are available", retryable=True)

            from bcv.examiner import repair_item_prompt

            session_id = secrets.token_urlsafe(16)
            item_map: dict[str, tuple[str, object]] = {}
            served: list[dict] = []
            for name, item in chosen:
                public_id = f"rc_{secrets.token_hex(4)}"
                item_map[public_id] = (name, item)
                served.append({"item_id": public_id, "domain": name, "prompt": repair_item_prompt(item)})
            self.sessions[session_id] = {
                "items": item_map,
                # A refresh may swap self.pools while this session is live.
                # Pin the exact pool snapshot that certified its master items.
                "pools": self.pools,
                "created_at": now,
                "expires_at": now + self.session_ttl,
                "client_ip": client_ip,
            }
        STATS.bump("report_card.sessions")
        return {
            "session_id": session_id,
            "expires_in_seconds": int(self.session_ttl),
            "items": served,
            "how_to_answer": (
                'Call report_card_submit once with {"session_id": ..., "answers": {"<item_id>": '
                '"<DSL predicate or the JSON reply the prompt asks for>"}}. One submission per '
                "session; unanswered items are graded as failures."
            ),
            "honesty_note": (
                "Disposable practice cohort. Items are minted from the repository's public "
                "frontier and graded by checker specs (any verified strict refinement passes; "
                "no answer key exists). Support retention is reported per passing item: a "
                "trivially narrow refinement passes the checker but flags as degenerate "
                "narrowing. This demonstrates the promotion-gate mechanism; it is not a "
                "private-bank credential."
            ),
        }

    def submit(self, session_id: str, answers: dict, client_ip: str) -> dict:
        if not isinstance(answers, dict):
            raise TierError("answers must be an object mapping item_id to your answer text")
        if not REPORT_SUBMIT_LIMIT.allow(client_ip):
            raise TierError("grading limit reached for this address; try again later")
        now = time.time()
        with self.lock:
            self._sweep_locked(now)
            session = self.sessions.pop(session_id, None)  # one-shot: gone even if grading fails
        if session is None:
            raise TierError("unknown or expired session (sessions are one-shot and time out)")
        if not GRADE_SLOT.acquire(blocking=False):
            # The session is spent — deliberately. A caller who hits the busy
            # slot restarts a session rather than queueing CPU work.
            raise TierError("grading worker is busy and this session is now spent; start a new session", retryable=True)
        try:
            return self._grade(session, answers, now)
        finally:
            GRADE_SLOT.release()

    def _grade(self, session: dict, answers: dict, now: float) -> dict:
        from bcv.domains import DOMAINS
        from bcv.examiner import grade_repair_answer
        from bcv.graph_agent import compile_feature_expression
        from bcv.refinery import _observe_all
        from bcv.transformers_client import extract_json
        from pathlib import Path

        graded: list[dict] = []
        passed_by_domain: dict[str, list[int]] = {}
        retentions: list[float] = []
        for public_id, (domain_name, item) in session["items"].items():
            raw = answers.get(public_id)
            expression: str | None = None
            if isinstance(raw, str) and raw.strip():
                parsed = extract_json(raw)
                if isinstance(parsed, dict) and isinstance(parsed.get("repair_expression"), str):
                    expression = parsed["repair_expression"]
                else:
                    expression = raw.strip()
            elif isinstance(raw, dict) and isinstance(raw.get("repair_expression"), str):
                expression = raw["repair_expression"]
            ok = bool(expression) and grade_repair_answer(
                item, expression, max_n=self.max_n, pool=session["pools"][domain_name]
            )
            row = {
                "item_id": public_id,
                "domain": domain_name,
                "answered": raw is not None,
                "passed": ok,
            }
            if ok:
                # Mode-collapse diagnostic: the checker spec accepts ANY verified
                # strict refinement, including a degenerate ultra-narrow one. We
                # report how much of the original's clean support the repair
                # retains instead of pretending the game does not exist.
                observations = _observe_all(DOMAINS[domain_name], self.max_n, Path(".unused"))
                predicate = compile_feature_expression(expression)
                original = compile_feature_expression(item.payload["original_expression"])
                clean = sum(1 for obs in observations if original(obs) and obs.greedy_is_optimal)
                kept = sum(1 for obs in observations if predicate(obs))
                retention = round(kept / clean, 4) if clean else 0.0
                row["support_retention"] = retention
                row["degenerate_narrowing"] = retention < 0.05
                retentions.append(retention)
            counts = passed_by_domain.setdefault(domain_name, [0, 0])
            counts[0] += int(ok)
            counts[1] += 1
            graded.append(row)
        total = len(graded)
        passed = sum(1 for row in graded if row["passed"])
        median_retention = sorted(retentions)[len(retentions) // 2] if retentions else None
        STATS.bump("report_card.graded")
        STATS.bump("report_card.items_graded", total)
        STATS.bump("report_card.items_passed", passed)
        for retention in retentions:
            STATS.retention_bucket(retention)
        item_ids = sorted(session["items"])
        answer_blob = json.dumps({key: answers.get(key) for key in item_ids}, sort_keys=True, default=str)
        return {
            "passed": passed,
            "total": total,
            "per_domain": {name: {"passed": p, "total": t} for name, (p, t) in sorted(passed_by_domain.items())},
            "items": graded,
            "verdict_line": (
                f"{passed}/{total} verified repairs — checker-spec graded (any verified strict "
                "refinement passes; no answer key exists)."
                + (
                    f" Median support retention {median_retention:.0%}"
                    + (" — degenerate-narrowing diagnostic FIRED." if median_retention < 0.05 else ".")
                    if median_retention is not None
                    else ""
                )
            ),
            "median_support_retention": median_retention,
            "item_set_sha256": hashlib.sha256(json.dumps(item_ids).encode("utf-8")).hexdigest(),
            "answers_sha256": hashlib.sha256(answer_blob.encode("utf-8")).hexdigest(),
            "graded_at_unix": int(now),
            "disposable_cohort": True,
            "session_spent": True,
        }


_HATCHERY: Hatchery | None = None
_HATCHERY_LOCK = threading.Lock()


def hatchery() -> Hatchery:
    global _HATCHERY
    with _HATCHERY_LOCK:
        if _HATCHERY is None:
            _HATCHERY = Hatchery()
        return _HATCHERY


def ensure_started() -> Hatchery:
    instance = hatchery()
    instance.start()
    return instance
