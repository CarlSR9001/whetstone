"""The self-escalating forge: mines adversary graphs until each frontier is
provably exhausted, then says so instead of pretending to work.

Design (from first principles, and from watching v1 idle):

- A *rung* is an effort level: annealing restarts/steps plus the anneal
  frontier ``anneal_ns``. Rung 0 is the v1 configuration, which saturated the
  library at 901 entries and then burned CPU on +0 cycles for days.
- Escalation is evidence-driven: PATIENCE consecutive zero-addition cycles at
  a rung means that rung's frontier is mined out for that domain — climb.
  Any find resets the clock (the rung is still paying).
- The ladder is deliberately finite. Exhaustive verification is exponential in
  n, so "just go deeper" has a wall; when a domain runs out of rungs it is
  EXHAUSTED, and when every domain is exhausted the forge drops to sentinel
  mode — one probe per day and a loud, explicit signal that further growth
  requires a new domain module, which is a research decision no amount of
  meta-optimization can manufacture.
- Every find is synced (atomically) into the public service's state dir, where
  the report-card hatchery folds it into both the minting-certification and
  grading pools: the public exam hardens as the forge learns, and items stay
  fair because both sides use the same pool.

State and logs live in the working directory (the systemd unit points this at
the forge state dir). The refinery runs as a subprocess per cycle so memory is
returned between cycles and a wedged anneal can be timed out.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

# (restarts, steps, anneal_ns) — stress_ns stays fixed: it is the
# certification pool the *exam* uses, and must remain comparable across time.
LADDER: tuple[dict, ...] = (
    {"restarts": 4, "steps": 500, "anneal_ns": (8, 9, 10, 11)},      # v1 baseline (known saturated)
    {"restarts": 10, "steps": 1500, "anneal_ns": (8, 9, 10, 11)},
    {"restarts": 16, "steps": 4000, "anneal_ns": (9, 10, 11, 12)},
    {"restarts": 24, "steps": 8000, "anneal_ns": (10, 11, 12, 13)},
    {"restarts": 40, "steps": 12000, "anneal_ns": (11, 12, 13)},
)
STRESS_NS = (7, 8, 10)
PATIENCE = 3           # zero-cycles at a rung before climbing
MAX_ERRORS = 5         # consecutive failures before a domain is benched
SLEEP_ACTIVE = 600.0
SLEEP_SENTINEL = 86_400.0
CYCLE_TIMEOUT = 21_600  # 6h hard cap per refinery run
LIBRARY_NAME = "adversary_library.jsonl"
STATE_NAME = "forge_controller_state.json"


@dataclass
class DomainState:
    rung: int = 1  # rung 0 is the empirically-exhausted v1 config
    zeros_at_rung: int = 0
    consecutive_errors: int = 0
    exhausted: bool = False
    exhausted_reason: str = ""
    cycles: int = 0
    finds: int = 0


@dataclass
class ForgeState:
    domains: dict = field(default_factory=dict)  # name -> DomainState as dict
    total_cycles: int = 0

    def domain(self, name: str) -> DomainState:
        raw = self.domains.get(name)
        state = DomainState(**raw) if raw else DomainState()
        self.domains[name] = asdict(state)
        return state

    def put(self, name: str, state: DomainState) -> None:
        self.domains[name] = asdict(state)


class Controller:
    def __init__(
        self,
        work_dir: Path,
        domains: tuple[str, ...] = ("coloring", "mis"),
        runner=None,
        sleeper=time.sleep,
        sync_dir: str | None = None,
    ) -> None:
        self.work_dir = Path(work_dir)
        self.domain_names = domains
        self.runner = runner or self._subprocess_runner
        self.sleeper = sleeper
        self.sync_dir = sync_dir if sync_dir is not None else os.environ.get("WHETSTONE_SYNC_DIR")
        self.library = self.work_dir / LIBRARY_NAME
        self.state_path = self.work_dir / STATE_NAME
        self.state = self._load_state()

    # ------------------------------------------------------------- plumbing

    def _load_state(self) -> ForgeState:
        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
            return ForgeState(domains=raw.get("domains", {}), total_cycles=int(raw.get("total_cycles", 0)))
        except (OSError, ValueError):
            return ForgeState()

    def _save_state(self) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self.state), indent=1), encoding="utf-8")
        tmp.replace(self.state_path)

    def log(self, message: str) -> None:
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {message}"
        print(line, flush=True)
        with open(self.work_dir / "forge.log", "a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def library_count(self) -> int:
        try:
            return sum(1 for line in self.library.read_text(encoding="utf-8").splitlines() if line.strip())
        except OSError:
            return 0

    def dedupe_library(self) -> int:
        """Keep the first entry per graph_id+domain. Returns entries removed."""
        try:
            lines = [line for line in self.library.read_text(encoding="utf-8").splitlines() if line.strip()]
        except OSError:
            return 0
        seen: set[tuple[str, str]] = set()
        kept: list[str] = []
        for line in lines:
            try:
                raw = json.loads(line)
                key = (str(raw.get("graph_id")), str(raw.get("domain")))
            except ValueError:
                continue  # drop unparseable lines
            if key not in seen:
                seen.add(key)
                kept.append(line)
        removed = len(lines) - len(kept)
        if removed or len(kept) != len(lines):
            tmp = self.library.with_suffix(".tmp")
            tmp.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
            tmp.replace(self.library)
        return removed

    def sync_library(self) -> None:
        """Atomically publish the library where the public service reads it."""
        if not self.sync_dir or not self.library.exists():
            return
        target = Path(self.sync_dir) / LIBRARY_NAME
        try:
            tmp = target.with_suffix(".sync-tmp")
            tmp.write_bytes(self.library.read_bytes())
            tmp.replace(target)
            self.log(f"synced library ({self.library_count()} entries) -> {target}")
        except OSError as error:
            self.log(f"sync FAILED: {type(error).__name__}")

    # ------------------------------------------------------------- cycles

    def _subprocess_runner(self, domain: str, rung_config: dict) -> tuple[int, float]:
        command = [
            sys.executable, "-m", "bcv.refinery",
            "--domain", domain,
            "--max-n", "6",
            "--stress-ns", *[str(n) for n in STRESS_NS],
            "--anneal-ns", *[str(n) for n in rung_config["anneal_ns"]],
            "--restarts", str(rung_config["restarts"]),
            "--steps", str(rung_config["steps"]),
            "--library", str(self.library),
            "--seed", str(int.from_bytes(os.urandom(3), "big")),
        ]
        started = time.perf_counter()
        try:
            result = subprocess.run(
                command, cwd=self.work_dir, timeout=CYCLE_TIMEOUT,
                stdout=open(self.work_dir / "refinery.out", "ab"),
                stderr=open(self.work_dir / "refinery.err", "ab"),
            )
            return result.returncode, time.perf_counter() - started
        except subprocess.TimeoutExpired:
            return 124, time.perf_counter() - started

    def run_cycle(self, domain: str) -> str:
        """One refinery run for one domain; applies the escalation policy."""
        state = self.state.domain(domain)
        if state.exhausted:
            # Sentinel probe: one shot at the top rung.
            rung_index = len(LADDER) - 1
        else:
            rung_index = min(state.rung, len(LADDER) - 1)
        config = LADDER[rung_index]

        before = self.library_count()
        rc, duration = self.runner(domain, config)
        self.dedupe_library()
        additions = self.library_count() - before

        state.cycles += 1
        self.state.total_cycles += 1

        if rc != 0:
            state.consecutive_errors += 1
            if state.consecutive_errors >= MAX_ERRORS and not state.exhausted:
                state.exhausted = True
                state.exhausted_reason = f"benched after {MAX_ERRORS} consecutive errors (rc={rc})"
                self.log(f"BENCHED domain={domain}: {state.exhausted_reason}")
        else:
            state.consecutive_errors = 0
            if additions > 0:
                state.finds += additions
                state.zeros_at_rung = 0
                if state.exhausted:
                    state.exhausted = False
                    state.exhausted_reason = ""
                    self.log(f"REVIVED domain={domain}: sentinel probe found {additions} new adversaries")
                self.sync_library()
            else:
                state.zeros_at_rung += 1
                if state.zeros_at_rung >= PATIENCE and not state.exhausted:
                    if rung_index < len(LADDER) - 1:
                        state.rung = rung_index + 1
                        state.zeros_at_rung = 0
                        self.log(
                            f"ESCALATE domain={domain} rung {rung_index}->{state.rung} "
                            f"(frontier at rung {rung_index} mined out)"
                        )
                    else:
                        state.exhausted = True
                        state.exhausted_reason = f"top rung mined out after {PATIENCE} zero cycles"
                        self.log(f"EXHAUSTED domain={domain}: {state.exhausted_reason}")

        self.state.put(domain, state)
        self._save_state()
        line = (
            f"cycle={self.state.total_cycles} domain={domain} rung={rung_index} "
            f"restarts={config['restarts']} steps={config['steps']} anneal_ns={list(config['anneal_ns'])} "
            f"rc={rc} dur={duration:.0f}s library={self.library_count()} (+{max(additions, 0)})"
        )
        self.log(line)
        return line

    def all_exhausted(self) -> bool:
        return all(self.state.domain(name).exhausted for name in self.domain_names)

    def main_loop(self) -> None:
        self.log(
            f"forge-controller start pid={os.getpid()} ladder_rungs={len(LADDER)} "
            f"patience={PATIENCE} domains={list(self.domain_names)} sync_dir={self.sync_dir}"
        )
        removed = self.dedupe_library()
        if removed:
            self.log(f"startup dedupe removed {removed} duplicate library entries")
        self.sync_library()
        while True:
            for name in self.domain_names:
                self.run_cycle(name)
            if self.all_exhausted():
                self.log(
                    "FRONTIER EXHAUSTED: every domain is mined out at the top rung. "
                    "Sentinel mode (one probe per domain per day). Further growth requires "
                    "a NEW DOMAIN MODULE — that is a research decision, not a knob."
                )
                self.sleeper(SLEEP_SENTINEL)
            else:
                self.sleeper(SLEEP_ACTIVE)


def main() -> None:
    controller = Controller(Path.cwd())
    controller.main_loop()


if __name__ == "__main__":
    main()
