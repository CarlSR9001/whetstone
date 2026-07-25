"""Shared rate-limit primitives for the public service surfaces.

Both the REST handlers and the MCP endpoint import the same limiter
instances, so a caller cannot double their budget by switching protocols.
"""

from __future__ import annotations

import threading
import time
from collections import deque


class SlidingWindowLimit:
    def __init__(self, requests: int, seconds: float, max_keys: int = 10_000) -> None:
        if requests <= 0 or seconds <= 0 or max_keys <= 0:
            raise ValueError("requests, seconds, and max_keys must be positive")
        self.requests = requests
        self.seconds = seconds
        self.max_keys = max_keys
        self.lock = threading.Lock()
        self.events: dict[str, deque[float]] = {}
        self._next_full_prune = 0.0
        self._prune_interval = min(60.0, max(1.0, seconds / 10.0))

    @staticmethod
    def _expire(queue: deque[float], cutoff: float) -> None:
        while queue and queue[0] <= cutoff:
            queue.popleft()

    def _prune_stale(self, now: float) -> None:
        cutoff = now - self.seconds
        for name, queue in list(self.events.items()):
            self._expire(queue, cutoff)
            if not queue:
                del self.events[name]
        self._next_full_prune = now + self._prune_interval

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self.lock:
            cutoff = now - self.seconds
            queue = self.events.get(key)
            if queue is not None:
                self._expire(queue, cutoff)
                if not queue:
                    del self.events[key]
                    queue = None

            if queue is None:
                if len(self.events) >= self.max_keys and now >= self._next_full_prune:
                    self._prune_stale(now)
                # Fail closed instead of evicting an active identity and giving
                # it a fresh quota. This also keeps memory strictly bounded.
                if len(self.events) >= self.max_keys:
                    return False
                queue = deque()
                self.events[key] = queue

            if len(queue) >= self.requests:
                return False
            queue.append(now)
            return True


# One budget per capability, shared across REST and MCP.
GENERAL_LIMIT = SlidingWindowLimit(60, 60)
HUNTER_LIMIT = SlidingWindowLimit(4, 600)
HUNTER_SLOT = threading.BoundedSemaphore(1)

# Tier 1 report cards: starting a session is cheap after warm-up but hands out
# exam prompts; grading holds the CPU. Both get deliberately small budgets.
REPORT_START_LIMIT = SlidingWindowLimit(4, 3600)
REPORT_SUBMIT_LIMIT = SlidingWindowLimit(8, 3600)
GRADE_SLOT = threading.BoundedSemaphore(1)
