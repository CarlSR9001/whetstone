"""Shared rate-limit primitives for the public service surfaces.

Both the REST handlers and the MCP endpoint import the same limiter
instances, so a caller cannot double their budget by switching protocols.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque


class SlidingWindowLimit:
    def __init__(self, requests: int, seconds: float) -> None:
        self.requests = requests
        self.seconds = seconds
        self.lock = threading.Lock()
        self.events: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self.lock:
            queue = self.events[key]
            while queue and queue[0] <= now - self.seconds:
                queue.popleft()
            if len(queue) >= self.requests:
                return False
            queue.append(now)
            if len(self.events) > 10_000:
                self.events = defaultdict(deque, {name: values for name, values in self.events.items() if values})
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
