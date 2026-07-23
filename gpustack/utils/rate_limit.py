"""Small bounded, process-local rate limiter for public HTTP endpoints."""

import threading
import time
from collections import OrderedDict, deque
from typing import Deque


class KeyedRateLimiter:
    """Sliding-window limiter with bounded key and timestamp storage.

    This protects each GPUStack server process from accidental or abusive
    bursts. Multi-replica deployments should additionally enforce a global
    limit at their trusted ingress.
    """

    def __init__(self, max_requests: int, window_seconds: int, max_keys: int = 10_000):
        if max_requests < 1 or window_seconds < 1 or max_keys < 1:
            raise ValueError("Rate-limit values must be positive")
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._max_keys = max_keys
        self._requests: OrderedDict[str, Deque[float]] = OrderedDict()
        self._lock = threading.Lock()

    def check(self, key: str) -> int:
        """Record a request and return zero, or seconds until retry is allowed."""
        now = time.monotonic()
        cutoff = now - self._window_seconds
        normalized_key = key or "unknown"

        with self._lock:
            attempts = self._requests.get(normalized_key)
            if attempts is None:
                if len(self._requests) >= self._max_keys:
                    self._requests.popitem(last=False)
                attempts = deque()
                self._requests[normalized_key] = attempts
            else:
                self._requests.move_to_end(normalized_key)

            while attempts and attempts[0] <= cutoff:
                attempts.popleft()
            if len(attempts) >= self._max_requests:
                return max(1, int(attempts[0] + self._window_seconds - now) + 1)

            attempts.append(now)
            return 0

    def clear(self) -> None:
        """Clear limiter state, primarily for isolated tests."""
        with self._lock:
            self._requests.clear()
