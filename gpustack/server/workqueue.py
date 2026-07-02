"""A generic, controller-runtime-style work queue for reconcile loops.

This is the Python analog of client-go's ``RateLimitingInterface`` (the
``DelayingQueue`` + ``RateLimiter`` stack). It exists because in-flight
reconciles previously re-observed downstream state by writing their own status
to the DB, which re-fired the event bus and re-enqueued the item — a
self-perpetuating poll loop. A work queue with in-memory ``add_after`` lets a
controller re-observe on a timer without touching the DB.

Guarantees:

* **Per-keys serialization** — a key handed out by :meth:`WorkQueue.get` is not
  handed out again until :meth:`WorkQueue.done` is called for it. Events that
  arrive for an in-flight key are coalesced and re-queued on ``done``.
* **DELETED priority + stickiness** — a ``DELETED`` event jumps to the front of
  the queue, cancels any pending delayed entry for the same key, bypasses the
  dedup window, and (via the default coalescer) is not displaced by a later
  non-``DELETED`` event.
* **Delay heap** — :meth:`add_after` schedules an event to become addable after
  a delay, backed by a min-heap so the consumer waits exactly until the next due
  time rather than polling.
* **Dedup / debounce window** — optional: coalesces same-key events over a short
  window and emits the last, bounded by a max-wait cap so a steady stream can't
  starve processing.
* **Exponential backoff** — :meth:`add_rate_limited` re-queues with a capped,
  jittered exponential delay per key; :meth:`forget` resets it on success.

Concurrency model: the queue assumes a **single asyncio event loop with a single
consumer** calling :meth:`get`. Producers (``add`` / ``add_after`` /
``add_rate_limited``) are synchronous and never await, so they run to completion
without interleaving; the queue therefore needs no internal locking. Consumers
fan out by spawning one task per key returned from ``get`` and calling ``done``
when that task finishes — this preserves unbounded per-key concurrency while
keeping each key strictly serial.
"""

import asyncio
import heapq
import itertools
import random
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional, Set, Tuple

Keys = Tuple[Any, ...]


class WorkEventType(Enum):
    """Watch-style event kind, distinct from the bus's CREATED/UPDATED/DELETED.

    Callers map their domain events onto these (e.g. CREATED -> ADDED,
    UPDATED -> MODIFIED, DELETED -> DELETED).
    """

    ADDED = 1
    MODIFIED = 2
    DELETED = 3

    def __str__(self) -> str:
        return self.name


@dataclass
class WorkEvent:
    """A unit of work.

    ``keys`` is a hashable tuple uniquely identifying the target (e.g.
    ``(cluster_id, id)`` upstream or ``(cluster_id, namespace, name)``
    downstream). ``object`` carries the payload; its authority is caller-defined
    (a hint for upstream re-fetch, or the authoritative downstream object).
    """

    keys: Keys
    type: WorkEventType
    object: Any = None


def _default_coalesce(existing: WorkEvent, incoming: WorkEvent) -> WorkEvent:
    """Latest-wins, but a pending DELETED is sticky.

    A later non-DELETED event never displaces a pending DELETED; a DELETED
    always upgrades whatever is pending.
    """
    if (
        existing.type == WorkEventType.DELETED
        and incoming.type != WorkEventType.DELETED
    ):
        return existing
    return incoming


class ExponentialBackoff:
    """Per-keys capped exponential backoff with optional jitter.

    ``when(keys)`` records a failure and returns the delay for the next retry:
    ``min(base * 2**(n-1), cap)`` plus up to ``jitter`` fraction of that delay.
    ``forget(keys)`` resets the failure count on success.
    """

    def __init__(
        self,
        base: float = 1.0,
        cap: float = 300.0,
        jitter: float = 0.0,
        rand: Callable[[], float] = random.random,
    ):
        self._base = base
        self._cap = cap
        self._jitter = jitter
        self._rand = rand
        self._failures: Dict[Keys, int] = {}

    def when(self, keys: Keys) -> float:
        n = self._failures.get(keys, 0)
        self._failures[keys] = n + 1
        delay = min(self._base * (2**n), self._cap)
        if self._jitter:
            delay += delay * self._jitter * self._rand()
        return delay

    def forget(self, keys: Keys) -> None:
        self._failures.pop(keys, None)

    def failures(self, keys: Keys) -> int:
        return self._failures.get(keys, 0)


class WorkQueue:
    """See module docstring. Single-loop, single-consumer."""

    def __init__(
        self,
        *,
        dedup_window: float = 0.0,
        dedup_max_wait: float = 0.0,
        backoff_base: float = 1.0,
        backoff_max: float = 300.0,
        backoff_jitter: float = 0.0,
        coalesce: Optional[Callable[[WorkEvent, WorkEvent], WorkEvent]] = None,
        rand: Callable[[], float] = random.random,
    ):
        # ``dedup_window`` 0 disables debouncing (events go through immediately).
        # ``dedup_max_wait`` caps how long a debounced key can slide; it defaults
        # to ``dedup_window`` (a fixed, non-sliding window).
        self._dedup_window = dedup_window
        self._dedup_max_wait = dedup_max_wait or dedup_window
        self._coalesce = coalesce or _default_coalesce
        self._backoff = ExponentialBackoff(
            base=backoff_base, cap=backoff_max, jitter=backoff_jitter, rand=rand
        )

        # Coalesced payload per key that still needs processing ("dirty").
        self._pending: Dict[Keys, WorkEvent] = {}
        # Keys ready to be handed out by ``get`` (a subset of ``_pending``),
        # in FIFO order with DELETED bumped to the front.
        self._order: Deque[Keys] = deque()
        self._queued: Set[Keys] = set()
        # Keys currently being processed (handed out, not yet ``done``).
        self._processing: Set[Keys] = set()

        # Min-heap of (due_time, seq, event) for ``add_after``.
        self._delayed: List[Tuple[float, int, WorkEvent]] = []
        self._seq = itertools.count()
        # key -> (deadline, first_seen, event) for the dedup window.
        self._debounce: Dict[Keys, Tuple[float, float, WorkEvent]] = {}

        self._wakeup = asyncio.Event()

    # -- producers (synchronous) -------------------------------------------- #

    def add(self, event: WorkEvent) -> None:
        """Enqueue immediately, coalescing with any pending payload."""
        self._add(event, now=time.monotonic())

    def add_after(self, event: WorkEvent, delay: float) -> None:
        """Enqueue after ``delay`` seconds (no backoff). requeueAfter."""
        if delay <= 0:
            self.add(event)
            return
        heapq.heappush(
            self._delayed, (time.monotonic() + delay, next(self._seq), event)
        )
        self._wakeup.set()

    def add_rate_limited(self, event: WorkEvent) -> None:
        """Enqueue after a per-keys capped exponential backoff delay."""
        self.add_after(event, self._backoff.when(event.keys))

    def forget(self, keys: Keys) -> None:
        """Reset the backoff counter for ``keys`` (call on success)."""
        self._backoff.forget(keys)

    def failures(self, keys: Keys) -> int:
        return self._backoff.failures(keys)

    # -- consumer ----------------------------------------------------------- #

    async def get(self) -> WorkEvent:
        """Return the next ready event, waiting as needed.

        Blocks until a key is ready and not in-flight, promoting due delayed and
        debounced entries first. The returned key stays in-flight until
        :meth:`done` is called.
        """
        while True:
            now = time.monotonic()
            self._drain_due(now)

            if self._order:
                keys = self._order.popleft()
                self._queued.discard(keys)
                event = self._pending.pop(keys)
                self._processing.add(keys)
                return event

            timeout = self._time_to_next_due(now)
            # Clear-then-recheck with no intervening await: a synchronous
            # producer cannot slip in between, so no wakeup is lost.
            self._wakeup.clear()
            if self._order or self._has_due(time.monotonic()):
                continue
            try:
                if timeout is None:
                    await self._wakeup.wait()
                else:
                    await asyncio.wait_for(self._wakeup.wait(), timeout)
            except asyncio.TimeoutError:
                pass

    def done(self, keys: Keys) -> None:
        """Mark ``keys`` no longer in-flight; re-queue if it went dirty again."""
        self._processing.discard(keys)
        if keys in self._pending and keys not in self._queued:
            self._enqueue_ready(keys, self._pending[keys].type)
            self._wakeup.set()

    def __len__(self) -> int:
        return len(self._pending)

    # -- internals ---------------------------------------------------------- #

    def _add(self, event: WorkEvent, now: float) -> None:
        keys = event.keys
        is_deleted = event.type == WorkEventType.DELETED

        if keys in self._pending:
            event = self._coalesce(self._pending[keys], event)
            is_deleted = event.type == WorkEventType.DELETED
        self._pending[keys] = event

        if is_deleted:
            # Priority: cancel pending delay/debounce and jump the queue.
            self._cancel_delayed(keys)
            self._debounce.pop(keys, None)
            self._promote_to_ready(keys, front=True)
            self._wakeup.set()
            return

        # Debounce MODIFIED events when a window is configured. ADDED and
        # DELETED always pass through immediately.
        if self._dedup_window > 0 and event.type == WorkEventType.MODIFIED:
            first_seen = now
            if keys in self._debounce:
                first_seen = self._debounce[keys][1]
            deadline = min(now + self._dedup_window, first_seen + self._dedup_max_wait)
            self._debounce[keys] = (deadline, first_seen, event)
            self._wakeup.set()
            return

        self._promote_to_ready(keys, front=False)
        self._wakeup.set()

    def _promote_to_ready(self, keys: Keys, front: bool) -> None:
        """Move a pending key onto the ready queue (or defer if in-flight)."""
        if keys in self._processing:
            # Stays dirty in ``_pending``; ``done`` will re-queue it.
            return
        if keys in self._queued:
            if front:
                # DELETED bumping an already-queued key to the front.
                try:
                    self._order.remove(keys)
                except ValueError:
                    pass
                self._order.appendleft(keys)
            return
        self._enqueue_ready(keys, WorkEventType.DELETED if front else None)

    def _enqueue_ready(self, keys: Keys, type_: Optional[WorkEventType]) -> None:
        self._queued.add(keys)
        if type_ == WorkEventType.DELETED:
            self._order.appendleft(keys)
        else:
            self._order.append(keys)

    def _cancel_delayed(self, keys: Keys) -> None:
        if not self._delayed:
            return
        filtered = [item for item in self._delayed if item[2].keys != keys]
        if len(filtered) != len(self._delayed):
            heapq.heapify(filtered)
            self._delayed = filtered

    def _drain_due(self, now: float) -> None:
        """Promote due delayed entries and expired debounced entries."""
        while self._delayed and self._delayed[0][0] <= now:
            _, _, event = heapq.heappop(self._delayed)
            self._add(event, now)

        if self._debounce:
            expired = [
                k for k, (deadline, _, _) in self._debounce.items() if deadline <= now
            ]
            for keys in expired:
                _, _, event = self._debounce.pop(keys)
                # Re-add without re-entering the debounce buffer.
                if keys in self._pending:
                    event = self._coalesce(self._pending[keys], event)
                self._pending[keys] = event
                self._promote_to_ready(keys, front=False)

    def _has_due(self, now: float) -> bool:
        if self._delayed and self._delayed[0][0] <= now:
            return True
        return any(deadline <= now for deadline, _, _ in self._debounce.values())

    def _time_to_next_due(self, now: float) -> Optional[float]:
        due_times: List[float] = []
        if self._delayed:
            due_times.append(self._delayed[0][0])
        for deadline, _, _ in self._debounce.values():
            due_times.append(deadline)
        if not due_times:
            return None
        return max(0.0, min(due_times) - now)
