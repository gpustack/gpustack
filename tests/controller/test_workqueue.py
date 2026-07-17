"""Unit tests for the generic controller work queue.

Covers the controller-runtime-style semantics the queue must guarantee:
per-keys serialization, DELETED priority + stickiness, the delay heap
(``add_after``) with second-precision requeue, the optional dedup/debounce
window with a max-wait cap, and the exponential backoff rate limiter
(``add_rate_limited`` / ``forget``).

Timing tests use small real delays and generous slack so they stay robust
without a fake clock; the backoff math is exercised deterministically via the
pure ``ExponentialBackoff`` helper.
"""

import asyncio
import time

import pytest

from gpustack.server.workqueue import (
    ExponentialBackoff,
    WorkEvent,
    WorkEventType,
    WorkQueue,
)


def _ev(key, type_=WorkEventType.MODIFIED, obj=None):
    return WorkEvent(keys=(key,), type=type_, object=obj)


# --------------------------------------------------------------------------- #
# Core add / get / done + coalescing
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_add_get_done_basic():
    q = WorkQueue()
    q.add(_ev("a", WorkEventType.ADDED, obj=1))

    event = await asyncio.wait_for(q.get(), timeout=1)
    assert event.keys == ("a",)
    assert event.object == 1
    q.done(event.keys)
    assert len(q) == 0


@pytest.mark.asyncio
async def test_coalesce_latest_wins():
    q = WorkQueue()
    q.add(_ev("a", WorkEventType.MODIFIED, obj="old"))
    q.add(_ev("a", WorkEventType.MODIFIED, obj="new"))

    assert len(q) == 1  # coalesced into a single slot
    event = await asyncio.wait_for(q.get(), timeout=1)
    assert event.object == "new"


@pytest.mark.asyncio
async def test_deleted_sticky_default_coalesce():
    q = WorkQueue()
    q.add(_ev("a", WorkEventType.MODIFIED, obj="m"))
    q.add(_ev("a", WorkEventType.DELETED, obj="d"))
    # A later non-DELETED event must not displace a pending DELETED.
    q.add(_ev("a", WorkEventType.MODIFIED, obj="m2"))

    event = await asyncio.wait_for(q.get(), timeout=1)
    assert event.type == WorkEventType.DELETED
    assert event.object == "d"


@pytest.mark.asyncio
async def test_deleted_takes_priority_front():
    q = WorkQueue()
    q.add(_ev("a"))
    q.add(_ev("b"))
    q.add(_ev("c", WorkEventType.DELETED))

    first = await asyncio.wait_for(q.get(), timeout=1)
    assert first.keys == ("c",)  # DELETED jumps ahead of a/b


@pytest.mark.asyncio
async def test_deleted_bumps_already_queued_key_to_front():
    q = WorkQueue()
    q.add(_ev("a"))
    q.add(_ev("b"))
    # b is already queued behind a; a DELETED for b moves it to the front.
    q.add(_ev("b", WorkEventType.DELETED))

    first = await asyncio.wait_for(q.get(), timeout=1)
    assert first.keys == ("b",)
    assert first.type == WorkEventType.DELETED


# --------------------------------------------------------------------------- #
# Per-keys serialization
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_per_keys_serialization():
    q = WorkQueue()
    q.add(_ev("a", obj="a1"))
    q.add(_ev("b"))

    first = await asyncio.wait_for(q.get(), timeout=1)
    assert first.keys == ("a",)  # a is now in-flight

    # A new event for the in-flight key must not be handed out again...
    q.add(_ev("a", obj="a2"))
    second = await asyncio.wait_for(q.get(), timeout=1)
    assert second.keys == ("b",)  # ...so b comes next, not a

    q.done(("a",))
    third = await asyncio.wait_for(q.get(), timeout=1)
    assert third.keys == ("a",)  # a re-handed only after done(), with latest payload
    assert third.object == "a2"


# --------------------------------------------------------------------------- #
# Delay heap: add_after / requeueAfter precision
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_add_after_promotes_after_delay():
    q = WorkQueue()
    start = time.monotonic()
    q.add_after(_ev("a"), 0.08)

    event = await asyncio.wait_for(q.get(), timeout=1)
    elapsed = time.monotonic() - start
    assert event.keys == ("a",)
    assert elapsed >= 0.06  # roughly honored the delay


@pytest.mark.asyncio
async def test_add_after_last_schedule_wins_per_key():
    # Re-scheduling the same key via add_after cancels the earlier delayed entry
    # so only the newest event (payload + delay) survives — no stale replay.
    q = WorkQueue()
    q.add_after(_ev("a", obj="old"), 10)  # far future, must be cancelled
    q.add_after(_ev("a", obj="new"), 0.05)  # supersedes it

    event = await asyncio.wait_for(q.get(), timeout=1)  # must not wait ~10s
    assert event.object == "new"
    with pytest.raises(asyncio.TimeoutError):  # no stale "old" left behind
        await asyncio.wait_for(q.get(), timeout=0.1)


@pytest.mark.asyncio
async def test_immediate_beats_delayed():
    q = WorkQueue()
    q.add_after(_ev("late"), 0.08)
    q.add(_ev("now"))

    first = await asyncio.wait_for(q.get(), timeout=1)
    assert first.keys == ("now",)
    second = await asyncio.wait_for(q.get(), timeout=1)
    assert second.keys == ("late",)


@pytest.mark.asyncio
async def test_deleted_cancels_pending_delayed():
    q = WorkQueue()
    q.add_after(_ev("a"), 10)  # far future
    q.add(_ev("a", WorkEventType.DELETED))  # should purge the delayed entry

    event = await asyncio.wait_for(q.get(), timeout=1)  # must not wait ~10s
    assert event.type == WorkEventType.DELETED


# --------------------------------------------------------------------------- #
# Dedup / debounce window
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_dedup_window_disabled_by_default():
    q = WorkQueue()  # dedup_window defaults to 0 -> immediate
    q.add(_ev("a", WorkEventType.MODIFIED, obj="v"))
    event = await asyncio.wait_for(q.get(), timeout=0.05)
    assert event.object == "v"


@pytest.mark.asyncio
async def test_dedup_window_debounces_modified():
    q = WorkQueue(dedup_window=0.08)
    q.add(_ev("a", WorkEventType.MODIFIED, obj="v1"))
    q.add(_ev("a", WorkEventType.MODIFIED, obj="v2"))

    assert len(q) == 1
    start = time.monotonic()
    event = await asyncio.wait_for(q.get(), timeout=1)
    elapsed = time.monotonic() - start
    assert event.object == "v2"  # emits the last within the window
    assert elapsed >= 0.06


@pytest.mark.asyncio
async def test_deleted_bypasses_dedup_window():
    q = WorkQueue(dedup_window=10)
    q.add(_ev("a", WorkEventType.MODIFIED, obj="v"))
    q.add(_ev("a", WorkEventType.DELETED))

    event = await asyncio.wait_for(q.get(), timeout=1)  # must not wait ~10s
    assert event.type == WorkEventType.DELETED


@pytest.mark.asyncio
async def test_dedup_max_wait_caps_window():
    # window is huge but max_wait is small: the item must fire at ~max_wait,
    # proving the max-wait cap bounds the (potentially sliding) window.
    q = WorkQueue(dedup_window=10, dedup_max_wait=0.08)
    start = time.monotonic()
    q.add(_ev("a", WorkEventType.MODIFIED, obj="v"))

    event = await asyncio.wait_for(q.get(), timeout=1)
    elapsed = time.monotonic() - start
    assert event.object == "v"
    assert elapsed < 1.0  # nowhere near the 10s window
    assert elapsed >= 0.06


@pytest.mark.asyncio
async def test_added_bypass_does_not_resurrect_stale_debounced_event():
    # An ADDED arriving mid-window bypasses the debounce and is promoted
    # immediately; the leftover debounced snapshot must NOT resurrect a stale
    # payload once its deadline passes (``_pending`` is authoritative).
    q = WorkQueue(dedup_window=0.05)
    q.add(_ev("a", WorkEventType.MODIFIED, obj="stale"))  # debounced
    q.add(_ev("a", WorkEventType.ADDED, obj="fresh"))  # bypasses the window

    event = await asyncio.wait_for(q.get(), timeout=0.2)
    assert event.object == "fresh"  # promoted immediately, latest payload
    q.done(("a",))

    # Let the stale debounce deadline pass; nothing must surface.
    await asyncio.sleep(0.1)
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(q.get(), timeout=0.1)
    assert len(q) == 0


# --------------------------------------------------------------------------- #
# Exponential backoff rate limiter
# --------------------------------------------------------------------------- #


def test_backoff_exponential_capped_and_reset():
    b = ExponentialBackoff(base=1.0, cap=8.0, jitter=0.0)
    keys = ("a",)
    assert b.when(keys) == 1.0
    assert b.when(keys) == 2.0
    assert b.when(keys) == 4.0
    assert b.when(keys) == 8.0
    assert b.when(keys) == 8.0  # capped
    assert b.failures(keys) == 5

    b.forget(keys)
    assert b.failures(keys) == 0
    assert b.when(keys) == 1.0  # reset back to base


def test_backoff_jitter_within_bounds():
    b = ExponentialBackoff(base=1.0, cap=100.0, jitter=0.5, rand=lambda: 1.0)
    # base delay 1.0 + up to 50% jitter -> 1.5 when rand()==1.0
    assert b.when(("a",)) == pytest.approx(1.5)


def test_backoff_is_per_keys():
    b = ExponentialBackoff(base=1.0, cap=100.0, jitter=0.0)
    assert b.when(("a",)) == 1.0
    assert b.when(("b",)) == 1.0  # independent counters
    assert b.when(("a",)) == 2.0


@pytest.mark.asyncio
async def test_add_rate_limited_increments_failures_and_delays():
    q = WorkQueue(backoff_base=0.05, backoff_max=10, backoff_jitter=0.0)
    start = time.monotonic()
    q.add_rate_limited(_ev("a"))
    assert q.failures(("a",)) == 1

    event = await asyncio.wait_for(q.get(), timeout=1)
    elapsed = time.monotonic() - start
    assert event.keys == ("a",)
    assert elapsed >= 0.03  # honored ~base delay

    q.forget(("a",))
    assert q.failures(("a",)) == 0
