"""Track when this process started observing cluster liveness.

Worker liveness is derived from the absolute ``workers.heartbeat_time`` column.
A server that was stopped for longer than a liveness grace period would
otherwise read its own downtime as worker downtime and reschedule healthy
instances. The observation start is a floor for those checks: time before this
process started observing cannot count as worker silence.

The timestamp is deliberately process-local. Persisting it would not survive a
database restored from backup, which carries the stale ``heartbeat_time`` back
with it, and a new leader must observe the cluster itself before acting on
stored heartbeats.
"""

from datetime import datetime, timezone
from typing import Optional

_observation_started_at: Optional[datetime] = None


def observation_started_at() -> Optional[datetime]:
    """When this process started observing the cluster, or None if it doesn't."""
    return _observation_started_at


def observation_elapsed_seconds() -> Optional[float]:
    """Seconds since this process started observing, or None if it isn't."""
    if _observation_started_at is None:
        return None
    return (datetime.now(timezone.utc) - _observation_started_at).total_seconds()


def mark_observation_start(at: Optional[datetime] = None) -> None:
    """Start a new observation window, defaulting to now.

    Called when the process starts, and again when it gains leadership so a
    standby promoted after a long outage gets the same grace period as a fresh
    process before stored heartbeats are treated as silence.
    """
    global _observation_started_at
    _observation_started_at = at or datetime.now(timezone.utc)
