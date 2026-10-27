"""Shared timezone for the usage views.

``model_usages`` is bucketed into daily rows in this timezone at ingest
(see ``metrics_collector``); the ``metered_usage`` read API reuses it so the
GPU-instance / storage breakdowns, Last Active and resource-event times all
line up in one operator-chosen calendar instead of a mix of UTC and local.

Resolution (``resolve_rollup_tz``):
  1. ``GPUSTACK_TIMEZONE`` (IANA name, e.g. ``Asia/Shanghai``; the legacy
     ``GPUSTACK_USAGE_ROLLUP_TIMEZONE`` is still honored as a deprecated alias)
  2. the OS local timezone (``TZ`` / ``/etc/localtime``)
  3. UTC as a last resort.

Read at call time (not cached) so it stays test-overridable.
"""

import logging
from datetime import datetime, timedelta, timezone, tzinfo
from typing import Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from gpustack import envs

logger = logging.getLogger(__name__)

# Emit the deprecation warning at most once, and only when the timezone is
# actually resolved — envs is imported before logging is configured, so the
# warning is deferred here rather than logged at import time.
_legacy_tz_warned = False


def resolve_rollup_tz() -> tzinfo:
    global _legacy_tz_warned
    using_legacy = getattr(envs, "USING_DEPRECATED_TIMEZONE", False)
    if using_legacy and not _legacy_tz_warned:
        logger.warning(
            "GPUSTACK_USAGE_ROLLUP_TIMEZONE is deprecated; use GPUSTACK_TIMEZONE "
            "instead. Honoring the legacy value for now."
        )
        _legacy_tz_warned = True

    tz_name = envs.TIMEZONE
    if tz_name:
        try:
            return ZoneInfo(tz_name)
        except (ZoneInfoNotFoundError, ValueError):
            # Name the variable the value actually came from, so the message
            # isn't misleading when it was set via the legacy alias.
            var = (
                "GPUSTACK_USAGE_ROLLUP_TIMEZONE"
                if using_legacy
                else "GPUSTACK_TIMEZONE"
            )
            logger.warning("Invalid %s=%r, falling back to OS local tz", var, tz_name)
    return datetime.now(timezone.utc).astimezone().tzinfo or timezone.utc


def rollup_offset_minutes(now: Optional[datetime] = None) -> int:
    """The rollup timezone's current UTC offset in whole minutes (e.g. 480 for
    +08:00, -300 for -05:00).

    Used to shift UTC ``bucket_start`` into the rollup timezone for SQL date
    bucketing with a plain interval add — portable across PostgreSQL / MySQL /
    SQLite with no per-dialect timezone tables. It's a *fixed* offset, so it's
    DST-naive (exact for non-DST zones like UTC / Asia/Shanghai; off by up to an
    hour near a DST transition otherwise). Point-in-time conversions
    (``to_rollup_aware``) use the real tz instead and stay DST-correct.
    """
    ref = now or datetime.now(timezone.utc)
    offset = ref.astimezone(resolve_rollup_tz()).utcoffset()
    return int(offset.total_seconds() // 60) if offset else 0


def rollup_fixed_tz(now: Optional[datetime] = None) -> timezone:
    """A fixed-offset ``timezone`` matching :func:`rollup_offset_minutes`.

    Used to *label* the SQL-bucketed timestamps (which were shifted by that same
    fixed offset) with an explicit offset so they serialize self-describing
    (e.g. ``+08:00``) — without a second, possibly DST-divergent, conversion.
    """
    return timezone(timedelta(minutes=rollup_offset_minutes(now)))


def to_rollup_aware(
    value: Optional[datetime], tz: Optional[tzinfo] = None
) -> Optional[datetime]:
    """Convert a (UTC) instant to an *aware* datetime in the rollup timezone —
    DST-correct. Keeps tzinfo so it serializes with an explicit offset (e.g.
    ``+08:00``): the API is self-describing and the UI renders the rollup-tz
    wall clock via ``dayjs.parseZone`` without re-converting to the browser's
    timezone. ``None`` passes through.

    Pass ``tz`` to reuse a timezone resolved once per request (avoids re-reading
    the env / re-parsing ``ZoneInfo`` for every row); defaults to resolving it.
    """
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(tz or resolve_rollup_tz())
