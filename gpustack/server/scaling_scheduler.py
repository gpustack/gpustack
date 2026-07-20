import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import sqlalchemy as sa
from croniter import croniter
from sqlmodel import col

from gpustack import envs
from gpustack.schemas.models import Model, ScalingSchedule, is_dp_node_per_instance
from gpustack.server.db import async_session
from gpustack.server.services import ModelService
from gpustack.utils.rollup_tz import resolve_rollup_tz

logger = logging.getLogger(__name__)


def compute_desired_replicas(
    schedule: ScalingSchedule, now: Optional[datetime] = None
) -> Optional[int]:
    """
    Given a scaling schedule, return the effective desired replica count for
    ``now`` (GCP scaling-schedule / KEDA Cron scaler semantics). A rule's window
    opens when ``start_cron`` fires and stays open for ``duration_seconds``;
    while ``now`` is inside it the model is driven to that rule's ``replicas``.
    When windows overlap, the one that started most recently wins; windows that
    started at the same instant resolve to the largest replica count, so the
    result never depends on rule order. Outside every window the count falls
    back to ``baseline_replicas``.

    Cron windows are evaluated in the platform-wide business timezone
    (``GPUSTACK_TIMEZONE``, shared with usage rollups) — see
    ``resolve_rollup_tz``. There is no per-model timezone.

    Known limitation: when a cron fires during a DST fall-back repeat (the same
    wall-clock hour occurring twice), croniter resolves the earlier occurrence
    and the window can read as closed for part of that hour.

    Returns None when the schedule is disabled, has no usable rules, or is
    outside every window with no baseline configured (nothing to enforce).
    """
    if not schedule or not schedule.enabled or not schedule.rules:
        return None

    tz = resolve_rollup_tz()

    if now is None:
        now = datetime.now(tz)
    elif now.tzinfo is None:
        # Naive input: assume it's already wall-clock in the business tz.
        now = now.replace(tzinfo=tz)
    else:
        # tz-aware input (e.g. a UTC caller): re-express in the business tz so
        # the cron is always evaluated against the platform calendar, not the
        # caller's zone.
        now = now.astimezone(tz)

    best_start: Optional[datetime] = None
    active_replicas: Optional[int] = None
    for rule in schedule.rules:
        if not rule.start_cron or not rule.duration_seconds:
            continue
        try:
            prev_start = croniter(rule.start_cron, now).get_prev(datetime)
        except Exception as e:
            logger.warning(f"Failed to evaluate cron {rule.start_cron!r}: {e}")
            continue
        # Add the duration in absolute (UTC) time, not wall-clock: adding a
        # timedelta to a zoned datetime keeps the original offset, so across a
        # DST transition the window would otherwise stretch or shrink by an hour.
        window_end = prev_start.astimezone(timezone.utc) + timedelta(
            seconds=rule.duration_seconds
        )
        # Inside the window when the most recent start already fired and did so
        # less than ``duration`` ago. The lower bound matters across a DST
        # spring-forward: for a cron whose time falls in the skipped hour,
        # croniter resolves a start *after* ``now``, which would open the window
        # early. Among overlapping active windows keep the most recent start, and
        # on a tie take the larger count so rule order can't change the outcome.
        if not prev_start <= now < window_end:
            continue
        if (
            best_start is None
            or prev_start > best_start
            or (prev_start == best_start and rule.replicas > active_replicas)
        ):
            best_start = prev_start
            active_replicas = rule.replicas

    if active_replicas is not None:
        return active_replicas
    return schedule.baseline_replicas


class ScalingScheduler:
    """
    Leader-only periodic task that drives model ``replicas`` on a cron
    timetable. It only sets the desired replica count; the existing model
    reconcile loop (`sync_replicas`) performs the actual scale up/down.
    """

    def __init__(self, interval: Optional[int] = None):
        self._interval = (
            interval if interval is not None else envs.SCALING_SCHEDULER_INTERVAL
        )

    async def start(self):
        while True:
            # Reconcile first, then sleep: a freshly elected leader corrects
            # replica counts immediately instead of after a full interval.
            try:
                await self._sync_scheduled_replicas()
            except Exception as e:
                logger.error(f"Failed to sync scheduled replicas: {e}")
            await asyncio.sleep(self._interval)

    async def _sync_scheduled_replicas(self):
        # Narrow to rows that actually carry a schedule, so a tick doesn't load
        # and deserialize every model. A model without one serializes to the
        # JSON literal 'null' rather than SQL NULL, so `IS NOT NULL` would match
        # everything; compare the text form instead, which also excludes rows
        # predating the column (true SQL NULL). The `enabled` flag lives inside
        # the JSON document and is checked in Python below.
        async with async_session() as session:
            models = await Model.all_by_fields(
                session,
                extra_conditions=[
                    col(Model.deleted_at).is_(None),
                    sa.cast(col(Model.scaling_schedule), sa.Text) != "null",
                ],
            )

        # Resolve "now" once so every model in this tick is evaluated against
        # the same instant (avoids cross-minute drift within a batch).
        now = datetime.now(resolve_rollup_tz())
        to_update: list[tuple[int, int]] = []
        for model in models:
            schedule = model.scaling_schedule
            if not schedule or not schedule.enabled:
                continue
            if is_dp_node_per_instance(model):
                # replicas is the DP node count fixed by --data-parallel-size;
                # driving it off a timetable would leave the surviving nodes
                # waiting on ranks that no longer exist.
                continue
            desired = compute_desired_replicas(schedule, now)
            if desired is None or desired == model.replicas:
                continue
            to_update.append((model.id, desired))

        if not to_update:
            return

        async with async_session() as session:
            service = ModelService(session)
            for model_id, desired in to_update:
                # Reload under this session and re-check to avoid clobbering a
                # concurrent manual change with a stale desired value.
                model = await Model.one_by_id(session, model_id)
                if model is None or model.deleted_at is not None:
                    continue
                if model.replicas == desired:
                    continue
                previous = model.replicas
                model.replicas = desired
                await service.update(model)
                logger.info(
                    f"Scheduled scaling: model {model.name} replicas "
                    f"{previous} -> {desired}"
                )
