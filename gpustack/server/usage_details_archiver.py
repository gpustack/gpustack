"""Periodic archival of ``model_usage_details`` rows past the retention window.

Hot rows older than ``USAGE_DETAILS_RETENTION_MONTHS`` (anchored on
``COALESCE(completed_at, created_at)``) are moved to
``model_usage_details_archive`` so the hot table stays bounded for fast
reconciliation queries while the audit trail is preserved indefinitely.

Runs once on server startup and then on the cron schedule defined by
``USAGE_DETAILS_ARCHIVE_CRON`` (UTC). Leader-only — both archive and hot
tables would race if multiple replicas ran the sweep concurrently.
"""

import asyncio
import calendar
import logging
from datetime import datetime, timezone

from apscheduler.triggers.cron import CronTrigger
from sqlalchemy import and_, delete, insert, or_, select

from gpustack import envs
from gpustack.schemas.model_usage_details import (
    ModelUsageDetails,
    ModelUsageDetailsArchive,
)
from gpustack.server.db import async_session

logger = logging.getLogger(__name__)


def _months_ago(dt: datetime, months: int) -> datetime:
    """Subtract ``months`` calendar months from ``dt``, clamping the day.

    Uses calendar arithmetic (not ``timedelta(days=30 * months)``) so the
    boundary is precise — ``13 months ago`` from May 7 is April 7, not
    a 5-day-off approximation.
    """
    target_year = dt.year
    target_month = dt.month - months
    while target_month <= 0:
        target_year -= 1
        target_month += 12
    last_day = calendar.monthrange(target_year, target_month)[1]
    target_day = min(dt.day, last_day)
    return dt.replace(year=target_year, month=target_month, day=target_day)


def _assert_archive_shape_aligned() -> None:
    """Bulk archival relies on hot and archive tables having identical column
    lists so ``INSERT ... SELECT`` lines up positionally. If they ever drift,
    fail loudly at server startup instead of silently dropping a column on
    the next sweep.
    """
    hot_cols = {c.name for c in ModelUsageDetails.__table__.columns}
    archive_cols = {c.name for c in ModelUsageDetailsArchive.__table__.columns}
    if hot_cols != archive_cols:
        only_hot = hot_cols - archive_cols
        only_archive = archive_cols - hot_cols
        raise RuntimeError(
            "model_usage_details ↔ model_usage_details_archive column "
            "mismatch — bulk archival requires identical column lists. "
            f"Only on hot: {sorted(only_hot)}; "
            f"only on archive: {sorted(only_archive)}."
        )


class UsageDetailsArchiver:
    """Leader-only loop that archives expired ``model_usage_details`` rows."""

    def __init__(self) -> None:
        self._retention_months = envs.USAGE_DETAILS_RETENTION_MONTHS
        self._batch_size = envs.USAGE_DETAILS_ARCHIVE_BATCH_SIZE
        # Validate the cron expression eagerly — a bad expression should
        # surface at startup, not silently degrade into an idle loop.
        # ``timezone.utc`` makes the schedule predictable across deployments
        # regardless of container TZ.
        try:
            self._trigger = CronTrigger.from_crontab(
                envs.USAGE_DETAILS_ARCHIVE_CRON, timezone=timezone.utc
            )
        except Exception as e:
            raise ValueError(
                "Invalid GPUSTACK_USAGE_DETAILS_ARCHIVE_CRON "
                f"(value={envs.USAGE_DETAILS_ARCHIVE_CRON!r}): {e}"
            ) from e
        # Surface schema drift at construction; the bulk SQL path below
        # would otherwise silently drop columns missing from the archive.
        _assert_archive_shape_aligned()
        # Cache the canonical column list once — used for INSERT ... SELECT.
        self._mirror_columns = [
            c.name for c in ModelUsageDetailsArchive.__table__.columns
        ]

    async def start(self) -> None:
        # Initial run: catches up on rows that aged out while no leader was
        # running (or while the previous leader was between cycles).
        try:
            await self.archive_once()
        except Exception as e:
            logger.error(f"Initial usage details archival failed: {e}")

        while True:
            sleep_seconds = self._seconds_until_next_fire()
            if sleep_seconds is None:
                # No future fire time — schedule is malformed in a way
                # CronTrigger accepted but can't satisfy. Bail out instead
                # of busy-looping.
                logger.error(
                    "Cron %r yielded no future fire time; archiver loop stopping.",
                    envs.USAGE_DETAILS_ARCHIVE_CRON,
                )
                return
            await asyncio.sleep(sleep_seconds)
            try:
                await self.archive_once()
            except Exception as e:
                logger.error(f"Usage details archival failed: {e}")

    def _seconds_until_next_fire(self) -> float | None:
        now = datetime.now(timezone.utc)
        next_fire = self._trigger.get_next_fire_time(None, now)
        if next_fire is None:
            return None
        # APScheduler returns a tz-aware datetime in the trigger's tz; subtract
        # works against ``now`` (also tz-aware UTC) without further coercion.
        return max(0.0, (next_fire - now).total_seconds())

    async def archive_once(self) -> int:
        """Drain rows older than the retention cutoff. Returns total moved."""
        cutoff = _months_ago(
            datetime.now(timezone.utc).replace(tzinfo=None),
            self._retention_months,
        )
        moved_total = 0
        while True:
            moved = await self._archive_batch(cutoff)
            if moved == 0:
                break
            moved_total += moved
        if moved_total > 0:
            logger.info(
                f"Archived {moved_total} model_usage_details rows older than "
                f"{cutoff.isoformat()} (retention={self._retention_months}mo)."
            )
        return moved_total

    async def _archive_batch(self, cutoff: datetime) -> int:
        """Move up to ``batch_size`` rows in a single transaction.

        Bulk SQL path: ``INSERT INTO archive (cols...) SELECT cols... FROM
        hot WHERE id IN (ids)`` keeps the row data inside the DB engine —
        Python only carries the id list. Avoids the per-row ORM hydration
        that would otherwise pull every column over the wire and rebuild
        it as a SQLAlchemy instance just to insert it back.
        """
        # Two-branch predicate (instead of COALESCE) so the planner can use
        # ix_..._completed_at on the modern fast path and fall back to
        # ix_..._created_at only for legacy rows missing completed_at.
        # PG combines them via BitmapOr; MySQL/SQLite may seq-scan but the
        # working set here is bounded by ``batch_size`` so it stays cheap.
        age_predicate = or_(
            and_(
                ModelUsageDetails.completed_at.is_not(None),
                ModelUsageDetails.completed_at < cutoff,
            ),
            and_(
                ModelUsageDetails.completed_at.is_(None),
                ModelUsageDetails.created_at < cutoff,
            ),
        )
        hot_table = ModelUsageDetails.__table__
        archive_table = ModelUsageDetailsArchive.__table__

        async with async_session() as session:
            ids = (
                await session.exec(
                    select(ModelUsageDetails.id)
                    .where(age_predicate)
                    .order_by(ModelUsageDetails.id)
                    .limit(self._batch_size)
                )
            ).all()
            if not ids:
                return 0

            # Defense against re-archival: if a previous sweep somehow left
            # an id in archive (replication quirk, mid-transaction rollback
            # of the delete leg), skip it on insert. INSERT and DELETE
            # share this transaction so the normal path can't hit this.
            existing_archive_ids = set(
                (
                    await session.exec(
                        select(ModelUsageDetailsArchive.id).where(
                            ModelUsageDetailsArchive.id.in_(ids)
                        )
                    )
                ).all()
            )
            ids_to_insert = [i for i in ids if i not in existing_archive_ids]

            if ids_to_insert:
                projection = select(
                    *[hot_table.c[name] for name in self._mirror_columns]
                ).where(hot_table.c.id.in_(ids_to_insert))
                await session.exec(
                    insert(archive_table).from_select(self._mirror_columns, projection)
                )

            await session.exec(
                delete(ModelUsageDetails).where(ModelUsageDetails.id.in_(ids))
            )
            await session.commit()
            return len(ids)
