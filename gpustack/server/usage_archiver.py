"""Generic hot→cold archiver for the usage tables.

Moves rows older than the retention window from a hot table to its
column-identical ``_archive`` cold table, so the hot table stays bounded while
the data is preserved for audit / metering reconciliation. Used for
``metered_usage`` and ``resource_events``.

Leader-only — both tables would race if multiple replicas swept concurrently.
Runs once on startup and then on the configured cron (UTC). The cold tables are
NOT exposed via any API; audit reads them directly via the database.

> Retention must be far larger than the collector's settlement horizon (hours).
> ``metered_usage`` is upserted into the current/recent bucket; with a 13-month
> retention no still-being-written bucket is ever archived.
"""

import asyncio
import calendar
import logging
from datetime import datetime, timezone

from apscheduler.triggers.cron import CronTrigger
from sqlalchemy import delete, insert, select

from gpustack.server.db import async_session

logger = logging.getLogger(__name__)


def _months_ago(dt: datetime, months: int) -> datetime:
    """Subtract ``months`` calendar months from ``dt``, clamping the day."""
    target_year = dt.year
    target_month = dt.month - months
    while target_month <= 0:
        target_year -= 1
        target_month += 12
    last_day = calendar.monthrange(target_year, target_month)[1]
    target_day = min(dt.day, last_day)
    return dt.replace(year=target_year, month=target_month, day=target_day)


class TableArchiver:
    """Leader-only loop that archives expired rows from one hot table."""

    def __init__(
        self,
        hot_model,
        archive_model,
        *,
        anchor_col: str,
        retention_months: int,
        cron: str,
        batch_size: int,
        label: str,
    ) -> None:
        self._hot = hot_model
        self._archive = archive_model
        self._anchor = getattr(hot_model, anchor_col)
        self._retention_months = retention_months
        self._batch_size = batch_size
        self._label = label
        try:
            self._trigger = CronTrigger.from_crontab(cron, timezone=timezone.utc)
        except Exception as e:
            raise ValueError(
                f"Invalid archive cron for {label} (value={cron!r}): {e}"
            ) from e
        self._assert_shape_aligned()
        self._mirror_columns = [c.name for c in archive_model.__table__.columns]

    def _assert_shape_aligned(self) -> None:
        hot_cols = {c.name for c in self._hot.__table__.columns}
        archive_cols = {c.name for c in self._archive.__table__.columns}
        if hot_cols != archive_cols:
            only_hot = hot_cols - archive_cols
            only_archive = archive_cols - hot_cols
            raise RuntimeError(
                f"{self._label}: hot ↔ archive column mismatch — bulk archival "
                f"requires identical column lists. Only on hot: {sorted(only_hot)}; "
                f"only on archive: {sorted(only_archive)}."
            )

    async def start(self) -> None:
        try:
            await self.archive_once()
        except Exception as e:
            logger.error(f"Initial {self._label} archival failed: {e}")

        while True:
            sleep_seconds = self._seconds_until_next_fire()
            if sleep_seconds is None:
                logger.error(
                    "%s: cron yielded no future fire time; archiver loop stopping.",
                    self._label,
                )
                return
            await asyncio.sleep(sleep_seconds)
            try:
                await self.archive_once()
            except Exception as e:
                logger.error(f"{self._label} archival failed: {e}")

    def _seconds_until_next_fire(self) -> float | None:
        now = datetime.now(timezone.utc)
        next_fire = self._trigger.get_next_fire_time(None, now)
        if next_fire is None:
            return None
        return max(0.0, (next_fire - now).total_seconds())

    async def archive_once(self) -> int:
        cutoff = _months_ago(
            datetime.now(timezone.utc).replace(tzinfo=None), self._retention_months
        )
        moved_total = 0
        while True:
            moved = await self._archive_batch(cutoff)
            if moved == 0:
                break
            moved_total += moved
        if moved_total > 0:
            logger.info(
                f"Archived {moved_total} {self._label} rows older than "
                f"{cutoff.isoformat()} (retention={self._retention_months}mo)."
            )
        return moved_total

    async def _archive_batch(self, cutoff: datetime) -> int:
        hot_table = self._hot.__table__
        archive_table = self._archive.__table__

        async with async_session() as session:
            id_rows = (
                await session.exec(
                    select(self._hot.id)
                    .where(self._anchor < cutoff)
                    .order_by(self._hot.id)
                    .limit(self._batch_size)
                )
            ).all()
            # ``session.exec(select(col))`` may yield Row tuples or scalars
            # depending on driver — normalize to plain ints.
            ids = [r if isinstance(r, int) else r[0] for r in id_rows]
            if not ids:
                return 0

            # Defense against re-archival (replication quirk / rollback).
            existing_rows = (
                await session.exec(
                    select(self._archive.id).where(self._archive.id.in_(ids))
                )
            ).all()
            existing_archive_ids = {
                r if isinstance(r, int) else r[0] for r in existing_rows
            }
            ids_to_insert = [i for i in ids if i not in existing_archive_ids]

            if ids_to_insert:
                projection = select(
                    *[hot_table.c[name] for name in self._mirror_columns]
                ).where(hot_table.c.id.in_(ids_to_insert))
                await session.exec(
                    insert(archive_table).from_select(self._mirror_columns, projection)
                )

            await session.exec(delete(self._hot).where(self._hot.id.in_(ids)))
            await session.commit()
            return len(ids)
