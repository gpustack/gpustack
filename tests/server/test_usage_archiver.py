"""Test the generic TableArchiver: rows older than retention move to the
column-identical archive table; recent rows stay."""

from contextlib import asynccontextmanager
from datetime import datetime
from unittest.mock import patch

import pytest
import pytest_asyncio
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel.ext.asyncio.session import AsyncSession

import gpustack.server.usage_archiver as ua
from gpustack.schemas.metered_usage import (
    METER_INSTANCE_UPTIME,
    RESOURCE_TYPE_GPU_INSTANCE,
    MeteredUsage,
    MeteredUsageArchive,
)
from gpustack.server.usage_archiver import TableArchiver

NOW = datetime(2026, 5, 26, 12, 0, 0)


def _row(resource_id, bucket_start):
    return MeteredUsage(
        owner_principal_id=1,
        creator_id=7,
        meter_key=METER_INSTANCE_UPTIME,
        resource_type=RESOURCE_TYPE_GPU_INSTANCE,
        resource_id=resource_id,
        resource_name=f"g{resource_id}",
        sku="nvidia-h100",
        sku_count=2,
        bucket_start=bucket_start,
        quantity=3600,
        unit="seconds",
        created_at=NOW,
        updated_at=NOW,
    )


@asynccontextmanager
async def _yield(session):
    yield session


@pytest_asyncio.fixture
async def session():
    engine = create_async_engine("sqlite+aiosqlite://")
    async with engine.begin() as conn:
        await conn.run_sync(MeteredUsage.__table__.create)
        await conn.run_sync(MeteredUsageArchive.__table__.create)
    async with AsyncSession(engine) as s:
        s.add_all(
            [
                _row(1, datetime(2020, 1, 1, 0, 0, 0)),  # ancient → archive
                _row(2, datetime(2026, 5, 26, 10, 0, 0)),  # recent → stays hot
            ]
        )
        await s.commit()
        yield s
    await engine.dispose()


@pytest.mark.asyncio
async def test_archiver_moves_old_rows(session):
    archiver = TableArchiver(
        MeteredUsage,
        MeteredUsageArchive,
        anchor_col="bucket_start",
        retention_months=13,
        cron="0 3 * * *",
        batch_size=100,
        label="metered_usage",
    )
    with patch.object(ua, "async_session", lambda: _yield(session)):
        moved = await archiver.archive_once()

    assert moved == 1
    hot = (await session.exec(select(func.count()).select_from(MeteredUsage))).first()[
        0
    ]
    arc = (
        await session.exec(select(func.count()).select_from(MeteredUsageArchive))
    ).first()[0]
    hot_ids = (await session.exec(select(MeteredUsage.resource_id))).all()
    arc_ids = (await session.exec(select(MeteredUsageArchive.resource_id))).all()
    assert hot == 1 and [r[0] for r in hot_ids] == [2]
    assert arc == 1 and [r[0] for r in arc_ids] == [1]


@pytest.mark.asyncio
async def test_archiver_shape_alignment_guard():
    # hot vs archive column lists are identical → no error at construction
    TableArchiver(
        MeteredUsage,
        MeteredUsageArchive,
        anchor_col="bucket_start",
        retention_months=13,
        cron="0 3 * * *",
        batch_size=100,
        label="metered_usage",
    )
