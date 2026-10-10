from contextlib import asynccontextmanager

import pytest
import pytest_asyncio
from sqlalchemy import event
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.mixins import active_record
from gpustack.schemas.models import (
    Model,
    ScalingSchedule,
    ScalingScheduleRule,
    SourceEnum,
)
from gpustack.server import scaling_scheduler
from gpustack.server.cache import delete_cache_by_key
from gpustack.server.services import ModelService


async def _drop_model_caches():
    await delete_cache_by_key(_key="Model.cached_all")
    for model_id, name in [(1, "batch-alpha"), (2, "batch-beta")]:
        await delete_cache_by_key(ModelService.get_by_id, model_id)
        await delete_cache_by_key(ModelService.get_by_name, name)


@pytest_asyncio.fixture
async def model_engine(tmp_path, monkeypatch):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'models.db'}")
    async with engine.begin() as connection:
        await connection.run_sync(Model.__table__.create)
    schedule = ScalingSchedule(
        enabled=True,
        baseline_replicas=2,
        rules=[
            ScalingScheduleRule(
                start_cron="0 0 * * *", duration_seconds=86400, replicas=2
            )
        ],
    )
    async with AsyncSession(engine) as session:
        session.add_all(
            [
                Model(
                    id=model_id,
                    name=name,
                    source=SourceEnum.HUGGING_FACE,
                    huggingface_repo_id="org/model",
                    replicas=0,
                    scaling_schedule=schedule.model_copy(deep=True),
                )
                for model_id, name in [(1, "batch-alpha"), (2, "batch-beta")]
            ]
        )
        await session.commit()
    monkeypatch.setattr(
        active_record,
        "async_session",
        lambda: AsyncSession(engine, expire_on_commit=False),
    )
    await _drop_model_caches()
    try:
        yield engine
    finally:
        await _drop_model_caches()
        await engine.dispose()


async def _read_cached_replicas(engine):
    async with AsyncSession(engine, expire_on_commit=False) as session:
        service = ModelService(session)
        by_id = [(await service.get_by_id(model_id)).replicas for model_id in (1, 2)]
        by_name = [
            (await service.get_by_name(name)).replicas
            for name in ("batch-alpha", "batch-beta")
        ]
        all_models = [model.replicas for model in await Model.cached_all()]
    return by_id, by_name, all_models


@pytest.mark.asyncio
async def test_batch_update_invalidates_real_id_name_and_all_caches(model_engine):
    assert await _read_cached_replicas(model_engine) == ([0, 0], [0, 0], [0, 0])
    commits = []
    event.listen(
        model_engine.sync_engine, "commit", lambda connection: commits.append(1)
    )

    async with AsyncSession(model_engine, expire_on_commit=False) as session:
        models = (await session.exec(select(Model).order_by(Model.id))).all()
        for model in models:
            model.replicas = 3
        assert await ModelService(session).batch_update(models) == 2

    assert len(commits) == 1
    assert await _read_cached_replicas(model_engine) == ([3, 3], [3, 3], [3, 3])


@pytest.mark.asyncio
async def test_scheduler_evicts_real_caches_refilled_before_batch_commit(
    model_engine,
    monkeypatch,
):
    observed = []

    class RefillingSession(AsyncSession):
        @asynccontextmanager
        async def begin(self):
            async with super().begin():
                yield
                # Independent readers see the committed row until the write commits.
                observed.append(await _read_cached_replicas(model_engine))

    monkeypatch.setattr(
        scaling_scheduler,
        "async_session",
        lambda: RefillingSession(model_engine, expire_on_commit=False),
    )
    assert await _read_cached_replicas(model_engine) == ([0, 0], [0, 0], [0, 0])
    await scaling_scheduler.ScalingScheduler()._sync_scheduled_replicas()

    assert observed == [([0, 0], [0, 0], [0, 0])]
    assert await _read_cached_replicas(model_engine) == ([2, 2], [2, 2], [2, 2])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change, expected",
    [("pause", 0), ("manual", 7), ("disable", 0), ("rules", 5), ("delete", 0)],
)
async def test_scheduler_reads_edits_committed_after_snapshot(
    model_engine, monkeypatch, change, expected
):
    sessions = 0

    @asynccontextmanager
    async def session():
        nonlocal sessions
        sessions += 1
        if sessions == 2:
            async with AsyncSession(model_engine, expire_on_commit=False) as editor:
                model = await Model.one_by_id(editor, 1)
                schedule = model.scaling_schedule.model_copy(deep=True)
                if change in ("pause", "manual"):
                    schedule.paused = True
                    if change == "manual":
                        model.replicas = 7
                elif change == "disable":
                    schedule.enabled = False
                elif change == "rules":
                    schedule.baseline_replicas = 5
                    schedule.rules[0].replicas = 5
                elif change == "delete":
                    from datetime import datetime, timezone

                    model.deleted_at = datetime.now(timezone.utc)
                model.scaling_schedule = schedule
                await editor.commit()
        async with AsyncSession(model_engine, expire_on_commit=False) as current:
            yield current

    monkeypatch.setattr(scaling_scheduler, "async_session", session)
    await scaling_scheduler.ScalingScheduler()._sync_scheduled_replicas()

    async with AsyncSession(model_engine) as reader:
        models = (await reader.exec(select(Model).order_by(Model.id))).all()
        assert [model.replicas for model in models] == [expected, 2]


@pytest.mark.asyncio
async def test_failed_batch_rolls_back_all_rows_and_can_retry(
    model_engine, monkeypatch
):
    async with AsyncSession(model_engine, expire_on_commit=False) as session:
        models = (await session.exec(select(Model).order_by(Model.id))).all()
        for model in models:
            model.replicas = 4
        original_commit = session.commit

        async def fail_commit():
            await session.flush()
            raise RuntimeError("commit failed")

        with monkeypatch.context() as patch:
            patch.setattr(session, "commit", fail_commit)
            with pytest.raises(RuntimeError, match="commit failed"):
                await ModelService(session).batch_update(models)

        async with AsyncSession(model_engine) as reader:
            rows = (await reader.exec(select(Model).order_by(Model.id))).all()
            assert [row.replicas for row in rows] == [0, 0]

        models = (await session.exec(select(Model).order_by(Model.id))).all()
        for model in models:
            model.replicas = 4
        assert session.commit == original_commit
        await ModelService(session).batch_update(models)
    assert await _read_cached_replicas(model_engine) == ([4, 4], [4, 4], [4, 4])
