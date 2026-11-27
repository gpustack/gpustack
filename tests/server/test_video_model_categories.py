from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from cachetools import TTLCache
from sqlalchemy.dialects import mysql, postgresql

from gpustack.server import video_model_categories as module
from tests.utils.model import new_model


@pytest.fixture
def setup_reconciler(monkeypatch):
    @asynccontextmanager
    async def session():
        yield SimpleNamespace(bind=SimpleNamespace(dialect=postgresql.dialect()))

    monkeypatch.setattr(module, "async_session", session)
    monkeypatch.setattr(module.Worker, "all_by_fields", AsyncMock(return_value=[]))
    update = AsyncMock()
    monkeypatch.setattr(module, "ModelService", lambda s: Mock(update=update))
    return module.VideoCategoryReconciler(), update


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "dialect, predicate",
    [(postgresql.asyncpg.dialect(), " ? "), (mysql.dialect(), "json_contains(")],
)
async def test_reconciler_queries_only_video_candidates(
    monkeypatch, setup_reconciler, dialect, predicate
):
    @asynccontextmanager
    async def session():
        yield SimpleNamespace(bind=SimpleNamespace(dialect=dialect))

    monkeypatch.setattr(module, "async_session", session)
    models = AsyncMock(return_value=[])
    monkeypatch.setattr(module.Model, "all_by_fields", models)
    reconciler, _ = setup_reconciler

    await reconciler.reconcile()

    query = models.await_args.kwargs
    assert query["fields"] == {"deleted_at": None}
    (condition,) = query["extra_conditions"]
    compiled = condition.compile(dialect=dialect)
    sql = str(compiled)
    assert "models.categories" in sql
    assert predicate in sql
    if dialect.name == "postgresql":
        assert "::VARCHAR" in sql
        assert "? $1::JSONB" not in sql
    assert module.CategoryEnum.VIDEO in compiled.params.values()
    module.Worker.all_by_fields.assert_not_awaited()


@pytest.mark.asyncio
async def test_explicit_image_categories_are_not_inspected_or_reclassified(
    monkeypatch, setup_reconciler
):
    reconciler, update = setup_reconciler
    video = new_model(
        1,
        "renamed",
        huggingface_repo_id="org/custom",
        backend="SGLang",
        categories=["image"],
    )
    image = new_model(
        2,
        "image",
        huggingface_repo_id="org/custom",
        backend="vLLM",
        categories=["image"],
    )
    declared = new_model(
        3,
        "explicit",
        huggingface_repo_id="org/custom",
        backend="SGLang",
        categories=["video"],
    )
    vision = new_model(
        4,
        "vision",
        huggingface_repo_id="org/custom",
        backend="SGLang",
        categories=["llm"],
    )
    monkeypatch.setattr(
        module.Model,
        "all_by_fields",
        AsyncMock(return_value=[video, image, declared, vision]),
    )
    monkeypatch.setattr(module.Model, "one_by_id", AsyncMock(return_value=video))
    read = AsyncMock(
        side_effect=lambda model, workers: {
            "_diffusers_version": "1",
            "_class_name": "WanPipeline" if model.id == 1 else "FluxPipeline",
        }
    )
    monkeypatch.setattr(module, "get_diffusers_model_index_with_workers", read)
    await reconciler.reconcile()
    update.assert_not_awaited()
    assert video.categories == ["image"]
    read.assert_awaited_once_with(declared, [])
    await reconciler.reconcile()
    assert read.await_count == 1


@pytest.mark.asyncio
async def test_running_h3_receives_metadata_without_category_change(
    monkeypatch, setup_reconciler
):
    reconciler, update = setup_reconciler
    model = new_model(
        1,
        "renamed",
        huggingface_repo_id="MiniMaxAI/MiniMax-H3",
        backend="SGLang",
        categories=["video"],
    )
    monkeypatch.setattr(module.Model, "all_by_fields", AsyncMock(return_value=[model]))
    monkeypatch.setattr(module.Model, "one_by_id", AsyncMock(return_value=model))
    read = AsyncMock(side_effect=AssertionError("Known source needs no Hub request"))
    monkeypatch.setattr(module, "get_diffusers_model_index_with_workers", read)
    await reconciler.reconcile()
    update.assert_awaited_once_with(model, source={"meta": model.meta})
    assert model.meta["video"]["default_task"] == "t2va"
    read.assert_not_awaited()


@pytest.mark.asyncio
async def test_unavailable_worker_metadata_retries_and_preserves_category(
    monkeypatch, setup_reconciler
):
    reconciler, update = setup_reconciler
    clock = [0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    model = new_model(
        1,
        "old",
        huggingface_repo_id="org/custom",
        backend="SGLang",
        categories=["video"],
    )
    monkeypatch.setattr(module.Model, "all_by_fields", AsyncMock(return_value=[model]))
    monkeypatch.setattr(module.Model, "one_by_id", AsyncMock(return_value=model))
    read = AsyncMock(
        side_effect=[
            TimeoutError("metadata unavailable"),
            {"_diffusers_version": "1", "_class_name": "WanPipeline"},
        ]
    )
    monkeypatch.setattr(module, "get_diffusers_model_index_with_workers", read)
    await reconciler.reconcile()
    update.assert_not_awaited()
    assert model.categories == ["video"]
    await reconciler.reconcile()
    assert read.await_count == 1
    clock[0] = 180
    await reconciler.reconcile()
    assert read.await_count == 2
    update.assert_not_awaited()


@pytest.mark.asyncio
async def test_persistent_failures_back_off_without_caching_absence(
    monkeypatch, setup_reconciler
):
    reconciler, update = setup_reconciler
    clock = [0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    model = new_model(
        1,
        "offline",
        backend="SGLang",
        categories=["video"],
        huggingface_repo_id="org/offline",
    )
    monkeypatch.setattr(module.Model, "all_by_fields", AsyncMock(return_value=[model]))
    read = AsyncMock(side_effect=TimeoutError("worker unavailable"))
    monkeypatch.setattr(module, "get_diffusers_model_index_with_workers", read)
    for now, calls in ((0, 1), (179, 1), (180, 2), (539, 2), (540, 3)):
        clock[0] = now
        await reconciler.reconcile()
        assert read.await_count == calls
    assert not reconciler._checked
    update.assert_not_awaited()


@pytest.mark.asyncio
async def test_worker_inspection_excludes_deleted_and_non_ready_workers(
    monkeypatch, setup_reconciler
):
    reconciler, _ = setup_reconciler
    model = new_model(
        1,
        "local",
        backend="SGLang",
        categories=["video"],
        huggingface_repo_id="org/local",
    )
    monkeypatch.setattr(module.Model, "all_by_fields", AsyncMock(return_value=[model]))
    worker = Mock()
    workers = AsyncMock(return_value=[worker])
    monkeypatch.setattr(module.Worker, "all_by_fields", workers)
    inspect = AsyncMock(return_value=None)
    monkeypatch.setattr(module, "get_diffusers_model_index_with_workers", inspect)
    await reconciler.reconcile()
    assert workers.call_args.kwargs["fields"] == {"deleted_at": None}
    condition = workers.call_args.kwargs["extra_conditions"][0]
    assert condition.right.value == module.WorkerStateEnum.READY
    inspect.assert_awaited_once_with(model, [worker])


@pytest.mark.asyncio
async def test_confirmed_missing_metadata_is_cached_until_the_next_ttl(
    monkeypatch, setup_reconciler
):
    reconciler, update = setup_reconciler
    clock = [0]
    reconciler._checked = TTLCache(maxsize=1024, ttl=3600, timer=lambda: clock[0])
    model = new_model(
        1,
        "image",
        backend="vLLM",
        categories=["video"],
        huggingface_repo_id="org/image",
    )
    monkeypatch.setattr(module.Model, "all_by_fields", AsyncMock(return_value=[model]))
    read = AsyncMock(return_value=None)
    monkeypatch.setattr(module, "get_diffusers_model_index_with_workers", read)

    await reconciler.reconcile()
    await reconciler.reconcile()
    assert read.await_count == 1
    update.assert_not_awaited()

    clock[0] = 3601
    await reconciler.reconcile()
    assert read.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["category", "source", "cluster", "deleted"])
async def test_metadata_refresh_does_not_overwrite_concurrent_changes(
    monkeypatch, setup_reconciler, change
):
    reconciler, update = setup_reconciler
    original = new_model(
        1,
        "old",
        backend="SGLang",
        categories=["video"],
        huggingface_repo_id="MiniMaxAI/MiniMax-H3",
    )
    current = new_model(
        1,
        "old",
        backend="SGLang",
        categories=["image"] if change == "category" else ["video"],
        huggingface_repo_id=(
            "org/new" if change == "source" else "MiniMaxAI/MiniMax-H3"
        ),
    )
    if change == "cluster":
        current.cluster_id = 7
    if change == "deleted":
        from datetime import datetime, timezone

        current.deleted_at = datetime.now(timezone.utc)
    monkeypatch.setattr(
        module.Model, "all_by_fields", AsyncMock(return_value=[original])
    )
    monkeypatch.setattr(module.Model, "one_by_id", AsyncMock(return_value=current))
    monkeypatch.setattr(
        module,
        "get_diffusers_model_index_with_workers",
        AsyncMock(
            return_value={"_diffusers_version": "1", "_class_name": "WanPipeline"}
        ),
    )
    await reconciler.reconcile()
    update.assert_not_awaited()


@pytest.mark.asyncio
async def test_video_reclassification_preserves_the_instance_generation(monkeypatch):
    from types import SimpleNamespace
    from gpustack.server import controllers

    monkeypatch.setattr(
        controllers, "_instance_type_snapshots", AsyncMock(return_value={})
    )
    model = new_model(1, "old", huggingface_repo_id="org/custom", categories=["image"])
    before = await controllers.model_spec_digest(SimpleNamespace(info={}), model)
    model.categories = ["video"]
    after = await controllers.model_spec_digest(SimpleNamespace(info={}), model)
    assert before == after
