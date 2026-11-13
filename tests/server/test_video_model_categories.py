from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, Mock

import pytest

from gpustack.server import video_model_categories as module
from tests.utils.model import new_model


@pytest.fixture
def setup_reconciler(monkeypatch):
    @asynccontextmanager
    async def session():
        yield object()

    monkeypatch.setattr(module, "async_session", session)
    monkeypatch.setattr(module.Worker, "all", AsyncMock(return_value=[]))
    update = AsyncMock()
    monkeypatch.setattr(module, "ModelService", lambda s: Mock(update=update))
    return module.VideoCategoryReconciler(), update


@pytest.mark.asyncio
async def test_existing_video_is_reclassified_without_restarting_instances(
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
    update.assert_awaited_once_with(video, source={"categories": ["video"]})
    assert read.await_count == 2
    await reconciler.reconcile()
    assert read.await_count == 2


@pytest.mark.asyncio
async def test_unavailable_worker_metadata_retries_and_preserves_category(
    monkeypatch, setup_reconciler
):
    reconciler, update = setup_reconciler
    model = new_model(
        1,
        "old",
        huggingface_repo_id="org/custom",
        backend="SGLang",
        categories=["image"],
    )
    monkeypatch.setattr(module.Model, "all_by_fields", AsyncMock(return_value=[model]))
    monkeypatch.setattr(module.Model, "one_by_id", AsyncMock(return_value=model))
    read = AsyncMock(
        side_effect=[None, {"_diffusers_version": "1", "_class_name": "WanPipeline"}]
    )
    monkeypatch.setattr(module, "get_diffusers_model_index_with_workers", read)
    await reconciler.reconcile()
    update.assert_not_awaited()
    assert model.categories == ["image"]
    await reconciler.reconcile()
    update.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["category", "source", "cluster", "deleted"])
async def test_reclassification_does_not_overwrite_concurrent_changes(
    monkeypatch, setup_reconciler, change
):
    reconciler, update = setup_reconciler
    original = new_model(
        1, "old", backend="SGLang", categories=["image"], huggingface_repo_id="org/old"
    )
    current = new_model(
        1,
        "old",
        backend="SGLang",
        categories=["llm"] if change == "category" else ["image"],
        huggingface_repo_id="org/new" if change == "source" else "org/old",
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
