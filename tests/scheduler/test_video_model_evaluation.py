from unittest.mock import AsyncMock
from types import SimpleNamespace

import pytest

from gpustack.scheduler import calculator, scheduler
from gpustack.schemas.models import CategoryEnum, SourceEnum, Model
from tests.utils.model import new_model


@pytest.mark.asyncio
@pytest.mark.parametrize("categories", [[], ["image"]])
@pytest.mark.parametrize("shape", ["dict", "list"])
async def test_video_artifact_reclassifies_legacy_image_model(
    monkeypatch, categories, shape
):
    index = {"_diffusers_version": "0.32", "_class_name": "WanPipeline"}
    monkeypatch.setattr(
        calculator,
        "get_diffusers_model_index",
        lambda *a, **k: index if shape == "dict" else [index],
    )
    model = new_model(
        1, "custom-alias", categories=categories, huggingface_repo_id="org/custom"
    )
    assert await scheduler.evaluate_diffusion_model(model)
    assert model.categories == [CategoryEnum.VIDEO]


@pytest.mark.asyncio
async def test_remote_local_path_video_index_is_preserved(monkeypatch):
    monkeypatch.setattr(calculator, "get_diffusers_model_index", lambda *a, **k: None)
    read = AsyncMock(
        return_value={
            "_diffusers_version": "0.32",
            "_class_name": "WanImageToVideoPipeline",
        }
    )
    monkeypatch.setattr(calculator, "read_local_path_file_from_workers", read)
    model = Model(
        id=1,
        name="local",
        source=SourceEnum.LOCAL_PATH,
        local_path="/worker/model",
        categories=["image"],
    )
    workers = [object()]
    assert await scheduler.evaluate_diffusion_model(model, workers)
    assert model.categories == [CategoryEnum.VIDEO]
    read.assert_awaited_once_with(model, "model_index.json", workers)


@pytest.mark.asyncio
async def test_manual_video_category_does_not_require_known_pipeline(monkeypatch):
    read = AsyncMock(side_effect=AssertionError("Explicit video needs no discovery"))
    monkeypatch.setattr(scheduler, "get_diffusers_model_index_with_workers", read)
    model = new_model(
        1, "custom", huggingface_repo_id="org/custom", categories=["video"]
    )
    assert await scheduler.evaluate_pretrained_config(model)
    assert model.categories == ["video"]
    read.assert_not_awaited()


@pytest.mark.asyncio
async def test_video_named_image_and_vision_models_are_not_reclassified(monkeypatch):
    read = AsyncMock(
        return_value={"_diffusers_version": "1", "_class_name": "FluxPipeline"}
    )
    monkeypatch.setattr(scheduler, "get_diffusers_model_index_with_workers", read)
    image = new_model(
        1, "wan-video-alias", huggingface_repo_id="org/custom", categories=["image"]
    )
    assert await scheduler.evaluate_diffusion_model(image)
    assert image.categories == ["image"]
    vision = new_model(
        2, "video-understanding", huggingface_repo_id="org/custom", categories=["llm"]
    )
    assert not await scheduler.evaluate_diffusion_model(vision)
    assert vision.categories == ["llm"]
    assert read.await_count == 1


@pytest.mark.asyncio
async def test_remote_video_metadata_uses_only_model_cluster_workers(monkeypatch):
    model = Model(
        id=1,
        name="local",
        source=SourceEnum.LOCAL_PATH,
        local_path="/worker/model",
        cluster_id=7,
    )
    selected = SimpleNamespace(cluster_id=7)
    other = SimpleNamespace(cluster_id=8)
    monkeypatch.setattr(calculator, "get_diffusers_model_index", lambda *a, **k: None)
    read = AsyncMock(
        return_value={"_diffusers_version": "1", "_class_name": "WanPipeline"}
    )
    monkeypatch.setattr(calculator, "read_local_path_file_from_workers", read)
    await calculator.get_diffusers_model_index_with_workers(model, [selected, other])
    read.assert_awaited_once_with(model, "model_index.json", [selected])
