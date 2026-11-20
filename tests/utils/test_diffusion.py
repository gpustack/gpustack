import json
from unittest.mock import Mock

import pytest

from gpustack.schemas.models import CategoryEnum, Model, SourceEnum
from gpustack.utils import hub
from gpustack.utils.diffusion import (
    VIDEO_PIPELINES,
    diffusion_model_category,
    normalize_diffusers_model_index,
)
from tests.utils.model import new_model


@pytest.mark.parametrize("pipeline", sorted(VIDEO_PIPELINES))
def test_known_video_pipeline_does_not_depend_on_repository_name(pipeline):
    index = {"_diffusers_version": "0.32.0", "_class_name": pipeline}
    assert diffusion_model_category(index) == CategoryEnum.VIDEO


@pytest.mark.parametrize("pipeline", ["FluxPipeline", "UnknownPipeline", None, []])
def test_unknown_and_image_pipelines_keep_image_category(pipeline):
    assert diffusion_model_category({"_class_name": pipeline}) == CategoryEnum.IMAGE


@pytest.mark.parametrize("data", [None, {}, [], [{"_class_name": "WanPipeline"}]])
def test_missing_diffusers_metadata_is_not_a_pipeline(data):
    assert normalize_diffusers_model_index(data) is None


@pytest.mark.parametrize("category", ["image", "video"])
def test_generation_weight_estimate_includes_pipeline_components(monkeypatch, category):
    model = new_model(
        1, "custom", categories=[category], huggingface_repo_id="org/custom"
    )
    monkeypatch.setattr(
        hub,
        "read_repo_file_content",
        lambda *a, **k: {
            "transformer": ["diffusers", "DiT"],
            "vae": ["diffusers", "VAE"],
        },
    )
    monkeypatch.setattr(
        hub,
        "list_repo",
        lambda *a, **k: [
            {"name": "transformer/model.safetensors", "size": 100},
            {"name": "vae/model.safetensors", "size": 20},
            {"name": "unrelated/model.safetensors", "size": 999},
            {"name": "transformer/config.json", "size": 10},
        ],
    )
    assert hub.get_diffusion_model_weight_size(model) == 120


def test_missing_hub_index_requires_an_accessible_repository(monkeypatch):
    model = new_model(1, "image", huggingface_repo_id="org/image")
    api = Mock()
    api.file_exists.return_value = False
    monkeypatch.setattr(hub, "HfApi", Mock(return_value=api))
    filesystem = Mock()
    monkeypatch.setattr(hub, "HfFileSystem", filesystem)
    assert hub.get_diffusers_model_index(model) is None
    api.repo_info.assert_called_once_with("org/image")
    filesystem.assert_not_called()


@pytest.mark.parametrize(
    "error", [PermissionError(), ConnectionError(), TimeoutError()]
)
def test_unavailable_hub_repository_is_not_a_missing_index(monkeypatch, error):
    model = new_model(1, "image", huggingface_repo_id="org/image")
    api = Mock()
    api.file_exists.return_value = False
    api.repo_info.side_effect = error
    monkeypatch.setattr(hub, "HfApi", Mock(return_value=api))
    with pytest.raises(type(error)):
        hub.get_diffusers_model_index(model)


def test_hub_index_read_failure_is_not_negative_metadata(monkeypatch):
    model = new_model(1, "image", huggingface_repo_id="org/image")
    api = Mock()
    api.file_exists.return_value = True
    monkeypatch.setattr(hub, "HfApi", Mock(return_value=api))
    filesystem = Mock()
    filesystem.open.side_effect = TimeoutError("hub read timed out")
    monkeypatch.setattr(hub, "HfFileSystem", Mock(return_value=filesystem))
    with pytest.raises(TimeoutError):
        hub.get_diffusers_model_index(model)


def test_local_absence_and_invalid_metadata_are_distinct(tmp_path):
    model = Model(name="local", source=SourceEnum.LOCAL_PATH, local_path=str(tmp_path))
    assert hub.get_diffusers_model_index(model) is None
    (tmp_path / "model_index.json").write_text("invalid json")
    with pytest.raises(json.JSONDecodeError):
        hub.get_diffusers_model_index(model)
    assert hub.read_repo_file_content(model, "model_index.json") is None
