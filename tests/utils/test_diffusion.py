import pytest

from gpustack.schemas.models import CategoryEnum
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
