import os
from unittest.mock import AsyncMock, patch

import pytest
from transformers import PretrainedConfig

from gpustack.policies.candidate_selectors import VLLMResourceFitSelector
from gpustack.policies.resource_view import ResourceView
from gpustack.scheduler.calculator import calculate_llm_model_weight_size
from gpustack.schemas.models import BackendEnum, Model, SourceEnum
from gpustack.utils.hub import get_model_weight_size
from tests.fixtures.workers.fixtures import linux_nvidia_1_4090_24gx1
from tests.utils.model import new_model

# The (file name, size) pairs of a repository's root directory, and the size of one
# copy of its weights.
WEIGHT_FILE_CASES = [
    pytest.param(
        {"model.safetensors": 100, "pytorch_model.bin": 110},
        100,
        id="safetensors-win-over-bin",
    ),
    pytest.param(
        {"pytorch_model-1.bin": 50, "pytorch_model-2.bin": 60},
        110,
        id="bin-only",
    ),
    pytest.param(
        {"model.safetensors": 100, "training_args.bin": 7},
        100,
        id="bin-beside-safetensors-is-not-weights",
    ),
    pytest.param(
        {"consolidated.safetensors": 100, "model-1.safetensors": 60},
        60,
        id="consolidated-duplicates-other-weights",
    ),
    pytest.param(
        {"consolidated.safetensors": 100, "params.json": 1},
        100,
        id="consolidated-is-the-only-copy",
    ),
    pytest.param(
        {"flow.pt": 30, "llm.pt": 40, "config.json": 1},
        70,
        id="pt-only",
    ),
    pytest.param(
        {"t3.safetensors": 30, "s3gen.pt": 40},
        70,
        id="pt-beside-safetensors-still-counts",
    ),
]


@pytest.mark.parametrize("files, expected", WEIGHT_FILE_CASES)
def test_hub_model_weight_size_counts_one_copy(files, expected):
    model = Model(source=SourceEnum.HUGGING_FACE, huggingface_repo_id="test/repo")
    listing = [{"name": name, "size": size} for name, size in files.items()]

    with patch("gpustack.utils.hub.list_repo", return_value=listing):
        assert get_model_weight_size(model) == expected


@pytest.mark.parametrize("files, expected", WEIGHT_FILE_CASES)
def test_local_model_weight_size_counts_one_copy(tmp_path, files, expected):
    for name, size in files.items():
        with open(tmp_path / name, "wb") as f:
            os.ftruncate(f.fileno(), size)

    assert calculate_llm_model_weight_size(str(tmp_path)) == expected


@pytest.mark.asyncio
async def test_model_with_both_formats_fits_the_gpu_its_weights_fit(config):
    # 14.5 GB of weights, shipped as both safetensors and bin, on a 24 GiB card.
    listing = [
        {"name": "model.safetensors", "size": 14_483_498_040},
        {"name": "pytorch_model.bin", "size": 15_007_851_703},
    ]
    model = new_model(
        1,
        "both-formats",
        huggingface_repo_id="test/both-formats",
        backend=BackendEnum.VLLM.value,
        cpu_offloading=False,
    )
    selector = VLLMResourceFitSelector(
        config, model, [], resource_view=ResourceView([])
    )
    pretrained = PretrainedConfig(
        num_hidden_layers=2, num_attention_heads=8, hidden_size=64
    )
    with (
        patch(
            "gpustack.policies.candidate_selectors.base_candidate_selector.get_pretrained_config_with_workers",
            new=AsyncMock(return_value=pretrained),
        ),
        patch("gpustack.utils.hub.list_repo", return_value=listing),
    ):
        candidates = await selector.select_candidates([linux_nvidia_1_4090_24gx1()])

    assert [c.gpu_indexes for c in candidates] == [[0]]
