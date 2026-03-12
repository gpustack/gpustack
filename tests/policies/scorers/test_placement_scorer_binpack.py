import pytest

from gpustack.policies.base import Allocatable
from gpustack.policies.scorers.placement_scorer import PlacementScorer, ScaleTypeEnum
from gpustack.schemas.models import ComputedResourceClaim, PlacementStrategyEnum
from tests.utils.model import new_model


@pytest.mark.asyncio
async def test_binpack_scale_down_single_gpu_handles_none_ram_claim():
    model = new_model(
        1,
        "m",
        huggingface_repo_id="Qwen/Qwen2.5-7B-Instruct",
        placement_strategy=PlacementStrategyEnum.BINPACK,
    )
    scorer = PlacementScorer(model, [])

    score = await scorer._score_binpack_item(
        gpu_indexes=[0],
        computed_resource_claim=ComputedResourceClaim(
            ram=None,
            vram={0: 10 * 1024**3},
        ),
        allocatable=Allocatable(
            ram=40 * 1024**3,
            vram={0: 24 * 1024**3},
        ),
        scale_type=ScaleTypeEnum.SCALE_DOWN,
    )

    assert score is not None
    assert score > 0


@pytest.mark.asyncio
async def test_binpack_scale_down_single_gpu_handles_none_vram_claim():
    model = new_model(
        2,
        "m2",
        huggingface_repo_id="Qwen/Qwen2.5-7B-Instruct",
        placement_strategy=PlacementStrategyEnum.BINPACK,
    )
    scorer = PlacementScorer(model, [])

    score = await scorer._score_binpack_item(
        gpu_indexes=[0],
        computed_resource_claim=ComputedResourceClaim(
            ram=10 * 1024**3,
            vram=None,
        ),
        allocatable=Allocatable(
            ram=40 * 1024**3,
            vram={0: 24 * 1024**3},
        ),
        scale_type=ScaleTypeEnum.SCALE_DOWN,
    )

    assert score is not None
    assert score > 0
