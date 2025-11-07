from unittest.mock import patch, AsyncMock

import pytest

from gpustack.policies.candidate_selectors.custom_backend_resource_fit_selector import (
    CustomBackendResourceFitSelector,
)
from gpustack.policies.scorers.placement_scorer import PlacementScorer
from tests.fixtures.workers.fixtures import linux_nvidia_4_4080_16gx4, linux_cpu_1
from tests.policies.candidate_selectors.vllm.test_vllm_resource_fit_selector import (
    make_model,
)


@pytest.mark.parametrize(
    "index, workers, model, expect_msg",
    [
        (
            1,
            [linux_nvidia_4_4080_16gx4()],
            make_model(1, None, "Qwen/Qwen2.5-Omni-7B"),
            [
                '- The model requires approximately 25.99 GiB of VRAM and 2.6 GiB of RAM.\n'
                '- The current available GPU only has 15.99 GiB allocatable VRAM.'
            ],
        ),
        (
            2,
            [linux_nvidia_4_4080_16gx4()],
            make_model(
                1,
                [
                    "host-4-4080:nvidia:0",
                    "host-4-4080:nvidia:1",
                    "host-4-4080:nvidia:2",
                ],
                "Qwen/Qwen3-8B",
            ),
            [
                '- The model requires approximately 19.31 GiB of VRAM and 2.0 GiB of RAM.\n'
            ],
        ),
    ],
)
@pytest.mark.asyncio
async def test_schedule_single_work_multi_gpu(
    config, index, workers, model, expect_msg
):
    m = model

    resource_fit_selector = CustomBackendResourceFitSelector(config, m)
    placement_scorer = PlacementScorer(m)

    if index == 1:
        # Simulate a scenario where the model's num_attention_heads cannot be evenly divided by the gpu_count through auto-scheduling.
        resource_fit_selector._num_attention_heads = 25

    with (
        patch(
            'gpustack.policies.utils.get_worker_model_instances',
            return_value=[],
        ),
        patch(
            'gpustack.policies.scorers.placement_scorer.get_model_instances',
            return_value=[],
        ),
        patch('sqlmodel.ext.asyncio.session.AsyncSession', AsyncMock()),
        patch(
            'gpustack.schemas.workers.Worker.all',
            return_value=workers,
        ),
    ):

        candidates = await resource_fit_selector.select_candidates(workers)
        _ = await placement_scorer.score(candidates)

        assert resource_fit_selector._messages == expect_msg


@pytest.mark.parametrize(
    "workers, model, env_overrides, expect_msg",
    [
        (
            [linux_nvidia_4_4080_16gx4()],
            make_model(1, None, "Qwen/Qwen2.5-Omni-7B"),
            {"GPUSTACK_MODEL_VRAM_CLAIM": str(27 * 1024**3)},
            [
                '- The model requires approximately 27.0 GiB of VRAM and 2.7 GiB of RAM.\n'
                '- The current available GPU only has 15.99 GiB allocatable VRAM.'
            ],
        ),
        (
            [linux_cpu_1()],
            make_model(1, None, "Qwen/Qwen2.5-7B-Instruct", cpu_offloading=True),
            {"GPUSTACK_MODEL_VRAM_CLAIM": str(700 * 1024**3)},
            [
                '- The model requires approximately 700.0 GiB of VRAM and 70.0 GiB of RAM.\n'
                '- CPU-only inference is supported. Requires at least 70.0 GiB RAM.'
            ],
        ),
    ],
)
@pytest.mark.asyncio
async def test_failed_cases_auto_schedule(
    config, workers, model, env_overrides, expect_msg
):
    model.env = env_overrides

    resource_fit_selector = CustomBackendResourceFitSelector(config, model)
    placement_scorer = PlacementScorer(model)

    with (
        patch(
            'gpustack.policies.utils.get_worker_model_instances',
            return_value=[],
        ),
        patch(
            'gpustack.policies.scorers.placement_scorer.get_model_instances',
            return_value=[],
        ),
        patch('sqlmodel.ext.asyncio.session.AsyncSession', AsyncMock()),
        patch(
            'gpustack.schemas.workers.Worker.all',
            return_value=workers,
        ),
    ):
        candidates = await resource_fit_selector.select_candidates(workers)
        _ = await placement_scorer.score(candidates)

        assert resource_fit_selector._messages == expect_msg
