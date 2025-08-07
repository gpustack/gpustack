from unittest.mock import patch, AsyncMock

import pytest

from gpustack.policies.candidate_selectors import (
    VLLMResourceFitSelector,
    AscendMindIEResourceFitSelector,
)
from gpustack.policies.scorers.placement_scorer import PlacementScorer
from gpustack.scheduler import scheduler
from gpustack.schemas.models import GPUSelector
from tests.fixtures.workers.fixtures import (
    linux_nvidia_4_4080_16gx4,
    linux_huawei_2_910b_64gx8,
)
from tests.utils.model import new_model


@pytest.mark.parametrize(
    "mock_flavor_list, model, expect_type_count",
    [
        (
            [
                "NVIDIA RTX 4080 16GB",
                "NVIDIA RTX 4080 16GB",
                "NVIDIA RTX 4090 24GB",
                "NVIDIA RTX 4090 24GB",
            ],
            new_model(
                1,
                "test_name",
                1,
                huggingface_repo_id="Qwen/Qwen2.5-7B-Instruct",
                cpu_offloading=False,
                gpu_selector=GPUSelector(
                    gpu_type="NVIDIA RTX 4080 16GB"  # Require specific GPU type
                ),
                backend_parameters=[],
            ),
            ("NVIDIA RTX 4080 16GB", 2),
        ),
        (
            [
                "NVIDIA RTX 4080 16GB",
                "NVIDIA RTX 4080 16GB",
                "NVIDIA RTX 4080 16GB",
                "NVIDIA RTX 4080 16GB",
            ],
            new_model(
                2,
                "test_name",
                1,
                huggingface_repo_id="Qwen/Qwen2.5-7B-Instruct",
                cpu_offloading=False,
                gpu_selector=GPUSelector(
                    gpu_type="NVIDIA RTX 4090 24GB",  # Require different GPU type
                    gpu_count=1,  # Specify gpu_count to avoid division by zero in attention heads check
                ),
                backend_parameters=[],
            ),
            ("NVIDIA RTX 4080 24GB", 0),
        ),
        (
            [
                "NVIDIA RTX 4080 16GB",
                "NVIDIA RTX 4080 16GB",
                "NVIDIA RTX 4090 16GB",
                "NVIDIA RTX 4090 16GB",
            ],
            new_model(
                3,
                "test_name",
                1,
                huggingface_repo_id="Qwen/Qwen2.5-7B-Instruct",
                cpu_offloading=False,
                gpu_selector=GPUSelector(
                    gpu_count=2,  # Require 2 GPUs
                    gpu_type="NVIDIA RTX 4080 16GB",  # Require specific GPU type
                ),
                backend_parameters=[],
            ),
            ("NVIDIA RTX 4080 16GB", 2),
        ),
    ],
)
@pytest.mark.asyncio
async def test_vllm_gpu_selector(config, mock_flavor_list, model, expect_type_count):
    """
    Test gpu_selector.gpu_type filtering in _find_single_worker_multi_gpu_full_offloading_candidates.
    When gpu_type is specified and GPUs don't match, they should be filtered out.
    """
    workers = [
        linux_nvidia_4_4080_16gx4(),  # 4 RTX 4080 GPUs
    ]

    # Create a worker with mixed GPU types by modifying the fixture
    worker = workers[0]

    # Mock GPU types
    for i in range(len(worker.status.gpu_devices)):
        worker.status.gpu_devices[i].flavor_name = mock_flavor_list[i]

    resource_fit_selector = VLLMResourceFitSelector(config, model)
    placement_scorer = PlacementScorer(model)

    with (
        patch(
            'gpustack.policies.utils.get_worker_model_instances',
            return_value=[],
        ),
        patch(
            'gpustack.policies.candidate_selectors.vllm_resource_fit_selector.get_worker_model_instances',
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
        candidates = await placement_scorer.score(candidates)
        candidate, _ = await scheduler.find_candidate(config, model, workers)

        # Should return candidates using only the matching GPU type
        if not candidates and expect_type_count[1] > 0:
            raise ValueError("No found candidates")
        # All selected GPUs should be of the specified type
        for candidate in candidates:
            for gpu_index in candidate.gpu_indexes:
                gpu = worker.status.gpu_devices[gpu_index]
                assert gpu.flavor_name == expect_type_count[0]
            assert len(candidate.gpu_indexes) == expect_type_count[1]


@pytest.mark.parametrize(
    "mock_flavor_list, model, expect_type_count",
    [
        (
            [
                "Huawei 910B1 64GB",
                "Huawei 910B1 64GB",
                "Huawei 910B2 64GB",
                "Huawei 910B2 64GB",
            ],
            new_model(
                1,
                "test_name",
                1,
                huggingface_repo_id="Qwen/Qwen3-32B",
                cpu_offloading=False,
                gpu_selector=GPUSelector(
                    gpu_type="Huawei 910B1 64GB"  # Require specific GPU type
                ),
                backend_parameters=[],
            ),
            ("Huawei 910B1 64GB", 2),
        ),
        (
            [
                "Huawei 910B1 64GB",
                "Huawei 910B1 64GB",
                "Huawei 910B1 64GB",
                "Huawei 910B1 64GB",
            ],
            new_model(
                2,
                "test_name",
                1,
                huggingface_repo_id="Qwen/Qwen2.5-7B-Instruct",
                cpu_offloading=False,
                gpu_selector=GPUSelector(
                    gpu_type="Huawei 910B2 64GB",  # Require different GPU type
                    gpu_count=1,  # Specify gpu_count to avoid division by zero in attention heads check
                ),
                backend_parameters=[],
            ),
            ("Huawei 910B2 64GB", 0),
        ),
        (
            [
                "Huawei 910B1 64GB",
                "Huawei 910B1 64GB",
                "Huawei 910B2 64GB",
                "Huawei 910B2 64GB",
            ],
            new_model(
                3,
                "test_name",
                1,
                huggingface_repo_id="Qwen/Qwen2.5-7B-Instruct",  # need 20GB vram
                cpu_offloading=False,
                gpu_selector=GPUSelector(
                    gpu_count=2,  # Require 2 GPUs
                    gpu_type="Huawei 910B1 64GB",  # Require specific GPU type
                ),
                backend_parameters=[],
            ),
            ("Huawei 910B1 64GB", 2),
        ),
    ],
)
@pytest.mark.asyncio
async def test_ascend_mindie_gpu_selector(
    config, mock_flavor_list, model, expect_type_count
):
    """
    Test gpu_selector.gpu_type filtering in AscendMindIEResourceFitSelector.
    When gpu_type is specified and NPUs don't match, they should be filtered out.
    """
    workers = [
        linux_huawei_2_910b_64gx8(),
    ]

    # Create a worker with mixed NPU types by modifying the fixture
    worker = workers[0]
    worker.status.gpu_devices = worker.status.gpu_devices[
        :4
    ]  # Become 4 NPU 910B devices

    # Mock NPU types
    for i in range(len(worker.status.gpu_devices)):
        worker.status.gpu_devices[i].flavor_name = mock_flavor_list[i]

    resource_fit_selector = AscendMindIEResourceFitSelector(config, model)

    with (
        patch('sqlmodel.ext.asyncio.session.AsyncSession', AsyncMock()),
        patch(
            'gpustack.policies.utils.get_worker_model_instances',
            return_value=[],
        ),
        patch(
            'gpustack.schemas.workers.Worker.all',
            return_value=workers,
        ),
    ):
        candidates = await resource_fit_selector.select_candidates(workers)

        # Should return candidates using only the matching GPU type
        if not candidates and expect_type_count[1] > 0:
            raise ValueError("No found candidates")
        # All selected NPUs should be of the specified type
        for candidate in candidates:
            for gpu_index in candidate.gpu_indexes:
                gpu = worker.status.gpu_devices[gpu_index]
                assert gpu.flavor_name == expect_type_count[0]
            assert len(candidate.gpu_indexes) == expect_type_count[1]
