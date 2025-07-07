from typing import List

import pytest
from unittest.mock import patch, AsyncMock
from tests.utils.model import new_model, new_model_instance
from gpustack.policies.candidate_selectors import VLLMResourceFitSelector
from gpustack.policies.scorers.placement_scorer import PlacementScorer
from gpustack.scheduler import scheduler
from gpustack.schemas.models import (
    CategoryEnum,
    ComputedResourceClaim,
    GPUSelector,
    RayActor,
    ModelInstanceStateEnum,
    ModelInstance,
)
from tests.fixtures.workers.fixtures import (
    linux_nvidia_1_4090_24gx1,
    linux_nvidia_22_H100_80gx8,
    linux_nvidia_23_H100_80gx8,
    linux_nvidia_24_H100_80gx8,
    linux_nvidia_25_H100_80gx8,
    linux_nvidia_3_4090_24gx1,
    linux_nvidia_4_4080_16gx4,
    linux_nvidia_5_a100_80gx2,
    linux_nvidia_6_a100_80gx2,
    linux_nvidia_7_a100_80gx2,
    linux_nvidia_2_4080_16gx2,
)
from tests.utils.scheduler import compare_candidates


@pytest.mark.asyncio
async def test_manual_schedule_to_2_worker_2_gpu(config):
    workers = [
        linux_nvidia_1_4090_24gx1(),
        linux_nvidia_3_4090_24gx1(),
    ]

    m = new_model(
        1,
        "test_name",
        1,
        huggingface_repo_id="Qwen/Qwen2.5-7B-Instruct",
        cpu_offloading=False,
        gpu_selector=GPUSelector(
            gpu_ids=[
                "host4090:cuda:0",
                "host-2-4090:cuda:0",
            ]
        ),
        backend_parameters=[],
    )

    resource_fit_selector = VLLMResourceFitSelector(config, m)
    placement_scorer = PlacementScorer(m)

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
        candidate, _ = await scheduler.find_candidate(config, m, workers)

        expected_candidates = [
            {
                "worker_id": 2,
                "worker_name": "host4090",
                "gpu_indexes": [0],
                "is_unified_memory": False,
                "vram": {
                    0: 23413653504,
                },
                "ray_actors": [
                    RayActor(
                        worker_id=0,
                        gpu_index=0,
                        computed_resource_claim=ComputedResourceClaim(
                            vram={0: 23413653504},
                        ),
                    )
                ],
            },
        ]

        assert len(candidates) == 1
        assert candidate == candidates[0]
        compare_candidates(candidates, expected_candidates)


@pytest.mark.asyncio
async def test_manual_schedule_to_2_worker_4_gpu_select_main_with_most_gpus(
    config,
):
    workers = [
        linux_nvidia_1_4090_24gx1(),
        linux_nvidia_4_4080_16gx4(),
    ]

    m = new_model(
        1,
        "test_name",
        1,
        huggingface_repo_id="Qwen/Qwen2.5-7B-Instruct",
        cpu_offloading=False,
        gpu_selector=GPUSelector(
            gpu_ids=[
                "host4090:cuda:0",
                "host-4-4080:cuda:0",
                "host-4-4080:cuda:1",
                "host-4-4080:cuda:2",
            ]
        ),
        backend_parameters=[],
    )

    resource_fit_selector = VLLMResourceFitSelector(config, m)
    placement_scorer = PlacementScorer(m)

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
        candidate, _ = await scheduler.find_candidate(config, m, workers)

        expected_candidates = [
            {
                "worker_id": 5,
                "worker_name": "host-4-4080",
                "gpu_indexes": [0, 1, 2],
                "is_unified_memory": False,
                "vram": {
                    0: 15454332518,
                    1: 15454332518,
                    2: 15454332518,
                },
                "ray_actors": [
                    RayActor(
                        worker_id=0,
                        gpu_index=0,
                        computed_resource_claim=ComputedResourceClaim(
                            vram={0: 23413653504},
                        ),
                    )
                ],
            },
        ]

        assert len(candidates) == 1
        assert candidate == candidates[0]
        compare_candidates(candidates, expected_candidates)


@pytest.mark.asyncio
async def test_manual_schedule_to_3_workers_4_gpus(
    config,
):
    def workers():
        return [
            linux_nvidia_5_a100_80gx2(),
            linux_nvidia_6_a100_80gx2(),
            linux_nvidia_7_a100_80gx2(),
        ]

    m = new_model(
        1,
        "test_name",
        1,
        huggingface_repo_id="Qwen/Qwen2.5-7B-Instruct",
        cpu_offloading=False,
        gpu_selector=GPUSelector(
            gpu_ids=[
                "llm01-A100:cuda:0",
                "llm01-A100:cuda:1",
                "llm02-A100:cuda:0",
                "llm03-A100:cuda:0",
            ]
        ),
        backend_parameters=[],
    )

    resource_fit_selector = VLLMResourceFitSelector(config, m)
    placement_scorer = PlacementScorer(m)

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
            return_value=workers(),
        ),
    ):

        candidates = await resource_fit_selector.select_candidates(workers())
        candidates = await placement_scorer.score(candidates)
        candidate, _ = await scheduler.find_candidate(config, m, workers())

        expected_candidates = [
            {
                "worker_id": 8,
                "worker_name": "llm01-A100",
                "gpu_indexes": [0, 1],
                "is_unified_memory": False,
                "vram": {
                    0: 77309411328,
                    1: 77309411328,
                },
                "ray_actors": [
                    RayActor(
                        worker_id=9,
                        gpu_index=0,
                        computed_resource_claim=ComputedResourceClaim(
                            vram={0: 77309411328},
                        ),
                    ),
                    RayActor(
                        worker_id=10,
                        gpu_index=0,
                        computed_resource_claim=ComputedResourceClaim(
                            vram={0: 77309411328},
                        ),
                    ),
                ],
            },
        ]

        assert len(candidates) == 1
        assert candidate == candidates[0]
        compare_candidates(candidates, expected_candidates)


@pytest.mark.asyncio
async def test_auto_schedule_to_2_worker_2_gpu(config):
    workers = [
        linux_nvidia_1_4090_24gx1(),
        linux_nvidia_3_4090_24gx1(),
    ]

    m = new_model(
        1,
        "test_name",
        1,
        huggingface_repo_id="Qwen/Qwen2.5-14B-Instruct",
        cpu_offloading=False,
        backend_parameters=[],
    )

    resource_fit_selector = VLLMResourceFitSelector(config, m)
    placement_scorer = PlacementScorer(m)

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
        candidates = await placement_scorer.score(candidates)
        candidate, _ = await scheduler.find_candidate(config, m, workers)

        expected_candidates = [
            {
                "worker_id": 2,
                "worker_name": "host4090",
                "gpu_indexes": [0],
                "is_unified_memory": False,
                "vram": {
                    0: 23413653504,
                },
                "ray_actors": [
                    RayActor(
                        worker_id=0,
                        gpu_index=0,
                        computed_resource_claim=ComputedResourceClaim(
                            vram={0: 23413653504},
                        ),
                    )
                ],
            },
        ]

        assert len(candidates) == 1
        assert candidate == candidates[0]
        compare_candidates(candidates, expected_candidates)


@pytest.mark.asyncio
async def test_auto_schedule_to_2_worker_16_gpu_deepseek_r1(config):
    workers = [
        linux_nvidia_1_4090_24gx1(),
        linux_nvidia_3_4090_24gx1(),
        linux_nvidia_22_H100_80gx8(),
        linux_nvidia_23_H100_80gx8(),
        linux_nvidia_24_H100_80gx8(),
        linux_nvidia_25_H100_80gx8(),
    ]

    m = new_model(
        1,
        "test_name",
        1,
        huggingface_repo_id="deepseek-ai/DeepSeek-R1",
        cpu_offloading=False,
        backend_parameters=[],
    )

    resource_fit_selector = VLLMResourceFitSelector(config, m)
    placement_scorer = PlacementScorer(m)

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
        candidates = await placement_scorer.score(candidates)
        candidate, _ = await scheduler.find_candidate(config, m, workers)

        expected_candidates = [
            {
                "worker_id": 22,
                "worker_name": "host22-h100",
                "gpu_indexes": [0, 1, 2, 3, 4, 5, 6, 7],
                "is_unified_memory": False,
                "vram": {
                    0: 77309411328,
                    1: 77309411328,
                    2: 77309411328,
                    3: 77309411328,
                    4: 77309411328,
                    5: 77309411328,
                    6: 77309411328,
                    7: 77309411328,
                },
                "ray_actors": [
                    RayActor(
                        worker_id=23,
                        gpu_indexes=[0, 1, 2, 3, 4, 5, 6, 7],
                        computed_resource_claim=ComputedResourceClaim(
                            vram={
                                0: 77309411328,
                                1: 77309411328,
                                2: 77309411328,
                                3: 77309411328,
                                4: 77309411328,
                                5: 77309411328,
                                6: 77309411328,
                                7: 77309411328,
                            },
                        ),
                    ),
                ],
            },
        ]

        assert len(candidates) == 1
        assert candidate == candidates[0]
        compare_candidates(candidates, expected_candidates)


@pytest.mark.asyncio
async def test_auto_schedule_embedding_models(config):
    workers = [
        linux_nvidia_1_4090_24gx1(),
    ]

    m = new_model(
        1,
        "test_name",
        1,
        huggingface_repo_id="BAAI/bge-base-en-v1.5",
        cpu_offloading=False,
        backend_parameters=[],
        categories=[CategoryEnum.EMBEDDING],
    )

    resource_fit_selector = VLLMResourceFitSelector(config, m)
    placement_scorer = PlacementScorer(m)

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
        candidates = await placement_scorer.score(candidates)
        candidate, _ = await scheduler.find_candidate(config, m, workers)

        expected_candidates = [
            {
                "worker_id": 2,
                "worker_name": "host4090",
                "gpu_indexes": [0],
                "is_unified_memory": False,
                "vram": {
                    0: 1588014354,
                },
            },
        ]

        assert len(candidates) == 1
        assert candidate == candidates[0]
        compare_candidates(candidates, expected_candidates)


@pytest.mark.asyncio
async def test_auto_schedule_single_work_single_gpu(config):
    workers = [
        linux_nvidia_1_4090_24gx1(),
    ]

    m = new_model(
        1,
        "test_name",
        1,
        huggingface_repo_id="Qwen/Qwen3-32B",
        cpu_offloading=False,
        backend_parameters=[],
    )

    resource_fit_selector = VLLMResourceFitSelector(config, m)
    placement_scorer = PlacementScorer(m)

    with (
        patch(
            'gpustack.policies.utils.get_worker_model_instances',
            return_value=[
                new_model_instance(
                    10,
                    "test-1",
                    1,
                    2,
                    ModelInstanceStateEnum.RUNNING,
                    [2],
                    ComputedResourceClaim(
                        is_unified_memory=False,
                        offload_layers=10,
                        total_layers=10,
                        ram=0,
                        vram={2: 500 * 1024 * 1024},
                    ),
                ),
            ],
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

        expect_msg = [
            """- The model requires approximately 75.23 GiB of VRAM.
- With --gpu-memory-utilization=0.9, All GPUs combined need to provide at least 83.59 GiB of total VRAM.
- The current available GPU only has 24.23 GiB allocatable VRAM (100.00%)."""
        ]
        assert resource_fit_selector._messages == expect_msg


@pytest.mark.asyncio
async def test_auto_schedule_single_work_multi_gpu(config):
    workers = [
        linux_nvidia_4_4080_16gx4(),
    ]

    m = new_model(
        1,
        "test_name",
        1,
        huggingface_repo_id="Qwen/Qwen3-32B",
        cpu_offloading=False,
        backend_parameters=[],
    )

    resource_fit_selector = VLLMResourceFitSelector(config, m)
    placement_scorer = PlacementScorer(m)

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

        expect_msg = [
            """- The model requires approximately 75.23 GiB of VRAM.
- With --gpu-memory-utilization=0.9, All GPUs combined need to provide at least 83.59 GiB of total VRAM.
- The largest available worker has 63.97 GiB allocatable VRAM, 4/4 of GPUs meet the VRAM utilization ratio, providing 57.57 GiB of allocatable VRAM."""
        ]

        assert expect_msg == resource_fit_selector._messages


@pytest.mark.asyncio
async def test_auto_schedule_multi_work_multi_gpu(config):
    workers = [linux_nvidia_2_4080_16gx2(), linux_nvidia_2_4080_16gx2()]
    workers[1].id += 1
    workers[1].name += "-1"
    m = new_model(
        1,
        "test_name",
        1,
        huggingface_repo_id="Qwen/Qwen3-32B",
        cpu_offloading=False,
        backend_parameters=[],
    )

    resource_fit_selector = VLLMResourceFitSelector(config, m)
    placement_scorer = PlacementScorer(m)

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

        expect_msg = [
            """- The model requires approximately 75.23 GiB of VRAM.
- With --gpu-memory-utilization=0.9, All GPUs combined need to provide at least 83.59 GiB of total VRAM.
- The optimal combination ['host4080', 'host4080-1'] provides 57.57 GiB of allocatable VRAM.
- Cannot find a suitable worker combination to run the model in distributed mode. If you are confident that the resources are sufficient, you may manually schedule the model by selecting the workers and GPUs."""
        ]

        assert resource_fit_selector._messages == expect_msg


@pytest.mark.asyncio
async def test_manual_schedule_multi_work_multi_gpu(config):
    workers = [
        linux_nvidia_1_4090_24gx1(),
        linux_nvidia_3_4090_24gx1(),
    ]
    m = new_model(
        1,
        "test_name",
        1,
        huggingface_repo_id="Qwen/Qwen3-32B",
        cpu_offloading=False,
        gpu_selector=GPUSelector(
            gpu_ids=[
                "host4090:cuda:0",
                "host-2-4090:cuda:0",
            ]
        ),
        backend_parameters=[],
    )

    def _get_mis() -> List[ModelInstance]:
        return [
            new_model_instance(
                10,
                "test-1",
                1,
                2,
                ModelInstanceStateEnum.RUNNING,
                [0],
                ComputedResourceClaim(
                    is_unified_memory=False,
                    offload_layers=10,
                    total_layers=10,
                    ram=0,
                    vram={0: 0},
                ),
            ),
            new_model_instance(
                11,
                "test-2",
                1,
                12,
                ModelInstanceStateEnum.RUNNING,
                [0],
                ComputedResourceClaim(
                    is_unified_memory=False,
                    offload_layers=10,
                    total_layers=10,
                    ram=0,
                    vram={0: 0},
                ),
            ),
        ]

    resource_fit_selector = VLLMResourceFitSelector(config, m)
    placement_scorer = PlacementScorer(m)

    resource_fit_selector2 = VLLMResourceFitSelector(config, m)
    placement_scorer2 = PlacementScorer(m)

    with (
        patch('sqlmodel.ext.asyncio.session.AsyncSession', return_value=AsyncMock()),
        patch(
            'gpustack.policies.candidate_selectors.VLLMResourceFitSelector._validate_distributed_vllm_limit_per_worker',
            return_value=True,
        ),
        patch(
            'gpustack.policies.utils.get_worker_model_instances',
            return_value=_get_mis(),
        ),
        patch(
            'gpustack.policies.scorers.placement_scorer.get_model_instances',
            side_effect=_get_mis,
        ),
        patch(
            'gpustack.schemas.workers.Worker.all',
            return_value=workers,
        ),
    ):

        candidates = await resource_fit_selector.select_candidates(workers)
        _ = await placement_scorer.score(candidates)

        expect_msg = [
            """- The model requires approximately 75.23 GiB of VRAM.
- With --gpu-memory-utilization=0.9, All GPUs combined need to provide at least 83.59 GiB of total VRAM.
- Selected GPUs have 47.22 GiB allocatable VRAM, 2/2 of GPUs meet the VRAM utilization ratio, providing 42.50 GiB of allocatable VRAM."""
        ]

        assert resource_fit_selector._messages == expect_msg

        # case 2

        for worker in workers:
            worker.system_reserved.vram = 12000000000

        candidates2 = await resource_fit_selector2.select_candidates(workers)
        _ = await placement_scorer2.score(candidates2)

        expect_msg2 = [
            """- The model requires approximately 75.23 GiB of VRAM.
- With --gpu-memory-utilization=0.9, All GPUs combined need to provide at least 83.59 GiB of total VRAM.
- Worker host4090 GPU indexes [0] and other 1 workers fails to meet the 90.00% allocatable VRAM ratio.
- Selected GPUs have 25.87 GiB allocatable VRAM, 0/2 of GPUs meet the VRAM utilization ratio, providing 0.00 GiB of allocatable VRAM."""
        ]

        assert resource_fit_selector2._messages == expect_msg2
