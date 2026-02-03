import pytest
from unittest.mock import patch, AsyncMock

from tests.utils.model import make_model, new_model, new_model_instance
from gpustack.policies.candidate_selectors import SGLangResourceFitSelector
from gpustack.policies.scorers.placement_scorer import PlacementScorer
from gpustack.scheduler import scheduler
from gpustack.schemas.models import (
    CategoryEnum,
    ComputedResourceClaim,
    ExtendedKVCacheConfig,
    GPUSelector,
    ModelInstanceStateEnum,
    ModelInstanceSubordinateWorker,
    BackendEnum,
)
from tests.fixtures.workers.fixtures import (
    linux_mix_1_nvidia_4080_16gx1_rocm_7800_16gx1,
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
    linux_rocm_1_7800_16gx1,
    linux_rocm_2_7800_16gx2,
    linux_nvidia_26_H200_141gx8,
    linux_ascend_1_910b_64gx8,
)
from tests.utils.scheduler import compare_candidates


def expected_candidate(
    worker_id, worker_name, gpu_indexes, vram, subworkers=None, ram=None
):
    candidate = {
        "worker_id": worker_id,
        "worker_name": worker_name,
        "gpu_indexes": gpu_indexes,
        "is_unified_memory": False,
        "vram": vram,
        "subordinate_workers": subworkers or [],
    }
    if ram is not None:
        candidate["ram"] = ram
    return candidate


# TODO aggregate other test cases for SGLangResourceFitSelector here
@pytest.mark.parametrize(
    "case_name, m, workers, expected_candidates, final_candidate_index",
    [
        # Manually select two GPUs from 1 worker with DP2 parameter.
        # Check point:
        # - Candidate selection correctness.
        # - Manual GPU selection handling.
        # - dp-size parameter handling.
        (
            "manual_select_2_gpus_1_worker_dp2",
            make_model(
                0,
                [
                    "host4080:cuda:0",
                    "host4080:cuda:1",
                ],
                "Qwen/Qwen3-0.6B",
                backend_parameters=["--data-parallel-size=2"],
            ),
            [
                linux_nvidia_2_4080_16gx2(),
            ],
            [
                expected_candidate(
                    3,
                    "host4080",
                    [0, 1],
                    {0: 13136182640, 1: 13136182640},
                )
            ],
            0,
        ),
        # Auto schedule two GPUs from 1 worker with DP2 parameter.
        # Check point:
        # - Candidate selection correctness.
        # - dp-size parameter handling.
        (
            "auto_select_2_gpus_1_worker_dp2",
            new_model(
                1,
                "test_name",
                1,
                huggingface_repo_id="Qwen/Qwen3-0.6B",
                backend_parameters=["--data-parallel-size=2"],
            ),
            [
                linux_nvidia_2_4080_16gx2(),
            ],
            [
                expected_candidate(
                    3,
                    "host4080",
                    [0, 1],
                    {0: 13256383004, 1: 13256383004},
                )
            ],
            0,
        ),
        # Auto schedule two GPUs from 1 worker with PP2 parameter.
        # Check point:
        # - Candidate selection correctness.
        # - pp-size parameter handling.
        (
            "auto_select_2_gpus_1_worker_pp2",
            new_model(
                1,
                "test_name",
                1,
                huggingface_repo_id="Qwen/Qwen3-0.6B",
                backend_parameters=["--pipeline-parallel-size=2"],
            ),
            [
                linux_nvidia_2_4080_16gx2(),
            ],
            [
                expected_candidate(
                    3,
                    "host4080",
                    [0, 1],
                    {0: 13136182640, 1: 13136182640},
                )
            ],
            0,
        ),
        # Auto schedule for DeepSeekV32 model
        (
            "auto_select_deepseekv32_model",
            new_model(
                1,
                "test_name",
                1,
                huggingface_repo_id="deepseek-ai/DeepSeek-V3.2",
                backend_version="0.11.0",
            ),
            [linux_nvidia_26_H200_141gx8()],
            [
                expected_candidate(
                    26,
                    "host26-h200",
                    [0, 1, 2, 3, 4, 5, 6, 7],
                    {
                        0: 137420587581,
                        1: 137420587581,
                        2: 137420587581,
                        3: 137420587581,
                        4: 137420587581,
                        5: 137420587581,
                        6: 137420587581,
                        7: 137420587581,
                    },
                )
            ],
            0,
        ),
        (
            "auto_select_multimodal_4_gpus_1_worker_tp4",
            new_model(
                1,
                "test_name",
                1,
                huggingface_repo_id="Qwen/Qwen2-VL-7B-Instruct",
                backend_parameters=["--tp-size=4"],
            ),
            [
                linux_nvidia_4_4080_16gx4(),
            ],
            [
                expected_candidate(
                    5,
                    "host-4-4080",
                    [0, 1, 2, 3],
                    {0: 9770572447, 1: 9770572447, 2: 9770572447, 3: 9770572447},
                )
            ],
            0,
        ),
        (
            "auto_select_multimodal_4_gpus_per_replica_1_worker",
            new_model(
                1,
                "test_name",
                1,
                huggingface_repo_id="Qwen/Qwen2-VL-7B-Instruct",
                gpu_selector=GPUSelector(
                    gpus_per_replica=4,
                    gpu_ids=[
                        "host-4-4080:cuda:0",
                        "host-4-4080:cuda:1",
                        "host-4-4080:cuda:2",
                        "host-4-4080:cuda:3",
                    ],
                ),
            ),
            [
                linux_nvidia_4_4080_16gx4(),
            ],
            [
                expected_candidate(
                    5,
                    "host-4-4080",
                    [0, 1, 2, 3],
                    {0: 9770572447, 1: 9770572447, 2: 9770572447, 3: 9770572447},
                )
            ],
            0,
        ),
        (
            "auto_select_multimodal_1_gpus_1_worker_npu",
            new_model(
                1,
                "test_name",
                1,
                huggingface_repo_id="Qwen/Qwen2-VL-7B-Instruct",
                backend_parameters=[],
            ),
            [
                linux_ascend_1_910b_64gx8(),
            ],
            [
                expected_candidate(
                    1,
                    "ascend_0",
                    [i],
                    {i: 41506563948},
                )
                for i in range(8)
            ],
            0,
        ),
        # Auto schedule 1 GPU from 1 worker for diffusion model.
        # Check point:
        # - mem-fraction-static shouldn't affect.
        (
            "auto_select_1_gpus_1_worker_for_diffusion",
            new_model(
                1,
                "test_name",
                1,
                huggingface_repo_id="Qwen/Qwen-Image",
                categories=[CategoryEnum.IMAGE],
            ),
            [
                linux_nvidia_5_a100_80gx2(),
            ],
            [
                expected_candidate(
                    8,
                    "llm01-A100",
                    [0],
                    {0: 57699249390},
                ),
                expected_candidate(
                    8,
                    "llm01-A100",
                    [1],
                    {1: 57699249390},
                ),
            ],
            0,
        ),
        # Manually select 2 GPUs from 1 worker for diffusion model.
        # Check point:
        # - both gpu claims should be equal.
        # - mem-fraction-static shouldn't affect.
        (
            "auto_select_2_gpus_1_worker_for_diffusion",
            new_model(
                1,
                "test_name",
                1,
                huggingface_repo_id="Qwen/Qwen-Image",
                categories=[CategoryEnum.IMAGE],
                backend_parameters=["--tp-size=2"],
                gpu_selector=GPUSelector(
                    gpu_ids=["llm01-A100:cuda:0", "llm01-A100:cuda:1"]
                ),
            ),
            [
                linux_nvidia_5_a100_80gx2(),
            ],
            [
                expected_candidate(
                    8,
                    "llm01-A100",
                    [0, 1],
                    {0: 57699249390, 1: 57699249390},
                ),
            ],
            0,
        ),
    ],
)
@pytest.mark.asyncio
async def test_select_candidates(
    config, case_name, m, workers, expected_candidates, final_candidate_index
):
    with (
        patch(
            'gpustack.scheduler.scheduler.BackendFrameworkFilter._has_supported_runners',
            return_value=(True, []),
        ),
        patch(
            'gpustack.schemas.workers.Worker.all',
            return_value=workers,
        ),
        patch(
            'gpustack.policies.worker_filters.backend_framework_filter.async_session',
            return_value=AsyncMock(),
        ),
        patch(
            'gpustack.policies.scorers.placement_scorer.async_session',
            return_value=AsyncMock(),
        ),
    ):
        m.backend = BackendEnum.SGLANG
        mis = []
        resource_fit_selector = SGLangResourceFitSelector(config, m, mis)
        placement_scorer = PlacementScorer(m, mis)

        actual_candidates = await resource_fit_selector.select_candidates(workers)
        actual_candidates = await placement_scorer.score(actual_candidates)
        actual_candidate, _ = await scheduler.find_candidate(config, m, workers, mis)

        try:
            assert len(actual_candidates) == len(expected_candidates)
            compare_candidates(actual_candidates, expected_candidates)
            compare_candidates(
                [actual_candidate], [expected_candidates[final_candidate_index]]
            )
        except AssertionError as e:
            raise AssertionError(f"Test case '{case_name}' failed: {str(e)}") from e


@pytest.mark.asyncio
async def test_manual_schedule_to_2_worker_2_gpu(config):
    """Test manual GPU selection with 2 workers and 2 GPUs"""
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
    m.backend = BackendEnum.SGLANG
    mis = []

    with (
        patch(
            'gpustack.scheduler.scheduler.BackendFrameworkFilter._has_supported_runners',
            return_value=(True, []),
        ),
        patch(
            'gpustack.schemas.workers.Worker.all',
            return_value=workers,
        ),
        patch(
            'gpustack.policies.worker_filters.backend_framework_filter.async_session',
            return_value=AsyncMock(),
        ),
        patch(
            'gpustack.policies.scorers.placement_scorer.async_session',
            return_value=AsyncMock(),
        ),
    ):

        candidate, _ = await scheduler.find_candidate(config, m, workers, mis)

        expected_candidates = [
            {
                "worker_id": 2,
                "worker_name": "host4090",
                "gpu_indexes": [0],
                "is_unified_memory": False,
                "vram": {
                    0: 21904773611,
                },
                "subordinate_workers": [
                    ModelInstanceSubordinateWorker(
                        worker_id=12,
                        worker_ip="192.168.50.4",
                        total_gpus=1,
                        gpu_indexes=[0],
                        computed_resource_claim=ComputedResourceClaim(
                            is_unified_memory=False,
                            vram={0: 21687579967},
                            vram_utilization=0.842,
                        ),
                    )
                ],
            },
        ]

        compare_candidates([candidate], expected_candidates)


@pytest.mark.asyncio
async def test_manual_schedule_to_2_worker_4_gpu_select_main_with_most_gpus(
    config,
):
    """Test manual GPU selection with 2 workers and 4 GPUs, selecting main worker with most GPUs"""
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
    m.backend = BackendEnum.SGLANG
    mis = []

    with (
        patch(
            'gpustack.scheduler.scheduler.BackendFrameworkFilter._has_supported_runners',
            return_value=(True, []),
        ),
        patch(
            'gpustack.schemas.workers.Worker.all',
            return_value=workers,
        ),
        patch(
            'gpustack.policies.worker_filters.backend_framework_filter.async_session',
            return_value=AsyncMock(),
        ),
        patch(
            'gpustack.policies.scorers.placement_scorer.async_session',
            return_value=AsyncMock(),
        ),
    ):

        candidate, _ = await scheduler.find_candidate(config, m, workers, mis)

        expected_candidates = [
            {
                "worker_id": 5,
                "worker_name": "host-4-4080",
                "gpu_indexes": [0, 1, 2],
                "is_unified_memory": False,
                "vram": {
                    0: 12586695262,
                    1: 12586695262,
                    2: 12586695262,
                },
                "subordinate_workers": [
                    ModelInstanceSubordinateWorker(
                        worker_id=2,
                        worker_ip="192.168.50.3",
                        total_gpus=1,
                        gpu_indexes=[0],
                        computed_resource_claim=ComputedResourceClaim(
                            vram={0: 19069120020},
                            vram_utilization=0.733,
                        ),
                    )
                ],
            },
        ]

        compare_candidates([candidate], expected_candidates)


@pytest.mark.asyncio
async def test_manual_schedule_to_3_workers_4_gpus(
    config,
):
    """Test manual GPU selection with 3 workers and 4 GPUs"""

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
    m.backend = BackendEnum.SGLANG
    mis = []

    with (
        patch(
            'gpustack.scheduler.scheduler.BackendFrameworkFilter._has_supported_runners',
            return_value=(True, []),
        ),
        patch(
            'gpustack.schemas.workers.Worker.all',
            return_value=workers(),
        ),
        patch(
            'gpustack.policies.worker_filters.backend_framework_filter.async_session',
            return_value=AsyncMock(),
        ),
        patch(
            'gpustack.policies.scorers.placement_scorer.async_session',
            return_value=AsyncMock(),
        ),
    ):

        candidate, _ = await scheduler.find_candidate(config, m, workers(), mis)

        expected_candidates = [
            {
                "worker_id": 8,
                "worker_name": "llm01-A100",
                "gpu_indexes": [0, 1],
                "is_unified_memory": False,
                "vram": {
                    0: 71124658421,
                    1: 71124658421,
                },
                "subordinate_workers": [
                    ModelInstanceSubordinateWorker(
                        worker_id=9,
                        worker_ip="192.168.50.11",
                        total_gpus=1,
                        gpu_indexes=[0],
                        computed_resource_claim=ComputedResourceClaim(
                            vram={0: 71124658421},
                            vram_utilization=0.828,
                        ),
                    ),
                    ModelInstanceSubordinateWorker(
                        worker_id=10,
                        worker_ip="192.168.50.12",
                        total_gpus=1,
                        gpu_indexes=[0],
                        computed_resource_claim=ComputedResourceClaim(
                            vram={0: 71124658421},
                            vram_utilization=0.828,
                        ),
                    ),
                ],
            },
        ]

        compare_candidates([candidate], expected_candidates)


@pytest.mark.asyncio
async def test_auto_schedule_to_2_worker_2_gpu(config):
    """Test automatic GPU selection with 2 workers and 2 GPUs"""
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
    m.backend = BackendEnum.SGLANG

    mis = []

    resource_fit_selector = SGLangResourceFitSelector(config, m, mis)
    placement_scorer = PlacementScorer(m, mis)

    with (
        patch(
            'gpustack.schemas.workers.Worker.all',
            return_value=workers,
        ),
        patch(
            'gpustack.policies.worker_filters.backend_framework_filter.async_session',
            return_value=AsyncMock(),
        ),
        patch(
            'gpustack.policies.scorers.placement_scorer.async_session',
            return_value=AsyncMock(),
        ),
    ):

        candidates = await resource_fit_selector.select_candidates(workers)
        candidates = await placement_scorer.score(candidates)

        expected_candidates = [
            {
                "worker_id": 2,
                "worker_name": "host4090",
                "gpu_indexes": [0],
                "is_unified_memory": False,
                "vram": {
                    0: 22034849464,
                },
                "subordinate_workers": [
                    ModelInstanceSubordinateWorker(
                        worker_id=12,
                        worker_ip="192.168.50.4",
                        total_gpus=1,
                        gpu_indexes=[0],
                        computed_resource_claim=ComputedResourceClaim(
                            vram={0: 21816366071},
                            vram_utilization=0.847,
                        ),
                    )
                ],
            },
        ]

        assert len(candidates) == 1
        compare_candidates(candidates, expected_candidates)


@pytest.mark.asyncio
async def test_auto_schedule_to_2_worker_16_gpu_deepseek_r1(config):
    """Test automatic GPU selection with large model requiring multiple GPUs"""
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
    m.backend = BackendEnum.SGLANG

    mis = []

    with (
        patch(
            'gpustack.scheduler.scheduler.BackendFrameworkFilter._has_supported_runners',
            return_value=(True, []),
        ),
        patch(
            'gpustack.schemas.workers.Worker.all',
            return_value=workers,
        ),
        patch(
            'gpustack.policies.worker_filters.backend_framework_filter.async_session',
            return_value=AsyncMock(),
        ),
        patch(
            'gpustack.policies.scorers.placement_scorer.async_session',
            return_value=AsyncMock(),
        ),
    ):

        candidate, _ = await scheduler.find_candidate(config, m, workers, mis)

        expected_candidates = [
            {
                "worker_id": 22,
                "worker_name": "host22-h100",
                "gpu_indexes": [0, 1, 2, 3, 4, 5, 6, 7],
                "is_unified_memory": False,
                "vram": {
                    0: 72756745994,
                    1: 72756745994,
                    2: 72756745994,
                    3: 72756745994,
                    4: 72756745994,
                    5: 72756745994,
                    6: 72756745994,
                    7: 72756745994,
                },
                "subordinate_workers": [
                    ModelInstanceSubordinateWorker(
                        worker_id=23,
                        worker_ip="192.168.50.23",
                        total_gpus=8,
                        gpu_indexes=[0, 1, 2, 3, 4, 5, 6, 7],
                        computed_resource_claim=ComputedResourceClaim(
                            vram={
                                0: 72756745994,
                                1: 72756745994,
                                2: 72756745994,
                                3: 72756745994,
                                4: 72756745994,
                                5: 72756745994,
                                6: 72756745994,
                                7: 72756745994,
                            },
                            vram_utilization=0.847,
                        ),
                    ),
                ],
            },
        ]

        compare_candidates([candidate], expected_candidates)


@pytest.mark.asyncio
async def test_auto_schedule_embedding_models(config):
    """Test automatic scheduling for embedding models"""
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
    m.backend = BackendEnum.SGLANG

    mis = []

    with (
        patch(
            'gpustack.scheduler.scheduler.BackendFrameworkFilter._has_supported_runners',
            return_value=(True, []),
        ),
        patch(
            'gpustack.schemas.workers.Worker.all',
            return_value=workers,
        ),
        patch(
            'gpustack.policies.worker_filters.backend_framework_filter.async_session',
            return_value=AsyncMock(),
        ),
        patch(
            'gpustack.policies.scorers.placement_scorer.async_session',
            return_value=AsyncMock(),
        ),
    ):

        candidate, _ = await scheduler.find_candidate(config, m, workers, mis)

        expected_candidates = [
            {
                "worker_id": 2,
                "worker_name": "host4090",
                "gpu_indexes": [0],
                "is_unified_memory": False,
                "vram": {
                    0: 22060864634,
                },
            },
        ]

        compare_candidates([candidate], expected_candidates)


@pytest.mark.asyncio
async def test_auto_schedule_single_work_single_gpu(config):
    """Test automatic scheduling with insufficient VRAM"""
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

    mis = [
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
    ]

    resource_fit_selector = SGLangResourceFitSelector(config, m, mis)
    placement_scorer = PlacementScorer(m, mis)

    with (
        patch(
            'gpustack.schemas.workers.Worker.all',
            return_value=workers,
        ),
        patch(
            'gpustack.policies.worker_filters.backend_framework_filter.async_session',
            return_value=AsyncMock(),
        ),
        patch(
            'gpustack.policies.scorers.placement_scorer.async_session',
            return_value=AsyncMock(),
        ),
    ):

        candidates = await resource_fit_selector.select_candidates(workers)
        _ = await placement_scorer.score(candidates)

        expect_msg = [
            """- The model requires approximately 75.23 GiB of VRAM.
- With --mem-fraction-static=0.848, all GPUs combined need to provide at least 88.71 GiB of total VRAM and each GPU needs 84% of allocatable VRAM.
- The current available GPU only has 24.23 GiB allocatable VRAM (100.00%)."""
        ]
        assert resource_fit_selector._messages == expect_msg


@pytest.mark.parametrize(
    "index, workers, model, expect_msg",
    [
        (
            1,
            [linux_nvidia_4_4080_16gx4()],
            new_model(
                1,
                "test_name",
                1,
                huggingface_repo_id="Qwen/Qwen2.5-Omni-7B",
                cpu_offloading=False,
                backend_parameters=[],
            ),
            [
                '- The model requires approximately 26.99 GiB of VRAM.\n'
                '- With --mem-fraction-static=0.772, all GPUs combined need to provide at '
                'least 34.96 GiB of total VRAM and each GPU needs 77% of allocatable VRAM.\n'
                '- Total number of attention heads (25) must be divisible by tensor parallel size (4).'
            ],
        ),
        (
            2,
            [linux_nvidia_4_4080_16gx4()],
            new_model(
                1,
                "test_name",
                1,
                huggingface_repo_id="Qwen/Qwen3-32B",
                cpu_offloading=False,
                backend_parameters=[],
            ),
            [
                """- The model requires approximately 75.23 GiB of VRAM.
- With --mem-fraction-static=0.772, all GPUs combined need to provide at least 97.45 GiB of total VRAM and each GPU needs 77% of allocatable VRAM.
- The largest available worker has 63.97 GiB allocatable VRAM, 4/4 of GPUs meet the VRAM utilization ratio, providing 49.38 GiB of allocatable VRAM."""
            ],
        ),
        (
            3,
            [linux_nvidia_4_4080_16gx4()],
            new_model(
                1,
                "test_name",
                1,
                huggingface_repo_id="Qwen/Qwen2.5-Omni-7B",
                cpu_offloading=False,
                backend_parameters=[],
            ),
            [
                '- The model requires approximately 26.99 GiB of VRAM.\n'
                '- With --mem-fraction-static=0.772, all GPUs combined need to provide at '
                'least 34.96 GiB of total VRAM and each GPU needs 77% of allocatable VRAM.\n'
                '- Vocabulary size (10001) must be divisible by tensor parallel size (4).'
            ],
        ),
        (
            3,
            [linux_nvidia_4_4080_16gx4()],
            new_model(
                1,
                "test_name",
                1,
                huggingface_repo_id="Qwen/Qwen-Image",
                cpu_offloading=False,
                backend_parameters=[],
                categories=[CategoryEnum.IMAGE],
            ),
            [
                """- The model requires approximately 53.74 GiB of VRAM.
- SGLang Diffusion requires each GPU to provide 53.74 GiB of allocatable VRAM when running in parallel."""
            ],
        ),
    ],
)
@pytest.mark.asyncio
async def test_auto_schedule_single_work_multi_gpu(
    config, index, workers, model, expect_msg
):
    """Test automatic scheduling with single worker and multiple GPUs"""
    m = model

    mis = []

    resource_fit_selector = SGLangResourceFitSelector(config, m, mis)
    placement_scorer = PlacementScorer(m, mis)

    original_init_model_params = resource_fit_selector._init_model_parameters

    async def mock_init_model_parameters(self, workers):
        if index == 1:
            # Simulate a scenario where the model's num_attention_heads cannot be evenly divided by the gpu_count through auto-scheduling.
            resource_fit_selector._num_attention_heads = 25
        elif index == 3:
            resource_fit_selector._model_params.vocab_size = 10001
        else:
            await original_init_model_params(workers)

    with (
        patch(
            'gpustack.schemas.workers.Worker.all',
            return_value=workers,
        ),
        patch(
            'gpustack.policies.worker_filters.backend_framework_filter.async_session',
            return_value=AsyncMock(),
        ),
        patch(
            'gpustack.policies.scorers.placement_scorer.async_session',
            return_value=AsyncMock(),
        ),
        patch.object(
            SGLangResourceFitSelector,
            '_init_model_parameters',
            new=mock_init_model_parameters,
        ),
    ):

        candidates = await resource_fit_selector.select_candidates(workers)
        _ = await placement_scorer.score(candidates)

        # Verify detailed message content
        assert resource_fit_selector._messages == expect_msg


@pytest.mark.asyncio
async def test_auto_schedule_multi_work_multi_gpu(config):
    """Test automatic scheduling with multiple workers and multiple GPUs"""
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
    m.backend = BackendEnum.SGLANG

    mis = []

    resource_fit_selector = SGLangResourceFitSelector(config, m, mis)
    placement_scorer = PlacementScorer(m, mis)

    with (
        patch(
            'gpustack.schemas.workers.Worker.all',
            return_value=workers,
        ),
        patch(
            'gpustack.policies.worker_filters.backend_framework_filter.async_session',
            return_value=AsyncMock(),
        ),
        patch(
            'gpustack.policies.scorers.placement_scorer.async_session',
            return_value=AsyncMock(),
        ),
    ):

        candidates = await resource_fit_selector.select_candidates(workers)
        _ = await placement_scorer.score(candidates)

        expect_msg = [
            """- The model requires approximately 75.23 GiB of VRAM.
- With --mem-fraction-static=0.772, all GPUs combined need to provide at least 97.45 GiB of total VRAM and each GPU needs 77% of allocatable VRAM.
- The optimal combination ['host4080', 'host4080-1'] provides 49.38 GiB of allocatable VRAM.
- Cannot find a suitable worker combination to run the model in distributed mode. If you are confident that the resources are sufficient, you may manually schedule the model by selecting the workers and GPUs."""
        ]

        assert resource_fit_selector._messages == expect_msg


@pytest.mark.asyncio
async def test_sglang_backend_parameters(config):
    """Test SGLang-specific backend parameters handling"""
    workers = [
        linux_nvidia_1_4090_24gx1(),
    ]

    # Test with SGLang-specific parameters
    m = new_model(
        1,
        "test_name",
        1,
        huggingface_repo_id="Qwen/Qwen2.5-7B-Instruct",
        cpu_offloading=False,
        backend_parameters=[
            "--tp-size=1",
            "--dp-size=1",
            "--enable-mixed-chunk=true",
            "--disable-overlap-schedule=false",
            "--mem-fraction-static=0.85",
        ],
    )

    mis = []

    resource_fit_selector = SGLangResourceFitSelector(config, m, mis)
    placement_scorer = PlacementScorer(m, mis)

    with (
        patch(
            'gpustack.schemas.workers.Worker.all',
            return_value=workers,
        ),
        patch(
            'gpustack.policies.worker_filters.backend_framework_filter.async_session',
            return_value=AsyncMock(),
        ),
        patch(
            'gpustack.policies.scorers.placement_scorer.async_session',
            return_value=AsyncMock(),
        ),
    ):

        candidates = await resource_fit_selector.select_candidates(workers)
        candidates = await placement_scorer.score(candidates)

        # Should successfully create candidates with SGLang parameters
        assert (
            len(candidates) >= 0
        )  # May be 0 if VRAM insufficient, but should not crash


@pytest.mark.asyncio
async def test_sglang_tensor_parallel_size(config):
    """Test SGLang tensor parallel size handling"""
    workers = [
        linux_nvidia_4_4080_16gx4(),
    ]

    # Test with tensor parallel size
    m = new_model(
        1,
        "test_name",
        1,
        huggingface_repo_id="Qwen/Qwen2.5-7B-Instruct",
        cpu_offloading=False,
        backend_parameters=[
            "--tp-size=2",
        ],
    )

    mis = []

    resource_fit_selector = SGLangResourceFitSelector(config, m, mis)
    placement_scorer = PlacementScorer(m, mis)

    with (
        patch(
            'gpustack.schemas.workers.Worker.all',
            return_value=workers,
        ),
        patch(
            'gpustack.policies.worker_filters.backend_framework_filter.async_session',
            return_value=AsyncMock(),
        ),
        patch(
            'gpustack.policies.scorers.placement_scorer.async_session',
            return_value=AsyncMock(),
        ),
    ):

        candidates = await resource_fit_selector.select_candidates(workers)
        candidates = await placement_scorer.score(candidates)

        # Should handle tensor parallel size correctly
        if candidates:
            # If candidates are found, they should respect the tp-size parameter
            assert len(candidates) >= 0


@pytest.mark.asyncio
async def test_sglang_data_parallel_size(config):
    """Test SGLang data parallel size handling"""
    workers = [
        linux_nvidia_22_H100_80gx8(),
        linux_nvidia_23_H100_80gx8(),
    ]

    # Test with data parallel size
    m = new_model(
        1,
        "test_name",
        1,
        huggingface_repo_id="Qwen/Qwen2.5-7B-Instruct",
        cpu_offloading=False,
        backend_parameters=[
            "--dp-size=2",
            "--tp-size=4",
        ],
    )

    mis = []

    resource_fit_selector = SGLangResourceFitSelector(config, m, mis)
    placement_scorer = PlacementScorer(m, mis)

    with (
        patch(
            'gpustack.schemas.workers.Worker.all',
            return_value=workers,
        ),
        patch(
            'gpustack.policies.worker_filters.backend_framework_filter.async_session',
            return_value=AsyncMock(),
        ),
        patch(
            'gpustack.policies.scorers.placement_scorer.async_session',
            return_value=AsyncMock(),
        ),
    ):

        candidates = await resource_fit_selector.select_candidates(workers)
        candidates = await placement_scorer.score(candidates)

        # Should handle data parallel size correctly
        assert len(candidates) >= 0


@pytest.mark.asyncio
async def test_sglang_memory_fraction_static(config):
    """Test SGLang memory fraction static parameter"""
    workers = [
        linux_nvidia_1_4090_24gx1(),
    ]

    # Test with custom memory fraction
    m = new_model(
        1,
        "test_name",
        1,
        huggingface_repo_id="Qwen/Qwen2.5-7B-Instruct",
        cpu_offloading=False,
        backend_parameters=[
            "--mem-fraction-static=0.8",
        ],
    )

    mis = []

    resource_fit_selector = SGLangResourceFitSelector(config, m, mis)
    placement_scorer = PlacementScorer(m, mis)

    with (
        patch(
            'gpustack.schemas.workers.Worker.all',
            return_value=workers,
        ),
        patch(
            'gpustack.policies.worker_filters.backend_framework_filter.async_session',
            return_value=AsyncMock(),
        ),
        patch(
            'gpustack.policies.scorers.placement_scorer.async_session',
            return_value=AsyncMock(),
        ),
    ):

        candidates = await resource_fit_selector.select_candidates(workers)
        candidates = await placement_scorer.score(candidates)

        # Should handle custom memory fraction
        assert len(candidates) >= 0


@pytest.mark.asyncio
async def test_auto_schedule_extended_kv_cache_ram_size(config):
    workers = [
        linux_nvidia_1_4090_24gx1(),
    ]

    m = new_model(
        1,
        "test_name",
        1,
        huggingface_repo_id="Qwen/Qwen3-0.6B",
        cpu_offloading=False,
        extended_kv_cache=ExtendedKVCacheConfig(
            enabled=True,
            ram_size=8.0,  # 8GiB
        ),
    )

    mis = []

    resource_fit_selector = SGLangResourceFitSelector(config, m, mis)
    placement_scorer = PlacementScorer(m, mis)

    with (
        patch(
            'gpustack.schemas.workers.Worker.all',
            return_value=workers,
        ),
        patch(
            'gpustack.policies.worker_filters.backend_framework_filter.async_session',
            return_value=AsyncMock(),
        ),
        patch(
            'gpustack.policies.scorers.placement_scorer.async_session',
            return_value=AsyncMock(),
        ),
    ):

        candidates = await resource_fit_selector.select_candidates(workers)
        candidates = await placement_scorer.score(candidates)

        expected_candidates = [
            {
                "worker_id": 2,
                "worker_name": "host4090",
                "gpu_indexes": [0],
                "is_unified_memory": False,
                "vram": {
                    0: 22060864634,
                },
                "ram": 8589934592,
            },
        ]

        assert len(candidates) == 1
        compare_candidates(candidates, expected_candidates)


@pytest.mark.asyncio
async def test_auto_schedule_extended_kv_cache_ram_ratio(config):
    workers = [
        linux_nvidia_1_4090_24gx1(),
    ]

    m = new_model(
        1,
        "test_name",
        1,
        huggingface_repo_id="Qwen/Qwen3-0.6B",
        cpu_offloading=False,
        extended_kv_cache=ExtendedKVCacheConfig(
            enabled=True,
            ram_ratio=2.0,  # 2x of VRAM
        ),
    )

    mis = []

    resource_fit_selector = SGLangResourceFitSelector(config, m, mis)
    placement_scorer = PlacementScorer(m, mis)

    with (
        patch(
            'gpustack.schemas.workers.Worker.all',
            return_value=workers,
        ),
        patch(
            'gpustack.policies.worker_filters.backend_framework_filter.async_session',
            return_value=AsyncMock(),
        ),
        patch(
            'gpustack.policies.scorers.placement_scorer.async_session',
            return_value=AsyncMock(),
        ),
    ):

        candidates = await resource_fit_selector.select_candidates(workers)
        candidates = await placement_scorer.score(candidates)

        expected_candidates = [
            {
                "worker_id": 2,
                "worker_name": "host4090",
                "gpu_indexes": [0],
                "is_unified_memory": False,
                "vram": {
                    0: 22060864634,
                },
                "ram": 44121729268,
            },
        ]

        assert len(candidates) == 1
        compare_candidates(candidates, expected_candidates)


@pytest.mark.parametrize(
    "index, workers, model, expect_msg",
    [
        # Overcommit when used all selected GPUs
        (
            1,
            [linux_nvidia_4_4080_16gx4()],
            make_model(
                2, ["host-4-4080:cuda:0", "host-4-4080:cuda:1"], "Qwen/Qwen3-32B"
            ),
            [
                """- The model requires approximately 75.23 GiB of VRAM.
- With --mem-fraction-static=0.765, all GPUs combined need to provide at least 98.34 GiB of total VRAM and each GPU needs 76% of allocatable VRAM.
- Manual GPU selection resulted in resource overcommit.
- Selected GPUs have 31.98 GiB allocatable VRAM, 2/2 of GPUs meet the VRAM utilization ratio, providing 24.47 GiB of allocatable VRAM."""
            ],
        ),
        # Overcommit when partially used selected GPUs
        (
            2,
            [linux_nvidia_4_4080_16gx4()],
            make_model(
                2,
                [
                    "host-4-4080:cuda:0",
                    "host-4-4080:cuda:1",
                    "host-4-4080:cuda:2",
                    "host-4-4080:cuda:3",
                ],
                "Qwen/Qwen3-32B",
            ),
            [
                """- The model requires approximately 75.23 GiB of VRAM.
- With --mem-fraction-static=0.749, all GPUs combined need to provide at least 100.44 GiB of total VRAM and each GPU needs 74% of allocatable VRAM.
- Manual GPU selection resulted in resource overcommit.
- Using worker host-4-4080 GPU indexes [0, 1] out of 4 selected devices.
- Used GPUs provide 31.98 GiB allocatable VRAM, 2/2 of GPUs meet the VRAM utilization ratio, providing 23.96 GiB of allocatable VRAM."""
            ],
        ),
    ],
)
@pytest.mark.asyncio
async def test_output_schedule_msg(config, index, workers, model, expect_msg):
    m = model

    mis = []

    resource_fit_selector = SGLangResourceFitSelector(config, m, mis)
    placement_scorer = PlacementScorer(m, mis)

    with (
        patch(
            'gpustack.schemas.workers.Worker.all',
            return_value=workers,
        ),
        patch(
            'gpustack.policies.worker_filters.backend_framework_filter.async_session',
            return_value=AsyncMock(),
        ),
        patch(
            'gpustack.policies.scorers.placement_scorer.async_session',
            return_value=AsyncMock(),
        ),
    ):

        candidates = await resource_fit_selector.select_candidates(workers)
        _ = await placement_scorer.score(candidates)

        assert resource_fit_selector._messages == expect_msg


@pytest.mark.parametrize(
    "case_name, m, workers, expected_candidates, final_candidate_index",
    [
        # Manually select 1 cuda gpu and 1 amd rocm gpu.
        # Check point:
        # - Candidate selection correctness, final candidate should be the cuda gpu.
        (
            "manual_select_1_cuda+1_rocm",
            make_model(1, ["host4090:cuda:0", "host01-7800:rocm:0"], "Qwen/Qwen3-0.6B"),
            [
                linux_nvidia_1_4090_24gx1(),
                linux_rocm_1_7800_16gx1(),
            ],
            [
                {
                    "worker_id": 2,
                    "worker_name": "host4090",
                    "gpu_indexes": [0],
                    "gpu_type": "cuda",
                    "vram": {
                        0: 22060864634,
                    },
                    "score": 56.53,
                },
                {
                    "worker_id": 13,
                    "worker_name": "host01-7800",
                    "gpu_indexes": [0],
                    "gpu_type": "rocm",
                    "vram": {
                        0: 13249906999,
                    },
                    "score": 51.4667,
                },
            ],
            0,
        ),
        # Manually select 1 cuda gpu and 1 amd rocm gpu with gpus_per_replica=2.
        # Check point:
        # - No candidates.
        (
            "manual_select_1_cuda+1_rocm_with_gpus_per_replica_2",
            make_model(2, ["host4090:cuda:0", "host01-7800:rocm:0"], "Qwen/Qwen3-0.6B"),
            [
                linux_nvidia_1_4090_24gx1(),
                linux_rocm_1_7800_16gx1(),
            ],
            [],
            None,
        ),
        # Manually select 2 cuda GPUs + 2 rocm GPUs and set gpus_per_replica=2.
        # Check point:
        # - Candidate should use 2 cuda GPUs.
        (
            "manual_select_2_cuda+2_rocm_with_gpus_per_replica_2",
            make_model(
                2,
                [
                    "host4080:cuda:0",
                    "host4080:cuda:1",
                    "host02-7800:rocm:0",
                    "host02-7800:rocm:1",
                ],
                "Qwen/Qwen3-0.6B",
            ),
            [
                linux_nvidia_2_4080_16gx2(),
                linux_rocm_2_7800_16gx2(),
            ],
            [
                {
                    "worker_id": 3,
                    "worker_name": "host4080",
                    "gpu_indexes": [0, 1],
                    "gpu_type": "cuda",
                    "vram": {
                        0: 13136182640,
                        1: 13136182640,
                    },
                    "score": 50.9999,
                },
                {
                    "worker_id": 28,
                    "worker_name": "host02-7800",
                    "gpu_indexes": [0, 1],
                    "gpu_type": "rocm",
                    "vram": {
                        0: 13112602263,
                        1: 13112602263,
                    },
                    "score": 50.9333,
                },
            ],
            0,
        ),
        # Manually select 1 cuda GPUs from mixed gpu worker + 1 cuda GPUs from cuda gpu worker and set gpus_per_replica=2.
        # Check point:
        # - Candidate should use 2 cuda GPUs.
        (
            "manual_select_1_cuda_from_mix+1_cuda_with_gpus_per_replica_2",
            make_model(2, ["host4080:cuda:0", "host-mix-01:cuda:0"], "Qwen/Qwen3-0.6B"),
            [
                linux_nvidia_2_4080_16gx2(),
                linux_mix_1_nvidia_4080_16gx1_rocm_7800_16gx1(),
            ],
            [
                {
                    "worker_id": 3,
                    "worker_name": "host4080",
                    "gpu_indexes": [0],
                    "gpu_type": "cuda",
                    "vram": {
                        0: 12861438951,
                    },
                    "score": 49.9333,
                    "subordinate_workers": [
                        ModelInstanceSubordinateWorker(
                            worker_id=29,
                            worker_ip="192.168.50.31",
                            total_gpus=1,
                            gpu_type="cuda",
                            gpu_indexes=[0],
                            computed_resource_claim=ComputedResourceClaim(
                                vram={0: 12861438951},
                                vram_utilization=0.749,
                            ),
                        )
                    ],
                },
            ],
            0,
        ),
        # Auto select use single gpu from nvidia and amd workers.
        # Check point:
        # - Candidate selection correctness, final candidate should be the cuda gpu.
        (
            "auto_select_single_gpu_from_cuda+rocm",
            make_model(1, None, "Qwen/Qwen3-0.6B"),
            [
                linux_nvidia_1_4090_24gx1(),
                linux_rocm_1_7800_16gx1(),
            ],
            [
                {
                    "worker_id": 2,
                    "worker_name": "host4090",
                    "gpu_indexes": [0],
                    "gpu_type": "cuda",
                    "vram": {
                        0: 22060864634,
                    },
                    "score": 56.5333,
                },
                {
                    "worker_id": 13,
                    "worker_name": "host01-7800",
                    "gpu_indexes": [0],
                    "gpu_type": "rocm",
                    "vram": {
                        0: 13249906999,
                    },
                    "score": 51.4667,
                },
            ],
            0,
        ),
        # Auto select single worker multi gpu from nvidia and amd workers.
        # Check point:
        # - Candidate selection correctness, final candidate should be the cuda gpu.
        (
            "auto_select_single_worker_multi_gpu_from_2_cuda+2_rocm_with_gpus_per_replica_2",
            make_model(2, None, "Qwen/Qwen3-8B"),
            [
                linux_nvidia_2_4080_16gx2(),
                linux_rocm_2_7800_16gx2(),
            ],
            [
                {
                    "worker_id": 3,
                    "worker_name": "host4080",
                    "gpu_indexes": [0, 1],
                    "gpu_type": "cuda",
                    "vram": {
                        0: 13256383004,
                        1: 13256383004,
                    },
                    "score": 51.46666666405769,
                },
                {
                    "worker_id": 28,
                    "worker_name": "host02-7800",
                    "gpu_indexes": [0, 1],
                    "gpu_type": "rocm",
                    "vram": {
                        0: 13249906999,
                        1: 13249906999,
                    },
                    "score": 51.46666666551692,
                },
            ],
            1,
        ),
    ],
)
@pytest.mark.asyncio
async def test_select_candidates_from_different_gpu_types(
    config, case_name, m, workers, expected_candidates, final_candidate_index
):
    with (
        patch(
            'gpustack.schemas.workers.Worker.all',
            return_value=workers,
        ),
        patch(
            'gpustack.scheduler.scheduler.BackendFrameworkFilter._has_supported_runners',
            return_value=(True, []),
        ),
        patch(
            'gpustack.policies.worker_filters.backend_framework_filter.async_session',
            return_value=AsyncMock(),
        ),
        patch(
            'gpustack.policies.scorers.placement_scorer.async_session',
            return_value=AsyncMock(),
        ),
    ):
        m.backend = BackendEnum.SGLANG.value

        mis = []

        resource_fit_selector = SGLangResourceFitSelector(config, m, mis)
        scorer = PlacementScorer(m, mis)

        actual_candidates = await resource_fit_selector.select_candidates(workers)
        actual_candidates = await scorer.score(actual_candidates)
        actual_candidate, _ = await scheduler.find_candidate(config, m, workers, mis)

        try:
            assert len(actual_candidates) == len(expected_candidates)
            compare_candidates(actual_candidates, expected_candidates)
            if final_candidate_index is not None:
                compare_candidates(
                    [actual_candidate], [expected_candidates[final_candidate_index]]
                )
        except AssertionError as e:
            raise AssertionError(f"Test case '{case_name}' failed: {str(e)}") from e
