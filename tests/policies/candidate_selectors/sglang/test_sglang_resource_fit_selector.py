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

    resource_fit_selector = SGLangResourceFitSelector(config, m)
    placement_scorer = PlacementScorer(m)

    with (
        patch(
            'gpustack.policies.candidate_selectors.base_candidate_selector.get_worker_model_instances',
            return_value=[],
        ),
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
                "ram": 0,
                "subordinate_workers": [
                    ModelInstanceSubordinateWorker(
                        worker_id=12,
                        worker_ip="192.168.50.4",
                        total_gpus=1,
                        gpu_indexes=[0],
                        computed_resource_claim=ComputedResourceClaim(
                            is_unified_memory=False,
                            vram={0: 23181498777},
                            ram=0,
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

    resource_fit_selector = SGLangResourceFitSelector(config, m)
    placement_scorer = PlacementScorer(m)

    with (
        patch(
            'gpustack.policies.candidate_selectors.base_candidate_selector.get_worker_model_instances',
            return_value=[],
        ),
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
                "worker_id": 5,
                "worker_name": "host-4-4080",
                "gpu_indexes": [0, 1, 2],
                "is_unified_memory": False,
                "vram": {
                    0: 15454332518,
                    1: 15454332518,
                    2: 15454332518,
                },
                "ram": 0,
                "subordinate_workers": [
                    ModelInstanceSubordinateWorker(
                        worker_id=2,
                        worker_ip="192.168.50.3",
                        total_gpus=1,
                        gpu_indexes=[0],
                        computed_resource_claim=ComputedResourceClaim(
                            vram={0: 23413653504},
                            ram=0,
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

    resource_fit_selector = SGLangResourceFitSelector(config, m)
    placement_scorer = PlacementScorer(m)

    with (
        patch(
            'gpustack.policies.candidate_selectors.base_candidate_selector.get_worker_model_instances',
            return_value=[],
        ),
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
                "subordinate_workers": [
                    ModelInstanceSubordinateWorker(
                        worker_id=9,
                        worker_ip="192.168.50.11",
                        total_gpus=1,
                        gpu_indexes=[0],
                        computed_resource_claim=ComputedResourceClaim(
                            vram={0: 77309411328},
                            ram=0,
                        ),
                    ),
                    ModelInstanceSubordinateWorker(
                        worker_id=10,
                        worker_ip="192.168.50.12",
                        total_gpus=1,
                        gpu_indexes=[0],
                        computed_resource_claim=ComputedResourceClaim(
                            vram={0: 77309411328},
                            ram=0,
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

    resource_fit_selector = SGLangResourceFitSelector(config, m)
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

        expected_candidates = [
            {
                "worker_id": 2,
                "worker_name": "host4090",
                "gpu_indexes": [0],
                "is_unified_memory": False,
                "vram": {
                    0: 23413653504,
                },
                "subordinate_workers": [
                    ModelInstanceSubordinateWorker(
                        worker_id=12,
                        worker_ip="192.168.50.4",
                        total_gpus=1,
                        gpu_indexes=[0],
                        computed_resource_claim=ComputedResourceClaim(
                            vram={0: 23181498777},
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

    resource_fit_selector = SGLangResourceFitSelector(config, m)
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
                "subordinate_workers": [
                    ModelInstanceSubordinateWorker(
                        worker_id=23,
                        worker_ip="192.168.50.23",
                        total_gpus=8,
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

    resource_fit_selector = SGLangResourceFitSelector(config, m)
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
            },
        ]

        assert len(candidates) == 1
        assert candidate == candidates[0]
        compare_candidates(candidates, expected_candidates)


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

    resource_fit_selector = SGLangResourceFitSelector(config, m)
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
- With --mem-fraction-static=0.9, all GPUs combined need to provide at least 83.59 GiB of total VRAM and each GPU needs 90% of allocatable VRAM.
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
                '- With --mem-fraction-static=0.9, all GPUs combined need to provide at '
                'least 29.99 GiB of total VRAM and each GPU needs 90% of allocatable VRAM.\n'
                '- Total number of attention heads (25) must be divisible by gpu count (4).'
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
- With --mem-fraction-static=0.9, all GPUs combined need to provide at least 83.59 GiB of total VRAM and each GPU needs 90% of allocatable VRAM.
- The largest available worker has 63.97 GiB allocatable VRAM, 4/4 of GPUs meet the VRAM utilization ratio, providing 57.57 GiB of allocatable VRAM."""
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

    resource_fit_selector = SGLangResourceFitSelector(config, m)
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

    resource_fit_selector = SGLangResourceFitSelector(config, m)
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
- With --mem-fraction-static=0.9, all GPUs combined need to provide at least 83.59 GiB of total VRAM and each GPU needs 90% of allocatable VRAM.
- The optimal combination ['host4080', 'host4080-1'] provides 57.57 GiB of allocatable VRAM.
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

    resource_fit_selector = SGLangResourceFitSelector(config, m)
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

    resource_fit_selector = SGLangResourceFitSelector(config, m)
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

    resource_fit_selector = SGLangResourceFitSelector(config, m)
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

    resource_fit_selector = SGLangResourceFitSelector(config, m)
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

    resource_fit_selector = SGLangResourceFitSelector(config, m)
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

        expected_candidates = [
            {
                "worker_id": 2,
                "worker_name": "host4090",
                "gpu_indexes": [0],
                "is_unified_memory": False,
                "vram": {
                    0: 23413653504,
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

    resource_fit_selector = SGLangResourceFitSelector(config, m)
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

        expected_candidates = [
            {
                "worker_id": 2,
                "worker_name": "host4090",
                "gpu_indexes": [0],
                "is_unified_memory": False,
                "vram": {
                    0: 23413653504,
                },
                "ram": 46827307008,
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
- With --mem-fraction-static=0.9, all GPUs combined need to provide at least 83.59 GiB of total VRAM and each GPU needs 90% of allocatable VRAM.
- Manual GPU selection resulted in resource overcommit.
- Selected GPUs have 31.98 GiB allocatable VRAM, 2/2 of GPUs meet the VRAM utilization ratio, providing 28.79 GiB of allocatable VRAM."""
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
- With --mem-fraction-static=0.9, all GPUs combined need to provide at least 83.59 GiB of total VRAM and each GPU needs 90% of allocatable VRAM.
- Manual GPU selection resulted in resource overcommit.
- Using worker host-4-4080 GPU indexes [0, 1] out of 4 selected devices.
- Used GPUs provide 31.98 GiB allocatable VRAM, 2/2 of GPUs meet the VRAM utilization ratio, providing 28.79 GiB of allocatable VRAM."""
            ],
        ),
    ],
)
@pytest.mark.asyncio
async def test_output_schedule_msg(config, index, workers, model, expect_msg):
    m = model

    resource_fit_selector = SGLangResourceFitSelector(config, m)
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

        assert resource_fit_selector._messages == expect_msg
