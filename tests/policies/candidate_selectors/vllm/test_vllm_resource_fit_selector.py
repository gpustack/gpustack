from typing import List

import pytest
from unittest.mock import patch, AsyncMock

from gpustack.policies.utils import get_model_num_attention_heads
from tests.utils.model import new_model, new_model_instance
from gpustack.policies.candidate_selectors import VLLMResourceFitSelector
from gpustack.policies.scorers.placement_scorer import PlacementScorer
from gpustack.scheduler import scheduler
from gpustack.schemas.models import (
    CategoryEnum,
    ComputedResourceClaim,
    ExtendedKVCacheConfig,
    GPUSelector,
    ModelInstanceStateEnum,
    ModelInstance,
    ModelInstanceSubordinateWorker,
    BackendEnum,
)
from tests.fixtures.workers.fixtures import (
    linux_nvidia_0_4090_24gx1,
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


def make_model(
    gpus_per_replica=2, gpu_ids=None, repo_id="Qwen/Qwen2.5-7B-Instruct", **kwargs
):
    gpu_selector = None
    if gpu_ids is not None:
        gpu_selector = GPUSelector(
            gpu_ids=gpu_ids,
            gpus_per_replica=gpus_per_replica,
        )

    return new_model(
        1,
        "test_name",
        1,
        huggingface_repo_id=repo_id,
        gpu_selector=gpu_selector,
        **kwargs,
    )


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


@pytest.mark.parametrize(
    "m, workers, expected_candidates, final_candidate_index",
    [
        # Manually select two GPUs from 2 worker.
        # Check point:
        # - Candidate selection correctness.
        (
            make_model(2, ["host4090:cuda:0", "host-2-4090:cuda:0"]),
            [
                linux_nvidia_1_4090_24gx1(),
                linux_nvidia_3_4090_24gx1(),
            ],
            [
                expected_candidate(
                    2,
                    "host4090",
                    [0],
                    {0: 23413653504},
                    [
                        ModelInstanceSubordinateWorker(
                            worker_id=12,
                            worker_ip="192.168.50.4",
                            total_gpus=1,
                            gpu_indexes=[0],
                            computed_resource_claim=ComputedResourceClaim(
                                is_unified_memory=False,
                                vram={0: 23181498777},
                            ),
                        )
                    ],
                )
            ],
            0,
        ),
        # Manually select 2 GPUs from 2 worker with gpus_per_replica=1.
        # Check point:
        # - Candidate should occupy 1 GPUs; 1 GPUs remain unused.
        (
            make_model(1, ["host4090:cuda:0", "host-2-4090:cuda:0"]),
            [
                linux_nvidia_1_4090_24gx1(),
                linux_nvidia_3_4090_24gx1(),
            ],
            [
                expected_candidate(2, "host4090", [0], {0: 23413653504}),
                expected_candidate(12, "host-2-4090", [0], {0: 23181498777}),
            ],
            1,
        ),
        # Manually select 3 GPUs from worker1 and 1 GPU from worker2.
        # Check point:
        # - Candidate selection correctness.
        # - Main inference worker should pick the worker with most GPUs.
        # - Subordinate worker assignment correctness.
        (
            make_model(
                4,
                [
                    "host4090:cuda:0",
                    "host-4-4080:cuda:0",
                    "host-4-4080:cuda:1",
                    "host-4-4080:cuda:2",
                ],
            ),
            [
                linux_nvidia_1_4090_24gx1(),
                linux_nvidia_4_4080_16gx4(),
            ],
            [
                expected_candidate(
                    5,
                    "host-4-4080",
                    [0, 1, 2],
                    {
                        0: 15454332518,
                        1: 15454332518,
                        2: 15454332518,
                    },
                    [
                        ModelInstanceSubordinateWorker(
                            worker_id=2,
                            worker_ip="192.168.50.3",
                            total_gpus=1,
                            gpu_indexes=[0],
                            computed_resource_claim=ComputedResourceClaim(
                                vram={0: 23413653504},
                            ),
                        )
                    ],
                )
            ],
            0,
        ),
        # Manually select 3 GPUs from worker1 and 1 GPU from worker2, set gpus_per_replica=2.
        # Check point:
        # - Candidate should occupy 2 GPUs; 2 GPUs remain unused.
        (
            make_model(
                2,
                [
                    "host4090:cuda:0",
                    "host-4-4080:cuda:0",
                    "host-4-4080:cuda:1",
                    "host-4-4080:cuda:2",
                ],
            ),
            [
                linux_nvidia_1_4090_24gx1(),
                linux_nvidia_4_4080_16gx4(),
            ],
            [
                expected_candidate(
                    5,
                    "host-4-4080",
                    [0, 1],
                    {
                        0: 15454332518,
                        1: 15454332518,
                    },
                )
            ],
            0,
        ),
        # Manually select 4 GPUs from 3 workers.
        # Check point:
        # - Candidate selection correctness.
        (
            make_model(
                4,
                [
                    "llm01-A100:cuda:0",
                    "llm01-A100:cuda:1",
                    "llm02-A100:cuda:0",
                    "llm03-A100:cuda:0",
                ],
            ),
            [
                linux_nvidia_5_a100_80gx2(),
                linux_nvidia_6_a100_80gx2(),
                linux_nvidia_7_a100_80gx2(),
            ],
            [
                expected_candidate(
                    8,
                    "llm01-A100",
                    [0, 1],
                    {
                        0: 77309411328,
                        1: 77309411328,
                    },
                    [
                        ModelInstanceSubordinateWorker(
                            worker_id=9,
                            worker_ip="192.168.50.11",
                            total_gpus=2,
                            gpu_indexes=[0],
                            computed_resource_claim=ComputedResourceClaim(
                                vram={0: 77309411328},
                            ),
                        ),
                        ModelInstanceSubordinateWorker(
                            worker_id=10,
                            worker_ip="192.168.50.12",
                            total_gpus=2,
                            gpu_indexes=[0],
                            computed_resource_claim=ComputedResourceClaim(
                                vram={0: 77309411328},
                            ),
                        ),
                    ],
                )
            ],
            0,
        ),
        # Auto schedule to 2 workers with 16 GPUs each for DeepSeek-R1 model.
        # Check point:
        # - Candidate selection correctness.
        (
            make_model(0, None, "deepseek-ai/DeepSeek-R1"),
            [
                linux_nvidia_1_4090_24gx1(),
                linux_nvidia_3_4090_24gx1(),
                linux_nvidia_22_H100_80gx8(),
                linux_nvidia_23_H100_80gx8(),
                linux_nvidia_24_H100_80gx8(),
                linux_nvidia_25_H100_80gx8(),
            ],
            [
                expected_candidate(
                    22,
                    "host22-h100",
                    [0, 1, 2, 3, 4, 5, 6, 7],
                    {
                        0: 77309411328,
                        1: 77309411328,
                        2: 77309411328,
                        3: 77309411328,
                        4: 77309411328,
                        5: 77309411328,
                        6: 77309411328,
                        7: 77309411328,
                    },
                    [
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
                )
            ],
            0,
        ),
        # Auto select 3 GPUs distributedly from 3 worker for PP3.
        # Check point:
        # - Candidate selection correctness.
        # - GPU count by PP size correctness.
        (
            make_model(
                0,
                None,
                "Qwen/Qwen3-0.6B",
                backend_parameters=["--pipeline-parallel-size=3"],
            ),
            [
                linux_nvidia_0_4090_24gx1(),
                linux_nvidia_1_4090_24gx1(),
                linux_nvidia_3_4090_24gx1(),
            ],
            [
                expected_candidate(
                    103,
                    "host4090-0",
                    [0],
                    {0: 23413653504},
                    [
                        ModelInstanceSubordinateWorker(
                            worker_id=2,
                            worker_ip="192.168.50.3",
                            total_gpus=1,
                            gpu_indexes=[0],
                            computed_resource_claim=ComputedResourceClaim(
                                is_unified_memory=False,
                                vram={0: 23413653504},
                            ),
                        ),
                        ModelInstanceSubordinateWorker(
                            worker_id=12,
                            worker_ip="192.168.50.4",
                            total_gpus=1,
                            gpu_indexes=[0],
                            computed_resource_claim=ComputedResourceClaim(
                                is_unified_memory=False,
                                vram={0: 23181498777},
                            ),
                        ),
                    ],
                )
            ],
            0,
        ),
        (
            new_model(
                1,
                "test_name",
                1,
                huggingface_repo_id="BAAI/bge-base-en-v1.5",
                cpu_offloading=False,
                backend_parameters=[],
                categories=[CategoryEnum.EMBEDDING],
            ),
            [linux_nvidia_1_4090_24gx1()],
            [expected_candidate(2, "host4090", [0], {0: 1588014354})],
            0,
        ),
        (
            # Auto schedule model with extended kv cache.
            # Check point:
            # - Candidate selection correctness.
            new_model(
                1,
                "test_name",
                1,
                huggingface_repo_id="Qwen/Qwen3-0.6B",
                cpu_offloading=False,
                extended_kv_cache=ExtendedKVCacheConfig(
                    enabled=True,
                    ram_size=8.0,  # 8GiB
                ),
            ),
            [linux_nvidia_1_4090_24gx1()],
            [expected_candidate(2, "host4090", [0], {0: 23413653504}, ram=8589934592)],
            0,
        ),
        (
            # Auto schedule model with extended kv cache and ram ratio.
            # Check point:
            # - Candidate selection correctness.
            new_model(
                1,
                "test_name",
                1,
                huggingface_repo_id="Qwen/Qwen3-0.6B",
                cpu_offloading=False,
                extended_kv_cache=ExtendedKVCacheConfig(
                    enabled=True,
                    ram_ratio=2.0,  # 2x of vram
                ),
            ),
            [linux_nvidia_1_4090_24gx1()],
            [expected_candidate(2, "host4090", [0], {0: 23413653504}, ram=46827307008)],
            0,
        ),
    ],
)
@pytest.mark.asyncio
async def test_select_candidates_2x_4090_24gx1(
    config, m, workers, expected_candidates, final_candidate_index
):
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
        m.backend = BackendEnum.VLLM.value
        resource_fit_selector = VLLMResourceFitSelector(config, m)
        placement_scorer = PlacementScorer(m)

        actual_candidates = await resource_fit_selector.select_candidates(workers)
        actual_candidates = await placement_scorer.score(actual_candidates)
        actual_candidate, _ = await scheduler.find_candidate(config, m, workers)

        assert len(actual_candidates) == len(expected_candidates)
        compare_candidates(actual_candidates, expected_candidates)
        compare_candidates(
            [actual_candidate], [expected_candidates[final_candidate_index]]
        )


@pytest.mark.asyncio
async def test_auto_schedule_single_work_single_gpu(config):
    workers = [
        linux_nvidia_1_4090_24gx1(),
    ]
    m = make_model(1, None, "Qwen/Qwen3-32B")

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
- With --gpu-memory-utilization=0.9, all GPUs combined need to provide at least 83.59 GiB of total VRAM and each GPU needs 90% of allocatable VRAM.
- The current available GPU only has 24.23 GiB allocatable VRAM (100.00%)."""
        ]
        assert resource_fit_selector._messages == expect_msg


@pytest.mark.parametrize(
    "index, workers, model, expect_msg",
    [
        (
            1,
            [linux_nvidia_4_4080_16gx4()],
            make_model(1, None, "Qwen/Qwen2.5-Omni-7B"),
            [
                '- The model requires approximately 26.99 GiB of VRAM.\n'
                '- With --gpu-memory-utilization=0.9, all GPUs combined need to provide at '
                'least 29.99 GiB of total VRAM and each GPU needs 90% of allocatable VRAM.\n'
                '- Total number of attention heads (25) must be divisible by gpu count (4).\n'
                '- The largest available worker has 63.97 GiB allocatable VRAM, 4/4 of GPUs '
                'meet the VRAM utilization ratio, providing 57.57 GiB of allocatable VRAM.'
            ],
        ),
        (
            2,
            [linux_nvidia_4_4080_16gx4()],
            make_model(1, None, "Qwen/Qwen3-32B"),
            [
                """- The model requires approximately 75.23 GiB of VRAM.
- With --gpu-memory-utilization=0.9, all GPUs combined need to provide at least 83.59 GiB of total VRAM and each GPU needs 90% of allocatable VRAM.
- The largest available worker has 63.97 GiB allocatable VRAM, 4/4 of GPUs meet the VRAM utilization ratio, providing 57.57 GiB of allocatable VRAM."""
            ],
        ),
    ],
)
@pytest.mark.asyncio
async def test_auto_schedule_single_work_multi_gpu(
    config, index, workers, model, expect_msg
):
    m = model

    resource_fit_selector = VLLMResourceFitSelector(config, m)
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


@pytest.mark.asyncio
async def test_auto_schedule_multi_work_multi_gpu(config):
    workers = [linux_nvidia_2_4080_16gx2(), linux_nvidia_2_4080_16gx2()]
    workers[1].id += 1
    workers[1].name += "-1"
    m = make_model(2, None, "Qwen/Qwen3-32B")

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
- With --gpu-memory-utilization=0.9, all GPUs combined need to provide at least 83.59 GiB of total VRAM and each GPU needs 90% of allocatable VRAM.
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

    m = make_model(2, ["host4090:cuda:0", "host-2-4090:cuda:0"], "Qwen/Qwen3-32B")

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
- With --gpu-memory-utilization=0.9, all GPUs combined need to provide at least 83.59 GiB of total VRAM and each GPU needs 90% of allocatable VRAM.
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
- With --gpu-memory-utilization=0.9, all GPUs combined need to provide at least 83.59 GiB of total VRAM and each GPU needs 90% of allocatable VRAM.
- Worker host4090 GPU indexes [0] and other 1 worker fail to meet the 90.00% allocatable VRAM ratio.
- Selected GPUs have 25.87 GiB allocatable VRAM, 0/2 of GPUs meet the VRAM utilization ratio, providing 0.00 GiB of allocatable VRAM."""
        ]

        assert resource_fit_selector2._messages == expect_msg2


@pytest.mark.parametrize(
    "pretrained_config, expect_num",
    [
        (
            {
                "_name_or_path": "Qwen/Qwen2.5-VL-72B-Instruct",
                "num_attention_heads": 64,
            },
            64,
        ),
        (
            {
                "model_type": "llama4",
                "vision_config": {
                    "model_type": "llama4_vision_model",
                    "num_attention_heads": 4,
                },
                "text_config": {
                    "model_type": "llama4_text",
                    "num_attention_heads": 10,
                },
            },
            10,
        ),
        (
            {
                "_name_or_path": "/mnt/petrelfs/wangweiyun/workspace_wwy/open_source/InternVL/internvl_chat/work_dirs/internvl_chat_v3_0/InternVL3_0-78B-MPO-try0-2/checkpoint-400",
                "llm_config": {
                    "num_attention_heads": 64,
                },
                "vision_config": {
                    "num_attention_heads": 25,
                },
            },
            64,
        ),
        (
            # case: num_attention_heads define error in config
            {
                "num_attention_heads": 128,
                "_name_or_path": "/mnt/petrelfs/wangweiyun/workspace_wwy/open_source/InternVL/internvl_chat/work_dirs/internvl_chat_v3_0/InternVL3_0-78B-MPO-try0-2/checkpoint-400",
                "llm_config": {
                    "num_attention_heads": 64,
                },
                "vision_config": {
                    "num_attention_heads": 25,
                },
            },
            64,
        ),
        (
            {
                # model_scope/LLM-Research/gemma-3-27b-it
                "text_config": {
                    "num_attention_heads": 32,
                },
                "vision_config": {
                    "num_attention_heads": 16,
                },
            },
            32,
        ),
        (
            # Qwen/Qwen2.5-Omni-7B
            {
                "thinker_config": {
                    "text_config": {
                        "num_attention_heads": 28,
                    }
                },
                "talker_config": {
                    "num_attention_heads": 12,
                },
            },
            28,
        ),
    ],
)
@pytest.mark.asyncio
async def test_num_attention_heads(config, pretrained_config, expect_num):
    class DictToObj:
        def __init__(self, dictionary):
            for key, value in dictionary.items():
                if isinstance(value, dict):
                    setattr(self, key, DictToObj(value))
                else:
                    setattr(self, key, value)

    pretrained_config_obj = DictToObj(pretrained_config)
    num_attention_heads = get_model_num_attention_heads(pretrained_config_obj)

    assert num_attention_heads == expect_num
