import tempfile
import shutil
from typing import List
import pytest
from gpustack.config.config import Config
from gpustack.policies.scorers.placement_scorer import PlacementScorer
from gpustack.policies.base import ModelInstanceScheduleCandidate
from gpustack.policies.candidate_selectors.gguf_resource_fit_selector import (
    GGUFResourceFitSelector,
)
from gpustack.policies.worker_filters.label_matching_filter import LabelMatchingFilter
from gpustack.scheduler.calculator import (
    GPUOffloadEnum,
    ModelInstanceResourceClaim,
)

from gpustack.scheduler.scheduler import Scheduler
from gpustack.schemas.models import (
    ComputedResourceClaim,
    GPUSelector,
    Model,
    ModelInstance,
    ModelInstanceRPCServer,
    ModelInstanceStateEnum,
    PlacementStrategyEnum,
)
from tests.fixtures.workers.fixtures import (
    linux_nvidia_19_4090_24gx2,
    linux_nvidia_3_4090_24gx1,
    linux_nvidia_4_4080_16gx4,
    linux_nvidia_5_a100_80gx2,
    linux_nvidia_6_a100_80gx2,
    linux_nvidia_7_a100_80gx2,
    linux_nvidia_8_3090_24gx8,
    linux_rocm_1_7800_16gx1,
    macos_metal_1_m1pro_21g,
    linux_nvidia_2_4080_16gx2,
    linux_nvidia_1_4090_24gx1,
    linux_cpu_1,
    linux_cpu_2,
    macos_metal_2_m2_24g,
)

from tests.fixtures.estimates.fixtures import (
    deepseek_r1_distill_qwen_32b_bf16_partial_offload,
    deepseek_r1_distill_qwen_32b_bf16_partial_offload_split_1main_1rpc,
    deepseek_r1_distill_qwen_32b_bf16_partial_offload_split_3_1,
    deepseek_r1_distill_qwen_32b_bf16_partial_offload_split_3_2,
    deepseek_r1_distill_qwen_32b_bf16_partial_offload_split_3_3,
    deepseek_r1_q4_k_m_partial_offload,
    deepseek_r1_q4_k_m_partial_offload_split_1main_1rpc,
    deepseek_r1_q4_k_m_partial_offload_split_6,
    deepseek_r1_ud_iq2_xxs_partial_offload,
    deepseek_r1_ud_iq2_xxs_partial_offload_split_2,
    deepseek_r1_ud_iq2_xxs_partial_offload_split_3,
    deepseek_r1_ud_iq2_xxs_partial_offload_split_4,
    deepseek_r1_ud_iq2_xxs_partial_offload_split_5,
    deepseek_r1_ud_iq2_xxs_partial_offload_split_6,
    deepseek_r1_ud_iq2_xxs_partial_offload_split_7,
    deepseek_r1_ud_iq2_xxs_partial_offload_split_8,
    llama3_70b_disable_offload,
    llama3_70b_partial_offload,
    llama3_70b_partial_offload_split_2_4080_4090,
    llama3_70b_partial_offload_split_2_4080,
    llama3_70b_partial_offload_split_2_4090,
    llama3_70b_partial_offload_split_3_4080,
    llama3_70b_partial_offload_split_3_4080_2,
    llama3_70b_partial_offload_split_3_4080_3,
    llama3_70b_partial_offload_split_3_4080_4,
    llama3_70b_partial_offload_split_3_4080_4090,
    llama3_70b_partial_offload_split_3_4080_4090_2,
    llama3_8b_disable_offload,
    llama3_8b_partial_offload,
    llama3_8b_partial_offload_split_1main_1rpc,
)

from unittest.mock import patch, AsyncMock

from tests.utils.model import new_model, new_model_instance


@pytest.fixture(scope="module", autouse=True)
def temp_dir():
    tmp_dir = tempfile.mkdtemp()
    print(f"Created temporary directory: {tmp_dir}")
    yield tmp_dir
    shutil.rmtree(tmp_dir)


@pytest.fixture(scope="module", autouse=True)
def config(temp_dir):
    cfg = Config(token="test", jwt_secret_key="test", data_dir=temp_dir)
    return cfg


@pytest.mark.asyncio
async def test_label_matching_filter():
    workers = [
        macos_metal_1_m1pro_21g(),
        linux_nvidia_1_4090_24gx1(),
        linux_nvidia_2_4080_16gx2(),
    ]

    labels = {"os": "Darwin"}
    m = new_model(1, "test", 1, "llama3:8b", worker_selector=labels)
    mi = new_model_instance(1, "test", 1)

    filter = LabelMatchingFilter(m, mi)
    candidates, _ = await filter.filter(workers)

    assert len(candidates) == 1
    assert candidates[0].labels == labels


@pytest.mark.asyncio
async def test_schedule_to_single_worker_single_gpu(config):
    workers = [
        macos_metal_1_m1pro_21g(),
        linux_nvidia_1_4090_24gx1(),
        linux_nvidia_2_4080_16gx2(),
    ]

    m = new_model(1, "test", 1, "llama3:8b")
    mi = new_model_instance(1, "test", 1)

    resource_fit_selector = GGUFResourceFitSelector(m, mi)
    placement_scorer = PlacementScorer(m, mi)
    scheduler = Scheduler(config)

    with (
        patch(
            'gpustack.policies.utils.get_worker_model_instances',
            return_value=[],
        ),
        patch(
            'gpustack.policies.candidate_selectors.gguf_resource_fit_selector.calculate_model_resource_claim',
            side_effect=mock_calculate_model_resource_claim,
        ),
    ):

        candidates = await resource_fit_selector.select_candidates(workers)
        candidates = await placement_scorer.score(candidates)
        candidate, _ = await scheduler.find_candidate(mi, m, workers)

        expected_candidates = [
            # non uma
            {
                "offload_layers": 33,
                "worker_id": 3,
                "worker_name": "host4080",
                "gpu_indexes": [0],
                "is_unified_memory": False,
                "ram": 337237976,
                "vram": {0: 6315049984},
                "score": 24.68491284676514,
            },
            {
                "offload_layers": 33,
                "worker_id": 3,
                "worker_name": "host4080",
                "gpu_indexes": [1],
                "is_unified_memory": False,
                "ram": 337237976,
                "vram": {1: 6315049984},
                "score": 24.68491284676514,
            },
            {
                "offload_layers": 33,
                "worker_id": 2,
                "worker_name": "host4090",
                "gpu_indexes": [0],
                "is_unified_memory": False,
                "ram": 337237976,
                "vram": {0: 6315049984},
                "score": 16.350315512573545,
            },
            # uma
            {
                "offload_layers": 33,
                "worker_id": 1,
                "worker_name": "hostmacosmetal",
                "gpu_indexes": [0],
                "is_unified_memory": True,
                "ram": 179951576,
                "vram": {0: 1074271232},
                "score": 3.3011152944948687,
            },
        ]

        assert len(candidates) == 4
        assert candidate == candidates[0]
        compare_candidates(candidates, expected_candidates)


@pytest.mark.asyncio
async def test_schedule_to_single_worker_multi_gpu(config):
    workers = [
        linux_nvidia_1_4090_24gx1(),
        linux_nvidia_2_4080_16gx2(),
        linux_nvidia_4_4080_16gx4(),
        linux_nvidia_19_4090_24gx2(),
    ]

    m = new_model(1, "test", 1, "llama3:70b")
    mi = new_model_instance(1, "test", 1)

    resource_fit_selector = GGUFResourceFitSelector(m, mi)
    placement_scorer = PlacementScorer(m, mi)
    scheduler = Scheduler(config)

    with (
        patch(
            'gpustack.policies.utils.get_worker_model_instances',
            return_value=[],
        ),
        patch(
            'gpustack.policies.candidate_selectors.gguf_resource_fit_selector.calculate_model_resource_claim',
            side_effect=mock_calculate_model_resource_claim,
        ),
    ):

        # filter
        candidates = await resource_fit_selector.select_candidates(workers)
        candidates = await placement_scorer.score(candidates)
        candidate, _ = await scheduler.find_candidate(mi, m, workers)

        expected_candidates = [
            {
                "offload_layers": 81,
                "worker_id": 4,
                "worker_name": "host-2-4090",
                "gpu_indexes": [0, 1],
                "is_unified_memory": False,
                "ram": 421610680,
                "vram": {0: 22564900864, 1: 22397527040},
                "score": 57.92957229698279,
                "tensor_split": [26015170560, 26015170560],
            },
        ]

        assert len(candidates) == 1
        assert candidate == candidates[0]
        compare_candidates(candidates, expected_candidates)


@pytest.mark.asyncio
async def test_schedule_to_single_worker_multi_gpu_with_deepseek_r1(config):
    workers = [
        linux_nvidia_8_3090_24gx8(),
    ]

    m = new_model(
        1,
        "DeepSeek-R1-GGUF",
        1,
        huggingface_repo_id="unsloth/DeepSeek-R1-GGUF",
        cpu_offloading=True,
        placement_strategy=PlacementStrategyEnum.SPREAD,
        huggingface_filename="DeepSeek-R1-UD-IQ2_XXS/DeepSeek-R1-UD-IQ2_XXS-00001-of-00004.gguf",
    )
    mi = new_model_instance(1, "test", 1)

    resource_fit_selector = GGUFResourceFitSelector(m, mi)
    placement_scorer_spread = PlacementScorer(m, mi)
    scheduler = Scheduler(config)

    with (
        patch(
            'gpustack.policies.utils.get_worker_model_instances',
            return_value=[],
        ),
        patch(
            'gpustack.policies.candidate_selectors.gguf_resource_fit_selector.calculate_model_resource_claim',
            side_effect=mock_calculate_model_resource_claim_for_deepseek_r1,
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

        spread_candidates = await resource_fit_selector.select_candidates(workers)
        spread_candidates = await placement_scorer_spread.score(spread_candidates)
        spread_candidate, _ = await scheduler.find_candidate(mi, m, workers)

        expected_candidates = [
            {
                "offload_layers": 55,
                "worker_id": 11,
                "worker_name": "host01-3090",
                "gpu_indexes": [0, 1, 2, 3, 4, 5, 6, 7],
                "is_unified_memory": False,
                "ram": 14750726248,
                "vram": {
                    0: 25728538112,
                    1: 24732937728,
                    2: 24732937728,
                    3: 24732937728,
                    4: 24732937728,
                    5: 24732937728,
                    6: 24732937728,
                    7: 21244001280,
                },
                "score": 100,
                "tensor_split": [
                    25769803776,
                    25769803776,
                    25769803776,
                    25769803776,
                    25769803776,
                    25769803776,
                    25769803776,
                    25769803776,
                ],
            },
        ]

        assert len(spread_candidates) == 1
        assert spread_candidate == spread_candidates[0]
        compare_candidates(spread_candidates, expected_candidates)


@pytest.mark.asyncio
async def test_schedule_to_single_worker_multi_gpu_with_binpack_spread(config):
    workers = [
        linux_nvidia_1_4090_24gx1(),
        linux_nvidia_2_4080_16gx2(),
        linux_nvidia_4_4080_16gx4(),
    ]

    m = new_model(
        1, "test", 1, "llama3:70b", placement_strategy=PlacementStrategyEnum.BINPACK
    )
    mi_binpack = new_model_instance(1, "test_binpack", 1)
    mi_spread = new_model_instance(2, "test_spread", 1)

    mis = [
        new_model_instance(
            10,
            "test-1",
            1,
            5,
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
        new_model_instance(
            11,
            "test-1",
            1,
            5,
            ModelInstanceStateEnum.RUNNING,
            [3],
            ComputedResourceClaim(
                is_unified_memory=False,
                offload_layers=10,
                total_layers=10,
                ram=0,
                vram={3: 600 * 1024 * 1024},
            ),
        ),
    ]

    resource_fit_selector_binpack = GGUFResourceFitSelector(m, mi_binpack)
    placement_scorer_binpack = PlacementScorer(m, mi_binpack)

    resource_fit_selector_spread = GGUFResourceFitSelector(m, mi_spread)
    placement_scorer_spread = PlacementScorer(m, mi_spread)

    scheduler = Scheduler(config)

    with (
        patch(
            'gpustack.policies.utils.get_worker_model_instances',
            return_value=mis,
        ),
        patch(
            'gpustack.policies.candidate_selectors.gguf_resource_fit_selector.calculate_model_resource_claim',
            side_effect=mock_calculate_model_resource_claim,
        ),
        patch(
            'gpustack.policies.scorers.placement_scorer.get_model_instances',
            return_value=mis,
        ),
    ):

        # binpack
        binpack_candidates = await resource_fit_selector_binpack.select_candidates(
            workers
        )
        binpack_candidates = await placement_scorer_binpack.score(binpack_candidates)
        binpack_candidate, _ = await scheduler.find_candidate(mi_binpack, m, workers)

        expected_candidates = [
            {
                "offload_layers": 81,
                "worker_id": 5,
                "worker_name": "host-4-4080",
                "gpu_indexes": [0, 1, 2],
                "is_unified_memory": False,
                "ram": 471942328,
                "vram": {
                    0: 16218530816,
                    1: 16217985024,
                    2: 15703068672,
                },
                "score": 63.08188756952499,
                "tensor_split": [17171480576, 17171480576, 16647192576],
            },
            {
                "offload_layers": 81,
                "worker_id": 5,
                "worker_name": "host-4-4080",
                "gpu_indexes": [0, 1, 3],
                "is_unified_memory": False,
                "ram": 471942328,
                "vram": {
                    0: 16218530816,
                    1: 16217985024,
                    3: 15703068672,
                },
                "score": 63.40144801834418,
                "tensor_split": [17171480576, 17171480576, 16542334976],
            },
            {
                "offload_layers": 81,
                "worker_id": 5,
                "worker_name": "host-4-4080",
                "gpu_indexes": [0, 2, 3],
                "is_unified_memory": False,
                "ram": 471942328,
                "vram": {
                    0: 16733447168,
                    2: 15703068672,
                    3: 15703068672,
                },
                "score": 65.08312111378692,
                "tensor_split": [17171480576, 16647192576, 16542334976],
            },
            {
                "offload_layers": 81,
                "worker_id": 5,
                "worker_name": "host-4-4080",
                "gpu_indexes": [1, 2, 3],
                "is_unified_memory": False,
                "ram": 471942328,
                "vram": {
                    1: 16733447168,
                    2: 15703068672,
                    3: 15703068672,
                },
                "score": 65.08312111378692,
                "tensor_split": [17171480576, 16647192576, 16542334976],
            },
        ]

        assert len(binpack_candidates) == 4
        assert binpack_candidate == binpack_candidates[2]
        compare_candidates(binpack_candidates, expected_candidates)

        # spread
        m.placement_strategy = PlacementStrategyEnum.SPREAD
        spread_candidates = await resource_fit_selector_spread.select_candidates(
            workers
        )
        spread_candidates = await placement_scorer_spread.score(spread_candidates)
        spread_candidate, _ = await scheduler.find_candidate(mi_spread, m, workers)

        expected_candidates = [
            {
                "gpu_indexes": [0, 1, 2],
                "score": 85.0,
                "tensor_split": [17171480576, 17171480576, 16647192576],
            },
            {
                "gpu_indexes": [0, 1, 3],
                "score": 85.0,
                "tensor_split": [17171480576, 17171480576, 16542334976],
            },
            {
                "gpu_indexes": [0, 2, 3],
                "score": 84.0,
                "tensor_split": [17171480576, 16647192576, 16542334976],
            },
            {
                "gpu_indexes": [1, 2, 3],
                "score": 84.0,
                "tensor_split": [17171480576, 16647192576, 16542334976],
            },
        ]

        assert len(spread_candidates) == 4
        assert spread_candidate == spread_candidates[0]
        compare_candidates(spread_candidates, expected_candidates)


@pytest.mark.asyncio
async def test_schedule_to_single_worker_multi_gpu_partial_offload(config):
    workers = [
        linux_nvidia_1_4090_24gx1(),
        linux_nvidia_2_4080_16gx2(),
    ]

    m = new_model(
        1,
        "test",
        1,
        "llama3:70b",
        cpu_offloading=True,
        distributed_inference_across_workers=False,
    )
    mi_binpack = new_model_instance(1, "test_binpack", 1)

    resource_fit_selector_binpack = GGUFResourceFitSelector(m, mi_binpack)
    placement_scorer_binpack = PlacementScorer(m, mi_binpack)
    scheduler = Scheduler(config)

    with (
        patch(
            'gpustack.policies.utils.get_worker_model_instances',
            return_value=[],
        ),
        patch(
            'gpustack.policies.candidate_selectors.gguf_resource_fit_selector.calculate_model_resource_claim',
            side_effect=mock_calculate_model_resource_claim,
        ),
        patch(
            'gpustack.policies.scorers.placement_scorer.get_model_instances',
            return_value=[],
        ),
    ):

        # binpack
        binpack_candidates = await resource_fit_selector_binpack.select_candidates(
            workers
        )
        binpack_candidates = await placement_scorer_binpack.score(binpack_candidates)
        binpack_candidate, _ = await scheduler.find_candidate(mi_binpack, m, workers)

        expected_candidates = [
            {
                "offload_layers": 60,
                "worker_id": 3,
                "worker_name": "host4080",
                "gpu_indexes": [0, 1],
                "is_unified_memory": False,
                "ram": 1093245112,
                "vram": {
                    0: 16900820992,
                    1: 16900820992,
                },
                "score": 66.15,
                "tensor_split": [17171480576, 17171480576],
            }
        ]

        assert len(binpack_candidates) == 1
        assert binpack_candidate == binpack_candidates[0]
        compare_candidates(binpack_candidates, expected_candidates)


@pytest.mark.asyncio
async def test_schedule_to_cpu_with_binpack_spread(config):
    workers = [
        linux_cpu_1(),
        linux_cpu_2(),
    ]

    m = new_model(1, "test", 1, "llama3:70b", cpu_offloading=True)
    mi_binpack = new_model_instance(4, "test_binpack", 1)
    mi_spread = new_model_instance(5, "test_spread", 1)

    mis = [
        new_model_instance(
            1,
            "test-1",
            1,
            6,
            ModelInstanceStateEnum.RUNNING,
            None,
            ComputedResourceClaim(
                is_unified_memory=False,
                offload_layers=0,
                total_layers=81,
                ram=20 * 1024 * 1024 * 1024,
            ),
        ),
        new_model_instance(
            2,
            "test-2",
            1,
            6,
            ModelInstanceStateEnum.RUNNING,
            None,
            ComputedResourceClaim(
                is_unified_memory=False,
                offload_layers=0,
                total_layers=81,
                ram=10 * 1024 * 1024 * 1024,
            ),
        ),
        new_model_instance(
            3,
            "test-3",
            1,
            7,
            ModelInstanceStateEnum.RUNNING,
            None,
            ComputedResourceClaim(
                is_unified_memory=False,
                offload_layers=0,
                total_layers=81,
                ram=10 * 1024 * 1024 * 1024,
            ),
        ),
    ]
    resource_fit_selector_binpack = GGUFResourceFitSelector(m, mi_binpack)
    placement_scorer_binpack = PlacementScorer(m, mi_binpack)
    scheduler = Scheduler(config)

    with (
        patch(
            'gpustack.policies.utils.get_worker_model_instances',
            return_value=mis,
        ),
        patch(
            'gpustack.policies.candidate_selectors.gguf_resource_fit_selector.calculate_model_resource_claim',
            side_effect=mock_calculate_model_resource_claim,
        ),
        patch(
            'gpustack.policies.scorers.placement_scorer.get_model_instances',
            return_value=mis,
        ),
    ):

        # binpack
        binpack_candidates = await resource_fit_selector_binpack.select_candidates(
            workers
        )
        binpack_candidates = await placement_scorer_binpack.score(binpack_candidates)
        binpack_candidate, _ = await scheduler.find_candidate(mi_binpack, m, workers)

        expected_candidates = [
            {
                "offload_layers": 0,
                "total_layers": 81,
                "worker_id": 6,
                "worker_name": "host-cpu-1",
                "is_unified_memory": False,
                "ram": 3106511032,
                "score": 8.5093054482,
            },
            {
                "offload_layers": 0,
                "total_layers": 81,
                "worker_id": 7,
                "worker_name": "host-cpu-2",
                "is_unified_memory": False,
                "ram": 3106511032,
                "score": 2.4518337732,
            },
        ]

        assert len(binpack_candidates) == 2
        assert binpack_candidate == binpack_candidates[0]
        compare_candidates(binpack_candidates, expected_candidates)

        # spread
        m.placement_strategy = PlacementStrategyEnum.SPREAD

        resource_fit_selector_spread = GGUFResourceFitSelector(m, mi_spread)
        placement_policy_spread = PlacementScorer(m, mi_spread)

        spread_candidates = await resource_fit_selector_spread.select_candidates(
            workers
        )
        spread_candidates = await placement_policy_spread.score(spread_candidates)
        spread_candidate, _ = await scheduler.find_candidate(mi_spread, m, workers)

        expected_spread_candidates = [
            {"worker_id": 6, "score": 83.3333333333},
            {"worker_id": 7, "score": 85.0},
        ]

        assert len(spread_candidates) == 2
        assert spread_candidate == spread_candidates[1]
        compare_candidates(spread_candidates, expected_spread_candidates)


@pytest.mark.asyncio
async def test_schedule_to_multi_worker_multi_gpu(config):
    workers = [
        linux_nvidia_1_4090_24gx1(),
        linux_nvidia_2_4080_16gx2(),
    ]

    m = new_model(1, "test", 1, "llama3:70b", cpu_offloading=False)
    mi_binpack = new_model_instance(1, "test_binpack", 1)

    resource_fit_selector_binpack = GGUFResourceFitSelector(m, mi_binpack)
    placement_scorer_binpack = PlacementScorer(m, mi_binpack)
    scheduler = Scheduler(config)

    with (
        patch(
            'gpustack.policies.utils.get_worker_model_instances',
            return_value=[],
        ),
        patch(
            'gpustack.policies.candidate_selectors.gguf_resource_fit_selector.calculate_model_resource_claim',
            side_effect=mock_calculate_model_resource_claim,
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

        # binpack
        binpack_candidates = await resource_fit_selector_binpack.select_candidates(
            workers
        )
        binpack_candidates = await placement_scorer_binpack.score(binpack_candidates)
        binpack_candidate, _ = await scheduler.find_candidate(mi_binpack, m, workers)

        expected_candidates = [
            {
                "offload_layers": 81,
                "worker_id": 3,
                "worker_name": "host4080",
                "gpu_indexes": [0, 1],
                "is_unified_memory": False,
                "ram": 421610680,
                "vram": {
                    0: 13811322880,
                    1: 13296406528,
                },
                "score": 53.10511012189776,
                "tensor_split": [26015170560, 17171480576, 17171480576],
                "rpc_servers": [
                    ModelInstanceRPCServer(
                        worker_id=2,
                        gpu_index=0,
                        computed_resource_claim=ComputedResourceClaim(
                            is_unified_memory=False,
                            offload_layers=33,
                            total_layers=81,
                            ram=0,
                            vram={0: 19308028928},
                        ),
                    )
                ],
            },
        ]

        assert len(binpack_candidates) == 1
        assert binpack_candidate == binpack_candidates[0]
        compare_candidates(binpack_candidates, expected_candidates)


@pytest.mark.asyncio
async def test_manual_schedule_to_multi_worker_multi_gpu(config):
    workers = [
        linux_nvidia_1_4090_24gx1(),
        linux_nvidia_2_4080_16gx2(),
    ]

    m = new_model(
        1,
        "test",
        1,
        "llama3:70b",
        cpu_offloading=False,
        gpu_selector=GPUSelector(
            gpu_ids=[
                "host4080:cuda:0",
                "host4080:cuda:1",
                "host4090:cuda:0",
            ]
        ),
    )
    mi_binpack = new_model_instance(1, "test_binpack", 1)

    resource_fit_selector_binpack = GGUFResourceFitSelector(m, mi_binpack)
    placement_scorer_binpack = PlacementScorer(m, mi_binpack)
    scheduler = Scheduler(config)

    with (
        patch(
            'gpustack.policies.utils.get_worker_model_instances',
            return_value=[],
        ),
        patch(
            'gpustack.policies.candidate_selectors.gguf_resource_fit_selector.calculate_model_resource_claim',
            side_effect=mock_calculate_model_resource_claim,
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

        # binpack
        binpack_candidates = await resource_fit_selector_binpack.select_candidates(
            workers
        )
        binpack_candidates = await placement_scorer_binpack.score(binpack_candidates)
        binpack_candidate, _ = await scheduler.find_candidate(mi_binpack, m, workers)

        expected_candidates = [
            {
                "offload_layers": 81,
                "worker_id": 3,
                "worker_name": "host4080",
                "gpu_indexes": [0, 1],
                "is_unified_memory": False,
                "ram": 421610680,
                "vram": {
                    0: 13811322880,
                    1: 13296406528,
                },
                "score": 53.10511012189776,
                "tensor_split": [26015170560, 17171480576, 17171480576],
                "rpc_servers": [
                    ModelInstanceRPCServer(
                        worker_id=2,
                        gpu_index=0,
                        computed_resource_claim=ComputedResourceClaim(
                            is_unified_memory=False,
                            offload_layers=33,
                            total_layers=81,
                            ram=0,
                            vram={0: 19308028928},
                        ),
                    )
                ],
            },
        ]

        assert len(binpack_candidates) == 1
        assert binpack_candidate == binpack_candidates[0]
        compare_candidates(binpack_candidates, expected_candidates)


@pytest.mark.asyncio
async def test_manual_schedule_to_multi_worker_multi_gpu_with_deepseek_r1(config):
    workers = [
        linux_nvidia_5_a100_80gx2(),
        linux_nvidia_6_a100_80gx2(),
        linux_nvidia_7_a100_80gx2(),
    ]

    m = new_model(
        1,
        "DeepSeek-R1-GGUF",
        1,
        huggingface_repo_id="unsloth/DeepSeek-R1-GGUF",
        cpu_offloading=False,
        gpu_selector=GPUSelector(
            gpu_ids=[
                "llm01-A100:cuda:0",
                "llm01-A100:cuda:1",
                "llm02-A100:cuda:0",
                "llm02-A100:cuda:1",
                "llm03-A100:cuda:0",
                "llm03-A100:cuda:1",
            ]
        ),
        placement_strategy=PlacementStrategyEnum.SPREAD,
        huggingface_filename="DeepSeek-R1-Q4_K_M/DeepSeek-R1-Q4_K_M-00001-of-00009.gguf",
    )
    mi_spread = new_model_instance(1, "test_spread", 1)

    resource_fit_selector_spread = GGUFResourceFitSelector(m, mi_spread)
    placement_scorer_spread = PlacementScorer(m, mi_spread)
    scheduler = Scheduler(config)

    with (
        patch(
            'gpustack.policies.utils.get_worker_model_instances',
            return_value=[],
        ),
        patch(
            'gpustack.policies.candidate_selectors.gguf_resource_fit_selector.calculate_model_resource_claim',
            side_effect=mock_calculate_model_resource_claim_for_deepseek_r1,
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

        spread_candidates = await resource_fit_selector_spread.select_candidates(
            workers
        )
        spread_candidates = await placement_scorer_spread.score(spread_candidates)
        spread_candidate, _ = await scheduler.find_candidate(mi_spread, m, workers)

        expected_candidates = [
            {
                "offload_layers": 62,
                "worker_id": 8,
                "worker_name": "llm01-A100",
                "gpu_indexes": [0, 1],
                "is_unified_memory": False,
                "ram": 1531856328,
                "vram": {
                    0: 70670915584,
                    1: 68910751744,
                },
                "score": 100,
                "tensor_split": [
                    85899345920,
                    85899345920,
                    85899345920,
                    85899345920,
                    85899345920,
                    85899345920,
                ],
                "rpc_servers": [
                    ModelInstanceRPCServer(
                        worker_id=9,
                        gpu_index=0,
                        computed_resource_claim=ComputedResourceClaim(
                            is_unified_memory=False,
                            offload_layers=11,
                            total_layers=62,
                            ram=0,
                            vram={0: 59939913728},
                        ),
                    ),
                    ModelInstanceRPCServer(
                        worker_id=9,
                        gpu_index=1,
                        computed_resource_claim=ComputedResourceClaim(
                            is_unified_memory=False,
                            offload_layers=10,
                            total_layers=62,
                            ram=0,
                            vram={1: 69698246656},
                        ),
                    ),
                    ModelInstanceRPCServer(
                        worker_id=10,
                        gpu_index=0,
                        computed_resource_claim=ComputedResourceClaim(
                            is_unified_memory=False,
                            offload_layers=10,
                            total_layers=62,
                            ram=0,
                            vram={0: 70670915584},
                        ),
                    ),
                    ModelInstanceRPCServer(
                        worker_id=10,
                        gpu_index=1,
                        computed_resource_claim=ComputedResourceClaim(
                            is_unified_memory=False,
                            offload_layers=11,
                            total_layers=62,
                            ram=0,
                            vram={1: 76345232384},
                        ),
                    ),
                ],
            },
        ]

        assert len(spread_candidates) == 1
        assert spread_candidate == spread_candidates[0]
        compare_candidates(spread_candidates, expected_candidates)


@pytest.mark.asyncio
async def test_manual_schedule_to_multi_worker_multi_gpu_with_deepseek_r1_distill_qwen(
    config,
):
    workers = [
        linux_nvidia_3_4090_24gx1(),
        macos_metal_2_m2_24g(),
        linux_rocm_1_7800_16gx1(),
    ]

    m = new_model(
        1,
        "DeepSeek-R1-GGUF",
        1,
        huggingface_repo_id="bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF",
        cpu_offloading=True,
        gpu_selector=GPUSelector(
            gpu_ids=[
                "host-2-4090:cuda:0",
                "host01-7800:rocm:0",
                "host02-metal:mps:0",
            ]
        ),
        placement_strategy=PlacementStrategyEnum.SPREAD,
        huggingface_filename="DeepSeek-R1-Distill-Qwen-32B-bf16/DeepSeek-R1-Distill-Qwen-32B-bf16-00001-of-00002.gguf",
    )
    mi = new_model_instance(1, "test", 1)

    resource_fit_selector = GGUFResourceFitSelector(m, mi)
    placement_scorer = PlacementScorer(m, mi)
    scheduler = Scheduler(config)

    with (
        patch(
            'gpustack.policies.utils.get_worker_model_instances',
            return_value=[],
        ),
        patch(
            'gpustack.policies.candidate_selectors.gguf_resource_fit_selector.calculate_model_resource_claim',
            side_effect=mock_calculate_model_resource_claim_for_deepseek_r1,
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
        candidate, _ = await scheduler.find_candidate(mi, m, workers)

        expected_candidates = [
            {
                "offload_layers": 40,
                "worker_id": 12,
                "worker_name": "host-2-4090",
                "gpu_indexes": [0],
                "is_unified_memory": False,
                "ram": 30171626040,
                "vram": {
                    0: 22050643968,
                },
                "score": 100,
                "tensor_split": [17163091968, 16106143744, 24683479040],
                "rpc_servers": [
                    ModelInstanceRPCServer(
                        worker_id=13,
                        gpu_index=0,
                        computed_resource_claim=ComputedResourceClaim(
                            is_unified_memory=False,
                            offload_layers=12,
                            total_layers=65,
                            ram=0,
                            vram={0: 16503328768},
                        ),
                    ),
                    ModelInstanceRPCServer(
                        worker_id=14,
                        gpu_index=0,
                        computed_resource_claim=ComputedResourceClaim(
                            is_unified_memory=True,
                            offload_layers=11,
                            total_layers=65,
                            ram=0,
                            vram={0: 12204093440},
                        ),
                    ),
                ],
            },
        ]

        assert len(candidates) == 1
        assert candidate == candidates[0]
        compare_candidates(candidates, expected_candidates)


@pytest.mark.asyncio
async def test_manual_schedule_to_single_worker_multi_gpu(config):
    workers = [
        linux_nvidia_1_4090_24gx1(),
        linux_nvidia_2_4080_16gx2(),
        linux_nvidia_4_4080_16gx4(),
    ]

    m = new_model(
        1,
        "test",
        1,
        "llama3:70b",
        placement_strategy=PlacementStrategyEnum.BINPACK,
        gpu_selector=GPUSelector(
            gpu_ids=["host-4-4080:cuda:0", "host-4-4080:cuda:2", "host-4-4080:cuda:3"]
        ),
    )
    mi = new_model_instance(1, "test", 1)

    mis = [
        new_model_instance(
            10,
            "test-1",
            1,
            5,
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
        new_model_instance(
            11,
            "test-1",
            1,
            5,
            ModelInstanceStateEnum.RUNNING,
            [3],
            ComputedResourceClaim(
                is_unified_memory=False,
                offload_layers=10,
                total_layers=10,
                ram=0,
                vram={3: 600 * 1024 * 1024},
            ),
        ),
    ]

    resource_fit_selector = GGUFResourceFitSelector(m, mi)
    placement_scorer = PlacementScorer(m, mi)

    with (
        patch(
            'gpustack.policies.utils.get_worker_model_instances',
            return_value=mis,
        ),
        patch(
            'gpustack.policies.candidate_selectors.gguf_resource_fit_selector.calculate_model_resource_claim',
            side_effect=mock_calculate_model_resource_claim,
        ),
        patch(
            'gpustack.policies.scorers.placement_scorer.get_model_instances',
            return_value=mis,
        ),
    ):

        candidates = await resource_fit_selector.select_candidates(workers)
        candidates = await placement_scorer.score(candidates)

        expected_candidates = [
            {
                "offload_layers": 81,
                "worker_id": 5,
                "worker_name": "host-4-4080",
                "gpu_indexes": [0, 2, 3],
                "is_unified_memory": False,
                "ram": 471942328,
                "vram": {
                    0: 16733447168,
                    2: 15703068672,
                    3: 15703068672,
                },
                "score": 65.08312111378692,
                "tensor_split": [17171480576, 16647192576, 16542334976],
            },
        ]

        assert len(candidates) == 1
        compare_candidates(candidates, expected_candidates)

        # update gpu selector
        m.gpu_selector = GPUSelector(
            gpu_ids=["host-4-4080:cuda:0", "host-4-4080:cuda:1", "host-4-4080:cuda:2"]
        )

        resource_fit_selector = GGUFResourceFitSelector(m, mi)
        placement_scorer = PlacementScorer(m, mi)

        candidates = await resource_fit_selector.select_candidates(workers)
        candidates = await placement_scorer.score(candidates)
        expected_candidates = [
            {
                "offload_layers": 81,
                "worker_id": 5,
                "worker_name": "host-4-4080",
                "gpu_indexes": [0, 1, 2],
                "is_unified_memory": False,
                "ram": 471942328,
                "vram": {
                    0: 16218530816,
                    1: 16217985024,
                    2: 15703068672,
                },
                "score": 63.08188756952499,
                "tensor_split": [17171480576, 17171480576, 16647192576],
            }
        ]

        assert len(candidates) == 1
        compare_candidates(candidates, expected_candidates)


@pytest.mark.asyncio
async def test_manual_schedule_to_single_worker_multi_gpu_partial_offload(config):
    workers = [
        linux_nvidia_1_4090_24gx1(),
        linux_nvidia_2_4080_16gx2(),
    ]

    m = new_model(
        1,
        "test",
        1,
        "llama3:70b",
        cpu_offloading=True,
        distributed_inference_across_workers=False,
        gpu_selector=GPUSelector(gpu_ids=["host4080:cuda:0", "host4080:cuda:1"]),
    )
    mi = new_model_instance(1, "test", 1)

    resource_fit_selector_binpack = GGUFResourceFitSelector(m, mi)
    placement_scorer_binpack = PlacementScorer(m, mi)

    with (
        patch(
            'gpustack.policies.utils.get_worker_model_instances',
            return_value=[],
        ),
        patch(
            'gpustack.policies.candidate_selectors.gguf_resource_fit_selector.calculate_model_resource_claim',
            side_effect=mock_calculate_model_resource_claim,
        ),
        patch(
            'gpustack.policies.scorers.placement_scorer.get_model_instances',
            return_value=[],
        ),
    ):

        candidates = await resource_fit_selector_binpack.select_candidates(workers)
        candidates = await placement_scorer_binpack.score(candidates)

        expected_candidates = [
            {
                "offload_layers": 60,
                "worker_id": 3,
                "worker_name": "host4080",
                "gpu_indexes": [0, 1],
                "is_unified_memory": False,
                "ram": 1093245112,
                "vram": {
                    0: 16900820992,
                    1: 16900820992,
                },
                "score": 66.15,
                "tensor_split": [17171480576, 17171480576],
            }
        ]

        assert len(candidates) == 1
        compare_candidates(candidates, expected_candidates)


def mock_calculate_model_resource_claim(
    model_instance: ModelInstance,
    model: Model,
    offload: GPUOffloadEnum = GPUOffloadEnum.Full,
    **kwargs,
) -> ModelInstanceResourceClaim:
    mock_estimate = AsyncMock()
    tensor_split = kwargs.get("tensor_split")
    if offload == GPUOffloadEnum.Partial:
        if model.ollama_library_model_name == "llama3:70b":
            mock_estimate = llama3_70b_partial_offload()
            if tensor_split:
                mapping = {
                    (
                        1,
                        1,
                    ): llama3_70b_partial_offload_split_2_4080,
                    (
                        17171480576,
                        17171480576,
                    ): llama3_70b_partial_offload_split_2_4080,
                    (
                        26015170560,
                        26015170560,
                    ): llama3_70b_partial_offload_split_2_4090,
                    (
                        17171480576,
                        26015170560,
                    ): llama3_70b_partial_offload_split_2_4080_4090,
                    (
                        17171480576,
                        17171480576,
                        16647192576,
                    ): llama3_70b_partial_offload_split_3_4080_2,
                    (
                        17171480576,
                        17171480576,
                        16542334976,
                    ): llama3_70b_partial_offload_split_3_4080_3,
                    (
                        17171480576,
                        16647192576,
                        16542334976,
                    ): llama3_70b_partial_offload_split_3_4080_4,
                    (
                        26015170560,
                        17171480576,
                        17171480576,
                    ): llama3_70b_partial_offload_split_3_4080_4090,
                    (
                        17171480576,
                        17171480576,
                        26015170560,
                    ): llama3_70b_partial_offload_split_3_4080_4090_2,
                    (
                        17171480576,
                        17171480576,
                        17171480576,
                    ): llama3_70b_partial_offload_split_3_4080,
                }
                mock_estimate = mapping[tuple(tensor_split)]()
        elif model.ollama_library_model_name == "llama3:8b":
            mock_estimate = llama3_8b_partial_offload()
            if tensor_split:
                mapping = {
                    (
                        1,
                        1,
                    ): llama3_8b_partial_offload_split_1main_1rpc,
                }
                mock_estimate = mapping[tuple(tensor_split)]()
    elif offload == GPUOffloadEnum.Disable:
        mock_estimate = llama3_70b_disable_offload()
        if model.ollama_library_model_name == "llama3:8b":
            return llama3_8b_disable_offload()
    return ModelInstanceResourceClaim(
        model_instance=model_instance, resource_claim_estimate=mock_estimate.estimate
    )


def mock_calculate_model_resource_claim_for_deepseek_r1(  # noqa: C901
    model_instance: ModelInstance,
    model: Model,
    offload: GPUOffloadEnum = GPUOffloadEnum.Full,
    **kwargs,
) -> ModelInstanceResourceClaim:
    mock_estimate = AsyncMock()
    tensor_split = kwargs.get("tensor_split")
    if offload == GPUOffloadEnum.Partial:
        if "deepseek-r1-q4_k_m" in model.huggingface_filename.lower():
            mock_estimate = deepseek_r1_q4_k_m_partial_offload()  # TODO
            if tensor_split:
                mapping = {
                    (
                        1,
                        1,
                    ): deepseek_r1_q4_k_m_partial_offload_split_1main_1rpc,
                    (
                        85899345920,
                        85899345920,
                        85899345920,
                        85899345920,
                        85899345920,
                        85899345920,
                    ): deepseek_r1_q4_k_m_partial_offload_split_6,
                }
                mock_estimate = mapping[tuple(tensor_split)]()

        if "deepseek-r1-ud-iq2_xxs" in model.huggingface_filename.lower():
            mock_estimate = deepseek_r1_ud_iq2_xxs_partial_offload()
            if tensor_split:
                mapping = {
                    (
                        1,
                        1,
                    ): deepseek_r1_ud_iq2_xxs_partial_offload_split_2,
                    (
                        25769803776,
                        25769803776,
                    ): deepseek_r1_ud_iq2_xxs_partial_offload_split_2,
                    (
                        25769803776,
                        25769803776,
                        25769803776,
                    ): deepseek_r1_ud_iq2_xxs_partial_offload_split_3,
                    (
                        25769803776,
                        25769803776,
                        25769803776,
                        25769803776,
                    ): deepseek_r1_ud_iq2_xxs_partial_offload_split_4,
                    (
                        25769803776,
                        25769803776,
                        25769803776,
                        25769803776,
                        25769803776,
                    ): deepseek_r1_ud_iq2_xxs_partial_offload_split_5,
                    (
                        25769803776,
                        25769803776,
                        25769803776,
                        25769803776,
                        25769803776,
                        25769803776,
                    ): deepseek_r1_ud_iq2_xxs_partial_offload_split_6,
                    (
                        25769803776,
                        25769803776,
                        25769803776,
                        25769803776,
                        25769803776,
                        25769803776,
                        25769803776,
                    ): deepseek_r1_ud_iq2_xxs_partial_offload_split_7,
                    (
                        25769803776,
                        25769803776,
                        25769803776,
                        25769803776,
                        25769803776,
                        25769803776,
                        25769803776,
                        25769803776,
                    ): deepseek_r1_ud_iq2_xxs_partial_offload_split_8,
                }
                mock_estimate = mapping[tuple(tensor_split)]()

        if "deepseek-r1-distill-qwen-32b-bf16" in model.huggingface_filename.lower():
            mock_estimate = deepseek_r1_distill_qwen_32b_bf16_partial_offload()
            if tensor_split:
                mapping = {
                    (
                        1,
                        1,
                    ): deepseek_r1_distill_qwen_32b_bf16_partial_offload_split_1main_1rpc,
                    (
                        17163091968,
                        16106143744,
                        24683479040,
                    ): deepseek_r1_distill_qwen_32b_bf16_partial_offload_split_3_1,
                    (
                        24683479040,
                        16106143744,
                        17163091968,
                    ): deepseek_r1_distill_qwen_32b_bf16_partial_offload_split_3_2,
                    (
                        24683479040,
                        17163091968,
                        16106143744,
                    ): deepseek_r1_distill_qwen_32b_bf16_partial_offload_split_3_3,
                }
                mock_estimate = mapping[tuple(tensor_split)]()
    return ModelInstanceResourceClaim(
        model_instance=model_instance, resource_claim_estimate=mock_estimate.estimate
    )


def compare_candidates(  # noqa: C901
    candidates: List[ModelInstanceScheduleCandidate], expected_candidates
):
    for i, expected in enumerate(expected_candidates):
        candidate = candidates[i]
        if "gpu_indexes" in expected:
            assert candidate.gpu_indexes == expected["gpu_indexes"]

        if "vram" in expected:
            assert candidate.computed_resource_claim.vram == expected["vram"]

        if "offload_layers" in expected:
            assert (
                candidate.computed_resource_claim.offload_layers
                == expected["offload_layers"]
            )

        if "worker_id" in expected:
            assert candidate.worker.id == expected["worker_id"]

        if "worker_name" in expected:
            assert candidate.worker.name == expected["worker_name"]

        if "is_unified_memory" in expected:
            assert (
                candidate.computed_resource_claim.is_unified_memory
                == expected["is_unified_memory"]
            )

        if "ram" in expected:
            assert candidate.computed_resource_claim.ram == expected["ram"]

        if "score" in expected:
            assert str(candidate.score)[:5] == str(expected["score"])[:5]

        if "rpc_servers" in expected:
            for i, rpc_server in enumerate(expected["rpc_servers"]):
                assert rpc_server.worker_id == expected["rpc_servers"][i].worker_id
                assert rpc_server.gpu_index == expected["rpc_servers"][i].gpu_index
                assert (
                    rpc_server.computed_resource_claim
                    == expected["rpc_servers"][i].computed_resource_claim
                )

        if "tensor_split" in expected:
            assert (
                candidate.computed_resource_claim.tensor_split
                == expected["tensor_split"]
            )
