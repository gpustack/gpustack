import tempfile
import shutil
from typing import List
import pytest
from gpustack.config.config import Config
from gpustack.policies.placement_policy import PlacementPolicy
from gpustack.policies.policy import ModelInstanceScheduleCandidate
from gpustack.policies.resource_fit_policy import ResourceFitPolicy
from gpustack.policies.label_matching_policy import LabelMatchingPolicy
from gpustack.policies.calculator import (
    GPUOffloadEnum,
    ModelInstanceResourceClaim,
)

from gpustack.scheduler.scheduler import Scheduler
from gpustack.schemas.models import (
    ComputedResourceClaim,
    Model,
    ModelInstance,
    ModelInstanceRPCServer,
    ModelInstanceStateEnum,
    PlacementStrategyEnum,
)
from tests.fixtures.workers.fixtures import (
    worker_linux_nvidia_2_4090_gpu,
    worker_linux_nvidia_4_4080_gpu,
    worker_macos_metal,
    worker_linux_nvidia_2_4080_gpu,
    worker_linux_nvidia_1_4090_gpu,
    worker_linux_cpu_1,
    worker_linux_cpu_2,
)

from tests.fixtures.estimates.fixtures import (
    llama3_70b_disable_offload_estimate_claim,
    llama3_70b_full_offload_estimate_claim,
    llama3_70b_full_offload_split_2_4080_estimate_claim,
    llama3_70b_full_offload_split_2_4080_estimate_claim_2,
    llama3_70b_full_offload_split_2_4080_estimate_claim_3,
    llama3_70b_full_offload_split_2_4090_estimate_claim,
    llama3_70b_full_offload_split_3_4080_estimate_claim,
    llama3_70b_full_offload_split_3_4080_estimate_claim_2,
    llama3_70b_full_offload_split_3_4080_estimate_claim_3,
    llama3_70b_full_offload_split_4_4080_estimate_claim,
    llama3_70b_partial_offload_estimate_claim,
    llama3_70b_partial_offload_split_2_4080_4090_estimate_claim,
    llama3_70b_partial_offload_split_2_4080_estimate_claim,
    llama3_70b_partial_offload_split_3_4080_4090_estimate_claim,
    llama3_70b_partial_offload_split_3_4080_4090_estimate_claim_2,
    llama3_8b_disable_offload_estimate_claim,
    llama3_8b_full_offload_estimate_claim,
    llama3_8b_partial_offload_estimate_claim,
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
async def test_label_matching_policy():
    workers = [
        worker_macos_metal(),
        worker_linux_nvidia_1_4090_gpu(),
        worker_linux_nvidia_2_4080_gpu(),
    ]

    labels = {"os": "Darwin"}
    m = new_model(1, "test", 1, worker_selector=labels)
    mi = new_model_instance(1, "test", 1)

    policy = LabelMatchingPolicy(m, mi)
    candidates = await policy.filter(workers)

    assert len(candidates) == 1
    assert candidates[0].labels == labels


@pytest.mark.asyncio
async def test_schedule_to_single_worker_single_gpu(config):
    workers = [
        worker_macos_metal(),
        worker_linux_nvidia_1_4090_gpu(),
        worker_linux_nvidia_2_4080_gpu(),
    ]

    m = new_model(1, "test", 1)
    mi = new_model_instance(1, "test", 1)

    resource_fit_policy = ResourceFitPolicy(m, mi)
    placement_policy = PlacementPolicy(m, mi)
    scheduler = Scheduler(config)

    with (
        patch(
            'gpustack.policies.resource_fit_policy.get_worker_model_instances',
            return_value=[],
        ),
        patch(
            'gpustack.policies.resource_fit_policy.calculate_model_resource_claim',
            side_effect=mock_calculate_model_resource_claim,
        ),
    ):

        candidates = await resource_fit_policy.filter(workers)
        candidates = await placement_policy.score(candidates)
        candidate = await scheduler.find_candidate(mi, m, workers)

        expected_candidates = [
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
            # non uma
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
        ]

        assert len(candidates) == 4
        assert candidate == candidates[2]
        compare_candidates(candidates, expected_candidates)


@pytest.mark.asyncio
async def test_schedule_to_single_worker_multi_gpu(config):
    workers = [
        worker_linux_nvidia_1_4090_gpu(),
        worker_linux_nvidia_2_4080_gpu(),
        worker_linux_nvidia_2_4090_gpu(),
        worker_linux_nvidia_4_4080_gpu(),
    ]

    m = new_model(1, "test", 1, "llama3:70b")
    mi = new_model_instance(1, "test", 1)

    resource_fit_policy = ResourceFitPolicy(m, mi)
    placement_policy = PlacementPolicy(m, mi)
    scheduler = Scheduler(config)

    with (
        patch(
            'gpustack.policies.resource_fit_policy.get_worker_model_instances',
            return_value=[],
        ),
        patch(
            'gpustack.policies.resource_fit_policy.calculate_model_resource_claim',
            side_effect=mock_calculate_model_resource_claim,
        ),
    ):

        # filter
        candidates = await resource_fit_policy.filter(workers)
        candidates = await placement_policy.score(candidates)
        candidate = await scheduler.find_candidate(mi, m, workers)

        expected_candidates = [
            {
                "offload_layers": 81,
                "worker_id": 4,
                "worker_name": "host-2-4090",
                "gpu_indexes": [0, 1],
                "is_unified_memory": False,
                "ram": 455165112,
                "vram": {0: 22912443392, 1: 22911897600},
                "score": 58.828511417553905,
            },
        ]

        assert len(candidates) == 1
        assert candidate == candidates[0]
        compare_candidates(candidates, expected_candidates)


@pytest.mark.asyncio
async def test_schedule_to_single_worker_multi_gpu_with_binpack_spread(config):
    workers = [
        worker_linux_nvidia_1_4090_gpu(),
        worker_linux_nvidia_2_4080_gpu(),
        worker_linux_nvidia_4_4080_gpu(),
    ]

    m = new_model(1, "test", 1, "llama3:70b", PlacementStrategyEnum.BINPACK)
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

    resource_fit_policy_binpack = ResourceFitPolicy(m, mi_binpack)
    placement_policy_binpack = PlacementPolicy(m, mi_binpack)

    resource_fit_policy_spread = ResourceFitPolicy(m, mi_spread)
    placement_policy_spread = PlacementPolicy(m, mi_spread)

    scheduler = Scheduler(config)

    with (
        patch(
            'gpustack.policies.resource_fit_policy.get_worker_model_instances',
            return_value=mis,
        ),
        patch(
            'gpustack.policies.resource_fit_policy.calculate_model_resource_claim',
            side_effect=mock_calculate_model_resource_claim,
        ),
        patch(
            'gpustack.policies.placement_policy.get_model_instances',
            return_value=mis,
        ),
    ):

        # binpack
        binpack_candidates = await resource_fit_policy_binpack.filter(workers)
        binpack_candidates = await placement_policy_binpack.score(binpack_candidates)
        binpack_candidate = await scheduler.find_candidate(mi_binpack, m, workers)

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
            },
        ]

        assert len(binpack_candidates) == 4
        assert binpack_candidate == binpack_candidates[2]
        compare_candidates(binpack_candidates, expected_candidates)

        # spread
        m.placement_strategy = PlacementStrategyEnum.SPREAD
        spread_candidates = await resource_fit_policy_spread.filter(workers)
        spread_candidates = await placement_policy_spread.score(spread_candidates)
        spread_candidate = await scheduler.find_candidate(mi_spread, m, workers)

        expected_candidates = [
            {
                "gpu_indexes": [0, 1, 2],
                "score": 85.0,
            },
            {
                "gpu_indexes": [0, 1, 3],
                "score": 85.0,
            },
            {
                "gpu_indexes": [0, 2, 3],
                "score": 84.0,
            },
            {
                "gpu_indexes": [1, 2, 3],
                "score": 84.0,
            },
        ]

        assert len(spread_candidates) == 4
        assert spread_candidate == spread_candidates[0]
        compare_candidates(spread_candidates, expected_candidates)


@pytest.mark.asyncio
async def test_schedule_to_single_worker_multi_gpu_partial_offload(config):
    workers = [
        worker_linux_nvidia_1_4090_gpu(),
        worker_linux_nvidia_2_4080_gpu(),
    ]

    m = new_model(1, "test", 1, "llama3:70b", cpu_offloading=True)
    mi_binpack = new_model_instance(1, "test_binpack", 1)

    resource_fit_policy_binpack = ResourceFitPolicy(m, mi_binpack)
    placement_policy_binpack = PlacementPolicy(m, mi_binpack)
    scheduler = Scheduler(config)

    with (
        patch(
            'gpustack.policies.resource_fit_policy.get_worker_model_instances',
            return_value=[],
        ),
        patch(
            'gpustack.policies.resource_fit_policy.calculate_model_resource_claim',
            side_effect=mock_calculate_model_resource_claim,
        ),
        patch(
            'gpustack.policies.placement_policy.get_model_instances',
            return_value=[],
        ),
    ):

        # binpack
        binpack_candidates = await resource_fit_policy_binpack.filter(workers)
        binpack_candidates = await placement_policy_binpack.score(binpack_candidates)
        binpack_candidate = await scheduler.find_candidate(mi_binpack, m, workers)

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
            }
        ]

        assert len(binpack_candidates) == 1
        assert binpack_candidate == binpack_candidates[0]
        compare_candidates(binpack_candidates, expected_candidates)


@pytest.mark.asyncio
async def test_schedule_to_cpu_with_binpack_spread(config):
    workers = [
        worker_linux_cpu_1(),
        worker_linux_cpu_2(),
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
    es = mock_calculate_model_resource_claim(
        mi_binpack, m, offload=GPUOffloadEnum.Disable
    )
    resource_fit_policy_binpack = ResourceFitPolicy(
        m, mi_binpack, estimate=es.resource_claim_estimate
    )
    placement_policy_binpack = PlacementPolicy(m, mi_binpack)
    scheduler = Scheduler(config)

    with (
        patch(
            'gpustack.policies.resource_fit_policy.get_worker_model_instances',
            return_value=mis,
        ),
        patch(
            'gpustack.policies.resource_fit_policy.calculate_model_resource_claim',
            side_effect=mock_calculate_model_resource_claim,
        ),
        patch(
            'gpustack.policies.placement_policy.get_model_instances',
            return_value=mis,
        ),
    ):

        # binpack
        binpack_candidates = await resource_fit_policy_binpack.filter(workers)
        binpack_candidates = await placement_policy_binpack.score(binpack_candidates)
        binpack_candidate = await scheduler.find_candidate(
            mi_binpack, m, workers, estimate=es.resource_claim_estimate
        )

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

        resource_fit_policy_spread = ResourceFitPolicy(
            m, mi_spread, estimate=es.resource_claim_estimate
        )
        placement_policy_spread = PlacementPolicy(m, mi_spread)

        spread_candidates = await resource_fit_policy_spread.filter(workers)
        spread_candidates = await placement_policy_spread.score(spread_candidates)
        spread_candidate = await scheduler.find_candidate(
            mi_spread, m, workers, estimate=es.resource_claim_estimate
        )

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
        worker_linux_nvidia_1_4090_gpu(),
        worker_linux_nvidia_2_4080_gpu(),
    ]

    m = new_model(1, "test", 1, "llama3:70b", cpu_offloading=False)
    mi_binpack = new_model_instance(1, "test_binpack", 1)
    es = mock_calculate_model_resource_claim(mi_binpack, m)

    resource_fit_policy_binpack = ResourceFitPolicy(
        m, mi_binpack, estimate=es.resource_claim_estimate
    )
    placement_policy_binpack = PlacementPolicy(m, mi_binpack)
    scheduler = Scheduler(config)

    with (
        patch(
            'gpustack.policies.resource_fit_policy.get_worker_model_instances',
            return_value=[],
        ),
        patch(
            'gpustack.policies.resource_fit_policy.calculate_model_resource_claim',
            side_effect=mock_calculate_model_resource_claim,
        ),
        patch(
            'gpustack.policies.placement_policy.get_model_instances',
            return_value=[],
        ),
        patch('sqlmodel.ext.asyncio.session.AsyncSession', AsyncMock()),
        patch(
            'gpustack.schemas.workers.Worker.all',
            return_value=workers,
        ),
    ):

        # binpack
        binpack_candidates = await resource_fit_policy_binpack.filter(workers)
        binpack_candidates = await placement_policy_binpack.score(binpack_candidates)
        binpack_candidate = await scheduler.find_candidate(
            mi_binpack, m, workers, estimate=es.resource_claim_estimate
        )

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
            {
                "offload_layers": 81,
                "worker_id": 2,
                "worker_name": "host4090",
                "gpu_indexes": [0],
                "is_unified_memory": False,
                "ram": 421610680,
                "vram": {
                    0: 19475402752,
                },
                "score": 65.6816841222221,
                "rpc_servers": [
                    ModelInstanceRPCServer(
                        worker_id=3,
                        gpu_index=0,
                        computed_resource_claim=ComputedResourceClaim(
                            is_unified_memory=False,
                            offload_layers=23,
                            total_layers=81,
                            ram=0,
                            vram={0: 13296406528},
                        ),
                    ),
                    ModelInstanceRPCServer(
                        worker_id=3,
                        gpu_index=1,
                        computed_resource_claim=ComputedResourceClaim(
                            is_unified_memory=False,
                            offload_layers=22,
                            total_layers=81,
                            ram=0,
                            vram={0: 13643949056},
                        ),
                    ),
                ],
            },
        ]

        assert len(binpack_candidates) == 2
        assert binpack_candidate == binpack_candidates[1]
        compare_candidates(binpack_candidates, expected_candidates)


def mock_calculate_model_resource_claim(
    model_instance: ModelInstance,
    model: Model,
    offload: GPUOffloadEnum = GPUOffloadEnum.Full,
    **kwargs,
) -> ModelInstanceResourceClaim:
    mock_estimate = AsyncMock()
    tensor_split = kwargs.get("tensor_split")
    if offload == GPUOffloadEnum.Full:
        mock_estimate = llama3_70b_full_offload_estimate_claim()
        if tensor_split:
            mapping = {
                (
                    17171480576,
                    17171480576,
                ): llama3_70b_full_offload_split_2_4080_estimate_claim,
                (
                    17171480576,
                    16647192576,
                ): llama3_70b_full_offload_split_2_4080_estimate_claim_2,
                (
                    17171480576,
                    16542334976,
                ): llama3_70b_full_offload_split_2_4080_estimate_claim_2,
                (
                    16647192576,
                    16542334976,
                ): llama3_70b_full_offload_split_2_4080_estimate_claim_3,
                (
                    26015170560,
                    26015170560,
                ): llama3_70b_full_offload_split_2_4090_estimate_claim,
                (
                    17171480576,
                    17171480576,
                    17171480576,
                ): llama3_70b_full_offload_split_3_4080_estimate_claim_3,
                (
                    17171480576,
                    17171480576,
                    16647192576,
                ): llama3_70b_full_offload_split_3_4080_estimate_claim,
                (
                    17171480576,
                    17171480576,
                    16542334976,
                ): llama3_70b_full_offload_split_3_4080_estimate_claim,
                (
                    17171480576,
                    16647192576,
                    16542334976,
                ): llama3_70b_full_offload_split_3_4080_estimate_claim_2,
                (
                    17171480576,
                    17171480576,
                    17171480576,
                    17171480576,
                ): llama3_70b_full_offload_split_4_4080_estimate_claim,
            }

            mock_estimate = mapping[tuple(tensor_split)]()

        if model.ollama_library_model_name == "llama3:8b":
            mock_estimate = llama3_8b_full_offload_estimate_claim()
    elif offload == GPUOffloadEnum.Partial:
        mock_estimate = llama3_70b_partial_offload_estimate_claim()
        if tensor_split:
            mapping = {
                (
                    17171480576,
                    17171480576,
                ): llama3_70b_partial_offload_split_2_4080_estimate_claim,
                (
                    17171480576,
                    26015170560,
                ): llama3_70b_partial_offload_split_2_4080_4090_estimate_claim,
                (
                    26015170560,
                    17171480576,
                    17171480576,
                ): llama3_70b_partial_offload_split_3_4080_4090_estimate_claim,
                (
                    17171480576,
                    17171480576,
                    26015170560,
                ): llama3_70b_partial_offload_split_3_4080_4090_estimate_claim_2,
            }
            mock_estimate = mapping[tuple(tensor_split)]()

        if model.ollama_library_model_name == "llama3:8b":
            mock_estimate = llama3_8b_partial_offload_estimate_claim()
    elif offload == GPUOffloadEnum.Disable:
        mock_estimate = llama3_70b_disable_offload_estimate_claim()
        if model.ollama_library_model_name == "llama3:8b":
            return llama3_8b_disable_offload_estimate_claim()
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

        if "instance_id" in expected:
            assert candidate.instance.id == expected["instance_id"]

        if "rpc_servers" in expected:
            for i, rpc_server in enumerate(expected["rpc_servers"]):
                assert rpc_server.worker_id == expected["rpc_servers"][i].worker_id
                assert rpc_server.gpu_index == expected["rpc_servers"][i].gpu_index
                assert (
                    rpc_server.computed_resource_claim
                    == expected["rpc_servers"][i].computed_resource_claim
                )
