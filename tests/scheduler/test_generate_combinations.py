from typing import Dict, Tuple
import pytest
from gpustack.policies.candidate_selectors.gguf_resource_fit_selector import (
    GGUFResourceFitSelector,
)

from gpustack.schemas.models import (
    GPUSelector,
    PlacementStrategyEnum,
)
from tests.fixtures.workers.fixtures import (
    linux_nvidia_10_3090x8,
    linux_nvidia_8_3090x8,
    linux_nvidia_9_3090x8,
)

from tests.utils.model import new_model, new_model_instance
from unittest.mock import patch


@pytest.mark.asyncio
async def test_generate_combinations_for_single_worker_gpus():
    workers = [
        linux_nvidia_8_3090x8(),
    ]

    m = new_model(1, "test", 1, "llama3:70b")
    mi = new_model_instance(1, "test", 1)

    resource_fit_selector = GGUFResourceFitSelector(m, mi)

    with (
        patch(
            'gpustack.policies.utils.get_worker_model_instances',
            return_value=[],
        ),
    ):

        allocatable = await resource_fit_selector._get_worker_allocatable_resource(
            workers[0]
        )
        combinations = (
            await resource_fit_selector._generate_combinations_for_single_worker_gpus(
                allocatable, workers[0]
            )
        )

        expected_total = 247
        expected_combinations = {
            # key: gpu count, value: combinations number
            2: 28,
            3: 56,
            4: 70,
            5: 56,
            6: 28,
            7: 8,
            8: 1,
        }

    compare_combinations(combinations, expected_combinations, expected_total)


@pytest.mark.asyncio
async def test_generate_combinations_for_worker_with_rpc_servers_with_manual_selected_gpus():
    workers = [
        linux_nvidia_8_3090x8(),
        linux_nvidia_9_3090x8(),
        linux_nvidia_10_3090x8(),
    ]

    m = new_model(
        1,
        "DeepSeek-R1-GGUF",
        1,
        huggingface_repo_id="unsloth/DeepSeek-R1-GGUF",
        cpu_offloading=False,
        gpu_selector=GPUSelector(
            gpu_ids=[
                # host01
                "host01-3090:cuda:0",
                "host01-3090:cuda:1",
                "host01-3090:cuda:2",
                "host01-3090:cuda:3",
                "host01-3090:cuda:4",
                "host01-3090:cuda:5",
                "host01-3090:cuda:6",
                "host01-3090:cuda:7",
                # host02
                "host02-3090:cuda:0",
                "host02-3090:cuda:1",
                "host02-3090:cuda:2",
                "host02-3090:cuda:3",
                "host02-3090:cuda:4",
                "host02-3090:cuda:5",
                "host02-3090:cuda:6",
                "host02-3090:cuda:7",
                # host03
                "host03-3090:cuda:0",
                "host03-3090:cuda:1",
                "host03-3090:cuda:2",
                "host03-3090:cuda:3",
                "host03-3090:cuda:4",
                "host03-3090:cuda:5",
                "host03-3090:cuda:6",
                "host03-3090:cuda:7",
            ]
        ),
        placement_strategy=PlacementStrategyEnum.SPREAD,
        huggingface_filename="DeepSeek-R1-Q4_K_M/DeepSeek-R1-Q4_K_M-00001-of-00009.gguf",
    )
    mi = new_model_instance(1, "test", 1)
    resource_fit_selector = GGUFResourceFitSelector(m, mi)

    with (
        patch(
            'gpustack.policies.utils.get_worker_model_instances',
            return_value=[],
        ),
    ):

        combinations, _, _ = (
            await resource_fit_selector._generate_combinations_for_worker_with_rpc_servers(
                workers
            )
        )

        expected_total = 3
        expected_combinations = {
            # key: gpu count, value: combinations number
            17: 3,
        }

    compare_combinations(combinations, expected_combinations, expected_total)


@pytest.mark.asyncio
async def test_generate_combinations_for_worker_with_rpc_servers_with_auto_selected_gpus():
    workers = [
        linux_nvidia_8_3090x8(),
        linux_nvidia_9_3090x8(),
        linux_nvidia_10_3090x8(),
    ]

    m = new_model(
        1,
        "DeepSeek-R1-GGUF",
        1,
        huggingface_repo_id="unsloth/DeepSeek-R1-GGUF",
        cpu_offloading=False,
        placement_strategy=PlacementStrategyEnum.SPREAD,
        huggingface_filename="DeepSeek-R1-Q4_K_M/DeepSeek-R1-Q4_K_M-00001-of-00009.gguf",
    )
    mi = new_model_instance(1, "test", 1)
    resource_fit_selector = GGUFResourceFitSelector(m, mi)

    with (
        patch(
            'gpustack.policies.utils.get_worker_model_instances',
            return_value=[],
        ),
    ):

        combinations, _, _ = (
            await resource_fit_selector._generate_combinations_for_worker_with_rpc_servers(
                workers
            )
        )

        expected_total = 117606
        expected_combinations = {
            # key: gpu count, value: combinations number
            2: 48,
            3: 360,
            4: 1680,
            5: 5460,
            6: 13104,
            7: 24024,
            8: 34320,
            9: 38610,
        }

    compare_combinations(combinations, expected_combinations, expected_total)


def compare_combinations(
    combinations: dict[Tuple[Tuple[int]]],
    expected_combinations: Dict[int, int],
    expected_total: int,
):
    actual_total = 0
    for e_gpu_count, e_comb_num in expected_combinations.items():
        a_comb = combinations[e_gpu_count]
        actual_total += len(a_comb)

        assert len(a_comb) == e_comb_num

    assert actual_total == expected_total
