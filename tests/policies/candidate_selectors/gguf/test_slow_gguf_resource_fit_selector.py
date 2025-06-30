import logging
import os
import random
import subprocess
from typing import Dict, List, Optional
import uuid
import pytest
from gpustack.config.config import Config, set_global_config
from gpustack.logging import setup_logging
from gpustack.policies.scorers.placement_scorer import PlacementScorer
from gpustack.policies.candidate_selectors import GGUFResourceFitSelector

from gpustack.scheduler.calculator import GPUOffloadEnum, _gguf_parser_command
from gpustack.schemas.models import (
    ComputedResourceClaim,
    GPUSelector,
    Model,
    ModelInstanceRPCServer,
    PlacementStrategyEnum,
)
from gpustack.schemas.workers import (
    GPUCoreInfo,
    MemoryInfo,
    SystemReserved,
    VendorEnum,
    Worker,
    WorkerStatus,
)
from gpustack.utils.platform import device_type_from_vendor
from tests.fixtures.workers.fixtures import (
    linux_cpu_1,
    linux_cpu_2,
    linux_cpu_3,
    linux_nvidia_10_3090_24gx8,
    linux_nvidia_11_V100_32gx2,
    linux_nvidia_12_A40_48gx2,
    linux_nvidia_13_A100_80gx8,
    linux_nvidia_14_A100_40gx2,
    linux_nvidia_15_4080_16gx8,
    linux_nvidia_16_5000_16gx8,
    linux_nvidia_17_4090_24gx8,
    linux_nvidia_18_4090_24gx4_4080_16gx4,
    linux_nvidia_1_4090_24gx1,
    linux_nvidia_20_3080_12gx8,
    linux_nvidia_21_4090_24gx4_3060_12gx4,
    linux_nvidia_2_4080_16gx2,
    linux_nvidia_19_4090_24gx2,
    linux_nvidia_3_4090_24gx1,
    linux_nvidia_4_4080_16gx4,
    linux_nvidia_8_3090_24gx8,
    linux_nvidia_9_3090_24gx8,
    linux_rocm_1_7800_16gx1,
    macos_metal_1_m1pro_21g,
    macos_metal_2_m2_24g,
)

from unittest.mock import patch, AsyncMock

from tests.policies.candidate_selectors.gguf.test_gguf_resource_fit_selector import (
    compare_candidates,
)
from tests.utils.model import new_model, new_model_instance

setup_logging(debug=True)

# Required: gguf-parser 0.13.10
# Set GGUF_PARSER_PATH environment variable to the path of gguf-parser executable


def check_parser(version: Optional[str] = "v0.13.10") -> bool:
    parser_path = os.getenv("GGUF_PARSER_PATH")
    if not parser_path:
        return False

    result = subprocess.run(
        [parser_path, "--version"], capture_output=True, text=True, check=True
    )
    parser_version = result.stdout.strip().split(" ")[-1]
    if parser_version != version:
        logging.error(
            f"Parser version mismatch: expected {version}, got {parser_version}"
        )
        return False

    return True


@pytest.mark.asyncio
async def test_schedule_with_deepseek_r1_bf16_end_in_multi_worker_multi_gpu_partial_offload(
    config,
):
    if not check_parser():
        pytest.skip("parser path is not available or version mismatch, skipping.")

    workers = [
        linux_nvidia_10_3090_24gx8(),
        linux_nvidia_11_V100_32gx2(),
        linux_nvidia_12_A40_48gx2(),
        linux_nvidia_13_A100_80gx8(),
        linux_nvidia_14_A100_40gx2(),
    ]

    m = new_model(
        1,
        "DeepSeek-R1-GGUF",
        1,
        huggingface_repo_id="unsloth/DeepSeek-R1-GGUF",
        cpu_offloading=True,
        placement_strategy=PlacementStrategyEnum.SPREAD,
        huggingface_filename="DeepSeek-R1-BF16/DeepSeek-R1.BF16-00001-of-00030.gguf",
        backend_parameters=["--ctx-size=32768"],
    )

    resource_fit_selector = GGUFResourceFitSelector(m)
    placement_scorer_spread = PlacementScorer(m)

    with (
        patch(
            'gpustack.policies.utils.get_worker_model_instances',
            return_value=[],
        ),
        patch(
            'gpustack.scheduler.calculator._gguf_parser_command',
            side_effect=mock_gguf_parser_command,
        ),
        patch(
            'gpustack.scheduler.calculator.hf_model_filename',
            return_value="DeepSeek-R1-BF16/DeepSeek-R1.BF16-00001-of-00030.gguf",
        ),
        patch(
            'gpustack.scheduler.calculator.hf_mmproj_filename',
            return_value=None,
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
        candidates = await placement_scorer_spread.score(candidates)

        expected_candidates = [
            {
                "offload_layers": 22,
                "worker_id": 19,
                "worker_name": "host01-a100",
                "gpu_indexes": [0, 1, 2, 3, 4, 5, 6, 7],
                "is_unified_memory": False,
                "ram": 940869368712,
                "vram": {
                    0: 77743332352,
                    1: 52040701952,
                    2: 77743332352,
                    3: 52040701952,
                    4: 77743332352,
                    5: 52040701952,
                    6: 77743332352,
                    7: 52040701952,
                },
                "score": 100,
                "tensor_split": [
                    34359738368,
                    34359738368,
                    85899345920,
                    85899345920,
                    85899345920,
                    85899345920,
                    85899345920,
                    85899345920,
                    85899345920,
                    85899345920,
                ],
                "rpc_servers": [
                    ModelInstanceRPCServer(
                        worker_id=17,
                        gpu_index=0,
                        computed_resource_claim=ComputedResourceClaim(
                            is_unified_memory=False,
                            offload_layers=1,
                            total_layers=62,
                            ram=0,
                            vram={0: 28426863616},
                        ),
                    ),
                    ModelInstanceRPCServer(
                        worker_id=17,
                        gpu_index=1,
                        computed_resource_claim=ComputedResourceClaim(
                            is_unified_memory=False,
                            offload_layers=1,
                            total_layers=62,
                            ram=0,
                            vram={1: 26338071552},
                        ),
                    ),
                ],
            },
        ]

        assert len(candidates) == 1
        compare_candidates(candidates, expected_candidates)


@pytest.mark.asyncio
async def test_schedule_with_deepseek_r1_bf16_end_in_patial_offload(temp_dir):
    cache_dir = os.path.join(
        os.path.dirname(__file__),
        "../../../fixtures/estimates/unsloth_DeepSeek-R1-GGUF_DeepSeek-R1-BF16",
    )
    config = Config(
        token="test",
        jwt_secret_key="test",
        data_dir=temp_dir,
        cache_dir=cache_dir,
        huggingface_token="",
    )
    set_global_config(config)

    if not check_parser():
        pytest.skip("parser path is not available or version mismatch, skipping.")

    workers = [
        linux_nvidia_2_4080_16gx2(),
        linux_nvidia_4_4080_16gx4(),
        linux_nvidia_19_4090_24gx2(),
        linux_nvidia_11_V100_32gx2(),
        linux_nvidia_12_A40_48gx2(),
        linux_nvidia_14_A100_40gx2(),
        linux_nvidia_15_4080_16gx8(),
        linux_nvidia_8_3090_24gx8(),
        linux_nvidia_9_3090_24gx8(),
        linux_nvidia_10_3090_24gx8(),
        macos_metal_2_m2_24g(),
        macos_metal_1_m1pro_21g(),
    ]

    m = new_model(
        1,
        "DeepSeek-R1-GGUF",
        1,
        huggingface_repo_id="unsloth/DeepSeek-R1-GGUF",
        cpu_offloading=True,
        placement_strategy=PlacementStrategyEnum.SPREAD,
        huggingface_filename="DeepSeek-R1-BF16/DeepSeek-R1.BF16-00001-of-00030.gguf",
        backend_parameters=["--ctx-size=32768"],
        distributed_inference_across_workers=False,
    )
    mi = new_model_instance(1, "test", 1)

    resource_fit_selector = GGUFResourceFitSelector(m, mi, cache_dir)
    placement_scorer_spread = PlacementScorer(m, mi)

    with (
        patch(
            'gpustack.policies.utils.get_worker_model_instances',
            return_value=[],
        ),
        patch(
            'gpustack.scheduler.calculator._gguf_parser_command',
            side_effect=mock_gguf_parser_command,
        ),
        patch(
            'gpustack.scheduler.calculator.hf_model_filename',
            return_value="DeepSeek-R1-BF16/DeepSeek-R1.BF16-00001-of-00030.gguf",
        ),
        patch(
            'gpustack.scheduler.calculator.hf_mmproj_filename',
            return_value=None,
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
        candidates = await placement_scorer_spread.score(candidates)

        expected_candidates = [
            {
                "offload_layers": 2,
                "worker_id": 18,
                "worker_name": "host01-a40",
                "gpu_indexes": [0, 1],
                "is_unified_memory": False,
                "ram": 1454921976712,
                "vram": {
                    0: 28426863616,
                    1: 26338071552,
                },
                "score": 100,
                "tensor_split": [51539607552, 51539607552],
            },
            {
                "offload_layers": 2,
                "worker_id": 20,
                "worker_name": "host02-a100",
                "gpu_indexes": [0, 1],
                "is_unified_memory": False,
                "ram": 1454921976712,
                "vram": {
                    0: 28426863616,
                    1: 26338071552,
                },
                "score": 100,
                "tensor_split": [42949672960, 42949672960],
            },
            {
                "offload_layers": 2,
                "worker_id": 17,
                "worker_name": "host01-v100",
                "gpu_indexes": [0, 1],
                "is_unified_memory": False,
                "ram": 1454921976712,
                "vram": {
                    0: 28426863616,
                    1: 26338071552,
                },
                "score": 100,
                "tensor_split": [34359738368, 34359738368],
            },
        ]

        assert len(candidates) == 3
        compare_candidates(candidates, expected_candidates)


@pytest.mark.asyncio
async def test_schedule_with_deepseek_r1_bf16_end_in_cpu(temp_dir):
    cache_dir = os.path.join(
        os.path.dirname(__file__),
        "../../../fixtures/estimates/unsloth_DeepSeek-R1-GGUF_DeepSeek-R1-BF16",
    )
    config = Config(
        token="test",
        jwt_secret_key="test",
        data_dir=temp_dir,
        cache_dir=cache_dir,
        huggingface_token="",
    )
    set_global_config(config)

    if not check_parser():
        pytest.skip("parser path is not available or version mismatch, skipping.")

    workers = [
        linux_nvidia_1_4090_24gx1(),
        linux_nvidia_2_4080_16gx2(),
        linux_nvidia_4_4080_16gx4(),
        linux_nvidia_3_4090_24gx1(),
        linux_nvidia_8_3090_24gx8(),
        linux_nvidia_9_3090_24gx8(),
        linux_nvidia_10_3090_24gx8(),
        linux_nvidia_15_4080_16gx8(),
        linux_nvidia_16_5000_16gx8(),
        linux_nvidia_17_4090_24gx8(),
        linux_nvidia_18_4090_24gx4_4080_16gx4(),
        linux_nvidia_19_4090_24gx2(),
        linux_nvidia_20_3080_12gx8(),
        linux_nvidia_21_4090_24gx4_3060_12gx4(),
        linux_rocm_1_7800_16gx1(),
        macos_metal_2_m2_24g(),
        macos_metal_1_m1pro_21g(),
        linux_cpu_1(),
        linux_cpu_2(),
        linux_cpu_3(),
    ]

    m = new_model(
        1,
        "DeepSeek-R1-GGUF",
        1,
        huggingface_repo_id="unsloth/DeepSeek-R1-GGUF",
        cpu_offloading=True,
        placement_strategy=PlacementStrategyEnum.SPREAD,
        huggingface_filename="DeepSeek-R1-BF16/DeepSeek-R1.BF16-00001-of-00030.gguf",
        backend_parameters=["--ctx-size=32768"],
    )
    mi = new_model_instance(1, "test", 1)

    resource_fit_selector = GGUFResourceFitSelector(m, mi, cache_dir)
    placement_scorer_spread = PlacementScorer(m, mi)

    with (
        patch(
            'gpustack.policies.utils.get_worker_model_instances',
            return_value=[],
        ),
        patch(
            'gpustack.scheduler.calculator._gguf_parser_command',
            side_effect=mock_gguf_parser_command,
        ),
        patch(
            'gpustack.scheduler.calculator.hf_model_filename',
            return_value="DeepSeek-R1-BF16/DeepSeek-R1.BF16-00001-of-00030.gguf",
        ),
        patch(
            'gpustack.scheduler.calculator.hf_mmproj_filename',
            return_value=None,
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
        candidates = await placement_scorer_spread.score(candidates)

        expected_candidates = [
            {
                "offload_layers": 0,
                "worker_id": 27,
                "worker_name": "host-cpu-3",
                "is_unified_memory": False,
                "ram": 1506297877384,
                "score": 100,
            },
            {
                "offload_layers": 0,
                "worker_id": 26,
                "worker_name": "host04-4090_24gx4_3060_12gx4",
                "is_unified_memory": False,
                "ram": 1506297877384,
                "score": 100,
            },
            {
                "offload_layers": 0,
                "worker_id": 25,
                "worker_name": "host01-3080",
                "is_unified_memory": False,
                "ram": 1506297877384,
                "score": 100,
            },
            {
                "offload_layers": 0,
                "worker_id": 23,
                "worker_name": "host03-4090",
                "is_unified_memory": False,
                "ram": 1506297877384,
                "score": 100,
            },
            {
                "offload_layers": 0,
                "worker_id": 24,
                "worker_name": "host04-4090_24gx4_4080_16gx4",
                "is_unified_memory": False,
                "ram": 1506297877384,
                "score": 100,
            },
            {
                "offload_layers": 0,
                "worker_id": 22,
                "worker_name": "host01-5000",
                "is_unified_memory": False,
                "ram": 1506297877384,
                "score": 100,
            },
            {
                "offload_layers": 0,
                "worker_id": 16,
                "worker_name": "host03-3090",
                "is_unified_memory": False,
                "ram": 1506297877384,
                "score": 100,
            },
        ]

        assert len(candidates) == 7
        compare_candidates(candidates, expected_candidates)


@pytest.mark.asyncio
async def test_schedule_with_deepseek_r1_bf16_with_30_workers(temp_dir):
    cache_dir = os.path.join(
        os.path.dirname(__file__),
        "../../../fixtures/estimates/unsloth_DeepSeek-R1-GGUF_DeepSeek-R1-BF16",
    )
    config = Config(
        token="test",
        jwt_secret_key="test",
        data_dir=temp_dir,
        cache_dir=cache_dir,
        huggingface_token="",
    )
    set_global_config(config)

    if not check_parser():
        pytest.skip("parser path is not available or version mismatch, skipping.")

    # worker_group1: 10 workers, each worker has 16 GPUs with vram in GiB: 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 82, 88, 94, 100, 106, 112
    worker_group1 = create_workers_with_gpu_step(10, 1, 2000, 16, 24, 6)
    # worker_group2: 10 workers, each worker has 16 GPUs with vram in GiB: 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72
    worker_group2 = create_workers_with_gpu_step(10, 11, 2000, 16, 12, 4)
    # worker_group3: 10 workers, each worker has 8 GPUs with vram in GiB: 30, 40, 50, 60, 70, 80, 90, 100
    worker_group3 = create_workers_with_gpu_step(10, 21, 2000, 8, 30, 10)

    workers = worker_group1 + worker_group2 + worker_group3

    m = new_model(
        1,
        "DeepSeek-R1-GGUF",
        1,
        huggingface_repo_id="unsloth/DeepSeek-R1-GGUF",
        cpu_offloading=True,
        placement_strategy=PlacementStrategyEnum.SPREAD,
        huggingface_filename="DeepSeek-R1-BF16/DeepSeek-R1.BF16-00001-of-00030.gguf",
        backend_parameters=["--ctx-size=32768"],
    )
    mi = new_model_instance(1, "test", 1)

    resource_fit_selector = GGUFResourceFitSelector(m, mi, cache_dir)
    placement_scorer_spread = PlacementScorer(m, mi)

    with (
        patch(
            'gpustack.policies.utils.get_worker_model_instances',
            return_value=[],
        ),
        patch(
            'gpustack.scheduler.calculator._gguf_parser_command',
            side_effect=mock_gguf_parser_command,
        ),
        patch(
            'gpustack.scheduler.calculator.hf_model_filename',
            return_value="DeepSeek-R1-BF16/DeepSeek-R1.BF16-00001-of-00030.gguf",
        ),
        patch(
            'gpustack.scheduler.calculator.hf_mmproj_filename',
            return_value=None,
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
        candidates = await placement_scorer_spread.score(candidates)
        assert len(candidates) == 1
        expected_candidates = [
            {
                "offload_layers": 34,
                "worker_id": 1,
                "worker_name": "host01",
                "is_unified_memory": False,
                "ram": 632437803912,
                "vram": {
                    15: 105534754816,
                    14: 103445962752,
                    13: 103445962752,
                    12: 77743332352,
                    11: 77743332352,
                    10: 77743332352,
                    9: 77743332352,
                    8: 52040701952,
                    7: 52040701952,
                    6: 52040701952,
                    5: 52040701952,
                    2: 26338071552,
                    1: 26338071552,
                },
                "score": 100,
                "gpu_indexes": [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 2, 1],
                "tensor_split": [
                    122406567936,
                    115964116992,
                    109521666048,
                    103079215104,
                    96636764160,
                    90194313216,
                    83751862272,
                    77309411328,
                    70866960384,
                    64424509440,
                    57982058496,
                    38654705664,
                    32212254720,
                ],
            },
        ]

        compare_candidates(candidates, expected_candidates)


@pytest.mark.asyncio
async def test_schedule_with_deepseek_r1_bf16_with_end_in_no_candidate(temp_dir):
    cache_dir = os.path.join(
        os.path.dirname(__file__),
        "../../../fixtures/estimates/unsloth_DeepSeek-R1-GGUF_DeepSeek-R1-BF16",
    )
    config = Config(
        token="test",
        jwt_secret_key="test",
        data_dir=temp_dir,
        cache_dir=cache_dir,
        huggingface_token="",
    )
    set_global_config(config)

    if not check_parser():
        pytest.skip("parser path is not available or version mismatch, skipping.")

    # worker_group1: 10 workers, each worker has 16 GPUs with vram in GiB: 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 82, 88, 94, 100, 106, 112
    worker_group1 = create_workers_with_gpu_step(10, 1, 2000, 16, 24, 6)
    # worker_group2: 10 workers, each worker has 16 GPUs with vram in GiB: 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72
    worker_group2 = create_workers_with_gpu_step(10, 11, 2000, 16, 12, 4)
    # worker_group3: 10 workers, each worker has 8 GPUs with vram in GiB: 30, 40, 50, 60, 70, 80, 90, 100
    worker_group3 = create_workers_with_gpu_step(10, 21, 2000, 8, 30, 10)

    workers = worker_group1 + worker_group2 + worker_group3

    m = new_model(
        1,
        "DeepSeek-R1-GGUF",
        1,
        huggingface_repo_id="unsloth/DeepSeek-R1-GGUF",
        placement_strategy=PlacementStrategyEnum.SPREAD,
        huggingface_filename="DeepSeek-R1-BF16/DeepSeek-R1.BF16-00001-of-00030.gguf",
        cpu_offloading=False,
        backend_parameters=["--ctx-size=32768"],
    )
    mi = new_model_instance(1, "test", 1)

    resource_fit_selector = GGUFResourceFitSelector(m, mi, cache_dir)

    with (
        patch(
            'gpustack.policies.utils.get_worker_model_instances',
            return_value=[],
        ),
        patch(
            'gpustack.scheduler.calculator._gguf_parser_command',
            side_effect=mock_gguf_parser_command,
        ),
        patch(
            'gpustack.scheduler.calculator.hf_model_filename',
            return_value="DeepSeek-R1-BF16/DeepSeek-R1.BF16-00001-of-00030.gguf",
        ),
        patch(
            'gpustack.scheduler.calculator.hf_mmproj_filename',
            return_value=None,
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
        messages = resource_fit_selector.get_messages()

        assert len(candidates) == 0
        assert (
            messages[0]
            == "The model requires approximately 1410.55 GiB VRAM and 3.74 GiB RAM. The largest available worker provides 1104.0 GiB VRAM and 2000.0 GiB RAM."
        )
        assert (
            messages[1]
            == "Too many candidate RPC servers, skipping distributed deployment. Use manual scheduling to select GPUs if needed."
        )


@pytest.mark.asyncio
async def test_schedule_with_deepseek_r1_q8_0_with_end_with_workerx2x80gx8(temp_dir):
    cache_dir = os.path.join(
        os.path.dirname(__file__),
        "../../../fixtures/estimates/unsloth_DeepSeek-R1-GGUF_DeepSeek-R1-Q8_0",
    )
    config = Config(
        token="test",
        jwt_secret_key="test",
        data_dir=temp_dir,
        cache_dir=cache_dir,
        huggingface_token="",
    )
    set_global_config(config)

    if not check_parser():
        pytest.skip("parser path is not available or version mismatch, skipping.")

    workers = create_workers_with_gpu_step(2, 1, 2000, 8, 80, 0)

    m = new_model(
        1,
        "DeepSeek-R1-GGUF",
        1,
        huggingface_repo_id="unsloth/DeepSeek-R1-GGUF",
        placement_strategy=PlacementStrategyEnum.SPREAD,
        huggingface_filename="DeepSeek-R1-Q8_0/DeepSeek-R1.Q8_0-00001-of-00015.gguf",
        cpu_offloading=True,
        distributable=True,
        backend_parameters=[],
    )

    mi = new_model_instance(1, "test", 1)

    resource_fit_selector = GGUFResourceFitSelector(m, mi, cache_dir)
    placement_scorer_spread = PlacementScorer(m, mi)

    with (
        patch(
            'gpustack.policies.utils.get_worker_model_instances',
            return_value=[],
        ),
        patch(
            'gpustack.scheduler.calculator._gguf_parser_command',
            side_effect=mock_gguf_parser_command,
        ),
        patch(
            'gpustack.scheduler.calculator.hf_model_filename',
            return_value="DeepSeek-R1-Q8_0/DeepSeek-R1.Q8_0-00001-of-00015.gguf",
        ),
        patch(
            'gpustack.scheduler.calculator.hf_mmproj_filename',
            return_value=None,
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
        candidates = await placement_scorer_spread.score(candidates)
        expected_candidates = [
            {
                "offload_layers": 62,
                "worker_id": 1,
                "worker_name": "host01",
                "is_unified_memory": False,
                "ram": 2232209448,
                "vram": {
                    0: 78003226624,
                    1: 78003226624,
                    2: 65100207104,
                    3: 78003226624,
                    4: 78003226624,
                    5: 65100207104,
                    6: 78003226624,
                    7: 53417763840,
                },
                "score": 100,
                "gpu_indexes": [0, 1, 2, 3, 4, 5, 6, 7],
                "tensor_split": [
                    85899345920,
                    85899345920,
                    85899345920,
                    85899345920,
                    85899345920,
                    85899345920,
                    85899345920,
                    85899345920,
                    85899345920,
                    85899345920,
                    85899345920,
                ],
                "rpc_servers": [
                    ModelInstanceRPCServer(
                        worker_id=2,
                        gpu_index=0,
                        computed_resource_claim=ComputedResourceClaim(
                            is_unified_memory=False,
                            offload_layers=6,
                            total_layers=62,
                            ram=0,
                            vram={0: 43167431680},
                        ),
                    ),
                    ModelInstanceRPCServer(
                        worker_id=2,
                        gpu_index=1,
                        computed_resource_claim=ComputedResourceClaim(
                            is_unified_memory=False,
                            offload_layers=6,
                            total_layers=62,
                            ram=0,
                            vram={0: 78003226624},
                        ),
                    ),
                    ModelInstanceRPCServer(
                        worker_id=2,
                        gpu_index=2,
                        computed_resource_claim=ComputedResourceClaim(
                            is_unified_memory=False,
                            offload_layers=5,
                            total_layers=62,
                            ram=0,
                            vram={0: 65100207104},
                        ),
                    ),
                ],
            },
        ]

        compare_candidates([candidates[0]], expected_candidates)


@pytest.mark.asyncio
async def test_schedule_with_deepseek_r1_q8_0_with_ngl_with_end_in_multi_worker_multi_gpu(
    temp_dir,
):
    cache_dir = os.path.join(
        os.path.dirname(__file__),
        "../../../fixtures/estimates/unsloth_DeepSeek-R1-GGUF_DeepSeek-R1-Q8_0",
    )
    config = Config(
        token="test",
        jwt_secret_key="test",
        data_dir=temp_dir,
        cache_dir=cache_dir,
        huggingface_token="",
    )
    set_global_config(config)

    if not check_parser():
        pytest.skip("parser path is not available or version mismatch, skipping.")

    workers = create_workers_with_gpu_step(2, 1, 2000, 8, 80, 0)

    m = new_model(
        1,
        "DeepSeek-R1-GGUF",
        1,
        huggingface_repo_id="unsloth/DeepSeek-R1-GGUF",
        placement_strategy=PlacementStrategyEnum.SPREAD,
        huggingface_filename="DeepSeek-R1-Q8_0/DeepSeek-R1.Q8_0-00001-of-00015.gguf",
        cpu_offloading=True,
        distributable=True,
        backend_parameters=["--ngl=50"],
    )

    mi = new_model_instance(1, "test", 1)

    resource_fit_selector = GGUFResourceFitSelector(m, mi, cache_dir)
    placement_scorer_spread = PlacementScorer(m, mi)

    with (
        patch(
            'gpustack.policies.utils.get_worker_model_instances',
            return_value=[],
        ),
        patch(
            'gpustack.scheduler.calculator._gguf_parser_command',
            side_effect=mock_gguf_parser_command,
        ),
        patch(
            'gpustack.scheduler.calculator.hf_model_filename',
            return_value="DeepSeek-R1-Q8_0/DeepSeek-R1.Q8_0-00001-of-00015.gguf",
        ),
        patch(
            'gpustack.scheduler.calculator.hf_mmproj_filename',
            return_value=None,
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
        candidates = await placement_scorer_spread.score(candidates)
        assert len(candidates) == 8
        expected_candidates = [
            {
                "offload_layers": 50,
                "worker_id": 1,
                "worker_name": "host01",
                "is_unified_memory": False,
                "ram": 109330175016,
                "vram": {
                    0: 78003226624,
                    1: 65100207104,
                    2: 78003226624,
                    3: 65100207104,
                    4: 78003226624,
                    5: 65100207104,
                    6: 78003226624,
                    7: 65100207104,
                },
                "score": 100,
                "gpu_indexes": [0, 1, 2, 3, 4, 5, 6, 7],
                "tensor_split": [
                    85899345920,
                    85899345920,
                    85899345920,
                    85899345920,
                    85899345920,
                    85899345920,
                    85899345920,
                    85899345920,
                    85899345920,
                ],
                "rpc_servers": [
                    ModelInstanceRPCServer(
                        worker_id=2,
                        gpu_index=0,
                        computed_resource_claim=ComputedResourceClaim(
                            is_unified_memory=False,
                            offload_layers=6,
                            total_layers=62,
                            ram=0,
                            vram={0: 79223257088},
                        ),
                    ),
                ],
            },
        ]

        compare_candidates([candidates[0]], expected_candidates)


@pytest.mark.asyncio
async def test_schedule_with_ngl_end_in_patial_offload(temp_dir):
    cache_dir = os.path.join(
        os.path.dirname(__file__),
        "../../../fixtures/estimates/unsloth_DeepSeek-R1-GGUF_DeepSeek-R1-BF16",
    )
    config = Config(
        token="test",
        jwt_secret_key="test",
        data_dir=temp_dir,
        cache_dir=cache_dir,
        huggingface_token="",
    )
    set_global_config(config)

    if not check_parser():
        pytest.skip("parser path is not available or version mismatch, skipping.")

    workers = [
        linux_nvidia_11_V100_32gx2(),
        linux_nvidia_12_A40_48gx2(),
        linux_nvidia_14_A100_40gx2(),
        linux_nvidia_8_3090_24gx8(),
        linux_nvidia_9_3090_24gx8(),
        linux_nvidia_10_3090_24gx8(),
    ]

    m = new_model(
        1,
        "DeepSeek-R1-GGUF",
        1,
        huggingface_repo_id="unsloth/DeepSeek-R1-GGUF",
        cpu_offloading=True,
        placement_strategy=PlacementStrategyEnum.SPREAD,
        huggingface_filename="DeepSeek-R1-BF16/DeepSeek-R1.BF16-00001-of-00030.gguf",
        backend_parameters=["--ctx-size=32768", "--ngl=2"],
        distributed_inference_across_workers=False,
    )
    mi = new_model_instance(1, "test", 1)

    resource_fit_selector = GGUFResourceFitSelector(m, mi, cache_dir)
    placement_scorer_spread = PlacementScorer(m, mi)

    with (
        patch(
            'gpustack.policies.utils.get_worker_model_instances',
            return_value=[],
        ),
        patch(
            'gpustack.scheduler.calculator._gguf_parser_command',
            side_effect=mock_gguf_parser_command,
        ),
        patch(
            'gpustack.scheduler.calculator.hf_model_filename',
            return_value="DeepSeek-R1-BF16/DeepSeek-R1.BF16-00001-of-00030.gguf",
        ),
        patch(
            'gpustack.scheduler.calculator.hf_mmproj_filename',
            return_value=None,
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
        candidates = await placement_scorer_spread.score(candidates)

        expected_candidates = [
            {
                "offload_layers": 2,
                "worker_id": 18,
                "worker_name": "host01-a40",
                "gpu_indexes": [0, 1],
                "is_unified_memory": False,
                "ram": 1454921976712,
                "vram": {
                    0: 28426863616,
                    1: 26338071552,
                },
                "score": 100,
                "tensor_split": [51539607552, 51539607552],
            },
            {
                "offload_layers": 2,
                "worker_id": 20,
                "worker_name": "host02-a100",
                "gpu_indexes": [0, 1],
                "is_unified_memory": False,
                "ram": 1454921976712,
                "vram": {
                    0: 28426863616,
                    1: 26338071552,
                },
                "score": 100,
                "tensor_split": [42949672960, 42949672960],
            },
            {
                "offload_layers": 2,
                "worker_id": 17,
                "worker_name": "host01-v100",
                "gpu_indexes": [0, 1],
                "is_unified_memory": False,
                "ram": 1454921976712,
                "vram": {
                    0: 28426863616,
                    1: 26338071552,
                },
                "score": 100,
                "tensor_split": [34359738368, 34359738368],
            },
        ]

        assert len(candidates) == 3
        compare_candidates(candidates, expected_candidates)


@pytest.mark.asyncio
async def test_schedule_with_ngl_end_in_cpu_offload(temp_dir):
    cache_dir = os.path.join(
        os.path.dirname(__file__),
        "../../../fixtures/estimates/bartowski_Meta-Llama-3.1-8B-Instruct-GGUF-Q8_0",
    )
    config = Config(
        token="test",
        jwt_secret_key="test",
        data_dir=temp_dir,
        cache_dir=cache_dir,
        huggingface_token="",
    )
    set_global_config(config)

    if not check_parser(version="v0.13.10"):
        pytest.skip("parser path is not available or version mismatch, skipping.")

    workers = [
        create_worker(
            1,
            22036840320,
            {0: 15275183032},
            SystemReserved(ram=2147483648, vram=1073741824),
        ),
        create_worker(
            2,
            67145928704,
            {0: 25757220864},
            SystemReserved(ram=2147483648, vram=1073741824),
        ),
    ]

    m = new_model(
        1,
        "Meta-Llama-3.1-8B-Instruct-Q8_0",
        1,
        huggingface_repo_id="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        cpu_offloading=True,
        placement_strategy=PlacementStrategyEnum.SPREAD,
        huggingface_filename="Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
        backend_parameters=["--ctx-size=8192", "--ngl=0"],
        distributed_inference_across_workers=True,
    )
    mi = new_model_instance(1, "test", 1)

    resource_fit_selector = GGUFResourceFitSelector(m, mi, cache_dir)
    placement_scorer_spread = PlacementScorer(m, mi)

    with (
        patch(
            'gpustack.policies.utils.get_worker_model_instances',
            return_value=[],
        ),
        patch(
            'gpustack.scheduler.calculator._gguf_parser_command',
            side_effect=mock_gguf_parser_command,
        ),
        patch(
            'gpustack.scheduler.calculator.hf_model_filename',
            return_value="Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
        ),
        patch(
            'gpustack.scheduler.calculator.hf_mmproj_filename',
            return_value=None,
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
        candidates = await placement_scorer_spread.score(candidates)

        expected_candidates = [
            {
                "offload_layers": 0,
                "worker_id": 2,
                "worker_name": "host02",
                "is_unified_memory": False,
                "ram": 9628795768,
                "score": 100,
            },
            {
                "offload_layers": 0,
                "worker_id": 1,
                "worker_name": "host01",
                "is_unified_memory": False,
                "ram": 9628795768,
                "score": 100,
            },
        ]

        assert len(candidates) == 2
        compare_candidates(candidates, expected_candidates)


@pytest.mark.asyncio
async def test_schedule_with_deepseek_r1_bf16_with_manual_selected_cant_offload_gpus(
    temp_dir,
):
    cache_dir = os.path.join(
        os.path.dirname(__file__),
        "../../../fixtures/estimates/unsloth_DeepSeek-R1-GGUF_DeepSeek-R1-BF16",
    )
    config = Config(
        token="test",
        jwt_secret_key="test",
        data_dir=temp_dir,
        cache_dir=cache_dir,
        huggingface_token="",
    )
    set_global_config(config)

    if not check_parser():
        pytest.skip("parser path is not available or version mismatch, skipping.")

    workers = [
        linux_nvidia_19_4090_24gx2(),
        linux_nvidia_11_V100_32gx2(),
        linux_nvidia_12_A40_48gx2(),
        linux_nvidia_14_A100_40gx2(),
        linux_nvidia_8_3090_24gx8(),
        linux_nvidia_15_4080_16gx8(),
    ]

    m = new_model(
        1,
        "DeepSeek-R1-GGUF",
        1,
        huggingface_repo_id="unsloth/DeepSeek-R1-GGUF",
        cpu_offloading=True,
        placement_strategy=PlacementStrategyEnum.SPREAD,
        huggingface_filename="DeepSeek-R1-BF16/DeepSeek-R1.BF16-00001-of-00030.gguf",
        backend_parameters=["--ctx-size=32768"],
        gpu_selector=GPUSelector(
            gpu_ids=[
                "host-5-4080:cuda:0",
                "host-5-4080:cuda:1",
                "host-5-4080:cuda:2",
                "host-5-4080:cuda:3",
                "host-5-4080:cuda:4",
                "host-5-4080:cuda:5",
                "host-5-4080:cuda:6",
                "host-5-4080:cuda:7",
                "host-2-4090:cuda:0",
                "host-2-4090:cuda:1",
            ]
        ),
    )
    mi = new_model_instance(1, "test", 1)

    resource_fit_selector = GGUFResourceFitSelector(m, mi, cache_dir)
    placement_scorer_spread = PlacementScorer(m, mi)

    with (
        patch(
            'gpustack.policies.utils.get_worker_model_instances',
            return_value=[],
        ),
        patch(
            'gpustack.scheduler.calculator._gguf_parser_command',
            side_effect=mock_gguf_parser_command,
        ),
        patch(
            'gpustack.scheduler.calculator.hf_model_filename',
            return_value="DeepSeek-R1-BF16/DeepSeek-R1.BF16-00001-of-00030.gguf",
        ),
        patch(
            'gpustack.scheduler.calculator.hf_mmproj_filename',
            return_value=None,
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
        candidates = await placement_scorer_spread.score(candidates)
        messages = resource_fit_selector.get_messages()
        assert len(candidates) == 0
        assert (
            messages[0]
            == "Selected GPU host-5-4080:cuda:0 lacks enough VRAM. At least 26.47 GiB is required."
        )


async def mock_gguf_parser_command(
    model: Model, offload: GPUOffloadEnum = GPUOffloadEnum.Full, **kwargs
):
    gguf_parser_path = os.getenv("GGUF_PARSER_PATH")
    execuable_command = await _gguf_parser_command(model, offload, **kwargs)
    execuable_command[0] = gguf_parser_path
    execuable_command.extend(["--cache-expiration", "438000h0m0s"])
    return execuable_command


def create_worker(
    worker_id: int,
    ram: int,
    gpu_memorys: Dict[int, int],
    system_reserved: Optional[SystemReserved],
    is_uma: bool = False,
    gpu_vendor: Optional[VendorEnum] = VendorEnum.NVIDIA.value,
) -> Worker:
    gpu_type = device_type_from_vendor(gpu_vendor)
    name = f"host{worker_id:02d}"
    return Worker(
        id=worker_id,
        name=name,
        hostname=name,
        ip=generate_random_ip(),
        port=random.randint(0, 65535),
        labels={},
        system_reserved=system_reserved,
        state="ready",
        status=WorkerStatus(
            memory=MemoryInfo(total=ram, is_unified_memory=is_uma),
            gpu_devices=[
                {
                    "uuid": generate_random_uuid(),
                    "name": "test",
                    "vendor": gpu_vendor,
                    "index": index,
                    "device_index": index,
                    "device_chip_index": 0,
                    "core": GPUCoreInfo(
                        total=0,
                    ),
                    "memory": MemoryInfo(
                        total=gpu_memory,
                        is_unified_memory=is_uma,
                    ),
                    "temperature": 0,
                    "labels": {},
                    "type": gpu_type,
                }
                for index, gpu_memory in gpu_memorys.items()
            ],
        ),
    )


def create_workers_with_gpu_step(
    worker_count: int,
    worker_id_begin: int,
    ram_in_gib: int,
    gpu_per_worker: int = 16,
    gpu_begin_vram_in_gib: int = 24,
    gpu_memory_increase_step_in_gib: int = 6,
) -> List[Worker]:
    workers = []
    for wi in range(worker_count):
        worker_id = worker_id_begin + wi
        gpu_memorys = {
            i: (gpu_begin_vram_in_gib + i * gpu_memory_increase_step_in_gib) * 1024**3
            for i in range(gpu_per_worker)
        }
        w = create_worker(
            worker_id, ram_in_gib * 1024**3, gpu_memorys, SystemReserved(ram=0, vram=0)
        )
        workers.append(w)

    return workers


def generate_random_ip() -> str:
    return f"{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}"


def generate_random_uuid() -> str:
    return str(uuid.uuid4())
