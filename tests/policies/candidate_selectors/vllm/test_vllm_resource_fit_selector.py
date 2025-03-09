import shutil
import tempfile
import pytest
from unittest.mock import patch, AsyncMock
from tests.utils.model import new_model, new_model_instance
from tenacity import retry, stop_after_attempt, wait_fixed
from gpustack.config.config import set_global_config, Config
from gpustack.policies.candidate_selectors.vllm_resource_fit_selector import (
    VLLMResourceFitSelector,
    get_hub_model_weight_size,
)
from gpustack.policies.scorers.placement_scorer import PlacementScorer
from gpustack.scheduler.scheduler import Scheduler
from gpustack.schemas.models import (
    ComputedResourceClaim,
    GPUSelector,
    Model,
    RayActor,
    SourceEnum,
)
from tests.fixtures.workers.fixtures import (
    linux_nvidia_1_4090_24gx1,
    linux_nvidia_3_4090_24gx1,
    linux_nvidia_4_4080_16gx4,
    linux_nvidia_5_a100_80gx2,
    linux_nvidia_6_a100_80gx2,
    linux_nvidia_7_a100_80gx2,
)
from tests.utils.scheduler import compare_candidates


def test_get_hub_model_weight_size():
    model_to_weight_sizes = [
        (
            Model(
                source=SourceEnum.HUGGING_FACE,
                huggingface_repo_id="Qwen/Qwen2-0.5B-Instruct",
            ),
            1_000_000_000,
        ),
        (
            Model(
                source=SourceEnum.HUGGING_FACE,
                huggingface_repo_id="Qwen/Qwen2-VL-7B-Instruct",
            ),
            14_000_000_000,
        ),
        (
            Model(
                source=SourceEnum.HUGGING_FACE,
                huggingface_repo_id="Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4",
            ),
            36_000_000_000,
        ),
        (
            Model(
                source=SourceEnum.HUGGING_FACE,
                huggingface_repo_id="unsloth/Llama-3.3-70B-Instruct-bnb-4bit",
            ),
            35_000_000_000,
        ),
        (
            Model(
                source=SourceEnum.MODEL_SCOPE,
                model_scope_model_id="Qwen/Qwen2-0.5B-Instruct",
            ),
            1_000_000_000,
        ),
        (
            Model(
                source=SourceEnum.MODEL_SCOPE,
                model_scope_model_id="Qwen/Qwen2-VL-7B-Instruct",
            ),
            14_000_000_000,
        ),
        (
            Model(
                source=SourceEnum.MODEL_SCOPE,
                model_scope_model_id="Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4",
            ),
            36_000_000_000,
        ),
        (
            Model(
                source=SourceEnum.MODEL_SCOPE,
                model_scope_model_id="unsloth/Llama-3.3-70B-Instruct-bnb-4bit",
            ),
            35_000_000_000,
        ),
    ]

    set_global_config(Config(data_dir="/tmp/test_data_dir"))

    for model, expected_weight_size in model_to_weight_sizes:
        computed = get_hub_model_weight_size_with_retry(model)
        assert (
            computed == expected_weight_size
        ), f"weight_size mismatch for {model}, computed: {computed}, expected: {expected_weight_size}"


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def get_hub_model_weight_size_with_retry(model: Model) -> int:
    return get_hub_model_weight_size(model)


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
async def test_manual_schedule_to_multi_worker_multi_gpu(config):
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
    mi = new_model_instance(1, "test_name", 1)

    resource_fit_selector = VLLMResourceFitSelector(m, mi)
    placement_scorer = PlacementScorer(m, mi)
    scheduler = Scheduler(config)

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
        candidate, _ = await scheduler.find_candidate(mi, m, workers)

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
async def test_manual_schedule_to_multi_worker_multi_gpu_select_main_with_most_gpus(
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
    mi = new_model_instance(1, "test_name", 1)

    resource_fit_selector = VLLMResourceFitSelector(m, mi)
    placement_scorer = PlacementScorer(m, mi)
    scheduler = Scheduler(config)

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
        candidate, _ = await scheduler.find_candidate(mi, m, workers)

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
async def test_manual_schedule_to_three_workers_four_gpus(
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
    mi = new_model_instance(1, "test_name", 1)

    resource_fit_selector = VLLMResourceFitSelector(m, mi)
    placement_scorer = PlacementScorer(m, mi)
    scheduler = Scheduler(config)

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
            return_value=workers(),
        ),
    ):

        candidates = await resource_fit_selector.select_candidates(workers())
        candidates = await placement_scorer.score(candidates)
        candidate, _ = await scheduler.find_candidate(mi, m, workers())

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
