from unittest.mock import AsyncMock, patch

import pytest
from transformers import PretrainedConfig

from gpustack.policies.candidate_selectors import (
    SGLangResourceFitSelector,
    VLLMResourceFitSelector,
)
from gpustack.policies.resource_view import ResourceView
from gpustack.schemas.models import BackendEnum, GPUSelector
from tests.fixtures.workers.fixtures import (
    linux_nvidia_4_4080_16gx4,
    linux_nvidia_5_a100_80gx2,
    linux_nvidia_6_a100_80gx2,
    linux_nvidia_7_a100_80gx2,
    linux_nvidia_19_4090_24gx2,
)
from tests.utils.model import new_model

SELECTORS = [
    (VLLMResourceFitSelector, BackendEnum.VLLM),
    (SGLangResourceFitSelector, BackendEnum.SGLANG),
]


def _zero_memory(worker, index):
    worker.status.gpu_devices[index].memory.total = 0
    return worker


async def _select(config, selector_class, workers, weight_size=1024**3, **model_fields):
    model = new_model(
        1,
        "zero-memory-gpu",
        huggingface_repo_id="test/zero-memory-gpu",
        cpu_offloading=False,
        **model_fields,
    )
    selector = selector_class(config, model, [], resource_view=ResourceView([]))
    pretrained = PretrainedConfig(
        num_hidden_layers=2,
        num_attention_heads=8,
        hidden_size=64,
        max_position_embeddings=8192,
    )
    with (
        patch(
            "gpustack.policies.candidate_selectors.base_candidate_selector.get_pretrained_config_with_workers",
            new=AsyncMock(return_value=pretrained),
        ),
        patch(
            "gpustack.policies.utils.get_model_weight_size", return_value=weight_size
        ),
    ):
        return await selector.select_candidates(workers)


@pytest.mark.parametrize("selector_class,backend", SELECTORS)
@pytest.mark.asyncio
async def test_single_gpu_placement_skips_a_gpu_without_memory(
    config, selector_class, backend
):
    worker = _zero_memory(linux_nvidia_19_4090_24gx2(), 0)

    candidates = await _select(config, selector_class, [worker], backend=backend.value)

    assert [c.gpu_indexes for c in candidates] == [[1]]


@pytest.mark.parametrize(
    "selector_class,backend,parallel_flag",
    [
        (VLLMResourceFitSelector, BackendEnum.VLLM, "--tensor-parallel-size=2"),
        (SGLangResourceFitSelector, BackendEnum.SGLANG, "--tp-size=2"),
    ],
)
@pytest.mark.asyncio
async def test_multi_gpu_placement_skips_a_gpu_without_memory(
    config, selector_class, backend, parallel_flag
):
    worker = _zero_memory(linux_nvidia_4_4080_16gx4(), 3)

    candidates = await _select(
        config,
        selector_class,
        [worker],
        backend=backend.value,
        backend_parameters=[parallel_flag],
    )

    assert len(candidates) == 1
    assert len(candidates[0].gpu_indexes) == 2
    assert 3 not in candidates[0].gpu_indexes


@pytest.mark.asyncio
async def test_vllm_multi_worker_placement_skips_a_worker_with_a_gpu_without_memory(
    config,
):
    bad_worker = _zero_memory(linux_nvidia_5_a100_80gx2(), 0)
    workers = [bad_worker, linux_nvidia_6_a100_80gx2(), linux_nvidia_7_a100_80gx2()]

    candidates = await _select(
        config,
        VLLMResourceFitSelector,
        workers,
        weight_size=150 * 1024**3,
        backend=BackendEnum.VLLM.value,
        distributed_inference_across_workers=True,
    )

    assert len(candidates) == 1
    used = {candidates[0].worker.name}
    used.update(sw.worker_name for sw in candidates[0].subordinate_workers)
    assert bad_worker.name not in used
    assert len(used) == 2


@pytest.mark.asyncio
async def test_sglang_multi_worker_placement_rejects_workers_with_a_gpu_without_memory(
    config,
):
    workers = [
        _zero_memory(linux_nvidia_5_a100_80gx2(), 0),
        _zero_memory(linux_nvidia_6_a100_80gx2(), 0),
    ]

    candidates = await _select(
        config,
        SGLangResourceFitSelector,
        workers,
        weight_size=150 * 1024**3,
        backend=BackendEnum.SGLANG.value,
        distributed_inference_across_workers=True,
    )

    assert candidates == []


@pytest.mark.parametrize("selector_class,backend", SELECTORS)
@pytest.mark.asyncio
async def test_manual_selection_reports_a_gpu_without_memory_as_overcommit(
    config, selector_class, backend
):
    worker = _zero_memory(linux_nvidia_19_4090_24gx2(), 0)
    gpu_ids = [f"{worker.name}:cuda:{i}" for i in (0, 1)]

    candidates = await _select(
        config,
        selector_class,
        [worker],
        backend=backend.value,
        gpu_selector=GPUSelector(gpu_ids=gpu_ids, gpus_per_replica=2),
    )

    assert len(candidates) == 1
    assert candidates[0].overcommit
