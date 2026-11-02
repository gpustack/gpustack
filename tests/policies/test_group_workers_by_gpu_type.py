from unittest.mock import AsyncMock, patch

import pytest
from transformers import PretrainedConfig

from gpustack.policies.candidate_selectors import VLLMResourceFitSelector
from gpustack.policies.resource_view import ResourceView
from gpustack.policies.utils import group_workers_by_gpu_type
from gpustack.schemas.models import BackendEnum
from tests.fixtures.workers.fixtures import (
    linux_mix_1_nvidia_4080_16gx1_rocm_7800_16gx1,
)
from tests.utils.model import new_model


def test_each_group_of_a_mixed_worker_keeps_its_own_devices():
    worker = linux_mix_1_nvidia_4080_16gx1_rocm_7800_16gx1()

    groups = group_workers_by_gpu_type([worker])

    assert {
        gpu_type: [gpu.type for gpu in workers[0].status.gpu_devices]
        for gpu_type, workers in groups.items()
    } == {"rocm": ["rocm"], "cuda": ["cuda"]}


def test_grouping_leaves_the_input_worker_unchanged():
    worker = linux_mix_1_nvidia_4080_16gx1_rocm_7800_16gx1()

    group_workers_by_gpu_type([worker])

    assert sorted(gpu.type for gpu in worker.status.gpu_devices) == ["cuda", "rocm"]


@pytest.mark.asyncio
async def test_auto_selection_offers_every_gpu_type_of_a_mixed_worker(config):
    model = new_model(
        1,
        "mixed-worker",
        huggingface_repo_id="test/mixed-worker",
        backend=BackendEnum.VLLM.value,
        cpu_offloading=False,
    )
    selector = VLLMResourceFitSelector(
        config, model, [], resource_view=ResourceView([])
    )
    pretrained = PretrainedConfig(
        num_hidden_layers=2, num_attention_heads=8, hidden_size=64
    )
    with (
        patch(
            "gpustack.policies.candidate_selectors.base_candidate_selector.get_pretrained_config_with_workers",
            new=AsyncMock(return_value=pretrained),
        ),
        patch("gpustack.policies.utils.get_model_weight_size", return_value=1024**3),
    ):
        candidates = await selector.select_candidates(
            [linux_mix_1_nvidia_4080_16gx1_rocm_7800_16gx1()]
        )

    assert sorted(c.gpu_type for c in candidates) == ["cuda", "rocm"]
