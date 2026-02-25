from typing import List
import pytest
from gpustack.policies.base import ModelInstanceScore

from gpustack.schemas.models import (
    ComputedResourceClaim,
    ModelInstanceStateEnum,
)
from gpustack.schemas.workers import WorkerStateEnum
from gpustack.server.controllers import find_scale_down_candidates
from tests.fixtures.workers.fixtures import (
    linux_nvidia_19_4090_24gx2,
    linux_nvidia_2_4080_16gx2,
    linux_cpu_1,
)

from unittest.mock import patch

from tests.utils.mock import mock_async_session

from tests.utils.model import new_model, new_model_instance


@pytest.mark.asyncio
async def test_find_scale_down_candidates():
    w1 = linux_nvidia_19_4090_24gx2()
    w1.state = WorkerStateEnum.NOT_READY
    workers = [
        w1,
        linux_nvidia_2_4080_16gx2(),
        linux_cpu_1(),
    ]

    m = new_model(1, "test", 3, "Meta-Llama-3-70B-Instruct-GGUF")
    mis = [
        new_model_instance(
            1,
            "test-1",
            1,
            4,
            ModelInstanceStateEnum.RUNNING,
            [0, 1],
            ComputedResourceClaim(
                is_unified_memory=False,
                offload_layers=81,
                total_layers=81,
                ram=455165112,
                vram={0: 22912443392, 1: 22911897600},
            ),
        ),
        new_model_instance(
            2,
            "test-2",
            1,
            3,
            ModelInstanceStateEnum.RUNNING,
            [0, 1],
            ComputedResourceClaim(
                is_unified_memory=False,
                offload_layers=60,
                total_layers=81,
                ram=1093245112,
                vram={0: 16900820992, 1: 16900820992},
            ),
        ),
        new_model_instance(
            3,
            "test-3",
            1,
            6,
            ModelInstanceStateEnum.RUNNING,
            None,
            ComputedResourceClaim(
                is_unified_memory=False,
                offload_layers=0,
                total_layers=81,
                ram=3106511032,
            ),
        ),
    ]

    with (
        patch(
            'gpustack.schemas.models.ModelInstance.all_by_field',
            return_value=mis,
        ),
        patch(
            'gpustack.schemas.models.ModelInstance.all',
            return_value=mis,
        ),
        patch(
            'gpustack.schemas.workers.Worker.all',
            return_value=workers,
        ),
        patch(
            'gpustack.policies.scorers.placement_scorer.async_session',
            return_value=mock_async_session(),
        ),
        patch(
            'gpustack.policies.scorers.status_scorer.async_session',
            return_value=mock_async_session(),
        ),
    ):

        candidates = await find_scale_down_candidates(mis, m, total_max_score=100)

        expected_candidates = [
            {
                "worker_id": 4,
                "instacnce_id": 1,
                "gpu_indexes": [0, 1],
                "score": 9.538995598356342,
            },
            {
                "worker_id": 6,
                "instacnce_id": 3,
                "score": 90.1308159326069,
            },
            {
                "worker_id": 3,
                "instacnce_id": 2,
                "score": 97.3594505895714,
            },
        ]

        compare_candidates(candidates, expected_candidates)


def compare_candidates(candidates: List[ModelInstanceScore], expected_candidates):
    for i, expected in enumerate(expected_candidates):
        candidate = candidates[i]
        instance = candidate.model_instance

        if "worker_id" in expected:
            assert instance.worker_id == expected["worker_id"]

        if "instance_id" in expected:
            assert instance.id == expected["instance_id"]

        if "score" in expected:
            assert str(candidate.score)[:5] == str(expected["score"])[:5]
