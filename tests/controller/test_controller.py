from typing import List
import pytest
from gpustack.policies.base import ModelInstanceScore

from gpustack.schemas.models import (
    ComputedResourceClaim,
    ModelInstanceStateEnum,
)
from gpustack.schemas.workers import WorkerStateEnum
from gpustack.server.controllers import (
    calculate_model_destinations,
    find_scale_down_candidates,
)
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


@pytest.mark.asyncio
async def test_calculate_model_destinations_manual_distributed_returns_empty():
    # Manual-distributed models opt out of gateway routing, so destinations must
    # be empty (no route points at an unregistered upstream). The early return
    # precedes any session access, so None is safe here.
    model = new_model(
        1,
        "manual-dp",
        1,
        "Meta-Llama-3-70B-Instruct-GGUF",
        env={"GPUSTACK_MANUAL_DISTRIBUTED": "1"},
    )
    destinations = await calculate_model_destinations(session=None, model=model)
    assert destinations == []


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


@pytest.mark.asyncio
async def test_find_scale_down_candidates_dp_node_per_instance():
    """DP-node-per-instance scale-down drops the highest dp_rank first and always keeps rank 0; nodes
    still lacking a rank go first. The DP-node-per-instance branch returns before the scorer chain, so
    no session/worker mocking is needed."""
    from gpustack.schemas.models import Model, BackendEnum

    model = Model(
        name="m",
        backend=BackendEnum.VLLM,
        backend_parameters=["--data-parallel-external-lb"],
        distributed_inference_across_workers=True,
        replicas=3,
    )
    instances = []
    for iid, rank in [(1, 0), (2, 1), (3, 2), (4, None)]:
        mi = new_model_instance(
            iid, f"m-{iid}", 1, state=ModelInstanceStateEnum.RUNNING
        )
        mi.dp_rank = rank
        instances.append(mi)

    candidates = await find_scale_down_candidates(instances, model)
    order = [c.model_instance.dp_rank for c in candidates]
    assert order == [None, 2, 1, 0]  # deleted front-first; rank 0 preserved last


@pytest.mark.asyncio
async def test_sync_replicas_backfills_dp_rank_on_existing_instances():
    """A model turning into a DP group keeps the instances it already had, and
    those carry dp_rank=None — invisible to the scheduler's DP fan-out and to
    the worker's DP-node detection. sync_replicas backfills them before handing
    ranks to the instances it creates, so the group ends up densely ranked."""
    from unittest.mock import AsyncMock, MagicMock
    from gpustack.schemas.models import Model, BackendEnum, SourceEnum
    from gpustack.server import controllers as controllers_module

    model = Model(
        id=1,
        name="m",
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="a/b",
        backend=BackendEnum.VLLM,
        backend_parameters=[
            "--data-parallel-external-lb",
            "--data-parallel-size=3",
        ],
        distributed_inference_across_workers=True,
        replicas=3,
    )
    existing = []
    for iid in (1, 2):
        mi = new_model_instance(
            iid, f"m-{iid}", 1, state=ModelInstanceStateEnum.RUNNING
        )
        mi.dp_rank = None
        existing.append(mi)

    service = MagicMock()
    service.create = AsyncMock()
    service.update = AsyncMock()
    with (
        patch.object(Model, "one_by_id", AsyncMock(return_value=model)),
        patch.object(
            controllers_module.ModelInstance,
            "all_by_field",
            AsyncMock(return_value=existing),
        ),
        patch.object(
            controllers_module, "ModelInstanceService", lambda session: service
        ),
    ):
        await controllers_module.sync_replicas(MagicMock(), model)

    assert [mi.dp_rank for mi in existing] == [0, 1]
    # The one new instance takes the only rank left, not a duplicate 0.
    created = [call.args[0] for call in service.create.await_args_list]
    assert [instance.dp_rank for instance in created] == [2]
