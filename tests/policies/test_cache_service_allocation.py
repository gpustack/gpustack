from types import SimpleNamespace

import pytest

from gpustack.policies.resource_view import ResourceView
from gpustack.policies.base import Allocated
from gpustack.schemas.cache_services import CacheServiceInstance, CacheServiceStateEnum
from gpustack.schemas.workers import Worker, WorkerStatus, MemoryInfo, SystemReserved
from gpustack.server.cache_service_resources import (
    cache_ram_by_worker,
    cache_resource_reservations,
)


@pytest.mark.parametrize("state", list(CacheServiceStateEnum))
def test_bound_instances_reserve_ram_in_every_execution_state(state):
    instance = CacheServiceInstance(
        id=1,
        name="cache",
        cache_service_id=1,
        worker_id=2,
        cluster_id=1,
        state=state,
        computed_resource_claim={"ram": 40},
    )
    assert cache_ram_by_worker([instance]) == {2: 40}


def test_deleted_and_other_worker_claims_are_not_billed_to_this_worker():
    instances = [
        SimpleNamespace(
            worker_id=1, computed_resource_claim={"ram": 20}, deleted_at=None
        ),
        SimpleNamespace(
            worker_id=2, computed_resource_claim={"ram": 30}, deleted_at=None
        ),
        SimpleNamespace(
            worker_id=1, computed_resource_claim={"ram": 80}, deleted_at=True
        ),
    ]
    worker = Worker(
        id=1,
        name="worker",
        hostname="worker",
        ip="127.0.0.1",
        status=WorkerStatus(memory=MemoryInfo(total=100)),
        system_reserved=SystemReserved(ram=10, vram=0),
    )
    assert (
        ResourceView([], cache_resource_reservations(instances)).allocatable(worker).ram
        == 70
    )


@pytest.mark.parametrize("gpu_type", [None, "cuda", "cann"])
def test_cache_ram_is_not_filtered_by_accelerator_type(gpu_type):
    worker = Worker(
        id=1,
        name="worker",
        hostname="worker",
        ip="127.0.0.1",
        status=WorkerStatus(memory=MemoryInfo(total=100)),
        system_reserved=SystemReserved(ram=10, vram=0),
    )
    assert (
        ResourceView([], {1: Allocated(ram=40, vram={})})
        .allocatable(worker, gpu_type)
        .ram
        == 50
    )


@pytest.mark.asyncio
async def test_cpu_candidate_requires_room_after_cache_reservation():
    from gpustack.scheduler.scheduler import build_candidate_selector
    from gpustack.schemas.models import Model

    worker = Worker(
        id=1,
        name="worker",
        hostname="worker",
        ip="127.0.0.1",
        status=WorkerStatus(memory=MemoryInfo(total=100)),
        system_reserved=SystemReserved(ram=10, vram=0),
    )
    model = Model(
        name="router",
        source="local_path",
        local_path="/models/router",
        backend="custom",
    )
    fitting = build_candidate_selector(
        SimpleNamespace(),
        model,
        [],
        cpu_only=True,
        ram_claim=20,
        resource_view=ResourceView([], {1: Allocated(ram=60, vram={})}),
    )
    full = build_candidate_selector(
        SimpleNamespace(),
        model,
        [],
        cpu_only=True,
        ram_claim=20,
        resource_view=ResourceView([], {1: Allocated(ram=80, vram={})}),
    )
    assert len(await fitting.select_candidates([worker])) == 1
    assert await full.select_candidates([worker]) == []


def test_unified_memory_caps_gpu_availability_after_cache_reservation():
    worker = Worker(
        id=1,
        name="worker",
        hostname="worker",
        ip="127.0.0.1",
        status=WorkerStatus(
            memory=MemoryInfo(total=100, is_unified_memory=True),
            gpu_devices=[{"index": 0, "type": "mps", "memory": {"total": 100}}],
        ),
        system_reserved=SystemReserved(ram=10, vram=5),
    )
    available = ResourceView([], {1: Allocated(ram=40, vram={})}).allocatable(worker)
    assert available.ram == 45
    assert available.vram == {0: 45}


def test_evaluation_cache_key_includes_reservations():
    from gpustack.scheduler.evaluator import make_hashable_key
    from gpustack.schemas.models import Model, SourceEnum

    model = Model(
        name="cache-aware", source=SourceEnum.LOCAL_PATH, local_path="/models/model"
    )
    assert make_hashable_key(
        model, [], extra={"reservations": {1: {"ram": 10, "vram": {}}}}
    ) != make_hashable_key(
        model, [], extra={"reservations": {1: {"ram": 20, "vram": {}}}}
    )
