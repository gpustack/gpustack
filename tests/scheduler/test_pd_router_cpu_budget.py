from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from gpustack.policies.resource_view import ResourceView
from gpustack.policies.worker_filters.backend_framework_filter import (
    BackendFrameworkFilter,
)
from gpustack.scheduler import port_budget, scheduler
from gpustack.scheduler.evaluator import make_hashable_key
from gpustack.scheduler.group_capacity import GroupCapacity
from gpustack.schemas.models import (
    BackendEnum,
    Model,
    RoleResources,
    RoleSpec,
    SourceEnum,
)
from gpustack.schemas.port_reservations import WorkerPortAvailability
from gpustack.schemas.workers import (
    KubernetesCPUBudget,
    MemoryInfo,
    SystemReserved,
    Worker,
    WorkerStateEnum,
    WorkerStatus,
)


def _worker(id, available, *, expired=False, enabled=True):
    expires = datetime.now(timezone.utc) + timedelta(hours=-1 if expired else 1)
    return Worker(
        id=id,
        name=f"worker-{id}",
        labels={"worker-name": f"worker-{id}"},
        state=WorkerStateEnum.READY,
        system_reserved=SystemReserved(ram=0, vram=0),
        status=WorkerStatus(
            memory=MemoryInfo(total=8 * 1024**3),
            kubernetes_cpu_budget=(
                KubernetesCPUBudget(available_millicores=available, expires_at=expires)
                if enabled
                else None
            ),
        ),
    )


def _model(cpu=None, allowed=None):
    return Model(
        id=1,
        name="pd",
        backend=BackendEnum.VLLM.value,
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="Qwen/Qwen3-0.6B",
        roles=[
            RoleSpec(
                name="router",
                replicas=1,
                resources=RoleResources(cpu=cpu) if cpu is not None else None,
                worker_selector={"worker-name": allowed} if allowed else None,
            )
        ],
    )


@pytest.fixture
def candidate_dependencies(monkeypatch):
    async def supported(_filter, workers):
        return workers, []

    async def port_snapshots(workers, exclude_instance_id=None):
        return {
            worker.id: WorkerPortAvailability(
                worker_id=worker.id, generation=uuid4(), port_range="40000-40063"
            )
            for worker in workers
        }

    monkeypatch.setattr(BackendFrameworkFilter, "filter", supported)
    monkeypatch.setattr(port_budget, "worker_port_snapshots", port_snapshots)
    monkeypatch.setattr(scheduler, "_group_topology", AsyncMock(return_value=None))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("workers", "cpu", "allowed", "placed", "slots", "diagnostic"),
    [
        ([_worker(1, 1425), _worker(2, 2000)], None, None, 2, {1: 0, 2: 1}, "1425m"),
        (
            [_worker(1, 1425), _worker(2, 2000)],
            None,
            "worker-1",
            None,
            {1: 0, 2: 0},
            "2000m",
        ),
        ([_worker(1, 1500)], 1.5, None, 1, {1: 1}, None),
        ([_worker(1, 1500)], 1.501, None, None, {1: 0}, "1501m"),
        (
            [_worker(1, 9000, expired=True), _worker(2, None)],
            None,
            None,
            None,
            {},
            "unavailable or stale",
        ),
        ([_worker(1, None, enabled=False)], None, None, 1, {1: 1}, None),
    ],
    ids=[
        "alternative",
        "selector",
        "override-fits",
        "override-fails",
        "unknown",
        "old-worker",
    ],
)
async def test_router_placement_and_group_capacity_agree(
    candidate_dependencies, workers, cpu, allowed, placed, slots, diagnostic
):
    model = _model(cpu, allowed)
    resource_view = ResourceView([])
    cfg = SimpleNamespace(cache_dir=None, huggingface_token=None)
    capacity = GroupCapacity(cfg, model, workers, [], resource_view=resource_view)

    measured = await capacity("router", [worker.id for worker in workers], [])
    candidate, messages = await scheduler.find_candidate(
        None, cfg, model, workers, [], role="router", resource_view=resource_view
    )

    assert measured == slots
    assert (candidate.worker.id if candidate else None) == placed
    if diagnostic is not None:
        assert any(diagnostic in message for message in messages)
        assert any(diagnostic in message for message in capacity.notes_for("router"))


def test_preflight_cache_tracks_usable_budgets_not_clock_ticks():
    model = _model()
    worker = _worker(1, 2000)
    baseline = make_hashable_key(model, [worker])
    budget = worker.status.kubernetes_cpu_budget

    budget.expires_at += timedelta(seconds=30)
    assert make_hashable_key(model, [worker]) == baseline

    budget.available_millicores = 1425
    insufficient = make_hashable_key(model, [worker])
    assert insufficient != baseline

    budget.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    assert make_hashable_key(model, [worker]) not in {baseline, insufficient}
