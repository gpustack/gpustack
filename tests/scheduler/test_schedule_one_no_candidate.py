from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from gpustack.policies.resource_view import compute_worker_allocated
from gpustack.scheduler.scheduler import Scheduler
from gpustack.schemas.models import (
    ComputedResourceClaim,
    DistributedServers,
    ModelInstanceStateEnum,
    ModelInstanceSubordinateWorker,
)
from tests.utils.model import new_model, new_model_instance


class _Session:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _Service:
    def __init__(self, session):
        pass

    async def update(self, instance):
        return instance


@pytest.mark.asyncio
@pytest.mark.parametrize("assigned", [False, True], ids=["control", "distributed"])
async def test_no_candidate_releases_placement(monkeypatch, assigned):
    old = datetime.now(timezone.utc) - timedelta(minutes=10)
    inst = new_model_instance(
        id=7,
        name="dist",
        model_id=1,
        worker_id=11 if assigned else None,
        state=ModelInstanceStateEnum.ANALYZING,
        gpu_indexes=[0] if assigned else [],
        computed_resource_claim=(
            ComputedResourceClaim(ram=1, vram={0: 7_000_000_000}) if assigned else None
        ),
    )
    inst.gpu_type = "cuda" if assigned else None
    inst.created_at = old - timedelta(hours=1)
    inst.updated_at = old
    inst.distributed_servers = (
        DistributedServers(
            subordinate_workers=[
                ModelInstanceSubordinateWorker(
                    worker_id=22,
                    worker_name="sub",
                    gpu_indexes=[1],
                    gpu_type="cuda",
                    computed_resource_claim=ComputedResourceClaim(
                        ram=0, vram={1: 7_000_000_000}
                    ),
                )
            ]
        )
        if assigned
        else None
    )
    model = new_model(1, "m", huggingface_repo_id="org/m", replicas=1)

    async def workers(session):
        return [SimpleNamespace(id=11)]

    async def one_model(session, _id):
        return model

    async def one_inst(session, _id):
        return inst

    async def all_inst(session, options=None):
        return [inst]

    async def no_candidate(*a, **k):
        nonlocal searches
        searches += 1
        return None, ["No workers meet the resource requirements.\n"]

    searches = 0
    p = "gpustack.scheduler.scheduler."
    monkeypatch.setattr(p + "async_session", lambda: _Session())
    monkeypatch.setattr(p + "Worker.all", workers)
    monkeypatch.setattr(p + "Model.one_by_id", one_model)
    monkeypatch.setattr(p + "ModelInstance.one_by_id", one_inst)
    monkeypatch.setattr(p + "ModelInstance.all", all_inst)
    monkeypatch.setattr(p + "find_candidate", no_candidate)
    monkeypatch.setattr(p + "ModelInstanceService", _Service)
    released = set()

    async def invalidate(worker_ids):
        released.update(worker_ids)

    monkeypatch.setattr(p + "invalidate_worker_ids", invalidate)

    scheduler = Scheduler(SimpleNamespace(cache_dir=None))
    await scheduler._schedule_one(inst)
    assert searches == 1
    assert inst.state == ModelInstanceStateEnum.PENDING
    assert inst.computed_resource_claim is None

    # Retry eligibility is evaluated after the normal re-enqueue delay.
    inst.updated_at = datetime.now(timezone.utc) - timedelta(minutes=10)
    allocated = dict(compute_worker_allocated([inst], 22, "cuda").vram)
    assert inst.worker_id is None
    assert inst.distributed_servers is None
    assert inst.gpu_indexes == []
    assert allocated == {}
    assert scheduler._should_schedule(inst) is True
    # The workers the instance just left get their allocation cache dropped.
    assert released == ({11, 22} if assigned else set())
