"""Server-side per-worker cache of ``Allocated`` aggregated from current
ModelInstance bindings.

Cached at one key per worker (``WorkerAllocated.worker.{id}``) so that a
single ModelInstance write only invalidates the workers actually bound to
that instance — heartbeats from unrelated workers still hit cache without
triggering recompute.

Invalidation point is ModelInstanceService's mutation methods — every
create / update / delete (and their batch variants) calls
``invalidate_workers_allocated`` with the affected worker_ids after
commit, so worker.status.allocated stays in sync with the ModelInstance
table.

UPDATE caveat: when an UPDATE reassigns an instance to a different
worker (rare; rescheduling), only the NEW worker's cache is dropped
here — the previous worker's cache stays until the next read after TTL
expires, or until the next mutation that touches it. The bounded
staleness is acceptable for the rescheduling case; chasing strict
correctness would require capturing the pre-update state via SQLAlchemy
history hooks.

Mirrors the K8s scheduler NodeInfo cache: derive Allocated from current
workload→node bindings; recompute is driven by workload mutations, not
by node heartbeats.
"""

import logging
from typing import Dict, Iterable, Optional, Set

from sqlmodel import select, or_

from gpustack.policies.base import Allocated
from gpustack.policies.resource_view import compute_worker_allocated
from gpustack.schemas.models import ModelInstance
from gpustack.server.cache import delete_cache_by_key, locked_cached
from gpustack.server.db import async_session

logger = logging.getLogger(__name__)


def vram_allocated_for_index(vram: Dict[int, int], index: Optional[int]) -> int:
    """Allocated VRAM for one GPU of a worker, given the worker's
    {gpu_index: vram} aggregation. 0 when the device index is unknown or no
    instance is assigned to it — shared fallback semantics for every place
    that injects allocated into a device payload (/v2/workers and
    /v1/gpu-devices, REST and watch)."""
    return vram.get(index, 0) if index is not None else 0


def _cache_key_for(worker_id: int) -> str:
    return f"WorkerAllocated.worker.{worker_id}"


def _key_builder(_f, *args, **kwargs):
    worker_id = kwargs.get("worker_id")
    if worker_id is None and args:
        worker_id = args[0]
    return _cache_key_for(worker_id)


@locked_cached(key=_key_builder)
async def _get_model_worker_allocated(worker_id: int) -> Allocated:
    """Return the cached ``Allocated`` for a single worker. On miss,
    aggregates current ModelInstance bindings for this worker (main +
    distributed subordinate)."""
    async with async_session() as session:
        # main: cheap indexed filter.
        # distributed subordinates live inside the distributed_servers JSON
        # column which can't be filtered portably across PG/MySQL/
        # OceanBase/openGauss for a specific worker_id, so fall back to
        # fetching all distributed instances and letting the helper pick
        # out the relevant ones in Python.
        rows = (
            await session.exec(
                select(ModelInstance).where(
                    or_(
                        ModelInstance.worker_id == worker_id,
                        ModelInstance.distributed_servers.is_not(None),
                    )
                )
            )
        ).all()
    return compute_worker_allocated(rows, worker_id)


async def get_worker_allocated(worker_id: int) -> Allocated:
    """Combine cached model allocations with current cache reservations."""
    from gpustack.server.cache_service_resources import get_cache_ram_by_worker

    model_allocated = await _get_model_worker_allocated(worker_id)
    async with async_session() as session:
        cache_ram = await get_cache_ram_by_worker(session, worker_id=worker_id)
    return Allocated(
        ram=model_allocated.ram + cache_ram.get(worker_id, 0),
        vram=dict(model_allocated.vram),
    )


async def get_workers_allocated(worker_ids: Iterable[int]) -> Dict[int, Allocated]:
    """Read current reservations once for a batch of workers.

    Workers whose model allocations cannot be read are omitted, so callers can
    conservatively report their capacity as unavailable.
    """
    from gpustack.server.cache_service_resources import get_cache_ram_by_worker

    worker_ids = list(set(worker_ids))
    if not worker_ids:
        return {}
    async with async_session() as session:
        cache_ram = await get_cache_ram_by_worker(session, worker_ids=worker_ids)
    allocated = {}
    for worker_id in worker_ids:
        try:
            model_allocated = await _get_model_worker_allocated(worker_id)
        except Exception:
            logger.exception("Could not read allocation for worker %s", worker_id)
            continue
        allocated[worker_id] = Allocated(
            ram=model_allocated.ram + cache_ram.get(worker_id, 0),
            vram=dict(model_allocated.vram),
        )
    return allocated


async def invalidate_workers_allocated(instances: Iterable[ModelInstance]) -> None:
    """Drop cached Allocated for every worker bound to any of the given
    instances — main worker plus distributed subordinates.

    Accepts an iterable so single-instance writes pass ``[instance]`` and
    batch writes pass the batch directly; deduplication across the union
    keeps the actual ``delete_cache_by_key`` calls minimal.
    """
    worker_ids: Set[int] = set()
    for inst in instances:
        if inst.worker_id is not None:
            worker_ids.add(inst.worker_id)
        dservers = inst.distributed_servers
        if dservers and dservers.subordinate_workers:
            for sw in dservers.subordinate_workers:
                if sw.worker_id is not None:
                    worker_ids.add(sw.worker_id)
    for wid in worker_ids:
        await delete_cache_by_key(_key=_cache_key_for(wid))
