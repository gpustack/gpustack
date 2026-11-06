"""Server-side allocation caches for worker and GPU resource displays.

Cached at one key per worker (``WorkerAllocated.worker.{id}``) so that a
single ModelInstance write only invalidates the workers actually bound to
that instance — heartbeats from unrelated workers still hit cache without
triggering recompute.

Invalidation point is ModelInstanceService's mutation methods — every
create / update / delete (and their batch variants) calls
``invalidate_workers_allocated`` with the affected worker_ids after
commit, so worker.status.allocated stays in sync with the ModelInstance
table.

Cache-service RAM uses a shared worker-indexed snapshot. CacheServiceInstanceService
and CacheServiceService invalidate it after committing instance mutations or
parent-service deletion, through the cross-server cache channel.
Scheduling reads cache reservations directly from its database session.

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

import asyncio
import logging
from typing import Dict, Iterable, Optional, Set
from uuid import uuid4

from sqlmodel import select, or_

from gpustack.policies.base import Allocated
from gpustack.policies.resource_view import compute_worker_allocated
from gpustack.schemas.models import ModelInstance
from gpustack.server.cache import delete_cache_by_key, locked_cached
from gpustack.server.db import async_session

logger = logging.getLogger(__name__)

CACHE_RAM_KEY = "CacheServiceInstance.worker_ram"
_cache_ram_lock = asyncio.Lock()
_cache_ram_revision: Optional[str] = None
_cache_ram: Dict[int, int] = {}


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


@locked_cached(key=CACHE_RAM_KEY)
async def _get_cache_ram_revision() -> str:
    """Use the shared cache's expiry and invalidation to version the snapshot."""
    return uuid4().hex


async def _get_cache_workers_ram() -> Dict[int, int]:
    """Share one reservation snapshot across worker lists and watch subscribers."""
    from gpustack.server.cache_service_resources import get_cache_ram_by_worker

    global _cache_ram_revision, _cache_ram
    revision = await _get_cache_ram_revision()
    if revision == _cache_ram_revision:
        return _cache_ram
    async with _cache_ram_lock:
        revision = await _get_cache_ram_revision()
        if revision != _cache_ram_revision:
            async with async_session() as session:
                ram = await get_cache_ram_by_worker(session)
            # A commit can invalidate the revision while this read is in flight.
            # Keep its original revision so subsequent readers reload the snapshot.
            _cache_ram, _cache_ram_revision = ram, revision
        return _cache_ram


async def invalidate_cache_workers_ram() -> None:
    """Invalidate the reservation snapshot locally and on other servers."""
    await delete_cache_by_key(_key=CACHE_RAM_KEY)


async def get_worker_vram_allocated(worker_id: int) -> Dict[int, int]:
    """Read model VRAM allocations without loading cache-service RAM claims."""
    return dict((await _get_model_worker_allocated(worker_id)).vram)


async def get_worker_allocated(worker_id: int) -> Allocated:
    """Combine cached model allocations with committed cache reservations."""
    model_allocated = await _get_model_worker_allocated(worker_id)
    cache_ram = await _get_cache_workers_ram()
    return Allocated(
        ram=model_allocated.ram + cache_ram.get(worker_id, 0),
        vram=dict(model_allocated.vram),
    )


async def get_workers_allocated(worker_ids: Iterable[int]) -> Dict[int, Allocated]:
    """Share the reservation snapshot with single-worker and watch lookups.

    Workers whose model allocations cannot be read are omitted, so callers can
    conservatively report their capacity as unavailable.
    """
    worker_ids = list(set(worker_ids))
    if not worker_ids:
        return {}
    cache_ram = await _get_cache_workers_ram()
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
