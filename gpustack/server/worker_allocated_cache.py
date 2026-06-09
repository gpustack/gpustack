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
from typing import Iterable, Set

from sqlmodel import select, or_

from gpustack.policies.base import Allocated
from gpustack.schemas.models import ModelInstance
from gpustack.server.cache import delete_cache_by_key, locked_cached
from gpustack.server.db import async_session

logger = logging.getLogger(__name__)


def _cache_key_for(worker_id: int) -> str:
    return f"WorkerAllocated.worker.{worker_id}"


def _key_builder(_f, *args, **kwargs):
    worker_id = kwargs.get("worker_id")
    if worker_id is None and args:
        worker_id = args[0]
    return _cache_key_for(worker_id)


@locked_cached(key=_key_builder)
async def get_worker_allocated(worker_id: int) -> Allocated:
    """Return the cached ``Allocated`` for a single worker. On miss,
    aggregates current ModelInstance bindings for this worker (main +
    distributed subordinate)."""
    # Lazy import to avoid a module-load cycle with policies.utils, which
    # imports server.services (and services imports this module).
    from gpustack.policies.utils import compute_worker_allocated

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
