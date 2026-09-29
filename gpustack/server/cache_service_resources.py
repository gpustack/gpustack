"""Read cache reservations without depending on worker telemetry."""

from typing import Dict, Iterable, List, Optional

from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.schemas.cache_services import CacheServiceInstance
from gpustack.schemas.models import ModelInstance
from gpustack.policies.base import Allocated
from gpustack.policies.resource_view import ResourceView


def cache_ram_by_worker(instances: Iterable[CacheServiceInstance]) -> Dict[int, int]:
    ram: Dict[int, int] = {}
    for instance in instances:
        if getattr(instance, "deleted_at", None) is not None:
            continue
        claim = getattr(instance, "computed_resource_claim", None)
        if claim:
            ram[instance.worker_id] = ram.get(instance.worker_id, 0) + (
                claim.get("ram") or 0
            )
    return ram


async def get_cache_ram_by_worker(
    session: AsyncSession,
    cluster_id: Optional[int] = None,
    worker_id: Optional[int] = None,
    worker_ids: Optional[List[int]] = None,
) -> Dict[int, int]:
    """Load the reservations used by one scheduling pass or worker API read.

    No execution state filter: preparing and restarting instances keep their
    reservations. A failed read propagates rather than suggesting free RAM.
    """
    fields = {}
    if cluster_id is not None:
        fields["cluster_id"] = cluster_id
    if worker_id is not None:
        fields["worker_id"] = worker_id
    if worker_ids is not None:
        if not worker_ids:
            return {}
        instances = await CacheServiceInstance.all_by_fields(
            session,
            fields,
            extra_conditions=[CacheServiceInstance.worker_id.in_(worker_ids)],
        )
    else:
        instances = await CacheServiceInstance.all_by_fields(session, fields)
    return cache_ram_by_worker(instances)


def cache_resource_reservations(
    instances: Iterable[CacheServiceInstance],
) -> Dict[int, Allocated]:
    """Adapt cache instance declarations to the scheduler's resource accounting."""
    return {
        worker_id: Allocated(ram=ram, vram={})
        for worker_id, ram in cache_ram_by_worker(instances).items()
    }


async def load_resource_view(
    session: AsyncSession,
    model_instances: List[ModelInstance],
    cluster_id: Optional[int] = None,
) -> ResourceView:
    """Build one accounting view for selection, scoring, and evaluation."""
    ram = await get_cache_ram_by_worker(session, cluster_id=cluster_id)
    return ResourceView(
        model_instances,
        {worker_id: Allocated(ram=value, vram={}) for worker_id, value in ram.items()},
    )
