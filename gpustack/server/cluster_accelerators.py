from typing import Optional, Set

from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.schemas.workers import Worker


async def cluster_vendors(session: AsyncSession, cluster_id: Optional[int]) -> Set[str]:
    """Accelerator manufacturer slugs the cluster's live workers report.

    One implementation for the two callers that must not disagree: the
    resolver that derives a PD recipe, and the request-time check that refuses
    one. Two copies of "what does this cluster have" would drift, and the
    failure would be a deployment the API accepted and the scheduler then
    found unplaceable.

    An empty set means **unknown**, not "no accelerators": a cluster whose
    workers have not reported devices yet reads the same as one with none, and
    neither may be judged as unable to run anything. Callers leave that case
    to scheduling.
    """
    if cluster_id is None:
        return set()

    workers = await Worker.all_by_fields(
        session,
        fields={"cluster_id": cluster_id},
        extra_conditions=[Worker.deleted_at.is_(None)],
    )
    return {
        (device.vendor or "").lower()
        for worker in workers
        for device in (
            worker.status.gpu_devices
            if worker.status and worker.status.gpu_devices
            else []
        )
        if device.vendor
    }
