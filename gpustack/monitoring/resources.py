"""Current authorized resource associations and owner-qualified hardware history."""

from collections import Counter
import math
import time
from typing import List, Optional, Set

import aiohttp
from sqlmodel import or_, select
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.api.exceptions import InvalidException, NotFoundException
from gpustack.api.tenant import (
    TenantContext,
    tenant_list_conditions,
    tenant_stream_filter,
)
from gpustack.schemas.models import Model, ModelInstance
from gpustack.schemas.model_metrics import MetricDimensions
from gpustack.schemas.principals import OrgRole
from gpustack.schemas.resource_metrics import (
    ResourceAssociation,
    ResourceMetrics,
    ResourcePlacement,
)
from gpustack.schemas.workers import Worker
from gpustack.utils.model_instance_workers import get_model_instance_worker_match

from . import hardware


def instance_gpu_indexes(instance: ModelInstance, worker_id: int) -> Set[int]:
    """Include main, subordinate and managed rank placements without readiness filters."""
    return hardware._allocated_gpus(instance).get(worker_id, set())


def inventory_devices(worker: Worker) -> List[hardware._Device]:
    """Include idle GPUs, keeping ambiguous identities visible but without history."""
    indexes = (
        sorted(
            {
                d.index
                for d in (worker.status.gpu_devices or [])
                if type(d.index) is int and d.index >= 0
            }
        )
        if worker.status
        else []
    )
    if len(indexes) > hardware.MAX_RESOURCE_TARGETS:
        raise InvalidException(message='Too many devices for one resource query')
    return [hardware._device(worker, index, set()) for index in indexes]


def select_device(
    devices: List[hardware._Device], device_id: Optional[str]
) -> Optional[hardware._Device]:
    """Resolve only an explicitly selected current device; never choose a fallback."""
    if device_id is None:
        return None
    device = next((d for d in devices if d.public.id == device_id), None)
    if device is None:
        raise NotFoundException(message='GPU device not found')
    return device


async def resource_associations(
    worker: Worker,
    devices: List[hardware._Device],
    ctx: TenantContext,
    session: AsyncSession,
    selected: Optional[hardware._Device] = None,
) -> List[ResourceAssociation]:
    """Expose only deployments the caller could read through the Model routes."""
    if not ctx.is_platform_admin and ctx.org_role != OrgRole.OWNER:
        return []
    result = await session.exec(
        select(ModelInstance, Model)
        .join(Model, Model.id == ModelInstance.model_id)
        .where(
            or_(
                ModelInstance.worker_id == worker.id,
                ModelInstance.distributed_servers.is_not(None),
            ),
            ModelInstance.deleted_at.is_(None),
            Model.deleted_at.is_(None),
            *tenant_list_conditions(ctx, ModelInstance),
            *tenant_list_conditions(ctx, Model),
        )
    )
    visible_instance = tenant_stream_filter(ctx, ModelInstance)
    visible_model = tenant_stream_filter(ctx, Model)
    rows = [
        (instance, model)
        for instance, model in result.all()
        if visible_instance(instance)
        and visible_model(model)
        and get_model_instance_worker_match(instance, worker_id=worker.id).matched
    ]
    workers = await _placement_workers(worker, [i for i, _ in rows], ctx, session)
    associations = []
    for instance, model in rows:
        indexes = instance_gpu_indexes(instance, worker.id)
        bound = [device for device in devices if device.public.index in indexes]
        for device in bound:
            device.public.model_instance_ids.append(instance.id)
        if selected and selected.public.index not in indexes:
            continue
        allocations = hardware._allocated_gpus(instance)
        associations.append(
            ResourceAssociation(
                id=instance.id,
                name=instance.name,
                model_id=model.id,
                model_name=model.name,
                state=instance.state.value,
                state_message=instance.state_message,
                role=instance.role,
                backend=instance.backend,
                backend_version=instance.backend_version,
                gpu_device_ids=[device.public.id for device in bound],
                spans_workers=len(allocations) > 1,
                placements=[
                    ResourcePlacement(
                        worker_id=worker_id,
                        worker_name=workers[worker_id].name,
                        worker_ip=workers[worker_id].ip,
                        gpu_indexes=sorted(allocations[worker_id]),
                        is_main=worker_id == instance.worker_id,
                    )
                    for worker_id in sorted(
                        allocations, key=lambda id: (id != instance.worker_id, id)
                    )
                    if worker_id in workers
                ],
                placements_complete=allocations.keys() <= workers.keys(),
            )
        )
    return sorted(associations, key=lambda item: (item.name, item.id))


async def _placement_workers(worker, instances, ctx, session):
    """Resolve peer names only after checking each Worker's tenant visibility."""
    workers = {worker.id: worker}
    peer_ids = {
        id for instance in instances for id in hardware._allocated_gpus(instance)
    } - {worker.id}
    if not peer_ids:
        return workers
    result = await session.exec(
        select(Worker).where(
            Worker.id.in_(peer_ids),
            Worker.deleted_at.is_(None),
            *tenant_list_conditions(ctx, Worker),
        )
    )
    visible = tenant_stream_filter(ctx, Worker)
    workers.update(
        (peer.id, peer)
        for peer in result.all()
        if peer.id in peer_ids and visible(peer) and peer.deleted_at is None
    )
    return workers


def filesystem_targets(worker: Worker) -> List[hardware._Target]:
    """Use only unambiguous current mounts and owner-qualified Worker labels."""
    mounts = worker.status.filesystem or [] if worker.status else []
    counts = Counter(m.mount_point for m in mounts)
    identity = hardware._worker_identity(worker)
    if len(counts) > hardware.MAX_RESOURCE_TARGETS:
        raise InvalidException(message='Too many filesystems for one resource query')
    return [
        hardware._Target(
            {**identity, 'mountpoint': mount.mount_point} if identity else None,
            MetricDimensions(worker_id=worker.id, mount_point=mount.mount_point),
            mount.mount_point,
            'filesystem',
            None if identity else 'source_identity_unresolved',
        )
        for mount in mounts
        if mount.mount_point and counts[mount.mount_point] == 1
    ]


async def collect_resource_metrics(
    worker: Worker,
    devices: List[hardware._Device],
    *,
    window_seconds: int,
    device_id: Optional[str] = None,
    start: Optional[float] = None,
    end: Optional[float] = None,
    client: Optional[aiohttp.ClientSession] = None,
) -> ResourceMetrics:
    """Collect a directly authorized Worker or one of its current stable GPU identities."""
    interval = hardware._resource_interval(window_seconds, start, end)
    selected = select_device(devices, device_id)
    scoped = [selected] if selected else devices
    response = ResourceMetrics(
        worker_id=worker.id,
        selected_device_id=device_id,
        start=interval.start,
        end=interval.end,
        step=max(15, math.ceil(interval.seconds / 240)),
        last_refreshed_at=interval.end,
        devices=[device.public for device in scoped],
    )
    workers = {worker.id: worker}
    extra = {} if selected else {'filesystem_usage': filesystem_targets(worker)}
    url = hardware.prometheus_query.prometheus_url()
    if url is None:
        response.reason = 'prometheus_unavailable'
        targets = {
            key: hardware._resource_targets(key, scoped, workers)
            for key in hardware.HARDWARE_KEYS
        }
        response.metrics = {
            key: hardware._all_metric(key, value, query_failed=True).model_copy(
                update={'reason': response.reason} if value else {}
            )
            for key, value in {**targets, **extra}.items()
        }
    elif client is None:
        async with aiohttp.ClientSession() as owned:
            await hardware._collect_all_history(
                scoped, workers, response, owned, url, extra
            )
    else:
        await hardware._collect_all_history(
            scoped, workers, response, client, url, extra
        )
    hardware._update_device_history(scoped, response)
    response.available = any(
        m.status in ('available', 'stale') for m in response.metrics.values()
    )
    if not response.available and response.reason is None:
        response.reason = (
            'query_failed'
            if any(m.status == 'query_error' for m in response.metrics.values())
            else 'no_samples'
        )
    if 'filesystem_usage' in response.metrics:
        metric = response.metrics['filesystem_usage']
        metric.measurement_scope = 'filesystem'
        if not extra.get('filesystem_usage'):
            metric.status, metric.reason = 'no_data', 'no_samples'
    response.last_refreshed_at = time.time()
    return response
