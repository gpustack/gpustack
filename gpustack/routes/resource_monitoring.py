"""Worker and GPU detail reads use their own resource visibility boundary."""

from typing import Optional
from urllib.parse import urlencode

from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse

from gpustack.api.exceptions import (
    ForbiddenException,
    InvalidException,
    NotFoundException,
)
from gpustack.api.tenant import (
    TenantContext,
    assert_resource_visible,
    assert_org_owned_writable,
)
from gpustack.api.exceptions import HTTPException
from gpustack.config.config import get_global_config
from gpustack.routes import workers as worker_routes
from gpustack.schemas.clusters import Cluster
from gpustack.schemas.gpu_devices import GPUDevice
from gpustack.schemas.model_metrics import MonitoringCapabilities
from gpustack.schemas.principals import PrincipalType
from gpustack.schemas.resource_metrics import ResourceDetail, ResourceMetrics
from gpustack.schemas.workers import Worker
from gpustack.server.deps import SessionDep, TenantContextDep
from gpustack.monitoring.prometheus import parse_window
from gpustack.monitoring.time_range import (
    monitoring_interval,
    collect_storage_policy,
)
from gpustack.monitoring.resources import (
    collect_resource_metrics,
    inventory_devices,
    resource_associations,
    select_device,
)
from gpustack.utils.grafana import resolve_grafana_base_url

router = APIRouter()


async def _resolve(session, ctx, id, gpu=False):
    if ctx is None or getattr(ctx.user, 'kind', None) != PrincipalType.USER:
        raise ForbiddenException(
            message='Resource monitoring requires a user principal'
        )
    if not ctx.is_platform_admin and ctx.current_principal_id is None:
        raise ForbiddenException(message='A resolved resource tenant is required')
    if (gpu and (not isinstance(id, str) or not id or len(id) > 512)) or (
        not gpu and (type(id) is not int or not 0 < id <= 2**63 - 1)
    ):
        raise InvalidException(message='Invalid resource identifier')
    row = await (GPUDevice if gpu else Worker).one_by_id(session, id)
    assert_resource_visible(ctx, row)
    worker = await Worker.one_by_id(session, row.worker_id) if gpu else row
    assert_resource_visible(ctx, worker, not_found_message='Worker not found')
    if worker.deleted_at is not None:
        raise NotFoundException(message='Worker not found')
    devices = inventory_devices(worker)
    selected = None
    if gpu:
        candidates = (
            [d for d in (worker.status.gpu_devices or []) if d.index == row.index]
            if worker.status
            else []
        )
        if (
            len(candidates) != 1
            or (candidates[0].uuid or '') != (row.uuid or '')
            or (candidates[0].device_chip_index or 0) != (row.device_chip_index or 0)
        ):
            raise NotFoundException(message='GPU device not found')
        selected = next(
            (d.public.id for d in devices if d.public.index == row.index), None
        )
        if selected is None:
            raise NotFoundException(message='GPU device not found')
    return worker, devices, selected


def _interval(window, start, end):
    try:
        return monitoring_interval(
            parse_window(window) if start is None and end is None else 0, start, end
        )
    except ValueError as exc:
        raise InvalidException(message=str(exc)) from exc


async def _detail(session, ctx, id, gpu, device_id=None):
    worker, devices, selected = await _resolve(session, ctx, id, gpu)
    selected = select_device(devices, selected if gpu else device_id)
    instances = await resource_associations(worker, devices, ctx, session, selected)
    current = worker_routes.to_worker_public(
        worker, False, await worker_routes._lookup_allocated(worker.id)
    )
    cluster = (
        await Cluster.one_by_id(session, worker.cluster_id)
        if worker.cluster_id
        else None
    )
    cluster_name = None
    if cluster is not None:
        try:
            assert_resource_visible(ctx, cluster)
            cluster_name = cluster.name
        except NotFoundException:
            pass
    return ResourceDetail(
        worker=current,
        can_edit=can_edit_worker(ctx, worker),
        cluster_name=cluster_name,
        selected_device_id=selected.public.id if selected else None,
        devices=[d.public for d in devices],
        instances=instances,
    )


def can_edit_worker(ctx: TenantContext, worker: Worker) -> bool:
    """Mirror the existing edit/private-key guard without expanding write access."""
    try:
        assert_org_owned_writable(ctx, worker, resource_label='worker')
        return True
    except HTTPException:
        return False


@router.get('/workers/{id}/detail', response_model=ResourceDetail)
async def get_worker_detail(
    session: SessionDep, ctx: TenantContextDep, id: int, device_id: Optional[str] = None
):
    return await _detail(session, ctx, id, False, device_id)


@router.get('/gpu-devices/{id}/detail', response_model=ResourceDetail)
async def get_gpu_detail(session: SessionDep, ctx: TenantContextDep, id: str):
    return await _detail(session, ctx, id, True)


async def _metrics(request, session, ctx, id, gpu, window, start, end, device_id=None):
    worker, devices, selected = await _resolve(session, ctx, id, gpu)
    interval = _interval(window, start, end)
    return await collect_resource_metrics(
        worker,
        devices,
        window_seconds=interval.seconds,
        device_id=selected if gpu else device_id,
        start=start,
        end=end,
        client=getattr(request.app.state, 'http_client_no_proxy', None),
    )


@router.get('/workers/{id}/metrics', response_model=ResourceMetrics)
async def get_worker_metrics(
    request: Request,
    session: SessionDep,
    ctx: TenantContextDep,
    id: int,
    window: str = '1h',
    start: Optional[float] = None,
    end: Optional[float] = None,
    device_id: Optional[str] = None,
):
    return await _metrics(
        request, session, ctx, id, False, window, start, end, device_id
    )


@router.get('/gpu-devices/{id}/metrics', response_model=ResourceMetrics)
async def get_gpu_metrics(
    request: Request,
    session: SessionDep,
    ctx: TenantContextDep,
    id: str,
    window: str = '1h',
    start: Optional[float] = None,
    end: Optional[float] = None,
):
    return await _metrics(request, session, ctx, id, True, window, start, end)


async def _capabilities(request, session, ctx, id, gpu):
    await _resolve(session, ctx, id, gpu)
    policy = await collect_storage_policy(
        getattr(request.app.state, 'http_client_no_proxy', None)
    )
    return MonitoringCapabilities(**policy)


@router.get(
    '/workers/{id}/monitoring-capabilities', response_model=MonitoringCapabilities
)
async def get_worker_capabilities(
    request: Request, session: SessionDep, ctx: TenantContextDep, id: int
):
    return await _capabilities(request, session, ctx, id, False)


@router.get(
    '/gpu-devices/{id}/monitoring-capabilities', response_model=MonitoringCapabilities
)
async def get_gpu_capabilities(
    request: Request, session: SessionDep, ctx: TenantContextDep, id: str
):
    return await _capabilities(request, session, ctx, id, True)


async def _dashboard(
    request, session, ctx, id, gpu, window, start, end, device_id=None
):
    worker, devices, selected = await _resolve(session, ctx, id, gpu)
    device = select_device(devices, selected if gpu else device_id)
    interval = _interval(window, start, end)
    cfg = get_global_config()
    if not cfg.get_grafana_url() or not cfg.grafana_worker_dashboard_uid:
        raise InvalidException(message='Grafana dashboard settings are not configured')
    query = {
        'var-worker_name': worker.name,
        'from': int(interval.start * 1000),
        'to': int(interval.end * 1000),
    }
    if worker.cluster_id:
        cluster = await Cluster.one_by_id(session, worker.cluster_id)
        if cluster:
            assert_resource_visible(ctx, cluster, not_found_message='Cluster not found')
            query['var-cluster_name'] = cluster.name
    if device:
        slots = [
            i
            for i, d in enumerate(worker.status.gpu_devices)
            if d.index == device.public.index
        ]
        if len(slots) != 1:
            raise InvalidException(message='GPU dashboard identity is ambiguous')
        # Node metrics use inventory positions and a compatibility chip value;
        # native history independently verifies the UUID/chip tuple.
        query['var-gpu_index'] = str(slots[0])
        query['var-gpu_chip_index'] = '0'
    base = resolve_grafana_base_url(cfg, request)
    return RedirectResponse(
        f'{base}/d/{cfg.grafana_worker_dashboard_uid}/gpustack-worker?{urlencode(query)}',
        status_code=302,
    )


@router.get('/workers/{id}/monitoring-dashboard')
async def get_worker_monitoring_dashboard(
    request: Request,
    session: SessionDep,
    ctx: TenantContextDep,
    id: int,
    window: str = '1h',
    start: Optional[float] = None,
    end: Optional[float] = None,
    device_id: Optional[str] = None,
):
    return await _dashboard(
        request, session, ctx, id, False, window, start, end, device_id
    )


@router.get('/gpu-devices/{id}/monitoring-dashboard')
async def get_gpu_monitoring_dashboard(
    request: Request,
    session: SessionDep,
    ctx: TenantContextDep,
    id: str,
    window: str = '1h',
    start: Optional[float] = None,
    end: Optional[float] = None,
):
    return await _dashboard(request, session, ctx, id, True, window, start, end)
