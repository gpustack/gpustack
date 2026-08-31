from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode, urlparse

import aiohttp
from fastapi import APIRouter, Request, status
from fastapi.responses import PlainTextResponse, RedirectResponse, StreamingResponse

from gpustack import envs
from gpustack.api.exceptions import (
    AlreadyExistsException,
    BadRequestException,
    InternalServerErrorException,
    NotFoundException,
)
from gpustack.api.responses import StreamingResponseWithStatusCode
from gpustack.api.tenant import (
    assert_resource_visible,
    bypass_tenant_filter,
    cluster_scoped_system,
    scoped_cluster_row_visible,
    tenant_list_conditions,
)
from gpustack.routes.models import assert_cluster_belongs_to_org
from gpustack.schemas.cache_providers import CUSTOM_VERSION
from gpustack.schemas.cache_services import (
    CacheServiceAttachedMetrics,
    CacheServiceMetricsPublic,
    CacheService,
    CacheServiceBase,
    CacheServiceConfig,
    CacheServiceCreate,
    CacheServiceEndpoint,
    CacheServiceInstance,
    CacheServiceInstancePublic,
    CacheServiceInstancesPublic,
    CacheServiceModeEnum,
    CacheServiceModelSummary,
    CacheServiceModelsPublic,
    CacheServicePublic,
    CacheServiceStateEnum,
    CacheServiceUpdate,
    CacheServicesPublic,
    TestCacheServiceConnectionRequest,
    TestCacheServiceConnectionResponse,
)
from gpustack.schemas.common import Pagination
from gpustack.config.config import get_global_config
from gpustack.schemas.clusters import Cluster
from gpustack.schemas.models import Model, ModelInstance, get_backend
from gpustack.schemas.principals import PrincipalType, platform_principal_id
from gpustack.schemas.workers import Worker
from gpustack.server.cache_provider_catalog import get_cache_provider
from gpustack.server.cache_services import probe_cache_service
from gpustack.server.db import async_session
from gpustack.schemas.principals import OrgRole
from gpustack.server.cache_service_metrics import (
    collect_cache_service_metrics,
    parse_window as parse_metrics_window,
)
from gpustack.server.deps import ListParamsDep, SessionDep, TenantContextDep
from gpustack.server.worker_request import request_to_worker, stream_to_worker
from gpustack.utils.grafana import resolve_grafana_base_url
from gpustack.worker.logs import LogOptionsDep

router = APIRouter()

SECRET_PLACEHOLDER = "********"
"""Stands in for declared password-typed values on user-facing reads; an
update carrying it back means "unchanged"."""


def _system_caller(ctx) -> bool:
    """Workers and cluster service accounts authenticate as SYSTEM. They
    render the real credentials into the cache server container, so
    secret redaction applies only to user-facing reads."""
    return ctx.user is not None and ctx.user.kind == PrincipalType.SYSTEM


def _secret_param_slots(provider, service) -> List[Tuple[Dict[str, Any], str]]:
    """(params dict, field name) pairs holding a declared password-typed
    value: the external endpoint params and each L2 storage entry's
    params."""
    slots: List[Tuple[Dict[str, Any], str]] = []
    endpoint = getattr(service, "endpoint", None)
    if endpoint is not None and endpoint.params:
        for field in provider.external_fields:
            if field.type == "password" and field.name in endpoint.params:
                slots.append((endpoint.params, field.name))
    config = getattr(service, "config", None)
    for storage in (config.l2_storages if config else None) or []:
        backend = provider.l2_backends.get(storage.backend)
        if backend is None or not storage.params:
            continue
        for field in backend.fields:
            if field.type == "password" and field.name in storage.params:
                slots.append((storage.params, field.name))
    return slots


def _redacted_for_user(cache_service) -> CacheServicePublic:
    """A detached copy of the service with declared password-typed values
    replaced by the placeholder. Detached because redaction must never
    write through to the row (or to the event payload other stream
    subscribers see)."""
    data = (
        # The dump-then-validate pair is the deep copy (validating from
        # attributes would pass nested objects through by reference and
        # let the redaction write through to the row). The dump itself
        # would warn on every call: the enum-typed columns deliberately
        # store plain strings (no DB enum casts), so ORM rows carry str
        # values that the validation below coerces — expected, not a bug.
        cache_service.model_dump(warnings=False)
        if hasattr(cache_service, "model_dump")
        else cache_service
    )
    public = CacheServicePublic.model_validate(data)
    provider = get_cache_provider(public.provider_name)
    if provider is None:
        return public
    for params, name in _secret_param_slots(provider, public):
        if params.get(name):
            params[name] = SECRET_PLACEHOLDER
    return public


def _restore_secret_params(
    cache_service_in: CacheServiceUpdate, existing: CacheService
) -> None:
    """Placeholder values in an update mean "unchanged": put the stored
    secret back so a redacted GET round-trips through an edit. A
    placeholder with no stored counterpart (e.g. on a freshly added L2
    entry) is dropped rather than stored literally. L2 entries match
    their stored counterpart by backend, positionally among same-backend
    entries."""
    provider = get_cache_provider(existing.provider_name)
    if provider is None:
        return

    def restore(params: Dict[str, Any], name: str, stored: Dict[str, Any]) -> None:
        if params.get(name) != SECRET_PLACEHOLDER:
            return
        if name in stored:
            params[name] = stored[name]
        else:
            params.pop(name, None)

    if cache_service_in.endpoint is not None and cache_service_in.endpoint.params:
        existing_endpoint = getattr(existing, "endpoint", None)
        stored_params = (existing_endpoint.params if existing_endpoint else None) or {}
        for field in provider.external_fields:
            if field.type == "password":
                restore(cache_service_in.endpoint.params, field.name, stored_params)

    existing_config = getattr(existing, "config", None)
    stored_by_backend: Dict[str, List[Dict[str, Any]]] = {}
    for storage in (existing_config.l2_storages if existing_config else None) or []:
        stored_by_backend.setdefault(storage.backend, []).append(storage.params or {})
    position: Dict[str, int] = {}
    for storage in (
        cache_service_in.config.l2_storages if cache_service_in.config else None
    ) or []:
        index = position.get(storage.backend, 0)
        position[storage.backend] = index + 1
        backend = provider.l2_backends.get(storage.backend)
        if backend is None or not storage.params:
            continue
        stored_entries = stored_by_backend.get(storage.backend) or []
        stored_params = stored_entries[index] if index < len(stored_entries) else {}
        for field in backend.fields:
            if field.type == "password":
                restore(storage.params, field.name, stored_params)


def _reject_placeholder_secrets(cache_service_in: CacheServiceBase) -> None:
    """A create has no stored value a placeholder could stand for;
    storing it literally would silently break the credential."""
    provider = get_cache_provider(cache_service_in.provider_name)
    if provider is None:
        return
    for params, name in _secret_param_slots(provider, cache_service_in):
        if params.get(name) == SECRET_PLACEHOLDER:
            raise BadRequestException(
                message=(
                    f"'{name}' carries the redaction placeholder "
                    f"'{SECRET_PLACEHOLDER}'; supply the actual secret value"
                )
            )


@router.get("", response_model=CacheServicesPublic)
async def get_cache_services(
    ctx: TenantContextDep,
    params: ListParamsDep,
    id: Optional[int] = None,
    cluster_id: Optional[int] = None,
    worker_id: Optional[int] = None,
    state: Optional[CacheServiceStateEnum] = None,
    mode: Optional[CacheServiceModeEnum] = None,
    provider_name: Optional[str] = None,
    search: Optional[str] = None,
):
    fields = {}
    search = search.strip() if search else None
    fuzzy_fields = {"name": search} if search else {}
    if id:
        fields["id"] = id

    if cluster_id:
        fields["cluster_id"] = cluster_id

    if worker_id:
        fields["worker_id"] = worker_id

    if state:
        fields["state"] = state

    if mode:
        fields["mode"] = mode

    if provider_name:
        fields["provider_name"] = provider_name

    # System principals (workers, cluster service accounts) and admin in
    # "All" mode must see every Org's cache services regardless of their
    # ``principal_id`` — otherwise a worker's awatch stream would silently
    # filter out services scheduled to it on clusters outside its
    # Personal Org.
    if ctx.current_principal_id is not None and not bypass_tenant_filter(ctx):
        fields["owner_principal_id"] = ctx.current_principal_id

    if params.watch:
        # Cluster-bound service accounts (worker / cluster bootstrap)
        # only stream cache services of their own cluster.
        filter_func = (
            (lambda data: scoped_cluster_row_visible(ctx, data))
            if cluster_scoped_system(ctx)
            else None
        )
        event_transform = None
        if not _system_caller(ctx):

            async def redact_event(event):
                event.data = _redacted_for_user(event.data)

            event_transform = redact_event
        return StreamingResponse(
            CacheService.streaming(
                fields=fields,
                fuzzy_fields=fuzzy_fields,
                filter_func=filter_func,
                event_transform=event_transform,
            ),
            media_type="text/event-stream",
        )

    async with async_session() as session:
        extra_conditions = tenant_list_conditions(ctx, CacheService)
        result = await CacheService.paginated_by_query(
            session=session,
            fields=fields,
            fuzzy_fields=fuzzy_fields,
            extra_conditions=extra_conditions,
            page=params.page,
            per_page=params.perPage,
        )
        if not _system_caller(ctx):
            result.items = [_redacted_for_user(item) for item in result.items]
        return result


@router.post("/test-connection", response_model=TestCacheServiceConnectionResponse)
async def test_cache_service_connection(
    connection_in: TestCacheServiceConnectionRequest,
):
    provider = get_cache_provider(connection_in.provider_name)
    if provider is None:
        raise BadRequestException(
            message=f"Unknown cache provider '{connection_in.provider_name}'"
        )

    reachable, message = await probe_cache_service(provider, connection_in.endpoint)
    return TestCacheServiceConnectionResponse(reachable=reachable, message=message)


async def _fetch_managed_cache_service(session, ctx, id: int) -> CacheService:
    cache_service = await CacheService.one_by_id(session, id)
    assert_resource_visible(
        ctx, cache_service, not_found_message="Cache service not found"
    )
    if cache_service.mode != CacheServiceModeEnum.MANAGED:
        raise BadRequestException(message="Only managed cache services have instances")
    return cache_service


async def _fetch_service_instance(
    session, cache_service: CacheService, instance_id: int
) -> CacheServiceInstance:
    instance = await CacheServiceInstance.one_by_id(session, instance_id)
    if instance is None or instance.cache_service_id != cache_service.id:
        raise NotFoundException(message="Cache service instance not found")
    return instance


@router.delete("/{id}/instances/{instance_id}")
async def delete_cache_service_instance(
    session: SessionDep,
    ctx: TenantContextDep,
    id: int,
    instance_id: int,
):
    """Delete one instance of the service, leaving siblings untouched.
    The owning worker tears down the cache server container on the
    delete event, and the controller immediately recreates a fresh
    PENDING row for the instance's worker (restart budget included), so
    deletion doubles as a relaunch from scratch."""
    cache_service = await _fetch_managed_cache_service(session, ctx, id)
    instance = await _fetch_service_instance(session, cache_service, instance_id)

    try:
        await instance.delete(session)
    except Exception as e:
        raise InternalServerErrorException(
            message=f"Failed to delete cache service instance: {e}"
        )


@router.get("/{id}/instances", response_model=CacheServiceInstancesPublic)
async def get_cache_service_instances_of_service(
    session: SessionDep,
    ctx: TenantContextDep,
    id: int,
):
    """The service's instances ordered by worker, for the detail page."""
    cache_service = await CacheService.one_by_id(session, id)
    assert_resource_visible(
        ctx, cache_service, not_found_message="Cache service not found"
    )

    instances = await CacheServiceInstance.all_by_fields(
        session, {"cache_service_id": cache_service.id}
    )
    instances = sorted(instances, key=lambda instance: instance.worker_id)
    items = [
        CacheServiceInstancePublic.model_validate(instance) for instance in instances
    ]
    return CacheServiceInstancesPublic(
        items=items,
        pagination=Pagination(
            page=1,
            perPage=max(len(items), 1),
            total=len(items),
            totalPage=1,
        ),
    )


async def _proxy_instance_logs(
    request: Request,
    instance: CacheServiceInstance,
    worker: Worker,
    log_options,
):
    """Proxy the instance's container logs from its worker. The worker
    resolves the workload name from the (service, instance) id pair."""
    timeout = aiohttp.ClientTimeout(total=envs.PROXY_TIMEOUT, sock_connect=5)
    params = {
        "tail": log_options.tail,
        "follow": log_options.follow,
        "cache_service_id": instance.cache_service_id,
    }

    if log_options.follow:

        def on_exception(e: Exception, t: aiohttp.ClientTimeout) -> tuple[str, int]:
            msg = (
                str(e)
                if not isinstance(e, TimeoutError)
                else f"Log stream timed out ({t.total} seconds). Please reopen the log page."
            )
            return f"\x1b[999;1H{msg}\n", status.HTTP_500_INTERNAL_SERVER_ERROR

        return StreamingResponseWithStatusCode(
            stream_to_worker(
                worker=worker,
                method="GET",
                path=f"cacheServiceInstanceLogs/{instance.id}",
                proxy_client=request.app.state.http_client,
                no_proxy_client=request.app.state.http_client_no_proxy,
                params=params,
                timeout=timeout,
                on_exception=on_exception,
                raw=True,
            ),
            media_type="application/octet-stream",
        )

    resp, body = await request_to_worker(
        worker=worker,
        method="GET",
        path=f"cacheServiceInstanceLogs/{instance.id}",
        proxy_client=request.app.state.http_client,
        no_proxy_client=request.app.state.http_client_no_proxy,
        params=params,
        timeout=timeout,
    )
    return PlainTextResponse(
        content=body.decode() if body else "", status_code=resp.status
    )


async def _fetch_instance_log_worker(session, instance: CacheServiceInstance) -> Worker:
    worker = await Worker.one_by_id(session, instance.worker_id)
    if not worker:
        raise NotFoundException(message="Cache service instance's worker not found")
    return worker


@router.get("/{id}/logs")
async def get_cache_service_logs(
    request: Request,
    ctx: TenantContextDep,
    id: int,
    log_options: LogOptionsDep,
):
    """Stream the managed cache server's container logs when the service
    runs a single instance; multi-instance services must be addressed per
    instance."""
    # Inline session released after the initial lookups so a long-lived
    # follow-log stream doesn't hold a database connection for its duration.
    async with async_session() as session:
        cache_service = await _fetch_managed_cache_service(session, ctx, id)

        instances = await CacheServiceInstance.all_by_fields(
            session, {"cache_service_id": cache_service.id}
        )
        if not instances:
            raise BadRequestException(message="Cache service has no instances yet")
        if len(instances) > 1:
            raise BadRequestException(
                message=(
                    "Cache service runs multiple instances; use "
                    "/cache-services/{id}/instances/{instance_id}/logs"
                )
            )

        instance = instances[0]
        worker = await _fetch_instance_log_worker(session, instance)

    return await _proxy_instance_logs(request, instance, worker, log_options)


@router.get("/{id}/instances/{instance_id}/logs")
async def get_cache_service_instance_logs(
    request: Request,
    ctx: TenantContextDep,
    id: int,
    instance_id: int,
    log_options: LogOptionsDep,
):
    """Stream one instance's cache server container logs from its worker."""
    async with async_session() as session:
        cache_service = await _fetch_managed_cache_service(session, ctx, id)
        instance = await _fetch_service_instance(session, cache_service, instance_id)
        worker = await _fetch_instance_log_worker(session, instance)

    return await _proxy_instance_logs(request, instance, worker, log_options)


@router.get("/{id}/models", response_model=CacheServiceModelsPublic)
async def get_cache_service_models(
    session: SessionDep,
    ctx: TenantContextDep,
    id: int,
):
    """Model deployments whose shared extended KV cache points at this
    service, for the service detail page."""
    cache_service = await CacheService.one_by_id(session, id)
    assert_resource_visible(
        ctx, cache_service, not_found_message="Cache service not found"
    )

    models = await _models_referencing_cache_service(session, cache_service)
    items = [
        CacheServiceModelSummary(
            id=model.id,
            name=model.name,
            replicas=model.replicas,
            ready_replicas=model.ready_replicas,
            backend=get_backend(model),
        )
        for model in sorted(models, key=lambda model: model.name)
    ]
    return CacheServiceModelsPublic(items=items)


@router.get("/{id}/dashboard")
async def get_cache_service_dashboard(
    session: SessionDep,
    ctx: TenantContextDep,
    id: int,
    request: Request,
):
    cache_service = await CacheService.one_by_id(session, id)
    assert_resource_visible(
        ctx, cache_service, not_found_message="Cache service not found"
    )

    cfg = get_global_config()
    # A provider may declare its own Grafana dashboard (provisioned
    # alongside the generic one); its services' Grafana entries land there
    # instead of the generic cache-service dashboard.
    provider = get_cache_provider(cache_service.provider_name)
    dashboard_uid = (
        provider.dashboard_uid if provider and provider.dashboard_uid else None
    ) or cfg.grafana_cache_service_dashboard_uid
    if not cfg.get_grafana_url() or not dashboard_uid:
        raise InternalServerErrorException(
            message="Grafana dashboard settings are not configured"
        )

    cluster = None
    if cache_service.cluster_id is not None:
        cluster = await Cluster.one_by_id(session, cache_service.cluster_id)

    query_params = {}
    if cluster is not None:
        query_params["var-cluster_name"] = cluster.name
    query_params["var-cache_service_name"] = cache_service.name

    grafana_base = resolve_grafana_base_url(cfg, request)
    dashboard_url = f"{grafana_base}/d/{dashboard_uid}/{dashboard_uid}"
    if query_params:
        dashboard_url = f"{dashboard_url}?{urlencode(query_params)}"

    return RedirectResponse(url=dashboard_url, status_code=302)


@router.get("/{id}/metrics", response_model=CacheServiceMetricsPublic)
async def get_cache_service_metrics(
    request: Request,
    session: SessionDep,
    ctx: TenantContextDep,
    id: int,
    window: str = "1h",
    workers: Optional[str] = None,
):
    """Chartable semantic metric series for the service, translated from
    the provider's catalog declaration and queried from the built-in
    Prometheus with a server-injected service-label selector. The whole
    router mounts Org-owner-only; the explicit assertion below keeps
    this telemetry gated on its own, independent of the mount policy."""
    cache_service = await CacheService.one_by_id(session, id)
    assert_resource_visible(
        ctx, cache_service, not_found_message="Cache service not found"
    )
    ctx.assert_org_role(OrgRole.OWNER)
    try:
        window_seconds = parse_metrics_window(window)
    except ValueError as e:
        raise BadRequestException(message=str(e))
    provider = get_cache_provider(cache_service.provider_name)
    # The database enumerates the attached deployments' instances (the
    # rows the UI shows); metrics only fill their numbers. The model-id
    # scope also bounds the Prometheus queries, so a caller only ever
    # reads the engines wired to this service. The cluster's models load
    # whole and filter in Python: extended_kv_cache is a JSON column and
    # the platform convention keeps JSON predicates out of SQL (the set
    # is bounded by one cluster's deployments).
    models = await Model.all_by_fields(
        session,
        fields={"cluster_id": cache_service.cluster_id},
        extra_conditions=[Model.deleted_at.is_(None)],
    )
    attached_models = {
        model.id: model
        for model in models
        if model.extended_kv_cache
        and model.extended_kv_cache.is_shared()
        and model.extended_kv_cache.cache_service_id == cache_service.id
    }
    attached = []
    if attached_models:
        instances = await ModelInstance.all_by_fields(
            session,
            extra_conditions=[ModelInstance.model_id.in_(attached_models.keys())],
        )
        attached = [
            CacheServiceAttachedMetrics(
                model_id=instance.model_id,
                model_name=attached_models[instance.model_id].name,
                model_instance_name=instance.name,
                worker_name=instance.worker_name,
            )
            for instance in instances
        ]
        # a stable row order regardless of how the batched query returns
        attached.sort(
            key=lambda row: (row.model_name or "", row.model_instance_name or "")
        )
    worker_names = (
        [name.strip() for name in workers.split(",") if name.strip()]
        if workers
        else None
    )
    if worker_names and len(worker_names) > 100:
        raise BadRequestException(message="workers filter accepts at most 100 names")
    if worker_names:
        # the worker scope applies to the whole response: charts filter
        # inside PromQL, the row set filters here
        selected = set(worker_names)
        attached = [row for row in attached if row.worker_name in selected]
    return await collect_cache_service_metrics(
        provider.metrics_for(cache_service.provider_version) if provider else None,
        cache_service.id,
        window_seconds,
        cluster_id=cache_service.cluster_id,
        attached=attached,
        worker_names=worker_names,
        client=getattr(request.app.state, "http_client_no_proxy", None),
    )


@router.get("/{id}", response_model=CacheServicePublic)
async def get_cache_service(
    session: SessionDep,
    ctx: TenantContextDep,
    id: int,
):
    cache_service = await CacheService.one_by_id(session, id)
    assert_resource_visible(
        ctx, cache_service, not_found_message="Cache service not found"
    )
    if _system_caller(ctx):
        return cache_service
    return _redacted_for_user(cache_service)


def _validate_cache_service_provider(cache_service_in: CacheServiceCreate) -> None:
    """Reject provider/mode/version combinations the catalog can't serve."""
    provider = get_cache_provider(cache_service_in.provider_name)
    if provider is None:
        raise BadRequestException(
            message=f"Unknown cache provider '{cache_service_in.provider_name}'"
        )

    if cache_service_in.mode.value not in provider.supported_modes:
        raise BadRequestException(
            message=(
                f"Cache provider '{provider.name}' does not support "
                f"mode '{cache_service_in.mode}'"
            )
        )

    # The reserved "custom" identifier is not a catalog version; it is
    # checked by _validate_cache_service_custom_version.
    if cache_service_in.provider_version == CUSTOM_VERSION:
        return

    # A version config carries the managed container's image and run
    # command; external services run no container, so their provider_version
    # is informational (the integrations' support matrix) and not resolved here.
    if cache_service_in.mode != CacheServiceModeEnum.MANAGED:
        return

    version_config, resolved_version = provider.get_version_config(
        cache_service_in.provider_version
    )
    if version_config is None:
        raise BadRequestException(
            message=(
                f"Cache provider '{provider.name}' has no "
                f"version '{resolved_version}'"
            )
        )


def _validate_cache_service_custom_version(cache_service_in: CacheServiceBase) -> None:
    """The reserved provider_version "custom" pins a user-supplied container
    image on a managed service: the provider must opt in and config.image
    carries the image. With any declared version the provider's image is
    rendered instead, so a supplied config.image would be silently ignored —
    reject it up front. External services don't run an image at all, so the
    custom version is rejected there too."""
    image = cache_service_in.config.image if cache_service_in.config else None

    if cache_service_in.provider_version != CUSTOM_VERSION:
        if image:
            raise BadRequestException(
                message=(
                    f"config.image is only applicable when provider_version "
                    f"is '{CUSTOM_VERSION}'"
                )
            )
        return

    if cache_service_in.mode != CacheServiceModeEnum.MANAGED:
        raise BadRequestException(
            message=(
                f"provider_version '{CUSTOM_VERSION}' is only applicable to "
                f"managed cache services; external services do not run a "
                f"container image"
            )
        )

    provider = get_cache_provider(cache_service_in.provider_name)
    if provider is None or not provider.custom_version:
        raise BadRequestException(
            message=(
                f"Cache provider '{cache_service_in.provider_name}' does not "
                f"allow the custom version"
            )
        )

    if not image or not image.strip():
        raise BadRequestException(
            message=(
                f"config.image is required when provider_version is "
                f"'{CUSTOM_VERSION}'"
            )
        )


def _validate_cache_service_config(config: Optional[CacheServiceConfig]) -> None:
    """Config parameters are a free-form escape hatch passed to the cache
    server command line; only items that can never form a valid command
    token are rejected."""
    if config is None or config.parameters is None:
        return
    for parameter in config.parameters:
        if not isinstance(parameter, str) or not parameter.strip():
            raise BadRequestException(
                message="config.parameters items must be non-empty strings"
            )


def _validate_cache_service_l2_storage(cache_service_in: CacheServiceBase) -> None:
    """L2 storage config only applies to managed services and must match the
    provider's declared adapter backends. Each entry is checked for a known
    backend key, all required fields set, no undeclared parameter names
    (typo guard), and numeric values for number-typed fields; optional adapter
    backends skip and clear their hidden fields while disabled. Across entries,
    no two may deliver a value through the same env var — env vars are
    process-global, so the values would clobber each other."""
    config = cache_service_in.config
    if config is None or config.l2_storages is None:
        return

    # An empty list means "no L2 storage"; store the canonical form.
    if not config.l2_storages:
        config.l2_storages = None
        return

    if cache_service_in.mode != CacheServiceModeEnum.MANAGED:
        raise BadRequestException(
            message=("config.l2_storages is only applicable to managed cache services")
        )

    provider = get_cache_provider(cache_service_in.provider_name)
    if provider is None or not provider.l2_adapter_flag:
        raise BadRequestException(
            message=(
                f"Cache provider '{cache_service_in.provider_name}' "
                f"does not support L2 storage"
            )
        )

    env_sources: Dict[str, str] = {}
    for l2_storage in config.l2_storages:
        backend_key = l2_storage.backend
        backend_spec = provider.l2_backends.get(backend_key)
        if backend_spec is None:
            raise BadRequestException(
                message=(
                    f"Cache provider '{provider.name}' has no L2 storage "
                    f"backend '{backend_key}'"
                )
            )

        if not backend_spec.adapter_flag_optional:
            if l2_storage.adapter_flag_enabled is not None:
                raise BadRequestException(
                    message=(
                        f"adapter_flag_enabled is not applicable to L2 "
                        f"backend '{backend_key}'"
                    )
                )
            adapter_enabled = True
        else:
            adapter_enabled = (
                backend_spec.adapter_flag_default
                if l2_storage.adapter_flag_enabled is None
                else l2_storage.adapter_flag_enabled
            )
            if not adapter_enabled:
                # Hidden fields have no runtime meaning while the adapter is
                # disabled. Drop stale form values so a later read mirrors
                # what the UI shows and no secret remains stored invisibly.
                l2_storage.params = {}
                continue

        params = l2_storage.params or {}
        declared = {field.name: field for field in backend_spec.fields}

        unknown = sorted(set(params) - set(declared))
        if unknown:
            raise BadRequestException(
                message=(
                    f"Unknown L2 storage parameter(s) for backend "
                    f"'{backend_key}': " + ", ".join(unknown)
                )
            )

        # Some adapters retain legacy fields while their current envelope
        # uses a nested backend_params mapping. For the current shape,
        # legacy required fields are irrelevant and must not prevent it from
        # being accepted; only an explicitly legacy payload uses them.
        required_fields = backend_spec.fields
        mapped_names = set(backend_spec.adapter_params.values())
        if backend_spec.adapter_backend and not mapped_names:
            mapped_names = {field.name for field in backend_spec.fields}
        legacy_shape = bool(
            backend_spec.adapter_params
            and "metadata_endpoint" in params
            and not (set(params) & mapped_names)
        )
        if (
            backend_spec.adapter_params or backend_spec.adapter_backend
        ) and not legacy_shape:
            required_fields = [
                field
                for field in backend_spec.fields
                if field.name in mapped_names or field.env_name
            ]

        missing = [
            field.name
            for field in required_fields
            if field.required
            and (params.get(field.name) is None or params.get(field.name) == "")
        ]
        if missing:
            raise BadRequestException(
                message=(
                    f"Missing required L2 storage parameter(s) for backend "
                    f"'{backend_key}': " + ", ".join(missing)
                )
            )

        for name, value in params.items():
            field = declared[name]
            if (
                field.type == "number"
                and value is not None
                and (isinstance(value, bool) or not isinstance(value, (int, float)))
            ):
                raise BadRequestException(
                    message=f"L2 storage parameter '{name}' must be a number"
                )

        # Mirror the rendering rule: only fields that resolve to a value
        # (explicitly or via default) reach the container env.
        for field in backend_spec.fields:
            if not field.env_name:
                continue
            value = params.get(field.name, field.default)
            if value is None or value == "":
                continue
            if field.env_name in env_sources:
                raise BadRequestException(
                    message=(
                        f"L2 storage entries '{env_sources[field.env_name]}' "
                        f"and '{backend_key}' both deliver the env var "
                        f"'{field.env_name}'; only one entry may set it"
                    )
                )
            env_sources[field.env_name] = backend_key


def _validate_cache_service_worker_selector(
    cache_service_in: CacheServiceBase,
) -> None:
    """worker_selector narrows which cluster workers a managed per-node
    service places instances on. On any other service shape it would be
    silently ignored, so it is rejected up front: external services have
    no server-driven placement, and singleton providers place on the
    explicitly picked worker_id."""
    selector = cache_service_in.worker_selector
    if not selector:
        # An empty selector means "every worker"; store the canonical form.
        cache_service_in.worker_selector = None
        return

    if cache_service_in.mode != CacheServiceModeEnum.MANAGED:
        raise BadRequestException(
            message=(
                "worker_selector is not applicable for external cache "
                "services; it scopes managed cache services whose provider "
                "runs one instance per worker node"
            )
        )

    provider = get_cache_provider(cache_service_in.provider_name)
    topology = provider.topology if provider else "singleton"
    if topology != "per_node":
        raise BadRequestException(
            message=(
                f"worker_selector is not applicable for cache provider "
                f"'{cache_service_in.provider_name}': it scopes providers "
                "that run one instance per worker node, while this provider "
                "runs a single instance on the picked worker_id"
            )
        )


def _validate_management_url(cache_service_in: CacheServiceCreate) -> None:
    """Validate the engine-management link and canonicalize blanks to None.

    The field rides ``config``, which managed and external services both
    accept, so this runs with the top-level validators for either mode —
    after ``_validate_cache_service_provider``, which already rejected
    unknown providers.
    """
    if cache_service_in.config is None:
        return
    management_url = (cache_service_in.config.management_url or "").strip()
    cache_service_in.config.management_url = management_url or None
    if not management_url:
        return
    provider = get_cache_provider(cache_service_in.provider_name)
    if provider is None or not provider.management_url:
        raise BadRequestException(
            message=(
                f"config.management_url is not supported by cache "
                f"provider '{cache_service_in.provider_name}'"
            )
        )
    if len(management_url) > 2048:
        raise BadRequestException(
            message="config.management_url must be at most 2048 characters"
        )
    # urlparse accepts whitespace inside the netloc, so guard separately
    parsed = urlparse(management_url)
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.netloc
        or any(ch.isspace() for ch in management_url)
    ):
        raise BadRequestException(
            message="config.management_url must be a valid http(s) URL"
        )
    # a display-only link must not smuggle credentials into stored config
    if parsed.username or parsed.password:
        raise BadRequestException(
            message="config.management_url must not embed credentials"
        )


async def _validate_cache_service_mode(
    session, cache_service_in: CacheServiceCreate
) -> None:
    """Enforce the mode-specific field contract.

    Managed services with a singleton-topology provider run on one worker
    the server deploys to, so a worker in the service's cluster must be
    chosen up front. Per-node providers derive their placement from the
    cluster's workers, so a worker pick would be meaningless. External
    services are reached at a caller-supplied endpoint, so worker_id must
    stay empty — the health checker would otherwise treat the service as
    managed.
    """
    if cache_service_in.mode == CacheServiceModeEnum.MANAGED:
        # An explicit capacity keeps the cache server's memory footprint
        # deliberate instead of falling through to an engine-internal
        # default the platform can't see.
        if cache_service_in.config is None or not cache_service_in.config.ram_size:
            raise BadRequestException(
                message="config.ram_size is required for managed cache services"
            )
        provider = get_cache_provider(cache_service_in.provider_name)
        topology = provider.topology if provider else "singleton"
        if topology == "per_node":
            if cache_service_in.worker_id is not None:
                raise BadRequestException(
                    message=(
                        f"worker_id is not applicable for cache provider "
                        f"'{cache_service_in.provider_name}': it runs one "
                        "instance per worker node, and instances follow "
                        "the cluster's workers"
                    )
                )
            return

        if not cache_service_in.worker_id:
            raise BadRequestException(
                message="worker_id is required for managed cache services"
            )
        worker = await Worker.one_by_id(session, cache_service_in.worker_id)
        if worker is None or worker.deleted_at is not None:
            raise BadRequestException(
                message=f"Worker {cache_service_in.worker_id} not found"
            )
        if worker.cluster_id != cache_service_in.cluster_id:
            raise BadRequestException(
                message="The selected worker does not belong to the selected cluster"
            )
    else:
        _validate_external_endpoint(cache_service_in.endpoint)
        if cache_service_in.worker_id is not None:
            raise BadRequestException(
                message="worker_id is not applicable for external cache services"
            )


def _validate_external_fields(cache_service_in: CacheServiceBase) -> None:
    """Every external field the provider declares required must have a
    value in endpoint.params. Only applies to external services; managed
    services carry no such fields."""
    if cache_service_in.mode != CacheServiceModeEnum.EXTERNAL:
        return
    provider = get_cache_provider(cache_service_in.provider_name)
    if provider is None or not provider.external_fields:
        return

    params = (
        cache_service_in.endpoint.params if cache_service_in.endpoint else {}
    ) or {}
    for field in provider.external_fields:
        if not field.required:
            continue
        value = params.get(field.name)
        if value is None or value == "":
            raise BadRequestException(
                message=(
                    f"endpoint.params.{field.name} is required for external "
                    f"cache services of provider '{provider.name}'"
                )
            )

    # Field values substitute into JSON connector templates (and config
    # files written through a heredoc) as plain strings; a quote,
    # backslash, or line break would corrupt the rendered artifact with
    # an engine-side failure far from its cause.
    for name, value in params.items():
        if isinstance(value, str) and any(
            ch in value for ch in ('"', "\\", "\n", "\r")
        ):
            raise BadRequestException(
                message=(
                    f"endpoint.params.{name} must not contain quotes, "
                    "backslashes, or line breaks"
                )
            )


def _validate_managed_fields(cache_service_in: CacheServiceBase) -> None:
    """config.fields must match the provider's managed_fields declaration:
    an unknown name is a typo (the worker would silently ignore it), and a
    non-numeric value or an off-list option would render into the cache
    server command and crash-loop the container through its restart
    budget."""
    config = cache_service_in.config
    values = (config.fields if config else None) or {}
    if not values:
        return
    provider = get_cache_provider(cache_service_in.provider_name)
    if provider is None:
        return
    declared = {field.name: field for field in provider.managed_fields}
    for name, value in values.items():
        field = declared.get(name)
        if field is None:
            raise BadRequestException(
                message=(
                    f"config.fields.{name} is not declared by provider "
                    f"'{cache_service_in.provider_name}'"
                )
            )
        if value is None or value == "":
            continue
        if field.type == "number":
            try:
                number = float(value)
            except (TypeError, ValueError):
                raise BadRequestException(
                    message=f"config.fields.{name} must be a number"
                )
            if (field.min is not None and number < field.min) or (
                field.max is not None and number > field.max
            ):
                raise BadRequestException(
                    message=(
                        f"config.fields.{name} must be between "
                        f"{field.min} and {field.max}"
                    )
                )
        elif field.type == "boolean":
            if not isinstance(value, bool):
                raise BadRequestException(
                    message=f"config.fields.{name} must be a boolean"
                )
        if field.options and str(value) not in field.options:
            raise BadRequestException(
                message=(
                    f"config.fields.{name} must be one of "
                    f"{', '.join(field.options)}"
                )
            )
        # String values reach shell tokens and JSON templates verbatim.
        if isinstance(value, str) and any(
            ch in value for ch in ('"', "\\", "\n", "\r")
        ):
            raise BadRequestException(
                message=(
                    f"config.fields.{name} must not contain quotes, "
                    "backslashes, or line breaks"
                )
            )


def _validate_external_endpoint(endpoint: Optional[CacheServiceEndpoint]) -> None:
    """An external service needs a connectable address: a fixed host+port
    (or url)."""
    if endpoint is None or not ((endpoint.host and endpoint.port) or endpoint.url):
        raise BadRequestException(
            message=(
                "endpoint with host and port (or url) is required "
                "for external cache services"
            )
        )


@router.post("", response_model=CacheServicePublic)
async def create_cache_service(
    session: SessionDep, ctx: TenantContextDep, cache_service_in: CacheServiceCreate
):
    # A cache service always lives on a concrete cluster — there is no
    # default-cluster resolution for it, so the caller must choose one.
    if not cache_service_in.cluster_id:
        raise BadRequestException(message="cluster_id is required")

    # Resolve the owning Org first — admin in "All" mode (no current
    # principal) inherits the chosen cluster's Org, or falls back to the
    # platform Org. The same value drives both the uniqueness pre-check
    # below and the row we stamp on insert.
    target_org_id = ctx.current_principal_id
    cluster = None
    if target_org_id is None:
        cluster = await Cluster.one_by_id(session, cache_service_in.cluster_id)
        if cluster is None:
            raise NotFoundException(
                message=f"Cluster {cache_service_in.cluster_id} not found"
            )
        target_org_id = cluster.owner_principal_id
    if target_org_id is None:
        target_org_id = platform_principal_id()

    # The chosen cluster must exist, be visible to the caller, and be owned
    # by the target Org.
    await assert_cluster_belongs_to_org(
        ctx, session, cache_service_in.cluster_id, target_org_id, cluster=cluster
    )

    # Cache service names are unique within their Org.
    existing = await CacheService.one_by_fields(
        session,
        {"name": cache_service_in.name, "owner_principal_id": target_org_id},
    )
    if existing:
        raise AlreadyExistsException(
            message=(
                f"Cache service with name '{cache_service_in.name}' already exists."
            )
        )

    _validate_cache_service_provider(cache_service_in)
    _validate_cache_service_custom_version(cache_service_in)
    _validate_cache_service_config(cache_service_in.config)
    _validate_management_url(cache_service_in)
    _validate_managed_fields(cache_service_in)
    _validate_cache_service_l2_storage(cache_service_in)
    _validate_cache_service_worker_selector(cache_service_in)
    _validate_external_fields(cache_service_in)
    _reject_placeholder_secrets(cache_service_in)
    await _validate_cache_service_mode(session, cache_service_in)

    cache_service_dict = cache_service_in.model_dump()
    cache_service_dict["owner_principal_id"] = target_org_id
    cache_service_dict["state"] = CacheServiceStateEnum.PENDING

    try:
        cache_service = await CacheService.create(session, cache_service_dict)
    except Exception as e:
        raise InternalServerErrorException(
            message=f"Failed to create cache service: {e}"
        )

    return cache_service


@router.put("/{id}", response_model=CacheServicePublic)
async def update_cache_service(
    session: SessionDep,
    ctx: TenantContextDep,
    id: int,
    cache_service_in: CacheServiceUpdate,
):
    cache_service = await CacheService.one_by_id(session, id)
    assert_resource_visible(
        ctx, cache_service, not_found_message="Cache service not found"
    )

    # SYSTEM principals (workers writing back state / port / health) may
    # touch any field. User callers cannot re-point the identity fields
    # that drive deployment — a provider/mode/cluster change is a
    # different service, not an edit. Config (and, for external mode,
    # endpoint) edits stay allowed.
    is_system = ctx.user is not None and ctx.user.kind == PrincipalType.SYSTEM
    if not is_system:
        immutable_fields = {
            "provider_name": cache_service.provider_name,
            "mode": cache_service.mode,
            "cluster_id": cache_service.cluster_id,
        }
        for field_name, current_value in immutable_fields.items():
            if getattr(cache_service_in, field_name) != current_value:
                raise BadRequestException(
                    message=f"Field '{field_name}' cannot be changed"
                )
        # Redacted reads round-trip through edits: placeholder values mean
        # "unchanged" and are restored from the stored row before the
        # validations below see them.
        _restore_secret_params(cache_service_in, cache_service)

        # Renames stay unique per Org — without the pre-check the unique
        # constraint surfaces as a 500 instead of a conflict.
        if cache_service_in.name != cache_service.name:
            existing = await CacheService.one_by_fields(
                session,
                {
                    "name": cache_service_in.name,
                    "owner_principal_id": cache_service.owner_principal_id,
                },
            )
            if existing:
                raise AlreadyExistsException(
                    message=(
                        f"Cache service with name '{cache_service_in.name}' "
                        "already exists."
                    )
                )

    # Update runs the same validation set as create: without it, a
    # provider_version unknown to the catalog or a cleared ram_size only
    # surfaces as a worker-side start failure, and worker_id could be
    # re-pointed across clusters.
    _validate_cache_service_provider(cache_service_in)
    _validate_cache_service_custom_version(cache_service_in)
    _validate_cache_service_config(cache_service_in.config)
    _validate_management_url(cache_service_in)
    _validate_managed_fields(cache_service_in)
    _validate_cache_service_l2_storage(cache_service_in)
    _validate_cache_service_worker_selector(cache_service_in)
    if cache_service_in.mode == CacheServiceModeEnum.EXTERNAL:
        _validate_external_fields(cache_service_in)
    # Restore resolved every placeholder above; anything still carrying
    # the literal sentinel would be stored as the secret itself.
    _reject_placeholder_secrets(cache_service_in)
    await _validate_cache_service_mode(session, cache_service_in)

    try:
        await cache_service.update(session, cache_service_in)
    except Exception as e:
        raise InternalServerErrorException(
            message=f"Failed to update cache service: {e}"
        )

    return cache_service


async def _models_referencing_cache_service(
    session, cache_service: CacheService
) -> List[Model]:
    """Models whose shared extended KV cache points at this service.
    ``extended_kv_cache`` is a JSON column, so the reference scan runs in
    Python over the owner Org's models rather than with dialect-specific
    JSON predicates."""
    models = await Model.all_by_fields(
        session, {"owner_principal_id": cache_service.owner_principal_id}
    )
    return [
        model
        for model in models
        if model.deleted_at is None
        and model.extended_kv_cache is not None
        and model.extended_kv_cache.is_shared()
        and model.extended_kv_cache.cache_service_id == cache_service.id
    ]


@router.delete("/{id}")
async def delete_cache_service(session: SessionDep, ctx: TenantContextDep, id: int):
    cache_service = await CacheService.one_by_id(session, id)
    assert_resource_visible(
        ctx, cache_service, not_found_message="Cache service not found"
    )

    # A model whose shared extended KV cache points at this service would
    # silently lose its cache backend on delete.
    models = await _models_referencing_cache_service(session, cache_service)
    referencing = sorted(model.name for model in models)
    if referencing:
        raise BadRequestException(
            message=("Cache service is in use by model(s): " + ", ".join(referencing))
        )

    try:
        await cache_service.delete(session)
    except Exception as e:
        raise InternalServerErrorException(
            message=f"Failed to delete cache service: {e}"
        )
