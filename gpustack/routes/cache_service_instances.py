from typing import Optional

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from sqlmodel import select

from gpustack.api.exceptions import (
    ForbiddenException,
    InternalServerErrorException,
    NotFoundException,
)
from gpustack.api.tenant import (
    bypass_tenant_filter,
    cluster_scoped_system,
    scoped_cluster_row_visible,
    tenant_list_conditions,
)
from gpustack.schemas.cache_services import (
    CacheService,
    CacheServiceInstance,
    CacheServiceInstancePublic,
    CacheServiceInstanceUpdate,
    CacheServiceInstancesPublic,
    CacheServiceStateEnum,
)
from gpustack.schemas.principals import PrincipalType
from gpustack.server.db import async_session
from gpustack.server.deps import ListParamsDep, SessionDep, TenantContextDep

router = APIRouter()


@router.get("", response_model=CacheServiceInstancesPublic)
async def get_cache_service_instances(
    ctx: TenantContextDep,
    params: ListParamsDep,
    id: Optional[int] = None,
    cache_service_id: Optional[int] = None,
    worker_id: Optional[int] = None,
    state: Optional[CacheServiceStateEnum] = None,
):
    fields = {}
    if id:
        fields["id"] = id

    if cache_service_id:
        fields["cache_service_id"] = cache_service_id

    if worker_id:
        fields["worker_id"] = worker_id

    if state:
        fields["state"] = state

    if params.watch:
        # Cluster-bound service accounts (worker / cluster bootstrap) only
        # stream instances of their own cluster (via the denormalized
        # cluster_id). Instances carry no owner_principal_id — tenant
        # visibility derives from the parent service, so Org-scoped
        # callers are filtered against the services they own at stream
        # start.
        if cluster_scoped_system(ctx):

            def filter_func(data):
                return scoped_cluster_row_visible(ctx, data)

        elif ctx.current_principal_id is not None and not bypass_tenant_filter(ctx):
            async with async_session() as session:
                services = await CacheService.all_by_fields(
                    session,
                    fields={"owner_principal_id": ctx.current_principal_id},
                    extra_conditions=[CacheService.deleted_at.is_(None)],
                )
            visible_service_ids = {service.id for service in services}

            def filter_func(data):
                return getattr(data, "cache_service_id", None) in visible_service_ids

        else:
            filter_func = None

        return StreamingResponse(
            CacheServiceInstance.streaming(
                fields=fields,
                filter_func=filter_func,
            ),
            media_type="text/event-stream",
        )

    async with async_session() as session:
        extra_conditions = tenant_list_conditions(ctx, CacheServiceInstance)
        if ctx.current_principal_id is not None and not bypass_tenant_filter(ctx):
            # Tenant scoping derives from the parent service: instances
            # have no owner_principal_id of their own.
            extra_conditions.append(
                CacheServiceInstance.cache_service_id.in_(
                    select(CacheService.id).where(
                        CacheService.owner_principal_id == ctx.current_principal_id,
                        CacheService.deleted_at.is_(None),
                    )
                )
            )
        return await CacheServiceInstance.paginated_by_query(
            session=session,
            fields=fields,
            extra_conditions=extra_conditions,
            page=params.page,
            per_page=params.perPage,
        )


@router.get("/{id}", response_model=CacheServiceInstancePublic)
async def get_cache_service_instance(
    session: SessionDep,
    ctx: TenantContextDep,
    id: int,
):
    """One instance by ID. Workers read an instance back through this
    endpoint whenever their watch-backed cache is not authoritative (e.g.
    during a stream reconnect), so their state write-backs must not depend
    on the cache being warm."""
    instance = await CacheServiceInstance.one_by_id(session, id)
    if instance is None:
        raise NotFoundException(message="Cache service instance not found")

    # Visibility mirrors the list endpoint: cluster-bound service accounts
    # see their own cluster's rows, and other tenant-scoped callers see the
    # instances of the services they own.
    if cluster_scoped_system(ctx):
        if not scoped_cluster_row_visible(ctx, instance):
            raise NotFoundException(message="Cache service instance not found")
    elif ctx.current_principal_id is not None and not bypass_tenant_filter(ctx):
        service = await CacheService.one_by_id(session, instance.cache_service_id)
        if (
            service is None
            or service.deleted_at is not None
            or service.owner_principal_id != ctx.current_principal_id
        ):
            raise NotFoundException(message="Cache service instance not found")

    return instance


@router.put("/{id}", response_model=CacheServiceInstancePublic)
async def update_cache_service_instance(
    session: SessionDep,
    ctx: TenantContextDep,
    id: int,
    instance_in: CacheServiceInstanceUpdate,
):
    """Worker write-back of instance runtime state (ports, state, health,
    restart bookkeeping). Users act on instances through the parent
    service's endpoints instead."""
    if ctx.user is None or ctx.user.kind != PrincipalType.SYSTEM:
        raise ForbiddenException(
            message="Only system principals may update cache service instances"
        )

    instance = await CacheServiceInstance.one_by_id(session, id)
    if instance is None:
        raise NotFoundException(message="Cache service instance not found")

    # Cluster-bound service accounts write their own cluster's rows only,
    # mirroring the read endpoints' scoping.
    if cluster_scoped_system(ctx) and not scoped_cluster_row_visible(ctx, instance):
        raise NotFoundException(message="Cache service instance not found")

    try:
        await instance.update(session, instance_in)
    except Exception as e:
        raise InternalServerErrorException(
            message=f"Failed to update cache service instance: {e}"
        )

    return instance
