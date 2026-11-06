from typing import Any, Awaitable, Callable, Optional

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from sqlmodel import false, select

from gpustack.api.streaming import tenant_streaming
from gpustack.api.exceptions import (
    ForbiddenException,
    InternalServerErrorException,
    NotFoundException,
)
from gpustack.api.tenant import (
    TenantContext,
    bypass_tenant_filter,
    cluster_scoped_system,
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
from gpustack.server.services import CacheServiceInstanceService

router = APIRouter()


def _system_instance_stream_filter(
    ctx: TenantContext,
) -> Callable[[Any], Awaitable[bool]]:
    # Parent ownership is immutable. Cache it per stream, including for deletion
    # events whose parent has already been removed. New services resolve lazily
    # so a long-lived worker stream can discover newly deployed workloads.
    owners = {}

    async def visible(instance: Any) -> bool:
        if (
            ctx.scoped_cluster_owner_id is None
            or getattr(instance, "cluster_id", None) != ctx.scoped_cluster_id
        ):
            return False
        service_id = getattr(instance, "cache_service_id", None)
        if service_id is None:
            return False
        if service_id not in owners:
            async with async_session() as session:
                # Include soft-deleted parents to authorize instance cleanup
                # events even when this stream has not cached the owner yet.
                service = await CacheService.one_by_id(session, service_id)
            if service is None:
                return False
            owners[service_id] = service.owner_principal_id
        return owners[service_id] == ctx.scoped_cluster_owner_id

    return visible


async def _assert_instance_visible(
    session, ctx: TenantContext, instance: Optional[CacheServiceInstance]
) -> None:
    not_found = "Cache service instance not found"
    if instance is None:
        raise NotFoundException(message=not_found)
    if cluster_scoped_system(ctx):
        if instance.cluster_id != ctx.scoped_cluster_id:
            raise NotFoundException(message=not_found)
        owner_id = ctx.scoped_cluster_owner_id
    elif bypass_tenant_filter(ctx):
        return
    else:
        owner_id = ctx.current_principal_id
    if owner_id is None:
        raise NotFoundException(message=not_found)
    service = await CacheService.one_by_id(session, instance.cache_service_id)
    if (
        service is None
        or service.deleted_at is not None
        or service.owner_principal_id != owner_id
    ):
        raise NotFoundException(message=not_found)


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
        if cluster_scoped_system(ctx):
            filter_func = _system_instance_stream_filter(ctx)
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

            def filter_func(data):
                return bypass_tenant_filter(ctx)

        return StreamingResponse(
            tenant_streaming(
                CacheServiceInstance,
                ctx,
                fields=fields,
                visibility_filter=filter_func,
            ),
            media_type="text/event-stream",
        )

    async with async_session() as session:
        extra_conditions = []
        if cluster_scoped_system(ctx):
            extra_conditions.append(
                CacheServiceInstance.cluster_id == ctx.scoped_cluster_id
            )
            owner_id = ctx.scoped_cluster_owner_id
        else:
            owner_id = ctx.current_principal_id
        if cluster_scoped_system(ctx) or not bypass_tenant_filter(ctx):
            # Instances derive tenant ownership from their parent service.
            extra_conditions.append(
                CacheServiceInstance.cache_service_id.in_(
                    select(CacheService.id).where(
                        (
                            CacheService.owner_principal_id == owner_id
                            if owner_id is not None
                            else false()
                        ),
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
    await _assert_instance_visible(session, ctx, instance)

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
    await _assert_instance_visible(session, ctx, instance)

    try:
        await CacheServiceInstanceService(session).update(
            instance, instance_in.model_dump(exclude_unset=True)
        )
    except Exception as e:
        raise InternalServerErrorException(
            message=f"Failed to update cache service instance: {e}"
        )

    return instance
