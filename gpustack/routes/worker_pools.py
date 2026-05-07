from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import selectinload

from gpustack.api.exceptions import (
    InternalServerErrorException,
    NotFoundException,
    ForbiddenException,
)
from gpustack.api.tenant import (
    assert_org_owned_writable,
    assert_resource_visible,
    tenant_list_conditions,
)
from gpustack.server.db import async_session
from gpustack.server.deps import ListParamsDep, SessionDep, TenantContextDep
from gpustack.schemas.clusters import (
    WorkerPoolPublic,
    WorkerPoolsPublic,
    WorkerPoolUpdate,
    WorkerPool,
)

WORKER_POOL_LOAD_OPTIONS = [selectinload(WorkerPool.pool_workers)]

router = APIRouter()


@router.get("", response_model=WorkerPoolsPublic)
async def list(
    ctx: TenantContextDep,
    params: ListParamsDep,
    name: str = None,
    search: str = None,
    cluster_id: int = None,
):
    fuzzy_fields = {}
    if search:
        fuzzy_fields = {"name": search}

    fields = {"deleted_at": None}

    if cluster_id:
        fields["cluster_id"] = cluster_id

    if name:
        fields["name"] = name

    if params.watch:
        return StreamingResponse(
            WorkerPool.streaming(
                fields=fields,
                fuzzy_fields=fuzzy_fields,
                options=WORKER_POOL_LOAD_OPTIONS,
            ),
            media_type="text/event-stream",
        )

    async with async_session() as session:
        # Worker pools mirror their cluster's owner_principal_id; same filter
        # rules as cloud_credentials apply.
        extra_conditions = tenant_list_conditions(ctx, WorkerPool)
        return await WorkerPool.paginated_by_query(
            session=session,
            fields=fields,
            fuzzy_fields=fuzzy_fields,
            extra_conditions=extra_conditions,
            page=params.page,
            per_page=params.perPage,
            options=WORKER_POOL_LOAD_OPTIONS,
        )


@router.get("/{id}", response_model=WorkerPoolPublic)
async def get(session: SessionDep, ctx: TenantContextDep, id: int):
    existing = await WorkerPool.one_by_id(session, id, options=WORKER_POOL_LOAD_OPTIONS)
    if not existing or existing.deleted_at is not None:
        raise NotFoundException(message=f"worker pool {id} not found")
    assert_resource_visible(
        ctx,
        existing,
        not_found_message=f"worker pool {id} not found",
    )
    return existing


@router.put("/{id}", response_model=WorkerPoolPublic)
async def update(
    session: SessionDep, ctx: TenantContextDep, id: int, input: WorkerPoolUpdate
):
    existing = await WorkerPool.one_by_id(session, id)
    if not existing or existing.deleted_at is not None:
        raise NotFoundException(message=f"worker pool {id} not found")
    assert_org_owned_writable(ctx, existing, resource_label="worker pool")

    try:
        await WorkerPool.update(existing, session=session, source=input)
    except Exception as e:
        raise InternalServerErrorException(
            message=f"Failed to update worker pool {id}: {e}"
        )

    return await WorkerPool.one_by_id(session, id, options=WORKER_POOL_LOAD_OPTIONS)


@router.delete("/{id}")
async def delete(session: SessionDep, ctx: TenantContextDep, id: int):
    existing = await WorkerPool.one_by_id(session, id, options=WORKER_POOL_LOAD_OPTIONS)
    if not existing or existing.deleted_at is not None:
        raise NotFoundException(message=f"worker pool {id} not found")
    assert_org_owned_writable(ctx, existing, resource_label="worker pool")
    if len(existing.pool_workers) > 0:
        raise ForbiddenException(
            message=f"worker pool {id} has workers, cannot be deleted"
        )
    try:
        await existing.delete(session=session)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to delete worker pool: {e}")
