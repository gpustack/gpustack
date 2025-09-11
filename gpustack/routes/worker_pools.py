from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from gpustack.api.exceptions import (
    InternalServerErrorException,
    NotFoundException,
    ForbiddenException,
)
from gpustack.server.deps import ListParamsDep, SessionDep, EngineDep
from gpustack.schemas.clusters import (
    WorkerPoolPublic,
    WorkerPoolsPublic,
    WorkerPoolUpdate,
    WorkerPool,
)

router = APIRouter()


@router.get("", response_model=WorkerPoolsPublic)
async def list(
    engine: EngineDep,
    session: SessionDep,
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
            WorkerPool.streaming(engine, fields=fields, fuzzy_fields=fuzzy_fields),
            media_type="text/event-stream",
        )

    return await WorkerPool.paginated_by_query(
        session=session,
        fields=fields,
        fuzzy_fields=fuzzy_fields,
        page=params.page,
        per_page=params.perPage,
    )


@router.get("/{id}", response_model=WorkerPoolPublic)
async def get(session: SessionDep, id: int):
    existing = await WorkerPool.one_by_id(session, id)
    if not existing or existing.deleted_at is not None:
        raise NotFoundException(message=f"worker pool {id} not found")

    return existing


@router.put("/{id}", response_model=WorkerPoolPublic)
async def update(session: SessionDep, id: int, input: WorkerPoolUpdate):
    existing = await WorkerPool.one_by_id(session, id)
    if not existing or existing.deleted_at is not None:
        raise NotFoundException(message=f"worker pool {id} not found")

    try:
        await WorkerPool.update(existing, session=session, source=input)
    except Exception as e:
        raise InternalServerErrorException(
            message=f"Failed to update worker pool {id}: {e}"
        )

    return await WorkerPool.one_by_id(session, id)


@router.delete("/{id}")
async def delete(session: SessionDep, id: int):
    existing = await WorkerPool.one_by_id(session, id)
    if not existing or existing.deleted_at is not None:
        raise NotFoundException(message=f"worker pool {id} not found")
    if len(existing.pool_workers) > 0:
        raise ForbiddenException(
            message=f"worker pool {id} has workers, cannot be deleted"
        )
    try:
        await existing.delete(session=session)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to delete worker pool: {e}")
