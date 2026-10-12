"""History operations scoped to an authorized model deployment."""

import math
from typing import Annotated, Any, Dict

from fastapi import APIRouter, Depends, Query, Response
from pydantic import ValidationError
from sqlalchemy import delete, func
from sqlmodel import select

from gpustack.api.exceptions import (
    BadRequestException,
    ConflictException,
    ForbiddenException,
    NotFoundException,
)
from gpustack.schemas.common import Pagination
from gpustack.schemas.deployment_document import (
    DeploymentChange,
    deployment_config_view,
)
from gpustack.schemas.model_revisions import (
    ModelRevision,
    ModelRevisionPublic,
    ModelRevisionSummary,
    ModelRevisionsPublic,
    ModelRollbackPreview,
    ModelRollbackRequest,
)
from gpustack.schemas.models import Model, ModelPublic, ModelUpdate
from gpustack.schemas.principals import PrincipalType
from gpustack.server.deps import SessionDep, TenantContextDep
from gpustack.server.model_revisions import (
    REVISION_FIELDS,
    deployment_spec,
    ensure_baseline,
    lock_model,
    prepare_history_read,
)


async def require_history_user(ctx: TenantContextDep) -> None:
    # Management authentication also accepts worker and cluster credentials.
    if ctx.user.kind == PrincipalType.SYSTEM:
        raise ForbiddenException(
            message="System principals cannot access model history"
        )


router = APIRouter(dependencies=[Depends(require_history_user)])


def _rollback_spec(model: Model, spec: Dict[str, Any]) -> Dict[str, Any]:
    # Absent fields carry no historical value; preserve their live configuration.
    return {
        **deployment_spec(model),
        **{field: spec[field] for field in REVISION_FIELDS if field in spec},
    }


async def _find_revision(
    session, model_id: int, revision: int, *, for_update: bool = False
) -> ModelRevision:
    statement = select(ModelRevision).where(
        ModelRevision.model_id == model_id, ModelRevision.revision == revision
    )
    if for_update:
        statement = statement.with_for_update()
    item = (await session.exec(statement)).one_or_none()
    if item is None:
        raise NotFoundException(message="Model revision not found")
    return item


@router.get("/{id}/revisions", response_model=ModelRevisionsPublic)
async def list_model_revisions(
    session: SessionDep,
    ctx: TenantContextDep,
    id: int,
    page: Annotated[int, Query(ge=1)] = 1,
    perPage: Annotated[int, Query(ge=1, le=100)] = 20,
) -> ModelRevisionsPublic:
    await prepare_history_read(session, ctx, id)
    total = (
        await session.exec(
            select(func.count())
            .select_from(ModelRevision)
            .where(ModelRevision.model_id == id)
        )
    ).one()
    # Project metadata in SQL so list requests never load configuration values.
    rows = (
        await session.exec(
            select(
                ModelRevision.id,
                ModelRevision.model_id,
                ModelRevision.revision,
                ModelRevision.created_at,
                ModelRevision.created_by,
            )
            .where(ModelRevision.model_id == id)
            .order_by(ModelRevision.revision.desc())
            .offset((page - 1) * perPage)
            .limit(perPage)
        )
    ).all()
    result = ModelRevisionsPublic(
        items=[ModelRevisionSummary.model_validate(dict(row._mapping)) for row in rows],
        pagination=Pagination(
            page=page,
            perPage=perPage,
            total=total,
            totalPage=math.ceil(total / perPage),
        ),
    )
    return result


@router.get("/{id}/revisions/{revision}", response_model=ModelRevisionPublic)
async def get_model_revision(
    session: SessionDep,
    ctx: TenantContextDep,
    id: int,
    revision: int,
) -> ModelRevisionPublic:
    await prepare_history_read(session, ctx, id)
    item = await _find_revision(session, id, revision)
    result = ModelRevisionPublic(
        **item.model_dump(exclude={"spec"}),
        spec=deployment_config_view(item.spec),
    )
    return result


@router.post("/{id}/rollback-preview", response_model=ModelRollbackPreview)
async def preview_model_rollback(
    session: SessionDep,
    ctx: TenantContextDep,
    id: int,
    body: ModelRollbackRequest,
) -> ModelRollbackPreview:
    model = await lock_model(session, ctx, id)
    latest = await ensure_baseline(session, model)
    target = await _find_revision(session, id, body.target_revision, for_update=True)
    current = deployment_spec(model)
    desired = _rollback_spec(model, target.spec)
    changes = [
        DeploymentChange(
            field=field, current=current.get(field), desired=desired.get(field)
        )
        for field in REVISION_FIELDS
        if current.get(field) != desired.get(field)
    ]
    result = ModelRollbackPreview(
        current_revision=latest.revision,
        target_revision=target.revision,
        current=deployment_config_view(current),
        desired=deployment_config_view(desired),
        changes=changes,
        changed=bool(changes),
    )
    await session.commit()
    return result


@router.post("/{id}/rollback", response_model=ModelPublic)
async def rollback_model(
    session: SessionDep,
    ctx: TenantContextDep,
    id: int,
    body: ModelRollbackRequest,
) -> ModelPublic:
    from gpustack.routes.models import save_model_update

    model = await lock_model(session, ctx, id)
    await ensure_baseline(session, model)
    target = await _find_revision(session, id, body.target_revision, for_update=True)
    # Identity, access settings and retention always come from the live resource.
    data = model.model_dump(include=set(ModelUpdate.model_fields))
    data.update(_rollback_spec(model, target.spec))
    try:
        model_in = ModelUpdate.model_validate(data)
    except ValidationError:
        raise BadRequestException(
            message="Historical deployment configuration is invalid"
        ) from None
    return await save_model_update(session, ctx, model, model_in)


@router.delete("/{id}/revisions/{revision}", status_code=204)
async def delete_model_revision(
    session: SessionDep,
    ctx: TenantContextDep,
    id: int,
    revision: int,
) -> Response:
    model = await lock_model(session, ctx, id)
    latest = await ensure_baseline(session, model)
    target = await _find_revision(session, id, revision, for_update=True)
    if target.revision == latest.revision:
        raise ConflictException(message="The latest model revision cannot be deleted")
    await session.execute(delete(ModelRevision).where(ModelRevision.id == target.id))
    await session.commit()
    return Response(status_code=204)
