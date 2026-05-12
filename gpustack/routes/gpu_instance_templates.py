from contextlib import asynccontextmanager
from typing import Optional

from fastapi import APIRouter, Depends
from starlette.responses import StreamingResponse

from gpustack.api.exceptions import (
    NotFoundException,
    InternalServerErrorException,
    AlreadyExistsException,
)
from gpustack.api.tenant import (
    TenantContext,
    bypass_tenant_filter,
    cluster_visibility_conditions,
    assert_cluster_visible,
    validate_owner_principal,
    assert_org_owned_writable,
)
from gpustack.schemas.principals import PLATFORM_PRINCIPAL_ID
from gpustack.server.db import async_session

from gpustack.schemas import (
    GPUInstanceTemplatesPublic,
    GPUInstanceTemplateListParams,
    GPUInstanceTemplate,
    GPUInstanceTemplatePublic,
    GPUInstanceTemplateCreate,
    GPUInstanceTemplateUpdate,
)
from gpustack.server.deps import SessionDep, TenantContextDep

router = APIRouter()


@router.get("", response_model=GPUInstanceTemplatesPublic)
async def get_gpu_instance_templates(
    ctx: TenantContextDep,
    params: GPUInstanceTemplateListParams = Depends(),
    manufacturer: Optional[str] = None,
    search: Optional[str] = None,
):

    owner_principal_id = ctx.current_principal_id or PLATFORM_PRINCIPAL_ID

    fields: dict = {
        "owner_principal_id": owner_principal_id,
    }
    if manufacturer:
        fields["manufacturer"] = manufacturer

    fuzzy_fields: dict = {}
    if search:
        fuzzy_fields["name"] = search

    if params.watch:
        return StreamingResponse(
            GPUInstanceTemplate.streaming(
                fields=fields,
                fuzzy_fields=fuzzy_fields,
                filter_func=lambda tmpl: _is_template_visible(tmpl, ctx),
            ),
            media_type="text/event-stream",
        )

    async with async_session() as session:
        return await GPUInstanceTemplate.paginated_by_query(
            session=session,
            fields=fields,
            fuzzy_fields=fuzzy_fields,
            order_by=params.order_by,
            page=params.page,
            per_page=params.perPage,
            extra_conditions=cluster_visibility_conditions(ctx, GPUInstanceTemplate),
        )


@router.get("/{id}", response_model=GPUInstanceTemplatePublic)
async def get_gpu_instance_template(
    session: SessionDep,
    ctx: TenantContextDep,
    id: int,
):
    return ensure_visible(
        await GPUInstanceTemplate.one_by_id(
            session=session,
            id=id,
        ),
        ctx,
    )


@router.post("", response_model=GPUInstanceTemplatePublic)
async def create_gpu_instance_template(
    session: SessionDep,
    ctx: TenantContextDep,
    create_obj: GPUInstanceTemplateCreate,
):
    if create_obj.owner_principal_id is None:
        create_obj.owner_principal_id = (
            ctx.current_principal_id or PLATFORM_PRINCIPAL_ID
        )
    validate_owner_principal(
        create_obj.owner_principal_id, ctx, resource_label="GPU instance template"
    )

    existed = await GPUInstanceTemplate.one_by_fields(
        session=session,
        fields={
            "owner_principal_id": create_obj.owner_principal_id,
            "name": create_obj.name,
            "deleted_at": None,
        },
    )
    if existed is not None:
        raise AlreadyExistsException(
            message="GPU instance template already exists",
        )

    async with handle_error(
        message="Failed to create GPU instance template",
    ):
        return await GPUInstanceTemplate.create(
            session=session,
            source=create_obj,
        )


@router.put("/{id}", response_model=GPUInstanceTemplatePublic)
async def update_gpu_instance_template(
    session: SessionDep,
    ctx: TenantContextDep,
    id: int,
    update_obj: GPUInstanceTemplateUpdate,
):
    ret = ensure_writable(
        await GPUInstanceTemplate.one_by_id(
            session=session,
            id=id,
        ),
        ctx,
    )

    async with handle_error(
        message="Failed to update GPU instance template",
    ):
        await ret.update(
            session=session,
            source=update_obj,
        )
        return ret


@router.delete("/{id}")
async def delete_gpu_instance_template(
    session: SessionDep,
    ctx: TenantContextDep,
    id: int,
):
    ret = ensure_writable(
        await GPUInstanceTemplate.one_by_id(
            session=session,
            id=id,
        ),
        ctx,
    )

    async with handle_error(
        message="Failed to delete GPU instance template",
    ):
        await ret.delete(
            session=session,
        )


def ensure_visible(obj, ctx, message: str = "GPU instance template not found"):
    if obj is None:
        raise NotFoundException(message=message)
    assert_cluster_visible(ctx, obj, not_found_message=message)
    return obj


def ensure_writable(obj, ctx, message: str = "GPU instance template not found"):
    if obj is None:
        raise NotFoundException(message=message)
    assert_org_owned_writable(ctx, obj, resource_label=message)
    return obj


@asynccontextmanager
async def handle_error(message: str):
    try:
        yield
    except Exception as e:
        raise InternalServerErrorException(
            message=message,
        ) from e


def _is_template_visible(tmpl: GPUInstanceTemplate, ctx: TenantContext) -> bool:
    if bypass_tenant_filter(ctx):
        return True
    if ctx.has_org_context:
        return tmpl.owner_principal_id == ctx.current_principal_id
    return False
