from contextlib import asynccontextmanager
from typing import Optional

from fastapi import APIRouter, Depends
from sqlalchemy.exc import IntegrityError
from starlette.responses import StreamingResponse

from gpustack.api.exceptions import (
    ConflictException,
    InternalServerErrorException,
    NotFoundException,
)
from gpustack.api.tenant import (
    bypass_tenant_filter,
    TenantContext,
    assert_org_owned_writable,
    validate_owner_principal,
)
from gpustack.gpu_instances import validate_k8s_object_name

from gpustack.schemas import (
    GPUInstancePersistentVolumeType,
    GPUInstancePersistentVolumeTypeUpdate,
    GPUInstancePersistentVolumeTypePublic,
    GPUInstancePersistentVolumeTypeListParams,
    GPUInstancePersistentVolumeTypesPublic,
    GPUInstancePersistentVolumeTypeCreate,
)
from gpustack.schemas.principals import platform_principal_id
from gpustack.server.db import async_session
from gpustack.server.deps import SessionDep, TenantContextDep

router = APIRouter()


@router.get("", response_model=GPUInstancePersistentVolumeTypesPublic)
async def get_gpu_instance_persistent_volume_types(
    ctx: TenantContextDep,
    params: GPUInstancePersistentVolumeTypeListParams = Depends(),
    search: Optional[str] = None,
):
    owner_principal_id = ctx.current_principal_id or platform_principal_id()
    if bypass_tenant_filter(ctx):
        owner_principal_id = None

    fields: dict = {}
    if owner_principal_id is not None:
        fields["owner_principal_id"] = owner_principal_id

    fuzzy_fields: dict = {}
    if search:
        fuzzy_fields["name"] = search

    if params.watch:
        return StreamingResponse(
            GPUInstancePersistentVolumeType.streaming(
                fields=fields,
                fuzzy_fields=fuzzy_fields,
            ),
            media_type="text/event-stream",
        )

    async with async_session() as session:
        return await GPUInstancePersistentVolumeType.paginated_by_query(
            session=session,
            fields=fields,
            fuzzy_fields=fuzzy_fields,
            order_by=params.order_by,
            page=params.page,
            per_page=params.perPage,
        )


@router.get("/{id}", response_model=GPUInstancePersistentVolumeTypePublic)
async def get_gpu_instance_persistent_volume_type(
    session: SessionDep,
    ctx: TenantContextDep,
    id: int,
):
    return ensure_visible(
        await GPUInstancePersistentVolumeType.one_by_id(
            session=session,
            id=id,
        ),
        ctx,
    )


@router.post("", response_model=GPUInstancePersistentVolumeTypePublic)
async def create_gpu_instance_persistent_volume_type(
    session: SessionDep,
    ctx: TenantContextDep,
    create_obj: GPUInstancePersistentVolumeTypeCreate,
):
    validate_owner_principal(
        create_obj.owner_principal_id,
        ctx,
        resource_label="GPU instance persistent volume type",
    )
    create_obj.owner_principal_id = ctx.current_principal_id or platform_principal_id()

    _validate_create_obj(create_obj)

    async with handle_error(
        message="Failed to create GPU instance persistent volume type",
    ):
        return await GPUInstancePersistentVolumeType.create(
            session=session,
            source=create_obj,
        )


@router.put("/{id}", response_model=GPUInstancePersistentVolumeTypePublic)
async def update_gpu_instance_persistent_volume_type(
    session: SessionDep,
    ctx: TenantContextDep,
    id: int,
    update_obj: GPUInstancePersistentVolumeTypeUpdate,
):
    ret = ensure_writable(
        await GPUInstancePersistentVolumeType.one_by_id(
            session=session,
            id=id,
        ),
        ctx,
    )

    async with handle_error(
        message="Failed to update GPU instance persistent volume type",
    ):
        await ret.update(
            session=session,
            source=update_obj,
        )
        return ret


@router.delete("/{id}")
async def delete_gpu_instance_persistent_volume_type(
    session: SessionDep,
    ctx: TenantContextDep,
    id: int,
):
    ret = ensure_writable(
        await GPUInstancePersistentVolumeType.one_by_id(
            session=session,
            id=id,
        ),
        ctx,
    )

    async with handle_error(
        message="Failed to delete GPU instance persistent volume type",
    ):
        await ret.delete(
            session=session,
        )


def ensure_visible(obj, ctx: TenantContext):
    if obj and is_visible(obj, ctx):
        return obj
    raise NotFoundException(message="GPU instance persistent volume type not found")


def ensure_writable(obj, ctx: TenantContext):
    if obj is None:
        raise NotFoundException(message="GPU instance persistent volume type not found")
    assert_org_owned_writable(
        ctx, obj, resource_label="GPU instance persistent volume type"
    )
    return obj


def is_visible(obj, ctx: TenantContext) -> bool:
    if bypass_tenant_filter(ctx):
        return True
    return ctx.current_principal_id == obj.owner_principal_id


@asynccontextmanager
async def handle_error(message: str):
    try:
        yield
    except IntegrityError as e:
        # ``ON DELETE RESTRICT`` from
        # gpu_instance_persistent_volumes.persistent_volume_type_id
        # surfaces here when a persistent volume still references this
        # type.
        raise ConflictException(
            message=message
            + ", as it is still referenced by existing GPU instance persistent volumes",
        ) from e
    except Exception as e:
        raise InternalServerErrorException(
            message=message,
        ) from e


def _validate_create_obj(create_obj: GPUInstancePersistentVolumeTypeCreate):
    validate_k8s_object_name(create_obj.name)
