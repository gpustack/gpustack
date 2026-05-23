from contextlib import asynccontextmanager
from typing import Optional

from fastapi import APIRouter, Depends
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.responses import StreamingResponse

from gpustack.api.exceptions import (
    ConflictException,
    InternalServerErrorException,
    InvalidException,
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
    GPUInstancePersistentVolume,
    GPUInstancePersistentVolumeType,
    GPUInstancePersistentVolumeUpdate,
    GPUInstancePersistentVolumePublic,
    GPUInstancePersistentVolumeListParams,
    GPUInstancePersistentVolumesPublic,
    GPUInstancePersistentVolumeCreate,
)
from gpustack.schemas.principals import platform_principal_id
from gpustack.server.db import async_session
from gpustack.server.deps import SessionDep, TenantContextDep

router = APIRouter()


@router.get("", response_model=GPUInstancePersistentVolumesPublic)
async def get_gpu_instance_persistent_volumes(
    ctx: TenantContextDep,
    params: GPUInstancePersistentVolumeListParams = Depends(),
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
            GPUInstancePersistentVolume.streaming(
                fields=fields,
                fuzzy_fields=fuzzy_fields,
            ),
            media_type="text/event-stream",
        )

    async with async_session() as session:
        return await GPUInstancePersistentVolume.paginated_by_query(
            session=session,
            fields=fields,
            fuzzy_fields=fuzzy_fields,
            order_by=params.order_by,
            page=params.page,
            per_page=params.perPage,
        )


@router.get("/{id}", response_model=GPUInstancePersistentVolumePublic)
async def get_gpu_instance_persistent_volume(
    session: SessionDep,
    ctx: TenantContextDep,
    id: int,
):
    return ensure_visible(
        await GPUInstancePersistentVolume.one_by_id(
            session=session,
            id=id,
        ),
        ctx,
    )


@router.post("", response_model=GPUInstancePersistentVolumePublic)
async def create_gpu_instance_persistent_volume(
    session: SessionDep,
    ctx: TenantContextDep,
    create_obj: GPUInstancePersistentVolumeCreate,
):
    validate_owner_principal(
        create_obj.owner_principal_id,
        ctx,
        resource_label="GPU instance persistent volume",
    )
    create_obj.owner_principal_id = ctx.current_principal_id or platform_principal_id()

    persistent_volume_type_id = await _validate_create_obj(session, create_obj)

    source = _build_create_source(create_obj, persistent_volume_type_id)
    async with handle_error(
        message="Failed to create GPU instance persistent volume",
    ):
        return await GPUInstancePersistentVolume.create(
            session=session,
            source=source,
        )


@router.put("/{id}", response_model=GPUInstancePersistentVolumePublic)
async def update_gpu_instance_persistent_volume(
    session: SessionDep,
    ctx: TenantContextDep,
    id: int,
    update_obj: GPUInstancePersistentVolumeUpdate,
):
    ret = ensure_writable(
        await GPUInstancePersistentVolume.one_by_id(
            session=session,
            id=id,
        ),
        ctx,
    )

    async with handle_error(
        message="Failed to update GPU instance persistent volume",
    ):
        await ret.update(
            session=session,
            source=update_obj,
        )
        return ret


@router.delete("/{id}")
async def delete_gpu_instance_persistent_volume(
    session: SessionDep,
    ctx: TenantContextDep,
    id: int,
):
    ret = ensure_writable(
        await GPUInstancePersistentVolume.one_by_id(
            session=session,
            id=id,
        ),
        ctx,
    )

    async with handle_error(
        message="Failed to delete GPU instance persistent volume",
    ):
        await ret.delete(
            session=session,
        )


def ensure_visible(obj, ctx: TenantContext):
    if obj and is_visible(obj, ctx):
        return obj
    raise NotFoundException(message="GPU instance persistent volume not found")


def ensure_writable(obj, ctx: TenantContext):
    if obj is None:
        raise NotFoundException(message="GPU instance persistent volume not found")
    assert_org_owned_writable(ctx, obj, resource_label="GPU instance persistent volume")
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
        # ``ON DELETE RESTRICT`` from gpu_instances.persistent_volume_id
        # surfaces here when a GPU instance still references this PV.
        raise ConflictException(
            message=message,
        ) from e
    except Exception as e:
        raise InternalServerErrorException(
            message=message,
        ) from e


async def _validate_create_obj(
    session: AsyncSession,
    create_obj: GPUInstancePersistentVolumeCreate,
) -> int:
    validate_k8s_object_name(create_obj.name)

    pvt = await GPUInstancePersistentVolumeType.first_by_fields(
        session=session,
        fields={
            "owner_principal_id": create_obj.owner_principal_id,
            "name": create_obj.spec.type_,
        },
    )
    if pvt is None:
        raise InvalidException(
            message=(
                f"GPU instance persistent volume type "
                f"'{create_obj.spec.type_}' not found"
            ),
        )

    return pvt.id


def _build_create_source(
    create_obj: GPUInstancePersistentVolumeCreate, persistent_volume_type_id: int
) -> dict:
    source: dict = create_obj.model_dump()
    source["persistent_volume_type_id"] = persistent_volume_type_id
    return source
