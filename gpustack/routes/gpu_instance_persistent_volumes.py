from contextlib import asynccontextmanager
from typing import Optional

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.responses import StreamingResponse

from gpustack.api.exceptions import (
    AlreadyExistsException,
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
from sqlmodel import select
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
    if create_obj.owner_principal_id is None:
        create_obj.owner_principal_id = (
            ctx.current_principal_id or platform_principal_id()
        )
    validate_owner_principal(
        create_obj.owner_principal_id,
        ctx,
        resource_label="GPU instance persistent volume",
        allow_member=True,
    )

    persistent_volume_type_id = await _validate_create_obj(session, ctx, create_obj)

    existed = await GPUInstancePersistentVolume.exist_by_fields(
        session=session,
        fields={
            "owner_principal_id": create_obj.owner_principal_id,
            "name": create_obj.name,
        },
    )
    if existed:
        raise AlreadyExistsException(
            message=(f"Storage with name '{create_obj.name}' already exists."),
        )

    source = _build_create_source(create_obj, ctx.user.id, persistent_volume_type_id)
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
    assert_org_owned_writable(
        ctx,
        obj,
        resource_label="GPU instance persistent volume",
        allow_member=True,
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
    except Exception as e:
        raise InternalServerErrorException(
            message=message,
        ) from e


async def _validate_create_obj(
    session: AsyncSession,
    ctx: TenantContext,
    create_obj: GPUInstancePersistentVolumeCreate,
) -> int:
    validate_k8s_object_name(create_obj.name)

    pvt = await resolve_pv_type_for_ctx(
        session,
        ctx,
        owner_principal_id=create_obj.owner_principal_id,
        name=create_obj.spec.type_,
    )
    if pvt is None:
        raise InvalidException(
            message=(
                f"GPU instance persistent volume type "
                f"'{create_obj.spec.type_}' not found"
            ),
        )

    return pvt.id


async def resolve_pv_type_for_ctx(
    session: AsyncSession,
    ctx: TenantContext,
    *,
    owner_principal_id: int,
    name: str,
) -> Optional[GPUInstancePersistentVolumeType]:
    """Resolve a PV type by name from any Org the caller can use:

    - the PV's own owner principal (caller's own types), and
    - any Org that owns a cluster the caller has access to
      (covers Default-Org "Everyone" grants and cross-Org cluster
      grants — Org1 grants its cluster to Org2 → Org2's user can
      reference Org1's PV types by name when provisioning).

    Returns the first matching row, or ``None`` if no Org in the
    accessible set defines a type with that name. ``ctx`` may bypass
    tenant scoping (admin / SYSTEM); in that case lookup is unscoped.
    """
    # Always scope by ``owner_principal_id`` — even for admin in
    # "All" mode. Two Orgs can independently define a PV type with
    # the same name; without the scope the lookup picks an
    # arbitrary one and the resulting PV would silently bind to the
    # wrong Org's storage backend. Tenant scoping is otherwise
    # unbounded here (admin / SYSTEM bypass): they may resolve a
    # type from any Org, but only the one matching the PV being
    # created.
    if bypass_tenant_filter(ctx):
        stmt = select(GPUInstancePersistentVolumeType).where(
            GPUInstancePersistentVolumeType.name == name,
            GPUInstancePersistentVolumeType.owner_principal_id == owner_principal_id,
        )
        return (await session.exec(stmt)).first()

    owner_ids = set(ctx.accessible_cluster_owner_ids)
    owner_ids.add(owner_principal_id)
    if not owner_ids:
        return None
    stmt = select(GPUInstancePersistentVolumeType).where(
        GPUInstancePersistentVolumeType.name == name,
        GPUInstancePersistentVolumeType.owner_principal_id.in_(owner_ids),
    )
    return (await session.exec(stmt)).first()


def _build_create_source(
    create_obj: GPUInstancePersistentVolumeCreate,
    creator_id: int,
    persistent_volume_type_id: int,
) -> dict:
    source: dict = create_obj.model_dump()
    source["creator_id"] = creator_id
    source["persistent_volume_type_id"] = persistent_volume_type_id
    return source
