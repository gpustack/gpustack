from contextlib import asynccontextmanager
from typing import Optional

from fastapi import APIRouter, Depends
from sqlalchemy.exc import IntegrityError
from starlette.responses import StreamingResponse

from gpustack.api.exceptions import (
    AlreadyExistsException,
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
    GPUInstancePersistentVolumeTypeStatus,
    GPUInstancePersistentVolumeTypePhase,
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
    mine: bool = False,
):
    """List PV types.

    Default visibility unions the caller's own types with types owned
    by every Org whose cluster the caller can use — covers Default-Org
    "Everyone" grants and cross-Org cluster grants, so PV-create
    pickers find any type referenceable from a cluster they target.

    ``mine=true`` restricts to types owned by the caller's current
    principal — drops cross-Org rows so the Storage Type management
    page doesn't surface read-only types from other Orgs that
    happened to grant the caller a cluster.
    """
    fuzzy_fields: dict = {}
    if search:
        fuzzy_fields["name"] = search

    extra_conditions: list = []
    filter_func = lambda row: is_visible(row, ctx)  # noqa: E731
    if not bypass_tenant_filter(ctx):
        if mine:
            if ctx.current_principal_id is None:
                extra_conditions = [GPUInstancePersistentVolumeType.id == -1]
            else:
                extra_conditions = [
                    GPUInstancePersistentVolumeType.owner_principal_id
                    == ctx.current_principal_id
                ]
            filter_func = lambda row: is_manageable(row, ctx)  # noqa: E731
        else:
            owner_ids = set(ctx.accessible_cluster_owner_ids)
            if ctx.current_principal_id is not None:
                owner_ids.add(ctx.current_principal_id)
            if not owner_ids:
                # No avenue; force empty result rather than leak.
                extra_conditions = [GPUInstancePersistentVolumeType.id == -1]
            else:
                extra_conditions = [
                    GPUInstancePersistentVolumeType.owner_principal_id.in_(owner_ids)
                ]

    if params.watch:
        return StreamingResponse(
            GPUInstancePersistentVolumeType.streaming(
                fields={},
                fuzzy_fields=fuzzy_fields,
                filter_func=filter_func,
            ),
            media_type="text/event-stream",
        )

    async with async_session() as session:
        return await GPUInstancePersistentVolumeType.paginated_by_query(
            session=session,
            fields={},
            fuzzy_fields=fuzzy_fields,
            order_by=params.order_by,
            page=params.page,
            per_page=params.perPage,
            extra_conditions=extra_conditions,
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
    if create_obj.owner_principal_id is None:
        create_obj.owner_principal_id = (
            ctx.current_principal_id or platform_principal_id()
        )
    validate_owner_principal(
        create_obj.owner_principal_id,
        ctx,
        resource_label="GPU instance persistent volume type",
    )

    _validate_create_obj(create_obj)

    existed = await GPUInstancePersistentVolumeType.exist_by_fields(
        session=session,
        fields={
            "owner_principal_id": create_obj.owner_principal_id,
            "name": create_obj.name,
        },
    )
    if existed:
        raise AlreadyExistsException(
            message=(f"Storage type with name '{create_obj.name}' already exists."),
        )

    source: dict = create_obj.model_dump()
    source["creator_id"] = ctx.user.id
    source["status"] = GPUInstancePersistentVolumeTypeStatus(
        phase=GPUInstancePersistentVolumeTypePhase.READY,
    )
    async with handle_error(
        message="Failed to create GPU instance persistent volume type",
    ):
        return await GPUInstancePersistentVolumeType.create(
            session=session,
            source=source,
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


@router.delete(
    "/{id}",
    status_code=202,
    response_model=GPUInstancePersistentVolumeTypePublic,
)
async def delete_gpu_instance_persistent_volume_type(
    session: SessionDep,
    ctx: TenantContextDep,
    id: int,
):
    """Mark the persistent volume type for deletion (soft delete).

    Stamps ``status.phase = Deleting`` and retains the row; the PVT
    finalizer controller waits until no persistent volume references it and
    the downstream types are cleared, then hard-deletes the row.
    """
    ret = ensure_writable(
        await GPUInstancePersistentVolumeType.one_by_id(
            session=session,
            id=id,
        ),
        ctx,
    )

    source = _build_delete_phase_source(ret)
    async with handle_error(
        message="Failed to delete GPU instance persistent volume type",
    ):
        await ret.update(
            session=session,
            source=source,
        )
        return ret


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
    # Caller's own Org always sees its PV types.
    if (
        ctx.current_principal_id is not None
        and obj.owner_principal_id == ctx.current_principal_id
    ):
        return True
    # PV types owned by an Org whose cluster the caller can use —
    # covers Default-Org "Everyone" grants and cross-Org cluster
    # grants (Org1 grants its cluster to Org2 → Org2 can pick Org1's
    # PV types when provisioning storage on that cluster).
    return obj.owner_principal_id in ctx.accessible_cluster_owner_ids


def is_manageable(obj, ctx: TenantContext) -> bool:
    """Manageability mirror of :func:`is_visible`: drops cross-Org rows
    that came in via cluster-access. Bypass for admin in "All" mode.
    """
    if bypass_tenant_filter(ctx):
        return True
    return (
        ctx.current_principal_id is not None
        and obj.owner_principal_id == ctx.current_principal_id
    )


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


def _build_delete_phase_source(existing: GPUInstancePersistentVolumeType) -> dict:
    """Stamp the soft-delete phase onto status, clearing any prior
    ``phase_message`` while preserving ``finalizing`` so a re-issued delete
    stays idempotent. The finalizer controller fills ``finalizing`` and
    hard-deletes the row once downstream cleanup completes."""
    base = existing.status or GPUInstancePersistentVolumeTypeStatus()
    return {
        "status": base.model_copy(
            update={
                "phase": GPUInstancePersistentVolumeTypePhase.DELETING,
                "phase_message": None,
            },
        ),
    }
