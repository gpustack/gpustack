from contextlib import asynccontextmanager
from typing import Optional

from fastapi import APIRouter, Depends
from starlette.responses import StreamingResponse

from gpustack.api.exceptions import (
    InternalServerErrorException,
    NotFoundException,
)
from gpustack.api.tenant import (
    bypass_tenant_filter,
    TenantContext,
    assert_org_owned_writable,
    validate_owner_principal,
)

from gpustack.schemas import (
    GPUInstanceSSHPublicKey,
    GPUInstanceSSHPublicKeyUpdate,
    GPUInstanceSSHPublicKeyPublic,
    GPUInstanceSSHPublicKeyListParams,
    GPUInstanceSSHPublicKeysPublic,
    GPUInstanceSSHPublicKeyCreate,
)
from gpustack.schemas.principals import platform_principal_id
from gpustack.server.db import async_session
from gpustack.server.deps import SessionDep, TenantContextDep

router = APIRouter()


@router.get("", response_model=GPUInstanceSSHPublicKeysPublic)
async def get_gpu_instance_ssh_public_keys(
    ctx: TenantContextDep,
    params: GPUInstanceSSHPublicKeyListParams = Depends(),
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
            GPUInstanceSSHPublicKey.streaming(
                fields=fields,
                fuzzy_fields=fuzzy_fields,
            ),
            media_type="text/event-stream",
        )

    async with async_session() as session:
        return await GPUInstanceSSHPublicKey.paginated_by_query(
            session=session,
            fields=fields,
            fuzzy_fields=fuzzy_fields,
            order_by=params.order_by,
            page=params.page,
            per_page=params.perPage,
        )


@router.get("/{id}", response_model=GPUInstanceSSHPublicKeyPublic)
async def get_gpu_instance_ssh_public_key(
    session: SessionDep,
    ctx: TenantContextDep,
    id: int,
):
    return ensure_visible(
        await GPUInstanceSSHPublicKey.one_by_id(
            session=session,
            id=id,
        ),
        ctx,
    )


@router.post("", response_model=GPUInstanceSSHPublicKeyPublic)
async def create_gpu_instance_ssh_public_key(
    session: SessionDep,
    ctx: TenantContextDep,
    create_obj: GPUInstanceSSHPublicKeyCreate,
):
    validate_owner_principal(
        create_obj.owner_principal_id, ctx, resource_label="GPU instance SSH public key"
    )
    create_obj.owner_principal_id = ctx.current_principal_id or platform_principal_id()

    async with handle_error(
        message="Failed to create GPU instance SSH public key",
    ):
        return await GPUInstanceSSHPublicKey.create(
            session=session,
            source=create_obj,
        )


@router.put("/{id}", response_model=GPUInstanceSSHPublicKeyPublic)
async def update_gpu_instance_ssh_public_key(
    session: SessionDep,
    ctx: TenantContextDep,
    id: int,
    update_obj: GPUInstanceSSHPublicKeyUpdate,
):
    ret = ensure_writable(
        await GPUInstanceSSHPublicKey.one_by_id(
            session=session,
            id=id,
        ),
        ctx,
    )

    async with handle_error(
        message="Failed to update GPU instance SSH public key",
    ):
        return await ret.update(
            session=session,
            source=update_obj,
        )


@router.delete("/{id}")
async def delete_gpu_instance_ssh_public_key(
    session: SessionDep,
    ctx: TenantContextDep,
    id: int,
):
    ret = ensure_writable(
        await GPUInstanceSSHPublicKey.one_by_id(
            session=session,
            id=id,
        ),
        ctx,
    )

    async with handle_error(
        message="Failed to delete GPU instance SSH public key",
    ):
        await ret.delete(
            session=session,
        )


def ensure_visible(obj, ctx: TenantContext):
    if obj and is_visible(obj, ctx):
        return obj
    raise NotFoundException(message="GPU instance SSH public key not found")


def ensure_writable(obj, ctx: TenantContext):
    if obj is None:
        raise NotFoundException(message="GPU instance SSH public key not found")
    assert_org_owned_writable(ctx, obj, resource_label="GPU instance SSH public key")
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
