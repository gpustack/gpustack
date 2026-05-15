from contextlib import asynccontextmanager

from fastapi import APIRouter

from gpustack import gpu_instances
from gpustack.api.exceptions import InternalServerErrorException, ForbiddenException
from gpustack.api.tenant import bypass_tenant_filter

from gpustack.schemas import (
    GPUInstanceSSHPublicKey,
    GPUInstanceSSHPublicKeyUpdate,
    GPUInstanceSSHPublicKeyPublic,
)
from gpustack.schemas.principals import platform_principal_id
from gpustack.server.deps import SessionDep, TenantContextDep

router = APIRouter()


@router.get("/data", response_model=GPUInstanceSSHPublicKeyPublic)
async def get_gpu_instance_ssh_public_key_data(
    session: SessionDep,
    ctx: TenantContextDep,
):
    if not bypass_tenant_filter(ctx) and not ctx.has_org_context:
        raise ForbiddenException(
            message="Organization context is required to get GPU instance SSH public key",
        )

    owner_principal_id = ctx.current_principal_id or platform_principal_id()
    name = gpu_instances.SSH_PUBLIC_KEY_NAME

    ret = await GPUInstanceSSHPublicKey.one_by_fields(
        session=session,
        fields={
            "owner_principal_id": owner_principal_id,
            "name": name,
        },
    )
    if ret is None:
        # Return a dummy record with empty data if not found.
        ret = GPUInstanceSSHPublicKeyPublic(
            name=name,
        )

    return ret


@router.put("/data", response_model=GPUInstanceSSHPublicKeyPublic)
async def update_gpu_instance_ssh_public_key_data(
    session: SessionDep,
    ctx: TenantContextDep,
    update_obj: GPUInstanceSSHPublicKeyUpdate,
):
    if not bypass_tenant_filter(ctx) and not ctx.has_org_context:
        raise ForbiddenException(
            message="Organization context is required to update GPU instance SSH public key",
        )

    owner_principal_id = ctx.current_principal_id or platform_principal_id()
    name = gpu_instances.SSH_PUBLIC_KEY_NAME

    if update_obj.name != name:
        raise ForbiddenException(
            message="Invalid name for GPU instance SSH public key",
        )
    update_obj.owner_principal_id = owner_principal_id

    ret = await GPUInstanceSSHPublicKey.one_by_fields(
        session=session,
        fields={
            "owner_principal_id": owner_principal_id,
            "name": name,
        },
    )

    async with handle_error(
        message="Failed to update GPU instance SSH public key",
    ):
        if ret is None:
            return await GPUInstanceSSHPublicKey.create(
                session=session,
                source=update_obj,
            )

        await ret.update(
            session=session,
            source=update_obj,
        )
        return ret


@asynccontextmanager
async def handle_error(message: str):
    try:
        yield
    except Exception as e:
        raise InternalServerErrorException(
            message=message,
        ) from e
