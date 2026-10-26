"""Deleting a GPU instance SSH public key is refused while an instance uses it."""

from types import SimpleNamespace

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.api.exceptions import ConflictException
from gpustack.routes import gpu_instance_ssh_public_keys as key_routes
from gpustack.schemas.gpu_instance_ssh_public_keys import (
    GPUInstanceSSHPublicKey,
    GPUInstanceSSHPublicKeySpec,
)
from gpustack.schemas.gpu_instances import GPUInstance
from gpustack.schemas.principals import PrincipalType

CTX = SimpleNamespace(
    user=SimpleNamespace(kind=PrincipalType.SYSTEM, id=1),
    is_platform_admin=True,
    current_principal_id=None,
)


@pytest_asyncio.fixture
async def session():
    engine = create_async_engine("sqlite+aiosqlite://")
    async with engine.begin() as conn:
        await conn.run_sync(GPUInstanceSSHPublicKey.__table__.create)
        await conn.run_sync(GPUInstance.__table__.create)
    async with AsyncSession(engine, expire_on_commit=False) as s:
        yield s
    await engine.dispose()


async def _add_key(session, id_=1, name="key1"):
    session.add(
        GPUInstanceSSHPublicKey(
            id=id_,
            name=name,
            owner_principal_id=1,
            spec=GPUInstanceSSHPublicKeySpec(data="ssh-ed25519 AAAA"),
        )
    )
    await session.commit()


async def _add_instance(session, *, id_, name, keys, phase=None):
    session.add(
        GPUInstance(
            id=id_,
            name=name,
            owner_principal_id=1,
            cluster_id=10,
            spec={
                "type_": "gpu",
                "image": "busybox",
                "ssh_public_keys": [{"name": k} for k in keys],
            },
            status={"phase": phase} if phase is not None else None,
        )
    )
    await session.commit()


@pytest.mark.asyncio
async def test_delete_referenced_key_rejected_with_409(session):
    await _add_key(session)
    await _add_instance(session, id_=1, name="gi-1", keys=["key1"], phase="Stopped")

    with pytest.raises(ConflictException) as exc_info:
        await key_routes.delete_gpu_instance_ssh_public_key(session, CTX, 1)

    assert exc_info.value.status_code == 409
    assert "gi-1" in exc_info.value.message
    assert await GPUInstanceSSHPublicKey.one_by_id(session=session, id=1) is not None


@pytest.mark.asyncio
async def test_delete_unreferenced_key_proceeds(session):
    await _add_key(session)
    await _add_instance(session, id_=1, name="gi-1", keys=["other"])

    await key_routes.delete_gpu_instance_ssh_public_key(session, CTX, 1)

    assert await GPUInstanceSSHPublicKey.one_by_id(session=session, id=1) is None


@pytest.mark.asyncio
async def test_delete_proceeds_when_only_holder_is_deleting(session):
    await _add_key(session)
    await _add_instance(session, id_=1, name="gi-1", keys=["key1"], phase="Deleting")

    await key_routes.delete_gpu_instance_ssh_public_key(session, CTX, 1)

    assert await GPUInstanceSSHPublicKey.one_by_id(session=session, id=1) is None
