"""A missing SSH public key must not block the instance's remaining keys."""

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.gpu_instances.controllers import GPUInstanceController
from gpustack.schemas.gpu_instance_ssh_public_keys import (
    GPUInstanceSSHPublicKey,
    GPUInstanceSSHPublicKeySpec,
)
from gpustack.schemas.gpu_instances import GPUInstance


@pytest_asyncio.fixture
async def session():
    engine = create_async_engine("sqlite+aiosqlite://")
    async with engine.begin() as conn:
        await conn.run_sync(GPUInstanceSSHPublicKey.__table__.create)
    async with AsyncSession(engine, expire_on_commit=False) as s:
        yield s
    await engine.dispose()


def _instance(keys):
    return GPUInstance.model_construct(
        name="gi-1",
        owner_principal_id=1,
        spec=GPUInstance.model_fields["spec"].annotation.model_validate(
            {
                "type_": "gpu",
                "image": "busybox",
                "ssh_public_keys": [{"name": k} for k in keys],
            }
        ),
    )


@pytest.mark.parametrize(
    "refs, expected",
    [
        (["key1", "key2"], "ssh-ed25519 KEY2"),
        (["key1"], ""),
        ([], ""),
    ],
)
@pytest.mark.asyncio
async def test_missing_key_is_skipped(session, refs, expected):
    session.add(
        GPUInstanceSSHPublicKey(
            id=2,
            name="key2",
            owner_principal_id=1,
            spec=GPUInstanceSSHPublicKeySpec(data="ssh-ed25519 KEY2"),
        )
    )
    await session.commit()

    data = await GPUInstanceController._aggregate_ssh_public_key_data(
        session, _instance(refs)
    )

    assert data == expected
