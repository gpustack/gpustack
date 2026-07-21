"""GPUInstanceType snapshot identity + table mapping tests.

``GPUInstanceType.compute_snapshot`` hashes ``(cluster_id, name,
definitional-spec)`` with the mutable ``display_name`` excluded. The spec now
holds only definitional fields (observed hardware lives on ``status.detail``),
so two definitions differing only by display name collide, while a
definitional-field / name / cluster change diverges. The table maps cleanly
over sqlite and enforces ``snapshot`` (the row's global identity) uniqueness.
"""

import re

import pytest
import pytest_asyncio
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.schemas.gpu_instance_types import (
    GPUInstanceType,
    GPUInstanceTypeSpec,
)

_SHA1_SNAPSHOT = re.compile(r"^sha1:[0-9a-f]{40}$")


def _type(cluster_id=1, name="a10g", **spec_kwargs):
    # Table models skip validation on init; snapshot is left unset because
    # compute_snapshot only reads cluster_id + name + spec.
    return GPUInstanceType(
        cluster_id=cluster_id,
        name=name,
        spec=GPUInstanceTypeSpec(**spec_kwargs),
    )


def _snap(cluster_id, instance_type):
    # Stamp cluster_id (part of the identity) onto the instance, then hash.
    instance_type.cluster_id = cluster_id
    return instance_type.compute_snapshot()


# --- snapshot identity ----------------------------------------------------- #


def test_snapshot_format_is_sha1_hex():
    snapshot = _snap(1, _type(accelerator_group="nvidia-a10g"))
    assert _SHA1_SNAPSHOT.match(snapshot)


def test_snapshot_is_deterministic_across_calls():
    it = _type(accelerator_group="nvidia-a10g")
    assert _snap(1, it) == _snap(1, it)


def test_display_name_only_diff_collides():
    a = _type(display_name="A10G Pool", accelerator_group="nvidia-a10g")
    b = _type(display_name="Renamed Pool", accelerator_group="nvidia-a10g")
    assert _snap(1, a) == _snap(1, b)


def test_definitional_field_diff_diverges():
    # A create-time definitional field (here local_storage) is part of identity.
    a = _type(accelerator_group="nvidia-a10g", local_storage="100Gi")
    b = _type(accelerator_group="nvidia-a10g", local_storage="200Gi")
    assert _snap(1, a) != _snap(1, b)


def test_name_diff_diverges():
    a = _type(name="a10g", accelerator_group="nvidia-a10g")
    b = _type(name="a100", accelerator_group="nvidia-a10g")
    assert _snap(1, a) != _snap(1, b)


def test_cluster_id_diff_diverges():
    it = _type(accelerator_group="nvidia-a10g")
    assert _snap(1, it) != _snap(2, it)


# --- table mapping --------------------------------------------------------- #


@pytest_asyncio.fixture
async def engine():
    e = create_async_engine("sqlite+aiosqlite://")
    async with e.begin() as conn:
        await conn.run_sync(GPUInstanceType.__table__.create)
    yield e
    await e.dispose()


@pytest.mark.asyncio
async def test_table_round_trip(engine):
    async with AsyncSession(engine, expire_on_commit=False) as s:
        s.add(
            GPUInstanceType(
                cluster_id=1,
                name="a10g",
                spec=GPUInstanceTypeSpec(accelerator_group="nvidia-a10g"),
                snapshot="sha1:deadbeef",
            )
        )
        await s.commit()

    async with AsyncSession(engine, expire_on_commit=False) as s:
        row = await GPUInstanceType.one_by_fields(s, {"cluster_id": 1, "name": "a10g"})
    assert row.spec.accelerator_group == "nvidia-a10g"
    assert row.snapshot == "sha1:deadbeef"
    assert row.is_deleted() is False


@pytest.mark.asyncio
async def test_same_name_different_cluster_coexist(engine):
    # Distinct snapshots (cluster_id feeds the hash), so both rows persist.
    async with AsyncSession(engine, expire_on_commit=False) as s:
        s.add(
            GPUInstanceType(
                cluster_id=1, name="a10g", spec=GPUInstanceTypeSpec(), snapshot="sha1:x"
            )
        )
        s.add(
            GPUInstanceType(
                cluster_id=2, name="a10g", spec=GPUInstanceTypeSpec(), snapshot="sha1:y"
            )
        )
        await s.commit()

    async with AsyncSession(engine, expire_on_commit=False) as s:
        rows = await GPUInstanceType.all_by_field(s, "name", "a10g")
    assert {r.cluster_id for r in rows} == {1, 2}


@pytest.mark.asyncio
async def test_snapshot_uniqueness_enforced(engine):
    async with AsyncSession(engine, expire_on_commit=False) as s:
        s.add(
            GPUInstanceType(
                cluster_id=1, name="a10g", spec=GPUInstanceTypeSpec(), snapshot="sha1:x"
            )
        )
        await s.commit()

    async with AsyncSession(engine, expire_on_commit=False) as s:
        s.add(
            GPUInstanceType(
                cluster_id=2, name="a100", spec=GPUInstanceTypeSpec(), snapshot="sha1:x"
            )
        )
        with pytest.raises(IntegrityError):
            await s.commit()
