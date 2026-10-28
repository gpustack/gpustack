"""Tenant bindings are immutable across ORM write paths without route guards."""

from typing import ClassVar, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from sqlalchemy import inspect
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import Field, Relationship, SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.api.exceptions import InvalidException
from gpustack.mixins import BaseModelMixin
from gpustack.mixins.immutable import allow_immutable_field_update


class BindingParent(SQLModel, table=True):
    __tablename__ = "test_binding_parents"
    id: Optional[int] = Field(default=None, primary_key=True)


class BindingResource(SQLModel, BaseModelMixin, table=True):
    __tablename__ = "test_binding_resources"
    __immutable_fields__: ClassVar[Tuple[str, ...]] = ("parent_id",)
    id: Optional[int] = Field(default=None, primary_key=True)
    owner_principal_id: Optional[int] = None
    parent_id: Optional[int] = Field(
        default=None, foreign_key="test_binding_parents.id"
    )
    name: str = "original"
    parent: Optional[BindingParent] = Relationship()


class BindingUpdate(SQLModel):
    owner_principal_id: Optional[int] = None
    name: Optional[str] = None


@pytest_asyncio.fixture
async def session(monkeypatch):
    engine = create_async_engine("sqlite+aiosqlite://")
    async with engine.begin() as connection:
        await connection.run_sync(BindingParent.__table__.create)
        await connection.run_sync(BindingResource.__table__.create)
    monkeypatch.setattr(BindingResource, "_publish_event_after_commit", MagicMock())
    monkeypatch.setattr(BindingResource, "_invalidate_cached_all", AsyncMock())
    monkeypatch.setattr(BindingResource, "_refresh_related_objects", AsyncMock())
    async with AsyncSession(engine, expire_on_commit=False) as session:
        session.add_all([BindingParent(id=1), BindingParent(id=2)])
        session.add_all(
            [BindingResource(id=i, owner_principal_id=8, parent_id=1) for i in (1, 2)]
        )
        await session.commit()
        yield session
    await engine.dispose()


WRITE_PATHS = [
    "dict-update",
    "schema-update",
    "upsert",
    "save",
    "flush",
    "commit",
    "batch",
    "deferred-batch",
    "detached",
    "merge",
]


async def write_owner(session, resource, owner, path):
    if path == "dict-update":
        await resource.update(session, {"name": "updated", "owner_principal_id": owner})
    elif path == "schema-update":
        await resource.update(session, BindingUpdate(owner_principal_id=owner))
    elif path == "upsert":
        await BindingResource.create_or_update(
            session, {"id": resource.id, "owner_principal_id": owner}
        )
    else:
        resource.owner_principal_id = owner
        if path == "save":
            await resource.save(session)
        elif path == "flush":
            await session.flush()
        elif path == "commit":
            await session.commit()
        elif path == "batch":
            await BindingResource.batch_update(session, [resource])
        elif path == "deferred-batch":
            await BindingResource.batch_update(session, [resource], auto_commit=False)
            await session.commit()
        elif path == "detached":
            session.expunge(resource)
            await resource.save(session)
        elif path == "merge":
            session.expunge(resource)
            await session.merge(resource)
            await session.commit()


@pytest.mark.asyncio
@pytest.mark.parametrize("path", WRITE_PATHS)
@pytest.mark.parametrize("owner", [None, 99])
async def test_owner_change_rejected_across_write_paths(session, path, owner):
    resource = await session.get(BindingResource, 1)
    with pytest.raises(InvalidException) as error:
        await write_owner(session, resource, owner, path)
    assert error.value.message == "owner_principal_id cannot be changed"
    await session.rollback()
    stored = await session.get(BindingResource, 1)
    assert stored.owner_principal_id == 8
    assert stored.name == "original"


@pytest.mark.asyncio
@pytest.mark.parametrize("path", WRITE_PATHS)
async def test_owner_echo_allowed_across_write_paths(session, path):
    resource = await session.get(BindingResource, 1)
    await write_owner(session, resource, 8, path)
    assert (await session.get(BindingResource, 1)).owner_principal_id == 8


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["dict-update", "save", "batch"])
@pytest.mark.parametrize("owner", [8, None, 99])
async def test_unloaded_owner_compared_with_storage(session, path, owner):
    resource = await session.get(BindingResource, 1)
    session.expire(resource, ["owner_principal_id"])
    if owner == 8:
        await write_owner(session, resource, owner, path)
    else:
        with pytest.raises(InvalidException):
            await write_owner(session, resource, owner, path)
        await session.rollback()
    assert (await session.get(BindingResource, 1)).owner_principal_id == 8


@pytest.mark.asyncio
async def test_omitted_owner_allows_normal_update(session):
    resource = await session.get(BindingResource, 1)
    await resource.update(session, BindingUpdate(name="updated"))
    assert resource.name == "updated"
    assert resource.owner_principal_id == 8


@pytest.mark.asyncio
async def test_invalid_patch_does_not_mutate_any_field(session):
    resource = await session.get(BindingResource, 1)
    with pytest.raises(InvalidException):
        await resource.update(session, {"name": "updated", "owner_principal_id": 99})
    assert resource.name == "original"
    assert resource.owner_principal_id == 8
    assert not inspect(resource).modified


@pytest.mark.asyncio
@pytest.mark.parametrize("owner", [None, 99])
async def test_new_rows_can_set_owner_before_first_flush(session, owner):
    resource = BindingResource(id=3, owner_principal_id=8)
    session.add(resource)
    resource.owner_principal_id = owner
    await session.commit()
    assert (await session.get(BindingResource, 3)).owner_principal_id == owner


@pytest.mark.asyncio
@pytest.mark.parametrize("relationship", [False, True])
async def test_model_declared_binding_is_immutable(session, relationship):
    resource = await session.get(BindingResource, 1)
    if relationship:
        resource.parent = await session.get(BindingParent, 2)
    else:
        resource.parent_id = 2
    with pytest.raises(InvalidException) as error:
        await session.commit()
    assert error.value.message == "parent_id cannot be changed"
    await session.rollback()
    assert (await session.get(BindingResource, 1)).parent_id == 1


@pytest.mark.asyncio
async def test_partial_flush_checks_its_target(session):
    resource = await session.get(BindingResource, 1)
    resource.owner_principal_id = 99
    with pytest.raises(InvalidException):
        await session.flush([resource])
    await session.rollback()
    assert (await session.get(BindingResource, 1)).owner_principal_id == 8


@pytest.mark.asyncio
async def test_batch_failure_does_not_commit_other_changes(session):
    rows = (
        await session.exec(select(BindingResource).order_by(BindingResource.id))
    ).all()
    rows[0].name = "updated"
    rows[1].owner_principal_id = 99
    with pytest.raises(InvalidException):
        await BindingResource.batch_update(session, rows)
    stored = await session.get(BindingResource, 1)
    assert stored.name == "original"


@pytest.mark.asyncio
async def test_allowance_permits_only_its_value_and_expires(session):
    resource = await session.get(BindingResource, 1)
    with allow_immutable_field_update(resource, "owner_principal_id", 99):
        with pytest.raises(InvalidException):
            await resource.update(session, {"owner_principal_id": None})
        await resource.update(session, {"owner_principal_id": 99}, auto_commit=False)
    await session.commit()
    assert resource.owner_principal_id == 99
    with pytest.raises(InvalidException):
        await resource.update(session, {"owner_principal_id": 8})


@pytest.mark.asyncio
@pytest.mark.parametrize("other_target", ["object", "field", "after-scope"])
async def test_allowance_does_not_authorize_other_writes(session, other_target):
    resource = await session.get(BindingResource, 1)
    other = await session.get(BindingResource, 2)
    with allow_immutable_field_update(resource, "owner_principal_id", 99):
        if other_target == "object":
            other.owner_principal_id = 99
        elif other_target == "field":
            resource.parent_id = 2
        else:
            resource.owner_principal_id = 99
        if other_target != "after-scope":
            with pytest.raises(InvalidException):
                await session.flush()
            await session.rollback()
    if other_target == "after-scope":
        with pytest.raises(InvalidException):
            await session.flush()
        await session.rollback()
    assert (await session.get(BindingResource, 1)).owner_principal_id == 8
    assert (await session.get(BindingResource, 2)).owner_principal_id == 8


@pytest.mark.asyncio
async def test_allowance_is_cleared_on_exception(session):
    resource = await session.get(BindingResource, 1)
    with pytest.raises(RuntimeError):
        with allow_immutable_field_update(resource, "owner_principal_id", 99):
            raise RuntimeError("reconciliation failed")
    assert "immutable_field_allowance" not in inspect(resource).info
    with pytest.raises(InvalidException):
        await resource.update(session, {"owner_principal_id": 99})
