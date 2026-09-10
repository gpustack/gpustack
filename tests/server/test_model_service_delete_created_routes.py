"""Deleting a model tears down the route it created in the same transaction.

The controller removed a model-created route only after the DELETED event
of the cascade-deleted target arrived. Creating a model with the same name
inside that window hit the unique-name check in ``create_model`` and got
``409 Model route with name '...' already exists.`` although the model was
already gone (#6197).
"""

from sqlalchemy import event
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession
import pytest
import pytest_asyncio

import gpustack.schemas  # noqa: F401  (registers every table)
from gpustack.schemas.model_routes import (
    ModelRoute,
    ModelRouteTarget,
    TargetStateEnum,
)
from gpustack.schemas.models import Model, SourceEnum
from gpustack.schemas.principals import (
    Principal,
    PrincipalType,
    platform_principal_id,
)
from gpustack.server.services import ModelService

ORG_ID = platform_principal_id()


@pytest_asyncio.fixture
async def session():
    engine = create_async_engine("sqlite+aiosqlite://")

    @event.listens_for(engine.sync_engine, "connect")
    def _enforce_foreign_keys(dbapi_connection, _record):
        # The production databases cascade target deletion through foreign
        # keys; SQLite only does so with the pragma switched on.
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    async with AsyncSession(engine, expire_on_commit=False) as s:
        s.add(Principal(id=ORG_ID, name="default", kind=PrincipalType.ORG))
        await s.commit()
        yield s
    await engine.dispose()


async def _model_with_created_route(session: AsyncSession, name: str) -> Model:
    model = await Model.create(
        session,
        source=Model(
            name=name,
            source=SourceEnum.HUGGING_FACE,
            huggingface_repo_id="org/repo",
            owner_principal_id=ORG_ID,
            enable_model_route=True,
        ),
    )
    route = await ModelRoute.create(
        session,
        source=ModelRoute(
            name=name, created_model_id=model.id, owner_principal_id=ORG_ID
        ),
        auto_commit=False,
    )
    await ModelRouteTarget.create(
        session,
        source=ModelRouteTarget(
            name=f"{name}-deployment",
            route_name=route.name,
            model_route=route,
            model=model,
            weight=100,
            state=TargetStateEnum.UNAVAILABLE,
        ),
    )
    return model


async def _route(session: AsyncSession, name: str):
    return await ModelRoute.one_by_fields(
        session, {"name": name, "owner_principal_id": ORG_ID}
    )


@pytest.mark.asyncio
async def test_delete_model_removes_the_route_it_created(session):
    model = await _model_with_created_route(session, "m1")
    assert await _route(session, "m1") is not None

    await ModelService(session).delete(model)

    # This is the lookup create_model uses for its 409 check.
    assert await _route(session, "m1") is None
    assert await ModelRouteTarget.all_by_fields(session, {"route_name": "m1"}) == []


@pytest.mark.asyncio
async def test_delete_model_keeps_a_created_route_other_targets_still_use(session):
    model = await _model_with_created_route(session, "shared")
    other = await Model.create(
        session,
        source=Model(
            name="other",
            source=SourceEnum.HUGGING_FACE,
            huggingface_repo_id="org/other",
            owner_principal_id=ORG_ID,
        ),
    )
    route = await _route(session, "shared")
    await ModelRouteTarget.create(
        session,
        source=ModelRouteTarget(
            name="other-deployment",
            route_name=route.name,
            model_route=route,
            model=other,
            weight=100,
            state=TargetStateEnum.UNAVAILABLE,
        ),
    )

    await ModelService(session).delete(model)

    kept = await _route(session, "shared")
    assert kept is not None and kept.created_model_id == model.id
    remaining = await ModelRouteTarget.all_by_fields(session, {"route_name": "shared"})
    assert [target.model_id for target in remaining] == [other.id]
