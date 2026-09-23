"""``categories`` / ``meta`` reaching the route the model created.

Neither field is known when the route is born — `POST /models` copies what the
caller sent, and the UI sends neither — so the scheduler fills them onto the
Model a second later and this owner carries them across. `/v1/models?
categories=llm` filters on ``ModelRoute.categories``, so a route left holding
its birth value is a deployment that silently stops being listed as an LLM.

The regression pinned hardest here is the one that made this owner a no-op in
production: by the time `_reconcile` calls it, the Model row has already been
loaded into the session twice, so a third fetch asking for
``selectinload(model_routes)`` hits the identity map, gets that same instance
back and never applies the loader option. ``model_routes`` is ``lazy="noload"``
and an unloaded collection reads as ``[]``, so the sync found nothing to do,
every pass, for every model, without raising or logging — no exception, no log
line, and a Model row that looked perfectly correct next to a route row that
did not.

These tests run against a real session rather than a mock one, because a mock
cannot be wrong in this particular way. That is not quite enough on its own:
SQLite does not reproduce the identity-map behaviour that PostgreSQL shows, so
the first test plants the empty collection directly rather than staging the
conditions that produce it. The rest exercise the real call sequence.
"""

from datetime import datetime, timezone

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.orm import selectinload
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.schemas.model_routes import ModelRoute, ModelRouteTarget
from gpustack.schemas.models import Model, SourceEnum
from gpustack.schemas.principals import Principal, PrincipalType
from gpustack.server.bus import Event, EventType
from gpustack.server.controllers import (
    notify_model_route_target,
    sync_categories_and_meta,
)


@pytest_asyncio.fixture
async def db():
    """An engine plus a `seed` that writes through a session of its own.

    Seeding through the session under test would hide the bug outright:
    ``session.add(route)`` puts the route in that session's identity map and
    SQLAlchemy wires the in-memory relationship up for free, so
    ``model.model_routes`` answers from memory no matter what the loader
    option did. `_reconcile` never has that luxury — it opens a fresh session
    per event and every row it touches comes off disk.
    """

    engine = create_async_engine("sqlite+aiosqlite://")
    async with engine.begin() as conn:
        await conn.run_sync(Principal.__table__.create)
        await conn.run_sync(Model.__table__.create)
        await conn.run_sync(ModelRoute.__table__.create)
        await conn.run_sync(ModelRouteTarget.__table__.create)

    async def seed(*rows):
        async with AsyncSession(engine) as writer:
            writer.add(Principal(id=1, name="default", kind=PrincipalType.ORG))
            for row in rows:
                writer.add(row)
            await writer.commit()

    # ``expire_on_commit=False`` matches `gpustack.server.db.async_session`,
    # and it matters: it keeps an already-loaded instance unexpired, so a
    # later `session.get` is answered from the identity map rather than
    # re-running the query.
    async with AsyncSession(engine, expire_on_commit=False) as session:
        session.seed = seed
        yield session
    await engine.dispose()


def _model(id=1, name="m", categories=None, meta=None) -> Model:
    """A model as the scheduler leaves it: categories and meta derived."""
    return Model(
        id=id,
        name=name,
        replicas=1,
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="org/repo",
        owner_principal_id=1,
        categories=["llm"] if categories is None else categories,
        meta={"max_model_len": 8192} if meta is None else meta,
    )


def _route(id=1, name="m", created_model_id=1, **kwargs) -> ModelRoute:
    """A route as `POST /models` leaves it: born with whatever the caller
    sent, which for a UI-created deployment is nothing."""
    return ModelRoute(
        id=id,
        name=name,
        created_model_id=created_model_id,
        owner_principal_id=1,
        categories=kwargs.pop("categories", []),
        meta=kwargs.pop("meta", {}),
        **kwargs,
    )


def _target(id=1, route_id=1, model_id=1) -> ModelRouteTarget:
    return ModelRouteTarget(
        id=id,
        name=f"m-deployment-{id}",
        route_name="m",
        route_id=route_id,
        model_id=model_id,
        weight=100,
    )


async def _as_reconcile_does(session, model_id) -> Model:
    """Leave the session in the state `_reconcile` hands the sync.

    Two prior loads of the same row: a plain one for `sync_model_status`, then
    one carrying a *different* loader option inside
    `notify_model_route_target`. On PostgreSQL that pair is what leaves a
    later ``selectinload(model_routes)`` unapplied.
    """
    await Model.one_by_id(session, model_id)
    return await Model.one_by_id(
        session, model_id, options=[selectinload(Model.model_route_targets)]
    )


def _updated(model: Model) -> Event:
    return Event(type=EventType.UPDATED, data=model)


@pytest.mark.asyncio
async def test_an_unloaded_relationship_does_not_stop_the_sync(db):
    """The regression that made this owner a no-op for every model in a live
    deployment, stated as the constraint it leaves behind: the sync must not
    read ``model.model_routes``.

    In production the relationship comes back empty here. `one_by_id` is
    `session.get`, `_reconcile` has already loaded this Model row twice by the
    time the sync runs, and a `get` that hits the identity map can answer
    without running the query — dropping the loader option. ``model_routes``
    is ``lazy="noload"``, and an unloaded collection reads as ``[]`` rather
    than raising, so the loop found nothing to do, every pass, for every
    model, silently. Verified against the live PostgreSQL database; SQLite
    does not reproduce the identity-map behaviour, which is exactly why a
    faithful-looking unit test could not have caught it.

    So this forces the empty collection directly instead of trying to
    reproduce the conditions that produce it. The assertion holds on any
    backend, and only holds for an implementation that queries ``ModelRoute``
    rather than walking the relationship.
    """
    await db.seed(_model(), _route(), _target())

    model = await Model.one_by_id(db, 1)
    # Bypass the ORM to plant precisely what production hands the sync.
    model.__dict__["model_routes"] = []

    await sync_categories_and_meta(db, model, _updated(model))

    assert (await ModelRoute.one_by_id(db, 1)).categories == ["llm"]


@pytest.mark.asyncio
async def test_categories_and_meta_reach_the_route(db):
    await db.seed(_model(), _route(), _target())

    model = await _as_reconcile_does(db, 1)
    await sync_categories_and_meta(db, model, _updated(model))

    route = await ModelRoute.one_by_id(db, 1)
    assert route.categories == ["llm"]
    assert route.meta == {"max_model_len": 8192}


@pytest.mark.asyncio
async def test_it_survives_the_call_that_precedes_it(db):
    """`notify_model_route_target` runs one line earlier and is what leaves the
    poisoned instance behind, so run the real pair in order rather than trusting
    a hand-built approximation of the session state."""
    await db.seed(_model(), _route(), _target())

    model = await Model.one_by_id(db, 1)
    event = _updated(model)
    await notify_model_route_target(session=db, model=model, event=event)
    await sync_categories_and_meta(db, model, event)

    assert (await ModelRoute.one_by_id(db, 1)).categories == ["llm"]


@pytest.mark.asyncio
async def test_an_up_to_date_route_is_left_alone(db):
    """The change gate: a pass over a settled world must not write, or every
    reconcile republishes a route event and every watcher churns."""
    await db.seed(
        _model(),
        _route(categories=["llm"], meta={"max_model_len": 8192}),
        _target(),
    )
    before = (await ModelRoute.one_by_id(db, 1)).updated_at

    model = await _as_reconcile_does(db, 1)
    await sync_categories_and_meta(db, model, _updated(model))

    assert (await ModelRoute.one_by_id(db, 1)).updated_at == before


@pytest.mark.asyncio
async def test_a_deletion_syncs_nothing(db):
    """The model row may already be gone; writing its categories anywhere would
    be writing a tombstone's opinion."""
    await db.seed(_model(), _route(), _target())

    model = await _as_reconcile_does(db, 1)
    await sync_categories_and_meta(db, model, Event(type=EventType.DELETED, data=model))

    assert (await ModelRoute.one_by_id(db, 1)).categories == []


@pytest.mark.asyncio
async def test_only_the_route_this_model_created(db):
    """A hand-created route (``created_model_id`` NULL) and a route another
    model created are both off limits, even when this model is one of their
    targets. Walking the relationship could not draw that line: it goes through
    ``ModelRouteTarget``, so a multi-target route reached from model 1 would
    have had model 1's categories written onto it regardless of who created
    it."""
    await db.seed(
        _model(),
        _model(id=2, name="other", categories=["embedding"]),
        _route(),
        _route(id=2, name="hand-made", created_model_id=None),
        _route(id=3, name="other", created_model_id=2),
        # Model 1 is a target of all three.
        _target(id=1, route_id=1, model_id=1),
        _target(id=2, route_id=2, model_id=1),
        _target(id=3, route_id=3, model_id=1),
    )

    model = await _as_reconcile_does(db, 1)
    await sync_categories_and_meta(db, model, _updated(model))

    assert (await ModelRoute.one_by_id(db, 1)).categories == ["llm"]
    assert (await ModelRoute.one_by_id(db, 2)).categories == []
    assert (await ModelRoute.one_by_id(db, 3)).categories == []


@pytest.mark.asyncio
async def test_a_soft_deleted_route_is_not_resurrected(db):
    deleted_at = datetime.now(timezone.utc).replace(tzinfo=None)
    await db.seed(_model(), _route(deleted_at=deleted_at), _target())

    model = await _as_reconcile_does(db, 1)
    await sync_categories_and_meta(db, model, _updated(model))

    assert (await ModelRoute.one_by_id(db, 1)).categories == []
