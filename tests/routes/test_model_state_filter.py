"""``GET /v2/models?state=`` and its watch-stream mirror.

The filter's name, values and meaning are frozen: ``ready`` is "at least one
member is ready", ``not_ready`` is "none is, but some were asked for",
``stopped`` is "none asked for". What changed is where the answer comes from —
``Model.state``, written by the one owner of the model's status, instead of a
second derivation from the replica counters.

Two things are pinned here: the SQL and the Python mirror admit exactly the
same rows (they are two implementations of one filter), and both keep
answering for rows whose ``state`` is still NULL, which is every row until the
owner first runs for it.
"""

from types import SimpleNamespace

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.routes.model_common import (
    ModelStateFilterEnum,
    model_state_condition,
    model_state_stream_filter,
)
from gpustack.schemas.models import Model, ModelStateEnum, SourceEnum

# (name, state, replicas, ready_replicas) — one row per interesting shape,
# including the pre-backfill rows that carry no state at all.
_ROWS = [
    ("running-full", ModelStateEnum.RUNNING, 2, 2),
    ("partial-scaling", ModelStateEnum.PARTIAL, 3, 1),
    ("pending", ModelStateEnum.PENDING, 2, 0),
    ("errored", ModelStateEnum.ERROR, 1, 0),
    ("stopped", ModelStateEnum.PENDING, 0, 0),
    ("legacy-ready", None, 2, 2),
    ("legacy-not-ready", None, 2, 0),
    ("legacy-stopped", None, 0, 0),
]


def _row(name, state, replicas, ready_replicas) -> Model:
    return Model(
        name=name,
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="org/repo",
        owner_principal_id=1,
        state=state,
        replicas=replicas,
        ready_replicas=ready_replicas,
    )


@pytest_asyncio.fixture
async def session():
    engine = create_async_engine("sqlite+aiosqlite://")
    async with engine.begin() as conn:
        await conn.run_sync(Model.__table__.create)
    async with AsyncSession(engine) as s:
        for row in _ROWS:
            s.add(_row(*row))
        await s.commit()
        yield s
    await engine.dispose()


async def _sql_matches(session, state) -> set:
    condition = model_state_condition(state)
    statement = select(Model.name)
    if condition is not None:
        statement = statement.where(condition)
    return set((await session.exec(statement)).all())


def _mirror_matches(state) -> set:
    return {
        name
        for name, model_state, replicas, ready_replicas in _ROWS
        if model_state_stream_filter(
            SimpleNamespace(
                state=model_state, replicas=replicas, ready_replicas=ready_replicas
            ),
            state,
        )
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "state, expected",
    [
        (
            ModelStateFilterEnum.READY,
            {"running-full", "partial-scaling", "legacy-ready"},
        ),
        (
            ModelStateFilterEnum.NOT_READY,
            {"pending", "errored", "legacy-not-ready"},
        ),
        (ModelStateFilterEnum.STOPPED, {"stopped", "legacy-stopped"}),
        (None, {name for name, _, _, _ in _ROWS}),
    ],
)
async def test_sql_filter_selects_by_state(session, state, expected):
    assert await _sql_matches(session, state) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "state",
    [
        ModelStateFilterEnum.READY,
        ModelStateFilterEnum.NOT_READY,
        ModelStateFilterEnum.STOPPED,
        None,
    ],
)
async def test_watch_mirror_agrees_with_the_sql(session, state):
    """The two are one filter with two implementations; a drift between them
    shows up as a list page and a watch stream disagreeing about a row."""
    assert _mirror_matches(state) == await _sql_matches(session, state)


@pytest.mark.asyncio
async def test_ready_and_not_ready_partition_the_deployed_models(session):
    """No row may fall out of both ``ready`` and ``not_ready``: that is what a
    three-valued ``state IN (...)`` would silently do to the rows that carry
    NULL, and it would look like the list page had lost them."""
    ready = await _sql_matches(session, ModelStateFilterEnum.READY)
    not_ready = await _sql_matches(session, ModelStateFilterEnum.NOT_READY)
    stopped = await _sql_matches(session, ModelStateFilterEnum.STOPPED)

    assert ready & not_ready == set()
    assert ready | not_ready | stopped == {name for name, _, _, _ in _ROWS}


@pytest.mark.parametrize(
    "state, expected",
    [
        (ModelStateFilterEnum.READY, True),
        (ModelStateFilterEnum.NOT_READY, True),
        (ModelStateFilterEnum.STOPPED, True),
    ],
)
def test_mirror_passes_id_only_delete_events(state, expected):
    """ID-only DELETED payloads carry no status. Dropping them would leave
    watch clients holding a row that no longer exists."""
    assert model_state_stream_filter({"id": 7}, state) is expected


def test_mirror_matches_the_old_counter_semantics_for_a_stateless_payload():
    """A payload from before the field existed is filtered by the counters,
    exactly as it was."""
    for ready_replicas, replicas, state, expected in [
        (2, 3, ModelStateFilterEnum.READY, True),
        (0, 3, ModelStateFilterEnum.READY, False),
        (0, 3, ModelStateFilterEnum.NOT_READY, True),
        (2, 3, ModelStateFilterEnum.NOT_READY, False),
        (0, 0, ModelStateFilterEnum.NOT_READY, False),
        (0, 0, ModelStateFilterEnum.STOPPED, True),
        (0, 3, ModelStateFilterEnum.STOPPED, False),
        (0, 3, None, True),
    ]:
        data = SimpleNamespace(ready_replicas=ready_replicas, replicas=replicas)
        assert model_state_stream_filter(data, state) is expected


def test_a_group_short_of_a_role_is_still_listed_as_ready():
    """The filter asks "is anything up", not "is it servable" — a group with
    its router down has members running, and hiding it from the list would
    hide the very row the user needs to look at."""
    group = SimpleNamespace(state=ModelStateEnum.PARTIAL, replicas=1, ready_replicas=4)

    assert model_state_stream_filter(group, ModelStateFilterEnum.READY) is True
    assert model_state_stream_filter(group, ModelStateFilterEnum.NOT_READY) is False
