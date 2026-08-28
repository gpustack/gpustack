"""``pagination.total`` has to count the rows ``items`` returned.

``paginated_by_query`` lowers both sides of each ``fuzzy_fields`` LIKE for the
item query and neither side for the COUNT. So a search whose case differs from
the stored value returns the row while reporting a ``total`` that does not
include it: the page renders items the pagination says are not there, and with
enough rows the later pages become unreachable. Reproduced on PostgreSQL 17 as
``items=['A10G', 'a10g-lower'] total=1``.

Nothing noticed because SQLite's LIKE is ASCII-case-insensitive, which makes both
predicates agree by accident; ``PRAGMA case_sensitive_like = ON`` gives the
fixture engine PostgreSQL's semantics, so the defect is reproducible without a
live database.

``GPUInstanceType`` stands in for every model with the shape, as in
``test_active_record_streaming``: the fix is in the mixin, so all the routes that
pass ``fuzzy_fields`` inherit it.
"""

import pytest
import pytest_asyncio
from sqlalchemy import event
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.schemas.gpu_instance_types import GPUInstanceType, GPUInstanceTypeSpec
from gpustack.server.bus import Event, EventType


@pytest_asyncio.fixture
async def engine():
    e = create_async_engine("sqlite+aiosqlite://")

    @event.listens_for(e.sync_engine, "connect")
    def _case_sensitive_like(dbapi_conn, _record):
        # Without this, SQLite matches 'a10' against 'A10G' in both predicates
        # and the divergence is invisible.
        dbapi_conn.execute("PRAGMA case_sensitive_like = ON")

    async with e.begin() as conn:
        await conn.run_sync(GPUInstanceType.__table__.create)
    yield e
    await e.dispose()


async def _seed(engine, *names):
    async with AsyncSession(engine, expire_on_commit=False) as session:
        for name in names:
            row = GPUInstanceType(
                cluster_id=1, name=name, spec=GPUInstanceTypeSpec(display_name=None)
            )
            row.snapshot = row.compute_snapshot()
            session.add(row)
        await session.commit()


async def _page(engine, needle):
    async with AsyncSession(engine, expire_on_commit=False) as session:
        return await GPUInstanceType.paginated_by_query(
            session=session, fuzzy_fields={"name": needle}
        )


@pytest.mark.asyncio
async def test_a_case_differing_search_is_counted(engine):
    # The regression: items held the row while total said there were none.
    await _seed(engine, "A10G")

    page = await _page(engine, "a10")

    assert [i.name for i in page.items] == ["A10G"]
    assert page.pagination.total == 1
    assert page.pagination.totalPage == 1


@pytest.mark.asyncio
async def test_every_matched_row_is_counted(engine):
    # The PostgreSQL reproduction's exact shape: two rows match, only the
    # case-identical one was counted.
    await _seed(engine, "A10G", "a10g-lower", "H100")

    page = await _page(engine, "a10")

    assert sorted(i.name for i in page.items) == ["A10G", "a10g-lower"]
    assert page.pagination.total == 2


@pytest.mark.asyncio
async def test_a_case_identical_search_still_counts_the_same(engine):
    # Positive control — the case-insensitive count must not start over-counting
    # what a case-sensitive one already got right.
    await _seed(engine, "A10G", "a10g-lower")

    page = await _page(engine, "a10g-lower")

    assert [i.name for i in page.items] == ["a10g-lower"]
    assert page.pagination.total == 1


@pytest.mark.asyncio
async def test_a_non_matching_search_counts_nothing(engine):
    await _seed(engine, "A10G")

    page = await _page(engine, "l40s")

    assert page.items == []
    assert (page.pagination.total, page.pagination.totalPage) == (0, 0)


#
# The needle is a LIKE pattern, so its wildcards have to be escaped or the search
# means something the user did not type — and something the watch stream, which
# does a plain Python substring test, never agreed with.
#


@pytest.mark.asyncio
async def test_a_percent_matches_a_literal_percent_not_everything(engine):
    await _seed(engine, "100%-reserved", "plain")

    page = await _page(engine, "%")

    assert [i.name for i in page.items] == ["100%-reserved"]
    assert page.pagination.total == 1


@pytest.mark.asyncio
async def test_an_underscore_matches_a_literal_underscore_not_any_character(engine):
    await _seed(engine, "team_a", "teamXa")

    page = await _page(engine, "team_a")

    assert [i.name for i in page.items] == ["team_a"]
    assert page.pagination.total == 1


@pytest.mark.asyncio
async def test_a_backslash_matches_a_literal_backslash(engine):
    # The escape character itself: escaped first, or escaping the wildcards would
    # re-escape it and the pattern would stop meaning what it says.
    await _seed(engine, "back\\slash", "plain")

    page = await _page(engine, "back\\slash")

    assert [i.name for i in page.items] == ["back\\slash"]
    assert page.pagination.total == 1


@pytest.mark.asyncio
async def test_the_stream_twin_agrees_on_wildcard_characters(engine):
    """The read and the watch stream have to accept the same rows. Before the
    escaping they did not: SQL read ``%`` as "everything" while the stream read it
    as the character, so a searched page listed a superset of the rows it then
    received updates for."""
    await _seed(engine, "100%-reserved", "team_a", "teamXa", "plain")
    rows = [
        GPUInstanceType(
            cluster_id=1, name=n, spec=GPUInstanceTypeSpec(display_name=None)
        )
        for n in ("100%-reserved", "team_a", "teamXa", "plain")
    ]

    for needle in ("%", "_", "team_a", "plain"):
        listed = {i.name for i in (await _page(engine, needle)).items}
        accepted = {
            r.name
            for r in rows
            if GPUInstanceType._match_fuzzy_fields(
                Event(type=EventType.UPDATED, data=r), {"name": needle}
            )
        }
        assert accepted == listed, needle
