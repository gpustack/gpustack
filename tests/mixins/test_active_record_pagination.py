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

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from sqlalchemy import event
from sqlalchemy import inspect, update
from sqlalchemy.dialects import mysql, postgresql
from sqlalchemy.exc import IntegrityError, InvalidRequestError
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.schemas.gpu_instance_types import GPUInstanceType, GPUInstanceTypeSpec
from gpustack.schemas.workers import Worker
from gpustack.server.bus import Event, EventType
from gpustack.mixins import active_record


@pytest_asyncio.fixture
async def engine():
    e = create_async_engine("sqlite+aiosqlite://")

    @event.listens_for(e.sync_engine, "connect")
    def _case_sensitive_like(dbapi_conn, _record):
        dbapi_conn.isolation_level = None
        # Without this, SQLite matches 'a10' against 'A10G' in both predicates
        # and the divergence is invisible.
        dbapi_conn.execute("PRAGMA case_sensitive_like = ON")

    @event.listens_for(e.sync_engine, "begin")
    def _begin(connection):
        # SAVEPOINT must remain inside a real outer database transaction.
        connection.exec_driver_sql("BEGIN")

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


async def _read(session, row_id, multiple, **kwargs):
    if multiple == "ids":
        rows = await GPUInstanceType.all_by_ids(session, [row_id], **kwargs)
        return rows[0] if rows else None
    if multiple:
        rows = await GPUInstanceType.all_by_field(session, "id", row_id, **kwargs)
        return rows[0] if rows else None
    return await GPUInstanceType.one_by_id(session, row_id, **kwargs)


async def _refreshed(session, row_id, multiple):
    return await _read(session, row_id, multiple, refresh=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("multiple", [False, True, "ids"])
@pytest.mark.parametrize("kwargs", [{}, {"refresh": False}])
async def test_refresh_is_opt_in(engine, multiple, kwargs):
    await _seed(engine, "original")
    async with AsyncSession(engine, expire_on_commit=False) as session:
        row = (await GPUInstanceType.all(session))[0]
        await session.exec(
            update(GPUInstanceType)
            .where(GPUInstanceType.id == row.id)
            .values(name="database")
            .execution_options(synchronize_session=False)
        )
        assert await _read(session, row.id, multiple, **kwargs) is row
        assert row.name == "original"


@pytest.mark.asyncio
@pytest.mark.parametrize("multiple", [True, "ids"])
@pytest.mark.parametrize("kwargs", [{}, {"refresh": False}])
async def test_normal_query_preserves_autoflush(engine, multiple, kwargs):
    await _seed(engine, "original")
    async with AsyncSession(engine, expire_on_commit=False) as session:
        row = (await GPUInstanceType.all(session))[0]
        row.name = "local"
        flushed = MagicMock()
        event.listen(session.sync_session, "before_flush", flushed)
        assert await _read(session, row.id, multiple, **kwargs) is row
        assert row.name == "local"
        flushed.assert_called_once()
        assert not session.is_modified(row)


@pytest.mark.asyncio
@pytest.mark.parametrize("multiple", [False, True, "ids"])
async def test_explicit_refresh_reloads_identity_map_without_changing_old_reader(
    engine, multiple
):
    await _seed(engine, "original")
    async with AsyncSession(engine, expire_on_commit=False) as session:
        row = (await GPUInstanceType.all(session))[0]
        await session.exec(
            update(GPUInstanceType)
            .where(GPUInstanceType.id == row.id)
            .values(name="database")
            .execution_options(synchronize_session=False)
        )
        assert (await GPUInstanceType.one_by_id(session, row.id)).name == "original"
        refreshed = await _refreshed(session, row.id, multiple)
        assert refreshed is row
        assert row.name == "database"
        assert inspect(row).session is session.sync_session
        assert session.in_transaction()


@pytest.mark.asyncio
@pytest.mark.parametrize("multiple", [False, True, "ids"])
@pytest.mark.parametrize("change", ["scalar", "json", "new", "deleted"])
async def test_refresh_rejects_unflushed_target_changes(engine, multiple, change):
    await _seed(engine, "original")
    async with AsyncSession(engine, expire_on_commit=False) as session:
        row = (await GPUInstanceType.all(session))[0]
        row_id = row.id
        if change == "scalar":
            row.name = "local"
        elif change == "json":
            row.spec = GPUInstanceTypeSpec(display_name="local")
        elif change == "new":
            session.add(
                GPUInstanceType(
                    cluster_id=1, name="pending", spec=GPUInstanceTypeSpec()
                )
            )
        else:
            await session.delete(row)
        flush = MagicMock()
        event.listen(session.sync_session, "before_flush", flush)
        with pytest.raises(InvalidRequestError, match="flush"):
            await _refreshed(session, row_id, multiple)
        flush.assert_not_called()
        assert session.in_transaction()
        if change == "scalar":
            assert row.name == "local"
        if change == "json":
            assert row.spec.display_name == "local"
        await session.rollback()
        assert (await _refreshed(session, row_id, multiple)).name == "original"


@pytest.mark.asyncio
@pytest.mark.parametrize("multiple", [False, True, "ids"])
async def test_refresh_preserves_pending_commit_event_and_does_not_touch_cache(
    engine, monkeypatch, multiple
):
    await _seed(engine, "original")
    publish = AsyncMock()
    invalidate = AsyncMock()
    monkeypatch.setattr(active_record.event_bus, "publish", publish)
    monkeypatch.setattr(active_record, "delete_cache_by_key", invalidate)
    async with AsyncSession(engine, expire_on_commit=False) as session:
        row = (await GPUInstanceType.all(session))[0]
        await row.update(session, {"name": "staged"}, auto_commit=False)
        pending = list(session.info["pending_events"])
        cache_calls = invalidate.await_count
        publish.assert_not_awaited()
        assert (await _refreshed(session, row.id, multiple)).name == "staged"
        assert session.info["pending_events"] == pending
        assert invalidate.await_count == cache_calls
        publish.assert_not_awaited()
        await session.commit()
        await asyncio.sleep(0)
        publish.assert_awaited_once()
        assert "name" in publish.await_args.args[1].changed_fields


@pytest.mark.asyncio
@pytest.mark.parametrize("multiple", [False, True, "ids"])
async def test_refreshed_read_does_not_publish_rolled_back_write(
    engine, monkeypatch, multiple
):
    await _seed(engine, "original")
    publish = AsyncMock()
    monkeypatch.setattr(active_record.event_bus, "publish", publish)
    monkeypatch.setattr(active_record, "delete_cache_by_key", AsyncMock())
    async with AsyncSession(engine, expire_on_commit=False) as session:
        row = (await GPUInstanceType.all(session))[0]
        row_id = row.id
        await row.update(session, {"name": "rolled-back"}, auto_commit=False)
        await session.rollback()
        assert (await _refreshed(session, row_id, multiple)).name == "original"
        await asyncio.sleep(0)
        publish.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("multiple", [False, True, "ids"])
@pytest.mark.parametrize("ending", ["commit", "rollback"])
async def test_refresh_connection_is_released_only_by_caller(engine, multiple, ending):
    await _seed(engine, "original")
    counts = {"checkout": 0, "checkin": 0}

    def checkout(*args):
        counts["checkout"] += 1

    def checkin(*args):
        counts["checkin"] += 1

    event.listen(engine.sync_engine, "checkout", checkout)
    event.listen(engine.sync_engine, "checkin", checkin)
    async with AsyncSession(engine, expire_on_commit=False) as session:
        row = (await GPUInstanceType.all(session))[0]
        row_id = row.id
        await _refreshed(session, row_id, multiple)
        assert counts == {"checkout": 1, "checkin": 0}
        await getattr(session, ending)()
        assert counts == {"checkout": 1, "checkin": 1}
        await _refreshed(session, row_id, multiple)
        assert counts["checkout"] == 2
    assert counts == {"checkout": 2, "checkin": 2}


@pytest.mark.asyncio
@pytest.mark.parametrize("multiple", [False, True, "ids"])
async def test_refresh_does_not_flush_another_models_pending_insert(engine, multiple):
    await _seed(engine, "original")
    async with AsyncSession(engine, expire_on_commit=False) as session:
        row_id = (await GPUInstanceType.all(session))[0].id
        pending = Worker(name="pending", hostname="pending", ip="127.0.0.1")
        session.add(pending)
        flush = MagicMock()
        event.listen(session.sync_session, "before_flush", flush)
        assert (await _refreshed(session, row_id, multiple)).name == "original"
        flush.assert_not_called()
        assert pending in session.new


@pytest.mark.asyncio
@pytest.mark.parametrize("multiple", [False, True, "ids"])
async def test_explicit_flush_preserves_tracked_json_before_refresh(engine, multiple):
    await _seed(engine, "original")
    async with AsyncSession(engine, expire_on_commit=False) as session:
        row = (await GPUInstanceType.all(session))[0]
        row.spec = GPUInstanceTypeSpec(display_name="updated")
        with pytest.raises(InvalidRequestError):
            await _refreshed(session, row.id, multiple)
        await session.flush()
        assert (
            await _refreshed(session, row.id, multiple)
        ).spec.display_name == "updated"


@pytest.mark.asyncio
@pytest.mark.parametrize("multiple", [False, True, "ids"])
async def test_refreshed_missing_row_keeps_query_contract(engine, multiple):
    async with AsyncSession(engine) as session:
        assert await _refreshed(session, 999, multiple) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("multiple", [False, True, "ids"])
async def test_refresh_keeps_load_hooks_without_using_application_cache(
    engine, monkeypatch, multiple
):
    await _seed(engine, "original")
    from gpustack.server import cache as cache_module

    monkeypatch.setattr(
        cache_module.cache,
        "get",
        AsyncMock(side_effect=AssertionError("application cache read")),
    )
    monkeypatch.setattr(
        cache_module.cache,
        "set",
        AsyncMock(side_effect=AssertionError("application cache write")),
    )
    monkeypatch.setattr(
        cache_module.cache,
        "delete",
        AsyncMock(side_effect=AssertionError("application cache invalidation")),
    )
    async with AsyncSession(engine) as session:
        row = (await GPUInstanceType.all(session))[0]
        refreshed = MagicMock()
        event.listen(GPUInstanceType, "refresh", refreshed)
        try:
            assert await _refreshed(session, row.id, multiple) is row
            refreshed.assert_called_once()
        finally:
            event.remove(GPUInstanceType, "refresh", refreshed)


@pytest.mark.asyncio
@pytest.mark.parametrize("multiple", [False, True, "ids"])
async def test_refresh_exception_leaves_cleanup_to_session_owner(engine, multiple):
    await _seed(engine, "original")
    checkin = MagicMock()
    event.listen(engine.sync_engine, "checkin", checkin)

    def fail(*args):
        raise RuntimeError("read failed")

    event.listen(engine.sync_engine, "before_cursor_execute", fail)
    try:
        with pytest.raises(RuntimeError, match="read failed"):
            async with AsyncSession(engine) as session:
                await _refreshed(session, 1, multiple)
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", fail)
    checkin.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("dialect", [postgresql.dialect(), mysql.dialect()])
@pytest.mark.parametrize("by_ids", [False, True])
async def test_refresh_batch_locks_rows_in_primary_key_order(dialect, by_ids):
    session = MagicMock()
    session.new = session.dirty = session.deleted = []
    result = MagicMock()
    result.all.return_value = []
    session.exec = AsyncMock(return_value=result)
    if by_ids:
        assert (
            await GPUInstanceType.all_by_ids(
                session, [3, 1, 3], for_update=True, refresh=True
            )
            == []
        )
    else:
        assert (
            await GPUInstanceType.all_by_field(
                session, "cluster_id", 1, for_update=True, refresh=True
            )
            == []
        )
    assert session.exec.await_count == 1
    statement = session.exec.await_args.args[0]
    sql = str(
        statement.compile(dialect=dialect, compile_kwargs={"literal_binds": True})
    )
    assert "ORDER BY gpu_instance_types.id" in sql
    assert "FOR UPDATE" in sql
    assert statement.get_execution_options()["populate_existing"] is True
    session.commit.assert_not_called()
    session.rollback.assert_not_called()
    session.close.assert_not_called()


@pytest.fixture
def published_events(monkeypatch):
    publish = AsyncMock()
    monkeypatch.setattr(active_record.event_bus, "publish", publish)
    monkeypatch.setattr(active_record, "delete_cache_by_key", AsyncMock())
    return publish


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["create", "update", "delete"])
async def test_rollback_events_cannot_escape_through_the_next_commit(
    engine, published_events, operation
):
    await _seed(engine, "original")
    async with AsyncSession(engine, expire_on_commit=False) as session:
        row = (await GPUInstanceType.all(session))[0]
        if operation == "create":
            pending = GPUInstanceType(
                cluster_id=1, name="created", spec=GPUInstanceTypeSpec()
            )
            pending.snapshot = pending.compute_snapshot()
            await GPUInstanceType.create(session, pending, auto_commit=False)
        elif operation == "update":
            await row.update(session, {"name": "discarded"}, auto_commit=False)
        else:
            await row.delete(session, auto_commit=False)
            await session.flush()
        await session.rollback()
        await row.refresh(session)
        assert row.name == "original"
        await session.commit()
        await asyncio.sleep(0)
        published_events.assert_not_awaited()
        assert "pending_events" not in session.info
        assert [item.name for item in await GPUInstanceType.all(session)] == [
            "original"
        ]


@pytest.mark.asyncio
@pytest.mark.parametrize("inner_commit", [False, True])
@pytest.mark.parametrize("outer_commit", [False, True])
async def test_savepoint_events_wait_for_the_outer_transaction(
    engine, published_events, inner_commit, outer_commit
):
    await _seed(engine, "outer", "inner")
    async with AsyncSession(engine, expire_on_commit=False) as session:
        rows = {row.name: row for row in await GPUInstanceType.all(session)}
        await rows["outer"].update(session, {"name": "outer-saved"}, auto_commit=False)
        nested = await session.begin_nested()
        await rows["inner"].update(session, {"name": "inner-saved"}, auto_commit=False)
        await (nested.commit() if inner_commit else nested.rollback())
        await asyncio.sleep(0)
        published_events.assert_not_awaited()
        await (session.commit() if outer_commit else session.rollback())
        await asyncio.sleep(0)
        names = [call.args[1].data.name for call in published_events.await_args_list]
        assert names == (
            ["outer-saved"] + (["inner-saved"] if inner_commit else [])
            if outer_commit
            else []
        )
        assert "pending_events" not in session.info
        assert "pending_event_savepoints" not in session.info
        actual = {row.name for row in await GPUInstanceType.all(session)}
        assert actual == {
            "outer-saved" if outer_commit else "outer",
            "inner-saved" if outer_commit and inner_commit else "inner",
        }


@pytest.mark.asyncio
async def test_savepoint_rollback_restores_outer_payload_and_changed_fields(
    engine, published_events
):
    await _seed(engine, "original")
    async with AsyncSession(engine, expire_on_commit=False) as session:
        row = (await GPUInstanceType.all(session))[0]
        row_id = row.id
        await row.update(session, {"name": "outer-saved"}, auto_commit=False)
        nested = await session.begin_nested()
        await row.update(
            session,
            {"name": "discarded", "spec": GPUInstanceTypeSpec(display_name="bad")},
            auto_commit=False,
        )
        await nested.rollback()
        assert inspect(row).expired_attributes
        queries = MagicMock()
        event.listen(engine.sync_engine, "before_cursor_execute", queries)
        await session.commit()
        queries.assert_not_called()
        await asyncio.sleep(0)
        published_events.assert_awaited_once()
        topic, notification = published_events.await_args.args
        assert topic == "gpuinstancetype"
        assert notification.id == notification.data.id == row_id
        assert notification.data.name == "outer-saved"
        assert notification.data.spec.display_name is None
        assert list(notification.changed_fields["name"][0]) == ["original"]
        assert list(notification.changed_fields["name"][1]) == ["outer-saved"]
        assert "spec" not in notification.changed_fields
        assert inspect(notification.data).session is None
        assert (await GPUInstanceType.one_by_id(session, row_id)).name == "outer-saved"


@pytest.mark.asyncio
async def test_committed_child_events_are_discarded_with_a_rolled_back_parent(
    engine, published_events
):
    await _seed(engine, "original")
    async with AsyncSession(engine, expire_on_commit=False) as session:
        row = (await GPUInstanceType.all(session))[0]
        await row.update(session, {"name": "outer-saved"}, auto_commit=False)
        middle = await session.begin_nested()
        await row.update(session, {"name": "middle"}, auto_commit=False)
        child = await session.begin_nested()
        await row.update(session, {"name": "child"}, auto_commit=False)
        await child.commit()
        await middle.rollback()
        await session.commit()
        await asyncio.sleep(0)
        published_events.assert_awaited_once()
        assert published_events.await_args.args[1].data.name == "outer-saved"


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["close", "reset", "context_exception"])
async def test_uncommitted_events_are_discarded_when_session_is_reused(
    engine, published_events, ending
):
    await _seed(engine, "original")
    async with AsyncSession(engine, expire_on_commit=False) as session:
        if ending == "context_exception":
            with pytest.raises(RuntimeError, match="abort"):
                async with session.begin():
                    row = (await GPUInstanceType.all(session))[0]
                    await row.update(session, {"name": "discarded"}, auto_commit=False)
                    raise RuntimeError("abort")
        else:
            row = (await GPUInstanceType.all(session))[0]
            await row.update(session, {"name": "discarded"}, auto_commit=False)
            await getattr(session, ending)()
        assert "pending_events" not in session.info
        assert "pending_event_savepoints" not in session.info
        row = (await GPUInstanceType.all(session))[0]
        assert row.name == "original"
        await row.update(session, {"name": "saved"})
        await asyncio.sleep(0)
        published_events.assert_awaited_once()
        assert published_events.await_args.args[1].data.name == "saved"


@pytest.mark.asyncio
async def test_flush_failure_and_recovery_do_not_publish_failed_events(
    engine, published_events
):
    await _seed(engine, "original")
    async with AsyncSession(engine, expire_on_commit=False) as session:
        row = (await GPUInstanceType.all(session))[0]
        duplicate = GPUInstanceType(
            cluster_id=1, name="original", spec=GPUInstanceTypeSpec()
        )
        duplicate.snapshot = duplicate.compute_snapshot()
        with pytest.raises(IntegrityError):
            await GPUInstanceType.create(session, duplicate)
        await row.refresh(session)
        await row.update(session, {"name": "recovered"})
        await asyncio.sleep(0)
        published_events.assert_awaited_once()
        assert published_events.await_args.args[1].type == EventType.UPDATED
        assert published_events.await_args.args[1].data.name == "recovered"


@pytest.mark.asyncio
async def test_committed_create_has_a_detached_payload_and_generated_id(
    engine, published_events
):
    row = GPUInstanceType(cluster_id=1, name="created", spec=GPUInstanceTypeSpec())
    row.snapshot = row.compute_snapshot()
    async with AsyncSession(engine, expire_on_commit=False) as session:
        await GPUInstanceType.create(session, row)
        await asyncio.sleep(0)
        published_events.assert_awaited_once()
        notification = published_events.await_args.args[1]
        assert notification.type == EventType.CREATED
        assert notification.id == notification.data.id == row.id
        assert notification.id is not None
        assert inspect(notification.data).session is None
        row.name = "later"
        assert notification.data.name == "created"


@pytest.mark.asyncio
async def test_commit_failure_discards_events_before_session_recovery(
    engine, published_events
):
    await _seed(engine, "original")
    async with AsyncSession(engine, expire_on_commit=False) as session:
        row = (await GPUInstanceType.all(session))[0]
        await row.update(session, {"name": "discarded"}, auto_commit=False)

        def fail_commit(connection):
            raise RuntimeError("commit failed")

        event.listen(engine.sync_engine, "commit", fail_commit)
        try:
            with pytest.raises(RuntimeError, match="commit failed"):
                await session.commit()
        finally:
            event.remove(engine.sync_engine, "commit", fail_commit)
        await session.rollback()
        await row.refresh(session)
        assert row.name == "original"
        await row.update(session, {"name": "recovered"})
        await asyncio.sleep(0)
        published_events.assert_awaited_once()
        assert published_events.await_args.args[1].data.name == "recovered"


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit_begin", [False, True])
async def test_event_scope_does_not_acquire_a_database_connection(
    published_events, explicit_begin
):
    # No bind: starting the ORM scope and discarding it must not execute SQL.
    async with AsyncSession(autobegin=not explicit_begin) as session:
        if explicit_begin:
            with pytest.raises(InvalidRequestError, match="Autobegin"):
                GPUInstanceType._publish_event_after_commit(
                    session, EventType.CREATED, GPUInstanceType(id=1)
                )
            assert "pending_events" not in session.info
            await session.begin()
        GPUInstanceType._publish_event_after_commit(
            session, EventType.CREATED, GPUInstanceType(id=1)
        )
        assert session.in_transaction()
        await session.close()
        assert "pending_events" not in session.info
        published_events.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_savepoint_flush_conflict_preserves_valid_outer_events(
    engine, published_events
):
    await _seed(engine, "original")
    async with AsyncSession(engine, expire_on_commit=False) as session:
        row = (await GPUInstanceType.all(session))[0]
        await row.update(session, {"name": "outer-saved"}, auto_commit=False)
        nested = await session.begin_nested()
        duplicate = GPUInstanceType(
            cluster_id=1,
            name="duplicate",
            snapshot=row.snapshot,
            spec=GPUInstanceTypeSpec(),
        )
        GPUInstanceType._publish_event_after_commit(
            session, EventType.CREATED, duplicate
        )
        session.add(duplicate)
        with pytest.raises(IntegrityError):
            await session.flush()
        await nested.rollback()
        await session.commit()
        await asyncio.sleep(0)
        published_events.assert_awaited_once()
        notification = published_events.await_args.args[1]
        assert notification.type == EventType.UPDATED
        assert notification.data.name == "outer-saved"


@pytest.mark.asyncio
async def test_outer_snapshot_keeps_successful_inner_changes_after_later_rollback(
    engine, published_events
):
    await _seed(engine, "original")
    async with AsyncSession(engine, expire_on_commit=False) as session:
        row = (await GPUInstanceType.all(session))[0]
        await row.update(session, {"name": "outer-saved"}, auto_commit=False)
        async with session.begin_nested():
            await row.update(session, {"name": "inner-saved"}, auto_commit=False)
        nested = await session.begin_nested()
        await row.update(session, {"name": "discarded"}, auto_commit=False)
        await nested.rollback()
        await session.commit()
        await asyncio.sleep(0)
        notifications = [call.args[1] for call in published_events.await_args_list]
        assert len(notifications) == 2
        assert all(item.data.name == "inner-saved" for item in notifications)
        assert all(
            "discarded" not in str(item.changed_fields) for item in notifications
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["commit", "rollback", "close"])
async def test_transaction_event_tracking_does_not_retain_connections(
    engine, published_events, ending
):
    await _seed(engine, "original")
    checkout, checkin = MagicMock(), MagicMock()
    event.listen(engine.sync_engine, "checkout", checkout)
    event.listen(engine.sync_engine, "checkin", checkin)
    async with AsyncSession(engine, expire_on_commit=False) as session:
        row = (await GPUInstanceType.all(session))[0]
        await row.update(session, {"name": "outer-saved"}, auto_commit=False)
        nested = await session.begin_nested()
        await row.update(session, {"name": "discarded"}, auto_commit=False)
        await nested.rollback()
        checkout.assert_called_once()
        checkin.assert_not_called()
        await getattr(session, ending)()
        assert "pending_events" not in session.info
        assert "pending_event_savepoints" not in session.info
        checkout.assert_called_once()
        checkin.assert_called_once()
    await asyncio.sleep(0)
    assert published_events.await_count == (1 if ending == "commit" else 0)


@pytest.mark.asyncio
async def test_batch_id_refresh_orders_deduplicates_and_empty_ids_do_not_query(engine):
    await _seed(engine, "a", "b", "c")
    async with AsyncSession(engine, expire_on_commit=False) as session:
        rows = await GPUInstanceType.all(session)
        ids = sorted(row.id for row in rows)
        found = await GPUInstanceType.all_by_ids(
            session, [ids[2], ids[0], ids[2], max(ids) + 100], refresh=True
        )
        assert [row.id for row in found] == [ids[0], ids[2]]
        executed = MagicMock()
        event.listen(engine.sync_engine, "before_cursor_execute", executed)
        try:
            assert await GPUInstanceType.all_by_ids(session, [], refresh=True) == []
            executed.assert_not_called()
        finally:
            event.remove(engine.sync_engine, "before_cursor_execute", executed)
        assert session.in_transaction()
