import pytest
from sqlalchemy import event
from sqlmodel import SQLModel

from gpustack import envs
from gpustack.server import init_db as init_db_module
from gpustack.server.init_db import (
    READ_ONLY_SQL_TRANSACTION_SQLSTATE,
    init_db_engine,
    listen_events,
    is_read_only_transaction_error,
    flag_readonly_error_as_disconnect,
)

POSTGRES_URL = "postgresql://user:pw@db.example.com:5432/gpustack"
MYSQL_URL = "mysql://user:pw@db.example.com:3306/gpustack"


class _FakeContext:
    """Stand-in for SQLAlchemy's ExceptionContext."""

    def __init__(self, exception):
        self.original_exception = exception
        self.is_disconnect = False


class _AsyncpgError(Exception):
    def __init__(self, sqlstate):
        super().__init__(sqlstate)
        self.sqlstate = sqlstate


async def _not_opengauss(_db_url):
    return False


@pytest.fixture
def contained_listen_events():
    """Undo the DDL listeners ``listen_events`` leaves on the shared metadata.

    ``listen_events`` registers ``after_create`` listeners on
    ``SQLModel.metadata``, which is process-global rather than per-engine, and
    it bakes the calling engine's dialect into the view DDL. Left in place they
    would accumulate across tests and fire against an unrelated engine.
    """
    before = list(SQLModel.metadata.dispatch.after_create)
    yield
    for listener in list(SQLModel.metadata.dispatch.after_create):
        if listener not in before:
            event.remove(SQLModel.metadata, "after_create", listener)


def test_read_only_sqlstate_is_recognised():
    """A demoted primary answers pool_pre_ping's SELECT 1 happily, so SQLSTATE
    25006 is the only signal that the connection is pinned to a node which can
    no longer be written to.
    """
    assert is_read_only_transaction_error(
        _AsyncpgError(READ_ONLY_SQL_TRANSACTION_SQLSTATE)
    )


def test_read_only_sqlstate_is_recognised_when_wrapped():
    """SQLAlchemy surfaces the driver error wrapped in its own Error type, with
    the asyncpg exception as __cause__.
    """
    wrapper = Exception("wrapped by SQLAlchemy")
    wrapper.__cause__ = _AsyncpgError(READ_ONLY_SQL_TRANSACTION_SQLSTATE)
    assert is_read_only_transaction_error(wrapper)


def test_read_only_sqlstate_is_recognised_through_orig():
    """Code that catches SQLAlchemy's wrapped DBAPIError sees the driver error
    as .orig rather than __cause__, so check that shape too.
    """
    wrapper = Exception("wrapped by SQLAlchemy")
    wrapper.orig = _AsyncpgError(READ_ONLY_SQL_TRANSACTION_SQLSTATE)
    assert is_read_only_transaction_error(wrapper)


def test_other_database_errors_are_not_read_only_failures():
    """Only the read-only state means the connection is on the wrong node. A
    constraint violation must not tear the pool down.
    """
    assert not is_read_only_transaction_error(_AsyncpgError("23505"))
    assert not is_read_only_transaction_error(Exception("no sqlstate at all"))
    unrelated = Exception("wrapped by SQLAlchemy")
    unrelated.orig = _AsyncpgError("23505")
    assert not is_read_only_transaction_error(unrelated)


def test_handler_flags_a_read_only_failure_as_a_disconnect():
    """Unless the error is reported as a disconnect the pool hands the same
    connection straight back and every following write fails too.
    """
    context = _FakeContext(_AsyncpgError(READ_ONLY_SQL_TRANSACTION_SQLSTATE))
    flag_readonly_error_as_disconnect(context)
    assert context.is_disconnect is True


def test_handler_leaves_unrelated_failures_alone():
    context = _FakeContext(_AsyncpgError("23505"))
    flag_readonly_error_as_disconnect(context)
    assert context.is_disconnect is False


@pytest.mark.asyncio
async def test_engine_bounds_pooled_connection_lifetime(monkeypatch):
    """Without pool_recycle SQLAlchemy never retires a connection, so a node
    demoted by a failover is reused for the life of the process.
    """
    monkeypatch.setattr(init_db_module, "is_opengauss", _not_opengauss)
    monkeypatch.setattr(envs, "DB_POOL_RECYCLE", 1800)
    engine = await init_db_engine(POSTGRES_URL)
    try:
        assert engine.pool._recycle == 1800
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_pool_recycle_can_be_disabled(monkeypatch):
    """0 means "never recycle", which SQLAlchemy spells -1."""
    monkeypatch.setattr(init_db_module, "is_opengauss", _not_opengauss)
    monkeypatch.setattr(envs, "DB_POOL_RECYCLE", 0)
    engine = await init_db_engine(POSTGRES_URL)
    try:
        assert engine.pool._recycle == -1
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_listen_events_registers_the_handler_on_postgres(
    monkeypatch, contained_listen_events
):
    monkeypatch.setattr(init_db_module, "is_opengauss", _not_opengauss)
    engine = await init_db_engine(POSTGRES_URL)
    try:
        listen_events(engine)
        assert event.contains(
            engine.sync_engine, "handle_error", flag_readonly_error_as_disconnect
        )
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_listen_events_skips_the_handler_on_mysql(contained_listen_events):
    """The handler keys on a PostgreSQL SQLSTATE, so it has no business on a
    MySQL engine.
    """
    engine = await init_db_engine(MYSQL_URL)
    try:
        listen_events(engine)
        assert not event.contains(
            engine.sync_engine, "handle_error", flag_readonly_error_as_disconnect
        )
    finally:
        await engine.dispose()
