from types import SimpleNamespace

import pytest
from sqlalchemy import event
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel

from gpustack.server.init_db import init_db_engine, listen_events

MYSQL_URL = "mysql://user:pw@db.example.com:3306/gpustack"
POSTGRES_URL = "postgresql://user:pw@db.example.com:5432/gpustack"
SQLITE_URL = "sqlite+aiosqlite:///:memory:"

VIEW_NAMES = ("gpu_devices_view", "non_admin_user_models", "principal_users")


def _drop_statement(url, view_name):
    """The drop statement ``view_name`` receives under ``url``'s dialect."""
    statement = f"DROP VIEW IF EXISTS {view_name}"
    if url == POSTGRES_URL and view_name == "gpu_devices_view":
        statement += " CASCADE"
    return statement


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


class _RecordingConnection:
    """Connection double that records executed SQL as strings."""

    def __init__(self, dialect_name):
        self.dialect = SimpleNamespace(name=dialect_name)
        self.executed = []

    def execute(self, element):
        self.executed.append(str(element))
        return SimpleNamespace(scalar=lambda: "")


async def _register_and_fire(url, monkeypatch, existing_views=()):
    """Register the view listeners for ``url``'s dialect and fire every
    metadata-level ``after_create`` listener against a recording connection.

    Returns the connection so callers can assert on the executed SQL. The
    ``inspect`` call inside the drop guard is pointed at ``existing_views``.
    """
    monkeypatch.setattr(
        "gpustack.server.init_db.sa_inspect",
        lambda _conn: SimpleNamespace(get_view_names=lambda: list(existing_views)),
    )
    monkeypatch.setattr("gpustack.server.init_db.is_opengauss", _not_opengauss)
    if url.startswith("sqlite"):
        # ``init_db_engine`` only accepts server URLs; the sqlite branch of
        # the listeners exists for the built-in single-node database.
        engine = create_async_engine(url)
    else:
        engine = await init_db_engine(url)
    try:
        listen_events(engine)
        connection = _RecordingConnection(engine.dialect.name)
        for listener in list(SQLModel.metadata.dispatch.after_create):
            listener(SQLModel.metadata, connection)
        return connection
    finally:
        await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("view_name", VIEW_NAMES)
async def test_missing_views_are_not_dropped_on_mysql(
    monkeypatch, contained_listen_events, view_name
):
    """A fresh MySQL or OceanBase database has none of the auxiliary views, so
    ``DROP VIEW IF EXISTS`` would succeed while still counting server warning
    1051 ("Unknown table"), which the driver logs. The drop must be skipped
    entirely when the view does not exist.
    """
    connection = await _register_and_fire(MYSQL_URL, monkeypatch, existing_views=[])
    assert _drop_statement(MYSQL_URL, view_name) not in connection.executed


@pytest.mark.asyncio
@pytest.mark.parametrize("view_name", VIEW_NAMES)
async def test_existing_views_are_still_dropped_on_mysql(
    monkeypatch, contained_listen_events, view_name
):
    """When a view survives from an earlier run it must be dropped before the
    create, so the refresh semantics of the listeners are unchanged.
    """
    from gpustack.schemas.stmt import (
        model_user_after_create_view_stmt,
        principal_users_after_create_view_stmt,
        worker_after_create_view_stmt_mysql,
    )

    connection = await _register_and_fire(
        MYSQL_URL, monkeypatch, existing_views=VIEW_NAMES
    )
    assert connection.executed.count(_drop_statement(MYSQL_URL, view_name)) == 1
    assert str(worker_after_create_view_stmt_mysql) in connection.executed
    assert str(principal_users_after_create_view_stmt()) in connection.executed
    assert str(model_user_after_create_view_stmt("mysql")) in connection.executed


@pytest.mark.asyncio
@pytest.mark.parametrize("url", [POSTGRES_URL, SQLITE_URL])
@pytest.mark.parametrize("view_name", VIEW_NAMES)
async def test_other_dialects_keep_unconditional_drops(
    monkeypatch, contained_listen_events, url, view_name
):
    """PostgreSQL and SQLite raise no warning for ``DROP VIEW IF EXISTS`` on a
    missing view, so those dialects keep dropping unconditionally and never
    pay the information_schema lookup.
    """
    monkeypatch.setattr(
        "gpustack.server.init_db.sa_inspect",
        lambda _conn: pytest.fail("non-mysql dialects must not inspect views"),
    )
    monkeypatch.setattr("gpustack.server.init_db.is_opengauss", _not_opengauss)
    connection = await _register_and_fire(url, monkeypatch)
    assert _drop_statement(url, view_name) in connection.executed
