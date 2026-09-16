from unittest.mock import MagicMock
from urllib.parse import urlparse

import pytest
from sqlalchemy.dialects.postgresql import base as pg_base

from gpustack.utils import db as db_utils
from gpustack.utils.db import patch_pg_version_info


def _version_info(version_string: str):
    patch_pg_version_info()
    connection = MagicMock()
    connection.exec_driver_sql.return_value.scalar.return_value = version_string
    dialect = pg_base.PGDialect.__new__(pg_base.PGDialect)
    return dialect._get_server_version_info(connection)


@pytest.mark.parametrize(
    "version_string, expected",
    [
        (
            "(openGauss 5.0.0 build 8e338bd1) compiled at 2023-03-29, 64-bit",
            (5, 0, 0),
        ),
        (
            "(openGauss-lite 7.0.0-RC3 build ) compiled at 2026-04-21 "
            "15:29:26 commit 0 last mr  release on aarch64-unknown-linux-gnu, "
            "compiled by g++ (GCC) 10.3.1, 64-bit",
            (7, 0, 0),
        ),
        ("(openGauss 5.0 build) 64-bit", (5, 0, 0)),
        (
            "PostgreSQL 14.5 on x86_64-pc-linux-gnu, compiled by gcc, 64-bit",
            (14, 5),
        ),
    ],
)
def test_pg_version_info(version_string, expected):
    assert _version_info(version_string) == expected


def test_pg_version_info_unparseable_raises():
    with pytest.raises(AssertionError):
        _version_info("TotallyUnknownDB 1.2.3")


class _FakeConnection:
    def __init__(self, version):
        self._version = version

    async def fetchval(self, _query):
        return self._version

    async def close(self):
        return None


@pytest.mark.asyncio
async def test_probe_dsn_drops_parameters_that_would_refuse_a_standby(monkeypatch):
    """The probe only reads version(), which every node in a cluster reports the
    same way. Honouring target_session_attrs here would fail startup whenever
    the DSN names a node that is not currently the primary, which is exactly
    when the parameter is worth setting.
    """
    captured = {}

    async def _fake_connect(dsn=None, **_kwargs):
        captured["dsn"] = dsn
        return _FakeConnection("PostgreSQL 16.1 on x86_64")

    monkeypatch.setattr(db_utils.asyncpg, "connect", _fake_connect)

    result = await db_utils.is_opengauss(
        "postgresql://user:pw@db.example.com:5432/gpustack"
        "?target_session_attrs=read-write&sslmode=require"
        "&options=-csearch_path=tenant_a"
    )

    assert result is False
    assert "target_session_attrs" not in captured["dsn"]
    assert "options" not in captured["dsn"]
    # Parameters that do not constrain which node answers are left alone.
    assert "sslmode=require" in captured["dsn"]


@pytest.mark.asyncio
async def test_opengauss_version_string_is_detected(monkeypatch):
    async def _fake_connect(dsn=None, **_kwargs):
        return _FakeConnection("(openGauss 7.0.0-RC3 build abc) compiled at ...")

    monkeypatch.setattr(db_utils.asyncpg, "connect", _fake_connect)
    assert await db_utils.is_opengauss("postgresql://user:pw@h:5432/gpustack") is True


@pytest.mark.asyncio
async def test_probe_dsn_moves_a_host_list_into_the_netloc(monkeypatch):
    """asyncpg reads the host list from the netloc and ignores host and port in
    the query string, so a DSN naming several nodes has to be respelled before
    the probe can try more than the one the netloc happens to name.
    """
    captured = {}

    async def _fake_connect(dsn=None, **_kwargs):
        captured["dsn"] = dsn
        return _FakeConnection("PostgreSQL 16.1 on x86_64")

    monkeypatch.setattr(db_utils.asyncpg, "connect", _fake_connect)

    await db_utils.is_opengauss(
        "postgresql://user:pw@db-a.example.com:5432/gpustack"
        "?host=db-a.example.com,db-b.example.com&port=5432,5433"
        "&target_session_attrs=read-write"
    )

    parsed = urlparse(captured["dsn"])
    assert parsed.netloc == "user:pw@db-a.example.com:5432,db-b.example.com:5433"
    assert "host=" not in (parsed.query or "")
    assert "port=" not in (parsed.query or "")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query, expected_netloc",
    [
        # One port applies to every host, the way libpq reads it.
        ("host=a,b&port=6432", "user:pw@a:6432,b:6432"),
        # No port at all leaves each host on the default.
        ("host=a,b", "user:pw@a:5432,b:5432"),
        # A bare IPv6 address carries colons of its own and needs bracketing.
        (
            "host=fd00::1,fd00::2&port=5432,5433",
            "user:pw@[fd00::1]:5432,[fd00::2]:5433",
        ),
    ],
)
async def test_probe_dsn_host_and_port_pairing(monkeypatch, query, expected_netloc):
    captured = {}

    async def _fake_connect(dsn=None, **_kwargs):
        captured["dsn"] = dsn
        return _FakeConnection("PostgreSQL 16.1 on x86_64")

    monkeypatch.setattr(db_utils.asyncpg, "connect", _fake_connect)

    await db_utils.is_opengauss(f"postgresql://user:pw@h:5432/gpustack?{query}")

    assert urlparse(captured["dsn"]).netloc == expected_netloc
