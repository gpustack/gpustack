"""Database-related utilities shared across GPUStack components."""

import re
from typing import List, Tuple
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

import asyncpg
from sqlalchemy.dialects.postgresql import base as pg_base

_pg_version_patched = False


# Query parameters dropped before the probe connects. ``options`` uses libpq's
# ``-c...`` syntax, which asyncpg does not accept. ``target_session_attrs``
# would refuse a node that cannot accept writes, and this probe only reads
# ``version()``, which every node in a cluster reports identically; leaving it
# in place would fail startup whenever the DSN happens to name a standby.
PROBE_EXCLUDED_PARAMS = {'options', 'target_session_attrs'}

DEFAULT_POSTGRES_PORT = '5432'


def _netloc_host_list(hosts: str, ports: str) -> str:
    """Render libpq's comma-separated host and port lists as one netloc.

    libpq allows a single port for every host; asyncpg wants the port spelled
    out next to each one. A count that matches neither one port nor one per
    host is left for SQLAlchemy to reject when it builds the engine, which it
    does with a message naming the mismatch.

    Args:
        hosts: Comma-separated host list, as it appeared in the query string.
        ports: Comma-separated port list, possibly empty.

    Returns:
        The ``host:port,host:port`` form asyncpg's DSN parser reads.
    """
    host_list = [h.strip() for h in hosts.split(',')]
    port_list = [p.strip() for p in ports.split(',')] if ports else []
    if len(port_list) == 1:
        port_list = port_list * len(host_list)
    if len(port_list) != len(host_list):
        port_list = [DEFAULT_POSTGRES_PORT] * len(host_list)
    return ','.join(
        # A bare IPv6 address carries colons of its own and has to be bracketed
        # before a port can be appended to it.
        f'[{host}]:{port}' if ':' in host else f'{host}:{port}'
        for host, port in zip(host_list, port_list)
    )


def _probe_dsn(db_url: str) -> str:
    """Build the DSN the openGauss probe connects with.

    asyncpg reads the host list from the netloc only: ``host`` and ``port`` in
    the query string are ignored outright, so a URL naming several nodes would
    leave the probe talking to whichever one the netloc happens to name. That
    is the node most likely to be down, since listing several is what an
    operator does when one of them may be, and the probe failing takes startup
    down with it before an engine is ever built. Moving the lists into the
    netloc lets asyncpg try each node in turn.

    Args:
        db_url: The PostgreSQL URL as configured.

    Returns:
        A DSN with ``PROBE_EXCLUDED_PARAMS`` removed and any host list moved
        into the netloc.
    """
    parsed = urlparse(db_url)
    params: List[Tuple[str, str]] = [
        (k, v)
        for k, v in parse_qsl(parsed.query, keep_blank_values=True)
        if k not in PROBE_EXCLUDED_PARAMS
    ]
    hosts = [v for k, v in params if k == 'host' and v]
    ports = [v for k, v in params if k == 'port' and v]
    if not hosts:
        return urlunparse(parsed._replace(query=urlencode(params)))

    params = [(k, v) for k, v in params if k not in ('host', 'port')]
    userinfo, sep, _ = parsed.netloc.rpartition('@')
    netloc = userinfo + sep + _netloc_host_list(hosts[-1], ports[-1] if ports else '')
    return urlunparse(parsed._replace(netloc=netloc, query=urlencode(params)))


async def is_opengauss(db_url: str) -> bool:
    """Return True when the PostgreSQL-shaped URL points at openGauss.

    Opens a one-off asyncpg connection and inspects ``SELECT version()`` —
    openGauss reports itself with ``openGauss`` in the version string
    rather than ``PostgreSQL``.

    Args:
        db_url: The PostgreSQL URL as configured.

    Returns:
        True when the server identifies itself as openGauss.
    """
    conn = await asyncpg.connect(dsn=_probe_dsn(db_url))
    try:
        version_str = await conn.fetchval("SELECT version()")
    finally:
        await conn.close()
    return 'openGauss' in (version_str or '')


def patch_pg_version_info() -> None:
    """Teach SQLAlchemy's PGDialect to parse openGauss version strings.

    openGauss presents itself with the PostgreSQL dialect but reports
    ``(openGauss X.Y.Z build ...)`` — or a variant such as
    ``(openGauss-lite X.Y.Z-RC3 build ...)`` — instead of
    ``PostgreSQL X.Y.Z``, which SQLAlchemy's default regex rejects
    with ``AssertionError``.
    We delegate to the original parser first so future upstream fixes
    are preserved, and only fall back to an openGauss regex on failure.

    Idempotent: safe to call multiple times.
    """
    global _pg_version_patched
    if _pg_version_patched:
        return
    _pg_version_patched = True

    orig_get_server_version_info = pg_base.PGDialect._get_server_version_info

    def _patched(self, connection):
        try:
            return orig_get_server_version_info(self, connection)
        except AssertionError:
            v = connection.exec_driver_sql("select pg_catalog.version()").scalar()
            m = re.search(r"openGauss\S* (\d+)\.(\d+)(?:\.(\d+))?", v or "")
            if not m:
                raise
            return tuple(int(x) if x is not None else 0 for x in m.group(1, 2, 3))

    pg_base.PGDialect._get_server_version_info = _patched
