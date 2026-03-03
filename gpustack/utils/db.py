"""Database-related utilities shared across GPUStack components."""

import re

from sqlalchemy.dialects.postgresql import base as pg_base

_pg_version_patched = False


def patch_pg_version_info() -> None:
    """Teach SQLAlchemy's PGDialect to parse openGauss version strings.

    openGauss presents itself with the PostgreSQL dialect but reports
    ``(openGauss X.Y.Z build ...)`` instead of ``PostgreSQL X.Y.Z``,
    which SQLAlchemy's default regex rejects with ``AssertionError``.
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
            m = re.search(r"openGauss (\d+)\.(\d+)(?:\.(\d+))?", v or "")
            if not m:
                raise
            return tuple(int(x) if x is not None else 0 for x in m.group(1, 2, 3))

    pg_base.PGDialect._get_server_version_info = _patched
