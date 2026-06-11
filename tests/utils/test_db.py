from unittest.mock import MagicMock

import pytest
from sqlalchemy.dialects.postgresql import base as pg_base

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
