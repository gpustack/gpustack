import inspect

import asyncpg
from sqlalchemy.dialects.postgresql import asyncpg as sa_asyncpg
from sqlalchemy.engine import make_url

from gpustack.server.init_db import build_postgres_connect_args

BASE_URL = "postgresql://user:pw@db.example.com:5432/gpustack"


def effective_asyncpg_kwargs(db_url, connect_args):
    """The keyword arguments asyncpg.connect() will actually receive.

    Asserting on these rather than on the intermediate dict covers the whole
    chain, including the translation SQLAlchemy's own dialect performs.
    """
    _, kwargs = sa_asyncpg.dialect().create_connect_args(make_url(db_url))
    kwargs.update(connect_args)
    return kwargs


def test_target_session_attrs_survives_the_url_rewrite():
    """The parameter that keeps the server on a writable node has to reach the
    driver. It used to be dropped along with the rest of the query string, which
    left an operator no way to ask for one.
    """
    kwargs = effective_asyncpg_kwargs(
        *build_postgres_connect_args(
            f"{BASE_URL}?target_session_attrs=read-write", opengauss=False
        )
    )
    assert kwargs["target_session_attrs"] == "read-write"


def test_multi_host_dsn_reaches_the_driver_as_a_host_list():
    """target_session_attrs is only useful when more than one node is on offer.
    SQLAlchemy's dialect is what splits the comma-separated lists, so the
    parameters have to stay in the URL for it to do that.
    """
    kwargs = effective_asyncpg_kwargs(
        *build_postgres_connect_args(
            f"{BASE_URL}?target_session_attrs=read-write"
            "&host=db-a.example.com,db-b.example.com&port=5432,5433",
            opengauss=False,
        )
    )
    assert kwargs["host"] == ["db-a.example.com", "db-b.example.com"]
    assert kwargs["port"] == [5432, 5433]
    assert kwargs["target_session_attrs"] == "read-write"


def test_sslmode_is_translated_to_the_name_asyncpg_accepts():
    """asyncpg has no sslmode parameter; forwarding one verbatim raises
    TypeError at connect time, so libpq's name has to be translated.
    """
    kwargs = effective_asyncpg_kwargs(
        *build_postgres_connect_args(f"{BASE_URL}?sslmode=disable", opengauss=False)
    )
    assert kwargs["ssl"] == "disable"
    assert "sslmode" not in kwargs


def test_embedded_database_url_yields_only_parameters_asyncpg_accepts():
    """Guards the default install: Config.get_database_url() appends
    ?sslmode=disable, and every one of these keys becomes an asyncpg.connect()
    keyword argument.
    """
    kwargs = effective_asyncpg_kwargs(
        *build_postgres_connect_args(
            "postgresql://root@127.0.0.1:5432/gpustack?sslmode=disable",
            opengauss=False,
        )
    )
    assert set(kwargs) <= set(inspect.signature(asyncpg.connect).parameters)


def test_search_path_option_still_becomes_a_server_setting():
    """libpq's options=-csearch_path=... has no asyncpg equivalent and must keep
    being turned into a server setting rather than forwarded.
    """
    db_url, connect_args = build_postgres_connect_args(
        f"{BASE_URL}?options=-csearch_path=tenant_a", opengauss=False
    )
    assert connect_args["server_settings"]["search_path"] == "tenant_a"
    assert "options" not in db_url


def test_unsupported_parameter_is_reported_not_silently_dropped(caplog):
    """A parameter asyncpg cannot accept is still dropped, because forwarding it
    would break startup, but it gets named in the log instead of vanishing.
    """
    with caplog.at_level("WARNING"):
        db_url, connect_args = build_postgres_connect_args(
            f"{BASE_URL}?connect_timeout=10", opengauss=False
        )
    assert "connect_timeout" not in effective_asyncpg_kwargs(db_url, connect_args)
    assert "connect_timeout" in caplog.text


def test_url_scheme_is_rewritten_for_asyncpg():
    db_url, _ = build_postgres_connect_args(BASE_URL, opengauss=False)
    assert db_url.startswith("postgresql+asyncpg://")


def test_url_without_parameters_is_unchanged_apart_from_the_scheme():
    db_url, connect_args = build_postgres_connect_args(BASE_URL, opengauss=False)
    assert db_url == "postgresql+asyncpg://user:pw@db.example.com:5432/gpustack"
    assert "server_settings" in connect_args


def test_opengauss_skips_the_idle_transaction_timeout():
    """openGauss rejects PostgreSQL's millisecond-scale value for it."""
    _, connect_args = build_postgres_connect_args(BASE_URL, opengauss=True)
    assert "idle_in_transaction_session_timeout" not in connect_args.get(
        "server_settings", {}
    )
