import inspect

import asyncpg
from sqlalchemy.dialects.postgresql import asyncpg as sa_asyncpg
from sqlalchemy.engine import make_url

from gpustack.server.init_db import (
    ASYNCPG_TYPED_PARAMS,
    build_postgres_connect_args,
    supported_asyncpg_params,
)

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
    driver, because it is the only way an operator can ask for one.
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

    The name here is deliberately one asyncpg will never grow, so the test keeps
    checking the behaviour rather than the status of one real parameter.
    """
    with caplog.at_level("WARNING"):
        db_url, connect_args = build_postgres_connect_args(
            f"{BASE_URL}?invalid_parameter=10", opengauss=False
        )
    assert "invalid_parameter" not in effective_asyncpg_kwargs(db_url, connect_args)
    assert "invalid_parameter" in caplog.text


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


def test_numeric_parameters_reach_the_driver_as_numbers():
    """asyncpg does arithmetic on these, so a str raises TypeError inside
    connect(). Out of a query string the dialect coerces only the port, so
    everything else has to be parsed before it is handed over.
    """
    kwargs = effective_asyncpg_kwargs(
        *build_postgres_connect_args(
            f"{BASE_URL}?timeout=5&statement_cache_size=0&command_timeout=2.5",
            opengauss=False,
        )
    )
    assert kwargs["timeout"] == 5.0
    assert isinstance(kwargs["timeout"], float)
    assert kwargs["statement_cache_size"] == 0
    assert isinstance(kwargs["statement_cache_size"], int)
    assert kwargs["command_timeout"] == 2.5


def test_boolean_parameter_is_parsed_rather_than_read_as_a_non_empty_string():
    """bool("false") is True, so a forwarded string turns direct TLS on for
    whoever asked to turn it off.
    """
    kwargs = effective_asyncpg_kwargs(
        *build_postgres_connect_args(f"{BASE_URL}?direct_tls=false", opengauss=False)
    )
    assert kwargs["direct_tls"] is False

    kwargs = effective_asyncpg_kwargs(
        *build_postgres_connect_args(f"{BASE_URL}?direct_tls=on", opengauss=False)
    )
    assert kwargs["direct_tls"] is True


def test_typed_parameter_that_does_not_parse_is_reported_not_forwarded(caplog):
    with caplog.at_level("WARNING"):
        kwargs = effective_asyncpg_kwargs(
            *build_postgres_connect_args(f"{BASE_URL}?timeout=soon", opengauss=False)
        )
    assert "timeout" not in kwargs
    assert "timeout=soon" in caplog.text


def test_typed_parameter_table_names_real_driver_parameters():
    """Guards the table against a driver rename: a name that asyncpg no longer
    takes would be parsed here and then rejected at connect time.
    """
    assert set(ASYNCPG_TYPED_PARAMS) <= supported_asyncpg_params()


def test_prepared_statement_cache_size_reaches_the_dialect(caplog):
    """SQLAlchemy's asyncpg dbapi pops this one before calling the driver, so it
    is absent from asyncpg.connect's signature while still being accepted.
    Setting it to 0 is the documented way to run through PgBouncer.
    """
    with caplog.at_level("WARNING"):
        kwargs = effective_asyncpg_kwargs(
            *build_postgres_connect_args(
                f"{BASE_URL}?prepared_statement_cache_size=0", opengauss=False
            )
        )
    assert kwargs["prepared_statement_cache_size"] == 0
    assert "prepared_statement_cache_size" not in caplog.text


def test_ssl_and_sslmode_together_are_reported(caplog):
    """They are one setting under two names, and only one of them can win."""
    with caplog.at_level("WARNING"):
        kwargs = effective_asyncpg_kwargs(
            *build_postgres_connect_args(
                f"{BASE_URL}?ssl=require&sslmode=disable", opengauss=False
            )
        )
    assert kwargs["ssl"] == "disable"
    assert "ssl" in caplog.text and "sslmode" in caplog.text


def test_parameter_without_a_value_is_reported_not_forwarded(caplog):
    """libpq reads an empty value as "use the default"; asyncpg raises
    ClientConfigurationError instead, so a blank cannot be passed on.
    """
    with caplog.at_level("WARNING"):
        kwargs = effective_asyncpg_kwargs(
            *build_postgres_connect_args(f"{BASE_URL}?sslmode=", opengauss=False)
        )
    assert "ssl" not in kwargs
    assert "sslmode" in caplog.text


def test_unhandled_options_value_is_reported_not_silently_dropped(caplog):
    """options is consumed before the generic report on unusable parameters
    runs, so a value that is not -csearch_path=... needs naming here or it
    would vanish without one.
    """
    with caplog.at_level("WARNING"):
        db_url, connect_args = build_postgres_connect_args(
            f"{BASE_URL}?options=-cstatement_timeout%3D30s", opengauss=False
        )
    assert "search_path" not in connect_args.get("server_settings", {})
    assert "options" not in db_url
    assert "statement_timeout" in caplog.text
