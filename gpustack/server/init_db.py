import asyncio
import logging
import threading
import time
import traceback
import re
from urllib.parse import urlparse, parse_qs, urlunparse
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    create_async_engine,
)
from sqlmodel import SQLModel
from sqlalchemy import DDL, event, text

from gpustack import envs
from gpustack.server import db
from gpustack.utils.db import is_opengauss
from gpustack.schemas.api_keys import ApiKey
from gpustack.schemas.inference_backend import InferenceBackend
from gpustack.schemas.model_usage import ModelUsage
from gpustack.schemas.models import Model, ModelInstance
from gpustack.schemas.system_load import SystemLoad
from gpustack.schemas.users import User
from gpustack.schemas.workers import Worker
from gpustack.schemas.clusters import (
    Cluster,
    CloudCredential,
    WorkerPool,
    Credential,
)
from gpustack.schemas.stmt import (
    worker_after_create_view_stmt_sqlite,
    worker_after_drop_view_stmt_sqlite,
    worker_after_create_view_stmt_postgres,
    worker_after_drop_view_stmt_postgres,
    worker_after_create_view_stmt_opengauss,
    worker_after_create_view_stmt_mysql,
    worker_after_drop_view_stmt_mysql,
    model_user_after_drop_view_stmt,
    model_user_after_create_view_stmt,
    principal_users_after_drop_view_stmt,
    principal_users_after_create_view_stmt,
)

logger = logging.getLogger(__name__)

SLOW_QUERY_THRESHOLD_SECOND = 0.5

# Query counter for performance monitoring
_query_counter = 0
_query_counter_lock = threading.Lock()


def increment_query_count_sync():
    """Increment the global query counter (synchronous version)."""
    global _query_counter
    with _query_counter_lock:
        _query_counter += 1


def get_query_count() -> int:
    """Get the current query count."""
    global _query_counter
    with _query_counter_lock:
        return _query_counter


async def init_db(db_url: str):
    if db.engine is None:
        db.engine = await init_db_engine(db_url)
        listen_events(db.engine)
    await create_db_and_tables(db.engine)


async def init_db_engine(db_url: str):
    connect_args = {}
    if db_url.startswith("postgresql://"):
        # Probe once to see whether we're talking to openGauss — it presents
        # as PostgreSQL but rejects PG's millisecond-scale value for
        # ``idle_in_transaction_session_timeout``. Skip the setting on openGauss.
        opengauss = await is_opengauss(db_url)

        db_url = re.sub(r'^postgresql://', 'postgresql+asyncpg://', db_url)
        parsed = urlparse(db_url)
        # rewrite the parameters to use asyncpg with custom database schema
        query_params = parse_qs(parsed.query)
        qoptions = query_params.pop('options', None)
        schema_name = None
        if qoptions is not None and len(qoptions) > 0:
            option = qoptions[0]
            if option.startswith('-csearch_path='):
                schema_name = option[len('-csearch_path=') :]
        server_settings = {}
        if schema_name:
            server_settings['search_path'] = schema_name
        if not opengauss and envs.DB_IDLE_IN_TRANSACTION_SESSION_TIMEOUT_SECONDS > 0:
            server_settings['idle_in_transaction_session_timeout'] = str(
                envs.DB_IDLE_IN_TRANSACTION_SESSION_TIMEOUT_SECONDS * 1000
            )
        if server_settings:
            connect_args['server_settings'] = server_settings
        new_parsed = parsed._replace(query={})
        db_url = urlunparse(new_parsed)

    elif db_url.startswith("mysql://"):
        db_url = re.sub(r'^mysql://', 'mysql+asyncmy://', db_url)
    else:
        raise Exception(f"Unsupported database URL: {db_url}")

    engine = create_async_engine(
        db_url,
        echo=envs.DB_ECHO,
        pool_size=envs.DB_POOL_SIZE,
        max_overflow=envs.DB_MAX_OVERFLOW,
        pool_timeout=envs.DB_POOL_TIMEOUT,
        pool_pre_ping=True,
        connect_args=connect_args,
    )
    return engine


async def create_db_and_tables(engine: AsyncEngine):
    async with engine.begin() as conn:
        await conn.run_sync(
            SQLModel.metadata.create_all,
            tables=[
                ApiKey.__table__,
                InferenceBackend.__table__,
                ModelUsage.__table__,
                Model.__table__,
                ModelInstance.__table__,
                SystemLoad.__table__,
                User.__table__,
                Worker.__table__,
                Cluster.__table__,
                CloudCredential.__table__,
                WorkerPool.__table__,
                Credential.__table__,
            ],
        )


def listen_events(engine: AsyncEngine):
    dialect_name = engine.dialect.name

    def _manage_worker_view(target, connection, **kw):
        d = connection.dialect.name
        if d == "postgresql":
            ver = connection.execute(text("SELECT version()")).scalar()
            create_stmt = (
                worker_after_create_view_stmt_opengauss
                if 'openGauss' in (ver or '')
                else worker_after_create_view_stmt_postgres
            )
            connection.execute(text(worker_after_drop_view_stmt_postgres))
            connection.execute(text(create_stmt))
        elif d == "mysql":
            connection.execute(text(worker_after_drop_view_stmt_mysql))
            connection.execute(text(worker_after_create_view_stmt_mysql))
        else:
            connection.execute(text(worker_after_drop_view_stmt_sqlite))
            connection.execute(text(worker_after_create_view_stmt_sqlite))

    event.listen(Worker.metadata, "after_create", _manage_worker_view)
    # ``non_admin_user_models`` references ``principal_users``; drop the
    # dependent view first and create the helper before the dependent.
    event.listen(
        SQLModel.metadata, "after_create", DDL(model_user_after_drop_view_stmt)
    )
    event.listen(
        SQLModel.metadata, "after_create", DDL(principal_users_after_drop_view_stmt)
    )
    event.listen(
        SQLModel.metadata,
        "after_create",
        DDL(principal_users_after_create_view_stmt()),
    )
    event.listen(
        SQLModel.metadata,
        "after_create",
        DDL(model_user_after_create_view_stmt(dialect_name)),
    )

    if engine.dialect.name == "sqlite":
        event.listen(engine.sync_engine, "connect", setup_sqlite_pragmas)
        event.listen(engine.sync_engine, "close", ignore_cancel_on_close)
        if logger.isEnabledFor(logging.DEBUG):
            # Log slow queries on debugging
            event.listen(
                engine.sync_engine, "before_cursor_execute", before_cursor_execute
            )
            event.listen(
                engine.sync_engine, "after_cursor_execute", after_cursor_execute
            )

    # Always count queries for performance monitoring
    event.listen(engine.sync_engine, "after_cursor_execute", count_query)


def count_query(conn, cursor, statement, parameters, context, executemany):
    """Increment the global query counter for each query executed."""
    increment_query_count_sync()
    _maybe_trace_sql(statement)


# Distinct call sites already logged, so a query that fires thousands of times
# is attributed once per caller rather than flooding the log.
_traced_call_sites = set()


def _maybe_trace_sql(statement: str):
    """When GPUSTACK_DB_TRACE_SQL_SUBSTR matches ``statement``, log the Python
    call stack once per distinct call site. Used to attribute a high-frequency
    query seen in DB_ECHO to the code that issues it."""
    substr = envs.DB_TRACE_SQL_SUBSTR
    if not substr or substr not in statement:
        return
    stack = traceback.extract_stack()
    frames = [
        f
        for f in stack
        if "/gpustack/" in f.filename and "server/init_db.py" not in f.filename
    ][-8:]
    sig = tuple((f.filename, f.lineno) for f in frames)
    if sig in _traced_call_sites:
        return
    _traced_call_sites.add(sig)
    logger.info(
        "[DB TRACE] matched %r; new call site:\n%s\nSQL: %s",
        substr,
        "".join(traceback.format_list(frames)),
        statement,
    )


def setup_sqlite_pragmas(conn, record):
    # Enable foreign keys for SQLite, since it's disabled by default
    conn.execute("PRAGMA foreign_keys=ON")

    # Performance tuning
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=normal")
    conn.execute("PRAGMA temp_store=memory")
    conn.execute("PRAGMA mmap_size=30000000000")


def ignore_cancel_on_close(dbapi_connection, connection_record):
    try:
        dbapi_connection.close()
    except asyncio.CancelledError:
        pass


def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    context._query_start_time = time.time()


def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    total = time.time() - context._query_start_time
    if total > SLOW_QUERY_THRESHOLD_SECOND:
        logger.debug(f"[SLOW SQL] {total:.3f}s\nSQL: {statement}\nParams: {parameters}")
