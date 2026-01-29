import asyncio
import logging
import time
import re
from urllib.parse import urlparse, parse_qs, urlunparse
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    create_async_engine,
)
from sqlmodel import SQLModel
from sqlalchemy import DDL, event

from gpustack import envs
from gpustack.server import db
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
    worker_after_create_view_stmt_mysql,
    worker_after_drop_view_stmt_mysql,
    model_user_after_drop_view_stmt,
    model_user_after_create_view_stmt,
)

logger = logging.getLogger(__name__)

SLOW_QUERY_THRESHOLD_SECOND = 0.5


async def init_db(db_url: str):
    if db.engine is None:
        db.engine = await init_db_engine(db_url)
        listen_events(db.engine)
    await create_db_and_tables(db.engine)


async def init_db_engine(db_url: str):
    connect_args = {}
    if db_url.startswith("postgresql://"):
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
        if schema_name:
            connect_args['server_settings'] = {'search_path': schema_name}
        new_parsed = parsed._replace(query={})
        db_url = urlunparse(new_parsed)

    elif db_url.startswith("mysql://"):
        db_url = re.sub(r'^mysql://', 'mysql+asyncmy://', db_url)
    else:
        raise Exception(f"Unsupported database URL: {db_url}")

    # Use the new init_engine function from db module
    # 使用 db 模块中的新 init_engine 函数
    engine = db.init_engine(
        db_url,
        pool_size=envs.DB_POOL_SIZE,
        max_overflow=envs.DB_MAX_OVERFLOW,
        connect_args=connect_args,
    )
    
    """
    变更：使用新的 init_engine 函数代替直接调用 create_async_engine
    Change: Use the new init_engine function instead of directly calling create_async_engine
    
    原因：
    - 提供连接池管理，优化多 Server 环境下的数据库连接
    - 支持连接池配置，如 pool_size 和 max_overflow
    - 确保数据库引擎的单例模式
    Reasons:
    - Provide connection pool management to optimize database connections in multi-server environments
    - Support connection pool configuration such as pool_size and max_overflow
    - Ensure singleton pattern for database engine
    """
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
    if engine.dialect.name == "postgresql":
        worker_after_drop_view_stmt = worker_after_drop_view_stmt_postgres
        worker_after_create_view_stmt = worker_after_create_view_stmt_postgres
    elif engine.dialect.name == "mysql":
        worker_after_drop_view_stmt = worker_after_drop_view_stmt_mysql
        worker_after_create_view_stmt = worker_after_create_view_stmt_mysql
    else:
        worker_after_drop_view_stmt = worker_after_drop_view_stmt_sqlite
        worker_after_create_view_stmt = worker_after_create_view_stmt_sqlite
    event.listen(Worker.metadata, "after_create", DDL(worker_after_drop_view_stmt))
    event.listen(Worker.metadata, "after_create", DDL(worker_after_create_view_stmt))
    event.listen(
        SQLModel.metadata, "after_create", DDL(model_user_after_drop_view_stmt)
    )
    event.listen(
        SQLModel.metadata,
        "after_create",
        DDL(model_user_after_create_view_stmt(engine.dialect.name)),
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
