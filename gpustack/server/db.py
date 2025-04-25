from contextlib import asynccontextmanager
import os
import re
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    create_async_engine,
)
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy import DDL, event

from gpustack.schemas.api_keys import ApiKey
from gpustack.schemas.model_usage import ModelUsage
from gpustack.schemas.models import Model, ModelInstance
from gpustack.schemas.system_load import SystemLoad
from gpustack.schemas.users import User
from gpustack.schemas.workers import Worker
from gpustack.schemas.stmt import (
    worker_after_create_view_stmt_sqlite,
    worker_after_drop_view_stmt_sqlite,
    worker_after_create_view_stmt_postgres,
    worker_after_drop_view_stmt_postgres,
    worker_after_create_view_stmt_mysql,
    worker_after_drop_view_stmt_mysql,
)

_engine = None

DB_ECHO = os.getenv("GPUSTACK_DB_ECHO", "false").lower() == "true"
DB_POOL_SIZE = int(os.getenv("GPUSTACK_DB_POOL_SIZE", 5))
DB_MAX_OVERFLOW = int(os.getenv("GPUSTACK_DB_MAX_OVERFLOW", 10))
DB_POOL_TIMEOUT = int(os.getenv("GPUSTACK_DB_POOL_TIMEOUT", 30))


def get_engine():
    return _engine


async def get_session():
    async with AsyncSession(_engine) as session:
        yield session


@asynccontextmanager
async def get_session_context():
    async with AsyncSession(_engine) as session:
        yield session


async def init_db(db_url: str):
    global _engine, _session_maker
    if _engine is None:
        connect_args = {}
        if db_url.startswith("sqlite://"):
            connect_args = {"check_same_thread": False}
            # use async driver
            db_url = re.sub(r'^sqlite://', 'sqlite+aiosqlite://', db_url)
        elif db_url.startswith("postgresql://"):
            db_url = re.sub(r'^postgresql://', 'postgresql+asyncpg://', db_url)
        elif db_url.startswith("mysql://"):
            db_url = re.sub(r'^mysql://', 'mysql+asyncmy://', db_url)
        else:
            raise Exception(f"Unsupported database URL: {db_url}")

        _engine = create_async_engine(
            db_url,
            echo=DB_ECHO,
            pool_size=DB_POOL_SIZE,
            max_overflow=DB_MAX_OVERFLOW,
            pool_timeout=DB_POOL_TIMEOUT,
            connect_args=connect_args,
        )
        listen_events(_engine)
    await create_db_and_tables(_engine)


async def create_db_and_tables(engine: AsyncEngine):
    async with engine.begin() as conn:
        await conn.run_sync(
            SQLModel.metadata.create_all,
            tables=[
                ApiKey.__table__,
                ModelUsage.__table__,
                Model.__table__,
                ModelInstance.__table__,
                SystemLoad.__table__,
                User.__table__,
                Worker.__table__,
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

    if engine.dialect.name == "sqlite":
        event.listen(engine.sync_engine, "connect", enable_sqlite_foreign_keys)


def enable_sqlite_foreign_keys(conn, record):
    # Enable foreign keys for SQLite, since it's disabled by default
    conn.execute("PRAGMA foreign_keys=ON")
