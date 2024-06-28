import re
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    create_async_engine,
)
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.schemas.api_keys import ApiKey
from gpustack.schemas.model_usage import ModelUsage
from gpustack.schemas.models import Model, ModelInstance
from gpustack.schemas.system_load import SystemLoad
from gpustack.schemas.users import User
from gpustack.schemas.workers import Worker


_engine = None


def get_engine():
    return _engine


async def get_session():
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
        else:
            raise Exception(f"Unsupported database URL: {db_url}")

        _engine = create_async_engine(db_url, echo=False, connect_args=connect_args)
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
