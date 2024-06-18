from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    create_async_engine,
)
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

_engine = None


def get_engine():
    return _engine


async def get_session():
    async with AsyncSession(_engine) as session:
        yield session


async def init_db(db_url: str):
    global _engine, _session_maker
    if _engine is None:
        connect_args = {"check_same_thread": False}
        _engine = create_async_engine(db_url, echo=False, connect_args=connect_args)
    await create_db_and_tables(_engine)


async def create_db_and_tables(engine: AsyncEngine):
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
