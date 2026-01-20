from sqlmodel.ext.asyncio.session import AsyncSession

engine = None


async def get_session():
    """
    Create a session for database operations.

    Note: expire_on_commit=False is required for async SQLAlchemy to prevent
    lazy loading errors after commit. See: https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html
    """
    async with AsyncSession(engine, expire_on_commit=False) as session:
        yield session


def async_session() -> AsyncSession:
    """
    Get an AsyncSession with default expire_on_commit=False.

    Returns:
        AsyncSession: Configured session with expire_on_commit=False
    """
    if engine is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")

    return AsyncSession(engine, expire_on_commit=False)
