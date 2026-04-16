from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.ext.asyncio import create_async_engine
import logging

logger = logging.getLogger(__name__)

engine = None


def init_engine(database_url, pool_size=10, max_overflow=20, connect_args=None):
    """
    Initialize the database engine with connection pool configuration.

    Args:
        database_url: Database connection URL
        pool_size: Number of connections to keep open in the pool
        max_overflow: Maximum number of connections to allow beyond pool_size
        connect_args: Additional connection arguments
    
    新增：实现连接池管理，优化多 Server 环境下的数据库连接
    Added: Implement connection pool management to optimize database connections in multi-server environments
    """
    global engine
    if engine is None:
        logger.info(f"Initializing database engine with pool_size={pool_size}, max_overflow={max_overflow}")
        engine = create_async_engine(
            database_url,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_pre_ping=True,
            pool_recycle=3600,  # Recycle connections every hour
            echo=False,  # Set to True for debugging
            connect_args=connect_args or {},
        )
    return engine


async def get_session():
    """
    Create a session for database operations.

    Note: expire_on_commit=False is required for async SQLAlchemy to prevent
    lazy loading errors after commit. See: https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html
    
    增强：添加错误处理和事务管理
    Enhanced: Add error handling and transaction management
    """
    if engine is None:
        raise RuntimeError("Database not initialized. Call init_engine() first.")
    
    async with AsyncSession(engine, expire_on_commit=False) as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            logger.error(f"Database session error: {e}")
            await session.rollback()
            raise


async def async_session() -> AsyncSession:
    """
    Get an AsyncSession with default expire_on_commit=False.

    Returns:
        AsyncSession: Configured session with expire_on_commit=False
    """
    if engine is None:
        raise RuntimeError("Database not initialized. Call init_engine() first.")

    return AsyncSession(engine, expire_on_commit=False)


async def close_engine():
    """
    Close the database engine and connection pool.
    
    新增：添加资源清理功能
    Added: Add resource cleanup functionality
    """
    global engine
    if engine is not None:
        logger.info("Closing database engine")
        await engine.dispose()
        engine = None
