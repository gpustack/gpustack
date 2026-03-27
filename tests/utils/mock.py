from unittest.mock import AsyncMock, MagicMock


def mock_async_session():
    """Create a mock for async_session that supports async context manager usage
    without producing 'coroutine was never awaited' warnings.

    Using a plain AsyncMock as the session causes warnings because any attribute
    access on AsyncMock implicitly creates a coroutine. Third-party code (pydantic,
    requests, inspect, etc.) accesses attributes without awaiting them, triggering:
        RuntimeWarning: coroutine 'AsyncMockMixin._execute_mock_call' was never awaited

    The fix: use MagicMock as the session (no spurious coroutines on attribute access),
    but explicitly set the async methods that need to be awaitable.
    """
    session = MagicMock()
    # exec returns a Result whose .all()/.first()/etc. are sync — use MagicMock as
    # the return value so that chained sync calls don't spawn unawaited coroutines.
    session.exec = AsyncMock(return_value=MagicMock())
    session.get = AsyncMock()
    session.refresh = AsyncMock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()

    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=session)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm
