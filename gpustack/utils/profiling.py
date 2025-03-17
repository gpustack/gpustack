import asyncio
import logging
import time


logger = logging.getLogger(__name__)


def time_decorator(func):
    """
    A decorator that logs the execution time of a function.
    """

    if asyncio.iscoroutinefunction(func):

        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            result = await func(*args, **kwargs)
            end_time = time.time()
            logger.debug(
                f"{func.__name__} execution time: {end_time - start_time:.2f} seconds"
            )
            return result

        return async_wrapper
    else:

        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            logger.debug(
                f"{func.__name__} execution time: {end_time - start_time} seconds"
            )
            return result

        return sync_wrapper
