import asyncio
import inspect
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
            model_info = get_model_info(func, args, kwargs)
            logger.debug(
                f"{func.__name__}{model_info} execution time: {end_time - start_time:.2f} seconds"
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


def get_model_info(func, args, kwargs) -> str:
    """
    Get model info from the function arguments."""
    sig = inspect.signature(func)
    bound_args = sig.bind_partial(*args, **kwargs).arguments

    model = bound_args.get("model")
    model_name = ""
    if model and hasattr(model, "name"):
        model_name = model.name

    if model and hasattr(model, "readable_source"):
        model_name = model.readable_source

    if model_name:
        return f"(model: '{model_name}')"
    return ""
