import asyncio
from functools import partial
import logging
import threading
import time
from typing import Callable, Optional
from gpustack.utils.process import threading_stop_event


logger = logging.getLogger(__name__)


def run_periodically(
    func: Callable[[], None],
    interval: float = 5,
    initial_delay: float = 0,
    stop_event: Optional[threading.Event] = None,
    *args,
    **kwargs,
) -> None:
    """
    Repeatedly run a function with a given interval.

    Args:
        func: The function to be executed.
        interval: The interval in seconds.
        initial_delay: The initial delay in seconds.
        stop_event: The event to stop the function.
        *args: Positional arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.
    """

    if stop_event is None:
        stop_event = threading.Event()

    if initial_delay > 0:
        time.sleep(initial_delay)

    while not stop_event.is_set():
        try:
            func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error running {func.__name__}: {e}")
            if stop_event.is_set():
                break
        time.sleep(interval)


def run_periodically_in_thread(
    func: Callable,
    interval: float,
    initial_delay: float = 0,
    stop_event: Optional[threading.Event] = threading_stop_event,
    *args,
    **kwargs,
) -> threading.Thread:
    """
    Repeatedly run a function asynchronously with a given interval.

    Args:
        func: The function to be executed.
        interval: The interval time in seconds.
        initial_delay: The initial delay in seconds.
        stop_event: Optional; The event used to stop the periodic execution.
        *args: Positional arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.

    Returns:
        The thread running the periodic function.
    """
    thread = threading.Thread(
        target=run_periodically,
        args=(func, interval, initial_delay, stop_event) + args,
        kwargs=kwargs,
        daemon=True,
    )
    thread.start()
    return thread


async def run_in_thread(sync_func, timeout: Optional[float] = None, *args, **kwargs):
    task = asyncio.to_thread(partial(sync_func, *args, **kwargs))

    if timeout is None:
        return await task

    return await asyncio.wait_for(task, timeout=timeout)
