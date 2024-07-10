import logging
import threading
import time
from typing import Callable, Optional


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
            print(f"Error running {func.__name__}: {e}")
        time.sleep(interval)


def run_periodically_in_thread(
    func: Callable,
    interval: float,
    initial_delay: float = 0,
    stop_event: Optional[threading.Event] = None,
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
