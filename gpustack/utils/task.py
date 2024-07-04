import threading
import time
from typing import Callable

stop_event = threading.Event()


def run_periodically(func: Callable[[], None], interval: float) -> None:
    """
    Repeatedly run a function with a given interval.

    Args:
        func: The function to be executed.
        interval: The interval time in seconds.
    """

    while not stop_event.is_set():
        func()
        time.sleep(interval)


def run_periodically_async(func: Callable[[], None], interval: float) -> None:
    """
    Repeatedly run a function asynchronously with a given interval.
    """

    threading.Thread(
        target=run_periodically,
        args=(func, interval),
        daemon=True,
    ).start()
