import asyncio
import logging
import signal
import os
import sys
import threading

import psutil

from gpustack.utils import platform

logger = logging.getLogger(__name__)

threading_stop_event = threading.Event()


def add_signal_handlers():
    signal.signal(signal.SIGTERM, shutdown_with_children)


def add_signal_handlers_in_loop():
    if platform.system() == "windows":
        # Windows does not support asyncio signal handlers.
        add_signal_handlers()
        return

    loop = asyncio.get_event_loop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        logger.debug(f"Adding signal handler for {sig}")
        loop.add_signal_handler(
            sig, lambda: asyncio.create_task(shutdown_event_loop(sig, loop))
        )


async def shutdown_event_loop(signal=None, loop=None):
    logger.debug(f"Received signal: {signal}. Shutting down gracefully...")

    threading_stop_event.set()

    try:
        tasks = [t for t in asyncio.all_tasks(loop) if t is not asyncio.current_task()]
        for task in tasks:
            task.cancel()

        # Wait for all tasks to complete
        await asyncio.gather(*tasks, return_exceptions=True)
    except asyncio.CancelledError:
        pass

    shutdown_with_children(signal=signal)


def shutdown_with_children(signal=None, frame=None):
    """
    Terminate the current process and all its children.
    """
    threading_stop_event.set()

    pid = os.getpid()
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return

    children = parent.children(recursive=True)
    for process in children:
        try:
            process.terminate()
        except Exception as e:
            logger.error(f"Error terminating child process {process.pid}: {e}")

    sys.exit(0)
