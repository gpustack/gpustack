"""Spawn targets that do not import the test modules in child processes."""

import multiprocessing
import signal
import threading
from multiprocessing.synchronize import Event


def wait_for_shutdown(ready: Event, mode: str = "terminate") -> None:
    """Signal readiness and wait for the process lifecycle test to stop this child."""
    if mode == "tree":
        child = multiprocessing.get_context("spawn").Process(
            target=wait_for_shutdown, args=(ready,)
        )
        child.start()
        child.join()
        return
    if mode == "kill":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    elif mode == "handler":
        from gpustack.utils.process import add_signal_handlers

        add_signal_handlers()
    ready.set()
    if mode != "exit":
        threading.Event().wait()
