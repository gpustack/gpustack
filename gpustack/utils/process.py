import psutil
import asyncio
import logging
import signal
import os
import threading


from gpustack.utils import platform

logger = logging.getLogger(__name__)

threading_stop_event = threading.Event()

termination_signal_handled = False


def add_signal_handlers():
    signal.signal(signal.SIGTERM, handle_termination_signal)


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


async def shutdown_event_loop(sig=None, loop=None):
    logger.debug(f"Received signal: {sig}. Shutting down gracefully...")

    threading_stop_event.set()

    try:
        tasks = [t for t in asyncio.all_tasks(loop) if t is not asyncio.current_task()]
        for task in tasks:
            task.cancel()

        # Wait for all tasks to complete
        await asyncio.gather(*tasks, return_exceptions=True)
    except asyncio.CancelledError:
        pass

    handle_termination_signal(sig=sig)


def handle_termination_signal(sig=None, frame=None):
    """
    Terminate the current process and all its children.
    """
    global termination_signal_handled
    if termination_signal_handled:
        return
    termination_signal_handled = True

    threading_stop_event.set()

    pid = os.getpid()
    terminate_process_tree(pid)


def terminate_process_tree(pid: int):
    try:
        process = psutil.Process(pid)
        children = process.children(recursive=True)

        # Terminate all child processes
        terminate_processes(children)

        # Terminate the parent process
        terminate_process(process)
    except psutil.NoSuchProcess:
        pass
    except Exception as e:
        logger.error(f"Error while terminating process tree: {e}")


def terminate_processes(processes):
    """
    Terminates a list of processes, attempting graceful termination first,
    then forcibly killing remaining ones if necessary.
    """
    for process in processes:
        try:
            process.terminate()
        except psutil.NoSuchProcess:
            continue

    # Wait for processes to terminate and kill if still alive
    _, alive_processes = psutil.wait_procs(processes, timeout=3)
    while alive_processes:
        for process in alive_processes:
            try:
                process.kill()
            except psutil.NoSuchProcess:
                continue
        _, alive_processes = psutil.wait_procs(alive_processes, timeout=1)


def terminate_process(process):
    """
    Terminates a single process, attempting graceful termination first,
    then forcibly killing it if necessary.
    """
    if process.is_running():
        try:
            process.terminate()
            process.wait(timeout=3)
        except psutil.NoSuchProcess:
            pass
        except psutil.TimeoutExpired:
            try:
                process.kill()
                process.wait(timeout=1)
            except (psutil.NoSuchProcess, psutil.TimeoutExpired):
                pass
