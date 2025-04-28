import asyncio
import logging
import signal
import os
import threading
import multiprocessing
import subprocess
import psutil

from gpustack.utils import platform

logger = logging.getLogger(__name__)

threading_stop_event = threading.Event()

termination_signal_handled = False

# Windows requires additional packages for proper process control.
if platform.system() == "windows":
    try:
        import win32api
        import win32con
        import win32job
    except (ModuleNotFoundError, ImportError) as e:
        win32api = None
        win32con = None
        win32job = None
        logger.warning(f"Failed to import win32api, win32con, win32job: {e}")


def _create_windows_job_object():
    """
    Creates a Windows job object that automatically terminates child processes
    when the parent process exits.
    """
    if platform.system() != "windows":
        return None

    if None in (win32api, win32con, win32job):
        logger.warning("Windows modules not available, cannot create job object")
        return None

    try:
        # Borrowed from https://github.com/ray-project/ray/blob/97e028bd23a2522efc3418171c044ed138a0b9ec/python/ray/dashboard/modules/job/job_supervisor.py#L217 noqa: E501
        win32_job_object = win32job.CreateJobObject(None, "")

        extendedInfo = win32job.QueryInformationJobObject(
            win32_job_object, win32job.JobObjectExtendedLimitInformation
        )

        basicLimitInformation = extendedInfo["BasicLimitInformation"]
        basicLimitInformation["LimitFlags"] = (
            win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        )

        win32job.SetInformationJobObject(
            win32_job_object,
            win32job.JobObjectExtendedLimitInformation,
            extendedInfo,
        )

        logger.debug("Windows job object created successfully")
        return win32_job_object
    except win32job.error as e:
        logger.error(f"Windows API error creating job object: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error creating job object: {e}")
        return None


_win32_job_object = _create_windows_job_object()


def _set_process_to_win_job_object(pid: int):
    """
    Set a process to the Windows job object, ensuring it will be terminated
    when the parent process exits.
    """
    if not pid or pid <= 0:
        logger.warning(f"Invalid process ID provided: {pid}")
        return False

    if _win32_job_object is None:
        logger.warning("No job handle available, cannot set process to job object")
        return False

    child_handle = None
    try:
        desired_access = win32con.PROCESS_TERMINATE | win32con.PROCESS_SET_QUOTA
        child_handle = win32api.OpenProcess(
            desired_access,
            False,  # inheritHandle
            pid,
        )

        if not child_handle:
            logger.warning(f"Failed to open process with PID {pid}")
            return False

        win32job.AssignProcessToJobObject(_win32_job_object, child_handle)
        logger.debug(f"Successfully set process {pid} to job object")
        return True
    except Exception as e:
        logger.error(f"Error setting process {pid} to job object: {e}")
        return False
    finally:
        if child_handle:
            try:
                win32api.CloseHandle(child_handle)
            except Exception as e:
                logger.warning(f"Failed to close process handle: {e}")


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


class Popen(subprocess.Popen):
    """Subclass of subprocess.Popen that sets the process to the Windows job object."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _set_process_to_win_job_object(self.pid)


class Process(multiprocessing.Process):
    """Subclass of multiprocessing.Process that sets the process to the Windows job object."""

    def start(self):
        super().start()
        _set_process_to_win_job_object(self.pid)
