"""GPUStack container-local task supervisor using only the Python standard library."""

import argparse
import codecs
from dataclasses import dataclass, field
import json
import math
import os
import re
import selectors
import signal
import subprocess
import sys
import time
from typing import Dict, List, Optional

_SUPERVISOR_NAME = "gpustack-supervisor"
_LOG_FRAGMENT_SIZE = 4096
_LOG_BUFFER_SIZE = 512 * 1024


@dataclass
class _Task:
    name: str
    argv: List[str]
    env: Dict[str, str]
    process: Optional[subprocess.Popen] = None
    decoder: codecs.IncrementalDecoder = field(
        default_factory=lambda: codecs.getincrementaldecoder("utf-8")(errors="replace")
    )
    buffer: str = ""
    group_confirmed_dead: bool = False


def _parse_task(value: dict) -> _Task:
    if not isinstance(value, dict):
        raise ValueError("Each task must be an object")
    name, argv, env = value.get("name"), value.get("argv"), value.get("env", {})
    if not isinstance(name, str) or not re.fullmatch(r"[\w.-]+", name):
        raise ValueError(
            "Each task requires a name containing letters, digits, _, . or -"
        )
    if (
        not isinstance(argv, list)
        or not argv
        or any(not isinstance(arg, str) or "\0" in arg for arg in argv)
    ):
        raise ValueError(f"Task {name} requires an argv list of strings")
    if not isinstance(env, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in env.items()
    ):
        raise ValueError(f"Task {name} environment values must be strings")
    return _Task(name=name, argv=argv, env=env)


def _load_config(path: str):
    with open(path, encoding="utf-8") as stream:
        config = json.load(stream)
    if not isinstance(config, dict) or not isinstance(config.get("tasks"), list):
        raise ValueError("Configuration requires a tasks list")
    tasks = [_parse_task(item) for item in config["tasks"]]
    if not tasks or len({task.name for task in tasks}) != len(tasks):
        raise ValueError("At least one task with a unique name is required")
    timeouts = []
    for name, default in (("shutdown_timeout", 10), ("cleanup_timeout", 5)):
        timeout = config.get(name, default)
        if (
            type(timeout) not in (int, float)
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError(f"{name} must be a finite positive number")
        timeouts.append(float(timeout))
    return tasks, *timeouts


class GPUStackSupervisor:
    def __init__(
        self, tasks: List[_Task], shutdown_timeout: float, cleanup_timeout: float = 5
    ):
        self.tasks = tasks
        self.shutdown_timeout = shutdown_timeout
        self.cleanup_timeout = cleanup_timeout
        self.stop_signal = 0
        self.selector = selectors.DefaultSelector()
        self.output = bytearray()
        self.stdout_fd = sys.stdout.fileno()
        self.dropped_logs = 0

    def _emit(self, name: str, message: str) -> None:
        block = f"[{name}] {message}\n".encode("utf-8")
        self._flush_logs()
        if len(self.output) + len(block) <= _LOG_BUFFER_SIZE:
            self.output.extend(block)
        else:
            # Container stdout can stall. Dropping output keeps process
            # monitoring and shutdown independent of the log consumer.
            self.dropped_logs += 1
        self._flush_logs()

    def _flush_logs(self) -> None:
        if not self.output:
            return
        try:
            written = os.write(self.stdout_fd, self.output)
        except OSError:
            # Keep bounded pending output for the next nonblocking attempt.
            return
        else:
            del self.output[:written]

    def _stop(self, signum, _frame) -> None:
        if not self.stop_signal:
            self.stop_signal = signum

    def _start_tasks(self) -> bool:
        for task in self.tasks:
            if self.stop_signal:
                break
            env = os.environ.copy()
            env.update(task.env)
            try:
                task.process = subprocess.Popen(
                    task.argv,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    bufsize=0,
                    start_new_session=True,
                )
            except OSError as exc:
                self._emit(_SUPERVISOR_NAME, f"failed to start {task.name}: {exc}")
                return False
            os.set_blocking(task.process.stdout.fileno(), False)
            self.selector.register(task.process.stdout, selectors.EVENT_READ, task)
            self._emit(_SUPERVISOR_NAME, f"started {task.name} pid={task.process.pid}")
        return True

    def _consume_logs(self, task: _Task, text: str, final: bool = False) -> None:
        task.buffer += text
        while task.buffer:
            newline = task.buffer.find("\n")
            if newline < 0 and len(task.buffer) <= _LOG_FRAGMENT_SIZE and not final:
                break
            length = min(
                newline if newline >= 0 else len(task.buffer), _LOG_FRAGMENT_SIZE
            )
            self._emit(task.name, task.buffer[:length].rstrip("\r"))
            consumed = length + 1 if newline == length else length
            task.buffer = task.buffer[consumed:]

    def _read_logs(self, timeout: float) -> None:
        self._flush_logs()
        for key, _ in self.selector.select(timeout):
            task = key.data
            try:
                block = os.read(key.fd, _LOG_FRAGMENT_SIZE)
            except BlockingIOError:
                continue
            self._consume_logs(
                task, task.decoder.decode(block, final=not block), not block
            )
            if not block:
                self.selector.unregister(key.fileobj)
                key.fileobj.close()

    def _monitor(self) -> int:
        while not self.stop_signal:
            for task in self.tasks:
                # Keep the exited leader waitable so its PID cannot be
                # recycled before the final process-group signal.
                status = os.waitid(
                    os.P_PID,
                    task.process.pid,
                    os.WEXITED | os.WNOHANG | os.WNOWAIT,
                )
                code = None
                if status is not None:
                    code = status.si_status
                    if status.si_code != os.CLD_EXITED:
                        code = -code
                if code is not None:
                    if self.stop_signal:
                        return 128 + self.stop_signal
                    self._emit(_SUPERVISOR_NAME, f"{task.name} exited with code {code}")
                    return 1
            self._read_logs(0.05)
        return 128 + self.stop_signal

    def _signal_groups(self, signum: int) -> None:
        for task in self.tasks:
            if task.process is not None and not task.group_confirmed_dead:
                try:
                    # The group can outlive its leader; poll() alone cannot
                    # establish that all descendants have stopped.
                    os.killpg(task.process.pid, signum)
                except ProcessLookupError:
                    task.group_confirmed_dead = True
                except PermissionError:
                    self._emit(
                        _SUPERVISOR_NAME, f"cannot signal {task.name} process group"
                    )

    def _groups_alive(self) -> bool:
        # Reap adopted descendants only after the final group signal and all
        # direct waits. Reaping a leader early would release its reserved PID.
        if all(
            task.process is None or task.process.returncode is not None
            for task in self.tasks
        ):
            for _ in range(128):
                try:
                    pid, _ = os.waitpid(-1, os.WNOHANG)
                except ChildProcessError:
                    break
                if pid == 0:
                    break
        for task in self.tasks:
            if task.process is None or task.group_confirmed_dead:
                continue
            try:
                os.killpg(task.process.pid, 0)
                return True
            except ProcessLookupError:
                task.group_confirmed_dead = True
            except PermissionError:
                return True
        return False

    def _cleanup(self) -> bool:
        started = time.monotonic()
        graceful_deadline = started + self.shutdown_timeout
        deadline = graceful_deadline + self.cleanup_timeout
        self._emit(
            _SUPERVISOR_NAME,
            f"shutdown: graceful={self.shutdown_timeout}s, cleanup={self.cleanup_timeout}s",
        )
        self._signal_groups(self.stop_signal or signal.SIGTERM)
        while time.monotonic() < graceful_deadline and self._groups_alive():
            self._read_logs(min(0.05, max(0, graceful_deadline - time.monotonic())))
        if self._groups_alive():
            self._emit(
                _SUPERVISOR_NAME,
                "shutdown grace expired; killing remaining process groups",
            )
        self._signal_groups(signal.SIGKILL)
        stopped = True
        for task in self.tasks:
            if task.process is None:
                continue
            try:
                task.process.wait(timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                self._emit(
                    _SUPERVISOR_NAME, f"{task.name} did not stop before the deadline"
                )
                stopped = False
        while time.monotonic() < deadline and (
            self.selector.get_map() or self._groups_alive()
        ):
            self._read_logs(min(0.01, max(0, deadline - time.monotonic())))
        if self._groups_alive():
            self._emit(
                _SUPERVISOR_NAME, "process groups remain unconfirmed at the deadline"
            )
            stopped = False
        for key in list(self.selector.get_map().values()):
            key.fileobj.close()
        self.selector.close()
        if self.dropped_logs:
            self._emit(_SUPERVISOR_NAME, f"dropped {self.dropped_logs} log fragments")
        self._flush_logs()
        return stopped

    def run(self) -> int:
        """Start every task, monitor exits, and stop all local process groups."""
        os.set_blocking(self.stdout_fd, False)
        handlers = {
            signum: signal.signal(signum, self._stop)
            for signum in (signal.SIGTERM, signal.SIGINT)
        }
        result = 1
        try:
            if self._start_tasks():
                result = (
                    self._monitor() if not self.stop_signal else 128 + self.stop_signal
                )
        except OSError as exc:
            self._emit(_SUPERVISOR_NAME, f"process supervision failed: {exc}")
        finally:
            if not self._cleanup():
                result = 1
            for signum, handler in handlers.items():
                signal.signal(signum, handler)
        return result


def main() -> int:
    parser = argparse.ArgumentParser(prog=_SUPERVISOR_NAME, description=__doc__)
    parser.add_argument(
        "config", help="JSON file containing tasks and shutdown_timeout"
    )
    args = parser.parse_args()
    if os.name != "posix":
        parser.error("Process supervision requires POSIX process groups")
    if not hasattr(os, "waitid"):
        parser.error("Process supervision requires os.waitid")
    try:
        tasks, timeout, cleanup_timeout = _load_config(args.config)
    except (OSError, ValueError) as exc:
        parser.error(f"Invalid supervisor configuration: {exc}")
    return GPUStackSupervisor(tasks, timeout, cleanup_timeout).run()


if __name__ == "__main__":
    sys.exit(main())
