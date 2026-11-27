"""Own research descendants for cleanup; this is not a security sandbox."""

import ctypes
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import List, Optional


def adopt_children() -> Optional[Path]:
    """Adopt detached descendants on Linux without affecting the controller."""
    if sys.platform != "linux":
        return None
    libc = ctypes.CDLL(None, use_errno=True)
    # PR_SET_CHILD_SUBREAPER keeps double-forked descendants under this supervisor.
    if libc.prctl(36, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "Cannot enable research child subreaper")
    children = Path(f"/proc/self/task/{os.getpid()}/children")
    children.read_text()
    return children


def reap_children(children: Path) -> None:
    """Kill and reap adopted descendants, including those in separate sessions."""
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        for value in children.read_text().split():
            try:
                os.kill(int(value), signal.SIGKILL)
            except ProcessLookupError:
                pass
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        if pid == 0:
            time.sleep(0.01)
    raise RuntimeError("Research descendant cleanup did not complete")


def supervise(argv: List[str]) -> int:
    """Forward the task streams and finish cleanup before reporting completion."""
    children = adopt_children()

    def stop(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    process = None
    try:
        process = subprocess.Popen(argv)
        return process.wait()
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        if children is not None:
            reap_children(children)


if __name__ == "__main__":
    raise SystemExit(supervise(sys.argv[1:]))
