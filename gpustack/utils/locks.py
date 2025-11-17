import os
import time
import threading
from typing import Optional
from filelock import SoftFileLock, Timeout


class HeartbeatSoftFileLock:
    def __init__(
        self,
        lock_path: str,
        ttl_seconds: int = 600,
        timeout_seconds: int = 120,
        heartbeat_seconds: int = 5,
    ):
        """
        Initialize a heartbeat-backed, lease-based soft file lock.

        Parameters:
        - lock_path: Path to the .lock file used for coordination.
        - ttl_seconds: Lease Time-To-Live (in seconds) for the lock file. If the
          current holder crashes and heartbeats stop, a subsequent acquirer will
          treat a lock file whose mtime is older than this TTL as stale and may
          delete it to recover.
        - timeout_seconds: Maximum total time (in seconds) to wait for acquiring
          the lock in this call. This is the overall budget; the implementation
          performs short, repeated acquire attempts and periodic stale-lock
          cleanup within this window. If exceeded, a TimeoutError is raised.
        - heartbeat_seconds: Interval (in seconds) at which the lock holder
          refreshes the lock file's modification time to signal liveness.
        """
        self._lock_path = lock_path
        self._ttl_seconds = ttl_seconds
        self._timeout_seconds = timeout_seconds
        self._heartbeat_seconds = heartbeat_seconds
        self._lock = SoftFileLock(lock_path)
        self._hb_stop = threading.Event()
        self._hb_thread: Optional[threading.Thread] = None

    def _cleanup_stale_lock(self):
        try:
            if os.path.exists(self._lock_path):
                mtime = os.path.getmtime(self._lock_path)
                if time.time() - mtime > self._ttl_seconds:
                    os.remove(self._lock_path)
        except Exception:
            # Swallow cleanup errors to avoid interfering with lock acquisition loop
            pass

    def __enter__(self):
        start = time.time()
        while True:
            try:
                # Attempt fast acquisition, fall through on contention
                self._lock.acquire(timeout=1)
                break
            except Timeout:
                self._cleanup_stale_lock()
                if time.time() - start > self._timeout_seconds:
                    raise TimeoutError(f"Failed to acquire lock {self._lock_path}")
                time.sleep(1)

        # Start heartbeat to keep lock mtime fresh
        self._hb_thread = threading.Thread(target=self._heartbeat, daemon=True)
        self._hb_thread.start()
        return self

    def _heartbeat(self):
        while not self._hb_stop.is_set():
            try:
                os.utime(self._lock_path, None)
            except Exception:
                pass
            self._hb_stop.wait(self._heartbeat_seconds)

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Stop heartbeat
        self._hb_stop.set()
        if self._hb_thread:
            self._hb_thread.join(timeout=1)
        # Release underlying lock (deletes the .lock file)
        try:
            self._lock.release()
        except Exception:
            # Defensive: if release fails, try to remove the file
            try:
                if os.path.exists(self._lock_path):
                    os.remove(self._lock_path)
            except Exception:
                pass


def cleanup_stale_lock_files(base_dir: str, ttl_seconds: int):
    """Scan base_dir recursively, remove .lock files older than ttl_seconds."""
    if not base_dir or not os.path.exists(base_dir):
        return
    now = time.time()
    for dirpath, _, filenames in os.walk(base_dir):
        for fname in filenames:
            if not fname.endswith(".lock"):
                continue
            fpath = os.path.join(dirpath, fname)
            try:
                if now - os.path.getmtime(fpath) > ttl_seconds:
                    os.remove(fpath)
            except Exception:
                # Ignore failures, best-effort cleanup
                pass
