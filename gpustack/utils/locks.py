import logging
import os
import time
import threading
import json
import socket
from typing import Optional
from filelock import SoftFileLock, Timeout
import fcntl
import errno
from modelscope.hub.utils.utils import model_id_to_group_owner_name

from gpustack.envs import DISABLE_OS_FILELOCK
from gpustack.schemas import ModelFile
from gpustack.schemas.models import SourceEnum

logger = logging.getLogger(__name__)


class HeartbeatSoftFileLock:
    def __init__(
        self,
        lock_path: str,
        ttl_seconds: int = 60,
        heartbeat_seconds: int = 5,
        owner_worker_id: Optional[int] = None,
    ):
        """
        Initialize a heartbeat-backed, lease-based soft file lock.

        Parameters:
        - lock_path: Path to the .lock file used for coordination.
        - ttl_seconds: Only for soft lock, Lease Time-To-Live (in seconds)
          for the lock file. If the
          current holder crashes and heartbeats stop, a subsequent acquirer will
          treat a lock file whose mtime is older than this TTL as stale and may
          delete it to recover.
        - heartbeat_seconds: Interval (in seconds) at which the lock holder
          refreshes the lock file's modification time to signal liveness.
        """
        self._lock_path = lock_path
        self._ttl_seconds = ttl_seconds
        self._heartbeat_seconds = heartbeat_seconds
        self._os_lock: Optional[int] = None
        self._lock = SoftFileLock(lock_path)
        self._hb_stop = threading.Event()
        self._hb_thread: Optional[threading.Thread] = None
        self._owner_worker_id = owner_worker_id
        self._using_soft_lock = False

    def _cleanup_stale_lock(self):
        try:
            if os.path.exists(self._lock_path):
                mtime = os.path.getmtime(self._lock_path)
                if time.time() - mtime > self._ttl_seconds:
                    os.remove(self._lock_path)
        except Exception as e:
            # Swallow cleanup errors to avoid interfering with lock acquisition loop
            logger.warning(f"Failed to cleanup stale lock: {e}")

    def __enter__(self):
        if DISABLE_OS_FILELOCK:
            self._using_soft_lock = True
        else:
            self._acquire_os_lock()

        if not self._using_soft_lock:
            return self

        while True:
            try:
                # Attempt fast acquisition, fall through on contention
                self._lock.acquire(timeout=1)
                break
            except Timeout:
                self._cleanup_stale_lock()
                time.sleep(1)
            except Exception as e:
                raise e

        try:
            info = {
                "worker_id": self._owner_worker_id,
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                "created_at": time.time(),
            }
            with open(self._lock_path, "w") as f:
                json.dump(info, f)
        except Exception as e:
            logger.warning(f"Failed to write lock info: {e}")

        # Start heartbeat to keep lock mtime fresh
        self._hb_thread = threading.Thread(target=self._heartbeat, daemon=True)
        self._hb_thread.start()
        return self

    def _heartbeat(self):
        while not self._hb_stop.is_set():
            try:
                os.utime(self._lock_path, None)
            except Exception as e:
                logger.warning(f"Failed to update lock mtime: {e}")
            self._hb_stop.wait(self._heartbeat_seconds)

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Stop heartbeat
        if self._using_soft_lock:
            self._hb_stop.set()
            if self._hb_thread:
                self._hb_thread.join(timeout=1)
            try:
                self._lock.release()
            except Exception:
                try:
                    if os.path.exists(self._lock_path):
                        os.remove(self._lock_path)
                except Exception as e:
                    logger.warning(f"Failed to release soft lock: {e}")
        if not self._using_soft_lock:
            self._release_os_lock()

    def _acquire_os_lock(self):
        dirpath = os.path.dirname(self._lock_path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        fd = os.open(self._lock_path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.lockf(fd, fcntl.LOCK_EX)
        except OSError as e:
            os.close(fd)
            if e.errno in (errno.ENOSYS, errno.ENOTSUP):
                # File system don't support fcntl
                self._using_soft_lock = True
            else:
                raise e
        else:
            self._os_lock = fd

    def _release_os_lock(self):
        fd = self._os_lock
        try:
            fcntl.lockf(fd, fcntl.LOCK_UN)
            if os.path.exists(self._lock_path):
                os.remove(self._lock_path)
        except Exception as e:
            logger.warning(f"Failed to release OS lock: {e}")
        finally:
            os.close(fd)
            self._os_lock = None


def get_lock_path(cache_dir: str, model_file: ModelFile):
    model_id = None
    if model_file.source == SourceEnum.HUGGING_FACE:
        model_id = model_file.huggingface_repo_id
    elif model_file.source == SourceEnum.MODEL_SCOPE:
        model_id = model_file.model_scope_model_id
    group_or_owner, name = model_id_to_group_owner_name(model_id)
    return os.path.join(
        os.path.join(cache_dir, model_file.source),
        group_or_owner,
        f"{name}.lock",
    )


def read_lock_info(lock_path: str) -> Optional[dict]:
    try:
        if not os.path.exists(lock_path):
            return None
        with open(lock_path, "r") as f:
            return json.load(f)
    except Exception:
        return None
