"""Cache deployment archives using the shared workload log layout and readers."""

import logging
from pathlib import Path
import shutil
import threading
from typing import Dict, Set, Tuple

from gpustack.schemas.cache_services import CacheServiceInstance
from gpustack.worker.container_logs import ContainerLogPersister
from gpustack.worker.log_sources import (
    CONTAINER_LOG_NAME,
    MAIN_LOG_NAME,
    find_instance_log_dir,
    main_log_path,
    restart_log_dirs,
)

logger = logging.getLogger(__name__)


def provision_log_path(log_dir: str, instance_id: int) -> Path:
    root = Path(log_dir) / "cache_services"
    restarts = restart_log_dirs(root, instance_id)
    if restarts:
        return restarts[max(restarts)] / MAIN_LOG_NAME
    return main_log_path(root, "", instance_id, 0)


class LogWriterBusyError(RuntimeError):
    """A live writer still owns files needed by the next launch."""


class CacheServiceLogManager(ContainerLogPersister):
    """Own one writer per instance and retain the latest two launch archives."""

    def __init__(self, log_dir: str):
        self.root = Path(log_dir) / "cache_services"
        self._writers: Dict[int, Tuple[threading.Thread, threading.Event, Path]] = {}
        self._lock = threading.RLock()

    def stop(self, instance_id: int) -> bool:
        """Fence writes before reusing or deleting an archive directory."""
        with self._lock:
            record = self._writers.get(instance_id)
            if record is None:
                return True
            thread, stop, _path = record
            stop.set()
            thread.join(timeout=1)
            if thread.is_alive():
                return False
            self._writers.pop(instance_id, None)
            return True

    def prepare(self, instance: CacheServiceInstance) -> Path:
        with self._lock:
            if not self.stop(instance.id):
                raise LogWriterBusyError(
                    "The previous container log writer is still stopping."
                )
            restarts = restart_log_dirs(self.root, instance.id)
            # The crash budget can reset after a healthy period. Disk generations
            # remain monotonic so that previous always identifies the last launch.
            generation = max(restarts, default=-1) + 1
            path = main_log_path(self.root, instance.name, instance.id, generation)
            path.parent.mkdir(parents=True, exist_ok=True)
            for old in sorted(restarts, reverse=True)[1:]:
                try:
                    shutil.rmtree(restarts[old])
                except OSError as e:
                    logger.warning(
                        f"Failed to remove cache log directory {restarts[old]}: {e}"
                    )
            return path

    def ensure(self, instance: CacheServiceInstance, path: Path) -> None:
        """Start or reattach the archive writer without truncating stored output."""
        with self._lock:
            record = self._writers.get(instance.id)
            if record is not None and record[0].is_alive():
                return
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch(exist_ok=True)
            archive = path.with_name(CONTAINER_LOG_NAME)
            stop = threading.Event()

            thread = threading.Thread(
                target=self._persist_container_logs,
                args=(instance.get_deployment_metadata().name, str(archive), stop),
                kwargs={"resume": True},
                daemon=True,
            )
            self._writers[instance.id] = (thread, stop, path)
            thread.start()

    def remove(self, instance_id: int) -> None:
        with self._lock:
            if not self.stop(instance_id):
                return
            directory = find_instance_log_dir(self.root, instance_id)
            if directory is not None:
                shutil.rmtree(directory, ignore_errors=True)

    def archive(self, instance: CacheServiceInstance, path: Path) -> bool:
        """Drain container output before its workload can be removed.

        Return False for a busy writer or failed runtime read. The caller decides
        whether teardown can wait for another attempt.
        """
        with self._lock:
            if not self.stop(instance.id):
                return False
            archive = path.with_name(CONTAINER_LOG_NAME)
            try:
                self._archive_container_logs(
                    instance.get_deployment_metadata().name, str(archive)
                )
                return True
            except Exception:
                logger.exception(
                    "Failed to archive cache instance %s logs", instance.id
                )
                return False

    def tracked_ids(self) -> Set[int]:
        with self._lock:
            ids = set(self._writers)
            if self.root.exists():
                for path in self.root.iterdir():
                    suffix = path.name.rsplit(".", 1)[-1]
                    if path.is_dir() and suffix.isdecimal():
                        ids.add(int(suffix))
            return ids
