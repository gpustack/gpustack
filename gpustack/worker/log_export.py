import logging
import os
import shutil
import threading
from pathlib import Path
from time import time_ns
from typing import TYPE_CHECKING, Dict, Optional
from uuid import uuid4

from gpustack.worker.log_sources import sanitize_instance_name

if TYPE_CHECKING:
    from gpustack.config.config import Config
    from gpustack.schemas.models import ModelInstance

logger = logging.getLogger(__name__)

ENGINE_LOG_EXPORT_DIR = "engine-export"
_RETENTION_NS = 5_000_000_000
_MAX_RETAINED_ARCHIVES = 32
_MAX_RETAINED_BYTES = 256 * 1024 * 1024


def pod_log_labels(
    config: "Config",
    instance: "ModelInstance",
    worker_id: int,
    role: Optional[str] = None,
) -> Dict[str, str]:
    """Label inference Pods for the Alloy instance in the same Helm release."""
    release = getattr(config, "log_pod_release", None)
    if not release:
        return {}
    return {
        "gpustack.ai/log-release": release,
        "gpustack.ai/log-source": "engine",
        "gpustack.ai/cluster-id": str(instance.cluster_id or 0),
        "gpustack.ai/model-id": str(instance.model_id),
        "gpustack.ai/model-instance-id": str(instance.id),
        "gpustack.ai/worker-id": str(worker_id),
        "gpustack.ai/restart-count": str(instance.restart_count or 0),
        "gpustack.ai/log-role": sanitize_instance_name(
            role or instance.role or "engine"
        ),
    }


class EngineLogExport:
    """Expose registered archive directories to filelog without copying output."""

    def __init__(self, log_dir: Path):
        self._root = log_dir / ENGINE_LOG_EXPORT_DIR
        self._archives = self._root / "archives"
        self._retired = self._root / "retired"
        self._lock = threading.Lock()

    def register(
        self, directory: Path, instance: "ModelInstance", worker_id: int
    ) -> None:
        """Publish trusted identity in the path used by the Collector."""
        with self._lock:
            # The inode distinguishes a replaced directory even when a name,
            # instance ID, and restart counter are reused.
            inode = directory.stat().st_ino
            role = sanitize_instance_name(instance.role or "engine")
            alias = self._archives / (
                f"{instance.cluster_id or 0}.{worker_id}.{instance.model_id}."
                f"{instance.id}.{instance.restart_count or 0}.{role}.{inode}"
            )
            self._archives.mkdir(parents=True, exist_ok=True)
            if not alias.is_symlink() or self._target(alias) != directory.absolute():
                self._link(alias, directory)

    def retain(self, directory: Path) -> None:
        """Keep registered generations briefly visible while their UI paths are removed."""
        with self._lock:
            self._prune()
            if not self._archives.is_dir():
                return
            directory = directory.absolute()
            for alias in self._archives.iterdir():
                if not alias.is_symlink():
                    continue
                source = self._target(alias)
                if source == directory or source.parent == directory:
                    self._retain_archive(alias, source)

    def prune(self) -> None:
        """Expire handoffs during the Worker's existing state sync."""
        with self._lock:
            self._prune()

    @staticmethod
    def _link(alias: Path, directory: Path) -> None:
        # Publish only the final path to the Collector's file glob.
        temporary = alias.parent.parent / f".{uuid4().hex}.tmp"
        try:
            temporary.symlink_to(os.path.relpath(directory, alias.parent))
            temporary.replace(alias)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _target(alias: Path) -> Path:
        return Path(os.path.abspath(alias.parent / alias.readlink()))

    @staticmethod
    def _size(directory: Path) -> int:
        # Serving archives contain files directly under each restart directory.
        size = 0
        for path in directory.iterdir():
            try:
                size += path.lstat().st_size
            except FileNotFoundError:
                # A file can be removed by log rotation during this scan.
                continue
        return size

    def _retain_archive(self, alias: Path, source: Path) -> None:
        try:
            self._retired.mkdir(parents=True, exist_ok=True)
            retained = list(self._retired.iterdir())
            if (
                len(retained) < _MAX_RETAINED_ARCHIVES
                and self._size(source) + sum(self._size(path) for path in retained)
                <= _MAX_RETAINED_BYTES
            ):
                retired = self._retired / f"{time_ns() + _RETENTION_NS}-{alias.name}"
                # Unpublish the old path before freeing it for another generation.
                self._link(alias, retired)
                source.rename(retired)
                return
            logger.warning("Log export retention budget exhausted for %s", alias.name)
        except OSError:
            logger.warning(
                "Cannot retain archive %s for log export", alias.name, exc_info=True
            )
        alias.unlink(missing_ok=True)

    def _prune(self) -> None:
        if self._archives.is_dir():
            for alias in self._archives.iterdir():
                if alias.is_symlink() and not alias.exists():
                    alias.unlink(missing_ok=True)
        if not self._retired.is_dir():
            return
        now = time_ns()
        for directory in self._retired.iterdir():
            deadline, _, name = directory.name.partition("-")
            if not deadline.isdecimal() or int(deadline) > now:
                continue
            alias = self._archives / name
            if alias.is_symlink() and self._target(alias) == directory.absolute():
                alias.unlink(missing_ok=True)
            shutil.rmtree(directory)
