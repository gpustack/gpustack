"""
Log source strategies for unified log streaming.

This module provides a strategy pattern + chain of responsibility approach
for handling different log sources (download logs, main logs, container logs).
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import AsyncGenerator, Callable, List, Optional

from gpustack.utils import file

logger = logging.getLogger(__name__)


async def has_log_content(log_file: Path) -> bool:
    """Check if log file has any actual content.

    Args:
        log_file: Path to log file

    Returns:
        True if file exists and has size > 0
    """
    return await asyncio.to_thread(
        lambda: log_file.exists() and log_file.stat().st_size > 0
    )


class LogSource(ABC):
    """Abstract base class for log sources.

    Each log source knows how to get its files and wait for them if needed.
    This follows the Strategy pattern used in the project
    (similar to LoadBalancingStrategy).
    """

    @abstractmethod
    async def get_files(self) -> List[Path]:
        """Get log files for this source.

        Returns:
            List of log file paths (may be empty)
        """
        pass

    @abstractmethod
    def get_file_pattern(self) -> str:
        """Get the file pattern for this source (for logging purposes).

        Returns:
            Pattern string for identification
        """
        pass

    async def is_valid_source(self) -> bool:
        """Check if this source is valid and should be waited for.

        Override this method to add preconditions for waiting.
        For example, DownloadLogSource returns False if log_path is None.

        Returns:
            True if the source is valid and should be waited for
        """
        return True

    async def get_files_with_log(self) -> List[Path]:
        """Get files with debug logging.

        Returns:
            List of log file paths
        """
        files = await self.get_files()
        if files:
            logger.debug(f"Found files for {self.get_file_pattern()}: {files}")
        return files

    async def wait_for_files(self, timeout: int = 300, **kwargs) -> List[Path]:
        """Wait for log files to appear.

        Uses the project's check_with_retries utility for consistent retry behavior.
        Subclasses should override is_valid_source() instead of this method
        unless they need completely different waiting logic.

        Args:
            timeout: Maximum time to wait in seconds
            **kwargs: Additional arguments for subclass implementations

        Returns:
            List of log file paths
        """
        if not await self.is_valid_source():
            return []

        async def check():
            files = await self.get_files()
            if not files:
                raise FileNotFoundError(
                    f"Log files not found for source: {self.get_file_pattern()}"
                )
            return files

        files = await file.check_with_retries(check, timeout=timeout)
        logger.debug(f"Found files after waiting: {self.get_file_pattern()}")
        return files

    async def wait_for_files_if_needed(
        self,
        follow: bool,
        timeout: int = 300,
        **kwargs,
    ) -> List[Path]:
        """Get files, waiting if necessary in follow mode.

        Args:
            follow: Whether in follow mode (triggers waiting if files not found)
            timeout: Maximum time to wait in seconds
            **kwargs: Additional arguments for wait_for_files

        Returns:
            List of log file paths
        """
        files = await self.get_files_with_log()
        if files:
            return files

        if follow and await self.is_valid_source():
            try:
                files = await self.wait_for_files(timeout=timeout, **kwargs)
            except Exception:
                pass

        return files


class DownloadLogSource(LogSource):
    """Log source for download logs."""

    def __init__(self, log_path: Optional[str]):
        self.log_path = Path(log_path) if log_path else None

    async def get_files(self) -> List[Path]:
        if not self.log_path:
            return []
        if await asyncio.to_thread(self.log_path.exists):
            return [self.log_path]
        return []

    def get_file_pattern(self) -> str:
        return str(self.log_path) if self.log_path else "download_log"

    async def is_valid_source(self) -> bool:
        return self.log_path is not None


class MainLogSource(LogSource):
    """Log source for main (historical) logs."""

    def __init__(
        self,
        log_dir: Path,
        model_instance_id: int,
        restart_count: Optional[int] = None,
        get_all_log_files_fn: Optional[Callable] = None,
    ):
        self.log_dir = log_dir
        self.model_instance_id = model_instance_id
        self.restart_count = restart_count
        self._get_all_log_files = get_all_log_files_fn

    async def get_files(self) -> List[Path]:
        if self._get_all_log_files:
            return await self._get_all_log_files(
                self.log_dir,
                self.model_instance_id,
                restart_count=self.restart_count,
            )
        return []

    def get_file_pattern(self) -> str:
        if self.restart_count is not None:
            return f"{self.model_instance_id}.*.{self.restart_count}.log"
        return f"{self.model_instance_id}.*.log"


class ContainerLogSource(LogSource):
    """Log source for container logs."""

    def __init__(
        self,
        log_dir: Path,
        model_instance_id: int,
        restart_count: Optional[int] = None,
        get_all_log_files_fn: Optional[Callable] = None,
        extract_restart_count_fn: Optional[Callable] = None,
    ):
        self.log_dir = log_dir
        self.model_instance_id = model_instance_id
        self.restart_count = restart_count
        self._get_all_log_files = get_all_log_files_fn
        self._extract_restart_count = extract_restart_count_fn

    async def get_files(self) -> List[Path]:
        if self._get_all_log_files:
            return await self._get_all_log_files(
                self.log_dir,
                self.model_instance_id,
                container=True,
                restart_count=self.restart_count,
            )
        return []

    def get_file_pattern(self) -> str:
        if self.restart_count is not None:
            return f"{self.model_instance_id}.container.{self.restart_count}.log"
        return f"{self.model_instance_id}.container.*.log"

    def _get_expected_file(self, main_log_files: List[Path]) -> Optional[Path]:
        """Infer expected container log file from main log files.

        Args:
            main_log_files: Main log files to infer container log name from

        Returns:
            Expected container log file path, or None if cannot infer
        """
        if not main_log_files or not self._extract_restart_count:
            return None

        expected_restart = self._extract_restart_count(main_log_files[-1].name)
        return self.log_dir / (
            f"{self.model_instance_id}.container.{expected_restart}.log"
        )

    async def wait_for_files(
        self,
        timeout: int = 300,
        main_log_files: Optional[List[Path]] = None,
    ) -> List[Path]:
        """Wait for container log files to appear.

        Container logs need special handling because the expected file name
        depends on the main log's restart count.

        Args:
            timeout: Maximum time to wait in seconds
            main_log_files: Main log files to infer expected container log name

        Returns:
            List of container log file paths
        """
        # First check if files already exist
        files = await self.get_files()
        if files:
            return files

        # Try to infer expected file from main logs
        expected_file = self._get_expected_file(main_log_files)
        if not expected_file:
            return []

        # Reuse base class retry logic with custom check function
        async def check():
            if not await asyncio.to_thread(expected_file.exists):
                raise FileNotFoundError(
                    f"Container log file not found: {expected_file}"
                )
            return [expected_file]

        files = await file.check_with_retries(check, timeout=timeout)
        logger.debug(f"Found container log after waiting: {expected_file}")
        return files

    async def wait_for_files_if_needed(
        self,
        follow: bool,
        timeout: int = 300,
        main_log_files: Optional[List[Path]] = None,
    ) -> List[Path]:
        """Get files, waiting if necessary in follow mode.

        Args:
            follow: Whether in follow mode (triggers waiting if files not found)
            timeout: Maximum time to wait in seconds
            main_log_files: Main log files to infer expected container log name

        Returns:
            List of log file paths
        """
        files = await self.get_files_with_log()
        if files:
            return files

        if follow:
            files = await self.wait_for_files(
                timeout=timeout, main_log_files=main_log_files
            )

        return files

    async def has_content(self) -> bool:
        """Check if any container log file has actual content."""
        files = await self.get_files()
        for f in files:
            if await has_log_content(f):
                return True
        return False


class LogSourceChain:
    """Chain of log sources for sequential log streaming.

    Similar to WorkerFilterChain in the project, this applies sources
    in order and yields log lines from each.
    """

    def __init__(
        self,
        sources: List[LogSource],
        log_generator_fn: Optional[Callable] = None,
    ):
        self.sources = sources
        self._log_generator = log_generator_fn

    async def stream_source(
        self,
        source: LogSource,
        options,
        stop_event: Optional[asyncio.Event] = None,
        follow: bool = True,
    ) -> AsyncGenerator[str, None]:
        """Stream logs from a single source.

        Args:
            source: Log source to stream from
            options: Log options (tail, follow)
            stop_event: Event to signal stopping
            follow: Whether to follow the log file

        Yields:
            Log lines from the source
        """
        from gpustack.worker.logs import LogOptions

        files = await source.get_files()

        if not files and follow:
            # Wait for files to appear in follow mode
            if isinstance(source, ContainerLogSource):
                return  # Container log waiting handled separately
            try:
                files = await source.wait_for_files()
            except Exception:
                logger.debug(f"Timeout waiting for files: {source.get_file_pattern()}")
                return

        if not files:
            return

        for log_file in files:
            if stop_event and stop_event.is_set():
                return

            file_options = LogOptions(
                tail=options.tail if files[-1] == log_file else -1,
                follow=options.follow if files[-1] == log_file else False,
                stop_event=stop_event,
            )

            if self._log_generator:
                async for line in self._log_generator(str(log_file), file_options):
                    if stop_event and stop_event.is_set():
                        return
                    yield line

    async def monitor_container_content(
        self,
        container_source: ContainerLogSource,
        stop_event: asyncio.Event,
        check_interval: float = 1.0,
    ) -> None:
        """Monitor container logs for content and signal stop when found.

        Args:
            container_source: Container log source to monitor
            stop_event: Event to set when container log has content
            check_interval: Time between checks in seconds
        """
        logger.debug("Starting background task to monitor container logs for content")
        initial_has_content = await container_source.has_content()

        while not stop_event.is_set():
            await asyncio.sleep(check_interval)

            if stop_event.is_set():
                return

            current_has_content = await container_source.has_content()

            if current_has_content and not initial_has_content:
                logger.debug("Container log now has content, stopping main log follow")
                stop_event.set()
                return
