import asyncio
import logging
from typing import List, Optional

from gpustack.client.worker_filesystem_client import WorkerFilesystemClient
from gpustack.schemas.workers import Worker

logger = logging.getLogger(__name__)


class WorkerSelector:
    """Service for selecting workers based on various criteria."""

    def __init__(self, filesystem_client: WorkerFilesystemClient):
        self._filesystem_client = filesystem_client

    async def find_worker_with_path(
        self,
        workers: List[Worker],
        worker_id: Optional[int] = None,
        path: Optional[str] = None,
    ) -> Optional[Worker]:
        """
        Find a worker that has access to the specified path.

        Args:
            workers: List of workers to search
            worker_id: Optional specific worker ID to use
            path: Optional path to check for existence

        Returns:
            Worker that has access to the path, or None if not found

        Raises:
            ValueError: If worker_id is specified but worker not found
        """
        # If worker_id is specified, use that worker
        if worker_id is not None:
            worker = next((w for w in workers if w.id == worker_id), None)
            if worker is None:
                logger.error(f"Worker with id {worker_id} not found")
                raise ValueError(f"Worker with id {worker_id} not found")

            # If path is specified, check if it exists on this worker
            if path is not None:
                try:
                    exists_response = await self._filesystem_client.path_exists(
                        worker, path
                    )
                    if not exists_response.exists:
                        logger.warning(
                            f"Path {path} does not exist on worker {worker.id}"
                        )
                        return None
                except Exception as e:
                    logger.error(
                        f"Failed to check path {path} on worker {worker.id}: {e}"
                    )
                    return None

            return worker

        # If no worker_id is specified, find a worker that has the path
        if path is not None:
            return await self._find_worker_with_path_concurrent(workers, path)

        return None

    async def _find_worker_with_path_concurrent(
        self,
        workers: List[Worker],
        path: str,
    ) -> Optional[Worker]:
        """
        Concurrently check multiple workers for the existence of a path.

        Args:
            workers: List of workers to check
            path: Path to check for existence

        Returns:
            First worker that has the path, or None if not found
        """

        async def check_worker(worker: Worker) -> tuple[Worker, bool]:
            """Check if a worker has the specified path."""
            try:
                exists_response = await self._filesystem_client.path_exists(
                    worker, path
                )
                return worker, exists_response.exists
            except Exception as e:
                logger.warning(
                    f"Failed to check path {path} on worker {worker.id}: {e}"
                )
                return worker, False

        # Create tasks for all workers
        tasks = [check_worker(worker) for worker in workers]

        # Execute tasks concurrently and get results as they complete
        for completed_task in asyncio.as_completed(tasks):
            worker, exists = await completed_task
            if exists:
                logger.info(f"Found path {path} on worker {worker.id}")
                return worker

        # No worker has the path
        logger.warning(f"Path {path} not found on any worker")
        return None
