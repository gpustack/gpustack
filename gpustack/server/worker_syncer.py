import asyncio
import logging
import aiohttp
from typing import Callable, Optional

from gpustack.schemas.workers import Worker, WorkerStateEnum
from gpustack.server.db import async_session
from gpustack.server.services import WorkerService
from gpustack.server.worker_request import is_worker_reachable
from gpustack import envs

logger = logging.getLogger(__name__)


class WorkerSyncer:
    """
    WorkerSyncer syncs worker status periodically.
    1. Performs worker reachability checks.
    2. Performs readiness checks based on heartbeats.
    """

    def __init__(
        self,
        http_client_getter: Callable[[], Optional[aiohttp.ClientSession]],
        http_client_no_proxy_getter: Callable[[], Optional[aiohttp.ClientSession]],
        interval=15,
        worker_unreachable_timeout=20,
    ):
        self._interval = interval
        self._worker_unreachable_timeout = worker_unreachable_timeout
        self._http_client_getter = http_client_getter
        self._http_client_no_proxy_getter = http_client_no_proxy_getter

        logger.debug(
            f"WorkerSyncer initialized with unreachable check mode: {envs.WORKER_UNREACHABLE_CHECK_MODE}"
        )

    async def start(self):
        client = self._http_client_getter()
        while True:
            await asyncio.sleep(self._interval)
            try:
                client = client or self._http_client_getter()
                if client is None:
                    logger.debug("HTTP client not available, skipping worker sync")
                    continue
                await self._sync_workers_states()
            except Exception as e:
                logger.error(f"Failed to sync workers: {e}")

    async def _sync_workers_states(self):
        """
        Sync workers' states by checking their reachability and readiness.
        """
        async with async_session() as session:
            all_workers = await Worker.all(session)

        if not all_workers:
            return

        if self._should_check_unreachable(len(all_workers)):
            tasks = [
                self._set_worker_unreachable(worker)
                for worker in all_workers
                if not worker.state.is_provisioning
            ]
            await asyncio.gather(*tasks)

        # only the unreachable flag carries over; readiness is recomputed below
        unreachable_by_id = {worker.id: worker.unreachable for worker in all_workers}

        state_to_worker_name = {
            WorkerStateEnum.NOT_READY: [],
            WorkerStateEnum.UNREACHABLE: [],
            WorkerStateEnum.READY: [],
            WorkerStateEnum.MAINTENANCE: [],
        }

        async with async_session() as session:
            for worker in all_workers:
                # bypasses get_by_id's cache, which flush_heartbeats() never invalidates
                to_update_worker = await Worker.one_by_id(session, worker.id)
                if to_update_worker is None:
                    continue

                original_state = to_update_worker.state
                original_state_message = to_update_worker.state_message

                if worker.id in unreachable_by_id:
                    to_update_worker.unreachable = unreachable_by_id[worker.id]
                to_update_worker.compute_state()

                if (
                    to_update_worker.state == original_state
                    and to_update_worker.state_message == original_state_message
                ):
                    continue

                await WorkerService(session).update(to_update_worker)
                if to_update_worker.state in state_to_worker_name:
                    state_to_worker_name[to_update_worker.state].append(
                        to_update_worker.name
                    )

        for state, worker_names in state_to_worker_name.items():
            if worker_names:
                logger.info(f"Marked worker {', '.join(worker_names)} as {state}")

    def _should_check_unreachable(self, worker_count: int) -> bool:
        """
        Determine if unreachable check should be performed based on mode and worker count.

        Args:
            worker_count: Total number of workers

        Returns:
            True if unreachable check should be performed, False otherwise
        """
        mode = envs.WORKER_UNREACHABLE_CHECK_MODE
        auto_threshold = 50  # Auto-disable threshold for worker count

        if mode == "disabled":
            return False
        elif mode == "enabled":
            return True
        elif mode == "auto":
            if worker_count > auto_threshold:
                return False
            return True
        else:
            logger.warning(
                f"Invalid WORKER_UNREACHABLE_CHECK_MODE: {mode}, defaulting to 'auto'"
            )
            # Default to auto behavior
            return worker_count <= auto_threshold

    async def _set_worker_unreachable(self, worker: Worker):
        worker.unreachable = not await is_worker_reachable(
            worker=worker,
            proxy_client=self._http_client_getter(),
            no_proxy_client=self._http_client_no_proxy_getter(),
            timeout_in_second=self._worker_unreachable_timeout,
        )
