import asyncio
import logging
from sqlmodel.ext.asyncio.session import AsyncSession
from gpustack.schemas.workers import Worker, WorkerStateEnum
from gpustack.server.db import get_engine
from gpustack.server.services import WorkerService
from gpustack.utils.network import is_url_reachable

logger = logging.getLogger(__name__)


class WorkerSyncer:
    """
    WorkerSyncer syncs worker status periodically.
    """

    def __init__(self, interval=15, worker_unreachable_timeout=20):
        self._engine = get_engine()
        self._interval = interval
        self._worker_unreachable_timeout = worker_unreachable_timeout

    async def start(self):
        while True:
            await asyncio.sleep(self._interval)
            try:
                await self._sync_workers_connectivity()
            except Exception as e:
                logger.error(f"Failed to sync workers: {e}")

    async def _sync_workers_connectivity(self):
        """
        Mark offline workers to not_ready state.
        """
        async with AsyncSession(self._engine) as session:
            workers = await Worker.all(session)
            if not workers:
                return

            tasks = [
                self._check_worker_connectivity(worker, session)
                for worker in workers
                if not worker.state.is_provisioning
            ]
            results = await asyncio.gather(*tasks)

            should_update_workers = []
            state_to_worker_name = {
                WorkerStateEnum.NOT_READY: [],
                WorkerStateEnum.UNREACHABLE: [],
                WorkerStateEnum.READY: [],
                WorkerStateEnum.MAINTENANCE: [],
            }
            for worker in results:
                if worker and worker.state in state_to_worker_name:
                    should_update_workers.append(worker)
                    state_to_worker_name[worker.state].append(worker.name)

            for worker in should_update_workers:
                await WorkerService(session).update(worker)

            for state, worker_names in state_to_worker_name.items():
                if worker_names:
                    logger.info(f"Marked worker {', '.join(worker_names)} as {state}")

    async def _check_worker_connectivity(self, worker: Worker, session: AsyncSession):
        original_worker_unreachable = worker.unreachable
        original_worker_state = worker.state
        original_worker_state_message = worker.state_message

        unreachable = not await self.is_worker_reachable(worker)
        worker = await Worker.one_by_id(session, worker.id)
        worker.unreachable = unreachable
        worker.compute_state()

        if (
            original_worker_unreachable != worker.unreachable
            or original_worker_state != worker.state
            or original_worker_state_message != worker.state_message
        ):
            return worker

        return None

    async def is_worker_reachable(
        self,
        worker: Worker,
    ) -> bool:
        address = worker.advertise_address or worker.ip
        healthz_url = f"http://{address}:{worker.port}/healthz"
        reachable = await is_url_reachable(
            healthz_url,
            self._worker_unreachable_timeout,
        )
        return reachable
