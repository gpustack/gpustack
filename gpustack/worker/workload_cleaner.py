import logging
from typing import Callable
from gpustack_runtime.deployer import (
    list_workloads,
    delete_workload,
)

from gpustack import envs
from gpustack.client.generated_clientset import ClientSet
from gpustack.utils import network
from gpustack.utils.datetimex import parse_iso8601_to_utc
from gpustack.schemas.benchmark import BenchmarkStateEnum
from gpustack.utils.runtime import is_benchmark_workload, is_cache_service_workload

logger = logging.getLogger(__name__)


_TERMINAL_BENCHMARK_STATES = frozenset(
    {
        BenchmarkStateEnum.COMPLETED,
        BenchmarkStateEnum.STOPPED,
        BenchmarkStateEnum.ERROR,
    }
)


class WorkloadCleaner:
    @property
    def _worker_id(self) -> int:
        return self._worker_id_getter()

    @property
    def _clientset(self) -> ClientSet:
        return self._clientset_getter()

    _clientset_getter: Callable[[], ClientSet]
    _worker_id_getter: Callable[[], int]

    def __init__(
        self,
        worker_id_getter: Callable[[], int],
        clientset_getter: Callable[[], ClientSet],
    ):
        self._worker_id_getter = worker_id_getter
        self._clientset_getter = clientset_getter

    def _current_model_instance_names(self) -> set:
        names = set()
        model_instances_page = self._clientset.model_instances.list()
        for model_instance in model_instances_page.items or []:
            deployment_metadata = model_instance.get_deployment_metadata(
                self._worker_id,
            )
            if deployment_metadata:
                names.add(deployment_metadata.name)
        return names

    def _live_benchmark_names(self) -> set:
        """Names of benchmarks whose row is still non-terminal.

        Deliberately *not* "every benchmark name": a workload whose row already
        reached COMPLETED/ERROR/STOPPED is garbage and should be swept, while a
        workload whose row is PENDING/QUEUED/RUNNING is a live run and must be
        left alone no matter what the container's momentary state says.

        `page=-1` for the same reason `_current_cache_service_instance_names`
        uses it: a truncated page would make live runs look orphaned.
        """
        names = set()
        benchmarks_page = self._clientset.benchmarks.list(params={"page": -1})
        for benchmark in benchmarks_page.items or []:
            if benchmark.state in _TERMINAL_BENCHMARK_STATES:
                continue
            deployment_metadata = benchmark.get_deployment_metadata()
            if deployment_metadata:
                names.add(deployment_metadata.name)
        return names

    def _current_cache_service_instance_names(self) -> set:
        names = set()
        instances_page = self._clientset.cache_service_instances.list(
            # page=-1 disables pagination: a truncated page would make the
            # cleaner treat live instances as orphans and delete their
            # running cache servers.
            params={"worker_id": self._worker_id, "page": -1}
        )
        for instance in instances_page.items or []:
            deployment_metadata = instance.get_deployment_metadata()
            if deployment_metadata:
                names.add(deployment_metadata.name)
        return names

    def cleanup_orphan_workloads(self):
        current_instance_names = self._current_model_instance_names()
        current_cache_service_names = self._current_cache_service_instance_names()

        # Fail closed on the benchmark list. A worker can reach the server over
        # a lossy link, and a failed read must not surface as "no benchmarks
        # exist" -- every benchmark workload past the 300 s grace period would
        # then look orphaned, and deleting a live run mid-flight loses its last
        # stage and wedges the worker's benchmark queue, because the deletion
        # bypasses the manager's teardown.
        try:
            live_benchmark_names = self._live_benchmark_names()
        except Exception as e:
            logger.warning(
                f"Skipping benchmark orphan cleanup this round; "
                f"could not read the benchmark list: {e}"
            )
            live_benchmark_names = None

        workloads = list_workloads()
        for w in workloads:
            create_at = parse_iso8601_to_utc(w.created_at)
            should_clean_orphan = False
            if is_benchmark_workload(w):
                if live_benchmark_names is None:
                    continue
                should_clean_orphan, _ = network.is_offline(
                    create_at,
                    envs.WORKER_ORPHAN_BENCHMARK_WORKLOAD_CLEANUP_GRACE_PERIOD,
                )
                # Past the grace period and no live row owns it. The row's state
                # is the authority, not the container's: a long multi-stage run
                # can read FAILED/INACTIVE between stages, and deleting on that
                # alone killed runs that were still producing results.
                if should_clean_orphan and w.name not in live_benchmark_names:
                    delete_workload(w.name)
                    logger.info(
                        f"Deleted orphan benchmark workload {w.name}, "
                        f"created at {w.created_at}."
                    )
            elif is_cache_service_workload(w):
                should_clean_orphan, _ = network.is_offline(
                    create_at, envs.WORKER_ORPHAN_WORKLOAD_CLEANUP_GRACE_PERIOD
                )
                if w.name not in current_cache_service_names and should_clean_orphan:
                    delete_workload(w.name)
                    logger.info(
                        f"Deleted orphan cache service workload {w.name}, "
                        f"created at {w.created_at}."
                    )
            else:
                should_clean_orphan, _ = network.is_offline(
                    create_at, envs.WORKER_ORPHAN_WORKLOAD_CLEANUP_GRACE_PERIOD
                )
                if w.name not in current_instance_names and should_clean_orphan:
                    delete_workload(w.name)
                    logger.info(
                        f"Deleted orphan workload {w.name}, created at {w.created_at}."
                    )
