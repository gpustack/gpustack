import logging
from typing import Callable
from gpustack_runtime.deployer import (
    list_workloads,
    WorkloadStatusStateEnum,
    delete_workload,
)

from gpustack import envs
from gpustack.client.generated_clientset import ClientSet
from gpustack.utils import network
from gpustack.utils.datetimex import parse_iso8601_to_utc
from gpustack.utils.runtime import is_benchmark_workload, is_cache_service_workload

logger = logging.getLogger(__name__)


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

    def _current_benchmark_names(self) -> set:
        names = set()
        benchmarks_page = self._clientset.benchmarks.list()
        for benchmark in benchmarks_page.items or []:
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
        current_benchmark_names = self._current_benchmark_names()
        current_cache_service_names = self._current_cache_service_instance_names()

        workloads = list_workloads()
        for w in workloads:
            create_at = parse_iso8601_to_utc(w.created_at)
            should_clean_orphan = False
            if is_benchmark_workload(w):
                should_clean_orphan, _ = network.is_offline(
                    create_at,
                    envs.WORKER_ORPHAN_BENCHMARK_WORKLOAD_CLEANUP_GRACE_PERIOD,
                )
                # Clean up benchmark workloads that are:
                # 1. In FAILED or INACTIVE state (regardless of whether they're in current_benchmark_names)
                # 2. Not in current_benchmark_names and past grace period
                if should_clean_orphan and (
                    w.state
                    in [
                        WorkloadStatusStateEnum.FAILED,
                        WorkloadStatusStateEnum.INACTIVE,
                    ]
                    or w.name not in current_benchmark_names
                ):
                    delete_workload(w.name)
                    logger.info(
                        f"Deleted orphan benchmark workload {w.name}, created at {w.created_at}."
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
