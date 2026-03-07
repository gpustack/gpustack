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
from gpustack.utils.runtime import is_benchmark_workload

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

    def cleanup_orphan_workloads(self):
        current_instance_names = set()
        model_instances_page = self._clientset.model_instances.list(
            params={"worker_id": str(self._worker_id)}
        )
        if model_instances_page.items:
            for model_instance in model_instances_page.items:
                deployment_metadata = model_instance.get_deployment_metadata(
                    self._worker_id,
                )
                if deployment_metadata:
                    current_instance_names.add(deployment_metadata.name)

        current_benchmark_names = set()
        benchmarks_page = self._clientset.benchmarks.list()
        if benchmarks_page.items:
            for benchmark in benchmarks_page.items:
                deployment_metadata = benchmark.get_deployment_metadata()
                if deployment_metadata:
                    current_benchmark_names.add(deployment_metadata.name)

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
            else:
                should_clean_orphan, _ = network.is_offline(
                    create_at, envs.WORKER_ORPHAN_WORKLOAD_CLEANUP_GRACE_PERIOD
                )
                if w.name not in current_instance_names and should_clean_orphan:
                    delete_workload(w.name)
                    logger.info(
                        f"Deleted orphan workload {w.name}, created at {w.created_at}."
                    )
