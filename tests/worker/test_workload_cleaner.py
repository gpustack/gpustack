from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest

from gpustack.schemas.benchmark import BenchmarkStateEnum

from gpustack.schemas.cache_services import CacheServiceInstance
from gpustack.worker.workload_cleaner import WorkloadCleaner
from gpustack_runtime.deployer import WorkloadStatusStateEnum


def _build_cleaner(worker_id: int = 1):
    clientset = MagicMock()
    clientset.model_instances.list.return_value = SimpleNamespace(items=[])
    clientset.benchmarks.list.return_value = SimpleNamespace(items=[])
    clientset.cache_service_instances.list.return_value = SimpleNamespace(items=[])
    cleaner = WorkloadCleaner(lambda: worker_id, lambda: clientset)
    return cleaner, clientset


def _cache_workload(name: str):
    return SimpleNamespace(
        name=name,
        labels={"type": "cache-service"},
        # Far past the orphan grace period.
        created_at="2020-01-01T00:00:00Z",
        state=WorkloadStatusStateEnum.RUNNING,
    )


def test_cleanup_keeps_workloads_of_this_workers_instances():
    """The live set of cache-service workload names comes from this
    worker's instance rows; only workloads outside it are orphans."""
    cleaner, clientset = _build_cleaner(worker_id=1)
    instance = CacheServiceInstance(
        id=11, name="svc-a1b2c", cache_service_id=5, worker_id=1, cluster_id=1
    )
    clientset.cache_service_instances.list.return_value = SimpleNamespace(
        items=[instance]
    )

    workloads = [
        _cache_workload(instance.name),
        _cache_workload("cache-svc-5-i99"),
    ]
    with (
        patch(
            "gpustack.worker.workload_cleaner.list_workloads",
            return_value=workloads,
        ),
        patch("gpustack.worker.workload_cleaner.delete_workload") as delete,
    ):
        cleaner.cleanup_orphan_workloads()

    assert clientset.cache_service_instances.list.call_args[1]["params"] == {
        "worker_id": 1,
        "page": -1,
    }
    delete.assert_called_once_with("cache-svc-5-i99")


def test_cleanup_spares_recent_cache_service_workloads():
    """An unmatched cache-service workload inside the grace period is left
    alone: its instance row may simply not have landed yet."""
    cleaner, _ = _build_cleaner(worker_id=1)

    workload = _cache_workload("cache-svc-5-i99")
    with (
        patch(
            "gpustack.worker.workload_cleaner.list_workloads",
            return_value=[workload],
        ),
        patch(
            "gpustack.worker.workload_cleaner.network.is_offline",
            return_value=(False, None),
        ),
        patch("gpustack.worker.workload_cleaner.delete_workload") as delete,
    ):
        cleaner.cleanup_orphan_workloads()

    delete.assert_not_called()


@pytest.mark.parametrize(
    "kinds",
    [
        [],
        ["model"],
        ["benchmark"],
        ["cache-service"],
        ["model", "benchmark", "cache-service"],
    ],
)
def test_cleanup_only_lists_resources_for_local_workload_types(kinds):
    cleaner, clientset = _build_cleaner()
    workloads = []
    for kind in kinds:
        workload = _cache_workload(kind)
        workload.labels = {"type": kind}
        workloads.append(workload)
    with (
        patch(
            "gpustack.worker.workload_cleaner.list_workloads", return_value=workloads
        ),
        patch("gpustack.worker.workload_cleaner.delete_workload") as delete,
    ):
        cleaner.cleanup_orphan_workloads()

    for kind, resource in (
        ("model", "model_instances"),
        ("benchmark", "benchmarks"),
        ("cache-service", "cache_service_instances"),
    ):
        listing = getattr(clientset, resource).list
        assert listing.call_count == int(kind in kinds)
    assert delete.call_args_list == [call(kind) for kind in kinds]
    if "benchmark" in kinds:
        clientset.benchmarks.list.assert_called_once_with(params={"page": -1})
    if "cache-service" in kinds:
        clientset.cache_service_instances.list.assert_called_once_with(
            params={"worker_id": 1, "page": -1}
        )


@pytest.mark.parametrize("fail_read", [False, True])
def test_cleanup_preserves_live_benchmarks_and_failed_reads(fail_read):
    cleaner, clientset = _build_cleaner()
    workload = _cache_workload("live-benchmark")
    workload.labels = {"type": "benchmark"}
    if fail_read:
        clientset.benchmarks.list.side_effect = RuntimeError("unavailable")
    else:
        clientset.benchmarks.list.return_value = SimpleNamespace(
            items=[
                SimpleNamespace(
                    state=BenchmarkStateEnum.RUNNING,
                    get_deployment_metadata=lambda: SimpleNamespace(name=workload.name),
                )
            ]
        )
    with (
        patch(
            "gpustack.worker.workload_cleaner.list_workloads", return_value=[workload]
        ),
        patch("gpustack.worker.workload_cleaner.delete_workload") as delete,
    ):
        cleaner.cleanup_orphan_workloads()
    delete.assert_not_called()
