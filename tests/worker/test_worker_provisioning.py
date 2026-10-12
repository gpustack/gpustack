import os
import threading
from unittest.mock import MagicMock, patch

import pytest
from kubernetes import client
from gpustack_runtime import envs as runtime_envs
from gpustack_runtime.deployer import (
    Container,
    ContainerExecution,
    ContainerResources,
    KubernetesDeployer,
    WorkloadPlan,
)

from gpustack.utils.runtime import transform_workload_plan
from gpustack.worker.provisioning import _provision, run_provisioning


def test_cancelled_launch_does_not_spawn(tmp_path):
    cancelled = threading.Event()
    cancelled.set()
    with patch("gpustack.worker.provisioning.multiprocessing.get_context") as context:
        assert (
            run_provisioning(MagicMock(), tmp_path / "startup.log", cancelled) is None
        )
    context.assert_not_called()


def test_cancellation_terminates_and_reaps_the_child(tmp_path):
    cancelled = threading.Event()
    process = MagicMock()
    process.pid = 12345
    process.start.side_effect = cancelled.set
    process.is_alive.side_effect = [True, False]
    with (
        patch("gpustack.worker.provisioning.multiprocessing.get_context") as context,
        patch("gpustack.worker.provisioning.terminate_process_tree") as terminate,
    ):
        context.return_value.Process.return_value = process
        assert (
            run_provisioning(MagicMock(), tmp_path / "startup.log", cancelled) is None
        )
    terminate.assert_called_once_with(12345)
    process.join.assert_called_once()
    process.close.assert_called_once()


def test_provisioning_failure_is_returned_to_the_status_writer(tmp_path):
    process = MagicMock()
    process.is_alive.return_value = False
    process.exitcode = 1
    with patch("gpustack.worker.provisioning.multiprocessing.get_context") as context:
        context.return_value.Process.return_value = process
        assert (
            run_provisioning(MagicMock(), tmp_path / "startup.log", threading.Event())
            == 1
        )
    process.join.assert_called_once()
    process.close.assert_called_once()


def test_runtime_stdout_is_written_to_the_instance_startup_log(tmp_path):
    path = tmp_path / "startup.log"
    with (
        patch("gpustack.worker.provisioning.add_signal_handlers"),
        patch("gpustack.worker.provisioning.setup_logging"),
        patch("gpustack.worker.provisioning.setup_runtime_logging"),
        patch(
            "gpustack.worker.provisioning.create_workload",
            side_effect=lambda plan: print("Pulling image: 42%", flush=True),
        ),
    ):
        _provision(MagicMock(), str(path), False)
    assert "Pulling image: 42%" in path.read_text()


INJECTION_POLICY = "GPUSTACK_RUNTIME_KUBERNETES_RESOURCE_INJECTION_POLICY"


@pytest.mark.parametrize("policy", ["Auto", "KDP", "Env"])
@pytest.mark.parametrize("gpu_access", [True, False])
def test_cache_plan_uses_env_injection_without_changing_worker_policy(
    tmp_path, policy, gpu_access
):
    """Cache Pods access GPUs without using device-plugin allocations."""
    plan = WorkloadPlan(
        name="cache-svc-1-i1",
        namespace="gpustack-system",
        labels={"type": "cache-service"},
        host_ipc=gpu_access,
        containers=[
            Container(
                name="default",
                image="example.com/lmcache:v1",
                execution=ContainerExecution(
                    privileged=False, command=["lmcache", "server"]
                ),
                resources=ContainerResources(
                    **({"nvidia.com/devices": "all"} if gpu_access else {}),
                    cpu=2,
                    memory="20Gi",
                ),
            )
        ],
    )
    deployer = KubernetesDeployer()
    deployer._client = MagicMock()
    deployer._node_name = "worker01"
    worker = client.V1Pod(
        metadata=client.V1ObjectMeta(name="worker", namespace="gpustack-system"),
        spec=client.V1PodSpec(
            runtime_class_name="nvidia",
            containers=[client.V1Container(name="default")],
        ),
    )
    core = MagicMock()
    core.read_namespaced_pod.return_value = None
    core.create_namespaced_pod.side_effect = lambda namespace, body: body
    # A spawned provisioning process has a fresh runtime configuration cache.
    runtime_envs.__getattr__.cache_clear()
    try:
        with (
            patch.dict(os.environ, {INJECTION_POLICY: policy}),
            patch("gpustack.worker.provisioning.add_signal_handlers"),
            patch("gpustack.worker.provisioning.setup_logging"),
            patch("gpustack.worker.provisioning.setup_runtime_logging"),
            patch(
                "gpustack.utils.runtime.DockerDeployer.is_supported", return_value=False
            ),
            patch.object(KubernetesDeployer, "is_supported", return_value=True),
            patch("gpustack_runtime.deployer._DEPLOYERS", [deployer]),
            patch.object(deployer, "_find_self_pod", return_value=worker),
            patch.object(deployer, "_probe_node_allocatable") as probe,
            patch.object(
                deployer, "get_runtime_visible_devices", return_value=["GPU-0", "GPU-1"]
            ) as devices,
            patch.object(deployer, "count_requested_devices", return_value=2),
            patch.object(
                deployer,
                "map_visible_devices_ordering",
                return_value={"CUDA_DEVICE_ORDER": "PCI_BUS_ID"},
            ),
            patch(
                "gpustack_runtime.deployer.kubernetes.kubernetes.client.CoreV1Api",
                return_value=core,
            ),
        ):
            plan = transform_workload_plan(None, plan)
            _provision(plan, str(tmp_path / "startup.log"), False)
            assert os.environ[INJECTION_POLICY] == policy
            assert (
                runtime_envs.GPUSTACK_RUNTIME_KUBERNETES_RESOURCE_INJECTION_POLICY
                == policy
            )
    finally:
        runtime_envs.__getattr__.cache_clear()

    pod = core.create_namespaced_pod.call_args.kwargs["body"]
    assert pod.spec.runtime_class_name == "nvidia"
    assert pod.spec.host_ipc is gpu_access
    container = pod.spec.containers[0]
    assert container.resources.requests == {"cpu": "2", "memory": "20Gi"}
    assert container.resources.limits == container.resources.requests
    assert container.security_context.privileged is gpu_access
    variables = {e.name: e.value for e in container.env}
    if gpu_access:
        assert variables == {
            "NVIDIA_VISIBLE_DEVICES": "GPU-0,GPU-1",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        }
        devices.assert_called_once_with("NVIDIA_VISIBLE_DEVICES", "plain")
    else:
        assert "NVIDIA_VISIBLE_DEVICES" not in variables
        devices.assert_not_called()
    probe.assert_not_called()
