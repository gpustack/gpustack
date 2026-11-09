import os
import multiprocessing
import signal
import threading
from unittest.mock import MagicMock, patch

import pytest
import psutil
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
from gpustack.utils.process import (
    add_signal_handlers,
    terminate_multiprocessing_process,
)
from gpustack.worker.provisioning import _provision, run_provisioning


def _wait_for_shutdown(ready, mode="terminate"):
    if mode == "tree":
        child = multiprocessing.get_context("spawn").Process(
            target=_wait_for_shutdown, args=(ready,)
        )
        child.start()
        child.join()
        return
    if mode == "kill":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    elif mode == "handler":
        add_signal_handlers()
    ready.set()
    if mode != "exit":
        threading.Event().wait()


@pytest.mark.parametrize("mode", ["exit", "terminate", "kill", "handler", "tree"])
def test_owned_process_stop_preserves_exit_status_and_cleans_children(mode):
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    process = context.Process(target=_wait_for_shutdown, args=(ready, mode))
    children = []
    try:
        process.start()
        assert ready.wait(15)
        if mode == "exit":
            process.join(timeout=5)
            assert process.exitcode == 0
        elif mode == "tree":
            children = psutil.Process(process.pid).children(recursive=True)
            assert children

        terminate_multiprocessing_process(process)
        assert not process.is_alive()
        assert process.exitcode is not None
        assert not psutil.pid_exists(process.pid)
        assert all(not child.is_running() for child in children)
        exitcode = process.exitcode
        terminate_multiprocessing_process(process)
        assert process.exitcode == exitcode
    finally:
        for child in children:
            if child.is_running():
                child.kill()
        if process.pid is not None:
            process.kill()
            process.join(timeout=5)
        if not process.is_alive():
            process.close()


@pytest.mark.parametrize("error_type", [psutil.AccessDenied, psutil.Error])
@pytest.mark.parametrize("failure_stage", ["lookup", "children", "cleanup"])
def test_owned_process_stop_reaps_parent_after_psutil_error(error_type, failure_stage):
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    process = context.Process(target=_wait_for_shutdown, args=(ready,))
    try:
        process.start()
        assert ready.wait(15)
        with (
            patch("gpustack.utils.process.psutil.Process") as lookup,
            patch("gpustack.utils.process.terminate_processes") as cleanup,
        ):
            failure_operation = {
                "lookup": lookup,
                "children": lookup.return_value.children,
                "cleanup": cleanup,
            }[failure_stage]
            failure_operation.side_effect = error_type()

            terminate_multiprocessing_process(process)

        assert not process.is_alive()
        assert process.exitcode is not None
        assert not psutil.pid_exists(process.pid)
    finally:
        if process.pid is not None:
            process.kill()
            process.join(timeout=5)
        if not process.is_alive():
            process.close()


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
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    process = context.Process(target=_wait_for_shutdown, args=(ready,))
    start = process.start

    def start_and_cancel(self):
        start()
        assert ready.wait(10)
        cancelled.set()

    try:
        with (
            patch(
                "gpustack.worker.provisioning.multiprocessing.get_context"
            ) as factory,
            patch.object(type(process), "start", start_and_cancel),
        ):
            factory.return_value.Process.return_value = process
            assert (
                run_provisioning(MagicMock(), tmp_path / "startup.log", cancelled)
                is None
            )
        with pytest.raises(ValueError, match="closed"):
            process.is_alive()
    finally:
        try:
            process.is_alive()
        except ValueError:
            pass
        else:
            if process.pid is not None:
                process.kill()
                process.join(timeout=5)
            if not process.is_alive():
                process.close()


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
