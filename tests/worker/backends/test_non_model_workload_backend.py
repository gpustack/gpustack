from types import SimpleNamespace
from unittest.mock import MagicMock

import docker
import pytest

from gpustack_runtime import envs as runtime_envs
from gpustack_runtime.deployer.docker import DockerDeployer, DockerWorkloadPlan
from gpustack_runtime.deployer import kubernetes as kubernetes_runtime

from gpustack.schemas.models import GPUSelector, GPUTypeSelector, Model
from gpustack.worker.backends import custom
from tests.utils.model import new_model_instance

GIB = 1024**3


@pytest.fixture
def frontend(config, monkeypatch):
    backend = custom.CustomServer.__new__(custom.CustomServer)
    backend._model = Model(
        name="frontend",
        source="local_path",
        local_path="/models/tokenizer",
        backend="Custom",
        image_name="test:frontend",
        run_command="python -m frontend",
        env={"GPUSTACK_NON_MODEL_WORKLOAD": "true"},
    )
    backend._model_instance = new_model_instance(1, "frontend", 1, worker_id=1)
    backend._config = config
    backend._worker = SimpleNamespace(id=1)
    backend._get_configured_image = lambda: "test:frontend"
    backend._get_configured_mounts = lambda: []
    backend._get_configured_ports = lambda: []
    backend._host_ipc_enabled = lambda: False
    monkeypatch.setattr(
        "gpustack.utils.runtime.DockerDeployer.is_supported", lambda *_: False
    )
    monkeypatch.setattr(DockerDeployer, "_get_client", lambda *_: None)
    monkeypatch.setattr(
        runtime_envs, "GPUSTACK_RUNTIME_DEPLOY_MIRRORED_DEPLOYMENT", False
    )
    return backend


def capture_workload(monkeypatch, backend, env=None):
    plans = []
    monkeypatch.setattr(custom, "create_workload", plans.append)
    if env is None:
        env = {
            "CUDA_VISIBLE_DEVICES": "-1",
            "NVIDIA_VISIBLE_DEVICES": "all",
            "APP_SETTING": "keep",
        }
    backend._create_workload(
        SimpleNamespace(name="frontend"),
        ["python3"],
        ["-m", "frontend"],
        env,
    )
    assert len(plans) == 1
    return plans[0]


@pytest.mark.parametrize("ram", [None, 0, 4])
def test_frontend_plan_contains_only_cpu_and_matching_ram(monkeypatch, frontend, ram):
    if ram is not None:
        frontend._model.env["GPUSTACK_MODEL_RAM_CLAIM"] = str(ram * GIB)
    # Even a stale instance binding cannot turn this service into a GPU request.
    frontend._model_instance.gpu_indexes = [0]
    plan = capture_workload(monkeypatch, frontend)
    assert plan.instance_type is None
    container = plan.containers[0]
    assert container.resources == {
        "cpu": 2,
        "memory": (2 if ram is None else ram) * GIB,
    }
    assert container.execution.privileged
    env = {item.name: item.value for item in container.envs}
    assert env["CUDA_VISIBLE_DEVICES"] == "-1"
    assert env["NVIDIA_VISIBLE_DEVICES"] == "all"
    assert "ASCEND_VISIBLE_DEVICES" not in env
    assert "TORCH_DEVICE_BACKEND_AUTOLOAD" not in env
    assert env["APP_SETTING"] == "keep"


@pytest.mark.parametrize(
    "device_env",
    [
        {},
        {
            "NVIDIA_VISIBLE_DEVICES": "0",
            "ASCEND_VISIBLE_DEVICES": "1",
            "CUDA_VISIBLE_DEVICES": "1",
            "TORCH_DEVICE_BACKEND_AUTOLOAD": "1",
        },
        {"CUDA_VISIBLE_DEVICES": "-1", "TORCH_DEVICE_BACKEND_AUTOLOAD": "0"},
    ],
)
def test_non_model_preserves_runtime_device_environment(
    monkeypatch, frontend, device_env
):
    plan = capture_workload(monkeypatch, frontend, env=device_env)
    env = {item.name: item.value for item in plan.containers[0].envs}
    assert env == device_env


@pytest.mark.parametrize("mount_all", [False, True])
@pytest.mark.parametrize("selector", ["gpu_selector", "gpu_type_selector"])
def test_worker_refuses_conflicting_gpu_requests(
    monkeypatch, frontend, selector, mount_all
):
    value = (
        GPUSelector(gpu_ids=["worker:cuda:0"])
        if selector == "gpu_selector"
        else GPUTypeSelector(type="test-slice")
    )
    setattr(frontend._model, selector, value)
    with pytest.raises(ValueError, match="not supported"):
        frontend._get_configured_resources(mount_all_devices=mount_all)
    with pytest.raises(ValueError, match="not supported"):
        capture_workload(monkeypatch, frontend)


@pytest.mark.parametrize("mirrored", [False, True])
@pytest.mark.parametrize(
    "device_env",
    [
        {},
        {"CUDA_VISIBLE_DEVICES": "-1", "TORCH_DEVICE_BACKEND_AUTOLOAD": "0"},
        {"CUDA_VISIBLE_DEVICES": "1", "TORCH_DEVICE_BACKEND_AUTOLOAD": "1"},
    ],
)
def test_docker_conversion_preserves_cpu_resources_and_gpu_visibility(
    monkeypatch, frontend, mirrored, device_env
):
    monkeypatch.setattr(
        "gpustack.utils.runtime.DockerDeployer.is_supported", lambda *_: True
    )
    monkeypatch.setattr(
        runtime_envs, "GPUSTACK_RUNTIME_DEPLOY_MIRRORED_DEPLOYMENT", mirrored
    )
    monkeypatch.setattr(runtime_envs, "GPUSTACK_RUNTIME_DEPLOY_MIRRORED_NAME", "worker")
    workload = capture_workload(monkeypatch, frontend, env=device_env)
    assert isinstance(workload, DockerWorkloadPlan)
    assert workload.containers[0].resources == {"cpu": 2, "memory": 2 * GIB}
    workload.validate_and_default()
    device_request = docker.types.DeviceRequest(
        driver="nvidia", count=-1, capabilities=[["gpu"]]
    )
    worker_env = {
        "CUDA_VISIBLE_DEVICES": "0,1",
        "NVIDIA_VISIBLE_DEVICES": "all",
        "TORCH_DEVICE_BACKEND_AUTOLOAD": "1",
    }
    worker = SimpleNamespace(
        short_id="worker",
        image=SimpleNamespace(attrs={"Config": {"Env": []}}),
        attrs={
            "HostConfig": {"DeviceRequests": [device_request]},
            "Config": {"Env": [f"{key}={value}" for key, value in worker_env.items()]},
            "Mounts": [],
        },
    )

    def get_container(name):
        if name == "worker":
            return worker
        raise docker.errors.NotFound("absent")

    deployer = DockerDeployer.__new__(DockerDeployer)
    deployer._client = MagicMock()
    deployer._client.containers.get.side_effect = get_container
    deployer._get_image = lambda image, *_: image
    deployer._mutate_create_options = None
    deployer._prepare_mirrored_deployment()
    deployer._create_containers(workload, {}, SimpleNamespace(id="pause"))
    options = deployer._client.containers.create.call_args.kwargs
    assert options.get("device_requests", []) == ([device_request] if mirrored else [])
    assert not options.get("devices")
    expected_env = {**(worker_env if mirrored else {}), **device_env}
    for key in worker_env:
        assert options["environment"].get(key) == expected_env.get(key)
    assert options["cpu_shares"] == 2048
    assert options["mem_limit"] == options["mem_reservation"] == 2 * GIB
    assert options["privileged"] is True


@pytest.mark.parametrize("injection", ["kdp", "env"])
def test_kubernetes_conversion_requests_no_gpu_resource(
    monkeypatch, frontend, injection
):
    plan = capture_workload(monkeypatch, frontend)
    workload = kubernetes_runtime.KubernetesWorkloadPlan(**plan.__dict__)
    workload.validate_and_default()
    deployer = kubernetes_runtime.KubernetesDeployer.__new__(
        kubernetes_runtime.KubernetesDeployer
    )
    deployer._client = MagicMock()
    deployer._node_name = "worker"
    deployer._mutate_create_pod = lambda pod: pod
    deployer._resolve_resource_injection_policy = lambda: injection
    api = MagicMock()
    api.read_namespaced_pod.return_value = None
    api.create_namespaced_pod.side_effect = lambda **kwargs: kwargs["body"]
    monkeypatch.setattr(
        kubernetes_runtime.kubernetes.client, "CoreV1Api", lambda _: api
    )
    monkeypatch.setattr(
        kubernetes_runtime, "_resolve_runtime_class_name", lambda *_: None
    )
    pod = deployer._create_pod(workload, {})
    container = pod.spec.containers[0]
    assert container.resources.requests == container.resources.limits
    assert set(container.resources.requests) == {"cpu", "memory"}
    assert container.security_context.privileged
    assert not any("instance-type" in key for key in pod.metadata.labels)
    env = {item.name: item.value for item in container.env}
    assert env["CUDA_VISIBLE_DEVICES"] == "-1"
    assert env["NVIDIA_VISIBLE_DEVICES"] == "all"
    assert "TORCH_DEVICE_BACKEND_AUTOLOAD" not in env


def test_disabled_flag_keeps_custom_gpu_plan_and_environment(monkeypatch, frontend):
    frontend._model.env = {"GPUSTACK_NON_MODEL_WORKLOAD": "false"}
    frontend._get_selected_gpu_devices = lambda: [SimpleNamespace(type="cuda", index=1)]
    plan = capture_workload(monkeypatch, frontend)
    container = plan.containers[0]
    key = runtime_envs.GPUSTACK_RUNTIME_DETECT_BACKEND_MAP_RESOURCE_KEY["cuda"]
    assert container.resources == {key: "1"}
    assert container.execution.privileged
    env = {item.name: item.value for item in container.envs}
    assert env["NVIDIA_VISIBLE_DEVICES"] == "all"
