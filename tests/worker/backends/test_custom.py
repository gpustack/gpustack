import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from gpustack.schemas.models import DisaggregationSpec, RoleSpec
from gpustack.server.pd_mode_catalog import load_pd_modes
from gpustack.worker.backends.custom import CustomServer


@pytest.fixture
def create_custom_workload(monkeypatch):
    load_pd_modes()

    def create(
        role="router",
        disaggregated=True,
        user_provided=False,
        command=None,
        arguments=None,
        image="gpustack/runner:test",
        router_image=None,
        model_image=None,
        router_command=None,
        router_backend=None,
        mode="vllm-nixl",
        definition=None,
    ):
        backend = CustomServer.__new__(CustomServer)
        backend._model_spec = SimpleNamespace(
            image_name=model_image,
            disaggregation=(DisaggregationSpec(mode=mode) if disaggregated else None),
            roles=[
                RoleSpec(
                    name="router",
                    backend=router_backend,
                    image_name=router_image
                    or ("custom/router:latest" if user_provided else None),
                    run_command=router_command
                    or ("custom-router" if user_provided else None),
                )
            ],
        )
        backend._model = SimpleNamespace(**vars(backend._model_spec))
        backend._model.image_name = (
            router_image or (model_image if router_backend is None else None) or image
        )
        backend._model.backend_version = None
        backend.inference_backend = definition
        if definition:
            from gpustack.utils.pd_router_launch import resolve_router_launch

            backend._router_launch = resolve_router_launch(
                backend._model_spec, definition
            )
        backend._model_instance = SimpleNamespace(role=role)
        backend._get_configured_image = lambda: backend._model.image_name
        backend._get_configured_resources = lambda: {}
        backend._get_configured_mounts = lambda: []
        backend._get_configured_ports = lambda: []
        backend._host_ipc_enabled = lambda: False
        backend._transform_workload_plan = lambda plan: plan
        workloads = []
        monkeypatch.setattr(
            "gpustack.worker.backends.custom.create_workload", workloads.append
        )
        backend._create_workload(
            deployment_metadata=SimpleNamespace(name="test-router"),
            command=command,
            command_args=(
                arguments
                if arguments is not None
                else ["vllm-router", "--model", "a model", "--port", "40006"]
            ),
            env={},
        )
        return workloads[0].containers[0].execution

    return create


@pytest.mark.parametrize("command", [None, ["tini", "--"]])
def test_managed_router_wraps_the_original_command(create_custom_workload, command):
    execution = create_custom_workload(command=command)

    assert execution.command_script is None
    router_command = execution.command or execution.args[:4]
    assert router_command[:2] == ["/bin/sh", "-c"]
    assert router_command[3] == "gpustack-pd-router"
    actual_args = execution.args if command else execution.args[4:]
    assert actual_args == (command or []) + [
        "vllm-router",
        "--model",
        "a model",
        "--port",
        "40006",
    ]


def test_router_keeps_the_image_entrypoint(create_custom_workload):
    execution = create_custom_workload()

    assert execution.command is None
    assert execution.command_script is None
    assert execution.args[:2] == ["/bin/sh", "-c"]


@pytest.mark.parametrize(
    "image",
    [
        "quay.io/gpustack/runner:cuda12.9-vllm0.27.1",
        "registry.example.com:5000/mirror/gpustack/runner@sha256:abcdef",
    ],
)
def test_runner_mirrors_receive_the_router_wrapper(create_custom_workload, image):
    execution = create_custom_workload(image=image)

    assert execution.args[:2] == ["/bin/sh", "-c"]


@pytest.mark.parametrize(
    "image",
    [
        "custom/router:latest",
        "ghcr.io/gpustack/vllm-router:0.1.15",
        "gpustack/gpustack:v2.3.0",
        "registry.example.com:5000/team/renamed-router:latest",
    ],
)
def test_managed_router_wrapping_does_not_depend_on_image_name(
    create_custom_workload, image
):
    execution = create_custom_workload(image=image)

    assert execution.command is None
    assert execution.command_script is None
    assert execution.args[:2] == ["/bin/sh", "-c"]
    assert execution.args[4:] == [
        "vllm-router",
        "--model",
        "a model",
        "--port",
        "40006",
    ]


def test_managed_router_requires_a_command(create_custom_workload):
    with pytest.raises(ValueError, match="Managed PD router requires.*command"):
        create_custom_workload(arguments=[])


@pytest.mark.parametrize(
    "role, disaggregated, user_provided",
    [
        (None, False, False),
        ("router", False, False),
        ("prefill", True, False),
        ("decode", True, False),
        ("router", True, True),
    ],
)
def test_other_custom_workloads_keep_their_execution(
    create_custom_workload, role, disaggregated, user_provided
):
    execution = create_custom_workload(
        role=role,
        disaggregated=disaggregated,
        user_provided=user_provided,
        command=["tini", "--"],
    )

    assert execution.command_script is None
    assert execution.command == ["tini", "--"]
    assert execution.args == [
        "vllm-router",
        "--model",
        "a model",
        "--port",
        "40006",
    ]


def _run_router(execution, soft, hard):
    import resource

    _, inherited_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if inherited_hard != resource.RLIM_INFINITY:
        hard = min(hard, inherited_hard)
    soft = min(soft, hard)
    bootstrap = (
        "import os, resource, sys; "
        "resource.setrlimit(resource.RLIMIT_NOFILE, "
        "(int(sys.argv[1]), int(sys.argv[2]))); "
        "os.execv(sys.argv[3], sys.argv[3:])"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            bootstrap,
            str(soft),
            str(hard),
            *(execution.command or []),
            *execution.args,
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result, (soft, hard)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX resource limits")
@pytest.mark.parametrize("mode", ["vllm-nixl", "sglang-mooncake", "sglang-nixl"])
@pytest.mark.parametrize("explicit_entrypoint", [False, True])
@pytest.mark.parametrize("image_source", ["role", "deployment", "both"])
@pytest.mark.parametrize(
    "router_image",
    [
        "custom/router:distroless",
        "registry.example.com:5000/mirror/gpustack/runner@sha256:abcdef",
    ],
)
def test_image_override_runs_directly_without_a_shell(
    create_custom_workload, mode, explicit_entrypoint, image_source, router_image
):
    arguments = ["a model", "$(exit 23)", "--port", "40006"]
    probe = (
        "import json, resource, sys; "
        "print(json.dumps({'limits': resource.getrlimit(resource.RLIMIT_NOFILE), "
        "'args': sys.argv[1:]}))"
    )
    command = [sys.executable, "-c", probe]
    execution = create_custom_workload(
        mode=mode,
        router_image=router_image if image_source in ("role", "both") else None,
        model_image=router_image if image_source in ("deployment", "both") else None,
        command=command if explicit_entrypoint else None,
        arguments=arguments if explicit_entrypoint else command + arguments,
    )

    assert execution.command_script is None
    assert execution.command == (command if explicit_entrypoint else command[:1])
    assert execution.args == (
        arguments if explicit_entrypoint else command[1:] + arguments
    )

    result, limits = _run_router(execution, 1024, 524288)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "limits": list(limits),
        "args": arguments,
    }


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX resource limits")
@pytest.mark.parametrize("mode", ["vllm-nixl", "sglang-mooncake", "sglang-nixl"])
@pytest.mark.parametrize("explicit_entrypoint", [False, True])
@pytest.mark.parametrize(
    "soft, hard",
    [
        (1024, 524288),
        (65536, 524288),
        (131072, 524288),
    ],
)
def test_router_process_receives_limits_and_literal_arguments(
    create_custom_workload, mode, explicit_entrypoint, soft, hard
):
    arguments = ["a model", "$(exit 23)", "--port", "40006"]
    probe = (
        "import json, resource, sys; "
        "print(json.dumps({'limits': resource.getrlimit(resource.RLIMIT_NOFILE), "
        "'args': sys.argv[1:]}))"
    )
    command = [sys.executable, "-c", probe]
    execution = create_custom_workload(
        mode=mode,
        command=command if explicit_entrypoint else None,
        arguments=arguments if explicit_entrypoint else command + arguments,
    )

    result, (soft, hard) = _run_router(execution, soft, hard)
    expected_soft = max(soft, min(65536, hard))

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "limits": [expected_soft, hard],
        "args": arguments,
    }
    assert f"soft={expected_soft}" in result.stderr
    assert f"hard={hard}" in result.stderr


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX resource limits")
@pytest.mark.parametrize("hard", [1024, 4096])
def test_router_uses_the_available_hard_limit(create_custom_workload, hard):
    probe = (
        "import json, resource; "
        "print(json.dumps(resource.getrlimit(resource.RLIMIT_NOFILE)))"
    )
    execution = create_custom_workload(arguments=[sys.executable, "-c", probe])

    result, (_, hard) = _run_router(execution, 1024, hard)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [hard, hard]
    assert "using the hard limit" in result.stderr


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell")
def test_router_refuses_to_start_if_nofile_adjustment_is_denied(
    create_custom_workload,
):
    script = create_custom_workload().args[2]
    denied = (
        'ulimit() { case "$*" in -Sn) echo 1024;; -Hn) echo 65536;; '
        '*) return 1;; esac; }\n'
    )

    result = subprocess.run(
        ["/bin/sh", "-c", denied + script, "router", "printf", "started"],
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode != 0
    assert result.stdout == ""
    assert "Failed to raise PD router nofile soft limit" in result.stderr


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell")
def test_router_keeps_an_unlimited_soft_limit(create_custom_workload):
    execution = create_custom_workload()
    script = execution.args[2]
    unlimited = 'ulimit() { [ "$#" -eq 1 ] || return 1; printf "%s\\n" unlimited; }\n'

    result = subprocess.run(
        ["/bin/sh", "-c", unlimited + script, "router", "printf", "started"],
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "started"
    assert "soft=unlimited, hard=unlimited" in result.stderr


def test_router_template_argv_is_not_rendered_a_second_time():
    import shlex
    from unittest.mock import MagicMock

    load_pd_modes()
    backend = CustomServer.__new__(CustomServer)
    args = [
        'my-router',
        '--label',
        'literal {{port}} with spaces;$(data)',
        '--port',
        '80',
    ]
    backend._model = SimpleNamespace(
        run_command=shlex.join(args),
        disaggregation=DisaggregationSpec(mode='vllm-nixl'),
        roles=[
            RoleSpec(
                name='router',
                image_name='custom:router',
                run_command='my-router --label {{model_name}} --port {{port}}',
            )
        ],
    )
    backend._model_instance = SimpleNamespace(role='router')
    backend.inference_backend = MagicMock()
    backend._get_selected_gpu_devices = lambda: []
    backend._flatten_backend_param = lambda: []
    backend._get_injected_backend_parameters = lambda *args: []
    assert backend._build_command_args()[0] == args
    backend.inference_backend.replace_command_param.assert_not_called()


@pytest.mark.parametrize("mode", ["vllm-nixl", "custom"])
def test_router_template_overrides_the_images_entrypoint(create_custom_workload, mode):
    execution = create_custom_workload(
        mode=mode,
        router_image="upstream/router:latest",
        router_command="my-router --port {{port}}",
        arguments=["my-router", "--port", "40006"],
    )
    # An application ENTRYPOINT in the image must not receive its own binary
    # name as a positional argument. The template is the full command.
    assert execution.command == ["my-router"]
    assert execution.args == ["--port", "40006"]


@pytest.mark.parametrize("entrypoint", [None, ["tini", "--"]])
def test_independent_router_startup_ignores_the_parent_image(
    create_custom_workload, entrypoint
):
    baseline = create_custom_workload(router_backend="vLLM", command=entrypoint)
    changed = create_custom_workload(
        router_backend="vLLM", command=entrypoint, model_image="unused:parent"
    )
    assert changed.command == baseline.command
    assert changed.args == baseline.args
    assert "gpustack-pd-router" in (changed.command or changed.args)


def test_role_router_argv_uses_the_source_template_for_command_ownership():
    import shlex
    from unittest.mock import MagicMock

    load_pd_modes()
    backend = CustomServer.__new__(CustomServer)
    backend._model_spec = SimpleNamespace(
        run_command="router --label {{model_name}} --peers {{decode_urls | join(',')}}",
        disaggregation=DisaggregationSpec(mode='custom'),
        roles=[
            RoleSpec(
                name='router',
                run_command="router --label {{model_name}} --peers {{decode_urls | join(',')}}",
            )
        ],
    )
    argv = ['router', '--label', 'literal {{port}}', '--peers', 'http://10.0.0.1:80']
    backend._model = SimpleNamespace(
        **{**vars(backend._model_spec), 'run_command': shlex.join(argv)}
    )
    backend._model_instance = SimpleNamespace(role='router')
    backend.inference_backend = MagicMock()
    backend._flatten_backend_param = lambda: []
    backend._get_injected_backend_parameters = lambda *args: []
    assert backend._build_command_args()[0] == argv
    backend.inference_backend.replace_command_param.assert_not_called()


@pytest.mark.parametrize('entrypoint', [None, 'router --entrypoint-flag'])
@pytest.mark.parametrize('mode', ['vllm-nixl', 'custom'])
def test_backend_router_command_preserves_entrypoint_and_needs_no_shell(
    create_custom_workload, entrypoint, mode
):
    from gpustack.schemas.inference_backend import InferenceBackend, VersionConfigDict
    import shlex

    definition = InferenceBackend(
        backend_name='router-custom',
        default_version='1',
        version_configs=VersionConfigDict(
            root={
                '1': {
                    'image_name': 'distroless:router',
                    'run_command': '--listen {{port}}',
                    'entrypoint': entrypoint,
                }
            }
        ),
    )
    execution = create_custom_workload(
        mode=mode,
        router_backend='router-custom',
        definition=definition,
        command=shlex.split(entrypoint) if entrypoint else None,
        arguments=['--listen', '9000'],
    )
    assert execution.command == (shlex.split(entrypoint) if entrypoint else None)
    assert execution.args == ['--listen', '9000']
    assert execution.command_script is None


def test_role_full_command_overrides_selected_backend_entrypoint_without_shell(
    create_custom_workload,
):
    from gpustack.schemas.inference_backend import InferenceBackend, VersionConfigDict

    definition = InferenceBackend(
        backend_name='router-custom',
        default_version='1',
        default_entrypoint='old-router',
        default_run_command='--old-flags',
        version_configs=VersionConfigDict(
            root={'1': {'image_name': 'distroless:router'}}
        ),
    )
    execution = create_custom_workload(
        router_backend='router-custom',
        definition=definition,
        router_command='new-router --port {{port}}',
        arguments=['new-router', '--port', '9000'],
    )
    assert execution.command == ['new-router']
    assert execution.args == ['--port', '9000']


@pytest.mark.parametrize('rendered_command', ['', '   ', "''"])
@pytest.mark.parametrize('image', [None, 'upstream:router'])
def test_router_parameters_cannot_supply_a_missing_executable(rendered_command, image):
    from unittest.mock import MagicMock

    backend = CustomServer.__new__(CustomServer)
    backend._model_spec = SimpleNamespace(
        disaggregation=DisaggregationSpec(mode='vllm-nixl'),
        roles=[RoleSpec(name='router', backend='Custom', image_name=image)],
    )
    backend._model = SimpleNamespace(run_command=rendered_command)
    backend._model_instance = SimpleNamespace(role='router')
    backend.inference_backend = None
    backend._flatten_backend_param = MagicMock(return_value=['--host', '10.0.0.3'])
    backend._get_injected_backend_parameters = lambda *args: []
    with pytest.raises(ValueError, match='Managed PD router requires.*command'):
        backend._build_command_args()
    backend._flatten_backend_param.assert_not_called()


def _ascend_device(index=0, arch_family="Ascend910B4"):
    return SimpleNamespace(
        index=index,
        vendor="ascend",
        type="cann",
        arch_family=arch_family,
    )


def _nvidia_device(index=0):
    return SimpleNamespace(
        index=index,
        vendor="nvidia",
        type="cuda",
        arch_family="Hopper",
    )


def _custom_env_server(
    *,
    gpu_devices,
    gpu_indexes,
    worker_ip="10.0.0.1",
    worker_ifname="eth0",
    worker_id=1,
    model_env=None,
    role=None,
):
    backend = CustomServer.__new__(CustomServer)
    backend._model = SimpleNamespace(env=model_env, disaggregation=None)
    backend._model_spec = None
    backend._model_path = None
    backend._model_instance = SimpleNamespace(
        role=role,
        gpu_indexes=gpu_indexes,
        gpu_type=None,
        distributed_servers=None,
        worker_id=worker_id,
    )
    backend._worker = SimpleNamespace(
        id=worker_id,
        ip=worker_ip,
        ifname=worker_ifname,
        status=SimpleNamespace(gpu_devices=gpu_devices),
    )
    backend._should_disable_cuda_compat = lambda: False
    backend._pd_injection_resolved = False
    backend._pd_injection_cache = None
    return backend


def test_custom_configured_env_ascend_single_node_sets_defaults(monkeypatch):
    monkeypatch.setattr(
        "gpustack.worker.backends.base.filter_env_vars", lambda _env: {}
    )
    server = _custom_env_server(
        gpu_devices=[_ascend_device()],
        gpu_indexes=[0],
    )

    env = server._get_configured_env()

    assert env == {
        "TASK_QUEUE_ENABLE": "1",
        "CPU_AFFINITY_CONF": "1",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "HCCL_CONNECT_TIMEOUT": "7200",
        "HCCL_RDMA_PCIE_DIRECT_POST_NOSTRICT": "TRUE",
        "HCCL_EXEC_TIMEOUT": "0",
        "HCCL_OP_EXPANSION_MODE": "AIV",
    }


def test_custom_configured_env_ascend_310p_omits_910_only_defaults(monkeypatch):
    monkeypatch.setattr(
        "gpustack.worker.backends.base.filter_env_vars", lambda _env: {}
    )
    server = _custom_env_server(
        gpu_devices=[_ascend_device(arch_family="Ascend310P1")],
        gpu_indexes=[0],
    )

    env = server._get_configured_env()

    assert env == {
        "TASK_QUEUE_ENABLE": "1",
        "CPU_AFFINITY_CONF": "1",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "HCCL_CONNECT_TIMEOUT": "7200",
        "HCCL_RDMA_PCIE_DIRECT_POST_NOSTRICT": "TRUE",
    }
    assert "HCCL_EXEC_TIMEOUT" not in env
    assert "HCCL_OP_EXPANSION_MODE" not in env


def test_custom_configured_env_preserves_ascend_user_overrides(monkeypatch):
    monkeypatch.setattr(
        "gpustack.worker.backends.base.filter_env_vars", lambda _env: {}
    )
    server = _custom_env_server(
        gpu_devices=[_ascend_device()],
        gpu_indexes=[0],
        model_env={
            "TASK_QUEUE_ENABLE": "0",
            "HCCL_CONNECT_TIMEOUT": "1800",
            "HCCL_OP_EXPANSION_MODE": "",
        },
    )

    env = server._get_configured_env()

    assert env["TASK_QUEUE_ENABLE"] == "0"
    assert env["HCCL_CONNECT_TIMEOUT"] == "1800"
    assert env["HCCL_OP_EXPANSION_MODE"] == ""
    assert env["CPU_AFFINITY_CONF"] == "1"
    assert env["PYTORCH_NPU_ALLOC_CONF"] == "expandable_segments:True"
    assert env["HCCL_RDMA_PCIE_DIRECT_POST_NOSTRICT"] == "TRUE"
    assert env["HCCL_EXEC_TIMEOUT"] == "0"


def test_custom_configured_env_distributed_ascend_derives_per_worker_env(
    monkeypatch,
):
    monkeypatch.setattr(
        "gpustack.worker.backends.base.filter_env_vars", lambda _env: {}
    )
    worker1 = _custom_env_server(
        gpu_devices=[_ascend_device()],
        gpu_indexes=[0],
        worker_ip="10.0.0.1",
        worker_ifname="eth0",
        worker_id=1,
    )
    worker2 = _custom_env_server(
        gpu_devices=[_ascend_device()],
        gpu_indexes=[0],
        worker_ip="10.0.0.2",
        worker_ifname="bond0",
        worker_id=2,
    )

    env1 = worker1._get_configured_env(is_distributed=True)
    env2 = worker2._get_configured_env(is_distributed=True)

    assert env1["HCCL_IF_IP"] == "10.0.0.1"
    assert env1["HCCL_SOCKET_IFNAME"] == "=eth0"
    assert env1["GLOO_SOCKET_IFNAME"] == "eth0"
    assert env1["TP_SOCKET_IFNAME"] == "eth0"
    assert env1["TASK_QUEUE_ENABLE"] == "1"
    assert "NCCL_SOCKET_IFNAME" not in env1

    assert env2["HCCL_IF_IP"] == "10.0.0.2"
    assert env2["HCCL_SOCKET_IFNAME"] == "=bond0"
    assert env2["GLOO_SOCKET_IFNAME"] == "bond0"
    assert env2["TP_SOCKET_IFNAME"] == "bond0"


def test_custom_configured_env_distributed_ascend_respects_user_overrides(
    monkeypatch,
):
    monkeypatch.setattr(
        "gpustack.worker.backends.base.filter_env_vars", lambda _env: {}
    )
    suppressed = _custom_env_server(
        gpu_devices=[_ascend_device()],
        gpu_indexes=[0],
        model_env={"HCCL_SOCKET_IFNAME": "custom0"},
    )
    suppressed_env = suppressed._get_configured_env(is_distributed=True)
    assert suppressed_env["HCCL_SOCKET_IFNAME"] == "custom0"
    assert "HCCL_IF_IP" not in suppressed_env
    assert "GLOO_SOCKET_IFNAME" not in suppressed_env
    assert "TP_SOCKET_IFNAME" not in suppressed_env

    partial = _custom_env_server(
        gpu_devices=[_ascend_device()],
        gpu_indexes=[0],
        model_env={
            "HCCL_IF_IP": "192.168.1.10",
            "GLOO_SOCKET_IFNAME": "ib0",
        },
    )
    partial_env = partial._get_configured_env(is_distributed=True)
    assert partial_env["HCCL_IF_IP"] == "192.168.1.10"
    assert partial_env["GLOO_SOCKET_IFNAME"] == "ib0"
    assert partial_env["HCCL_SOCKET_IFNAME"] == "=eth0"
    assert partial_env["TP_SOCKET_IFNAME"] == "eth0"


def test_custom_configured_env_distributed_non_ascend_sets_nccl_and_gloo(
    monkeypatch,
):
    monkeypatch.setattr(
        "gpustack.worker.backends.base.filter_env_vars", lambda _env: {}
    )
    server = _custom_env_server(
        gpu_devices=[_nvidia_device()],
        gpu_indexes=[0],
        worker_ip="10.0.0.1",
        worker_ifname="eth0",
    )
    env = server._get_configured_env(is_distributed=True)
    assert env == {
        "NCCL_SOCKET_IFNAME": "=eth0",
        "GLOO_SOCKET_IFNAME": "eth0",
    }

    suppressed = _custom_env_server(
        gpu_devices=[_nvidia_device()],
        gpu_indexes=[0],
        model_env={"NCCL_SOCKET_IFNAME": "custom0"},
    )
    suppressed_env = suppressed._get_configured_env(is_distributed=True)
    assert suppressed_env == {"NCCL_SOCKET_IFNAME": "custom0"}

    partial = _custom_env_server(
        gpu_devices=[_nvidia_device()],
        gpu_indexes=[0],
        model_env={"GLOO_SOCKET_IFNAME": "ib0"},
    )
    partial_env = partial._get_configured_env(is_distributed=True)
    assert partial_env == {
        "GLOO_SOCKET_IFNAME": "ib0",
        "NCCL_SOCKET_IFNAME": "=eth0",
    }


@pytest.mark.parametrize(
    "gpu_devices,gpu_indexes",
    [
        ([_nvidia_device()], [0]),
        ([], []),
    ],
)
def test_custom_configured_env_single_node_non_ascend_or_cpu_sets_no_defaults(
    monkeypatch, gpu_devices, gpu_indexes
):
    monkeypatch.setattr(
        "gpustack.worker.backends.base.filter_env_vars", lambda _env: {}
    )
    server = _custom_env_server(
        gpu_devices=gpu_devices,
        gpu_indexes=gpu_indexes,
        role="router" if not gpu_devices else None,
    )

    assert server._get_configured_env() == {}


@pytest.mark.parametrize("missing_ifname", [None, ""])
def test_custom_configured_env_distributed_skips_missing_ifname(
    monkeypatch, missing_ifname
):
    monkeypatch.setattr(
        "gpustack.worker.backends.base.filter_env_vars", lambda _env: {}
    )
    ascend_server = _custom_env_server(
        gpu_devices=[_ascend_device()],
        gpu_indexes=[0],
        worker_ip="10.0.0.1",
        worker_ifname=missing_ifname,
    )
    ascend_env = ascend_server._get_configured_env(is_distributed=True)
    assert ascend_env["HCCL_IF_IP"] == "10.0.0.1"
    assert "HCCL_SOCKET_IFNAME" not in ascend_env
    assert "GLOO_SOCKET_IFNAME" not in ascend_env
    assert "TP_SOCKET_IFNAME" not in ascend_env

    nvidia_server = _custom_env_server(
        gpu_devices=[_nvidia_device()],
        gpu_indexes=[0],
        worker_ip="10.0.0.1",
        worker_ifname=missing_ifname,
    )
    nvidia_env = nvidia_server._get_configured_env(is_distributed=True)
    assert nvidia_env == {}


@pytest.mark.parametrize("distributed", [False, True])
def test_custom_start_passes_distributed_flag_to_configured_env(distributed):
    recorded = []
    created = []

    server = CustomServer.__new__(CustomServer)
    server._model = SimpleNamespace(backend_version=None)
    server._model_instance = SimpleNamespace(id=1, name="custom-test")
    server.inference_backend = None
    server._get_deployment_metadata = lambda: SimpleNamespace(distributed=distributed)

    def fake_get_configured_env(is_distributed=False):
        recorded.append(is_distributed)
        return {"TEST_ENV": str(is_distributed)}

    server._get_configured_env = fake_get_configured_env
    server._build_command_args = lambda entrypoint=None: (["serve"], [])
    server._update_model_instance = lambda *args, **kwargs: None
    server._create_workload = lambda **kwargs: created.append(kwargs)

    server._start()

    assert recorded == [distributed]
    assert len(created) == 1
    assert created[0]["env"] == {"TEST_ENV": str(distributed)}
