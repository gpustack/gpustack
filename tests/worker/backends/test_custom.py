import json
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

from gpustack.schemas.models import DisaggregationSpec, RoleSpec
from gpustack.server.pd_mode_catalog import get_pd_mode, load_pd_modes
from gpustack.utils.pd_router_launch import resolve_router_launch
from gpustack.worker.backends.custom import CustomServer
from gpustack.worker.pd_router import PeerAddress, render_router


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
        env=None,
        log_collection=False,
        trace_endpoint=None,
        return_container=False,
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
        backend._router_launch = resolve_router_launch(backend._model_spec, definition)
        backend._config = SimpleNamespace(
            log_otlp_endpoint="http://otel/v1/logs" if log_collection else None,
            effective_trace_otlp_endpoint=trace_endpoint,
        )
        backend._model_instance = SimpleNamespace(
            id=7, name="test-router", model_id=42, role=role
        )
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
            env=dict(env or {}),
        )
        container = workloads[0].containers[0]
        return container if return_container else container.execution

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


@pytest.mark.parametrize(
    "options, json_logs",
    [
        ({"log_collection": False}, False),
        ({"mode": "vllm-nixl"}, False),
        ({"mode": "sglang-nixl"}, False),
        ({"role": "prefill"}, False),
        ({"image": "gpustack/runner:cuda13.0-sglang0.5.21"}, True),
        ({"image": "mirror.example/team/sglang:custom"}, True),
        ({"router_image": "custom/runner:cuda13.0-sglang0.5.22"}, False),
        ({"model_image": "custom/runner:cuda13.0-sglang0.5.22"}, False),
        ({"router_command": "custom-router"}, False),
        ({"command": ["tini", "--"]}, False),
        ({"arguments": ["smg", "launch"]}, False),
        ({"env": {"PYPI_PACKAGES_INSTALL": "sglang-router==0.2.0"}}, False),
        ({}, True),
        ({"trace_endpoint": "collector:4317", "log_collection": False}, False),
        ({"trace_endpoint": "collector:4317"}, True),
        ({"env": {"RUST_LOG": "smg=warn"}}, True),
        (
            {
                "arguments": [
                    "python3",
                    "-m",
                    "sglang_router.launch_router",
                    "--log-level=warn",
                ]
            },
            True,
        ),
        (
            {
                "arguments": [
                    "python3",
                    "-m",
                    "sglang_router.launch_router",
                    "--json-log",
                ]
            },
            True,
        ),
    ],
)
def test_managed_sglang_router_request_logging(
    create_custom_workload, options, json_logs
):
    image = "gpustack/runner:cuda13.0-sglang0.5.22"
    plan = render_router(
        get_pd_mode("sglang-mooncake"),
        {
            "worker_ip": "10.0.0.1",
            "port": 40006,
            "ports.prometheus": 40007,
            "runner_image": image,
        },
        {
            "prefill": [PeerAddress("10.0.0.2", 40006, {"bootstrap": 8998})],
            "decode": [PeerAddress("10.0.0.3", 40006)],
        },
    )
    defaults = dict(
        image=image,
        arguments=plan.command,
        mode="sglang-mooncake",
        log_collection=True,
    )
    container = create_custom_workload(**{**defaults, **options}, return_container=True)

    env = {item.name: item.value for item in container.envs}
    args = container.execution.args
    assert args.count("--json-log") == int(json_logs)
    # The bundled router preserves the incoming trace only with its exporter off.
    assert {arg.partition("=")[0] for arg in args}.isdisjoint(
        {"--enable-trace", "--otlp-traces-endpoint"}
    )
    default_filter = (
        "smg=info,smg::observability::events=debug"
        if json_logs and "--log-level=warn" not in args
        else None
    )
    assert env.get("RUST_LOG") == options.get("env", {}).get("RUST_LOG", default_filter)


@pytest.mark.parametrize("log_collection", [False, True])
@pytest.mark.parametrize(
    "image",
    [
        "mirror.example/team/runner:cuda13.0-vllm0.32.0",
        "gpustack/runner:cuda13.0-vllm0.31.0",
        "gpustack/runner:cuda13.0-vllm0.33.0",
        "mirror.example/team/vllm:custom",
    ],
)
def test_managed_vllm_router_launches_native_tracing(
    create_custom_workload, monkeypatch, log_collection, image
):
    endpoint = "collector:4317"
    router_args = [
        "--vllm-pd-disaggregation",
        "--enable-igw",
        "--prefill",
        "http://prefill:8000",
        "--decode",
        "http://decode:8000",
    ]
    container = create_custom_workload(
        image=image,
        arguments=["vllm-router", *router_args],
        trace_endpoint=endpoint,
        log_collection=log_collection,
        env={
            "OTEL_RESOURCE_ATTRIBUTES": "deployment.environment=qa,gpustack.model_id=1",
            "OTEL_EXPORTER_OTLP_PROTOCOL": "http/protobuf",
            "OTEL_EXPORTER_OTLP_TRACES_PROTOCOL": "grpc",
        },
        return_container=True,
    )
    env = {item.name: item.value for item in container.envs}
    assert env["OTEL_RESOURCE_ATTRIBUTES"] == (
        "deployment.environment=qa,gpustack.engine_instance_id=7,"
        "gpustack.engine_instance_name=test-router,"
        "gpustack.model_id=42,gpustack.engine_role=router"
    )
    execution = container.execution
    assert execution.args[:2] == ["/bin/sh", "-c"]
    assert execution.args[4:6] == ["python3", "-c"]
    script = execution.args[6]
    assert execution.args[7:] == [endpoint, *router_args]

    parsed_args = SimpleNamespace()
    launched = []

    def parse_router_args(args):
        assert args == router_args
        return parsed_args

    launcher = ModuleType("vllm_router.launch_router")
    launcher.parse_router_args = parse_router_args
    launcher.launch_router = launched.append
    monkeypatch.setitem(sys.modules, launcher.__name__, launcher)
    monkeypatch.setattr(sys, "argv", ["-c", *execution.args[7:]])
    exec(script, {})
    assert launched == [parsed_args]
    assert vars(parsed_args) == dict(enable_trace=True, otlp_traces_endpoint=endpoint)


def test_managed_vllm_router_tracing_falls_back_on_missing_python_entrypoint(
    create_custom_workload, monkeypatch
):
    router_args = ["--vllm-pd-disaggregation", "--enable-igw"]
    container = create_custom_workload(
        image="gpustack/runner:cuda13.0-vllm0.32.0",
        arguments=["vllm-router", *router_args],
        trace_endpoint="collector:4317",
        return_container=True,
    )
    script = container.execution.args[6]
    monkeypatch.setattr(sys, "argv", ["-c", *container.execution.args[7:]])

    # Missing Python tracing helper falls back to the plain CLI via execvp.
    empty_launcher = ModuleType("vllm_router.launch_router")
    monkeypatch.setitem(sys.modules, empty_launcher.__name__, empty_launcher)
    exec_calls = []

    class _ExecInvoked(BaseException):
        pass

    def fake_execvp(file, args):
        exec_calls.append((file, args))
        raise _ExecInvoked()

    monkeypatch.setattr("os.execvp", fake_execvp)
    with pytest.raises(_ExecInvoked):
        exec(script, {})
    assert exec_calls == [("vllm-router", ["vllm-router", *router_args])]

    # Real startup failures inside launch_router must propagate rather than falling back.
    failing_launcher = ModuleType("vllm_router.launch_router")
    failing_launcher.parse_router_args = lambda args: SimpleNamespace()

    def fail_launch(_args):
        raise RuntimeError("router failed to bind")

    failing_launcher.launch_router = fail_launch
    monkeypatch.setitem(sys.modules, failing_launcher.__name__, failing_launcher)
    with pytest.raises(RuntimeError, match="router failed to bind"):
        exec(script, {})


@pytest.mark.parametrize(
    "overrides",
    [
        {"trace_endpoint": None},
        {"router_command": "vllm-router"},
        {"command": ["tini", "--"]},
        {"arguments": ["custom-launcher", "vllm-router"]},
        {"router_image": "custom/runner:cuda13.0-vllm0.32.0"},
        {"model_image": "custom/runner:cuda13.0-vllm0.32.0"},
        {"env": {"PYPI_PACKAGES_INSTALL": "vllm-router==0.1.17"}},
        {"env": {"OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": "custom:4317"}},
        {"env": {"OTEL_EXPORTER_OTLP_ENDPOINT": "custom:4317"}},
        {"env": {"OTEL_EXPORTER_OTLP_TRACES_PROTOCOL": "http/protobuf"}},
        {"env": {"OTEL_EXPORTER_OTLP_PROTOCOL": "http/protobuf"}},
        {"arguments": ["vllm-router", "--otlp-traces-endpoint=custom:4317"]},
        {"arguments": ["vllm-router", "--enable-trace"]},
    ],
)
def test_router_retains_existing_command_when_tracing_is_not_injected(
    create_custom_workload, overrides
):
    options = {
        "image": "gpustack/runner:cuda13.0-vllm0.32.0",
        "trace_endpoint": "collector:4317",
        "arguments": ["vllm-router", "--enable-igw"],
        **overrides,
    }
    container = create_custom_workload(**options, return_container=True)
    assert {item.name: item.value for item in container.envs} == options.get("env", {})
    execution = container.execution
    actual_args = (execution.command or []) + execution.args
    if actual_args[:2] == ["/bin/sh", "-c"]:
        actual_args = actual_args[4:]
    assert actual_args == options.get("command", []) + options["arguments"]


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
