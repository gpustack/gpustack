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
        mode="vllm-nixl",
    ):
        backend = CustomServer.__new__(CustomServer)
        backend._model_spec = SimpleNamespace(
            image_name=model_image,
            disaggregation=(DisaggregationSpec(mode=mode) if disaggregated else None),
            roles=[
                RoleSpec(
                    name="router",
                    image_name=router_image
                    or ("custom/router:latest" if user_provided else None),
                    run_command="custom-router" if user_provided else None,
                )
            ],
        )
        backend._model = SimpleNamespace(**vars(backend._model_spec))
        backend._model.image_name = router_image or model_image or image
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
    assert execution.command == (command if explicit_entrypoint else None)
    assert execution.args == (arguments if explicit_entrypoint else command + arguments)

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
