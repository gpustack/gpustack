import types

import pytest

from gpustack.schemas.inference_backend import (
    InferenceBackend,
    ParameterFormatEnum,
    VersionConfig,
    VersionConfigDict,
)
from gpustack.schemas.models import BackendEnum
from gpustack.utils.config import apply_registry_override_to_image
from gpustack.worker.backends.custom import CustomServer
from gpustack.worker.backends.sglang import (
    SGLangServer,
    get_access_log_arguments as get_sglang_access_log_arguments,
    get_cache_report_arguments as get_sglang_cache_report_arguments,
)
from gpustack.worker.backends.vllm import (
    VLLMServer,
    get_access_log_arguments as get_vllm_access_log_arguments,
    get_cache_report_arguments as get_vllm_cache_report_arguments,
)
from gpustack.worker.backends.vox_box import VoxBoxServer


@pytest.mark.parametrize(
    "image_name, container_registry, expect_image_name, fallback_registry",
    [
        (
            "ghcr.io/ggml-org/llama.cpp:server",
            "test-registry.io",
            "ghcr.io/ggml-org/llama.cpp:server",
            None,
        ),
        (
            "gpustack/runner:cuda12.8-vllm0.10.2",
            "test-registry.io",
            "test-registry.io/gpustack/runner:cuda12.8-vllm0.10.2",
            None,
        ),
        (
            "foo/bar",
            "test-registry.io",
            "test-registry.io/foo/bar",
            None,
        ),
        ("ubuntu:24.04", "test-registry.io", "test-registry.io/ubuntu:24.04", None),
        (
            "gpustack/runner:cuda12.8-vllm0.10.2",
            None,
            "quay.io/gpustack/runner:cuda12.8-vllm0.10.2",
            "quay.io",
        ),
        (
            "lmsysorg/sglang:v0.5.5",
            "",
            "lmsysorg/sglang:v0.5.5",
            None,
        ),
    ],
)
@pytest.mark.asyncio
async def test_apply_registry_override(
    image_name,
    container_registry,
    expect_image_name,
    fallback_registry,
    monkeypatch,
):
    backend = CustomServer.__new__(CustomServer)
    # CustomServer inherits _apply_registry_override from InferenceServer,
    # and _apply_registry_override accesses self._config.system_default_container_registry.
    # Since we constructed the instance via __new__ (without __init__),
    # the _config attribute does not exist. We attach a minimal stub config here.
    backend._config = types.SimpleNamespace(
        system_default_container_registry=container_registry,
    )
    backend._fallback_registry = fallback_registry

    assert (
        apply_registry_override_to_image(
            backend._config, image_name, backend._fallback_registry
        )
        == expect_image_name
    )

    if container_registry:
        backend._config = types.SimpleNamespace(system_default_container_registry=None)
        assert (
            apply_registry_override_to_image(
                backend._config, image_name, backend._fallback_registry
            )
            == image_name
        )


@pytest.mark.parametrize(
    "backend_parameters, expected",
    [
        (
            ["--ctx-size 1024"],
            ["--ctx-size", "1024"],
        ),
        (
            ["--served-model-name foo"],
            ["--served-model-name", "foo"],
        ),
        (
            ['--served-model-name "foo bar"'],
            ["--served-model-name", "foo bar"],
        ),
        (
            ['--arg1', '--arg2 "val with spaces"'],
            ['--arg1', '--arg2', 'val with spaces'],
        ),
        (
            ['--arg1 "val with spaces"', '--arg2="val with spaces"'],
            ['--arg1', 'val with spaces', '--arg2="val with spaces"'],
        ),
        (
            [
                """--hf-overrides '{"architectures": ["NewModel"]}'""",
                """--hf-overrides={"architectures": ["NewModel"]}""",
            ],
            [
                '--hf-overrides',
                '{"architectures": ["NewModel"]}',
                """--hf-overrides={"architectures": ["NewModel"]}""",
            ],
        ),
        # Test cases for whitespace handling
        (
            [" --ctx-size=1024"],
            ["--ctx-size=1024"],
        ),
        (
            ["--ctx-size =1024"],
            ["--ctx-size=1024"],
        ),
        (
            ["  --ctx-size  =1024"],
            ["--ctx-size=1024"],
        ),
        (
            ["--ctx-size  =  1024"],
            ["--ctx-size=1024"],
        ),
        (
            ["  --ctx-size 1024"],
            ["--ctx-size", "1024"],
        ),
        (
            [" --max-model-len=8192"],
            ["--max-model-len=8192"],
        ),
        (
            ["--foo =bar", "  --baz  =  qux"],
            ["--foo=bar", "--baz=qux"],
        ),
        (
            None,
            [],
        ),
    ],
)
def test_flatten_backend_param(backend_parameters, expected):
    backend = CustomServer.__new__(CustomServer)
    backend._model = types.SimpleNamespace(backend_parameters=backend_parameters)
    backend.inference_backend = None
    assert backend._flatten_backend_param() == expected


@pytest.mark.parametrize(
    "backend_parameters, parameter_format, expected",
    [
        # Test space format conversion
        (["ctx-size=1024"], ParameterFormatEnum.SPACE, ["--ctx-size", "1024"]),
        (["--ctx-size=1024"], ParameterFormatEnum.SPACE, ["--ctx-size", "1024"]),
        (["n-gpu-layers=0"], ParameterFormatEnum.SPACE, ["--n-gpu-layers", "0"]),
        (["-n-gpu-layers=0"], ParameterFormatEnum.SPACE, ["--n-gpu-layers", "0"]),
        # Test equal format conversion
        (["--ctx-size 1024"], ParameterFormatEnum.EQUAL, ["--ctx-size=1024"]),
        (["ctx-size 1024"], ParameterFormatEnum.EQUAL, ["--ctx-size=1024"]),
        (["-ctx-size 1024"], ParameterFormatEnum.EQUAL, ["--ctx-size=1024"]),
        # Test no conversion (None)
        (["--ctx-size 1024"], None, ["--ctx-size", "1024"]),
        (["--ctx-size=1024"], None, ["--ctx-size=1024"]),
        # Test flag parameters (no value)
        (["--verbose"], ParameterFormatEnum.SPACE, ["--verbose"]),
        (["--verbose"], ParameterFormatEnum.EQUAL, ["--verbose"]),
        (["verbose"], ParameterFormatEnum.SPACE, ["--verbose"]),
        (["-verbose"], ParameterFormatEnum.EQUAL, ["--verbose"]),
        # Test parameters with spaces in value
        (['--name "my model"'], ParameterFormatEnum.SPACE, ["--name", "my model"]),
        (['--name "my model"'], ParameterFormatEnum.EQUAL, ["--name=my model"]),
        (['name "my model"'], ParameterFormatEnum.SPACE, ["--name", "my model"]),
        # Test multiple parameters
        (
            ["ctx-size=1024", "n-gpu-layers=0"],
            ParameterFormatEnum.SPACE,
            ["--ctx-size", "1024", "--n-gpu-layers", "0"],
        ),
        (
            ["ctx-size 1024", "n-gpu-layers 0"],
            ParameterFormatEnum.EQUAL,
            ["--ctx-size=1024", "--n-gpu-layers=0"],
        ),
        # Test edge cases
        (["  ctx-size = 1024  "], ParameterFormatEnum.SPACE, ["--ctx-size", "1024"]),
        (["  ctx-size   1024  "], ParameterFormatEnum.EQUAL, ["--ctx-size=1024"]),
        ([""], ParameterFormatEnum.SPACE, []),
        (None, ParameterFormatEnum.SPACE, []),
        # Test mixed formats with conversion
        (
            ["--ctx-size=1024", "n-gpu-layers 0"],
            ParameterFormatEnum.SPACE,
            ["--ctx-size", "1024", "--n-gpu-layers", "0"],
        ),
        (
            ["ctx-size 1024", "--n-gpu-layers=0"],
            ParameterFormatEnum.EQUAL,
            ["--ctx-size=1024", "--n-gpu-layers=0"],
        ),
        # Test parameters with multiple values
        (
            ['--arg "value1 value2 value3"'],
            ParameterFormatEnum.SPACE,
            ["--arg", "value1 value2 value3"],
        ),
        (
            ['--arg "value1 value2 value3"'],
            ParameterFormatEnum.EQUAL,
            ["--arg=value1 value2 value3"],
        ),
    ],
)
def test_flatten_backend_param_with_format_conversion(
    backend_parameters, parameter_format, expected
):
    backend = CustomServer.__new__(CustomServer)
    backend._model = types.SimpleNamespace(backend_parameters=backend_parameters)

    # Mock the inference backend with parameter_format configuration
    if parameter_format is not None:
        inference_backend = types.SimpleNamespace(parameter_format=parameter_format)
        backend.inference_backend = inference_backend
    else:
        backend.inference_backend = None

    assert backend._flatten_backend_param() == expected


@pytest.mark.parametrize(
    "backend_parameters, backend_version, expected",
    [
        (None, None, []),
        ([], "0.15.2", []),
        ([], "0.16.0", ["--disable-access-log-for-endpoints", "/metrics"]),
        (
            ["--disable-access-log-for-endpoints=/health,/metrics"],
            "0.16.0",
            [],
        ),
        (
            ["--disable-access-log-for-endpoints", "/health,/metrics"],
            "0.16.0",
            [],
        ),
    ],
)
def test_vllm_access_log_arguments(backend_parameters, backend_version, expected):
    assert (
        get_vllm_access_log_arguments(backend_parameters, backend_version) == expected
    )


@pytest.mark.parametrize(
    "backend_parameters, backend_version, expected",
    [
        (None, None, []),
        ([], "0.5.8", []),
        ([], "0.5.8.post1", ["--uvicorn-access-log-exclude-prefixes", "/metrics"]),
        (
            ["--uvicorn-access-log-exclude-prefixes=/health"],
            "0.5.8.post1",
            [],
        ),
        (
            ["--uvicorn-access-log-exclude-prefixes", "/health"],
            "0.5.8.post1",
            [],
        ),
    ],
)
def test_sglang_access_log_arguments(backend_parameters, backend_version, expected):
    assert (
        get_sglang_access_log_arguments(backend_parameters, backend_version) == expected
    )


@pytest.mark.parametrize(
    "backend_parameters, backend_version, expected",
    [
        # Unknown version: do not inject (we cannot version-gate it).
        (None, None, []),
        # Below the v0.9.0.1 cutoff: skipped (V1 silently dropped the field).
        ([], "0.9.0", []),
        # At/after the cutoff: injected.
        ([], "0.9.0.1", ["--enable-prompt-tokens-details"]),
        ([], "0.10.0", ["--enable-prompt-tokens-details"]),
        # User explicitly opted in: do not duplicate.
        (["--enable-prompt-tokens-details"], "0.10.0", []),
        # User explicitly opted out: respect their choice.
        (["--no-enable-prompt-tokens-details"], "0.10.0", []),
        # Prefix-caching flags are not GPUStack's responsibility — left to the user.
        (["--enable-prefix-caching"], "0.10.0", ["--enable-prompt-tokens-details"]),
    ],
)
def test_vllm_cache_report_arguments(backend_parameters, backend_version, expected):
    assert (
        get_vllm_cache_report_arguments(backend_parameters, backend_version) == expected
    )


@pytest.mark.parametrize(
    "backend_parameters, backend_version, expected",
    [
        # Unknown version: do not inject (we cannot version-gate it).
        (None, None, []),
        # Below the v0.3.4 cutoff: skipped.
        ([], "0.3.3", []),
        # At/after the cutoff: injected.
        ([], "0.3.4", ["--enable-cache-report"]),
        ([], "0.5.8.post1", ["--enable-cache-report"]),
        # User already passed it: do not duplicate.
        (["--enable-cache-report"], "0.5.8.post1", []),
    ],
)
def test_sglang_cache_report_arguments(backend_parameters, backend_version, expected):
    assert (
        get_sglang_cache_report_arguments(backend_parameters, backend_version)
        == expected
    )


def test_vllm_set_cache_env_defaults_to_config_cache_dir(tmp_path):
    backend = VLLMServer.__new__(VLLMServer)
    backend._config = types.SimpleNamespace(cache_dir=str(tmp_path))

    env = {}
    backend._set_cache_env(env)

    expected = tmp_path / "vllm"
    assert env["VLLM_CACHE_ROOT"] == str(expected)
    assert expected.is_dir()


def test_vllm_set_cache_env_respects_user_override(tmp_path):
    backend = VLLMServer.__new__(VLLMServer)
    backend._config = types.SimpleNamespace(cache_dir=str(tmp_path))

    env = {"VLLM_CACHE_ROOT": "/custom/cache"}
    backend._set_cache_env(env)

    assert env["VLLM_CACHE_ROOT"] == "/custom/cache"
    # Default cache dir should not be created when the user overrode it.
    assert not (tmp_path / "vllm").exists()


def test_vllm_command_args_include_late_system_flags_as_injected():
    backend = VLLMServer.__new__(VLLMServer)
    backend.inference_backend = None
    backend._model_path = "/models/llm"
    backend._worker = types.SimpleNamespace(ip="192.168.50.10")
    backend._model_instance = types.SimpleNamespace(
        model_name="llm",
        gpu_indexes=[],
        ports=[4000],
        computed_resource_claim=None,
    )
    backend._model = types.SimpleNamespace(
        backend=BackendEnum.VLLM,
        backend_parameters=[],
        backend_version=None,
        categories=[],
        extended_kv_cache=None,
        speculative_config=None,
    )
    backend._derive_max_model_len = lambda: None
    backend._get_speculative_arguments = lambda: []
    backend._get_selected_gpu_devices = lambda: [
        types.SimpleNamespace(vendor="NVIDIA", arch_family=None)
    ]

    arguments, injected = backend._build_command_args(port=4000, is_distributed=False)

    assert arguments[-6:] == [
        "--host",
        "192.168.50.10",
        "--port",
        "4000",
        "--served-model-name",
        "llm",
    ]
    assert injected == [
        "--host",
        "192.168.50.10",
        "--port",
        "4000",
        "--served-model-name",
        "llm",
    ]


def test_vllm_command_args_exclude_user_backend_parameters_from_injected():
    backend = VLLMServer.__new__(VLLMServer)
    backend.inference_backend = None
    backend._model_path = "/models/llm"
    backend._worker = types.SimpleNamespace(ip="192.168.50.10")
    backend._model_instance = types.SimpleNamespace(
        model_name="llm",
        gpu_indexes=[],
        ports=[4000],
        computed_resource_claim=None,
    )
    backend._model = types.SimpleNamespace(
        backend=BackendEnum.VLLM,
        backend_parameters=["--host", "0.0.0.0", "--temperature", "0.2"],
        backend_version=None,
        categories=[],
        extended_kv_cache=None,
        speculative_config=None,
    )
    backend._derive_max_model_len = lambda: None
    backend._get_speculative_arguments = lambda: []
    backend._get_selected_gpu_devices = lambda: [
        types.SimpleNamespace(vendor="NVIDIA", arch_family=None)
    ]

    arguments, injected = backend._build_command_args(port=4000, is_distributed=False)

    assert "--temperature" in arguments
    assert "--temperature" not in injected
    assert "--host" not in injected
    assert injected == ["--port", "4000", "--served-model-name", "llm"]


def test_sglang_command_args_include_model_and_late_system_flags_as_injected():
    backend = SGLangServer.__new__(SGLangServer)
    backend.inference_backend = None
    backend._model_path = "/models/llm"
    backend._worker = types.SimpleNamespace(ip="192.168.50.10")
    backend._model_instance = types.SimpleNamespace(
        gpu_indexes=[],
        ports=[4000],
        computed_resource_claim=None,
    )
    backend._model = types.SimpleNamespace(
        backend_parameters=[],
        backend_version=None,
        env={"GPUSTACK_DISABLE_METRICS": "1"},
        extended_kv_cache=None,
        speculative_config=None,
    )
    backend._derive_max_model_len = lambda: None
    backend._get_model_architecture = lambda: []
    backend._get_speculative_arguments = lambda: []
    backend._get_hicache_arguments = lambda: []
    backend._get_selected_gpu_devices = lambda: [
        types.SimpleNamespace(vendor="NVIDIA", arch_family=None)
    ]

    _, injected = backend._build_command_args(
        port=4000,
        is_distributed=False,
        is_distributed_leader=False,
    )

    assert injected == [
        "--model-path",
        "/models/llm",
        "--host",
        "192.168.50.10",
        "--port",
        "4000",
    ]


def test_vox_box_command_args_return_injected_parameters():
    backend = VoxBoxServer.__new__(VoxBoxServer)
    backend.inference_backend = None
    backend._model_path = "/models/audio"
    backend._config = types.SimpleNamespace(data_dir="/var/lib/gpustack")
    backend._worker = types.SimpleNamespace(ip="192.168.50.10")
    backend._model_instance = types.SimpleNamespace(gpu_indexes=[1])
    backend._model = types.SimpleNamespace(backend_parameters=[], backend_version=None)

    _, injected = backend._build_command_args(port=4000)

    assert injected == [
        "--model",
        "/models/audio",
        "--data-dir",
        "/var/lib/gpustack",
        "--host",
        "192.168.50.10",
        "--port",
        "4000",
        "--device",
        "cuda:1",
    ]


def test_custom_command_args_return_injected_parameters_after_entrypoint():
    backend = CustomServer.__new__(CustomServer)
    backend._model_path = "/models/custom"
    backend._worker = types.SimpleNamespace(ip="192.168.50.10")
    backend._model_instance = types.SimpleNamespace(ports=[4000])
    backend._model = types.SimpleNamespace(
        backend_parameters=["--temperature", "0.2"],
        backend_version=None,
        env={},
        name="custom-model",
        run_command="python -m custom.launch --model-path {{model_path}} --port {{port}}",
    )
    backend.inference_backend = types.SimpleNamespace(
        replace_command_param=lambda **_: (
            "python -m custom.launch --model-path /models/custom --port 4000"
        )
    )

    arguments, injected = backend._build_command_args()

    assert arguments[-2:] == ["--temperature", "0.2"]
    assert injected == ["--model-path", "/models/custom", "--port", "4000"]


def test_custom_command_args_include_short_flags_as_injected():
    backend = CustomServer.__new__(CustomServer)
    backend._model_path = "/models/custom"
    backend._worker = types.SimpleNamespace(ip="192.168.50.10")
    backend._model_instance = types.SimpleNamespace(ports=[4000])
    backend._model = types.SimpleNamespace(
        backend_parameters=["-u", "1"],
        backend_version=None,
        env={},
        name="custom-model",
        run_command="custom-server -s 0.0.0.0 -t 4",
    )
    backend.inference_backend = types.SimpleNamespace(
        replace_command_param=lambda **_: "custom-server -s 0.0.0.0 -t 4"
    )

    _, injected = backend._build_command_args()

    assert injected == ["-s", "0.0.0.0", "-t", "4"]


def test_injected_parameters_start_at_zero_with_explicit_container_entrypoint():
    backend = CustomServer.__new__(CustomServer)
    backend._model_path = "/models/custom"
    backend._worker = types.SimpleNamespace(ip="192.168.50.10")
    backend._model_instance = types.SimpleNamespace(ports=[4000])
    backend._model = types.SimpleNamespace(
        backend_parameters=["-u", "1"],
        backend_version=None,
        env={},
        name="custom-model",
        run_command="-m /models/custom -t 4",
    )
    backend.inference_backend = types.SimpleNamespace(
        replace_command_param=lambda **_: "-m /models/custom -t 4"
    )

    _, injected = backend._build_command_args(entrypoint=["llama-server"])

    assert injected == ["-m", "/models/custom", "-t", "4"]


@pytest.mark.parametrize(
    "default_entrypoint, version_entrypoint, default_run_command, expected_entrypoint, expected_injected",
    [
        (
            "llama-server",
            None,
            "-m {{model_path}} -p {{port}}",
            ["llama-server"],
            ["-m", "/models/custom", "-p", "4000"],
        ),
        (
            "unused-entrypoint",
            "python -m custom.launch",
            "--model-path {{model_path}} --port {{port}}",
            ["python", "-m", "custom.launch"],
            ["--model-path", "/models/custom", "--port", "4000"],
        ),
    ],
)
def test_custom_backend_configured_entrypoint_injected_parameters(
    default_entrypoint,
    version_entrypoint,
    default_run_command,
    expected_entrypoint,
    expected_injected,
):
    backend = CustomServer.__new__(CustomServer)
    backend._model_path = "/models/custom"
    backend._worker = types.SimpleNamespace(ip="192.168.50.10")
    backend._model_instance = types.SimpleNamespace(ports=[4000])
    backend._model = types.SimpleNamespace(
        backend_parameters=["--user-param", "1"],
        backend_version="cpu",
        env={},
        name="custom-model",
        run_command=None,
    )
    backend.inference_backend = InferenceBackend(
        backend_name="custom-entrypoint-backend",
        default_version="cpu",
        default_entrypoint=default_entrypoint,
        default_run_command=default_run_command,
        version_configs=VersionConfigDict(
            root={
                "cpu": VersionConfig(
                    image_name="custom/backend:cpu",
                    entrypoint=version_entrypoint,
                    custom_framework="cpu",
                )
            }
        ),
    )

    entrypoint = backend.inference_backend.get_container_entrypoint("cpu")
    arguments, injected = backend._build_command_args(entrypoint=entrypoint)

    assert entrypoint == expected_entrypoint
    assert arguments[-2:] == ["--user-param", "1"]
    assert injected == expected_injected
