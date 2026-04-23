import types

import pytest

from gpustack.utils.config import apply_registry_override_to_image
from gpustack.worker.backends.custom import CustomServer
from gpustack.worker.backends.sglang import (
    get_access_log_arguments as get_sglang_access_log_arguments,
)
from gpustack.worker.backends.vllm import (
    VLLMServer,
    get_access_log_arguments as get_vllm_access_log_arguments,
)


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
    assert backend._flatten_backend_param() == expected


@pytest.mark.parametrize(
    "backend_parameters, expected",
    [
        (None, ["--disable-access-log-for-endpoints", "/metrics"]),
        ([], ["--disable-access-log-for-endpoints", "/metrics"]),
        (
            ["--disable-access-log-for-endpoints=/health,/metrics"],
            [],
        ),
        (
            ["--disable-access-log-for-endpoints", "/health,/metrics"],
            [],
        ),
    ],
)
def test_vllm_access_log_arguments(backend_parameters, expected):
    assert get_vllm_access_log_arguments(backend_parameters) == expected


@pytest.mark.parametrize(
    "backend_parameters, expected",
    [
        (None, ["--uvicorn-access-log-exclude-prefixes", "/metrics"]),
        ([], ["--uvicorn-access-log-exclude-prefixes", "/metrics"]),
        (
            ["--uvicorn-access-log-exclude-prefixes=/health"],
            [],
        ),
        (
            ["--uvicorn-access-log-exclude-prefixes", "/health"],
            [],
        ),
    ],
)
def test_sglang_access_log_arguments(backend_parameters, expected):
    assert get_sglang_access_log_arguments(backend_parameters) == expected


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
