import types

import pytest

from gpustack.schemas.inference_backend import ParameterFormatEnum
from gpustack.utils.config import apply_registry_override_to_image
from gpustack.worker.backends.custom import CustomServer


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
