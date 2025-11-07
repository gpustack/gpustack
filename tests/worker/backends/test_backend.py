import types

import pytest

from gpustack.worker.backends.custom import CustomServer


@pytest.mark.parametrize(
    "image_name, container_registry, expect_image_name",
    [
        (
            "ghcr.io/ggml-org/llama.cpp:server",
            "test-registry.io",
            "ghcr.io/ggml-org/llama.cpp:server",
        ),
        (
            "gpustack/runner:cuda12.8-vllm0.10.2",
            "test-registry.io",
            "test-registry.io/gpustack/runner:cuda12.8-vllm0.10.2",
        ),
        (
            "foo/bar",
            "test-registry.io",
            "test-registry.io/foo/bar",
        ),
        (
            "ubuntu:24.04",
            "test-registry.io",
            "test-registry.io/ubuntu:24.04",
        ),
    ],
)
@pytest.mark.asyncio
async def test_apply_registry_override(
    image_name, container_registry, expect_image_name
):
    backend = CustomServer.__new__(CustomServer)
    # CustomServer inherits _apply_registry_override from InferenceServer,
    # and _apply_registry_override accesses self._config.system_default_container_registry.
    # Since we constructed the instance via __new__ (without __init__),
    # the _config attribute does not exist. We attach a minimal stub config here.
    backend._config = types.SimpleNamespace(
        system_default_container_registry=container_registry
    )
    assert backend._apply_registry_override(image_name) == expect_image_name

    backend._config = types.SimpleNamespace(system_default_container_registry=None)
    assert backend._apply_registry_override(image_name) == image_name
