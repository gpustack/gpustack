import types

import pytest

from gpustack.worker.backends.custom import CustomServer


@pytest.mark.parametrize(
    "image_name, container_registry, expect_image_name, can_connet_dockerhub",
    [
        (
            "ghcr.io/ggml-org/llama.cpp:server",
            "test-registry.io",
            "ghcr.io/ggml-org/llama.cpp:server",
            True,
        ),
        (
            "gpustack/runner:cuda12.8-vllm0.10.2",
            "test-registry.io",
            "test-registry.io/gpustack/runner:cuda12.8-vllm0.10.2",
            True,
        ),
        (
            "foo/bar",
            "test-registry.io",
            "test-registry.io/foo/bar",
            True,
        ),
        (
            "ubuntu:24.04",
            "test-registry.io",
            "test-registry.io/ubuntu:24.04",
            True,
        ),
        (
            "gpustack/runner:cuda12.8-vllm0.10.2",
            "",
            "quay.io/gpustack/runner:cuda12.8-vllm0.10.2",
            False,
        ),
        (
            "lmsysorg/sglang:v0.5.5",
            "",
            "lmsysorg/sglang:v0.5.5",
            False,
        ),
    ],
)
@pytest.mark.asyncio
async def test_apply_registry_override(
    image_name, container_registry, expect_image_name, can_connet_dockerhub, monkeypatch
):
    backend = CustomServer.__new__(CustomServer)
    # CustomServer inherits _apply_registry_override from InferenceServer,
    # and _apply_registry_override accesses self._config.system_default_container_registry.
    # Since we constructed the instance via __new__ (without __init__),
    # the _config attribute does not exist. We attach a minimal stub config here.
    backend._config = types.SimpleNamespace(
        system_default_container_registry=container_registry
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.base.get_dockerhub_reachable",
        lambda: can_connet_dockerhub,
    )
    assert backend._apply_registry_override(image_name) == expect_image_name

    if container_registry:
        backend._config = types.SimpleNamespace(system_default_container_registry=None)
        assert backend._apply_registry_override(image_name) == image_name
