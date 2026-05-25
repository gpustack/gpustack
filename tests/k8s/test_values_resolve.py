from gpustack_runtime.detector import ManufacturerEnum

from gpustack.schemas.clusters import (
    ImageCredential,
    K8sOptions,
    K8sOptionsOverride,
)


def _cred(registry):
    return ImageCredential(registry=registry, username="u", password="p")


def test_resolve_no_runtime_strips_overrides():
    values = K8sOptions(
        node_selector={"env": "prod"},
        gpu_vendor_overrides={
            ManufacturerEnum.NVIDIA: K8sOptionsOverride(
                node_selector={"nvidia.com/gpu": "true"}
            )
        },
    )
    out = values.resolve_for(None)
    assert out.node_selector == {"env": "prod"}
    assert out.gpu_vendor_overrides is None


def test_resolve_runtime_without_matching_override_returns_base():
    values = K8sOptions(
        node_selector={"env": "prod"},
        gpu_vendor_overrides={
            ManufacturerEnum.NVIDIA: K8sOptionsOverride(
                node_selector={"nvidia.com/gpu": "true"}
            )
        },
    )
    out = values.resolve_for(ManufacturerEnum.AMD)
    assert out.node_selector == {"env": "prod"}
    assert out.gpu_vendor_overrides is None


def test_resolve_node_selector_merge_override_wins_per_key():
    values = K8sOptions(
        node_selector={"env": "prod", "zone": "a"},
        gpu_vendor_overrides={
            ManufacturerEnum.NVIDIA: K8sOptionsOverride(
                node_selector={"zone": "b", "nvidia.com/gpu": "true"}
            )
        },
    )
    out = values.resolve_for(ManufacturerEnum.NVIDIA)
    assert out.node_selector == {
        "env": "prod",
        "zone": "b",
        "nvidia.com/gpu": "true",
    }


def test_resolve_passes_through_base_image_credentials():
    """image_credentials is registry-scoped, not vendor-scoped — base flows through unchanged."""
    values = K8sOptions(
        image_credentials=[_cred("harbor.example.com"), _cred("ghcr.io")],
        gpu_vendor_overrides={
            ManufacturerEnum.NVIDIA: K8sOptionsOverride(
                node_selector={"nvidia.com/gpu": "true"}
            )
        },
    )
    out = values.resolve_for(ManufacturerEnum.NVIDIA)
    assert [c.registry for c in out.image_credentials] == [
        "harbor.example.com",
        "ghcr.io",
    ]
    assert out.node_selector == {"nvidia.com/gpu": "true"}


def test_resolve_with_only_override_node_selector():
    values = K8sOptions(
        gpu_vendor_overrides={
            ManufacturerEnum.ASCEND: K8sOptionsOverride(
                node_selector={"huawei.com/Ascend910": "true"}
            )
        },
    )
    out = values.resolve_for(ManufacturerEnum.ASCEND)
    assert out.node_selector == {"huawei.com/Ascend910": "true"}
    assert out.image_credentials is None


def test_resolve_empty_override_block_leaves_base_intact():
    values = K8sOptions(
        node_selector={"env": "prod"},
        image_credentials=[_cred("harbor.example.com")],
        gpu_vendor_overrides={ManufacturerEnum.NVIDIA: K8sOptionsOverride()},
    )
    out = values.resolve_for(ManufacturerEnum.NVIDIA)
    assert out.node_selector == {"env": "prod"}
    assert [c.registry for c in out.image_credentials] == ["harbor.example.com"]


def test_resolve_no_gpu_vendor_overrides_field():
    values = K8sOptions(node_selector={"env": "prod"})
    out = values.resolve_for(ManufacturerEnum.NVIDIA)
    assert out.node_selector == {"env": "prod"}
    assert out.gpu_vendor_overrides is None


def test_resolve_by_alias_payload_round_trip():
    payload = {
        "imageCredentials": [
            {
                "registry": "harbor.example.com",
                "username": "alice",
                "password": "s3cret",
            }
        ],
        "nodeSelector": {"env": "prod"},
        "gpuVendorOverrides": {
            "nvidia": {
                "nodeSelector": {"nvidia.com/gpu": "true"},
            }
        },
    }
    values = K8sOptions.model_validate(payload)
    out = values.resolve_for(ManufacturerEnum.NVIDIA)
    assert [c.registry for c in out.image_credentials] == ["harbor.example.com"]
    assert out.node_selector == {"env": "prod", "nvidia.com/gpu": "true"}
