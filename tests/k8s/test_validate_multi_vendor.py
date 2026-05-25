import pytest
from gpustack_runtime.detector import ManufacturerEnum

from gpustack.api.exceptions import InvalidException
from gpustack.routes.clusters import _validate_multi_vendor_overrides
from gpustack.schemas.clusters import K8sOptions, K8sOptionsOverride


def test_no_runtimes_skips_validation():
    _validate_multi_vendor_overrides(None, None)
    _validate_multi_vendor_overrides([], None)


def test_single_runtime_skips_validation_even_without_override():
    # Single-vendor mode does not require a vendor override.
    _validate_multi_vendor_overrides([ManufacturerEnum.NVIDIA], None)


def test_repeated_same_runtime_treated_as_single_vendor():
    _validate_multi_vendor_overrides(
        [ManufacturerEnum.NVIDIA, ManufacturerEnum.NVIDIA], None
    )


def test_unknown_excluded_from_distinct_count():
    # UNKNOWN + NVIDIA distinct count = 1, so single-vendor → no validation.
    _validate_multi_vendor_overrides(
        [ManufacturerEnum.UNKNOWN, ManufacturerEnum.NVIDIA], None
    )


def test_multi_vendor_without_k8s_options_raises():
    with pytest.raises(InvalidException) as exc:
        _validate_multi_vendor_overrides(
            [ManufacturerEnum.NVIDIA, ManufacturerEnum.ASCEND],
            None,
        )
    assert "nvidia" in exc.value.message
    assert "ascend" in exc.value.message


def test_multi_vendor_with_partial_overrides_raises():
    k8s_options = K8sOptions(
        gpu_vendor_overrides={
            ManufacturerEnum.NVIDIA: K8sOptionsOverride(
                node_selector={"nvidia.com/gpu": "true"}
            )
        }
    )
    with pytest.raises(InvalidException) as exc:
        _validate_multi_vendor_overrides(
            [ManufacturerEnum.NVIDIA, ManufacturerEnum.ASCEND],
            k8s_options,
        )
    assert "ascend" in exc.value.message
    assert "nvidia" not in exc.value.message


def test_multi_vendor_with_empty_override_node_selector_raises():
    """An override entry without a populated node_selector does not satisfy
    the requirement — the CPU DS still gets no exclusion key from it."""
    k8s_options = K8sOptions(
        gpu_vendor_overrides={
            ManufacturerEnum.NVIDIA: K8sOptionsOverride(
                node_selector={"nvidia.com/gpu": "true"}
            ),
            ManufacturerEnum.ASCEND: K8sOptionsOverride(),
        }
    )
    with pytest.raises(InvalidException) as exc:
        _validate_multi_vendor_overrides(
            [ManufacturerEnum.NVIDIA, ManufacturerEnum.ASCEND],
            k8s_options,
        )
    assert "ascend" in exc.value.message


def test_multi_vendor_with_all_overrides_passes():
    k8s_options = K8sOptions(
        gpu_vendor_overrides={
            ManufacturerEnum.NVIDIA: K8sOptionsOverride(
                node_selector={"nvidia.com/gpu": "true"}
            ),
            ManufacturerEnum.ASCEND: K8sOptionsOverride(
                node_selector={"huawei.com/Ascend910": "true"}
            ),
        }
    )
    _validate_multi_vendor_overrides(
        [ManufacturerEnum.NVIDIA, ManufacturerEnum.ASCEND],
        k8s_options,
    )


def test_multi_vendor_duplicate_override_node_selector_raises():
    """Two vendors with identical nodeSelector dicts target the same node
    set — podAntiAffinity would let one DS win each node and the other
    would stay Pending forever."""
    k8s_options = K8sOptions(
        gpu_vendor_overrides={
            ManufacturerEnum.NVIDIA: K8sOptionsOverride(node_selector={"gpu": "true"}),
            ManufacturerEnum.ASCEND: K8sOptionsOverride(node_selector={"gpu": "true"}),
        }
    )
    with pytest.raises(InvalidException) as exc:
        _validate_multi_vendor_overrides(
            [ManufacturerEnum.NVIDIA, ManufacturerEnum.ASCEND],
            k8s_options,
        )
    assert "nvidia+ascend" in exc.value.message or "ascend+nvidia" in exc.value.message


def test_multi_vendor_partial_key_overlap_is_allowed():
    """Overrides may share a key (e.g. a shared zone constraint) as long
    as the dicts are not fully equal — different values still scope each
    vendor to a distinct node set."""
    k8s_options = K8sOptions(
        gpu_vendor_overrides={
            ManufacturerEnum.NVIDIA: K8sOptionsOverride(
                node_selector={"gpu-type": "nvidia", "zone": "a"}
            ),
            ManufacturerEnum.ASCEND: K8sOptionsOverride(
                node_selector={"gpu-type": "ascend", "zone": "a"}
            ),
        }
    )
    _validate_multi_vendor_overrides(
        [ManufacturerEnum.NVIDIA, ManufacturerEnum.ASCEND],
        k8s_options,
    )


def test_multi_vendor_base_key_in_override_raises():
    """A base nodeSelector key reused in any vendor override creates a
    self-contradicting CPU DS (requires X AND DoesNotExist X)."""
    k8s_options = K8sOptions(
        node_selector={"gpu": "true"},
        gpu_vendor_overrides={
            ManufacturerEnum.NVIDIA: K8sOptionsOverride(
                node_selector={"gpu": "true", "nvidia.com/gpu": "true"}
            ),
            ManufacturerEnum.ASCEND: K8sOptionsOverride(
                node_selector={"huawei.com/Ascend910": "true"}
            ),
        },
    )
    with pytest.raises(InvalidException) as exc:
        _validate_multi_vendor_overrides(
            [ManufacturerEnum.NVIDIA, ManufacturerEnum.ASCEND],
            k8s_options,
        )
    assert "gpu" in exc.value.message


def test_multi_vendor_disjoint_base_and_override_keys_passes():
    """Base provides shared constraint, overrides provide vendor-only
    keys — no clash."""
    k8s_options = K8sOptions(
        node_selector={"env": "prod"},
        gpu_vendor_overrides={
            ManufacturerEnum.NVIDIA: K8sOptionsOverride(
                node_selector={"nvidia.com/gpu": "true"}
            ),
            ManufacturerEnum.ASCEND: K8sOptionsOverride(
                node_selector={"huawei.com/Ascend910": "true"}
            ),
        },
    )
    _validate_multi_vendor_overrides(
        [ManufacturerEnum.NVIDIA, ManufacturerEnum.ASCEND],
        k8s_options,
    )
