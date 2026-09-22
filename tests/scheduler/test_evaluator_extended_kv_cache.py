"""Compatibility reporting for vLLM's local extended KV cache.

``evaluate_local_extended_kv_cache`` answers the model evaluation the UI
runs before a deploy: it must name the same reason the worker skips the
connector for, so an accelerator or backend version that cannot carry the
cache is refused at configuration time instead of starting degraded.
"""

import types

import pytest

from gpustack.scheduler.evaluator import evaluate_local_extended_kv_cache
from gpustack.schemas.models import ExtendedKVCacheConfig, KVCacheModeEnum


def _worker(*gpus, name="worker-1"):
    return types.SimpleNamespace(
        name=name,
        status=types.SimpleNamespace(
            gpu_devices=[
                types.SimpleNamespace(
                    vendor=vendor,
                    arch_family=arch_family,
                    type="cann" if vendor == "ascend" else "cuda",
                    index=index,
                )
                for index, (vendor, arch_family) in enumerate(gpus)
            ]
        ),
    )


def _model(extended_kv_cache, backend_version=None, gpu_ids=None):
    return types.SimpleNamespace(
        extended_kv_cache=extended_kv_cache,
        backend_version=backend_version,
        gpu_selector=(
            types.SimpleNamespace(gpu_ids=gpu_ids) if gpu_ids is not None else None
        ),
    )


def _local():
    return ExtendedKVCacheConfig(enabled=True, ram_size=20)


@pytest.mark.parametrize(
    "gpus, backend_version",
    [
        ((("nvidia", None),), None),
        ((("amd", None),), None),
        ((("ascend", "Ascend910B3"),), None),
        ((("ascend", "Ascend910B3"),), "0.23.0"),
        # One capable NPU in the cluster is enough: the deployment can land
        # on it even though the 310P beside it cannot carry the cache.
        ((("ascend", "Ascend310P3"), ("ascend", "Ascend910B3")), "0.23.0"),
    ],
)
def test_supported_targets_report_no_incompatibility(gpus, backend_version):
    assert (
        evaluate_local_extended_kv_cache(
            _model(_local(), backend_version), [_worker(*gpus)]
        )
        is None
    )


def test_ascend_backend_version_below_floor_is_reported():
    message = evaluate_local_extended_kv_cache(
        _model(_local(), "0.20.2"), [_worker(("ascend", "Ascend910B3"))]
    )

    assert message is not None
    assert "0.21.0" in message
    assert "0.20.2" in message


def test_ascend_310p_is_reported():
    message = evaluate_local_extended_kv_cache(
        _model(_local(), "0.23.0"), [_worker(("ascend", "Ascend310P3"))]
    )

    assert message is not None
    assert "310P" in message


@pytest.mark.parametrize(
    "gpus",
    [
        (("ascend", "Ascend310P3"), ("ascend", "Ascend910B3")),
        (("ascend", "Ascend910B3"), ("ascend", "Ascend310P3")),
    ],
)
def test_every_blocking_reason_is_reported(gpus):
    """A 310P beside a 910B below the version floor blocks for two different
    reasons; naming only the card that happens to come first would send the
    user after hardware when a version bump is also needed. The verdict does
    not depend on the order the cards are listed in."""
    message = evaluate_local_extended_kv_cache(
        _model(_local(), "0.20.2"), [_worker(*gpus)]
    )

    assert message is not None
    assert "310P" in message
    assert "0.21.0" in message


def test_pinned_gpu_outside_the_evaluated_workers_reports_nothing():
    """The selected GPU's worker is not in the evaluated set (it is down, say).
    Scheduling reports that accurately, and this check runs first — a verdict
    here would replace it with a claim about the fleet's accelerators."""
    workers = [_worker(("nvidia", None), name="worker-up")]

    assert (
        evaluate_local_extended_kv_cache(
            _model(_local(), gpu_ids=["worker-down:cann:0"]), workers
        )
        is None
    )


def test_manual_selection_is_judged_on_the_picked_gpus():
    """A deployment pinned to a 310P is refused even though the cluster holds
    a capable NPU the deployment can never land on."""
    workers = [_worker(("ascend", "Ascend310P3"), ("ascend", "Ascend910B3"))]

    message = evaluate_local_extended_kv_cache(
        _model(_local(), "0.23.0", gpu_ids=["worker-1:cann:0"]), workers
    )

    assert message is not None
    assert "310P" in message


def test_manual_selection_of_a_capable_gpu_reports_nothing():
    workers = [_worker(("ascend", "Ascend310P3"), ("ascend", "Ascend910B3"))]

    assert (
        evaluate_local_extended_kv_cache(
            _model(_local(), "0.23.0", gpu_ids=["worker-1:cann:1"]), workers
        )
        is None
    )


def test_other_vendor_is_reported():
    message = evaluate_local_extended_kv_cache(
        _model(_local()), [_worker(("mthreads", None))]
    )

    assert message is not None
    assert "NVIDIA, AMD or Ascend" in message


@pytest.mark.parametrize(
    "extended_kv_cache",
    [
        None,
        ExtendedKVCacheConfig(enabled=False),
        # Shared mode is the provider catalog's call, not this check's.
        ExtendedKVCacheConfig(
            enabled=True, mode=KVCacheModeEnum.SHARED, cache_service_id=3
        ),
    ],
)
def test_cache_not_locally_enabled_reports_nothing(extended_kv_cache):
    assert (
        evaluate_local_extended_kv_cache(
            _model(extended_kv_cache, "0.20.2"), [_worker(("ascend", "Ascend310P3"))]
        )
        is None
    )
