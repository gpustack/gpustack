from typing import Dict, List, Optional

from packaging.version import Version
from packaging.specifiers import SpecifierSet
from gpustack_runtime.detector import ManufacturerEnum
from gpustack_runtime.detector.ascend import get_ascend_cann_variant

from gpustack.schemas.model_sets import ModelSpec
from gpustack.schemas.workers import GPUDeviceStatus


def filter_specs_by_gpu(
    gpus: List[GPUDeviceStatus], specs: List[ModelSpec]
) -> List[ModelSpec]:
    """Filter model specs based on the GPUs available."""

    # Matched specs mapping by mode (standard, throughput, latency, etc.).
    filtered: Dict[str, ModelSpec] = {}

    gpu_vendors = {gpu.vendor.lower() for gpu in gpus if gpu.vendor}

    # Vendor variants. Now only Ascend CANN variants are supported.
    # An unmapped SoC yields None and is left out.
    vendor_variants = {
        variant.lower()
        for gpu in gpus
        if gpu.arch_family is not None and gpu.vendor == ManufacturerEnum.ASCEND
        if (variant := get_ascend_cann_variant(gpu.arch_family))
    }

    for spec in specs:
        # If already selected for this mode, skip
        if spec.mode in filtered:
            continue

        gf = spec.gpu_filters
        if gf is None:
            filtered[spec.mode] = spec
            continue

        # --- GPU Vendor check ---
        if gf.vendor:
            wanted = {v.lower() for v in gf.vendor}
            if wanted.isdisjoint(gpu_vendors):
                continue

        # --- Compute Capability check ---
        if gf.compute_capability:
            if not any(
                match_compute_capability(gf.compute_capability, gpu.compute_capability)
                for gpu in gpus
            ):
                continue

        # --- Vendor Variant check ---
        if gf.vendor_variant:
            wanted = {v.lower() for v in gf.vendor_variant}
            if wanted.isdisjoint(vendor_variants):
                continue

        filtered[spec.mode] = spec

    result = list(filtered.values())

    # Sort by mode priority in case catalog items are messy. These are our conventional priorities.
    ORDER = {"throughput": 0, "latency": 1, "standard": 2}
    result.sort(key=lambda spec: (ORDER.get(spec.mode, 999), spec.mode))

    return result


def match_compute_capability(filter_str: Optional[str], gpu_cc: Optional[str]) -> bool:
    """Check if the GPU compute capability matches the given filter string.

    Args:
        filter_str (Optional[str]): The pip-style version specifier string.
        gpu_cc (Optional[str]): The GPU compute capability version string.

    Returns:
        bool: True if the GPU compute capability matches the filter, False otherwise.
    """
    if not filter_str:
        return True

    if not gpu_cc:
        return False

    try:
        spec_set = SpecifierSet(filter_str)
        cc_version = Version(gpu_cc)
        return cc_version in spec_set
    except Exception:
        return False
