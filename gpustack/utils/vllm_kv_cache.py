"""Which targets can carry vLLM's local extended KV cache.

Lives in ``utils`` (not in ``worker.backends``) so scheduler / policies code
can import it without dragging in worker-only dependencies such as
``gpustack_runtime.deployer`` — the same reason ``vllm_topology`` sits here.
The worker skips the connector on whatever reason this returns and the model
evaluator reports it, so the compatibility check and the running instance
cannot drift apart.
"""

from typing import Optional

from gpustack_runtime.deployer.__utils__ import compare_versions
from gpustack_runtime.detector.ascend import get_ascend_cann_variant

# The release from which vllm-ascend registers its NPU-native worker under
# the SimpleCPUOffloadConnector name; on cann the backend version is the
# vllm-ascend version.
ASCEND_LOCAL_KV_CACHE_MIN_VERSION = "0.21.0"


def ascend_local_kv_cache_unsupported_reason(
    arch_family: Optional[str],
    backend_version: Optional[str],
) -> Optional[str]:
    """Why local extended KV cache cannot run on this Ascend target.

    ``None`` when it can, which includes an unresolved backend version: "Auto"
    resolves to the newest runner the worker can pull, and the resolved
    version is written back onto the model before the arguments are built.

    Args:
        arch_family: SoC name of the target device, e.g. "Ascend910B3".
        backend_version: The deployment's vLLM Ascend version, if pinned.

    Returns:
        A user-facing reason, or None when the target is supported.
    """
    if get_ascend_cann_variant(arch_family) == "310p":
        # The runner images leave every KV offload component out of the 310P
        # build, and upstream validates the connector on A2/A3 only.
        return "Extended KV cache is not supported on Ascend 310P devices."

    if (
        backend_version
        and compare_versions(backend_version, ASCEND_LOCAL_KV_CACHE_MIN_VERSION) < 0
    ):
        return (
            f"Extended KV cache on Ascend requires vLLM Ascend "
            f"{ASCEND_LOCAL_KV_CACHE_MIN_VERSION} or later, "
            f"but version {backend_version} is selected."
        )

    return None
