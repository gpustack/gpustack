"""Parse non-model workload resources without scheduler or worker imports."""

import math
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from gpustack.schemas.models import ModelSpecBase

MODEL_RAM_CLAIM_ENV = "GPUSTACK_MODEL_RAM_CLAIM"
MODEL_CPU_CLAIM_ENV = "GPUSTACK_MODEL_CPU_CLAIM"
NON_MODEL_MIN_RAM_BYTES = 64 * 1024**2


def get_non_model_ram_claim(model: "ModelSpecBase") -> Optional[int]:
    """Read an optional non-model container memory limit in bytes.

    Args:
        model: The deployment whose environment contains the declaration.

    Returns:
        The declared RAM, or None when absent.

    Raises:
        ValueError: The declaration is invalid or below 64 MiB.
    """
    value = (getattr(model, "env", None) or {}).get(MODEL_RAM_CLAIM_ENV)
    if value is None:
        return None
    try:
        claim = int(value)
        if claim >= NON_MODEL_MIN_RAM_BYTES:
            return claim
    except (ValueError, TypeError, OverflowError):
        pass
    raise ValueError(
        f"{MODEL_RAM_CLAIM_ENV} must be an integer of at least "
        f"{NON_MODEL_MIN_RAM_BYTES} bytes (64 MiB)."
    )


def get_non_model_cpu_claim(model: "ModelSpecBase") -> Optional[float]:
    """Read an optional non-model CPU declaration in cores.

    Args:
        model: The deployment whose environment contains the declaration.

    Returns:
        The declared CPU cores, or None when absent.

    Raises:
        ValueError: The declaration is not a finite number greater than zero.
    """
    value = (getattr(model, "env", None) or {}).get(MODEL_CPU_CLAIM_ENV)
    if value is None:
        return None
    try:
        claim = float(value)
        if math.isfinite(claim) and claim > 0:
            return claim
    except (ValueError, TypeError, OverflowError):
        pass
    raise ValueError(
        f"{MODEL_CPU_CLAIM_ENV} must be a finite number greater than zero (cores)."
    )
