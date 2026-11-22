"""Deployment resource declarations shared by scheduling and container creation."""

from gpustack.utils.model_env import (
    get_non_model_cpu_claim,
    get_non_model_ram_claim,
)
from gpustack.schemas.models import (
    ModelSpecBase,
    RoleResources,
)

NON_MODEL_WORKLOAD_ENV = "GPUSTACK_NON_MODEL_WORKLOAD"


def non_model_workload_enabled(model: ModelSpecBase) -> bool:
    """Read the deployment's opt-in for a service that loads no model weights."""
    value = (getattr(model, "env", None) or {}).get(NON_MODEL_WORKLOAD_ENV, "")
    return str(value).strip().lower() in {"true", "1", "yes", "on"}


def non_model_workload_resources(model: ModelSpecBase) -> RoleResources:
    """Read user-configured CPU and memory for a non-model workload.

    Values come from the deployment's environment variables. Unset resources
    remain None; no default resource values are added.
    """
    return RoleResources(
        cpu=get_non_model_cpu_claim(model),
        memory=get_non_model_ram_claim(model),
    )
