"""Deployment resource declarations shared by scheduling and container creation."""

from gpustack.policies.utils import get_ram_claim_from_model_env
from gpustack.schemas.models import (
    ModelSpecBase,
    RoleResources,
    ROUTER_DEFAULT_CPU,
    ROUTER_DEFAULT_MEMORY,
)

NON_MODEL_WORKLOAD_ENV = "GPUSTACK_NON_MODEL_WORKLOAD"


def non_model_workload_enabled(model: ModelSpecBase) -> bool:
    """Read the deployment's opt-in for a service that loads no model weights."""
    value = (getattr(model, "env", None) or {}).get(NON_MODEL_WORKLOAD_ENV, "")
    return str(value).strip().lower() in {"true", "1", "yes", "on"}


def non_model_workload_resources(model: ModelSpecBase) -> RoleResources:
    """Use the same RAM declaration for admission and the service's container."""
    ram = get_ram_claim_from_model_env(model)
    return RoleResources(
        cpu=ROUTER_DEFAULT_CPU,
        memory=ram if ram is not None else ROUTER_DEFAULT_MEMORY,
    )
