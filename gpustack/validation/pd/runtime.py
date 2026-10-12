"""Resolve model metadata and validate accelerator eligibility for PD modes."""

import logging
from typing import Any, Optional, Union

from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.api.exceptions import BadRequestException
from gpustack.schemas.models import (
    BackendEnum,
    Model,
    ModelCreate,
    ModelSpecBase,
    ModelUpdate,
)
from gpustack.server.cluster_accelerators import cluster_vendors
from gpustack.server.pd_mode_catalog import get_pd_mode
from gpustack.server.pd_mode_resolver import resolve_pd_mode
from gpustack.validation.pd.common import model_field
from gpustack.validation.pd.environment import environment_pairing_needs_config

logger = logging.getLogger(__name__)


async def resolve_pairing_model_config(
    model_in: Union[ModelCreate, ModelUpdate, ModelSpecBase],
    stored: Optional[Model] = None,
) -> Optional[Any]:
    """Resolve model-dependent NIXL factors only when the spec needs metadata.

    Args:
        model_in: The submitted deployment specification.
        stored: The stored deployment supplying omitted update fields.

    Returns:
        The source configuration, or None when it is unavailable or unnecessary.
    """
    if not environment_pairing_needs_config(model_in, stored=stored):
        return
    from gpustack.scheduler.calculator import get_pretrained_config_with_workers

    # A sparse update may inherit source, role and runtime fields from storage.
    effective = (
        stored.model_copy(
            update={name: getattr(model_in, name) for name in model_in.model_fields_set}
        )
        if stored is not None
        else model_in
    )
    try:
        config = await get_pretrained_config_with_workers(effective)
    except Exception:
        # An offline or worker-only source cannot justify a compatibility refusal.
        logger.debug("Model configuration unavailable for NIXL environment pairing")
        return
    return config


async def validate_pd_mode_runtime(
    session: AsyncSession,
    model_in: Union[ModelCreate, ModelUpdate, ModelSpecBase],
    *,
    cluster_id: Optional[int] = None,
    stored: Optional[Model] = None,
) -> None:
    """Reject a pd mode no accelerator in the cluster can run.

    Every built-in recipe is accelerator-specific: `vllm-ascend-mooncake`
    injects an Ascend-only connector plus HCCL variables, and the NVIDIA
    recipes inject connectors no other runtime can read. Injecting one into
    the wrong accelerator fails inside the connector rather than at submit
    time. `PDModeRuntimeFilter` also drops the mismatched workers during
    scheduling; this check exists so the answer is a readable refusal instead
    of an empty candidate list.

    `custom` is never rejected -- it declares no `gpu_filters`, so an
    unsupported engine × accelerator pair means "no built-in recipe", never
    "no PD".

    Shares `resolve_pd_mode` with the resolve endpoint the form reads, so the
    API cannot refuse a combination the form just told the user was fine.

    Accelerator-less clusters are left to scheduling: a cluster whose workers
    have not reported devices yet must not be judged as unable to run
    anything.
    """
    disaggregation = model_field(model_in, stored, "disaggregation")
    if not disaggregation:
        return

    mode_name = disaggregation.mode.value
    mode = get_pd_mode(mode_name)
    if mode is None or mode.gpu_filters is None or not mode.gpu_filters.vendor:
        return

    effective_cluster_id = cluster_id or getattr(model_in, "cluster_id", None)
    vendors = await cluster_vendors(session, effective_cluster_id)
    if not vendors:
        return

    backend = model_field(model_in, stored, "backend") or BackendEnum.VLLM.value
    resolution = resolve_pd_mode(
        backend, vendors, vendor=getattr(disaggregation, "vendor", None)
    )
    verdict = next(
        (option for option in resolution.options if option.name == mode_name), None
    )
    if verdict is not None and not verdict.eligible:
        raise BadRequestException(
            message=(
                f"pd mode '{mode_name}' cannot run here: "
                f"{verdict.ineligible_reason}"
            )
        )
