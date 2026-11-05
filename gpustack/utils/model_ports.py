"""Ordered model port plans shared by candidate admission and Worker allocation."""

import logging
from typing import List, Optional, Tuple

from gpustack.schemas.models import BackendEnum, Model, get_backend
from gpustack.schemas.pd_modes import PDPortScopeEnum
from gpustack.utils.command import (
    find_int_parameter,
    find_last_parameter,
    resolve_executor_backend,
)
from gpustack.utils.model_instance_workers import (
    sglang_component_port_requests,
    sglang_smg_mode,
)
from gpustack.utils.vllm_topology import (
    VLLM_MP_CONNECTING_PORT_COUNT,
    managed_lb_port_requests,
    parse_user_parallelism,
    validate_multinode_topology,
    vllm_lb_mode,
    manual_dp_enabled,
    normalize_vllm_parameters,
)

logger = logging.getLogger(__name__)


class PDPortScopeUnsupportedError(Exception):
    """Role-scoped port bands require a registry shared by all role members."""


def connector_port_requests(
    model: Model, role: Optional[str], allocation: List[int]
) -> List[Tuple[str, int]]:
    """Resolve connector bands in catalog order, leaving unknown widths unresolved."""
    from gpustack.worker.pd_injection import (
        ACCELERATOR_COUNT_KEY,
        band_count_key,
        band_specs_for,
        member_band_width,
    )

    requests = []
    mode = getattr(getattr(model, "disaggregation", None), "mode", None)
    for spec in band_specs_for(model, role, context=f"Role {role!r}"):
        if spec.scope == PDPortScopeEnum.ROLE:
            raise PDPortScopeUnsupportedError(
                f"Role '{role}' (PD mode '{mode}') declares port band '{spec.name}' "
                f"with scope '{PDPortScopeEnum.ROLE.value}'. Role-scoped bands need "
                "a registry shared by the members of one role, which does not exist "
                "yet; allocating one per instance would give each member a different base."
            )
        count = member_band_width(spec, model, allocation)
        if count is None:
            why = (
                "has no accelerators assigned to size it from"
                if band_count_key(spec) == ACCELERATOR_COUNT_KEY
                else "declares no parallelism parameter answering to that name"
            )
            logger.warning(
                "Role '%s' declares port band '%s' with count '%s' but %s. "
                "No ports are reserved for it, leaving its placeholder unresolved.",
                role,
                spec.name,
                spec.count,
                why,
            )
            continue
        requests.append((spec.name, count))
    return requests


def model_port_requests(
    model: Model,
    role: Optional[str],
    allocation: List[int],
    *,
    managed: Optional[bool] = None,
) -> List[Tuple[str, int, Optional[int]]]:
    """Return named bands in first-fit allocation order, including fixed listeners.

    The Worker supplies its persisted orchestration mode; the scheduler plans the
    mode that a new placement will use. Published positional ports are separate
    from this allocation order.
    """
    backend = get_backend(model)
    parameters = (
        normalize_vllm_parameters(model.backend_parameters)
        if backend == BackendEnum.VLLM
        else model.backend_parameters
    )
    if managed is None:
        managed = not (
            backend == BackendEnum.VLLM
            and manual_dp_enabled(getattr(model, 'env', None))
        )
    if managed and role is None and sglang_smg_mode(model):
        return [
            (name, count, None)
            for name, count in sglang_component_port_requests(model, allocation)
        ]
    connector = connector_port_requests(model, role, allocation)
    bands = [
        (name, count, None)
        for name, count in sorted(connector, key=lambda item: item[1], reverse=True)
    ]
    mode = (
        vllm_lb_mode(parameters, tokenized=True)
        if managed and backend == BackendEnum.VLLM
        else "internal"
    )
    if (
        managed
        and role in (None, "prefill", "decode")
        and mode != "internal"
        and min(allocation) > 0
    ):
        requests = [
            (name, count, None)
            for name, count in managed_lb_port_requests(
                parameters, allocation, tokenized=True
            )
        ]
        fixed = find_last_parameter(
            parameters, ["data-parallel-supervisor-port"], tokenized=True
        )
        if mode == "multi_port_external" and fixed is not None:
            requests.insert(0, ("_vllm_multi_port_external_supervisor", 1, int(fixed)))
        return requests + bands

    connecting, cross = 0, 0
    if len(allocation) > 1:
        connecting = 1
        if backend == BackendEnum.VLLM:
            executor = resolve_executor_backend(
                parameters,
                getattr(model, "backend_version", None),
                getattr(model, "image_name", None),
                tokenized=True,
            )
            if executor == "mp":
                topology = validate_multinode_topology(
                    allocation, parse_user_parallelism(parameters, tokenized=True)
                )
                if topology.shape in ("dp_only", "nested"):
                    connecting = VLLM_MP_CONNECTING_PORT_COUNT
                cross = 3
            else:
                dp = find_int_parameter(
                    parameters, ["data-parallel-size", "dp"], tokenized=True
                )
                cross = 1 if dp and dp > 1 else 0
    requests = [("_connecting", connecting, None)] if connecting > 1 else []
    requests += bands
    if connecting == 1:
        requests.append(("_connecting", 1, None))
    requests.append(("_api", 1, None))
    requests.extend((f"_cross_{index}", 1, None) for index in range(cross))
    return requests
