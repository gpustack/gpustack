"""Ordered model port plans shared by admission and Worker allocation.

Port inventory
==============
This inventory covers GPUStack's built-in SGLang/vLLM, PD, SMG and Ray paths.
Keys are conditional on the deployment mode; a custom PD catalog can add keys.
D denotes the global DP/routing rank count; G is the instance's GPU count across
all participating Workers. A reservation is a band, not proof of a live listener.

Executable rules belong to:
* sglang_port_layout.py: native SGLang requests, offsets and resolved bindings.
* vllm_topology.py: managed vLLM native requests and parallelism.
* assets/pd-modes.yaml plus worker/pd_injection.py: PD declarations and widths.
* worker/backends/vllm.py::_build_ray_configuration: the separate Ray range.
Native requests and catalog requests are combined here before allocation.

Persisted native named_ports keys
--------------------------------
Key                                  | Width  | Purpose
_sglang_engine_api                   | 1      | Managed Engine HTTP API.
_sglang_connecting                   | varies | Internal communication band.
_sglang_smg_api                      | 1      | SMG HTTP API.
_sglang_smg_metrics                  | 1      | SMG Prometheus endpoint.
_vllm_external_api                   | D / 1  | Rank APIs / shared Hybrid API.
_vllm_external_rpc                   | 1      | DP coordinator RPC.
_vllm_mp_connecting                  | 10     | MP/DP protection window.
_vllm_external_store                 | D      | Cross-node DP-group TCPStores.
_vllm_multi_port_external_supervisor | 1      | Multi-port external supervisor.

The SGLang internal band is normally one port without DPA, 14 for Python DPA,
and max(14, D+8) for the explicit Rust DPA contract. The vLLM MP band and store
bands are requested only by the applicable topology/LB paths. Both protection
windows can include unused ports. Engine metrics exposed on an Engine HTTP API
do not require a second API reservation.

Persisted PD/catalog named_ports keys
------------------------------------
Key             | Modes / roles                 | Width
kv_side_channel | vLLM NIXL, P and D            | D
bootstrap       | vLLM Mooncake, P              | 1; D for External-LB
bootstrap       | SGLang Mooncake/NIXL, P       | 1
kv_port         | vLLM-Ascend Mooncake, P and D | G
prometheus      | Built-in PD Router            | 1

PD Router HTTP APIs use the generic API slot below. The Engine's DP/LB control
ports and the PD connector's KV/bootstrap ports are distinct reservations.

SGLang internal offsets (I = _sglang_connecting.base)
----------------------------------------------------
Purpose               | Position     | Native path
Rendezvous/base       | I            | Distributed initialization
Tokenizer             | I+1          | DPA
Detokenizer           | I+2          | DPA
RPC                   | I+3          | DPA
Internal metrics      | I+4          | DPA
Controller input      | I+5          | DPA
Load collector        | I+6          | Applicable DPA paths
NCCL argument         | I+7          | Python DPA
Controller handshake  | I+13         | Python multi-node DPA
Rank scheduler inputs | I+7 .. I+6+D | Rust DPA
NCCL argument         | I+7+D        | Rust DPA

These offsets are covered by the parent band, not separate named_ports keys.
Preparation rejects overlapping named bands, so child aliases must not be
persisted alongside their parent as independent reservations. An external
distributed-init override can replace the actual TCPStore; the native base
still controls DPA-derived addresses. See SGLangPortLayout.internal_offsets.

Planning keys not persisted as named_ports in legacy/generic allocation
---------------------------------------------------------------------
Plan key    | Storage        | Purpose
_api        | port, ports[0] | Generic Engine or PD Router HTTP API.
_cross_0    | ports[1]       | Legacy vLLM DP RPC.
_cross_1    | ports[2]       | Legacy vLLM MP master/TCPStore.
_cross_2    | ports[3]       | Legacy vLLM VLLM_PORT compatibility slot.
_connecting | ports[-1]      | Legacy connecting base.

Only applicable paths allocate these slots; some allocated slots are unused.
A wide legacy vLLM _connecting band is persisted as _vllm_mp_connecting.

Ray configuration (R = ray_port_range start; not instance named_ports)
-------------------------------------------------------------------
Position    | Purpose
R           | GCS, primary only.
R+1         | Ray client compatibility slot; not enabled.
R+2         | Dashboard, primary only.
R+3         | Dashboard gRPC compatibility slot.
R+4         | Dashboard agent gRPC.
R+5         | Dashboard agent listen.
R+6         | Metrics export.
R+7         | Node manager.
R+8         | Object manager.
R+9         | Runtime-env agent.
R+10 .. end | Ray worker port range; Ray chooses concrete worker ports.

Runtime-selected ports outside static named_ports reservations
-------------------------------------------------------------
SGLang's layout describes native_nccl, Python dp_scheduler_input, pd_kv_rank,
legacy native_dpa_tcp and external rendezvous ownership. These are not fixed
bands allocated here. vLLM and UCX/Mooncake can also create runtime sockets
outside their static plans. Separate cache-service ports belong to the cache
instance's own lifecycle and reservation ledger, not this model's named_ports.
"""

import logging
from typing import List, Optional, Tuple

from gpustack.schemas.models import BackendEnum, Model, get_backend
from gpustack.schemas.pd_modes import PDPortScopeEnum
from gpustack.utils.command import (
    find_int_parameter,
    find_last_parameter,
    resolve_executor_backend,
)
from gpustack.utils.sglang_port_layout import sglang_port_layout
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
    connector = connector_port_requests(model, role, allocation)
    bands = [
        (name, count, None)
        for name, count in sorted(connector, key=lambda item: item[1], reverse=True)
    ]
    if backend == BackendEnum.SGLANG:
        requests = sglang_port_layout(
            model, role, allocation, managed=managed
        ).port_requests()
        # Generic APIs follow connector bands; component layouts publish their
        # native order and place connector bands afterwards.
        api_index = next(
            (index for index, (name, _, _) in enumerate(requests) if name == '_api'),
            len(requests),
        )
        return requests[:api_index] + bands + requests[api_index:]
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
