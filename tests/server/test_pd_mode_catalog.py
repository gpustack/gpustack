import pytest

from gpustack.schemas.models import PD_MODE_BACKENDS, BackendEnum, PDModeEnum
from gpustack.schemas.pd_modes import (
    PDInjectTargetEnum,
    PDKVLeaseTargetEnum,
    PDMembershipAPI,
    PDTransferMetrics,
    PDMode,
    PDNetDevicePlaneEnum,
    PDPeerStyleEnum,
    PDPortScopeEnum,
    PDRouter,
    PDRouterProtocolEnum,
    PDTensorParallelPairingEnum,
)
from gpustack.server.pd_mode_catalog import (
    PDModeCatalogError,
    get_composed_cache,
    get_kv_lease,
    get_kv_leases,
    get_pd_mode,
    get_pd_modes,
    get_transfer_metrics,
    load_pd_mode_catalog,
    load_pd_modes,
    parse_pd_mode_catalog,
)


def _document(modes, kv_leases=None, kv_transfer_metrics=None, composed_cache=None):
    """A minimal catalog document whose mode names satisfy the enum
    assertion, so a test can isolate the assertion it is after.

    Every declared connector gets an all-null transfer-metrics entry unless
    the caller supplies one: the loader requires the two registries to cover
    the same connectors, and a test about something else should not have to
    restate that."""
    entries = {mode["name"]: mode for mode in modes}
    for name, backends in PD_MODE_BACKENDS.items():
        entries.setdefault(name, {"name": name, "backends": list(backends)})
    leases = kv_leases or []
    if kv_transfer_metrics is None:
        kv_transfer_metrics = [{"connector": lease["connector"]} for lease in leases]
    document = {
        "kv_leases": leases,
        "kv_transfer_metrics": kv_transfer_metrics,
        "modes": list(entries.values()),
    }
    if composed_cache is not None:
        document["composed_cache"] = composed_cache
    return document


def test_catalog_asset_loads():
    modes = load_pd_modes(reload=True)
    assert modes, "bundled pd-modes.yaml should yield at least one mode"
    # Every shipped entry parses into a typed model, not a raw dict.
    assert all(isinstance(mode, PDMode) for mode in modes)


def test_catalog_names_are_exactly_the_enum():
    """The load-time assertion's happy path: the catalog is looked up by
    mode name, so the two sets have to be equal, not merely overlapping."""
    modes = load_pd_modes()
    assert {mode.name for mode in modes} == {mode.value for mode in PDModeEnum}


def test_enum_mismatch_fails_the_load():
    """The mismatch that already happened once: `ascend-mooncake` where the
    enum says `vllm-ascend-mooncake`. It has to fail loudly, because a table
    miss injects nothing and the deployment still comes up."""
    document = _document([])
    for entry in document["modes"]:
        if entry["name"] == PDModeEnum.VLLM_ASCEND_MOONCAKE.value:
            entry["name"] = "ascend-mooncake"
    with pytest.raises(PDModeCatalogError) as excinfo:
        parse_pd_mode_catalog(document)
    message = str(excinfo.value)
    assert "PDModeEnum" in message
    assert "vllm-ascend-mooncake" in message
    assert "ascend-mooncake" in message


def test_missing_and_extra_names_both_fail_the_load():
    with pytest.raises(PDModeCatalogError):
        parse_pd_mode_catalog({"modes": [{"name": PDModeEnum.CUSTOM.value}]})
    with pytest.raises(PDModeCatalogError):
        parse_pd_mode_catalog(
            _document([{"name": "vllm-moriio", "backends": [BackendEnum.VLLM.value]}])
        )


def test_catalog_backends_agree_with_the_validation_table():
    """PD_MODE_BACKENDS exists only so request validation need not read the
    catalog. The catalog is authoritative; this is the check that keeps the
    copy honest."""
    for mode in load_pd_modes():
        assert sorted(mode.backends) == sorted(PD_MODE_BACKENDS[mode.name])
    # `custom` injects nothing, so it constrains nothing.
    assert get_pd_mode(PDModeEnum.CUSTOM.value).backends == []


def test_backends_mismatch_fails_the_load():
    document = _document(
        [{"name": PDModeEnum.VLLM_NIXL.value, "backends": [BackendEnum.SGLANG.value]}]
    )
    with pytest.raises(PDModeCatalogError) as excinfo:
        parse_pd_mode_catalog(document)
    message = str(excinfo.value)
    assert "PD_MODE_BACKENDS" in message
    assert "vllm-nixl" in message


def test_every_built_in_recipe_declares_its_accelerator():
    """The shipped catalog must not contain a recipe that would be offered on
    every accelerator. This is the assertion that keeps the AMD case closed:
    the shipped catalog has no AMD recipe, so an unconstrained NVIDIA recipe
    would be selectable there and fail inside the connector."""
    for mode in get_pd_modes():
        injects = bool(mode.roles) or (
            mode.router is not None and mode.router.protocol.value != "user_provided"
        )
        if injects:
            assert (
                mode.gpu_filters is not None and mode.gpu_filters.vendor
            ), f"{mode.name} injects configuration but names no accelerator"


def test_every_built_in_recipe_names_its_transport():
    """The derived one-liner shows the transport alone — the engine is already
    named in the field above it, so "vLLM + NIXL" next to a "Backend: vLLM"
    field repeats itself. `display_name` keeps the engine because the picker
    lists several engines' recipes side by side and there it has to say *which*
    NIXL."""
    for mode in get_pd_modes():
        injects = bool(mode.roles) or (
            mode.router is not None and mode.router.protocol.value != "user_provided"
        )
        if injects:
            assert mode.transport, f"{mode.name} names no transport"
            assert mode.transport not in mode.display_name.split(" + ")[0], (
                f"{mode.transport} should be the transport alone, not the "
                f"engine-qualified name"
            )


def test_transport_is_unique_per_engine_accelerator_pair():
    """The picker labels rows by transport alone, and it hides recipes that do
    not fit the chosen engine and accelerator. So two recipes in one cell with
    the same transport would render as two identical, indistinguishable rows.

    Holds today because SGLang-on-NVIDIA is the only cell with two candidates
    and they use different transports — this is the guard that keeps a third
    recipe from breaking it silently."""
    from collections import defaultdict

    cells = defaultdict(list)
    for mode in get_pd_modes():
        if not mode.transport:
            continue
        for backend in mode.backends:
            for vendor in mode.gpu_filters.vendor if mode.gpu_filters else []:
                cells[(backend, vendor, mode.transport)].append(mode.name)
    for cell, names in cells.items():
        assert len(names) == 1, f"{cell} is claimed by {names}"


def test_custom_names_no_transport():
    """It has none of its own: the user supplies the connector."""
    assert get_pd_mode(PDModeEnum.CUSTOM.value).transport is None


def test_custom_declares_no_accelerator_constraint():
    """`custom` is the escape hatch for every engine × accelerator pair we
    ship no recipe for. Constraining it would turn "no built-in recipe" into
    "no PD"."""
    custom = get_pd_mode(PDModeEnum.CUSTOM.value)
    assert custom.gpu_filters is None or not custom.gpu_filters.vendor
    assert not custom.backends


def test_an_undeclared_accelerator_fails_the_load():
    """A recipe that injects a connector but names no accelerator would be
    offered on every one of them -- exactly how three NVIDIA-only recipes came
    to be selectable on Ascend."""
    document = _document(
        [
            {
                "name": PDModeEnum.VLLM_NIXL.value,
                "backends": [BackendEnum.VLLM.value],
                "roles": {"prefill": {}, "decode": {}},
            }
        ]
    )
    with pytest.raises(PDModeCatalogError) as excinfo:
        parse_pd_mode_catalog(document)
    assert "gpu_filters" in str(excinfo.value)
    assert "vllm-nixl" in str(excinfo.value)


def test_constraining_custom_fails_the_load():
    """The other half: `custom` injects nothing, so a constraint on it only
    removes the escape hatch."""
    document = _document(
        [
            {
                "name": PDModeEnum.CUSTOM.value,
                "backends": [],
                "gpu_filters": {"vendor": "nvidia"},
            }
        ]
    )
    with pytest.raises(PDModeCatalogError) as excinfo:
        parse_pd_mode_catalog(document)
    assert "custom" in str(excinfo.value)


def test_tensor_parallel_direction_is_the_connectors_to_declare():
    """One rule for every recipe rejects a working Ascend deployment:
    Huawei's Ascend reference is prefill TP4 / decode TP1, which NIXL's
    "decode at least prefill" forbids. So the direction is a recipe
    declaration. The default is NIXL's, and `custom`, which injects no
    connector, has no direction to vouch for."""
    nixl = get_pd_mode(PDModeEnum.VLLM_NIXL.value)
    ascend = get_pd_mode(PDModeEnum.VLLM_ASCEND_MOONCAKE.value)
    custom = get_pd_mode(PDModeEnum.CUSTOM.value)
    assert nixl.pairing.tensor_parallel is PDTensorParallelPairingEnum.DECODE_GE_PREFILL
    assert (
        ascend.pairing.tensor_parallel
        is not PDTensorParallelPairingEnum.DECODE_GE_PREFILL
    )
    assert custom.pairing.tensor_parallel is PDTensorParallelPairingEnum.ANY


def test_exactly_one_recipe_is_preferred_per_engine_accelerator_pair():
    """`preferred` is the tie-break for a cell with more than one candidate.
    Two preferred recipes in one cell would make the derived answer depend on
    catalog order."""
    from collections import defaultdict

    cells = defaultdict(list)
    for mode in get_pd_modes():
        if not mode.preferred:
            continue
        for backend in mode.backends:
            for vendor in mode.gpu_filters.vendor if mode.gpu_filters else []:
                cells[(backend, vendor)].append(mode.name)
    for cell, names in cells.items():
        assert len(names) == 1, f"{cell} has multiple preferred recipes: {names}"


def test_backends_use_the_backend_enum_spelling():
    """The table is keyed by BackendEnum values and a role's `backend` is
    compared against these, so "vllm" instead of "vLLM" would silently
    never match."""
    known = {backend.value for backend in BackendEnum}
    for mode in load_pd_modes():
        assert set(mode.backends) <= known


# ---------------------------------------------------------------------------
# Where injected content lands differs per engine.
# ---------------------------------------------------------------------------


def test_injection_target_is_per_engine_not_uniform():
    nixl = get_pd_mode(PDModeEnum.VLLM_NIXL.value)
    ascend = get_pd_mode(PDModeEnum.VLLM_ASCEND_MOONCAKE.value)
    sglang = get_pd_mode(PDModeEnum.SGLANG_MOONCAKE.value)

    # NIXL: the side-channel port is an env var, and the host is a bind
    # address the engine defaults to localhost.
    nixl_prefill = nixl.role("prefill")
    assert nixl_prefill.ports[0].name == "kv_side_channel"
    assert nixl_prefill.ports[0].inject_to == PDInjectTargetEnum.ENV
    assert nixl_prefill.env["VLLM_NIXL_SIDE_CHANNEL_HOST"] == "{{worker_ip}}"
    assert (
        nixl_prefill.env["VLLM_NIXL_SIDE_CHANNEL_PORT"] == "{{ports.kv_side_channel}}"
    )
    assert nixl_prefill.env["UCX_NET_DEVICES"] == "{{net_device}}"

    # Mooncake on Ascend: the same port is a field of the connector
    # descriptor, which renders into --kv-transfer-config. "The transfer
    # config carries no addresses" holds for NIXL only.
    ascend_prefill = ascend.role("prefill")
    assert ascend_prefill.ports[0].name == "kv_port"
    assert ascend_prefill.ports[0].inject_to == PDInjectTargetEnum.ARGS
    assert ascend_prefill.connector["kv_port"] == "{{ports.kv_port}}"
    assert "VLLM_NIXL_SIDE_CHANNEL_PORT" not in ascend_prefill.env
    # Ascend's handshake variables have nothing in common with NIXL's.
    assert ascend_prefill.env["HCCL_IF_IP"] == "{{worker_ip}}"
    assert [key for key in ascend_prefill.env if key.endswith("_SOCKET_IFNAME")] == [
        "HCCL_SOCKET_IFNAME",
        "GLOO_SOCKET_IFNAME",
        "TP_SOCKET_IFNAME",
    ]

    # SGLang: a command-line flag, and no connector descriptor at all.
    sglang_prefill = sglang.role("prefill")
    assert sglang_prefill.ports[0].inject_to == PDInjectTargetEnum.ARGS
    assert "--disaggregation-bootstrap-port" in sglang_prefill.args
    assert "{{ports.bootstrap}}" in sglang_prefill.args
    assert sglang_prefill.connector == {}


def test_sglang_mooncake_roles_survive_a_tcp_transport():
    """Two engine-side defaults make a Mooncake group over TCP fail under any
    real load, and both are turned off by declaration rather than by asking
    the operator.

    Unpooled TCP opens a connection per KV transfer, which exhausts the host's
    ephemeral ports within seconds, and the first transfer failure blacklists
    the decode session for the life of the process. The pool applies to
    whichever side dials; the probe that clears the blacklist only runs on
    prefill, which is where the set lives.
    """
    mode = get_pd_mode(PDModeEnum.SGLANG_MOONCAKE.value)

    for role in ("prefill", "decode"):
        assert mode.role(role).env["MC_TCP_ENABLE_CONNECTION_POOL"] == "1"

    assert mode.role("prefill").env["SGLANG_ENABLE_FAILED_SESSION_PROBE"] == "true"
    assert "SGLANG_ENABLE_FAILED_SESSION_PROBE" not in mode.role("decode").env


def test_files_is_a_declarable_injection_target():
    """The third target, needed by connectors that read a config file and
    nothing else (Mooncake's transfer engine reads only the JSON that
    MOONCAKE_CONFIG_PATH points at). No shipped entry uses it yet, so the
    schema is what has to prove it."""
    mode = PDMode(
        name="file-fed",
        roles={
            "prefill": {
                "ports": [{"name": "kv_port", "inject_to": "files"}],
                "env": {"MOONCAKE_CONFIG_PATH": "/tmp/gpustack-pd-mooncake.json"},
                "files": {
                    "/tmp/gpustack-pd-mooncake.json": '{"port": {{ports.kv_port}}}'
                },
            }
        },
    )
    port = mode.role("prefill").ports[0]
    assert port.inject_to == PDInjectTargetEnum.FILES
    assert (
        "{{ports.kv_port}}"
        in mode.role("prefill").files["/tmp/gpustack-pd-mooncake.json"]
    )


def test_a_band_must_be_consumed_where_it_says_it_is():
    """A wrong inject_to is exactly the NIXL-vs-Mooncake mistake this
    schema exists to prevent, so it cannot sit there looking plausible."""
    with pytest.raises(ValueError, match="declares inject_to 'env'"):
        PDMode(
            name="misdeclared",
            roles={
                "prefill": {
                    "ports": [{"name": "kv_port", "inject_to": "env"}],
                    "args": ["--kv-port", "{{ports.kv_port}}"],
                }
            },
        )


# ---------------------------------------------------------------------------
# A band's width is the connector's rule, not a formula.
# ---------------------------------------------------------------------------


def test_port_band_count_is_declared_by_the_connector():
    ascend = get_pd_mode(PDModeEnum.VLLM_ASCEND_MOONCAKE.value)
    nixl = get_pd_mode(PDModeEnum.VLLM_NIXL.value)

    # kv_port is a base address and Mooncake binds one port per *worker rank*
    # (rank 0 -> base+0, rank 1 -> base+2), so the width is the member's card
    # count, resolved at allocation time. Reading it as the tensor-parallel
    # size is indistinguishable when dp is 1 and under-reserves by a factor of
    # dp on every DP member.
    for role in ("prefill", "decode"):
        band = ascend.role(role).ports[0]
        assert band.count == "{{accelerator_count}}"
        assert band.scope == PDPortScopeEnum.INSTANCE
    # NIXL's side channel is offset per DP index instead; no shipped recipe
    # uses local DP, so the declared width is one.
    assert nixl.role("prefill").ports[0].count == 1


def test_shorthand_port_declaration_expands():
    mode = PDMode(
        name="shorthand",
        roles={
            "prefill": {
                "ports": ["kv_side_channel"],
                "env": {"P": "{{ports.kv_side_channel}}"},
            }
        },
    )
    band = mode.role("prefill").ports[0]
    assert band.name == "kv_side_channel"
    assert band.count == 1
    assert band.inject_to == PDInjectTargetEnum.ENV


def test_port_band_count_rejects_arithmetic():
    with pytest.raises(ValueError, match="neither an integer nor a single placeholder"):
        PDMode(
            name="bad-count",
            roles={
                "prefill": {
                    "ports": [{"name": "kv", "count": "{{tp}}*2"}],
                    "env": {"P": "{{ports.kv}}"},
                }
            },
        )


# ---------------------------------------------------------------------------
# Cross-role references.
# ---------------------------------------------------------------------------


def test_cross_role_references_render_both_directions():
    """On Ascend both sides carry both sides' parallelism —
    prefill's connector config names decode's TP and vice versa. NIXL has no
    equivalent coupling, which is why this is declarable rather than
    hardcoded."""
    ascend = get_pd_mode(PDModeEnum.VLLM_ASCEND_MOONCAKE.value)
    for role in ("prefill", "decode"):
        extra = ascend.role(role).connector["kv_connector_extra_config"]
        assert extra["prefill"] == {
            "tp_size": "{{roles.prefill.tensor_parallel_size}}",
            "dp_size": "{{roles.prefill.data_parallel_size}}",
        }
        assert extra["decode"] == {
            "tp_size": "{{roles.decode.tensor_parallel_size}}",
            "dp_size": "{{roles.decode.data_parallel_size}}",
        }
    # The NIXL path declares no cross-role reference at all.
    nixl = get_pd_mode(PDModeEnum.VLLM_NIXL.value)
    assert "roles." not in repr(nixl.roles)


def test_cross_role_reference_to_an_undeclared_role_fails():
    with pytest.raises(ValueError, match="undeclared role 'decode'"):
        PDMode(
            name="dangling",
            roles={"prefill": {"connector": {"tp": "{{roles.decode.tp_size}}"}}},
        )


def test_placeholders_reject_inner_spaces():
    """The renderer's pattern takes no spaces, so "{{ worker_ip }}" would
    reach the container verbatim — measured as a ZMQError on a bind
    address."""
    with pytest.raises(ValueError, match="malformed placeholder"):
        PDMode(name="spaced", roles={"prefill": {"env": {"H": "{{ worker_ip }}"}}})


# ---------------------------------------------------------------------------
# Router capabilities.
# ---------------------------------------------------------------------------


def test_router_capabilities_are_declared_per_mode():
    nixl = get_pd_mode(PDModeEnum.VLLM_NIXL.value)
    assert nixl.router.protocol == PDRouterProtocolEnum.TWO_HOP
    assert nixl.router.capabilities.metrics is True
    assert nixl.router.capabilities.models_endpoint is True
    assert nixl.router.capabilities.kv_expired_metric is True
    assert nixl.router.health_path == "/health"

    # Ascend runs the same router as the vLLM recipe, so it serves /metrics,
    # /v1/models and /health. What stays false is the connector's KV-expiry
    # counter: Mooncake exports none, and a router with metrics does not
    # conjure an engine-side counter that was never written.
    ascend = get_pd_mode(PDModeEnum.VLLM_ASCEND_MOONCAKE.value)
    assert ascend.router.capabilities.metrics is True
    assert ascend.router.capabilities.models_endpoint is True
    assert ascend.router.capabilities.kv_expired_metric is False
    assert ascend.router.health_path == "/health"


def test_router_capabilities_default_to_absent():
    """An undeclared endpoint must read as absent, not assumed present."""
    mode = PDMode(
        name="bare",
        router={
            "protocol": "two_hop",
            "command": ["router"],
            "peers": {"style": "repeated_flag"},
        },
    )
    assert mode.router.capabilities.metrics is False
    assert mode.router.capabilities.models_endpoint is False
    assert mode.router.capabilities.kv_expired_metric is False
    assert mode.router.health_path is None


def test_router_peer_styles_and_prometheus_band():
    nixl = get_pd_mode(PDModeEnum.VLLM_NIXL.value)
    ascend = get_pd_mode(PDModeEnum.VLLM_ASCEND_MOONCAKE.value)
    sglang = get_pd_mode(PDModeEnum.SGLANG_MOONCAKE.value)
    custom = get_pd_mode(PDModeEnum.CUSTOM.value)

    assert nixl.router.peers.style == PDPeerStyleEnum.REPEATED_FLAG
    assert nixl.router.peers.prefill == {
        "flag": "--prefill",
        "value": "http://{{peer.ip}}:{{peer.port}}",
    }
    # The router's own Prometheus port is fixed at 29000 upstream and
    # always binds, so a second group's router on one host panics; it is a
    # managed band like any other.
    assert [band.name for band in nixl.router.ports] == ["prometheus"]
    assert "{{ports.prometheus}}" in nixl.router.command
    # And the exposition has to be bound to the worker's address, not to
    # the default loopback: on loopback the exposition is served only on-host,
    # which puts the ratio's denominator out of reach of the collector. Same
    # class of bug as VLLM_NIXL_SIDE_CHANNEL_HOST, same fix.
    assert (
        nixl.router.command[nixl.router.command.index("--prometheus-host") + 1]
        == "{{worker_ip}}"
    )
    # Fast failure detection is the circuit breaker's, not the health
    # check's: the breaker runs on the request path and sees a dead worker at
    # real traffic rate, while a short health-check interval lands near the
    # engine's HTTP keep-alive and ejects healthy workers on a stale socket.
    # See tests/worker/test_pd_router.py for the full argument.
    for flag in ("--cb-failure-threshold", "--retry-max-retries"):
        assert flag in nixl.router.command
    assert not [c for c in nixl.router.command if str(c).startswith("--health")]

    # Hosts and ports as two parallel flags.
    # The `parallel_lists` renderer is still supported and still tested,
    # through a synthetic mode in tests/worker/test_pd_router.py, since no
    # shipped mode uses it.
    assert ascend.router.peers.style == PDPeerStyleEnum.REPEATED_FLAG
    assert ascend.router.peers.prefill == {
        "flag": "--prefill",
        "value": "http://{{peer.ip}}:{{peer.port}}",
    }
    # `nixl` on a Mooncake-transport mode is deliberate: the flag names the
    # wire protocol shape, not the transport. Choosing `mooncake` makes the
    # router wait forever on a bootstrap server vllm-ascend does not run, and
    # no request ever completes.
    assert ascend.router.command[ascend.router.command.index("--kv-connector") + 1] == (
        "nixl"
    )
    assert [band.name for band in ascend.router.ports] == ["prometheus"]

    # A prefill peer carries THAT PEER'S bootstrap band as a second positional
    # value. `peer.ports.` and not `ports.`: the latter is the deployment scope
    # and means the router's own band of that name, so it renders verbatim into
    # the address and a two-prefill group cannot be expressed correctly.
    assert "{{peer.ports.bootstrap}}" in sglang.router.peers.prefill["value"]
    assert "{{ports.bootstrap}}" not in sglang.router.peers.prefill["value"]

    # custom supplies nothing: image, command and ports are the user's.
    assert custom.router.protocol == PDRouterProtocolEnum.USER_PROVIDED
    assert custom.router.command == []
    assert custom.router.image is None
    assert custom.roles == {}


def test_user_provided_router_must_not_declare_a_command():
    with pytest.raises(ValueError, match="user_provided router"):
        PDMode(
            name="contradiction",
            router={"protocol": "user_provided", "command": ["whatever"]},
        )


def test_unknown_peer_style_fails_the_load():
    """A new wire style needs a renderer branch, so an unknown value must
    not be quietly accepted and rendered wrong."""
    with pytest.raises(ValueError):
        PDMode(
            name="future",
            router={
                "protocol": "two_hop",
                "command": ["router"],
                "peers": {"style": "grpc_delegate"},
            },
        )


# ---------------------------------------------------------------------------
# KV lease / abort window per connector.
# ---------------------------------------------------------------------------


def test_kv_lease_windows_are_per_connector():
    leases = get_kv_leases()

    # NIXL: a lease with heartbeat renewal, in the connector's extra
    # config, and the one window with a Prometheus counter.
    nixl = leases["nixl"]
    assert nixl.param == "kv_lease_duration"
    assert nixl.inject_to == PDKVLeaseTargetEnum.CONNECTOR_EXTRA_CONFIG
    assert nixl.engine_default == 30
    assert nixl.gpustack_default == 60
    assert nixl.expired_metric == "vllm:nixl_num_kv_expired_reqs"
    assert nixl.settable is True

    # Mooncake: an env var, 16x the window, and no Prometheus counter at
    # all — an expiry is only visible in the engine log.
    mooncake = leases["mooncake"]
    assert mooncake.param == "VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT"
    assert mooncake.inject_to == PDKVLeaseTargetEnum.ENV
    assert mooncake.engine_default == 480
    assert mooncake.gpustack_default == 60
    assert mooncake.expired_metric is None

    # MoRIIO: the window that reclaims prefill's blocks is `defer_timeout`
    # and it IS settable. The 3600 read-abort constant is a different
    # deadline; tracking that one instead read as "an hour, unshortenable",
    # which was wrong on both halves.
    moriio = leases["moriio"]
    assert moriio.param == "defer_timeout"
    assert moriio.inject_to == PDKVLeaseTargetEnum.CONNECTOR_EXTRA_CONFIG
    assert moriio.engine_default == 60
    assert moriio.settable is True
    # No shipped mode uses MoRIIO, so nothing is injected and no platform
    # default is claimed.
    assert moriio.gpustack_default is None
    assert all(mode.kv_lease is not moriio for mode in load_pd_modes())

    # Both SGLang backends share one window, because the timeout lives in
    # the base both extend: nixl/conn.py and mooncake/conn.py each call
    # CommonKVSender._check_bootstrap_timeout(). Declaring only one of them
    # as settable was the error this asserts against.
    for name in ("sglang-mooncake", "sglang-nixl"):
        lease = leases[name]
        assert lease.param == "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT", name
        assert lease.inject_to == PDKVLeaseTargetEnum.ENV, name
        assert lease.settable is True, name
        assert lease.engine_default == 300, name
        # Deliberately unset: this is not a lease but "how long prefill
        # waits for decode's KV indices", so compressing it to the vLLM
        # connectors' 60s would fail healthy requests whose decode queued.
        assert lease.gpustack_default is None, name
        assert lease.expired_metric is None, name


def test_modes_resolve_their_connector_window():
    assert get_pd_mode(PDModeEnum.VLLM_NIXL.value).kv_lease is get_kv_lease("nixl")
    assert get_pd_mode(PDModeEnum.VLLM_ASCEND_MOONCAKE.value).kv_lease is get_kv_lease(
        "mooncake"
    )
    assert get_pd_mode(PDModeEnum.SGLANG_NIXL.value).kv_lease is get_kv_lease(
        "sglang-nixl"
    )
    # custom configures nothing, including the window.
    assert get_pd_mode(PDModeEnum.CUSTOM.value).kv_lease is None
    # The declared window is what the connector's own config template
    # renders from.
    nixl = get_pd_mode(PDModeEnum.VLLM_NIXL.value)
    extra = nixl.role("prefill").connector["kv_connector_extra_config"]
    assert extra["kv_lease_duration"] == "{{kv_lease_duration}}"


def test_unknown_kv_lease_reference_fails_the_load():
    with pytest.raises(PDModeCatalogError, match="references kv_lease"):
        parse_pd_mode_catalog(
            _document([{"name": PDModeEnum.CUSTOM.value, "kv_lease": "no-such"}])
        )


def test_expired_metric_claim_must_match_the_window():
    """Two spellings of one fact: the router capability is what the metrics
    collector reads, the lease carries the metric's name."""
    document = _document(
        [
            {
                "name": PDModeEnum.VLLM_NIXL.value,
                "backends": PD_MODE_BACKENDS[PDModeEnum.VLLM_NIXL.value],
                "kv_lease": "mooncake",
                "router": {
                    "protocol": "two_hop",
                    "command": ["router"],
                    "peers": {"style": "repeated_flag"},
                    "capabilities": {"kv_expired_metric": True},
                },
            }
        ],
        kv_leases=[
            {
                "connector": "mooncake",
                "param": "VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT",
                "inject_to": "env",
                "engine_default": 480,
            }
        ],
    )
    with pytest.raises(PDModeCatalogError, match="kv_expired_metric"):
        parse_pd_mode_catalog(document)


def test_unsettable_window_must_not_claim_an_injection_target():
    with pytest.raises(PDModeCatalogError, match="not settable"):
        parse_pd_mode_catalog(
            _document(
                [],
                kv_leases=[
                    {
                        "connector": "moriio",
                        "param": "VLLM_MORI_READ_ABORT_REQUEST_TIMEOUT",
                        "inject_to": "env",
                        "settable": False,
                        "engine_default": 3600,
                    }
                ],
            )
        )


# ---------------------------------------------------------------------------
# Transfer counters.
# ---------------------------------------------------------------------------


def test_the_read_side_is_declared_because_reading_the_wrong_one_inverts_it():
    """Which side to read is the connector's semantics, not its spelling, so
    normalization cannot absorb it. Read the wrong side and a healthy pair
    reports "no KV ever moved": the exact alarm this exists to raise, fired at
    a deployment that is fine.
    """
    nixl = get_transfer_metrics("nixl")
    assert nixl.read_from_role == "decode"

    for connector in ("sglang-mooncake", "sglang-nixl"):
        assert get_transfer_metrics(connector).read_from_role == "decode", connector


def test_the_metric_names_are_gone_from_this_catalog():
    """The regression this pins, and it is about a *removal*.

    The names moved to `metrics_config.yaml`, where the aggregator normalizes
    every engine's spelling and units onto one set of `gpustack:pd_*` series.
    A second copy here would be two files to edit for one rename -- and worse,
    nothing reads this one any more, so it would look authoritative and change
    nothing. Adding a metric name back is the mistake this catches.
    """
    fields = set(PDTransferMetrics.model_fields)
    assert fields == {"connector", "read_from_role"}, fields


def test_sglang_is_read_on_decode_even_though_prefill_is_the_sender():
    """The case where "who moves the bytes" and "who can be read" part.

    SGLang pushes, so prefill is the sender and owns the byte, speed and wire
    -time families. It writes none of them -- nor the handshake and allocation
    timings that share their code path -- unless the connector hands it a
    transfer size: `compute_and_observe_kv_transfer_metrics` returns early on
    `transfer_total_bytes is None` and takes the whole group with it. Decode's
    `set_decode_transfer_queue_entry_time` records `bootstrap_ms` and
    `alloc_ms` unconditionally.

    A running pair therefore exposes no `kv_transfer_*` family on prefill and
    two on decode. Reading prefill reports a working group as never having
    moved any KV, which is why this is decode despite prefill being the
    sender.
    """
    for connector in ("sglang-mooncake", "sglang-nixl"):
        metrics = get_transfer_metrics(connector)
        assert metrics.read_from_role == "decode", connector


def test_modes_resolve_their_connector_counters_through_one_reference():
    """The connector id names the transport once; the lease window and the
    counters both hang off it, so the two cannot come to disagree."""
    for name in (PDModeEnum.VLLM_NIXL.value, PDModeEnum.SGLANG_NIXL.value):
        mode = get_pd_mode(name)
        assert mode.transfer_metrics is get_transfer_metrics(mode.kv_lease.connector)
    assert get_pd_mode(PDModeEnum.CUSTOM.value).transfer_metrics is None


def test_a_connector_missing_from_the_counter_registry_fails_the_load():
    """An all-null entry declares "exports nothing"; a missing entry behaves
    identically at runtime while meaning nobody looked."""
    with pytest.raises(PDModeCatalogError, match="Missing: \\['nixl'\\]"):
        parse_pd_mode_catalog(
            _document(
                [],
                kv_leases=[
                    {
                        "connector": "nixl",
                        "param": "kv_lease_duration",
                        "inject_to": "connector_extra_config",
                    }
                ],
                kv_transfer_metrics=[],
            )
        )


def test_the_denominator_names_left_this_catalog():
    """The router's request-counter names are not declared here.

    They live in `metrics_config.yaml`, where the worker's aggregator
    normalizes every router's spelling onto `gpustack:pd_router_*`. A copy in
    this catalog would be read by nothing: a declaration that looks
    authoritative and changes nothing.

    What stays on the router is what normalization cannot absorb: whether it
    serves an exposition at all (`capabilities.metrics`) and where
    (`ports`, the band the API port is not on).
    """
    fields = set(PDRouter.model_fields)
    assert "request_metrics" not in fields
    assert {"capabilities", "ports", "membership_api"} <= fields


def test_catalog_is_cached_and_reloadable():
    first = load_pd_mode_catalog()
    assert load_pd_mode_catalog() is first
    assert load_pd_mode_catalog(reload=True) is not first


def test_mode_lookup_is_case_insensitive():
    assert get_pd_mode("VLLM-NIXL") is not None
    assert get_pd_mode("no-such-mode") is None
    assert get_pd_mode("") is None


def test_malformed_document_fails_the_load():
    with pytest.raises(PDModeCatalogError, match="must be a mapping"):
        parse_pd_mode_catalog([{"name": "vllm-nixl"}])
    with pytest.raises(PDModeCatalogError, match="must be mappings"):
        parse_pd_mode_catalog({"modes": ["vllm-nixl"]})


def test_every_shipped_mode_declares_a_router_and_two_engine_roles():
    for mode in load_pd_modes():
        assert mode.router is not None, mode.name
        if mode.name == PDModeEnum.CUSTOM.value:
            continue
        assert set(mode.roles) == {"prefill", "decode"}, mode.name
        assert mode.display_name and mode.description, mode.name


def test_all_five_capabilities_round_trip_through_serialization():
    """What the endpoint serves is the parsed catalog re-serialized, and
    re-validating the dump re-runs every load-time check, so this covers
    both directions of the round trip."""
    for mode in load_pd_modes():
        assert PDMode(**mode.model_dump()) == mode
        assert PDMode.model_validate_json(mode.model_dump_json()) == mode

    dumped = {mode.name: mode.model_dump(mode="json") for mode in load_pd_modes()}
    nixl = dumped[PDModeEnum.VLLM_NIXL.value]
    ascend = dumped[PDModeEnum.VLLM_ASCEND_MOONCAKE.value]
    # 1: injection target, 2: band width, 3: cross-role reference,
    # 4: router capabilities, 5: the lease window.
    assert nixl["roles"]["prefill"]["ports"][0]["inject_to"] == "env"
    assert ascend["roles"]["prefill"]["ports"][0]["inject_to"] == "args"
    assert ascend["roles"]["decode"]["ports"][0]["count"] == "{{accelerator_count}}"
    assert (
        ascend["roles"]["prefill"]["connector"]["kv_connector_extra_config"]["decode"][
            "tp_size"
        ]
        == "{{roles.decode.tensor_parallel_size}}"
    )
    assert ascend["router"]["capabilities"] == {
        "metrics": True,
        "models_endpoint": True,
        # The connector's, not the router's: Mooncake exports no counter for an
        # expired lease, so there is nothing for a metrics-serving router to
        # forward.
        "kv_expired_metric": False,
    }
    assert nixl["kv_lease"]["inject_to"] == "connector_extra_config"
    assert nixl["kv_lease"]["engine_default"] == 30
    assert ascend["kv_lease"]["param"] == "VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT"
    assert ascend["kv_lease"]["engine_default"] == 480


# ---------------------------------------------------------------------------
# The router's membership API.
# ---------------------------------------------------------------------------


def test_a_router_with_no_membership_api_is_a_valid_declaration():
    """The regression this exists for.

    Empty must mean "re-render and restart", not "the catalog is incomplete".
    Two of the five modes have no readable PD-mode membership shape, and a
    consumer that treated absence as an error would refuse to deploy them —
    for a capability whose whole purpose is to make one operation cheaper.
    """
    api = PDMembershipAPI()
    assert api.available is False
    assert api.add is None and api.probe is None
    assert api.requires_args == []


def test_add_without_a_read_back_is_not_available():
    """An accepted `add` is not a joined member: upstream polls the peer and
    drops it silently on timeout, with a default window tuned for small
    models. Without `probe`, a scale-out that quietly failed looks exactly
    like one that worked — so the pair, not `add` alone, is what gates."""
    assert PDMembershipAPI(add="POST /workers").available is False
    assert PDMembershipAPI(probe="GET /workers").available is False
    assert PDMembershipAPI(add="POST /workers", probe="GET /workers").available


def test_the_two_vllm_router_recipes_declare_the_same_membership_shape():
    """Same binary, so the same API. Both Ascend and CUDA now launch
    `vllm-router` — Ascend's own proxy example is gone from the catalog — and a
    divergence here would mean one of the two was updated and the other
    forgotten."""
    dumped = {mode.name: mode.router for mode in load_pd_modes()}
    nixl = dumped[PDModeEnum.VLLM_NIXL.value].membership_api
    ascend = dumped[PDModeEnum.VLLM_ASCEND_MOONCAKE.value].membership_api

    assert nixl == ascend
    assert nixl.add == "POST /workers"
    assert nixl.remove == "DELETE /workers/{url}"
    assert nixl.probe == "GET /workers"
    # The field the single-router path drops, which is why it is named rather
    # than assumed: the read-back has to be checked for it.
    assert nixl.role_field == "worker_type"
    assert nixl.role_values == {"prefill": "prefill", "decode": "decode"}


def test_declared_and_usable_move_together_once_the_flag_is_launched():
    """The distinction the whole capability turns on.

    `membership_api_usable` is not "upstream serves this API"; it is "this
    recipe launches the process in the mode where the API works". The two can
    disagree whenever a recipe declares the API but omits an argument it
    requires.

    The *rule* is what this asserts, not today's answer: a recipe that
    declares the API and launches every argument it requires is usable, and
    one that does not is not. Written as a rule, a flip stays a one-line
    change.
    """
    for mode in load_pd_modes():
        router = mode.router
        if not router.membership_api.available:
            assert not router.membership_api_usable, mode.name
            continue
        launched = set(str(token) for token in (router.command or []))
        required = set(router.membership_api.requires_args or [])
        assert router.membership_api_usable == required.issubset(launched), mode.name


def test_the_vllm_recipes_launch_what_their_membership_api_requires():
    """Without the flag there is no membership API at all on that fork
    (`POST /workers` -> 400), so a recipe that declares one
    and omits it promises a scale-out that will fail on first use."""
    seen = 0
    for mode in load_pd_modes():
        router = mode.router
        if not router.membership_api.available:
            continue
        if BackendEnum.VLLM not in mode.backends:
            continue
        seen += 1
        assert router.membership_api.requires_args == ["--enable-igw"]
        assert "--enable-igw" in [str(t) for t in (router.command or [])], mode.name
        assert router.membership_api_usable, mode.name
    assert seen == 2, "both vLLM-family recipes declare a membership API"


def test_the_sglang_recipes_declare_membership_without_a_flag_to_launch():
    """On `sglang_router`, `POST /workers` is accepted (202, queued), the
    member is in `GET /workers` on the next read, and traffic reaches it —
    with no flag, because on this gateway the command-line peers and the API
    go through the same registration job and the PD router reads the shared
    registry per request.

    So the assertion that matters is the *absence* of a required flag: copying
    the vLLM shape wholesale is what this file guards against, and
    `requires_args: [--enable-igw]` here would be exactly that copy — it would
    make `membership_api_usable` False and send every ratio change through a
    restart the router does not need.
    """
    for name in (PDModeEnum.SGLANG_MOONCAKE.value, PDModeEnum.SGLANG_NIXL.value):
        mode = next(m for m in load_pd_modes() if m.name == name)
        api = mode.router.membership_api
        assert api.available is True, name
        assert api.requires_args == [], name
        assert mode.router.membership_api_usable is True, name
        # `{id}`, not `{url}`: from v0.5.7 a member's id is a UUID the registry
        # mints, and the address form answers 400. The id comes off the probe
        # because nothing on this side can construct it.
        assert api.remove == "DELETE /workers/{id}", name
        # And the floor that shape belongs to is declared, so a reader does not
        # have to infer which dialect the recipe speaks.
        assert mode.backend_versions == ">=0.5.7", name
        assert api.probe == "GET /workers", name
        # The prefill's bootstrap port must reach the body, or the member
        # registers and then hangs every request routed to it.
        assert (api.body or {}).get("bootstrap_port") == "{{peer.ports.bootstrap}}"


def test_membership_api_round_trips_through_serialization():
    """The endpoint serves the re-serialized catalog, so a field that does not
    survive the dump is a field the UI never sees."""
    for mode in load_pd_modes():
        dumped = mode.model_dump(mode="json")
        assert "membership_api" in dumped["router"]
        assert PDMode.model_validate(dumped).router.membership_api == (
            mode.router.membership_api
        )


def test_every_shipped_router_classifies_its_invocation():
    """The refusal list is derived, so an unclassified router permits everything.

    Not a style rule. `platform_owned_flags` reads `connection_args`, and a
    router that declared its whole invocation as a bare `command` would expose
    an empty list — which turns "you may not set --prefill" into "you may".
    Both shipped routers declare `--prefill` as `action="append"`, so a second
    one does not replace the injected peer: it adds one the router forwards to
    and cannot reach.
    """
    from gpustack.schemas.pd_modes import PDRouterProtocolEnum

    for mode in load_pd_modes():
        router = mode.router
        if router is None or router.protocol == PDRouterProtocolEnum.USER_PROVIDED:
            continue
        assert router.entrypoint, f"{mode.name}: no entrypoint"
        assert router.connection_args, f"{mode.name}: no connection_args"
        # The flags a deployment may not set, and the ones it may, must be
        # disjoint — the model validator enforces it, this asserts the shipped
        # catalog actually exercises both sides.
        owned = set(router.platform_owned_flags)
        tunable = {arg.flag for arg in router.tunable_args}
        assert owned, f"{mode.name}: derived refusal list is empty"
        assert not (owned & tunable), f"{mode.name}: {owned & tunable} on both sides"


def test_the_composed_command_is_the_three_parts_in_order():
    """`command` stays readable as the whole invocation, so every existing
    consumer — the renderer, the read-only view — keeps seeing one list.

    Order is the contract, not an accident: a deployment's own parameters are
    appended after all of these, and appending only overrides a tunable
    default because repeated flags are last-wins.
    """
    from gpustack.schemas.pd_modes import PDRouterProtocolEnum

    for mode in load_pd_modes():
        router = mode.router
        if router is None or router.protocol == PDRouterProtocolEnum.USER_PROVIDED:
            continue
        expected = list(router.entrypoint) + list(router.connection_args)
        for arg in router.tunable_args:
            expected.extend(arg.tokens)
        assert router.command == expected, mode.name


def test_a_membership_body_may_only_name_a_band_the_mode_allocates():
    """The one template whose failure is silent, so it is caught at load.

    Everywhere else an unresolved placeholder reaches the process verbatim and
    the engine dies with it in the message. A membership value that cannot
    resolve is dropped from the body instead — deliberately, so one `body` can
    serve a prefill that has a bootstrap band and a decode that has none. A
    typo'd band would therefore register a prefill without its
    `bootstrap_port`, and every request routed to that member would hang with
    nothing anywhere saying why.
    """
    shipped = next(
        m for m in load_pd_modes() if m.name == PDModeEnum.SGLANG_MOONCAKE.value
    )
    dumped = shipped.model_dump(mode="json")
    dumped["router"]["membership_api"]["body"][
        "bootstrap_port"
    ] = "{{peer.ports.bootstrp}}"

    with pytest.raises(ValueError, match="nothing in this mode allocates"):
        PDMode.model_validate(dumped)

    # And the shipped spelling still validates, so this is not vacuous.
    assert PDMode.model_validate(shipped.model_dump(mode="json")).name == shipped.name


# ---------------------------------------------------------------------------
# Which plane {{net_device}} rides is the recipe's to declare.
# ---------------------------------------------------------------------------


def test_the_net_device_plane_differs_between_the_recipes_that_inject_it():
    """One placeholder, two natures — which is the whole reason it is declared.

    Ascend's three `*_SOCKET_IFNAME` carry handshake sockets while the KV bytes
    ride the cards' RoCE ports, so the worker's own management NIC is the right
    answer and the multi-NIC refusal only costs an operator a value the platform
    already holds. NIXL's `UCX_NET_DEVICES` is the data plane on the same kind
    of host, where the management NIC is usually the wrong fabric.
    """
    ascend = get_pd_mode(PDModeEnum.VLLM_ASCEND_MOONCAKE.value)
    assert ascend.net_device_plane == PDNetDevicePlaneEnum.CONTROL

    for name in (PDModeEnum.VLLM_NIXL.value, PDModeEnum.SGLANG_NIXL.value):
        mode = get_pd_mode(name)
        assert mode.net_device_plane == PDNetDevicePlaneEnum.DATA, name


def test_a_recipe_that_says_nothing_keeps_the_stricter_plane():
    """The default has to be `data`: forgetting to classify a recipe then costs
    one `kv_ifname`, where the reverse default would silently put KV bytes on
    the management NIC of every unclassified recipe."""
    assert PDMode(name="x").net_device_plane == PDNetDevicePlaneEnum.DATA
    # And the shipped recipes that leave the NIC to the engine are unaffected.
    assert (
        get_pd_mode(PDModeEnum.SGLANG_MOONCAKE.value).net_device_plane
        == PDNetDevicePlaneEnum.DATA
    )


def test_declaring_the_control_plane_where_no_net_device_is_injected_fails():
    """The field changes exactly one placeholder's value, so declaring it on a
    recipe that never injects `{{net_device}}` is a statement that does nothing
    — and the mistake it would come from is the expensive one: relaxing the
    refusal for `sglang-mooncake`, whose NIC is the engine's own business, would
    have no effect and send somebody looking for the reason in code."""
    document = _document(
        [
            {
                "name": PDModeEnum.SGLANG_MOONCAKE.value,
                "backends": list(PD_MODE_BACKENDS[PDModeEnum.SGLANG_MOONCAKE.value]),
                "net_device_plane": "control",
            }
        ]
    )

    with pytest.raises(PDModeCatalogError, match="changes nothing"):
        parse_pd_mode_catalog(document)


def test_the_plane_round_trips_through_serialization():
    """The endpoint serves `model_dump()` back, and re-validating our own output
    has to be a no-op."""
    shipped = get_pd_mode(PDModeEnum.VLLM_ASCEND_MOONCAKE.value)
    dumped = shipped.model_dump(mode="json")
    assert dumped["net_device_plane"] == "control"
    assert (
        PDMode.model_validate(dumped).net_device_plane == PDNetDevicePlaneEnum.CONTROL
    )


# --- what an engine needs before a cache is folded in beside the connector -- #


def test_the_shipped_catalog_declares_the_multiconnector_floor():
    """vLLM below 0.26.0 hands a non-chosen child empty blocks, so a composed
    cache stores nothing and the pool can never warm itself. The requirement
    belongs to the engine, not to any one mode or provider."""
    declared = get_composed_cache("vLLM")

    assert declared is not None
    assert declared.min_version == "0.26.0"
    assert "46865" in (declared.reference or "")


def test_which_role_may_take_a_cache_is_readable_from_the_document():
    """Answerable from the catalog alone: decode is refused, prefill is not.
    Keyed on role names rather than the connector's `kv_role` because every
    mode has roles while only the vLLM ones declare a kv_role — a rule keyed on
    a field two of five shipped modes leave unset would go quiet exactly where
    nobody would notice."""
    refused = set(get_composed_cache("vLLM").refuse_cache_on.roles)

    assert refused == {"decode"}
    for name in (PDModeEnum.VLLM_NIXL.value, PDModeEnum.VLLM_ASCEND_MOONCAKE.value):
        assert set(get_pd_mode(name).roles) >= {"prefill", "decode"}


def test_a_refused_role_no_recipe_declares_is_caught_at_load():
    """The refusal is matched on the member's role name, so 'decoder' instead
    of 'decode' would let a cache through onto the side that dies on a hit."""
    document = _document(
        [],
        composed_cache=[{"backend": "vLLM", "refuse_cache_on": {"roles": ["decoder"]}}],
    )

    with pytest.raises(PDModeCatalogError, match="refuses roles no recipe"):
        parse_pd_mode_catalog(document)


def test_an_engine_that_never_composes_declares_nothing():
    """SGLang attaches its cache through --enable-lmcache and a config file,
    which never reaches the connector flag. Absence is the answer — an entry
    with a permissive version would read as 'checked' when there is nothing
    to check."""
    assert get_composed_cache("SGLang") is None


def test_a_composed_cache_floor_for_an_unknown_backend_is_refused():
    """The lookup keys on the backend string, so a spelling slip would stop
    applying in silence rather than failing."""
    document = _document(
        [],
        composed_cache=[{"backend": "vllm", "min_version": "0.26.0"}],
    )

    with pytest.raises(PDModeCatalogError, match="names backends no mode declares"):
        parse_pd_mode_catalog(document)


def test_two_floors_for_one_backend_are_refused():
    document = _document(
        [],
        composed_cache=[
            {"backend": "vLLM", "min_version": "0.26.0"},
            {"backend": "vLLM", "min_version": "0.27.0"},
        ],
    )

    with pytest.raises(PDModeCatalogError, match="duplicate composed_cache"):
        parse_pd_mode_catalog(document)
