"""Port assignment on the worker: named connector bands, fencing, refill.

The whole point of these is that a port handed out twice does not fail
cleanly — the second engine crash-loops on bind and the instance sits in
`starting` forever — so every path that can leak a port back into the free
pool is pinned here.
"""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from gpustack.schemas.models import (
    BackendEnum,
    DisaggregationSpec,
    DistributedServers,
    ModelInstanceSubordinateWorker,
    PDModeEnum,
    PortBand,
    RoleSpec,
    role_effective_model,
)
from gpustack.schemas.pd_modes import (
    PDInjectTargetEnum,
    PDMode,
    PDModeRole,
    PDPortScopeEnum,
    PDPortSpec,
)
from gpustack.server.pd_mode_catalog import get_pd_modes
from gpustack.utils import network
from gpustack.worker.serve_manager import (
    _VLLM_MP_CONNECTING_BAND,
    PDPortScopeUnsupportedError,
    ServeManager,
)
from tests.utils.model import new_model, new_model_instance

PORT_RANGE = "40000-40063"


def _manager(port_range: str = PORT_RANGE, worker_id: int = 1) -> ServeManager:
    clientset = MagicMock()
    clientset.model_instances.list.return_value = SimpleNamespace(items=[])
    cfg = SimpleNamespace(
        log_dir="/tmp",
        service_port_range=port_range,
        system_default_container_registry=None,
    )
    manager = ServeManager(lambda: worker_id, lambda: clientset, cfg)
    manager._inference_backend_manager = MagicMock()
    return manager


def _pd_model(mode: PDModeEnum = PDModeEnum.VLLM_NIXL, **kwargs):
    return new_model(
        1,
        "pd-model",
        huggingface_repo_id="Qwen/Qwen2.5-7B-Instruct",
        disaggregation=DisaggregationSpec(mode=mode),
        **kwargs,
    )


def _instance(instance_id: int = 1, role: str = "prefill", cards: int = 0):
    mi = new_model_instance(
        instance_id,
        f"pd-{instance_id}",
        1,
        worker_id=1,
        gpu_indexes=list(range(cards)) or None,
    )
    mi.worker_ip = "127.0.0.1"
    mi.role = role
    return mi


def _parallelism_band_mode(placeholder: str = "{{tensor_parallel_size}}") -> PDMode:
    """A mode that sizes its band from a parallelism parameter.

    The shipped Ascend entry sizes `kv_port` by card count instead, so the
    parameter-driven branch of the resolver needs a mode of its own to stay
    covered — a connector that sizes its band by TP/DP/PP is a YAML change,
    not a code change, and that has to keep working.
    """
    return PDMode(
        name="test-parallelism-band",
        roles={
            "prefill": PDModeRole(
                ports=[
                    PDPortSpec(
                        name="kv_port",
                        count=placeholder,
                        inject_to=PDInjectTargetEnum.ARGS,
                    )
                ],
                connector={"kv_port": "{{ports.kv_port}}"},
            ),
            "decode": PDModeRole(
                ports=[
                    PDPortSpec(
                        name="kv_port",
                        count=placeholder,
                        inject_to=PDInjectTargetEnum.ARGS,
                    )
                ],
                connector={"kv_port": "{{ports.kv_port}}"},
            ),
        },
    )


@pytest.fixture(autouse=True)
def all_ports_free(monkeypatch):
    """The allocator's scan, not the host's port table, is what's under test."""
    monkeypatch.setattr(network, "is_port_available", lambda port, host=None: True)


def _wide_band_mode(count: int = 4) -> PDMode:
    """A mode whose prefill role wants a band wider than one port."""
    return PDMode(
        name="test-wide-band",
        roles={
            "prefill": PDModeRole(
                ports=[
                    PDPortSpec(
                        name="kv_port",
                        count=count,
                        inject_to=PDInjectTargetEnum.ARGS,
                    )
                ],
                connector={"kv_port": "{{ports.kv_port}}"},
            )
        },
    )


def test_named_band_lands_in_both_indexes():
    """`named_ports` is the new index; `mi.ports` is the one the runtime turns
    into hostPorts, and it is the only reason a same-host collision shows up
    as a scheduling event instead of a crash loop."""
    manager = _manager()
    mi = _instance(role="prefill")

    manager._assign_ports(mi, _pd_model(), BackendEnum.VLLM)

    band = mi.named_ports["kv_side_channel"]
    assert band.count == 1
    assert mi.ports[0] == mi.port
    assert band.base in mi.ports
    assert band.base != mi.port


def test_router_role_reads_the_router_declaration():
    """A router's bands sit on `mode.router`, not in `mode.roles`."""
    manager = _manager()
    mi = _instance(role="router")

    manager._assign_ports(mi, _pd_model(), BackendEnum.VLLM)

    assert set(mi.named_ports) == {"prometheus"}
    assert mi.named_ports["prometheus"].base in mi.ports


def test_whole_band_is_fenced_not_just_its_base():
    """Every port a connector derives from the base is bound just as surely
    as the base, so the fence has to cover the run."""
    manager = _manager()
    mi = _instance(1)

    with patch(
        "gpustack.worker.pd_injection.get_pd_mode", return_value=_wide_band_mode(4)
    ):
        manager._assign_ports(mi, _pd_model(), BackendEnum.VLLM)

    band = mi.named_ports["kv_port"]
    expected = list(range(band.base, band.base + 4))
    assert band.count == 4
    assert [p for p in mi.ports if p != mi.port] == expected
    assert set(expected) <= manager._assigned_ports[mi.id]


def test_a_second_instance_gets_a_disjoint_band():
    manager = _manager()
    first, second = _instance(1), _instance(2)

    with patch(
        "gpustack.worker.pd_injection.get_pd_mode", return_value=_wide_band_mode(4)
    ):
        manager._assign_ports(first, _pd_model(), BackendEnum.VLLM)
        manager._assign_ports(second, _pd_model(), BackendEnum.VLLM)

    assert not set(first.ports) & set(second.ports)


def test_named_bands_go_before_the_connecting_port():
    """The distributed backends read the connecting port as `ports[-1]`
    (VLLM_DP_MASTER_PORT / VLLM_PORT), so "append at the tail" must not
    quietly redefine which port that is."""
    manager = _manager()
    mi = _instance()
    mi.distributed_servers = DistributedServers(
        subordinate_workers=[ModelInstanceSubordinateWorker(worker_id=2)]
    )

    with patch(
        "gpustack.worker.pd_injection.get_pd_mode", return_value=_wide_band_mode(4)
    ):
        manager._assign_ports(mi, _pd_model(), BackendEnum.VLLM)

    band = mi.named_ports["kv_port"]
    named = set(range(band.base, band.base + band.count))
    assert mi.ports[-1] not in named
    assert mi.ports[0] == mi.port
    # And the band is still there, just not last.
    assert named <= set(mi.ports)


def test_a_non_pd_instance_takes_the_old_path_unchanged():
    manager = _manager()
    mi = new_model_instance(1, "plain", 1, worker_id=1)
    mi.worker_ip = "127.0.0.1"
    model = new_model(1, "plain", huggingface_repo_id="Qwen/Qwen2.5-7B-Instruct")

    manager._assign_ports(mi, model, BackendEnum.VLLM)

    assert mi.ports == [mi.port]
    assert not mi.named_ports


def test_a_role_the_mode_does_not_declare_allocates_nothing():
    manager = _manager()
    mi = _instance(role="decode")

    with patch(
        "gpustack.worker.pd_injection.get_pd_mode", return_value=_wide_band_mode(4)
    ):
        manager._assign_ports(mi, _pd_model(), BackendEnum.VLLM)

    assert mi.ports == [mi.port]
    assert not mi.named_ports


def test_early_return_refills_the_registry_with_the_whole_band():
    """A restart reuses the persisted ports and must re-register them: without
    that, this process sees the entire band as free and hands it to the next
    instance."""
    manager = _manager()
    mi = _instance()
    mi.port = 40000
    mi.ports = [40000, 40010, 40011, 40012, 40013]
    mi.named_ports = {"kv_port": PortBand(base=40010, count=4)}

    manager._assign_ports(mi, _pd_model(), BackendEnum.VLLM)

    assert manager._assigned_ports[mi.id] == {40000, 40010, 40011, 40012, 40013}
    # Nothing was reallocated.
    assert mi.port == 40000


def test_early_return_expands_a_band_absent_from_mi_ports():
    """Bands persisted before the append-to-`mi.ports` rule existed are still
    fenced: `count` is what defines the run, not the list."""
    manager = _manager()
    mi = _instance()
    mi.port = 40000
    mi.ports = [40000]
    mi.named_ports = {"kv_port": PortBand(base=40020, count=3)}

    manager._assign_ports(mi, _pd_model(), BackendEnum.VLLM)

    assert manager._assigned_ports[mi.id] == {40000, 40020, 40021, 40022}


def test_refilled_ports_are_honoured_by_the_next_allocation():
    manager = _manager()
    restarted = _instance(1)
    restarted.port = 40000
    restarted.ports = [40000]
    restarted.named_ports = {"kv_port": PortBand(base=40001, count=4)}
    manager._assign_ports(restarted, _pd_model(), BackendEnum.VLLM)

    fresh = _instance(2)
    with patch(
        "gpustack.worker.pd_injection.get_pd_mode", return_value=_wide_band_mode(4)
    ):
        manager._assign_ports(fresh, _pd_model(), BackendEnum.VLLM)

    assert not set(fresh.ports) & {40000, 40001, 40002, 40003, 40004}


def test_the_kv_band_is_as_wide_as_the_members_cards():
    """Mooncake's `kv_port` is a base and the connector binds one port per
    *worker rank*:

        handshake_port = kv_port + dp_rank * tp * pp + (pp_rank + pcp_rank) * tp
                       + tp_rank

    so the band is `dp x tp x pp` wide -- the member's own card count. TP8/DP1
    was measured holding 41100-41107 and DP2xTP2 holding 20001-20004; reading
    the width as the tensor-parallel size fits the first sample and
    under-reserves the second by a factor of dp.
    """
    manager = _manager(port_range="40000-40063")
    mi = _instance(role="prefill", cards=8)
    model = _pd_model(
        PDModeEnum.VLLM_ASCEND_MOONCAKE,
        # DP2xTP4 across the 8 cards: neither factor alone gives 8.
        backend_parameters=["--tensor-parallel-size", "4", "--data-parallel-size", "2"],
    )

    manager._assign_ports(mi, model, BackendEnum.VLLM)

    band = mi.named_ports["kv_port"]
    assert band.count == 8
    expected = set(range(band.base, band.base + 8))
    # The whole run is fenced, not just the base — that is the point.
    assert expected <= set(mi.ports)
    assert expected <= manager._assigned_ports[mi.id]


def test_a_card_count_band_does_not_follow_tensor_parallel_size():
    """The regression this replaces: a DP member whose band was sized by TP
    reserved `tp` ports and bound `dp x tp`, and the second rank to bind died
    on `Address already in use` — an instance wedged in `starting`."""
    manager = _manager()
    mi = _instance(role="prefill", cards=4)
    model = _pd_model(
        PDModeEnum.VLLM_ASCEND_MOONCAKE,
        backend_parameters=["--tensor-parallel-size", "2", "--data-parallel-size", "2"],
    )

    manager._assign_ports(mi, model, BackendEnum.VLLM)

    assert mi.named_ports["kv_port"].count == 4


def test_a_member_without_cards_gets_no_band_and_says_why(caplog):
    """`{{accelerator_count}}` on a member with nothing scheduled onto it is
    not a one-port band: same refusal as an unresolvable parallelism
    parameter, so the placeholder reaches the launch and names itself."""
    manager = _manager()
    mi = _instance(role="prefill", cards=0)

    with caplog.at_level(logging.WARNING, logger="gpustack.worker.serve_manager"):
        manager._assign_ports(
            mi, _pd_model(PDModeEnum.VLLM_ASCEND_MOONCAKE), BackendEnum.VLLM
        )

    assert not mi.named_ports
    assert mi.ports == [mi.port]
    assert "kv_port" in caplog.text
    assert "no accelerators assigned" in caplog.text


@pytest.mark.parametrize(
    "parameters, expected",
    [
        (["--tensor-parallel-size", "4"], 4),
        (["-tp", "2"], 2),
        (["--tp-size=8"], 8),
    ],
)
def test_templated_count_accepts_every_spelling_of_tp(parameters, expected):
    """The engines accept three spellings and users write all three."""
    manager = _manager()
    mi = _instance(role="prefill")
    model = _pd_model(backend_parameters=parameters)

    with patch(
        "gpustack.worker.pd_injection.get_pd_mode",
        return_value=_parallelism_band_mode(),
    ):
        manager._assign_ports(mi, model, BackendEnum.VLLM)

    assert mi.named_ports["kv_port"].count == expected


@pytest.mark.parametrize(
    "placeholder, parameters",
    [
        ("{{data_parallel_size}}", ["--data-parallel-size", "3"]),
        ("{{pipeline_parallel_size}}", ["-pp", "3"]),
    ],
)
def test_the_other_parallelism_placeholders_resolve_too(placeholder, parameters):
    """A connector that sizes its band by DP or PP is a YAML change, not a
    code change."""
    manager = _manager()
    mi = _instance(role="prefill")
    mode = PDMode(
        name="test-wide-band",
        roles={
            "prefill": PDModeRole(
                ports=[
                    PDPortSpec(
                        name="kv_port",
                        count=placeholder,
                        inject_to=PDInjectTargetEnum.ARGS,
                    )
                ],
                connector={"kv_port": "{{ports.kv_port}}"},
            )
        },
    )
    model = _pd_model(backend_parameters=parameters)

    with patch("gpustack.worker.pd_injection.get_pd_mode", return_value=mode):
        manager._assign_ports(mi, model, BackendEnum.VLLM)

    assert mi.named_ports["kv_port"].count == 3


def test_templated_count_reads_the_roles_own_override():
    """`_get_model` hands this allocator the role's projection, so a role that
    overrides `backend_parameters` sizes its band from its own value and not
    from the Model's."""
    manager = _manager()
    mi = _instance(role="decode")
    model = _pd_model(
        backend_parameters=["--tensor-parallel-size", "8"],
        roles=[
            RoleSpec(name="prefill"),
            RoleSpec(name="decode", backend_parameters=["--tensor-parallel-size", "2"]),
        ],
    )

    with patch(
        "gpustack.worker.pd_injection.get_pd_mode",
        return_value=_parallelism_band_mode(),
    ):
        manager._assign_ports(
            mi, role_effective_model(model, "decode"), BackendEnum.VLLM
        )

    assert mi.named_ports["kv_port"].count == 2


def test_unresolvable_templated_count_allocates_nothing_and_says_why(caplog):
    """vLLM's default TP is 1, but GPUStack injects a tensor-parallel size of
    its own further down, so "no -tp written" does not mean "one rank".
    Guessing 1 would fence one port where the connector binds eight, and that
    surfaces as an instance wedged in `starting`; not allocating the band
    leaves the placeholder in the launch, which names itself."""
    manager = _manager()
    mi = _instance(role="prefill")

    with (
        caplog.at_level(logging.WARNING, logger="gpustack.worker.serve_manager"),
        patch(
            "gpustack.worker.pd_injection.get_pd_mode",
            return_value=_parallelism_band_mode(),
        ),
    ):
        manager._assign_ports(mi, _pd_model(), BackendEnum.VLLM)

    assert not mi.named_ports
    assert mi.ports == [mi.port]
    assert "kv_port" in caplog.text
    assert "tensor_parallel_size" in caplog.text
    assert "No ports are reserved" in caplog.text


def test_a_role_scoped_band_is_refused_not_downgraded():
    """Role scope means "every member of this role shares one band". Handing
    each member its own band is not a partial implementation of that, it is a
    band the connector cannot meet on — and it only fails at handshake."""
    manager = _manager()
    mi = _instance(role="prefill")
    mode = PDMode(
        name="test-wide-band",
        roles={
            "prefill": PDModeRole(
                ports=[
                    PDPortSpec(
                        name="kv_port",
                        count=2,
                        inject_to=PDInjectTargetEnum.ARGS,
                        scope=PDPortScopeEnum.ROLE,
                    )
                ],
                connector={"kv_port": "{{ports.kv_port}}"},
            )
        },
    )

    with pytest.raises(PDPortScopeUnsupportedError) as excinfo:
        with patch("gpustack.worker.pd_injection.get_pd_mode", return_value=mode):
            manager._assign_ports(mi, _pd_model(), BackendEnum.VLLM)

    message = str(excinfo.value)
    assert "kv_port" in message
    assert "registry" in message
    assert not mi.named_ports


def test_no_shipped_band_declares_role_scope():
    """The guard above is only tolerable while nothing needs role scope. This
    is the tripwire: adding `scope: role` to the catalog fails here, where the
    fix is "implement the registry", rather than on a worker at handshake."""
    for mode in get_pd_modes():
        holders = list(mode.roles.values())
        if mode.router is not None:
            holders.append(mode.router)
        for holder in holders:
            for spec in holder.ports or []:
                assert spec.scope == PDPortScopeEnum.INSTANCE, (
                    f"PD mode '{mode.name}' band '{spec.name}' declares "
                    f"scope '{spec.scope}', which the worker allocator refuses"
                )


def test_unknown_mode_warns_and_assigns_no_bands(caplog):
    manager = _manager()
    mi = _instance()

    with caplog.at_level(logging.WARNING, logger="gpustack.worker.serve_manager"):
        with patch("gpustack.worker.pd_injection.get_pd_mode", return_value=None):
            manager._assign_ports(mi, _pd_model(), BackendEnum.VLLM)

    assert not mi.named_ports
    assert "not in the catalog" in caplog.text


def test_exhaustion_names_the_role_and_the_band():
    """The allocator knows the arithmetic but not who was asking, and "which
    role of which deployment" is the first thing an operator needs."""
    manager = _manager(port_range="40000-40000")
    mi = _instance(role="prefill")

    with pytest.raises(network.PortRangeExhaustedError) as excinfo:
        with patch(
            "gpustack.worker.pd_injection.get_pd_mode", return_value=_wide_band_mode(4)
        ):
            manager._assign_ports(mi, _pd_model(), BackendEnum.VLLM)

    message = str(excinfo.value)
    assert "role 'prefill'" in message
    assert "'kv_port'" in message
    assert "Widen the port range" in message


# ---------------------------------------------------------------------------
# The vLLM mp connecting band. Not a PD path — it predates PD and is shared by
# every multi-node vLLM deployment — but the same three bugs: the ports vLLM
# derives from VLLM_DP_MASTER_PORT were never probed, never registered, and
# silently clamped to fewer than ten near the end of the range.
# ---------------------------------------------------------------------------


def _distributed(mi):
    mi.distributed_servers = DistributedServers(
        subordinate_workers=[ModelInstanceSubordinateWorker(worker_id=2)]
    )
    return mi


def _mp_model(**kwargs):
    return new_model(
        1,
        "mp-model",
        huggingface_repo_id="Qwen/Qwen2.5-7B-Instruct",
        backend_parameters=["--distributed-executor-backend", "mp"],
        **kwargs,
    )


def _plain_instance(instance_id: int = 1):
    mi = new_model_instance(instance_id, f"mp-{instance_id}", 1, worker_id=1)
    mi.worker_ip = "127.0.0.1"
    return _distributed(mi)


def test_mp_connecting_band_keeps_the_positional_layout():
    """`ports[1..3]` and `ports[-1]` are read by name in vllm.py, so nothing
    may be appended after the connecting port."""
    manager = _manager()
    mi = _plain_instance()

    manager._assign_ports(mi, _mp_model(), BackendEnum.VLLM)

    connecting = mi.ports[-1]
    assert mi.ports[0] == mi.port
    band = set(range(connecting, connecting + _VLLM_MP_CONNECTING_BAND))
    # The cross ports (DP RPC, master, VLLM_PORT) stay outside the band.
    assert not band & set(mi.ports[1:4])


def test_the_mp_container_spec_is_unchanged():
    """The nine derived ports are recorded but NOT put in `mi.ports`.

    `mi.ports` is what the runtime turns into host ports, so adding them there
    would rewrite the container spec of every existing multi-worker vLLM
    deployment — a change to non-disaggregated behaviour, made as a side effect
    of a disaggregation change. HTTP plus three cross ports plus the band's
    base is what it has always been.
    """
    manager = _manager()
    mi = _plain_instance()

    manager._assign_ports(mi, _mp_model(), BackendEnum.VLLM)

    assert len(mi.ports) == 5


def test_mp_connecting_band_is_registered_whole():
    """The whole band is fenced, not just the call that allocated it: fencing
    the base alone would leave nine of its ten ports free to the next instance
    on this worker."""
    manager = _manager()
    mi = _plain_instance()

    manager._assign_ports(mi, _mp_model(), BackendEnum.VLLM)

    connecting = mi.ports[-1]
    band = set(range(connecting, connecting + _VLLM_MP_CONNECTING_BAND))
    assert band <= manager._assigned_ports[mi.id]


def test_the_mp_band_survives_a_worker_restart():
    """Recorded in `named_ports`, which the early-return refill re-expands from
    `base`/`count` — the whole reason the band needed a persisted home at all.
    """
    manager = _manager()
    mi = _plain_instance()
    manager._assign_ports(mi, _mp_model(), BackendEnum.VLLM)
    connecting = mi.ports[-1]

    fresh = _manager()
    fresh._register_assigned_ports(mi)

    band = set(range(connecting, connecting + _VLLM_MP_CONNECTING_BAND))
    assert band <= fresh._assigned_ports[mi.id]


def test_a_second_mp_instance_gets_a_disjoint_band():
    """Fencing the band only within the call that allocated it would leave the
    next instance on the worker seeing nine of its ten ports as free, and vLLM
    would bind them anyway."""
    manager = _manager()
    first, second = _plain_instance(1), _plain_instance(2)

    manager._assign_ports(first, _mp_model(), BackendEnum.VLLM)
    manager._assign_ports(second, _mp_model(), BackendEnum.VLLM)

    assert not set(first.ports) & set(second.ports)
    # The derived ports are the point: comparing only `mi.ports` would compare
    # the two bases and miss the nine ports on either side of them.
    first_band = set(range(first.ports[-1], first.ports[-1] + _VLLM_MP_CONNECTING_BAND))
    second_band = set(
        range(second.ports[-1], second.ports[-1] + _VLLM_MP_CONNECTING_BAND)
    )
    assert not first_band & second_band


def test_mp_connecting_band_is_probed(monkeypatch):
    """Every port of the band is bound by vLLM, so every port of it has to be
    probed, not just the base."""
    busy = {40003, 40011}
    monkeypatch.setattr(
        network, "is_port_available", lambda port, host=None: port not in busy
    )
    manager = _manager()
    mi = _plain_instance()

    manager._assign_ports(mi, _mp_model(), BackendEnum.VLLM)

    connecting = mi.ports[-1]
    band = set(range(connecting, connecting + _VLLM_MP_CONNECTING_BAND))
    assert not band & busy


def test_mp_band_at_the_range_end_is_refused_not_clamped():
    """The old implementation clamped the fence to the end of the range and
    handed out the base anyway, so vLLM bound ports nobody had reserved. A
    range that cannot hold the band now fails where the message can be read."""
    # 40000 goes to the HTTP port, leaving nine — one short of the band.
    manager = _manager(port_range="40000-40009")
    mi = _plain_instance()

    with pytest.raises(network.PortRangeExhaustedError) as excinfo:
        manager._assign_ports(mi, _mp_model(), BackendEnum.VLLM)

    message = str(excinfo.value)
    assert "10 consecutive free port(s)" in message
    assert "Widen the port range" in message


def test_restarted_mp_instance_refills_the_whole_band():
    """The band is persisted in `mi.ports`, so the refill on a restart covers
    it without a second index to expand."""
    manager = _manager()
    restarted = _plain_instance(1)
    restarted.port = 40000
    restarted.ports = [40000, 40020, 40021, 40022] + list(range(40031, 40040)) + [40030]

    manager._assign_ports(restarted, _mp_model(), BackendEnum.VLLM)

    assert manager._assigned_ports[restarted.id] == set(restarted.ports)
    fresh = _plain_instance(2)
    manager._assign_ports(fresh, _mp_model(), BackendEnum.VLLM)
    assert not set(fresh.ports) & set(restarted.ports)


def test_the_ray_path_still_takes_a_single_connecting_port():
    """Only vLLM's mp executor derives ports from the connecting port; fencing
    ten on the Ray path would burn nine ports per instance."""
    manager = _manager()
    mi = _plain_instance()
    model = new_model(
        1,
        "ray-model",
        huggingface_repo_id="Qwen/Qwen2.5-7B-Instruct",
        backend_parameters=["--distributed-executor-backend", "ray", "--dp", "2"],
    )

    manager._assign_ports(mi, model, BackendEnum.VLLM)

    # [http, dp_rpc, connecting] — nothing else.
    assert len(mi.ports) == 3
    assert manager._assigned_ports[mi.id] == set(mi.ports)


def test_a_non_vllm_distributed_backend_takes_a_single_connecting_port():
    manager = _manager()
    mi = _plain_instance()
    model = new_model(1, "mindie-model", huggingface_repo_id="Qwen/Qwen2.5-7B-Instruct")

    manager._assign_ports(mi, model, BackendEnum.ASCEND_MINDIE)

    assert len(mi.ports) == 2
    assert mi.ports[0] == mi.port


def test_start_model_instance_persists_named_ports(tmp_path):
    """The bands are allocated on the worker and read on the server (the
    router's peer config), so they persist the way `port`/`ports` do."""
    manager = _manager()
    manager._serve_log_dir = str(tmp_path)
    mi = _instance()
    model = _pd_model()

    with (
        patch.object(manager, "_get_model", return_value=model),
        patch.object(manager, "_start_container_log_persistence"),
        patch.object(manager, "_update_model_instance") as update,
        patch("gpustack.worker.serve_manager.multiprocessing.Process") as process,
    ):
        process.return_value.pid = 4242
        manager._start_model_instance(mi)

    patch_dict = update.call_args.kwargs
    assert patch_dict["named_ports"] == mi.named_ports
    assert patch_dict["named_ports"]["kv_side_channel"].base in patch_dict["ports"]
