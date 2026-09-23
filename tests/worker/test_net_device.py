import logging
import socket
from collections import namedtuple

import pytest

from gpustack.config.config import Config
from gpustack.schemas.pd_modes import PDNetDevicePlaneEnum
from gpustack.schemas.workers import (
    GPUDeviceStatus,
    GPUNetworkInfo,
    Worker,
    WorkerStatus,
)
from gpustack.worker import net_device
from gpustack.worker.net_device import candidate_kv_interfaces, derive_net_device


@pytest.fixture
def config(tmp_path):
    def _config(**kwargs) -> Config:
        return Config(data_dir=str(tmp_path / "data"), **kwargs)

    return _config


@pytest.fixture(autouse=True)
def single_nic_host(monkeypatch):
    """Pin the host NIC enumeration for every test that does not care about it.

    Without this the results would depend on whatever NICs the machine running
    the suite happens to have -- a developer box with Docker plus a second
    uplink would flip the multi-NIC refusal on and fail the fallback tests.
    """
    monkeypatch.setattr(net_device, "candidate_kv_interfaces", lambda: ["eth0"])


@pytest.fixture
def nics(monkeypatch):
    def _set(names):
        monkeypatch.setattr(net_device, "candidate_kv_interfaces", lambda: list(names))

    return _set


def _worker(ifname: str = "eth0", gpu_devices=None) -> Worker:
    status = WorkerStatus.get_default_status()
    if gpu_devices is not None:
        status.gpu_devices = gpu_devices
    return Worker(
        id=1,
        name="test-worker",
        hostname="test-host",
        ip="192.168.1.100",
        ifname=ifname,
        port=10150,
        worker_uuid="test-uuid",
        cluster_id=1,
        status=status,
    )


def test_kv_ifname_overrides_worker_ifname(config):
    """The escape hatch exists precisely because the auto-detected NIC is the
    management one; if it did not win, setting it would be a no-op."""
    assert derive_net_device(_worker(ifname="eth0"), config(kv_ifname="ib0")) == "ib0"


def test_kv_ifname_wins_on_a_multi_nic_host(config, nics):
    """The refusal below is what kv_ifname is the answer to, so it must not be
    reachable once the operator has answered."""
    nics(["eno1", "ib0"])
    assert derive_net_device(_worker(ifname="eno1"), config(kv_ifname="ib0")) == "ib0"


def test_kv_ifname_is_honoured_even_when_it_is_not_a_candidate(config, nics):
    """The candidate filter gates the automatic path only. An explicit value is
    an instruction, not a proposal -- an RDMA-only netdev, or one this host's
    naming does not resemble, must still get through."""
    nics(["eno1"])
    assert (
        derive_net_device(_worker(ifname="eno1"), config(kv_ifname="mlx5_0"))
        == "mlx5_0"
    )


def test_falls_back_to_worker_ifname(config, nics):
    nics(["bond0"])
    assert derive_net_device(_worker(ifname="bond0"), config()) == "bond0"


def test_returns_none_when_no_source_has_a_value(config):
    """No source means no answer. Anything else -- notably ``all`` -- turns a
    clean failure into an unroutable address inside the NIXL metadata."""
    assert derive_net_device(_worker(ifname=""), config()) is None


def test_placeholder_worker_row_is_not_a_value(config):
    """Pool-provisioned workers are stored with ifname="" long before they
    report in, so emptiness has to read as unknown, not as an interface name."""
    assert derive_net_device(_worker(ifname="   "), config()) is None


def test_blank_kv_ifname_defers_instead_of_blanking_the_result(config, nics):
    nics(["eth1"])
    assert derive_net_device(_worker(ifname="eth1"), config(kv_ifname="  ")) == "eth1"


def test_ascend_per_card_iface_is_never_used(config, nics):
    """hccn_tool reports eth0-eth7 for the card-internal ports. Those devices do
    not exist in the host netns, so UCX cannot bind to them -- the host NIC is
    still the only usable answer."""
    ascend_cards = [
        GPUDeviceStatus(
            vendor="Huawei",
            type="cann",
            index=i,
            name="Ascend 910B2",
            network=GPUNetworkInfo(
                status="up",
                inet=f"10.10.0.{i + 1}",
                iface=f"eth{i}",
                mtu=8192,
            ),
        )
        for i in range(8)
    ]
    worker = _worker(ifname="enp1s0f0", gpu_devices=ascend_cards)
    nics(["enp1s0f0"])

    assert derive_net_device(worker, config()) == "enp1s0f0"


def test_ascend_per_card_iface_does_not_rescue_a_missing_worker_ifname(config):
    """The per-card data being present must not make the None case disappear --
    that would smuggle a card identifier in as a UCX device name."""
    worker = _worker(
        ifname="",
        gpu_devices=[
            GPUDeviceStatus(
                vendor="Huawei",
                type="cann",
                index=0,
                name="Ascend 910B2",
                network=GPUNetworkInfo(status="up", inet="10.10.0.1", iface="eth0"),
            )
        ],
    )

    assert derive_net_device(worker, config()) is None


def test_kv_ifname_defaults_to_none(config):
    assert config().kv_ifname is None


#
# The multi-NIC refusal.
#


def test_multi_nic_host_refuses_instead_of_using_the_management_nic(
    config, nics, caplog
):
    """The design's requirement: silently resolving to the management plane on a
    multi-NIC host is how the KV traffic ends up on the wrong fabric."""
    nics(["eno1", "ib0"])
    with caplog.at_level(logging.ERROR):
        assert derive_net_device(_worker(ifname="eno1"), config()) is None
    assert "Refusing to derive" in caplog.text


def test_the_refusal_names_the_candidates_so_the_operator_can_transcribe_one(
    config, nics, caplog
):
    nics(["eno1", "ib0", "ib1"])
    with caplog.at_level(logging.ERROR):
        derive_net_device(_worker(ifname="eno1"), config())
    for name in ("eno1", "ib0", "ib1"):
        assert name in caplog.text
    # The remedy has to be in the message, not only in the docs.
    assert "kv_ifname" in caplog.text


def test_the_refusal_reports_the_management_nic_as_the_likely_answer(
    config, nics, caplog
):
    nics(["eno1", "ib0"])
    with caplog.at_level(logging.ERROR):
        derive_net_device(_worker(ifname="eno1"), config())
    assert "management-plane NIC is 'eno1'" in caplog.text


def test_multi_nic_refusal_is_logged_at_error_not_warning(config, nics, caplog):
    """The one caller downgrades exceptions to WARNING, so ERROR is the only way
    this stays distinguishable from the ordinary "nothing detected yet" case."""
    nics(["eno1", "ib0"])
    with caplog.at_level(logging.DEBUG):
        derive_net_device(_worker(ifname="eno1"), config())
    levels = {r.levelno for r in caplog.records if "Refusing to derive" in r.message}
    assert levels == {logging.ERROR}


def test_multi_nic_refusal_does_not_raise(config, nics):
    """Raising would be swallowed into a WARNING by the caller and would lose
    the log level; returning None keeps the placeholder unresolved instead."""
    nics(["eno1", "ib0"])
    assert derive_net_device(_worker(ifname="eno1"), config()) is None


def test_zero_candidates_keeps_the_worker_ifname_fallback(config, nics):
    """An enumeration that finds nothing must not be able to withhold a value:
    the gate exists to stop a guess, not to invent a new failure."""
    nics([])
    assert derive_net_device(_worker(ifname="eth0"), config()) == "eth0"


def test_enumeration_failure_falls_back_instead_of_refusing(config, monkeypatch):
    def _boom():
        raise OSError("no netlink for you")

    monkeypatch.setattr(net_device, "candidate_kv_interfaces", _boom)
    assert derive_net_device(_worker(ifname="eth0"), config()) == "eth0"


def test_management_nic_outside_the_candidate_set_warns_but_still_resolves(
    config, nics, caplog
):
    """A worker that reaches the server through docker0 reports docker0. That is
    known-bad, but the judgement rests on a hand-written prefix list, so a false
    positive there must not break a deployment that works today."""
    nics(["eno1"])
    with caplog.at_level(logging.WARNING):
        assert derive_net_device(_worker(ifname="docker0"), config()) == "docker0"
    assert "not among the candidate KV interfaces" in caplog.text


#
# The control plane: a recipe whose {{net_device}} carries handshake sockets.
#


def test_control_plane_derives_on_a_multi_nic_host_instead_of_refusing(
    config, nics, caplog
):
    """A host with many candidate NICs must not make `kv_ifname` mandatory for
    a value the platform already holds: `HCCL_IF_IP` is the worker's
    registration IP and HCCL requires it to sit on the interface
    `HCCL_SOCKET_IFNAME` names, which is what `Worker.ifname` is."""
    nics(["bond0", "bond1", "enp189s0f0", "enp189s0f1", "eno1", "eno2"])
    with caplog.at_level(logging.ERROR):
        assert (
            derive_net_device(
                _worker(ifname="bond1"), config(), PDNetDevicePlaneEnum.CONTROL
            )
            == "bond1"
        )
    assert "Refusing to derive" not in caplog.text


def test_the_data_plane_still_refuses_on_the_same_host(config, nics, caplog):
    """The regression that matters. Relaxing the control plane must not relax
    NIXL's: there the management NIC is usually the wrong fabric, and the
    symptom is `NIXL_ERR_BACKEND` at the peer — a whole handshake away from the
    machine that was misconfigured, and far dearer than typing one value."""
    nics(["bond0", "bond1", "enp189s0f0", "enp189s0f1", "eno1", "eno2"])
    with caplog.at_level(logging.ERROR):
        assert derive_net_device(_worker(ifname="bond1"), config()) is None
    assert "Refusing to derive" in caplog.text


def test_the_data_plane_is_what_an_undeclared_caller_gets(config, nics):
    """`plane` defaults to the stricter side, so a caller that has not been
    taught about planes keeps today's behaviour byte for byte."""
    nics(["eno1", "ib0"])
    assert derive_net_device(_worker(ifname="eno1"), config()) is None


def test_kv_ifname_still_wins_on_the_control_plane(config, nics):
    """The escape hatch is per-worker and no plane may bypass it: a machine
    whose handshake really does belong on another NIC must stay expressible."""
    nics(["bond1", "eno1"])
    assert (
        derive_net_device(
            _worker(ifname="bond1"),
            config(kv_ifname="eno1"),
            PDNetDevicePlaneEnum.CONTROL,
        )
        == "eno1"
    )


def test_control_plane_still_invents_nothing_without_a_worker_ifname(config, caplog):
    """Relaxing the gate must not relax the principle behind it. No source, no
    value — the placeholder survives into the launch, where HCCL fails with the
    literal `{{net_device}}` in its message rather than binding a wrong NIC."""
    with caplog.at_level(logging.WARNING):
        assert (
            derive_net_device(
                _worker(ifname=""), config(), PDNetDevicePlaneEnum.CONTROL
            )
            is None
        )
    assert "kv_ifname" in caplog.text


def test_control_plane_does_not_second_guess_the_worker_ifname(config, monkeypatch):
    """The host's NIC list answers a different question, so it is not consulted
    at all here — not even to warn. An interface the prefix list calls virtual
    is still the route this worker's own handshake has to take."""

    def _boom():
        raise AssertionError("the control plane must not enumerate host NICs")

    monkeypatch.setattr(net_device, "candidate_kv_interfaces", _boom)
    assert (
        derive_net_device(
            _worker(ifname="docker0"), config(), PDNetDevicePlaneEnum.CONTROL
        )
        == "docker0"
    )


#
# Host NIC enumeration.
#

_Addr = namedtuple("_Addr", ["family", "address"])
_Stats = namedtuple("_Stats", ["isup", "duplex", "speed", "mtu", "flags"])


def _up(flags="up,broadcast,running,multicast"):
    return _Stats(isup=True, duplex=0, speed=0, mtu=1500, flags=flags)


def _down():
    return _Stats(isup=False, duplex=0, speed=0, mtu=1500, flags="broadcast,multicast")


def _v4(address):
    return _Addr(family=socket.AF_INET, address=address)


def _v6(address):
    return _Addr(family=socket.AF_INET6, address=address)


@pytest.fixture
def host(monkeypatch):
    """Fake a host's NIC table: {name: (addrs, stats)}."""

    def _set(table):
        monkeypatch.setattr(
            net_device.psutil,
            "net_if_addrs",
            lambda: {name: addrs for name, (addrs, _) in table.items()},
        )
        monkeypatch.setattr(
            net_device.psutil,
            "net_if_stats",
            lambda: {name: stats for name, (_, stats) in table.items() if stats},
        )

    return _set


def test_enumeration_keeps_the_single_physical_nic(host):
    host(
        {
            "lo": ([_v4("127.0.0.1"), _v6("::1")], _up("up,loopback,running")),
            "eno1": ([_v4("192.168.50.15")], _up()),
        }
    )
    assert candidate_kv_interfaces() == ["eno1"]


def test_enumeration_excludes_the_container_and_cni_bridges(host):
    """The exact M0 host: five virtual devices, every one of them with an
    address UCX would happily advertise and no peer could route."""
    host(
        {
            "lo": ([_v4("127.0.0.1")], _up("up,loopback,running")),
            "eno1": ([_v4("192.168.50.15")], _up()),
            "docker0": ([_v4("172.17.0.1")], _up()),
            "flannel.1": ([_v4("10.42.0.0")], _up()),
            "cni0": ([_v4("10.42.0.1")], _up()),
            "br-e9790e8eabab": ([_v4("172.18.0.1")], _up()),
            "veth472ae2e": ([_v6("fe80::10ec:fcff:fec9:4849%veth472ae2e")], _up()),
        }
    )
    assert candidate_kv_interfaces() == ["eno1"]


def test_enumeration_excludes_down_and_unaddressed_ports(host):
    """Onboard ports nobody plugged in are not a reason to refuse."""
    host(
        {
            "eno1": ([_v4("192.168.50.15")], _up()),
            "eno2": ([], _down()),
            "wlp0s20f3": ([], _down()),
            "eno3": ([_v4("10.0.0.7")], _down()),
        }
    )
    assert candidate_kv_interfaces() == ["eno1"]


def test_enumeration_excludes_link_local_only_interfaces(host):
    host(
        {
            "eno1": ([_v4("192.168.50.15")], _up()),
            "eno2": ([_v4("169.254.3.4")], _up()),
            "eno3": ([_v6("fe80::1%eno3")], _up()),
        }
    )
    assert candidate_kv_interfaces() == ["eno1"]


def test_enumeration_reports_a_real_dual_homed_host(host):
    """The case the refusal is for: a management port and a fabric port, both
    real, both up, and nothing here able to tell which carries the KV plane."""
    host(
        {
            "lo": ([_v4("127.0.0.1")], _up("up,loopback,running")),
            "eno1": ([_v4("192.168.50.15")], _up()),
            "ib0": ([_v4("10.10.0.1")], _up()),
            "docker0": ([_v4("172.17.0.1")], _up()),
        }
    )
    assert candidate_kv_interfaces() == ["eno1", "ib0"]


def test_enumeration_keeps_an_ipv6_only_fabric(host):
    """A global IPv6 address is routable, so excluding it would shrink the
    candidate set and turn a refusal into a confident wrong answer."""
    host({"eno1": ([_v6("2001:db8::5")], _up())})
    assert candidate_kv_interfaces() == ["eno1"]


def test_enumeration_keeps_a_hand_made_br0_but_not_dockers_br_hex(host):
    """Docker names its user-defined bridges ``br-<hex>``; a bare ``br0`` is
    frequently the host's real uplink after the address moves onto the bridge."""
    host(
        {
            "br0": ([_v4("192.168.50.15")], _up()),
            "br-e9790e8eabab": ([_v4("172.18.0.1")], _up()),
        }
    )
    assert candidate_kv_interfaces() == ["br0"]


def test_enumeration_tolerates_a_missing_stats_entry(host):
    """psutil can report an interface in one table and not the other; dropping
    the NIC in that case would silently shrink the candidate set."""
    host({"eno1": ([_v4("192.168.50.15")], None)})
    assert candidate_kv_interfaces() == ["eno1"]


def test_enumeration_ignores_non_ip_address_families(host):
    """AF_PACKET/AF_LINK entries carry a MAC, which is not something a peer can
    connect back to."""
    link = _Addr(family=net_device.psutil.AF_LINK, address="be:fc:e7:53:52:6b")
    host({"eno1": ([link], _up())})
    assert candidate_kv_interfaces() == []


def test_the_real_host_enumeration_runs_and_excludes_loopback():
    """One unmocked call, to catch a psutil API drift the fakes would hide."""
    for name in candidate_kv_interfaces():
        assert name not in ("lo", "lo0")
