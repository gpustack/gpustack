"""Unit tests for IPv4/IPv6 network utilities and internal URL formatting."""

from collections import namedtuple
import socket
from types import SimpleNamespace

import pytest

from gpustack.exporter.exporter import _parse_scrape_address
from gpustack.gateway.utils import (
    _worker_reserve_proxy_registry,
    _worker_tunnel_proxy_registry,
    model_instance_registry,
)
from gpustack.schemas.config import ModelInstanceProxyModeEnum
from gpustack.schemas.models import ModelInstance, PortBand
from gpustack.schemas.workers import Worker
from gpustack.server import pd_membership
from gpustack.server.worker_request import _build_url
from gpustack.utils import network
from gpustack.utils.network import (
    _get_no_proxy_cidrs,
    format_bind_address,
    format_url_host,
    get_first_non_loopback_ip,
    get_ifname_by_ip_hostname,
    host_cidr,
    join_host_port,
    loopback_host,
    resolve_bind_host,
    strip_host_brackets,
    use_proxy_env_for_url,
)
from gpustack.utils.sglang_port_layout import SGLANG_CONNECTING_BAND, sglang_port_layout
from gpustack.websocket_proxy.proxy_server import HTTPSProxyServer, _split_authority
from gpustack.worker.pd_router import PeerAddress, _peer_scope

_SnicAddr = namedtuple("_SnicAddr", ["family", "address"])


@pytest.mark.parametrize(
    ("raw", "expected_host", "expected_host_port"),
    [
        ("10.0.0.5", "10.0.0.5", "10.0.0.5:10150"),
        ("worker.internal", "worker.internal", "worker.internal:10150"),
        ("240b:8168:10:4::64", "[240b:8168:10:4::64]", "[240b:8168:10:4::64]:10150"),
        ("[240b:8168:10:4::64]", "[240b:8168:10:4::64]", "[240b:8168:10:4::64]:10150"),
        ("fe80::1%eth0", "[fe80::1]", "[fe80::1]:10150"),
        ("[fe80::1%eth0]", "[fe80::1]", "[fe80::1]:10150"),
        ("::1", "[::1]", "[::1]:10150"),
        ("[::1]", "[::1]", "[::1]:10150"),
    ],
)
def test_format_url_host_and_join_host_port(raw, expected_host, expected_host_port):
    assert format_url_host(raw) == expected_host
    assert join_host_port(raw, 10150) == expected_host_port


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("[::1]", "::1"),
        ("::1", "::1"),
        ("10.0.0.1", "10.0.0.1"),
        ("", ""),
    ],
)
def test_strip_host_brackets(raw, expected):
    assert strip_host_brackets(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("10.0.0.5", "10.0.0.5/32"),
        ("240b:8168:10:4::64", "240b:8168:10:4::64/128"),
        ("[240b:8168:10:4::64]", "240b:8168:10:4::64/128"),
        ("fd00::1%eth0", "fd00::1/128"),
        (None, None),
        ("", None),
        ("not-an-ip", None),
    ],
)
def test_host_cidr(raw, expected):
    assert host_cidr(raw) == expected


@pytest.mark.parametrize(
    ("configured", "v4_ok", "v6_ok", "expected_bind", "expected_loopback"),
    [
        (None, True, True, None, "127.0.0.1"),
        (None, True, False, "0.0.0.0", "127.0.0.1"),
        (None, False, True, "::", "::1"),
        ("[::1]", True, True, "::1", "127.0.0.1"),
        ("240b::64%eth0", False, True, "240b::64", "::1"),
    ],
)
def test_resolve_bind_host_and_loopback_host(
    monkeypatch, configured, v4_ok, v6_ok, expected_bind, expected_loopback
):
    monkeypatch.setattr(
        network,
        "_can_bind_ephemeral",
        lambda family, _host: v4_ok if family == socket.AF_INET else v6_ok,
    )
    assert resolve_bind_host(configured) == expected_bind
    assert loopback_host() == expected_loopback


def test_format_bind_address():
    assert format_bind_address(None, 8080) == "0.0.0.0:8080 and [::]:8080"
    assert format_bind_address("0.0.0.0", 8080) == "0.0.0.0:8080"
    assert format_bind_address("::", 8080) == "[::]:8080"


class TestGetFirstNonLoopbackIP:
    def test_prefers_ipv4_on_dual_stack_hosts(self, monkeypatch):
        monkeypatch.setattr(
            network.psutil,
            "net_if_addrs",
            lambda: {
                "eth0": [
                    _SnicAddr(socket.AF_INET6, "240b:8168:10:4::64"),
                    _SnicAddr(socket.AF_INET, "10.20.30.40"),
                ]
            },
        )
        assert get_first_non_loopback_ip() == "10.20.30.40"

    def test_falls_back_to_global_ipv6_on_pure_ipv6_hosts(self, monkeypatch):
        monkeypatch.setattr(
            network.psutil,
            "net_if_addrs",
            lambda: {
                "lo": [_SnicAddr(socket.AF_INET6, "::1")],
                "eth0": [
                    _SnicAddr(socket.AF_INET6, "fe80::1%eth0"),
                    _SnicAddr(socket.AF_INET6, "fd00::10%eth0"),
                    _SnicAddr(socket.AF_INET6, "240b:8168:10:4::64%eth0"),
                ],
            },
        )
        assert get_first_non_loopback_ip() == "240b:8168:10:4::64"

    def test_accepts_ula_ipv6_when_no_global_address_exists(self, monkeypatch):
        monkeypatch.setattr(
            network.psutil,
            "net_if_addrs",
            lambda: {
                "eth0": [
                    _SnicAddr(socket.AF_INET6, "fe80::1%eth0"),
                    _SnicAddr(socket.AF_INET6, "fd00::42%eth0"),
                ],
            },
        )
        assert get_first_non_loopback_ip(expected_ifname="eth0") == "fd00::42"

    def test_raises_when_only_loopback_and_link_local_exist(self, monkeypatch):
        monkeypatch.setattr(
            network.psutil,
            "net_if_addrs",
            lambda: {
                "lo": [
                    _SnicAddr(socket.AF_INET, "127.0.0.1"),
                    _SnicAddr(socket.AF_INET6, "::1"),
                ],
                "eth0": [
                    _SnicAddr(socket.AF_INET, "169.254.1.2"),
                    _SnicAddr(socket.AF_INET6, "fe80::1%eth0"),
                ],
            },
        )
        with pytest.raises(Exception, match="No non-loopback IP address found"):
            get_first_non_loopback_ip()


class TestGetIfnameByIPHostname:
    def test_matches_local_ipv6_with_scope_and_non_canonical_spelling(
        self, monkeypatch
    ):
        monkeypatch.setattr(
            network.psutil,
            "net_if_addrs",
            lambda: {
                "eth1": [_SnicAddr(socket.AF_INET6, "240b:8168:10:4::64%eth1")],
            },
        )
        assert get_ifname_by_ip_hostname("240b:8168:0010:0004::0064") == "eth1"
        assert get_ifname_by_ip_hostname("[240b:8168:10:4::64]") == "eth1"

    def test_resolves_remote_ipv6_route_via_udp_socket(self, monkeypatch):
        monkeypatch.setattr(
            network.psutil,
            "net_if_addrs",
            lambda: {
                "eno1": [_SnicAddr(socket.AF_INET6, "240b:8168:10:4::64")],
            },
        )

        class _FakeUDPSocket:
            def __init__(self, af, _socktype):
                assert af == socket.AF_INET6

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def connect(self, _addr):
                return None

            def getsockname(self):
                return ("240b:8168:10:4::64", 54321, 0, 0)

        monkeypatch.setattr(network.socket, "socket", _FakeUDPSocket)
        assert get_ifname_by_ip_hostname("240b:8168:10:4::1") == "eno1"


def test_use_proxy_env_for_url_supports_mixed_ipv4_and_ipv6_no_proxy(monkeypatch):
    _get_no_proxy_cidrs.cache_clear()
    monkeypatch.setenv("NO_PROXY", "10.0.0.0/8,[fd00::]/8,fe80::%eth0/64,localhost")
    try:
        assert use_proxy_env_for_url("http://[fd00::1234]:10150/healthz") is False
        assert use_proxy_env_for_url("http://[fe80::abcd]:10150/healthz") is False
        assert use_proxy_env_for_url("http://[240b:8168::1]:10150/healthz") is True
        assert use_proxy_env_for_url("http://10.1.2.3:10150/healthz") is False
        assert use_proxy_env_for_url("http://203.0.113.5:10150/healthz") is True
    finally:
        _get_no_proxy_cidrs.cache_clear()


class TestInternalIPv6URLCallers:
    def test_worker_request_build_url_brackets_ipv6(self):
        worker = Worker(
            id=1,
            name="w-v6",
            hostname="w-v6",
            ip="240b:8168:10:4::64",
            port=10150,
            proxy_mode=ModelInstanceProxyModeEnum.WORKER,
        )
        assert (
            _build_url(worker, "/healthz")
            == "http://[240b:8168:10:4::64]:10150/healthz"
        )

    def test_worker_compute_state_unreachable_brackets_ipv6(self):
        from datetime import datetime, timezone

        worker = Worker(
            id=1,
            name="w-v6",
            hostname="w-v6",
            ip="240b:8168:10:4::64",
            port=10150,
            unreachable=True,
            heartbeat_time=datetime.now(timezone.utc),
        )
        worker.compute_state()
        assert "http://[240b:8168:10:4::64]:10150/healthz" in worker.state_message

    def test_gateway_registries_bracket_ipv6_domains(self):
        mi = ModelInstance(
            id=10,
            name="mi-1",
            model_id=1,
            model_name="qwen",
            worker_id=2,
            worker_ip="240b:8168:10:4::64",
            port=40001,
        )
        direct_reg = model_instance_registry(mi)
        assert direct_reg.type == "static"
        assert direct_reg.domain == "[240b:8168:10:4::64]:40001"

        worker = Worker(
            id=2,
            cluster_id=1,
            name="w-v6",
            hostname="w-v6",
            ip="240b:8168:10:4::64",
            port=10150,
            proxy_mode=ModelInstanceProxyModeEnum.WORKER,
        )
        worker_reg = _worker_reserve_proxy_registry(worker)
        assert worker_reg.type == "static"
        assert worker_reg.domain == "[240b:8168:10:4::64]:10150"

        tunnel_worker = Worker(
            id=3,
            cluster_id=1,
            name="w-tun",
            hostname="w-tun",
            ip="fd00::99",
            port=10150,
            proxy_mode=ModelInstanceProxyModeEnum.TUNNEL,
            proxy_address="http://[240b:8168:10:4::1]:30079",
        )
        tunnel_reg = _worker_tunnel_proxy_registry(tunnel_worker)
        assert tunnel_reg.type == "static"
        assert tunnel_reg.domain == "[240b:8168:10:4::1]:30079"

    def test_sglang_multinode_dist_init_addr_brackets_ipv6(self):
        model = SimpleNamespace(
            backend="SGLang",
            categories=["llm"],
            backend_parameters=[],
            backend_version="v0.4.9",
            env={},
            disaggregation=None,
        )
        layout = sglang_port_layout(model, None, [1, 1])
        instance = SimpleNamespace(
            worker_ip="240b:8168:10:4::64",
            named_ports={
                SGLANG_CONNECTING_BAND: PortBand(
                    name=SGLANG_CONNECTING_BAND, base=41000, count=2
                )
            },
            port=40001,
            ports=[40001],
            distributed_servers=None,
        )
        bindings = layout.bind(instance, "240b:8168:10:4::64", is_leader=True)
        assert bindings.arguments[:2] == (
            "--dist-init-addr",
            "[240b:8168:10:4::64]:41000",
        )

    def test_pd_membership_and_router_peer_scope_bracket_ipv6(self):
        scope = pd_membership._scope(
            "http://[240b:8168:10:4::64]:40010", "qwen", {"bootstrap": 9002}
        )
        assert (
            pd_membership._render("http://{{peer.ip}}:{{peer.port}}", scope)
            == "http://[240b:8168:10:4::64]:40010"
        )
        assert pd_membership._render("{{peer.ip}}", scope) == "240b:8168:10:4::64"

        router_scope = _peer_scope(
            PeerAddress(ip="240b:8168:10:4::64", port=40010, ports={"bootstrap": 9002})
        )
        assert router_scope["peer.ip"] == "[240b:8168:10:4::64]"
        assert router_scope["peer.url"] == "http://[240b:8168:10:4::64]:40010"

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (
                "http://[240b:8168:10:4::64]:9090/metrics",
                ("[240b:8168:10:4::64]:9090", "/metrics", "http"),
            ),
            (
                "[240b:8168:10:4::64]:9100",
                ("[240b:8168:10:4::64]:9100", "/metrics", "http"),
            ),
        ],
    )
    def test_exporter_parse_scrape_address_brackets_ipv6(self, raw, expected):
        assert _parse_scrape_address(raw) == expected

    @pytest.mark.asyncio
    async def test_websocket_proxy_extracts_bare_ipv6_target(self):
        assert _split_authority("[240b:8168:10:4::64]:10150") == (
            "240b:8168:10:4::64",
            10150,
        )
        assert _split_authority("240b:8168:10:4::64", default_port=443) == (
            "240b:8168:10:4::64",
            443,
        )

        proxy = HTTPSProxyServer(
            host=None, port=0, connection_manager_getter=lambda _: None
        )
        assert (
            await proxy._get_target_ip("CONNECT", "[240b:8168:10:4::64]:10150", {})
            == "240b:8168:10:4::64"
        )
        assert (
            await proxy._get_target_ip(
                "GET", "/healthz", {"host": "[240b:8168:10:4::64]:10150"}
            )
            == "240b:8168:10:4::64"
        )

    @pytest.mark.asyncio
    async def test_remote_connection_manager_brackets_ipv6_connect_target(
        self, monkeypatch
    ):
        from gpustack.websocket_proxy import connection_manager as cm_mod

        captured = {}

        class _FakeReader:
            async def read(self, _n: int) -> bytes:
                return b"HTTP/1.1 200 Connection Established\r\n\r\n"

        class _FakeWriter:
            def write(self, data: bytes) -> None:
                captured["request"] = data.decode("utf-8")

            async def drain(self) -> None:
                return None

            def close(self) -> None:
                return None

            async def wait_closed(self) -> None:
                return None

        async def _fake_open_connection(host: str, port: int):
            captured["peer"] = (host, port)
            return _FakeReader(), _FakeWriter()

        monkeypatch.setattr(cm_mod.asyncio, "open_connection", _fake_open_connection)
        mgr = cm_mod.RemoteConnectionManager("[240b:8168:10:4::1]", 30079)
        await mgr.connect("tcp://[240b:8168:10:4::64]:10150")

        assert captured["peer"] == ("240b:8168:10:4::1", 30079)
        assert captured["request"] == (
            "CONNECT [240b:8168:10:4::64]:10150 HTTP/1.1\r\n"
            "Host: [240b:8168:10:4::64]:10150\r\n\r\n"
        )
