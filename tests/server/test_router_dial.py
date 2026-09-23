"""Which address the server dials to reach a group's router.

The two paths want different answers, and getting it wrong is silent. A
worker's `ip` is what that worker reported about itself; on a cloud host it is
routinely a VPC address only that host's own network can route. The server
sits outside that network, so dialling `ip` directly fails — while the tunnel
proxy, which leaves from the worker's own side, needs exactly that `ip` and
would be broken by the published address instead.

Measured on a cloud worker whose `ip` was `10.0.0.37` and whose
`advertise_address` was reachable: every registry read failed, so no member
was ever registered, and the group served 503s while every member reported
RUNNING.
"""

from types import SimpleNamespace

import pytest

from gpustack.schemas.config import ModelInstanceProxyModeEnum
from gpustack.schemas.workers import Worker
from gpustack.server import controllers
from gpustack.server.controllers import _router_dial


def _instance(worker_id=1, worker_ip="10.0.0.37", port=40008):
    return SimpleNamespace(worker_id=worker_id, worker_ip=worker_ip, port=port)


def _worker(
    ip,
    advertise_address=None,
    proxy_mode=ModelInstanceProxyModeEnum.WORKER,
    proxy_address=None,
):
    return Worker(
        name="w",
        hostname="w",
        ip=ip,
        advertise_address=advertise_address,
        proxy_mode=proxy_mode,
        proxy_address=proxy_address,
        token="worker-token",
    )


@pytest.fixture
def load_worker(monkeypatch):
    """Patch the row lookup; nothing here needs a database."""

    def _install(result):
        async def _one_by_id(_session, _id):
            if isinstance(result, Exception):
                raise result
            return result

        monkeypatch.setattr(controllers.Worker, "one_by_id", _one_by_id)

    return _install


@pytest.mark.asyncio
async def test_the_tunnel_dials_the_workers_own_ip_not_the_published_one(load_worker):
    """The proxy forwards from inside the worker's network.

    Handing it the published address would send the request back out and in
    again, which is the one address that network may not resolve.
    """
    load_worker(
        _worker(
            ip="192.168.50.15",
            advertise_address="203.0.113.9",
            proxy_mode=ModelInstanceProxyModeEnum.TUNNEL,
            proxy_address="http://127.0.0.1:30079",
        )
    )

    host, proxy, token = await _router_dial(None, _instance(worker_ip="192.168.50.15"))

    assert host == "192.168.50.15"
    assert proxy == "http://127.0.0.1:30079"
    assert token == "worker-token"


@pytest.mark.asyncio
async def test_direct_dials_the_address_the_worker_publishes(load_worker):
    """The regression: `ip` is unroutable from the server, the published one is not."""
    load_worker(_worker(ip="10.0.0.37", advertise_address="89.169.112.94"))

    host, proxy, token = await _router_dial(None, _instance(worker_ip="10.0.0.37"))

    assert host == "89.169.112.94"
    assert proxy is None
    # No hop to authorise, so handing over the worker's token would be giving
    # a credential to something that never asked for one.
    assert token is None


@pytest.mark.asyncio
async def test_direct_falls_back_to_the_reported_ip_when_nothing_is_published(
    load_worker,
):
    """A single-network deployment publishes nothing and must keep working."""
    load_worker(_worker(ip="192.168.50.15", advertise_address=None))

    host, proxy, _ = await _router_dial(None, _instance(worker_ip="192.168.50.15"))

    assert host == "192.168.50.15"
    assert proxy is None


@pytest.mark.asyncio
async def test_an_unreadable_worker_row_still_dials_something(load_worker):
    """Degrade to the instance's own address rather than to no attempt at all."""
    load_worker(RuntimeError("database is gone"))

    host, proxy, token = await _router_dial(None, _instance(worker_ip="10.0.0.37"))

    assert host == "10.0.0.37"
    assert (proxy, token) == (None, None)


@pytest.mark.asyncio
async def test_a_missing_worker_row_still_dials_something(load_worker):
    load_worker(None)

    host, _, _ = await _router_dial(None, _instance(worker_ip="10.0.0.37"))

    assert host == "10.0.0.37"


@pytest.mark.asyncio
async def test_an_instance_with_no_worker_dials_its_own_recorded_address():
    """No row to consult, so the address the instance carries is all there is."""
    host, proxy, token = await _router_dial(
        None, _instance(worker_id=None, worker_ip="10.0.0.37")
    )

    assert host == "10.0.0.37"
    assert (proxy, token) == (None, None)


def test_the_published_address_wins_for_a_direct_dial():
    """`get_dial_address` is the one place this preference is expressed."""
    assert (
        _worker(ip="10.0.0.37", advertise_address="89.169.112.94").get_dial_address()
        == "89.169.112.94"
    )
    assert _worker(ip="10.0.0.37").get_dial_address() == "10.0.0.37"


@pytest.mark.parametrize(
    "proxy_mode",
    [
        ModelInstanceProxyModeEnum.WORKER,
        ModelInstanceProxyModeEnum.DIRECT,
        ModelInstanceProxyModeEnum.DELEGATED,
    ],
)
@pytest.mark.asyncio
async def test_every_non_tunnel_mode_dials_direct_at_the_published_address(
    load_worker, proxy_mode
):
    """Only `tunnel` has a hop. The rest all dial the worker themselves, so
    they all need the address the worker publishes rather than the one it sees
    itself at."""
    load_worker(
        _worker(
            ip="10.0.0.37",
            advertise_address="89.169.112.94",
            proxy_mode=proxy_mode,
        )
    )

    host, proxy, token = await _router_dial(None, _instance(worker_ip="10.0.0.37"))

    assert host == "89.169.112.94"
    assert (proxy, token) == (None, None)


@pytest.mark.parametrize(
    "proxy_mode",
    [
        ModelInstanceProxyModeEnum.WORKER,
        ModelInstanceProxyModeEnum.DIRECT,
        ModelInstanceProxyModeEnum.DELEGATED,
        ModelInstanceProxyModeEnum.TUNNEL,
    ],
)
@pytest.mark.asyncio
async def test_a_worker_that_publishes_nothing_is_dialled_exactly_as_before(
    load_worker, proxy_mode
):
    """The no-regression guarantee for every existing deployment.

    A worker that was never given `--advertise-address` reports the address it
    sees itself at (`collector.py`: `advertise_address or worker_ip`), so both
    fields hold the same value and every mode keeps dialling what it always
    dialled.
    """
    load_worker(
        _worker(
            ip="192.168.50.15",
            advertise_address="192.168.50.15",
            proxy_mode=proxy_mode,
            proxy_address="http://127.0.0.1:30079",
        )
    )

    host, _, _ = await _router_dial(None, _instance(worker_ip="192.168.50.15"))

    assert host == "192.168.50.15"
