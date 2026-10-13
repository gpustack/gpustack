"""Security invariants of the worker-proxy registration plane.

Verified here:
- The production wiring never routes a connection into the server
  federation branch, which admits caller-supplied x-server-* identity.
- A client may not claim the whole address space or sockets outside the
  runtime-path allowlist, on either the direct registration path or the
  peer-broadcast path.
- An authenticated worker may only claim CIDRs covering one of its own
  addresses, and a registration never replaces a route already owned by
  another client.
"""

import asyncio
import contextlib
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock


import pytest

from gpustack.websocket_proxy import message_server
from gpustack.websocket_proxy.message import (
    ClientUpdateInfo,
    ClientUpdateMessage,
    RegisteredClientInfo,
)
from gpustack.websocket_proxy.message_server import (
    MessageServerHandler,
    websocket_endpoint,
)


def _handler() -> MessageServerHandler:
    return MessageServerHandler(
        listen_address="127.0.0.1",
        listen_port=0,
        proxy_port=0,
    )


def _websocket(headers: dict) -> MagicMock:
    websocket = MagicMock()
    websocket.headers = headers
    websocket.scope = {}
    websocket.accept = AsyncMock()
    websocket.close = AsyncMock()
    websocket.app.state = MagicMock(
        message_server_handler=None,
        websocket_authenticator=message_server.NoOpAuthenticator(),
    )
    return websocket


class TestFederationDisabledInProduction:
    """The production endpoint must not admit federation connections."""

    @pytest.mark.asyncio
    async def test_bearer_connection_with_server_headers_is_not_routed_to_federation(
        self,
    ):
        # A worker-style connection that also carries x-server-* headers
        # must be treated as a plain client connection: the federation
        # branch trusts caller-supplied identity and is not part of the
        # product deployment.
        handler = _handler()
        handler._server_info_getter = lambda websocket: None
        handler.handle_server_federation = AsyncMock()

        websocket = _websocket(
            {
                "x-server-id": str(uuid.uuid4()),
                "x-server-listen-address": "10.9.8.7",
                "x-server-proxy-port": "8080",
            }
        )

        await websocket_endpoint(websocket, handler, message_server.NoOpAuthenticator())

        handler.handle_server_federation.assert_not_awaited()
        assert handler.serving_peers == {}
        # No client info either, so the endpoint simply rejects the frame.
        websocket.close.assert_awaited()

    @pytest.mark.asyncio
    async def test_default_wiring_still_routes_server_headers_to_federation(self):
        # Guard the guard: the generic handler keeps federation for the
        # standalone deployment (websocket_proxy.main), so the production
        # fix must live in the app wiring, not in the protocol module.
        handler = _handler()
        server_id = uuid.uuid4()
        handler.handle_server_federation = AsyncMock()

        websocket = _websocket({"x-server-id": str(server_id)})

        await websocket_endpoint(websocket, handler, message_server.NoOpAuthenticator())

        handler.handle_server_federation.assert_awaited_once()


class TestClientClaimValidation:
    """A client's CIDR / unix-socket claims are policy-checked."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "cidrs",
        [
            ["0.0.0.0/0"],
            ["::/0"],
            ["10.0.0.0/7"],
            ["not-a-cidr"],
        ],
    )
    async def test_connection_with_broad_cidr_claim_is_rejected(self, cidrs):
        handler = _handler()
        client_info = RegisteredClientInfo(
            client_id=uuid.uuid4(), cidrs=cidrs, unix_sockets=[]
        )

        await handler.handle_client_connection(_websocket({}), client_info)

        assert handler.client_registry == {}
        assert handler._cidr_registry.find_best_match("10.1.2.3") is None

    @pytest.mark.asyncio
    async def test_connection_with_out_of_allowlist_socket_is_rejected(self):
        handler = _handler()
        client_info = RegisteredClientInfo(
            client_id=uuid.uuid4(),
            cidrs=["10.0.0.5/32"],
            unix_sockets=["/home/user/.docker/run/docker.sock"],
        )

        await handler.handle_client_connection(_websocket({}), client_info)

        assert handler.client_registry == {}

    @pytest.mark.asyncio
    async def test_connection_with_worker_style_claims_is_accepted(self):
        handler = _handler()
        client_id = uuid.uuid4()
        client_info = RegisteredClientInfo(
            client_id=client_id,
            cidrs=["10.0.0.5/32"],
            unix_sockets=["/var/run/docker.sock"],
        )
        websocket = _websocket({})
        # Keep handle_client from looping on a mocked receive.
        websocket.receive = AsyncMock(side_effect=[{"type": "websocket.disconnect"}])

        await handler.handle_client_connection(websocket, client_info)

        # Registration passed the claim checks: the connection was accepted
        # and never closed with a policy violation.
        websocket.accept.assert_awaited_once()
        websocket.close.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_peer_update_with_broad_cidr_claim_is_dropped(self):
        handler = _handler()
        client_id = uuid.uuid4()

        msg = ClientUpdateMessage(
            server_id=uuid.uuid4(),
            updates=[
                ClientUpdateInfo(client_id=client_id, action="add", cidrs=["0.0.0.0/0"])
            ],
        )
        await handler.handle_peer_client_update(msg)

        assert client_id not in handler.client_registry

    @pytest.mark.asyncio
    async def test_peer_update_with_valid_claims_is_applied(self):
        handler = _handler()
        client_id = uuid.uuid4()

        msg = ClientUpdateMessage(
            server_id=uuid.uuid4(),
            updates=[
                ClientUpdateInfo(
                    client_id=client_id, action="add", cidrs=["10.0.0.9/32"]
                )
            ],
        )
        await handler.handle_peer_client_update(msg)

        assert client_id in handler.client_registry


class _WorkerAuthenticator:
    """Authenticator that binds a worker identity to the connection.

    Mirrors what ``BearerTokenAuthenticator`` does in production: the
    authenticated worker is stored on the websocket scope and is the
    identity every claim is validated against. ``ip=None`` models a
    worker that reported no address.
    """

    def __init__(self, ip=None):
        self.worker = SimpleNamespace(ip=ip or "", advertise_address=None)

    async def authenticate(self, websocket) -> bool:
        websocket.scope["user"] = SimpleNamespace(worker=self.worker)
        return True


async def _register_through_endpoint(handler, authenticator, headers) -> MagicMock:
    """Run a client registration through ``websocket_endpoint``."""
    websocket = _websocket(headers)
    websocket.receive = AsyncMock(side_effect=[{"type": "websocket.disconnect"}])
    await websocket_endpoint(websocket, handler, authenticator)
    return websocket


async def _wait_for_registration(handler, client_id: uuid.UUID) -> None:
    for _ in range(100):
        if client_id in handler.client_registry:
            return
        await asyncio.sleep(0.01)
    assert client_id in handler.client_registry


class TestRouteOwnership:
    """A registration may never capture another worker's route."""

    @pytest.mark.asyncio
    async def test_worker_cannot_hijack_another_workers_route(self):
        # Two workers: the victim registers 10.0.0.5/32 and stays
        # connected; the attacker, authenticated as its own worker with
        # its own x-client-id, then claims the victim's exact /32. The
        # registration must be rejected and the victim's IP must keep
        # resolving to the victim's connection manager.
        handler = _handler()
        handler._server_info_getter = lambda websocket: None

        victim_id = uuid.uuid4()
        victim_ws = _websocket(
            {"x-client-id": str(victim_id), "x-cidrs": "10.0.0.5/32"}
        )
        # Block in the receive loop so the victim stays registered.
        victim_ws.receive = AsyncMock(side_effect=asyncio.Event().wait)

        victim_task = asyncio.create_task(
            websocket_endpoint(victim_ws, handler, _WorkerAuthenticator(ip="10.0.0.5"))
        )
        await _wait_for_registration(handler, victim_id)

        attacker_id = uuid.uuid4()
        attacker_ws = await _register_through_endpoint(
            handler,
            _WorkerAuthenticator(ip="10.0.0.9"),
            {"x-client-id": str(attacker_id), "x-cidrs": "10.0.0.5/32"},
        )

        assert attacker_id not in handler.client_registry
        attacker_ws.close.assert_awaited()
        assert (
            handler.get_connection_manager_by_ip_in_cidr("10.0.0.5")
            is handler.connection_managers[victim_id]
        )

        victim_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await victim_task

    @pytest.mark.asyncio
    async def test_worker_claim_outside_its_addresses_is_rejected(self):
        handler = _handler()
        handler._server_info_getter = lambda websocket: None

        attacker_id = uuid.uuid4()
        attacker_ws = await _register_through_endpoint(
            handler,
            _WorkerAuthenticator(ip="10.0.0.9"),
            {"x-client-id": str(attacker_id), "x-cidrs": "10.0.0.99/32"},
        )

        assert handler.client_registry == {}
        attacker_ws.close.assert_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "cidrs",
        [
            ["10.0.0.0/25"],
            ["10.0.0.0/24"],
            ["10.0.0.8/31"],
        ],
    )
    async def test_worker_subnet_claim_is_rejected_in_production_wiring(self, cidrs):
        # Even a subnet that covers the worker's own address is rejected:
        # a more specific claim inside another worker's subnet would win
        # longest-prefix match and steal the overlapped traffic. Subnet
        # claims are only legal in the generic protocol module.
        handler = _handler()
        handler._server_info_getter = lambda websocket: None

        worker_id = uuid.uuid4()
        worker_ws = await _register_through_endpoint(
            handler,
            _WorkerAuthenticator(ip="10.0.0.9"),
            {"x-client-id": str(worker_id), "x-cidrs": ",".join(cidrs)},
        )

        assert handler.client_registry == {}
        worker_ws.close.assert_awaited()

    @pytest.mark.asyncio
    async def test_subnet_claim_stays_legal_in_generic_module(self):
        # The standalone protocol module keeps subnet registration: no
        # authorized addresses are passed, so only the generic policy
        # (prefix length / socket allowlist) applies.
        handler = _handler()
        client_id = uuid.uuid4()
        websocket = _websocket({})
        websocket.receive = AsyncMock(side_effect=[{"type": "websocket.disconnect"}])

        await handler.handle_client_connection(
            websocket,
            RegisteredClientInfo(
                client_id=client_id, cidrs=["10.0.0.0/24"], unix_sockets=[]
            ),
        )

        websocket.accept.assert_awaited_once()
        websocket.close.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "cidrs",
        [
            ["10.0.0.0/8"],
            ["10.0.0.9/32"],
            ["10.0.0.0/25", "10.0.0.9/32"],
        ],
    )
    async def test_addressless_worker_may_not_claim_any_route(self, cidrs):
        # An authenticated worker that reported no address must not fall
        # back to the generic protocol policy (which permits subnets): an
        # empty authorized list rejects every CIDR claim instead.
        handler = _handler()
        handler._server_info_getter = lambda websocket: None

        worker_id = uuid.uuid4()
        worker_ws = await _register_through_endpoint(
            handler,
            _WorkerAuthenticator(ip=None),
            {"x-client-id": str(worker_id), "x-cidrs": ",".join(cidrs)},
        )

        assert handler.client_registry == {}
        worker_ws.close.assert_awaited()

    @pytest.mark.asyncio
    async def test_worker_own_address_claim_is_accepted(self):
        handler = _handler()
        handler._server_info_getter = lambda websocket: None

        worker_id = uuid.uuid4()
        worker_ws = await _register_through_endpoint(
            handler,
            _WorkerAuthenticator(ip="10.0.0.9"),
            {"x-client-id": str(worker_id), "x-cidrs": "10.0.0.9/32"},
        )

        # The registration passed every claim check: the connection was
        # accepted and never closed with a policy violation. (The mocked
        # receive then disconnects, so the registry cannot be asserted.)
        worker_ws.accept.assert_awaited_once()
        worker_ws.close.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_same_client_may_re_register_its_own_route(self):
        # A worker reconnects (same client_id) while its previous
        # connection is still registered: re-claiming and extending its
        # own routes must stay possible.
        handler = _handler()

        worker_id = uuid.uuid4()
        first_ws = _websocket({})
        first_ws.receive = AsyncMock(side_effect=asyncio.Event().wait)
        first_task = asyncio.create_task(
            handler.handle_client_connection(
                first_ws,
                RegisteredClientInfo(
                    client_id=worker_id, cidrs=["10.0.0.5/32"], unix_sockets=[]
                ),
            )
        )
        await _wait_for_registration(handler, worker_id)

        reconnected_ws = _websocket({})
        reconnected_ws.receive = AsyncMock(side_effect=asyncio.Event().wait)
        reconnected_task = asyncio.create_task(
            handler.handle_client_connection(
                reconnected_ws,
                RegisteredClientInfo(
                    client_id=worker_id,
                    cidrs=["10.0.0.5/32", "10.0.0.7/32"],
                    unix_sockets=[],
                ),
            )
        )
        for _ in range(100):
            if handler._cidr_registry.find_exact("10.0.0.7/32") == worker_id:
                break
            await asyncio.sleep(0.01)

        # The re-claim of its own /32 was not treated as a foreign claim.
        assert handler._cidr_registry.find_exact("10.0.0.5/32") == worker_id
        assert handler._cidr_registry.find_exact("10.0.0.7/32") == worker_id

        for task in (first_task, reconnected_task):
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    @pytest.mark.asyncio
    async def test_peer_update_cannot_replace_another_clients_route(self):
        handler = _handler()

        victim_id = uuid.uuid4()
        attacker_id = uuid.uuid4()
        server_id = uuid.uuid4()

        await handler.handle_peer_client_update(
            ClientUpdateMessage(
                server_id=server_id,
                updates=[
                    ClientUpdateInfo(
                        client_id=victim_id, action="add", cidrs=["10.0.0.5/32"]
                    )
                ],
            )
        )
        await handler.handle_peer_client_update(
            ClientUpdateMessage(
                server_id=server_id,
                updates=[
                    ClientUpdateInfo(
                        client_id=attacker_id, action="add", cidrs=["10.0.0.5/32"]
                    )
                ],
            )
        )

        assert attacker_id not in handler.client_registry
        assert handler._cidr_registry.find_exact("10.0.0.5/32") == victim_id
