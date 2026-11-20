"""Tunnel liveness, release, and peer-update ordering invariants.

Verified here:
- A released client's own socket close does not fire the disconnect callback
  (generation advanced by the release), so a deleted worker's proxy_address
  is not resurrected.
- A tunnel that has proven liveness (heartbeat) and then goes silent is
  reaped; a client that never heartbeats is never reaped.
- ``force_release`` reaches the owning server through the ``force_remove``
  peer update when the local replica holds nothing.
- A reconnect's registration waits for an in-flight ``remove`` broadcast, so
  peers observe ``remove`` before ``add`` for the same client.
"""

import asyncio
import contextlib
import logging
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from gpustack.websocket_proxy.message import (
    ClientUpdateInfo,
    ClientUpdateMessage,
    HeartbeatMessage,
    RegisteredClientInfo,
    pack_message,
    parse_message,
)
from gpustack.websocket_proxy.message_server import MessageServerHandler


def _handler(**kwargs) -> MessageServerHandler:
    return MessageServerHandler(
        listen_address="127.0.0.1",
        listen_port=0,
        proxy_port=0,
        **kwargs,
    )


def _websocket() -> MagicMock:
    websocket = MagicMock()
    websocket.accept = AsyncMock()
    websocket.close = AsyncMock()
    websocket.send_bytes = AsyncMock()
    return websocket


def _scripted_websocket(messages):
    """A websocket whose receive() yields the scripted messages, then idles.

    ``receive`` returns a disconnect once the socket is closed, so the reap
    watchdog's ``close()`` unwinds the receive loop.
    """
    inbox = asyncio.Queue()
    for message in messages:
        inbox.put_nowait(message)
    websocket = _websocket()
    websocket.closed = False

    async def close(**kwargs):
        websocket.closed = True

    async def receive():
        while True:
            if websocket.closed:
                return {"type": "websocket.disconnect"}
            try:
                return await asyncio.wait_for(inbox.get(), timeout=0.005)
            except asyncio.TimeoutError:
                continue

    websocket.close = AsyncMock(side_effect=close)
    websocket.receive = receive
    return websocket


def _disconnecting_websocket():
    """A websocket whose receive() blocks until ``disconnect`` is called."""
    inbox = asyncio.Queue()
    websocket = _websocket()
    websocket.receive = inbox.get
    websocket.disconnect = lambda: inbox.put_nowait({"type": "websocket.disconnect"})
    return websocket


async def _wait_for_registration(handler, client_id, websocket):
    for _ in range(100):
        manager = handler.connection_managers.get(client_id)
        if manager is not None and manager.websocket is websocket:
            return
        await asyncio.sleep(0.01)
    pytest.fail(f"client {client_id} did not register")


def _heartbeat_frame() -> dict:
    return {"bytes": pack_message(HeartbeatMessage(timestamp=1))}


class TestReleasedClientCallbackSuppressed:
    @pytest.mark.asyncio
    async def test_socket_close_after_release_does_not_fire_disconnect_callback(
        self,
    ):
        handler = _handler()
        events = []

        async def on_disconnect(server, client):
            events.append("disconnect")

        handler._callback_on_disconnect = on_disconnect
        client_id = uuid.uuid4()
        websocket = _disconnecting_websocket()

        task = asyncio.create_task(
            handler.handle_client_connection(
                websocket,
                RegisteredClientInfo(client_id=client_id, cidrs=["10.0.0.1/32"]),
            )
        )
        await _wait_for_registration(handler, client_id, websocket)

        await handler.release_client(client_id)
        websocket.disconnect()
        await asyncio.wait_for(task, timeout=5)

        assert events == []
        assert handler.get_connection_manager("10.0.0.1") is None


class TestStaleTunnelReaping:
    @pytest.mark.asyncio
    async def test_silent_tunnel_with_heartbeat_is_reaped(self):
        handler = _handler(client_read_timeout=0.05)
        websocket = _scripted_websocket([_heartbeat_frame()])
        client_id = uuid.uuid4()

        await asyncio.wait_for(
            handler.handle_client_connection(
                websocket,
                RegisteredClientInfo(client_id=client_id, cidrs=["10.0.0.2/32"]),
            ),
            timeout=5,
        )

        assert handler.get_connection_manager("10.0.0.2") is None
        assert client_id not in handler.client_registry

    @pytest.mark.asyncio
    async def test_tunnel_without_heartbeat_is_not_reaped(self):
        handler = _handler(client_read_timeout=0.05)
        websocket = _scripted_websocket([])
        client_id = uuid.uuid4()

        task = asyncio.create_task(
            handler.handle_client_connection(
                websocket,
                RegisteredClientInfo(client_id=client_id, cidrs=["10.0.0.3/32"]),
            )
        )
        await _wait_for_registration(handler, client_id, websocket)

        # Several read timeouts pass; the silent old-version client stays.
        await asyncio.sleep(0.2)
        assert not task.done()
        assert handler.get_connection_manager("10.0.0.3") is not None

        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


class TestForcedRelease:
    @pytest.mark.asyncio
    async def test_force_remove_releases_tunnel_on_owning_server(self):
        handler = _handler()
        handler._enqueue_client_update = MagicMock()
        client_id = uuid.uuid4()
        websocket = _disconnecting_websocket()

        task = asyncio.create_task(
            handler.handle_client_connection(
                websocket,
                RegisteredClientInfo(client_id=client_id, cidrs=["10.0.0.4/32"]),
            )
        )
        await _wait_for_registration(handler, client_id, websocket)

        update = ClientUpdateMessage(
            server_id=uuid.uuid4(),
            updates=[
                ClientUpdateInfo(
                    client_id=client_id, action="force_remove", cidrs=["10.0.0.4/32"]
                )
            ],
        )
        await handler.handle_peer_client_update(update)

        assert handler.get_connection_manager("10.0.0.4") is None
        assert client_id not in handler.client_registry
        # The owning server announces the ordinary remove to its peers.
        handler._enqueue_client_update.assert_called_with(
            "remove", client_id, ["10.0.0.4/32"], []
        )
        websocket.close.assert_awaited()

        # The closed socket's receive loop unwinds; its cleanup must be a
        # no-op (the release already dropped the registration and advanced
        # the generation).
        websocket.disconnect()
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(task, timeout=0.5)

    @pytest.mark.asyncio
    async def test_force_release_without_local_registration_broadcasts_force_remove(
        self,
    ):
        handler = _handler()
        handler._enqueue_client_update = MagicMock()

        released = await handler.force_release(uuid.uuid4())

        assert released is False
        handler._enqueue_client_update.assert_called_once()
        action = handler._enqueue_client_update.call_args.args[0]
        assert action == "force_remove"

    @pytest.mark.asyncio
    async def test_force_release_with_only_a_mirror_still_broadcasts_force_remove(
        self,
    ):
        # A non-owning replica holds the peer's mirrored registry entry but no
        # tunnel websocket. The mirror release succeeds locally, yet only a
        # force_remove makes the owning replica close the live socket.
        handler = _handler()
        handler._enqueue_client_update = MagicMock()
        client_id = uuid.uuid4()
        handler.client_registry[client_id] = RegisteredClientInfo(
            client_id=client_id,
            cidrs=["10.0.0.9/32"],
            unix_sockets=[],
            server_id=uuid.uuid4(),
        )
        handler._cidr_registry.insert("10.0.0.9/32", client_id)

        released = await handler.force_release(client_id)

        assert released is True
        actions = [
            call.args[0] for call in handler._enqueue_client_update.call_args_list
        ]
        assert actions == ["remove", "force_remove"]

    @pytest.mark.asyncio
    async def test_force_remove_drops_mirrored_registry_entry(self):
        handler = _handler()
        client_id = uuid.uuid4()
        handler.client_registry[client_id] = RegisteredClientInfo(
            client_id=client_id,
            cidrs=["10.0.0.5/32"],
            unix_sockets=[],
            server_id=uuid.uuid4(),
        )
        handler._cidr_registry.insert("10.0.0.5/32", client_id)

        update = ClientUpdateMessage(
            server_id=uuid.uuid4(),
            updates=[
                ClientUpdateInfo(
                    client_id=client_id, action="force_remove", cidrs=["10.0.0.5/32"]
                )
            ],
        )
        await handler.handle_peer_client_update(update)

        assert client_id not in handler.client_registry
        assert handler.get_connection_manager("10.0.0.5") is None

    @pytest.mark.asyncio
    async def test_force_remove_update_round_trips_through_the_protocol(self):
        client_id = uuid.uuid4()
        msg = ClientUpdateMessage(
            server_id=uuid.uuid4(),
            updates=[
                ClientUpdateInfo(
                    client_id=client_id, action="force_remove", cidrs=["10.0.0.6/32"]
                )
            ],
        )
        parsed = parse_message(pack_message(msg))
        assert parsed.updates[0].action == "force_remove"
        assert parsed.updates[0].client_id == client_id


class TestPeerUpdateOrdering:
    @pytest.mark.asyncio
    async def test_peer_updates_are_chained_remove_before_add(self):
        # A reconnect enqueues its add while the previous socket's remove is
        # still being sent. Sends are chained per client, so peers observe
        # remove before add even though the reconnection itself registered
        # without waiting on the in-flight send.
        handler = _handler()
        first_send = asyncio.Event()
        release_first = asyncio.Event()
        sent = []

        real_send = MessageServerHandler._send_client_update_to_peers

        async def slow_first_send(self, msg_data):
            if not first_send.is_set():
                first_send.set()
                await release_first.wait()
            sent.append(msg_data)
            await real_send(self, msg_data)

        handler._send_client_update_to_peers = slow_first_send.__get__(handler)
        client_id = uuid.uuid4()

        old_ws = _disconnecting_websocket()
        old_task = asyncio.create_task(
            handler.handle_client_connection(
                old_ws,
                RegisteredClientInfo(client_id=client_id, cidrs=["10.0.0.7/32"]),
            )
        )
        await _wait_for_registration(handler, client_id, old_ws)
        await first_send.wait()  # the add send is now wedged

        old_ws.disconnect()
        await old_task

        new_ws = _disconnecting_websocket()
        new_task = asyncio.create_task(
            handler.handle_client_connection(
                new_ws,
                RegisteredClientInfo(client_id=client_id, cidrs=["10.0.0.7/32"]),
            )
        )
        await _wait_for_registration(handler, client_id, new_ws)

        # The reconnection registered while the first send was still wedged,
        # and its update is queued behind it.
        assert sent == []
        release_first.set()
        for _ in range(200):
            if len(sent) >= 3:
                break
            await asyncio.sleep(0.01)

        # add (from registration), then remove, then add — peers observe the
        # client's updates strictly in the order the server produced them.
        actions = [parse_message(data).updates[0].action for data in sent]
        assert actions == ["add", "remove", "add"]

        new_ws.disconnect()
        await asyncio.wait_for(new_task, timeout=5)


class TestReconnectReplacesClaimSet:
    @pytest.mark.asyncio
    async def test_reconnect_with_changed_cidr_drops_the_old_route(self):
        # A worker that re-registers with a different address must not keep
        # its previous CIDR claimed: the old address could then be routed to
        # the new tunnel, and another worker claiming it would be rejected
        # as a foreign claim.
        handler = _handler()
        handler._enqueue_client_update = MagicMock()
        client_id = uuid.uuid4()

        old_ws = _scripted_websocket([])
        old_task = asyncio.create_task(
            handler.handle_client_connection(
                old_ws,
                RegisteredClientInfo(client_id=client_id, cidrs=["10.0.1.1/32"]),
            )
        )
        await _wait_for_registration(handler, client_id, old_ws)

        new_ws = _disconnecting_websocket()
        new_task = asyncio.create_task(
            handler.handle_client_connection(
                new_ws,
                RegisteredClientInfo(client_id=client_id, cidrs=["10.0.1.2/32"]),
            )
        )
        await _wait_for_registration(handler, client_id, new_ws)

        assert handler.get_connection_manager("10.0.1.1") is None
        assert handler.get_connection_manager("10.0.1.2").websocket is new_ws

        new_ws.disconnect()
        await asyncio.wait_for(new_task, timeout=5)
        old_ws.closed = True
        await asyncio.wait_for(old_task, timeout=5)


class TestConcurrentClaims:
    @pytest.mark.asyncio
    async def test_two_clients_claiming_one_cidr_concurrently_one_wins(self):
        # The ownership check and the claim write must be one atomic critical
        # section across clients: if both registrations could observe "no
        # owner", the later write would silently steal the other's route.
        handler = _handler()
        handler._enqueue_client_update = MagicMock()
        cidr = ["10.0.2.7/32"]

        ws_a = _disconnecting_websocket()
        ws_b = _disconnecting_websocket()
        tasks = [
            asyncio.create_task(
                handler.handle_client_connection(
                    ws,
                    RegisteredClientInfo(client_id=client_id, cidrs=cidr),
                )
            )
            for ws, client_id in (
                (ws_a, uuid.uuid4()),
                (ws_b, uuid.uuid4()),
            )
        ]

        # Let both registrations run to completion; exactly one registered.
        await asyncio.sleep(0.1)
        registered = [
            client_id
            for client_id in handler.client_registry
            if cidr[0] in handler.client_registry[client_id].cidrs
        ]
        rejected = [ws for ws in (ws_a, ws_b) if ws.close.await_count > 0]
        assert len(registered) == 1
        assert len(rejected) == 1

        for ws in (ws_a, ws_b):
            ws.disconnect()
        for task in tasks:
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(task, timeout=2)


class TestClientHeartbeatSender:
    @pytest.mark.asyncio
    async def test_failed_heartbeat_send_closes_the_tunnel(self, monkeypatch):
        # The server reaps a tunnel whose heartbeat has gone silent. If a
        # heartbeat send fails, the client must close the connection so its
        # retry loop reconnects immediately, rather than staying open in
        # one-way silence until the server's reaper fires.
        from gpustack.websocket_proxy import message_client

        monkeypatch.setattr(message_client, "HEARTBEAT_INTERVAL", 0.01)
        websocket = MagicMock()
        websocket.send = AsyncMock(side_effect=RuntimeError("write buffer full"))
        websocket.close = AsyncMock()
        client = message_client.MessageClient(
            server_endpoint="http://127.0.0.1:1", client_id=uuid.uuid4()
        )

        task = asyncio.create_task(client._send_heartbeats(websocket))
        await asyncio.wait_for(task, timeout=5)

        websocket.send.assert_awaited()
        websocket.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_healthy_heartbeat_sender_keeps_sending(self, monkeypatch):
        from gpustack.websocket_proxy import message_client

        monkeypatch.setattr(message_client, "HEARTBEAT_INTERVAL", 0.01)
        websocket = MagicMock()
        websocket.send = AsyncMock()
        websocket.close = AsyncMock()
        client = message_client.MessageClient(
            server_endpoint="http://127.0.0.1:1", client_id=uuid.uuid4()
        )

        task = asyncio.create_task(client._send_heartbeats(websocket))
        for _ in range(100):
            if websocket.send.await_count >= 3:
                break
            await asyncio.sleep(0.01)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

        assert websocket.send.await_count >= 3
        websocket.close.assert_not_awaited()


class TestRegistrationCancelled:
    @pytest.mark.asyncio
    async def test_cancellation_during_registration_releases_claims(self):
        # The claims are written before the first await after them (the
        # generation bump). A task cancelled at that await must still release
        # the registration via the shielded fallback, or the CIDR stays
        # claimed until a server restart.
        handler = _handler()
        handler._enqueue_client_update = MagicMock()

        async def cancel_at_generation(client_id):
            raise asyncio.CancelledError()

        handler._get_next_generation = cancel_at_generation
        client_id = uuid.uuid4()

        task = asyncio.create_task(
            handler.handle_client_connection(
                _disconnecting_websocket(),
                RegisteredClientInfo(client_id=client_id, cidrs=["10.0.3.1/32"]),
            )
        )
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=5)

        # The shielded fallback releases asynchronously after cancellation.
        for _ in range(100):
            if handler.get_connection_manager("10.0.3.1") is None:
                break
            await asyncio.sleep(0.01)
        assert handler.get_connection_manager("10.0.3.1") is None
        assert client_id not in handler.client_registry


class TestRejectionLogs:
    @pytest.mark.asyncio
    async def test_foreign_claim_rejection_names_the_owner(self, caplog):
        handler = _handler()
        owner = uuid.uuid4()
        handler.client_registry[owner] = RegisteredClientInfo(
            client_id=owner, cidrs=["10.0.0.8/32"]
        )
        handler._cidr_registry.insert("10.0.0.8/32", owner)
        websocket = _websocket()

        with caplog.at_level(logging.WARNING):
            await handler.handle_client_connection(
                websocket,
                RegisteredClientInfo(client_id=uuid.uuid4(), cidrs=["10.0.0.8/32"]),
            )

        assert any(
            "owned by another client" in record.message and str(owner) in record.message
            for record in caplog.records
        )
        websocket.close.assert_awaited_once()
