"""
Test suite for server-to-server federation functionality.

Tests:
- Server connection via WebSocket handshake with header-based registration
- Client registration broadcast to peers
- Client disconnection broadcast to peers
- Peer removal
- Peer authentication with HMAC
"""

import asyncio
import pytest
import uuid
import uvicorn
from fastapi import FastAPI

from gpustack.websocket_proxy.message_server import MessageServerHandler, router
from gpustack.websocket_proxy.message_client import MessageClient
from gpustack.websocket_proxy.authenticator import create_authenticator


def get_free_port(host: str = "127.0.0.1") -> int:
    """Get a free port on the host"""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


async def start_server(message_handler: MessageServerHandler, host: str, port: int):
    """Start WebSocket server using uvicorn"""
    app = FastAPI()
    app.state.message_server_handler = message_handler
    app.include_router(router)

    # Get actual port if port=0 (dynamic allocation)
    actual_port = get_free_port(host) if port == 0 else port

    config = uvicorn.Config(app, host=host, port=actual_port, log_level="error")
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve())

    await asyncio.sleep(0.5)  # Wait for server to start

    return server, server_task, actual_port


class TestServerFederation:
    """Test server-to-server federation"""

    @pytest.mark.asyncio
    async def test_server_connection_via_headers(self):
        """Test that two servers can connect using header-based registration"""
        server1_id = uuid.uuid4()
        server2_id = uuid.uuid4()

        # Create two server handlers
        handler1 = MessageServerHandler(
            server_id=server1_id,
            listen_address="127.0.0.1",
            listen_port=0,
            proxy_port=0,
        )
        handler2 = MessageServerHandler(
            server_id=server2_id,
            listen_address="127.0.0.1",
            listen_port=0,
            proxy_port=0,
        )

        # Start both servers
        _, task1, port1 = await start_server(handler1, "127.0.0.1", 0)
        _, task2, port2 = await start_server(handler2, "127.0.0.1", 0)

        try:
            # Server1 connects to Server2 as a peer
            peer_id = await handler1.add_peer("127.0.0.1", port2)
            assert peer_id == server2_id

            # Server2 should have server1 in serving_peers
            assert server1_id in handler2.serving_peers
            assert handler2.serving_peers[server1_id].server_id == server1_id

            # Server1 should have server2 in peers
            assert server2_id in handler1.peers
            assert handler1.peers[server2_id].server_id == server2_id

        finally:
            task1.cancel()
            task2.cancel()
            try:
                await task1
            except asyncio.CancelledError:
                pass
            try:
                await task2
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_client_registration_broadcast(self):
        """Test that client registration is broadcast to peers"""
        server1_id = uuid.uuid4()
        server2_id = uuid.uuid4()

        handler1 = MessageServerHandler(
            server_id=server1_id,
            listen_address="127.0.0.1",
            listen_port=0,
            proxy_port=0,
        )
        handler2 = MessageServerHandler(
            server_id=server2_id,
            listen_address="127.0.0.1",
            listen_port=0,
            proxy_port=0,
        )

        _, task1, port1 = await start_server(handler1, "127.0.0.1", 0)
        _, task2, port2 = await start_server(handler2, "127.0.0.1", 0)

        try:
            # Connect servers as peers
            await handler1.add_peer("127.0.0.1", port2)
            await asyncio.sleep(0.5)  # Wait for connection to establish

            # Now connect a client to server1 (use valid UUID for client_id)
            client_uuid = uuid.uuid4()
            client = MessageClient(
                server_endpoint=f"ws://127.0.0.1:{port1}",
                client_id=client_uuid,
                cidrs=["192.168.1.0/24"],
            )
            client_task = asyncio.create_task(client.run())

            await asyncio.sleep(0.5)  # Wait for client registration

            # Server1 should have the client (key is UUID)
            assert client_uuid in handler1.client_registry
            assert handler1.client_registry[client_uuid].cidrs == ["192.168.1.0/24"]

            # Server2 should receive the client update from server1
            # (it takes a moment for the broadcast to propagate)
            await asyncio.sleep(0.5)

            # The client should be registered in server2's registry via peer update
            assert client_uuid in handler2.client_registry
            assert handler2.client_registry[client_uuid].cidrs == ["192.168.1.0/24"]
            assert handler2.client_registry[client_uuid].server_id == server1_id

            client_task.cancel()
            try:
                await client_task
            except asyncio.CancelledError:
                pass

        finally:
            task1.cancel()
            task2.cancel()
            try:
                await task1
            except asyncio.CancelledError:
                pass
            try:
                await task2
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_peer_sends_existing_clients_on_connect(self):
        """Test that when ServerA connects to ServerB, ServerA receives ServerB's existing clients"""
        server1_id = uuid.uuid4()
        server2_id = uuid.uuid4()

        handler1 = MessageServerHandler(
            server_id=server1_id,
            listen_address="127.0.0.1",
            listen_port=0,
            proxy_port=0,
        )
        handler2 = MessageServerHandler(
            server_id=server2_id,
            listen_address="127.0.0.1",
            listen_port=0,
            proxy_port=0,
        )

        _, task1, port1 = await start_server(handler1, "127.0.0.1", 0)
        _, task2, port2 = await start_server(handler2, "127.0.0.1", 0)

        try:
            # First, connect a client to Server2 (before Server1 connects)
            client_uuid = uuid.uuid4()
            client = MessageClient(
                server_endpoint=f"ws://127.0.0.1:{port2}",
                client_id=client_uuid,
                cidrs=["10.0.0.0/8"],
            )
            client_task = asyncio.create_task(client.run())

            await asyncio.sleep(0.5)  # Wait for client registration

            # Server2 should have the client (key is UUID)
            assert client_uuid in handler2.client_registry
            assert handler2.client_registry[client_uuid].cidrs == ["10.0.0.0/8"]

            # Now Server1 connects to Server2
            await handler1.add_peer("127.0.0.1", port2)
            await asyncio.sleep(0.5)  # Wait for connection and client list sync

            # Server1 should have received Server2's existing clients
            assert (
                client_uuid in handler1.client_registry
            ), f"Server1 did not receive Server2's client. Registry keys: {list(handler1.client_registry.keys())}"
            assert handler1.client_registry[client_uuid].cidrs == ["10.0.0.0/8"]
            assert handler1.client_registry[client_uuid].server_id == server2_id

            client_task.cancel()
            try:
                await client_task
            except asyncio.CancelledError:
                pass

        finally:
            task1.cancel()
            task2.cancel()
            try:
                await task1
            except asyncio.CancelledError:
                pass
            try:
                await task2
            except asyncio.CancelledError:
                pass
            try:
                await task1
            except asyncio.CancelledError:
                pass
            try:
                await task2
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_remove_peer(self):
        """Test that peer removal works correctly"""
        server1_id = uuid.uuid4()
        server2_id = uuid.uuid4()

        handler1 = MessageServerHandler(
            server_id=server1_id,
            listen_address="127.0.0.1",
            listen_port=0,
            proxy_port=0,
        )
        handler2 = MessageServerHandler(
            server_id=server2_id,
            listen_address="127.0.0.1",
            listen_port=0,
            proxy_port=0,
        )

        _, task1, port1 = await start_server(handler1, "127.0.0.1", 0)
        _, task2, port2 = await start_server(handler2, "127.0.0.1", 0)

        try:
            # Connect servers as peers
            await handler1.add_peer("127.0.0.1", port2)
            await asyncio.sleep(0.5)

            # Verify connection
            assert server2_id in handler1.peers
            assert server1_id in handler2.serving_peers

            # Remove peer by UUID
            result = await handler1.remove_peer(server2_id)
            assert result is True

            # Verify removal
            assert server2_id not in handler1.peers
            # server2 should still have server1 in serving_peers until it detects disconnect
            await asyncio.sleep(0.5)

        finally:
            task1.cancel()
            task2.cancel()
            try:
                await task1
            except asyncio.CancelledError:
                pass
            try:
                await task2
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_bidirectional_peer_connection(self):
        """Test that two servers can connect to each other as peers"""
        server1_id = uuid.uuid4()
        server2_id = uuid.uuid4()

        handler1 = MessageServerHandler(
            server_id=server1_id,
            listen_address="127.0.0.1",
            listen_port=0,
            proxy_port=0,
        )
        handler2 = MessageServerHandler(
            server_id=server2_id,
            listen_address="127.0.0.1",
            listen_port=0,
            proxy_port=0,
        )

        _, task1, port1 = await start_server(handler1, "127.0.0.1", 0)
        _, task2, port2 = await start_server(handler2, "127.0.0.1", 0)

        try:
            # Server1 connects to Server2
            await handler1.add_peer("127.0.0.1", port2)
            await asyncio.sleep(0.3)

            # Server2 connects to Server1
            await handler2.add_peer("127.0.0.1", port1)
            await asyncio.sleep(0.3)

            # Both should have each other as peers (server1 has server2 as outgoing,
            # server2 has server1 as outgoing, but server1 is also in server2's serving_peers)
            assert server2_id in handler1.peers
            assert server1_id in handler2.peers
            assert server1_id in handler2.serving_peers  # server1 connected to server2

        finally:
            task1.cancel()
            task2.cancel()
            try:
                await task1
            except asyncio.CancelledError:
                pass
            try:
                await task2
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_clients_removed_when_peer_disconnects(self):
        """Test that clients synced from a peer are removed when the peer connection is closed.

        Scenario:
        1. Server1 connects to Server2 (server1.add_peer(server2))
        2. Server2 connects a client to itself
        3. Server2 syncs its client to Server1 (stored with server_id=server2_id)
        4. Server1 disconnects from Server2 (server1.remove_peer(server2))
        5. Server1 should clean up clients with server_id=server2_id
        """
        server1_id = uuid.uuid4()
        server2_id = uuid.uuid4()

        handler1 = MessageServerHandler(
            server_id=server1_id,
            listen_address="127.0.0.1",
            listen_port=0,
            proxy_port=0,
        )
        handler2 = MessageServerHandler(
            server_id=server2_id,
            listen_address="127.0.0.1",
            listen_port=0,
            proxy_port=0,
        )

        _, task1, _ = await start_server(handler1, "127.0.0.1", 0)
        _, task2, port2 = await start_server(handler2, "127.0.0.1", 0)

        try:
            # Step 1: Server1 connects to Server2 as peer
            await handler1.add_peer("127.0.0.1", port2)
            await asyncio.sleep(0.5)

            # Step 2: Connect a client to Server2 (not Server1)
            client_uuid = uuid.uuid4()
            client = MessageClient(
                server_endpoint=f"ws://127.0.0.1:{port2}",
                client_id=client_uuid,
                cidrs=["192.168.1.0/24"],
            )
            client_task = asyncio.create_task(client.run())
            await asyncio.sleep(0.5)

            # Verify client is in Server2 (its own registry)
            assert client_uuid in handler2.client_registry

            # Step 3: Server2 syncs its client to Server1 via peer connection
            await asyncio.sleep(0.5)
            assert client_uuid in handler1.client_registry
            assert handler1.client_registry[client_uuid].server_id == server2_id

            # Step 4: Server1 disconnects from Server2
            await handler1.remove_peer(server2_id)
            await asyncio.sleep(0.5)

            # Step 5: Server1 should clean up clients synced from Server2
            assert client_uuid not in handler1.client_registry

            # Server2 should still have its own client (disconnect was on Server1's outgoing connection)
            assert client_uuid in handler2.client_registry

            client_task.cancel()
            try:
                await client_task
            except asyncio.CancelledError:
                pass

        finally:
            task1.cancel()
            task2.cancel()
            try:
                await task1
            except asyncio.CancelledError:
                pass
            try:
                await task2
            except asyncio.CancelledError:
                pass


class TestServerFederationWithAuth:
    """Integration tests for server federation with authentication"""

    @pytest.mark.asyncio
    async def test_peer_connection_with_authenticator(self):
        """Test that two servers can connect when both use the same authenticator"""
        server1_id = uuid.uuid4()
        server2_id = uuid.uuid4()
        secret = "shared-secret"

        handler1 = MessageServerHandler(
            server_id=server1_id,
            listen_address="127.0.0.1",
            listen_port=0,
            proxy_port=0,
            authenticator=create_authenticator(secret),
        )
        handler2 = MessageServerHandler(
            server_id=server2_id,
            listen_address="127.0.0.1",
            listen_port=0,
            proxy_port=0,
            authenticator=create_authenticator(secret),
        )

        _, task1, port1 = await start_server(handler1, "127.0.0.1", 0)
        _, task2, port2 = await start_server(handler2, "127.0.0.1", 0)

        try:
            # Server1 connects to Server2
            peer_id = await handler1.add_peer("127.0.0.1", port2)
            assert peer_id == server2_id

            # Verify connection established
            await asyncio.sleep(0.5)
            assert server2_id in handler1.peers
            assert server1_id in handler2.serving_peers

        finally:
            task1.cancel()
            task2.cancel()
            try:
                await task1
            except asyncio.CancelledError:
                pass
            try:
                await task2
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_peer_connection_fails_with_wrong_secret(self):
        """Test that peer connection fails when secrets don't match"""
        server1_id = uuid.uuid4()
        server2_id = uuid.uuid4()

        handler1 = MessageServerHandler(
            server_id=server1_id,
            listen_address="127.0.0.1",
            listen_port=0,
            proxy_port=0,
            authenticator=create_authenticator("secret-a"),
        )
        handler2 = MessageServerHandler(
            server_id=server2_id,
            listen_address="127.0.0.1",
            listen_port=0,
            proxy_port=0,
            authenticator=create_authenticator("secret-b"),
        )

        _, task1, _ = await start_server(handler1, "127.0.0.1", 0)
        _, task2, port2 = await start_server(handler2, "127.0.0.1", 0)

        try:
            # Server1 tries to connect to Server2 with different secret
            _ = await handler1.add_peer("127.0.0.1", port2)
        except Exception as e:
            # Expect connection to fail due to authentication error
            # Server sends HTTP 403 during WebSocket handshake
            assert "403" in str(e) or "Authentication failed" in str(e)

        finally:
            task1.cancel()
            task2.cancel()
            try:
                await task1
            except asyncio.CancelledError:
                pass
            try:
                await task2
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_peer_connection_with_noop_authenticator(self):
        """Test that peer connection works when no authenticator is set"""
        server1_id = uuid.uuid4()
        server2_id = uuid.uuid4()

        handler1 = MessageServerHandler(
            server_id=server1_id,
            listen_address="127.0.0.1",
            listen_port=0,
            proxy_port=0,
            # No authenticator - uses default NoOpAuthenticator
        )
        handler2 = MessageServerHandler(
            server_id=server2_id,
            listen_address="127.0.0.1",
            listen_port=0,
            proxy_port=0,
            # No authenticator - uses default NoOpAuthenticator
        )

        _, task1, port1 = await start_server(handler1, "127.0.0.1", 0)
        _, task2, port2 = await start_server(handler2, "127.0.0.1", 0)

        try:
            # Server1 connects to Server2
            peer_id = await handler1.add_peer("127.0.0.1", port2)
            assert peer_id == server2_id

            await asyncio.sleep(0.5)
            assert server2_id in handler1.peers
            assert server1_id in handler2.serving_peers

        finally:
            task1.cancel()
            task2.cancel()
            try:
                await task1
            except asyncio.CancelledError:
                pass
            try:
                await task2
            except asyncio.CancelledError:
                pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
