#!/usr/bin/env python3
"""
Message Server - Handles WebSocket connections and message protocol
"""

import ipaddress
import uuid
import asyncio
import logging
from typing import Optional, Dict, List, Callable, Coroutine, Any, TypeAlias

from .patricia_trie import CIDRRegistry
from fastapi import WebSocket, Depends, APIRouter
from websockets.asyncio.client import connect as ws_connect
from .connection_manager import ConnectionManager, RemoteConnectionManager
from .message import (
    SessionBaseMessage,
    HeartbeatMessage,
    ClientUpdateMessage,
    ClientUpdateInfo,
    RegisteredClientInfo,
    ServerInfo,
    ServerPeer,
    parse_message,
    pack_message,
)
from .authenticator import Authenticator, NoOpAuthenticator
from .constants import default_connect_path

logger = logging.getLogger(__name__)


VERSION = "0.1.0"


# ==================== Type Aliases ====================

ServerInfoGetter: TypeAlias = Callable[[WebSocket], Optional['ServerInfo']]
RegisteredClientInfoGetter: TypeAlias = Callable[
    [WebSocket], Optional['RegisteredClientInfo']
]
WSCallback: TypeAlias = Callable[
    [Optional['ServerInfo'], Optional['RegisteredClientInfo']],
    Coroutine[Any, Any, None],
]


# ==================== Helper Functions ====================


def default_server_info_getter(websocket: WebSocket) -> Optional['ServerInfo']:
    """Extract server info from headers if this is a server connection"""
    return ServerInfo.from_headers(websocket.headers)


def default_client_info_getter(
    websocket: WebSocket,
) -> Optional['RegisteredClientInfo']:
    """Extract client info from headers if this is a client connection"""
    return RegisteredClientInfo.from_headers(websocket.headers)


def ip_in_cidrs(ip: str, cidrs: List[str]) -> bool:
    """Check if an IP address is in any of the given CIDR ranges"""
    try:
        ip_obj = ipaddress.ip_address(ip)
        for cidr in cidrs:
            network = ipaddress.ip_network(cidr, strict=False)
            if ip_obj in network:
                return True
    except ValueError:
        return False
    return False


# ==================== Dataclasses ====================


class MessageServerHandler:
    """Simple server that handles message protocol and connect-based proxy"""

    _server_info: ServerInfo

    def __init__(
        self,
        listen_address: str,
        listen_port: int,
        proxy_port: int,
        server_id: Optional[uuid.UUID] = None,
        client_info_getter: RegisteredClientInfoGetter = default_client_info_getter,
        server_info_getter: ServerInfoGetter = default_server_info_getter,
        authenticator: Authenticator = None,
        callback_on_connect: Optional[WSCallback] = None,
        callback_on_disconnect: Optional[WSCallback] = None,
    ):
        self._server_info = ServerInfo(
            server_id=server_id or uuid.uuid4(),
            listen_address=listen_address,
            listen_port=listen_port,
            proxy_port=proxy_port,
        )
        self._callback_on_connect = callback_on_connect
        self._callback_on_disconnect = callback_on_disconnect

        # Info extractors - can be overridden for customization
        self._client_info_getter = client_info_getter
        self._server_info_getter = server_info_getter
        self._authenticator = authenticator or NoOpAuthenticator()

        # Client management
        self.client_registry: Dict[uuid.UUID, RegisteredClientInfo] = (
            {}
        )  # client_id -> client_info
        self.connection_managers: Dict[uuid.UUID, ConnectionManager] = {}
        self._cidr_registry = CIDRRegistry()

        # Generation tracking for disconnect callbacks
        self._client_generations: Dict[uuid.UUID, int] = {}  # client_id -> generation
        self._generation_lock = asyncio.Lock()

        # Server federation
        self.peers: Dict[uuid.UUID, ServerPeer] = {}  # Outgoing: servers I connected to
        self.serving_peers: Dict[uuid.UUID, ServerPeer] = (
            {}
        )  # Incoming: servers that connected to me
        self.peer_tasks: Dict[uuid.UUID, asyncio.Task] = {}  # server_id -> task

    async def _get_next_generation(self, client_id: uuid.UUID) -> int:
        """Get the next generation number for a client (thread-safe)."""
        async with self._generation_lock:
            gen = self._client_generations.get(client_id, 0) + 1
            self._client_generations[client_id] = gen
            return gen

    async def _safe_callback(
        self,
        callback: WSCallback,
        server_info: Optional[ServerInfo],
        client_info: Optional[RegisteredClientInfo],
    ) -> None:
        """Execute callback with error handling, does not raise exceptions."""
        try:
            await callback(server_info, client_info)
        except Exception as e:
            logger.error(f"[Server] callback_on_connect error: {e}")

    async def _safe_disconnect_callback(
        self,
        callback: WSCallback,
        client_info: Optional[RegisteredClientInfo],
        generation: int,
    ) -> None:
        """Execute disconnect callback with error handling and stale callback filtering."""
        if client_info is None:
            return
        async with self._generation_lock:
            current_gen = self._client_generations.get(client_info.client_id, 0)
            if generation < current_gen:
                logger.debug(
                    f"[Server] Stale disconnect callback ignored: "
                    f"client={client_info.client_id}, callback_gen={generation}, "
                    f"current_gen={current_gen}"
                )
                return
        try:
            await callback(None, client_info)
        except Exception as e:
            logger.error(f"[Server] callback_on_disconnect error: {e}")

    def _find_peer_by_server_id(self, server_id: uuid.UUID) -> Optional[ServerPeer]:
        """Find a peer by server_id, checking both outgoing and incoming peers"""
        peer = self.peers.get(server_id)
        if peer:
            return peer
        return self.serving_peers.get(server_id)

    def get_connection_manager_by_ip_in_cidr(
        self, target_ip: str
    ) -> Optional[ConnectionManager]:
        """Find a ConnectionManager by matching IP against registered CIDRs using Patricia Trie.

        Returns local ConnectionManager for local clients, or RemoteConnectionManager
        for peer clients.
        """
        client_id = self._cidr_registry.find_best_match(target_ip)
        if not client_id:
            return None

        client_info = self.client_registry.get(client_id)
        if not client_info:
            return None

        # Check if this is a local client (registered on this server)
        if client_info.server_id == self._server_info.server_id:
            return self.connection_managers.get(client_id)

        # Peer client - find the peer and return RemoteConnectionManager
        peer = self._find_peer_by_server_id(client_info.server_id)
        if peer and peer.listen_address and peer.proxy_port:
            return RemoteConnectionManager(peer.listen_address, peer.proxy_port)

        return None

    def get_connection_manager(self, target_ip: str) -> Optional[ConnectionManager]:
        """Get connection manager for target IP (local or remote)."""
        return self.get_connection_manager_by_ip_in_cidr(target_ip)

    async def add_peer(self, address: str, port: int) -> Optional[uuid.UUID]:
        """Add a peer server and connect to it. Returns the peer_id when connected."""
        # Create a future that will be resolved when the peer connects
        future = asyncio.Future()

        asyncio.create_task(self.connect_to_peer(address, port, future))

        # Wait for the peer to connect (with timeout)
        try:
            peer_id = await asyncio.wait_for(future, timeout=10.0)
            return peer_id
        except asyncio.TimeoutError:
            logger.debug(
                f"[Server] Timeout waiting for peer to connect: {address}:{port}"
            )
            return None

    async def remove_peer(self, peer_id: uuid.UUID):
        """Remove a peer server by UUID"""
        logger.debug(f"[Server] Attempting to remove peer by UUID: {peer_id}")
        return await self._remove_peer_impl(peer_id)

    async def remove_peer_by_address(self, address: str):
        """Remove a peer server by address (host:port)"""
        logger.debug(f"[Server] Attempting to remove peer by address: {address}")
        target_peer_id = self._get_peer_id_by_address(address)
        if not target_peer_id:
            return False
        return await self._remove_peer_impl(target_peer_id)

    def _get_peer_id_by_address(self, address: str) -> Optional[uuid.UUID]:
        """Helper to find peer_id by address (host:port)"""
        for peer_id, peer in self.peers.items():
            peer_addr = f"{peer.listen_address}:{peer.listen_port}"
            if peer_addr == address:
                return peer_id
        for peer_id, peer in self.serving_peers.items():
            peer_addr = f"{peer.listen_address}:{peer.listen_port}"
            if peer_addr == address:
                return peer_id
        logger.debug(f"[Server] No peer found with address: {address}")
        return None

    async def _remove_peer_impl(self, peer_id: uuid.UUID) -> bool:
        """Internal implementation for removing a peer (checks both peers and serving_peers)"""
        # Check outgoing peers first
        peer = self.peers.pop(peer_id, None)
        if not peer:
            # Check incoming serving_peers
            peer = self.serving_peers.pop(peer_id, None)

        if peer:
            logger.debug(f"[Server] Found peer to remove: {peer.server_id}")
            if peer.websocket:
                await peer.websocket.close()
                logger.debug(f"[Server] Closed websocket for peer: {peer_id}")
        else:
            logger.debug(f"[Server] Peer not found: {peer_id}")
            return False

        task = self.peer_tasks.pop(peer_id, None)
        if task:
            task.cancel()
            logger.debug(f"[Server] Cancelled task for peer: {peer_id}")

        logger.debug(f"[Server] Removed peer: {peer_id}")
        return True

    async def connect_to_peer(self, host: str, port: int, future: asyncio.Future):
        """Connect to a peer server"""
        peer_key = f"{host}:{port}"
        try:
            ws_uri = f"ws://{host}:{port}{default_connect_path}"
            logger.debug(f"[Server] Connecting to peer: {ws_uri}")

            # Connect with server info in headers (header-based registration)
            headers = {'x-server-id': str(self._server_info.server_id)}
            self._authenticator.inject_headers(headers)
            websocket = await ws_connect(ws_uri, additional_headers=headers)

            # Get peer info from response headers
            peer_info = ServerInfo.from_headers(dict(websocket.response.headers))
            if not peer_info:
                logger.debug("[Server] Peer did not provide valid registration headers")
                await websocket.close()
                if not future.done():
                    future.set_result(None)
                return

            peer_server_id = peer_info.server_id
            peer = ServerPeer(
                server_id=peer_info.server_id,
                listen_address=peer_info.listen_address,
                listen_port=peer_info.listen_port,
                proxy_port=peer_info.proxy_port,
                websocket=websocket,
                connected=True,
            )
            self.peers[peer_server_id] = peer
            logger.debug(f"[Server] Registered with peer: {peer_server_id}")

            # Resolve the future to notify add_peer that connection is complete
            if not future.done():
                future.set_result(peer_server_id)

            # Start handling messages from peer
            task = asyncio.create_task(self.handle_peer(websocket, peer_server_id))
            self.peer_tasks[peer_server_id] = task

        except Exception as e:
            # Connection failed - peer may not be running or rejected connection
            logger.debug(f"[Server] Failed to connect to peer {peer_key}: {e}")
            if not future.done():
                future.set_exception(e)

    async def handle_peer(self, websocket, peer_server_id: uuid.UUID):
        """Handle messages from a peer server"""
        try:
            # Check if this is a Starlette WebSocket or websockets WebSocket
            if hasattr(websocket, 'receive'):
                # Starlette WebSocket
                while True:
                    message = await websocket.receive()
                    if message.get("type") == "websocket.disconnect":
                        break
                    raw_data = message.get("bytes") or message.get("text", "").encode()
                    msg = parse_message(raw_data)
                    logger.trace(f"[Server] Received from peer: {msg.get_type()}")

                    if isinstance(msg, ClientUpdateMessage):
                        await self.handle_peer_client_update(msg)
            else:
                # websockets library WebSocket
                while True:
                    raw_data = await websocket.recv()
                    msg = parse_message(raw_data)
                    logger.trace(f"[Server] Received from peer: {msg.get_type()}")

                    if isinstance(msg, ClientUpdateMessage):
                        await self.handle_peer_client_update(msg)

        except Exception as e:
            logger.debug(f"[Server] Peer connection error: {e}")
        finally:
            # Clean up peer on disconnect (check both peers and serving_peers)
            if peer_server_id in self.peers:
                del self.peers[peer_server_id]
            if peer_server_id in self.serving_peers:
                del self.serving_peers[peer_server_id]
            if peer_server_id in self.peer_tasks:
                del self.peer_tasks[peer_server_id]

            # Clean up clients registered through this peer
            await self._remove_clients_from_peer(peer_server_id)

            logger.debug(f"[Server] Peer disconnected: {peer_server_id}")

    async def handle_peer_client_update(self, msg: ClientUpdateMessage):
        """Handle client update from peer"""
        for update in msg.updates:
            if update.action == "add":
                # Add client's CIDRs to local registry
                client_info = RegisteredClientInfo(
                    client_id=update.client_id,
                    cidrs=update.cidrs,
                    unix_sockets=update.unix_sockets,
                    server_id=msg.server_id,
                )
                self.client_registry[update.client_id] = client_info
                # Index CIDRs for efficient lookup
                for cidr in update.cidrs:
                    self._cidr_registry.insert(cidr, update.client_id)
                logger.debug(f"[Server] Added client from peer: {update.client_id}")
            elif update.action == "remove":
                # Remove client from local registry
                if update.client_id in self.client_registry:
                    self._cidr_registry.remove_client(update.client_id)
                    del self.client_registry[update.client_id]
                    logger.debug(
                        f"[Server] Removed client from peer: {update.client_id}"
                    )

    async def _remove_clients_from_peer(self, peer_server_id: uuid.UUID):
        """Remove all clients that were registered through a peer server."""
        clients_to_remove = [
            client_id
            for client_id, client_info in self.client_registry.items()
            if client_info.server_id == peer_server_id
        ]

        for client_id in clients_to_remove:
            self._cidr_registry.remove_client(client_id)
            del self.client_registry[client_id]
            logger.debug(
                f"[Server] Removed client {client_id} from disconnected peer {peer_server_id}"
            )

    async def send_client_update_to_peer(self, websocket, action: str):
        """Send client updates to a peer"""
        updates = []
        for client_id, client_info in self.client_registry.items():
            # Only send clients owned by this server
            if client_info.server_id == self._server_info.server_id:
                updates.append(
                    ClientUpdateInfo(
                        client_id=client_id,
                        action=action,
                        cidrs=client_info.cidrs,
                        unix_sockets=client_info.unix_sockets,
                    )
                )

        if updates:
            msg = ClientUpdateMessage(
                server_id=self._server_info.server_id, updates=updates
            )
            msg_data = pack_message(msg)
            if hasattr(websocket, 'send_bytes'):
                await websocket.send_bytes(msg_data)
            else:
                await websocket.send(msg_data)

    async def broadcast_client_update(
        self,
        action: str,
        client_id: uuid.UUID,
        cidrs: List[str],
        unix_sockets: List[str],
    ):
        """Broadcast client update to all peers (both outgoing and incoming)"""
        update = ClientUpdateInfo(
            client_id=client_id,
            action=action,
            cidrs=cidrs,
            unix_sockets=unix_sockets,
        )
        msg = ClientUpdateMessage(
            server_id=self._server_info.server_id, updates=[update]
        )

        # Broadcast to outgoing peers
        for peer_id, peer in self.peers.items():
            if peer.connected and peer.websocket:
                try:
                    msg_data = pack_message(msg)
                    if hasattr(peer.websocket, 'send_bytes'):
                        await peer.websocket.send_bytes(msg_data)
                    else:
                        await peer.websocket.send(msg_data)
                except Exception as e:
                    logger.debug(
                        f"[Server] Error sending update to peer {peer_id}: {e}"
                    )

        # Broadcast to incoming serving_peers
        for peer_id, peer in self.serving_peers.items():
            if peer.connected and peer.websocket:
                try:
                    msg_data = pack_message(msg)
                    if hasattr(peer.websocket, 'send_bytes'):
                        await peer.websocket.send_bytes(msg_data)
                    else:
                        await peer.websocket.send(msg_data)
                except Exception as e:
                    logger.debug(
                        f"[Server] Error sending update to serving_peer {peer_id}: {e}"
                    )

    async def handle_client_connection(
        self, websocket: WebSocket, client_info: RegisteredClientInfo
    ):
        """Handle a client WebSocket connection"""
        client_id = client_info.client_id
        cidr_list = client_info.cidrs
        socket_list = client_info.unix_sockets

        connection_manager = ConnectionManager(websocket)
        self.connection_managers[client_id] = connection_manager

        # Set server_id so send_client_update_to_peer can filter correctly
        client_info.server_id = self._server_info.server_id
        self.client_registry[client_id] = client_info

        # Index CIDRs for efficient lookup
        for cidr in cidr_list:
            self._cidr_registry.insert(cidr, client_id)

        logger.debug(
            f"[Server] Client registered via WS: {client_id}, CIDRs: {cidr_list}"
        )

        # Broadcast new client to peers
        await self.broadcast_client_update("add", client_id, cidr_list, socket_list)

        # Get generation for this connection (used to filter stale disconnect callbacks)
        generation = await self._get_next_generation(client_id)

        await websocket.accept()
        if self._callback_on_connect:
            await self._safe_callback(self._callback_on_connect, None, client_info)
        await self.handle_client(websocket, client_id, generation)

    async def handle_server_federation(
        self,
        websocket: WebSocket,
        server_info: ServerInfo,
    ):
        our_server_id = self._server_info.server_id
        """Handle incoming server-to-server federation connection"""
        logger.debug(
            f"[Server] handle_server_federation: incoming={server_info.server_id}, self={our_server_id}"
        )
        # Prevent adding self as peer
        if server_info.server_id == our_server_id:
            logger.debug("[Server] Ignoring self-connection attempt")
            await websocket.close()
            return

        # Accept with our server info in response headers
        await websocket.accept(headers=self._server_info.to_bytes_headers())

        # Add to serving_peers (incoming connections)
        peer = ServerPeer(
            server_id=server_info.server_id,
            listen_address=server_info.listen_address,
            listen_port=server_info.listen_port,
            proxy_port=server_info.proxy_port,
            websocket=websocket,
            connected=True,
        )
        self.serving_peers[server_info.server_id] = peer

        logger.debug(f"[Server] Serving peer connected: {server_info.server_id}")

        if self._callback_on_connect:
            await self._safe_callback(self._callback_on_connect, server_info, None)

        # Send existing clients to new peer
        await self.send_client_update_to_peer(websocket, "add")

        await self.handle_peer(websocket, server_info.server_id)

    async def handle_client(
        self, websocket: WebSocket, client_id: uuid.UUID, generation: int
    ):
        """Handle a client connection"""
        try:
            while True:
                try:
                    message = await websocket.receive()
                    if message.get("type") == "websocket.disconnect":
                        break

                    if "text" in message:
                        raw_data = (
                            message["text"].encode()
                            if isinstance(message["text"], str)
                            else message["text"]
                        )
                    elif "bytes" in message:
                        raw_data = message["bytes"]
                    else:
                        continue

                    msg = parse_message(raw_data)
                    logger.trace(f"[Server] Received: {msg.get_type()}")

                    if isinstance(msg, SessionBaseMessage):
                        if client_id in self.connection_managers:
                            await self.connection_managers[client_id].dispatch(msg)
                    elif isinstance(msg, HeartbeatMessage):
                        response = HeartbeatMessage(timestamp=msg.timestamp)
                        await websocket.send_bytes(pack_message(response))

                except asyncio.CancelledError:
                    # Client is being closed, exit gracefully
                    break
                except Exception as e:
                    logger.debug(f"[Server] Error processing message: {e}")

        except Exception as e:
            logger.debug(f"[Server] Client error: {e}")
        finally:
            # Get client info before removing
            client_info = self.client_registry.get(client_id)
            cidr_list = client_info.cidrs if client_info else []
            socket_list = client_info.unix_sockets if client_info else []

            if client_id and client_id in self.client_registry:
                self._cidr_registry.remove_client(client_id)
                del self.client_registry[client_id]
            if client_id and client_id in self.connection_managers:
                del self.connection_managers[client_id]

            # Broadcast client disconnection to peers
            await self.broadcast_client_update(
                "remove", client_id, cidr_list, socket_list
            )

            # Call disconnect callback (filtered by generation to avoid stale callbacks)
            if self._callback_on_disconnect:
                await self._safe_disconnect_callback(
                    self._callback_on_disconnect, client_info, generation
                )

            logger.debug(f"[Server] Client disconnected: {client_id}")


def handler_getter(websocket: WebSocket) -> MessageServerHandler:
    return getattr(websocket.app.state, "message_server_handler", None)


def authenticator_getter(websocket: WebSocket) -> Authenticator:
    authenticator: Authenticator = getattr(
        websocket.app.state, "websocket_authenticator", NoOpAuthenticator()
    )
    return authenticator


router = APIRouter()


@router.websocket(default_connect_path)
async def websocket_endpoint(
    websocket: WebSocket,
    handler: MessageServerHandler = Depends(handler_getter),
    authenticator: Authenticator = Depends(authenticator_getter),
):
    """WebSocket endpoint - handles both client and server connections"""
    try:
        if not await authenticator.authenticate(websocket):
            logger.debug("[Server] Authentication failed for connection")
            await websocket.close(code=4001, reason="Authentication failed")
            return
    except Exception as e:
        logger.debug(f"Server failed with: {e}")
        await websocket.close(code=1008, reason="Server Error")
        return

    # Check if this is a server federation connection
    server_info = handler._server_info_getter(websocket)
    if server_info:
        logger.debug(
            f"[Server] Detected server federation connection: {server_info.server_id}"
        )
        await handler.handle_server_federation(websocket, server_info)
        return

    client_info = handler._client_info_getter(websocket)
    if client_info:
        logger.debug(f"[Server] Detected client connection: {client_info.client_id}")
        await handler.handle_client_connection(websocket, client_info)
        return

    logger.debug(
        "[Server] No valid server or client info found in headers, rejecting connection"
    )
    await websocket.close(code=1008, reason="not a valid client or server connection")
