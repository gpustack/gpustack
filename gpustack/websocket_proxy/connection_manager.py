#!/usr/bin/env python3
"""
Connection Managers - Handle tunnel connections lifecycle
"""

import asyncio
import logging
import urllib.parse
import uuid
from typing import Optional, Dict, TYPE_CHECKING, Union, Protocol

if TYPE_CHECKING:
    from websockets.client import ClientConnection
    from websockets.server import ServerConnection
    from starlette.websockets import WebSocket as StarletteWebSocket

from .connection import TunnelConnection, IOConnection, AsyncIOConnection, tunnel
from .message import (
    SessionBaseMessage,
    ConnectRequestMessage,
    ConnectResponseMessage,
    DataMessage,
    DisconnectMessage,
    pack_message,
)

logger = logging.getLogger(__name__)


# ==================== Independent Handlers ====================


class ConnectionManager:
    """Manages all tunnel connections lifecycle (server-side)"""

    def __init__(
        self,
        websocket: Optional[
            Union["ClientConnection", "ServerConnection", "StarletteWebSocket"]
        ] = None,
    ) -> None:
        self.websocket = websocket
        self._connections: Dict[uuid.UUID, TunnelConnection] = {}

    async def _send_to_websocket(self, data: bytes) -> None:
        """Send data to WebSocket, compatible with Starlette and websockets library"""
        if self.websocket is None:
            return
        if hasattr(self.websocket, 'send_bytes'):
            await self.websocket.send_bytes(data)
        else:
            await self.websocket.send(data)

    async def _direct_connect(
        self, session_id: uuid.UUID, target_url: str
    ) -> IOConnection:
        parsed = urllib.parse.urlparse(target_url)
        if parsed.scheme == "unix":
            reader, writer = await asyncio.open_unix_connection(parsed.path)
        else:
            reader, writer = await asyncio.open_connection(parsed.hostname, parsed.port)
        connection = AsyncIOConnection(reader=reader, writer=writer)
        self._connections[session_id] = connection
        return connection

    async def _websocket_connect(
        self, session_id: uuid.UUID, target_url: str
    ) -> TunnelConnection:
        connection = TunnelConnection(session_id, self.websocket)
        self._connections[session_id] = connection

        message = ConnectRequestMessage(session_id=session_id, target_url=target_url)
        await self._send_to_websocket(pack_message(message))
        logger.trace(
            f"[ConnectionManager] Sent CONNECT_REQUEST for {target_url} (session={session_id})"
        )

        try:
            await asyncio.wait_for(connection.connect_result, timeout=30)
        except asyncio.TimeoutError:
            raise TimeoutError(f"Connection to {target_url} timed out")
        except asyncio.CancelledError:
            raise

        return connection

    async def connect(
        self,
        target_url: str,
    ) -> IOConnection:
        """Create new connection: use WebSocket tunnel or direct TCP/Unix

        URL format: tcp://host:port or unix:///path/to/socket
        """
        session_id = uuid.uuid4()
        if self.websocket is None:
            connection = await self._direct_connect(session_id, target_url)
        else:
            connection = await self._websocket_connect(session_id, target_url)
        return connection

    def get_connection(self, session_id: uuid.UUID) -> Optional[TunnelConnection]:
        """Get connection by session_id"""
        return self._connections.get(session_id)

    def pop_connection(self, session_id: uuid.UUID) -> Optional[TunnelConnection]:
        """Remove and return connection by session_id"""
        return self._connections.pop(session_id, None)

    def connections(self) -> Dict[uuid.UUID, TunnelConnection]:
        """Get all connections"""
        return self._connections

    async def dispatch(self, msg: SessionBaseMessage) -> None:
        """Dispatch message to appropriate handler based on message type"""
        connection = self.get_connection(msg.session_id)
        if connection is None and not isinstance(msg, DisconnectMessage):
            logger.error(
                f"[ConnectionManager] WARNING: No connection found for session_id={msg.session_id}, message type={type(msg).__name__}"
            )
            return

        if isinstance(msg, ConnectResponseMessage):
            if msg.success:
                connection.set_connected()
            else:
                connection.connect_error(Exception(f"Connection failed: {msg.error}"))
        elif isinstance(msg, DataMessage):
            await connection.handle_data(msg.data)
        elif isinstance(msg, DisconnectMessage):
            connection = self.pop_connection(msg.session_id)
            if connection:
                await connection.close()


class BaseConnectionManager(Protocol):
    """Abstract base class for connection managers"""

    def connections(self) -> Dict:
        """Get all connections"""
        ...

    async def connect(self, target_url: str) -> IOConnection:
        """Establish connection and return IOConnection

        URL format: tcp://host:port or unix:///path/to/socket
        """
        ...


class ClientConnectionManager:
    """Client-side ConnectionManager, handles CONNECT_REQUEST and forwards data"""

    def __init__(
        self,
        websocket: Union["ClientConnection", "ServerConnection", "StarletteWebSocket"],
    ) -> None:
        self.websocket = websocket
        self._connections: Dict[uuid.UUID, TunnelConnection] = {}

    async def _send_to_websocket(self, data: bytes) -> None:
        """Send data to WebSocket, compatible with Starlette and websockets library"""
        logger.trace(
            f"[ClientConnectionManager] Sending {len(data)} bytes to WebSocket"
        )
        if hasattr(self.websocket, 'send_bytes'):
            await self.websocket.send_bytes(data)
        else:
            await self.websocket.send(data)

    def get_connection(self, session_id: uuid.UUID) -> Optional[TunnelConnection]:
        """Get connection by session_id"""
        return self._connections.get(session_id)

    def pop_connection(self, session_id: uuid.UUID) -> Optional[TunnelConnection]:
        """Remove and return connection by session_id"""
        return self._connections.pop(session_id, None)

    def connections(self) -> Dict[uuid.UUID, TunnelConnection]:
        """Get all connections"""
        return self._connections

    async def dispatch(self, msg: SessionBaseMessage) -> None:
        """Dispatch message to appropriate handler based on message type"""
        connection = self.get_connection(msg.session_id)

        if isinstance(msg, ConnectRequestMessage):
            await self.handle_client_connect_request(msg)
        elif isinstance(msg, DataMessage):
            if connection:
                await connection.handle_data(msg.data)
        elif isinstance(msg, DisconnectMessage):
            connection = self.pop_connection(msg.session_id)
            if connection:
                await connection.close()

    async def handle_client_connect_request(self, msg: ConnectRequestMessage) -> None:
        """Handle CONNECT_REQUEST: establish connection and respond"""
        logger.trace(
            f"[ClientConnectionManager] Handling CONNECT_REQUEST for {msg.target_url} (session_id={msg.session_id})"
        )
        try:
            parsed = urllib.parse.urlparse(msg.target_url)
            if parsed.scheme == "unix":
                reader, writer = await asyncio.wait_for(
                    asyncio.open_unix_connection(parsed.path), timeout=5.0
                )
            else:
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(parsed.hostname, parsed.port),
                    timeout=5.0,
                )
            target_connection = AsyncIOConnection(reader=reader, writer=writer)
            connection = TunnelConnection(
                session_id=msg.session_id,
                websocket=self.websocket,
            )
            connection.set_connected()
            self._connections[msg.session_id] = connection

            response = ConnectResponseMessage(session_id=msg.session_id, success=True)
            await self._send_to_websocket(pack_message(response))
            logger.trace(f"[ClientConnectionManager] Connected to {msg.target_url}")

            async def tunnel_and_close(session_id: uuid.UUID = msg.session_id):
                try:
                    await tunnel(
                        connection,
                        target_connection,
                        name="ClientConnectionManager Tunnel",
                    )
                except Exception as e:
                    logger.error(f"[ClientConnectionManager] Tunnel error: {e}")
                finally:
                    conn = self.pop_connection(session_id)
                    if conn:
                        await conn.close()

            asyncio.create_task(tunnel_and_close())

        except Exception as e:
            logger.error(f"[ClientConnectionManager] Failed to connect: {e}")
            response = ConnectResponseMessage(
                session_id=msg.session_id, success=False, error=str(e)
            )
            await self._send_to_websocket(pack_message(response))


class RemoteConnectionManager:
    """Connection manager that forwards requests to a remote peer's HTTP proxy

    Only supports TCP connections. Unix socket targets will raise an error.
    """

    def __init__(self, peer_address: str, proxy_port: int):
        self.peer_address = peer_address
        self.proxy_port = proxy_port

    async def connect(
        self,
        target_url: str,
    ) -> IOConnection:
        """Forward HTTP request to remote peer's proxy

        URL format: tcp://host:port
        Note: Unix socket URLs are not supported and will raise an error.
        """
        parsed = urllib.parse.urlparse(target_url)

        if parsed.scheme == "unix":
            raise ValueError(
                "RemoteConnectionManager does not support Unix socket connections"
            )

        # TCP connection
        target = f"{parsed.hostname}:{parsed.port}"

        logger.trace(
            f"[RemoteConnectionManager] Forwarding to {self.peer_address}:{self.proxy_port} -> {target}"
        )

        # Connect to the remote proxy
        reader, writer = await asyncio.open_connection(
            self.peer_address, self.proxy_port
        )

        # For HTTP CONNECT method, we need to send CONNECT request to the proxy
        # The proxy will then connect to the target
        connect_request = f"CONNECT {target} HTTP/1.1\r\nHost: {target}\r\n\r\n"
        writer.write(connect_request.encode())
        await writer.drain()

        # Read response from proxy
        response = await reader.read(4096)
        response_str = response.decode('utf-8', errors='ignore')

        # Check if proxy accepted the connection
        if not response_str.startswith("HTTP/1.1 200"):
            logger.error(
                f"[RemoteConnectionManager] Proxy rejected connection: {response_str}"
            )
            writer.close()
            await writer.wait_closed()
            raise Exception(f"Proxy connection failed: {response_str}")

        logger.trace(
            f"[RemoteConnectionManager] Connected to {target} via remote proxy"
        )

        # Create a tunnel connection for this forward
        connection = AsyncIOConnection(reader=reader, writer=writer)

        return connection

    def connections(self) -> Dict:
        """RemoteConnectionManager does not track active connections, so return empty dict"""
        return {}
