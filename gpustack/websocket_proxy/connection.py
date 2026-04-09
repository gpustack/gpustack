#!/usr/bin/env python3
"""
TunnelConnection - Represents a tunnel connection to target server
"""

from abc import ABC, abstractmethod
import asyncio
import logging
import uuid
from functools import partial
from typing import Optional, Union, TYPE_CHECKING, Callable, Coroutine, Any
from fastapi import HTTPException

if TYPE_CHECKING:
    from websockets.client import ClientConnection
    from websockets.server import ServerConnection
    from starlette.websockets import WebSocket as StarletteWebSocket

from .message import DataMessage, DisconnectMessage, pack_message

logger = logging.getLogger(__name__)


class IOConnection(ABC):
    @abstractmethod
    async def read(self, n: int = -1, timeout: Optional[float] = None) -> bytes:
        pass

    @abstractmethod
    async def write(self, data: bytes) -> None:
        pass

    @abstractmethod
    async def close(self) -> None:
        pass


class AsyncIOConnection(IOConnection):
    writer: asyncio.StreamWriter
    reader: asyncio.StreamReader

    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        self.reader = reader
        self.writer = writer

    async def read(self, n: int = -1, timeout: Optional[float] = None) -> bytes:
        if timeout is not None:
            return await asyncio.wait_for(self.reader.read(n), timeout)
        return await self.reader.read(n)

    async def write(self, data: bytes) -> None:
        self.writer.write(data)
        await self.writer.drain()

    async def close(self) -> None:
        self.writer.close()
        await self.writer.wait_closed()

    async def send_error(self, code: int, message: str) -> None:
        """Send error response to client with message in body"""
        reason_phrases = {
            400: "Bad Request",
            502: "Bad Gateway",
            503: "Service Unavailable",
        }
        reason = reason_phrases.get(code, "Error")
        response = f"HTTP/1.1 {code} {reason}\r\n"
        response += "Content-Type: text/plain\r\n"
        response += f"Content-Length: {len(message)}\r\n"
        response += "Connection: close\r\n"
        response += "\r\n"
        response += message
        await self.write(response.encode())

    async def write_connect_established(self) -> None:
        """Send 200 Connection Established response to client"""
        response = b"HTTP/1.1 200 Connection Established\r\n\r\n"
        await self.write(response)


class TunnelConnection(IOConnection):
    """Represents a tunnel connection to target server"""

    def __init__(
        self,
        session_id: uuid.UUID,
        websocket: Union[
            "ClientConnection", "ServerConnection", "StarletteWebSocket", None
        ],
    ) -> None:
        self.session_id = session_id
        self.websocket = websocket
        # Connection state
        self._pending_future: Optional[asyncio.Future[bool]] = (
            asyncio.get_event_loop().create_future()
        )
        # Response tracking queue for WebSocket tunnel mode
        self._response_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=100)
        self._connection_error: Optional[Exception] = None

    @property
    def is_pending(self) -> bool:
        """Check if connection is in pending state (waiting for response)"""
        return self._pending_future is not None and not self._pending_future.done()

    @property
    def is_connected(self) -> bool:
        """Check if connection is established"""
        return self._pending_future is None and self._connection_error is None

    @property
    def connect_result(self) -> asyncio.Future[bool]:
        """Get the future representing the connection state"""
        return self._pending_future

    def set_connected(self) -> None:
        """Mark connection as connected"""
        if self._pending_future is None:
            logger.warning(
                f"set_connected called but no pending future, session_id={self.session_id}"
            )
            return
        if self._pending_future and not self._pending_future.done():
            self._pending_future.set_result(True)
        self._pending_future = None

    def connect_error(self, error: Exception) -> None:
        """Mark connection as failed with error"""
        if self._pending_future is None:
            logger.warning(
                f"connect_error called but no pending future, session_id={self.session_id}, error={error}"
            )
            return
        if self._pending_future and not self._pending_future.done():
            self._pending_future.set_exception(error)
        self._connection_error = error
        self._pending_future = None

    async def _send_to_websocket(self, data: bytes) -> None:
        """Send data to WebSocket, compatible with Starlette and websockets library"""
        if hasattr(self.websocket, 'send_bytes'):
            await self.websocket.send_bytes(data)
        else:
            await self.websocket.send(data)

    async def handle_data(self, data: bytes) -> None:
        """Handle data received from WebSocket, forward to target"""
        logger.trace(
            f"[Tunnel] handle_data: session_id={self.session_id}, {len(data)} bytes"
        )
        if not self.is_connected:
            logger.warning(
                f"[Tunnel] Connection is pending or failed, ignoring data until connected, session_id={self.session_id}"
            )
            return

        if not data:
            logger.trace("[Tunnel] Empty data received, signaling EOF")
        else:
            logger.trace(f"[Tunnel] Queuing {len(data)} bytes for response tracking")
        await self._response_queue.put(data)
        return

    # Followings methods are for compatibility with IOConnection interface, used in tunnel function
    async def close(self) -> None:
        """Close connection"""
        # Send disconnect message to WebSocket
        logger.trace(
            f"[Tunnel] Closing connection, sending DisconnectMessage, session_id={self.session_id}"
        )
        msg = DisconnectMessage(session_id=self.session_id)
        await self._send_to_websocket(pack_message(msg))
        await self._response_queue.put(b"")  # Unblock any pending reads

    async def read(self, _n: int = -1, timeout: float = 30.0) -> bytes:
        return await asyncio.wait_for(self._response_queue.get(), timeout=timeout)

    async def write(self, data: bytes) -> None:
        """Send data to WebSocket"""
        logger.trace(
            f"[Tunnel] Sending {str(data)} to WebSocket, session_id={self.session_id}"
        )
        msg = DataMessage(session_id=self.session_id, data=data)
        await self._send_to_websocket(pack_message(msg))


async def relay(
    reader: IOConnection,
    writer: IOConnection,
    name: str,
) -> None:
    try:
        while True:
            data = await reader.read(8192)
            if not data:
                logger.trace(f"{name}: read EOF")
                break
            logger.trace(f"{name}: forwarding {len(data)} bytes")
            await writer.write(data)
    except Exception as e:
        logger.error(f"{name}: error {e}")
    finally:
        logger.trace(f"{name}: closing {type(writer).__name__} writer")
        await writer.close()


async def tunnel(
    client_connection: IOConnection,
    remote_connection: IOConnection,
    name: str = "Tunnel",
    request: Optional[str] = None,
    response_relay: Optional[
        Callable[[IOConnection, IOConnection], Coroutine[Any, Any, None]]
    ] = None,
) -> None:
    """Tunnel data between client and remote connections"""
    if request is not None:
        logger.trace(
            f"[{name}] client->remote: Sending initial request data through tunnel:\n{request[:500]}..."
        )
        try:
            await remote_connection.write(request.encode())
        except Exception as e:
            logger.error(f"[{name}] Error sending initial request data: {e}")
            await remote_connection.close()
            raise HTTPException(
                status_code=502, detail="Failed to send initial request data"
            )
    if response_relay is None:
        response_relay = partial(relay, name=f"[{name}] remote->client")
    await asyncio.gather(
        response_relay(remote_connection, client_connection),
        relay(client_connection, remote_connection, f"[{name}] client->remote"),
        return_exceptions=True,
    )
    logger.trace(f"[{name}] Tunnel closed")
