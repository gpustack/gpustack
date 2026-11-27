#!/usr/bin/env python3
"""
Message Client - Client that handles CONNECT_REQUEST from server
"""

import asyncio
import contextlib
import logging
import random
import time
import uuid
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed
from typing import List, Optional
from gpustack.ssl_context import make_ssl_context
from .connection_manager import ClientConnectionManager
from .message import (
    BaseClientInfo,
    HeartbeatMessage,
    SessionBaseMessage,
    pack_message,
    parse_message,
)
from .authenticator import Authenticator, create_authenticator
from .constants import default_connect_path

logger = logging.getLogger(__name__)

# Reconnect constants
INITIAL_RECONNECT_DELAY = 1.0
MAX_RECONNECT_DELAY = 60.0
RECONNECT_JITTER_FACTOR = 0.3

# Liveness signal cadence. The server reaps a tunnel whose websocket has
# been silent past its read timeout once the client has sent at least one
# heartbeat; keep the interval well below that timeout.
HEARTBEAT_INTERVAL = 10.0


class MessageClient:
    """Client that handles CONNECT_REQUEST from server using ClientConnectionManager"""

    _client_info: BaseClientInfo

    def __init__(
        self,
        server_endpoint: str,
        client_id: uuid.UUID,
        cidrs: Optional[List[str]] = None,
        unix_sockets: Optional[List[str]] = None,
        authenticator: Optional[Authenticator] = None,
    ) -> None:
        # replace http(s):// with ws(s):// and append connect path
        self.server_uri = (
            server_endpoint.replace('https://', 'wss://').replace('http://', 'ws://')
            + default_connect_path
        )
        self._client_info = BaseClientInfo(
            client_id=client_id,
            cidrs=cidrs or [],
            unix_sockets=unix_sockets or [],
        )
        self._authenticator = (
            authenticator if authenticator is not None else create_authenticator(None)
        )
        self._lock = asyncio.Lock()
        self._websocket = None

    async def update_cidrs(self, cidrs: List[str]) -> None:
        """Update CIDRs for the client (thread-safe)"""
        async with self._lock:
            self._client_info.cidrs = cidrs
        logger.debug(f"[Client] Updated CIDRs: {cidrs}")
        if self._websocket and not self._websocket.close_code:
            await self._websocket.close(
                code=1008, reason="CIDRs updated"
            )  # Trigger reconnect to update server with new CIDRs

    async def _send_heartbeats(self, websocket) -> None:
        """Keep the liveness signal flowing so the server can reap stale tunnels."""
        try:
            while True:
                await asyncio.sleep(HEARTBEAT_INTERVAL)
                await websocket.send(
                    pack_message(HeartbeatMessage(timestamp=int(time.time())))
                )
        except asyncio.CancelledError:
            pass
        except Exception as e:
            # A failed heartbeat send means this connection can no longer
            # prove liveness; the server would reap it as a stale tunnel
            # after its idle timeout while the client still believes it is
            # healthy. Close now so run() reconnects immediately instead of
            # lingering in one-way silence.
            logger.debug(f"[Client] Heartbeat send failed, closing tunnel: {e}")
            with contextlib.suppress(Exception):
                await websocket.close()

    async def run(self) -> None:
        """Connect to server and handle incoming messages with automatic reconnect"""
        reconnect_delay = INITIAL_RECONNECT_DELAY

        while True:
            async with self._lock:
                headers = self._client_info.to_headers()
            self._authenticator.inject_headers(headers)
            try:
                self._websocket = await connect(
                    self.server_uri,
                    proxy=None,
                    additional_headers=headers,
                    ssl=(
                        make_ssl_context()
                        if self.server_uri.startswith('wss://')
                        else None
                    ),
                )
                logger.debug(
                    f"[Client] Connected to {self.server_uri} "
                    f"with client_id: {self._client_info.client_id}"
                )
                connection_manager = ClientConnectionManager(self._websocket)
                heartbeat_task = asyncio.create_task(
                    self._send_heartbeats(self._websocket)
                )
                reconnect_delay = (
                    INITIAL_RECONNECT_DELAY  # Reset delay on successful connection
                )

                try:
                    async for raw_data in self._websocket:
                        msg = parse_message(raw_data)
                        logger.trace(f"[Client] Received: {msg.get_type()}")

                        if isinstance(msg, SessionBaseMessage):
                            await connection_manager.dispatch(msg)
                finally:
                    heartbeat_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await heartbeat_task

            except ConnectionClosed:
                # Suppress ConnectionClosed - reconnect automatically
                logger.debug("[Client] Server disconnected, reconnecting...")

            except asyncio.CancelledError:
                # Task was cancelled externally - exit gracefully
                logger.debug("[Client] Task was cancelled, stopping")
                return

            except Exception as e:
                logger.error(f"[Client] Unexpected error: {e}, reconnecting...")

            # Exponential backoff with jitter
            jitter = (
                reconnect_delay * RECONNECT_JITTER_FACTOR * (2 * random.random() - 1)
            )
            actual_delay = min(reconnect_delay + jitter, MAX_RECONNECT_DELAY)
            logger.debug(f"[Client] Reconnecting in {actual_delay:.2f} seconds")
            await asyncio.sleep(actual_delay)
            reconnect_delay = min(reconnect_delay * 2, MAX_RECONNECT_DELAY)
