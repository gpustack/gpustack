#!/usr/bin/env python3
"""
Message Client - Client that handles CONNECT_REQUEST from server
"""

import asyncio
import logging
import random
import uuid
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed
from typing import List, Optional
from .connection_manager import ClientConnectionManager
from .message import (
    BaseClientInfo,
    SessionBaseMessage,
    parse_message,
)
from .authenticator import Authenticator, create_authenticator
from .constants import default_connect_path

logger = logging.getLogger(__name__)

# Reconnect constants
INITIAL_RECONNECT_DELAY = 1.0
MAX_RECONNECT_DELAY = 60.0
RECONNECT_JITTER_FACTOR = 0.3


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

    async def update_cidrs(self, cidrs: List[str]) -> None:
        """Update CIDRs for the client (thread-safe)"""
        async with self._lock:
            self._client_info.cidrs = cidrs
        logger.debug(f"[Client] Updated CIDRs: {cidrs}")
        if self._websocket and not self._websocket.close_code:
            await self._websocket.close(
                code=1008, reason="CIDRs updated"
            )  # Trigger reconnect to update server with new CIDRs

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
                )
                logger.debug(
                    f"[Client] Connected to {self.server_uri} with client_id: {self._client_info.client_id}"
                )
                connection_manager = ClientConnectionManager(self._websocket)
                reconnect_delay = (
                    INITIAL_RECONNECT_DELAY  # Reset delay on successful connection
                )

                async for raw_data in self._websocket:
                    msg = parse_message(raw_data)
                    logger.trace(f"[Client] Received: {msg.get_type()}")

                    if isinstance(msg, SessionBaseMessage):
                        await connection_manager.dispatch(msg)

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
