#!/usr/bin/env python3
"""
HTTP Proxy Server based on asyncio

Supports HTTP/1.1 proxy protocol:
- HTTP requests: Client sends full URI (http://host:port/path)
- HTTPS requests: CONNECT method to establish tunnel

Run:
    python -m src.aio_proxy --host 0.0.0.0 --port 8080

Test:
    curl --proxy http://localhost:8080 http://example.com
    curl --proxy http://localhost:8080 https://example.com
"""
import logging
import asyncio
from fastapi import HTTPException
from typing import Optional, Dict, Callable, Coroutine, Any, Tuple, TypeAlias
from urllib.parse import urlparse

from .connection_manager import BaseConnectionManager
from .connection import AsyncIOConnection, tunnel, IOConnection

logger = logging.getLogger(__name__)


# ==================== Handler Functions ====================
ConnectionManagerGetter: TypeAlias = Callable[[str], Optional[BaseConnectionManager]]
HeaderAuthenticator: TypeAlias = Callable[[Dict[str, str]], Coroutine[Any, Any, bool]]
HeaderRouter: TypeAlias = Callable[
    [Dict[str, str]], Coroutine[Any, Any, Tuple[Optional[str], int]]
]


async def _read_line(reader: asyncio.StreamReader) -> Optional[str]:
    """Read a line from the client"""
    try:
        line = await reader.readline()
        if not line:
            return None
        return line.decode('utf-8').strip()
    except Exception:
        return None


async def _read_headers(reader: asyncio.StreamReader) -> Dict[str, str]:
    """Read all headers from the client"""
    headers: Dict[str, str] = {}
    while True:
        line = await _read_line(reader)
        if not line:
            break
        if line == "":
            break
        if ":" in line:
            key, value = line.split(":", 1)
            headers[key.strip().lower()] = value.strip()
    return headers


async def _handle_connect(
    client_connection: AsyncIOConnection,
    target: str,
    connection_manager: BaseConnectionManager,
) -> None:
    """Handle CONNECT request for HTTPS tunnel"""
    logger.debug(f"[Proxy] CONNECT target: {target}")

    # Parse host and port
    if ":" in target:
        host, port_str = target.rsplit(":", 1)
        try:
            port = int(port_str)
        except ValueError:
            await client_connection.send_error(400, "Invalid port")
            return
    else:
        host = target
        port = 443

    try:
        target_url = f"tcp://{host}:{port}"
        logger.debug(
            f"[Proxy] Connecting to {target_url} with manager: {connection_manager}"
        )
        connection = await asyncio.wait_for(
            connection_manager.connect(target_url), timeout=30
        )
        logger.debug(f"[Proxy] Connection established: {connection}")
    except Exception as e:
        logger.exception(
            f"[Proxy] Error connecting to target {host}:{port}, error: {e}"
        )
        await client_connection.send_error(502, str(e))
        return

    # Send 200 Connection Established
    await client_connection.write_connect_established()
    await tunnel(client_connection, connection)


# Hop-by-hop headers that should not be forwarded
# Per RFC 7230 Section 6.1, these headers are hop-by-hop:
# connection, keep-alive, proxy-authenticate, proxy-authorization,
# proxy-connection, te, trailer, transfer-encoding, upgrade
# Note: Upgrade is NOT in this list - it's an end-to-end header
# that should be forwarded to the backend server (needed for WebSocket)
HOP_BY_HOP_HEADERS = {
    'connection',
    'keep-alive',
    'proxy-authenticate',
    'proxy-authorization',
    'proxy-connection',
    'te',
    'trailer',
    # Do not filter 'transfer-encoding' as it may be needed for chunked encoding, and the backend server should handle it correctly.
    # 'transfer-encoding',
    'host',
}


def _filter_headers(headers) -> list[tuple[str, str]]:
    """Filter out hop-by-hop headers that should not be forwarded.
    Supports list of tuples (for response headers) or dict (for request headers).
    """
    filtered: list[tuple[str, str]] = []
    connection_values: set[str] = set()

    # Extract values from Connection header
    if isinstance(headers, dict):
        conn_header = headers.get('connection', '')
    elif isinstance(headers, list):
        conn_header = _header_get(headers, 'connection')
    if conn_header:
        for v in conn_header.split(','):
            connection_values.add(v.strip().lower())

    # Iterate over headers (supports both dict and list of tuples)
    if isinstance(headers, dict):
        items = headers.items()
    else:
        items = headers
    for key, value in items:
        # Skip hop-by-hop headers
        if key.lower() in HOP_BY_HOP_HEADERS:
            continue
        # Skip headers listed in Connection header
        if key.lower() in connection_values:
            continue
        filtered.append((key, value))

    return filtered


def _get_request(
    method: str,
    path: str,
    headers: Dict[str, str],
) -> str:
    """Construct HTTP request line and headers for forwarding"""
    request = f"{method} {path} HTTP/1.1\r\n"
    for key, value in headers:
        request += f"{key}: {value}\r\n"
    request += "\r\n"
    return request


async def _handle_http(
    client_connection: AsyncIOConnection,
    method: str,
    uri: str,
    headers: Dict[str, str],
    connection_manager: BaseConnectionManager,
    header_router: Optional[HeaderRouter] = None,
) -> None:
    """Handle HTTP request with full URI"""
    parsed = urlparse(uri)
    host, port = await header_router(headers) if header_router else (None, 0)
    if host is None:
        host = parsed.hostname
        port = parsed.port or (80 if parsed.scheme == "http" else 443)

    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"

    if not host:
        await client_connection.send_error(400, "Invalid URI")
        return

    logger.debug(f"[Proxy] {method} {uri} -> {host}:{port}")

    # Filter hop-by-hop headers
    filtered_headers: list[tuple[str, str]] = _filter_headers(headers)
    host_value = f"{host}:{port}" if port != 80 else host
    filtered_headers.append(('host', host_value))

    try:
        target_url = f"tcp://{host}:{port}"
        connection = await asyncio.wait_for(
            connection_manager.connect(target_url), timeout=30
        )
    except Exception as e:
        logger.exception(
            f"[Proxy] Error connecting to target {host}:{port}, error: {e}"
        )
        await client_connection.send_error(502, str(e))
        return

    try:
        await tunnel(
            client_connection,
            connection,
            request=_get_request(method, path, filtered_headers),
            response_relay=wait_for_complete_response,
        )
    except HTTPException as e:
        await client_connection.send_error(e.status_code, e.detail)
    finally:
        if connection:
            await connection.close()


def _header_get(headers: list[tuple[str, str]], key: str) -> str:
    """Get header value from headers list. Returns empty string if not found."""
    for k, v in headers:
        if k == key.lower():
            return v
    return ""


async def wait_for_complete_response(  # noqa: C901
    remote_reader: IOConnection,
    client_writer: IOConnection,
) -> None:
    """Wait for complete HTTP response from WebSocket tunnel and forward to client.

    For WebSocket tunnel responses, we forward the raw response data directly without
    header filtering, as the response comes from a trusted internal source and may use
    chunked transfer encoding that would be broken by header parsing/reconstruction.
    """
    if client_writer is None:
        logger.debug("[Proxy] Client writer is None")
        return

    pending_data = b''
    headers_sent = False
    content_length: Optional[int] = None
    body_remaining: Optional[int] = None

    try:
        while True:
            chunk = await remote_reader.read()
            if not chunk:
                logger.debug(
                    "[Proxy] Remote connection closed while waiting for response"
                )
                break
            pending_data += chunk

            # Parse headers on first chunk
            if not headers_sent and b'\r\n\r\n' in pending_data:
                header_end = pending_data.find(b'\r\n\r\n')
                header_part = pending_data[:header_end].decode('utf-8', errors='ignore')
                body_start = header_end + 4

                # Look for Content-Length
                for line in header_part.split('\r\n'):
                    if line.lower().startswith('content-length:'):
                        content_length = int(line.split(':')[1].strip())
                        body_remaining = content_length
                        break

                # Forward headers + any body data already received
                if body_start < len(pending_data):
                    body_data = pending_data[body_start:]
                    # Send headers first
                    await client_writer.write(pending_data[:body_start])
                    # Then send initial body
                    if body_data:
                        if body_remaining is not None:
                            to_write = min(len(body_data), body_remaining)
                            await client_writer.write(body_data[:to_write])
                            body_remaining -= to_write
                            pending_data = b''
                        else:
                            # Chunked - send and continue
                            await client_writer.write(body_data)
                            pending_data = b''
                else:
                    await client_writer.write(pending_data)
                    pending_data = b''

                headers_sent = True
                logger.trace(
                    f"[Proxy] Headers sent, content_length={content_length}, body_remaining={body_remaining}"
                )

                if body_remaining is not None and body_remaining <= 0:
                    return
                continue

            # Forward subsequent body data
            if body_remaining is not None:
                to_write = min(len(pending_data), body_remaining)
                if to_write > 0:
                    await client_writer.write(pending_data[:to_write])
                    pending_data = pending_data[to_write:]
                    body_remaining -= to_write
                    logger.trace(
                        f"[Proxy] Forwarded {to_write} bytes, body_remaining={body_remaining}"
                    )
                if body_remaining <= 0:
                    return
            else:
                # No Content-Length: stream until source closes (chunked encoding)
                if pending_data:
                    await client_writer.write(pending_data)
                    pending_data = b''
    except asyncio.TimeoutError:
        logger.debug("[Proxy] Timeout waiting for response")
        return
    except Exception as e:
        logger.debug(f"[Proxy] Error waiting for complete response: {e}")
        return
    finally:
        await client_writer.close()


# ==================== Server Class ====================


class HTTPSProxyServer:
    """Async HTTP/HTTPS Proxy Server"""

    def __init__(
        self,
        host: str,
        port: int,
        connection_manager_getter: ConnectionManagerGetter,
        authenticator: Optional[HeaderAuthenticator] = None,
        header_router: Optional[HeaderRouter] = None,
    ) -> None:
        self.host = host
        self.port = port
        self.server: Optional[asyncio.Server] = None
        self.connection_manager_getter = connection_manager_getter
        self.authenticator = authenticator
        self.header_router = header_router

    async def start(self) -> None:
        """Start the proxy server"""
        self.server = await asyncio.start_server(
            self._handle_client, self.host, self.port
        )
        logger.debug(f"[Proxy] Server started on {self.host}:{self.port}")

        async with self.server:
            await self.server.serve_forever()

    async def stop(self) -> None:
        """Stop the proxy server"""
        if self.server:
            self.server.close()
            await self.server.wait_closed()

    async def _get_target_ip(
        self, method: str, uri: str, headers: Dict[str, str]
    ) -> Optional[str]:
        """Extract target IP/hostname from request"""
        if method == "CONNECT":
            # CONNECT target is the host:port (e.g., "example.com:443")
            return uri.split(":")[0] if ":" in uri else uri
        elif self.header_router:
            target_ip, _ = await self.header_router(headers)
            if target_ip:
                return target_ip

        # HTTP request: parse URI (e.g., "http://example.com:8080/path")
        parsed = urlparse(uri)
        if parsed.hostname:
            return parsed.hostname
        # Fallback to Host header
        return headers.get("host", "").split(":")[0] or None

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        client_connection = AsyncIOConnection(reader=reader, writer=writer)
        """Handle incoming client connection"""
        client_addr = writer.get_extra_info('peername')

        try:
            request_line = await _read_line(reader)
            if not request_line:
                return

            parts = request_line.split()
            if len(parts) < 3:
                await client_connection.send_error(400, "Bad Request")
                return

            method, uri, _ = parts[0], parts[1], parts[2]

            headers = await _read_headers(reader)
            logger.debug(f"[Proxy] Received request: {method} {uri} from {client_addr}")
            logger.trace(f"[Proxy] Headers: {headers}")

            # Authenticate before any other processing
            # Skip authenticator for /metrics path on non-CONNECT requests
            result = urlparse(uri)
            should_skip_auth = method == "GET" and result.path == "/metrics"
            if self.authenticator and not should_skip_auth:
                if not await self.authenticator(headers):
                    await client_connection.send_error(401, "Unauthorized")
                    return
            # Extract target address from request
            target_ip = await self._get_target_ip(method, uri, headers)
            if not target_ip:
                await client_connection.send_error(
                    400, "Bad Request: No target address"
                )
                return

            # Get connection manager by target IP
            connection_manager = (
                self.connection_manager_getter(target_ip)
                if self.connection_manager_getter
                else None
            )
            if connection_manager is None:
                # failed to get connection manager, return error.
                logger.debug(
                    f"[Proxy] No connection manager available for target: {target_ip}"
                )
                await client_connection.send_error(
                    502, "Bad Gateway: No connection manager available"
                )
                return

            if method == "CONNECT":
                await _handle_connect(client_connection, uri, connection_manager)
            else:
                await _handle_http(
                    client_connection,
                    method,
                    uri,
                    headers,
                    connection_manager,
                    self.header_router,
                )

        except Exception as e:
            logger.debug(f"[Proxy] Error handling {client_addr}: {e}")
        finally:
            await client_connection.close()
